"""
评估带分类头的Transform MLP模型

评估内容：
1. 分类性能：Accuracy, Precision, Recall, F1, AUC-ROC
2. 跨语言对齐：同一问题不同语言的表示相似度
3. 安全性分离：有害和无害问题的表示分离度
4. OOD泛化：在未见过的有害数据上的表现
"""

import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, silhouette_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from model import TransformMLPWithClassifier
from dataset import (
    load_dataset, collate_fn, load_safety_data, 
    extract_hidden_states, MultilingualHiddenStateDataset
)
from transformers import AutoTokenizer, AutoModelForCausalLM


def convert_to_serializable(obj):
    """递归地将numpy类型转换为Python原生类型"""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def compute_classification_metrics(
    logits: torch.Tensor,
    labels: np.ndarray,
    languages: list = None
) -> dict:
    """
    计算分类评估指标
    
    Args:
        logits: 模型输出的logits, shape [N, 1]
        labels: 真实标签, shape [N]
        languages: 语言列表（可选，用于按语言统计）
    
    Returns:
        指标字典
    """
    probs = torch.sigmoid(logits).squeeze(-1).numpy()
    preds = (probs > 0.5).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, zero_division=0),
        'recall': recall_score(labels, preds, zero_division=0),
        'f1': f1_score(labels, preds, zero_division=0),
    }
    
    # AUC-ROC
    if len(np.unique(labels)) > 1:
        metrics['auc_roc'] = roc_auc_score(labels, probs)
    else:
        metrics['auc_roc'] = 0.0
    
    # 混淆矩阵
    cm = confusion_matrix(labels, preds)
    metrics['confusion_matrix'] = cm.tolist()
    
    # True Positive, True Negative, False Positive, False Negative
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negative'] = int(tn)
        metrics['false_positive'] = int(fp)
        metrics['false_negative'] = int(fn)
        metrics['true_positive'] = int(tp)
        
        # 特异度 (Specificity) = Benign分类准确率
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics['benign_accuracy'] = metrics['specificity']  # 别名，更直观
        
        # Recall = Harmful分类准确率
        metrics['harmful_accuracy'] = metrics['recall']  # 别名，更直观
        
        # 各类别样本数
        metrics['total_benign'] = int(tn + fp)
        metrics['total_harmful'] = int(fn + tp)
    
    # 按语言统计
    if languages is not None:
        metrics['by_language'] = {}
        unique_langs = sorted(set(languages))
        
        for lang in unique_langs:
            lang_indices = [i for i, l in enumerate(languages) if l == lang]
            lang_labels = labels[lang_indices]
            lang_preds = preds[lang_indices]
            lang_probs = probs[lang_indices]
            
            lang_metrics = {
                'count': len(lang_indices),
                'safe_count': int(sum(lang_labels == 0)),
                'unsafe_count': int(sum(lang_labels == 1)),
                'accuracy': accuracy_score(lang_labels, lang_preds),
                'precision': precision_score(lang_labels, lang_preds, zero_division=0),
                'recall': recall_score(lang_labels, lang_preds, zero_division=0),
                'f1': f1_score(lang_labels, lang_preds, zero_division=0),
            }
            
            # 计算该语言的 benign_accuracy 和 harmful_accuracy
            lang_cm = confusion_matrix(lang_labels, lang_preds, labels=[0, 1])
            if lang_cm.shape == (2, 2):
                lang_tn, lang_fp, lang_fn, lang_tp = lang_cm.ravel()
                # Benign准确率 = TN / (TN + FP)
                lang_metrics['benign_accuracy'] = lang_tn / (lang_tn + lang_fp) if (lang_tn + lang_fp) > 0 else 0.0
                # Harmful准确率 = TP / (TP + FN) = Recall
                lang_metrics['harmful_accuracy'] = lang_tp / (lang_tp + lang_fn) if (lang_tp + lang_fn) > 0 else 0.0
            else:
                # 只有一个类别
                if sum(lang_labels == 0) > 0:  # 只有benign
                    lang_metrics['benign_accuracy'] = accuracy_score(lang_labels, lang_preds)
                    lang_metrics['harmful_accuracy'] = 0.0
                else:  # 只有harmful
                    lang_metrics['benign_accuracy'] = 0.0
                    lang_metrics['harmful_accuracy'] = accuracy_score(lang_labels, lang_preds)
            
            if len(np.unique(lang_labels)) > 1:
                lang_metrics['auc_roc'] = roc_auc_score(lang_labels, lang_probs)
            else:
                lang_metrics['auc_roc'] = 0.0
            
            metrics['by_language'][lang] = lang_metrics
    
    return metrics


def compute_embedding_metrics(
    embeddings: torch.Tensor,
    group_ids: list,
    safety_labels: list,
    languages: list
) -> dict:
    """
    计算embedding相关的评估指标
    
    Args:
        embeddings: 变换后的表示, shape [N, dim]
        group_ids: 问题ID列表
        safety_labels: 安全标签列表
        languages: 语言列表
    
    Returns:
        指标字典
    """
    # L2归一化
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    metrics = {}
    
    # ========== 1. 跨语言对齐指标 ==========
    group_to_indices = defaultdict(list)
    for i, gid in enumerate(group_ids):
        group_to_indices[gid].append(i)
    
    # 同组内相似度
    intra_group_sims = []
    for gid, indices in group_to_indices.items():
        if len(indices) < 2:
            continue
        
        group_emb = embeddings[indices]
        sim_matrix = torch.matmul(group_emb, group_emb.T)
        mask = ~torch.eye(len(indices), dtype=torch.bool)
        intra_group_sims.extend(sim_matrix[mask].tolist())
    
    metrics['intra_group_similarity'] = np.mean(intra_group_sims) if intra_group_sims else 0.0
    metrics['intra_group_std'] = np.std(intra_group_sims) if intra_group_sims else 0.0
    
    # ========== 2. 安全性区分指标 ==========
    safe_indices = [i for i, l in enumerate(safety_labels) if l == 0]
    unsafe_indices = [i for i, l in enumerate(safety_labels) if l == 1]
    
    if safe_indices and unsafe_indices:
        safe_emb = embeddings[safe_indices]
        unsafe_emb = embeddings[unsafe_indices]
        
        # Safe-Unsafe相似度
        cross_sim = torch.matmul(safe_emb, unsafe_emb.T)
        metrics['safe_unsafe_similarity'] = cross_sim.mean().item()
        
        # Safe-Safe相似度
        if len(safe_indices) >= 2:
            safe_sim = torch.matmul(safe_emb, safe_emb.T)
            mask = ~torch.eye(len(safe_indices), dtype=torch.bool)
            metrics['safe_safe_similarity'] = safe_sim[mask].mean().item()
        else:
            metrics['safe_safe_similarity'] = 0.0
        
        # Unsafe-Unsafe相似度
        if len(unsafe_indices) >= 2:
            unsafe_sim = torch.matmul(unsafe_emb, unsafe_emb.T)
            mask = ~torch.eye(len(unsafe_indices), dtype=torch.bool)
            metrics['unsafe_unsafe_similarity'] = unsafe_sim[mask].mean().item()
        else:
            metrics['unsafe_unsafe_similarity'] = 0.0
        
        # 安全性分离度
        avg_within = (metrics['safe_safe_similarity'] + metrics['unsafe_unsafe_similarity']) / 2
        metrics['safety_separation_gap'] = avg_within - metrics['safe_unsafe_similarity']
    
    # ========== 3. 聚类质量指标 ==========
    if len(set(safety_labels)) > 1 and len(embeddings) > 10:
        try:
            silhouette = silhouette_score(
                embeddings.numpy(),
                safety_labels,
                metric='cosine'
            )
            metrics['safety_silhouette_score'] = silhouette
        except:
            metrics['safety_silhouette_score'] = 0.0
    
    return metrics


def visualize_classification(
    probs: np.ndarray,
    labels: np.ndarray,
    output_dir: str
):
    """
    可视化分类结果
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. ROC曲线
    ax = axes[0, 0]
    if len(np.unique(labels)) > 1:
        fpr, tpr, thresholds = roc_curve(labels, probs)
        auc = roc_auc_score(labels, probs)
        ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc:.4f})')
        ax.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random')
        ax.fill_between(fpr, tpr, alpha=0.2)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # 2. Precision-Recall曲线
    ax = axes[0, 1]
    if len(np.unique(labels)) > 1:
        precision, recall, thresholds = precision_recall_curve(labels, probs)
        ax.plot(recall, precision, 'g-', linewidth=2)
        ax.fill_between(recall, precision, alpha=0.2)
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 3. 概率分布直方图
    ax = axes[1, 0]
    safe_probs = probs[labels == 0]
    unsafe_probs = probs[labels == 1]
    ax.hist(safe_probs, bins=50, alpha=0.6, label='Benign (label=0)', color='blue', density=True)
    ax.hist(unsafe_probs, bins=50, alpha=0.6, label='Harmful (label=1)', color='red', density=True)
    ax.axvline(x=0.5, color='black', linestyle='--', linewidth=1, label='Threshold=0.5')
    ax.set_xlabel('Predicted Probability (Harmful)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Prediction Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 混淆矩阵
    ax = axes[1, 1]
    preds = (probs > 0.5).astype(int)
    cm = confusion_matrix(labels, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Benign', 'Harmful'],
                yticklabels=['Benign', 'Harmful'])
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'classification_metrics.png'), dpi=300)
    plt.close()
    
    print(f"Classification visualization saved to {os.path.join(output_dir, 'classification_metrics.png')}")


def visualize_embeddings(
    embeddings: torch.Tensor,
    safety_labels: list,
    languages: list,
    probs: np.ndarray,
    output_dir: str
):
    """
    可视化embeddings
    """
    print("Computing t-SNE projection...")
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings.numpy())
    
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    
    # 1. 按真实标签着色
    ax = axes[0]
    colors = ['blue' if l == 0 else 'red' for l in safety_labels]
    ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, alpha=0.5, s=10)
    ax.set_title('Ground Truth Labels\n(Blue=Benign, Red=Harmful)', fontsize=14)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    
    # 2. 按预测概率着色
    ax = axes[1]
    scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                        c=probs, cmap='RdYlBu_r', alpha=0.5, s=10, vmin=0, vmax=1)
    plt.colorbar(scatter, ax=ax, label='P(Harmful)')
    ax.set_title('Predicted Probability\n(Harmful)', fontsize=14)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    
    # 3. 按语言着色
    ax = axes[2]
    unique_langs = sorted(set(languages))
    color_map = plt.cm.tab10(np.linspace(0, 1, len(unique_langs)))
    lang_to_color = {lang: color_map[i] for i, lang in enumerate(unique_langs)}
    colors = [lang_to_color[lang] for lang in languages]
    
    ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, alpha=0.5, s=10)
    ax.set_title('By Language', fontsize=14)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    
    # 添加语言图例
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=lang_to_color[lang], 
                          markersize=10, label=lang) for lang in unique_langs]
    ax.legend(handles=handles, loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'embeddings_tsne.png'), dpi=300)
    plt.close()
    
    print(f"t-SNE visualization saved to {os.path.join(output_dir, 'embeddings_tsne.png')}")


def visualize_question_clustering(
    original_embeddings: torch.Tensor,
    transformed_embeddings: torch.Tensor,
    group_ids: list,
    safety_labels: list,
    languages: list,
    output_dir: str,
    n_safe_questions: int = 5,
    n_unsafe_questions: int = 5,
    max_variants_per_question: int = 10
):
    """
    可视化问题级别的聚类效果
    
    选择N个问题（safe和unsafe各一半），每个问题选取多种表述（不同语言），
    用问题ID作为label来着色，展示MLP是否能将同一问题的不同表述聚类在一起。
    
    Args:
        original_embeddings: 原始hidden states, shape [N, dim]
        transformed_embeddings: MLP变换后的embeddings, shape [N, dim]
        group_ids: 问题ID列表
        safety_labels: 安全标签列表
        languages: 语言列表
        output_dir: 输出目录
        n_safe_questions: 选择的safe问题数量
        n_unsafe_questions: 选择的unsafe问题数量
        max_variants_per_question: 每个问题最多选择的表述数量
    """
    print(f"\nGenerating question clustering visualization...")
    print(f"  Selecting {n_safe_questions} safe + {n_unsafe_questions} unsafe questions")
    
    # 按group_id组织样本索引
    group_to_indices = defaultdict(list)
    for i, gid in enumerate(group_ids):
        group_to_indices[gid].append(i)
    
    # 按安全性分组
    safe_groups = {}
    unsafe_groups = {}
    for gid, indices in group_to_indices.items():
        label = safety_labels[indices[0]]
        if label == 0:
            safe_groups[gid] = indices
        else:
            unsafe_groups[gid] = indices
    
    # 筛选有足够表述变体的问题（至少2个变体）
    valid_safe_groups = {gid: indices for gid, indices in safe_groups.items() if len(indices) >= 2}
    valid_unsafe_groups = {gid: indices for gid, indices in unsafe_groups.items() if len(indices) >= 2}
    
    print(f"  Available safe questions with >=2 variants: {len(valid_safe_groups)}")
    print(f"  Available unsafe questions with >=2 variants: {len(valid_unsafe_groups)}")
    
    if len(valid_safe_groups) < n_safe_questions or len(valid_unsafe_groups) < n_unsafe_questions:
        print(f"  Warning: Not enough questions with multiple variants. Adjusting...")
        n_safe_questions = min(n_safe_questions, len(valid_safe_groups))
        n_unsafe_questions = min(n_unsafe_questions, len(valid_unsafe_groups))
    
    if n_safe_questions == 0 and n_unsafe_questions == 0:
        print("  Error: No valid questions found for clustering visualization.")
        return
    
    # 随机选择问题（按表述变体数量排序，优先选择变体多的）
    np.random.seed(42)  # 可复现
    
    sorted_safe = sorted(valid_safe_groups.items(), key=lambda x: len(x[1]), reverse=True)
    sorted_unsafe = sorted(valid_unsafe_groups.items(), key=lambda x: len(x[1]), reverse=True)
    
    # 选择变体数量较多的问题
    selected_safe = [gid for gid, _ in sorted_safe[:n_safe_questions]]
    selected_unsafe = [gid for gid, _ in sorted_unsafe[:n_unsafe_questions]]
    
    # 收集选中问题的样本索引
    selected_indices = []
    question_labels = []  # 用于着色的问题标签（0-9）
    question_ids = []     # 原始的group_id
    is_safe_list = []     # 是否为safe问题
    lang_list = []        # 语言
    
    question_idx = 0
    for gid in selected_safe:
        indices = valid_safe_groups[gid][:max_variants_per_question]
        for idx in indices:
            selected_indices.append(idx)
            question_labels.append(question_idx)
            question_ids.append(gid)
            is_safe_list.append(True)
            lang_list.append(languages[idx])
        question_idx += 1
    
    for gid in selected_unsafe:
        indices = valid_unsafe_groups[gid][:max_variants_per_question]
        for idx in indices:
            selected_indices.append(idx)
            question_labels.append(question_idx)
            question_ids.append(gid)
            is_safe_list.append(False)
            lang_list.append(languages[idx])
        question_idx += 1
    
    print(f"  Selected {len(selected_indices)} samples from {n_safe_questions + n_unsafe_questions} questions")
    
    # 提取选中样本的embeddings
    original_selected = original_embeddings[selected_indices].numpy()
    transformed_selected = transformed_embeddings[selected_indices].numpy()
    
    # 计算t-SNE
    print("  Computing t-SNE for original embeddings...")
    perplexity = min(30, len(selected_indices) - 1)
    tsne_original = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    original_2d = tsne_original.fit_transform(original_selected)
    
    print("  Computing t-SNE for transformed embeddings...")
    tsne_transformed = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    transformed_2d = tsne_transformed.fit_transform(transformed_selected)
    
    # 创建颜色映射（使用tab20，支持20种颜色）
    n_questions = n_safe_questions + n_unsafe_questions
    if n_questions <= 10:
        cmap = plt.cm.tab10
    else:
        cmap = plt.cm.tab20
    colors = [cmap(q / n_questions) for q in question_labels]
    
    # 创建标记映射（safe用圆形，unsafe用三角形）
    markers = ['o' if is_safe else '^' for is_safe in is_safe_list]
    
    # 创建2x2子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # ============ 1. Original embeddings - by question ID ============
    ax = axes[0, 0]
    for i in range(len(selected_indices)):
        ax.scatter(original_2d[i, 0], original_2d[i, 1], 
                  c=[colors[i]], marker=markers[i], s=80, alpha=0.7, edgecolors='white', linewidths=0.5)
    ax.set_title('Original Hidden States\n(Colored by Question ID)', fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    
    # ============ 2. Transformed embeddings - by question ID ============
    ax = axes[0, 1]
    for i in range(len(selected_indices)):
        ax.scatter(transformed_2d[i, 0], transformed_2d[i, 1], 
                  c=[colors[i]], marker=markers[i], s=80, alpha=0.7, edgecolors='white', linewidths=0.5)
    ax.set_title('MLP Transformed Embeddings\n(Colored by Question ID)', fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    
    # ============ 3. Original embeddings - by language ============
    ax = axes[1, 0]
    unique_langs = sorted(set(lang_list))
    lang_cmap = plt.cm.Set3(np.linspace(0, 1, len(unique_langs)))
    lang_to_color = {lang: lang_cmap[i] for i, lang in enumerate(unique_langs)}
    for i in range(len(selected_indices)):
        ax.scatter(original_2d[i, 0], original_2d[i, 1], 
                  c=[lang_to_color[lang_list[i]]], marker=markers[i], s=80, alpha=0.7, 
                  edgecolors='white', linewidths=0.5)
    ax.set_title('Original Hidden States\n(Colored by Language)', fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    # 添加语言图例
    lang_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=lang_to_color[lang], 
                               markersize=10, label=lang) for lang in unique_langs]
    ax.legend(handles=lang_handles, loc='upper right', fontsize=8, title='Language')
    
    # ============ 4. Transformed embeddings - by language ============
    ax = axes[1, 1]
    for i in range(len(selected_indices)):
        ax.scatter(transformed_2d[i, 0], transformed_2d[i, 1], 
                  c=[lang_to_color[lang_list[i]]], marker=markers[i], s=80, alpha=0.7, 
                  edgecolors='white', linewidths=0.5)
    ax.set_title('MLP Transformed Embeddings\n(Colored by Language)', fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend(handles=lang_handles, loc='upper right', fontsize=8, title='Language')
    
    # 添加全局图例
    fig.suptitle(f'Question Clustering Analysis\n({n_safe_questions} Safe Questions [○] + {n_unsafe_questions} Unsafe Questions [△])', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    # 创建问题ID图例
    question_handles = []
    for q in range(n_questions):
        is_safe = q < n_safe_questions
        marker = 'o' if is_safe else '^'
        color = cmap(q / n_questions)
        label = f"Q{q+1} ({'Safe' if is_safe else 'Unsafe'})"
        handle = plt.Line2D([0], [0], marker=marker, color='w', markerfacecolor=color, 
                           markersize=10, label=label, markeredgecolor='gray', markeredgewidth=0.5)
        question_handles.append(handle)
    
    # 在图的底部添加问题图例
    fig.legend(handles=question_handles, loc='lower center', ncol=min(5, n_questions), 
               fontsize=9, title='Question IDs', bbox_to_anchor=(0.5, -0.02))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'question_clustering_tsne.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # ============ 计算聚类质量指标 ============
    from sklearn.metrics import silhouette_score
    
    clustering_metrics = {}
    
    # 原始embedding的问题聚类质量
    if len(set(question_labels)) > 1:
        try:
            original_silhouette = silhouette_score(original_selected, question_labels, metric='cosine')
            clustering_metrics['original_question_silhouette'] = float(original_silhouette)
        except:
            clustering_metrics['original_question_silhouette'] = None
        
        try:
            transformed_silhouette = silhouette_score(transformed_selected, question_labels, metric='cosine')
            clustering_metrics['transformed_question_silhouette'] = float(transformed_silhouette)
        except:
            clustering_metrics['transformed_question_silhouette'] = None
        
        if clustering_metrics.get('original_question_silhouette') and clustering_metrics.get('transformed_question_silhouette'):
            improvement = clustering_metrics['transformed_question_silhouette'] - clustering_metrics['original_question_silhouette']
            clustering_metrics['silhouette_improvement'] = float(improvement)
    
    # 计算同问题内的平均余弦相似度
    from torch.nn.functional import normalize
    
    original_norm = normalize(torch.tensor(original_selected), p=2, dim=1)
    transformed_norm = normalize(torch.tensor(transformed_selected), p=2, dim=1)
    
    original_intra_sims = []
    transformed_intra_sims = []
    
    for q in range(n_questions):
        q_indices = [i for i, ql in enumerate(question_labels) if ql == q]
        if len(q_indices) < 2:
            continue
        
        # 原始embedding的同问题相似度
        q_original = original_norm[q_indices]
        sim_matrix = torch.matmul(q_original, q_original.T)
        mask = ~torch.eye(len(q_indices), dtype=torch.bool)
        original_intra_sims.extend(sim_matrix[mask].tolist())
        
        # 变换后embedding的同问题相似度
        q_transformed = transformed_norm[q_indices]
        sim_matrix = torch.matmul(q_transformed, q_transformed.T)
        transformed_intra_sims.extend(sim_matrix[mask].tolist())
    
    if original_intra_sims:
        clustering_metrics['original_intra_question_similarity'] = float(np.mean(original_intra_sims))
        clustering_metrics['transformed_intra_question_similarity'] = float(np.mean(transformed_intra_sims))
        clustering_metrics['intra_similarity_improvement'] = float(
            np.mean(transformed_intra_sims) - np.mean(original_intra_sims)
        )
    
    # 保存聚类指标
    with open(os.path.join(output_dir, 'question_clustering_metrics.json'), 'w') as f:
        json.dump(clustering_metrics, f, indent=2)
    
    print(f"\n  Question Clustering Metrics:")
    for key, value in clustering_metrics.items():
        if value is not None:
            print(f"    {key}: {value:.4f}")
    
    print(f"\n  Visualization saved to {os.path.join(output_dir, 'question_clustering_tsne.png')}")
    print(f"  Metrics saved to {os.path.join(output_dir, 'question_clustering_metrics.json')}")
    
    return clustering_metrics


def visualize_by_language(metrics: dict, output_dir: str):
    """
    按语言可视化分类性能
    """
    if 'by_language' not in metrics:
        return
    
    by_lang = metrics['by_language']
    languages = list(by_lang.keys())
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 准备数据
    accuracies = [by_lang[l]['accuracy'] for l in languages]
    f1_scores = [by_lang[l]['f1'] for l in languages]
    auc_scores = [by_lang[l]['auc_roc'] for l in languages]
    benign_accs = [by_lang[l].get('benign_accuracy', 0) for l in languages]
    harmful_accs = [by_lang[l].get('harmful_accuracy', 0) for l in languages]
    
    x = np.arange(len(languages))
    width = 0.6
    
    # 1. Overall Accuracy
    ax = axes[0, 0]
    bars = ax.bar(x, accuracies, width, color='steelblue', alpha=0.8)
    ax.axhline(y=metrics['accuracy'], color='red', linestyle='--', label=f'Overall: {metrics["accuracy"]:.3f}')
    ax.set_ylabel('Accuracy')
    ax.set_title('Overall Accuracy by Language', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(languages)
    ax.legend()
    ax.set_ylim(0, 1.1)
    for bar, v in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Benign Accuracy (Specificity)
    ax = axes[0, 1]
    bars = ax.bar(x, benign_accs, width, color='blue', alpha=0.8)
    if 'benign_accuracy' in metrics:
        ax.axhline(y=metrics['benign_accuracy'], color='red', linestyle='--', 
                   label=f'Overall: {metrics["benign_accuracy"]:.3f}')
    ax.set_ylabel('Benign Accuracy')
    ax.set_title('Benign (Safe) Classification Accuracy', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(languages)
    ax.legend()
    ax.set_ylim(0, 1.1)
    for bar, v in zip(bars, benign_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Harmful Accuracy (Recall)
    ax = axes[0, 2]
    bars = ax.bar(x, harmful_accs, width, color='red', alpha=0.8)
    if 'harmful_accuracy' in metrics:
        ax.axhline(y=metrics['harmful_accuracy'], color='blue', linestyle='--', 
                   label=f'Overall: {metrics["harmful_accuracy"]:.3f}')
    ax.set_ylabel('Harmful Accuracy')
    ax.set_title('Harmful Classification Accuracy (Recall)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(languages)
    ax.legend()
    ax.set_ylim(0, 1.1)
    for bar, v in zip(bars, harmful_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 4. F1 Score
    ax = axes[1, 0]
    bars = ax.bar(x, f1_scores, width, color='forestgreen', alpha=0.8)
    ax.axhline(y=metrics['f1'], color='red', linestyle='--', label=f'Overall: {metrics["f1"]:.3f}')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score by Language', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(languages)
    ax.legend()
    ax.set_ylim(0, 1.1)
    for bar, v in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 5. AUC-ROC
    ax = axes[1, 1]
    bars = ax.bar(x, auc_scores, width, color='darkorange', alpha=0.8)
    ax.axhline(y=metrics['auc_roc'], color='red', linestyle='--', label=f'Overall: {metrics["auc_roc"]:.3f}')
    ax.set_ylabel('AUC-ROC')
    ax.set_title('AUC-ROC by Language', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(languages)
    ax.legend()
    ax.set_ylim(0, 1.1)
    for bar, v in zip(bars, auc_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 6. Sample Count
    ax = axes[1, 2]
    safe_counts = [by_lang[l]['safe_count'] for l in languages]
    unsafe_counts = [by_lang[l]['unsafe_count'] for l in languages]
    ax.bar(x - width/4, safe_counts, width/2, label='Benign', color='blue', alpha=0.7)
    ax.bar(x + width/4, unsafe_counts, width/2, label='Harmful', color='red', alpha=0.7)
    ax.set_ylabel('Sample Count')
    ax.set_title('Sample Distribution by Language', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(languages)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_by_language.png'), dpi=300)
    plt.close()
    
    print(f"Language metrics saved to {os.path.join(output_dir, 'metrics_by_language.png')}")


def evaluate(
    model: TransformMLPWithClassifier,
    dataset,
    device: str,
    output_dir: str,
    batch_size: int = 64,
    visualize: bool = True,
    n_safe_questions: int = 5,
    n_unsafe_questions: int = 5,
    max_variants_per_question: int = 10
):
    """
    评估模型
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # 收集所有输出
    all_logits = []
    all_embeddings = []
    all_original = []
    all_group_ids = []
    all_safety_labels = []
    all_languages = []
    
    with torch.no_grad():
        for batch in dataloader:
            hidden_states = batch['hidden_states'].to(device)
            
            outputs = model(hidden_states, return_embeddings=True)
            
            all_logits.append(outputs['logits'].cpu())
            all_embeddings.append(outputs['embeddings'].cpu())
            all_original.append(hidden_states.cpu())
            all_group_ids.extend(batch['group_ids'].tolist())
            all_safety_labels.extend(batch['safety_labels'].tolist())
            all_languages.extend(batch['languages'])
    
    all_logits = torch.cat(all_logits, dim=0)
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_original = torch.cat(all_original, dim=0)
    all_safety_labels = np.array(all_safety_labels)
    
    print("\n" + "=" * 60)
    print("Classifier Evaluation Results")
    print("=" * 60)
    
    # ========== 分类指标 ==========
    print("\n--- Classification Metrics ---")
    classification_metrics = compute_classification_metrics(
        all_logits, all_safety_labels, all_languages
    )
    
    print(f"  Accuracy:    {classification_metrics['accuracy']:.4f}")
    print(f"  Precision:   {classification_metrics['precision']:.4f}")
    print(f"  Recall:      {classification_metrics['recall']:.4f}")
    print(f"  F1 Score:    {classification_metrics['f1']:.4f}")
    print(f"  AUC-ROC:     {classification_metrics['auc_roc']:.4f}")
    
    # 分类别准确率
    print(f"\n  Per-Class Accuracy:")
    if 'benign_accuracy' in classification_metrics:
        print(f"    Benign (label=0):  {classification_metrics['benign_accuracy']:.4f} "
              f"({classification_metrics.get('total_benign', 'N/A')} samples)")
    if 'harmful_accuracy' in classification_metrics:
        print(f"    Harmful (label=1): {classification_metrics['harmful_accuracy']:.4f} "
              f"({classification_metrics.get('total_harmful', 'N/A')} samples)")
    
    print(f"\n  Confusion Matrix:")
    print(f"    TN={classification_metrics.get('true_negative', 'N/A')}, "
          f"FP={classification_metrics.get('false_positive', 'N/A')}")
    print(f"    FN={classification_metrics.get('false_negative', 'N/A')}, "
          f"TP={classification_metrics.get('true_positive', 'N/A')}")
    
    # ========== Embedding指标 ==========
    print("\n--- Original Hidden States ---")
    original_embedding_metrics = compute_embedding_metrics(
        all_original, all_group_ids, all_safety_labels.tolist(), all_languages
    )
    
    for key, value in original_embedding_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n--- Transformed Embeddings ---")
    transformed_embedding_metrics = compute_embedding_metrics(
        all_embeddings, all_group_ids, all_safety_labels.tolist(), all_languages
    )
    
    for key, value in transformed_embedding_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # ========== 提升 ==========
    print("\n--- Embedding Improvement ---")
    for key in ['intra_group_similarity', 'safety_separation_gap', 'safety_silhouette_score']:
        if key in original_embedding_metrics and key in transformed_embedding_metrics:
            improvement = transformed_embedding_metrics[key] - original_embedding_metrics[key]
            print(f"  {key}: {improvement:+.4f}")
    
    # 保存结果
    results = {
        'classification': {k: v for k, v in classification_metrics.items() if k != 'by_language'},
        'classification_by_language': classification_metrics.get('by_language', {}),
        'original_embeddings': original_embedding_metrics,
        'transformed_embeddings': transformed_embedding_metrics
    }
    
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)
    
    print(f"\nResults saved to {os.path.join(output_dir, 'evaluation_results.json')}")
    
    # 可视化
    if visualize:
        print("\nGenerating visualizations...")
        
        probs = torch.sigmoid(all_logits).squeeze(-1).numpy()
        
        # 分类指标可视化
        visualize_classification(probs, all_safety_labels, output_dir)
        
        # Embedding可视化
        visualize_embeddings(
            all_embeddings, all_safety_labels.tolist(), all_languages, probs, output_dir
        )
        
        # 按语言的指标可视化
        visualize_by_language(classification_metrics, output_dir)
        
        # 问题级别聚类可视化
        visualize_question_clustering(
            original_embeddings=all_original,
            transformed_embeddings=all_embeddings,
            group_ids=all_group_ids,
            safety_labels=all_safety_labels.tolist(),
            languages=all_languages,
            output_dir=output_dir,
            n_safe_questions=n_safe_questions,
            n_unsafe_questions=n_unsafe_questions,
            max_variants_per_question=max_variants_per_question
        )
    
    return results


def prepare_ood_dataset(
    ood_data_path: str,
    model_name_or_path: str,
    layer_idx: int = 20,
    device: str = 'cuda',
    batch_size: int = 8,
    max_samples: int = None
):
    """
    准备OOD测试数据集
    """
    print(f"Loading OOD data from {ood_data_path}...")
    
    ood_data = load_safety_data(ood_data_path, max_samples=max_samples)
    print(f"Loaded {len(ood_data)} OOD samples")
    
    print(f"Loading model for hidden state extraction...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    texts = [item['prompt'] for item in ood_data]
    hidden_states = extract_hidden_states(
        model, tokenizer, texts,
        layer_idx=layer_idx,
        device=device,
        batch_size=batch_size
    )
    
    group_ids = [item['idx'] for item in ood_data]
    safety_labels = [1] * len(ood_data)  # OOD harmful data
    languages = [item['language'] for item in ood_data]
    
    dataset = MultilingualHiddenStateDataset(
        hidden_states=hidden_states,
        group_ids=group_ids,
        safety_labels=safety_labels,
        languages=languages,
        texts=texts
    )
    
    return dataset


def evaluate_ood(
    model: TransformMLPWithClassifier,
    ood_dataset,
    device: str,
    output_dir: str,
    batch_size: int = 64
):
    """
    评估OOD测试集上的分类性能
    
    OOD数据应该全部被预测为harmful (label=1)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    
    dataloader = DataLoader(
        ood_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    all_logits = []
    all_embeddings = []
    all_safety_labels = []
    all_languages = []
    all_group_ids = []
    
    with torch.no_grad():
        for batch in dataloader:
            hidden_states = batch['hidden_states'].to(device)
            
            outputs = model(hidden_states, return_embeddings=True)
            
            all_logits.append(outputs['logits'].cpu())
            all_embeddings.append(outputs['embeddings'].cpu())
            all_safety_labels.extend(batch['safety_labels'].tolist())
            all_languages.extend(batch['languages'])
            all_group_ids.extend(batch['group_ids'].tolist())
    
    all_logits = torch.cat(all_logits, dim=0)
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_safety_labels = np.array(all_safety_labels)
    
    probs = torch.sigmoid(all_logits).squeeze(-1).numpy()
    preds = (probs > 0.5).astype(int)
    
    print("\n" + "=" * 60)
    print("OOD Evaluation Results")
    print("=" * 60)
    
    # OOD数据应该全部是harmful，所以我们主要看recall (检测率)
    print(f"\n--- OOD Detection (All samples are harmful) ---")
    print(f"  Total samples: {len(all_safety_labels)}")
    print(f"  Predicted harmful: {sum(preds)} ({100*sum(preds)/len(preds):.1f}%)")
    print(f"  Predicted benign: {len(preds)-sum(preds)} ({100*(len(preds)-sum(preds))/len(preds):.1f}%)")
    
    # 按语言统计
    print(f"\n--- By Language ---")
    unique_langs = sorted(set(all_languages))
    
    ood_by_language = {}
    for lang in unique_langs:
        lang_indices = [i for i, l in enumerate(all_languages) if l == lang]
        lang_preds = preds[lang_indices]
        lang_probs = probs[lang_indices]
        
        detection_rate = sum(lang_preds) / len(lang_preds)
        avg_prob = np.mean(lang_probs)
        
        print(f"  {lang}: {len(lang_indices)} samples, "
              f"Detection Rate: {detection_rate:.4f}, "
              f"Avg P(Harmful): {avg_prob:.4f}")
        
        ood_by_language[lang] = {
            'count': len(lang_indices),
            'detection_rate': detection_rate,
            'avg_harmful_prob': float(avg_prob),
            'min_harmful_prob': float(np.min(lang_probs)),
            'max_harmful_prob': float(np.max(lang_probs))
        }
    
    # 整体指标
    overall_detection_rate = sum(preds) / len(preds)
    avg_prob_overall = np.mean(probs)
    
    results = {
        'total_samples': len(all_safety_labels),
        'overall_detection_rate': float(overall_detection_rate),
        'avg_harmful_probability': float(avg_prob_overall),
        'min_harmful_probability': float(np.min(probs)),
        'max_harmful_probability': float(np.max(probs)),
        'by_language': ood_by_language
    }
    
    # 可视化OOD检测
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. 概率分布
    ax = axes[0]
    ax.hist(probs, bins=50, alpha=0.7, color='red', edgecolor='darkred')
    ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold=0.5')
    ax.axvline(x=avg_prob_overall, color='blue', linestyle='-', linewidth=2, 
               label=f'Mean={avg_prob_overall:.3f}')
    ax.set_xlabel('Predicted P(Harmful)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('OOD Harmful Probability Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 按语言的检测率
    ax = axes[1]
    langs = list(ood_by_language.keys())
    detection_rates = [ood_by_language[l]['detection_rate'] for l in langs]
    x = np.arange(len(langs))
    bars = ax.bar(x, detection_rates, color='crimson', alpha=0.8)
    ax.axhline(y=overall_detection_rate, color='blue', linestyle='--', 
               label=f'Overall: {overall_detection_rate:.3f}')
    ax.set_ylabel('Detection Rate (Recall)', fontsize=12)
    ax.set_title('OOD Detection Rate by Language', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(langs)
    ax.legend()
    ax.set_ylim(0, 1.1)
    for bar, v in zip(bars, detection_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ood_evaluation.png'), dpi=300)
    plt.close()
    
    # 保存结果
    with open(os.path.join(output_dir, 'ood_evaluation_results.json'), 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)
    
    print(f"\nOOD results saved to {os.path.join(output_dir, 'ood_evaluation_results.json')}")
    print(f"OOD visualization saved to {os.path.join(output_dir, 'ood_evaluation.png')}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Classifier Model")
    
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained classifier model checkpoint")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to evaluation dataset")
    parser.add_argument("--eval_languages", type=str, nargs="+", default=None,
                        help="Only evaluate on these languages")
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                        help="Output directory")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device")
    parser.add_argument("--no_visualize", action="store_true",
                        help="Skip visualization")
    
    # 模型架构参数（需要与训练时一致）
    parser.add_argument("--hidden_dim", type=int, default=4096,
                        help="Hidden dimension")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Number of MLP layers")
    parser.add_argument("--use_residual", action="store_true",
                        help="Use residual connection")
    parser.add_argument("--classifier_hidden_dim", type=int, default=512,
                        help="Classifier hidden dimension")
    parser.add_argument("--identity_mlp", action="store_true",
                        help="Use identity mapping for MLP (output = input)")
    
    # OOD测试参数
    parser.add_argument("--ood_data_path", type=str, default=None,
                        help="Path to OOD test data")
    parser.add_argument("--llm_model_path", type=str, default=None,
                        help="Path to LLM for extracting hidden states")
    parser.add_argument("--layer_idx", type=int, default=20,
                        help="Layer index for hidden state extraction")
    parser.add_argument("--ood_max_samples", type=int, default=None,
                        help="Maximum OOD samples to use")
    
    # 问题聚类可视化参数
    parser.add_argument("--n_safe_questions", type=int, default=5,
                        help="Number of safe questions for clustering visualization")
    parser.add_argument("--n_unsafe_questions", type=int, default=5,
                        help="Number of unsafe questions for clustering visualization")
    parser.add_argument("--max_variants_per_question", type=int, default=10,
                        help="Max number of variants (languages) per question for clustering")
    
    args = parser.parse_args()
    
    # 加载数据
    print(f"Loading dataset from {args.data_path}...")
    dataset = load_dataset(args.data_path)
    print(f"Dataset size: {len(dataset)}")
    
    # 可选：只在指定语言上评估
    if args.eval_languages:
        allowed = set(args.eval_languages)
        keep_indices = [i for i, lang in enumerate(dataset.languages) if lang in allowed]
        if len(keep_indices) == 0:
            raise ValueError(f"No samples found for languages: {allowed}")
        
        def _subset(seq):
            return [seq[i] for i in keep_indices]
        
        dataset = MultilingualHiddenStateDataset(
            hidden_states=_subset(dataset.hidden_states),
            group_ids=_subset(dataset.group_ids),
            safety_labels=_subset(dataset.safety_labels),
            languages=_subset(dataset.languages),
            texts=_subset(dataset.texts),
        )
        print(f"Filtered to languages {sorted(allowed)}, remaining samples: {len(dataset)}")
    
    # 获取输入维度
    input_dim = dataset.hidden_states[0].shape[0]
    print(f"Input dimension: {input_dim}")
    
    # 尝试从checkpoint目录读取config.json来获取identity_mlp设置
    identity_mlp = args.identity_mlp
    checkpoint_dir = os.path.dirname(args.model_path)
    config_path = os.path.join(checkpoint_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            if 'identity_mlp' in config:
                identity_mlp = config['identity_mlp']
                print(f"Found identity_mlp={identity_mlp} in config.json")
    
    # 创建模型
    model = TransformMLPWithClassifier(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=input_dim,
        num_layers=args.num_layers,
        use_residual=args.use_residual,
        use_layer_norm=True,
        classifier_hidden_dim=args.classifier_hidden_dim,
        identity_mlp=identity_mlp
    ).to(args.device)
    
    # 加载模型权重
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully")
    if identity_mlp:
        print(">>> Using identity MLP (output = input)")
    
    if 'val_metrics' in checkpoint:
        print(f"Checkpoint metrics: F1={checkpoint['val_metrics'].get('f1', 'N/A')}, "
              f"AUC={checkpoint['val_metrics'].get('auc_roc', 'N/A')}")
    
    # 评估
    results = evaluate(
        model=model,
        dataset=dataset,
        device=args.device,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        visualize=not args.no_visualize,
        n_safe_questions=args.n_safe_questions,
        n_unsafe_questions=args.n_unsafe_questions,
        max_variants_per_question=args.max_variants_per_question
    )
    
    # OOD评估
    if args.ood_data_path and args.llm_model_path:
        print("\n" + "=" * 60)
        print("Running OOD Evaluation")
        print("=" * 60)
        
        ood_dataset = prepare_ood_dataset(
            ood_data_path=args.ood_data_path,
            model_name_or_path=args.llm_model_path,
            layer_idx=args.layer_idx,
            device=args.device,
            batch_size=8,
            max_samples=args.ood_max_samples
        )
        
        ood_results = evaluate_ood(
            model=model,
            ood_dataset=ood_dataset,
            device=args.device,
            output_dir=args.output_dir,
            batch_size=args.batch_size
        )
    elif args.ood_data_path:
        print("\nWarning: --ood_data_path provided but --llm_model_path is missing. "
              "Skipping OOD evaluation.")
    
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()


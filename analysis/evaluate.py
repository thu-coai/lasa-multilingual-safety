"""
评估脚本

评估训练好的Transform MLP模型
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
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from model import TransformMLP
from dataset import load_dataset, collate_fn, load_safety_data, extract_hidden_states, MultilingualHiddenStateDataset
from transformers import AutoTokenizer, AutoModelForCausalLM


def convert_to_serializable(obj):
    """递归地将numpy类型转换为Python原生类型，以便JSON序列化"""
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


def compute_detailed_metrics(
    embeddings: torch.Tensor,
    group_ids: list,
    safety_labels: list,
    languages: list
) -> dict:
    """
    计算详细的评估指标
    
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
    
    # 不同组之间的相似度（随机采样）
    all_indices = list(range(len(embeddings)))
    inter_group_sims = []
    num_samples = min(10000, len(all_indices) * (len(all_indices) - 1) // 2)
    
    for _ in range(num_samples):
        i, j = np.random.choice(all_indices, 2, replace=False)
        if group_ids[i] != group_ids[j]:
            sim = torch.dot(embeddings[i], embeddings[j]).item()
            inter_group_sims.append(sim)
    
    metrics['inter_group_similarity'] = np.mean(inter_group_sims) if inter_group_sims else 0.0
    metrics['alignment_gap'] = metrics['intra_group_similarity'] - metrics['inter_group_similarity']
    
    # ========== 2. 安全性区分指标 ==========
    safe_indices = [i for i, l in enumerate(safety_labels) if l == 0]
    unsafe_indices = [i for i, l in enumerate(safety_labels) if l == 1]
    
    if safe_indices and unsafe_indices:
        safe_emb = embeddings[safe_indices]
        unsafe_emb = embeddings[unsafe_indices]
        
        # Safe-Unsafe相似度
        cross_sim = torch.matmul(safe_emb, unsafe_emb.T)
        metrics['safe_unsafe_similarity'] = cross_sim.mean().item()
        metrics['safe_unsafe_std'] = cross_sim.std().item()
        
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
    else:
        metrics['safe_unsafe_similarity'] = 0.0
        metrics['safe_safe_similarity'] = 0.0
        metrics['unsafe_unsafe_similarity'] = 0.0
        metrics['safety_separation_gap'] = 0.0
    
    # ========== 3. 按语言的对齐指标（分别统计safe和unsafe） ==========
    metrics['by_language'] = {}
    unique_langs = sorted(set(languages))
    
    for lang in unique_langs:
        # 该语言的所有样本
        lang_indices = [i for i, l in enumerate(languages) if l == lang]
        
        # 分别获取safe和unsafe的索引
        lang_safe_indices = [i for i in lang_indices if safety_labels[i] == 0]
        lang_unsafe_indices = [i for i in lang_indices if safety_labels[i] == 1]
        
        lang_stats = {
            'count': len(lang_indices),
            'safe_count': len(lang_safe_indices),
            'unsafe_count': len(lang_unsafe_indices)
        }
        
        # 计算该语言内所有样本的相似度
        if len(lang_indices) >= 2:
            lang_emb = embeddings[lang_indices]
            lang_sim = torch.matmul(lang_emb, lang_emb.T)
            mask = ~torch.eye(len(lang_indices), dtype=torch.bool)
            lang_stats['mean_similarity'] = lang_sim[mask].mean().item()
            lang_stats['std_similarity'] = lang_sim[mask].std().item()
        else:
            lang_stats['mean_similarity'] = 0.0
            lang_stats['std_similarity'] = 0.0
        
        # 计算该语言内 safe 样本的相似度
        if len(lang_safe_indices) >= 2:
            safe_emb = embeddings[lang_safe_indices]
            safe_sim = torch.matmul(safe_emb, safe_emb.T)
            mask = ~torch.eye(len(lang_safe_indices), dtype=torch.bool)
            lang_stats['safe_mean_similarity'] = safe_sim[mask].mean().item()
            lang_stats['safe_std_similarity'] = safe_sim[mask].std().item()
        else:
            lang_stats['safe_mean_similarity'] = 0.0
            lang_stats['safe_std_similarity'] = 0.0
        
        # 计算该语言内 unsafe 样本的相似度
        if len(lang_unsafe_indices) >= 2:
            unsafe_emb = embeddings[lang_unsafe_indices]
            unsafe_sim = torch.matmul(unsafe_emb, unsafe_emb.T)
            mask = ~torch.eye(len(lang_unsafe_indices), dtype=torch.bool)
            lang_stats['unsafe_mean_similarity'] = unsafe_sim[mask].mean().item()
            lang_stats['unsafe_std_similarity'] = unsafe_sim[mask].std().item()
        else:
            lang_stats['unsafe_mean_similarity'] = 0.0
            lang_stats['unsafe_std_similarity'] = 0.0
        
        # 计算该语言内 safe-unsafe 之间的相似度
        if len(lang_safe_indices) >= 1 and len(lang_unsafe_indices) >= 1:
            safe_emb = embeddings[lang_safe_indices]
            unsafe_emb = embeddings[lang_unsafe_indices]
            cross_sim = torch.matmul(safe_emb, unsafe_emb.T)
            lang_stats['safe_unsafe_similarity'] = cross_sim.mean().item()
        else:
            lang_stats['safe_unsafe_similarity'] = 0.0
        
        metrics['by_language'][lang] = lang_stats
    
    # ========== 4. 聚类质量指标 ==========
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
    else:
        metrics['safety_silhouette_score'] = 0.0
    
    return metrics


def visualize_embeddings(
    embeddings: torch.Tensor,
    safety_labels: list,
    languages: list,
    output_dir: str,
    prefix: str = ""
):
    """
    可视化embeddings
    """
    print("Computing t-SNE projection...")
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings.numpy())
    
    # 1. 按安全性标签着色
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    ax = axes[0]
    colors = ['blue' if l == 0 else 'red' for l in safety_labels]
    ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, alpha=0.5, s=10)
    ax.set_title('Embeddings by Safety Label\n(Blue=Safe, Red=Unsafe)', fontsize=14)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    
    # 2. 按语言着色
    ax = axes[1]
    unique_langs = sorted(set(languages))
    color_map = plt.cm.tab10(np.linspace(0, 1, len(unique_langs)))
    lang_to_color = {lang: color_map[i] for i, lang in enumerate(unique_langs)}
    colors = [lang_to_color[lang] for lang in languages]
    
    scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, alpha=0.5, s=10)
    ax.set_title('Embeddings by Language', fontsize=14)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    
    # 添加图例
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=lang_to_color[lang], 
                          markersize=10, label=lang) for lang in unique_langs]
    ax.legend(handles=handles, loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}embeddings_tsne.png'), dpi=300)
    plt.close()
    
    print(f"t-SNE visualization saved to {os.path.join(output_dir, f'{prefix}embeddings_tsne.png')}")
    
    # 3. 相似度热力图（按语言对，分别统计safe和unsafe）
    print("Computing language pair similarity heatmaps (safe/unsafe separated)...")
    
    # 分别收集safe和unsafe的embeddings
    lang_embeddings_safe = defaultdict(list)
    lang_embeddings_unsafe = defaultdict(list)
    lang_embeddings_all = defaultdict(list)
    
    for i, lang in enumerate(languages):
        lang_embeddings_all[lang].append(embeddings[i])
        if safety_labels[i] == 0:
            lang_embeddings_safe[lang].append(embeddings[i])
        else:
            lang_embeddings_unsafe[lang].append(embeddings[i])
    
    unique_langs = sorted(lang_embeddings_all.keys())
    
    # 创建3个子图：All, Safe, Unsafe
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    
    def compute_lang_sim_matrix(lang_embeddings_dict, unique_langs):
        """计算语言对之间的相似度矩阵"""
        sim_matrix = np.zeros((len(unique_langs), len(unique_langs)))
        for i, lang1 in enumerate(unique_langs):
            for j, lang2 in enumerate(unique_langs):
                if lang1 in lang_embeddings_dict and lang2 in lang_embeddings_dict:
                    if len(lang_embeddings_dict[lang1]) > 0 and len(lang_embeddings_dict[lang2]) > 0:
                        emb1 = torch.stack(lang_embeddings_dict[lang1])
                        emb2 = torch.stack(lang_embeddings_dict[lang2])
                        emb1 = F.normalize(emb1, p=2, dim=1)
                        emb2 = F.normalize(emb2, p=2, dim=1)
                        sim = torch.matmul(emb1, emb2.T).mean().item()
                        sim_matrix[i, j] = sim
        return sim_matrix
    
    # All samples
    sim_matrix_all = compute_lang_sim_matrix(lang_embeddings_all, unique_langs)
    sns.heatmap(sim_matrix_all, annot=True, fmt='.3f', cmap='coolwarm',
                xticklabels=unique_langs, yticklabels=unique_langs,
                vmin=-1, vmax=1, ax=axes[0])
    axes[0].set_title('All Samples', fontsize=14, fontweight='bold')
    
    # Safe samples
    sim_matrix_safe = compute_lang_sim_matrix(lang_embeddings_safe, unique_langs)
    sns.heatmap(sim_matrix_safe, annot=True, fmt='.3f', cmap='coolwarm',
                xticklabels=unique_langs, yticklabels=unique_langs,
                vmin=-1, vmax=1, ax=axes[1])
    axes[1].set_title('Safe Samples Only', fontsize=14, fontweight='bold')
    
    # Unsafe samples
    sim_matrix_unsafe = compute_lang_sim_matrix(lang_embeddings_unsafe, unique_langs)
    sns.heatmap(sim_matrix_unsafe, annot=True, fmt='.3f', cmap='coolwarm',
                xticklabels=unique_langs, yticklabels=unique_langs,
                vmin=-1, vmax=1, ax=axes[2])
    axes[2].set_title('Unsafe Samples Only', fontsize=14, fontweight='bold')
    
    plt.suptitle('Average Similarity Between Languages', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}language_similarity_heatmap.png'), dpi=300)
    plt.close()
    
    print(f"Heatmap saved to {os.path.join(output_dir, f'{prefix}language_similarity_heatmap.png')}")


def evaluate(
    model: TransformMLP,
    dataset,
    device: str,
    output_dir: str,
    batch_size: int = 64,
    visualize: bool = True
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
    
    # 收集所有embeddings
    all_original = []
    all_transformed = []
    all_group_ids = []
    all_safety_labels = []
    all_languages = []
    
    with torch.no_grad():
        for batch in dataloader:
            hidden_states = batch['hidden_states'].to(device)
            
            transformed = model(hidden_states)
            
            all_original.append(hidden_states.cpu())
            all_transformed.append(transformed.cpu())
            all_group_ids.extend(batch['group_ids'].tolist())
            all_safety_labels.extend(batch['safety_labels'].tolist())
            all_languages.extend(batch['languages'])
    
    all_original = torch.cat(all_original, dim=0)
    all_transformed = torch.cat(all_transformed, dim=0)
    
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    
    # 评估原始表示
    print("\n--- Original Hidden States ---")
    original_metrics = compute_detailed_metrics(
        all_original, all_group_ids, all_safety_labels, all_languages
    )
    
    for key, value in original_metrics.items():
        if key != 'by_language':
            print(f"  {key}: {value:.4f}")
    
    # 评估变换后的表示
    print("\n--- Transformed Embeddings ---")
    transformed_metrics = compute_detailed_metrics(
        all_transformed, all_group_ids, all_safety_labels, all_languages
    )
    
    for key, value in transformed_metrics.items():
        if key != 'by_language':
            print(f"  {key}: {value:.4f}")
    
    # 计算提升
    print("\n--- Improvement ---")
    for key in ['intra_group_similarity', 'alignment_gap', 'safety_separation_gap', 'safety_silhouette_score']:
        if key in original_metrics and key in transformed_metrics:
            improvement = transformed_metrics[key] - original_metrics[key]
            print(f"  {key}: {improvement:+.4f}")
    
    # 保存结果
    results = {
        'original': {k: v for k, v in original_metrics.items() if k != 'by_language'},
        'transformed': {k: v for k, v in transformed_metrics.items() if k != 'by_language'},
        'by_language_original': original_metrics.get('by_language', {}),
        'by_language_transformed': transformed_metrics.get('by_language', {})
    }
    
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)
    
    print(f"\nResults saved to {os.path.join(output_dir, 'evaluation_results.json')}")
    
    # 可视化
    if visualize:
        print("\nGenerating visualizations...")
        
        # 原始表示可视化
        visualize_embeddings(
            all_original, all_safety_labels, all_languages,
            output_dir, prefix="original_"
        )
        
        # 变换后表示可视化
        visualize_embeddings(
            all_transformed, all_safety_labels, all_languages,
            output_dir, prefix="transformed_"
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
    
    Args:
        ood_data_path: OOD数据路径 (如 harmbench_translated.json)
        model_name_or_path: 用于提取hidden states的模型路径
        layer_idx: 提取的层
        device: 设备
        batch_size: 批大小
        max_samples: 最大样本数
    
    Returns:
        MultilingualHiddenStateDataset
    """
    print(f"Loading OOD data from {ood_data_path}...")
    
    # 加载OOD数据（使用与safety数据相同的格式）
    ood_data = load_safety_data(ood_data_path, max_samples=max_samples)
    print(f"Loaded {len(ood_data)} OOD samples")
    
    # 加载模型
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
    
    # 提取hidden states
    texts = [item['prompt'] for item in ood_data]
    hidden_states = extract_hidden_states(
        model, tokenizer, texts,
        layer_idx=layer_idx,
        device=device,
        batch_size=batch_size
    )
    
    # 组织数据 - OOD数据全部标记为unsafe (safety_label=1)
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
    model: TransformMLP,
    train_dataset,
    ood_dataset,
    device: str,
    output_dir: str,
    batch_size: int = 64
):
    """
    评估OOD测试集
    
    检验：
    1. OOD数据与训练集中safe数据的分离度
    2. OOD数据与训练集中unsafe数据的相似度
    3. OOD数据的跨语言对齐
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    
    # 收集训练集的embeddings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    train_original = []
    train_transformed = []
    train_safety_labels = []
    
    with torch.no_grad():
        for batch in train_loader:
            hidden_states = batch['hidden_states'].to(device)
            transformed = model(hidden_states)
            train_original.append(hidden_states.cpu())
            train_transformed.append(transformed.cpu())
            train_safety_labels.extend(batch['safety_labels'].tolist())
    
    train_original = torch.cat(train_original, dim=0)
    train_transformed = torch.cat(train_transformed, dim=0)
    
    # 收集OOD的embeddings
    ood_loader = DataLoader(
        ood_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    ood_original = []
    ood_transformed = []
    ood_group_ids = []
    ood_languages = []
    
    with torch.no_grad():
        for batch in ood_loader:
            hidden_states = batch['hidden_states'].to(device)
            transformed = model(hidden_states)
            ood_original.append(hidden_states.cpu())
            ood_transformed.append(transformed.cpu())
            ood_group_ids.extend(batch['group_ids'].tolist())
            ood_languages.extend(batch['languages'])
    
    ood_original = torch.cat(ood_original, dim=0)
    ood_transformed = torch.cat(ood_transformed, dim=0)
    
    # 归一化
    train_original_norm = F.normalize(train_original, p=2, dim=1)
    train_transformed_norm = F.normalize(train_transformed, p=2, dim=1)
    ood_original_norm = F.normalize(ood_original, p=2, dim=1)
    ood_transformed_norm = F.normalize(ood_transformed, p=2, dim=1)
    
    # 分离训练集的safe和unsafe
    train_safe_indices = [i for i, l in enumerate(train_safety_labels) if l == 0]
    train_unsafe_indices = [i for i, l in enumerate(train_safety_labels) if l == 1]
    
    results = {'original': {}, 'transformed': {}}
    
    for name, train_emb, ood_emb in [
        ('original', train_original_norm, ood_original_norm),
        ('transformed', train_transformed_norm, ood_transformed_norm)
    ]:
        train_safe_emb = train_emb[train_safe_indices]
        train_unsafe_emb = train_emb[train_unsafe_indices]
        
        # 1. OOD与训练集safe的相似度
        ood_safe_sim = torch.matmul(ood_emb, train_safe_emb.T)
        results[name]['ood_train_safe_similarity'] = ood_safe_sim.mean().item()
        
        # 2. OOD与训练集unsafe的相似度
        ood_unsafe_sim = torch.matmul(ood_emb, train_unsafe_emb.T)
        results[name]['ood_train_unsafe_similarity'] = ood_unsafe_sim.mean().item()
        
        # 3. 分离度：OOD应该与unsafe更相似，与safe更不相似
        results[name]['ood_safety_gap'] = (
            results[name]['ood_train_unsafe_similarity'] - 
            results[name]['ood_train_safe_similarity']
        )
        
        # 4. OOD内部的跨语言对齐
        group_to_indices = defaultdict(list)
        for i, gid in enumerate(ood_group_ids):
            group_to_indices[gid].append(i)
        
        intra_group_sims = []
        for gid, indices in group_to_indices.items():
            if len(indices) < 2:
                continue
            group_emb = ood_emb[indices]
            sim_matrix = torch.matmul(group_emb, group_emb.T)
            mask = ~torch.eye(len(indices), dtype=torch.bool)
            intra_group_sims.extend(sim_matrix[mask].tolist())
        
        results[name]['ood_intra_group_similarity'] = np.mean(intra_group_sims) if intra_group_sims else 0.0
        
        # 5. 按语言统计
        results[name]['ood_by_language'] = {}
        for lang in sorted(set(ood_languages)):
            lang_indices = [i for i, l in enumerate(ood_languages) if l == lang]
            if len(lang_indices) >= 2:
                lang_emb = ood_emb[lang_indices]
                lang_sim = torch.matmul(lang_emb, lang_emb.T)
                mask = ~torch.eye(len(lang_indices), dtype=torch.bool)
                
                # 与训练集safe的相似度
                lang_safe_sim = torch.matmul(lang_emb, train_safe_emb.T).mean().item()
                # 与训练集unsafe的相似度  
                lang_unsafe_sim = torch.matmul(lang_emb, train_unsafe_emb.T).mean().item()
                
                results[name]['ood_by_language'][lang] = {
                    'count': len(lang_indices),
                    'mean_similarity': lang_sim[mask].mean().item(),
                    'train_safe_similarity': lang_safe_sim,
                    'train_unsafe_similarity': lang_unsafe_sim,
                    'safety_gap': lang_unsafe_sim - lang_safe_sim
                }
    
    # 打印结果
    print("\n" + "=" * 60)
    print("OOD Evaluation Results")
    print("=" * 60)
    
    print("\n--- Original Hidden States ---")
    for key, value in results['original'].items():
        if key != 'ood_by_language':
            print(f"  {key}: {value:.4f}")
    
    print("\n--- Transformed Embeddings ---")
    for key, value in results['transformed'].items():
        if key != 'ood_by_language':
            print(f"  {key}: {value:.4f}")
    
    print("\n--- Improvement ---")
    print(f"  ood_safety_gap: {results['transformed']['ood_safety_gap'] - results['original']['ood_safety_gap']:+.4f}")
    print(f"  ood_intra_group_similarity: {results['transformed']['ood_intra_group_similarity'] - results['original']['ood_intra_group_similarity']:+.4f}")
    
    # 保存结果
    with open(os.path.join(output_dir, 'ood_evaluation_results.json'), 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)
    
    print(f"\nOOD results saved to {os.path.join(output_dir, 'ood_evaluation_results.json')}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Transform MLP")
    
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to evaluation dataset")
    parser.add_argument("--eval_languages", type=str, nargs="+", default=None,
                        help="Only evaluate on these languages (e.g., ar vi th)")
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                        help="Output directory")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device")
    parser.add_argument("--no_visualize", action="store_true",
                        help="Skip visualization")
    
    # OOD测试参数
    parser.add_argument("--ood_data_path", type=str, default=None,
                        help="Path to OOD test data (e.g., harmbench_translated.json)")
    parser.add_argument("--llm_model_path", type=str, default=None,
                        help="Path to LLM for extracting hidden states (required for OOD test)")
    parser.add_argument("--layer_idx", type=int, default=20,
                        help="Layer index for hidden state extraction")
    parser.add_argument("--ood_max_samples", type=int, default=None,
                        help="Maximum OOD samples to use")
    
    args = parser.parse_args()
    
    # 加载数据
    print(f"Loading dataset from {args.data_path}...")
    dataset = load_dataset(args.data_path)
    print(f"Dataset size: {len(dataset)}")
    
    # 可选：只在指定语言上评估，用于测试跨语言泛化
    if args.eval_languages:
        allowed = set(args.eval_languages)
        keep_indices = [i for i, lang in enumerate(dataset.languages) if lang in allowed]
        if len(keep_indices) == 0:
            raise ValueError(f"No evaluation samples found for languages: {allowed}")

        def _subset(seq):
            return [seq[i] for i in keep_indices]

        dataset = MultilingualHiddenStateDataset(
            hidden_states=_subset(dataset.hidden_states),
            group_ids=_subset(dataset.group_ids),
            safety_labels=_subset(dataset.safety_labels),
            languages=_subset(dataset.languages),
            texts=_subset(dataset.texts),
        )
        print(f"Filtered eval languages to {sorted(allowed)}, remaining samples: {len(dataset)}")
    
    # 获取输入维度
    input_dim = dataset.hidden_states[0].shape[0]
    print(f"Input dimension: {input_dim}")
    
    # 加载模型
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=args.device, weights_only=False)
    
    # 从checkpoint推断模型配置
    state_dict = checkpoint['model_state_dict']
    
    # 尝试推断模型结构
    model = TransformMLP(
        input_dim=input_dim,
        hidden_dim=input_dim,
        output_dim=input_dim,
        num_layers=2,
        use_residual=True,
        use_layer_norm=True
    ).to(args.device)
    
    model.load_state_dict(state_dict)
    
    print("Model loaded successfully")
    
    # 评估
    results = evaluate(
        model=model,
        dataset=dataset,
        device=args.device,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        visualize=not args.no_visualize
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
            train_dataset=dataset,
            ood_dataset=ood_dataset,
            device=args.device,
            output_dir=args.output_dir,
            batch_size=args.batch_size
        )
    elif args.ood_data_path:
        print("\nWarning: --ood_data_path provided but --llm_model_path is missing. Skipping OOD evaluation.")
    
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()


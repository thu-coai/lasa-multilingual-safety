"""
LASA semantic bottleneck analysis.

This script tests the paper's representation-level hypothesis: early/late
layers often preserve language-specific structure, while an intermediate
semantic bottleneck clusters multilingual prompts by shared query semantics.
It computes language-vs-query silhouette scores and can render t-SNE plots.

Legacy experiment note:
层级聚类分析实验

验证假设：
1. 第1层和最后1层：hidden states按语言聚类
2. 中间层（如第15层）：hidden states按问题（query id）聚类

通过t-SNE可视化和聚类指标（silhouette score）来验证这一假设
"""

import os
import argparse
import json
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

from transformers import AutoTokenizer, AutoModelForCausalLM

# 尝试导入本地模块
try:
    from dataset import load_ultrafeedback_data, load_safety_data
except ImportError:
    # 如果直接运行，添加当前目录到路径
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from dataset import load_ultrafeedback_data, load_safety_data


def extract_hidden_states_multi_layer(
    model,
    tokenizer,
    texts: List[str],
    layer_indices: List[int],
    device: str = 'cuda',
    batch_size: int = 8,
    max_length: int = 512,
    token_position: int = -1
) -> Dict[int, List[torch.Tensor]]:
    """
    批量提取多个层的hidden states
    
    Args:
        model: 预训练模型
        tokenizer: tokenizer
        texts: 文本列表
        layer_indices: 要提取的层索引列表
        device: 设备
        batch_size: 批大小
        max_length: 最大序列长度
        token_position: 使用哪个位置的token，-1表示最后一个，-2表示倒数第二个
    
    Returns:
        字典: {layer_idx: [hidden_states]}
    """
    model.eval()
    hidden_states_dict = {layer_idx: [] for layer_idx in layer_indices}
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting hidden states"):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Forward
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        # 获取每个指定层的hidden states
        for layer_idx in layer_indices:
            # 安全检查：确保层索引在有效范围内
            if layer_idx < 0 or layer_idx >= len(outputs.hidden_states):
                raise IndexError(
                    f"Layer index {layer_idx} is out of range. "
                    f"Model has {len(outputs.hidden_states) - 1} transformer layers "
                    f"(hidden_states indices: 0 to {len(outputs.hidden_states) - 1})."
                )
            layer_output = outputs.hidden_states[layer_idx]
            
            # 获取每个样本指定位置的hidden state
            for j in range(len(batch_texts)):
                attention_mask = inputs['attention_mask'][j]
                seq_len = attention_mask.sum().item()
                
                # 根据token_position计算实际索引
                # token_position=-1 表示最后一个token
                # token_position=-2 表示倒数第二个token
                token_idx = max(0, seq_len + token_position)
                
                hidden_state = layer_output[j, token_idx, :].cpu()
                hidden_states_dict[layer_idx].append(hidden_state)
    
    return hidden_states_dict


def compute_clustering_metrics(
    embeddings: torch.Tensor,
    language_labels: List[str],
    query_ids: List[int]
) -> Dict:
    """
    计算聚类指标
    
    Args:
        embeddings: hidden states, shape [N, dim]
        language_labels: 语言标签列表
        query_ids: 问题ID列表
    
    Returns:
        指标字典
    """
    # 转换为float32（模型输出可能是bfloat16，numpy不支持）
    embeddings = embeddings.float()
    
    # L2归一化
    embeddings_np = F.normalize(embeddings, p=2, dim=1).numpy()
    
    metrics = {}
    
    # 将语言标签转换为数值
    unique_langs = sorted(set(language_labels))
    lang_to_idx = {lang: i for i, lang in enumerate(unique_langs)}
    lang_labels_numeric = [lang_to_idx[lang] for lang in language_labels]
    
    # 将query_id转换为数值（因为可能不连续）
    unique_queries = sorted(set(query_ids))
    query_to_idx = {qid: i for i, qid in enumerate(unique_queries)}
    query_labels_numeric = [query_to_idx[qid] for qid in query_ids]
    
    # ========== 1. Silhouette Score ==========
    # 按语言聚类的silhouette score
    if len(unique_langs) > 1:
        try:
            metrics['silhouette_language'] = silhouette_score(
                embeddings_np, lang_labels_numeric, metric='cosine'
            )
        except:
            metrics['silhouette_language'] = 0.0
    else:
        metrics['silhouette_language'] = 0.0
    
    # 按问题聚类的silhouette score（只采样一部分，否则太慢）
    if len(unique_queries) > 1:
        try:
            # 如果query太多，采样
            if len(unique_queries) > 100:
                # 只保留有多个语言版本的query
                query_counts = defaultdict(int)
                for qid in query_ids:
                    query_counts[qid] += 1
                valid_queries = {qid for qid, cnt in query_counts.items() if cnt >= 2}
                
                if len(valid_queries) > 100:
                    valid_queries = set(list(valid_queries)[:100])
                
                valid_indices = [i for i, qid in enumerate(query_ids) if qid in valid_queries]
                
                if len(valid_indices) > 10:
                    sampled_embeddings = embeddings_np[valid_indices]
                    sampled_query_labels = [query_labels_numeric[i] for i in valid_indices]
                    metrics['silhouette_query'] = silhouette_score(
                        sampled_embeddings, sampled_query_labels, metric='cosine'
                    )
                else:
                    metrics['silhouette_query'] = 0.0
            else:
                metrics['silhouette_query'] = silhouette_score(
                    embeddings_np, query_labels_numeric, metric='cosine'
                )
        except Exception as e:
            print(f"Warning: silhouette_query computation failed: {e}")
            metrics['silhouette_query'] = 0.0
    else:
        metrics['silhouette_query'] = 0.0
    
    # ========== 2. 同语言内相似度 vs 跨语言相似度 ==========
    embeddings_tensor = torch.tensor(embeddings_np, dtype=torch.float32)
    
    intra_lang_sims = []
    inter_lang_sims = []
    
    # 按语言分组
    lang_to_indices = defaultdict(list)
    for i, lang in enumerate(language_labels):
        lang_to_indices[lang].append(i)
    
    # 计算同语言相似度
    for lang, indices in lang_to_indices.items():
        if len(indices) >= 2:
            lang_emb = embeddings_tensor[indices]
            sim_matrix = torch.matmul(lang_emb, lang_emb.T)
            mask = ~torch.eye(len(indices), dtype=torch.bool)
            intra_lang_sims.extend(sim_matrix[mask].tolist())
    
    # 计算跨语言相似度（采样）
    all_indices = list(range(len(embeddings)))
    num_samples = min(10000, len(all_indices) * (len(all_indices) - 1) // 2)
    
    for _ in range(num_samples):
        i, j = np.random.choice(all_indices, 2, replace=False)
        if language_labels[i] != language_labels[j]:
            sim = torch.dot(embeddings_tensor[i], embeddings_tensor[j]).item()
            inter_lang_sims.append(sim)
    
    metrics['intra_language_similarity'] = np.mean(intra_lang_sims) if intra_lang_sims else 0.0
    metrics['inter_language_similarity'] = np.mean(inter_lang_sims) if inter_lang_sims else 0.0
    metrics['language_clustering_gap'] = metrics['intra_language_similarity'] - metrics['inter_language_similarity']
    
    # ========== 3. 同问题内相似度 vs 跨问题相似度 ==========
    query_to_indices = defaultdict(list)
    for i, qid in enumerate(query_ids):
        query_to_indices[qid].append(i)
    
    # 计算同问题相似度（不同语言版本）
    intra_query_sims = []
    for qid, indices in query_to_indices.items():
        if len(indices) >= 2:
            query_emb = embeddings_tensor[indices]
            sim_matrix = torch.matmul(query_emb, query_emb.T)
            mask = ~torch.eye(len(indices), dtype=torch.bool)
            intra_query_sims.extend(sim_matrix[mask].tolist())
    
    # 计算跨问题相似度
    inter_query_sims = []
    for _ in range(num_samples):
        i, j = np.random.choice(all_indices, 2, replace=False)
        if query_ids[i] != query_ids[j]:
            sim = torch.dot(embeddings_tensor[i], embeddings_tensor[j]).item()
            inter_query_sims.append(sim)
    
    metrics['intra_query_similarity'] = np.mean(intra_query_sims) if intra_query_sims else 0.0
    metrics['inter_query_similarity'] = np.mean(inter_query_sims) if inter_query_sims else 0.0
    metrics['query_clustering_gap'] = metrics['intra_query_similarity'] - metrics['inter_query_similarity']
    
    # ========== 4. 比较两种聚类哪个更强 ==========
    # 正值表示语言聚类更强，负值表示问题聚类更强
    metrics['language_vs_query_clustering'] = (
        metrics['language_clustering_gap'] - metrics['query_clustering_gap']
    )
    
    return metrics


def visualize_layer_embeddings(
    embeddings: torch.Tensor,
    language_labels: List[str],
    query_ids: List[int],
    layer_idx: int,
    output_dir: str,
    max_samples: int = 2000,
    perplexity: int = 30
):
    """
    可视化单层的embeddings（保留用于兼容性，主要使用 visualize_all_layers_tsne）
    """
    # 这个函数现在主要作为辅助，核心可视化由 visualize_all_layers_tsne 完成
    pass


def visualize_all_layers_tsne(
    hidden_states_dict: Dict[int, List[torch.Tensor]],
    language_labels: List[str],
    query_ids: List[int],
    layer_indices: List[int],
    output_dir: str,
    perplexity: int = 30,
    all_metrics: Dict = None
):
    """
    创建综合的多层t-SNE可视化图（学术论文风格 - ACL标准）
    
    生成：
    1. 2×N 的t-SNE图（第一行按语言，第二行按Query）
    2. 单独的折线图文件
    
    Args:
        hidden_states_dict: 每层的hidden states字典
        language_labels: 语言标签列表
        query_ids: 问题ID列表
        layer_indices: 层索引列表
        output_dir: 输出目录
        perplexity: t-SNE的perplexity参数
        all_metrics: 每层的聚类指标（用于折线图）
    """
    import matplotlib
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'
    matplotlib.rcParams['font.size'] = 18
    matplotlib.rcParams['axes.linewidth'] = 1.0
    
    os.makedirs(output_dir, exist_ok=True)
    
    n_layers = len(layer_indices)
    layers = sorted(layer_indices)
    
    # 深色配色方案
    LANGUAGE_COLORS = {
        'en': '#b71c1c',  # 深红
        'zh': '#1565c0',  # 深蓝
        'ar': '#2e7d32',  # 深绿
        'ko': '#6a1b9a',  # 深紫
        'vi': '#e65100',  # 深橙
        'th': '#f9a825',  # 深黄
        'it': '#4e342e',  # 深棕
        'bn': '#ad1457',  # 深粉
        'sw': '#37474f',  # 深灰
        'jw': '#00695c',  # 深青
    }
    
    # Query深色配色
    QUERY_COLORS = [
        '#d32f2f',  # 红
        '#1976d2',  # 蓝
        '#388e3c',  # 绿
        '#7b1fa2',  # 紫
        '#f57c00',  # 橙
        '#fbc02d',  # 黄
        '#5d4037',  # 棕
        '#c2185b',  # 粉
        '#455a64',  # 灰
        '#00796b',  # 青
    ]
    
    unique_langs = sorted(set(language_labels))
    lang_to_color = {lang: LANGUAGE_COLORS.get(lang, '#333333') for lang in unique_langs}
    
    # Query分组
    unique_queries = sorted(set(query_ids))
    query_to_indices_global = defaultdict(list)
    for i, qid in enumerate(query_ids):
        query_to_indices_global[qid].append(i)
    
    multilingual_queries = [qid for qid in unique_queries 
                           if len(query_to_indices_global[qid]) >= 2]
    n_queries = len(multilingual_queries)
    
    query_to_color = {qid: QUERY_COLORS[i % len(QUERY_COLORS)] for i, qid in enumerate(multilingual_queries)}
    
    # 预计算所有层的t-SNE
    print("Computing t-SNE for all layers...")
    tsne_results = {}
    for layer_idx in layers:
        embeddings = torch.stack(hidden_states_dict[layer_idx]).float()
        embeddings_np = F.normalize(embeddings, p=2, dim=1).numpy()
        
        print(f"  Layer {layer_idx}...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=1000)
        tsne_results[layer_idx] = tsne.fit_transform(embeddings_np)
    
    # 找到中间层（语义瓶颈层）
    middle_layer_idx = len(layers) // 2
    semantic_bottleneck_layer = layers[middle_layer_idx]
    
    # ========== 1. 单独保存折线图 ==========
    if all_metrics:
        # 使用大字体
        plt.rcParams.update({'font.size': 18})
        
        # 创建两个子图，左右排列，尺寸加大
        fig_line, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
        
        sil_lang = [all_metrics[l].get('silhouette_language', 0) for l in layers]
        sil_query = [all_metrics[l].get('silhouette_query', 0) for l in layers]
        
        # 统一Y轴范围以方便对比
        y_min = min(min(sil_lang), min(sil_query))
        y_max = max(max(sil_lang), max(sil_query))
        padding = (y_max - y_min) * 0.1 if y_max != y_min else 0.1
        y_limit = (y_min - padding, y_max + padding)
        # 或者固定范围 [-0.1, 1.0] ? 还是自适应比较好。自适应。
        
        # 图1: Language Clustering
        ax1.set_facecolor('#fafafa')
        ax1.plot(layers, sil_lang, 'o-', color='#c62828', linewidth=4, 
                 markersize=14, markeredgecolor='white', markeredgewidth=2,
                 label='Language Clustering', zorder=3)
        
        ax1.set_xlabel('Layer', fontsize=20, fontweight='bold')
        ax1.set_ylabel('Silhouette Score', fontsize=20, fontweight='bold')
        ax1.set_title('Language Clustering', fontsize=24, fontweight='bold', pad=20)
        ax1.set_xticks(layers)
        ax1.grid(True, alpha=0.3, linestyle='--', zorder=0)
        ax1.set_xlim(layers[0] - 1, layers[-1] + 1)
        ax1.set_ylim(y_limit)
        
        # 图2: Query Clustering
        ax2.set_facecolor('#fafafa')
        ax2.plot(layers, sil_query, 's-', color='#1565c0', linewidth=4, 
                 markersize=14, markeredgecolor='white', markeredgewidth=2,
                 label='Query Clustering', zorder=3)
        
        ax2.set_xlabel('Layer', fontsize=20, fontweight='bold')
        ax2.set_ylabel('Silhouette Score', fontsize=20, fontweight='bold')
        ax2.set_title('Query Clustering', fontsize=24, fontweight='bold', pad=20)
        ax2.set_xticks(layers)
        ax2.grid(True, alpha=0.3, linestyle='--', zorder=0)
        ax2.set_xlim(layers[0] - 1, layers[-1] + 1)
        ax2.set_ylim(y_limit)
        
        # 美化刻度和边框
        for ax in [ax1, ax2]:
            for spine in ax.spines.values():
                spine.set_edgecolor('#333333')
                spine.set_linewidth(1.5)
            ax.tick_params(axis='both', which='major', labelsize=18, width=1.5, length=8)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'clustering_quality_line.pdf'), 
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(os.path.join(output_dir, 'clustering_quality_line.png'), 
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("  Saved clustering quality line plot (split view)")
    
    # ========== 2. 创建 2×N 的t-SNE图（每格正方形，大小一致） ==========
    cell_size = 4.0  # 加大格子大小
    fig_width = cell_size * n_layers + 2.0  # 额外空间给图例
    fig_height = cell_size * 2 + 0.8
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor('white')
    
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, n_layers, figure=fig, wspace=0.05, hspace=0.12,
                  left=0.04, right=0.88, top=0.92, bottom=0.05)
    
    POINT_SIZE = 55
    ALPHA = 0.85
    
    for col_idx, layer_idx in enumerate(layers):
        embeddings_2d = tsne_results[layer_idx]
        is_bottleneck = (layer_idx == semantic_bottleneck_layer)
        
        # ========== 第一行：按语言着色 ==========
        ax = fig.add_subplot(gs[0, col_idx])
        ax.set_aspect('equal', adjustable='datalim')
        
        if is_bottleneck:
            ax.set_facecolor('#e3f2fd')
        else:
            ax.set_facecolor('white')
        
        for lang in unique_langs:
            indices = [i for i, l in enumerate(language_labels) if l == lang]
            ax.scatter(
                embeddings_2d[indices, 0], 
                embeddings_2d[indices, 1],
                c=[lang_to_color[lang]],
                marker='o',
                label=lang.upper(),
                alpha=ALPHA,
                s=POINT_SIZE,
                edgecolors='white',
                linewidths=0.2,
                zorder=2
            )
        
        ax.set_title(f'Layer {layer_idx}', fontsize=18, fontweight='bold', pad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # 统一坐标范围
        margin = 0.1
        x_range = embeddings_2d[:, 0].max() - embeddings_2d[:, 0].min()
        y_range = embeddings_2d[:, 1].max() - embeddings_2d[:, 1].min()
        max_range = max(x_range, y_range)
        x_center = (embeddings_2d[:, 0].max() + embeddings_2d[:, 0].min()) / 2
        y_center = (embeddings_2d[:, 1].max() + embeddings_2d[:, 1].min()) / 2
        ax.set_xlim(x_center - max_range/2 * (1+margin), x_center + max_range/2 * (1+margin))
        ax.set_ylim(y_center - max_range/2 * (1+margin), y_center + max_range/2 * (1+margin))
        
        for spine in ax.spines.values():
            spine.set_edgecolor('#888888')
            spine.set_linewidth(0.6)
        
        # 只在最后一列添加Language图例
        if col_idx == n_layers - 1:
            handles = [plt.Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor=lang_to_color[lang], 
                                  markersize=7, label=lang.upper())
                      for lang in unique_langs]
            ax.legend(
                handles=handles,
                loc='center left',
                bbox_to_anchor=(1.02, 0.5),
                fontsize=14,
                frameon=True,
                fancybox=False,
                edgecolor='#888888',
                ncol=1,
                handletextpad=0.2,
                labelspacing=0.2,
                title='Lang',
                title_fontsize=16
            )
        
        # ========== 第二行：按Query着色 + Star Plot ==========
        ax = fig.add_subplot(gs[1, col_idx])
        ax.set_aspect('equal', adjustable='datalim')
        
        if is_bottleneck:
            ax.set_facecolor('#e3f2fd')
        else:
            ax.set_facecolor('white')
        
        # Star Plot连线到质心
        for qid in multilingual_queries:
            indices = query_to_indices_global[qid]
            if len(indices) >= 2:
                points = embeddings_2d[indices]
                centroid = points.mean(axis=0)
                
                for point in points:
                    ax.plot(
                        [centroid[0], point[0]], 
                        [centroid[1], point[1]],
                        color='#555555',  # 更深的颜色
                        alpha=0.7,
                        linewidth=1.2,  # 更粗的线
                        zorder=1
                    )
                
                ax.scatter(centroid[0], centroid[1], 
                          c=[query_to_color[qid]], marker='+', 
                          s=30, linewidths=1.2, alpha=0.9, zorder=2)
        
        # 散点
        for q_idx, qid in enumerate(multilingual_queries):
            indices = query_to_indices_global[qid]
            ax.scatter(
                embeddings_2d[indices, 0],
                embeddings_2d[indices, 1],
                c=[query_to_color[qid]],
                marker='o',
                s=POINT_SIZE,
                alpha=ALPHA,
                edgecolors='white',
                linewidths=0.2,
                zorder=3,
                label=f'Q{q_idx + 1}'
            )
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(x_center - max_range/2 * (1+margin), x_center + max_range/2 * (1+margin))
        ax.set_ylim(y_center - max_range/2 * (1+margin), y_center + max_range/2 * (1+margin))
        
        for spine in ax.spines.values():
            spine.set_edgecolor('#888888')
            spine.set_linewidth(0.6)
        
        # 只在最后一列添加Query图例
        if col_idx == n_layers - 1:
            handles = [plt.Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor=query_to_color[qid], 
                                  markersize=7, label=f'Q{i+1}')
                      for i, qid in enumerate(multilingual_queries)]
            ax.legend(
                handles=handles,
                loc='center left',
                bbox_to_anchor=(1.02, 0.5),
                fontsize=14,
                frameon=True,
                fancybox=False,
                edgecolor='#888888',
                ncol=1,
                handletextpad=0.2,
                labelspacing=0.2,
                title='Query',
                title_fontsize=16
            )
    
    # 行标签
    fig.text(0.01, 0.72, 'By Language', fontsize=18, fontweight='bold', 
             rotation=90, va='center', ha='left')
    fig.text(0.01, 0.28, 'By Query', fontsize=18, fontweight='bold', 
             rotation=90, va='center', ha='left')
    
    # 保存
    save_path = os.path.join(output_dir, 'layer_clustering_tsne_combined.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    
    save_path_png = os.path.join(output_dir, 'layer_clustering_tsne_combined.png')
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    
    plt.close()
    
    print(f"  Saved combined t-SNE visualization to {save_path}")
    print(f"  Saved PNG version to {save_path_png}")


def visualize_all_layers_comparison(
    all_metrics: Dict[int, Dict],
    layer_indices: List[int],
    output_dir: str
):
    """
    可视化所有层的聚类指标对比
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 提取数据
    layers = sorted(layer_indices)
    silhouette_lang = [all_metrics[l]['silhouette_language'] for l in layers]
    silhouette_query = [all_metrics[l]['silhouette_query'] for l in layers]
    lang_gap = [all_metrics[l]['language_clustering_gap'] for l in layers]
    query_gap = [all_metrics[l]['query_clustering_gap'] for l in layers]
    
    # 创建图表，加大尺寸以适应大字体
    plt.rcParams.update({'font.size': 16})
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1. Silhouette Score对比
    ax = axes[0, 0]
    x = np.arange(len(layers))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, silhouette_lang, width, label='Language Clustering', color='steelblue')
    bars2 = ax.bar(x + width/2, silhouette_query, width, label='Query Clustering', color='coral')
    
    ax.set_xlabel('Layer Index', fontsize=18)
    ax.set_ylabel('Silhouette Score', fontsize=18)
    ax.set_title('Silhouette Score by Clustering Type', fontsize=20, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(layers, fontsize=16)
    ax.legend(fontsize=16)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.tick_params(axis='y', labelsize=16)
    
    # 2. Clustering Gap对比
    ax = axes[0, 1]
    ax.bar(x - width/2, lang_gap, width, label='Language Gap (intra - inter)', color='steelblue')
    ax.bar(x + width/2, query_gap, width, label='Query Gap (intra - inter)', color='coral')
    
    ax.set_xlabel('Layer Index', fontsize=18)
    ax.set_ylabel('Clustering Gap', fontsize=18)
    ax.set_title('Clustering Gap: Same-group vs Different-group', fontsize=20, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(layers, fontsize=16)
    ax.legend(fontsize=16)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.tick_params(axis='y', labelsize=16)
    
    # 3. 折线图：聚类趋势
    ax = axes[1, 0]
    ax.plot(layers, silhouette_lang, 'o-', color='steelblue', linewidth=3, 
            markersize=10, label='Language Silhouette')
    ax.plot(layers, silhouette_query, 's-', color='coral', linewidth=3, 
            markersize=10, label='Query Silhouette')
    
    ax.set_xlabel('Layer Index', fontsize=18)
    ax.set_ylabel('Silhouette Score', fontsize=18)
    ax.set_title('Clustering Trend Across Layers', fontsize=20, fontweight='bold')
    ax.legend(fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=16)
    
    # 标注关键层
    for i, layer in enumerate(layers):
        if layer in [1, len(layers)-1] or silhouette_lang[i] > silhouette_query[i]:
            ax.annotate(f'L{layer}', (layer, silhouette_lang[i]), 
                       textcoords="offset points", xytext=(0,15), ha='center', fontsize=14)
    
    # 4. 语言 vs 问题聚类差异
    ax = axes[1, 1]
    lang_vs_query = [all_metrics[l]['language_vs_query_clustering'] for l in layers]
    
    colors = ['steelblue' if v > 0 else 'coral' for v in lang_vs_query]
    ax.bar(layers, lang_vs_query, color=colors, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Layer Index', fontsize=18)
    ax.set_ylabel('Language Gap - Query Gap', fontsize=18)
    ax.set_title('Language vs Query Clustering Dominance', fontsize=20, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.tick_params(axis='both', labelsize=16)
    
    # 添加标注
    for i, (layer, val) in enumerate(zip(layers, lang_vs_query)):
        if val > 0:
            ax.text(layer, val + 0.02, 'Lang', ha='center', va='bottom', fontsize=14, color='steelblue')
        else:
            ax.text(layer, val - 0.02, 'Query', ha='center', va='top', fontsize=14, color='coral')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'layer_clustering_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Saved clustering comparison plot to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Analyze layer-wise clustering patterns")
    
    # 数据参数
    parser.add_argument("--data_type", type=str, choices=['ultrafeedback', 'safety', 'both'], 
                        default='ultrafeedback',
                        help="Data type to use")
    parser.add_argument("--ultrafeedback_dir", type=str, default=None,
                        help="Path to ultrafeedback data directory")
    parser.add_argument("--safety_data_path", type=str, default=None,
                        help="Path to safety data file")
    parser.add_argument("--languages", type=str, nargs="+", 
                        default=['en', 'zh', 'ar', 'ko', 'vi', 'th'],
                        help="Languages to include")
    parser.add_argument("--max_samples_per_lang", type=int, default=100,
                        help="Maximum samples per language")
    parser.add_argument("--max_queries", type=int, default=10,
                        help="Maximum number of queries to keep (only keep queries with multiple language versions)")
    
    # 模型参数
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Path to the LLM model")
    parser.add_argument("--layer_indices", type=int, nargs="+", 
                        default=None,
                        help="Layer indices to analyze (e.g., 1 8 15 20 31). If not specified, will auto-select.")
    
    # 其他参数
    parser.add_argument("--output_dir", type=str, default="./layer_clustering_analysis",
                        help="Output directory")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for hidden state extraction")
    parser.add_argument("--max_vis_samples", type=int, default=2000,
                        help="Maximum samples for t-SNE visualization")
    parser.add_argument("--skip_visualization", action="store_true",
                        help="Skip t-SNE visualization (faster)")
    parser.add_argument("--token_position", type=int, default=-1,
                        help="Token position to extract hidden state from (-1=last, -2=second last)")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ========== 1. 加载数据 ==========
    print("=" * 60)
    print("Loading data...")
    print("=" * 60)
    
    all_data = []
    
    if args.data_type in ['ultrafeedback', 'both'] and args.ultrafeedback_dir:
        ultrafeedback_data = load_ultrafeedback_data(
            args.ultrafeedback_dir,
            languages=args.languages,
            max_samples_per_lang=args.max_samples_per_lang
        )
        print(f"Loaded {len(ultrafeedback_data)} ultrafeedback samples")
        all_data.extend(ultrafeedback_data)
    
    if args.data_type in ['safety', 'both'] and args.safety_data_path:
        safety_data = load_safety_data(
            args.safety_data_path,
            max_samples=None  # 先加载全部，后面根据语言过滤
        )
        # 根据指定的语言列表过滤
        allowed_langs = set(args.languages)
        safety_data = [item for item in safety_data if item['language'] in allowed_langs]
        print(f"Loaded {len(safety_data)} safety samples (filtered by languages: {args.languages})")
        all_data.extend(safety_data)
    
    if not all_data:
        raise ValueError("No data loaded! Please specify --ultrafeedback_dir or --safety_data_path")
    
    print(f"Total samples: {len(all_data)}")
    
    # 统计语言分布
    lang_counts = defaultdict(int)
    for item in all_data:
        lang_counts[item['language']] += 1
    print(f"Language distribution: {dict(lang_counts)}")
    
    # 统计问题分布
    query_counts = defaultdict(int)
    for item in all_data:
        query_counts[item['idx']] += 1
    multilingual_queries = sum(1 for cnt in query_counts.values() if cnt >= 2)
    print(f"Unique queries: {len(query_counts)}, Multilingual queries: {multilingual_queries}")
    
    # ========== 过滤：只保留前N个有多语言版本的问题 ==========
    if args.max_queries is not None and args.max_queries > 0:
        # 找出有多语言版本的问题（至少2种语言）
        multilingual_query_ids = [qid for qid, cnt in query_counts.items() if cnt >= 2]
        
        # 只保留前max_queries个
        selected_queries = set(multilingual_query_ids[:args.max_queries])
        
        # 过滤数据
        all_data = [item for item in all_data if item['idx'] in selected_queries]
        
        print(f"\nFiltered to {args.max_queries} queries with multiple languages")
        print(f"Remaining samples: {len(all_data)}")
        
        # 重新统计语言分布
        lang_counts = defaultdict(int)
        for item in all_data:
            lang_counts[item['language']] += 1
        print(f"Language distribution after filtering: {dict(lang_counts)}")
    
    # ========== 2. 加载模型 ==========
    print("\n" + "=" * 60)
    print("Loading model...")
    print("=" * 60)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 获取模型层数
    num_layers = model.config.num_hidden_layers
    print(f"Model has {num_layers} layers")
    
    # 确定要分析的层
    if args.layer_indices is None:
        # 自动选择：第1层、1/4处、1/2处、3/4处、最后一层
        args.layer_indices = [
            1,                      # 第1层（接近embedding）
            num_layers // 4,        # 1/4处
            num_layers // 2,        # 中间层
            3 * num_layers // 4,    # 3/4处
            num_layers              # 最后一层
        ]
        # 去重并排序
        args.layer_indices = sorted(set(args.layer_indices))
    
    # 验证并过滤层索引：hidden_states 的有效索引是 0 到 num_layers（共 num_layers+1 个元素）
    # 但 layer_idx 从 1 开始（1 表示第一层 transformer），所以最大有效值是 num_layers
    valid_layer_indices = [idx for idx in args.layer_indices if 1 <= idx <= num_layers]
    invalid_layer_indices = [idx for idx in args.layer_indices if idx < 1 or idx > num_layers]
    
    if invalid_layer_indices:
        print(f"Warning: Some layer indices are out of range for {num_layers}-layer model: {invalid_layer_indices}")
        print(f"  These layers will be skipped.")
    
    if not valid_layer_indices:
        raise ValueError(f"No valid layer indices! All specified layers {args.layer_indices} are out of range for {num_layers}-layer model.")
    
    args.layer_indices = sorted(set(valid_layer_indices))
    print(f"Analyzing layers: {args.layer_indices}")
    print(f"Token position: {args.token_position} ({'last token' if args.token_position == -1 else 'second last token' if args.token_position == -2 else f'{args.token_position}'})")
    
    # ========== 3. 提取hidden states ==========
    print("\n" + "=" * 60)
    print("Extracting hidden states from multiple layers...")
    print("=" * 60)
    
    texts = [item['prompt'] for item in all_data]
    query_ids = [item['idx'] for item in all_data]
    languages = [item['language'] for item in all_data]
    
    hidden_states_dict = extract_hidden_states_multi_layer(
        model, tokenizer, texts,
        layer_indices=args.layer_indices,
        device=args.device,
        batch_size=args.batch_size,
        token_position=args.token_position
    )
    
    # 释放模型内存
    del model
    torch.cuda.empty_cache()
    
    # ========== 4. 计算聚类指标 ==========
    print("\n" + "=" * 60)
    print("Computing clustering metrics...")
    print("=" * 60)
    
    all_metrics = {}
    
    for layer_idx in args.layer_indices:
        print(f"\nLayer {layer_idx}:")
        
        embeddings = torch.stack(hidden_states_dict[layer_idx])
        metrics = compute_clustering_metrics(embeddings, languages, query_ids)
        all_metrics[layer_idx] = metrics
        
        print(f"  Silhouette (Language): {metrics['silhouette_language']:.4f}")
        print(f"  Silhouette (Query):    {metrics['silhouette_query']:.4f}")
        print(f"  Language Gap:          {metrics['language_clustering_gap']:.4f}")
        print(f"  Query Gap:             {metrics['query_clustering_gap']:.4f}")
        print(f"  Lang vs Query:         {metrics['language_vs_query_clustering']:.4f}")
        
        if metrics['language_vs_query_clustering'] > 0:
            print(f"  → Language clustering is STRONGER")
        else:
            print(f"  → Query clustering is STRONGER")
    
    # ========== 5. 保存指标结果 ==========
    print("\n" + "=" * 60)
    print("Saving results...")
    print("=" * 60)
    
    # 转换为可序列化格式
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {str(k): convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    results = {
        'model': args.model_name_or_path,
        'num_samples': len(all_data),
        'languages': list(lang_counts.keys()),
        'layer_indices': args.layer_indices,
        'metrics': convert_to_serializable(all_metrics)
    }
    
    with open(os.path.join(args.output_dir, 'clustering_metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Metrics saved to {os.path.join(args.output_dir, 'clustering_metrics.json')}")
    
    # ========== 6. 可视化 ==========
    if not args.skip_visualization:
        print("\n" + "=" * 60)
        print("Generating visualizations...")
        print("=" * 60)
        
        # 生成综合的 3×N t-SNE可视化图（学术论文风格）
        visualize_all_layers_tsne(
            hidden_states_dict=hidden_states_dict,
            language_labels=languages,
            query_ids=query_ids,
            layer_indices=args.layer_indices,
            output_dir=args.output_dir,
            all_metrics=all_metrics
        )
        
        # 生成层对比图（聚类指标）
        visualize_all_layers_comparison(all_metrics, args.layer_indices, args.output_dir)
    
    # ========== 7. 打印总结 ==========
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print("\n聚类模式分析:")
    for layer_idx in args.layer_indices:
        metrics = all_metrics[layer_idx]
        dominant = "Language" if metrics['language_vs_query_clustering'] > 0 else "Query"
        print(f"  Layer {layer_idx:2d}: {dominant:8s} dominant "
              f"(Lang Sil: {metrics['silhouette_language']:+.4f}, "
              f"Query Sil: {metrics['silhouette_query']:+.4f})")
    
    print("\n假设验证:")
    first_layer = args.layer_indices[0]
    last_layer = args.layer_indices[-1]
    middle_layer = args.layer_indices[len(args.layer_indices) // 2]
    
    first_dominant = "Language" if all_metrics[first_layer]['language_vs_query_clustering'] > 0 else "Query"
    last_dominant = "Language" if all_metrics[last_layer]['language_vs_query_clustering'] > 0 else "Query"
    middle_dominant = "Language" if all_metrics[middle_layer]['language_vs_query_clustering'] > 0 else "Query"
    
    print(f"  First layer ({first_layer}):  {first_dominant} clustering dominant")
    print(f"  Middle layer ({middle_layer}): {middle_dominant} clustering dominant")
    print(f"  Last layer ({last_layer}):  {last_dominant} clustering dominant")
    
    hypothesis_holds = (
        all_metrics[first_layer]['language_vs_query_clustering'] > 0 and
        all_metrics[last_layer]['language_vs_query_clustering'] > 0 and
        all_metrics[middle_layer]['language_vs_query_clustering'] < 0
    )
    
    if hypothesis_holds:
        print("\n✓ 假设得到支持: 第一层和最后一层按语言聚类，中间层按问题聚类")
    else:
        print("\n✗ 假设需要进一步验证")
        print("  建议：尝试不同的中间层或调整数据")
    
    print(f"\n完整结果已保存至: {args.output_dir}")


if __name__ == "__main__":
    main()

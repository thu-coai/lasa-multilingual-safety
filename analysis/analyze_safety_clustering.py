"""
LASA safety-semantics clustering analysis.

This script visualizes whether hidden states at candidate bottleneck layers
separate benign and harmful semantics while retaining multilingual semantic
alignment. It reports safety/query silhouette scores and writes t-SNE plots.

Legacy experiment note:
安全性聚类分析实验

实验1: 5条safety + 5条benign，中间层按safe/unsafe聚类
       左图按safe label着色，右图按query id聚类

实验2: 100条safety + 100条benign，en和sw两种语言
       在1/16/32层按safe label观察聚类
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
import matplotlib
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from transformers import AutoTokenizer, AutoModelForCausalLM

# 导入本地模块
try:
    from dataset import load_ultrafeedback_data, load_safety_data
except ImportError:
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
    """
    model.eval()
    hidden_states_dict = {layer_idx: [] for layer_idx in layer_indices}
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting hidden states"):
        batch_texts = texts[i:i+batch_size]
        
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        for layer_idx in layer_indices:
            layer_output = outputs.hidden_states[layer_idx]
            
            for j in range(len(batch_texts)):
                attention_mask = inputs['attention_mask'][j]
                seq_len = attention_mask.sum().item()
                token_idx = max(0, seq_len + token_position)
                hidden_state = layer_output[j, token_idx, :].cpu()
                hidden_states_dict[layer_idx].append(hidden_state)
    
    return hidden_states_dict


def compute_safety_clustering_metrics(
    embeddings: torch.Tensor,
    safety_labels: List[int],
    query_ids: List[int]
) -> Dict:
    """
    计算安全性聚类指标
    """
    embeddings = embeddings.float()
    embeddings_np = F.normalize(embeddings, p=2, dim=1).numpy()
    
    metrics = {}
    
    # 按safety label的silhouette score
    unique_safety = sorted(set(safety_labels))
    if len(unique_safety) > 1:
        try:
            metrics['silhouette_safety'] = silhouette_score(
                embeddings_np, safety_labels, metric='cosine'
            )
        except:
            metrics['silhouette_safety'] = 0.0
    else:
        metrics['silhouette_safety'] = 0.0
    
    # 按query id的silhouette score
    unique_queries = sorted(set(query_ids))
    if len(unique_queries) > 1:
        try:
            query_to_idx = {qid: i for i, qid in enumerate(unique_queries)}
            query_labels_numeric = [query_to_idx[qid] for qid in query_ids]
            metrics['silhouette_query'] = silhouette_score(
                embeddings_np, query_labels_numeric, metric='cosine'
            )
        except:
            metrics['silhouette_query'] = 0.0
    else:
        metrics['silhouette_query'] = 0.0
    
    # 计算同类相似度
    embeddings_tensor = torch.tensor(embeddings_np, dtype=torch.float32)
    
    # Safe vs Unsafe相似度
    safe_indices = [i for i, l in enumerate(safety_labels) if l == 0]
    unsafe_indices = [i for i, l in enumerate(safety_labels) if l == 1]
    
    intra_safe_sims = []
    intra_unsafe_sims = []
    inter_safety_sims = []
    
    if len(safe_indices) >= 2:
        safe_emb = embeddings_tensor[safe_indices]
        sim_matrix = torch.matmul(safe_emb, safe_emb.T)
        mask = ~torch.eye(len(safe_indices), dtype=torch.bool)
        intra_safe_sims = sim_matrix[mask].tolist()
    
    if len(unsafe_indices) >= 2:
        unsafe_emb = embeddings_tensor[unsafe_indices]
        sim_matrix = torch.matmul(unsafe_emb, unsafe_emb.T)
        mask = ~torch.eye(len(unsafe_indices), dtype=torch.bool)
        intra_unsafe_sims = sim_matrix[mask].tolist()
    
    # 计算跨类相似度
    for si in safe_indices:
        for ui in unsafe_indices:
            sim = torch.dot(embeddings_tensor[si], embeddings_tensor[ui]).item()
            inter_safety_sims.append(sim)
    
    metrics['intra_safe_similarity'] = np.mean(intra_safe_sims) if intra_safe_sims else 0.0
    metrics['intra_unsafe_similarity'] = np.mean(intra_unsafe_sims) if intra_unsafe_sims else 0.0
    metrics['inter_safety_similarity'] = np.mean(inter_safety_sims) if inter_safety_sims else 0.0
    metrics['safety_clustering_gap'] = (
        (metrics['intra_safe_similarity'] + metrics['intra_unsafe_similarity']) / 2 
        - metrics['inter_safety_similarity']
    )
    
    return metrics


def visualize_experiment1(
    hidden_states: List[torch.Tensor],
    safety_labels: List[int],
    query_ids: List[int],
    data_sources: List[str],
    layer_idx: int,
    output_dir: str,
    perplexity: int = 5
):
    """
    实验1可视化：左图按safe label，右图按query id
    """
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'
    matplotlib.rcParams['font.size'] = 11
    matplotlib.rcParams['axes.linewidth'] = 1.0
    
    os.makedirs(output_dir, exist_ok=True)
    
    embeddings = torch.stack(hidden_states).float()
    embeddings_np = F.normalize(embeddings, p=2, dim=1).numpy()
    
    # t-SNE降维
    print(f"  Computing t-SNE for layer {layer_idx}...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings_np)
    
    # 创建1×2图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('white')
    
    POINT_SIZE = 150
    ALPHA = 0.9
    
    # ========== 左图：按Safe Label着色 ==========
    ax = axes[0]
    ax.set_facecolor('white')
    
    SAFETY_COLORS = {
        0: '#2e7d32',  # Safe - 绿色
        1: '#c62828',  # Unsafe - 红色
    }
    SAFETY_LABELS = {
        0: 'Safe (Benign)',
        1: 'Unsafe (Harmful)'
    }
    
    for label in [0, 1]:
        indices = [i for i, l in enumerate(safety_labels) if l == label]
        ax.scatter(
            embeddings_2d[indices, 0],
            embeddings_2d[indices, 1],
            c=[SAFETY_COLORS[label]],
            marker='o',
            label=SAFETY_LABELS[label],
            alpha=ALPHA,
            s=POINT_SIZE,
            edgecolors='white',
            linewidths=0.5,
            zorder=2
        )
    
    ax.set_title(f'Layer {layer_idx} - By Safety Label', fontsize=14, fontweight='bold', pad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc='upper right', fontsize=10, frameon=True, fancybox=False, edgecolor='#666666')
    
    for spine in ax.spines.values():
        spine.set_edgecolor('#888888')
        spine.set_linewidth(0.6)
    
    # ========== 右图：按Query ID着色 ==========
    ax = axes[1]
    ax.set_facecolor('white')
    
    unique_queries = sorted(set(query_ids))
    n_queries = len(unique_queries)
    
    # 使用不同的颜色和形状区分safe/unsafe
    QUERY_COLORS = plt.cm.tab10(np.linspace(0, 1, min(n_queries, 10)))
    
    query_to_indices = defaultdict(list)
    for i, qid in enumerate(query_ids):
        query_to_indices[qid].append(i)
    
    # 绘制连线（同一query不同语言）
    for qid in unique_queries:
        indices = query_to_indices[qid]
        if len(indices) >= 2:
            points = embeddings_2d[indices]
            centroid = points.mean(axis=0)
            for point in points:
                ax.plot(
                    [centroid[0], point[0]],
                    [centroid[1], point[1]],
                    color='#555555',
                    alpha=0.5,
                    linewidth=1.0,
                    zorder=1
                )
    
    # 绘制点
    for q_idx, qid in enumerate(unique_queries):
        indices = query_to_indices[qid]
        color = QUERY_COLORS[q_idx % len(QUERY_COLORS)]
        
        for i in indices:
            marker = 'o' if safety_labels[i] == 0 else 's'  # 圆形=safe, 方形=unsafe
            ax.scatter(
                embeddings_2d[i, 0],
                embeddings_2d[i, 1],
                c=[color],
                marker=marker,
                s=POINT_SIZE,
                alpha=ALPHA,
                edgecolors='white',
                linewidths=0.5,
                zorder=3,
                label=f'Q{q_idx+1} ({"Safe" if safety_labels[i] == 0 else "Unsafe"})' if i == indices[0] else None
            )
    
    ax.set_title(f'Layer {layer_idx} - By Query ID', fontsize=14, fontweight='bold', pad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # 添加形状图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Safe'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=10, label='Unsafe'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, frameon=True, fancybox=False, edgecolor='#666666')
    
    for spine in ax.spines.values():
        spine.set_edgecolor('#888888')
        spine.set_linewidth(0.6)
    
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f'experiment1_layer{layer_idx}_safety_vs_query.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    save_path_png = save_path.replace('.pdf', '.png')
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"  Saved experiment1 visualization to {save_path}")


def visualize_experiment2(
    hidden_states_dict: Dict[int, List[torch.Tensor]],
    safety_labels: List[int],
    languages: List[str],
    layer_indices: List[int],
    output_dir: str,
    perplexity: int = 30,
    all_metrics: Dict = None
):
    """
    实验2可视化：分两个图
    1. 按 safe/unsafe 分组着色
    2. 按语言分组着色
    """
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'
    matplotlib.rcParams['font.size'] = 11
    matplotlib.rcParams['axes.linewidth'] = 1.0
    
    os.makedirs(output_dir, exist_ok=True)
    
    n_layers = len(layer_indices)
    layers = sorted(layer_indices)
    
    # 预计算所有层的t-SNE
    print("Computing t-SNE for all layers...")
    tsne_results = {}
    for layer_idx in layers:
        embeddings = torch.stack(hidden_states_dict[layer_idx]).float()
        embeddings_np = F.normalize(embeddings, p=2, dim=1).numpy()
        
        print(f"  Layer {layer_idx}...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=1000)
        tsne_results[layer_idx] = tsne.fit_transform(embeddings_np)
    
    # 颜色定义
    SAFETY_COLORS = {
        0: '#2e7d32',  # Safe - 绿色
        1: '#c62828',  # Unsafe - 红色
    }
    SAFETY_LABELS = {
        0: 'Safe (Benign)',
        1: 'Unsafe (Harmful)'
    }
    
    LANG_COLORS = {
        'en': '#1565c0',  # 蓝色
        'sw': '#e65100',  # 橙色
    }
    
    unique_langs = sorted(set(languages))
    
    POINT_SIZE = 80
    ALPHA = 0.85
    
    # ========== 图1: 按 Safe/Unsafe 分组 ==========
    cell_size = 3.5
    fig_width = cell_size * n_layers + 1.5
    fig_height = cell_size + 0.5
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor('white')
    
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(1, n_layers, figure=fig, wspace=0.1,
                  left=0.04, right=0.88, top=0.85, bottom=0.08)
    
    for col_idx, layer_idx in enumerate(layers):
        embeddings_2d = tsne_results[layer_idx]
        
        ax = fig.add_subplot(gs[0, col_idx])
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_facecolor('white')
        
        # 按安全标签分组绘制
        for safety in [0, 1]:
            indices = [i for i in range(len(safety_labels)) if safety_labels[i] == safety]
            
            ax.scatter(
                embeddings_2d[indices, 0],
                embeddings_2d[indices, 1],
                c=[SAFETY_COLORS[safety]],
                marker='o',
                label=SAFETY_LABELS[safety],
                alpha=ALPHA,
                s=POINT_SIZE,
                edgecolors='white',
                linewidths=0.3,
                zorder=2
            )
        
        ax.set_title(f'Layer {layer_idx}', fontsize=13, fontweight='bold', pad=8)
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
        
        # 最后一列添加图例
        if col_idx == n_layers - 1:
            ax.legend(
                loc='center left',
                bbox_to_anchor=(1.02, 0.5),
                fontsize=10,
                frameon=True,
                fancybox=False,
                edgecolor='#888888',
                ncol=1,
                handletextpad=0.3,
                labelspacing=0.4
            )
    
    # 添加聚类指标信息
    if all_metrics:
        metrics_text = "Silhouette (Safety): " + " | ".join(
            [f"L{l}: {all_metrics[l].get('silhouette_safety', 0):.3f}" for l in layers]
        )
        fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=10, style='italic')
    
    plt.suptitle('Safety Clustering Across Layers (By Safe/Unsafe)', fontsize=14, fontweight='bold', y=0.96)
    
    save_path = os.path.join(output_dir, 'experiment2_by_safety.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"  Saved safety clustering visualization to {save_path}")
    
    # ========== 图2: 按语言分组 ==========
    fig = plt.figure(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor('white')
    
    gs = GridSpec(1, n_layers, figure=fig, wspace=0.1,
                  left=0.04, right=0.88, top=0.85, bottom=0.08)
    
    for col_idx, layer_idx in enumerate(layers):
        embeddings_2d = tsne_results[layer_idx]
        
        ax = fig.add_subplot(gs[0, col_idx])
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_facecolor('white')
        
        # 按语言分组绘制
        for lang in unique_langs:
            indices = [i for i in range(len(languages)) if languages[i] == lang]
            
            color = LANG_COLORS.get(lang, '#333333')
            
            ax.scatter(
                embeddings_2d[indices, 0],
                embeddings_2d[indices, 1],
                c=[color],
                marker='o',
                label=lang.upper(),
                alpha=ALPHA,
                s=POINT_SIZE,
                edgecolors='white',
                linewidths=0.3,
                zorder=2
            )
        
        ax.set_title(f'Layer {layer_idx}', fontsize=13, fontweight='bold', pad=8)
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
        
        # 最后一列添加图例
        if col_idx == n_layers - 1:
            ax.legend(
                loc='center left',
                bbox_to_anchor=(1.02, 0.5),
                fontsize=10,
                frameon=True,
                fancybox=False,
                edgecolor='#888888',
                ncol=1,
                handletextpad=0.3,
                labelspacing=0.4
            )
    
    plt.suptitle('Language Distribution Across Layers (By Language)', fontsize=14, fontweight='bold', y=0.96)
    
    save_path = os.path.join(output_dir, 'experiment2_by_language.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"  Saved language clustering visualization to {save_path}")
    
    # ========== 图3: Safe/Unsafe + 语言标签混合图 ==========
    COMBINED_COLORS = {
        ('en', 0): '#1565c0',  # EN-Safe: 蓝色
        ('en', 1): '#b71c1c',  # EN-Unsafe: 深红
        ('sw', 0): '#2e7d32',  # SW-Safe: 绿色
        ('sw', 1): '#e65100',  # SW-Unsafe: 橙色
    }
    LANG_MARKERS = {
        'en': 'o',  # 圆形
        'sw': 's',  # 方形
    }
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor('white')
    
    gs = GridSpec(1, n_layers, figure=fig, wspace=0.1,
                  left=0.04, right=0.85, top=0.85, bottom=0.08)
    
    for col_idx, layer_idx in enumerate(layers):
        embeddings_2d = tsne_results[layer_idx]
        
        ax = fig.add_subplot(gs[0, col_idx])
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_facecolor('white')
        
        # 按语言和安全标签分组绘制
        for lang in unique_langs:
            for safety in [0, 1]:
                indices = [i for i in range(len(safety_labels)) 
                          if languages[i] == lang and safety_labels[i] == safety]
                
                if not indices:
                    continue
                
                color = COMBINED_COLORS.get((lang, safety), '#333333')
                marker = LANG_MARKERS.get(lang, 'o')
                safety_name = 'Safe' if safety == 0 else 'Unsafe'
                label = f'{lang.upper()}-{safety_name}'
                
                ax.scatter(
                    embeddings_2d[indices, 0],
                    embeddings_2d[indices, 1],
                    c=[color],
                    marker=marker,
                    label=label,
                    alpha=ALPHA,
                    s=POINT_SIZE,
                    edgecolors='white',
                    linewidths=0.3,
                    zorder=2
                )
        
        ax.set_title(f'Layer {layer_idx}', fontsize=13, fontweight='bold', pad=8)
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
        
        # 最后一列添加图例
        if col_idx == n_layers - 1:
            ax.legend(
                loc='center left',
                bbox_to_anchor=(1.02, 0.5),
                fontsize=9,
                frameon=True,
                fancybox=False,
                edgecolor='#888888',
                ncol=1,
                handletextpad=0.3,
                labelspacing=0.4
            )
    
    if all_metrics:
        metrics_text = "Silhouette (Safety): " + " | ".join(
            [f"L{l}: {all_metrics[l].get('silhouette_safety', 0):.3f}" for l in layers]
        )
        fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=10, style='italic')
    
    plt.suptitle('Safety × Language Clustering (Combined)', fontsize=14, fontweight='bold', y=0.96)
    
    save_path = os.path.join(output_dir, 'experiment2_combined.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"  Saved combined clustering visualization to {save_path}")
    
    # ========== 单独保存折线图 ==========
    if all_metrics:
        fig_line = plt.figure(figsize=(6, 4))
        ax_line = fig_line.add_subplot(111)
        ax_line.set_facecolor('#fafafa')
        
        sil_safety = [all_metrics[l].get('silhouette_safety', 0) for l in layers]
        sil_query = [all_metrics[l].get('silhouette_query', 0) for l in layers]
        
        ax_line.plot(layers, sil_safety, 'o-', color='#c62828', linewidth=2.5, 
                     markersize=10, markeredgecolor='white', markeredgewidth=1.5,
                     label='Safety Clustering', zorder=3)
        ax_line.plot(layers, sil_query, 's--', color='#1565c0', linewidth=2.5, 
                     markersize=9, markeredgecolor='white', markeredgewidth=1.5,
                     label='Query Clustering', zorder=3)
        
        ax_line.set_xlabel('Layer', fontsize=12, fontweight='bold')
        ax_line.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
        ax_line.set_title('Safety vs Query Clustering Quality', fontsize=14, fontweight='bold', pad=10)
        ax_line.legend(loc='best', fontsize=10, frameon=True, fancybox=False, edgecolor='#666666')
        ax_line.set_xticks(layers)
        ax_line.grid(True, alpha=0.3, linestyle='--')
        
        for spine in ax_line.spines.values():
            spine.set_edgecolor('#666666')
            spine.set_linewidth(0.8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'experiment2_clustering_quality_line.pdf'), 
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(os.path.join(output_dir, 'experiment2_clustering_quality_line.png'), 
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("  Saved clustering quality line plot")


def run_experiment1(args):
    """
    实验1: 5条safety + 5条benign，中间层按safe/unsafe聚类
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Small-scale Safety Clustering")
    print("  5 safety + 5 benign queries, middle layer analysis")
    print("=" * 60)
    
    output_dir = os.path.join(args.output_dir, "experiment1")
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    print("\nLoading data...")
    
    # Benign data (safe)
    safe_data = load_ultrafeedback_data(
        args.ultrafeedback_dir,
        languages=args.languages,
        max_samples_per_lang=args.n_samples_exp1
    )
    # 只取前n_samples_exp1个unique query
    unique_safe_queries = []
    seen_idx = set()
    for item in safe_data:
        if item['idx'] not in seen_idx and len(unique_safe_queries) < args.n_samples_exp1:
            seen_idx.add(item['idx'])
            unique_safe_queries.append(item['idx'])
    
    safe_data = [item for item in safe_data if item['idx'] in unique_safe_queries]
    print(f"  Safe (benign) samples: {len(safe_data)} ({len(unique_safe_queries)} queries)")
    
    # Safety data (unsafe)
    unsafe_data = load_safety_data(args.safety_data_path, max_samples=args.n_samples_exp1)
    
    # 过滤语言
    allowed_langs = set(args.languages)
    unsafe_data = [item for item in unsafe_data if item['language'] in allowed_langs]
    
    # 只取前n_samples_exp1个unique query
    unique_unsafe_queries = []
    seen_idx = set()
    for item in unsafe_data:
        if item['idx'] not in seen_idx and len(unique_unsafe_queries) < args.n_samples_exp1:
            seen_idx.add(item['idx'])
            unique_unsafe_queries.append(item['idx'])
    
    unsafe_data = [item for item in unsafe_data if item['idx'] in unique_unsafe_queries]
    print(f"  Unsafe (safety) samples: {len(unsafe_data)} ({len(unique_unsafe_queries)} queries)")
    
    # 合并数据
    all_data = []
    safe_offset = 0
    unsafe_offset = 10000  # 确保ID不重叠
    
    for item in safe_data:
        all_data.append({
            'prompt': item['prompt'],
            'language': item['language'],
            'idx': item['idx'] + safe_offset,
            'safety_label': 0,  # Safe
            'source': 'benign'
        })
    
    for item in unsafe_data:
        all_data.append({
            'prompt': item['prompt'],
            'language': item['language'],
            'idx': item['idx'] + unsafe_offset,
            'safety_label': 1,  # Unsafe
            'source': 'safety'
        })
    
    print(f"  Total samples: {len(all_data)}")
    
    # 加载模型
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    num_layers = model.config.num_hidden_layers
    middle_layer = num_layers // 2
    print(f"  Model has {num_layers} layers, using middle layer: {middle_layer}")
    
    # 提取hidden states
    print("\nExtracting hidden states...")
    texts = [item['prompt'] for item in all_data]
    
    hidden_states_dict = extract_hidden_states_multi_layer(
        model, tokenizer, texts,
        layer_indices=[middle_layer],
        device=args.device,
        batch_size=args.batch_size,
        token_position=args.token_position
    )
    
    # 释放模型内存
    del model
    torch.cuda.empty_cache()
    
    # 准备标签
    safety_labels = [item['safety_label'] for item in all_data]
    query_ids = [item['idx'] for item in all_data]
    data_sources = [item['source'] for item in all_data]
    
    # 计算聚类指标
    print("\nComputing clustering metrics...")
    embeddings = torch.stack(hidden_states_dict[middle_layer])
    metrics = compute_safety_clustering_metrics(embeddings, safety_labels, query_ids)
    
    print(f"\nLayer {middle_layer} Metrics:")
    print(f"  Silhouette (Safety):    {metrics['silhouette_safety']:.4f}")
    print(f"  Silhouette (Query):     {metrics['silhouette_query']:.4f}")
    print(f"  Safety Clustering Gap:  {metrics['safety_clustering_gap']:.4f}")
    
    # 可视化
    print("\nGenerating visualization...")
    visualize_experiment1(
        hidden_states_dict[middle_layer],
        safety_labels,
        query_ids,
        data_sources,
        middle_layer,
        output_dir,
        perplexity=min(5, len(all_data) // 2 - 1)  # 小数据集使用较小的perplexity
    )
    
    # 保存结果
    results = {
        'experiment': 'experiment1',
        'model': args.model_name_or_path,
        'n_safe_queries': len(unique_safe_queries),
        'n_unsafe_queries': len(unique_unsafe_queries),
        'n_total_samples': len(all_data),
        'middle_layer': middle_layer,
        'metrics': {str(k): float(v) if isinstance(v, (np.floating, float)) else v 
                   for k, v in metrics.items()}
    }
    
    with open(os.path.join(output_dir, 'experiment1_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")


def run_experiment2(args):
    """
    实验2: 100条safety + 100条benign，en和sw两种语言
           在1/16/32层按safe label观察聚类
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Multi-layer Safety Clustering")
    print("  100 safety + 100 benign queries, EN & SW languages")
    print("  Layers: 1, 16, 32")
    print("=" * 60)
    
    output_dir = os.path.join(args.output_dir, "experiment2")
    os.makedirs(output_dir, exist_ok=True)
    
    # 只使用en和sw两种语言
    exp2_languages = ['en', 'sw']
    
    # 加载数据
    print("\nLoading data...")
    
    # Benign data (safe)
    safe_data = load_ultrafeedback_data(
        args.ultrafeedback_dir,
        languages=exp2_languages,
        max_samples_per_lang=args.n_samples_exp2
    )
    
    # 只取前n_samples_exp2个unique query
    unique_safe_queries = []
    seen_idx = set()
    for item in safe_data:
        if item['idx'] not in seen_idx and len(unique_safe_queries) < args.n_samples_exp2:
            seen_idx.add(item['idx'])
            unique_safe_queries.append(item['idx'])
    
    safe_data = [item for item in safe_data if item['idx'] in unique_safe_queries]
    print(f"  Safe (benign) samples: {len(safe_data)} ({len(unique_safe_queries)} queries)")
    
    # Safety data (unsafe)
    unsafe_data = load_safety_data(args.safety_data_path, max_samples=args.n_samples_exp2)
    
    # 只使用en和sw
    unsafe_data = [item for item in unsafe_data if item['language'] in exp2_languages]
    
    # 只取前n_samples_exp2个unique query
    unique_unsafe_queries = []
    seen_idx = set()
    for item in unsafe_data:
        if item['idx'] not in seen_idx and len(unique_unsafe_queries) < args.n_samples_exp2:
            seen_idx.add(item['idx'])
            unique_unsafe_queries.append(item['idx'])
    
    unsafe_data = [item for item in unsafe_data if item['idx'] in unique_unsafe_queries]
    print(f"  Unsafe (safety) samples: {len(unsafe_data)} ({len(unique_unsafe_queries)} queries)")
    
    # 合并数据
    all_data = []
    safe_offset = 0
    unsafe_offset = 10000
    
    for item in safe_data:
        all_data.append({
            'prompt': item['prompt'],
            'language': item['language'],
            'idx': item['idx'] + safe_offset,
            'safety_label': 0,
            'source': 'benign'
        })
    
    for item in unsafe_data:
        all_data.append({
            'prompt': item['prompt'],
            'language': item['language'],
            'idx': item['idx'] + unsafe_offset,
            'safety_label': 1,
            'source': 'safety'
        })
    
    print(f"  Total samples: {len(all_data)}")
    
    # 统计语言分布
    lang_counts = defaultdict(lambda: {'safe': 0, 'unsafe': 0})
    for item in all_data:
        safety_type = 'safe' if item['safety_label'] == 0 else 'unsafe'
        lang_counts[item['language']][safety_type] += 1
    print(f"  Language distribution: {dict(lang_counts)}")
    
    # 加载模型
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    num_layers = model.config.num_hidden_layers
    
    # 使用1/16/32层（如果模型层数不够，自动调整）
    layer_indices = args.layer_indices_exp2 if args.layer_indices_exp2 else [1, 16, 32]
    layer_indices = [l for l in layer_indices if l <= num_layers]
    print(f"  Model has {num_layers} layers, analyzing layers: {layer_indices}")
    
    # 提取hidden states
    print("\nExtracting hidden states...")
    texts = [item['prompt'] for item in all_data]
    
    hidden_states_dict = extract_hidden_states_multi_layer(
        model, tokenizer, texts,
        layer_indices=layer_indices,
        device=args.device,
        batch_size=args.batch_size,
        token_position=args.token_position
    )
    
    # 释放模型内存
    del model
    torch.cuda.empty_cache()
    
    # 准备标签
    safety_labels = [item['safety_label'] for item in all_data]
    query_ids = [item['idx'] for item in all_data]
    languages = [item['language'] for item in all_data]
    
    # 计算每层的聚类指标
    print("\nComputing clustering metrics for each layer...")
    all_metrics = {}
    
    for layer_idx in layer_indices:
        embeddings = torch.stack(hidden_states_dict[layer_idx])
        metrics = compute_safety_clustering_metrics(embeddings, safety_labels, query_ids)
        all_metrics[layer_idx] = metrics
        
        print(f"\nLayer {layer_idx}:")
        print(f"  Silhouette (Safety):    {metrics['silhouette_safety']:.4f}")
        print(f"  Silhouette (Query):     {metrics['silhouette_query']:.4f}")
        print(f"  Safety Clustering Gap:  {metrics['safety_clustering_gap']:.4f}")
    
    # 可视化
    print("\nGenerating visualizations...")
    visualize_experiment2(
        hidden_states_dict,
        safety_labels,
        languages,
        layer_indices,
        output_dir,
        perplexity=min(30, len(all_data) // 2 - 1),
        all_metrics=all_metrics
    )
    
    # 保存结果
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
        'experiment': 'experiment2',
        'model': args.model_name_or_path,
        'languages': exp2_languages,
        'n_safe_queries': len(unique_safe_queries),
        'n_unsafe_queries': len(unique_unsafe_queries),
        'n_total_samples': len(all_data),
        'layer_indices': layer_indices,
        'metrics': convert_to_serializable(all_metrics)
    }
    
    with open(os.path.join(output_dir, 'experiment2_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Safety Clustering Analysis")
    
    # 数据参数
    parser.add_argument("--ultrafeedback_dir", type=str, required=True,
                        help="Path to ultrafeedback data directory (benign data)")
    parser.add_argument("--safety_data_path", type=str, required=True,
                        help="Path to safety data file (unsafe data)")
    parser.add_argument("--languages", type=str, nargs="+", 
                        default=['en', 'zh', 'sw'],
                        help="Languages to include for experiment 1")
    
    # 模型参数
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Path to the LLM model")
    
    # 实验参数
    parser.add_argument("--experiment", type=str, choices=['1', '2', 'both'], 
                        default='both',
                        help="Which experiment to run")
    parser.add_argument("--n_samples_exp1", type=int, default=5,
                        help="Number of queries for experiment 1")
    parser.add_argument("--n_samples_exp2", type=int, default=100,
                        help="Number of queries for experiment 2")
    parser.add_argument("--layer_indices_exp2", type=int, nargs="+", 
                        default=[1, 16, 32],
                        help="Layer indices for experiment 2")
    
    # 其他参数
    parser.add_argument("--output_dir", type=str, default="./output_analysis/safety_clustering",
                        help="Output directory")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for hidden state extraction")
    parser.add_argument("--token_position", type=int, default=-1,
                        help="Token position to extract hidden state from")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.experiment in ['1', 'both']:
        run_experiment1(args)
    
    if args.experiment in ['2', 'both']:
        run_experiment2(args)
    
    print("\n" + "=" * 60)
    print("All experiments completed!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

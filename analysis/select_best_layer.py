"""
Select the LASA semantic bottleneck layer.

For each layer, compute silhouette scores under language labels and query
labels, then choose the layer maximizing silhouette_query - silhouette_language.
A larger value indicates stronger language-agnostic semantic clustering.

Legacy usage note:
最佳层选择脚本

基于聚类分析选择最佳的语义层：
1. 遍历模型的所有层
2. 对每层计算按语言聚类和按问题聚类的 Silhouette Score
3. 计算差值 (silhouette_query - silhouette_language)
4. 选择差值最大的层（即语义聚类最强的层）

使用方法：
    python select_best_layer.py \
        --model_name_or_path /path/to/model \
        --safety_data_path /path/to/safety_data.json \
        --output_dir ./output_layer_selection
"""

import os
import argparse
import json
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

from transformers import AutoTokenizer, AutoModelForCausalLM

# 尝试导入本地模块
try:
    from dataset import load_safety_data
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from dataset import load_safety_data


def extract_hidden_states_single_layer(
    model,
    tokenizer,
    texts: List[str],
    layer_idx: int,
    device: str = 'cuda',
    batch_size: int = 8,
    max_length: int = 512,
    token_position: int = -1
) -> List[torch.Tensor]:
    """
    提取单层的 hidden states
    
    Args:
        model: 预训练模型
        tokenizer: tokenizer
        texts: 文本列表
        layer_idx: 要提取的层索引
        device: 设备
        batch_size: 批大小
        max_length: 最大序列长度
        token_position: 使用哪个位置的token，-1表示最后一个，-2表示倒数第二个
    
    Returns:
        hidden states列表
    """
    model.eval()
    hidden_states_list = []
    
    for i in range(0, len(texts), batch_size):
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
        
        # 获取指定层的hidden states
        layer_output = outputs.hidden_states[layer_idx]
        
        # 获取每个样本指定位置的hidden state
        for j in range(len(batch_texts)):
            attention_mask = inputs['attention_mask'][j]
            seq_len = attention_mask.sum().item()
            
            # 根据token_position计算实际索引
            token_idx = max(0, seq_len + token_position)
            
            hidden_state = layer_output[j, token_idx, :].cpu()
            hidden_states_list.append(hidden_state)
    
    return hidden_states_list


def compute_silhouette_scores(
    embeddings: torch.Tensor,
    language_labels: List[str],
    query_ids: List[int]
) -> Dict[str, float]:
    """
    计算按语言和按问题聚类的 Silhouette Score
    
    Args:
        embeddings: hidden states, shape [N, dim]
        language_labels: 语言标签列表
        query_ids: 问题ID列表
    
    Returns:
        包含两种 Silhouette Score 的字典
    """
    # 转换为float32
    embeddings = embeddings.float()
    
    # L2归一化
    embeddings_np = F.normalize(embeddings, p=2, dim=1).numpy()
    
    scores = {
        'silhouette_language': 0.0,
        'silhouette_query': 0.0,
        'difference': 0.0  # silhouette_query - silhouette_language
    }
    
    # 将语言标签转换为数值
    unique_langs = sorted(set(language_labels))
    lang_to_idx = {lang: i for i, lang in enumerate(unique_langs)}
    lang_labels_numeric = [lang_to_idx[lang] for lang in language_labels]
    
    # 将query_id转换为数值
    unique_queries = sorted(set(query_ids))
    query_to_idx = {qid: i for i, qid in enumerate(unique_queries)}
    query_labels_numeric = [query_to_idx[qid] for qid in query_ids]
    
    # 计算按语言聚类的 Silhouette Score
    if len(unique_langs) > 1:
        try:
            scores['silhouette_language'] = silhouette_score(
                embeddings_np, lang_labels_numeric, metric='cosine'
            )
        except Exception as e:
            print(f"Warning: Language silhouette computation failed: {e}")
            scores['silhouette_language'] = 0.0
    
    # 计算按问题聚类的 Silhouette Score
    if len(unique_queries) > 1:
        try:
            scores['silhouette_query'] = silhouette_score(
                embeddings_np, query_labels_numeric, metric='cosine'
            )
        except Exception as e:
            print(f"Warning: Query silhouette computation failed: {e}")
            scores['silhouette_query'] = 0.0
    
    # 计算差值
    scores['difference'] = scores['silhouette_query'] - scores['silhouette_language']
    
    return scores


def visualize_layer_scores(
    all_scores: Dict[int, Dict],
    output_dir: str,
    best_layer: int
):
    """
    可视化各层的 Silhouette Score
    
    Args:
        all_scores: 每层的分数字典
        output_dir: 输出目录
        best_layer: 最佳层索引
    """
    import matplotlib
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'
    matplotlib.rcParams['font.size'] = 11
    
    os.makedirs(output_dir, exist_ok=True)
    
    layers = sorted(all_scores.keys())
    sil_lang = [all_scores[l]['silhouette_language'] for l in layers]
    sil_query = [all_scores[l]['silhouette_query'] for l in layers]
    difference = [all_scores[l]['difference'] for l in layers]
    
    # 创建图表
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # 图1: Silhouette Scores
    ax1 = axes[0]
    ax1.plot(layers, sil_lang, 'o-', color='#c62828', linewidth=2, 
             markersize=6, label='Language Clustering')
    ax1.plot(layers, sil_query, 's-', color='#1565c0', linewidth=2, 
             markersize=6, label='Query (Semantic) Clustering')
    ax1.axvline(x=best_layer, color='green', linestyle='--', linewidth=2, 
                alpha=0.7, label=f'Best Layer: {best_layer}')
    
    ax1.set_xlabel('Layer Index', fontsize=12)
    ax1.set_ylabel('Silhouette Score', fontsize=12)
    ax1.set_title('Silhouette Scores Across Layers', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(min(layers) - 0.5, max(layers) + 0.5)
    
    # 图2: Difference (Query - Language)
    ax2 = axes[1]
    colors = ['#1565c0' if d > 0 else '#c62828' for d in difference]
    ax2.bar(layers, difference, color=colors, edgecolor='black', linewidth=0.5, alpha=0.8)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.axvline(x=best_layer, color='green', linestyle='--', linewidth=2, 
                alpha=0.7, label=f'Best Layer: {best_layer}')
    
    ax2.set_xlabel('Layer Index', fontsize=12)
    ax2.set_ylabel('Score Difference (Query - Language)', fontsize=12)
    ax2.set_title('Query vs Language Clustering Difference\n(Positive = Query clustering stronger)', 
                  fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xlim(min(layers) - 0.5, max(layers) + 0.5)
    
    plt.tight_layout()
    
    # 保存图表
    save_path = os.path.join(output_dir, 'layer_selection_analysis.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Select the best layer based on clustering analysis")
    
    # 数据参数
    parser.add_argument("--safety_data_path", type=str, required=True,
                        help="Path to safety data file")
    parser.add_argument("--languages", type=str, nargs="+", 
                        default=['en', 'zh', 'ar', 'ko', 'vi', 'th', 'it'],
                        help="Languages to include")
    parser.add_argument("--max_queries", type=int, default=10,
                        help="Number of safety queries to use (default: 10)")
    
    # 模型参数
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Path to the LLM model")
    
    # 其他参数
    parser.add_argument("--output_dir", type=str, default="./output_layer_selection",
                        help="Output directory")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for hidden state extraction")
    parser.add_argument("--token_position", type=int, default=-1,
                        help="Token position to extract hidden state from (-1=last, -2=second last)")
    parser.add_argument("--skip_visualization", action="store_true",
                        help="Skip generating visualization plots")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ========== 1. 加载数据 ==========
    print("=" * 60)
    print("Loading safety data...")
    print("=" * 60)
    
    safety_data = load_safety_data(args.safety_data_path, max_samples=None)
    
    # 按语言过滤
    allowed_langs = set(args.languages)
    safety_data = [item for item in safety_data if item['language'] in allowed_langs]
    print(f"Loaded {len(safety_data)} samples (languages: {args.languages})")
    
    # 统计每个问题的语言版本数
    query_lang_counts = defaultdict(set)
    for item in safety_data:
        query_lang_counts[item['idx']].add(item['language'])
    
    # 只保留有多语言版本的问题
    multilingual_query_ids = [
        qid for qid, langs in query_lang_counts.items() 
        if len(langs) >= 2
    ]
    
    # 选择前 max_queries 个问题
    selected_queries = set(multilingual_query_ids[:args.max_queries])
    safety_data = [item for item in safety_data if item['idx'] in selected_queries]
    
    print(f"Selected {len(selected_queries)} queries with multiple languages")
    print(f"Total samples after filtering: {len(safety_data)}")
    
    # 统计语言分布
    lang_counts = defaultdict(int)
    for item in safety_data:
        lang_counts[item['language']] += 1
    print(f"Language distribution: {dict(lang_counts)}")
    
    if len(safety_data) < 10:
        raise ValueError(f"Not enough data samples ({len(safety_data)}). Need at least 10 samples.")
    
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
    print(f"Model: {os.path.basename(args.model_name_or_path)}")
    print(f"Number of layers: {num_layers}")
    
    # ========== 3. 遍历所有层，计算 Silhouette Scores ==========
    print("\n" + "=" * 60)
    print("Analyzing all layers...")
    print("=" * 60)
    
    texts = [item['prompt'] for item in safety_data]
    query_ids = [item['idx'] for item in safety_data]
    languages = [item['language'] for item in safety_data]
    
    all_scores = {}
    best_layer = 1
    best_difference = float('-inf')
    
    # 遍历所有层 (从第1层到最后一层，0层是embedding层)
    for layer_idx in tqdm(range(1, num_layers + 1), desc="Processing layers"):
        # 提取该层的 hidden states
        hidden_states = extract_hidden_states_single_layer(
            model, tokenizer, texts,
            layer_idx=layer_idx,
            device=args.device,
            batch_size=args.batch_size,
            token_position=args.token_position
        )
        
        # 计算 Silhouette Scores
        embeddings = torch.stack(hidden_states)
        scores = compute_silhouette_scores(embeddings, languages, query_ids)
        all_scores[layer_idx] = scores
        
        # 检查是否是最佳层
        if scores['difference'] > best_difference:
            best_difference = scores['difference']
            best_layer = layer_idx
        
        # 打印进度
        print(f"  Layer {layer_idx:2d}: Language={scores['silhouette_language']:+.4f}, "
              f"Query={scores['silhouette_query']:+.4f}, "
              f"Diff={scores['difference']:+.4f}")
    
    # 释放模型内存
    del model
    torch.cuda.empty_cache()
    
    # ========== 4. 输出结果 ==========
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"\n🎯 Best Layer: {best_layer}")
    print(f"   - Silhouette (Language): {all_scores[best_layer]['silhouette_language']:+.4f}")
    print(f"   - Silhouette (Query):    {all_scores[best_layer]['silhouette_query']:+.4f}")
    print(f"   - Difference:            {all_scores[best_layer]['difference']:+.4f}")
    
    # 找出 Top-5 层
    sorted_layers = sorted(all_scores.items(), key=lambda x: x[1]['difference'], reverse=True)
    print(f"\nTop-5 layers by difference (Query - Language):")
    for i, (layer, scores) in enumerate(sorted_layers[:5]):
        print(f"  {i+1}. Layer {layer:2d}: diff={scores['difference']:+.4f} "
              f"(query={scores['silhouette_query']:+.4f}, lang={scores['silhouette_language']:+.4f})")
    
    # ========== 5. 保存结果 ==========
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
        'model_name': os.path.basename(args.model_name_or_path),
        'num_layers': num_layers,
        'num_queries': len(selected_queries),
        'num_samples': len(safety_data),
        'languages': list(lang_counts.keys()),
        'token_position': args.token_position,
        'best_layer': best_layer,
        'best_difference': float(best_difference),
        'all_layer_scores': convert_to_serializable(all_scores),
        'top5_layers': [
            {
                'layer': layer,
                'silhouette_language': float(scores['silhouette_language']),
                'silhouette_query': float(scores['silhouette_query']),
                'difference': float(scores['difference'])
            }
            for layer, scores in sorted_layers[:5]
        ]
    }
    
    output_file = os.path.join(args.output_dir, 'best_layer_selection.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")
    
    # ========== 6. 可视化 ==========
    if not args.skip_visualization:
        print("\nGenerating visualization...")
        visualize_layer_scores(all_scores, args.output_dir, best_layer)
    
    # ========== 7. 总结 ==========
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Model:          {os.path.basename(args.model_name_or_path)}")
    print(f"Total Layers:   {num_layers}")
    print(f"Queries Used:   {len(selected_queries)}")
    print(f"Languages:      {list(lang_counts.keys())}")
    print(f"Token Position: {args.token_position}")
    print(f"")
    print(f"✅ Best Layer:  {best_layer}")
    print(f"   (This layer has the strongest semantic/query clustering)")
    print(f"")
    print(f"Output saved to: {args.output_dir}")
    
    return best_layer


if __name__ == "__main__":
    main()

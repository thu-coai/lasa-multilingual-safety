"""
实验评估脚本

评估训练集、测试集、OOD集的效果
"""

import os
import argparse
import json
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from collections import defaultdict
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

from model import TransformMLP
from dataset import MultilingualHiddenStateDataset, collate_fn


def load_dataset(data_path):
    """加载数据集"""
    data = torch.load(data_path, weights_only=False)
    
    return MultilingualHiddenStateDataset(
        hidden_states=data['hidden_states'],
        group_ids=data['group_ids'],
        safety_labels=data['safety_labels'],
        languages=data.get('languages'),
        texts=data.get('texts')
    ), data.get('sources', ['unknown'] * len(data['hidden_states']))


def get_embeddings(model, dataset, device, batch_size=64):
    """获取变换前后的embeddings"""
    model.eval()
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    original = []
    transformed = []
    group_ids = []
    safety_labels = []
    languages = []
    
    with torch.no_grad():
        for batch in dataloader:
            hidden_states = batch['hidden_states'].to(device)
            trans = model(hidden_states)
            
            original.append(hidden_states.cpu())
            transformed.append(trans.cpu())
            group_ids.extend(batch['group_ids'].tolist())
            safety_labels.extend(batch['safety_labels'].tolist())
            languages.extend(batch['languages'])
    
    return {
        'original': torch.cat(original, dim=0),
        'transformed': torch.cat(transformed, dim=0),
        'group_ids': group_ids,
        'safety_labels': safety_labels,
        'languages': languages
    }


def compute_metrics(embeddings, group_ids, safety_labels, languages, prefix=""):
    """计算评估指标"""
    # L2归一化
    emb = F.normalize(embeddings, p=2, dim=1)
    
    metrics = {}
    
    # 1. 跨语言对齐（同组内相似度）
    group_to_indices = defaultdict(list)
    for i, gid in enumerate(group_ids):
        group_to_indices[gid].append(i)
    
    intra_group_sims = []
    for gid, indices in group_to_indices.items():
        if len(indices) < 2:
            continue
        group_emb = emb[indices]
        sim_matrix = torch.matmul(group_emb, group_emb.T)
        mask = ~torch.eye(len(indices), dtype=torch.bool)
        intra_group_sims.extend(sim_matrix[mask].tolist())
    
    metrics[f'{prefix}intra_group_similarity'] = np.mean(intra_group_sims) if intra_group_sims else 0.0
    
    # 2. 安全性区分
    safe_indices = [i for i, l in enumerate(safety_labels) if l == 0]
    unsafe_indices = [i for i, l in enumerate(safety_labels) if l == 1]
    
    if safe_indices and unsafe_indices:
        safe_emb = emb[safe_indices]
        unsafe_emb = emb[unsafe_indices]
        
        # Safe-Unsafe相似度
        cross_sim = torch.matmul(safe_emb, unsafe_emb.T)
        metrics[f'{prefix}safe_unsafe_similarity'] = cross_sim.mean().item()
        
        # Safe-Safe相似度
        if len(safe_indices) >= 2:
            safe_sim = torch.matmul(safe_emb, safe_emb.T)
            mask = ~torch.eye(len(safe_indices), dtype=torch.bool)
            metrics[f'{prefix}safe_safe_similarity'] = safe_sim[mask].mean().item()
        else:
            metrics[f'{prefix}safe_safe_similarity'] = 0.0
        
        # Unsafe-Unsafe相似度
        if len(unsafe_indices) >= 2:
            unsafe_sim = torch.matmul(unsafe_emb, unsafe_emb.T)
            mask = ~torch.eye(len(unsafe_indices), dtype=torch.bool)
            metrics[f'{prefix}unsafe_unsafe_similarity'] = unsafe_sim[mask].mean().item()
        else:
            metrics[f'{prefix}unsafe_unsafe_similarity'] = 0.0
        
        # 安全性分离度
        avg_within = (metrics[f'{prefix}safe_safe_similarity'] + metrics[f'{prefix}unsafe_unsafe_similarity']) / 2
        metrics[f'{prefix}safety_separation_gap'] = avg_within - metrics[f'{prefix}safe_unsafe_similarity']
    else:
        metrics[f'{prefix}safe_unsafe_similarity'] = 0.0
        metrics[f'{prefix}safe_safe_similarity'] = 0.0
        metrics[f'{prefix}unsafe_unsafe_similarity'] = 0.0
        metrics[f'{prefix}safety_separation_gap'] = 0.0
    
    # 3. 按语言统计
    by_lang = {}
    for lang in sorted(set(languages)):
        lang_safe_indices = [i for i, (l, s) in enumerate(zip(languages, safety_labels)) if l == lang and s == 0]
        lang_unsafe_indices = [i for i, (l, s) in enumerate(zip(languages, safety_labels)) if l == lang and s == 1]
        
        lang_stats = {
            'safe_count': len(lang_safe_indices),
            'unsafe_count': len(lang_unsafe_indices)
        }
        
        if len(lang_safe_indices) >= 2:
            lang_safe_emb = emb[lang_safe_indices]
            sim = torch.matmul(lang_safe_emb, lang_safe_emb.T)
            mask = ~torch.eye(len(lang_safe_indices), dtype=torch.bool)
            lang_stats['safe_similarity'] = sim[mask].mean().item()
        else:
            lang_stats['safe_similarity'] = 0.0
        
        if len(lang_unsafe_indices) >= 2:
            lang_unsafe_emb = emb[lang_unsafe_indices]
            sim = torch.matmul(lang_unsafe_emb, lang_unsafe_emb.T)
            mask = ~torch.eye(len(lang_unsafe_indices), dtype=torch.bool)
            lang_stats['unsafe_similarity'] = sim[mask].mean().item()
        else:
            lang_stats['unsafe_similarity'] = 0.0
        
        if lang_safe_indices and lang_unsafe_indices:
            cross = torch.matmul(emb[lang_safe_indices], emb[lang_unsafe_indices].T)
            lang_stats['safe_unsafe_similarity'] = cross.mean().item()
        else:
            lang_stats['safe_unsafe_similarity'] = 0.0
        
        by_lang[lang] = lang_stats
    
    metrics[f'{prefix}by_language'] = by_lang
    
    return metrics


def compute_ood_metrics(train_emb, ood_emb, train_safety_labels, ood_group_ids, ood_languages):
    """计算OOD指标"""
    train_emb = F.normalize(train_emb, p=2, dim=1)
    ood_emb = F.normalize(ood_emb, p=2, dim=1)
    
    train_safe_indices = [i for i, l in enumerate(train_safety_labels) if l == 0]
    train_unsafe_indices = [i for i, l in enumerate(train_safety_labels) if l == 1]
    
    train_safe_emb = train_emb[train_safe_indices]
    train_unsafe_emb = train_emb[train_unsafe_indices]
    
    metrics = {}
    
    # OOD与训练集的相似度
    ood_safe_sim = torch.matmul(ood_emb, train_safe_emb.T).mean().item()
    ood_unsafe_sim = torch.matmul(ood_emb, train_unsafe_emb.T).mean().item()
    
    metrics['ood_train_safe_similarity'] = ood_safe_sim
    metrics['ood_train_unsafe_similarity'] = ood_unsafe_sim
    metrics['ood_safety_gap'] = ood_unsafe_sim - ood_safe_sim
    
    # OOD内部的跨语言对齐
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
    
    metrics['ood_intra_group_similarity'] = np.mean(intra_group_sims) if intra_group_sims else 0.0
    
    # 按语言统计
    by_lang = {}
    for lang in sorted(set(ood_languages)):
        lang_indices = [i for i, l in enumerate(ood_languages) if l == lang]
        if len(lang_indices) >= 2:
            lang_emb = ood_emb[lang_indices]
            
            # 与训练集safe的相似度
            safe_sim = torch.matmul(lang_emb, train_safe_emb.T).mean().item()
            # 与训练集unsafe的相似度
            unsafe_sim = torch.matmul(lang_emb, train_unsafe_emb.T).mean().item()
            
            by_lang[lang] = {
                'count': len(lang_indices),
                'train_safe_similarity': safe_sim,
                'train_unsafe_similarity': unsafe_sim,
                'safety_gap': unsafe_sim - safe_sim
            }
    
    metrics['ood_by_language'] = by_lang
    
    return metrics


def visualize_all(train_data, test_data, ood_data, output_dir, title_prefix=""):
    """可视化所有数据"""
    # 合并所有数据
    all_emb = torch.cat([train_data['transformed'], test_data['transformed'], ood_data['transformed']], dim=0)
    all_safety = train_data['safety_labels'] + test_data['safety_labels'] + ood_data['safety_labels']
    all_split = (['train'] * len(train_data['safety_labels']) + 
                 ['test'] * len(test_data['safety_labels']) +
                 ['ood'] * len(ood_data['safety_labels']))
    
    # t-SNE
    print("Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    emb_2d = tsne.fit_transform(all_emb.numpy())
    
    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 按安全性着色
    ax = axes[0]
    colors = ['blue' if s == 0 else 'red' for s in all_safety]
    ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=colors, alpha=0.5, s=10)
    ax.set_title(f'{title_prefix}By Safety (Blue=Safe, Red=Unsafe)', fontsize=14)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    
    # 按数据集着色
    ax = axes[1]
    split_colors = {'train': 'green', 'test': 'orange', 'ood': 'purple'}
    colors = [split_colors[s] for s in all_split]
    ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=colors, alpha=0.5, s=10)
    ax.set_title(f'{title_prefix}By Dataset (Green=Train, Orange=Test, Purple=OOD)', fontsize=14)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_embeddings_tsne.png'), dpi=300)
    plt.close()
    
    print(f"Saved t-SNE visualization to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate experiment")
    
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--test_data_path", type=str, required=True)
    parser.add_argument("--ood_data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据
    print("Loading datasets...")
    train_dataset, train_sources = load_dataset(args.train_data_path)
    test_dataset, test_sources = load_dataset(args.test_data_path)
    ood_dataset, ood_sources = load_dataset(args.ood_data_path)
    
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    print(f"  OOD: {len(ood_dataset)} samples")
    
    # 加载模型
    print(f"\nLoading model from {args.model_path}...")
    input_dim = train_dataset.hidden_states[0].shape[0]
    
    model = TransformMLP(
        input_dim=input_dim,
        hidden_dim=input_dim,
        output_dim=input_dim,
        num_layers=2,
        use_residual=True,
        use_layer_norm=True
    ).to(args.device)
    
    checkpoint = torch.load(args.model_path, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully")
    
    # 获取embeddings
    print("\nGetting embeddings...")
    train_data = get_embeddings(model, train_dataset, args.device, args.batch_size)
    test_data = get_embeddings(model, test_dataset, args.device, args.batch_size)
    ood_data = get_embeddings(model, ood_dataset, args.device, args.batch_size)
    
    # 计算指标
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    
    results = {}
    
    # 训练集指标
    print("\n--- Training Set ---")
    train_orig_metrics = compute_metrics(
        train_data['original'], train_data['group_ids'],
        train_data['safety_labels'], train_data['languages'], prefix=""
    )
    train_trans_metrics = compute_metrics(
        train_data['transformed'], train_data['group_ids'],
        train_data['safety_labels'], train_data['languages'], prefix=""
    )
    
    results['train'] = {
        'original': {k: v for k, v in train_orig_metrics.items() if not k.endswith('by_language')},
        'transformed': {k: v for k, v in train_trans_metrics.items() if not k.endswith('by_language')},
        'by_language_original': train_orig_metrics.get('by_language', {}),
        'by_language_transformed': train_trans_metrics.get('by_language', {})
    }
    
    print("Original:")
    for k, v in results['train']['original'].items():
        print(f"  {k}: {v:.4f}")
    print("Transformed:")
    for k, v in results['train']['transformed'].items():
        print(f"  {k}: {v:.4f}")
    
    # 测试集指标
    print("\n--- Test Set ---")
    test_orig_metrics = compute_metrics(
        test_data['original'], test_data['group_ids'],
        test_data['safety_labels'], test_data['languages'], prefix=""
    )
    test_trans_metrics = compute_metrics(
        test_data['transformed'], test_data['group_ids'],
        test_data['safety_labels'], test_data['languages'], prefix=""
    )
    
    results['test'] = {
        'original': {k: v for k, v in test_orig_metrics.items() if not k.endswith('by_language')},
        'transformed': {k: v for k, v in test_trans_metrics.items() if not k.endswith('by_language')},
        'by_language_original': test_orig_metrics.get('by_language', {}),
        'by_language_transformed': test_trans_metrics.get('by_language', {})
    }
    
    print("Original:")
    for k, v in results['test']['original'].items():
        print(f"  {k}: {v:.4f}")
    print("Transformed:")
    for k, v in results['test']['transformed'].items():
        print(f"  {k}: {v:.4f}")
    
    # OOD指标
    print("\n--- OOD Set (Multijail) ---")
    ood_orig_metrics = compute_ood_metrics(
        train_data['original'], ood_data['original'],
        train_data['safety_labels'], ood_data['group_ids'], ood_data['languages']
    )
    ood_trans_metrics = compute_ood_metrics(
        train_data['transformed'], ood_data['transformed'],
        train_data['safety_labels'], ood_data['group_ids'], ood_data['languages']
    )
    
    results['ood'] = {
        'original': {k: v for k, v in ood_orig_metrics.items() if not k.endswith('by_language')},
        'transformed': {k: v for k, v in ood_trans_metrics.items() if not k.endswith('by_language')},
        'by_language_original': ood_orig_metrics.get('ood_by_language', {}),
        'by_language_transformed': ood_trans_metrics.get('ood_by_language', {})
    }
    
    print("Original:")
    for k, v in results['ood']['original'].items():
        print(f"  {k}: {v:.4f}")
    print("Transformed:")
    for k, v in results['ood']['transformed'].items():
        print(f"  {k}: {v:.4f}")
    
    # 计算改进
    print("\n--- Improvements ---")
    print("Train:")
    for k in ['intra_group_similarity', 'safety_separation_gap']:
        if k in results['train']['original'] and k in results['train']['transformed']:
            imp = results['train']['transformed'][k] - results['train']['original'][k]
            print(f"  {k}: {imp:+.4f}")
    
    print("Test:")
    for k in ['intra_group_similarity', 'safety_separation_gap']:
        if k in results['test']['original'] and k in results['test']['transformed']:
            imp = results['test']['transformed'][k] - results['test']['original'][k]
            print(f"  {k}: {imp:+.4f}")
    
    print("OOD:")
    for k in ['ood_safety_gap', 'ood_intra_group_similarity']:
        if k in results['ood']['original'] and k in results['ood']['transformed']:
            imp = results['ood']['transformed'][k] - results['ood']['original'][k]
            print(f"  {k}: {imp:+.4f}")
    
    # 保存结果
    with open(os.path.join(args.output_dir, 'experiment_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {os.path.join(args.output_dir, 'experiment_results.json')}")
    
    # 可视化
    print("\nGenerating visualizations...")
    visualize_all(train_data, test_data, ood_data, args.output_dir, title_prefix="Transformed ")
    
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()


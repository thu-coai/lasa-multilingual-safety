"""
Train LASA's Safety Semantic Interpreter (SSI) on one bottleneck layer.

The base LLM is frozen. This script extracts prompt hidden states from the
selected semantic bottleneck layer and trains a lightweight MLP to classify
benign vs. harmful semantics across languages.

Recommended usage:
    MODEL_PATH=/path/to/model LAYER_IDX=14 bash scripts/run_train_safety_classifier.sh

Direct usage:
    python train_safety_classifier.py \
        --model_path /path/to/model \
        --layer_idx 15 \
        --classifier_hidden_dim 256 \
        --ultrafeedback_dir data/ultrafeedback \
        --safety_data_path data/ssi/unsafe_train.json \
        --output_dir ./output
"""

import os
import argparse
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForCausalLM

# 从 cross_lingual_layer_analysis 导入共享组件
from cross_lingual_layer_analysis import (
    SimpleMLP,
    DeepMLP,
    HiddenStateDataset,
    collate_fn,
    load_ultrafeedback_data,
    load_safety_data,
    load_harmbench_data,
    train_classifier,
    evaluate_classifier,
    evaluate_per_language,
    find_optimal_threshold
)


def extract_hidden_states_single_layer(
    model,
    tokenizer,
    texts: List[str],
    layer_idx: int,
    device: str = 'cuda',
    batch_size: int = 8,
    max_length: int = 512
) -> List[torch.Tensor]:
    """
    提取指定层的 hidden states
    
    Args:
        model: 预训练模型
        tokenizer: tokenizer
        texts: 文本列表
        layer_idx: 层索引
        device: 设备
        batch_size: 批大小
        max_length: 最大序列长度
    
    Returns:
        该层的 hidden states 列表
    """
    model.eval()
    hidden_states = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc=f"Extracting layer {layer_idx} hidden states"):
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
        
        # 获取指定层的 hidden states
        layer_output = outputs.hidden_states[layer_idx]
        
        for j in range(len(batch_texts)):
            attention_mask = inputs['attention_mask'][j]
            last_token_idx = attention_mask.sum() - 1
            
            # 使用最后一个有效 token 的 hidden state
            hidden_state = layer_output[j, last_token_idx, :].cpu()
            hidden_states.append(hidden_state)
    
    return hidden_states


def train_single_layer_classifier(
    hidden_states: List[torch.Tensor],
    safety_labels: List[int],
    all_data: List[Dict],
    layer_idx: int,
    output_dir: str,
    ood_hidden_states: List[torch.Tensor] = None,
    ood_data: List[Dict] = None,
    device: str = 'cuda',
    classifier_hidden_dim: int = 256,
    train_epochs: int = 20,
    val_split: float = 0.2,
    seed: int = 42,
    train_languages: List[str] = None,
    test_languages: List[str] = None,
    use_deep_mlp: bool = True,
    classifier_dropout: float = 0.2
) -> Dict:
    """
    在单层上训练安全分类器
    
    Args:
        hidden_states: 该层的 hidden states 列表
        safety_labels: 安全标签列表 (0=safe, 1=unsafe)
        all_data: 数据列表
        layer_idx: 层索引
        output_dir: 输出目录
        ood_hidden_states: OOD 数据的 hidden states
        ood_data: OOD 数据列表
        device: 设备
        classifier_hidden_dim: 分类器隐藏层维度
        train_epochs: 训练轮数
        val_split: 验证集比例
        seed: 随机种子
        train_languages: 训练语言列表
        test_languages: 测试语言列表
        use_deep_mlp: 是否使用 DeepMLP
        classifier_dropout: Dropout 率
    
    Returns:
        训练结果字典
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    os.makedirs(output_dir, exist_ok=True)
    
    hidden_dim = hidden_states[0].shape[0]
    
    print(f"\n{'='*60}")
    print(f"Training Safety Classifier on Layer {layer_idx}")
    print(f"{'='*60}")
    print(f"Total samples: {len(all_data)}")
    print(f"Model hidden dimension: {hidden_dim}")
    print(f"Classifier hidden dimension: {classifier_hidden_dim}")
    
    has_ood = ood_hidden_states is not None and len(ood_hidden_states) > 0
    if has_ood:
        print(f"OOD samples: {len(ood_data)}")
    
    # 获取语言列表
    languages = sorted(list(set(item['language'] for item in all_data)))
    num_languages = len(languages)
    print(f"Languages ({num_languages}): {languages}")
    
    # 按语言分割训练/测试集
    if train_languages is not None or test_languages is not None:
        train_langs = set(train_languages) if train_languages else set(languages)
        test_langs = set(test_languages) if test_languages else set(languages)
        
        train_indices = [i for i, item in enumerate(all_data) if item['language'] in train_langs]
        val_indices = [i for i, item in enumerate(all_data) if item['language'] in test_langs]
        
        np.random.shuffle(train_indices)
        inner_val_size = int(len(train_indices) * 0.1)
        inner_val_indices = train_indices[:inner_val_size]
        train_indices = train_indices[inner_val_size:]
        
        print(f"\n>>> Cross-lingual generalization mode:")
        print(f"    Train languages: {sorted(train_langs)}")
        print(f"    Test languages: {sorted(test_langs)}")
        print(f"\nTrain samples: {len(train_indices)}")
        print(f"Inner val samples: {len(inner_val_indices)}")
        print(f"Test samples: {len(val_indices)}")
        
        use_cross_lingual = True
    else:
        indices = list(range(len(all_data)))
        np.random.shuffle(indices)
        
        val_size = int(len(indices) * val_split)
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]
        inner_val_indices = None
        
        print(f"\nTrain samples: {len(train_indices)}")
        print(f"Val samples: {len(val_indices)}")
        
        use_cross_lingual = False
    
    # 分割数据
    train_hs = [hidden_states[i] for i in train_indices]
    test_hs = [hidden_states[i] for i in val_indices]
    
    train_safety = [safety_labels[i] for i in train_indices]
    test_safety = [safety_labels[i] for i in val_indices]
    
    if use_cross_lingual and inner_val_indices is not None:
        inner_val_hs = [hidden_states[i] for i in inner_val_indices]
        inner_val_safety = [safety_labels[i] for i in inner_val_indices]
    else:
        inner_val_hs = test_hs
        inner_val_safety = test_safety
    
    # 创建 DataLoader
    safety_train_dataset = HiddenStateDataset(train_hs, train_safety)
    safety_val_dataset = HiddenStateDataset(inner_val_hs, inner_val_safety)
    
    safety_train_loader = DataLoader(
        safety_train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn
    )
    safety_val_loader = DataLoader(
        safety_val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn
    )
    
    # 创建分类器
    if use_deep_mlp:
        classifier = DeepMLP(
            input_dim=hidden_dim,
            hidden_dims=[classifier_hidden_dim, classifier_hidden_dim // 2],
            output_dim=1,
            dropout=classifier_dropout,
            use_layer_norm=True,
            use_residual=False,
            activation='gelu'
        )
        print(f"\nUsing DeepMLP: {hidden_dim} -> {classifier_hidden_dim} -> {classifier_hidden_dim//2} -> 1")
    else:
        classifier = SimpleMLP(
            input_dim=hidden_dim,
            hidden_dim=classifier_hidden_dim,
            output_dim=1,
            dropout=classifier_dropout
        )
        print(f"\nUsing SimpleMLP: {hidden_dim} -> {classifier_hidden_dim} -> 1")
    
    # 训练
    print(f"\nTraining classifier...")
    train_result = train_classifier(
        classifier, safety_train_loader, safety_val_loader,
        num_classes=1, device=device, num_epochs=train_epochs, verbose=True,
        pos_weight=None, optimize_threshold=True
    )
    
    best_threshold = train_result.get('best_threshold', 0.5)
    val_acc = train_result['best_val_acc']
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {val_acc:.4f}")
    print(f"Optimal threshold: {best_threshold:.3f}")
    
    # 在测试集上评估
    test_safety_dataset = HiddenStateDataset(test_hs, test_safety)
    test_safety_loader = DataLoader(test_safety_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    
    test_result = evaluate_classifier(
        classifier, test_safety_loader, num_classes=1, device=device, threshold=best_threshold
    )
    
    print(f"\n{'='*60}")
    print(f"Test Set Results")
    print(f"{'='*60}")
    print(f"Test Accuracy: {test_result['accuracy']:.4f}")
    print(f"Test F1: {test_result['f1']:.4f}")
    print(f"Test TPR: {test_result['tpr']:.4f}")
    print(f"Test FPR: {test_result['fpr']:.4f}")
    
    # 每个语言的详细评估
    per_lang_results = evaluate_per_language(
        classifier, hidden_states, safety_labels, all_data,
        device=device, threshold=best_threshold
    )
    
    print(f"\n{'='*60}")
    print(f"Per-Language Results")
    print(f"{'='*60}")
    print(f"{'Language':<10} {'Accuracy':>10} {'FPR':>10} {'TPR':>10} {'Safe':>8} {'Unsafe':>8}")
    print("-" * 60)
    for lang in sorted(per_lang_results.keys()):
        info = per_lang_results[lang]
        print(f"{lang:<10} {info['accuracy']:>10.4f} {info['fpr']:>10.4f} {info['tpr']:>10.4f} {info['num_safe']:>8} {info['num_unsafe']:>8}")
    
    avg_fpr = np.mean([v['fpr'] for v in per_lang_results.values() if v['num_safe'] > 0])
    avg_tpr = np.mean([v['tpr'] for v in per_lang_results.values() if v['num_unsafe'] > 0])
    
    print("-" * 60)
    print(f"{'Average':<10} {'':<10} {avg_fpr:>10.4f} {avg_tpr:>10.4f}")
    
    # OOD 评估
    ood_accuracy = 0.0
    ood_per_lang = {}
    if has_ood:
        ood_labels = [1] * len(ood_hidden_states)
        ood_dataset = HiddenStateDataset(ood_hidden_states, ood_labels)
        ood_loader = DataLoader(ood_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
        
        ood_result = evaluate_classifier(
            classifier, ood_loader, num_classes=1, device=device, threshold=best_threshold
        )
        ood_accuracy = ood_result['accuracy']
        
        ood_per_lang = evaluate_per_language(
            classifier, ood_hidden_states, ood_labels, ood_data,
            device=device, threshold=best_threshold
        )
        
        print(f"\n{'='*60}")
        print(f"OOD (Out-of-Distribution) Results")
        print(f"{'='*60}")
        print(f"Overall OOD Accuracy: {ood_accuracy:.4f}")
        print(f"\n{'Language':<10} {'Accuracy':>10} {'Total':>10}")
        print("-" * 35)
        for lang in sorted(ood_per_lang.keys()):
            info = ood_per_lang[lang]
            print(f"{lang:<10} {info['accuracy']:>10.4f} {info['total']:>10}")
        
        ood_avg_acc = np.mean([v['accuracy'] for v in ood_per_lang.values()])
        print("-" * 35)
        print(f"{'Average':<10} {ood_avg_acc:>10.4f}")
    
    # 保存模型
    threshold_int = int(round(best_threshold * 100))
    threshold_str = f"{threshold_int:03d}"
    model_save_path = os.path.join(
        output_dir,
        f'classifier_layer{layer_idx}_hidden{classifier_hidden_dim}_threshold{threshold_str}.pt'
    )
    
    torch.save({
        'model_state_dict': classifier.state_dict(),
        'layer_idx': layer_idx,
        'hidden_dim': hidden_dim,
        'classifier_hidden_dim': classifier_hidden_dim,
        'val_acc': val_acc,
        'test_acc': test_result['accuracy'],
        'threshold': best_threshold,
        'avg_fpr': avg_fpr,
        'avg_tpr': avg_tpr,
        'ood_accuracy': ood_accuracy,
        'use_deep_mlp': use_deep_mlp,
        'classifier_dropout': classifier_dropout,
        'train_languages': list(train_languages) if train_languages else None,
        'test_languages': list(test_languages) if test_languages else None
    }, model_save_path)
    print(f"\nModel saved to {model_save_path}")
    
    # 保存详细结果
    final_summary = {
        'model_path': model_save_path,
        'layer_idx': layer_idx,
        'classifier_hidden_dim': classifier_hidden_dim,
        'use_deep_mlp': use_deep_mlp,
        'classifier_dropout': classifier_dropout,
        'threshold': best_threshold,
        'in_distribution': {
            'val_accuracy': val_acc,
            'test_accuracy': test_result['accuracy'],
            'avg_fpr': avg_fpr,
            'avg_tpr': avg_tpr,
            'per_lang_accuracy': {lang: info['accuracy'] for lang, info in per_lang_results.items()},
            'per_lang_fpr': {lang: info['fpr'] for lang, info in per_lang_results.items()},
            'per_lang_tpr': {lang: info['tpr'] for lang, info in per_lang_results.items()}
        },
        'out_of_distribution': {
            'overall_accuracy': ood_accuracy,
            'per_lang_accuracy': {lang: info['accuracy'] for lang, info in ood_per_lang.items()} if ood_per_lang else {}
        }
    }
    
    summary_path = os.path.join(output_dir, 'final_model_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(final_summary, f, indent=2)
    print(f"Summary saved to {summary_path}")
    
    # 打印最终汇总
    print(f"\n{'='*70}")
    print(f"FINAL MODEL PERFORMANCE SUMMARY")
    print(f"{'='*70}")
    print(f"Layer: {layer_idx} | Hidden Dim: {classifier_hidden_dim} | Threshold: {best_threshold:.3f}")
    print(f"\n[IN-DISTRIBUTION]")
    print(f"Val Accuracy: {val_acc:.4f} | Test Accuracy: {test_result['accuracy']:.4f}")
    print(f"Avg FPR: {avg_fpr:.4f} | Avg TPR: {avg_tpr:.4f}")
    if has_ood:
        print(f"\n[OUT-OF-DISTRIBUTION]")
        print(f"OOD Accuracy: {ood_accuracy:.4f}")
    print(f"{'='*70}")
    
    return {
        'layer_idx': layer_idx,
        'classifier_hidden_dim': classifier_hidden_dim,
        'threshold': best_threshold,
        'val_acc': val_acc,
        'test_acc': test_result['accuracy'],
        'avg_fpr': avg_fpr,
        'avg_tpr': avg_tpr,
        'ood_accuracy': ood_accuracy,
        'per_lang_results': per_lang_results,
        'ood_per_lang': ood_per_lang,
        'model_save_path': model_save_path
    }


def main():
    parser = argparse.ArgumentParser(description="Train Safety Classifier on Specified Layer")
    
    # 模型参数
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to pretrained model")
    
    # 层选择参数
    parser.add_argument("--layer_idx", type=int, required=True,
                        help="Layer index to train classifier on")
    
    # 数据参数
    parser.add_argument("--ultrafeedback_dir", type=str, required=True,
                        help="Path to ultrafeedback data directory")
    parser.add_argument("--safety_data_path", type=str, required=True,
                        help="Path to safety data file")
    parser.add_argument("--ood_data_path", type=str, default=None,
                        help="Path to OOD data file (e.g. multijail)")
    parser.add_argument("--additional_unsafe_paths", type=str, nargs='*', default=[],
                        help="Additional unsafe data paths for training")
    parser.add_argument("--languages", type=str, nargs="+", 
                        default=['en', 'zh', 'ar', 'it', 'ko', 'vi'],
                        help="All languages to use")
    parser.add_argument("--train_languages", type=str, nargs="+", default=None,
                        help="Languages to use for training (default: all)")
    parser.add_argument("--test_languages", type=str, nargs="+", default=None,
                        help="Languages to use for testing (default: all)")
    parser.add_argument("--max_safe_per_lang", type=int, default=200,
                        help="Max safe samples per language")
    parser.add_argument("--max_unsafe", type=int, default=0,
                        help="Max unsafe samples (0 = all)")
    parser.add_argument("--max_ood", type=int, default=200,
                        help="Max OOD samples")
    
    # 分类器参数
    parser.add_argument("--classifier_hidden_dim", type=int, default=256,
                        help="Hidden dimension for classifier")
    parser.add_argument("--use_deep_mlp", action="store_true", default=True,
                        help="Use DeepMLP (2 hidden layers)")
    parser.add_argument("--use_simple_mlp", action="store_true",
                        help="Use SimpleMLP (1 hidden layer)")
    parser.add_argument("--classifier_dropout", type=float, default=0.2,
                        help="Dropout rate for classifier")
    parser.add_argument("--train_epochs", type=int, default=20,
                        help="Training epochs")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Validation split ratio")
    
    # 其他参数
    parser.add_argument("--output_dir", type=str, default="./output/single_layer_training",
                        help="Output directory")
    parser.add_argument("--hidden_states_cache", type=str, default=None,
                        help="Path to cached hidden states file")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for hidden state extraction")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"layer{args.layer_idx}_hidden{args.classifier_hidden_dim}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存配置
    config = vars(args)
    config['timestamp'] = timestamp
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # 检查是否有缓存
    cache_path = args.hidden_states_cache
    cached_data = None
    
    if cache_path and os.path.exists(cache_path):
        print(f"\n>>> Loading cached hidden states from {cache_path}")
        cached_data = torch.load(cache_path, weights_only=False)
        
        safe_data = cached_data['safe_data']
        unsafe_data = cached_data['unsafe_data']
        ood_data = cached_data.get('ood_data')
        
        # 检查是否有该层的 hidden states
        if 'all_layer_hidden_states' in cached_data:
            all_layer_hs = cached_data['all_layer_hidden_states']
            if args.layer_idx in all_layer_hs:
                hidden_states = all_layer_hs[args.layer_idx]
                print(f"Loaded layer {args.layer_idx} hidden states from cache")
            else:
                print(f"Layer {args.layer_idx} not found in cache, will extract...")
                hidden_states = None
        else:
            hidden_states = None
        
        # OOD hidden states
        ood_hidden_states = None
        if ood_data and 'ood_layer_hidden_states' in cached_data:
            ood_layer_hs = cached_data['ood_layer_hidden_states']
            if ood_layer_hs and args.layer_idx in ood_layer_hs:
                ood_hidden_states = ood_layer_hs[args.layer_idx]
                print(f"Loaded OOD layer {args.layer_idx} hidden states from cache")
        
        print(f"Loaded {len(safe_data)} safe, {len(unsafe_data)} unsafe samples")
        if ood_data:
            print(f"Loaded {len(ood_data)} OOD samples")
        
        # 如果需要提取 hidden states
        if hidden_states is None:
            print(f"\nLoading model to extract layer {args.layer_idx} hidden states...")
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16,
                device_map=args.device,
                trust_remote_code=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            all_data = safe_data + unsafe_data
            all_texts = [item['prompt'] for item in all_data]
            
            hidden_states = extract_hidden_states_single_layer(
                model, tokenizer, all_texts,
                layer_idx=args.layer_idx,
                device=args.device, batch_size=args.batch_size
            )
            
            if ood_data and ood_hidden_states is None:
                ood_texts = [item['prompt'] for item in ood_data]
                ood_hidden_states = extract_hidden_states_single_layer(
                    model, tokenizer, ood_texts,
                    layer_idx=args.layer_idx,
                    device=args.device, batch_size=args.batch_size
                )
            
            del model
            del tokenizer
            torch.cuda.empty_cache()
    else:
        # 加载模型
        print(f"\nLoading model from {args.model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map=args.device,
            trust_remote_code=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 加载数据
        print("\nLoading data...")
        safe_data = load_ultrafeedback_data(
            args.ultrafeedback_dir,
            languages=args.languages,
            max_samples_per_lang=args.max_safe_per_lang
        )
        
        unsafe_data = load_safety_data(
            args.safety_data_path,
            max_samples=args.max_unsafe
        )
        
        # 加载额外有害数据
        if args.additional_unsafe_paths:
            for additional_path in args.additional_unsafe_paths:
                if os.path.exists(additional_path):
                    additional_data = load_harmbench_data(additional_path, max_samples=None)
                    unsafe_data.extend(additional_data)
                    print(f"Loaded {len(additional_data)} additional unsafe samples from {additional_path}")
        
        # 筛选语言
        if args.languages:
            unsafe_data = [item for item in unsafe_data if item['language'] in args.languages]
        
        print(f"Loaded {len(safe_data)} safe samples, {len(unsafe_data)} unsafe samples")
        
        # 加载 OOD 数据
        ood_data = None
        ood_hidden_states = None
        if args.ood_data_path and os.path.exists(args.ood_data_path):
            max_unsafe_idx = max(item['idx'] for item in unsafe_data) if unsafe_data else 0
            ood_data = load_harmbench_data(
                args.ood_data_path,
                max_samples=args.max_ood,
                idx_offset=max_unsafe_idx + 10000
            )
            
            if args.languages:
                ood_data = [item for item in ood_data if item['language'] in args.languages]
            
            print(f"Loaded {len(ood_data)} OOD samples")
        
        # 提取 hidden states
        all_data = safe_data + unsafe_data
        all_texts = [item['prompt'] for item in all_data]
        
        print(f"\nExtracting layer {args.layer_idx} hidden states...")
        hidden_states = extract_hidden_states_single_layer(
            model, tokenizer, all_texts,
            layer_idx=args.layer_idx,
            device=args.device, batch_size=args.batch_size
        )
        
        # 提取 OOD hidden states
        if ood_data:
            ood_texts = [item['prompt'] for item in ood_data]
            print(f"Extracting OOD layer {args.layer_idx} hidden states...")
            ood_hidden_states = extract_hidden_states_single_layer(
                model, tokenizer, ood_texts,
                layer_idx=args.layer_idx,
                device=args.device, batch_size=args.batch_size
            )
        
        # 释放模型内存
        del model
        del tokenizer
        torch.cuda.empty_cache()
    
    # 准备数据
    all_data = safe_data + unsafe_data
    safety_labels = [0 if not item['is_unsafe'] else 1 for item in all_data]
    
    # 决定是否使用 DeepMLP
    use_deep = args.use_deep_mlp and not args.use_simple_mlp
    print(f"\nClassifier type: {'DeepMLP (2 hidden layers)' if use_deep else 'SimpleMLP (1 hidden layer)'}")
    print(f"Classifier hidden dim: {args.classifier_hidden_dim}")
    print(f"Dropout: {args.classifier_dropout}")
    
    # 训练分类器
    results = train_single_layer_classifier(
        hidden_states=hidden_states,
        safety_labels=safety_labels,
        all_data=all_data,
        layer_idx=args.layer_idx,
        output_dir=output_dir,
        ood_hidden_states=ood_hidden_states,
        ood_data=ood_data,
        device=args.device,
        classifier_hidden_dim=args.classifier_hidden_dim,
        train_epochs=args.train_epochs,
        val_split=args.val_split,
        seed=args.seed,
        train_languages=args.train_languages,
        test_languages=args.test_languages,
        use_deep_mlp=use_deep,
        classifier_dropout=args.classifier_dropout
    )
    
    print(f"\nTraining completed! Results saved to {output_dir}")


if __name__ == "__main__":
    main()

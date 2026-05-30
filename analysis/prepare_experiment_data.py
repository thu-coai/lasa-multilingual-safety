"""
实验数据准备脚本

精细控制训练/测试/OOD数据的划分
"""

import os
import argparse
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from dataset import (
    extract_hidden_states,
    MultilingualHiddenStateDataset
)


def load_ultrafeedback_split(
    data_dir: str,
    languages: list,
    train_samples: int,
    test_samples: int
):
    """
    加载ultrafeedback数据并划分训练/测试集
    
    按idx排序，前train_samples作为训练集，接下来test_samples作为测试集
    """
    train_data = []
    test_data = []
    
    for lang in languages:
        file_path = os.path.join(data_dir, f'monolingual_first1000_{lang}.json')
        
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lang_data = json.load(f)
        
        # 按idx排序
        lang_data = sorted(lang_data, key=lambda x: x['idx'])
        
        # 划分训练集和测试集
        for item in lang_data[:train_samples]:
            train_data.append({
                'idx': item['idx'],
                'prompt': item['prompt'],
                'language': lang,
                'source': 'ultrafeedback'
            })
        
        for item in lang_data[train_samples:train_samples + test_samples]:
            test_data.append({
                'idx': item['idx'],
                'prompt': item['prompt'],
                'language': lang,
                'source': 'ultrafeedback'
            })
    
    return train_data, test_data


def load_safety_split(
    data_path: str,
    train_samples: int,
    test_samples: int,
    source_name: str = 'safety'
):
    """
    加载safety/harmbench数据并划分训练/测试集
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # 按idx排序
    raw_data = sorted(raw_data, key=lambda x: x['idx'])
    
    train_data = []
    test_data = []
    
    # 语言代码映射
    lang_mapping = {
        'chinese': 'zh', 'arabic': 'ar', 'bengali': 'bn',
        'italian': 'it', 'javanese': 'jw', 'korean': 'ko',
        'swahili': 'sw', 'thai': 'th', 'vietnamese': 'vi'
    }
    
    # 训练集
    for item in raw_data[:train_samples]:
        idx = item['idx']
        
        # 英文原文
        train_data.append({
            'idx': idx,
            'prompt': item['original_query'],
            'language': 'en',
            'source': source_name
        })
        
        # 翻译版本
        for lang, translation in item.get('translations', {}).items():
            lang_code = lang_mapping.get(lang.lower(), lang.lower())
            train_data.append({
                'idx': idx,
                'prompt': translation,
                'language': lang_code,
                'source': source_name
            })
    
    # 测试集
    for item in raw_data[train_samples:train_samples + test_samples]:
        idx = item['idx']
        
        test_data.append({
            'idx': idx,
            'prompt': item['original_query'],
            'language': 'en',
            'source': source_name
        })
        
        for lang, translation in item.get('translations', {}).items():
            lang_code = lang_mapping.get(lang.lower(), lang.lower())
            test_data.append({
                'idx': idx,
                'prompt': translation,
                'language': lang_code,
                'source': source_name
            })
    
    return train_data, test_data


def load_multijail_data(data_path: str, max_samples: int = None):
    """
    加载multijail数据作为OOD测试集
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # multijail的id是按语言重复的，需要重新组织
    # id // 10 表示问题编号，id % 10 表示语言
    data = []
    
    # 按问题编号排序
    raw_data = sorted(raw_data, key=lambda x: (x['id'] // 10, x['id'] % 10))
    
    # 只取前max_samples个不同的问题
    if max_samples:
        seen_questions = set()
        filtered_data = []
        for item in raw_data:
            question_id = item['id'] // 10
            if question_id < max_samples:
                filtered_data.append(item)
        raw_data = filtered_data
    
    for item in raw_data:
        data.append({
            'idx': item['id'] // 10,  # 问题编号
            'prompt': item['prompt'],
            'language': item['lingual'],
            'source': 'multijail'
        })
    
    return data


def create_dataset(
    data: list,
    model,
    tokenizer,
    layer_idx: int,
    device: str,
    batch_size: int,
    is_safe: bool,
    idx_offset: int = 0
):
    """
    从数据列表创建数据集
    """
    texts = [item['prompt'] for item in data]
    
    hidden_states = extract_hidden_states(
        model, tokenizer, texts,
        layer_idx=layer_idx,
        device=device,
        batch_size=batch_size
    )
    
    group_ids = [item['idx'] + idx_offset for item in data]
    safety_labels = [0 if is_safe else 1 for _ in data]
    languages = [item['language'] for item in data]
    sources = [item.get('source', 'unknown') for item in data]
    
    return hidden_states, group_ids, safety_labels, languages, texts, sources


def main():
    parser = argparse.ArgumentParser(description="Prepare experiment dataset")
    
    # 模型参数
    parser.add_argument("--model_name_or_path", type=str, required=True)
    
    # 数据路径
    parser.add_argument("--ultrafeedback_dir", type=str, required=True)
    parser.add_argument("--safety_data_path", type=str, required=True)
    parser.add_argument("--harmbench_path", type=str, required=True)
    parser.add_argument("--multijail_path", type=str, required=True)
    
    # 样本数量
    parser.add_argument("--train_samples", type=int, default=100,
                        help="每个来源的训练样本数")
    parser.add_argument("--test_samples", type=int, default=100,
                        help="每个来源的测试样本数")
    parser.add_argument("--ood_samples", type=int, default=100,
                        help="OOD测试样本数（问题数）")
    
    # 提取参数
    parser.add_argument("--layer_idx", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    
    # 输出路径
    parser.add_argument("--train_output", type=str, required=True)
    parser.add_argument("--test_output", type=str, required=True)
    parser.add_argument("--ood_output", type=str, required=True)
    
    args = parser.parse_args()
    
    languages = ['en', 'zh', 'ar', 'bn', 'it', 'jw', 'ko', 'sw', 'th', 'vi']
    
    print("=" * 60)
    print("Experiment Data Preparation")
    print("=" * 60)
    
    # 加载模型
    print(f"\nLoading model from {args.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    
    # ========== 1. 加载并划分数据 ==========
    print("\n--- Loading and splitting data ---")
    
    # Safe数据 (ultrafeedback)
    print(f"Loading ultrafeedback (safe) data...")
    safe_train, safe_test = load_ultrafeedback_split(
        args.ultrafeedback_dir, languages,
        args.train_samples, args.test_samples
    )
    print(f"  Train: {len(safe_train)}, Test: {len(safe_test)}")
    
    # Unsafe数据 (safety_train)
    print(f"Loading safety_train (unsafe) data...")
    safety_train, safety_test = load_safety_split(
        args.safety_data_path,
        args.train_samples, args.test_samples,
        source_name='safety'
    )
    print(f"  Train: {len(safety_train)}, Test: {len(safety_test)}")
    
    # Harmbench数据 (unsafe)
    print(f"Loading harmbench (unsafe) data...")
    harmbench_train, harmbench_test = load_safety_split(
        args.harmbench_path,
        args.train_samples, args.test_samples,
        source_name='harmbench'
    )
    print(f"  Train: {len(harmbench_train)}, Test: {len(harmbench_test)}")
    
    # Multijail数据 (OOD unsafe)
    print(f"Loading multijail (OOD) data...")
    multijail_data = load_multijail_data(args.multijail_path, args.ood_samples)
    print(f"  OOD: {len(multijail_data)}")
    
    # ========== 2. 提取hidden states ==========
    print(f"\n--- Extracting hidden states from layer {args.layer_idx} ---")
    
    # 计算idx偏移，避免不同数据源的idx冲突
    safe_offset = 0
    safety_offset = 10000
    harmbench_offset = 20000
    multijail_offset = 30000
    
    # 训练集
    print("\nProcessing training data...")
    train_hidden = []
    train_group_ids = []
    train_safety_labels = []
    train_languages = []
    train_texts = []
    train_sources = []
    
    # Safe训练数据
    print("  - Safe data...")
    h, g, s, l, t, src = create_dataset(
        safe_train, model, tokenizer, args.layer_idx, args.device,
        args.batch_size, is_safe=True, idx_offset=safe_offset
    )
    train_hidden.extend(h)
    train_group_ids.extend(g)
    train_safety_labels.extend(s)
    train_languages.extend(l)
    train_texts.extend(t)
    train_sources.extend(src)
    
    # Safety训练数据
    print("  - Safety (unsafe) data...")
    h, g, s, l, t, src = create_dataset(
        safety_train, model, tokenizer, args.layer_idx, args.device,
        args.batch_size, is_safe=False, idx_offset=safety_offset
    )
    train_hidden.extend(h)
    train_group_ids.extend(g)
    train_safety_labels.extend(s)
    train_languages.extend(l)
    train_texts.extend(t)
    train_sources.extend(src)
    
    # Harmbench训练数据
    print("  - Harmbench (unsafe) data...")
    h, g, s, l, t, src = create_dataset(
        harmbench_train, model, tokenizer, args.layer_idx, args.device,
        args.batch_size, is_safe=False, idx_offset=harmbench_offset
    )
    train_hidden.extend(h)
    train_group_ids.extend(g)
    train_safety_labels.extend(s)
    train_languages.extend(l)
    train_texts.extend(t)
    train_sources.extend(src)
    
    # 测试集
    print("\nProcessing test data...")
    test_hidden = []
    test_group_ids = []
    test_safety_labels = []
    test_languages = []
    test_texts = []
    test_sources = []
    
    print("  - Safe data...")
    h, g, s, l, t, src = create_dataset(
        safe_test, model, tokenizer, args.layer_idx, args.device,
        args.batch_size, is_safe=True, idx_offset=safe_offset + 1000
    )
    test_hidden.extend(h)
    test_group_ids.extend(g)
    test_safety_labels.extend(s)
    test_languages.extend(l)
    test_texts.extend(t)
    test_sources.extend(src)
    
    print("  - Safety (unsafe) data...")
    h, g, s, l, t, src = create_dataset(
        safety_test, model, tokenizer, args.layer_idx, args.device,
        args.batch_size, is_safe=False, idx_offset=safety_offset + 1000
    )
    test_hidden.extend(h)
    test_group_ids.extend(g)
    test_safety_labels.extend(s)
    test_languages.extend(l)
    test_texts.extend(t)
    test_sources.extend(src)
    
    print("  - Harmbench (unsafe) data...")
    h, g, s, l, t, src = create_dataset(
        harmbench_test, model, tokenizer, args.layer_idx, args.device,
        args.batch_size, is_safe=False, idx_offset=harmbench_offset + 1000
    )
    test_hidden.extend(h)
    test_group_ids.extend(g)
    test_safety_labels.extend(s)
    test_languages.extend(l)
    test_texts.extend(t)
    test_sources.extend(src)
    
    # OOD数据
    print("\nProcessing OOD data (multijail)...")
    h, g, s, l, t, src = create_dataset(
        multijail_data, model, tokenizer, args.layer_idx, args.device,
        args.batch_size, is_safe=False, idx_offset=multijail_offset
    )
    ood_hidden = h
    ood_group_ids = g
    ood_safety_labels = s
    ood_languages = l
    ood_texts = t
    ood_sources = src
    
    # ========== 3. 保存数据集 ==========
    print("\n--- Saving datasets ---")
    
    os.makedirs(os.path.dirname(args.train_output), exist_ok=True)
    
    # 保存训练集
    torch.save({
        'hidden_states': train_hidden,
        'group_ids': train_group_ids,
        'safety_labels': train_safety_labels,
        'languages': train_languages,
        'texts': train_texts,
        'sources': train_sources,
        'layer_idx': args.layer_idx,
        'model_name': args.model_name_or_path
    }, args.train_output)
    print(f"Train dataset saved to {args.train_output}")
    print(f"  Total: {len(train_hidden)} samples")
    print(f"  Safe: {sum(1 for l in train_safety_labels if l == 0)}")
    print(f"  Unsafe: {sum(1 for l in train_safety_labels if l == 1)}")
    
    # 保存测试集
    torch.save({
        'hidden_states': test_hidden,
        'group_ids': test_group_ids,
        'safety_labels': test_safety_labels,
        'languages': test_languages,
        'texts': test_texts,
        'sources': test_sources,
        'layer_idx': args.layer_idx,
        'model_name': args.model_name_or_path
    }, args.test_output)
    print(f"\nTest dataset saved to {args.test_output}")
    print(f"  Total: {len(test_hidden)} samples")
    print(f"  Safe: {sum(1 for l in test_safety_labels if l == 0)}")
    print(f"  Unsafe: {sum(1 for l in test_safety_labels if l == 1)}")
    
    # 保存OOD数据集
    torch.save({
        'hidden_states': ood_hidden,
        'group_ids': ood_group_ids,
        'safety_labels': ood_safety_labels,
        'languages': ood_languages,
        'texts': ood_texts,
        'sources': ood_sources,
        'layer_idx': args.layer_idx,
        'model_name': args.model_name_or_path
    }, args.ood_output)
    print(f"\nOOD dataset saved to {args.ood_output}")
    print(f"  Total: {len(ood_hidden)} samples")
    
    print("\n" + "=" * 60)
    print("Data preparation completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()


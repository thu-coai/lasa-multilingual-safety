"""
数据准备脚本

从预训练模型中提取hidden states并保存
"""

import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from dataset import (
    load_ultrafeedback_data,
    load_safety_data,
    extract_hidden_states,
    MultilingualHiddenStateDataset
)


def main():
    parser = argparse.ArgumentParser(description="Prepare hidden states dataset")
    
    # 模型参数
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Path to the model")
    
    # 数据参数
    parser.add_argument("--ultrafeedback_dir", type=str, required=True,
                        help="Path to ultrafeedback data directory")
    parser.add_argument("--safety_data_path", type=str, required=True,
                        help="Path to safety_train_translated.json")
    parser.add_argument("--additional_unsafe_paths", type=str, nargs='*', default=[],
                        help="Additional unsafe data paths (e.g., harmbench_translated.json) for data augmentation")
    
    # 提取参数
    parser.add_argument("--layer_idx", type=int, default=20,
                        help="Layer index to extract hidden states from (0=embedding)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for extraction")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    
    # 采样参数
    parser.add_argument("--max_safe_samples_per_lang", type=int, default=None,
                        help="Maximum safe samples per language")
    parser.add_argument("--max_unsafe_samples", type=int, default=None,
                        help="Maximum unsafe samples")
    parser.add_argument("--languages", type=str, nargs='+',
                        default=['en', 'zh', 'ar', 'bn', 'it', 'jw', 'ko', 'sw', 'th', 'vi'],
                        help="Languages to include")
    
    # 输出参数
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output path for the dataset")
    
    # 设备参数
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Hidden States Extraction")
    print("=" * 60)
    print(f"Model: {args.model_name_or_path}")
    print(f"Layer: {args.layer_idx}")
    print(f"Languages: {args.languages}")
    print(f"Device: {args.device}")
    print("=" * 60)
    
    # 加载模型
    print(f"\nLoading model from {args.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True
    )
    
    # 设置padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    
    # 检查层数
    num_layers = model.config.num_hidden_layers + 1
    print(f"Model has {num_layers} layers (including embedding)")
    
    if args.layer_idx >= num_layers:
        print(f"Warning: layer_idx {args.layer_idx} >= num_layers {num_layers}, using layer {num_layers - 1}")
        args.layer_idx = num_layers - 1
    
    # 加载无害数据
    print(f"\nLoading ultrafeedback (safe) data from {args.ultrafeedback_dir}...")
    safe_data = load_ultrafeedback_data(
        args.ultrafeedback_dir,
        languages=args.languages,
        max_samples_per_lang=args.max_safe_samples_per_lang
    )
    print(f"Loaded {len(safe_data)} safe samples")
    
    # 统计语言分布
    lang_counts = {}
    for item in safe_data:
        lang = item['language']
        lang_counts[lang] = lang_counts.get(lang, 0) + 1
    print(f"Language distribution: {lang_counts}")
    
    # 加载有害数据（主要来源）
    print(f"\nLoading safety (unsafe) data from {args.safety_data_path}...")
    unsafe_data = load_safety_data(
        args.safety_data_path,
        max_samples=args.max_unsafe_samples
    )
    print(f"Loaded {len(unsafe_data)} unsafe samples from main source")
    
    # 加载额外的有害数据（用于数据增强，提高泛化能力）
    additional_unsafe_data = []
    if args.additional_unsafe_paths:
        for additional_path in args.additional_unsafe_paths:
            if os.path.exists(additional_path):
                print(f"Loading additional unsafe data from {additional_path}...")
                additional_data = load_safety_data(additional_path, max_samples=None)
                # 为额外数据的 idx 添加偏移，避免与主数据冲突
                max_idx = max(item['idx'] for item in unsafe_data) + 1 if unsafe_data else 0
                for item in additional_data:
                    item['idx'] = item['idx'] + max_idx
                additional_unsafe_data.extend(additional_data)
                print(f"  Added {len(additional_data)} samples")
            else:
                print(f"Warning: Additional unsafe data path not found: {additional_path}")
        
        unsafe_data.extend(additional_unsafe_data)
        print(f"Total unsafe samples after augmentation: {len(unsafe_data)}")
    
    # 统计语言分布
    lang_counts = {}
    for item in unsafe_data:
        lang = item['language']
        lang_counts[lang] = lang_counts.get(lang, 0) + 1
    print(f"Language distribution: {lang_counts}")
    
    # 提取hidden states
    print(f"\nExtracting hidden states from layer {args.layer_idx}...")
    
    all_texts = [item['prompt'] for item in safe_data + unsafe_data]
    all_hidden_states = extract_hidden_states(
        model, tokenizer, all_texts,
        layer_idx=args.layer_idx,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length
    )
    
    # 组织数据
    hidden_states = []
    group_ids = []
    safety_labels = []
    languages_list = []
    texts = []
    
    # 为有害问题和无害问题分配不同的group_id偏移
    safe_offset = 0
    unsafe_offset = max(item['idx'] for item in safe_data) + 1 if safe_data else 0
    
    for i, item in enumerate(safe_data):
        hidden_states.append(all_hidden_states[i])
        group_ids.append(item['idx'] + safe_offset)
        safety_labels.append(0)
        languages_list.append(item['language'])
        texts.append(item['prompt'])
    
    for i, item in enumerate(unsafe_data):
        hidden_states.append(all_hidden_states[len(safe_data) + i])
        group_ids.append(item['idx'] + unsafe_offset)
        safety_labels.append(1)
        languages_list.append(item['language'])
        texts.append(item['prompt'])
    
    # 创建数据集
    dataset = MultilingualHiddenStateDataset(
        hidden_states=hidden_states,
        group_ids=group_ids,
        safety_labels=safety_labels,
        languages=languages_list,
        texts=texts
    )
    
    # 保存数据
    print(f"\nSaving dataset to {args.output_path}...")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    torch.save({
        'hidden_states': hidden_states,
        'group_ids': group_ids,
        'safety_labels': safety_labels,
        'languages': languages_list,
        'texts': texts,
        'layer_idx': args.layer_idx,
        'model_name': args.model_name_or_path,
        'hidden_dim': hidden_states[0].shape[0]
    }, args.output_path)
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    print(f"Total samples: {len(dataset)}")
    print(f"Hidden dimension: {hidden_states[0].shape[0]}")
    print(f"Safe samples: {sum(1 for l in safety_labels if l == 0)}")
    print(f"Unsafe samples: {sum(1 for l in safety_labels if l == 1)}")
    print(f"Unique groups (safe): {len(set(g for g, l in zip(group_ids, safety_labels) if l == 0))}")
    print(f"Unique groups (unsafe): {len(set(g for g, l in zip(group_ids, safety_labels) if l == 1))}")
    
    # 按语言统计
    print("\nBy language:")
    for lang in sorted(set(languages_list)):
        safe_count = sum(1 for l, s in zip(languages_list, safety_labels) if l == lang and s == 0)
        unsafe_count = sum(1 for l, s in zip(languages_list, safety_labels) if l == lang and s == 1)
        print(f"  {lang}: {safe_count} safe, {unsafe_count} unsafe")
    
    print(f"\nDataset saved to {args.output_path}")


if __name__ == "__main__":
    main()


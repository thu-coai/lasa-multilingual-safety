"""
数据集处理模块

处理两种数据：
1. ultrafeedback: 无害问题的多语言版本
2. safety_train_translated: 有害问题的多语言版本
"""

import json
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class HiddenStateItem:
    """单个hidden state数据项"""
    hidden_state: torch.Tensor  # [hidden_dim]
    group_id: int  # 问题ID（同一问题不同语言共享）
    language: str  # 语言代码
    is_unsafe: bool  # 是否为有害问题
    text: str  # 原始文本


class MultilingualHiddenStateDataset(Dataset):
    """多语言Hidden State数据集"""
    
    def __init__(
        self,
        hidden_states: List[torch.Tensor],
        group_ids: List[int],
        safety_labels: List[int],
        languages: Optional[List[str]] = None,
        texts: Optional[List[str]] = None
    ):
        """
        Args:
            hidden_states: hidden state列表
            group_ids: 问题ID列表
            safety_labels: 安全标签列表（0=无害，1=有害）
            languages: 语言列表
            texts: 原始文本列表
        """
        self.hidden_states = hidden_states
        self.group_ids = group_ids
        self.safety_labels = safety_labels
        self.languages = languages or ['unknown'] * len(hidden_states)
        self.texts = texts or [''] * len(hidden_states)
    
    def __len__(self):
        return len(self.hidden_states)
    
    def __getitem__(self, idx):
        return {
            'hidden_state': self.hidden_states[idx],
            'group_id': self.group_ids[idx],
            'safety_label': self.safety_labels[idx],
            'language': self.languages[idx],
            'text': self.texts[idx]
        }


def collate_fn(batch: List[dict]) -> dict:
    """DataLoader的collate函数"""
    hidden_states = torch.stack([item['hidden_state'] for item in batch])
    # 确保转换为 float32，避免 bfloat16 和 float32 不匹配的问题
    hidden_states = hidden_states.float()
    group_ids = torch.tensor([item['group_id'] for item in batch], dtype=torch.long)
    safety_labels = torch.tensor([item['safety_label'] for item in batch], dtype=torch.long)
    languages = [item['language'] for item in batch]
    texts = [item['text'] for item in batch]
    
    return {
        'hidden_states': hidden_states,
        'group_ids': group_ids,
        'safety_labels': safety_labels,
        'languages': languages,
        'texts': texts
    }


def load_ultrafeedback_data(
    data_dir: str,
    languages: List[str] = None,
    max_samples_per_lang: int = None
) -> List[Dict]:
    """
    加载ultrafeedback多语言数据
    
    Args:
        data_dir: 数据目录
        languages: 要加载的语言列表
        max_samples_per_lang: 每个语言最多加载的样本数
    
    Returns:
        数据列表，每个元素包含 {idx, prompt, language}
    """
    if languages is None:
        # 默认语言列表（从文件名推断）
        languages = ['en', 'zh', 'ar', 'bn', 'it', 'jw', 'ko', 'sw', 'th', 'vi']
    
    data = []
    
    for lang in languages:
        file_path = os.path.join(data_dir, f'monolingual_first1000_{lang}.json')
        
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lang_data = json.load(f)
        
        if max_samples_per_lang:
            lang_data = lang_data[:max_samples_per_lang]
        
        for item in lang_data:
            data.append({
                'idx': item['idx'],
                'prompt': item['prompt'],
                'language': lang
            })
    
    return data


def load_safety_data(
    data_path: str,
    max_samples: int = None
) -> List[Dict]:
    """
    加载有害问题数据
    
    支持多种格式：
    1. 翻译格式：包含 original_query 和 translations 字段
    2. 简单格式：包含 prompt 和 lingual 字段（如 harmbench_training.json, multijail_prepared.json）
    3. 新格式（tranquery_nllb.json, tranquery_googletrans.json）：包含 prompt[0]["content"] 和 original_en 字段
    
    Args:
        data_path: 数据文件路径
        max_samples: 最多加载的样本数
    
    Returns:
        数据列表，每个元素包含 {idx, prompt, language}
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    if max_samples:
        raw_data = raw_data[:max_samples]
    
    data = []
    
    # 检测数据格式
    if not raw_data:
        return data
    
    first_item = raw_data[0]
    
    # 格式1: 翻译格式（safety_train_translated.json）
    if 'original_query' in first_item:
        for item in raw_data:
            idx = item['idx']
            
            # 英文原文
            data.append({
                'idx': idx,
                'prompt': item['original_query'],
                'language': 'en'
            })
            
            # 翻译版本
            for lang, translation in item['translations'].items():
                # 统一语言代码
                lang_code = lang.lower()
                lang_mapping = {
                    'chinese': 'zh',
                    'arabic': 'ar',
                    'bengali': 'bn',
                    'italian': 'it',
                    'javanese': 'jw',
                    'korean': 'ko',
                    'swahili': 'sw',
                    'thai': 'th',
                    'vietnamese': 'vi'
                }
                lang_code = lang_mapping.get(lang_code, lang_code)
                
                data.append({
                    'idx': idx,
                    'prompt': translation,
                    'language': lang_code
                })
    
    # 格式2: 新格式（tranquery_nllb.json, tranquery_googletrans.json）
    elif 'prompt' in first_item and isinstance(first_item['prompt'], list) and len(first_item['prompt']) > 0:
        # 使用 original_en 作为 query id 的标识
        # 创建一个映射：original_en -> idx
        original_en_to_idx = {}
        current_idx = 0
        
        # 第一遍：收集所有有 original_en 的条目，建立映射
        # 同时收集英文版本的 prompt，用于后续匹配
        en_prompts = {}  # original_en -> en_prompt_text
        
        for item in raw_data:
            original_en = item.get('original_en', '')
            if original_en:
                if original_en not in original_en_to_idx:
                    original_en_to_idx[original_en] = current_idx
                    current_idx += 1
                # 记录英文版本的 prompt
                if isinstance(item['prompt'], list) and len(item['prompt']) > 0:
                    prompt_text = item['prompt'][0].get('content', '')
                else:
                    prompt_text = item.get('prompt', '')
                language = item.get('language', 'en')
                if language == 'en':
                    en_prompts[original_en] = prompt_text
        
        # 第二遍：处理所有条目
        for item in raw_data:
            # 提取问题文本
            if isinstance(item['prompt'], list) and len(item['prompt']) > 0:
                prompt_text = item['prompt'][0].get('content', '')
            else:
                prompt_text = item.get('prompt', '')
            
            # 获取语言
            language = item.get('language', 'en')
            
            # 获取 original_en（用于标识同一个query）
            original_en = item.get('original_en', '')
            
            # 如果没有 original_en
            if not original_en:
                if language == 'en':
                    # 英文版本：使用 prompt 文本作为 original_en
                    original_en = prompt_text
                    if original_en not in original_en_to_idx:
                        original_en_to_idx[original_en] = current_idx
                        current_idx += 1
                        en_prompts[original_en] = prompt_text
                else:
                    # 非英文版本：尝试通过 prompt 文本匹配找到对应的 original_en
                    # 这里我们使用一个简单的策略：如果找不到匹配，就创建一个新的 idx
                    # 在实际使用中，可能需要更复杂的匹配逻辑
                    # 为了简化，我们暂时为每个非英文条目创建新的 idx
                    # 但更好的方法是预先处理数据，确保所有条目都有 original_en
                    idx = current_idx
                    current_idx += 1
                    data.append({
                        'idx': idx,
                        'prompt': prompt_text,
                        'language': language
                    })
                    continue
            
            # 为每个 unique original_en 分配一个 idx
            if original_en not in original_en_to_idx:
                original_en_to_idx[original_en] = current_idx
                current_idx += 1
            
            idx = original_en_to_idx[original_en]
            
            data.append({
                'idx': idx,
                'prompt': prompt_text,
                'language': language
            })
    
    # 格式3: 简单格式（harmbench_training.json, multijail_prepared.json）
    else:
        for item in raw_data:
            idx = item.get('idx', item.get('id', 0))
            prompt = item.get('prompt', '')
            # 如果 prompt 是列表格式，提取第一个元素的 content
            if isinstance(prompt, list) and len(prompt) > 0:
                prompt = prompt[0].get('content', '')
            data.append({
                'idx': idx,
                'prompt': prompt,
                'language': item.get('lingual', item.get('language', 'en'))
            })
    
    return data


def extract_hidden_states(
    model,
    tokenizer,
    texts: List[str],
    layer_idx: int = 20,
    device: str = 'cuda',
    batch_size: int = 8,
    max_length: int = 512
) -> List[torch.Tensor]:
    """
    批量提取hidden states
    
    Args:
        model: 预训练模型
        tokenizer: tokenizer
        texts: 文本列表
        layer_idx: 要提取的层（从0开始，0是embedding层）
        device: 设备
        batch_size: 批大小
        max_length: 最大序列长度
    
    Returns:
        hidden states列表
    """
    model.eval()
    hidden_states_list = []
    
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
        
        # 获取指定层的hidden states
        # outputs.hidden_states: tuple of (batch, seq_len, hidden_dim)
        layer_output = outputs.hidden_states[layer_idx]
        
        # 获取每个样本的最后一个有效token的hidden state
        for j in range(len(batch_texts)):
            # 找到最后一个非padding token
            attention_mask = inputs['attention_mask'][j]
            last_token_idx = attention_mask.sum() - 1
            
            # 根据文本末尾符号决定使用哪个token
            text = batch_texts[j]
            if text and text[-1] in '.?!':
                # 如果以标点结尾，使用倒数第二个token
                token_idx = max(0, last_token_idx - 1)
            else:
                token_idx = last_token_idx
            
            hidden_state = layer_output[j, token_idx, :].cpu()
            hidden_states_list.append(hidden_state)
    
    return hidden_states_list


def prepare_dataset(
    model_name_or_path: str,
    ultrafeedback_dir: str,
    safety_data_path: str,
    output_path: str,
    layer_idx: int = 20,
    device: str = 'cuda',
    languages: List[str] = None,
    max_safe_samples: int = None,
    max_unsafe_samples: int = None,
    batch_size: int = 8
) -> MultilingualHiddenStateDataset:
    """
    准备训练数据集
    
    Args:
        model_name_or_path: 模型路径
        ultrafeedback_dir: ultrafeedback数据目录
        safety_data_path: safety数据路径
        output_path: 输出路径（保存提取的hidden states）
        layer_idx: 要提取的层
        device: 设备
        languages: 语言列表
        max_safe_samples: 每语言最大无害样本数
        max_unsafe_samples: 最大有害样本数
        batch_size: 批大小
    
    Returns:
        准备好的数据集
    """
    print(f"Loading model from {model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    
    # 设置padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载无害数据
    print("Loading ultrafeedback (safe) data...")
    safe_data = load_ultrafeedback_data(
        ultrafeedback_dir,
        languages=languages,
        max_samples_per_lang=max_safe_samples
    )
    print(f"Loaded {len(safe_data)} safe samples")
    
    # 加载有害数据
    print("Loading safety (unsafe) data...")
    unsafe_data = load_safety_data(
        safety_data_path,
        max_samples=max_unsafe_samples
    )
    print(f"Loaded {len(unsafe_data)} unsafe samples")
    
    # 提取hidden states
    print(f"\nExtracting hidden states from layer {layer_idx}...")
    
    all_texts = [item['prompt'] for item in safe_data + unsafe_data]
    all_hidden_states = extract_hidden_states(
        model, tokenizer, all_texts,
        layer_idx=layer_idx,
        device=device,
        batch_size=batch_size
    )
    
    # 组织数据
    hidden_states = []
    group_ids = []
    safety_labels = []
    languages_list = []
    texts = []
    
    # 为有害问题和无害问题分配不同的group_id偏移
    # 无害问题: group_id = idx
    # 有害问题: group_id = idx + offset (确保不重叠)
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
    print(f"\nSaving dataset to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    torch.save({
        'hidden_states': hidden_states,
        'group_ids': group_ids,
        'safety_labels': safety_labels,
        'languages': languages_list,
        'texts': texts,
        'layer_idx': layer_idx,
        'model_name': model_name_or_path
    }, output_path)
    
    print(f"Dataset saved with {len(dataset)} samples")
    print(f"  - Safe samples: {sum(1 for l in safety_labels if l == 0)}")
    print(f"  - Unsafe samples: {sum(1 for l in safety_labels if l == 1)}")
    print(f"  - Unique groups (safe): {len(set(g for g, l in zip(group_ids, safety_labels) if l == 0))}")
    print(f"  - Unique groups (unsafe): {len(set(g for g, l in zip(group_ids, safety_labels) if l == 1))}")
    
    return dataset


def load_dataset(data_path: str) -> MultilingualHiddenStateDataset:
    """
    加载保存的数据集
    
    Args:
        data_path: 数据文件路径
    
    Returns:
        数据集
    """
    data = torch.load(data_path, weights_only=False)
    
    return MultilingualHiddenStateDataset(
        hidden_states=data['hidden_states'],
        group_ids=data['group_ids'],
        safety_labels=data['safety_labels'],
        languages=data.get('languages'),
        texts=data.get('texts')
    )


class BalancedBatchSampler:
    """
    平衡采样器：确保每个batch中有足够的同组样本和安全/不安全样本
    """
    
    def __init__(
        self,
        dataset: MultilingualHiddenStateDataset,
        batch_size: int = 32,
        min_group_samples: int = 2,
        balance_safety: bool = True
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.min_group_samples = min_group_samples
        self.balance_safety = balance_safety
        
        # 按group_id组织样本索引
        self.group_to_indices = {}
        for i, group_id in enumerate(dataset.group_ids):
            if group_id not in self.group_to_indices:
                self.group_to_indices[group_id] = []
            self.group_to_indices[group_id].append(i)
        
        # 按安全标签组织
        self.safe_indices = [i for i, l in enumerate(dataset.safety_labels) if l == 0]
        self.unsafe_indices = [i for i, l in enumerate(dataset.safety_labels) if l == 1]
        
        # 筛选有多语言版本的group
        self.valid_groups = [
            gid for gid, indices in self.group_to_indices.items()
            if len(indices) >= min_group_samples
        ]
    
    def __iter__(self):
        # 打乱group顺序
        np.random.shuffle(self.valid_groups)
        
        batch = []
        
        for group_id in self.valid_groups:
            group_indices = self.group_to_indices[group_id]
            
            # 采样该组的样本
            if len(group_indices) <= self.min_group_samples:
                sampled = group_indices
            else:
                sampled = np.random.choice(
                    group_indices,
                    size=min(self.min_group_samples, len(group_indices)),
                    replace=False
                ).tolist()
            
            batch.extend(sampled)
            
            if len(batch) >= self.batch_size:
                yield batch[:self.batch_size]
                batch = batch[self.batch_size:]
        
        if len(batch) > 0:
            yield batch
    
    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


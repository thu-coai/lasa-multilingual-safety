"""
跨语言层分析预实验 V2 - 基于Silhouette Score的层选择

核心思路：
    1. 对于每一层，使用少量样本（N个问题的多语言版本）计算聚类指标
    2. 计算 silhouette_query - silhouette_language 差值
    3. 选择差值最大的层（即query聚类更强、language聚类更弱的层）
    4. 只在选定的最佳层上进行完整的Safety分类器训练

这种方法的优势：
    - 快速：只需计算Silhouette Score，无需训练分类器
    - 直接：差值大意味着同一问题的不同语言版本更接近，不同问题更分开
    - 理论依据：中间层更倾向于按语义（问题内容）聚类，而非语言聚类
"""

import os
import argparse
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, silhouette_score
from transformers import AutoTokenizer, AutoModelForCausalLM


# ==================== Silhouette Score 计算 ====================

def compute_silhouette_scores(
    embeddings: torch.Tensor,
    language_labels: List[str],
    query_ids: List[int]
) -> Dict:
    """
    计算按语言和按问题聚类的Silhouette Score
    
    Args:
        embeddings: hidden states, shape [N, dim]
        language_labels: 语言标签列表
        query_ids: 问题ID列表
    
    Returns:
        包含silhouette_language, silhouette_query, score_diff的字典
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
    
    # ========== 1. 按语言聚类的Silhouette Score ==========
    if len(unique_langs) > 1:
        try:
            metrics['silhouette_language'] = silhouette_score(
                embeddings_np, lang_labels_numeric, metric='cosine'
            )
        except Exception as e:
            print(f"Warning: silhouette_language computation failed: {e}")
            metrics['silhouette_language'] = 0.0
    else:
        metrics['silhouette_language'] = 0.0
    
    # ========== 2. 按问题聚类的Silhouette Score ==========
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
    
    # ========== 3. 计算差值（正值表示query聚类更强） ==========
    metrics['score_diff'] = metrics['silhouette_query'] - metrics['silhouette_language']
    
    return metrics


def select_best_layer_by_silhouette(
    all_layer_hidden_states: Dict[int, List[torch.Tensor]],
    data: List[Dict],
    num_queries: int = 10,
    device: str = 'cuda'
) -> Tuple[int, Dict[int, Dict]]:
    """
    使用Silhouette Score选择最佳层
    
    Args:
        all_layer_hidden_states: 每层的hidden states字典 {layer_idx: [hidden_states]}
        data: 数据列表，每个元素包含 {idx, prompt, language, is_unsafe}
        num_queries: 用于评估的问题数量
        device: 设备
    
    Returns:
        (best_layer_idx, layer_metrics_dict)
    """
    print(f"\n{'='*60}")
    print(f"Selecting Best Layer using Silhouette Score")
    print(f"{'='*60}")
    
    # 统计每个问题的语言数量，选择有多语言版本的问题
    query_to_items = defaultdict(list)
    for i, item in enumerate(data):
        query_to_items[item['idx']].append(i)
    
    # 选择有多语言版本的问题
    multilingual_queries = [
        (qid, indices) for qid, indices in query_to_items.items()
        if len(indices) >= 2
    ]
    
    # 按问题数量排序，优先选择语言版本更多的问题
    multilingual_queries.sort(key=lambda x: len(x[1]), reverse=True)
    
    # 选择前N个问题
    selected_queries = multilingual_queries[:num_queries]
    selected_indices = []
    for qid, indices in selected_queries:
        selected_indices.extend(indices)
    
    print(f"Selected {len(selected_queries)} queries with {len(selected_indices)} samples for layer selection")
    
    # 获取选中样本的信息
    selected_data = [data[i] for i in selected_indices]
    languages = [item['language'] for item in selected_data]
    query_ids = [item['idx'] for item in selected_data]
    
    print(f"Languages: {sorted(set(languages))}")
    print(f"Queries: {sorted(set(query_ids))}")
    
    # 计算每层的Silhouette Score
    num_layers = len(all_layer_hidden_states)
    layer_metrics = {}
    
    print(f"\nComputing Silhouette Scores for {num_layers} layers...")
    print(f"{'Layer':<8} {'Sil_Lang':<12} {'Sil_Query':<12} {'Diff (Q-L)':<12} {'Status':<10}")
    print("-" * 56)
    
    for layer_idx in range(num_layers):
        # 获取选中样本的hidden states
        layer_hs = all_layer_hidden_states[layer_idx]
        selected_hs = torch.stack([layer_hs[i] for i in selected_indices])
        
        # 计算Silhouette Score
        metrics = compute_silhouette_scores(selected_hs, languages, query_ids)
        layer_metrics[layer_idx] = metrics
        
        # 判断状态
        if metrics['score_diff'] > 0:
            status = "Query↑"
        else:
            status = "Lang↑"
        
        print(f"{layer_idx:<8} {metrics['silhouette_language']:<12.4f} "
              f"{metrics['silhouette_query']:<12.4f} {metrics['score_diff']:<12.4f} {status:<10}")
    
    # 选择差值最大的层
    best_layer = max(layer_metrics.keys(), key=lambda k: layer_metrics[k]['score_diff'])
    best_metrics = layer_metrics[best_layer]
    
    print(f"\n{'='*60}")
    print(f"Best Layer: {best_layer}")
    print(f"  Silhouette (Language): {best_metrics['silhouette_language']:.4f}")
    print(f"  Silhouette (Query):    {best_metrics['silhouette_query']:.4f}")
    print(f"  Score Diff (Q - L):    {best_metrics['score_diff']:.4f}")
    print(f"{'='*60}")
    
    return best_layer, layer_metrics


# ==================== MLP分类器 ====================

class SimpleMLP(nn.Module):
    """简单的MLP分类器（单hidden layer）"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class DeepMLP(nn.Module):
    """更深的MLP分类器（多hidden layer）"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [512, 256],
        output_dim: int = 1,
        dropout: float = 0.2,
        use_layer_norm: bool = True,
        activation: str = 'gelu'
    ):
        super().__init__()
        
        if activation == 'gelu':
            act_fn = nn.GELU()
        elif activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'silu':
            act_fn = nn.SiLU()
        else:
            act_fn = nn.GELU()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(act_fn)
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# ==================== 数据集类 ====================

class HiddenStateDataset(Dataset):
    """Hidden State 数据集"""
    
    def __init__(
        self,
        hidden_states: List[torch.Tensor],
        labels: List[int],
        label_names: Optional[List[str]] = None
    ):
        self.hidden_states = hidden_states
        self.labels = labels
        self.label_names = label_names
    
    def __len__(self):
        return len(self.hidden_states)
    
    def __getitem__(self, idx):
        return {
            'hidden_state': self.hidden_states[idx],
            'label': self.labels[idx]
        }


def collate_fn(batch: List[dict]) -> dict:
    """DataLoader的collate函数"""
    hidden_states = torch.stack([item['hidden_state'] for item in batch])
    hidden_states = hidden_states.float()
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    return {'hidden_states': hidden_states, 'labels': labels}


# ==================== 数据加载函数 ====================

def load_ultrafeedback_data(data_dir: str, languages: List[str] = None, max_samples_per_lang: int = None) -> List[Dict]:
    """加载 ultrafeedback 多语言数据（无害问题）"""
    if languages is None:
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
                'language': lang,
                'is_unsafe': False
            })
    
    return data


def load_safety_data(data_path: str, max_samples: int = None, idx_offset: int = 0) -> List[Dict]:
    """Load harmful prompts from translated or flattened SSI safety data."""
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    if max_samples:
        raw_data = raw_data[:max_samples]
    
    if not raw_data:
        return []
    
    lang_mapping = {
        'chinese': 'zh', 'arabic': 'ar', 'bengali': 'bn',
        'italian': 'it', 'javanese': 'jw', 'korean': 'ko',
        'swahili': 'sw', 'thai': 'th', 'vietnamese': 'vi'
    }
    
    if 'original_query' not in raw_data[0]:
        data = []
        for item in raw_data:
            prompt = item.get('prompt', '')
            if isinstance(prompt, list) and prompt:
                prompt = prompt[0].get('content', '')
            language = item.get('lingual', item.get('language', 'en'))
            language = lang_mapping.get(str(language).lower(), str(language).lower())
            data.append({
                'idx': item.get('idx', item.get('id', 0)) + idx_offset,
                'prompt': prompt,
                'language': language,
                'is_unsafe': True,
                'source': item.get('source', 'safety')
            })
        return data
    
    data = []
    for item in raw_data:
        idx = item['idx'] + idx_offset
        
        # 英文原文
        data.append({
            'idx': idx,
            'prompt': item['original_query'],
            'language': 'en',
            'is_unsafe': True
        })
        
        # 翻译版本
        for lang, translation in item['translations'].items():
            lang_code = lang_mapping.get(lang.lower(), lang.lower())
            data.append({
                'idx': idx,
                'prompt': translation,
                'language': lang_code,
                'is_unsafe': True
            })
    
    return data


def load_harmbench_data(data_path: str, max_samples: int = None, idx_offset: int = 0) -> List[Dict]:
    """加载 harmbench 或 multijail 数据（OOD 有害问题）"""
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    if max_samples and len(raw_data) > max_samples:
        raw_data = raw_data[:max_samples]
    
    if raw_data and 'original_query' in raw_data[0]:
        return load_safety_data(data_path, max_samples, idx_offset)
    else:
        data = []
        for item in raw_data:
            idx = item.get('idx', item.get('id', 0)) + idx_offset
            data.append({
                'idx': idx,
                'prompt': item['prompt'],
                'language': item.get('lingual', 'en'),
                'is_unsafe': True
            })
        return data


# ==================== Hidden State 提取 ====================

def extract_hidden_states_all_layers(
    model,
    tokenizer,
    texts: List[str],
    device: str = 'cuda',
    batch_size: int = 8,
    max_length: int = 512
) -> Dict[int, List[torch.Tensor]]:
    """提取所有层的 hidden states"""
    model.eval()
    
    num_layers = model.config.num_hidden_layers + 1
    all_hidden_states = {layer_idx: [] for layer_idx in range(num_layers)}
    
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
        
        for layer_idx in range(num_layers):
            layer_output = outputs.hidden_states[layer_idx]
            
            for j in range(len(batch_texts)):
                attention_mask = inputs['attention_mask'][j]
                last_token_idx = attention_mask.sum() - 1
                hidden_state = layer_output[j, last_token_idx, :].cpu()
                all_hidden_states[layer_idx].append(hidden_state)
    
    return all_hidden_states


# ==================== 训练和评估函数 ====================

def find_optimal_threshold(
    probs: List[float],
    labels: List[int],
    optimize_for: str = 'accuracy'
) -> Tuple[float, Dict]:
    """找到最优分类阈值"""
    probs = np.array(probs)
    labels = np.array(labels)
    
    best_threshold = 0.5
    best_score = -float('inf')
    best_metrics = {}
    
    for threshold in np.arange(0.30, 0.71, 0.01):
        preds = (probs > threshold).astype(int)
        
        tp = ((preds == 1) & (labels == 1)).sum()
        tn = ((preds == 0) & (labels == 0)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * precision * tpr / (precision + tpr) if (precision + tpr) > 0 else 0
        acc = (tp + tn) / len(labels)
        
        if optimize_for == 'accuracy':
            score = acc
        elif optimize_for == 'f1':
            score = f1
        else:
            score = acc
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_metrics = {
                'tpr': tpr,
                'fpr': fpr,
                'precision': precision,
                'f1': f1,
                'accuracy': acc
            }
    
    return best_threshold, best_metrics


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str = 'cuda',
    num_epochs: int = 20,
    learning_rate: float = 1e-3,
    early_stopping_patience: int = 5,
    verbose: bool = True,
    optimize_threshold: bool = True
) -> Dict:
    """训练分类器"""
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)
    
    best_val_acc = 0.0
    patience_counter = 0
    best_state = None
    history = {'train_loss': [], 'val_acc': [], 'val_f1': []}
    
    for epoch in range(num_epochs):
        # 训练
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            hidden_states = batch['hidden_states'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            logits = model(hidden_states)
            loss = criterion(logits.squeeze(-1), labels.float())
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        history['train_loss'].append(avg_loss)
        
        # 验证
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                hidden_states = batch['hidden_states'].to(device)
                labels = batch['labels']
                logits = model(hidden_states)
                preds = (torch.sigmoid(logits.squeeze(-1)) > 0.5).long().cpu()
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.tolist())
        
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
        
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        scheduler.step(val_acc)
        
        if verbose:
            print(f"  Epoch {epoch+1}/{num_epochs}: Loss={avg_loss:.4f}, Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}")
        
        # 早停
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1}")
                break
    
    # 恢复最佳模型
    if best_state:
        model.load_state_dict(best_state)
    
    # 找最优阈值
    best_threshold = 0.5
    if optimize_threshold:
        model.eval()
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                hidden_states = batch['hidden_states'].to(device)
                labels = batch['labels']
                logits = model(hidden_states)
                probs = torch.sigmoid(logits.squeeze(-1)).cpu()
                all_probs.extend(probs.tolist())
                all_labels.extend(labels.tolist())
        
        best_threshold, _ = find_optimal_threshold(all_probs, all_labels, optimize_for='accuracy')
        if verbose:
            print(f"  Optimal threshold: {best_threshold:.3f}")
    
    return {
        'best_val_acc': best_val_acc,
        'best_val_f1': max(history['val_f1']),
        'best_threshold': best_threshold,
        'history': history,
        'model_state_dict': {k: v.cpu().clone() for k, v in model.state_dict().items()}
    }


def evaluate_classifier(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = 'cuda',
    threshold: float = 0.5
) -> Dict:
    """评估分类器"""
    model.eval()
    model = model.to(device)
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in test_loader:
            hidden_states = batch['hidden_states'].to(device)
            labels = batch['labels']
            logits = model(hidden_states)
            probs = torch.sigmoid(logits.squeeze(-1))
            preds = (probs > threshold).long().cpu()
            all_probs.extend(probs.cpu().tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    
    tp = ((all_preds == 1) & (all_labels == 1)).sum()
    tn = ((all_preds == 0) & (all_labels == 0)).sum()
    fp = ((all_preds == 1) & (all_labels == 0)).sum()
    fn = ((all_preds == 0) & (all_labels == 1)).sum()
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return {
        'accuracy': acc,
        'f1': f1,
        'tpr': tpr,
        'fpr': fpr,
        'threshold': threshold
    }


def evaluate_per_language(
    model: nn.Module,
    hidden_states: List[torch.Tensor],
    labels: List[int],
    data: List[Dict],
    device: str = 'cuda',
    threshold: float = 0.5
) -> Dict[str, Dict]:
    """计算每个语言的分类准确率"""
    model.eval()
    model = model.to(device)
    
    lang_indices = defaultdict(list)
    for i, item in enumerate(data):
        lang_indices[item['language']].append(i)
    
    per_lang_results = {}
    
    with torch.no_grad():
        for lang, indices in lang_indices.items():
            if len(indices) == 0:
                continue
            
            lang_hs = [hidden_states[i] for i in indices]
            lang_labels = [labels[i] for i in indices]
            
            hs_tensor = torch.stack([
                h if isinstance(h, torch.Tensor) else torch.tensor(h)
                for h in lang_hs
            ]).to(device).float()
            labels_tensor = torch.tensor(lang_labels, device=device)
            
            outputs = model(hs_tensor)
            if outputs.dim() > 1 and outputs.size(1) == 1:
                outputs = outputs.squeeze(1)
            probs = torch.sigmoid(outputs)
            preds = (probs > threshold).long()
            
            correct = (preds == labels_tensor).sum().item()
            total = len(indices)
            acc = correct / total
            
            num_safe = sum(1 for l in lang_labels if l == 0)
            num_unsafe = sum(1 for l in lang_labels if l == 1)
            
            preds_np = preds.cpu().numpy()
            labels_np = labels_tensor.cpu().numpy()
            
            tp = ((preds_np == 1) & (labels_np == 1)).sum()
            tn = ((preds_np == 0) & (labels_np == 0)).sum()
            fp = ((preds_np == 1) & (labels_np == 0)).sum()
            fn = ((preds_np == 0) & (labels_np == 1)).sum()
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            per_lang_results[lang] = {
                'accuracy': acc,
                'total': total,
                'num_safe': num_safe,
                'num_unsafe': num_unsafe,
                'correct': correct,
                'tpr': tpr,
                'fpr': fpr
            }
    
    return per_lang_results


# ==================== 可视化函数 ====================

def plot_layer_silhouette_analysis(layer_metrics: Dict[int, Dict], output_dir: str):
    """绘制每层的Silhouette Score分析图"""
    layers = sorted(layer_metrics.keys())
    sil_lang = [layer_metrics[l]['silhouette_language'] for l in layers]
    sil_query = [layer_metrics[l]['silhouette_query'] for l in layers]
    score_diff = [layer_metrics[l]['score_diff'] for l in layers]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Silhouette Score对比
    ax = axes[0]
    ax.plot(layers, sil_lang, 'b-o', linewidth=2, markersize=6, label='Language')
    ax.plot(layers, sil_query, 'r-s', linewidth=2, markersize=6, label='Query')
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Silhouette Score', fontsize=12)
    ax.set_title('Silhouette Score by Clustering Type', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Score Diff
    ax = axes[1]
    colors = ['green' if d > 0 else 'red' for d in score_diff]
    ax.bar(layers, score_diff, color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Score Diff (Query - Language)', fontsize=12)
    ax.set_title('Query vs Language Clustering\n(>0: Query dominant)', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # 标记最佳层
    best_layer = max(layer_metrics.keys(), key=lambda k: layer_metrics[k]['score_diff'])
    ax.scatter([best_layer], [layer_metrics[best_layer]['score_diff']], 
               color='gold', s=200, zorder=5, edgecolor='black', linewidth=2, 
               marker='*', label=f'Best: Layer {best_layer}')
    ax.legend()
    
    # 3. 折线图
    ax = axes[2]
    ax.plot(layers, score_diff, 'g-o', linewidth=2, markersize=6)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.scatter([best_layer], [layer_metrics[best_layer]['score_diff']], 
               color='gold', s=200, zorder=5, edgecolor='black', linewidth=2, marker='*')
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Score Diff (Query - Language)', fontsize=12)
    ax.set_title('Layer Selection Score', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'layer_silhouette_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved layer analysis plot to {output_dir}")


# ==================== 主分析流程 ====================

def analyze_with_silhouette_selection(
    model,
    tokenizer,
    safe_data: List[Dict],
    unsafe_data: List[Dict],
    output_dir: str,
    ood_data: List[Dict] = None,
    device: str = 'cuda',
    batch_size: int = 8,
    num_queries_for_selection: int = 10,
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
    使用Silhouette Score选择最佳层，然后在该层训练分类器
    
    Args:
        model: 预训练模型
        tokenizer: tokenizer
        safe_data: 无害数据列表
        unsafe_data: 有害数据列表
        output_dir: 输出目录
        ood_data: OOD数据列表
        device: 设备
        batch_size: 批大小
        num_queries_for_selection: 用于层选择的问题数量
        classifier_hidden_dim: 分类器隐藏层维度
        train_epochs: 训练轮数
        val_split: 验证集比例
        seed: 随机种子
        train_languages: 训练语言列表
        test_languages: 测试语言列表
        use_deep_mlp: 是否使用DeepMLP
        classifier_dropout: Dropout率
    
    Returns:
        分析结果
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 合并数据
    all_data = safe_data + unsafe_data
    all_texts = [item['prompt'] for item in all_data]
    
    print(f"\n{'='*60}")
    print(f"Cross-Lingual Layer Analysis with Silhouette Selection")
    print(f"{'='*60}")
    print(f"Total samples: {len(all_data)}")
    print(f"  - Safe samples: {len(safe_data)}")
    print(f"  - Unsafe samples: {len(unsafe_data)}")
    
    has_ood = ood_data is not None and len(ood_data) > 0
    if has_ood:
        ood_texts = [item['prompt'] for item in ood_data]
        print(f"  - OOD samples: {len(ood_data)}")
    
    languages = sorted(list(set(item['language'] for item in all_data)))
    print(f"Languages ({len(languages)}): {languages}")
    
    # ========== 1. 提取所有层的Hidden States ==========
    print(f"\nExtracting hidden states from all layers...")
    all_layer_hidden_states = extract_hidden_states_all_layers(
        model, tokenizer, all_texts,
        device=device, batch_size=batch_size
    )
    
    if has_ood:
        print(f"Extracting OOD hidden states...")
        ood_layer_hidden_states = extract_hidden_states_all_layers(
            model, tokenizer, ood_texts,
            device=device, batch_size=batch_size
        )
    else:
        ood_layer_hidden_states = None
    
    num_layers = len(all_layer_hidden_states)
    hidden_dim = all_layer_hidden_states[0][0].shape[0]
    print(f"Number of layers: {num_layers}")
    print(f"Hidden dimension: {hidden_dim}")
    
    # ========== 2. 使用Silhouette Score选择最佳层 ==========
    best_layer, layer_metrics = select_best_layer_by_silhouette(
        all_layer_hidden_states=all_layer_hidden_states,
        data=all_data,
        num_queries=num_queries_for_selection,
        device=device
    )
    
    # 绘制层分析图
    plot_layer_silhouette_analysis(layer_metrics, output_dir)
    
    # 保存层选择结果
    layer_selection_results = {
        'best_layer': best_layer,
        'num_queries_for_selection': num_queries_for_selection,
        'layer_metrics': {str(k): v for k, v in layer_metrics.items()}
    }
    with open(os.path.join(output_dir, 'layer_selection_results.json'), 'w') as f:
        json.dump(layer_selection_results, f, indent=2)
    
    # ========== 3. 在最佳层上训练Safety分类器 ==========
    print(f"\n{'='*60}")
    print(f"Training Safety Classifier on Best Layer {best_layer}")
    print(f"{'='*60}")
    
    best_layer_hs = all_layer_hidden_states[best_layer]
    safety_labels = [0 if not item['is_unsafe'] else 1 for item in all_data]
    
    # 按语言分割训练/测试集
    if train_languages is not None or test_languages is not None:
        train_langs = set(train_languages) if train_languages else set(languages)
        test_langs = set(test_languages) if test_languages else set(languages)
        
        train_indices = [i for i, item in enumerate(all_data) if item['language'] in train_langs]
        test_indices = [i for i, item in enumerate(all_data) if item['language'] in test_langs]
        
        np.random.shuffle(train_indices)
        inner_val_size = int(len(train_indices) * 0.1)
        val_indices = train_indices[:inner_val_size]
        train_indices = train_indices[inner_val_size:]
        
        print(f"\n>>> Cross-lingual generalization mode:")
        print(f"    Train languages: {sorted(train_langs)}")
        print(f"    Test languages: {sorted(test_langs)}")
    else:
        indices = list(range(len(all_data)))
        np.random.shuffle(indices)
        
        val_size = int(len(indices) * val_split)
        test_size = int(len(indices) * val_split)
        train_indices = indices[val_size + test_size:]
        val_indices = indices[:val_size]
        test_indices = indices[val_size:val_size + test_size]
    
    print(f"Train samples: {len(train_indices)}")
    print(f"Val samples: {len(val_indices)}")
    print(f"Test samples: {len(test_indices)}")
    
    # 准备数据
    train_hs = [best_layer_hs[i] for i in train_indices]
    val_hs = [best_layer_hs[i] for i in val_indices]
    test_hs = [best_layer_hs[i] for i in test_indices]
    
    train_labels = [safety_labels[i] for i in train_indices]
    val_labels = [safety_labels[i] for i in val_indices]
    test_labels = [safety_labels[i] for i in test_indices]
    
    train_dataset = HiddenStateDataset(train_hs, train_labels)
    val_dataset = HiddenStateDataset(val_hs, val_labels)
    test_dataset = HiddenStateDataset(test_hs, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    
    # 创建分类器
    if use_deep_mlp:
        classifier = DeepMLP(
            input_dim=hidden_dim,
            hidden_dims=[classifier_hidden_dim, classifier_hidden_dim // 2],
            output_dim=1,
            dropout=classifier_dropout,
            use_layer_norm=True,
            activation='gelu'
        )
        print(f"Using DeepMLP: {hidden_dim} -> {classifier_hidden_dim} -> {classifier_hidden_dim//2} -> 1")
    else:
        classifier = SimpleMLP(
            input_dim=hidden_dim,
            hidden_dim=classifier_hidden_dim,
            output_dim=1,
            dropout=classifier_dropout
        )
        print(f"Using SimpleMLP: {hidden_dim} -> {classifier_hidden_dim} -> 1")
    
    # 训练
    train_result = train_classifier(
        classifier, train_loader, val_loader,
        device=device, num_epochs=train_epochs, verbose=True,
        optimize_threshold=True
    )
    
    # 加载最佳模型
    classifier.load_state_dict(train_result['model_state_dict'])
    classifier = classifier.to(device)
    
    best_threshold = train_result['best_threshold']
    
    # ========== 4. 评估 ==========
    print(f"\n{'='*60}")
    print(f"Evaluation Results")
    print(f"{'='*60}")
    
    # 测试集评估
    test_result = evaluate_classifier(classifier, test_loader, device=device, threshold=best_threshold)
    print(f"\nTest Set Performance:")
    print(f"  Accuracy: {test_result['accuracy']:.4f}")
    print(f"  F1: {test_result['f1']:.4f}")
    print(f"  TPR: {test_result['tpr']:.4f}")
    print(f"  FPR: {test_result['fpr']:.4f}")
    
    # 每个语言的评估
    per_lang_results = evaluate_per_language(
        classifier, best_layer_hs, safety_labels, all_data,
        device=device, threshold=best_threshold
    )
    
    print(f"\nPer-Language Results:")
    print(f"{'Language':<10} {'Accuracy':>10} {'FPR':>10} {'TPR':>10} {'Safe':>8} {'Unsafe':>8}")
    print("-" * 60)
    for lang in sorted(per_lang_results.keys()):
        info = per_lang_results[lang]
        print(f"{lang:<10} {info['accuracy']:>10.4f} {info['fpr']:>10.4f} {info['tpr']:>10.4f} "
              f"{info['num_safe']:>8} {info['num_unsafe']:>8}")
    
    # OOD评估
    ood_result = None
    ood_per_lang = {}
    if has_ood and ood_layer_hidden_states:
        ood_hs = ood_layer_hidden_states[best_layer]
        ood_labels = [1] * len(ood_data)
        
        ood_dataset = HiddenStateDataset(ood_hs, ood_labels)
        ood_loader = DataLoader(ood_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
        
        ood_result = evaluate_classifier(classifier, ood_loader, device=device, threshold=best_threshold)
        
        print(f"\nOOD (MultiJail) Performance:")
        print(f"  Accuracy: {ood_result['accuracy']:.4f}")
        
        ood_per_lang = evaluate_per_language(
            classifier, ood_hs, ood_labels, ood_data,
            device=device, threshold=best_threshold
        )
        
        print(f"\nOOD Per-Language Results:")
        print(f"{'Language':<10} {'Accuracy':>10} {'Total':>10}")
        print("-" * 35)
        for lang in sorted(ood_per_lang.keys()):
            info = ood_per_lang[lang]
            print(f"{lang:<10} {info['accuracy']:>10.4f} {info['total']:>10}")
    
    # ========== 5. 保存结果 ==========
    # 保存模型
    threshold_str = f"{int(round(best_threshold * 100)):03d}"
    model_save_path = os.path.join(
        output_dir,
        f'classifier_layer{best_layer}_hidden{classifier_hidden_dim}_threshold{threshold_str}.pt'
    )
    
    avg_fpr = np.mean([v['fpr'] for v in per_lang_results.values() if v['num_safe'] > 0])
    avg_tpr = np.mean([v['tpr'] for v in per_lang_results.values() if v['num_unsafe'] > 0])
    
    torch.save({
        'model_state_dict': classifier.state_dict(),
        'layer_idx': best_layer,
        'hidden_dim': hidden_dim,
        'classifier_hidden_dim': classifier_hidden_dim,
        'threshold': best_threshold,
        'test_accuracy': test_result['accuracy'],
        'avg_fpr': avg_fpr,
        'avg_tpr': avg_tpr,
        'ood_accuracy': ood_result['accuracy'] if ood_result else 0.0,
        'train_languages': list(train_languages) if train_languages else None,
        'test_languages': list(test_languages) if test_languages else None,
        'layer_selection_method': 'silhouette_score_diff'
    }, model_save_path)
    
    print(f"\nModel saved to {model_save_path}")
    
    # 保存完整结果
    final_summary = {
        'model_path': model_save_path,
        'layer_idx': best_layer,
        'layer_selection_method': 'silhouette_score_diff',
        'layer_selection_score': layer_metrics[best_layer]['score_diff'],
        'classifier_hidden_dim': classifier_hidden_dim,
        'threshold': best_threshold,
        'in_distribution': {
            'test_accuracy': test_result['accuracy'],
            'avg_fpr': avg_fpr,
            'avg_tpr': avg_tpr,
            'per_lang_accuracy': {lang: info['accuracy'] for lang, info in per_lang_results.items()},
            'per_lang_fpr': {lang: info['fpr'] for lang, info in per_lang_results.items()},
            'per_lang_tpr': {lang: info['tpr'] for lang, info in per_lang_results.items()}
        },
        'out_of_distribution': {
            'overall_accuracy': ood_result['accuracy'] if ood_result else 0.0,
            'per_lang_accuracy': {lang: info['accuracy'] for lang, info in ood_per_lang.items()} if ood_per_lang else {}
        }
    }
    
    with open(os.path.join(output_dir, 'final_model_summary.json'), 'w') as f:
        json.dump(final_summary, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"FINAL MODEL PERFORMANCE SUMMARY")
    print(f"{'='*70}")
    print(f"Layer: {best_layer} (selected by Silhouette Score Diff: {layer_metrics[best_layer]['score_diff']:.4f})")
    print(f"Threshold: {best_threshold:.3f}")
    print(f"Test Accuracy: {test_result['accuracy']:.4f}")
    print(f"Avg FPR: {avg_fpr:.4f}")
    print(f"Avg TPR: {avg_tpr:.4f}")
    if ood_result:
        print(f"OOD Accuracy: {ood_result['accuracy']:.4f}")
    print(f"{'='*70}")
    
    return {
        'best_layer': best_layer,
        'layer_metrics': layer_metrics,
        'test_result': test_result,
        'per_lang_results': per_lang_results,
        'ood_result': ood_result,
        'final_summary': final_summary
    }


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description="Cross-Lingual Layer Analysis with Silhouette Selection")
    
    # 模型参数
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to pretrained model")
    
    # 数据参数
    parser.add_argument("--ultrafeedback_dir", type=str, required=True,
                        help="Path to ultrafeedback data directory")
    parser.add_argument("--safety_data_path", type=str, required=True,
                        help="Path to safety data file")
    parser.add_argument("--ood_data_path", type=str, default=None,
                        help="Path to OOD data file (e.g. multijail)")
    parser.add_argument("--languages", type=str, nargs="+",
                        default=['en', 'zh', 'ar', 'it', 'ko', 'vi'],
                        help="All languages to use")
    parser.add_argument("--train_languages", type=str, nargs="+", default=None,
                        help="Languages for training (default: all)")
    parser.add_argument("--test_languages", type=str, nargs="+", default=None,
                        help="Languages for testing (default: all)")
    parser.add_argument("--max_safe_per_lang", type=int, default=200,
                        help="Max safe samples per language")
    parser.add_argument("--max_unsafe", type=int, default=200,
                        help="Max unsafe samples")
    parser.add_argument("--max_ood", type=int, default=200,
                        help="Max OOD samples")
    
    # 层选择参数
    parser.add_argument("--num_queries_for_selection", type=int, default=10,
                        help="Number of queries to use for layer selection")
    
    # 分类器参数
    parser.add_argument("--classifier_hidden_dim", type=int, default=256,
                        help="Hidden dimension for classifier")
    parser.add_argument("--use_deep_mlp", action="store_true", default=True,
                        help="Use DeepMLP")
    parser.add_argument("--use_simple_mlp", action="store_true",
                        help="Use SimpleMLP instead")
    parser.add_argument("--classifier_dropout", type=float, default=0.2,
                        help="Dropout rate")
    parser.add_argument("--train_epochs", type=int, default=20,
                        help="Training epochs")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Validation split ratio")
    
    # 其他参数
    parser.add_argument("--output_dir", type=str, default="./output/layer_analysis_silhouette",
                        help="Output directory")
    parser.add_argument("--hidden_states_cache", type=str, default=None,
                        help="Path to cache hidden states")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"analysis_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存配置
    config = vars(args)
    config['timestamp'] = timestamp
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # 检查缓存
    cache_path = args.hidden_states_cache
    
    if cache_path and os.path.exists(cache_path):
        print(f"\n>>> Loading cached hidden states from {cache_path}")
        cached_data = torch.load(cache_path, weights_only=False)
        
        safe_data = cached_data['safe_data']
        unsafe_data = cached_data['unsafe_data']
        ood_data = cached_data.get('ood_data')
        all_layer_hidden_states = cached_data['all_layer_hidden_states']
        ood_layer_hidden_states = cached_data.get('ood_layer_hidden_states')
        
        print(f"Loaded {len(safe_data)} safe, {len(unsafe_data)} unsafe samples")
        
        # 不需要加载模型，直接使用缓存的hidden states进行分析
        # 这里需要一个不需要模型的分析函数
        analyze_with_cached_hidden_states(
            safe_data=safe_data,
            unsafe_data=unsafe_data,
            all_layer_hidden_states=all_layer_hidden_states,
            output_dir=output_dir,
            ood_data=ood_data,
            ood_layer_hidden_states=ood_layer_hidden_states,
            device=args.device,
            num_queries_for_selection=args.num_queries_for_selection,
            classifier_hidden_dim=args.classifier_hidden_dim,
            train_epochs=args.train_epochs,
            val_split=args.val_split,
            seed=args.seed,
            train_languages=args.train_languages,
            test_languages=args.test_languages,
            use_deep_mlp=args.use_deep_mlp and not args.use_simple_mlp,
            classifier_dropout=args.classifier_dropout
        )
    else:
        # 加载模型
        print(f"Loading model from {args.model_path}...")
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
        
        # 筛选语言
        if args.languages:
            unsafe_data = [item for item in unsafe_data if item['language'] in args.languages]
        
        print(f"Loaded {len(safe_data)} safe samples, {len(unsafe_data)} unsafe samples")
        
        # 加载 OOD 数据
        ood_data = None
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
        
        # 运行分析
        results = analyze_with_silhouette_selection(
            model=model,
            tokenizer=tokenizer,
            safe_data=safe_data,
            unsafe_data=unsafe_data,
            output_dir=output_dir,
            ood_data=ood_data,
            device=args.device,
            batch_size=args.batch_size,
            num_queries_for_selection=args.num_queries_for_selection,
            classifier_hidden_dim=args.classifier_hidden_dim,
            train_epochs=args.train_epochs,
            val_split=args.val_split,
            seed=args.seed,
            train_languages=args.train_languages,
            test_languages=args.test_languages,
            use_deep_mlp=args.use_deep_mlp and not args.use_simple_mlp,
            classifier_dropout=args.classifier_dropout
        )
        
        # 保存缓存
        if cache_path:
            print(f"\n>>> Saving hidden states cache to {cache_path}")
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            
            all_data = safe_data + unsafe_data
            all_texts = [item['prompt'] for item in all_data]
            
            all_layer_hidden_states = extract_hidden_states_all_layers(
                model, tokenizer, all_texts,
                device=args.device, batch_size=args.batch_size
            )
            
            ood_layer_hidden_states = None
            if ood_data:
                ood_texts = [item['prompt'] for item in ood_data]
                ood_layer_hidden_states = extract_hidden_states_all_layers(
                    model, tokenizer, ood_texts,
                    device=args.device, batch_size=args.batch_size
                )
            
            torch.save({
                'safe_data': safe_data,
                'unsafe_data': unsafe_data,
                'ood_data': ood_data,
                'all_layer_hidden_states': all_layer_hidden_states,
                'ood_layer_hidden_states': ood_layer_hidden_states,
                'model_path': args.model_path
            }, cache_path)
            print(f"Cache saved!")
        
        # 释放模型内存
        del model
        torch.cuda.empty_cache()
    
    print(f"\nResults saved to {output_dir}")


def analyze_with_cached_hidden_states(
    safe_data: List[Dict],
    unsafe_data: List[Dict],
    all_layer_hidden_states: Dict[int, List[torch.Tensor]],
    output_dir: str,
    ood_data: List[Dict] = None,
    ood_layer_hidden_states: Dict[int, List[torch.Tensor]] = None,
    device: str = 'cuda',
    num_queries_for_selection: int = 10,
    classifier_hidden_dim: int = 256,
    train_epochs: int = 20,
    val_split: float = 0.2,
    seed: int = 42,
    train_languages: List[str] = None,
    test_languages: List[str] = None,
    use_deep_mlp: bool = True,
    classifier_dropout: float = 0.2
) -> Dict:
    """使用缓存的hidden states进行分析"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    all_data = safe_data + unsafe_data
    
    print(f"\n{'='*60}")
    print(f"Cross-Lingual Layer Analysis with Silhouette Selection (Cached)")
    print(f"{'='*60}")
    print(f"Total samples: {len(all_data)}")
    print(f"  - Safe samples: {len(safe_data)}")
    print(f"  - Unsafe samples: {len(unsafe_data)}")
    
    has_ood = ood_data is not None and len(ood_data) > 0
    if has_ood:
        print(f"  - OOD samples: {len(ood_data)}")
    
    languages = sorted(list(set(item['language'] for item in all_data)))
    print(f"Languages ({len(languages)}): {languages}")
    
    num_layers = len(all_layer_hidden_states)
    hidden_dim = all_layer_hidden_states[0][0].shape[0]
    print(f"Number of layers: {num_layers}")
    print(f"Hidden dimension: {hidden_dim}")
    
    # ========== 1. 使用Silhouette Score选择最佳层 ==========
    best_layer, layer_metrics = select_best_layer_by_silhouette(
        all_layer_hidden_states=all_layer_hidden_states,
        data=all_data,
        num_queries=num_queries_for_selection,
        device=device
    )
    
    # 绘制层分析图
    plot_layer_silhouette_analysis(layer_metrics, output_dir)
    
    # 保存层选择结果
    layer_selection_results = {
        'best_layer': best_layer,
        'num_queries_for_selection': num_queries_for_selection,
        'layer_metrics': {str(k): v for k, v in layer_metrics.items()}
    }
    with open(os.path.join(output_dir, 'layer_selection_results.json'), 'w') as f:
        json.dump(layer_selection_results, f, indent=2)
    
    # ========== 2. 在最佳层上训练Safety分类器 ==========
    print(f"\n{'='*60}")
    print(f"Training Safety Classifier on Best Layer {best_layer}")
    print(f"{'='*60}")
    
    best_layer_hs = all_layer_hidden_states[best_layer]
    safety_labels = [0 if not item['is_unsafe'] else 1 for item in all_data]
    
    # 按语言分割训练/测试集
    if train_languages is not None or test_languages is not None:
        train_langs = set(train_languages) if train_languages else set(languages)
        test_langs = set(test_languages) if test_languages else set(languages)
        
        train_indices = [i for i, item in enumerate(all_data) if item['language'] in train_langs]
        test_indices = [i for i, item in enumerate(all_data) if item['language'] in test_langs]
        
        np.random.shuffle(train_indices)
        inner_val_size = int(len(train_indices) * 0.1)
        val_indices = train_indices[:inner_val_size]
        train_indices = train_indices[inner_val_size:]
    else:
        indices = list(range(len(all_data)))
        np.random.shuffle(indices)
        
        val_size = int(len(indices) * val_split)
        test_size = int(len(indices) * val_split)
        train_indices = indices[val_size + test_size:]
        val_indices = indices[:val_size]
        test_indices = indices[val_size:val_size + test_size]
    
    print(f"Train samples: {len(train_indices)}")
    print(f"Val samples: {len(val_indices)}")
    print(f"Test samples: {len(test_indices)}")
    
    # 准备数据
    train_hs = [best_layer_hs[i] for i in train_indices]
    val_hs = [best_layer_hs[i] for i in val_indices]
    test_hs = [best_layer_hs[i] for i in test_indices]
    
    train_labels = [safety_labels[i] for i in train_indices]
    val_labels = [safety_labels[i] for i in val_indices]
    test_labels = [safety_labels[i] for i in test_indices]
    
    train_dataset = HiddenStateDataset(train_hs, train_labels)
    val_dataset = HiddenStateDataset(val_hs, val_labels)
    test_dataset = HiddenStateDataset(test_hs, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    
    # 创建分类器
    if use_deep_mlp:
        classifier = DeepMLP(
            input_dim=hidden_dim,
            hidden_dims=[classifier_hidden_dim, classifier_hidden_dim // 2],
            output_dim=1,
            dropout=classifier_dropout,
            use_layer_norm=True,
            activation='gelu'
        )
        print(f"Using DeepMLP: {hidden_dim} -> {classifier_hidden_dim} -> {classifier_hidden_dim//2} -> 1")
    else:
        classifier = SimpleMLP(
            input_dim=hidden_dim,
            hidden_dim=classifier_hidden_dim,
            output_dim=1,
            dropout=classifier_dropout
        )
        print(f"Using SimpleMLP: {hidden_dim} -> {classifier_hidden_dim} -> 1")
    
    # 训练
    train_result = train_classifier(
        classifier, train_loader, val_loader,
        device=device, num_epochs=train_epochs, verbose=True,
        optimize_threshold=True
    )
    
    # 加载最佳模型
    classifier.load_state_dict(train_result['model_state_dict'])
    classifier = classifier.to(device)
    
    best_threshold = train_result['best_threshold']
    
    # ========== 3. 评估 ==========
    print(f"\n{'='*60}")
    print(f"Evaluation Results")
    print(f"{'='*60}")
    
    test_result = evaluate_classifier(classifier, test_loader, device=device, threshold=best_threshold)
    print(f"\nTest Set Performance:")
    print(f"  Accuracy: {test_result['accuracy']:.4f}")
    print(f"  F1: {test_result['f1']:.4f}")
    print(f"  TPR: {test_result['tpr']:.4f}")
    print(f"  FPR: {test_result['fpr']:.4f}")
    
    per_lang_results = evaluate_per_language(
        classifier, best_layer_hs, safety_labels, all_data,
        device=device, threshold=best_threshold
    )
    
    print(f"\nPer-Language Results:")
    print(f"{'Language':<10} {'Accuracy':>10} {'FPR':>10} {'TPR':>10} {'Safe':>8} {'Unsafe':>8}")
    print("-" * 60)
    for lang in sorted(per_lang_results.keys()):
        info = per_lang_results[lang]
        print(f"{lang:<10} {info['accuracy']:>10.4f} {info['fpr']:>10.4f} {info['tpr']:>10.4f} "
              f"{info['num_safe']:>8} {info['num_unsafe']:>8}")
    
    # OOD评估
    ood_result = None
    ood_per_lang = {}
    if has_ood and ood_layer_hidden_states:
        ood_hs = ood_layer_hidden_states[best_layer]
        ood_labels = [1] * len(ood_data)
        
        ood_dataset = HiddenStateDataset(ood_hs, ood_labels)
        ood_loader = DataLoader(ood_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
        
        ood_result = evaluate_classifier(classifier, ood_loader, device=device, threshold=best_threshold)
        
        print(f"\nOOD Performance:")
        print(f"  Accuracy: {ood_result['accuracy']:.4f}")
        
        ood_per_lang = evaluate_per_language(
            classifier, ood_hs, ood_labels, ood_data,
            device=device, threshold=best_threshold
        )
    
    # ========== 4. 保存结果 ==========
    threshold_str = f"{int(round(best_threshold * 100)):03d}"
    model_save_path = os.path.join(
        output_dir,
        f'classifier_layer{best_layer}_hidden{classifier_hidden_dim}_threshold{threshold_str}.pt'
    )
    
    avg_fpr = np.mean([v['fpr'] for v in per_lang_results.values() if v['num_safe'] > 0])
    avg_tpr = np.mean([v['tpr'] for v in per_lang_results.values() if v['num_unsafe'] > 0])
    
    torch.save({
        'model_state_dict': classifier.state_dict(),
        'layer_idx': best_layer,
        'hidden_dim': hidden_dim,
        'classifier_hidden_dim': classifier_hidden_dim,
        'threshold': best_threshold,
        'test_accuracy': test_result['accuracy'],
        'avg_fpr': avg_fpr,
        'avg_tpr': avg_tpr,
        'ood_accuracy': ood_result['accuracy'] if ood_result else 0.0,
        'layer_selection_method': 'silhouette_score_diff'
    }, model_save_path)
    
    print(f"\nModel saved to {model_save_path}")
    
    final_summary = {
        'model_path': model_save_path,
        'layer_idx': best_layer,
        'layer_selection_method': 'silhouette_score_diff',
        'layer_selection_score': layer_metrics[best_layer]['score_diff'],
        'classifier_hidden_dim': classifier_hidden_dim,
        'threshold': best_threshold,
        'in_distribution': {
            'test_accuracy': test_result['accuracy'],
            'avg_fpr': avg_fpr,
            'avg_tpr': avg_tpr,
            'per_lang_accuracy': {lang: info['accuracy'] for lang, info in per_lang_results.items()}
        },
        'out_of_distribution': {
            'overall_accuracy': ood_result['accuracy'] if ood_result else 0.0,
            'per_lang_accuracy': {lang: info['accuracy'] for lang, info in ood_per_lang.items()} if ood_per_lang else {}
        }
    }
    
    with open(os.path.join(output_dir, 'final_model_summary.json'), 'w') as f:
        json.dump(final_summary, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"FINAL MODEL PERFORMANCE SUMMARY")
    print(f"{'='*70}")
    print(f"Layer: {best_layer} (selected by Silhouette Score Diff: {layer_metrics[best_layer]['score_diff']:.4f})")
    print(f"Threshold: {best_threshold:.3f}")
    print(f"Test Accuracy: {test_result['accuracy']:.4f}")
    if ood_result:
        print(f"OOD Accuracy: {ood_result['accuracy']:.4f}")
    print(f"{'='*70}")
    
    return {
        'best_layer': best_layer,
        'layer_metrics': layer_metrics,
        'test_result': test_result,
        'per_lang_results': per_lang_results,
        'ood_result': ood_result,
        'final_summary': final_summary
    }


if __name__ == "__main__":
    main()

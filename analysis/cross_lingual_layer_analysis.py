"""
跨语言层分析预实验

目标：找出模型中跨语言相似度最高的层
方法：
    对于每一层 ℓ:
    1. 训练 Safety 分类器 -> 得到 SafetyAcc(ℓ)
    2. 训练 Language 分类器 -> 得到 LangAcc(ℓ)
    3. 计算分数: Score(ℓ) = SafetyAcc(ℓ) - α⋅LangAcc(ℓ)
    
    高分意味着该层:
    - 能很好地保留安全性信息（区分 safe/unsafe）
    - 但丢失了语言信息（无法区分语言）
    这正是跨语言对齐的理想状态
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


# ==================== MLP分类器 ====================

class SimpleMLP(nn.Module):
    """
    简单的MLP分类器（单hidden layer）
    
    结构: input_dim -> hidden_dim -> output_dim
    """
    
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
            # nn.Linear(hidden_dim, hidden_dim//2),
            # nn.LayerNorm(hidden_dim//2),
            # nn.GELU(),
            # nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        if logits.shape[-1] == 1:
            return torch.sigmoid(logits)
        else:
            return torch.softmax(logits, dim=-1)


class DeepMLP(nn.Module):
    """
    更深的MLP分类器（多hidden layer），用于Safety分类以提升OOD泛化
    
    结构: input_dim -> hidden_dim1 -> hidden_dim2 -> output_dim
    
    增强泛化的设计:
    1. 多层hidden layer
    2. BatchNorm/LayerNorm 用于稳定训练
    3. Dropout 防止过拟合
    4. 可选的残差连接
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [512, 256],
        output_dim: int = 1,
        dropout: float = 0.2,
        use_layer_norm: bool = True,
        use_residual: bool = False,
        activation: str = 'gelu'
    ):
        super().__init__()
        
        self.use_residual = use_residual
        
        # 激活函数
        if activation == 'gelu':
            act_fn = nn.GELU()
        elif activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'silu':
            act_fn = nn.SiLU()
        else:
            act_fn = nn.GELU()
        
        # 构建多层网络
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(act_fn)
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # 残差连接的投影层（如果维度不匹配）
        if use_residual and input_dim != output_dim:
            self.residual_proj = nn.Linear(input_dim, output_dim)
        else:
            self.residual_proj = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.mlp(x)
        
        if self.use_residual:
            if self.residual_proj is not None:
                residual = self.residual_proj(x)
            else:
                residual = x
            out = out + 0.1 * residual  # 小的残差权重
        
        return out
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        if logits.shape[-1] == 1:
            return torch.sigmoid(logits)
        else:
            return torch.softmax(logits, dim=-1)


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
    hidden_states = hidden_states.float()  # 确保 float32
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
    """加载 harmbench 或 multijail 数据（OOD 有害问题）
    
    支持两种格式：
    1. 翻译格式：包含 original_query 和 translations 字段
    2. 简单格式：包含 prompt 和 lingual 字段（如 multijail_prepared.json）
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    if max_samples and len(raw_data) > max_samples:
        raw_data = raw_data[:max_samples]
    
    # 检测数据格式
    if raw_data and 'original_query' in raw_data[0]:
        # 翻译格式，使用原有的 load_safety_data
        return load_safety_data(data_path, max_samples, idx_offset)
    else:
        # 简单格式（multijail_prepared.json）
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
    """
    提取所有层的 hidden states
    
    Args:
        model: 预训练模型
        tokenizer: tokenizer
        texts: 文本列表
        device: 设备
        batch_size: 批大小
        max_length: 最大序列长度
    
    Returns:
        字典 {layer_idx: [hidden_states]}
    """
    model.eval()
    
    # 获取模型层数
    num_layers = model.config.num_hidden_layers + 1  # +1 for embedding layer
    
    # 初始化存储
    all_hidden_states = {layer_idx: [] for layer_idx in range(num_layers)}
    
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
        
        # 获取每层的 hidden states
        for layer_idx in range(num_layers):
            layer_output = outputs.hidden_states[layer_idx]
            
            for j in range(len(batch_texts)):
                attention_mask = inputs['attention_mask'][j]
                last_token_idx = attention_mask.sum() - 1
                
                # 使用最后一个有效 token 的 hidden state
                hidden_state = layer_output[j, last_token_idx, :].cpu()
                all_hidden_states[layer_idx].append(hidden_state)
    
    return all_hidden_states


# ==================== 训练和评估函数 ====================

def train_classifier(
    model: SimpleMLP,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    device: str = 'cuda',
    num_epochs: int = 20,
    learning_rate: float = 1e-3,
    early_stopping_patience: int = 5,
    verbose: bool = False,
    pos_weight: float = None,
    optimize_threshold: bool = True
) -> Dict:
    """
    训练分类器
    
    Args:
        model: 分类器模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        num_classes: 类别数（1 表示二分类）
        device: 设备
        num_epochs: 训练轮数
        learning_rate: 学习率
        early_stopping_patience: 早停耐心值
        verbose: 是否打印详细信息
        pos_weight: 正样本权重（用于处理类别不平衡，unsafe类的权重）
        optimize_threshold: 是否在验证集上优化分类阈值
    
    Returns:
        训练结果字典
    """
    model = model.to(device)
    
    # 损失函数 - 支持类别权重
    if num_classes == 1:
        if pos_weight is not None:
            pw = torch.tensor([pos_weight], device=device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
        else:
            criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)
    
    best_val_acc = 0.0
    patience_counter = 0
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
            
            if num_classes == 1:
                loss = criterion(logits.squeeze(-1), labels.float())
            else:
                loss = criterion(logits, labels)
            
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
                
                if num_classes == 1:
                    preds = (torch.sigmoid(logits.squeeze(-1)) > 0.5).long().cpu()
                else:
                    preds = logits.argmax(dim=-1).cpu()
                
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.tolist())
        
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='macro' if num_classes > 1 else 'binary')
        
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        scheduler.step(val_acc)
        
        if verbose:
            print(f"  Epoch {epoch+1}/{num_epochs}: Loss={avg_loss:.4f}, Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}")
        
        # 早停
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1}")
                break
    
    # 恢复最佳模型
    model.load_state_dict(best_state)
    
    # 在验证集上找最优阈值
    best_threshold = 0.5
    if optimize_threshold and num_classes == 1:
        model.eval()
        all_probs_val = []
        all_labels_val = []
        
        with torch.no_grad():
            for batch in val_loader:
                hidden_states = batch['hidden_states'].to(device)
                labels = batch['labels']
                logits = model(hidden_states)
                probs = torch.sigmoid(logits.squeeze(-1)).cpu()
                all_probs_val.extend(probs.tolist())
                all_labels_val.extend(labels.tolist())
        
        # 尝试不同阈值，优化目标：最大化 accuracy
        best_threshold, threshold_metrics = find_optimal_threshold(
            all_probs_val, all_labels_val, optimize_for='accuracy'
        )
        
        if verbose:
            print(f"  Optimal threshold: {best_threshold:.3f} "
                  f"(Acc={threshold_metrics['accuracy']:.4f})")
    
    return {
        'best_val_acc': best_val_acc,
        'best_val_f1': max(history['val_f1']),
        'best_threshold': best_threshold,
        'history': history
    }


def find_optimal_threshold(
    probs: List[float],
    labels: List[int],
    optimize_for: str = 'accuracy',
    max_fpr: float = 0.01
) -> Tuple[float, Dict]:
    """
    找到最优分类阈值
    
    Args:
        probs: 预测概率列表
        labels: 真实标签列表
        optimize_for: 优化目标
            - 'accuracy': 最大化准确率（默认）
            - 'f1': 最大化 F1
            - 'balanced': 平衡 TPR 和 FPR
            - 'low_fpr': 在 FPR <= max_fpr 的约束下最大化 TPR
        max_fpr: 最大允许的假阳性率（仅在 optimize_for='low_fpr' 时使用）
    
    Returns:
        最优阈值和对应的指标
    """
    probs = np.array(probs)
    labels = np.array(labels)
    
    best_threshold = 0.5
    best_score = -float('inf')
    best_metrics = {}
    
    # 阈值搜索范围限制在 0.30-0.70（步长 0.01）
    for threshold in np.arange(0.30, 0.71, 0.01):
        preds = (probs > threshold).astype(int)
        
        # 计算指标
        tp = ((preds == 1) & (labels == 1)).sum()
        tn = ((preds == 0) & (labels == 0)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # 真阳性率 (Recall for unsafe)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # 假阳性率 (Safe被误判为Unsafe)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * precision * tpr / (precision + tpr) if (precision + tpr) > 0 else 0
        acc = (tp + tn) / len(labels)
        
        if optimize_for == 'accuracy':
            score = acc
        elif optimize_for == 'f1':
            score = f1
        elif optimize_for == 'balanced':
            # 平衡 TPR 和 (1 - FPR)
            score = 0.5 * tpr + 0.5 * (1 - fpr)
        elif optimize_for == 'low_fpr':
            # 严格约束：FPR 必须 <= max_fpr
            if fpr <= max_fpr:
                # 在满足 FPR 约束的阈值中，选择 TPR 最高的
                score = tpr + 1000  # 加1000确保满足约束的总是优于不满足的
            else:
                # 不满足约束，使用负FPR作为分数（越接近约束越好）
                score = -fpr
        else:
            score = acc  # 默认优化accuracy
        
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


def evaluate_classifier(
    model: SimpleMLP,
    test_loader: DataLoader,
    num_classes: int,
    device: str = 'cuda',
    threshold: float = 0.5
) -> Dict:
    """
    评估分类器
    
    Args:
        model: 分类器模型
        test_loader: 测试数据加载器
        num_classes: 类别数
        device: 设备
        threshold: 分类阈值（仅用于二分类）
    
    Returns:
        包含 accuracy, f1, fpr, tpr 的字典
    """
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
            
            if num_classes == 1:
                probs = torch.sigmoid(logits.squeeze(-1))
                preds = (probs > threshold).long().cpu()  # 使用传入的阈值
                all_probs.extend(probs.cpu().tolist())
            else:
                probs = torch.softmax(logits, dim=-1)
                preds = logits.argmax(dim=-1).cpu()
                all_probs.extend(probs.cpu().tolist())
            
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro' if num_classes > 1 else 'binary', zero_division=0)
    
    # 计算 FPR 和 TPR
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    tp = ((all_preds == 1) & (all_labels == 1)).sum()
    tn = ((all_preds == 0) & (all_labels == 0)).sum()
    fp = ((all_preds == 1) & (all_labels == 0)).sum()
    fn = ((all_preds == 0) & (all_labels == 1)).sum()
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall for unsafe
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False positive rate
    
    return {
        'accuracy': acc, 
        'f1': f1,
        'tpr': tpr,
        'fpr': fpr,
        'threshold': threshold
    }


def evaluate_per_language(
    model: SimpleMLP,
    hidden_states: List[torch.Tensor],
    labels: List[int],
    data: List[Dict],
    device: str = 'cuda',
    threshold: float = 0.5
) -> Dict[str, Dict]:
    """
    计算每个语言的分类准确率和假阳性率
    
    Args:
        model: 训练好的分类器
        hidden_states: hidden states 列表
        labels: 标签列表
        data: 数据列表，包含 language 字段
        device: 设备
        threshold: 分类阈值
    
    Returns:
        每个语言的准确率、假阳性率等
    """
    from collections import defaultdict
    
    model.eval()
    model = model.to(device)
    
    # 按语言分组
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
            
            # 转换为 tensor，确保类型为 float32
            hs_tensor = torch.stack([
                h if isinstance(h, torch.Tensor) else torch.tensor(h)
                for h in lang_hs
            ]).to(device).float()
            labels_tensor = torch.tensor(lang_labels, device=device)
            
            # 预测
            outputs = model(hs_tensor)
            if outputs.dim() > 1 and outputs.size(1) == 1:
                outputs = outputs.squeeze(1)
            probs = torch.sigmoid(outputs)
            preds = (probs > threshold).long()
            
            # 计算准确率
            correct = (preds == labels_tensor).sum().item()
            total = len(indices)
            acc = correct / total
            
            # 统计 safe/unsafe 数量
            num_safe = sum(1 for l in lang_labels if l == 0)
            num_unsafe = sum(1 for l in lang_labels if l == 1)
            
            # 计算 FPR 和 TPR
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


def compute_cross_lingual_similarity(
    hidden_states: List[torch.Tensor],
    data: List[Dict],
    device: str = 'cuda'
) -> float:
    """
    计算同一问题在不同语言下与英语的 hidden state 相似度
    
    对于每个问题（通过 idx 标识），计算所有非英语版本与英语版本之间的余弦相似度，
    然后取平均值作为该层的跨语言相似度分数
    
    Args:
        hidden_states: 该层的 hidden states 列表
        data: 数据列表，每个元素包含 idx 和 language
        device: 设备
    
    Returns:
        平均跨语言相似度（越高越好，表示不同语言的表示与英语越接近）
    """
    from collections import defaultdict
    import torch.nn.functional as F
    
    # 按 idx 分组
    idx_to_items = defaultdict(dict)
    for i, item in enumerate(data):
        idx_to_items[item['idx']][item['language']] = i
    
    all_similarities = []
    
    for idx, lang_to_idx in idx_to_items.items():
        # 必须有英语版本
        if 'en' not in lang_to_idx:
            continue
        
        # 获取英语版本的 hidden state
        en_idx = lang_to_idx['en']
        en_hs = hidden_states[en_idx]
        if isinstance(en_hs, torch.Tensor):
            en_hs = en_hs.to(device)
        else:
            en_hs = torch.tensor(en_hs, device=device)
        
        # 计算其他语言与英语的相似度
        for lang, data_idx in lang_to_idx.items():
            if lang == 'en':
                continue
            
            other_hs = hidden_states[data_idx]
            if isinstance(other_hs, torch.Tensor):
                other_hs = other_hs.to(device)
            else:
                other_hs = torch.tensor(other_hs, device=device)
            
            sim = F.cosine_similarity(
                en_hs.unsqueeze(0), 
                other_hs.unsqueeze(0)
            ).item()
            all_similarities.append(sim)
    
    if len(all_similarities) == 0:
        return 0.0
    
    return np.mean(all_similarities)


# ==================== 主分析流程 ====================

def analyze_all_layers(
    model,
    tokenizer,
    safe_data: List[Dict],
    unsafe_data: List[Dict],
    output_dir: str,
    ood_data: List[Dict] = None,
    device: str = 'cuda',
    batch_size: int = 8,
    classifier_hidden_dim: int = 256,
    train_epochs: int = 20,
    val_split: float = 0.2,
    alpha: float = 1.0,
    seed: int = 42,
    train_languages: List[str] = None,
    test_languages: List[str] = None,
    use_deep_mlp: bool = True,
    classifier_dropout: float = 0.2
) -> Dict:
    """
    分析所有层的跨语言相似度
    
    Args:
        model: 预训练模型
        tokenizer: tokenizer
        safe_data: 无害数据列表
        unsafe_data: 有害数据列表
        output_dir: 输出目录
        ood_data: OOD数据列表（如harmbench）
        device: 设备
        batch_size: 批大小
        classifier_hidden_dim: 分类器隐藏层维度
        train_epochs: 训练轮数
        val_split: 验证集比例
        alpha: 语言分类器权重
        seed: 随机种子
        train_languages: 训练语言列表（默认使用所有语言）
        test_languages: 测试语言列表（默认使用所有语言）
    
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
    print(f"Cross-Lingual Layer Analysis")
    print(f"{'='*60}")
    print(f"Total samples: {len(all_data)}")
    print(f"  - Safe samples: {len(safe_data)}")
    print(f"  - Unsafe samples: {len(unsafe_data)}")
    
    # OOD数据处理
    has_ood = ood_data is not None and len(ood_data) > 0
    if has_ood:
        ood_texts = [item['prompt'] for item in ood_data]
        print(f"  - OOD (harmbench) samples: {len(ood_data)}")
    
    # 获取语言列表
    languages = sorted(list(set(item['language'] for item in all_data)))
    lang_to_idx = {lang: idx for idx, lang in enumerate(languages)}
    num_languages = len(languages)
    print(f"Languages ({num_languages}): {languages}")
    
    # 提取所有层的 hidden states
    print(f"\nExtracting hidden states from all layers...")
    all_layer_hidden_states = extract_hidden_states_all_layers(
        model, tokenizer, all_texts,
        device=device, batch_size=batch_size
    )
    
    # 提取 OOD 数据的 hidden states
    if has_ood:
        print(f"Extracting OOD hidden states from all layers...")
        ood_layer_hidden_states = extract_hidden_states_all_layers(
            model, tokenizer, ood_texts,
            device=device, batch_size=batch_size
        )
    
    num_layers = len(all_layer_hidden_states)
    hidden_dim = all_layer_hidden_states[0][0].shape[0]
    print(f"Number of layers: {num_layers}")
    print(f"Hidden dimension: {hidden_dim}")
    
    # 准备标签
    safety_labels = [0 if not item['is_unsafe'] else 1 for item in all_data]
    
    # 按语言分割训练/测试集
    if train_languages is not None or test_languages is not None:
        # 跨语言泛化模式：用特定语言训练，其他语言测试
        train_langs = set(train_languages) if train_languages else set(languages)
        test_langs = set(test_languages) if test_languages else set(languages)
        
        train_indices = [i for i, item in enumerate(all_data) if item['language'] in train_langs]
        val_indices = [i for i, item in enumerate(all_data) if item['language'] in test_langs]
        
        # 在训练集内部做一个小的验证划分用于早停
        np.random.shuffle(train_indices)
        inner_val_size = int(len(train_indices) * 0.1)
        inner_val_indices = train_indices[:inner_val_size]
        train_indices = train_indices[inner_val_size:]
        
        print(f"\n>>> Cross-lingual generalization mode:")
        print(f"    Train languages: {sorted(train_langs)}")
        print(f"    Test languages: {sorted(test_langs)}")
        print(f"\nTrain samples: {len(train_indices)}")
        print(f"Inner val samples (for early stopping): {len(inner_val_indices)}")
        print(f"Test samples (unseen languages): {len(val_indices)}")
        
        use_cross_lingual = True
    else:
        # 标准模式：随机分割
        indices = list(range(len(all_data)))
        np.random.shuffle(indices)
        
        val_size = int(len(indices) * val_split)
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]
        inner_val_indices = None
        
        print(f"\nTrain samples: {len(train_indices)}")
        print(f"Val samples: {len(val_indices)}")
        
        use_cross_lingual = False
    
    # 存储每层的结果
    layer_results = {}
    
    for layer_idx in tqdm(range(num_layers), desc="Analyzing layers"):
        layer_hidden_states = all_layer_hidden_states[layer_idx]
        
        # 分割数据
        train_hs = [layer_hidden_states[i] for i in train_indices]
        test_hs = [layer_hidden_states[i] for i in val_indices]
        
        train_safety = [safety_labels[i] for i in train_indices]
        test_safety = [safety_labels[i] for i in val_indices]
        
        # 跨语言模式下，用 inner_val 作为早停验证集
        if use_cross_lingual and inner_val_indices is not None:
            inner_val_hs = [layer_hidden_states[i] for i in inner_val_indices]
            inner_val_safety = [safety_labels[i] for i in inner_val_indices]
        else:
            inner_val_hs = test_hs
            inner_val_safety = test_safety
        
        # ========== 训练 Safety 分类器 ==========
        safety_train_dataset = HiddenStateDataset(train_hs, train_safety)
        safety_val_dataset = HiddenStateDataset(inner_val_hs, inner_val_safety)
        
        safety_train_loader = DataLoader(
            safety_train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn
        )
        safety_val_loader = DataLoader(
            safety_val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn
        )
        
        # 不使用类别权重，直接优化accuracy
        pos_weight = None
        
        # 创建 Safety 分类器
        if use_deep_mlp:
            # DeepMLP（2层hidden layer）以提升OOD泛化
            # 结构: input -> hidden_dim -> hidden_dim/2 -> 1
            safety_classifier = DeepMLP(
                input_dim=hidden_dim,
                hidden_dims=[classifier_hidden_dim, classifier_hidden_dim],
                output_dim=1,
                dropout=classifier_dropout,
                use_layer_norm=True,
                use_residual=True,
                activation='gelu'
            )
        else:
            # SimpleMLP（单层hidden layer）
            safety_classifier = SimpleMLP(
                input_dim=hidden_dim,
                hidden_dim=classifier_hidden_dim,
                output_dim=1,
                dropout=classifier_dropout
            )
        
        safety_result = train_classifier(
            safety_classifier, safety_train_loader, safety_val_loader,
            num_classes=1, device=device, num_epochs=train_epochs, verbose=False,
            pos_weight=pos_weight, optimize_threshold=True
        )
        
        safety_acc = safety_result['best_val_acc']
        best_threshold = safety_result.get('best_threshold', 0.5)
        
        # ========== 在 OOD 数据上评估 Safety 分类器 ==========
        ood_safety_acc = 0.0
        ood_fpr = 0.0
        if has_ood:
            ood_hs = ood_layer_hidden_states[layer_idx]
            # OOD 数据全部是 unsafe (label=1)
            ood_safety_labels = [1] * len(ood_hs)
            
            ood_dataset = HiddenStateDataset(ood_hs, ood_safety_labels)
            ood_loader = DataLoader(
                ood_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn
            )
            
            ood_result = evaluate_classifier(
                safety_classifier, ood_loader, num_classes=1, device=device,
                threshold=best_threshold
            )
            ood_safety_acc = ood_result['accuracy']
        
        # ========== 在测试语言上评估 Safety 分类器（跨语言泛化）==========
        test_safety_acc = 0.0
        test_fpr = 0.0
        if use_cross_lingual:
            test_safety_dataset = HiddenStateDataset(test_hs, test_safety)
            test_safety_loader = DataLoader(
                test_safety_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn
            )
            test_safety_result = evaluate_classifier(
                safety_classifier, test_safety_loader, num_classes=1, device=device,
                threshold=best_threshold
            )
            test_safety_acc = test_safety_result['accuracy']
            test_fpr = test_safety_result.get('fpr', 0.0)
        
        # ========== 计算每个语言的准确率 ==========
        per_lang_acc = evaluate_per_language(
            safety_classifier, layer_hidden_states, safety_labels, all_data, 
            device=device, threshold=best_threshold
        )
        
        # 计算 OOD 每个语言的准确率
        per_lang_ood_acc = {}
        if has_ood:
            ood_hs = ood_layer_hidden_states[layer_idx]
            ood_safety_labels_list = [1] * len(ood_data)  # OOD 全是 unsafe
            per_lang_ood_acc = evaluate_per_language(
                safety_classifier, ood_hs, ood_safety_labels_list, ood_data, 
                device=device, threshold=best_threshold
            )
        
        # ========== 计算跨语言相似度 ==========
        # 使用同一问题在不同语言下的 hidden state 相似度
        cross_lingual_sim = compute_cross_lingual_similarity(
            layer_hidden_states, all_data, device=device
        )
        
        # 计算跨语言分数
        # Score = SafetyAcc + α * CrossLingualSim（相似度越高越好）
        # 在跨语言模式下，使用测试语言的safety acc
        effective_safety_acc = test_safety_acc if use_cross_lingual else safety_acc
        score = effective_safety_acc + alpha * cross_lingual_sim
        
        # 计算平均 FPR
        avg_fpr = np.mean([v['fpr'] for v in per_lang_acc.values() if v['num_safe'] > 0])
        
        layer_results[layer_idx] = {
            'safety_acc': safety_acc,
            'test_safety_acc': test_safety_acc,
            'cross_lingual_sim': cross_lingual_sim,
            'score': score,
            'ood_safety_acc': ood_safety_acc,
            'avg_fpr': avg_fpr,
            'threshold': best_threshold,
            'per_lang_acc': per_lang_acc,
            'per_lang_ood_acc': per_lang_ood_acc,
            'safety_history': safety_result['history'],
            # 保存模型状态，以便后续直接使用最优层的模型
            'model_state_dict': {k: v.cpu().clone() for k, v in safety_classifier.state_dict().items()},
            'pos_weight': pos_weight
        }
        
        print(f"Layer {layer_idx:2d}: TrainAcc={safety_acc:.4f}, TestAcc={test_safety_acc:.4f}, cross_lingual_sim={cross_lingual_sim:.4f}, "
            f"FPR={avg_fpr:.4f}, Threshold={best_threshold:.2f}, "
            f"OOD={ood_safety_acc:.4f}, Score={score:.4f}")
        
        # Log 每个语言的准确率和FPR
        print(f"         Per-lang acc (FPR): ", end="")
        for lang in sorted(per_lang_acc.keys()):
            info = per_lang_acc[lang]
            print(f"{lang}={info['accuracy']:.3f}({info['fpr']:.2f})", end=" ")
        print()
        if has_ood and per_lang_ood_acc:
            print(f"         Per-lang OOD: ", end="")
            for lang in sorted(per_lang_ood_acc.keys()):
                print(f"{lang}={per_lang_ood_acc[lang]['accuracy']:.3f}", end=" ")
            print()
    
    # 找到最佳层
    best_layer = max(layer_results.keys(), key=lambda k: layer_results[k]['score'])
    best_score = layer_results[best_layer]['score']
    
    print(f"\n{'='*60}")
    print(f"Best Layer: {best_layer} (Score: {best_score:.4f})")
    print(f"  SafetyAcc (train lang): {layer_results[best_layer]['safety_acc']:.4f}")
    if use_cross_lingual:
        print(f"  SafetyAcc (test lang): {layer_results[best_layer]['test_safety_acc']:.4f}")
    print(f"  CrossLingualSim: {layer_results[best_layer]['cross_lingual_sim']:.4f}")
    print(f"{'='*60}")
    
    # 保存结果
    results = {
        'config': {
            'alpha': alpha,
            'classifier_hidden_dim': classifier_hidden_dim,
            'train_epochs': train_epochs,
            'val_split': val_split,
            'num_layers': num_layers,
            'hidden_dim': hidden_dim,
            'num_languages': num_languages,
            'languages': languages,
            'train_languages': list(train_languages) if train_languages else None,
            'test_languages': list(test_languages) if test_languages else None,
            'num_safe_samples': len(safe_data),
            'num_unsafe_samples': len(unsafe_data),
            'num_ood_samples': len(ood_data) if has_ood else 0
        },
        'layer_results': {
            str(k): {
                'safety_acc': v['safety_acc'],
                'test_safety_acc': v['test_safety_acc'],
                'cross_lingual_sim': v['cross_lingual_sim'],
                'score': v['score'],
                'ood_safety_acc': v['ood_safety_acc'],
                'per_lang_acc': {lang: info['accuracy'] for lang, info in v['per_lang_acc'].items()},
                'per_lang_ood_acc': {lang: info['accuracy'] for lang, info in v['per_lang_ood_acc'].items()} if v['per_lang_ood_acc'] else {}
            } for k, v in layer_results.items()
        },
        'best_layer': best_layer,
        'best_score': best_score
    }
    
    with open(os.path.join(output_dir, 'layer_analysis_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # 绘图
    plot_layer_analysis(layer_results, alpha, output_dir, has_ood=has_ood)
    
    # ========== 使用层分析时的最优模型（不重新训练）==========
    print(f"\n{'='*60}")
    print(f"Loading Best Layer {best_layer} Model (No Retraining)")
    print(f"{'='*60}")
    
    best_layer_hs = all_layer_hidden_states[best_layer]
    best_layer_result = layer_results[best_layer]
    
    # 直接使用层分析时保存的模型状态
    final_threshold = best_layer_result['threshold']
    final_val_acc = best_layer_result['safety_acc']
    final_pos_weight = best_layer_result['pos_weight']
    
    # 重建模型并加载状态
    if use_deep_mlp:
        best_classifier = DeepMLP(
            input_dim=hidden_dim,
            hidden_dims=[classifier_hidden_dim, classifier_hidden_dim],
            output_dim=1,
            dropout=classifier_dropout,
            use_layer_norm=True,
            use_residual=False,
            activation='gelu'
        )
        print(f"Using DeepMLP: {hidden_dim} -> {classifier_hidden_dim} -> {classifier_hidden_dim//2} -> 1")
    else:
        best_classifier = SimpleMLP(
            input_dim=hidden_dim,
            hidden_dim=classifier_hidden_dim,
            output_dim=1,
            dropout=classifier_dropout
        )
        print(f"Using SimpleMLP: {hidden_dim} -> {classifier_hidden_dim} -> 1")
    
    # 加载层分析时保存的模型权重
    best_classifier.load_state_dict(best_layer_result['model_state_dict'])
    best_classifier = best_classifier.to(device)
    best_classifier.eval()
    
    print(f"Loaded model from layer analysis (no retraining)")
    print(f"Layer: {best_layer}, Threshold: {final_threshold:.3f}, cross_lingual_sim={cross_lingual_sim:.4f}")
    print(f"Training Val Acc: {final_val_acc:.4f}, OOD Acc: {best_layer_result['ood_safety_acc']:.4f}")
    
    # 保存模型 - 阈值格式修复：乘以100后取整
    threshold_int = int(round(final_threshold * 100))
    threshold_str = f"{threshold_int:03d}"  # 0.75 -> 075, 0.50 -> 050
    model_save_path = os.path.join(
        output_dir, 
        f'classifier_layer{best_layer}_hidden{classifier_hidden_dim}_threshold{threshold_str}.pt'
    )
    
    # ========== 详细评估每个语言 ==========
    print(f"\n{'='*60}")
    print(f"Detailed Per-Language Evaluation")
    print(f"{'='*60}")
    print(f"Using threshold: {final_threshold:.3f}")
    
    # 在所有数据上评估
    final_per_lang = evaluate_per_language(
        best_classifier, best_layer_hs, safety_labels, all_data, 
        device=device, threshold=final_threshold
    )
    
    print(f"\n{'='*50}")
    print(f"In-Distribution Data (All Languages)")
    print(f"{'='*50}")
    print(f"{'Language':<10} {'Acc':>8} {'FPR':>8} {'Safe':>8} {'Unsafe':>8}")
    print("-" * 50)
    for lang in sorted(final_per_lang.keys()):
        info = final_per_lang[lang]
        print(f"{lang:<10} {info['accuracy']:>8.4f} {info['fpr']:>8.4f} {info['num_safe']:>8} {info['num_unsafe']:>8}")
    
    # 计算整体 FPR
    total_safe = sum(v['num_safe'] for v in final_per_lang.values())
    total_fp = sum(int(v['fpr'] * v['num_safe']) for v in final_per_lang.values() if v['num_safe'] > 0)
    overall_fpr = total_fp / total_safe if total_safe > 0 else 0
    print("-" * 50)
    print(f"{'Overall FPR':<10} {'':<8} {overall_fpr:>8.4f}")
    
    # OOD 评估
    if has_ood:
        ood_layer_hs = ood_layer_hidden_states[best_layer]
        ood_labels = [1] * len(ood_data)
        final_per_lang_ood = evaluate_per_language(
            best_classifier, ood_layer_hs, ood_labels, ood_data, 
            device=device, threshold=final_threshold
        )
        
        print(f"\n{'='*40}")
        print(f"OOD Data (MultiJail) - All Languages")
        print(f"{'='*40}")
        print(f"{'Language':<10} {'Acc':>8} {'Total':>8}")
        print("-" * 28)
        for lang in sorted(final_per_lang_ood.keys()):
            info = final_per_lang_ood[lang]
            print(f"{lang:<10} {info['accuracy']:>8.4f} {info['total']:>8}")
        
        # 计算总体 OOD 准确率
        total_correct = sum(v['correct'] for v in final_per_lang_ood.values())
        total_samples = sum(v['total'] for v in final_per_lang_ood.values())
        overall_ood_acc = total_correct / total_samples if total_samples > 0 else 0
        print("-" * 28)
        print(f"{'Overall':<10} {overall_ood_acc:>8.4f} {total_samples:>8}")
    
    # 保存详细结果
    detailed_results = {
        'best_layer': best_layer,
        'best_score': best_score,
        'final_val_acc': final_val_acc,
        'per_lang_results': {
            lang: {
                'accuracy': info['accuracy'],
                'total': info['total'],
                'num_safe': info['num_safe'],
                'num_unsafe': info['num_unsafe']
            } for lang, info in final_per_lang.items()
        }
    }
    
    if has_ood:
        detailed_results['per_lang_ood_results'] = {
            lang: {
                'accuracy': info['accuracy'],
                'total': info['total']
            } for lang, info in final_per_lang_ood.items()
        }
        detailed_results['overall_ood_acc'] = overall_ood_acc
    
    with open(os.path.join(output_dir, 'best_layer_detailed_results.json'), 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    # 计算汇总指标
    avg_fpr = np.mean([v['fpr'] for v in final_per_lang.values() if v['num_safe'] > 0])
    avg_tpr = np.mean([v['tpr'] for v in final_per_lang.values() if v['num_unsafe'] > 0])
    final_ood_acc = overall_ood_acc if has_ood else 0.0
    
    # 保存模型（包含完整评估指标）
    torch.save({
        'model_state_dict': best_classifier.state_dict(),
        'layer_idx': best_layer,
        'hidden_dim': hidden_dim,
        'classifier_hidden_dim': classifier_hidden_dim,
        'val_acc': final_val_acc,
        'test_acc': final_val_acc,  # 这里 val 就是 test
        'threshold': final_threshold,
        'pos_weight': final_pos_weight,
        'avg_fpr': avg_fpr,
        'avg_tpr': avg_tpr,
        'ood_accuracy': final_ood_acc,
        'train_languages': list(train_languages) if train_languages else None,
        'test_languages': list(test_languages) if test_languages else None
    }, model_save_path)
    print(f"\nModel saved to {model_save_path}")
    
    # OOD 汇总指标
    ood_avg_acc = 0.0
    if has_ood:
        ood_avg_acc = np.mean([v['accuracy'] for v in final_per_lang_ood.values()])
    
    # 保存 final_model_summary.json（分开汇报 In-Distribution 和 OOD）
    final_summary = {
        'model_path': model_save_path,
        'layer_idx': best_layer,
        'classifier_hidden_dim': classifier_hidden_dim,
        'threshold': final_threshold,
        'pos_weight': final_pos_weight,
        # In-Distribution 结果（原始训练数据集）
        'in_distribution': {
            'test_accuracy': final_val_acc,
            'avg_fpr': avg_fpr,
            'avg_tpr': avg_tpr,
            'per_lang_accuracy': {lang: info['accuracy'] for lang, info in final_per_lang.items()},
            'per_lang_fpr': {lang: info['fpr'] for lang, info in final_per_lang.items()},
            'per_lang_tpr': {lang: info['tpr'] for lang, info in final_per_lang.items()}
        },
        # OOD 结果（Harmbench/MultiJail）
        'out_of_distribution': {
            'overall_accuracy': final_ood_acc,
            'avg_accuracy': ood_avg_acc,
            'per_lang_accuracy': {lang: info['accuracy'] for lang, info in final_per_lang_ood.items()} if has_ood else {}
        },
        # 兼容旧格式
        'test_accuracy': final_val_acc,
        'avg_fpr': avg_fpr,
        'avg_tpr': avg_tpr,
        'ood_accuracy': final_ood_acc
    }
    
    summary_path = os.path.join(output_dir, 'final_model_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(final_summary, f, indent=2)
    print(f"Summary saved to {summary_path}")
    
    # 添加到 results 中
    results['final_summary'] = final_summary
    
    print(f"\n{'='*70}")
    print(f"FINAL MODEL PERFORMANCE SUMMARY")
    print(f"{'='*70}")
    print(f"Layer: {best_layer} | Hidden Dim: {classifier_hidden_dim} | Threshold: {final_threshold:.3f}")
    
    print(f"\n{'='*70}")
    print(f"[IN-DISTRIBUTION] Original Training Dataset")
    print(f"{'='*70}")
    print(f"Test Accuracy: {final_val_acc:.4f} | Avg FPR: {avg_fpr:.4f} | Avg TPR: {avg_tpr:.4f}")
    
    if has_ood:
        print(f"\n{'='*70}")
        print(f"[OUT-OF-DISTRIBUTION] Harmbench/MultiJail")
        print(f"{'='*70}")
        print(f"Overall Accuracy: {final_ood_acc:.4f} | Avg Accuracy: {ood_avg_acc:.4f}")
    
    print(f"{'='*70}")
    
    return results


def analyze_all_layers_with_cache(
    safe_data: List[Dict],
    unsafe_data: List[Dict],
    all_layer_hidden_states: Dict[int, List[torch.Tensor]],
    output_dir: str,
    ood_data: List[Dict] = None,
    ood_layer_hidden_states: Dict[int, List[torch.Tensor]] = None,
    device: str = 'cuda',
    classifier_hidden_dim: int = 256,
    train_epochs: int = 20,
    val_split: float = 0.2,
    alpha: float = 1.0,
    seed: int = 42,
    train_languages: List[str] = None,
    test_languages: List[str] = None,
    use_deep_mlp: bool = False,
    classifier_dropout: float = 0.2
) -> Dict:
    """
    使用预提取的 hidden states 分析所有层（避免重复提取）
    
    Args:
        safe_data: 无害数据列表
        unsafe_data: 有害数据列表
        all_layer_hidden_states: 预提取的所有层 hidden states
        output_dir: 输出目录
        ood_data: OOD数据列表
        ood_layer_hidden_states: OOD数据的 hidden states
        其他参数同 analyze_all_layers
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 合并数据
    all_data = safe_data + unsafe_data
    
    print(f"\n{'='*60}")
    print(f"Cross-Lingual Layer Analysis (Using Cached Hidden States)")
    print(f"{'='*60}")
    print(f"Total samples: {len(all_data)}")
    print(f"  - Safe samples: {len(safe_data)}")
    print(f"  - Unsafe samples: {len(unsafe_data)}")
    
    has_ood = ood_data is not None and len(ood_data) > 0
    if has_ood:
        print(f"  - OOD samples: {len(ood_data)}")
    
    # 获取语言列表
    languages = sorted(list(set(item['language'] for item in all_data)))
    lang_to_idx = {lang: idx for idx, lang in enumerate(languages)}
    num_languages = len(languages)
    print(f"Languages ({num_languages}): {languages}")
    
    num_layers = len(all_layer_hidden_states)
    hidden_dim = all_layer_hidden_states[0][0].shape[0]
    print(f"Number of layers: {num_layers}")
    print(f"Model hidden dimension: {hidden_dim}")
    print(f"Classifier hidden dimension: {classifier_hidden_dim}")
    
    # 准备标签
    safety_labels = [0 if not item['is_unsafe'] else 1 for item in all_data]
    
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
    
    # 存储每层的结果
    layer_results = {}
    
    for layer_idx in tqdm(range(num_layers), desc="Analyzing layers"):
        layer_hidden_states = all_layer_hidden_states[layer_idx]
        
        # 分割数据
        train_hs = [layer_hidden_states[i] for i in train_indices]
        test_hs = [layer_hidden_states[i] for i in val_indices]
        
        train_safety = [safety_labels[i] for i in train_indices]
        test_safety = [safety_labels[i] for i in val_indices]
        
        if use_cross_lingual and inner_val_indices is not None:
            inner_val_hs = [layer_hidden_states[i] for i in inner_val_indices]
            inner_val_safety = [safety_labels[i] for i in inner_val_indices]
        else:
            inner_val_hs = test_hs
            inner_val_safety = test_safety
        
        # 训练 Safety 分类器
        safety_train_dataset = HiddenStateDataset(train_hs, train_safety)
        safety_val_dataset = HiddenStateDataset(inner_val_hs, inner_val_safety)
        
        safety_train_loader = DataLoader(
            safety_train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn
        )
        safety_val_loader = DataLoader(
            safety_val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn
        )
        
        # 不使用类别权重，直接优化accuracy
        pos_weight = None
        
        if use_deep_mlp:
            safety_classifier = DeepMLP(
                input_dim=hidden_dim,
                hidden_dims=[classifier_hidden_dim, classifier_hidden_dim],
                output_dim=1,
                dropout=classifier_dropout,
                use_layer_norm=True,
                use_residual=True,
                activation='gelu'
            )
        else:
            safety_classifier = SimpleMLP(
                input_dim=hidden_dim,
                hidden_dim=classifier_hidden_dim,
                output_dim=1,
                dropout=classifier_dropout
            )
        
        safety_result = train_classifier(
            safety_classifier, safety_train_loader, safety_val_loader,
            num_classes=1, device=device, num_epochs=train_epochs, verbose=False,
            pos_weight=pos_weight, optimize_threshold=True
        )
        
        safety_acc = safety_result['best_val_acc']
        best_threshold = safety_result.get('best_threshold', 0.5)
        
        # OOD 评估
        ood_safety_acc = 0.0
        if has_ood and ood_layer_hidden_states:
            ood_hs = ood_layer_hidden_states[layer_idx]
            ood_safety_labels = [1] * len(ood_hs)
            
            ood_dataset = HiddenStateDataset(ood_hs, ood_safety_labels)
            ood_loader = DataLoader(ood_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
            
            ood_result = evaluate_classifier(
                safety_classifier, ood_loader, num_classes=1, device=device, threshold=best_threshold
            )
            ood_safety_acc = ood_result['accuracy']
        
        # 测试语言评估
        test_safety_acc = 0.0
        if use_cross_lingual:
            test_safety_dataset = HiddenStateDataset(test_hs, test_safety)
            test_safety_loader = DataLoader(test_safety_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
            test_safety_result = evaluate_classifier(
                safety_classifier, test_safety_loader, num_classes=1, device=device, threshold=best_threshold
            )
            test_safety_acc = test_safety_result['accuracy']
        
        # 每个语言的准确率
        per_lang_acc = evaluate_per_language(
            safety_classifier, layer_hidden_states, safety_labels, all_data,
            device=device, threshold=best_threshold
        )
        
        per_lang_ood_acc = {}
        if has_ood and ood_layer_hidden_states:
            ood_hs = ood_layer_hidden_states[layer_idx]
            ood_safety_labels_list = [1] * len(ood_data)
            per_lang_ood_acc = evaluate_per_language(
                safety_classifier, ood_hs, ood_safety_labels_list, ood_data,
                device=device, threshold=best_threshold
            )
        
        # 跨语言相似度
        cross_lingual_sim = compute_cross_lingual_similarity(layer_hidden_states, all_data, device=device)
        
        # 计算分数
        effective_safety_acc = test_safety_acc if use_cross_lingual else safety_acc
        score = effective_safety_acc + alpha * cross_lingual_sim
        
        # 计算平均 FPR
        avg_fpr = np.mean([v['fpr'] for v in per_lang_acc.values() if v['num_safe'] > 0])
        
        layer_results[layer_idx] = {
            'safety_acc': safety_acc,
            'test_safety_acc': test_safety_acc,
            'cross_lingual_sim': cross_lingual_sim,
            'score': score,
            'ood_safety_acc': ood_safety_acc,
            'avg_fpr': avg_fpr,
            'threshold': best_threshold,
            'per_lang_acc': per_lang_acc,
            'per_lang_ood_acc': per_lang_ood_acc,
            'safety_history': safety_result['history'],
            # 保存模型状态，以便后续直接使用最优层的模型
            'model_state_dict': {k: v.cpu().clone() for k, v in safety_classifier.state_dict().items()},
            'pos_weight': pos_weight
        }
        
        print(f"Layer {layer_idx:2d}: TrainAcc={safety_acc:.4f}, TestAcc={test_safety_acc:.4f}, cross_lingual_sim={cross_lingual_sim:.4f}"
            f"FPR={avg_fpr:.4f}, Threshold={best_threshold:.2f}, OOD={ood_safety_acc:.4f}")
    
    # 找到最佳层
    best_layer = max(layer_results.keys(), key=lambda k: layer_results[k]['score'])
    best_score = layer_results[best_layer]['score']
    
    print(f"\n{'='*60}")
    print(f"Best Layer: {best_layer} (Score: {best_score:.4f})")
    print(f"  SafetyAcc: {layer_results[best_layer]['safety_acc']:.4f}")
    print(f"  Threshold: {layer_results[best_layer]['threshold']:.3f}")
    print(f"  Avg FPR: {layer_results[best_layer]['avg_fpr']:.4f}")
    print(f"{'='*60}")
    
    # 保存结果
    results = {
        'config': {
            'alpha': alpha,
            'classifier_hidden_dim': classifier_hidden_dim,
            'train_epochs': train_epochs,
            'val_split': val_split,
            'num_layers': num_layers,
            'hidden_dim': hidden_dim,
            'num_languages': num_languages,
            'languages': languages,
            'train_languages': list(train_languages) if train_languages else None,
            'test_languages': list(test_languages) if test_languages else None,
            'num_safe_samples': len(safe_data),
            'num_unsafe_samples': len(unsafe_data),
            'num_ood_samples': len(ood_data) if has_ood else 0
        },
        'layer_results': {
            str(k): {
                'safety_acc': v['safety_acc'],
                'test_safety_acc': v['test_safety_acc'],
                'cross_lingual_sim': v['cross_lingual_sim'],
                'score': v['score'],
                'ood_safety_acc': v['ood_safety_acc'],
                'avg_fpr': v['avg_fpr'],
                'threshold': v['threshold'],
                'per_lang_acc': {lang: info['accuracy'] for lang, info in v['per_lang_acc'].items()},
                'per_lang_ood_acc': {lang: info['accuracy'] for lang, info in v['per_lang_ood_acc'].items()} if v['per_lang_ood_acc'] else {}
            } for k, v in layer_results.items()
        },
        'best_layer': best_layer,
        'best_score': best_score
    }
    
    with open(os.path.join(output_dir, 'layer_analysis_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # 绘图
    plot_layer_analysis(layer_results, alpha, output_dir, has_ood=has_ood)
    
    # ========== 使用层分析时的最优模型（不重新训练）==========
    print(f"\n{'='*60}")
    print(f"Loading Best Layer {best_layer} Model (No Retraining)")
    print(f"{'='*60}")
    
    best_layer_hs = all_layer_hidden_states[best_layer]
    best_layer_result = layer_results[best_layer]
    
    # 直接使用层分析时保存的模型状态
    final_threshold = best_layer_result['threshold']
    final_val_acc = best_layer_result['safety_acc']
    final_pos_weight = best_layer_result['pos_weight']
    
    # 重建模型并加载状态
    if use_deep_mlp:
        best_classifier = DeepMLP(
            input_dim=hidden_dim,
            hidden_dims=[classifier_hidden_dim, classifier_hidden_dim],
            output_dim=1,
            dropout=classifier_dropout,
            use_layer_norm=True,
            use_residual=True,
            activation='gelu'
        )
        print(f"Using DeepMLP: {hidden_dim} -> {classifier_hidden_dim} -> {classifier_hidden_dim//2} -> 1")
    else:
        best_classifier = SimpleMLP(
            input_dim=hidden_dim,
            hidden_dim=classifier_hidden_dim,
            output_dim=1,
            dropout=classifier_dropout
        )
        print(f"Using SimpleMLP: {hidden_dim} -> {classifier_hidden_dim} -> 1")
    
    # 加载层分析时保存的模型权重
    best_classifier.load_state_dict(best_layer_result['model_state_dict'])
    best_classifier = best_classifier.to(device)
    best_classifier.eval()
    
    print(f"Loaded model from layer analysis (no retraining)")
    print(f"Layer: {best_layer}, Threshold: {final_threshold:.3f}")
    print(f"Training Val Acc: {final_val_acc:.4f}, OOD Acc: {best_layer_result['ood_safety_acc']:.4f}")
    
    # 保存模型 - 阈值格式修复：乘以100后取整
    threshold_int = int(round(final_threshold * 100))
    threshold_str = f"{threshold_int:03d}"  # 0.75 -> 075, 0.50 -> 050
    model_save_path = os.path.join(
        output_dir,
        f'classifier_layer{best_layer}_hidden{classifier_hidden_dim}_threshold{threshold_str}.pt'
    )
    
    # ========== 用保存的模型和阈值重新测试 ==========
    print(f"\n{'='*60}")
    print(f"Final Verification with Saved Model")
    print(f"{'='*60}")
    
    best_classifier.eval()
    
    # 测试集评估
    test_hs = [best_layer_hs[i] for i in val_indices]
    test_safety = [safety_labels[i] for i in val_indices]
    test_dataset = HiddenStateDataset(test_hs, test_safety)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    
    test_result = evaluate_classifier(
        best_classifier, test_loader, num_classes=1, device=device, threshold=final_threshold
    )
    
    # 每个语言的详细评估
    final_per_lang = evaluate_per_language(
        best_classifier, best_layer_hs, safety_labels, all_data,
        device=device, threshold=final_threshold
    )
    
    # OOD 评估
    final_ood_acc = 0.0
    final_ood_per_lang = {}
    if has_ood and ood_layer_hidden_states:
        ood_hs = ood_layer_hidden_states[best_layer]
        ood_labels = [1] * len(ood_data)
        ood_dataset = HiddenStateDataset(ood_hs, ood_labels)
        ood_loader = DataLoader(ood_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
        
        ood_result = evaluate_classifier(
            best_classifier, ood_loader, num_classes=1, device=device, threshold=final_threshold
        )
        final_ood_acc = ood_result['accuracy']
        
        final_ood_per_lang = evaluate_per_language(
            best_classifier, ood_hs, ood_labels, ood_data,
            device=device, threshold=final_threshold
        )
    
    # 计算汇总指标
    avg_fpr = np.mean([v['fpr'] for v in final_per_lang.values() if v['num_safe'] > 0])
    avg_tpr = np.mean([v['tpr'] for v in final_per_lang.values() if v['num_unsafe'] > 0])
    
    # OOD 汇总指标
    ood_avg_acc = 0.0
    if has_ood and final_ood_per_lang:
        ood_avg_acc = np.mean([v['accuracy'] for v in final_ood_per_lang.values()])
    
    print(f"\n{'='*70}")
    print(f"FINAL MODEL PERFORMANCE SUMMARY")
    print(f"{'='*70}")
    print(f"Layer: {best_layer} | Hidden Dim: {classifier_hidden_dim} | Threshold: {final_threshold:.3f}")
    
    # ===== 原始训练数据集结果 =====
    print(f"\n{'='*70}")
    print(f"[IN-DISTRIBUTION] Original Training Dataset Results")
    print(f"{'='*70}")
    print(f"Overall Test Accuracy: {test_result['accuracy']:.4f}")
    print(f"Average FPR (Safe→Unsafe): {avg_fpr:.4f}")
    print(f"Average TPR (Unsafe detected): {avg_tpr:.4f}")
    print(f"\n{'Language':<10} {'Accuracy':>10} {'FPR':>10} {'TPR':>10} {'Safe':>8} {'Unsafe':>8}")
    print("-" * 60)
    for lang in sorted(final_per_lang.keys()):
        info = final_per_lang[lang]
        print(f"{lang:<10} {info['accuracy']:>10.4f} {info['fpr']:>10.4f} {info['tpr']:>10.4f} {info['num_safe']:>8} {info['num_unsafe']:>8}")
    
    # ===== Harmbench/OOD 结果 =====
    if has_ood and final_ood_per_lang:
        print(f"\n{'='*70}")
        print(f"[OUT-OF-DISTRIBUTION] Harmbench/MultiJail Results")
        print(f"{'='*70}")
        print(f"Overall OOD Accuracy: {final_ood_acc:.4f}")
        print(f"Average OOD Accuracy: {ood_avg_acc:.4f}")
        print(f"\n{'Language':<10} {'Accuracy':>10} {'Total':>10}")
        print("-" * 35)
        for lang in sorted(final_ood_per_lang.keys()):
            info = final_ood_per_lang[lang]
            print(f"{lang:<10} {info['accuracy']:>10.4f} {info['total']:>10}")
    
    print(f"\n{'='*70}")
    
    # 保存详细结果摘要（分开汇报 In-Distribution 和 OOD）
    final_summary = {
        'model_path': model_save_path,
        'layer_idx': best_layer,
        'classifier_hidden_dim': classifier_hidden_dim,
        'threshold': final_threshold,
        'pos_weight': final_pos_weight,
        # In-Distribution 结果（原始训练数据集）
        'in_distribution': {
            'test_accuracy': test_result['accuracy'],
            'avg_fpr': avg_fpr,
            'avg_tpr': avg_tpr,
            'per_lang_accuracy': {lang: info['accuracy'] for lang, info in final_per_lang.items()},
            'per_lang_fpr': {lang: info['fpr'] for lang, info in final_per_lang.items()},
            'per_lang_tpr': {lang: info['tpr'] for lang, info in final_per_lang.items()}
        },
        # OOD 结果（Harmbench/MultiJail）
        'out_of_distribution': {
            'overall_accuracy': final_ood_acc,
            'avg_accuracy': ood_avg_acc if has_ood else 0.0,
            'per_lang_accuracy': {lang: info['accuracy'] for lang, info in final_ood_per_lang.items()} if final_ood_per_lang else {}
        },
        # 兼容旧格式
        'test_accuracy': test_result['accuracy'],
        'avg_fpr': avg_fpr,
        'avg_tpr': avg_tpr,
        'ood_accuracy': final_ood_acc
    }
    
    # 保存模型
    torch.save({
        'model_state_dict': best_classifier.state_dict(),
        'layer_idx': best_layer,
        'hidden_dim': hidden_dim,
        'classifier_hidden_dim': classifier_hidden_dim,
        'val_acc': final_val_acc,
        'test_acc': test_result['accuracy'],
        'threshold': final_threshold,
        'pos_weight': final_pos_weight,
        'avg_fpr': avg_fpr,
        'avg_tpr': avg_tpr,
        'ood_accuracy': final_ood_acc,
        'train_languages': list(train_languages) if train_languages else None,
        'test_languages': list(test_languages) if test_languages else None
    }, model_save_path)
    print(f"\nModel saved to {model_save_path}")
    
    # 保存摘要到 JSON
    summary_path = os.path.join(output_dir, 'final_model_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(final_summary, f, indent=2)
    print(f"Summary saved to {summary_path}")
    
    # 添加到 results 中
    results['final_summary'] = final_summary
    
    return results


def train_on_single_layer_with_cache(
    safe_data: List[Dict],
    unsafe_data: List[Dict],
    all_layer_hidden_states: Dict[int, List[torch.Tensor]],
    target_layer: int,
    output_dir: str,
    ood_data: List[Dict] = None,
    ood_layer_hidden_states: Dict[int, List[torch.Tensor]] = None,
    device: str = 'cuda',
    classifier_hidden_dim: int = 256,
    train_epochs: int = 20,
    val_split: float = 0.2,
    seed: int = 42,
    train_languages: List[str] = None,
    test_languages: List[str] = None,
    use_deep_mlp: bool = False,
    classifier_dropout: float = 0.2
) -> Dict:
    """
    在指定层上直接训练分类器（跳过层选择）
    
    Args:
        safe_data: 无害数据列表
        unsafe_data: 有害数据列表
        all_layer_hidden_states: 预提取的所有层 hidden states
        target_layer: 目标层索引
        output_dir: 输出目录
        ood_data: OOD数据列表
        ood_layer_hidden_states: OOD数据的 hidden states
        其他参数同 analyze_all_layers_with_cache
    
    Returns:
        训练结果字典
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 合并数据
    all_data = safe_data + unsafe_data
    
    print(f"\n{'='*60}")
    print(f"Training on Single Layer (Skip Layer Selection)")
    print(f"{'='*60}")
    print(f"Target Layer: {target_layer}")
    print(f"Classifier Hidden Dim: {classifier_hidden_dim}")
    print(f"Total samples: {len(all_data)}")
    print(f"  - Safe samples: {len(safe_data)}")
    print(f"  - Unsafe samples: {len(unsafe_data)}")
    
    has_ood = ood_data is not None and len(ood_data) > 0
    if has_ood:
        print(f"  - OOD samples: {len(ood_data)}")
    
    # 获取语言列表
    languages = sorted(list(set(item['language'] for item in all_data)))
    num_languages = len(languages)
    print(f"Languages ({num_languages}): {languages}")
    
    num_layers = len(all_layer_hidden_states)
    hidden_dim = all_layer_hidden_states[0][0].shape[0]
    print(f"Number of layers: {num_layers}")
    print(f"Model hidden dimension: {hidden_dim}")
    
    # 验证目标层是否有效
    if target_layer < 0 or target_layer >= num_layers:
        raise ValueError(f"Invalid target_layer {target_layer}, must be in [0, {num_layers-1}]")
    
    # 准备标签
    safety_labels = [0 if not item['is_unsafe'] else 1 for item in all_data]
    
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
    
    # 获取目标层的 hidden states
    layer_hidden_states = all_layer_hidden_states[target_layer]
    
    # 分割数据
    train_hs = [layer_hidden_states[i] for i in train_indices]
    test_hs = [layer_hidden_states[i] for i in val_indices]
    
    train_safety = [safety_labels[i] for i in train_indices]
    test_safety = [safety_labels[i] for i in val_indices]
    
    if use_cross_lingual and inner_val_indices is not None:
        inner_val_hs = [layer_hidden_states[i] for i in inner_val_indices]
        inner_val_safety = [safety_labels[i] for i in inner_val_indices]
    else:
        inner_val_hs = test_hs
        inner_val_safety = test_safety
    
    # 训练 Safety 分类器
    print(f"\n{'='*60}")
    print(f"Training Safety Classifier on Layer {target_layer}")
    print(f"{'='*60}")
    
    safety_train_dataset = HiddenStateDataset(train_hs, train_safety)
    safety_val_dataset = HiddenStateDataset(inner_val_hs, inner_val_safety)
    
    safety_train_loader = DataLoader(
        safety_train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn
    )
    safety_val_loader = DataLoader(
        safety_val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn
    )
    
    pos_weight = None
    
    if use_deep_mlp:
        classifier = DeepMLP(
            input_dim=hidden_dim,
            hidden_dims=[classifier_hidden_dim, classifier_hidden_dim],
            output_dim=1,
            dropout=classifier_dropout,
            use_layer_norm=True,
            use_residual=True,
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
    
    train_result = train_classifier(
        classifier, safety_train_loader, safety_val_loader,
        num_classes=1, device=device, num_epochs=train_epochs, verbose=True,
        pos_weight=pos_weight, optimize_threshold=True
    )
    
    safety_acc = train_result['best_val_acc']
    best_threshold = train_result.get('best_threshold', 0.5)
    
    print(f"\nTraining completed!")
    print(f"Best Val Accuracy: {safety_acc:.4f}")
    print(f"Optimal Threshold: {best_threshold:.3f}")
    
    # OOD 评估
    ood_safety_acc = 0.0
    ood_per_lang = {}
    if has_ood and ood_layer_hidden_states:
        ood_hs = ood_layer_hidden_states[target_layer]
        ood_safety_labels = [1] * len(ood_hs)
        
        ood_dataset = HiddenStateDataset(ood_hs, ood_safety_labels)
        ood_loader = DataLoader(ood_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
        
        ood_result = evaluate_classifier(
            classifier, ood_loader, num_classes=1, device=device, threshold=best_threshold
        )
        ood_safety_acc = ood_result['accuracy']
        
        ood_per_lang = evaluate_per_language(
            classifier, ood_hs, ood_safety_labels, ood_data,
            device=device, threshold=best_threshold
        )
    
    # 测试语言评估
    test_safety_acc = 0.0
    if use_cross_lingual:
        test_safety_dataset = HiddenStateDataset(test_hs, test_safety)
        test_safety_loader = DataLoader(test_safety_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
        test_safety_result = evaluate_classifier(
            classifier, test_safety_loader, num_classes=1, device=device, threshold=best_threshold
        )
        test_safety_acc = test_safety_result['accuracy']
    
    # 每个语言的准确率
    per_lang_acc = evaluate_per_language(
        classifier, layer_hidden_states, safety_labels, all_data,
        device=device, threshold=best_threshold
    )
    
    # 计算汇总指标
    avg_fpr = np.mean([v['fpr'] for v in per_lang_acc.values() if v['num_safe'] > 0])
    avg_tpr = np.mean([v['tpr'] for v in per_lang_acc.values() if v['num_unsafe'] > 0])
    
    # 保存模型
    threshold_int = int(round(best_threshold * 100))
    threshold_str = f"{threshold_int:03d}"
    model_save_path = os.path.join(
        output_dir,
        f'classifier_layer{target_layer}_hidden{classifier_hidden_dim}_threshold{threshold_str}.pt'
    )
    
    torch.save({
        'model_state_dict': classifier.state_dict(),
        'layer_idx': target_layer,
        'hidden_dim': hidden_dim,
        'classifier_hidden_dim': classifier_hidden_dim,
        'val_acc': safety_acc,
        'test_acc': test_safety_acc if use_cross_lingual else safety_acc,
        'threshold': best_threshold,
        'pos_weight': pos_weight,
        'avg_fpr': avg_fpr,
        'avg_tpr': avg_tpr,
        'ood_accuracy': ood_safety_acc,
        'train_languages': list(train_languages) if train_languages else None,
        'test_languages': list(test_languages) if test_languages else None
    }, model_save_path)
    print(f"\nModel saved to {model_save_path}")
    
    # OOD 汇总指标
    ood_avg_acc = 0.0
    if has_ood and ood_per_lang:
        ood_avg_acc = np.mean([v['accuracy'] for v in ood_per_lang.values()])
    
    # 保存 final_model_summary.json
    final_summary = {
        'model_path': model_save_path,
        'layer_idx': target_layer,
        'classifier_hidden_dim': classifier_hidden_dim,
        'threshold': best_threshold,
        'pos_weight': pos_weight,
        'mode': 'single_layer_training',
        # In-Distribution 结果
        'in_distribution': {
            'test_accuracy': test_safety_acc if use_cross_lingual else safety_acc,
            'avg_fpr': avg_fpr,
            'avg_tpr': avg_tpr,
            'per_lang_accuracy': {lang: info['accuracy'] for lang, info in per_lang_acc.items()},
            'per_lang_fpr': {lang: info['fpr'] for lang, info in per_lang_acc.items()},
            'per_lang_tpr': {lang: info['tpr'] for lang, info in per_lang_acc.items()}
        },
        # OOD 结果
        'out_of_distribution': {
            'overall_accuracy': ood_safety_acc,
            'avg_accuracy': ood_avg_acc,
            'per_lang_accuracy': {lang: info['accuracy'] for lang, info in ood_per_lang.items()} if ood_per_lang else {}
        },
        # 兼容旧格式
        'test_accuracy': test_safety_acc if use_cross_lingual else safety_acc,
        'avg_fpr': avg_fpr,
        'avg_tpr': avg_tpr,
        'ood_accuracy': ood_safety_acc
    }
    
    summary_path = os.path.join(output_dir, 'final_model_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(final_summary, f, indent=2)
    print(f"Summary saved to {summary_path}")
    
    # 打印最终结果
    print(f"\n{'='*70}")
    print(f"FINAL MODEL PERFORMANCE SUMMARY")
    print(f"{'='*70}")
    print(f"Layer: {target_layer} | Hidden Dim: {classifier_hidden_dim} | Threshold: {best_threshold:.3f}")
    
    print(f"\n{'='*70}")
    print(f"[IN-DISTRIBUTION] Original Training Dataset Results")
    print(f"{'='*70}")
    print(f"Overall Test Accuracy: {test_safety_acc if use_cross_lingual else safety_acc:.4f}")
    print(f"Average FPR: {avg_fpr:.4f} | Average TPR: {avg_tpr:.4f}")
    print(f"\n{'Language':<10} {'Accuracy':>10} {'FPR':>10} {'TPR':>10} {'Safe':>8} {'Unsafe':>8}")
    print("-" * 60)
    for lang in sorted(per_lang_acc.keys()):
        info = per_lang_acc[lang]
        print(f"{lang:<10} {info['accuracy']:>10.4f} {info['fpr']:>10.4f} {info['tpr']:>10.4f} {info['num_safe']:>8} {info['num_unsafe']:>8}")
    
    if has_ood and ood_per_lang:
        print(f"\n{'='*70}")
        print(f"[OUT-OF-DISTRIBUTION] Harmbench/MultiJail Results")
        print(f"{'='*70}")
        print(f"Overall OOD Accuracy: {ood_safety_acc:.4f}")
        print(f"Average OOD Accuracy: {ood_avg_acc:.4f}")
        print(f"\n{'Language':<10} {'Accuracy':>10} {'Total':>10}")
        print("-" * 35)
        for lang in sorted(ood_per_lang.keys()):
            info = ood_per_lang[lang]
            print(f"{lang:<10} {info['accuracy']:>10.4f} {info['total']:>10}")
    
    print(f"\n{'='*70}")
    
    return {
        'layer_idx': target_layer,
        'classifier_hidden_dim': classifier_hidden_dim,
        'threshold': best_threshold,
        'test_accuracy': test_safety_acc if use_cross_lingual else safety_acc,
        'avg_fpr': avg_fpr,
        'avg_tpr': avg_tpr,
        'ood_accuracy': ood_safety_acc,
        'model_path': model_save_path,
        'final_summary': final_summary
    }


def plot_layer_analysis(layer_results: Dict, alpha: float, output_dir: str, has_ood: bool = False):
    """绘制层分析结果"""
    layers = sorted(layer_results.keys())
    safety_accs = [layer_results[l]['safety_acc'] for l in layers]
    cross_lingual_sims = [layer_results[l]['cross_lingual_sim'] for l in layers]
    scores = [layer_results[l]['score'] for l in layers]
    
    if has_ood:
        ood_accs = [layer_results[l]['ood_safety_acc'] for l in layers]
    
    # 根据是否有OOD数据决定绘图布局
    num_cols = 4 if has_ood else 3
    fig, axes = plt.subplots(1, num_cols, figsize=(5*num_cols, 5))
    
    # 1. Safety Accuracy
    ax = axes[0]
    ax.plot(layers, safety_accs, 'b-o', linewidth=2, markersize=6)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Safety Accuracy', fontsize=12)
    ax.set_title('Safety Classification Accuracy per Layer', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    # 2. Cross-Lingual Similarity
    ax = axes[1]
    ax.plot(layers, cross_lingual_sims, 'g-o', linewidth=2, markersize=6)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Cross-Lingual Similarity', fontsize=12)
    ax.set_title('Cross-Lingual Hidden State Similarity per Layer', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    # 3. OOD Safety Accuracy (如果有)
    if has_ood:
        ax = axes[2]
        ax.plot(layers, ood_accs, 'm-o', linewidth=2, markersize=6)
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('OOD Safety Accuracy', fontsize=12)
        ax.set_title('OOD (MultiJail) Safety Accuracy per Layer', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
    
    # 4. Cross-Lingual Score
    ax = axes[num_cols - 1]
    ax.plot(layers, scores, 'c-o', linewidth=2, markersize=6)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel(f'Score (SafetyAcc + {alpha}×CrossLingSim)', fontsize=12)
    ax.set_title('Cross-Lingual Alignment Score per Layer', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # 标记最佳层
    best_layer = max(layer_results.keys(), key=lambda k: layer_results[k]['score'])
    best_score = layer_results[best_layer]['score']
    ax.scatter([best_layer], [best_score], color='gold', s=200, zorder=5, 
               edgecolor='black', linewidth=2, marker='*', label=f'Best Layer: {best_layer}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'layer_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制组合图
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(layers, safety_accs, 'b-o', linewidth=2, markersize=6, label='Safety Acc')
    ax.plot(layers, cross_lingual_sims, 'g-o', linewidth=2, markersize=6, label='Cross-Lingual Sim')
    if has_ood:
        ax.plot(layers, ood_accs, 'm-o', linewidth=2, markersize=6, label='OOD Safety Acc')
    ax.plot(layers, scores, 'c-o', linewidth=2, markersize=6, label=f'Score (α={alpha})')
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.scatter([best_layer], [best_score], color='gold', s=200, zorder=5,
               edgecolor='black', linewidth=2, marker='*', label=f'Best: Layer {best_layer}')
    
    ax.set_xlabel('Layer', fontsize=14)
    ax.set_ylabel('Value', fontsize=14)
    ax.set_title('Cross-Lingual Layer Analysis', fontsize=16)
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'layer_analysis_combined.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlots saved to {output_dir}")


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description="Cross-Lingual Layer Analysis")
    
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
    parser.add_argument("--additional_unsafe_paths", type=str, nargs='*', default=[],
                        help="Additional unsafe data paths for training (e.g. harmbench_training)")
    parser.add_argument("--languages", type=str, nargs="+", 
                        default=['en', 'zh', 'ar', 'it', 'ko', 'vi'],
                        help="All languages to use (for data extraction)")
    parser.add_argument("--train_languages", type=str, nargs="+", default=None,
                        help="Languages to use for training (default: all languages)")
    parser.add_argument("--test_languages", type=str, nargs="+", default=None,
                        help="Languages to use for testing (default: all languages)")
    parser.add_argument("--max_safe_per_lang", type=int, default=200,
                        help="Max safe samples per language")
    parser.add_argument("--max_unsafe", type=int, default=200,
                        help="Max unsafe samples")
    parser.add_argument("--max_ood", type=int, default=200,
                        help="Max OOD samples")
    
    # 分类器参数
    parser.add_argument("--classifier_hidden_dim", type=int, default=256,
                        help="Hidden dimension for classifier (first layer)")
    parser.add_argument("--use_deep_mlp", action="store_true", default=True,
                        help="Use DeepMLP (2 hidden layers) for safety classifier")
    parser.add_argument("--use_simple_mlp", action="store_true",
                        help="Use SimpleMLP (1 hidden layer) instead of DeepMLP")
    parser.add_argument("--classifier_dropout", type=float, default=0.2,
                        help="Dropout rate for classifier")
    parser.add_argument("--train_epochs", type=int, default=20,
                        help="Training epochs per layer")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Validation split ratio")
    
    # 分数计算参数
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Weight for language accuracy in score")
    
    # 直接指定层训练（跳过层选择）
    parser.add_argument("--target_layer", type=int, default=None,
                        help="Target layer index to train on (skip layer selection if specified)")
    parser.add_argument("--skip_layer_selection", action="store_true",
                        help="Skip layer selection and train directly on target_layer")
    
    # 其他参数
    parser.add_argument("--output_dir", type=str, default="./output/layer_analysis",
                        help="Output directory")
    parser.add_argument("--hidden_states_cache", type=str, default=None,
                        help="Path to cached hidden states file (will load if exists, save if not)")
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
    output_dir = os.path.join(args.output_dir, f"analysis_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存配置
    config = vars(args)
    config['timestamp'] = timestamp
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # 检查是否有缓存的 hidden states
    cache_path = args.hidden_states_cache
    cached_data = None
    
    if cache_path and os.path.exists(cache_path):
        print(f"\n>>> Loading cached hidden states from {cache_path}")
        cached_data = torch.load(cache_path, weights_only=False)
        
        safe_data = cached_data['safe_data']
        unsafe_data = cached_data['unsafe_data']
        ood_data = cached_data.get('ood_data')
        all_layer_hidden_states = cached_data['all_layer_hidden_states']
        ood_layer_hidden_states = cached_data.get('ood_layer_hidden_states')
        
        print(f"Loaded {len(safe_data)} safe, {len(unsafe_data)} unsafe samples")
        if ood_data:
            print(f"Loaded {len(ood_data)} OOD samples")
        print(f"Hidden states: {len(all_layer_hidden_states)} layers")
        
        # 不需要加载模型
        model = None
        tokenizer = None
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
        
        # 加载额外的有害数据（如 harmbench_training）
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
        if args.ood_data_path and os.path.exists(args.ood_data_path):
            max_unsafe_idx = max(item['idx'] for item in unsafe_data) if unsafe_data else 0
            ood_data = load_harmbench_data(
                args.ood_data_path,
                max_samples=args.max_ood,
                idx_offset=max_unsafe_idx + 10000
            )
            
            if args.languages:
                ood_data = [item for item in ood_data if item['language'] in args.languages]
            
            print(f"Loaded {len(ood_data)} OOD (harmbench) samples")
        
        # 提取 hidden states
        all_data = safe_data + unsafe_data
        all_texts = [item['prompt'] for item in all_data]
        
        print(f"\nExtracting hidden states from all layers...")
        all_layer_hidden_states = extract_hidden_states_all_layers(
            model, tokenizer, all_texts,
            device=args.device, batch_size=args.batch_size
        )
        
        # 提取 OOD hidden states
        ood_layer_hidden_states = None
        if ood_data:
            ood_texts = [item['prompt'] for item in ood_data]
            print(f"Extracting OOD hidden states...")
            ood_layer_hidden_states = extract_hidden_states_all_layers(
                model, tokenizer, ood_texts,
                device=args.device, batch_size=args.batch_size
            )
        
        # 保存缓存
        if cache_path:
            print(f"\n>>> Saving hidden states cache to {cache_path}")
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
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
        del tokenizer
        torch.cuda.empty_cache()
    
    # 决定是否使用 DeepMLP
    use_deep = args.use_deep_mlp and not args.use_simple_mlp
    print(f"\nClassifier type: {'DeepMLP (2 hidden layers)' if use_deep else 'SimpleMLP (1 hidden layer)'}")
    print(f"Classifier hidden dim: {args.classifier_hidden_dim}")
    print(f"Dropout: {args.classifier_dropout}")
    
    # 判断是否跳过层选择
    if args.skip_layer_selection or args.target_layer is not None:
        # 直接在指定层训练
        if args.target_layer is None:
            raise ValueError("--target_layer must be specified when using --skip_layer_selection")
        
        print(f"\n>>> SKIP LAYER SELECTION MODE <<<")
        print(f">>> Training directly on layer {args.target_layer}")
        
        results = train_on_single_layer_with_cache(
            safe_data=safe_data,
            unsafe_data=unsafe_data,
            all_layer_hidden_states=all_layer_hidden_states,
            target_layer=args.target_layer,
            output_dir=output_dir,
            ood_data=ood_data,
            ood_layer_hidden_states=ood_layer_hidden_states,
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
        
        print(f"\nResults saved to {output_dir}")
        print(f"\nTrained on layer: {args.target_layer}")
    else:
        # 运行完整的层分析（使用预提取的 hidden states）
        results = analyze_all_layers_with_cache(
            safe_data=safe_data,
            unsafe_data=unsafe_data,
            all_layer_hidden_states=all_layer_hidden_states,
            output_dir=output_dir,
            ood_data=ood_data,
            ood_layer_hidden_states=ood_layer_hidden_states,
            device=args.device,
            classifier_hidden_dim=args.classifier_hidden_dim,
            train_epochs=args.train_epochs,
            val_split=args.val_split,
            alpha=args.alpha,
            seed=args.seed,
            train_languages=args.train_languages,
            test_languages=args.test_languages,
            use_deep_mlp=use_deep,
            classifier_dropout=args.classifier_dropout
        )
        
        print(f"\nResults saved to {output_dir}")
        print(f"\nBest layer for cross-lingual alignment: Layer {results['best_layer']}")


if __name__ == "__main__":
    main()

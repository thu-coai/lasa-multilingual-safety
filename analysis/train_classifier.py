"""
训练带分类头的Transform MLP模型

目标：
1. 跨语言对齐：不同语言同一问题的编码尽可能接近
2. 安全性区分：有害问题和无害问题的距离尽可能远
3. 分类任务：直接预测hidden state是harmful还是benign
"""

import os
import argparse
import json
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from model import (
    TransformMLPWithClassifier, 
    CombinedLoss, 
    ClassificationOnlyLoss,
    ContrastiveLoss
)
from dataset import (
    MultilingualHiddenStateDataset,
    load_dataset,
    collate_fn
)


def set_seed(seed: int):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_classification_metrics(
    model: TransformMLPWithClassifier,
    dataloader: DataLoader,
    device: str
) -> dict:
    """
    计算分类评估指标
    
    指标：
    1. Accuracy, Precision, Recall, F1
    2. AUC-ROC
    """
    model.eval()
    
    all_logits = []
    all_labels = []
    all_embeddings = []
    all_group_ids = []
    
    with torch.no_grad():
        for batch in dataloader:
            hidden_states = batch['hidden_states'].to(device)
            safety_labels = batch['safety_labels']
            
            outputs = model(hidden_states, return_embeddings=True)
            
            all_logits.append(outputs['logits'].cpu())
            all_embeddings.append(outputs['embeddings'].cpu())
            all_labels.extend(safety_labels.tolist())
            all_group_ids.extend(batch['group_ids'].tolist())
    
    all_logits = torch.cat(all_logits, dim=0)
    all_embeddings = torch.cat(all_embeddings, dim=0)
    
    # 计算预测概率和预测标签
    probs = torch.sigmoid(all_logits).squeeze(-1).numpy()
    preds = (probs > 0.5).astype(int)
    labels = np.array(all_labels)
    
    # 分类指标
    metrics = {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, zero_division=0),
        'recall': recall_score(labels, preds, zero_division=0),
        'f1': f1_score(labels, preds, zero_division=0),
    }
    
    # AUC-ROC (如果有两个类别)
    if len(np.unique(labels)) > 1:
        metrics['auc_roc'] = roc_auc_score(labels, probs)
    else:
        metrics['auc_roc'] = 0.0
    
    # 计算embedding相关指标（对齐和分离度）
    all_embeddings_norm = nn.functional.normalize(all_embeddings, p=2, dim=1)
    
    # 1. 计算同组内平均相似度
    group_to_indices = {}
    for i, gid in enumerate(all_group_ids):
        if gid not in group_to_indices:
            group_to_indices[gid] = []
        group_to_indices[gid].append(i)
    
    intra_group_sims = []
    for gid, indices in group_to_indices.items():
        if len(indices) < 2:
            continue
        
        group_embeddings = all_embeddings_norm[indices]
        sim_matrix = torch.matmul(group_embeddings, group_embeddings.T)
        
        mask = ~torch.eye(len(indices), dtype=torch.bool)
        intra_group_sims.extend(sim_matrix[mask].tolist())
    
    metrics['intra_group_similarity'] = np.mean(intra_group_sims) if intra_group_sims else 0.0
    
    # 2. 计算safe-unsafe之间的平均距离
    safe_indices = [i for i, l in enumerate(all_labels) if l == 0]
    unsafe_indices = [i for i, l in enumerate(all_labels) if l == 1]
    
    if safe_indices and unsafe_indices:
        safe_embeddings = all_embeddings_norm[safe_indices]
        unsafe_embeddings = all_embeddings_norm[unsafe_indices]
        
        cross_sim = torch.matmul(safe_embeddings, unsafe_embeddings.T)
        metrics['safe_unsafe_similarity'] = cross_sim.mean().item()
    else:
        metrics['safe_unsafe_similarity'] = 0.0
    
    return metrics


def train_epoch(
    model: TransformMLPWithClassifier,
    dataloader: DataLoader,
    criterion,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int,
    use_combined_loss: bool = True
) -> dict:
    """训练一个epoch"""
    model.train()
    
    total_loss = 0.0
    total_classify_loss = 0.0
    total_align_loss = 0.0
    total_safety_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch in pbar:
        hidden_states = batch['hidden_states'].to(device)
        group_ids = batch['group_ids'].to(device)
        safety_labels = batch['safety_labels'].to(device)
        
        optimizer.zero_grad()
        
        # Forward
        outputs = model(hidden_states, return_embeddings=use_combined_loss)
        
        # Compute loss
        if use_combined_loss:
            losses = criterion(
                outputs['logits'], 
                outputs['embeddings'], 
                group_ids, 
                safety_labels
            )
        else:
            losses = criterion(outputs['logits'], safety_labels)
        
        # Backward
        losses['total_loss'].backward()
        
        # Gradient clipping (只对需要梯度的参数)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if trainable_params:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        
        optimizer.step()
        
        total_loss += losses['total_loss'].item()
        total_classify_loss += losses['classify_loss'].item()
        
        if use_combined_loss:
            total_align_loss += losses.get('alignment_loss', torch.tensor(0.0)).item()
            total_safety_loss += losses.get('safety_loss', torch.tensor(0.0)).item()
        
        num_batches += 1
        
        postfix = {
            'loss': f"{losses['total_loss'].item():.4f}",
            'cls': f"{losses['classify_loss'].item():.4f}"
        }
        if use_combined_loss:
            postfix['align'] = f"{losses.get('alignment_loss', 0):.4f}"
            postfix['safety'] = f"{losses.get('safety_loss', 0):.4f}"
        
        pbar.set_postfix(postfix)
    
    result = {
        'loss': total_loss / num_batches,
        'classify_loss': total_classify_loss / num_batches
    }
    
    if use_combined_loss:
        result['alignment_loss'] = total_align_loss / num_batches
        result['safety_loss'] = total_safety_loss / num_batches
    
    return result


def train(
    model: TransformMLPWithClassifier,
    train_dataset: MultilingualHiddenStateDataset,
    val_dataset: MultilingualHiddenStateDataset,
    output_dir: str,
    num_epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    temperature: float = 0.07,
    margin: float = 1.0,
    lambda_align: float = 1.0,
    lambda_safety: float = 1.0,
    lambda_classify: float = 1.0,
    device: str = 'cuda',
    loss_type: str = 'combined',  # 'combined' or 'classify_only'
    save_every: int = 10
):
    """
    训练模型
    
    Args:
        model: 带分类头的MLP模型
        train_dataset: 训练数据集
        val_dataset: 验证数据集
        output_dir: 输出目录
        num_epochs: 训练轮数
        batch_size: 批大小
        learning_rate: 学习率
        weight_decay: 权重衰减
        temperature: 对比学习温度
        margin: margin损失的margin值
        lambda_align: 对齐损失权重
        lambda_safety: 安全性损失权重（对比学习）
        lambda_classify: 分类损失权重
        device: 设备
        loss_type: 损失类型 ('combined' or 'classify_only')
        save_every: 每多少epoch保存一次
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # 损失函数
    use_combined_loss = (loss_type == 'combined')
    
    if use_combined_loss:
        criterion = CombinedLoss(
            temperature=temperature,
            margin=margin,
            lambda_align=lambda_align,
            lambda_safety=lambda_safety,
            lambda_classify=lambda_classify
        )
    else:
        criterion = ClassificationOnlyLoss()
    
    # 优化器：只优化需要梯度的参数
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        trainable_params,
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=learning_rate * 0.01
    )
    
    # 训练历史
    history = {
        'train_loss': [],
        'train_classify_loss': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'val_auc_roc': [],
        'val_intra_group_sim': [],
        'val_safe_unsafe_sim': []
    }
    
    if use_combined_loss:
        history['train_align_loss'] = []
        history['train_safety_loss'] = []
    
    best_f1 = 0.0
    best_auc = 0.0
    
    print(f"\nStarting training...")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Loss type: {loss_type}")
    if use_combined_loss:
        print(f"  Lambda align: {lambda_align}")
        print(f"  Lambda safety: {lambda_safety}")
        print(f"  Lambda classify: {lambda_classify}")
    
    for epoch in range(1, num_epochs + 1):
        # 训练
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, 
            use_combined_loss=use_combined_loss
        )
        
        # 验证
        val_metrics = compute_classification_metrics(model, val_loader, device)
        
        # 更新学习率
        scheduler.step()
        
        # 记录历史
        history['train_loss'].append(train_metrics['loss'])
        history['train_classify_loss'].append(train_metrics['classify_loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_auc_roc'].append(val_metrics['auc_roc'])
        history['val_intra_group_sim'].append(val_metrics['intra_group_similarity'])
        history['val_safe_unsafe_sim'].append(val_metrics['safe_unsafe_similarity'])
        
        if use_combined_loss:
            history['train_align_loss'].append(train_metrics['alignment_loss'])
            history['train_safety_loss'].append(train_metrics['safety_loss'])
        
        print(f"\nEpoch {epoch}/{num_epochs}:")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
              f"Classify: {train_metrics['classify_loss']:.4f}")
        if use_combined_loss:
            print(f"         Align: {train_metrics['alignment_loss']:.4f}, "
                  f"Safety: {train_metrics['safety_loss']:.4f}")
        print(f"  Val   - Acc: {val_metrics['accuracy']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}, "
              f"AUC: {val_metrics['auc_roc']:.4f}")
        print(f"         IntraGroup: {val_metrics['intra_group_similarity']:.4f}, "
              f"Safe-Unsafe: {val_metrics['safe_unsafe_similarity']:.4f}")
        
        # 保存最佳模型（基于F1或AUC）
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'history': history
            }, os.path.join(output_dir, 'best_model_f1.pt'))
            print(f"  -> Saved best model (F1: {best_f1:.4f})")
        
        if val_metrics['auc_roc'] > best_auc:
            best_auc = val_metrics['auc_roc']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'history': history
            }, os.path.join(output_dir, 'best_model_auc.pt'))
            print(f"  -> Saved best model (AUC: {best_auc:.4f})")
        
        # 定期保存
        if epoch % save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history
            }, os.path.join(output_dir, f'checkpoint_epoch{epoch}.pt'))
    
    # 保存最终模型
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'history': history
    }, os.path.join(output_dir, 'final_model.pt'))
    
    # 保存训练历史
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # 绘制训练曲线
    plot_training_curves(history, output_dir, use_combined_loss)
    
    return history


def plot_training_curves(history: dict, output_dir: str, use_combined_loss: bool = True):
    """绘制训练曲线"""
    if use_combined_loss:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
    
    # 1. 训练损失
    ax = axes[0, 0] if use_combined_loss else axes[0]
    ax.plot(history['train_loss'], label='Total Loss')
    ax.plot(history['train_classify_loss'], label='Classify Loss')
    if use_combined_loss:
        ax.plot(history['train_align_loss'], label='Alignment Loss')
        ax.plot(history['train_safety_loss'], label='Safety Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 分类指标 - Accuracy & F1
    ax = axes[0, 1] if use_combined_loss else axes[1]
    ax.plot(history['val_accuracy'], label='Accuracy', color='blue')
    ax.plot(history['val_f1'], label='F1 Score', color='green')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title('Classification Metrics (Higher is Better)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Precision & Recall
    ax = axes[0, 2] if use_combined_loss else axes[2]
    ax.plot(history['val_precision'], label='Precision', color='orange')
    ax.plot(history['val_recall'], label='Recall', color='red')
    ax.plot(history['val_auc_roc'], label='AUC-ROC', color='purple')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title('Precision, Recall & AUC-ROC')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 组内相似度
    ax = axes[1, 0] if use_combined_loss else axes[3]
    ax.plot(history['val_intra_group_sim'], label='Intra-Group Similarity', color='blue')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Similarity')
    ax.set_title('Cross-Lingual Alignment (Higher is Better)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Safe-Unsafe相似度
    if use_combined_loss:
        ax = axes[1, 1]
        ax.plot(history['val_safe_unsafe_sim'], label='Safe-Unsafe Similarity', color='red')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Similarity')
        ax.set_title('Safe-Unsafe Similarity (Lower is Better)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. 分离度
        ax = axes[1, 2]
        separation = [1 - s for s in history['val_safe_unsafe_sim']]  # 转换为分离度
        ax.plot(separation, label='Separation', color='green')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Separation')
        ax.set_title('Safety Separation (Higher is Better)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300)
    plt.close()
    
    print(f"Training curves saved to {os.path.join(output_dir, 'training_curves.png')}")


def main():
    parser = argparse.ArgumentParser(description="Train Transform MLP with Classifier")
    
    # 数据参数
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to prepared hidden states dataset")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Validation split ratio")
    parser.add_argument("--train_languages", type=str, nargs="+", default=None,
                        help="Only use these languages for training/val")
    
    # 模型参数
    parser.add_argument("--hidden_dim", type=int, default=4096,
                        help="Hidden dimension for transform MLP")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Number of MLP layers")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    parser.add_argument("--use_residual", action="store_true",
                        help="Use residual connection")
    parser.add_argument("--classifier_hidden_dim", type=int, default=512,
                        help="Hidden dimension for classifier head")
    parser.add_argument("--identity_mlp", action="store_true",
                        help="Use identity mapping for MLP (output = input)")
    parser.add_argument("--freeze_mlp", action="store_true",
                        help="Freeze MLP parameters, only train classifier head")
    
    # 训练参数
    parser.add_argument("--num_epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay")
    
    # 损失函数参数
    parser.add_argument("--loss_type", type=str, default="combined",
                        choices=["combined", "classify_only"],
                        help="Loss type: 'combined' (contrastive + classify) or 'classify_only'")
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="Temperature for contrastive loss")
    parser.add_argument("--margin", type=float, default=1.0,
                        help="Margin for safety loss")
    parser.add_argument("--lambda_align", type=float, default=1.0,
                        help="Weight for alignment loss")
    parser.add_argument("--lambda_safety", type=float, default=1.0,
                        help="Weight for safety contrastive loss")
    parser.add_argument("--lambda_classify", type=float, default=1.0,
                        help="Weight for classification loss")
    
    # 其他参数
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Output directory")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Custom run name")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save checkpoint every N epochs")
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 加载数据
    print(f"Loading dataset from {args.data_path}...")
    dataset = load_dataset(args.data_path)
    print(f"Dataset size: {len(dataset)}")
    
    # 统计数据分布
    safe_count = sum(1 for l in dataset.safety_labels if l == 0)
    unsafe_count = sum(1 for l in dataset.safety_labels if l == 1)
    print(f"  Safe (benign) samples: {safe_count}")
    print(f"  Unsafe (harmful) samples: {unsafe_count}")
    
    # 可选：按语言过滤
    if args.train_languages:
        allowed = set(args.train_languages)
        keep_indices = [i for i, lang in enumerate(dataset.languages) if lang in allowed]
        if len(keep_indices) == 0:
            raise ValueError(f"No samples found for languages: {allowed}")
        
        def _subset(seq):
            return [seq[i] for i in keep_indices]
        
        dataset = MultilingualHiddenStateDataset(
            hidden_states=_subset(dataset.hidden_states),
            group_ids=_subset(dataset.group_ids),
            safety_labels=_subset(dataset.safety_labels),
            languages=_subset(dataset.languages),
            texts=_subset(dataset.texts),
        )
        print(f"Filtered to languages {sorted(allowed)}, remaining samples: {len(dataset)}")
    
    # 分割训练/验证集
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    
    val_size = int(len(indices) * args.val_split)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    train_dataset = MultilingualHiddenStateDataset(
        hidden_states=[dataset.hidden_states[i] for i in train_indices],
        group_ids=[dataset.group_ids[i] for i in train_indices],
        safety_labels=[dataset.safety_labels[i] for i in train_indices],
        languages=[dataset.languages[i] for i in train_indices],
        texts=[dataset.texts[i] for i in train_indices]
    )
    
    val_dataset = MultilingualHiddenStateDataset(
        hidden_states=[dataset.hidden_states[i] for i in val_indices],
        group_ids=[dataset.group_ids[i] for i in val_indices],
        safety_labels=[dataset.safety_labels[i] for i in val_indices],
        languages=[dataset.languages[i] for i in val_indices],
        texts=[dataset.texts[i] for i in val_indices]
    )
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    # 获取输入维度
    input_dim = dataset.hidden_states[0].shape[0]
    print(f"Input dimension: {input_dim}")
    
    # 创建模型
    model = TransformMLPWithClassifier(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=input_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_residual=args.use_residual,
        use_layer_norm=True,
        classifier_hidden_dim=args.classifier_hidden_dim,
        identity_mlp=args.identity_mlp
    ).to(args.device)
    
    # 如果设置了freeze_mlp，冻结MLP参数
    if args.freeze_mlp:
        model.freeze_mlp()
        print("\n>>> MLP parameters frozen, only training classifier head")
    
    print(f"\nModel architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    if args.freeze_mlp or args.identity_mlp:
        print(f"Frozen parameters: {total_params - trainable_params:,}")
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.run_name:
        run_folder = f"{args.run_name}_{timestamp}"
    else:
        run_folder = f"classifier_run_{timestamp}"
    output_dir = os.path.join(args.output_dir, run_folder)
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存配置
    config = vars(args)
    config['input_dim'] = input_dim
    config['timestamp'] = timestamp
    
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # 训练
    history = train(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        margin=args.margin,
        lambda_align=args.lambda_align,
        lambda_safety=args.lambda_safety,
        lambda_classify=args.lambda_classify,
        device=args.device,
        loss_type=args.loss_type,
        save_every=args.save_every
    )
    
    print(f"\nTraining completed!")
    print(f"Results saved to: {output_dir}")
    print(f"\nBest metrics:")
    print(f"  Best F1: {max(history['val_f1']):.4f}")
    print(f"  Best AUC-ROC: {max(history['val_auc_roc']):.4f}")
    print(f"  Best Accuracy: {max(history['val_accuracy']):.4f}")


if __name__ == "__main__":
    main()


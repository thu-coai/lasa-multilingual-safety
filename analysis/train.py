"""
训练脚本

训练MLP网络实现：
1. 跨语言对齐：不同语言同一问题的编码尽可能接近
2. 安全性区分：有害问题和无害问题的距离尽可能远
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

from model import TransformMLP, ContrastiveLoss, TripletContrastiveLoss
from dataset import (
    MultilingualHiddenStateDataset,
    load_dataset,
    collate_fn,
    BalancedBatchSampler,
    prepare_dataset
)


def set_seed(seed: int):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(
    model: TransformMLP,
    dataloader: DataLoader,
    device: str
) -> dict:
    """
    计算评估指标
    
    指标：
    1. 跨语言对齐：同组内平均相似度
    2. 安全性区分：safe-unsafe之间的平均距离
    """
    model.eval()
    
    all_embeddings = []
    all_group_ids = []
    all_safety_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            hidden_states = batch['hidden_states'].to(device)
            
            embeddings = model(hidden_states)
            
            all_embeddings.append(embeddings.cpu())
            all_group_ids.extend(batch['group_ids'].tolist())
            all_safety_labels.extend(batch['safety_labels'].tolist())
    
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_embeddings = nn.functional.normalize(all_embeddings, p=2, dim=1)
    
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
        
        group_embeddings = all_embeddings[indices]
        sim_matrix = torch.matmul(group_embeddings, group_embeddings.T)
        
        # 排除对角线
        mask = ~torch.eye(len(indices), dtype=torch.bool)
        intra_group_sims.extend(sim_matrix[mask].tolist())
    
    avg_intra_group_sim = np.mean(intra_group_sims) if intra_group_sims else 0.0
    
    # 2. 计算safe-unsafe之间的平均距离
    safe_indices = [i for i, l in enumerate(all_safety_labels) if l == 0]
    unsafe_indices = [i for i, l in enumerate(all_safety_labels) if l == 1]
    
    if safe_indices and unsafe_indices:
        safe_embeddings = all_embeddings[safe_indices]
        unsafe_embeddings = all_embeddings[unsafe_indices]
        
        cross_sim = torch.matmul(safe_embeddings, unsafe_embeddings.T)
        avg_cross_sim = cross_sim.mean().item()
    else:
        avg_cross_sim = 0.0
    
    # 3. 计算同类型内的相似度（作为baseline）
    same_type_sims = []
    
    # Safe内部
    if len(safe_indices) >= 2:
        safe_emb = all_embeddings[safe_indices]
        safe_sim = torch.matmul(safe_emb, safe_emb.T)
        mask = ~torch.eye(len(safe_indices), dtype=torch.bool)
        same_type_sims.extend(safe_sim[mask].tolist())
    
    # Unsafe内部
    if len(unsafe_indices) >= 2:
        unsafe_emb = all_embeddings[unsafe_indices]
        unsafe_sim = torch.matmul(unsafe_emb, unsafe_emb.T)
        mask = ~torch.eye(len(unsafe_indices), dtype=torch.bool)
        same_type_sims.extend(unsafe_sim[mask].tolist())
    
    avg_same_type_sim = np.mean(same_type_sims) if same_type_sims else 0.0
    
    return {
        'intra_group_similarity': avg_intra_group_sim,
        'safe_unsafe_similarity': avg_cross_sim,
        'same_type_similarity': avg_same_type_sim,
        'separation_gap': avg_same_type_sim - avg_cross_sim  # 越大越好
    }


def train_epoch(
    model: TransformMLP,
    dataloader: DataLoader,
    criterion,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int
) -> dict:
    """训练一个epoch"""
    model.train()
    
    total_loss = 0.0
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
        embeddings = model(hidden_states)
        
        # Compute loss
        losses = criterion(embeddings, group_ids, safety_labels)
        
        # Backward
        losses['total_loss'].backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += losses['total_loss'].item()
        total_align_loss += losses['alignment_loss'].item()
        total_safety_loss += losses['safety_loss'].item()
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f"{losses['total_loss'].item():.4f}",
            'align': f"{losses['alignment_loss'].item():.4f}",
            'safety': f"{losses['safety_loss'].item():.4f}"
        })
    
    return {
        'loss': total_loss / num_batches,
        'alignment_loss': total_align_loss / num_batches,
        'safety_loss': total_safety_loss / num_batches
    }


def train(
    model: TransformMLP,
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
    device: str = 'cuda',
    loss_type: str = 'contrastive',
    save_every: int = 10
):
    """
    训练模型
    
    Args:
        model: MLP模型
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
        lambda_safety: 安全性损失权重
        device: 设备
        loss_type: 损失类型 ('contrastive' or 'triplet')
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
    if loss_type == 'contrastive':
        criterion = ContrastiveLoss(
            temperature=temperature,
            margin=margin,
            lambda_align=lambda_align,
            lambda_safety=lambda_safety
        )
    else:
        criterion = TripletContrastiveLoss(
            margin=margin,
            safety_margin=margin * 2,
            lambda_align=lambda_align,
            lambda_safety=lambda_safety
        )
    
    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
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
        'train_align_loss': [],
        'train_safety_loss': [],
        'val_intra_group_sim': [],
        'val_safe_unsafe_sim': [],
        'val_separation_gap': []
    }
    
    best_gap = -float('inf')
    
    print(f"\nStarting training...")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Loss type: {loss_type}")
    print(f"  Lambda align: {lambda_align}")
    print(f"  Lambda safety: {lambda_safety}")
    
    for epoch in range(1, num_epochs + 1):
        # 训练
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # 验证
        val_metrics = compute_metrics(model, val_loader, device)
        
        # 更新学习率
        scheduler.step()
        
        # 记录历史
        history['train_loss'].append(train_metrics['loss'])
        history['train_align_loss'].append(train_metrics['alignment_loss'])
        history['train_safety_loss'].append(train_metrics['safety_loss'])
        history['val_intra_group_sim'].append(val_metrics['intra_group_similarity'])
        history['val_safe_unsafe_sim'].append(val_metrics['safe_unsafe_similarity'])
        history['val_separation_gap'].append(val_metrics['separation_gap'])
        
        print(f"\nEpoch {epoch}/{num_epochs}:")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
              f"Align: {train_metrics['alignment_loss']:.4f}, "
              f"Safety: {train_metrics['safety_loss']:.4f}")
        print(f"  Val - IntraGroup Sim: {val_metrics['intra_group_similarity']:.4f}, "
              f"Safe-Unsafe Sim: {val_metrics['safe_unsafe_similarity']:.4f}, "
              f"Gap: {val_metrics['separation_gap']:.4f}")
        
        # 保存最佳模型
        if val_metrics['separation_gap'] > best_gap:
            best_gap = val_metrics['separation_gap']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'history': history
            }, os.path.join(output_dir, 'best_model.pt'))
            print(f"  -> Saved best model (gap: {best_gap:.4f})")
        
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
    plot_training_curves(history, output_dir)
    
    return history


def plot_training_curves(history: dict, output_dir: str):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 训练损失
    ax = axes[0, 0]
    ax.plot(history['train_loss'], label='Total Loss')
    ax.plot(history['train_align_loss'], label='Alignment Loss')
    ax.plot(history['train_safety_loss'], label='Safety Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 组内相似度
    ax = axes[0, 1]
    ax.plot(history['val_intra_group_sim'], label='Intra-Group Similarity', color='blue')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Similarity')
    ax.set_title('Cross-Lingual Alignment (Higher is Better)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Safe-Unsafe相似度
    ax = axes[1, 0]
    ax.plot(history['val_safe_unsafe_sim'], label='Safe-Unsafe Similarity', color='red')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Similarity')
    ax.set_title('Safe-Unsafe Similarity (Lower is Better)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 分离度
    ax = axes[1, 1]
    ax.plot(history['val_separation_gap'], label='Separation Gap', color='green')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gap')
    ax.set_title('Safety Separation Gap (Higher is Better)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300)
    plt.close()
    
    print(f"Training curves saved to {os.path.join(output_dir, 'training_curves.png')}")


def main():
    parser = argparse.ArgumentParser(description="Train Transform MLP")
    
    # 数据参数
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to prepared hidden states dataset")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Validation split ratio")
    parser.add_argument("--train_languages", type=str, nargs="+", default=None,
                        help="Only use these languages for training/val (e.g., en zh ko)")
    
    # 模型参数
    parser.add_argument("--hidden_dim", type=int, default=4096,
                        help="Hidden dimension")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Number of MLP layers")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    parser.add_argument("--use_residual", action="store_true",
                        help="Use residual connection")
    
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
    parser.add_argument("--loss_type", type=str, default="contrastive",
                        choices=["contrastive", "triplet"],
                        help="Loss type")
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="Temperature for contrastive loss")
    parser.add_argument("--margin", type=float, default=1.0,
                        help="Margin for triplet/safety loss")
    parser.add_argument("--lambda_align", type=float, default=1.0,
                        help="Weight for alignment loss")
    parser.add_argument("--lambda_safety", type=float, default=1.0,
                        help="Weight for safety loss")
    
    # 其他参数
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Output directory")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Custom run name (default: auto-generated with timestamp)")
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

    # 可选：按语言过滤训练/验证数据，用于“只用少量语言训练、测试泛化”
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
    model = TransformMLP(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=input_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_residual=args.use_residual,
        use_layer_norm=True
    ).to(args.device)
    
    print(f"\nModel architecture:")
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.run_name:
        run_folder = f"{args.run_name}_{timestamp}"
    else:
        run_folder = f"run_{timestamp}"
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
        device=args.device,
        loss_type=args.loss_type,
        save_every=args.save_every
    )
    
    print(f"\nTraining completed!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()


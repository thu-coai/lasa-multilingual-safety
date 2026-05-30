"""
Transform MLP Network for Hidden State Alignment

目标：
1. 不同语言同一问题的编码尽可能接近（跨语言对齐）
2. 有害问题和无害问题的距离尽可能远（安全性区分）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformMLP(nn.Module):
    """
    MLP网络，用于变换hidden state
    
    特点：
    - 残差连接：保留原始信息
    - 层归一化：提高训练稳定性
    - 可选的多层设计
    """
    
    def __init__(
        self,
        input_dim: int = 4096,
        hidden_dim: int = 4096,
        output_dim: int = 4096,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_residual: bool = True,
        use_layer_norm: bool = True,
        activation: str = 'gelu'
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_residual = use_residual and (input_dim == output_dim)
        self.use_layer_norm = use_layer_norm
        
        # 激活函数
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.GELU()
        
        # 构建层
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            next_dim = hidden_dim if i < num_layers - 1 else output_dim
            
            layers.append(nn.Linear(current_dim, next_dim))
            
            if i < num_layers - 1:  # 最后一层不加激活和dropout
                if use_layer_norm:
                    layers.append(nn.LayerNorm(next_dim))
                layers.append(self.activation)
                layers.append(nn.Dropout(dropout))
            
            current_dim = next_dim
        
        self.mlp = nn.Sequential(*layers)
        
        # 输出层归一化
        if use_layer_norm:
            self.output_norm = nn.LayerNorm(output_dim)
        
        # 残差连接的缩放因子（可学习）
        if self.use_residual:
            self.residual_scale = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入hidden state, shape [batch_size, input_dim]
        
        Returns:
            变换后的hidden state, shape [batch_size, output_dim]
        """
        out = self.mlp(x)
        
        if self.use_residual:
            # 残差连接：output = input + scale * transformed
            out = x + self.residual_scale * out
        
        if self.use_layer_norm:
            out = self.output_norm(out)
        
        return out


class ContrastiveLoss(nn.Module):
    """
    对比学习损失函数
    
    结合两个目标：
    1. 跨语言对齐：同一问题不同语言的表示应该接近
    2. 安全性区分：有害问题和无害问题的表示应该远离
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        margin: float = 1.0,
        lambda_align: float = 1.0,
        lambda_safety: float = 1.0
    ):
        super().__init__()
        
        self.temperature = temperature
        self.margin = margin
        self.lambda_align = lambda_align
        self.lambda_safety = lambda_safety
    
    def alignment_loss(self, embeddings: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
        """
        跨语言对齐损失（InfoNCE）
        
        Args:
            embeddings: 变换后的表示, shape [batch_size, dim]
            group_ids: 问题ID，同一ID表示同一问题的不同语言版本
        
        Returns:
            对齐损失
        """
        batch_size = embeddings.size(0)
        
        if batch_size <= 1:
            return torch.tensor(0.0, device=embeddings.device)
        
        # L2归一化
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # 计算相似度矩阵
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # 创建正样本mask：同一group_id的样本为正样本
        group_ids = group_ids.view(-1, 1)
        positive_mask = (group_ids == group_ids.T).float()
        
        # 移除对角线（自己与自己）
        identity_mask = torch.eye(batch_size, device=embeddings.device)
        positive_mask = positive_mask - identity_mask
        
        # 检查是否有正样本
        if positive_mask.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device)
        
        # InfoNCE loss
        # 对每个样本，将正样本的相似度最大化，负样本的相似度最小化
        exp_sim = torch.exp(sim_matrix)
        
        # 负样本：移除对角线
        neg_mask = 1 - identity_mask
        
        # 分母：所有负样本的exp相似度之和
        neg_sum = (exp_sim * neg_mask).sum(dim=1, keepdim=True)
        
        # 对每个正样本对计算loss
        log_prob = sim_matrix - torch.log(exp_sim + neg_sum + 1e-8)
        
        # 只对正样本计算损失
        loss = -(log_prob * positive_mask).sum() / (positive_mask.sum() + 1e-8)
        
        return loss
    
    def safety_loss(
        self,
        safe_embeddings: torch.Tensor,
        unsafe_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        安全性区分损失（Margin-based）
        
        目标：让有害问题和无害问题的表示之间保持足够的距离
        
        Args:
            safe_embeddings: 无害问题的表示, shape [n_safe, dim]
            unsafe_embeddings: 有害问题的表示, shape [n_unsafe, dim]
        
        Returns:
            安全性损失
        """
        if safe_embeddings.size(0) == 0 or unsafe_embeddings.size(0) == 0:
            return torch.tensor(0.0, device=safe_embeddings.device)
        
        # L2归一化
        safe_embeddings = F.normalize(safe_embeddings, p=2, dim=1)
        unsafe_embeddings = F.normalize(unsafe_embeddings, p=2, dim=1)
        
        # 计算所有safe-unsafe对的相似度
        # shape: [n_safe, n_unsafe]
        sim_matrix = torch.matmul(safe_embeddings, unsafe_embeddings.T)
        
        # 我们希望相似度足够小（距离足够大）
        # Loss = max(0, similarity + margin)
        # 当similarity < -margin时，loss为0
        loss = F.relu(sim_matrix + self.margin).mean()
        
        return loss
    
    def forward(
        self,
        embeddings: torch.Tensor,
        group_ids: torch.Tensor,
        safety_labels: torch.Tensor
    ) -> dict:
        """
        计算总损失
        
        Args:
            embeddings: 变换后的表示, shape [batch_size, dim]
            group_ids: 问题ID
            safety_labels: 安全标签，0=无害，1=有害
        
        Returns:
            损失字典
        """
        # 跨语言对齐损失
        align_loss = self.alignment_loss(embeddings, group_ids)
        
        # 安全性区分损失
        safe_mask = safety_labels == 0
        unsafe_mask = safety_labels == 1
        
        safe_embeddings = embeddings[safe_mask]
        unsafe_embeddings = embeddings[unsafe_mask]
        
        safety_loss = self.safety_loss(safe_embeddings, unsafe_embeddings)
        
        # 总损失
        total_loss = self.lambda_align * align_loss + self.lambda_safety * safety_loss
        
        return {
            'total_loss': total_loss,
            'alignment_loss': align_loss,
            'safety_loss': safety_loss
        }


class TransformMLPWithClassifier(nn.Module):
    """
    带分类头的Transform MLP网络
    
    在TransformMLP的基础上添加一个分类头，用于预测样本是harmful还是benign
    hidden_dim -> 1 的二分类任务
    """
    
    def __init__(
        self,
        input_dim: int = 4096,
        hidden_dim: int = 4096,
        output_dim: int = 4096,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_residual: bool = True,
        use_layer_norm: bool = True,
        activation: str = 'gelu',
        classifier_hidden_dim: int = 512,
        identity_mlp: bool = False
    ):
        super().__init__()
        
        self.identity_mlp = identity_mlp
        
        # Transform MLP 部分
        if not identity_mlp:
            self.transform_mlp = TransformMLP(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_layers=num_layers,
                dropout=dropout,
                use_residual=use_residual,
                use_layer_norm=use_layer_norm,
                activation=activation
            )
        else:
            # 如果使用恒等映射，仍然创建MLP结构（用于兼容性），但forward时会跳过
            self.transform_mlp = TransformMLP(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_layers=num_layers,
                dropout=dropout,
                use_residual=use_residual,
                use_layer_norm=use_layer_norm,
                activation=activation
            )
        
        # 分类头：hidden_dim -> classifier_hidden_dim -> 1
        self.classifier = nn.Sequential(
            nn.Linear(output_dim, classifier_hidden_dim),
            nn.LayerNorm(classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, 1)
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_embeddings: bool = False
    ) -> dict:
        """
        前向传播
        
        Args:
            x: 输入hidden state, shape [batch_size, input_dim]
            return_embeddings: 是否返回中间表示
        
        Returns:
            字典包含:
                - logits: 分类logits, shape [batch_size, 1]
                - embeddings: 变换后的hidden state (如果return_embeddings=True)
        """
        # Transform部分：如果identity_mlp=True，直接返回输入（恒等映射）
        if self.identity_mlp:
            embeddings = x
        else:
            embeddings = self.transform_mlp(x)
        
        # 分类部分
        logits = self.classifier(embeddings)
        
        result = {'logits': logits}
        
        if return_embeddings:
            result['embeddings'] = embeddings
        
        return result
    
    def get_transform_mlp(self) -> TransformMLP:
        """获取内部的TransformMLP模块，方便单独使用"""
        return self.transform_mlp
    
    def freeze_mlp(self):
        """冻结MLP参数，只训练分类头"""
        for param in self.transform_mlp.parameters():
            param.requires_grad = False
    
    def unfreeze_mlp(self):
        """解冻MLP参数"""
        for param in self.transform_mlp.parameters():
            param.requires_grad = True


class CombinedLoss(nn.Module):
    """
    组合损失函数
    
    结合三个目标：
    1. 跨语言对齐：同一问题不同语言的表示应该接近
    2. 安全性区分：有害问题和无害问题的表示应该远离（对比学习）
    3. 分类损失：直接预测样本是harmful还是benign
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        margin: float = 1.0,
        lambda_align: float = 1.0,
        lambda_safety: float = 1.0,
        lambda_classify: float = 1.0
    ):
        super().__init__()
        
        self.contrastive_loss = ContrastiveLoss(
            temperature=temperature,
            margin=margin,
            lambda_align=lambda_align,
            lambda_safety=lambda_safety
        )
        
        self.lambda_classify = lambda_classify
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(
        self,
        logits: torch.Tensor,
        embeddings: torch.Tensor,
        group_ids: torch.Tensor,
        safety_labels: torch.Tensor
    ) -> dict:
        """
        计算总损失
        
        Args:
            logits: 分类logits, shape [batch_size, 1]
            embeddings: 变换后的表示, shape [batch_size, dim]
            group_ids: 问题ID
            safety_labels: 安全标签，0=无害，1=有害
        
        Returns:
            损失字典
        """
        # 对比学习损失
        contrastive_losses = self.contrastive_loss(embeddings, group_ids, safety_labels)
        
        # 分类损失
        # safety_labels: 0=benign, 1=harmful
        classify_loss = self.bce_loss(
            logits.squeeze(-1), 
            safety_labels.float()
        )
        
        # 总损失
        total_loss = (
            contrastive_losses['total_loss'] + 
            self.lambda_classify * classify_loss
        )
        
        return {
            'total_loss': total_loss,
            'alignment_loss': contrastive_losses['alignment_loss'],
            'safety_loss': contrastive_losses['safety_loss'],
            'classify_loss': classify_loss
        }


class ClassificationOnlyLoss(nn.Module):
    """
    仅分类损失函数
    
    只做二分类任务：预测样本是harmful还是benign
    """
    
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(
        self,
        logits: torch.Tensor,
        safety_labels: torch.Tensor
    ) -> dict:
        """
        计算分类损失
        
        Args:
            logits: 分类logits, shape [batch_size, 1]
            safety_labels: 安全标签，0=无害(benign)，1=有害(harmful)
        
        Returns:
            损失字典
        """
        classify_loss = self.bce_loss(
            logits.squeeze(-1), 
            safety_labels.float()
        )
        
        return {
            'total_loss': classify_loss,
            'classify_loss': classify_loss
        }


class TripletContrastiveLoss(nn.Module):
    """
    基于Triplet的对比损失
    
    更直接的方式：
    - Anchor: 某语言的问题
    - Positive: 同一问题的其他语言版本
    - Negative: 不同问题的表示
    """
    
    def __init__(
        self,
        margin: float = 1.0,
        safety_margin: float = 2.0,
        lambda_align: float = 1.0,
        lambda_safety: float = 1.0
    ):
        super().__init__()
        
        self.margin = margin
        self.safety_margin = safety_margin
        self.lambda_align = lambda_align
        self.lambda_safety = lambda_safety
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)
    
    def forward(
        self,
        embeddings: torch.Tensor,
        group_ids: torch.Tensor,
        safety_labels: torch.Tensor
    ) -> dict:
        """
        计算总损失
        """
        batch_size = embeddings.size(0)
        device = embeddings.device
        
        # L2归一化
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        
        # ========== 跨语言对齐损失 ==========
        align_loss = torch.tensor(0.0, device=device)
        triplet_count = 0
        
        unique_groups = group_ids.unique()
        
        for group_id in unique_groups:
            group_mask = group_ids == group_id
            group_indices = torch.where(group_mask)[0]
            
            if len(group_indices) < 2:
                continue
            
            # 该组外的负样本
            neg_mask = ~group_mask
            neg_indices = torch.where(neg_mask)[0]
            
            if len(neg_indices) == 0:
                continue
            
            # 对组内每对样本创建triplet
            for i, anchor_idx in enumerate(group_indices):
                for positive_idx in group_indices[i+1:]:
                    # 随机选择一个负样本
                    neg_idx = neg_indices[torch.randint(len(neg_indices), (1,))]
                    
                    anchor = embeddings_norm[anchor_idx:anchor_idx+1]
                    positive = embeddings_norm[positive_idx:positive_idx+1]
                    negative = embeddings_norm[neg_idx]
                    
                    align_loss += self.triplet_loss(anchor, positive, negative)
                    triplet_count += 1
        
        if triplet_count > 0:
            align_loss = align_loss / triplet_count
        
        # ========== 安全性区分损失 ==========
        safe_mask = safety_labels == 0
        unsafe_mask = safety_labels == 1
        
        safe_embeddings = embeddings_norm[safe_mask]
        unsafe_embeddings = embeddings_norm[unsafe_mask]
        
        safety_loss = torch.tensor(0.0, device=device)
        
        if safe_embeddings.size(0) > 0 and unsafe_embeddings.size(0) > 0:
            # 计算safe和unsafe之间的相似度
            sim_matrix = torch.matmul(safe_embeddings, unsafe_embeddings.T)
            # 希望相似度足够小
            safety_loss = F.relu(sim_matrix + self.safety_margin).mean()
        
        total_loss = self.lambda_align * align_loss + self.lambda_safety * safety_loss
        
        return {
            'total_loss': total_loss,
            'alignment_loss': align_loss,
            'safety_loss': safety_loss
        }


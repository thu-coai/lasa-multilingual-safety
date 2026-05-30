"""
Transform Training Module

训练MLP网络实现：
1. 跨语言对齐：不同语言同一问题的编码尽可能接近
2. 安全性区分：有害问题和无害问题的距离尽可能远
"""

from .model import TransformMLP, ContrastiveLoss, TripletContrastiveLoss
from .dataset import (
    MultilingualHiddenStateDataset,
    load_dataset,
    load_ultrafeedback_data,
    load_safety_data,
    extract_hidden_states,
    collate_fn
)

__all__ = [
    'TransformMLP',
    'ContrastiveLoss',
    'TripletContrastiveLoss',
    'MultilingualHiddenStateDataset',
    'load_dataset',
    'load_ultrafeedback_data',
    'load_safety_data',
    'extract_hidden_states',
    'collate_fn'
]


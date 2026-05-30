"""
Safety Semantic Interpreter (SSI) module for LASA.

SSI is the lightweight MLP classifier trained on hidden states from the
semantic bottleneck layer. It predicts whether the prompt semantics are
benign or harmful, independent of the surface language.
"""
import os
import json
import torch
import torch.nn as nn
from typing import Tuple
from loguru import logger


class SimpleSafetyClassifier(nn.Module):
    """
    Lightweight SSI MLP for binary safety prediction.
    Input: semantic bottleneck hidden state -> Output: 0=benign, 1=harmful.
    """
    
    def __init__(
        self,
        input_dim: int = 4096,
        hidden_dim: int = 4096,
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
    
    def predict_safety(
        self,
        x: torch.Tensor,
        threshold: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict SSI safety labels.
        
        Args:
            x: Bottleneck hidden states [batch_size, hidden_dim]
            threshold: Classification threshold
        
        Returns:
            Tuple of (predictions [batch_size], probabilities [batch_size])
            predictions: 0=benign, 1=harmful
        """
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits).squeeze(-1)
            return (probs > threshold).long(), probs


def load_safety_classifier(
    classifier_dir: str,
    device: str = 'cuda:0'
) -> Tuple[SimpleSafetyClassifier, int, float]:
    """
    Load an SSI classifier from a directory containing checkpoint.pt and config.json.
    
    Args:
        classifier_dir: Path to SSI directory containing checkpoint.pt and config.json
        device: Device to load the module on
    
    Returns:
        Tuple of (SSI module, semantic bottleneck layer_idx, threshold)
    """
    # Load config
    config_path = os.path.join(classifier_dir, 'config.json')
    checkpoint_path = os.path.join(classifier_dir, 'checkpoint.pt')
    
    logger.info(f"Loading SSI classifier from {classifier_dir}...")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    layer_idx = config['layer_idx']
    threshold = config['threshold']
    hidden_dim = config['hidden_dim']
    
    logger.info(f"Config: layer_idx={layer_idx}, threshold={threshold}, hidden_dim={hidden_dim}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model_state_dict']
    
    # Parse model dimensions from state dict
    input_dim = state_dict['mlp.0.weight'].shape[1]
    output_dim = state_dict['mlp.4.weight'].shape[0]
    
    module = SimpleSafetyClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        dropout=0.1
    )
    module.load_state_dict(state_dict)
    
    module = module.to(device)
    module.eval()
    for param in module.parameters():
        param.requires_grad = False
    
    logger.info(f"Loaded SimpleSafetyClassifier (input_dim={input_dim}, hidden_dim={hidden_dim})")
    
    return module, layer_idx, threshold

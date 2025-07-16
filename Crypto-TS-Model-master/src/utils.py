import torch
import yaml
from datetime import datetime
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any, Optional, Union, Tuple 
import numpy as np 
from torch import nn
import torch.nn.functional as F

def seed_everything(seed: int = 42):
    """Cố định seed cho tất cả thư viện"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

class TrainingTracker:
    """Theo dõi quá trình training"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        log_dir = Path(config['training']['log_dir']) / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(log_dir)
        self._save_config()

    def _save_config(self):
        """Lưu config vào thư mục log"""
        with open(Path(self.writer.log_dir) / "config.yaml", 'w') as f:
            yaml.dump(self.config, f)

    def log(self, tag: str, value: float, step: int):
        """Ghi log metrics"""
        self.writer.add_scalar(tag, value, step)

    def close(self):
        """Đóng SummaryWriter"""
        self.writer.close()

class EarlyStopper:
    def __init__(self, config: Dict[str, Any]):  
        self.patience = config['training']['patience']
        self.min_delta = config['training'].get('min_delta', 0.01)
        self.counter = 0
        self.best_loss = float('inf')

    def check(self, current_loss: float) -> bool:
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

class CompositeLoss(nn.Module):
    """combine many loss function with weight"""
    def __init__(self, losses: list, weights: list):
        super().__init__()
        self.losses = losses
        self.weights = weights
        
    def forward(self, pred, target):
        total_loss = 0
        for loss, weight in zip(self.losses, self.weights):
            total_loss += weight * loss(pred, target)
        return total_loss

class QuantileLoss(nn.Module):
    """Loss function for quantile regression"""
    def __init__(self, quantiles: list):
        super().__init__()
        self.quantiles = quantiles
        
    def forward(self, pred, target):
        # Ensure shapes match
        if target.shape[1] != pred.shape[1]:
            min_len = min(target.shape[1], pred.shape[1])
            target = target[:, :min_len, :]
            pred = pred[:, :min_len, :]
            
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - pred  # Use same shape predictions
            losses.append(torch.max((q-1) * errors, q * errors).unsqueeze(1))
        return torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))

class ModelEMA:
    """Exponential Moving Average for model weights"""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                if name not in self.shadow:
                    self.shadow[name] = param.data.clone()
                else:
                    self.shadow[name] = (self.decay * self.shadow[name] 
                                      + (1 - self.decay) * param.data)
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}
    
    @property
    def module(self):
        return self.model

class DirectionLoss(nn.Module):
    def __init__(self, alpha=0.3):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        
    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        
        pred_directions = torch.sign(pred[:, 1:] - pred[:, :-1])
        true_directions = torch.sign(target[:, 1:] - target[:, :-1])
        direction_loss = F.binary_cross_entropy_with_logits(
            pred_directions.float(), 
            (true_directions > 0).float()
        )
        
        return (1-self.alpha)*mse_loss + self.alpha*direction_loss
    
class VolatilityWeightedLoss(nn.Module):
    def __init__(self, base_loss=nn.HuberLoss(delta=0.5)):
        super().__init__()
        self.base_loss = base_loss
        
    def forward(self, pred, target):
        residuals = (pred - target).abs()
        volatility = residuals.unfold(1, 5, 1).std(dim=-1)
        weights = 1 / (volatility + 1e-6)
        return (self.base_loss(pred, target) * weights).mean()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple 

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * 
                   -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class VolatilityEmbedding(nn.Module):
    def __init__(self, d_model, lookback=11):
        super().__init__()
        assert lookback % 2 == 1, "Lookback must be odd number"
        self.lookback = lookback
        self.proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, x_close):  # x_close: [B,T,1]
        B, T, _ = x_close.shape
        
        # 1. Tính returns với padding đầu
        returns = F.pad(x_close.diff(dim=1).abs(), (0,0,1,0), value=0)  # [B,T,1]
        
        # 2. Tính rolling volatility
        volatility = returns.unfold(1, self.lookback, 1).std(dim=-1, keepdim=True)  # [B,T-lookback+1,1]
        
        # 3. Padding đối xứng chính xác để giữ nguyên kích thước
        pad_front = (self.lookback - 1) // 2
        pad_back = (self.lookback - 1) // 2
        volatility = F.pad(volatility, (0,0,pad_front,pad_back), mode='replicate')  # [B,T,1]
        
        # 4. Project cuối cùng
        return self.proj(volatility)  # [B,T,D]
    
class CryptoTokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model, patch_size):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(patch_size * c_in, d_model),  
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        
    def forward(self, x):
        return self.projection(x)

class CryptoTimeEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.minute_embed = nn.Embedding(60, d_model)
        self.hour_embed = nn.Embedding(24, d_model)
        
    def forward(self, x_mark):
        # Lấy sample tương ứng với các patches
        if x_mark.size(1) > 35:  # Nếu sequence dài hơn số patches
            indices = torch.linspace(0, x_mark.size(1)-1, 35).long()
            x_mark = x_mark[:, indices, :]
            
        minute_x = self.minute_embed(x_mark[..., 0].long())  # Phút
        hour_x = self.hour_embed(x_mark[..., 1].long())      # Giờ
        return minute_x + hour_x  # [B, 35, D]

class CryptoDataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, patch_size=16, lookback=11, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size
        
        # 1. Token Embedding
        self.token_embedding = nn.Sequential(
            nn.Linear(patch_size * c_in, d_model),
            nn.LayerNorm(d_model)
        )
        
        # 2. Volatility Embedding (ĐÃ SỬA)
        self.volatility_embedding = VolatilityEmbedding(d_model, lookback)
        
        # 3. Time Embedding (ĐÃ SỬA)
        self.time_embedding = CryptoTimeEmbedding(d_model)
        
        # 4. Positional Embedding
        self.position_embedding = PositionalEmbedding(d_model)
        
        # 5. Gate & Dropout
        self.volatility_gate = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

class CryptoDataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, patch_size=16, lookback=11, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size
        
        # 1. Token Embedding
        self.token_embedding = nn.Sequential(
            nn.Linear(patch_size * c_in, d_model),
            nn.LayerNorm(d_model)
        )
        
        # 2. Volatility Embedding (ĐÃ SỬA)
        self.volatility_embedding = VolatilityEmbedding(d_model, lookback)
        
        # 3. Time Embedding (ĐÃ SỬA)
        self.time_embedding = CryptoTimeEmbedding(d_model)
        
        # 4. Positional Embedding
        self.position_embedding = PositionalEmbedding(d_model)
        
        # 5. Gate & Dropout
        self.volatility_gate = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mark=None):
        B, T, _ = x.shape
        
        # 1. Token Embedding
        x_embed = self.token_embedding(x)  # [B,T,D]
        
        # 2. Volatility Embedding (FIXED)
        volatility = self.volatility_embedding(x[:, :, -1:])  # [B,T,D]
        
        # 3. Time Embedding (FIXED)
        if x_mark is not None:
            # Lấy mẫu time features theo patches
            patch_indices = torch.linspace(0, x_mark.size(1)-1, T).long()
            time_embed = self.time_embedding(x_mark[:, patch_indices, :])  # [B,T,D]
        else:
            time_embed = 0
        
        # 4. Positional Embedding
        pos_embed = self.position_embedding(x)[:, :T, :]  # [1,T,D]
        
        # 5. Tính toán output
        gate = torch.sigmoid(self.volatility_gate(volatility))
        out = (x_embed + time_embed + pos_embed) * gate + volatility
        
        return self.dropout(out)

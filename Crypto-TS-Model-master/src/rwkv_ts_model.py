import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math
from dataclasses import dataclass
from typing import Dict, Any
from embed import CryptoDataEmbedding

# --------------------------
# 1. Cấu trúc Config
# --------------------------
@dataclass
class RWKVConfig:
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    bias: bool = True

class ModelConfig:
    def __init__(self, config_dict: Dict[str, Any]):
        model_config = config_dict['model']
        
        # Các tham số bắt buộc
        self.enc_in = model_config['enc_in']
        self.d_model = model_config['d_model']
        self.n_heads = model_config['n_heads']
        self.e_layers = model_config['e_layers']
        self.patch_size = model_config['patch_size']
        self.c_out = model_config['c_out']
        self.seq_len = model_config['seq_len']
        self.pred_len = model_config['pred_len']
        
        # Các tham số có giá trị mặc định
        self.dropout = model_config.get('dropout', 0.1)
        self.embed = model_config.get('embed', 'fixed')
        self.freq = model_config.get('freq', 'h')
        self.d_ff = model_config.get('d_ff', 2048)

# --------------------------
# 2. Layer Normalization
# --------------------------
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

# --------------------------
# 3. Patching Module
# --------------------------
class Patching(nn.Module):
    def __init__(self, patch_size, stride):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride

    def forward(self, x):  # [B,L,C]
        B, L, C = x.shape
        # Tính số patches chính xác
        num_patches = (L - self.patch_size) // self.stride + 1
        # Tạo patches [B,num_patches,patch_size*C]
        x = x.unfold(1, self.patch_size, self.stride).reshape(B, num_patches, -1)
        return x

# --------------------------
# 4. Time Mixing Module
# --------------------------
class CryptoRWKV_TimeMix(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.head_size = config.n_embd // config.n_head
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # Time-aware parameters
        with torch.no_grad():
            ratio_0_to_1 = layer_id / (config.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / config.n_layer)  # 1 to ~0
            
            # Time weighting curve
            ddd = torch.ones(1, 1, config.n_embd)
            for i in range(config.n_embd):
                ddd[0, 0, i] = i / config.n_embd  # Linear from 0 to 1

            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            # Time decay
            decay_speed = torch.ones(self.n_head)
            for h in range(self.n_head):
                decay_speed[h] = -6 + 5 * (h / (self.n_head - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.unsqueeze(-1))  # [n_head, 1]

        # Volatility projection
        self.volatility_proj = nn.Linear(1, config.n_embd)
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))  # Shift along time dimension

        # Linear transformations
        self.key = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.receptance = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.output = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Normalization and regularization
        self.ln_x = nn.GroupNorm(self.n_head, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, volatility=None):
        B, T, C = x.size()  # Batch, Time, Channels
        H, N = self.n_head, self.head_size
        
        # 1. Volatility Adjustment (Optional)
        if volatility is not None:
            v_emb = torch.sigmoid(self.volatility_proj(volatility.unsqueeze(-1)))  # [B,T,1] -> [B,T,C]
            x = x * v_emb  # Scale features by volatility

        # 2. Time Shift and Difference
        xx = self.time_shift(x) - x  # [B,T,C]
        
        # 3. Time-aware Feature Adjustment
        xk = x + xx * self.time_maa_k  # Time-adjusted keys
        xv = x + xx * self.time_maa_v  # Time-adjusted values
        xr = x + xx * self.time_maa_r  # Time-adjusted receptance

        # 4. Linear Projections
        r = self.receptance(xr).view(B, T, H, N).transpose(1, 2)  # [B,H,T,N]
        k = self.key(xk).view(B, T, H, N).permute(0, 2, 3, 1)     # [B,H,N,T]
        v = self.value(xv).view(B, T, H, N).transpose(1, 2)       # [B,H,T,N]

        # 5. Time Decay Weights
        w = torch.exp(-torch.exp(self.time_decay.float()))  # [H,1]
        wk = w.view(1, H, 1, 1)  # For state update
        wb = wk.transpose(-2, -1).flip(2)  # For backward pass

        # 6. Chunked Processing (for long sequences)
        state = torch.zeros(B, H, N, N, device=x.device, dtype=x.dtype)
        y = torch.zeros(B, H, T, N, device=x.device, dtype=x.dtype)
        
        chunk_size = 256  # Optimized for GPU memory
        for i in range((T + chunk_size - 1) // chunk_size):
            s = i * chunk_size
            e = min(s + chunk_size, T)
            
            rr = r[:, :, s:e, :]  # [B,H,chunk,N]
            kk = k[:, :, :, s:e]  # [B,H,N,chunk]
            vv = v[:, :, s:e, :]  # [B,H,chunk,N]
            
            # Core RWKV operation
            y[:, :, s:e, :] = ((rr @ kk) * w) @ vv + (rr @ state) * wb
            state = wk * state + (kk * wk) @ vv

        # 7. Output Processing
        y = y.transpose(1, 2).contiguous().view(B * T, C)  # [B*T,C]
        y = self.ln_x(y).view(B, T, C)                     # [B,T,C]
        return self.dropout(self.output(y))

# --------------------------
# 5. Channel Mixing Module 
# --------------------------
class RWKV_ChannelMix(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        
        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / config.n_layer)
            ddd = torch.ones(1, 1, config.n_embd)
            for i in range(config.n_embd):
                ddd[0, 0, i] = i / config.n_embd
                
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))

        self.key = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.value = nn.Linear(3 * config.n_embd, config.n_embd, bias=config.bias)
        self.receptance = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        xx = self.time_shift(x) - x
        xk = x + xx * self.time_maa_k
        xr = x + xx * self.time_maa_r

        x = self.key(xk)
        x = torch.relu(x) ** 2  # Squared ReLU
        x = self.value(x)
        x = torch.sigmoid(self.receptance(xr)) * x  # Gating
        return self.dropout(x)

# --------------------------
# 6. Block chính
# --------------------------
class Block(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.ln1 = LayerNorm(config.n_embd, config.bias)
        self.tmix = CryptoRWKV_TimeMix(config, layer_id)
        self.ln2 = LayerNorm(config.n_embd, config.bias)
        self.cmix = RWKV_ChannelMix(config, layer_id)

    def forward(self, x, volatility=None):
        x = x + self.tmix(self.ln1(x), volatility)
        x = x + self.cmix(self.ln2(x))
        return x

# --------------------------
# 7. Main Model
# --------------------------
class CryptoRWKV_TS(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.configs = ModelConfig(config)
        self.num_patches = (self.configs.seq_len - self.configs.patch_size) // (self.configs.patch_size // 2) + 1
        self.patching = Patching(
            patch_size=self.configs.patch_size,
            stride=self.configs.patch_size // 2
        )
        
        self.embedding = CryptoDataEmbedding(
            c_in=self.configs.enc_in, 
            d_model=self.configs.d_model,
            patch_size=self.configs.patch_size,
            lookback=config['model'].get('volatility_lookback', 11),
            dropout=self.configs.dropout
        )
        
        # RWKV Blocks
        rwkv_config = RWKVConfig(
            n_layer=self.configs.e_layers,
            n_head=self.configs.n_heads,
            n_embd=self.configs.d_model,
            dropout=self.configs.dropout
        )
        
        self.blocks = nn.ModuleList([
            Block(rwkv_config, i) 
            for i in range(rwkv_config.n_layer)
        ])
        
        # Predictor
        self.predictor = nn.Sequential(
            nn.Linear(self.configs.d_model, self.configs.d_ff),
            nn.GELU(),
            nn.Linear(self.configs.d_ff, self.configs.c_out * self.configs.patch_size),
            nn.Unflatten(-1, (self.configs.patch_size, self.configs.c_out))
        )
        
        # Technical Analysis
        self.ta_encoder = nn.Linear(7, self.configs.d_model)

    def compute_volatility(self, x):
        returns = x[:, :, 3:4].diff(dim=1).abs()
        return returns.unfold(1, 10, 1).std(dim=-1)

    def _overlap_add(self, patches: torch.Tensor) -> torch.Tensor:
        # Thêm reshape để đảm bảo đúng 4 chiều
        B, T, _ = patches.shape
        patches = patches.reshape(B, T, self.configs.patch_size, self.configs.c_out)
        
        stride = self.configs.patch_size // 2
        output_len = (T - 1) * stride + self.configs.patch_size
        
        output = torch.zeros(B, output_len, self.configs.c_out, device=patches.device)
        count = torch.zeros(B, output_len, 1, device=patches.device)
        
        for i in range(T):
            start = i * stride
            end = start + self.configs.patch_size
            output[:, start:end] += patches[:, i]
            count[:, start:end] += 1
        
        return output / (count + 1e-6)

    def forward(self, x_enc: torch.Tensor, x_mark_enc=None) -> torch.Tensor:
        B, L, M = x_enc.shape
        assert M == self.configs.enc_in, f"Input features {M} không khớp với config enc_in {self.configs.enc_in}"
        
        # Normalization
        median = x_enc.median(dim=1, keepdim=True).values
        mad = torch.median(torch.abs(x_enc - median), dim=1, keepdim=True).values
        x_norm = (x_enc - median) / (mad + 1e-6)
        
        # Patching
        x_patched = self.patching(x_norm)  # [B, num_patches, patch_size * M]
        volatility = self.compute_volatility(x_patched)  # Tính toán volatility trực tiếp
        
        # Embedding
        x = self.embedding(x_patched, x_mark_enc)
        
        # Technical Analysis (nếu có đủ features)
        if M > 5:
            ta_features = self.ta_encoder(x_enc[:, :, -5:])
            x = x + ta_features[:, :self.num_patches]  # Cắt cho khớp số patches
        
        # RWKV Blocks
        for block in self.blocks:
            x = block(x, volatility)
        
        # Prediction
        pred_patches = self.predictor(x)  # [B, num_patches, patch_size * c_out]
        pred = self._overlap_add(pred_patches)
        
        # Denormalize (chỉ cho feature thứ 3 - index 2)
        return pred[:, -self.configs.pred_len:] * mad[:, :, 2:3] + median[:, :, 2:3]

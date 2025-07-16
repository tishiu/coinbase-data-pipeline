import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Any

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class LightAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B, T, _ = x.shape
        Q = self.query(x)  # (B, T, d_model)
        K = self.key(x)    # (B, T, d_model)
        
        # Attention đơn giản
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.d_model)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Kết hợp residual
        out = x + self.gamma * torch.bmm(attn, x)
        return out

class LSTMAttentionHybrid(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        model_cfg = config['model']
        self.d_model = d_model = model_cfg['d_model']
        self.pred_len = model_cfg['pred_len']
        self.out_dim = model_cfg.get('output_dim', 1)
        
        # Phần đầu vào
        self.input_proj = nn.Sequential(
            nn.Linear(model_cfg['enc_in'], d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(0.1),
            nn.GELU()
        )
        
        # LSTM layers
        self.lstm1 = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model//2,  
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        self.lstm2 = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True
        )
        
        # Attention và gate điều khiển
        self.attention = LightAttention(d_model)
        self.attention_gate = nn.Sequential(
            nn.Linear(2*d_model, d_model),
            nn.Sigmoid()
        )
        
        # Phần đầu ra
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, self.pred_len * self.out_dim)
        )
        
        self.pos_encoder = PositionalEncoding(d_model)

    def forward(self, x, time_features=None):
        # Phần đầu vào
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        
        # LSTM tầng thấp (bidirectional)
        lstm1_out, _ = self.lstm1(x)
        lstm1_out = lstm1_out[:, :, :self.d_model//2] + lstm1_out[:, :, self.d_model//2:]  # Combine bidirectional
        
        # Ghép với đầu vào ban đầu
        lstm1_out = torch.cat([x, lstm1_out], dim=-1)
        
        # Attention có gate điều khiển
        gate = self.attention_gate(lstm1_out)
        attn_out = self.attention(lstm1_out * gate)
        
        # LSTM tầng cao
        lstm2_out, _ = self.lstm2(attn_out)
        
        # Pooling và đầu ra
        context = lstm2_out.mean(dim=1)  # Global average pooling
        output = self.output_proj(context)
        
        return output.view(-1, self.pred_len, self.out_dim)
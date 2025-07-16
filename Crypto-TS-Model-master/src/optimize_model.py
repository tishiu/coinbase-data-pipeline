import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultiScaleTemporalAttention(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.1, scales=[1, 3, 6]):
        super().__init__()
        self.scales = scales
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Các lớp attention cho từng scale
        self.attention_heads = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True
            ) for _ in scales
        ])
        
        # Lớp downsample cho các scale lớn hơn 1
        self.downsamplers = nn.ModuleList([
            nn.Sequential(
                nn.AvgPool1d(kernel_size=scale, stride=scale),
                nn.Conv1d(d_model, d_model, kernel_size=1)
            ) if scale > 1 else nn.Identity()
            for scale in scales
        ])
        
        # Lớp tổng hợp
        self.aggregate = nn.Sequential(
            nn.Linear(d_model * len(scales), d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.GELU()
        )
        
    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]
        """
        B, T, _ = x.shape
        outputs = []
        
        for scale, attention, downsample in zip(self.scales, self.attention_heads, self.downsamplers):
            # Xử lý theo từng scale
            if scale > 1:
                # Downsample sequence
                x_down = downsample(x.transpose(1, 2)).transpose(1, 2)
            else:
                x_down = x
                
            # Self-attention
            attn_out, _ = attention(x_down, x_down, x_down)
            
            # Upsample trở lại độ dài ban đầu nếu cần
            if scale > 1:
                attn_out = F.interpolate(attn_out.permute(0, 2, 1), size=T).permute(0, 2, 1)
            
            outputs.append(attn_out)
        
        # Kết hợp các kết quả từ nhiều scale
        combined = torch.cat(outputs, dim=-1)
        return self.aggregate(combined)
    
class OptimizedLSTMAttentionModel(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        model_cfg = config['model']
        d_model = model_cfg['d_model']
        enc_in = model_cfg['enc_in']
        self.pred_len = model_cfg['pred_len']
        self.out_dim = model_cfg.get('output_dim', 1)
        self.dropout_rate = model_cfg.get('dropout', 0.3)

        input_dim = enc_in 

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(self.dropout_rate),
            nn.GELU()
        )

        self.pos_encoder = PositionalEncoding(d_model)

        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=2,
            batch_first=True,
            dropout=self.dropout_rate
        )

        self.temporal_attention = MultiScaleTemporalAttention(
            d_model=d_model,
            n_heads=config['model'].get('n_heads', 4),
            dropout=self.dropout_rate,
            scales=[1, 3, 6]  # scale
        )

        self.skip_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU()
        )
        
        self.attn_norm = nn.LayerNorm(d_model)
        self.lstm_norm = nn.LayerNorm(d_model)
        self.residual_norm = nn.LayerNorm(d_model)
        self.pool_norm = nn.LayerNorm(d_model)
        
        # residual connection
        self.residual = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(self.dropout_rate)
        )
        
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model*2),
            nn.LayerNorm(d_model*2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(d_model*2, self.pred_len * self.out_dim)
        )

    def attention_weighted_pooling(self, x):
        attn_weights = F.softmax(x.mean(dim=-1), dim=1)  # [B, T]
        attn_weights = F.dropout(attn_weights, p=self.dropout_rate, training=self.training)
        pooled = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)
        return pooled
    
    def forward(self, x, time_features=None):
        B, T, _ = x.shape
        
        # Input projection
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        
        # LSTM + Norm
        lstm_out, _ = self.lstm(x)
        lstm_out = self.lstm_norm(lstm_out)  # add norm
        
        # Multi-scale temporal attention
        attn_out = self.temporal_attention(lstm_out)
        
        # Skip connection từ đầu vào LSTM
        skip = self.skip_conv(lstm_out.permute(0, 2, 1)).permute(0, 2, 1)
        skip = self.pool_norm(skip)  

        attn_out = attn_out + skip
        
        # Attention pooling 
        context = self.attention_weighted_pooling(attn_out)
        
        # Residual + Norm
        context = context + self.residual(context)
        context = self.residual_norm(context)  # add norm
        
        # Output
        out = self.output_proj(context)
        return out.view(-1, self.pred_len, self.out_dim)

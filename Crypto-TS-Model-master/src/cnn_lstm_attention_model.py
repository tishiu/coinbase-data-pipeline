import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

class CNNLSTMAttentionModel(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        model_cfg = config['model']

        self.input_dim = model_cfg['enc_in']
        self.seq_len = model_cfg['seq_len']
        self.pred_len = model_cfg['pred_len']
        self.cnn_out_channels = model_cfg.get('cnn_out_channels', 64)
        self.lstm_hidden_dim = model_cfg.get('d_model', 128)
        self.attention_dim = model_cfg.get('attn_dim', 64)
        self.dropout = model_cfg.get('dropout', 0.3)

        # CNN module
        self.cnn = nn.Sequential(
            nn.Conv1d(self.input_dim, self.cnn_out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(self.dropout)
        )

        # LSTM module
        self.lstm = nn.LSTM(
            input_size=self.cnn_out_channels,
            hidden_size=self.lstm_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        # Attention module
        self.attn_fc = nn.Linear(self.lstm_hidden_dim, self.attention_dim)
        self.attn_score = nn.Linear(self.attention_dim, 1, bias=False)

        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(self.lstm_hidden_dim, self.lstm_hidden_dim),
            nn.LayerNorm(self.lstm_hidden_dim),
            nn.GELU(),
            nn.Linear(self.lstm_hidden_dim, self.pred_len)
        )

    def attention_layer(self, lstm_out):
        score = torch.tanh(self.attn_fc(lstm_out))
        score = self.attn_score(score).squeeze(-1)
        attn_weights = F.softmax(score, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        return context

    def forward(self, x_enc: torch.Tensor, x_mark_enc=None):
        x = x_enc.permute(0, 2, 1)  # [batch, input_dim, seq_len]
        x = self.cnn(x)             # [batch, cnn_out, seq_len//2]
        x = x.permute(0, 2, 1)      # [batch, seq_len//2, cnn_out]
        lstm_out, _ = self.lstm(x)
        context = self.attention_layer(lstm_out)
        out = self.output_layer(context)
        return out.unsqueeze(-1)

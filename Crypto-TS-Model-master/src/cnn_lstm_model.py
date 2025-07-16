import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

class CNNLSTMModel(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        model_cfg = config['model']

        self.input_dim = model_cfg['enc_in']
        self.seq_len = model_cfg['seq_len']
        self.pred_len = model_cfg['pred_len']
        self.cnn_out_channels = model_cfg.get('cnn_out_channels', 64)
        self.lstm_hidden_dim = model_cfg.get('d_model', 128)
        self.dropout = model_cfg.get('dropout', 0.3)

        # CNN module
        self.cnn = nn.Sequential(
            nn.Conv1d(self.input_dim, self.cnn_out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
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

        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(self.lstm_hidden_dim, self.lstm_hidden_dim),
            nn.LayerNorm(self.lstm_hidden_dim),
            nn.GELU(),
            nn.Linear(self.lstm_hidden_dim, self.pred_len)
        )

    def forward(self, x_enc: torch.Tensor, x_mark_enc=None):
        # x_enc: [batch, seq_len, input_dim]
        x = x_enc.permute(0, 2, 1)  # [batch, input_dim, seq_len]
        x = self.cnn(x)             # [batch, cnn_out_channels, seq_len]
        x = x.permute(0, 2, 1)      # [batch, seq_len, cnn_out_channels]
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, lstm_hidden_dim]
        last_output = lstm_out[:, -1, :]   # Lấy output cuối cùng
        out = self.output_layer(last_output)       # [batch, pred_len]
        return out.unsqueeze(-1)                    # [batch, pred_len, 1]

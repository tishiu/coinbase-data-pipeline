import torch
import torch.nn as nn
from typing import Dict, Any

class LSTMModel(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        model_cfg = config['model']

        self.input_dim = model_cfg['enc_in']
        self.seq_len = model_cfg['seq_len']
        self.pred_len = model_cfg['pred_len']
        self.hidden_dim = model_cfg.get('d_model', 128)
        self.dropout = model_cfg.get('dropout', 0.3)

        self.input_net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(self.dropout)
        )

        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.output_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.pred_len)
        )

    def forward(self, x_enc: torch.Tensor, x_mark_enc=None):
        batch_size, seq_len, _ = x_enc.shape
        x = self.input_net(x_enc.view(-1, self.input_dim))
        x = x.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        out = self.output_net(last_output)
        return out.unsqueeze(-1)

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Union, Dict, Optional

class CryptoMetrics:
    """Class tính toán các metrics đặc thù cho crypto trading"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.vol_window = config.get('metrics', {}).get('volatility_window', 10) if config else 10
        
    @staticmethod
    def smape(y_true: Union[torch.Tensor, np.ndarray], 
              y_pred: Union[torch.Tensor, np.ndarray]) -> float:
        """Tính SMAPE với epsilon tránh chia 0"""
        if isinstance(y_true, torch.Tensor):
            y_true, y_pred = y_true.cpu().numpy(), y_pred.cpu().numpy()
        return 200 * np.mean(
            np.abs(y_pred - y_true) / 
            (np.abs(y_true) + np.abs(y_pred) + 1e-8
        ))

    def directional_accuracy(self, 
                          y_true: Union[torch.Tensor, np.ndarray],
                          y_pred: Union[torch.Tensor, np.ndarray]) -> float:
        """Tính độ chính xác hướng dự đoán"""
        if isinstance(y_true, torch.Tensor):
            y_true, y_pred = y_true.cpu().numpy(), y_pred.cpu().numpy()
        true_dir = np.sign(y_true[..., 1:] - y_true[..., :-1])
        pred_dir = np.sign(y_pred[..., 1:] - y_pred[..., :-1])
        return np.mean(true_dir == pred_dir) * 100

    def volatility_rmse(self,
                      y_true: Union[torch.Tensor, np.ndarray],
                      y_pred: Union[torch.Tensor, np.ndarray]) -> float:
        """RMSE của biến động giá"""
        if isinstance(y_true, torch.Tensor):
            y_true, y_pred = y_true.cpu().numpy(), y_pred.cpu().numpy()
        
        true_vol = self._rolling_volatility(y_true)
        pred_vol = self._rolling_volatility(y_pred)
        return np.sqrt(mean_squared_error(true_vol, pred_vol))

    def _rolling_volatility(self, data: np.ndarray) -> np.ndarray:
        """Tính biến động rolling"""
        returns = np.diff(data, axis=-1)
        return np.sqrt(
            np.convolve(returns**2, np.ones(self.vol_window)/self.vol_window, 'valid')
        )

    def evaluate_all(self,
                   y_true: Union[torch.Tensor, np.ndarray],
                   y_pred: Union[torch.Tensor, np.ndarray]) -> Dict[str, float]:
        """Tính toán tất cả metrics"""
        return {
            'smape': self.smape(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'vol_rmse': self.volatility_rmse(y_true, y_pred),
            'dir_acc': self.directional_accuracy(y_true, y_pred)
        }

    def plot_predictions(self,
                       y_true: Union[torch.Tensor, np.ndarray],
                       y_pred: Union[torch.Tensor, np.ndarray],
                       title: str = "Predictions") -> None:
        """Visualize kết quả dự đoán"""
        if isinstance(y_true, torch.Tensor):
            y_true, y_pred = y_true.cpu().numpy(), y_pred.cpu().numpy()
        
        plt.figure(figsize=(12, 6))
        plt.plot(y_true[0], label='Actual')
        plt.plot(y_pred[0], label='Predicted')
        plt.title(f"{title} | SMAPE: {self.smape(y_true, y_pred):.2f}%")
        plt.legend()
        plt.show()

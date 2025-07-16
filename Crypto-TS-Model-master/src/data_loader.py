import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler, MinMaxScaler
import ta
from typing import Optional, Dict, Any, Tuple
import yaml
from pathlib import Path
from sklearn.base import TransformerMixin
import torch.nn.functional as F
import random

class CryptoDataset(Dataset):
    def __init__(self, 
                 data_path: str, 
                 config: Dict[str, Any],
                 train: bool = True,
                 scalers: Optional[Dict[str, TransformerMixin]] = None,
                 test_mode: bool = False):
        self.seq_len = config['model']['seq_len']
        self.pred_len = config['model']['pred_len']
        self.freq = config['data']['freq']
        self.train = train
        self.test_mode = test_mode
        self.current_epoch = 0
        self.max_epoch = 100 
        
        # Tải và tiền xử lý dữ liệu
        raw_df = self._load_and_clean(data_path)
        self.data = self._enhance_crypto_features(raw_df, self.freq)
        
        # Khởi tạo scalers
        self.scalers = scalers if scalers is not None else {
            'price': RobustScaler(),
            'volume': RobustScaler(),
            'indicators': MinMaxScaler(feature_range=(-1, 1)),
            'time': MinMaxScaler()
        }
        
        # Fit và transform dữ liệu
        if self.train:
            self._fit_scalers()
        self._scale_data()

    def _load_and_clean(self, path: str) -> pd.DataFrame:
        """Tải và làm sạch dữ liệu thô"""
        df = pd.read_csv(path, float_precision='high')
        
        # Chuẩn hóa tên cột
        column_map = {
            'timestamp': 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }
        df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})
        
        # Xử lý timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        
        # Xử lý missing values và outliers
        df['volume'] = df['volume'].replace(0, np.nan)
        df['volume'] = df['volume'].fillna(df['volume'].rolling(12, min_periods=1).median())
        
        # Clip outliers cho các cột quan trọng
        for col in ['close', 'volume']:
            q1 = df[col].quantile(0.01)
            q3 = df[col].quantile(0.99)
            df[col] = df[col].clip(lower=q1, upper=q3)
            
        return df.replace([np.inf, -np.inf], np.nan).ffill().bfill()

    def _enhance_crypto_features(self, df: pd.DataFrame, freq: str) -> pd.DataFrame:

        # Resample nếu tần số khác 5 phút
        if freq != '5T':
            ohlc_dict = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }
            df = df.resample(freq).apply(ohlc_dict).dropna()
        
        # 1. Features giá và lợi nhuận
        df['log_returns'] = np.log1p(df['close'].pct_change())
        df['price_ma_ratio'] = df['close'] / df['close'].rolling(24, min_periods=1).mean()
        df['price_spread'] = (df['high'] - df['low']) / df['close']
        
        # 2. Features volume
        df['volume_zscore'] = (df['volume'] - df['volume'].rolling(24).mean()) / df['volume'].rolling(24).std()
        df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(24, min_periods=1).mean()
        df['liquidity'] = np.log1p(df['volume'] * df['close'])
        
        # 3. Technical indicators
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        df['macd'] = ta.trend.MACD(df['close'], window_slow=26, window_fast=12).macd_diff()
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        
        # 4. Volatility features
        for window in [6, 12, 24]:  # 30 phút, 1 giờ, 2 giờ
            df[f'volatility_{window}'] = df['log_returns'].rolling(window).std()
        
        # 5. Momentum features
        df['momentum_3_6'] = df['close'].rolling(3).mean() - df['close'].rolling(6).mean()
        df['momentum_6_12'] = df['close'].rolling(6).mean() - df['close'].rolling(12).mean()
        
        # 6. Time features
        df['hour'] = df.index.hour
        df['dayofweek'] = df.index.dayofweek
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        df['is_market_open'] = ((df.index.hour >= 8) & (df.index.hour < 20)).astype(int)
        
        # 7. Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        
        return df.dropna()

    def _fit_scalers(self):

        if not self.train:
            return
        
        price_cols = ['open', 'high', 'low', 'close', 'price_ma_ratio', 'price_spread']
        volume_cols = ['volume', 'volume_zscore', 'volume_ma_ratio', 'liquidity']
        indicator_cols = ['rsi', 'macd', 'atr', 'obv', 'log_returns'] + \
                        [f'volatility_{w}' for w in [6, 12, 24]] + \
                        ['momentum_3_6', 'momentum_6_12']
        time_cols = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend', 'is_market_open']
        
        self.scalers['price'].fit(self.data[price_cols])
        self.scalers['volume'].fit(self.data[volume_cols])
        self.scalers['indicators'].fit(self.data[indicator_cols])
        self.scalers['time'].fit(self.data[time_cols])

    def _scale_data(self):

        price_cols = ['open', 'high', 'low', 'close', 'price_ma_ratio', 'price_spread']
        volume_cols = ['volume', 'volume_zscore', 'volume_ma_ratio', 'liquidity']
        indicator_cols = ['rsi', 'macd', 'atr', 'obv', 'log_returns'] + \
                        [f'volatility_{w}' for w in [6, 12, 24]] + \
                        ['momentum_3_6', 'momentum_6_12']
        time_cols = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend', 'is_market_open']
        
        scaled_data = pd.DataFrame(index=self.data.index)
        scaled_data[price_cols] = self.scalers['price'].transform(self.data[price_cols])
        scaled_data[volume_cols] = self.scalers['volume'].transform(self.data[volume_cols])
        scaled_data[indicator_cols] = self.scalers['indicators'].transform(self.data[indicator_cols])
        scaled_data[time_cols] = self.scalers['time'].transform(self.data[time_cols])
        
        self.scaled_data = scaled_data.values
        self.feature_names = price_cols + volume_cols + indicator_cols + time_cols
        
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx: int) -> dict:
        start_idx = idx
        end_idx = idx + self.seq_len
        pred_end_idx = end_idx + self.pred_len

        x = self.scaled_data[start_idx:end_idx].copy()
        y = self.scaled_data[end_idx:pred_end_idx, self.feature_names.index('close')]

        if self.train and not self.test_mode:
            # 1. Epoch tracking
            current_epoch = getattr(self, 'current_epoch', 0)
            max_epoch = getattr(self, 'max_epoch', 100)
            progress = min(1.0, current_epoch / (max_epoch * 0.5))
            noise_level = 0.02 * progress
            mask_ratio = 0.15 * progress

            # 2. Local Mean Masking
            if random.random() > 0.5:
                mask_length = max(1, int(self.seq_len * mask_ratio))
                mask_start = random.randint(0, max(0, self.seq_len - mask_length - 1))
                window_start = max(0, mask_start - 5)
                window_end = min(self.seq_len, mask_start + mask_length + 5)
                local_mean = np.mean(x[window_start:window_end], axis=0, keepdims=True)
                x[mask_start:mask_start + mask_length] = local_mean

            # 3. Adaptive Noise (có clip std để tránh NaN)
            feature_stds = np.clip(np.std(x, axis=0, keepdims=True), 1e-6, None)
            noise = np.random.normal(0, noise_level * feature_stds, size=x.shape)
            x = np.clip(x + noise, -3, 3)

            # 4. Smart Scaling
            if random.random() > 0.5:
                non_close_features = [i for i, name in enumerate(self.feature_names) if name != 'close']
                if non_close_features:
                    scale_factors = np.random.uniform(0.9, 1.1, size=(1, len(non_close_features)))
                    x[:, non_close_features] *= scale_factors

            # 5. Time Warping (F.interpolate)
            if random.random() > 0.7:
                x_tensor = torch.FloatTensor(x).unsqueeze(0)  # [1, T, D]
                x_tensor = x_tensor.permute(0, 2, 1)  # [1, D, T]
                warp_factor = random.uniform(0.8, 1.2)
                x_tensor = F.interpolate(x_tensor, scale_factor=warp_factor, mode='linear', align_corners=False)
                x_tensor = x_tensor.permute(0, 2, 1).squeeze(0)  # [new_T, D]
                x = x_tensor.numpy()
                # Pad hoặc cắt lại đúng self.seq_len
                if x.shape[0] >= self.seq_len:
                    x = x[:self.seq_len]
                else:
                    pad_len = self.seq_len - x.shape[0]
                    x = np.pad(x, ((0, pad_len), (0, 0)), mode='edge')

            # 6. Feature Dropout
            if random.random() > 0.5:
                drop_mask = (np.random.rand(x.shape[1]) > 0.1).astype(np.float32)  # numpy version
                x *= drop_mask  # broadcasted multiply

        return {
            'x': torch.FloatTensor(x),
            'y': torch.FloatTensor(y).unsqueeze(-1)
        }
                
    def set_epoch(self, epoch, max_epoch):
        self.current_epoch = epoch
        self.max_epoch = max_epoch
        
    @classmethod
    def from_cassandra_rows(cls, rows: list, config: Dict[str, Any], scalers: Optional[Dict] = None):
        """Tạo dataset từ dữ liệu real-time"""
        data = {
            'timestamp': [row.timestamp for row in rows],
            'open': [float(row.open) for row in rows],
            'high': [float(row.high) for row in rows],
            'low': [float(row.low) for row in rows],
            'close': [float(row.close) for row in rows],
            'volume': [float(row.volume) for row in rows]
        }
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        
        instance = cls.__new__(cls)
        instance.config = config
        instance.seq_len = config['model']['seq_len']
        instance.pred_len = config['model']['pred_len']
        instance.freq = config['data']['freq']
        instance.train = False
        instance.test_mode = True
        
        instance.data = instance._enhance_crypto_features(df, instance.freq)
        instance.scalers = scalers if scalers else {
            'price': RobustScaler(),
            'volume': RobustScaler(),
            'indicators': MinMaxScaler(feature_range=(-1, 1)),
            'time': MinMaxScaler()
        }
        instance._fit_scalers()
        instance._scale_data()
        
        return instance

class CryptoDataLoader:
    def __init__(self, config_path: str = None, data_path: Optional[str] = None):
        config_path = config_path or str(Path(__file__).parent.parent / "configs/train_config.yaml")
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        if data_path:
            self.config['data']['path'] = data_path
            
        # Load full dataset
        full_data = CryptoDataset(
            data_path=self.config['data']['path'],
            config=self.config,
            train=True
        )
        
        # Chia dữ liệu theo thời gian 
        split_idx = int(len(full_data) * self.config['data']['train_ratio'])
        train_idx = np.arange(split_idx)
        test_idx = np.arange(split_idx, len(full_data))
        
        self.train_data = SubsetWithAttributes(full_data, train_idx)
        self.test_data = SubsetWithAttributes(full_data, test_idx)
        
        # Lưu scalers và feature names
        self.scalers = full_data.scalers
        self.feature_names = full_data.feature_names
        
        # Tạo data loaders
        self.batch_size = self.config['training']['batch_size']
        self.train_loader = self._create_loader(self.train_data, shuffle=True)
        self.test_loader = self._create_loader(self.test_data, shuffle=False)
    
    def _create_loader(self, dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=4, 
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True,
            drop_last=True
        )
        
class SubsetWithAttributes(torch.utils.data.Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        
        for attr in dir(dataset):
            if not attr.startswith('_') and attr != 'indices':
                setattr(self, attr, getattr(dataset, attr))
    
    def set_epoch(self, epoch, max_epoch):
        self.dataset.set_epoch(epoch, max_epoch)

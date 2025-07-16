import logging
import torch
import numpy as np
import pandas as pd
import yaml
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
from sklearn.preprocessing import RobustScaler, MinMaxScaler
import sys
import os
from pathlib import Path

# Add model source path TRƯỚC KHI import
current_dir = Path(__file__).parent
model_src_path = current_dir.parent / 'model' / 'src'
sys.path.append(str(model_src_path))

# Import tất cả model classes
MODEL_CLASSES = {}
try:
    from lstm_model import LSTMModel
    MODEL_CLASSES['lstm'] = LSTMModel
    
    from lstm_attention_model import LSTMAttentionModel
    MODEL_CLASSES['lstm_attention'] = LSTMAttentionModel
    
    from optimize_model import OptimizedLSTMAttentionModel
    MODEL_CLASSES['optimize'] = OptimizedLSTMAttentionModel
    
    from cnn_lstm_model import CNNLSTMModel
    MODEL_CLASSES['cnn_lstm'] = CNNLSTMModel
    
    from cnn_lstm_attention_model import CNNLSTMAttentionModel
    MODEL_CLASSES['cnn_lstm_attention'] = CNNLSTMAttentionModel
    
    from lstm_attention_hybrid_model import LSTMAttentionHybrid
    MODEL_CLASSES['lstm_hybridattention'] = LSTMAttentionHybrid
    
    logger = logging.getLogger(__name__)
    logger.info(f"Successfully imported {len(MODEL_CLASSES)} model classes: {list(MODEL_CLASSES.keys())}")
    
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error(f"Error importing some model classes: {str(e)}")
    # At minimum, we need the basic LSTM model
    try:
        from lstm_model import LSTMModel
        MODEL_CLASSES['lstm'] = LSTMModel
        logger.info("Fallback: Only LSTM model available")
    except ImportError:
        logger.critical("Cannot import any model classes!")
        raise

class CryptoPricePredictor:
    """Class thực hiện dự đoán giá tiền điện tử - SỬ DỤNG CÙNG LOGIC VỚI NOTEBOOK 16.ipynb"""
    
    def __init__(self, 
                 model_config_path: str,
                 model_checkpoint_path: str,
                 device: str = None):
        
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Load config - GIỐNG NOTEBOOK 16
        with open(model_config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Validate config structure
        self._validate_config()
        
        # Load model - GIỐNG NOTEBOOK 16
        self.model = self._load_model(model_checkpoint_path)
        
        # Scalers sẽ được set từ bên ngoài (từ CryptoDataset)
        self.scalers = None
        
        logger.info(f"Predictor initialized on device: {self.device}")
        logger.info(f"Model type: {self.config['model'].get('model_type', 'lstm')}")
    
    def _validate_config(self):
        """Validate config structure"""
        required_fields = ['model']
        model_required = ['model_type', 'seq_len', 'pred_len', 'enc_in', 'd_model']
        
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required config field: {field}")
        
        for field in model_required:
            if field not in self.config['model']:
                raise ValueError(f"Missing required model config field: {field}")
        
        # Log config for debugging
        logger.info(f"Model config validation passed:")
        logger.info(f"  model_type: {self.config['model']['model_type']}")
        logger.info(f"  seq_len: {self.config['model']['seq_len']}")
        logger.info(f"  pred_len: {self.config['model']['pred_len']}")
        logger.info(f"  enc_in: {self.config['model']['enc_in']}")
        logger.info(f"  d_model: {self.config['model']['d_model']}")
    
    def _load_model(self, checkpoint_path: str):
        """Load model từ checkpoint - COPY TỪNG DÒNG TỪ NOTEBOOK 16"""
        try:
            # Get model type from config
            model_type = self.config['model'].get('model_type', 'lstm').lower()
            logger.info(f"Loading model type: {model_type}")
            
            # Select model class from imported classes
            if model_type in MODEL_CLASSES:
                model_class = MODEL_CLASSES[model_type]
                logger.info(f"Using model class: {model_class.__name__}")
            else:
                logger.warning(f"Model type '{model_type}' not found, using default LSTM")
                model_class = MODEL_CLASSES.get('lstm', LSTMModel)
            
            # Khởi tạo model - GIỐNG NOTEBOOK 16
            model = model_class(self.config).to(self.device)
            
            # Load weights với proper error handling
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Xử lý các định dạng checkpoint khác nhau - GIỐNG NOTEBOOK 16
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                logger.info("Loading from model_state_dict")
            else:
                state_dict = checkpoint
                logger.info("Loading from direct state dict")
            
            # Try loading with strict=True first, fallback to strict=False
            try:
                model.load_state_dict(state_dict, strict=True)
                logger.info("Model loaded with strict=True (exact match)")
            except RuntimeError as e:
                logger.warning(f"Strict loading failed: {str(e)[:200]}...")
                logger.info("Trying with strict=False (allow missing/extra keys)")
                
                # Load with strict=False
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                
                if missing_keys:
                    logger.warning(f"Missing keys: {len(missing_keys)} keys")
                    for key in missing_keys[:5]:  # Show first 5
                        logger.warning(f"  Missing: {key}")
                
                if unexpected_keys:
                    logger.warning(f"Unexpected keys: {len(unexpected_keys)} keys")
                    for key in unexpected_keys[:5]:  # Show first 5
                        logger.warning(f"  Unexpected: {key}")
                
                logger.info("Model loaded with strict=False - some parameters may be randomly initialized")
            
            model.eval()  # QUAN TRỌNG: Set eval mode
            
            logger.info(f"Model loaded successfully from {checkpoint_path}")
            logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.error(f"Model type attempted: {model_type}")
            logger.error(f"Available model types: {list(MODEL_CLASSES.keys())}")
            logger.error(f"Checkpoint path: {checkpoint_path}")
            raise
    
    def set_scalers(self, scalers: Dict):
        """Set scalers từ CryptoDataset"""
        self.scalers = scalers
        logger.info("Scalers set successfully")
    
    def predict(self, df_processed: pd.DataFrame, scalers: Dict = None) -> Tuple[np.ndarray, List[datetime]]:
        """
        Thực hiện dự đoán - LOGIC GIỐNG NOTEBOOK 16.ipynb
        
        Args:
            df_processed: DataFrame đã được xử lý bằng CryptoDataset (có tất cả features)
            scalers: Scalers từ CryptoDataset
            
        Returns:
            Tuple of (predictions, target_timestamps)
        """
        try:
            if scalers:
                self.scalers = scalers
                
            if self.scalers is None:
                raise ValueError("Scalers must be provided or set before prediction")
            
            seq_len = self.config['model']['seq_len']
            pred_len = self.config['model']['pred_len']
            
            if len(df_processed) < seq_len:
                raise ValueError(f"Insufficient data: need {seq_len}, got {len(df_processed)}")
            
            # Lấy sequence mới nhất
            recent_data = df_processed.tail(seq_len).copy()
            
            # Scale data - SỬ DỤNG CÙNG LOGIC VỚI CryptoDataset
            scaled_data = self._scale_data_like_dataset(recent_data)
            
            # Tạo tensor - GIỐNG NOTEBOOK 16
            x = torch.FloatTensor(scaled_data).unsqueeze(0).to(self.device)  # [1, seq_len, features]
            
            # Prediction - GIỐNG NOTEBOOK 16
            with torch.no_grad():
                pred = self.model(x)  # Model output shape tùy thuộc vào model type
                
                # Handle different output shapes
                if len(pred.shape) == 3:  # [batch, pred_len, 1]
                    pred = pred.squeeze(-1)  # [batch, pred_len]
                pred = pred.squeeze(0).cpu().numpy()  # [pred_len] hoặc scalar
                
                # Ensure pred is array
                if np.isscalar(pred):
                    pred = np.array([pred])
                elif len(pred.shape) == 0:
                    pred = np.array([pred.item()])
            
            # DENORMALIZE predictions về giá USD thực
            pred_denormalized = self._denormalize_predictions(pred, recent_data)
            
            # Tạo target timestamps
            last_timestamp = recent_data.index[-1]
            target_timestamps = []
            for i in range(len(pred_denormalized)):
                target_timestamps.append(last_timestamp + timedelta(minutes=5 * (i + 1)))
            
            logger.info(f"Prediction completed: {len(pred_denormalized)} values")
            logger.info(f"First prediction (scaled): {pred[0]:.6f}")
            logger.info(f"First prediction (USD): ${pred_denormalized[0]:.2f}")
            
            return pred_denormalized, target_timestamps
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise
    
    def _denormalize_predictions(self, scaled_predictions: np.ndarray, recent_data: pd.DataFrame) -> np.ndarray:
        """
        Denormalize predictions từ scaled values về USD thực
        """
        try:
            # Method 1: Sử dụng close price scaler để denormalize
            if 'price' in self.scalers and hasattr(self.scalers['price'], 'center_') and hasattr(self.scalers['price'], 'scale_'):
                # RobustScaler: X_scaled = (X - center_) / scale_
                # X = X_scaled * scale_ + center_
                
                # Lấy close price index trong price scaler
                price_cols = ['open', 'high', 'low', 'close', 'price_ma_ratio', 'price_spread']
                if 'close' in price_cols:
                    close_idx = price_cols.index('close')
                    if close_idx < len(self.scalers['price'].center_):
                        center = self.scalers['price'].center_[close_idx]
                        scale = self.scalers['price'].scale_[close_idx]
                        
                        denormalized = scaled_predictions * scale + center
                        logger.info(f"Denormalized using RobustScaler: center={center:.2f}, scale={scale:.2f}")
                        return denormalized
            
            # Method 2: Fallback - sử dụng recent price làm reference
            last_price = recent_data['close'].iloc[-1]
            
            # Giả định scaled prediction trong khoảng [-3, 3] tương ứng với ±20% price change
            price_change_factor = 0.2  # 20% max change
            max_scaled_value = 3.0
            
            # Convert scaled value to price change percentage
            price_change_pct = (scaled_predictions / max_scaled_value) * price_change_factor
            denormalized = last_price * (1 + price_change_pct)
            
            logger.info(f"Denormalized using fallback method: base_price=${last_price:.2f}")
            return denormalized
            
        except Exception as e:
            logger.error(f"Error denormalizing predictions: {str(e)}")
            # Last resort: return original scaled values
            return scaled_predictions
    
    def predict_single_step(self, df_processed: pd.DataFrame, scalers: Dict = None) -> Tuple[float, datetime]:
        """Dự đoán 1 bước tiếp theo"""
        pred, timestamps = self.predict(df_processed, scalers)
        return float(pred[0]), timestamps[0]
    
    def _scale_data_like_dataset(self, df: pd.DataFrame) -> np.ndarray:
        """
        Scale dữ liệu giống như trong CryptoDataset._scale_data()
        SỬ DỤNG LẠI LOGIC TỪ data_loader.py
        """
        try:
            # Feature groups - GIỐNG TRONG data_loader.py
            price_cols = ['open', 'high', 'low', 'close', 'price_ma_ratio', 'price_spread']
            volume_cols = ['volume', 'volume_zscore', 'volume_ma_ratio', 'liquidity']
            indicator_cols = ['rsi', 'macd', 'atr', 'obv', 'log_returns'] + \
                            [f'volatility_{w}' for w in [6, 12, 24]] + \
                            ['momentum_3_6', 'momentum_6_12']
            time_cols = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend', 'is_market_open']
            
            # Feature names theo thứ tự - GIỐNG data_loader.py
            feature_names = price_cols + volume_cols + indicator_cols + time_cols
            
            scaled_data = pd.DataFrame(index=df.index)
            
            # Scale từng nhóm - GIỐNG data_loader.py
            available_price_cols = [col for col in price_cols if col in df.columns]
            available_volume_cols = [col for col in volume_cols if col in df.columns]
            available_indicator_cols = [col for col in indicator_cols if col in df.columns]
            available_time_cols = [col for col in time_cols if col in df.columns]
            
            if available_price_cols:
                scaled_data[available_price_cols] = self.scalers['price'].transform(df[available_price_cols])
            if available_volume_cols:
                scaled_data[available_volume_cols] = self.scalers['volume'].transform(df[available_volume_cols])
            if available_indicator_cols:
                scaled_data[available_indicator_cols] = self.scalers['indicators'].transform(df[available_indicator_cols])
            if available_time_cols:
                scaled_data[available_time_cols] = self.scalers['time'].transform(df[available_time_cols])
            
            # Đảm bảo thứ tự features đúng - GIỐNG data_loader.py
            feature_data = []
            for feature_name in feature_names:
                if feature_name in scaled_data.columns:
                    feature_data.append(scaled_data[feature_name].values)
                else:
                    logger.warning(f"Missing feature: {feature_name}, filling with zeros")
                    feature_data.append(np.zeros(len(scaled_data)))
            
            result = np.column_stack(feature_data)
            logger.info(f"Scaled data shape: {result.shape}")
            return result
            
        except Exception as e:
            logger.error(f"Error scaling data: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Lấy thông tin về model"""
        return {
            'model_type': self.config['model'].get('model_type', 'lstm'),
            'seq_len': self.config['model']['seq_len'],
            'pred_len': self.config['model']['pred_len'],
            'device': str(self.device),
            'scalers_available': self.scalers is not None
        }
    
    def validate_input_data(self, df: pd.DataFrame) -> bool:
        """Kiểm tra tính hợp lệ của input data"""
        try:
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                return False
            
            # Kiểm tra NaN values
            if df[required_cols].isnull().any().any():
                logger.warning("Found NaN values in required columns")
                return False
            
            # Kiểm tra độ dài dữ liệu
            min_required = self.config['model']['seq_len']
            if len(df) < min_required:
                logger.error(f"Insufficient data: need {min_required}, got {len(df)}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating input data: {str(e)}")
            return False
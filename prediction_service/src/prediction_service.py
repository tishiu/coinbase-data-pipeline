import logging
import time
import threading
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path
import signal
import json

# Add model paths to sys.path
current_dir = Path(__file__).parent
model_src_path = current_dir.parent / 'model' / 'src'
sys.path.append(str(model_src_path))

from cassandra_client import CassandraClient
from data_fetcher import CryptoDataFetcher
from predictor import CryptoPricePredictor
from data_writer import PredictionDataWriter

# Configure logging
def setup_logging():
    """Setup logging with fallback paths"""
    log_paths = [
        '/app/logs/prediction_service.log',  # Docker path
        './logs/prediction_service.log',     # Local path
        './prediction_service.log',          # Fallback
        None  # Console only
    ]
    
    handlers = [logging.StreamHandler()]  # Always have console output
    
    # Try to add file handler
    for log_path in log_paths:
        if log_path is None:
            break
        try:
            # Create directory if it doesn't exist
            log_dir = Path(log_path).parent
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Add file handler
            file_handler = logging.FileHandler(log_path)
            handlers.append(file_handler)
            break
        except (OSError, PermissionError):
            continue
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

# Call setup_logging at module level
setup_logging()
logger = logging.getLogger(__name__)

class PredictionService:
    """Service chính để thực hiện dự đoán giá tiền điện tử theo thời gian thực"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.running = False
        self.prediction_threads = {}
        
        # Initialize components
        self._init_cassandra()
        self._init_predictor()
        self._init_data_components()
        
        # Service configuration
        self.product_ids = config.get('product_ids', ['BTC-USD', 'ETH-USD', 'XRP-USD'])
        self.prediction_interval = config.get('prediction_interval', 300)  # 5 minutes
        self.model_name = config.get('model_name', 'lstm_model_v1')
        
        # Cache scalers for each product
        self.product_scalers = {}
        
        # Health check
        self.last_prediction_times = {}
        self.prediction_errors = {}
        
        logger.info("PredictionService initialized successfully")
    
    def _init_cassandra(self):
        """Khởi tạo Cassandra client"""
        try:
            cassandra_hosts = self.config.get('cassandra_hosts', ['localhost'])
            if isinstance(cassandra_hosts, str):
                cassandra_hosts = [cassandra_hosts]
            
            cassandra_port = self.config.get('cassandra_port', 9042)
            keyspace = self.config.get('cassandra_keyspace', 'coinbase')
            
            self.cassandra_client = CassandraClient(
                hosts=cassandra_hosts,
                port=cassandra_port,
                keyspace=keyspace
            )
            
            logger.info("Cassandra client initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Cassandra client: {str(e)}")
            raise
    
    def _init_predictor(self):
        """Khởi tạo predictor"""
        try:
            model_config_path = self.config.get('model_config_path')
            model_checkpoint_path = self.config.get('model_checkpoint_path')
            device = self.config.get('device', 'cpu')
            
            if not model_config_path or not model_checkpoint_path:
                raise ValueError("Model config and checkpoint paths are required")
            
            self.predictor = CryptoPricePredictor(
                model_config_path=model_config_path,
                model_checkpoint_path=model_checkpoint_path,
                device=device
            )
            
            logger.info("Predictor initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize predictor: {str(e)}")
            raise
    
    def _init_data_components(self):
        """Khởi tạo data fetcher và writer"""
        try:
            self.data_fetcher = CryptoDataFetcher(self.cassandra_client)
            self.data_writer = PredictionDataWriter(self.cassandra_client)
            
            logger.info("Data components initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize data components: {str(e)}")
            raise
    
    def _get_or_create_scalers(self, product_id: str):
        """Lấy hoặc tạo scalers cho một product"""
        if product_id not in self.product_scalers:
            logger.info(f"Creating scalers for {product_id}")
            scalers = self.data_fetcher.create_scalers_from_historical_data(
                product_id=product_id,
                days_back=30
            )
            if scalers:
                self.product_scalers[product_id] = scalers
                logger.info(f"Scalers created successfully for {product_id}")
                return scalers
            else:
                logger.error(f"Failed to create scalers for {product_id}")
                return None
        
        return self.product_scalers[product_id]
    
    def predict_for_product(self, product_id: str) -> bool:
        """
        Thực hiện dự đoán cho một sản phẩm cụ thể - SỬ DỤNG CryptoDataset
        
        Args:
            product_id: ID của sản phẩm (VD: BTC-USD)
            
        Returns:
            bool: True nếu thành công
        """
        try:
            logger.info(f"Starting prediction for {product_id}")
            
            # Lấy hoặc tạo scalers cho product này
            scalers = self._get_or_create_scalers(product_id)
            if not scalers:
                logger.error(f"Cannot get scalers for {product_id} - skipping prediction")
                return False
            
            # Lấy dữ liệu mới nhất và xử lý bằng CryptoDataset
            df_processed = self.data_fetcher.get_latest_data_for_prediction(
                product_id=product_id,
                seq_len=self.predictor.config['model']['seq_len'],
                config=self.predictor.config,
                existing_scalers=scalers  # Pass scalers đã fit
            )
            
            if df_processed.empty:
                logger.warning(f"No processed data available for {product_id}")
                return False
            
            # Validate dữ liệu
            if not self.predictor.validate_input_data(df_processed):
                logger.error(f"Invalid input data for {product_id}")
                return False
            
            # Thực hiện dự đoán - GIỐNG NOTEBOOK 16
            predictions, target_timestamps = self.predictor.predict(df_processed, scalers)
            
            # Log both scaled and USD values for debugging
            logger.info(f"Predictions shape: {predictions.shape}")
            logger.info(f"First prediction value: {predictions[0]:.2f}")
            
            # Ghi kết quả vào database - SỬ DỤNG GIÁ TRỊ ĐÃ DENORMALIZED
            success = self.data_writer.write_predictions(
                product_id=product_id,
                model_name=self.model_name,
                predictions=predictions.tolist(),  # Đã là USD values
                target_timestamps=target_timestamps,
                confidence_intervals=None,  # Có thể thêm sau
                model_version=self.config.get('model_version', '1.0'),
                metadata=json.dumps({
                    'data_points_used': len(df_processed),
                    'last_price': float(df_processed['close'].iloc[-1]),
                    'prediction_method': 'dataset_reuse',
                    'features_count': df_processed.shape[1],
                    'denormalized': True  # Flag để biết đã denormalize
                })
            )
            
            if success:
                from datetime import timezone
                self.last_prediction_times[product_id] = datetime.now(timezone.utc).replace(tzinfo=None)
                if product_id in self.prediction_errors:
                    del self.prediction_errors[product_id]
                    del self.prediction_errors[product_id]
                
                logger.info(f"Successfully completed prediction for {product_id}")
                logger.info(f"Next price prediction: ${predictions[0]:.2f} at {target_timestamps[0]}")
                return True
            else:
                raise Exception("Failed to write predictions to database")
                
        except Exception as e:
            error_msg = f"Error predicting for {product_id}: {str(e)}"
            logger.error(error_msg)
            from datetime import timezone
            self.prediction_errors[product_id] = {
                'error': error_msg,
                'time': datetime.now(timezone.utc).replace(tzinfo=None)
            }
            return False
    
    def prediction_worker(self, product_id: str):
        """Worker thread cho việc dự đoán một sản phẩm"""
        logger.info(f"Starting prediction worker for {product_id}")
        
        retry_count = 0
        max_retries = 5
        base_delay = 30  # 30 seconds
        
        while self.running:
            try:
                # Thực hiện dự đoán
                success = self.predict_for_product(product_id)
                
                if success:
                    retry_count = 0  # Reset retry count on success
                    time.sleep(self.prediction_interval)
                else:
                    # Backoff strategy for failed predictions
                    retry_count += 1
                    if retry_count <= max_retries:
                        delay = base_delay * (2 ** (retry_count - 1))  # Exponential backoff
                        logger.warning(f"Prediction failed for {product_id}, retrying in {delay}s (attempt {retry_count}/{max_retries})")
                        time.sleep(delay)
                    else:
                        logger.error(f"Max retries reached for {product_id}, waiting for next cycle")
                        retry_count = 0
                        time.sleep(self.prediction_interval)
                
            except Exception as e:
                logger.error(f"Error in prediction worker for {product_id}: {str(e)}")
                time.sleep(60)  # Wait 1 minute on unexpected errors
    
    def start(self):
        """Bắt đầu service"""
        if self.running:
            logger.warning("Service is already running")
            return
        
        logger.info("Starting PredictionService...")
        self.running = True
        
        # Kiểm tra kết nối trước khi bắt đầu
        if not self._health_check():
            raise Exception("Health check failed - cannot start service")
        
        # Khởi tạo threads cho từng product
        for product_id in self.product_ids:
            thread = threading.Thread(
                target=self.prediction_worker,
                args=(product_id,),
                name=f"predictor-{product_id}",
                daemon=True
            )
            thread.start()
            self.prediction_threads[product_id] = thread
            
            # Stagger startup để tránh tải đồng thời
            time.sleep(5)
        
        logger.info(f"PredictionService started with {len(self.prediction_threads)} prediction threads")
    
    def stop(self):
        """Dừng service"""
        if not self.running:
            logger.warning("Service is not running")
            return
        
        logger.info("Stopping PredictionService...")
        self.running = False
        
        # Chờ các threads kết thúc
        for product_id, thread in self.prediction_threads.items():
            logger.info(f"Waiting for {product_id} prediction thread to stop...")
            thread.join(timeout=30)
        
        # Đóng kết nối Cassandra
        self.cassandra_client.close()
        
        logger.info("PredictionService stopped")
    
    def _health_check(self) -> bool:
        """Kiểm tra sức khỏe của service"""
        try:
            # Kiểm tra kết nối Cassandra
            if not self.cassandra_client.health_check():
                logger.error("Cassandra health check failed")
                return False
            
            # Kiểm tra model
            model_info = self.predictor.get_model_info()
            if not model_info:
                logger.error("Model health check failed")
                return False
            
            # Kiểm tra dữ liệu có sẵn
            available_products = self.data_fetcher.get_available_products()
            missing_products = [p for p in self.product_ids if p not in available_products]
            
            if missing_products:
                logger.warning(f"Some products not available in database: {missing_products}")
                # Không fail hoàn toàn, chỉ warning
            
            logger.info("Health check passed")
            return True
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Lấy trạng thái hiện tại của service"""
        return {
            'running': self.running,
            'product_ids': self.product_ids,
            'active_threads': len([t for t in self.prediction_threads.values() if t.is_alive()]),
            'last_prediction_times': {
                pid: time.isoformat() for pid, time in self.last_prediction_times.items()
            },
            'prediction_errors': {
                pid: {
                    'error': error['error'],
                    'time': error['time'].isoformat()
                } for pid, error in self.prediction_errors.items()
            },
            'model_info': self.predictor.get_model_info(),
            'cassandra_health': self.cassandra_client.health_check(),
            'scalers_cached': list(self.product_scalers.keys())
        }
    
    def force_prediction(self, product_id: Optional[str] = None) -> Dict[str, bool]:
        """Buộc thực hiện dự đoán ngay lập tức"""
        results = {}
        
        if product_id:
            if product_id in self.product_ids:
                results[product_id] = self.predict_for_product(product_id)
            else:
                logger.error(f"Product {product_id} not in configured products")
                results[product_id] = False
        else:
            # Predict for all products
            for pid in self.product_ids:
                results[pid] = self.predict_for_product(pid)
        
        return results

def create_service_from_env() -> PredictionService:
    """Tạo service từ environment variables"""
    config = {
        'cassandra_hosts': os.getenv('CASSANDRA_HOSTS', 'localhost').split(','),
        'cassandra_port': int(os.getenv('CASSANDRA_PORT', '9042')),
        'cassandra_keyspace': os.getenv('CASSANDRA_KEYSPACE', 'coinbase'),
        'product_ids': os.getenv('PRODUCT_IDS', 'BTC-USD,ETH-USD,XRP-USD').split(','),
        'prediction_interval': int(os.getenv('PREDICTION_INTERVAL', '300')),
        'model_config_path': os.getenv('MODEL_CONFIG_PATH'),
        'model_checkpoint_path': os.getenv('MODEL_CHECKPOINT_PATH'),
        'model_name': os.getenv('MODEL_NAME', 'lstm_model_v1'),
        'model_version': os.getenv('MODEL_VERSION', '1.0'),
        'device': os.getenv('DEVICE', 'cpu')
    }
    
    return PredictionService(config)

# Health check endpoint cho Docker
def health_check_endpoint():
    """Simple health check for Docker"""
    try:
        from flask import Flask, jsonify
        app = Flask(__name__)
        
        @app.route('/health')
        def health():
            from datetime import datetime, timezone
            return jsonify({'status': 'healthy', 'timestamp': datetime.now(timezone.utc).isoformat()})
        
        return app
    except ImportError:
        logger.warning("Flask not available for health check endpoint")
        return None

def main():
    """Main function để chạy service"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Crypto Price Prediction Service')
    parser.add_argument('--config', help='Config file path')
    parser.add_argument('--health-check', action='store_true', help='Enable health check endpoint')
    args = parser.parse_args()
    
    # Create service
    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        service = PredictionService(config)
    else:
        service = create_service_from_env()
    
    # Setup signal handling
    def signal_handler(signum, frame):
        logger.info("Received stop signal")
        service.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start health check endpoint if requested
        if args.health_check or os.getenv('ENABLE_HEALTH_CHECK', 'false').lower() == 'true':
            health_app = health_check_endpoint()
            if health_app:
                import threading
                from werkzeug.serving import run_simple
                
                def run_health_server():
                    run_simple('0.0.0.0', 8000, health_app, 
                             use_reloader=False, use_debugger=False, threaded=True)
                
                health_thread = threading.Thread(target=run_health_server, daemon=True)
                health_thread.start()
                logger.info("Health check endpoint started on port 8000")
        
        # Start prediction service
        service.start()
        
        # Keep main thread alive
        while service.running:
            time.sleep(10)
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Service error: {str(e)}")
    finally:
        service.stop()

if __name__ == "__main__":
    main()
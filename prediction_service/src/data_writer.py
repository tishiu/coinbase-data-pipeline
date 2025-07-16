import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from cassandra_client import CassandraClient
import time

logger = logging.getLogger(__name__)

class PredictionDataWriter:
    """Class để ghi kết quả dự đoán vào Cassandra"""
    
    def __init__(self, cassandra_client: CassandraClient):
        self.client = cassandra_client
        self.session = cassandra_client.get_session()
        
        # Tạo tables nếu chưa tồn tại
        self._create_prediction_tables()
        
        # Prepare statements
        self._prepare_statements()
    
    def _create_prediction_tables(self):
        """Tạo các bảng cần thiết để lưu predictions"""
        try:
            # Bảng predictions chính
            create_predictions_table = """
            CREATE TABLE IF NOT EXISTS predictions (
                product_id TEXT,
                model_name TEXT,
                prediction_time TIMESTAMP,
                target_time TIMESTAMP,
                predicted_price DOUBLE,
                confidence_lower DOUBLE,
                confidence_upper DOUBLE,
                model_version TEXT,
                metadata TEXT,
                PRIMARY KEY ((product_id, model_name), prediction_time, target_time)
            ) WITH CLUSTERING ORDER BY (prediction_time DESC, target_time ASC)
            """
            
            # Bảng predictions theo horizon để query dễ hơn
            create_predictions_by_horizon_table = """
            CREATE TABLE IF NOT EXISTS predictions_by_horizon (
                product_id TEXT,
                model_name TEXT,
                prediction_horizon INT,
                prediction_time TIMESTAMP,
                target_time TIMESTAMP,
                predicted_price DOUBLE,
                confidence_lower DOUBLE,
                confidence_upper DOUBLE,
                PRIMARY KEY ((product_id, model_name, prediction_horizon), prediction_time)
            ) WITH CLUSTERING ORDER BY (prediction_time DESC)
            """
            
            # Bảng để track model performance
            create_model_metrics_table = """
            CREATE TABLE IF NOT EXISTS model_metrics (
                product_id TEXT,
                model_name TEXT,
                evaluation_time TIMESTAMP,
                horizon INT,
                mae DOUBLE,
                rmse DOUBLE,
                mape DOUBLE,
                directional_accuracy DOUBLE,
                sample_count INT,
                PRIMARY KEY ((product_id, model_name), evaluation_time, horizon)
            ) WITH CLUSTERING ORDER BY (evaluation_time DESC, horizon ASC)
            """
            
            self.session.execute(create_predictions_table)
            self.session.execute(create_predictions_by_horizon_table)
            self.session.execute(create_model_metrics_table)
            
            logger.info("Prediction tables created/verified successfully")
            
        except Exception as e:
            logger.error(f"Error creating prediction tables: {str(e)}")
            raise
    
    def _prepare_statements(self):
        """Prepare các statements thường dùng"""
        try:
            # Insert prediction statement
            self.insert_prediction_stmt = self.session.prepare("""
                INSERT INTO predictions (
                    product_id, model_name, prediction_time, target_time, 
                    predicted_price, confidence_lower, confidence_upper, 
                    model_version, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """)
            
            # Insert prediction by horizon statement
            self.insert_prediction_by_horizon_stmt = self.session.prepare("""
                INSERT INTO predictions_by_horizon (
                    product_id, model_name, prediction_horizon, prediction_time, 
                    target_time, predicted_price, confidence_lower, confidence_upper
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """)
            
            # Insert model metrics statement
            self.insert_metrics_stmt = self.session.prepare("""
                INSERT INTO model_metrics (
                    product_id, model_name, evaluation_time, horizon,
                    mae, rmse, mape, directional_accuracy, sample_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """)
            
            # Query statements
            self.get_recent_predictions_stmt = self.session.prepare("""
                SELECT * FROM predictions 
                WHERE product_id = ? AND model_name = ? 
                AND prediction_time >= ?
                ORDER BY prediction_time DESC, target_time ASC
                LIMIT ?
            """)
            
        except Exception as e:
            logger.error(f"Error preparing statements: {str(e)}")
            raise
    
    def write_predictions(self,
                         product_id: str,
                         model_name: str,
                         predictions: List[float],
                         target_timestamps: List[datetime],
                         prediction_time: datetime = None,
                         confidence_intervals: Optional[Dict[str, List[float]]] = None,
                         model_version: str = "1.0",
                         metadata: str = "") -> bool:
        """
        Ghi predictions vào database
        
        Args:
            product_id: ID của sản phẩm (VD: BTC-USD)
            model_name: Tên model
            predictions: List các giá trị dự đoán
            target_timestamps: List thời gian tương ứng với dự đoán
            prediction_time: Thời gian thực hiện dự đoán
            confidence_intervals: Dict với 'lower' và 'upper' bounds
            model_version: Version của model
            metadata: Metadata bổ sung
            
        Returns:
            bool: True nếu thành công
        """
        try:
            if prediction_time is None:
                prediction_time = datetime.utcnow()
            
            if len(predictions) != len(target_timestamps):
                raise ValueError("Predictions and timestamps must have same length")
            
            # Prepare batch statements
            batch_statements = []
            
            for i, (pred_price, target_time) in enumerate(zip(predictions, target_timestamps)):
                # Default confidence intervals
                conf_lower = confidence_intervals['lower'][i] if confidence_intervals else pred_price * 0.95
                conf_upper = confidence_intervals['upper'][i] if confidence_intervals else pred_price * 1.05
                
                # Main predictions table
                batch_statements.append((
                    self.insert_prediction_stmt,
                    (product_id, model_name, prediction_time, target_time,
                     float(pred_price), float(conf_lower), float(conf_upper),
                     model_version, metadata)
                ))
                
                # Predictions by horizon table
                horizon = i + 1  # Horizon starts from 1
                batch_statements.append((
                    self.insert_prediction_by_horizon_stmt,
                    (product_id, model_name, horizon, prediction_time,
                     target_time, float(pred_price), float(conf_lower), float(conf_upper))
                ))
            
            # Execute batch
            self.client.execute_batch(batch_statements)
            
            logger.info(f"Successfully wrote {len(predictions)} predictions for {product_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error writing predictions: {str(e)}")
            return False
    
    def write_model_metrics(self,
                           product_id: str,
                           model_name: str,
                           metrics: Dict[str, Any],
                           evaluation_time: datetime = None) -> bool:
        """
        Ghi model performance metrics
        
        Args:
            product_id: ID sản phẩm
            model_name: Tên model
            metrics: Dict chứa các metrics
            evaluation_time: Thời gian đánh giá
            
        Returns:
            bool: True nếu thành công
        """
        try:
            if evaluation_time is None:
                evaluation_time = datetime.utcnow()
            
            # Extract metrics by horizon if available
            if 'horizon_metrics' in metrics:
                # Multi-horizon metrics
                for horizon, horizon_metrics in metrics['horizon_metrics'].items():
                    self.session.execute(
                        self.insert_metrics_stmt,
                        (product_id, model_name, evaluation_time, int(horizon),
                         float(horizon_metrics.get('mae', 0.0)),
                         float(horizon_metrics.get('rmse', 0.0)),
                         float(horizon_metrics.get('mape', 0.0)),
                         float(horizon_metrics.get('directional_accuracy', 0.0)),
                         int(horizon_metrics.get('sample_count', 0)))
                    )
            else:
                # Overall metrics (horizon = 0)
                self.session.execute(
                    self.insert_metrics_stmt,
                    (product_id, model_name, evaluation_time, 0,
                     float(metrics.get('mae', 0.0)),
                     float(metrics.get('rmse', 0.0)),
                     float(metrics.get('mape', 0.0)),
                     float(metrics.get('directional_accuracy', 0.0)),
                     int(metrics.get('sample_count', 0)))
                )
            
            logger.info(f"Successfully wrote metrics for {product_id} - {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error writing model metrics: {str(e)}")
            return False
    
    def get_recent_predictions(self,
                              product_id: str,
                              model_name: str,
                              hours_back: int = 24,
                              limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Lấy predictions gần đây
        
        Args:
            product_id: ID sản phẩm
            model_name: Tên model
            hours_back: Số giờ trước đó
            limit: Giới hạn số records
            
        Returns:
            List các prediction records
        """
        try:
            from datetime import datetime, timedelta
            
            start_time = datetime.utcnow() - timedelta(hours=hours_back)
            
            rows = self.session.execute(
                self.get_recent_predictions_stmt,
                (product_id, model_name, start_time, limit)
            )
            
            predictions = []
            for row in rows:
                predictions.append({
                    'product_id': row.product_id,
                    'model_name': row.model_name,
                    'prediction_time': row.prediction_time,
                    'target_time': row.target_time,
                    'predicted_price': row.predicted_price,
                    'confidence_lower': row.confidence_lower,
                    'confidence_upper': row.confidence_upper,
                    'model_version': row.model_version,
                    'metadata': row.metadata
                })
            
            logger.info(f"Retrieved {len(predictions)} recent predictions for {product_id}")
            return predictions
            
        except Exception as e:
            logger.error(f"Error retrieving recent predictions: {str(e)}")
            return []
    
    def cleanup_old_predictions(self, days_to_keep: int = 30) -> bool:
        """
        Xóa predictions cũ để tiết kiệm dung lượng
        
        Args:
            days_to_keep: Số ngày muốn giữ lại
            
        Returns:
            bool: True nếu thành công
        """
        try:
            from datetime import datetime, timedelta
            
            cutoff_time = datetime.utcnow() - timedelta(days=days_to_keep)
            
            # Note: Cassandra không hỗ trợ DELETE với time range trực tiếp
            # Cần implement logic phức tạp hơn hoặc sử dụng TTL
            logger.warning("Cleanup old predictions requires TTL setup or custom implementation")
            
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning up old predictions: {str(e)}")
            return False
    
    def get_prediction_stats(self, product_id: str, model_name: str) -> Dict[str, Any]:
        """Lấy thống kê về predictions"""
        try:
            query = """
                SELECT COUNT(*) as count, MIN(prediction_time) as earliest, MAX(prediction_time) as latest
                FROM predictions 
                WHERE product_id = ? AND model_name = ?
            """
            
            row = self.session.execute(query, (product_id, model_name)).one()
            
            return {
                'product_id': product_id,
                'model_name': model_name,
                'total_predictions': row.count,
                'earliest_prediction': row.earliest,
                'latest_prediction': row.latest
            }
            
        except Exception as e:
            logger.error(f"Error getting prediction stats: {str(e)}")
            return {}
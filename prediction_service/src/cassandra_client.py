import logging
from typing import List, Optional, Dict, Any
from cassandra.cluster import Cluster, Session
from cassandra.auth import PlainTextAuthProvider
from cassandra.policies import DCAwareRoundRobinPolicy
import time
import os

logger = logging.getLogger(__name__)

class CassandraClient:
    """Client để kết nối và thao tác với Cassandra"""
    
    def __init__(self, 
                 hosts: List[str] = None,
                 port: int = 9042,
                 keyspace: str = 'coinbase',
                 username: Optional[str] = None,
                 password: Optional[str] = None):
        
        self.hosts = hosts or [os.getenv('CASSANDRA_HOSTS', 'localhost')]
        if isinstance(self.hosts[0], str) and ',' in self.hosts[0]:
            self.hosts = self.hosts[0].split(',')
        
        self.port = port
        self.keyspace = keyspace
        self.cluster = None
        self.session = None
        
        # Authentication
        self.auth_provider = None
        if username and password:
            self.auth_provider = PlainTextAuthProvider(username=username, password=password)
        
        self._connect()
    
    def _connect(self, max_retries: int = 10, retry_delay: int = 5):
        """Kết nối đến Cassandra với retry logic"""
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempting to connect to Cassandra (attempt {attempt + 1}/{max_retries})")
                
                self.cluster = Cluster(
                    contact_points=self.hosts,
                    port=self.port,
                    auth_provider=self.auth_provider,
                    load_balancing_policy=DCAwareRoundRobinPolicy(),
                    connect_timeout=30,
                    control_connection_timeout=30
                )
                
                self.session = self.cluster.connect()
                
                # Test connection
                self.session.execute("SELECT release_version FROM system.local")
                
                # Set keyspace
                if self.keyspace:
                    self.session.set_keyspace(self.keyspace)
                
                logger.info(f"Successfully connected to Cassandra keyspace: {self.keyspace}")
                return
                
            except Exception as e:
                logger.warning(f"Connection attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    raise ConnectionError(f"Failed to connect to Cassandra after {max_retries} attempts")
    
    def execute_query(self, query: str, parameters: tuple = None) -> Any:
        """Thực thi query với error handling"""
        try:
            if parameters:
                return self.session.execute(query, parameters)
            return self.session.execute(query)
        except Exception as e:
            logger.error(f"Query execution failed: {query[:100]}... Error: {str(e)}")
            raise
    
    def execute_batch(self, statements: List[tuple]) -> None:
        """Thực thi batch statements"""
        from cassandra.query import BatchStatement
        try:
            batch = BatchStatement()
            for statement, params in statements:
                if isinstance(statement, str):
                    # If statement is string, prepare it first
                    prepared_stmt = self.session.prepare(statement)
                    batch.add(prepared_stmt, params)
                else:
                    # If statement is already prepared
                    batch.add(statement, params)
            
            self.session.execute(batch)
        except Exception as e:
            logger.error(f"Batch execution failed: {str(e)}")
            raise
    
    def prepare_statement(self, query: str):
        """Prepare statement để tăng hiệu suất"""
        try:
            return self.session.prepare(query)
        except Exception as e:
            logger.error(f"Statement preparation failed: {query[:100]}... Error: {str(e)}")
            raise
    
    def get_session(self) -> Session:
        """Lấy session để sử dụng trực tiếp"""
        if not self.session:
            self._connect()
        return self.session
    
    def close(self):
        """Đóng kết nối"""
        if self.cluster:
            self.cluster.shutdown()
            logger.info("Cassandra connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def health_check(self) -> bool:
        """Kiểm tra sức khỏe kết nối"""
        try:
            result = self.session.execute("SELECT release_version FROM system.local")
            return result is not None
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False
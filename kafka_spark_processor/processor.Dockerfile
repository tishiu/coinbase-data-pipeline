FROM apache/spark:3.5.5-scala2.12-java17-python3-r-ubuntu

USER root

# Install required tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    netcat \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create checkpoint directories
RUN mkdir -p /tmp/spark-ticker-checkpoint /tmp/spark-ticker-candles-checkpoint && \
    chmod 777 /tmp/spark-ticker-checkpoint /tmp/spark-ticker-candles-checkpoint

# Copy Spark processor application
COPY kafka_spark_processor/spark_processor.py .

# Tạo script khởi động tối ưu
RUN echo '#!/bin/bash \n\
\n\
export SPARK_HOME=${SPARK_HOME:-/opt/spark} \n\
export PATH=$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH \n\
\n\
echo "=== Thông tin cấu hình ===" \n\
echo "SPARK_HOME: $SPARK_HOME" \n\
echo "PATH: $PATH" \n\
echo "Kafka Host: ${KAFKA_HOST:-kafka}" \n\
echo "Kafka Port: ${KAFKA_PORT:-9092}" \n\
echo "Cassandra Host: ${CASSANDRA_HOST:-cassandra}" \n\
echo "Cassandra Port: ${CASSANDRA_PORT:-9042}" \n\
\n\
# Kiểm tra spark-submit \n\
SPARK_SUBMIT=$(which spark-submit || echo "$SPARK_HOME/bin/spark-submit") \n\
if [ ! -f "$SPARK_SUBMIT" ]; then \n\
  echo "CẢNH BÁO: Không tìm thấy spark-submit tại $SPARK_SUBMIT" \n\
  echo "Tìm spark-submit trong hệ thống..." \n\
  SPARK_SUBMIT=$(find / -name "spark-submit" -type f | head -1) \n\
  if [ -z "$SPARK_SUBMIT" ]; then \n\
    echo "LỖI: Không thể tìm thấy spark-submit ở bất kỳ đâu!" \n\
    exit 1 \n\
  fi \n\
fi \n\
echo "Sử dụng spark-submit tại: $SPARK_SUBMIT" \n\
\n\
# Kiểm tra kết nối Kafka \n\
echo "=== Kiểm tra kết nối Kafka ===" \n\
until nc -z ${KAFKA_HOST:-kafka} ${KAFKA_PORT:-9092} 2>/dev/null; do \n\
  echo "Đang đợi Kafka..." \n\
  sleep 5 \n\
done \n\
\n\
echo "=== Kiểm tra kết nối Cassandra ===" \n\
until nc -z ${CASSANDRA_HOST:-cassandra} ${CASSANDRA_PORT:-9042} 2>/dev/null; do \n\
  echo "Đang đợi Cassandra..." \n\
  sleep 5 \n\
done \n\
\n\
echo "=== Khởi động ứng dụng Spark ===" \n\
$SPARK_SUBMIT \\\n\
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.5,org.apache.kafka:kafka-clients:3.9.0,com.datastax.spark:spark-cassandra-connector_2.12:3.5.1 \\\n\
  --conf spark.cassandra.connection.host=${CASSANDRA_HOST:-cassandra} \\\n\
  --conf spark.cassandra.connection.port=${CASSANDRA_PORT:-9042} \\\n\
  --conf spark.sql.shuffle.partitions=10 \\\n\
  --conf spark.executor.memory=1g \\\n\
  --conf spark.driver.memory=1g \\\n\
  /app/spark_processor.py' > /app/start.sh && chmod +x /app/start.sh

CMD ["/app/start.sh"]
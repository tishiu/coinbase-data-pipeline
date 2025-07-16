FROM python:3.13-slim

WORKDIR /app

# Basic setup
RUN apt-get update && \
    apt-get install -y curl netcat-traditional dnsutils iputils-ping && \
    rm -rf /var/lib/apt/lists/*

# Copy source code
COPY coinbase_kafka_producer/producer.py .
COPY coinbase_kafka_producer/requirements.txt .

# Install python dependencies
RUN pip install -r requirements.txt

# Start script + logs
RUN echo '#!/bin/bash\n\
echo "=== Bắt đầu kiểm tra môi trường ==="\n\
echo "Ngày giờ hiện tại: $(date)"\n\
echo "Thư mục làm việc: $(pwd)"\n\
\n\
# Kiểm tra phiên bản Python\n\
echo "Phiên bản Python:"\n\
python --version 2>&1 || { echo "Lỗi: Python không được cài đặt hoặc không chạy được"; exit 1; }\n\
\n\
# Kiểm tra các thư viện Python cần thiết\n\
echo "Kiểm tra các thư viện Python..."\n\
pip list | grep -E "kafka-python|websocket-client|coinbase" || echo "Cảnh báo: Một số thư viện (kafka-python, websocket-client, coinbase) có thể chưa được cài đặt"\n\
echo "Danh sách tất cả thư viện đã cài đặt:"\n\
pip list\n\
\n\
# Kiểm tra sự tồn tại của producer.py\n\
echo "Kiểm tra file producer.py..."\n\
if [ -f /app/producer.py ]; then\n\
    echo "File producer.py tồn tại"\n\
    ls -l /app/producer.py\n\
else\n\
    echo "Lỗi: File producer.py không tồn tại trong /app"\n\
    exit 1\n\
fi\n\
\n\
# Kiểm tra biến môi trường\n\
echo "=== Kiểm tra biến môi trường ==="\n\
echo "BOOTSTRAP_SERVERS: ${BOOTSTRAP_SERVERS:-Chưa thiết lập}"\n\
echo "COINBASE_API_KEY: ${COINBASE_API_KEY:-Chưa thiết lập}"\n\
echo "COINBASE_API_SECRET: ${COINBASE_API_SECRET:-Chưa thiết lập}"\n\
\n\
# Kiểm tra kết nối Kafka\n\
echo "=== Kiểm tra kết nối Kafka ==="\n\
echo "Thử kết nối đến: $BOOTSTRAP_SERVERS"\n\
KAFKA_HOST=$(echo $BOOTSTRAP_SERVERS | cut -d: -f1)\n\
KAFKA_PORT=$(echo $BOOTSTRAP_SERVERS | cut -d: -f2)\n\
echo "Kiểm tra kết nối đến $KAFKA_HOST:$KAFKA_PORT..."\n\
timeout 60 bash -c "until nc -z $KAFKA_HOST $KAFKA_PORT 2>/dev/null; do echo \"Đang chờ Kafka khởi động...\"; sleep 5; done"\n\
if [ $? -eq 0 ]; then\n\
    echo "Kết nối thành công đến Kafka."\n\
else\n\
    echo "Warning: Không thể kết nối đến Kafka sau 60 giây!"\n\
    echo "Sẽ tiếp tục khởi động Producer nhưng có thể sẽ không kết nối được."\n\
fi\n\
\n\
# Chạy producer.py với log chi tiết\n\
echo "=== Khởi động Producer ==="\n\
echo "Chạy python /app/producer.py..."\n\
python /app/producer.py 2>&1 | tee /app/producer.log\n\
EXIT_CODE=$?\n\
echo "Mã thoát của producer.py: $EXIT_CODE"\n\
if [ $EXIT_CODE -ne 0 ]; then\n\
    echo "Lỗi: producer.py không chạy thành công. Kiểm tra /app/producer.log để biết chi tiết."\n\
    cat /app/producer.log\n\
    exit $EXIT_CODE\n\
else\n\
    echo "producer.py đã chạy thành công."\n\
fi\n' > /app/start.sh && chmod +x /app/start.sh

ENTRYPOINT ["/app/start.sh"]
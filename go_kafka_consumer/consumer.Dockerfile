FROM golang:latest AS builder

WORKDIR /app

# Cài đặt librdkafka-dev để hỗ trợ confluent-kafka-go
RUN apt-get update && \
    apt-get install -y librdkafka-dev pkg-config && \
    rm -rf /var/lib/apt/lists/*

# Copy go.mod and go.sum first for better caching
COPY go_kafka_consumer/go.mod go_kafka_consumer/go.sum ./
RUN go mod download

# Copy source code
COPY go_kafka_consumer/consumer.go .

# Build with CGO enabled (confluent-kafka-go)
RUN CGO_ENABLED=1 GOOS=linux go build -o consumer ./consumer.go

FROM debian:bookworm-slim

# Dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    netcat-traditional \
    librdkafka1 \
    curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/consumer .

# Startup script
RUN echo '#!/bin/bash \n\
# Check environment variables \n\
echo "=== Configuration Check ===" \n\
echo "Kafka Bootstrap Servers: ${BOOTSTRAP_SERVERS:-kafka:9092}" \n\
echo "S3 Endpoint: ${AWS_ENDPOINT:-AWS S3}" \n\
echo "S3 Bucket: ${S3_BUCKET:-ie212-coinbase-data}" \n\
echo "AWS Region: ${AWS_REGION:-ap-southeast-1}" \n\
\n\
# Check Kafka connection \n\
echo "=== Checking Kafka Connection ===" \n\
KAFKA_HOST=$(echo ${BOOTSTRAP_SERVERS:-kafka:9092} | cut -d: -f1) \n\
KAFKA_PORT=$(echo ${BOOTSTRAP_SERVERS:-kafka:9092} | cut -d: -f2) \n\
echo "Waiting for Kafka at $KAFKA_HOST:$KAFKA_PORT..." \n\
timeout 60 bash -c "until nc -z $KAFKA_HOST $KAFKA_PORT 2>/dev/null; do echo \"Waiting for Kafka to start...\"; sleep 5; done" \n\
if [ $? -ne 0 ]; then \n\
  echo "WARNING: Could not connect to Kafka after 60 seconds!" \n\
  echo "Consumer may not work until Kafka is ready." \n\
fi \n\
\n\
# Check MinIO/S3 connection \n\
if [ ! -z "$AWS_ENDPOINT" ] && [[ "$AWS_ENDPOINT" != "AWS S3" ]]; then \n\
  echo "=== Checking MinIO/S3 Connection ===" \n\
  # Parse URL to get host and port \n\
  if [[ "$AWS_ENDPOINT" == http://* ]] || [[ "$AWS_ENDPOINT" == https://* ]]; then \n\
    # Remove protocol part (http:// or https://) \n\
    S3_URL=$(echo "$AWS_ENDPOINT" | sed -e "s|^[^/]*//||") \n\
    # Split host and port \n\
    S3_HOST=$(echo "$S3_URL" | cut -d: -f1) \n\
    S3_PORT=$(echo "$S3_URL" | cut -d: -f2) \n\
    \n\
    # Set default port if not specified \n\
    if [ "$S3_HOST" = "$S3_PORT" ]; then \n\
      if [[ "$AWS_ENDPOINT" == https://* ]]; then \n\
        S3_PORT=443 \n\
      else \n\
        S3_PORT=80 \n\
      fi \n\
    fi \n\
    \n\
    # Remove path after host:port if any \n\
    S3_PORT=$(echo "$S3_PORT" | cut -d/ -f1) \n\
    \n\
    echo "Waiting for MinIO/S3 at $S3_HOST:$S3_PORT..." \n\
    timeout 60 bash -c "until nc -z $S3_HOST $S3_PORT 2>/dev/null; do echo \"Waiting for MinIO/S3 to start...\"; sleep 5; done" \n\
    if [ $? -ne 0 ]; then \n\
      echo "WARNING: Could not connect to MinIO/S3 after 60 seconds!" \n\
      echo "Consumer may not work properly without S3 connection." \n\
    else \n\
      echo "Successfully connected to MinIO/S3 at $S3_HOST:$S3_PORT" \n\
    fi \n\
  else \n\
    echo "AWS_ENDPOINT not in http(s)://host:port format, skipping connection check" \n\
  fi \n\
else \n\
  echo "=== Skipping MinIO/S3 Connection Check ===" \n\
  echo "No AWS_ENDPOINT found or using default AWS S3" \n\
fi \n\
\n\
echo "=== Starting Combined Consumer ===" \n\
./consumer' > /app/start.sh && chmod +x /app/start.sh

CMD ["/app/start.sh"]
# Dockerfile for Crypto Price Prediction Service
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY prediction_service/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files (from the main Crypto-TS-Model directory)
COPY Crypto-TS-Model-master/ ./model/

# Copy prediction service code
COPY prediction_service/src/ ./src/

# Copy configs
COPY prediction_service/configs/ ./configs/

# Create logs directory
RUN mkdir -p /app/logs

# Set Python path to include model source
ENV PYTHONPATH="/app/model/src:/app/src:$PYTHONPATH"

# Expose health check port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the prediction service
CMD ["python", "src/prediction_service.py", "--health-check"]
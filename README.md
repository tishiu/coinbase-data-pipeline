
# Hệ thống truy xuất và dự đoán dữ liệu tài chính theo thời gian thực từ Coinbase

##  Giới thiệu
Đồ án này xây dựng một hệ thống thu thập, xử lý và dự đoán giá tiền điện tử theo thời gian thực từ sàn **Coinbase**, sử dụng **kiến trúc Kappa**. Hệ thống hỗ trợ nhà đầu tư ra quyết định nhanh chóng và chính xác trong môi trường thị trường có độ biến động cao.

##  Mục tiêu
- Truy xuất và xử lý dữ liệu tiền điện tử thời gian thực.
- Dự đoán giá ngắn hạn (3 giờ tới) sử dụng LSTM, CNN-LSTM và CNN-LSTM-Attention.
- Lưu trữ dữ liệu hiệu quả với độ trễ thấp.
- Trực quan hóa dữ liệu và dự đoán bằng Grafana.

##  Công nghệ sử dụng

| Thành phần             | Công nghệ / Công cụ                        |
|------------------------|--------------------------------------------|
| Thu thập dữ liệu       | Coinbase WebSocket API                     |
| Streaming              | Apache Kafka, PySpark Structured Streaming |
| Lưu trữ                | AWS S3 (MinIO), Apache Cassandra           |
| Huấn luyện mô hình     | PyTorch, PySpark + TorchDistributor        |
| Dự đoán thời gian thực | Python, mô hình deep learning              |
| Trực quan hóa          | Grafana                                    |
| Quản lý dịch vụ        | Docker Compose                             |

##  Kiến trúc hệ thống (Kappa Architecture)
<img width="735" height="228" alt="image" src="https://github.com/user-attachments/assets/c88e6673-c469-4671-bcbb-c45c33097dc6" />

##  Hướng dẫn chạy hệ thống

### Yêu cầu
- Docker + Docker Compose
- Python 3.9+
- Go 1.18+
- PyTorch
- Spark + Hadoop
- Grafana

### Khởi động toàn bộ hệ thống

```bash
# Clone repo
git clone https://github.com/tishiu/coinbase-data-pipeline.git
cd coinbase-data-pipeline

# Khởi chạy 
docker-compose up --build



# BGE-M3 Embedding Service

Microservice độc lập cho BGE-M3 text embeddings.

## Tính năng

- Encode nhiều texts cùng lúc
- Encode single text
- Health check endpoint
- Model info endpoint
- CORS enabled cho cross-origin requests

## Cài đặt

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Chạy service

```bash
python main.py
```

Hoặc với uvicorn trực tiếp:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 3004
```

### 3. Chạy với Docker

```bash
docker-compose up -d
```

## API Endpoints

### Health Check
```
GET /health
```

### Model Info
```
GET /info
```

### Encode Multiple Texts
```
POST /encode
Content-Type: application/json

{
  "texts": ["text 1", "text 2", "text 3"],
  "normalize": true
}
```

### Encode Single Text
```
POST /encode/single
Content-Type: application/json

{
  "text": "your text here",
  "normalize": true
}
```

## Configuration

Tạo file `.env` hoặc set environment variables:

- `MODEL_NAME`: Tên model (default: BAAI/bge-m3)
- `EMBEDDING_DIM`: Dimension của embedding (default: 1024)
- `HOST`: Host để bind (default: 0.0.0.0)
- `PORT`: Port để chạy service (default: 3004)
- `LOG_LEVEL`: Log level (default: INFO)

## API Documentation

Sau khi chạy service, truy cập:
- Swagger UI: http://localhost:3004/docs
- ReDoc: http://localhost:3004/redoc



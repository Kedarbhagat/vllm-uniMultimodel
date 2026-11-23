# Title Generation Service - Setup & Usage Guide

## Overview

The **Title Generation Service** is a **separate microservice** that generates conversation titles asynchronously, preventing blocking of your chat server.

## Architecture

```
┌──────────────────────┐
│   Chat Server        │
│   (port 8000)        │
├──────────────────────┤
│ /chat/send           │──┐
│ (BackgroundTasks)    │  │ HTTP Request
└──────────────────────┘  │ (non-blocking)
                          │
                    ┌─────▼────────────────────┐
                    │ Title Service            │
                    │ (port 8001)              │
                    ├──────────────────────────┤
                    │ - Run ML inference       │
                    │ - Generate titles        │
                    │ - Update DB directly     │
                    │ - Cache results          │
                    └──────────────────────────┘
```

## How It Works

### 1. **Non-Blocking Title Generation**
When a user sends the **first message** in a thread:

```
Timeline:
t=0ms   → Chat server receives message
t=1ms   → Message stored in DB
t=2ms   → Server sends async HTTP request to title service
t=3ms   → Server returns streaming response to client (NO WAIT)
         ↓
         (In background on title service)
t=500ms → Title service loads ML model (if not cached)
t=600ms → Title service generates title
t=610ms → Title service updates thread title in DB
```

**Key advantage**: Client gets instant response without waiting for ML inference!

### 2. **Two Operating Modes**

#### **Async Mode (Default - Recommended)**
```python
{
    "thread_id": "abc123",
    "messages": [...],
    "async_mode": true
}
```
- Returns immediately with `status: "queued"`
- Title updates in background
- Perfect for chat server (non-blocking)

#### **Sync Mode (For batch operations)**
```python
{
    "thread_id": "abc123",
    "messages": [...],
    "async_mode": false
}
```
- Waits for title generation
- Returns completed title
- Use for background batch jobs

## Setup Instructions

### Step 1: Start the Title Service

```bash
# Option A: Direct Python
python RAG/title_service.py
# Service runs on http://localhost:8001

# Option B: With custom port
TITLE_SERVICE_PORT=8001 python RAG/title_service.py

# Option C: From different machine
TITLE_SERVICE_URL=http://192.168.1.100:8001 python RAG/server.py
```

### Step 2: Verify Service is Running

```bash
curl http://localhost:8001/health
```

Expected response:
```json
{
  "status": "healthy",
  "service": "Title Generation Service",
  "model": "google/flan-t5-base"
}
```

### Step 3: Chat Server Uses It Automatically

The chat server now automatically calls the title service. No code changes needed on the client side!

## API Endpoints

### 1. **Generate Title** (Main endpoint)
```http
POST /generate_title
Content-Type: application/json

{
  "thread_id": "abc123",
  "messages": [
    {"role": "human", "content": "What is machine learning?"},
    {"role": "assistant", "content": "Machine learning is..."}
  ],
  "async_mode": true
}
```

**Response (Async Mode):**
```json
{
  "thread_id": "abc123",
  "title": "Generating...",
  "status": "queued",
  "timestamp": "2026-01-11T10:30:00"
}
```

**Response (Sync Mode):**
```json
{
  "thread_id": "abc123",
  "title": "machine learning basics explained",
  "status": "success",
  "timestamp": "2026-01-11T10:30:00"
}
```

### 2. **Batch Generate Titles**
Generate titles for multiple threads at once:

```http
POST /batch_generate_titles
Content-Type: application/json

{
  "thread_ids": ["thread1", "thread2", "thread3"]
}
```

**Response:**
```json
{
  "status": "queued",
  "thread_count": 3,
  "message": "Titles are being generated in background"
}
```

### 3. **Health Check**
```http
GET /health
```

### 4. **Service Stats**
```http
GET /stats
```

## Configuration

### Environment Variables

```bash
# Set custom title service URL (if running on different machine/port)
export TITLE_SERVICE_URL=http://192.168.1.100:8001

# Or in .env file:
TITLE_SERVICE_URL=http://title-service.company.com:8001
```

### Logging

Check logs to verify title generation:
```bash
# In title service output:
✅ Title summarizer model loaded
✅ Generated title: 'machine learning basics'
✅ Updated thread abc123 with title: 'machine learning basics'
```

## Scaling Options

### Option 1: Single Instance (Current)
- One title service handles all requests
- First request loads model (~3-5s), then cached
- Sufficient for < 100 concurrent users

### Option 2: Load Balanced (Production)
```
Load Balancer (nginx)
    ├── Title Service Instance 1
    ├── Title Service Instance 2
    └── Title Service Instance 3
```

Use environment variable:
```bash
TITLE_SERVICE_URL=http://nginx-loadbalancer:8001
```

### Option 3: Queue-Based (High Volume)
```
Chat Server
    ├─ RabbitMQ / Redis
    └── Title Service (consumer)
```

Would require updating `title_service.py` to consume from queue.

## Troubleshooting

### Title Service Not Starting

```bash
# Check port availability
netstat -an | grep 8001

# Try different port
python RAG/title_service.py --port 8002
```

### Title Service Unreachable

```bash
# Check if service is running
curl http://localhost:8001/health

# Check chat server logs for errors
# Look for: "Failed to call title service"

# Verify network connectivity
ping localhost
```

### Slow Title Generation

First request takes ~3-5 seconds (model loading). Subsequent requests: ~500ms-1s.

If consistently slow:
1. Check CPU usage: `top`, `Task Manager`
2. Consider GPU acceleration by changing `device=-1` to `device=0`
3. Use lighter model: `google/flan-t5-small` (faster, lower quality)

### Titles Not Updating

1. Check title service logs for errors
2. Verify DB connection in title service
3. Check if `TITLE_SERVICE_URL` is correct in chat server

## Performance Metrics

| Metric | Value |
|--------|-------|
| First model load | ~3-5 seconds |
| Subsequent inference | 500ms - 1 sec |
| HTTP request overhead | ~50-100ms |
| DB update time | ~10-20ms |
| **Total non-blocking time** | **50-100ms** |

✅ Chat response unaffected - all processing in background!

## Advanced: Custom Models

Want a better/faster title generator?

```python
# In title_service.py, change:
_title_summarizer = pipeline(
    task="text2text-generation",
    model="google/flan-t5-base",  # ← Change this
    device=-1,
)

# Options:
# - "google/flan-t5-small"      (faster, lower quality)
# - "google/flan-t5-large"      (slower, higher quality)
# - "facebook/bart-large-cnn"   (news summarization)
# - "google/pegasus-cnn_dailymail" (document summarization)
```

## Docker Deployment

```dockerfile
# Dockerfile for title service
FROM python:3.11-slim
WORKDIR /app
COPY RAG/title_service.py .
COPY requirements.txt .
RUN pip install -r requirements.txt
EXPOSE 8001
CMD ["python", "title_service.py"]
```

```bash
docker build -t title-service .
docker run -p 8001:8001 title-service
```

## Summary

✅ **Separate service** keeps chat server fast  
✅ **Non-blocking** title generation (async HTTP)  
✅ **Scalable** can run multiple instances  
✅ **Flexible** can use different models/hardware  
✅ **Robust** handles errors gracefully  

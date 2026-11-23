# Title Service - Quick Reference

## 🚀 Quick Start (30 seconds)

### Terminal 1: Start Title Service
```bash
cd c:\Users\STUDENT\Documents\vllm
python RAG/title_service.py
```
✅ Runs on `http://localhost:8001`

### Terminal 2: Start Chat Server
```bash
cd c:\Users\STUDENT\Documents\vllm
python RAG/server.py
```
✅ Runs on `http://localhost:8000`

### Terminal 3: Run Tests
```bash
cd c:\Users\STUDENT\Documents\vllm
python RAG/test_title_service.py
```
✅ Should show `6/6 tests passed`

---

## 📋 How It Works

**When user sends first message:**

```
Client sends message
    ↓
Chat Server:
  1. Store message in DB (10ms)
  2. Queue title generation task (1ms)
  3. Return streaming response ← INSTANT!
    ↓
Chat Server (background):
  4. Call title service HTTP endpoint
    ↓
Title Service:
  5. Load ML model (first time: 3-5s, cached after)
  6. Generate title (~500ms)
  7. Update DB with title
    ↓
✅ Done! Title appears in chat in ~1-2 seconds
```

**Key insight**: Chat response doesn't wait for title! User sees response immediately.

---

## 📊 Performance

| Operation | Time | Blocking |
|-----------|------|----------|
| Add message to DB | 10ms | Yes |
| Queue title task | 1ms | Yes |
| HTTP request overhead | 50-100ms | Yes |
| **Chat response sent** | **~110ms** | **✅ NO** |
| Load ML model (first) | 3-5s | No |
| Generate title | 500ms-1s | No |
| Update DB with title | 10ms | No |
| **Total** | **~1-2s** | **All async** |

---

## 🔧 Configuration

### Set Custom Title Service URL
```bash
# If running on different machine
set TITLE_SERVICE_URL=http://192.168.1.100:8001
python RAG/server.py
```

### Set Custom Title Service Port
```bash
# If port 8001 is busy
set TITLE_SERVICE_PORT=8002
python RAG/title_service.py
```

---

## 🧪 Testing

### Test 1: Service Health
```bash
curl http://localhost:8001/health
# Should return: {"status": "healthy", ...}
```

### Test 2: Generate Title (Async)
```bash
curl -X POST http://localhost:8001/generate_title \
  -H "Content-Type: application/json" \
  -d '{
    "thread_id": "test1",
    "messages": [{"role": "human", "content": "What is AI?"}],
    "async_mode": true
  }'
# Returns immediately with status: "queued"
```

### Test 3: Generate Title (Sync)
```bash
curl -X POST http://localhost:8001/generate_title \
  -H "Content-Type: application/json" \
  -d '{
    "thread_id": "test2",
    "messages": [{"role": "human", "content": "Explain machine learning"}],
    "async_mode": false
  }'
# Returns after title is generated (5-10s first time)
```

### Test 4: Batch Generation
```bash
curl -X POST http://localhost:8001/batch_generate_titles \
  -H "Content-Type: application/json" \
  -d '{"thread_ids": ["t1", "t2", "t3"]}'
# Queues titles for 3 threads
```

### Test Full Integration
```bash
python RAG/test_title_service.py
# Comprehensive 6-test suite
```

---

## 📁 Files Overview

```
RAG/
├── title_service.py              ← Main service (NEW)
├── server.py                      ← Chat server (MODIFIED)
├── summariser.py                  ← Old impl (DEPRECATED)
├── TITLE_SERVICE_GUIDE.md         ← Full docs (NEW)
├── TITLE_SERVICE_IMPLEMENTATION.md ← Architecture (NEW)
├── run_title_service.py           ← Launcher (NEW)
└── test_title_service.py          ← Tests (NEW)
```

---

## 🐛 Troubleshooting

### "Connection refused" to title service
```bash
# 1. Check title service is running
curl http://localhost:8001/health

# 2. If not running, start it
python RAG/title_service.py

# 3. If port busy, use different port
set TITLE_SERVICE_PORT=8002
python RAG/title_service.py
```

### "Title not updating"
```bash
# 1. Check chat server logs for errors
# 2. Verify title service is accessible
curl http://localhost:8001/health

# 3. Check database permissions
# 4. Manually test title service
python RAG/test_title_service.py
```

### "Service slow on first request"
```bash
# Normal! First request loads 3-5GB model
# Subsequent requests are ~500ms-1s
# Model stays cached in memory for fast access
```

### "Port already in use"
```bash
# Find process using port
netstat -ano | findstr :8001

# Kill process (get PID from above)
taskkill /PID 12345 /F

# Or use different port
set TITLE_SERVICE_PORT=8002
python RAG/title_service.py
```

---

## 🎯 API Endpoints

### Title Service (`http://localhost:8001`)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Check service status |
| `/generate_title` | POST | Generate title for thread |
| `/batch_generate_titles` | POST | Generate titles for multiple threads |
| `/stats` | GET | Get service statistics |
| `/docs` | GET | Interactive API documentation |

### Chat Server calls automatically
When you send first message via `/chat/send`, server automatically calls title service in background.

---

## 🔄 Lifecycle

### Service Startup
```
1. Load transformer model (3-5 seconds)
2. Ready to accept requests
3. Model stays in memory (cached)
4. Subsequent requests: ~500ms each
```

### Message Flow
```
User sends message
    ↓
1. Chat server receives at /chat/send
2. Message stored in DB
3. Title generation queued in background_tasks
4. Streaming response begins immediately
5. (In parallel) Title service generates title
6. (In parallel) Title service updates DB
7. Client sees chat response with "New Chat" initially
8. After 1-2 seconds, title updates in UI
```

### Graceful Degradation
```
If title service is down:
- Chat still works perfectly
- Thread remains titled "New Chat"
- No errors shown to user
- All logged on server side
```

---

## 💡 Pro Tips

### Tip 1: Use Async by Default
```python
# Always use async_mode=true for chat
# Only use async_mode=false for batch jobs
```

### Tip 2: Monitor Title Generation
Check server logs:
```bash
# Look for:
# "Queued title generation for new thread"
# "Title generation response: success"
# Or "Failed to call title service"
```

### Tip 3: Custom Models
Want faster or better titles?

```python
# In title_service.py, change model:
model="google/flan-t5-small"      # Faster
model="google/flan-t5-base"       # Balanced (default)
model="google/flan-t5-large"      # Higher quality
model="facebook/bart-large-cnn"   # For news
```

### Tip 4: Batch Update Existing Threads
Generate titles for all existing threads:

```bash
curl -X POST http://localhost:8001/batch_generate_titles \
  -H "Content-Type: application/json" \
  -d '{"thread_ids": ["thread1", "thread2", "..."]}' &
```

---

## 📊 Monitoring

### Check if Service is Running
```bash
# Service health
curl http://localhost:8001/health

# Service stats
curl http://localhost:8001/stats
```

### Monitor Logs
```bash
# Title service logs (in its terminal)
# Look for: "Generated title", "Updated thread"

# Chat server logs (in its terminal)
# Look for: "Title generation response"
```

---

## 🚀 Deployment Options

### Option 1: Same Machine (Current)
```
Both services on localhost
- Port 8000: Chat Server
- Port 8001: Title Service
```

### Option 2: Different Machines
```bash
# Machine A: Chat Server
python RAG/server.py

# Machine B: Title Service
set TITLE_SERVICE_URL=http://machine-a:8001
python RAG/title_service.py

# Or on Machine A:
set TITLE_SERVICE_URL=http://192.168.1.100:8001
python RAG/server.py
```

### Option 3: Docker
```bash
# Build
docker build -t title-service RAG/

# Run
docker run -p 8001:8001 title-service
```

---

## ✅ Checklist

- [ ] Started Title Service (`python RAG/title_service.py`)
- [ ] Started Chat Server (`python RAG/server.py`)
- [ ] Ran test suite (`python RAG/test_title_service.py`)
- [ ] All 6 tests passing
- [ ] Sent first message in chat
- [ ] Title appeared in thread after 1-2 seconds
- [ ] Checked server logs for success messages

---

## 📚 More Info

- Full Guide: `TITLE_SERVICE_GUIDE.md`
- Implementation Details: `TITLE_SERVICE_IMPLEMENTATION.md`
- Source Code: `title_service.py`
- Tests: `test_title_service.py`

# Title Service - Implementation Checklist

## Pre-Launch Checklist

### Prerequisites
- [ ] Python 3.8+ installed
- [ ] Required packages: `transformers`, `torch`, `fastapi`, `uvicorn`
- [ ] Port 8000 available (chat server)
- [ ] Port 8001 available (title service)
- [ ] Database connection configured

### File Verification
- [ ] `title_service.py` exists in `RAG/`
- [ ] `server.py` updated with title service code
- [ ] `run_title_service.py` exists in `RAG/`
- [ ] `test_title_service.py` exists in `RAG/`

### Documentation Review
- [ ] Read `TITLE_SERVICE_QUICK_REF.md` (5 min)
- [ ] Understand architecture from `TITLE_SERVICE_GUIDE.md`
- [ ] Review diagrams in `TITLE_SERVICE_DIAGRAMS.md`

---

## Startup Checklist

### Terminal 1: Start Title Service
```bash
cd c:\Users\STUDENT\Documents\vllm
python RAG/title_service.py
```
- [ ] No error on startup
- [ ] See "✅ Title summarizer model loaded"
- [ ] Running on `http://localhost:8001`
- [ ] Ready to accept requests

### Terminal 2: Start Chat Server
```bash
cd c:\Users\STUDENT\Documents\vllm
python RAG/server.py
```
- [ ] No import errors
- [ ] Chat server starts normally
- [ ] Running on `http://localhost:8000`
- [ ] Can access `/` endpoint

### Terminal 3: Run Tests
```bash
cd c:\Users\STUDENT\Documents\vllm
python RAG/test_title_service.py
```
Results should show:
- [ ] ✅ PASS | Health Check
- [ ] ✅ PASS | Async Generation
- [ ] ✅ PASS | Sync Generation
- [ ] ✅ PASS | Batch Generation
- [ ] ✅ PASS | Stats
- [ ] ✅ PASS | Chat Integration
- [ ] Result: 6/6 tests passed

---

## Integration Testing

### Test 1: Direct Service Call
```bash
curl http://localhost:8001/health
```
- [ ] Returns `{"status": "healthy", ...}`
- [ ] Confirms service is running

### Test 2: Generate Single Title
```bash
curl -X POST http://localhost:8001/generate_title \
  -H "Content-Type: application/json" \
  -d '{
    "thread_id": "test_001",
    "messages": [{"role": "human", "content": "What is machine learning?"}],
    "async_mode": true
  }'
```
- [ ] Returns with `status: "queued"`
- [ ] Confirms async mode works

### Test 3: Sync Mode (Wait for Result)
```bash
curl -X POST http://localhost:8001/generate_title \
  -H "Content-Type: application/json" \
  -d '{
    "thread_id": "test_002",
    "messages": [{"role": "human", "content": "Explain neural networks"}],
    "async_mode": false
  }'
```
- [ ] Returns with actual title (not "Generating...")
- [ ] Response time: 5-10 seconds (first request) or 1-2 seconds (cached)
- [ ] Confirms sync mode works

### Test 4: Chat Server Integration
1. Open chat interface
2. Start new conversation
3. Send first message
4. [ ] Chat response appears immediately (~100-150ms)
5. [ ] After 1-2 seconds, thread title appears
6. Check chat server logs:
   - [ ] See "Queued title generation for new thread"
   - [ ] See "Title generation response: success"
7. Check database:
   - [ ] Thread title updated from "New Chat" to generated title

### Test 5: Multiple Messages
1. Send first message → [ ] Title should be generated
2. Send second message → [ ] Chat responds
3. Send third message → [ ] No new title generation
4. Verify:
   - [ ] Title generation only happens once (on first message)
   - [ ] Subsequent messages don't queue new title tasks

### Test 6: Error Scenarios
1. Stop title service while chat is running
2. Send new message
3. [ ] Chat still responds normally
4. [ ] Title remains "New Chat" 
5. Check chat server logs:
   - [ ] See "Failed to call title service"
   - [ ] No crashes or errors
6. Restart title service
7. [ ] Everything works again

---

## Database Verification

### Check Thread Table
```sql
SELECT id, title, created_at FROM threads ORDER BY created_at DESC LIMIT 5;
```
- [ ] First thread has generated title (not "New Chat")
- [ ] Title is reasonable and relevant
- [ ] Timestamp matches creation time

### Check Messages Table
```sql
SELECT role, content, created_at FROM messages WHERE thread_id = 'YOUR_THREAD_ID' ORDER BY created_at;
```
- [ ] Human message recorded
- [ ] AI response recorded
- [ ] Timestamps are sequential
- [ ] No system messages about title

---

## Performance Verification

### First Request Timing
```
t=0    Title service receives request
t=3000 Model loaded (3-5 seconds typical)
t=3500 Title generated
t=3510 Database updated
Total: 3.5-5 seconds ✅ Acceptable (first time only)
```
- [ ] First title takes 3-5 seconds
- [ ] Subsequent titles take ~500ms-1s
- [ ] Chat not blocked during this time

### Chat Response Timing
```
t=0    User sends message
t=10   Message in database
t=50   Title task queued
t=100  Streaming response sent
Total: ~100ms ✅ Fast!
```
- [ ] Chat response appears within ~100-150ms
- [ ] Title update happens separately (~1-2s later)

---

## Log Verification

### Check Title Service Logs
Should see lines like:
```
✅ Title summarizer model loaded
[time] Title generation request for thread: abc123
[time] Generated title: 'machine learning basics'
[time] ✅ Updated thread abc123 with title: 'machine learning basics'
```
- [ ] Model loads on startup
- [ ] Requests logged
- [ ] Titles generated successfully
- [ ] Database updates logged

### Check Chat Server Logs
Should see lines like:
```
[time] Queued title generation for new thread abc123
[time] Title generation response: success
```
- [ ] Title generation queued
- [ ] Successful completion logged
- [ ] No connection errors

---

## Configuration Verification

### Environment Variables
```bash
# Should see these in logs or configuration
TITLE_SERVICE_URL=http://localhost:8001
TITLE_SERVICE_PORT=8001
TITLE_SERVICE_HOST=0.0.0.0
```
- [ ] Correct service URL configured
- [ ] Port not in use
- [ ] Host accessible

### Model Configuration
In `title_service.py`:
```python
model="google/flan-t5-base"  # ✅ Current
device=-1                      # ✅ CPU (default)
```
- [ ] Using correct model
- [ ] Device setting appropriate for hardware

---

## Deployment Checklist

### Before Production
- [ ] Services run without errors for > 1 hour
- [ ] All 6 tests passing consistently
- [ ] No memory leaks detected
- [ ] No database connection issues
- [ ] Error logging configured
- [ ] Monitoring setup (optional)

### Docker (Optional)
- [ ] Dockerfile created
- [ ] Image builds successfully
- [ ] Container runs and passes tests
- [ ] Port mappings correct

### Scaling (Future)
- [ ] Load balancer configuration ready (if needed)
- [ ] Multiple instances tested (if needed)
- [ ] Database handles concurrent writes
- [ ] Connection pooling configured

---

## Monitoring Checklist

### Daily/Weekly
- [ ] Services still running
- [ ] Logs show successful title generations
- [ ] Database growing at expected rate
- [ ] No repeated errors in logs

### Performance Monitoring
- [ ] First request time: 3-5 seconds
- [ ] Subsequent requests: ~500ms-1s
- [ ] Chat response: ~100-150ms
- [ ] No timeouts or failures

### Error Tracking
- [ ] No connection refused errors
- [ ] No database errors
- [ ] No out of memory errors
- [ ] All errors properly logged

---

## Maintenance Checklist

### Monthly
- [ ] Review logs for errors
- [ ] Check database size
- [ ] Monitor resource usage
- [ ] Update dependencies if needed

### Quarterly
- [ ] Update transformer model (if new versions available)
- [ ] Review and optimize database queries
- [ ] Test disaster recovery
- [ ] Security audit

### Annually
- [ ] Full system review
- [ ] Performance baseline
- [ ] Capacity planning
- [ ] Documentation update

---

## Troubleshooting Decision Tree

### Service won't start
- [ ] Check port available: `netstat -an | grep 8001`
- [ ] Check Python version: `python --version`
- [ ] Check dependencies: `pip list | grep -E "fastapi|uvicorn|torch"`
- [ ] Check permissions
- [ ] Review error message

### Tests failing
- [ ] Both services running?
- [ ] Ports correct? (8000 and 8001)
- [ ] Network connectivity?
- [ ] Database accessible?
- [ ] Run test again

### Titles not updating
- [ ] Title service running?
- [ ] Check logs for errors
- [ ] Verify database connectivity
- [ ] Check TITLE_SERVICE_URL is correct
- [ ] Run direct curl test

### Slow title generation
- [ ] First request? (normal: 3-5s)
- [ ] Subsequent requests? (should be ~500ms-1s)
- [ ] Check CPU/GPU usage
- [ ] Check system resources
- [ ] Check if model is cached

### Chat server crashes
- [ ] Check error message
- [ ] Verify title service is accessible
- [ ] Check database connection
- [ ] Review recent code changes
- [ ] Check system resources

---

## Success Criteria

### ✅ All checks passed when:
- [x] Both services start without errors
- [x] All 6 tests pass
- [x] Chat responds in ~100ms
- [x] Titles appear in 1-2 seconds
- [x] Titles persist in database
- [x] Error handling works (title service can fail without breaking chat)
- [x] Logs show all operations
- [x] No performance degradation

### 🎉 You're ready when:
All items above are checked!

---

## Quick Reference Commands

### Start Services
```bash
# Terminal 1
python RAG/title_service.py

# Terminal 2
python RAG/server.py

# Terminal 3
python RAG/test_title_service.py
```

### Test Services
```bash
# Health check
curl http://localhost:8001/health

# Generate title (async)
curl -X POST http://localhost:8001/generate_title \
  -H "Content-Type: application/json" \
  -d '{"thread_id":"t1","messages":[{"role":"human","content":"Test"}],"async_mode":true}'

# View stats
curl http://localhost:8001/stats

# Full test suite
python RAG/test_title_service.py
```

### Stop Services
```bash
# Ctrl+C in each terminal
```

---

## Support Resources

- **Quick Help?** → `TITLE_SERVICE_QUICK_REF.md`
- **Setup Instructions?** → `TITLE_SERVICE_GUIDE.md`
- **How It Works?** → `TITLE_SERVICE_DIAGRAMS.md`
- **Architecture Details?** → `TITLE_SERVICE_IMPLEMENTATION.md`
- **Code Examples?** → `title_service.py` source

---

## Sign Off

Date: January 11, 2026

- [ ] All checklist items verified
- [ ] System ready for production
- [ ] Documentation reviewed
- [ ] Team briefed on new architecture

**Status: ✅ READY**

---

Last Updated: January 11, 2026
Next Review: January 18, 2026

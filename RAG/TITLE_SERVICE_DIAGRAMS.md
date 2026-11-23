# Title Service - Visual Flow Diagrams

## Diagram 1: Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        CLIENT (Browser)                      │
│                                                              │
│  ┌──────────────────────┐           ┌──────────────────┐    │
│  │ Send Message Input   │           │ Chat Display     │    │
│  └──────────────────────┘           └──────────────────┘    │
│           │ POST /chat/send                ▲                │
│           │                                │ Stream          │
│           └────────────────────────────────┘                │
└────────────────────────┬──────────────────────────────────┬─┘
                         │                                  │
                    HTTP │                                  │ HTTP
                         │                                  │
         ┌───────────────▼─────────────────┐                │
         │   CHAT SERVER (Port 8000)       │                │
         │                                 │                │
         │  ┌─────────────────────────┐   │                │
         │  │ /chat/send Handler      │   │                │
         │  │ - Receive message       │   │                │
         │  │ - Store in DB           │   │ Streaming      │
         │  │ - Queue title task      │   │ Response       │
         │  │ - Return immediately ◄──────────────────────┘
         │  └─────────────────────────┘   │
         │              │                  │
         │              │ BackgroundTask   │
         │              │ HTTP Request     │
         │              │ (non-blocking)   │
         │              ▼                  │
         │  ┌─────────────────────────┐   │
         │  │ request_title_          │   │
         │  │ generation()            │   │
         │  │ ├─ Call Title Service   │   │
         │  │ └─ Handle response      │   │
         │  └─────────────────────────┘   │
         └───────────────┬────────────────┘
                         │
                    HTTP │ /generate_title
                         │
         ┌───────────────▼──────────────────┐
         │  TITLE SERVICE (Port 8001)       │
         │                                  │
         │  ┌──────────────────────────┐   │
         │  │ /generate_title Handler  │   │
         │  │ - Receive HTTP request   │   │
         │  │ - Load ML model (async)  │   │
         │  │ - Generate title         │   │
         │  │ - Update DB directly     │   │
         │  │ - Return HTTP response   │   │
         │  └──────────────────────────┘   │
         │              │                  │
         │              ▼                  │
         │  ┌──────────────────────────┐   │
         │  │ Database                 │   │
         │  │ (Update thread title)    │   │
         │  └──────────────────────────┘   │
         └──────────────────────────────────┘
```

---

## Diagram 2: Timeline - Request vs Title Generation

```
CLIENT TIMELINE                           SERVER TIMELINE
═════════════════════════════════════════════════════════════════

User types message                       
and clicks Send                          
       │
       │                                 
       ├─────── POST /chat/send ────────►  t=0ms: Receive request
       │                                   │
       │                                   ├─ t=5ms: Add to DB
       │                                   │
       │                                   ├─ t=8ms: Queue background task
       │                                   │
       │                                   ├─ t=10ms: Start streaming
       │                                   │
       │◄──────── Streaming response ─────┤
       │                                   │
       ├─ Chat appears in UI              ├─ t=50ms: Http request to title svc
       │  (INSTANT! ~100ms total)         │
       │                                   
       │ (User reading response...)       
       │                                   
       │                                   ┌─ TITLE SERVICE (Port 8001)
       │                                   │
       │                                   ├─ t=500ms: Receive request
       │                                   │
       │                                   ├─ t=1000ms: Load model (first time)
       │                                   │
       │                                   ├─ t=1500ms: Generate title
       │                                   │
       │◄──── Update thread title ────────┤ t=1510ms: Update DB
       │  (After 1-2 seconds)             │
       │                                   └─ t=1520ms: Done!
       │
       └─ Chat now shows proper title

```

---

## Diagram 3: Data Flow - First User Message

```
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: User sends first message                            │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
     ┌───────────────────────┐
     │ POST /chat/send       │
     │ {                     │
     │   "thread_id": "t123" │
     │   "message": "Hello"  │
     │ }                     │
     └────────┬──────────────┘
              │
              ▼
     ┌──────────────────────────────────────┐
     │ STEP 2: Chat Server Processing       │
     │                                      │
     │ a) add_message(                      │
     │    thread_id="t123",                 │
     │    role="human",                     │
     │    content="Hello"                   │
     │    )                                 │
     │                                      │
     │ b) raw_history = get_messages(...)   │
     │    [{"role":"human","content":.."} ] │
     │                                      │
     │ c) Check: len(user_messages) == 1    │
     │    YES! First message detected       │
     │                                      │
     │ d) background_tasks.add_task(        │
     │      request_title_generation,       │
     │      thread_id="t123",               │
     │      messages=[...],                 │
     │      async_mode=True                 │
     │    )                                 │
     │    ✅ Queued (non-blocking!)        │
     └────────┬─────────────────────────────┘
              │
              ├──► CONTINUE CHAT PROCESSING
              │    (start streaming response)
              │
              └──► BACKGROUND: HTTP to Title Service
                   
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: Response Streaming (Starts Immediately)             │
│                                                              │
│ ✅ User sees chat response WITHOUT waiting for title        │
│                                                              │
│ Timeline: ~100ms from POST to response visible              │
└─────────────────────────────────────────────────────────────┘

                    (Meanwhile in background)
                            │
                            ▼
                ┌──────────────────────────┐
                │ STEP 4: Title Service    │
                │ Receives HTTP request    │
                │ /generate_title          │
                └────────┬─────────────────┘
                         │
                         ▼
                ┌──────────────────────────┐
                │ STEP 5: Load ML Model    │
                │ (First time: 3-5 seconds)│
                │ (Cached after)           │
                └────────┬─────────────────┘
                         │
                         ▼
                ┌──────────────────────────┐
                │ STEP 6: Generate Title   │
                │ Input: "Hello"           │
                │ Output: "greeting hello" │
                │ Time: ~500ms             │
                └────────┬─────────────────┘
                         │
                         ▼
                ┌──────────────────────────┐
                │ STEP 7: Update Database  │
                │                          │
                │ UPDATE threads           │
                │ SET title = "greeting.." │
                │ WHERE id = "t123"        │
                └────────┬─────────────────┘
                         │
                         ▼
                ┌──────────────────────────┐
                │ STEP 8: Return Response  │
                │ {                        │
                │   "status": "success",   │
                │   "title": "greeting.."  │
                │ }                        │
                └────────┬─────────────────┘
                         │
                         ▼
                    ✅ Complete!
           (Total background time: 1-2 seconds)

┌─────────────────────────────────────────────────────────────┐
│ RESULT: User sees chat immediately, title appears shortly   │
│ Chat doesn't wait for title generation!                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Diagram 4: Request/Response Sequence

```
Client              Chat Server         Title Service          Database
  │                     │                     │                   │
  │ POST /chat/send     │                     │                   │
  │────────────────────►│                     │                   │
  │                     │ Store message       │                   │
  │                     ├────────────────────────────────────────►│
  │                     │                     │                   │
  │                     │                     │◄────────────────────│
  │                     │                     │    Stored OK       │
  │                     │                     │                   │
  │                     │ Queue background    │                   │
  │                     │ title task          │                   │
  │                     │                     │                   │
  │                     │ POST /generate_     │                   │
  │                     │ title (async)       │                   │
  │                     ├────────────────────►│                   │
  │                     │                     │                   │
  │ Stream Response     │                     │                   │
  │◄────────────────────┤                     │                   │
  │                     │                     │                   │
  │ Chat appears!       │                     │ Load model...     │
  │ (in ~100ms)         │                     │                   │
  │                     │                     │ Generate title... │
  │                     │                     │                   │
  │                     │                     │ Update thread     │
  │                     │                     ├───────────────────►│
  │                     │                     │                   │
  │ (After 1-2s)        │                     │◄────────────────────│
  │ Title updates       │                     │    Updated!        │
  │                     │                     │                   │

KEY INSIGHT:
- User sees response in ~100ms (no wait!)
- Title updates in ~1-2 seconds (in background)
- No blocking at any point
```

---

## Diagram 5: Error Handling Flow

```
Request to Title Service
        │
        ▼
    ┌─────────────┐
    │ Try to call │
    │ service     │
    └──┬──────┬───┘
       │      │
    OK │      │ Error
       │      │
       ▼      ▼
    ┌─────┐ ┌──────────────────┐
    │ ✅  │ │ Service Down/    │
    │Recv │ │ Network Error    │
    │Title│ └────┬─────────────┘
    └────┘      │
       │         ▼
       │    ┌──────────────────────┐
       │    │ Catch Exception      │
       │    │ Log error            │
       │    │ return None          │
       │    └────┬─────────────────┘
       │         │
       └─────┬───┘
             │
             ▼
      ┌────────────────────────┐
      │ RESULT:                │
      │                        │
      │ ✅ Chat works fine     │
      │ ❌ Title = "New Chat"  │
      │ 📝 Error logged        │
      │ 😊 User doesn't see    │
      │    error               │
      └────────────────────────┘

BENEFIT: Graceful degradation
         Service availability not coupled
         Chat quality unaffected
```

---

## Diagram 6: Scaling Scenarios

### Current (Single Instance)
```
User 1 ──┐
User 2 ──┤─► Chat Server (8000)  ──┐
User 3 ──┘                         ├─► Title Service (8001)
...                                │
                                   └─► Database
```

### Future: Load Balanced
```
User 1 ──┐                         ┌─► Title Service (1)
User 2 ──┤─► Chat Server (8000) ──┤
User 3 ──┤    (multiple)           ├─► Title Service (2)  ──┐
User 4 ──┘                         │                       ├─► Database
...                                └─► Title Service (3) ──┘
```

### Advanced: Queue-Based
```
Chat Server ──► Redis/RabbitMQ ┬─► Title Worker (1)
               (Message Queue)  ├─► Title Worker (2) ──┐
                                ├─► Title Worker (3)   ├─► Database
                                └─► Title Worker (N) ──┘
```

---

## Diagram 7: Model Loading (First Request)

```
First request arrives
        │
        ▼
    ┌──────────────────────────┐
    │ Load transformer model   │
    │ from HuggingFace         │
    │                          │
    │ google/flan-t5-base      │
    │ ~3-5GB of data           │
    │                          │
    │ Time: 3-5 seconds ⏳     │
    └────────┬─────────────────┘
             │
             ▼
    ┌──────────────────────────┐
    │ Model loaded in memory   │
    │ ✅ Cached ✅             │
    │                          │
    │ Subsequent requests:     │
    │ Skip this step!          │
    │ Time: ~500ms instead     │
    └────────┬─────────────────┘
             │
             ▼
    ┌──────────────────────────┐
    │ Process request          │
    │ Generate title           │
    │ Return response          │
    └──────────────────────────┘
```

---

## Diagram 8: Database State Timeline

```
Time    Chat Server Action           Database State
───────────────────────────────────────────────────
 t=0    POST /chat/send received
        
 t=5ms  INSERT message                
        (human)                        
                                      threads:
                                      ├─ id: "t123"
 t=10   Queue title task              ├─ title: "New Chat"
        (starts async)                └─ created_at: now
        
        START STREAMING RESPONSE
        (user sees chat)
                                      messages:
 t=50   HTTP to title service         ├─ id: "m1"
                                      ├─ role: "human"
                                      ├─ content: "Hello"
                                      └─ created_at: t=5ms
        
 t=500  Title service loads model
 
 t=1000 Title generated: "greeting"
        
 t=1010 UPDATE thread title          threads:
                                      ├─ id: "t123"
                                      ├─ title: "greeting"  ◄── CHANGED!
                                      └─ created_at: now
        
 t=2000 COMPLETE!


OBSERVATION:
- Database updated in two stages
- Message inserted first (t=5ms)
- Title updated later (t=1010ms)
- Both operations tracked and logged
```

---

## Summary

✅ **Non-blocking**: Chat returns immediately (~100ms)  
✅ **Async**: Title generates in background (~1-2s)  
✅ **Reliable**: Graceful error handling  
✅ **Scalable**: Independent services  
✅ **Fast**: ML model cached after first request  

# 🐛 Bug Report & Issues Scan

## Critical Issues 🔴

### 1. **Duplicate Imports in server.py**
- **Location**: Lines 3, 36
- **Issue**: `import time` declared twice
- **Impact**: Confusing and wastes memory
- **Fix**: Remove duplicate at line 36

### 2. **Duplicate ChatOpenAI Imports**
- **Location**: Lines 12 (langchain_community) and 19 (langchain_openai)
- **Issue**: Two different versions of ChatOpenAI imported
- **Impact**: Potential conflicts or using wrong class
- **Fix**: Keep only `from langchain_openai import ChatOpenAI` (line 19)

### 3. **No Authentication on Thread Access**
- **Location**: `/chat/history`, `/chat/threads`, `/chat/rag` endpoints
- **Issue**: Anyone who knows a thread_id can access it
- **Impact**: Major security breach - users can see other users' conversations
- **Fix**: Add session check - verify current user owns the thread
```python
@app.get("/chat/history")
def chat_history(thread_id: str, request: Request):
    user = request.session.get("user")
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    # Verify user owns this thread
```

### 4. **Different Default ALLOWED_DOMAIN**
- **Location**: auth/main.py (line 76) vs RAG/server.py (line 95)
- **Issue**: Flask defaults to "gmail.com", FastAPI defaults to "nmit.ac.in"
- **Impact**: Different validation rules between servers
- **Fix**: Use same default from .env environment variable

### 5. **Session Cookie Not Secure**
- **Location**: auth/main.py (line 41)
- **Issue**: `SESSION_COOKIE_SECURE=False` means cookies sent over HTTP
- **Impact**: Session hijacking vulnerability
- **Fix**: Set to `True` in production (or use environment variable)

---

## High Priority Issues 🟠

### 6. **No Rate Limiting on Authentication Endpoints**
- **Location**: `/api/register`, `/api/login`, `/api/update-password`
- **Issue**: Endpoints can be hammered for brute force attacks
- **Impact**: Account takeover attacks possible
- **Recommendation**: Implement rate limiting (e.g., 5 attempts per minute)

### 7. **No Input Validation on Password Updates**
- **Location**: `PasswordUpdateInput` model (line 331)
- **Issue**: No validation for password format/complexity
- **Impact**: Users can set weak passwords
- **Fix**: Add validation:
```python
class PasswordUpdateInput(BaseModel):
    email: str
    current_password: str
    new_password: str
    
    @validator('new_password')
    def validate_password(cls, v):
        if len(v) < 8:  # Minimum length
            raise ValueError('Password must be at least 8 characters')
        # Add more rules as needed
        return v
```

### 8. **Hardcoded Model Infrastructure IPs**
- **Location**: API_PORT_MAPPING (line 163)
- **Issue**: IP addresses `172.31.21.186` hardcoded - not flexible
- **Impact**: Can't easily change infrastructure without code change
- **Fix**: Move to environment variables

### 9. **No Database Connection Health Check**
- **Location**: Server startup
- **Issue**: Server starts even if database is unreachable
- **Impact**: Runtime errors when first query happens
- **Fix**: Add health check on startup:
```python
@app.on_event("startup")
async def startup():
    try:
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
        logger.info("✅ Database connection OK")
    except Exception as e:
        logger.error(f"❌ Database not available: {e}")
        raise
```

### 10. **Agent Client Cache Never Cleared**
- **Location**: agent_clients dict (line 179)
- **Issue**: Dictionary grows indefinitely, models cached forever
- **Impact**: Memory leak with many model switches
- **Fix**: Implement cache expiration or size limit

---

## Medium Priority Issues 🟡

### 11. **No File Size/Type Validation**
- **Location**: `/chat/upload_doc` endpoint (line 433)
- **Issue**: No checks on file size or MIME type
- **Impact**: Can upload malicious files or exhaust disk
- **Fix**: Add validation:
```python
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_TYPES = {'pdf', 'txt', 'docx'}

if file.size > MAX_FILE_SIZE:
    raise HTTPException(status_code=413, detail="File too large")
```

### 12. **Hardcoded Document Processing Parameters**
- **Location**: Line 474 in upload endpoint
- **Issue**: `max_sentences=10` hardcoded
- **Impact**: Can't adjust for different document types
- **Fix**: Make configurable via environment or request parameter

### 13. **No Error Handling in Stream Response**
- **Location**: `/chat/rag` endpoint stream_response() (line 613)
- **Issue**: If generator fails mid-stream, connection may hang
- **Impact**: Resource leaks
- **Fix**: Add proper exception handling and cleanup

### 14. **No User Name Sanitization**
- **Location**: User registration (line 241)
- **Issue**: Only email sanitized, user names can contain HTML/scripts
- **Impact**: XSS vulnerability if name displayed without escaping
- **Recommendation**: Always escape output in frontend

### 15. **CORS allows all methods**
- **Location**: Line 133 (allow_methods=["*"])
- **Issue**: Allows DELETE, PUT, PATCH on any endpoint
- **Impact**: More attack surface
- **Fix**: Specify only needed methods: `["GET", "POST", "OPTIONS"]`

---

## Low Priority Issues 🟢

### 16. **Hardcoded Document Query Parameters**
- **Location**: TestVectorstoreInput.query default (line 220)
- **Issue**: "what is india goal for 2030" hardcoded default
- **Impact**: Confusing in tests
- **Fix**: Use empty string or generic default

### 17. **Missing Logging in Authentication**
- **Location**: Flask routes (auth/main.py)
- **Issue**: No logs for login attempts or failures
- **Impact**: Can't audit security events
- **Recommendation**: Add logging for all auth events

### 18. **RAG Endpoint Returns Dict on Error**
- **Location**: Line 596 - `return {"status": "error", ...}`
- **Issue**: Inconsistent with HTTPException error format
- **Impact**: Client needs to handle two different error formats
- **Fix**: Raise HTTPException instead

### 19. **No Session Timeout Handling**
- **Location**: Both servers
- **Issue**: No explicit session expiration check on API calls
- **Impact**: Expired sessions might still work for a while
- **Recommendation**: Add session validation middleware

### 20. **Database Transaction Isolation Not Set**
- **Location**: SQLAlchemy engine creation (line 62-68 in RAG, line 48-54 in Flask)
- **Issue**: Using default isolation level
- **Impact**: Potential concurrency issues
- **Recommendation**: Set isolation_level explicitly for PostgreSQL

---

## Warnings & Observations ⚠️

### 21. **Default Secret Key Used**
- **Location**: Both servers
- **Issue**: "your-secret-key-here" used as default
- **Impact**: Weak session security if .env not set
- **Recommendation**: Require .env to be set in production

### 22. **No Explicit Resource Cleanup**
- **Location**: Throughout
- **Issue**: Database connections not always explicitly closed
- **Impact**: Connection pool exhaustion possible
- **Recommendation**: Use context managers consistently

### 23. **No Endpoint Rate Limiting**
- **Location**: All endpoints
- **Issue**: No global or per-endpoint rate limiting
- **Impact**: DoS attacks possible
- **Recommendation**: Implement using middleware

### 24. **Inconsistent Error Response Format**
- **Location**: HTTP endpoints return different formats
- **Issue**: Some return JSONResponse, some return dict
- **Impact**: Client parsing inconsistency
- **Recommendation**: Standardize error response format

### 25. **SQLAlchemy ORM Model Sync Issue**
- **Location**: server.py defines User model separately from auth/main.py
- **Issue**: Two separate User models could get out of sync
- **Impact**: Column mismatches possible
- **Recommendation**: Use shared model or validate schema consistency

---

## Summary

| Severity | Count | Status |
|----------|-------|--------|
| 🔴 Critical | 5 | Must Fix |
| 🟠 High | 5 | Should Fix |
| 🟡 Medium | 5 | Nice to Fix |
| 🟢 Low | 5 | For Later |
| ⚠️ Warnings | 5 | Document |
| **Total** | **25** | |

---

## Quick Fix Priority Order

1. **Add thread ownership validation** (Issue #3) - Security critical
2. **Fix ALLOWED_DOMAIN defaults** (Issue #4) - Causes bugs
3. **Add database connection health check** (Issue #9) - Prevents runtime errors
4. **Add input validation for passwords** (Issue #7) - Security
5. **Remove duplicate imports** (Issues #1, #2) - Code quality
6. **Add file upload validation** (Issue #11) - Security
7. **Standardize error responses** (Issue #24) - Better UX


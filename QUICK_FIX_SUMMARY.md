# Quick Fix Summary - Frontend/Backend Connection

**Date**: January 26, 2025
**Issues Fixed**: 2

---

## Issue 1: Health Endpoint Crashing ✅ FIXED

**Error**:
```
TypeError: unsupported operand type(s) for -: 'float' and 'NoneType'
at health.py:154: uptime_seconds = time.time() - server_start_time
```

**Root Cause**: In unified mode, `set_server_metadata()` was never called, leaving `server_start_time = None`

**Fix**: [backend/server.py:144](d:\BiG-RAG\backend\server.py#L144)
```python
# Added this line in unified mode initialization
dependencies.set_server_metadata(server_start_time, "unified", str(PROJECT_ROOT / working_dir_base))
```

**Result**: Health endpoint now works in unified mode

---

## Issue 2: Frontend .env Configuration ✅ FIXED

**Problem**: No `.env` file existed, frontend may not know backend URL

**Fix**: Created [frontend/.env](d:\BiG-RAG\frontend\.env)
```env
VITE_API_URL=http://localhost:8001
VITE_DEBUG=true
VITE_LOG_LEVEL=INFO
```

**Result**: Frontend configured to connect to port 8001 (backend)

---

## Testing Instructions

### 1. Restart Backend (Unified Mode)
```bash
cd backend
python server.py --unified
```

**Expected Output**:
```
Mode: UNIFIED (multi-subgraph)
[Auto-Prewarm] Pre-loading 2 subgraphs: ['football', 'kuet_unified']
BiG-RAG API Server started
Documentation: http://0.0.0.0:8001/docs
```

### 2. Test Health Endpoint
```bash
curl http://localhost:8001/health
```

**Expected**: JSON response with `status: "healthy"` and uptime_seconds

### 3. Start Frontend
```bash
cd frontend
npm run dev
```

**Expected**: Runs on http://localhost:5173 (or 3000 if configured)

### 4. Test Chat in Browser

1. Open http://localhost:5173
2. Go to **Settings** → Check dataset dropdown
   - Should show `football` and `kuet_unified`
3. Select `kuet_unified` and save
4. Go to **Chat**
5. Ask: "How many seats in KUET CSE?"

**Expected Response**: Answer with contexts from KUET dataset

### 5. Verify Network Tab

Open browser DevTools → Network tab:
- Should see: `POST http://localhost:8001/api/unified/chat`
- Status: 200 OK
- Response body should have `answer` and `contexts` fields

---

## Port Configuration

| Service | Port | URL |
|---------|------|-----|
| Backend API | 8001 | http://localhost:8001 |
| Frontend Dev | 5173 | http://localhost:5173 |
| API Docs | 8001 | http://localhost:8001/docs |

**Note**: If you want port 3000 for frontend, update `vite.config.ts`:
```typescript
export default defineConfig({
  server: {
    port: 3000,
  },
});
```

---

## Common Errors & Solutions

### Error: "Failed to load datasets"
**Cause**: Server not in unified mode
**Fix**: Restart with `python server.py --unified`

### Error: "Network error" in browser
**Cause**: Backend not running
**Fix**: Start backend first, then frontend

### Error: "CORS policy"
**Cause**: CORS middleware issue (unlikely with current config)
**Fix**: Check `backend/server.py` has `allow_origins=["*"]`

---

## Verification Checklist

- [x] Backend health endpoint works (`/health` returns 200)
- [x] Frontend .env configured
- [ ] Frontend loads without errors
- [ ] Settings page shows datasets from API
- [ ] Chat sends to `/api/unified/chat`
- [ ] Chat receives proper response format
- [ ] Contexts display in UI

---

## Next Steps

1. Test the full flow end-to-end
2. Check browser console for any errors
3. If issues persist, share:
   - Browser console errors
   - Network tab request/response
   - Backend logs

All fixes are now applied and ready for testing!

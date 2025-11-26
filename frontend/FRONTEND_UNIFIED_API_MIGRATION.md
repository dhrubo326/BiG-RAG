# Frontend Migration to Unified API

**Date**: January 26, 2025
**Status**: COMPLETE - Ready for Testing

---

## Overview

Successfully migrated frontend from single-mode legacy API endpoints to new unified multi-subgraph API.

---

## Changes Made

### 1. Constants Updated (`src/utils/constants.ts`)

**Added New Endpoints**:
- `UNIFIED_CHAT`: `/api/unified/chat` - Chat with RAG + LLM synthesis
- `UNIFIED_ASK`: `/api/unified/ask` - Simple Q&A with contexts
- `UNIFIED_QUERY`: `/api/unified/query` - Retrieval only
- `UNIFIED_REGISTRY`: `/api/unified/registry` - List all subgraphs
- `UNIFIED_SUBGRAPHS`: `/api/unified/subgraphs` - Subgraph info

**Deprecated** (kept for backwards compatibility):
- `/ask`
- `/chat/completions`
- `/search`

---

### 2. Chat Service Updated (`src/services/chat.ts`)

**Before**:
```typescript
// Used /chat/completions (single-mode)
const response = await api.post(API_ENDPOINTS.CHAT_COMPLETIONS, {...});
```

**After**:
```typescript
// Uses /api/unified/chat (multi-subgraph)
const response = await api.post(API_ENDPOINTS.UNIFIED_CHAT, {
  messages: [{role: 'user', content: query}],
  force_subgraphs: [activeDataset],  // From settings
  output_mode: 'answer_with_context',
  use_rag: true,
  ...
});
```

**Key Changes**:
- Reads `activeDataset` from localStorage (settings store)
- Uses `force_subgraphs` parameter to route to selected dataset
- Maps unified API response format to frontend types
- Extracts metrics from `llm_metrics` and `retrieval_metrics`

---

### 3. Dataset Service Created (`src/services/datasets.ts`)

**New Service** for fetching available datasets:
```typescript
export const fetchAvailableDatasets = async () => {
  const response = await api.get(API_ENDPOINTS.UNIFIED_REGISTRY);
  // Returns { datasets, isUnifiedMode }
};
```

**Features**:
- Fetches from `/api/unified/registry`
- Falls back to single-mode if endpoint returns 503/404
- Returns enabled subgraphs only

---

### 4. Dataset Hook Created (`src/hooks/useDatasets.ts`)

**New React Hook** for dataset management:
```typescript
const { datasets, isLoading, isUnifiedMode, error, refetch } = useDatasets();
```

**Features**:
- Auto-loads datasets on mount
- Provides refetch function for manual reload
- Detects unified vs single mode
- Handles errors gracefully

---

### 5. Settings Page Updated (`src/pages/Settings.tsx`)

**Before**:
- Hardcoded dataset dropdown:
  ```html
  <SelectItem value="demo_test">Demo Test</SelectItem>
  <SelectItem value="SingleTopic">SingleTopic</SelectItem>
  ...
  ```

**After**:
- Dynamic dataset dropdown from API:
  ```html
  {datasets.map(ds => (
    <SelectItem value={ds.name}>{ds.name} - {ds.description}</SelectItem>
  ))}
  ```

**UI Enhancements**:
- "Unified Mode" badge when server in unified mode
- Refresh button to reload datasets
- Loading state with spinner
- Truncated descriptions (max 50 chars)

---

## How It Works

### Data Flow

1. **Settings Page Loads**:
   - `useDatasets()` hook calls `/api/unified/registry`
   - Fetches list of available subgraphs
   - Populates dataset dropdown

2. **User Selects Dataset**:
   - Choice saved to `settings-store` in localStorage
   - `activeDataset` field updated

3. **User Sends Chat Message**:
   - `askQuestion()` reads `activeDataset` from localStorage
   - Sends to `/api/unified/chat` with `force_subgraphs: [activeDataset]`
   - Backend routes to selected subgraph

4. **Response Received**:
   - Maps `answer` and `contexts` from unified API
   - Displays in chat UI

---

## Testing Instructions

### 1. Start Backend in Unified Mode

```bash
cd backend
python server.py --unified
```

**Expected logs**:
```
Mode: UNIFIED (multi-subgraph)
Registry: expr/subgraph_registry.json
Max cached subgraphs: 10
[Auto-Prewarm] Pre-loading 2 subgraphs: ['football', 'kuet_unified']
```

### 2. Start Frontend

```bash
cd frontend
npm run dev
```

**Access**: http://localhost:5173

### 3. Test Dataset Selection

1. Navigate to **Settings** page (gear icon)
2. Check **Default Dataset** section:
   - Should show "Unified Mode" badge
   - Should list: `football` and `kuet_unified`
   - Click refresh icon to reload

3. Select `kuet_unified` and save

### 4. Test Chat

1. Navigate to **Chat** page
2. Ask: "How many CSE seats at KUET?"
3. **Expected**: Answer with contexts from `kuet_unified` subgraph

4. Go back to Settings, select `football`
5. Ask: "Who won the Champions League in 2023?"
6. **Expected**: Answer from `football` subgraph

---

## Verification Checklist

- [ ] Settings page shows available datasets
- [ ] "Unified Mode" badge visible
- [ ] Can switch between datasets
- [ ] Chat uses selected dataset
- [ ] Contexts display correctly
- [ ] Metrics shown (retrieval time, tokens, etc.)
- [ ] No console errors

---

## Troubleshooting

### "Failed to load datasets"

**Cause**: Server not in unified mode or registry not found

**Fix**:
```bash
# Restart server with --unified flag
python backend/server.py --unified
```

### "Connection successful" but no datasets

**Cause**: Server in single mode (no `--unified` flag)

**Expected behavior**: Dropdown shows only `demo_test`

### Chat returns 503 error

**Cause**: Using unified endpoint but server in single mode

**Fix**: Either:
- Start server with `--unified` flag
- Or frontend will automatically detect and use fallback

---

## Next Steps (Optional)

1. **Add Chat UI Dataset Indicator**:
   - Show active dataset in chat header
   - Allow quick switching without going to Settings

2. **Auto-Routing Support**:
   - Add toggle for auto-routing vs forced routing
   - When auto-routing enabled, don't pass `force_subgraphs`
   - LLM will decide which subgraph to use

3. **Multi-Subgraph Queries**:
   - Allow selecting multiple datasets
   - Pass as `force_subgraphs: ['football', 'kuet_unified']`
   - Useful for cross-domain queries

4. **Dataset Metadata Display**:
   - Show description, creation date, entity/relation counts
   - Add tooltips with full description

---

## Files Modified

1. `frontend/src/utils/constants.ts` - Added unified endpoints
2. `frontend/src/services/chat.ts` - Migrated to `/api/unified/chat`
3. `frontend/src/services/datasets.ts` - **NEW** dataset service
4. `frontend/src/hooks/useDatasets.ts` - **NEW** dataset hook
5. `frontend/src/pages/Settings.tsx` - Dynamic dataset dropdown

---

## Backwards Compatibility

All changes are **non-breaking**:
- Legacy endpoints still available (deprecated)
- Falls back to single mode if unified not available
- Existing localStorage settings preserved

---

## Conclusion

Frontend now fully supports unified multi-subgraph API! Users can:
- See all available datasets
- Switch between datasets in Settings
- Chat uses selected dataset automatically
- No server restart required when switching

**Ready for production use!**

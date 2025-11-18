# Per-Query Language Parameter - Implementation Status

## Current Status

✅ **Backend: COMPLETE** (Committed in 3ce7cd9)
⏳ **Frontend: PENDING** (Reverted due to white screen error - to be implemented later)

---

## Backend Changes (✅ Complete & Committed)

### Files Modified:
1. **`bigrag/base.py`** - Added optional `language` field to QueryParam
2. **`backend/api/models/models.py`** - Added language to request models (ChatCompletionRequest, AskRequest)
3. **`backend/api/routes/llm.py`** - Pass language from request to QueryParam
4. **`backend/api/routes/retrieval.py`** - Pass language from request to QueryParam
5. **`bigrag/operate.py`** - Cascading language priority logic + enhanced logging
6. **`backend/server.py`** - Language configuration from .env
7. **`CLAUDE.md`** - Documentation and usage examples

### How It Works:
**Cascading Language Priority:**
1. **Priority 1**: `language` parameter in API request (per-query override)
2. **Priority 2**: `DEFAULT_LANGUAGE` from `.env` file (currently: Bangla)
3. **Priority 3**: Hardcoded default (English)

### Supported Languages:
English, Bangla, Hindi, Arabic, Chinese, Spanish, French, German, Japanese, Korean

### Backend Testing:
```bash
# Test with cURL
curl -X POST http://localhost:8001/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "KUET e CSE te koyti seat ache?"}],
    "use_rag": true,
    "language": "Bangla"
  }'

# Verify in backend logs:
# [Query Preprocess] Using per-query language override: Bangla
# LANGUAGE: Bangla
```

---

## Frontend Changes (⏳ Pending Implementation)

### Why Pending?
Initial frontend implementation caused white screen error. Changes were reverted to maintain stability. Backend is fully functional via API.

### Files to Modify (Step-by-Step Guide):

#### Step 1: Add Language Constants
**File**: `frontend/src/utils/constants.ts`
**Location**: After existing constants (around line 96)

```typescript
// Query languages (for per-query language override)
// null = Auto (uses server default from .env)
export const QUERY_LANGUAGES = [
  { value: null, label: 'Auto (Server Default)' },
  { value: 'English', label: 'English' },
  { value: 'Bangla', label: 'বাংলা (Bangla)' },
  { value: 'Hindi', label: 'हिन्दी (Hindi)' },
  { value: 'Arabic', label: 'العربية (Arabic)' },
  { value: 'Chinese', label: '中文 (Chinese)' },
  { value: 'Spanish', label: 'Español (Spanish)' },
  { value: 'French', label: 'Français (French)' },
  { value: 'German', label: 'Deutsch (German)' },
  { value: 'Japanese', label: '日本語 (Japanese)' },
  { value: 'Korean', label: '한국어 (Korean)' },
] as const;
```

#### Step 2: Add Language to Type Definition
**File**: `frontend/src/types/api.ts`
**Location**: In QueryParams interface (around line 83-93)

```typescript
export interface QueryParams {
  query: string;
  top_k?: number;
  mode?: 'local' | 'global' | 'hybrid' | 'naive';
  enable_reranking?: boolean;
  dataset?: string;
  model?: string;
  temperature?: number;
  max_tokens?: number;
  llm_provider?: string;
  language?: string | null;  // ADD THIS LINE
}
```

#### Step 3: Add Language State to Store
**File**: `frontend/src/stores/chat.ts`

**Add to interface** (around line 24):
```typescript
language: string | null;  // Query language override (null = Auto/Server Default)
```

**Add to actions interface** (around line 41):
```typescript
setLanguage: (language: string | null) => void;
```

**Add to initial state** (around line 61):
```typescript
language: null,  // Auto (uses server default)
```

**Add action implementation** (around line 118):
```typescript
setLanguage: (language) => set({ language }),
```

**Add to resetSettings** (around line 127):
```typescript
language: null,  // Reset to Auto
```

**Add to persistence** (around line 140):
```typescript
language: state.language,
```

#### Step 4: Add Language Dropdown to Settings UI
**File**: `frontend/src/components/chat/ChatSettings.tsx`

**Add import** (line 2):
```typescript
import { AVAILABLE_MODELS, QUERY_MODES, QUERY_LANGUAGES } from '../../utils/constants';
```

**Add to props interface** (around line 16):
```typescript
language: string | null;
onLanguageChange: (language: string | null) => void;
```

**Destructure props** (around line 27):
```typescript
language,
onLanguageChange,
```

**Add JSX** (between Query Mode and Enable Reranking sections, around line 130):
```typescript
{/* Query Language */}
<div>
  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
    Query Language
  </label>
  <select
    value={language === null ? 'null' : language}
    onChange={(e) => onLanguageChange(e.target.value === 'null' ? null : e.target.value)}
    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
  >
    {QUERY_LANGUAGES.map((option) => (
      <option key={option.value === null ? 'null' : option.value} value={option.value === null ? 'null' : option.value}>
        {option.label}
      </option>
    ))}
  </select>
  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
    Auto uses server default from .env
  </p>
</div>
```

#### Step 5: Connect Chat Page to Settings
**File**: `frontend/src/pages/Chat.tsx`

**Add to useChat destructuring** (around line 33):
```typescript
language,
setLanguage,
```

**Pass to ChatSettings component** (around line 183):
```typescript
<ChatSettings
  model={model}
  temperature={temperature}
  topK={topK}
  enableReranking={enableReranking}
  queryMode={queryMode || 'hybrid'}
  language={language}  // ADD THIS
  onModelChange={setModel}
  onTemperatureChange={setTemperature}
  onTopKChange={setTopK}
  onRerankingChange={setEnableReranking}
  onQueryModeChange={setQueryMode}
  onLanguageChange={setLanguage}  // ADD THIS
  onReset={resetSettings}
  onClose={() => setShowSettings(false)}
/>
```

#### Step 6: Update useChat Hook
**File**: `frontend/src/hooks/useChat.ts`

**Add to store destructuring** (around line 20):
```typescript
language,
setLanguage,
```

**Add to params object** (around line 70):
```typescript
const params: QueryParams = {
  query: content,
  top_k: topK,
  mode: queryMode,
  enable_reranking: enableReranking,
  model,
  temperature,
  max_tokens: 4096,
  language,  // ADD THIS LINE
};
```

**Add to dependency array** (around line 122):
```typescript
[
  messages,
  model,
  temperature,
  topK,
  enableReranking,
  queryMode,
  language,  // ADD THIS LINE
  addMessage,
  setRetrievedContexts,
  setThinking,
  setLoading,
  setError,
]
```

**Add to return statement** (around line 350):
```typescript
return {
  // ... existing returns
  language,  // ADD THIS
  // ... existing returns
  setLanguage,  // ADD THIS (in settings section around line 369)
};
```

#### Step 7: Update API Service
**File**: `frontend/src/services/chat.ts`

**Update askQuestion function** (around line 9-30):
```typescript
export const askQuestion = async (params: QueryParams): Promise<QueryResponse> => {
  const requestBody: any = {
    model: params.model || 'gpt-4o-mini',
    messages: [
      {
        role: 'user',
        content: params.query,
      },
    ],
    temperature: params.temperature || 0.7,
    max_tokens: params.max_tokens || 4096,
    llm_provider: params.llm_provider,
    use_rag: true,
    enable_reranking: params.enable_reranking,
  };

  // Only include language if explicitly set (not null/undefined)
  if (params.language !== undefined && params.language !== null) {
    requestBody.language = params.language;
  }

  const response = await api.post(API_ENDPOINTS.CHAT_COMPLETIONS, requestBody);
  // ... rest of function
```

**Update context fetch** (around line 35-54):
```typescript
try {
  const askBody: any = {
    question: params.query,
    top_k: params.top_k || 5,
    mode: params.mode || 'hybrid',
    enable_reranking: params.enable_reranking,
  };

  // Only include language if explicitly set (not null/undefined)
  if (params.language !== undefined && params.language !== null) {
    askBody.language = params.language;
  }

  const contextResponse = await api.post(API_ENDPOINTS.ASK, askBody);
  // ... rest of function
```

**Update streamChat function** (around line 85-116):
```typescript
export const streamChat = async (
  params: QueryParams,
  onChunk: (chunk: string) => void,
  onContexts?: (contexts: RetrievedContext[]) => void,
  abortSignal?: AbortSignal
): Promise<void> => {
  const streamBody: any = {
    question: params.query,
    top_k: params.top_k,
    mode: params.mode,
    enable_reranking: params.enable_reranking,
    model: params.model,
    temperature: params.temperature,
    stream: true,
  };

  // Only include language if explicitly set (not null/undefined)
  if (params.language !== undefined && params.language !== null) {
    streamBody.language = params.language;
  }

  const response = await fetch(`${api.defaults.baseURL}${API_ENDPOINTS.ASK}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(streamBody),
    signal: abortSignal,
  });
  // ... rest of function
```

---

## Frontend Testing Checklist

After implementing all frontend changes:

1. ✅ Start dev server: `cd frontend && npm run dev`
2. ✅ No TypeScript errors in console
3. ✅ No white screen error
4. ✅ Chat page loads successfully
5. ✅ Settings panel opens (gear icon)
6. ✅ Language dropdown appears between "Query Mode" and "Enable Reranking"
7. ✅ Language selection persists after page refresh
8. ✅ Backend logs show correct language when querying:
   - `[Query Preprocess] Using per-query language override: English`
   - `LANGUAGE: English`

### Test Scenarios:
1. **Auto (default)**: Leave as "Auto" → uses Bangla from .env
2. **English override**: Select "English" → normalizes to English
3. **Bangla explicit**: Select "বাংলা (Bangla)" → normalizes to Bangla
4. **Banglish input**: Type "KUET e CSE te koyti seat ache?" with Bangla selected → converts to proper Bangla

---

## Troubleshooting

### White Screen Error
**Symptoms**: Frontend shows blank white screen after changes
**Cause**: TypeScript compilation error or missing type definitions
**Fix**:
1. Revert all frontend changes: `git checkout frontend/src/`
2. Implement changes incrementally (one file at a time)
3. Test after each file change
4. Check browser console for errors

### Language Not Passed to Backend
**Symptoms**: Backend logs don't show language parameter
**Fix**:
1. Check browser Network tab → verify language in request body
2. Verify `params.language` is not undefined in `services/chat.ts`
3. Check conditional inclusion logic (should only add if not null/undefined)

### Language Dropdown Not Showing
**Symptoms**: Settings panel missing language dropdown
**Fix**:
1. Verify `QUERY_LANGUAGES` imported in ChatSettings.tsx
2. Check props are passed from Chat.tsx to ChatSettings
3. Verify JSX placement (should be between Query Mode and Enable Reranking)

---

## Notes

- Backend is fully functional and can be tested via cURL/Postman
- Frontend implementation can be done incrementally when ready
- Use conditional inclusion (`if (params.language !== undefined && params.language !== null)`) to avoid sending null values
- Test each file change individually to isolate white screen issues
- Backend logs provide clear indication of which language is being used

---

## Related Documentation

- **CLAUDE.md**: Usage examples and API documentation
- **Backend commit**: 3ce7cd9 (update backend)

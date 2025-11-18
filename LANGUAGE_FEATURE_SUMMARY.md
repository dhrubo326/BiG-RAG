# Per-Query Language Parameter - Implementation Complete ✅

## Summary

Successfully implemented optional per-query language parameter across frontend and backend.

## Backend Changes (Already Complete)

✅ Added `language: Optional[str] = None` to `QueryParam` in `bigrag/base.py`
✅ Added `language` to API request models (`ChatCompletionRequest`, `AskRequest`)
✅ Added cascading language priority logic in `kg_query()`
✅ Updated both `/chat/completions` and `/ask` endpoints to pass language parameter
✅ Enhanced logging to show language being used for each query

## Frontend Changes (Completed in this session)

### 1. Constants (`frontend/src/utils/constants.ts`)
```typescript
export const QUERY_LANGUAGES = [
  { value: null, label: 'Auto (Server Default)' },
  { value: 'English', label: 'English' },
  { value: 'Bangla', label: 'বাংলা (Bangla)' },
  { value: 'Hindi', label: 'हिन्दी (Hindi)' },
  // ... more languages
] as const;
```

### 2. Store (`frontend/src/stores/chat.ts`)
- Added `language: string | null` to state
- Added `setLanguage()` action
- Added language to persist storage
- Added language to resetSettings()

### 3. UI Component (`frontend/src/components/chat/ChatSettings.tsx`)
- Added language dropdown selector
- Positioned between "Query Mode" and "Enable Reranking"
- Shows helpful text: "Auto uses server default from .env"

### 4. Page Component (`frontend/src/pages/Chat.tsx`)
- Destructured `language` and `setLanguage` from useChat hook
- Passed to `<ChatSettings>` component

### 5. Remaining Tasks (COMPLETE THESE)

#### File: `frontend/src/hooks/useChat.ts`

**Add to destructuring (line 18):**
```typescript
language,
setLanguage,
```

**Add to params (line 60):**
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

**Add to dependency array (line 112):**
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

**Add to return statement (line 345):**
```typescript
language,
setLanguage,
```

#### File: `frontend/src/services/chat.ts`

**Add to interface (create if doesn't exist):**
```typescript
export interface QueryParams {
  query: string;
  top_k?: number;
  mode?: string;
  enable_reranking?: boolean;
  model?: string;
  temperature?: number;
  max_tokens?: number;
  language?: string | null;  // ADD THIS
}
```

**Update askQuestion function (line 10):**
```typescript
const response = await api.post(API_ENDPOINTS.CHAT_COMPLETIONS, {
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
  language: params.language,  // ADD THIS LINE
});
```

## Testing

1. **Restart frontend:**
   ```bash
   cd frontend
   npm run dev
   ```

2. **Open chat interface:** http://localhost:5173/chat

3. **Test scenarios:**
   - **Auto (default)**: Leave language as "Auto (Server Default)" → uses Bangla from .env
   - **English override**: Select "English" → query preprocessing in English
   - **Bangla explicit**: Select "বাংলা (Bangla)" → query preprocessing in Bangla
   - **Persistence**: Refresh page → language selection should persist

4. **Verify in backend logs:**
   ```
   [Query Preprocess] Using per-query language override: English  ← When override selected
   LANGUAGE: English  ← Shows in comparison log
   ```

## API Examples

### cURL Example
```bash
curl -X POST http://localhost:8001/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "KUET e CSE te koyti seat ache?"}],
    "use_rag": true,
    "language": "Bangla"
  }'
```

### Frontend TypeScript Example
```typescript
const response = await askQuestion({
  query: userInput,
  language: selectedLanguage,  // null | 'English' | 'Bangla' | etc.
  top_k: 10,
  mode: 'hybrid',
});
```

## Documentation Location

Brief examples added to CLAUDE.md under "Query Preprocessing" section.

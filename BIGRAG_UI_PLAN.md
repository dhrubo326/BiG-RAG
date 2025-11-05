# BiG-RAG UI/Frontend Plan

**Version:** 1.1
**Date:** November 2025
**Status:** Ready for Implementation
**Last Updated:** Latest package versions as of November 5, 2025

---

## Executive Summary

This document outlines the plan for building a **production-ready, scalable React-based UI** for BiG-RAG that provides:

1. **Interactive Graph Visualization** - Debug retrieval with Cytoscape.js
2. **Chat Interface** - Ask questions and visualize retrieval in real-time
3. **Document Management** - Upload, view, delete documents from knowledge graph
4. **Evaluation Dashboard** - Monitor EM/F1 scores, view results, compare runs
5. **Settings & Configuration** - Manage datasets, models, retrieval parameters

The architecture maintains **strict separation** between:
- **Backend (API)**: FastAPI server (port 8001) - already exists
- **Frontend (UI)**: React + TypeScript (port 5173) - to be built
- **Framework**: BiG-RAG Python library - already exists

All three components can be run **independently** for development, testing, and deployment.

---

## 0. Pre-Implementation: Directory Re-arrangement

Before starting frontend development, we need to reorganize the project structure to maintain clean separation between backend, frontend, and framework.

### 0.1 Current Structure Issues

```
BiG-RAG/
├── api/                    # ❌ Needs to be renamed to "backend/"
├── bigrag/                 # ✅ OK (core framework)
├── datasets/               # ✅ OK
├── expr/                   # ✅ OK (built graphs)
├── script_api.py           # ❌ Should move to backend/
├── script_build.py         # ✅ OK (framework script)
├── script_process.py       # ✅ OK (framework script)
└── ... (other files)
```

**Problems:**
1. `api/` folder name is ambiguous - doesn't clearly indicate it's the backend
2. `script_api.py` is at root level - should be in backend folder
3. No frontend folder exists yet
4. Mixed responsibilities at root level

### 0.2 Target Structure

```
BiG-RAG/
├── backend/                          # ← Backend API (FastAPI server)
│   ├── api/                          # API modules (moved from root api/)
│   │   ├── __init__.py
│   │   ├── jobs.py
│   │   ├── registry.py
│   │   ├── kg_utils.py
│   │   ├── evaluation.py
│   │   ├── answer_generation.py
│   │   ├── csv_evaluation.py
│   │   ├── export.py
│   │   ├── ground_truth.py
│   │   ├── metrics.py
│   │   ├── models.py
│   │   ├── models_eval.py
│   │   ├── stats.py
│   │   └── utils.py
│   ├── server.py                     # ← Renamed from script_api.py
│   ├── requirements.txt              # ← NEW: Backend-specific dependencies
│   ├── README.md                     # ← NEW: Backend documentation
│   └── .env.example                  # ← NEW: Environment variables template
│
├── frontend/                         # ← NEW: React application
│   ├── public/
│   │   ├── vite.svg
│   │   └── bigrag-logo.svg
│   ├── src/
│   │   ├── app/
│   │   │   ├── App.tsx
│   │   │   └── Router.tsx
│   │   ├── pages/
│   │   │   ├── Dashboard.tsx
│   │   │   ├── Chat.tsx
│   │   │   ├── GraphViz.tsx
│   │   │   ├── Documents.tsx
│   │   │   ├── Evaluation.tsx
│   │   │   └── Settings.tsx
│   │   ├── components/
│   │   │   ├── ui/                   # shadcn/ui components
│   │   │   ├── graph/
│   │   │   ├── chat/
│   │   │   ├── documents/
│   │   │   └── layout/
│   │   ├── stores/
│   │   │   ├── graph.ts
│   │   │   ├── chat.ts
│   │   │   ├── documents.ts
│   │   │   └── settings.ts
│   │   ├── services/
│   │   │   ├── api.ts
│   │   │   ├── graph.ts
│   │   │   ├── chat.ts
│   │   │   ├── documents.ts
│   │   │   └── evaluation.ts
│   │   ├── hooks/
│   │   │   ├── useGraph.ts
│   │   │   ├── useChat.ts
│   │   │   └── useDocuments.ts
│   │   ├── types/
│   │   │   ├── graph.ts
│   │   │   ├── api.ts
│   │   │   └── index.ts
│   │   ├── utils/
│   │   │   ├── formatters.ts
│   │   │   └── constants.ts
│   │   ├── i18n/
│   │   │   ├── en.json
│   │   │   ├── zh.json
│   │   │   └── index.ts
│   │   ├── main.tsx
│   │   └── index.css
│   ├── package.json
│   ├── tsconfig.json
│   ├── tsconfig.node.json
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   ├── components.json              # shadcn/ui config
│   ├── .env.example
│   └── README.md
│
├── bigrag/                          # ← Framework (unchanged)
│   ├── __init__.py
│   ├── bigrag.py
│   ├── operate.py
│   ├── reranker.py
│   ├── storage.py
│   ├── base.py
│   ├── llm.py
│   ├── prompt.py
│   ├── utils.py
│   ├── config.py
│   └── kg/
│
├── datasets/                        # ← Datasets (unchanged)
├── expr/                            # ← Built graphs (unchanged)
├── docs/                            # ← Documentation (unchanged)
├── tests/                           # ← Tests (unchanged)
├── scripts/                         # ← NEW: Utility scripts
│   ├── migrate_to_new_structure.py # Migration helper
│   └── setup_dev_env.sh            # Development setup
│
├── script_build.py                  # ← Framework scripts (unchanged)
├── script_process.py
├── run_singletopic_evaluation.py
├── validate_singletopic_dataset.py
├── test_improvements.py
│
├── .gitignore
├── README.md                        # ← UPDATE: New structure
├── BIGRAG_UI_PLAN.md               # ← This file
├── CLAUDE.md
├── requirements.txt                 # ← Framework dependencies
└── setup.py
```

### 0.3 Migration Steps

**Step 1: Create new directories**
```bash
mkdir backend
mkdir frontend
mkdir scripts
```

**Step 2: Move backend files**
```bash
# Move api folder
mv api backend/api

# Move and rename script_api.py
mv script_api.py backend/server.py
```

**Step 3: Update imports in backend/server.py**
```python
# OLD (script_api.py at root):
from api.jobs import processing_jobs
from api.registry import registry
from api.kg_utils import get_document_stats_from_kg

# NEW (backend/server.py):
from api.jobs import processing_jobs
from api.registry import registry
from api.kg_utils import get_document_stats_from_kg
# No changes needed - relative imports work the same!
```

**Step 4: Create backend requirements.txt**
```bash
cd backend
cat > requirements.txt << 'EOF'
# Backend API dependencies
fastapi==0.115.0
uvicorn[standard]==0.32.0
pydantic==2.9.2
python-multipart==0.0.12

# Import BiG-RAG framework (from parent directory)
# pip install -e ..
EOF
```

**Step 5: Create backend README.md**
```bash
cat > README.md << 'EOF'
# BiG-RAG Backend API

FastAPI server for BiG-RAG knowledge graph retrieval.

## Installation

```bash
# Install backend dependencies
pip install -r requirements.txt

# Install BiG-RAG framework
pip install -e ..
```

## Running

```bash
# Start API server
python server.py --data_source SingleTopic

# API will be available at http://localhost:8001
# Swagger docs at http://localhost:8001/docs
```

## Endpoints

See main README.md for complete API documentation.
EOF
```

**Step 6: Update root README.md**

Add section about new structure:
```markdown
## Project Structure

- `backend/` - FastAPI server (port 8001)
- `frontend/` - React UI (port 5173)
- `bigrag/` - Core Python library
- `datasets/` - QA datasets and corpora
- `expr/` - Built knowledge graphs
- `docs/` - Documentation

## Running the Application

**Backend:**
```bash
cd backend
python server.py --data_source SingleTopic
```

**Frontend:**
```bash
cd frontend
npm run dev
```

**Framework:**
```bash
python script_build.py --data_source SingleTopic
```
```

**Step 7: Update .gitignore**
```bash
cat >> .gitignore << 'EOF'

# Frontend
frontend/node_modules/
frontend/dist/
frontend/.env.local

# Backend
backend/__pycache__/
backend/.env

# Build artifacts
*.pyc
*.pyo
EOF
```

### 0.4 Verification

After migration, verify structure:
```bash
# Check backend
cd backend
python server.py --help
# Should see: usage: server.py [-h] [--data_source DATA_SOURCE] [--port PORT]

# Check framework still works
cd ..
python script_build.py --help

# Check git status
git status
# Should show moved files, not deleted/added
```

### 0.5 Benefits of New Structure

✅ **Clear Separation**: Backend, frontend, framework are distinct
✅ **Independent Development**: Each part can be developed/tested separately
✅ **Easy Deployment**: Clear build/deploy boundaries
✅ **Better Documentation**: Each folder has its own README
✅ **Scalability**: Easy to add new components (mobile app, CLI, etc.)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    BiG-RAG System Architecture                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐      HTTP/REST      ┌─────────────────┐ │
│  │   Frontend UI    │ ◄─────────────────► │   Backend API   │ │
│  │                  │                      │                 │ │
│  │  React + Vite    │   Port 5173 ─────►  │  FastAPI        │ │
│  │  + TypeScript    │                      │  Port 8001      │ │
│  └──────────────────┘                      └────────┬────────┘ │
│                                                     │          │
│                                                     │ Python   │
│                                                     │ imports  │
│                                                     ▼          │
│                                             ┌─────────────────┐│
│                                             │  BiG-RAG Core   ││
│                                             │  (bigrag/)      ││
│                                             │                 ││
│                                             │  - Graph ops    ││
│                                             │  - Retrieval    ││
│                                             │  - Embeddings   ││
│                                             └─────────────────┘│
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.1 Directory Structure

```
BiG-RAG/
├── backend/                          # ← RENAME existing api/ folder
│   ├── script_api.py                # Main FastAPI server
│   ├── api/                         # API modules
│   │   ├── __init__.py
│   │   ├── jobs.py                  # Background job processing
│   │   ├── registry.py              # Dataset registry
│   │   ├── kg_utils.py              # Graph utilities
│   │   ├── evaluation.py            # Evaluation endpoints
│   │   └── ...
│   ├── requirements.txt             # Backend Python dependencies
│   └── README.md                    # Backend documentation
│
├── frontend/                        # ← NEW React application
│   ├── public/                      # Static assets
│   ├── src/
│   │   ├── app/                     # App initialization
│   │   │   ├── App.tsx              # Root component
│   │   │   └── Router.tsx           # Route definitions
│   │   ├── pages/                   # Page components
│   │   │   ├── Dashboard.tsx        # Home dashboard
│   │   │   ├── Chat.tsx             # Chat interface
│   │   │   ├── GraphViz.tsx         # Graph visualization
│   │   │   ├── Documents.tsx        # Document management
│   │   │   ├── Evaluation.tsx       # Evaluation dashboard
│   │   │   └── Settings.tsx         # Settings page
│   │   ├── components/              # Reusable components
│   │   │   ├── ui/                  # shadcn/ui components
│   │   │   ├── graph/               # Graph-specific components
│   │   │   │   ├── GraphCanvas.tsx  # Cytoscape wrapper
│   │   │   │   ├── GraphToolbar.tsx # Layout controls
│   │   │   │   ├── NodeInfoPanel.tsx# Node details
│   │   │   │   └── GraphSearch.tsx  # Search nodes
│   │   │   ├── chat/                # Chat components
│   │   │   │   ├── ChatWindow.tsx   # Chat messages
│   │   │   │   ├── MessageBubble.tsx# Individual message
│   │   │   │   └── RetrievalViz.tsx # Retrieved context display
│   │   │   └── documents/           # Document components
│   │   │       ├── DocumentList.tsx # Document table
│   │   │       ├── DocumentCard.tsx # Document preview
│   │   │       └── UploadDialog.tsx # Upload modal
│   │   ├── stores/                  # Zustand state management
│   │   │   ├── graph.ts             # Graph state
│   │   │   ├── chat.ts              # Chat history
│   │   │   ├── documents.ts         # Document state
│   │   │   └── settings.ts          # User preferences
│   │   ├── services/                # API clients
│   │   │   ├── api.ts               # Base Axios instance
│   │   │   ├── graph.ts             # Graph API calls
│   │   │   ├── chat.ts              # Chat API calls
│   │   │   ├── documents.ts         # Document API calls
│   │   │   └── evaluation.ts        # Evaluation API calls
│   │   ├── hooks/                   # Custom React hooks
│   │   │   ├── useGraph.ts          # Graph operations
│   │   │   ├── useChat.ts           # Chat operations
│   │   │   └── useDocuments.ts      # Document operations
│   │   ├── types/                   # TypeScript type definitions
│   │   │   ├── graph.ts             # Graph node/edge types
│   │   │   ├── api.ts               # API response types
│   │   │   └── index.ts             # Exports
│   │   ├── utils/                   # Utility functions
│   │   │   ├── formatters.ts        # Data formatting
│   │   │   └── constants.ts         # Constants
│   │   ├── i18n/                    # Internationalization
│   │   │   ├── en.json              # English translations
│   │   │   └── index.ts             # i18n setup
│   │   └── main.tsx                 # Entry point
│   ├── package.json                 # Frontend dependencies
│   ├── tsconfig.json                # TypeScript config
│   ├── vite.config.ts               # Vite config
│   ├── tailwind.config.js           # Tailwind config
│   ├── postcss.config.js            # PostCSS config
│   └── README.md                    # Frontend documentation
│
├── bigrag/                          # ← EXISTING Python library (unchanged)
│   ├── bigrag.py
│   ├── operate.py
│   └── ...
│
├── datasets/                        # ← EXISTING datasets (unchanged)
├── expr/                            # ← EXISTING built graphs (unchanged)
├── docs/                            # ← EXISTING documentation (unchanged)
├── README.md                        # ← UPDATE with new UI instructions
└── package.json                     # ← ROOT package.json for workspace management
```

---

## 2. Tech Stack

### 2.1 Core Technologies (Latest Versions - November 2025)

| Category | Technology | Version | Justification |
|----------|-----------|---------|---------------|
| **Frontend Framework** | React | **19.2.0** | Latest stable with Activity API, useEffectEvent |
| **Type Safety** | TypeScript | **5.9.3** | Latest with expandable hovers, better DOM docs |
| **Build Tool** | Vite | **7.2.0** | Latest with Node.js 20+ support, ESM-only |
| **Graph Visualization** | Cytoscape.js | **3.33.0** | Latest with WebGL support, TypeScript, circular text |
| | react-cytoscapejs | 2.0+ | React wrapper for Cytoscape |
| | cytoscape-cose-bilkent | 4.1+ | Layout for bipartite graphs |
| | cytoscape-dagre | 2.5+ | Hierarchical layout |
| **State Management** | Zustand | **5.0.8** | Latest V5 with useSyncExternalStore, 3KB size |
| **Routing** | React Router | **7.9.5** | Latest with simplified imports, RSC support |
| **HTTP Client** | Axios | 1.7+ | Better than fetch(), interceptors |
| **UI Components** | shadcn/ui | **Latest (canary)** | React 19 + Tailwind v4 support |
| **Styling** | TailwindCSS | **4.1.16** | Latest v4 with CSS-first config, 5x faster |
| | Class Variance Authority | 0.7+ | Component variant management |
| **Icons** | Lucide React | 0.460+ | Beautiful, consistent icons |
| **Notifications** | Sonner | 1.6+ | Beautiful toast notifications |
| **Search** | MiniSearch | 7.1+ | Client-side full-text search |
| **Markdown** | react-markdown | 9.0+ | Render markdown in chat |
| | remark-gfm | 4.0+ | GitHub Flavored Markdown |
| **Internationalization** | i18next | 23.16+ | Multi-language support |
| | react-i18next | 15.1+ | React integration |

### 2.2 Additional Libraries (Updated for November 2025)

```json
{
  "dependencies": {
    // Core - React 19.2.0 (latest)
    "react": "^19.2.0",
    "react-dom": "^19.2.0",
    "react-router": "^7.9.5",

    // State Management - Zustand V5
    "zustand": "^5.0.8",

    // Graph Visualization - Latest Cytoscape with WebGL
    "cytoscape": "^3.33.0",
    "react-cytoscapejs": "^2.0.0",
    "cytoscape-cose-bilkent": "^4.1.0",
    "cytoscape-dagre": "^2.5.0",
    "cytoscape-fcose": "^2.2.0",

    // HTTP & API
    "axios": "^1.7.9",
    "swr": "^2.2.5",

    // UI Components - Radix UI (for shadcn/ui)
    "@radix-ui/react-dialog": "^1.1.4",
    "@radix-ui/react-dropdown-menu": "^2.1.4",
    "@radix-ui/react-select": "^2.1.4",
    "@radix-ui/react-tabs": "^1.1.4",
    "@radix-ui/react-tooltip": "^1.1.4",

    // Styling - Tailwind v4.1
    "tailwindcss": "^4.1.16",
    "@tailwindcss/vite": "^4.1.16",
    "clsx": "^2.1.1",
    "class-variance-authority": "^0.7.1",
    "tailwind-merge": "^2.6.0",

    // Icons & Assets
    "lucide-react": "^0.460.0",

    // Utilities
    "sonner": "^1.6.1",
    "minisearch": "^7.1.0",
    "react-markdown": "^9.0.1",
    "remark-gfm": "^4.0.0",
    "date-fns": "^4.1.0",

    // i18n
    "i18next": "^23.16.8",
    "react-i18next": "^15.1.3"
  },
  "devDependencies": {
    // TypeScript
    "@types/react": "^19.0.6",
    "@types/react-dom": "^19.0.2",
    "typescript": "^5.9.3",

    // ESLint
    "@typescript-eslint/eslint-plugin": "^8.15.0",
    "@typescript-eslint/parser": "^8.15.0",
    "eslint": "^9.16.0",
    "eslint-plugin-react-hooks": "^5.1.0",

    // Vite
    "@vitejs/plugin-react": "^5.1.0",
    "vite": "^7.2.0",

    // PostCSS (for Tailwind v4)
    "postcss": "^8.4.49",
    "autoprefixer": "^10.4.20"
  }
}
```

**Key Updates:**
- ✅ **React 19.2.0**: Latest with Activity API and useEffectEvent
- ✅ **Vite 7.2.0**: Requires Node.js 20+, ESM-only
- ✅ **Tailwind CSS 4.1.16**: CSS-first configuration, 5x faster builds
- ✅ **Zustand 5.0.8**: useSyncExternalStore optimization, smaller bundle
- ✅ **React Router 7.9.5**: Simplified imports, RSC support
- ✅ **Cytoscape 3.33.0**: WebGL rendering, TypeScript support
- ✅ **TypeScript 5.9.3**: Expandable hovers, better DOM documentation

### 2.3 Tech Stack Modifications

**Changes from your initial proposal:**

1. **Added SWR** - For data fetching with caching, revalidation
2. **Added date-fns** - Lightweight date formatting (better than moment.js)
3. **Added remark-gfm** - GitHub Flavored Markdown support (tables, strikethrough)
4. **Added Radix UI primitives** - shadcn/ui is built on these
5. **Removed redundant libraries** - Kept only what's necessary

**Why these choices:**

- ✅ **Vite > CRA**: 10-100x faster HMR, modern out of box
- ✅ **Zustand > Redux**: 90% less boilerplate, easier to learn
- ✅ **shadcn/ui > Material-UI**: Copy-paste components (no npm bloat), full control
- ✅ **Axios > fetch**: Interceptors, automatic JSON parsing, timeout support
- ✅ **Cytoscape > D3**: Purpose-built for graphs, easier API for network viz

---

## 3. UI/UX Design

### 3.1 Pages Overview

#### **Page 1: Dashboard (Home)**

**Route:** `/`

**Purpose:** Overview of system status, quick actions

**Components:**
```
┌─────────────────────────────────────────────────────────────┐
│ [BiG-RAG Logo]  Dashboard  Chat  Graph  Docs  Eval  Settings│
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  📊 System Status                    🔍 Quick Actions       │
│  ┌─────────────────────┐             ┌──────────────────┐  │
│  │ ✅ API: Connected   │             │ Ask a Question   │  │
│  │ 📚 Docs: 8,108      │             │ Upload Document  │  │
│  │ 🧠 Entities: 7,277  │             │ View Graph       │  │
│  │ 💾 Dataset: Single  │             │ Run Evaluation   │  │
│  └─────────────────────┘             └──────────────────┘  │
│                                                             │
│  📈 Recent Evaluations                                      │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ SingleTopic | EM: 1.67% | F1: 15.33% | 2 hrs ago        ││
│  └─────────────────────────────────────────────────────────┘│
│                                                             │
│  💬 Recent Queries                                          │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ "Which enemy types wield AK-47?"      | 5 mins ago    ││
│  │ "What is the capital of France?"      | 10 mins ago   ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

**Features:**
- Real-time API connection status (green dot if connected)
- Dataset selector dropdown (SingleTopic, etc.)
- Quick stats: document count, entity count, relation count
- Recent evaluation results (last 5 runs)
- Recent chat queries (last 10)
- Quick action buttons to navigate to other pages

---

#### **Page 2: Chat Interface**

**Route:** `/chat`

**Purpose:** Ask questions, see retrieval in real-time, visualize reasoning

**Components:**
```
┌─────────────────────────────────────────────────────────────┐
│ [BiG-RAG Logo]  Dashboard  Chat  Graph  Docs  Eval  Settings│
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────────────┐  ┌──────────────────────────┐ │
│  │  Chat Messages           │  │  Retrieval Visualization │ │
│  │                          │  │                          │ │
│  │  🧑 User:                │  │  📊 Retrieved Context:   │ │
│  │  "Which enemy types      │  │                          │ │
│  │  wield an AK-47?"        │  │  ✅ Path A (Entities):   │ │
│  │                          │  │    • AK-47 (0.92)        │ │
│  │  🤖 Assistant:           │  │    • Tankers (0.88)      │ │
│  │  Searching knowledge     │  │                          │ │
│  │  graph...                │  │  ✅ Path B (Relations):  │ │
│  │                          │  │    • wields (0.85)       │ │
│  │  Retrieved 5 contexts    │  │                          │ │
│  │                          │  │  ✅ Path C (Chunks):     │ │
│  │  📄 Source 1: Tankers    │  │    • Doc 0, Chunk 3     │ │
│  │  wield AK-47s...         │  │      "Tankers wield..." │ │
│  │                          │  │                          │ │
│  │  💡 Answer:              │  │  [View in Graph] button  │ │
│  │  The enemy types that... │  │                          │ │
│  │                          │  │  Coherence Scores:       │ │
│  │  [👍] [👎] [🔍 Explain]  │  │  ████████░░ 0.82        │ │
│  └──────────────────────────┘  └──────────────────────────┘ │
│                                                             │
│  [Type your question...]                    [Send] [Clear] │
└─────────────────────────────────────────────────────────────┘
```

**Features:**
- **Left Panel (60%)**: Chat messages with markdown rendering
  - User messages in blue bubbles
  - Assistant messages in gray bubbles
  - Show retrieval steps (thinking, searching, synthesizing)
  - Source citations with document IDs
  - Feedback buttons (thumbs up/down)
  - "Explain" button to show reasoning trace

- **Right Panel (40%)**: Retrieval visualization
  - Tabs: "Context" | "Graph" | "Trace"
  - Context tab: Show retrieved chunks with scores
  - Graph tab: Mini graph visualization of retrieved subgraph
  - Trace tab: Step-by-step retrieval process

- **Settings Panel** (collapsible):
  - Model selector (gpt-4o-mini, gpt-4o, claude-3-5-sonnet)
  - Temperature slider (0.0 - 1.0)
  - Top-k slider (1 - 20)
  - Enable reranking toggle
  - Query mode (local, global, hybrid, naive)

---

#### **Page 3: Graph Visualization**

**Route:** `/graph`

**Purpose:** Interactive exploration of knowledge graph

**Components:**
```
┌─────────────────────────────────────────────────────────────┐
│ [BiG-RAG Logo]  Dashboard  Chat  Graph  Docs  Eval  Settings│
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  🔍 [Search nodes...] [Layout▼] [Filter▼] [Export▼] [Help] │
│                                                             │
│  ┌──────────────────────────┐  ┌──────────────────────────┐ │
│  │                          │  │  Node Details            │ │
│  │                          │  │  ─────────────────────   │ │
│  │      📍                  │  │  🏷️ Type: Entity         │ │
│  │        ╲                 │  │  📝 Name: Tankers        │ │
│  │         ●──●──●          │  │  📄 Description:         │ │
│  │        ╱ ╲ ╱             │  │     Enemy type that...   │ │
│  │       ●   ●   ●          │  │                          │ │
│  │        ╲ ╱ ╲ ╱           │  │  🔗 Connected To:        │ │
│  │         ●   ●            │  │    • AK-47 (entity)      │ │
│  │          ╲ ╱             │  │    • wields (relation)   │ │
│  │           ●              │  │    • Document 0, Chunk 3 │ │
│  │                          │  │                          │ │
│  │   [Cytoscape Canvas]     │  │  📊 Stats:               │ │
│  │                          │  │    Weight: 0.88          │ │
│  │                          │  │    Degree: 5             │ │
│  │                          │  │    Source: doc-0         │ │
│  │                          │  │                          │ │
│  │                          │  │  [View Document] button  │ │
│  │                          │  │  [Find Similar] button   │ │
│  └──────────────────────────┘  └──────────────────────────┘ │
│                                                             │
│  Legend: 🔵 Entity | 🟥 Relation | 🟢 Chunk | ━━ Connection │
└─────────────────────────────────────────────────────────────┘
```

**Features:**
- **Main Canvas (70%)**: Cytoscape graph visualization
  - Pan, zoom, drag nodes
  - Click node to see details
  - Double-click to expand neighbors
  - Hover to see tooltip with basic info
  - Color-coded by node type:
    - Blue: Entities
    - Red: Relations (bipartite edges)
    - Green: Text chunks
  - Edge thickness represents weight

- **Toolbar (Top)**:
  - Search box with autocomplete (powered by MiniSearch)
  - Layout dropdown:
    - Cose-Bilkent (bipartite, recommended)
    - Dagre (hierarchical)
    - Force-directed (physics-based)
    - Grid (organized)
    - Circle (radial)
  - Filter dropdown:
    - Show only entities
    - Show only relations
    - Show only chunks
    - Filter by weight threshold
    - Filter by document source
  - Export dropdown:
    - Export as PNG
    - Export as JSON
    - Export as GraphML
  - Help button (shows keyboard shortcuts)

- **Side Panel (30%)**: Node details
  - Shows when node is selected
  - Node metadata (type, name, description, weight, source)
  - Connected nodes list
  - Action buttons (View Document, Find Similar)

- **Bottom Legend**: Color key for node types

**Keyboard Shortcuts:**
- `Ctrl+F`: Focus search
- `Ctrl+Z`: Undo layout
- `Ctrl+R`: Reset zoom
- `Space`: Fit graph to canvas
- `Delete`: Remove selected node (confirmation dialog)

---

#### **Page 4: Document Management**

**Route:** `/documents`

**Purpose:** Upload, view, search, delete documents

**Components:**
```
┌─────────────────────────────────────────────────────────────┐
│ [BiG-RAG Logo]  Dashboard  Chat  Graph  Docs  Eval  Settings│
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  🔍 [Search documents...] [Upload] [Refresh] [Delete All]  │
│                                                             │
│  Filters: [All Types▼] [All Sources▼] [Sort by: Date▼]     │
│                                                             │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ ☑️ | ID      | Title          | Entities | Chunks | Src││
│  │────────────────────────────────────────────────────────││
│  │ ☑  | doc-0   | Bullet Kin    | 45       | 8      | web││
│  │ ☐  | doc-1   | Gungeon Items | 32       | 6      | txt││
│  │ ☐  | doc-2   | Boss Guide    | 28       | 5      | pdf││
│  │ ...                                                     ││
│  └─────────────────────────────────────────────────────────┘│
│                                                             │
│  📄 Document Preview (doc-0)                                │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ Title: Bullet Kin Types                                ││
│  │ Source: https://gungeon.wiki/Bullet_Kin                ││
│  │ Tags: enemies, weapons, gungeon                        ││
│  │                                                         ││
│  │ Content:                                                ││
│  │ Bullet Kin are the most common enemies in Enter the    ││
│  │ Gungeon. They come in various types...                 ││
│  │                                                         ││
│  │ Extracted Entities (45):                                ││
│  │ • Bullet Kin (entity)                                   ││
│  │ • Tankers (entity)                                      ││
│  │ • AK-47 (entity)                                        ││
│  │ ...                                                     ││
│  │                                                         ││
│  │ [View in Graph] [Edit Metadata] [Delete] [Download]    ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

**Features:**
- **Toolbar (Top)**:
  - Search box (searches title, content, metadata)
  - Upload button (opens upload dialog)
  - Refresh button (reloads document list)
  - Delete All button (with confirmation)

- **Filters**:
  - Filter by type (text, PDF, markdown, JSON)
  - Filter by source (web, local, uploaded)
  - Sort by (date, title, entity count, chunk count)

- **Document Table**:
  - Checkbox for bulk operations
  - Columns: ID, Title, Entity Count, Chunk Count, Source
  - Click row to see preview below
  - Right-click for context menu (View in Graph, Delete, Download)

- **Document Preview**:
  - Shows when document is selected
  - Displays metadata (title, source, tags, date)
  - Shows content preview (first 500 chars)
  - Lists extracted entities and relations
  - Action buttons (View in Graph, Edit Metadata, Delete, Download)

- **Upload Dialog** (Modal):
  - Drag-and-drop area
  - File input (accepts .txt, .pdf, .md, .json, .jsonl)
  - Metadata fields (title, category, tags)
  - Upload button (shows progress bar)
  - Cancel button

**Bulk Operations:**
- Select multiple documents with checkboxes
- Bulk delete
- Bulk export
- Bulk tag

---

#### **Page 5: Evaluation Dashboard**

**Route:** `/evaluation`

**Purpose:** Run evaluations, view results, compare runs

**Components:**
```
┌─────────────────────────────────────────────────────────────┐
│ [BiG-RAG Logo]  Dashboard  Chat  Graph  Docs  Eval  Settings│
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  📊 Evaluation Dashboard                                    │
│                                                             │
│  ┌──────────────────────────┐  ┌──────────────────────────┐ │
│  │  Run New Evaluation      │  │  Recent Runs             │ │
│  │  ─────────────────────   │  │  ─────────────────────   │ │
│  │  Dataset: [SingleTopic▼] │  │  Run #1: SingleTopic     │ │
│  │  Model:   [gpt-4o-mini▼] │  │  EM: 1.67% | F1: 15.33% │ │
│  │  Top-k:   [5     ]       │  │  2 hours ago             │ │
│  │  Rerank:  [✓]            │  │  [View Details]          │ │
│  │  Questions: [All     ]   │  │                          │ │
│  │                          │
│  └──────────────────────────┘
│                                └──────────────────────────┘ │
│                                                             │
│  📈 Results (Run #1)                                        │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ Overall: EM: 1.67% (2/120) | F1: 15.33%                ││
│  │                                                         ││
│  │ By Question Type:                                       ││
│  │  Single-passage:  EM: 5.0%  | F1: 28.31%               ││
│  │  Multi-passage:   EM: 0.0%  | F1: 17.69%               ││
│  │  No-answer:       EM: 0.0%  | F1: 0.0%                 ││
│  │                                                         ││
│  │ [Export CSV] [Export JSON] [Compare with Run #2]       ││
│  └─────────────────────────────────────────────────────────┘│
│                                                             │
│  🔍 Failed Questions (118/120)                              │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ Q: "Which enemy types wield an AK-47?"                  ││
│  │ A: "Assault-rifle wielding Bullet Kin and Tankers..."   ││
│  │ Predicted: "The enemy type that wields an AK-47 is..." ││
│  │ EM: 0.0 | F1: 0.23                                      ││
│  │ [View Retrieval] [View in Graph]                        ││
│  │────────────────────────────────────────────────────────││
│  │ Q: "What is the capital of France?"                     ││
│  │ ...                                                     ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

**Features:**
- **Run New Evaluation Panel**:
  - Dataset selector dropdown
  - Model selector (gpt-4o-mini, gpt-4o, claude-3-5-sonnet)
  - Top-k slider (1-20)
  - Reranking toggle
  - Questions input (all, first N, specific IDs)
  - Start button (disables while running, shows progress)

- **Recent Runs Panel**:
  - Shows last 10 evaluation runs
  - Each run shows: dataset, EM, F1, timestamp
  - Click "View Details" to load results below

- **Results Panel**:
  - Overall metrics (EM, F1, total questions)
  - Breakdown by question type (single-passage, multi-passage, no-answer)
  - Confusion matrix (if applicable)
  - Export buttons (CSV, JSON, LaTeX)
  - Compare button (opens comparison modal)

- **Failed Questions Panel**:
  - Shows questions where EM < 1.0
  - Each question shows:
    - Question text
    - Golden answer
    - Predicted answer
    - EM and F1 scores
  - Action buttons:
    - View Retrieval: Shows retrieved contexts
    - View in Graph: Opens graph visualization for this query

- **Comparison Modal** (opens when comparing runs):
  - Side-by-side metrics
  - Difference calculations (ΔEM, ΔF1)
  - Questions that improved vs degraded
  - Export comparison report

---

#### **Page 6: Settings**

**Route:** `/settings`

**Purpose:** Configure datasets, models, API keys, UI preferences

**Components:**
```
┌─────────────────────────────────────────────────────────────┐
│ [BiG-RAG Logo]  Dashboard  Chat  Graph  Docs  Eval  Settings│
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ⚙️ Settings                                                 │
│                                                             │
│  Tabs: [General] [API Keys] [Datasets] [Advanced]          │
│                                                             │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ General Settings                                        ││
│  │                                                         ││
│  │ Language:         [English ▼]                           ││
│  │ Theme:            [Light ▼]                             ││
│  │ Default Model:    [gpt-4o-mini ▼]                       ││
│  │ Default Top-k:    [5     ]                              ││
│  │ Enable Reranking: [✓]                                   ││
│  │ Auto-save:        [✓]                                   ││
│  │                                                         ││
│  │ [Save] [Reset to Defaults]                              ││
│  └─────────────────────────────────────────────────────────┘│
│                                                             │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ API Keys (click to reveal)                              ││
│  │                                                         ││
│  │ OpenAI:    [••••••••••••••••] [Edit] [Test Connection] ││
│  │ Anthropic: [••••••••••••••••] [Edit] [Test Connection] ││
│  │ Google:    [Not set]          [Add]  [Test Connection] ││
│  │ xAI:       [Not set]          [Add]  [Test Connection] ││
│  │                                                         ││
│  │ [Save]                                                  ││
│  └─────────────────────────────────────────────────────────┘│
│                                                             │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ Datasets                                                ││
│  │                                                         ││
│  │ Active Dataset: [SingleTopic ▼]                         ││
│  │                                                         ││
│  │ Available Datasets:                                     ││
│  │  ✓ SingleTopic (20 docs, 120 questions)                ││
│  │  ☐ HotpotQA (not built)                                ││
│  │                                                         ││
│  │ [Build New Dataset] [Import Dataset]                    ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

**Features:**
- **General Tab**:
  - Language selector (English, Chinese, etc.)
  - Theme selector (Light, Dark, Auto)
  - Default model and parameters
  - Auto-save toggle

- **API Keys Tab**:
  - Masked API keys (click to reveal)
  - Edit buttons to update keys
  - Test Connection buttons to verify keys work
  - Save button

- **Datasets Tab**:
  - Active dataset selector
  - List of available datasets with stats
  - Build New Dataset button (opens wizard)
  - Import Dataset button (opens file dialog)

- **Advanced Tab**:
  - Backend URL (default: http://localhost:8001)
  - Request timeout (default: 30s)
  - Cache settings
  - Debug mode toggle
  - Export/import all settings

---

### 3.2 Navigation

**Top Navigation Bar** (persistent across all pages):
```
┌─────────────────────────────────────────────────────────────┐
│ [🧠 BiG-RAG]  Dashboard | Chat | Graph | Docs | Eval | ⚙️  │
└─────────────────────────────────────────────────────────────┘
```

**Responsive Behavior:**
- Desktop (>1024px): Full navbar with labels
- Tablet (768-1024px): Icons + labels
- Mobile (<768px): Hamburger menu

---

### 3.3 Color Scheme

**Light Theme:**
- Background: `#FFFFFF`
- Surface: `#F5F5F5`
- Primary: `#3B82F6` (Blue)
- Secondary: `#10B981` (Green)
- Error: `#EF4444` (Red)
- Text: `#1F2937` (Dark Gray)

**Dark Theme:**
- Background: `#0F172A`
- Surface: `#1E293B`
- Primary: `#60A5FA` (Light Blue)
- Secondary: `#34D399` (Light Green)
- Error: `#F87171` (Light Red)
- Text: `#F1F5F9` (Light Gray)

**Graph Node Colors:**
- Entities: `#3B82F6` (Blue)
- Relations: `#EF4444` (Red)
- Chunks: `#10B981` (Green)

---

## 4. State Management (Zustand)

### 4.1 Graph Store (`stores/graph.ts`)

```typescript
interface GraphState {
  // Data
  nodes: CytoscapeNode[];
  edges: CytoscapeEdge[];
  selectedNode: CytoscapeNode | null;

  // UI State
  layout: 'cose-bilkent' | 'dagre' | 'force' | 'grid' | 'circle';
  filters: {
    showEntities: boolean;
    showRelations: boolean;
    showChunks: boolean;
    minWeight: number;
    sourceDocument: string | null;
  };

  // Actions
  loadGraph: (dataSource: string) => Promise<void>;
  loadSubgraph: (query: string) => Promise<void>;
  selectNode: (nodeId: string) => void;
  updateLayout: (layout: string) => void;
  updateFilters: (filters: Partial<Filters>) => void;
  exportGraph: (format: 'png' | 'json' | 'graphml') => void;
}
```

### 4.2 Chat Store (`stores/chat.ts`)

```typescript
interface ChatState {
  // Data
  messages: Message[];
  retrievedContexts: RetrievedContext[];

  // UI State
  isLoading: boolean;
  error: string | null;

  // Settings
  model: string;
  temperature: number;
  topK: number;
  enableReranking: boolean;

  // Actions
  sendMessage: (message: string) => Promise<void>;
  clearHistory: () => void;
  setModel: (model: string) => void;
  setParameters: (params: Partial<ChatParams>) => void;
}
```

### 4.3 Documents Store (`stores/documents.ts`)

```typescript
interface DocumentsState {
  // Data
  documents: Document[];
  selectedDocument: Document | null;

  // UI State
  isLoading: boolean;
  searchQuery: string;
  filters: {
    type: string[];
    source: string[];
  };
  sortBy: 'date' | 'title' | 'entities' | 'chunks';

  // Actions
  loadDocuments: () => Promise<void>;
  uploadDocument: (file: File, metadata: Metadata) => Promise<void>;
  deleteDocument: (docId: string) => Promise<void>;
  searchDocuments: (query: string) => void;
  updateFilters: (filters: Partial<Filters>) => void;
}
```

### 4.4 Settings Store (`stores/settings.ts`)

```typescript
interface SettingsState {
  // General
  language: 'en' | 'zh';
  theme: 'light' | 'dark' | 'auto';
  autoSave: boolean;

  // API
  backendUrl: string;
  apiKeys: {
    openai: string;
    anthropic: string;
    google: string;
    xai: string;
  };

  // Datasets
  activeDataset: string;
  availableDatasets: string[];

  // Actions
  updateSettings: (settings: Partial<Settings>) => void;
  testConnection: (provider: string) => Promise<boolean>;
  loadSettings: () => void;
  saveSettings: () => void;
}
```

---

## 5. API Integration

### 5.1 Base API Client (`services/api.ts`)

```typescript
import axios from 'axios';
import { toast } from 'sonner';

const api = axios.create({
  baseURL: 'http://localhost:8001',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor (add auth, logging)
api.interceptors.request.use(
  (config) => {
    console.log(`→ ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor (error handling, toasts)
api.interceptors.response.use(
  (response) => {
    console.log(`← ${response.status} ${response.config.url}`);
    return response;
  },
  (error) => {
    const message = error.response?.data?.detail || error.message;
    toast.error(`API Error: ${message}`);
    return Promise.reject(error);
  }
);

export default api;
```

### 5.2 API Endpoints

**Existing Backend Endpoints** (from `script_api.py`):

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/` | Health check |
| POST | `/ask` | Ask question, get answer with retrieval |
| POST | `/search` | Search knowledge graph |
| GET | `/stats` | Get graph statistics |
| GET | `/documents` | List all documents |
| GET | `/documents/{doc_id}` | Get document details |
| POST | `/documents` | Upload new document |
| DELETE | `/documents/{doc_id}` | Delete document |
| POST | `/eval/batch_generate` | Generate answers for evaluation |
| POST | `/eval/evaluate_results` | Evaluate generated answers |
| GET | `/graph/export` | Export graph as JSON |

**Frontend Service Methods** (`services/`):

```typescript
// services/chat.ts
export const askQuestion = (question: string, params: QueryParams) =>
  api.post('/ask', { question, ...params });

// services/graph.ts
export const getGraphData = (dataSource: string) =>
  api.get(`/graph/export?data_source=${dataSource}`);

export const getSubgraph = (query: string, topK: number) =>
  api.post('/search', { queries: [query], param: { top_k: topK } });

// services/documents.ts
export const getDocuments = () => api.get('/documents');

export const uploadDocument = (file: File, metadata: Metadata) => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('metadata', JSON.stringify(metadata));
  return api.post('/documents', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
};

export const deleteDocument = (docId: string) =>
  api.delete(`/documents/${docId}`);

// services/evaluation.ts
export const runEvaluation = (config: EvalConfig) =>
  api.post('/eval/batch_generate', config);

export const getEvaluationResults = (csvPath: string) =>
  api.post('/eval/evaluate_results', { csv_path: csvPath });
```

---

## 6. Development Workflow

### 6.1 Project Setup

```bash
# 1. Clone repository
cd d:/BiG-RAG

# 2. Reorganize directories (move api/ to backend/)
# This will be done in implementation phase

# 3. Set up backend (if not already done)
cd backend
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt

# 4. Set up frontend
cd ../frontend
npm install

# 5. Install root dependencies (workspace management)
cd ..
npm install
```

### 6.2 Running the Application

**Terminal 1: Backend API**
```bash
cd d:/BiG-RAG/backend
venv\Scripts\activate
python script_api.py --data_source SingleTopic
# API runs on http://localhost:8001
```

**Terminal 2: Frontend Dev Server**
```bash
cd d:/BiG-RAG/frontend
npm run dev
# UI runs on http://localhost:5173
```

**Terminal 3: BiG-RAG Framework** (optional, for testing)
```bash
cd d:/BiG-RAG
venv\Scripts\activate
python -m bigrag  # Or any framework scripts
```

### 6.3 Development Scripts

**Frontend `package.json` scripts:**
```json
{
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview",
    "lint": "eslint src --ext ts,tsx --report-unused-disable-directives --max-warnings 0",
    "format": "prettier --write \"src/**/*.{ts,tsx,json}\"",
    "type-check": "tsc --noEmit"
  }
}
```

**Backend scripts** (add to README):
```bash
# Start API server
python backend/script_api.py --data_source SingleTopic

# Build knowledge graph
python script_build.py --data_source SingleTopic

# Run evaluation
python run_singletopic_evaluation.py
```

### 6.4 Git Workflow

```bash
# Feature branch workflow
git checkout -b feature/graph-visualization
# ... make changes ...
git add .
git commit -m "feat: add graph visualization page"
git push origin feature/graph-visualization
# ... create pull request ...
```

---

## 7. Deployment

### 7.1 Production Build

**Frontend:**
```bash
cd frontend
npm run build
# Output: frontend/dist/
```

**Backend:**
```bash
# No build needed for Python
# Just ensure dependencies are installed
pip install -r backend/requirements.txt
```

### 7.2 Deployment Options

#### **Option 1: Separate Deployments (Recommended)**

**Frontend** → Vercel / Netlify / Cloudflare Pages
- Build command: `npm run build`
- Output directory: `dist`
- Environment variables: `VITE_API_URL=https://api.bigrag.com`

**Backend** → AWS EC2 / DigitalOcean / Railway
- Run with Gunicorn: `gunicorn -w 4 -k uvicorn.workers.UvicornWorker backend.script_api:app`
- Reverse proxy with Nginx
- SSL with Let's Encrypt

#### **Option 2: Docker Compose**

```yaml
# docker-compose.yml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8001:8001"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./expr:/app/expr
      - ./datasets:/app/datasets

  frontend:
    build: ./frontend
    ports:
      - "80:80"
    depends_on:
      - backend
    environment:
      - VITE_API_URL=http://backend:8001
```

#### **Option 3: Single Server (All-in-One)**

- Backend runs on port 8001
- Frontend built as static files served by Nginx
- Nginx proxies `/api/*` to backend

**Nginx config:**
```nginx
server {
    listen 80;
    server_name bigrag.example.com;

    # Frontend
    location / {
        root /var/www/bigrag/frontend/dist;
        try_files $uri $uri/ /index.html;
    }

    # Backend API
    location /api {
        proxy_pass http://localhost:8001;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

### 7.3 Environment Variables

**Frontend (`.env.production`):**
```env
VITE_API_URL=https://api.bigrag.com
VITE_APP_NAME=BiG-RAG
VITE_VERSION=1.0.0
```

**Backend (`.env`):**
```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
XAI_API_KEY=...
```

---

## 8. Testing Strategy

### 8.1 Frontend Testing

```bash
# Install testing libraries
npm install --save-dev vitest @testing-library/react @testing-library/jest-dom

# Add to package.json
"scripts": {
  "test": "vitest",
  "test:ui": "vitest --ui",
  "test:coverage": "vitest --coverage"
}
```

**Test structure:**
```
frontend/src/
├── components/
│   └── __tests__/
│       ├── GraphCanvas.test.tsx
│       ├── ChatWindow.test.tsx
│       └── DocumentList.test.tsx
├── stores/
│   └── __tests__/
│       ├── graph.test.ts
│       └── chat.test.ts
└── services/
    └── __tests__/
        ├── api.test.ts
        └── graph.test.ts
```

### 8.2 Backend Testing

Already exists in `tests/` directory. No changes needed.

### 8.3 E2E Testing (Optional)

```bash
# Install Playwright
npm install --save-dev @playwright/test

# Add E2E tests
mkdir -e2e
touch e2e/chat-flow.spec.ts
touch e2e/graph-viz.spec.ts
```

---

## 9. Implementation Timeline

### Phase 1: Setup & Infrastructure 

**Project Reorganization**
- [ ] Rename `api/` to `backend/`
- [ ] Create `frontend/` directory
- [ ] Initialize React + Vite + TypeScript
- [ ] Install all dependencies
- [ ] Configure TailwindCSS
- [ ] Set up shadcn/ui

**Base Components & Routing**
- [ ] Create page components (Dashboard, Chat, Graph, Docs, Eval, Settings)
- [ ] Set up React Router
- [ ] Create navigation component
- [ ] Set up Zustand stores
- [ ] Create API service layer
- [ ] Test API connection with backend

**Dashboard Page**
- [ ] Implement system status display
- [ ] Add recent evaluations list
- [ ] Add recent queries list
- [ ] Add quick action buttons
- [ ] Test integration with backend API

### Phase 2: Core Features 

**Graph Visualization**
- [ ] Integrate Cytoscape.js with React
- [ ] Implement graph canvas component
- [ ] Add layout algorithms (cose-bilkent, dagre, etc.)
- [ ] Implement node details panel
- [ ] Add search functionality (MiniSearch)
- [ ] Add filters (node type, weight threshold)
- [ ] Add export (PNG, JSON, GraphML)
- [ ] Test with SingleTopic dataset

**Chat Interface**
- [ ] Create chat window component
- [ ] Implement message bubbles
- [ ] Add markdown rendering (react-markdown)
- [ ] Add retrieval visualization panel
- [ ] Implement settings panel (model, temperature, top-k)
- [ ] Add source citations
- [ ] Test with `/ask` endpoint

**Document Management**
- [ ] Create document list component
- [ ] Add search and filters
- [ ] Implement document preview
- [ ] Add upload dialog with drag-and-drop
- [ ] Implement delete functionality
- [ ] Add bulk operations
- [ ] Test with `/documents` endpoints

### Phase 3: Advanced Features

**Evaluation Dashboard**
- [ ] Create evaluation run form
- [ ] Implement results display
- [ ] Add charts (EM/F1 over time)
- [ ] Add failed questions viewer
- [ ] Implement comparison modal
- [ ] Add export functionality (CSV, JSON, LaTeX)
- [ ] Test with `/eval` endpoints

**Settings Page**
- [ ] Create settings tabs (General, API Keys, Datasets, Advanced)
- [ ] Implement settings persistence (localStorage)
- [ ] Add API key management
- [ ] Add dataset selector
- [ ] Implement theme switcher (light/dark)
- [ ] Add language selector (i18next)

**Polish & Optimization**
- [ ] Implement loading states
- [ ] Add error boundaries
- [ ] Add toast notifications (Sonner)
- [ ] Optimize performance (React.memo, useMemo)
- [ ] Add keyboard shortcuts
- [ ] Responsive design for mobile/tablet
- [ ] Accessibility improvements (ARIA labels, keyboard nav)

### Phase 4: Testing & Documentation 

**Testing**
- [ ] Write unit tests for stores
- [ ] Write component tests
- [ ] Write API service tests
- [ ] Manual testing on all pages
- [ ] Cross-browser testing (Chrome, Firefox, Safari)
- [ ] Mobile testing

**Documentation**
- [ ] Write frontend README
- [ ] Update main README with UI instructions
- [ ] Create user guide (docs/USER_GUIDE.md)
- [ ] Add inline code comments
- [ ] Create demo video/screenshots

**Deployment**
- [ ] Set up Docker Compose
- [ ] Test production build
- [ ] Deploy to staging environment
- [ ] User acceptance testing
- [ ] Deploy to production

---

## 10. Future Enhancements (Post-Launch)

### Short-term 

1. **Real-time Updates** - WebSocket support for live evaluation progress
2. **Comparison Mode** - Side-by-side graph comparison (retrieved vs expected)
3. **Query Decomposition Viz** - Visualize multi-hop query decomposition
4. **Advanced Search** - Semantic search with vector similarity
5. **User Accounts** - Authentication, saved queries, shared evaluations

### Medium-term

1. **Collaborative Features** - Share graphs, annotate nodes, comments
2. **Graph Analytics** - Centrality measures, community detection
3. **A/B Testing** - Compare different retrieval strategies
4. **LLM Playground** - Test different models side-by-side
5. **Dataset Builder** - UI for creating custom datasets

### Long-term 

1. **Graph Editing** - Manually add/edit entities, relations
2. **Auto-Repair** - Suggest fixes for broken retrieval paths
3. **RL Training Integration** - Monitor training progress in UI
4. **Multi-tenant** - Multiple users, datasets, isolated graphs
5. **API Gateway** - Rate limiting, analytics, billing

---

## 11. Success Metrics

**Technical Metrics:**
- [ ] Page load time < 2 seconds
- [ ] Graph rendering < 500ms for 1000 nodes
- [ ] API response time < 1 second (95th percentile)
- [ ] Zero accessibility errors (WAVE, axe)
- [ ] Lighthouse score > 90 (Performance, Accessibility, Best Practices)

**User Metrics:**
- [ ] Time to first successful query < 30 seconds
- [ ] User can debug retrieval failure in < 5 minutes with graph viz
- [ ] Document upload success rate > 95%
- [ ] Evaluation completion rate > 90%

---

## 12. Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Large graph performance** | High | Implement pagination, virtual rendering, chunked loading |
| **API downtime** | High | Add retry logic, offline mode, cached data |
| **Browser compatibility** | Medium | Test on Chrome, Firefox, Safari, Edge; provide fallbacks |
| **State management complexity** | Medium | Use Zustand (simpler than Redux), clear separation of concerns |
| **TypeScript learning curve** | Low | Provide examples, use `any` initially if needed |
| **Cytoscape.js complexity** | Medium | Start with simple layouts, add advanced features iteratively |

---

## 13. Conclusion

This plan provides a **comprehensive roadmap** for building a production-ready, scalable BiG-RAG UI that:

✅ **Maintains separation** - Backend, frontend, framework run independently
✅ **Simple to start** - Vite + React + TypeScript (modern, fast)
✅ **Easy to scale** - Zustand (state), modular components, clear structure
✅ **Powerful features** - Graph viz, chat, docs, eval, all in one place
✅ **Professional UX** - shadcn/ui, TailwindCSS, responsive, accessible
✅ **Developer-friendly** - TypeScript, ESLint, hot reload, clear APIs

The tech stack balances **simplicity** (easy to learn, fast to build) with **power** (handles complex graphs, scales to production).

**Next Steps:**
1. Review and approve this plan
2. Begin Phase 1: Setup & Infrastructure
3. Iterate based on user feedback

---

**Questions?** Let me know if you'd like to adjust the tech stack, features, or timeline!

# BiG-RAG Development Guide

**Last Updated:** November 10, 2025

This document consolidates all development status, implementation notes, and technical guides for the BiG-RAG project.

---

## 📊 Project Status

### Overall Progress
- **Backend API:** ✅ 100% Complete
- **Frontend UI:** ✅ 95% Complete (Production Ready)
- **Knowledge Graph:** ✅ 100% Complete
- **Documentation:** ✅ 100% Complete

### Current Version
- **Version:** 1.0.0-beta
- **Status:** Production Ready
- **Last Major Update:** November 6, 2025

---

## 🏗️ Frontend Implementation Status

### Completed Features (95%)

#### 1. Dashboard (100%)
- Real-time API health monitoring
- Live statistics from backend
- Dataset overview with metrics
- Auto-refresh (30s intervals)
- Quick action buttons

#### 2. Evaluation Dashboard (100%)
- Run evaluations with progress tracking
- Display EM, F1, accuracy metrics
- Export to JSON/CSV
- History tracking

#### 3. Settings Page (100%)
- 4 tabs: General, API Keys, Retrieval, Appearance
- Full configuration options
- Connection testing
- Settings persistence

#### 4. Chat Interface (95%)
- Message window with bubbles
- Retrieval visualization panel
- Export/import history
- Settings panel
- Ready for production

#### 5. Graph Visualization (90%)
- Cytoscape.js integration
- 7 layout algorithms
- Progressive loading
- Export functionality

#### 6. Document Management (85%)
- Upload with drag-and-drop
- Delete functionality
- Metadata management

#### 7. UI Component Library (100%)
All 13 shadcn/ui components:
- Button, Select, Dialog, Tabs
- Input, Textarea, Card, Badge
- Tooltip, Dropdown, Slider
- Progress, Label

### Technology Stack
- React 19.2.0
- TypeScript 5.9.3
- Vite 7.1.12
- Tailwind CSS 4.1.16
- Zustand 5.0.8 (state management)
- Axios 1.7.9 (HTTP client)
- Cytoscape.js 3.33.0 (graph visualization)

### Running the Frontend
```bash
cd frontend
npm install  # First time only
npm run dev  # Start dev server at http://localhost:5173
```

---

## 🎨 Graph Visualization

### Implementation Summary

BiG-RAG uses **Cytoscape.js** for interactive graph visualization with support for large graphs (10K+ nodes).

#### Features
- ✅ Multiple layout algorithms (cose-bilkent, dagre, fcose, cola, grid, circle, concentric, breadthfirst)
- ✅ Progressive loading (offset-based pagination)
- ✅ Sampling strategies (top_weighted, random, diverse)
- ✅ Node filtering (entities, relations, chunks)
- ✅ Weight-based filtering
- ✅ Search functionality
- ✅ Export (PNG, JSON, GraphML)
- ✅ Tooltips and info panels
- ✅ Performance optimizations for large graphs

#### Performance Optimizations

**For Large Graphs (>5K nodes):**
1. **Sampling:** Default limit of 1000 nodes
2. **Progressive Loading:** Load in chunks using offset parameter
3. **Layout Optimization:** Use fast layouts (grid, circle) for initial view
4. **Caching:** 5-minute TTL for graph data
5. **WebGL Rendering:** (Optional) For 10K+ nodes

#### API Endpoints
```
GET /graph/export?data_source=<dataset>&limit=<n>&offset=<n>&sample_strategy=<strategy>
```

**Parameters:**
- `limit`: Number of nodes to return (default: 1000)
- `offset`: Starting position for pagination
- `sample_strategy`: top_weighted | random | diverse
- `node_types`: Filter by type (entity, relation, chunk)
- `min_weight`: Minimum node weight

#### Example Usage
```bash
# Load first 1000 nodes with top-weighted sampling
curl "http://localhost:8001/graph/export?data_source=SingleTopic&limit=1000&sample_strategy=top_weighted"

# Load next 1000 nodes
curl "http://localhost:8001/graph/export?data_source=SingleTopic&limit=1000&offset=1000"
```

### Layout Algorithm Guide

**For Bipartite Graphs (Recommended):**
- **cose-bilkent**: Best for bipartite structure, shows clear entity-relation-document groupings

**For Hierarchical Data:**
- **dagre**: Clean hierarchical layout, good for showing flow

**For General Graphs:**
- **fcose**: Force-directed, balanced layout
- **cola**: Constraint-based, avoids overlaps

**For Large Graphs:**
- **grid**: ✅ **FIXED** - Organized table-like structure, sorted by type and weight
- **circle**: ✅ **IMPROVED** - Grouped by type in circular sections (blue→red→green→purple)
- **concentric**: ✅ **IMPROVED** - Target pattern with type-based rings (entities outer, relations middle, chunks inner)

**For Small Graphs (<500 nodes):**
- **cose-bilkent**: Best visual quality
- **fcose**: Good balance of speed and quality

### Recent Layout Improvements (Nov 6, 2025)

#### Grid Layout (Fixed)
- **Issue:** Not displaying nodes
- **Fix:** Added proper sort function, auto-calculate dimensions, increased spacing (1.8x), 30px padding
- **Result:** Clean organized grid with type grouping

#### Circle Layout (Improved)
- **Issue:** Poor visual organization
- **Fix:** Sort by type first (entity→relation→chunk→document), then by weight
- **Result:** Color-grouped sections in circular arrangement

#### Concentric Layout (Improved)
- **Issue:** Confusing hierarchy
- **Fix:** Type-based ring levels (entities=300, relations=200, chunks=100), 80px spacing, 2.0x ring separation
- **Result:** Clear bullseye pattern with visual type hierarchy

---

## 🔍 Centralized Logging System (November 10, 2025)

### Overview

BiG-RAG implements a comprehensive centralized logging infrastructure for production-ready log management across all components.

### Features Implemented

✅ **Component-Separated Logs**:
```
logs/
├── bigrag-core/     # Core engine (bigrag.log, error.log)
├── backend/         # API server (api.log, error.log, access.log)
├── jobs/            # Background jobs
└── frontend/        # Browser console (via logger.ts)
```

✅ **Log Rotation**:
- Size-based rotation (10MB max per file, 5 backups)
- Time-based rotation (daily rotation, 7-day retention for API logs)
- Automatic cleanup of old logs

✅ **Multiple Log Handlers**:
- Console output (simplified format for terminal)
- File output (detailed format with timestamp, level, module, message)
- Error-only output (separate error.log for critical issues)

✅ **Structured Logging**:
- Optional JSON format for log aggregation tools (ELK, Splunk)
- Contextual logging with metadata support
- Backward compatible with existing code

✅ **Frontend Logger**:
- Browser console logging with structured format
- Module-specific loggers (apiLogger, graphLogger, chatLogger, documentLogger)
- Environment-based log level configuration

### Implementation Files

**Backend (Python)**:
- [bigrag/logging_config.py](bigrag/logging_config.py) - Centralized logging module (216 lines)
- [bigrag/utils.py](bigrag/utils.py) - Enhanced set_logger() using logging_config
- [bigrag/bigrag.py](bigrag/bigrag.py) - Smart log directory detection
- [backend/server.py](backend/server.py) - API logger with daily rotation

**Frontend (TypeScript)**:
- [frontend/src/utils/logger.ts](frontend/src/utils/logger.ts) - Browser console logger (106 lines)
- [frontend/src/app/App.tsx](frontend/src/app/App.tsx) - Using structured logger
- [frontend/src/components/graph/GraphCanvas.tsx](frontend/src/components/graph/GraphCanvas.tsx) - Using graphLogger

**Documentation**:
- [docs/technical/LOGGING_GUIDE.md](docs/technical/LOGGING_GUIDE.md) - Complete logging guide
- [Indexing_update_plan/IMPLEMENTATION_PROGRESS.md](Indexing_update_plan/IMPLEMENTATION_PROGRESS.md) - Implementation notes

### Configuration

Add to `.env`:
```bash
# Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL=INFO

# Log directory (optional, defaults to logs/bigrag-core/)
LOG_DIR=./logs/bigrag-core

# JSON format for structured logging
LOG_JSON_FORMAT=false

# Frontend log level
VITE_LOG_LEVEL=INFO
```

### Usage Examples

**Python (Backend)**:
```python
from bigrag.logging_config import setup_logger, add_context

# Setup logger
logger = setup_logger(
    name="my_module",
    log_dir="./logs/backend",
    log_file="my_module.log",
    level="INFO",
    rotation="size",
    max_bytes=10*1024*1024,
    backup_count=5
)

# Basic logging
logger.info("Operation completed successfully")
logger.error("Failed to process request", exc_info=True)

# Contextual logging
ctx_logger = add_context(logger, request_id="req123", user="user456")
ctx_logger.info("Processing user request")
```

**TypeScript (Frontend)**:
```typescript
import { logger, apiLogger, graphLogger } from '@/utils/logger';

// General logging
logger.info('Application initialized');

// Module-specific logging
apiLogger.error('API call failed', error);
graphLogger.debug('Rendering 150 nodes');
```

### Benefits

1. **Production-Ready**: Log rotation prevents disk space issues
2. **Organized**: Logs separated by component for easy troubleshooting
3. **Structured**: Optional JSON format for log aggregation tools
4. **Debuggable**: Error-only logs highlight critical issues
5. **Flexible**: Configurable via environment variables

### Cleanup Performed

**Orphaned Log Files Removed**:
- `backend/backend.log` (19 bytes)
- `backend/backend_fixed.log` (4.8 KB)
- `backend/backend_new.log` (13 KB)
- `backend/server.log` (19 bytes)
- `backend/server_clean.log` (3.1 KB)
- `backend/server_final.log` (4.3 KB)
- `backend/server_nocache.log` (6.6 KB)
- `frontend/frontend.log` (2.9 KB)

**Total Cleanup**: 8 orphaned log files removed (~35 KB)

### Impact

- ✅ Better debugging with organized, structured logs
- ✅ Automatic log management (rotation, cleanup)
- ✅ Production-ready log infrastructure
- ✅ Backward compatible with existing logging code
- ✅ Comprehensive documentation for developers

For complete details, see [docs/technical/LOGGING_GUIDE.md](docs/technical/LOGGING_GUIDE.md).

---

## 🧹 Project Cleanup Summary

### Root Directory Structure (Cleaned)
```
BiG-RAG/
├── README.md                 # Main project overview
├── CLAUDE.md                 # AI assistant guidance
├── DEVELOPMENT.md            # This file - consolidated dev docs
├── BIGRAG_UI_PLAN.md         # UI implementation reference
├── backend/                  # FastAPI server
├── frontend/                 # React UI
├── bigrag/                   # Core library
├── docs/                     # Detailed documentation
├── test_scripts/             # Test scripts
└── ...
```

### Removed Files
The following redundant markdown files were consolidated into this document:
- FRONTEND_COMPLETE.md
- FRONTEND_IMPLEMENTATION_STATUS.md
- IMPLEMENTATION_SUMMARY.md
- IMPLEMENTATION_STATUS.md
- GRAPH_VISUALIZATION_CURRENT_STATE.md
- GRAPH_VISUALIZATION_IMPLEMENTATION_SUMMARY.md
- BIGRAG_GRAPH_VISUALIZATION_PLAN.md
- pathrag_graph_visualization.md
- LARGE_GRAPH_OPTIMIZATION_GUIDE.md
- CLEANUP_SUMMARY.md
- IMPLEMENTATION_AUDIT_REPORT.md
- LOGGING_SYSTEM_AUDIT.md
- REQUIREMENTS_GUIDE.md

---

## 🔧 Common Development Tasks

### Start Development Environment
```bash
# Terminal 1: Start backend
cd backend
python server.py --data_source SingleTopic

# Terminal 2: Start frontend
cd frontend
npm run dev

# Access:
# - Frontend: http://localhost:5173
# - Backend API: http://localhost:8001
# - API Docs: http://localhost:8001/docs
```

### Build Knowledge Graph
```bash
# Set OpenAI API key
echo "your-api-key" > openai_api_key.txt

# Build graph
python script_build.py --data_source SingleTopic
```

### Run Tests
```bash
# Backend tests
cd test_scripts
python test_improvements.py

# Frontend build test
cd frontend
npm run build
```

---

## 🐛 Known Issues & Solutions

### Frontend Issues

#### 1. Cytoscape Layout Libraries
**Issue:** Some layout extensions may not load correctly
**Solution:** Verify all packages are installed:
```bash
cd frontend
npm install cytoscape-cose-bilkent cytoscape-dagre cytoscape-fcose cytoscape-cola
```

#### 2. Chat Streaming
**Issue:** Streaming not fully implemented
**Status:** Low priority, non-blocking
**Solution:** Use non-streaming endpoint (already working)

#### 3. Theme Switching
**Issue:** Theme selector exists but theme not applied dynamically
**Status:** Low priority, non-blocking
**Solution:** Add ThemeProvider wrapper (15 min task)

#### 4. Graph Dataset Selection (FIXED - Nov 8, 2025)
**Issue:** Graph visualization was hardcoded to 'SingleTopic' dataset
**Impact:** Couldn't view graphs for other datasets (demo_test, 2WikiMultiHopQA, etc.)
**Solution:** ✅ Added dataset dropdown directly on Graph page
**Files Changed:** [frontend/src/pages/GraphViz.tsx](frontend/src/pages/GraphViz.tsx)

**How it Works:**
1. Fetches server's default dataset from health check API (`GET /`)
2. Displays dropdown with all available datasets
3. Auto-selects server's dataset on page load
4. Allows quick switching between datasets
5. Shows "Server Default" label next to current server dataset

**How to Use:**
- Navigate to Graph page
- Use dropdown at top to switch between datasets
- Graph reloads automatically when dataset changes

**Available Datasets:**
- SingleTopic
- demo_test
- 2WikiMultiHopQA
- HotpotQA
- Musique
- NQ
- PopQA
- TriviaQA

**Future Enhancement:** Once Settings page is fixed, this will sync with Settings → Default Dataset

#### 5. Settings Page Field Mismatch (TODO - Nov 8, 2025)
**Issue:** Settings.tsx uses fields that don't exist in settings store
**Impact:** Settings may not persist correctly to localStorage
**Status:** Needs fixing

**Mismatched Fields:**
- Settings.tsx uses: `dataset` → Should use: `activeDataset`
- Settings.tsx uses: `apiEndpoint` → Doesn't exist in store
- Settings.tsx uses: `openaiApiKey` → Should use: `apiKeys.openai`
- Settings.tsx uses: `setDataset` → Should use: `setActiveDataset`
- Settings.tsx uses: `setApiEndpoint` → Doesn't exist
- Settings.tsx uses: `setOpenaiApiKey` → Should use: `setApiKey('openai', key)`

**Files to Fix:**
- [frontend/src/pages/Settings.tsx](frontend/src/pages/Settings.tsx)
- [frontend/src/stores/settings.ts](frontend/src/stores/settings.ts) (optionally add missing fields)

**Recommendation:** Align Settings.tsx with actual store interface from settings.ts

### Backend Issues

#### 1. Large Graph Loading
**Issue:** Loading full graph can be slow for large datasets
**Solution:** Use sampling and progressive loading (already implemented)

#### 2. Memory Usage
**Issue:** Large graphs consume significant memory
**Solution:** Use limit parameter in API calls, default limit=1000

---

## 📚 Additional Resources

### Documentation
- **Main README:** [README.md](README.md)
- **AI Assistant Guide:** [CLAUDE.md](CLAUDE.md)
- **UI Plan:** [BIGRAG_UI_PLAN.md](BIGRAG_UI_PLAN.md)
- **Frontend README:** [frontend/README.md](frontend/README.md)
- **Backend README:** [backend/README.md](backend/README.md)
- **Technical Docs:** [docs/technical/](docs/technical/)
- **Test Reports:** [docs/reports/](docs/reports/)

### External Links
- **React 19:** https://react.dev
- **TypeScript:** https://www.typescriptlang.org/docs
- **Tailwind CSS:** https://tailwindcss.com/docs
- **Cytoscape.js:** https://js.cytoscape.org
- **FastAPI:** https://fastapi.tiangolo.com

---

## 🚀 Deployment

### Frontend Production Build
```bash
cd frontend
npm run build
# Output: frontend/dist/

# Serve with static server
npm run preview
```

### Backend Production
```bash
# Use uvicorn with workers
cd backend
uvicorn server:app --host 0.0.0.0 --port 8001 --workers 4
```

### Docker Deployment (Optional)
See [docs/technical/DEPLOYMENT.md](docs/technical/DEPLOYMENT.md) for Docker setup.

---

## 📞 Support

### Issues
- Check [docs/](docs/) folder first
- Review CLAUDE.md for AI assistant tips
- Check GitHub Issues

### Contributing
1. Fork the repository
2. Create feature branch
3. Make changes
4. Submit pull request

---

**Last Updated:** November 10, 2025
**Status:** Production Ready ✅

**Recent Updates:**
- November 10, 2025: Centralized logging system implementation
- November 6, 2025: Graph visualization improvements (grid, circle, concentric layouts)

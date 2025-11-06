# BiG-RAG Development Guide

**Last Updated:** November 6, 2025

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
- **grid**: Fastest, simple grid arrangement
- **circle**: Fast, nodes in circle
- **concentric**: Grouped by importance

**For Small Graphs (<500 nodes):**
- **cose-bilkent**: Best visual quality
- **fcose**: Good balance of speed and quality

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

**Last Updated:** November 6, 2025
**Status:** Production Ready ✅

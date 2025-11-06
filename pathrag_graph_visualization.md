Complete Graph Visualization Architecture Guide
A Framework-Agnostic Deep Dive into Interactive Knowledge Graph Systems
This guide documents the complete architecture of a production-grade interactive graph visualization system, providing patterns and principles applicable to any graph framework.
TABLE OF CONTENTS
Tech Stack Overview
Architecture & Data Flow
Core Components Breakdown
Graph Initialization Process
Interactive Features Implementation
Layout Algorithms & Animation
State Management Patterns
Performance Optimizations
Framework-Agnostic Principles
1. TECH STACK OVERVIEW {#tech-stack}
Core Libraries
{
  "graphology": "^0.26.0",           // Graph data structure
  "sigma": "^3.0.2",                 // WebGL-based renderer
  "graphology-layout": "^0.6.1",     // Layout utilities
  "graphology-layout-forceatlas2": "^0.10.1"  // Force-directed layout
}
Why This Stack?
Library	Purpose	Alternative Options
Graphology	Pure JS graph data structure with efficient traversal	NetworkX (Python), JGraphT (Java)
Sigma.js	WebGL-based rendering for performance	D3.js, Cytoscape.js, Vis.js, React Flow
ForceAtlas2	Physics-based layout algorithm	Force-Directed, Kamada-Kawai, Hierarchical
React	Component framework	Vue, Angular, Svelte
Key Characteristics
WebGL Rendering: Can handle 10,000+ nodes smoothly
Separation of Concerns: Graphology (data) + Sigma (rendering)
Reactive Updates: React state triggers re-renders
Event-Driven: Mouse events bubble from Sigma to React
2. ARCHITECTURE & DATA FLOW {#architecture}
System Architecture Diagram
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Query Form   │  │ Graph Canvas │  │ Control Panel│          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
└─────────┼──────────────────┼──────────────────┼─────────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    REACT COMPONENT LAYER                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ KnowledgeGraphPage (Container)                           │  │
│  │  - State Management (graphData, loading, error)          │  │
│  │  - API Orchestration                                     │  │
│  │  - Lifecycle Management                                  │  │
│  └─────────────────────────┬────────────────────────────────┘  │
│                            │                                     │
│  ┌─────────────────────────▼────────────────────────────────┐  │
│  │ Graph Component                                           │  │
│  │  - Sigma.js Initialization                               │  │
│  │  - Event Handling (hover, click, zoom)                   │  │
│  │  - Layout Control                                        │  │
│  │  - Modal Management                                      │  │
│  └─────────────────────────┬────────────────────────────────┘  │
└────────────────────────────┼────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  VISUALIZATION LAYER                             │
│  ┌──────────────────┐  ┌──────────────────┐                    │
│  │  Graphology      │◄─┤   Sigma.js       │                    │
│  │  Graph Instance  │  │   Renderer       │                    │
│  │  (Data Model)    │  │   (WebGL)        │                    │
│  └────────┬─────────┘  └────────┬─────────┘                    │
│           │ add/update/delete   │ on(event)                     │
│           │ traverse nodes      │ refresh()                     │
└───────────┼─────────────────────┼───────────────────────────────┘
            │                     │
            ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API LAYER                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ knowledgeGraphAPI                                         │  │
│  │  - getGraph()                                            │  │
│  │  - queryGraph(query)                                     │  │
│  │  - updateEntity(name, data)                              │  │
│  │  - createRelationship(src, tgt, data)                    │  │
│  │  - deleteRelationship(src, tgt)                          │  │
│  └─────────────────────────┬────────────────────────────────┘  │
└────────────────────────────┼────────────────────────────────────┘
                             │ HTTP (REST API)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      BACKEND SERVER                              │
│  - Knowledge Graph Database                                      │
│  - Entity/Relationship Management                                │
│  - Query Processing                                              │
│  - AI Suggestion Engine                                          │
└─────────────────────────────────────────────────────────────────┘
Data Flow: From API to Screen
// STEP 1: Fetch Data from Backend
const response = await knowledgeGraphAPI.getGraph();
// Returns: { nodes: [...], edges: [...] }

// STEP 2: React State Update
setGraphData(response.data);

// STEP 3: useEffect Triggers (dependency: data)
useEffect(() => {
  // STEP 4: Create Graphology Instance
  const graph = new Graph();
  
  // STEP 5: Transform & Add Nodes
  data.nodes.forEach(node => {
    graph.addNode(node.id, {
      label: node.label,
      nodeType: normalizeType(node.type),
      size: 10,
      color: getColorByType(node.type),
      x: Math.random() * 100,  // Initial position
      y: Math.random() * 100
    });
  });
  
  // STEP 6: Transform & Add Edges
  data.edges.forEach(edge => {
    graph.addEdge(edge.source, edge.target, {
      label: edge.label,
      weight: edge.weight || 1,
      size: Math.sqrt(edge.weight)
    });
  });
  
  // STEP 7: Apply Initial Layout
  circular.assign(graph);
  
  // STEP 8: Initialize Renderer
  const renderer = new Sigma(graph, containerRef.current, config);
  
  // STEP 9: Attach Event Listeners
  renderer.on('enterNode', handleHover);
  renderer.on('clickNode', handleClick);
  
  // STEP 10: Start Layout Animation
  startForceDirectedLayout();
  
}, [data]);
3. CORE COMPONENTS BREAKDOWN {#components}
Component Hierarchy
KnowledgeGraphPage (Smart Component)
├── QueryForm (User Input)
├── Graph (Presentation Component)
│   ├── Sigma Canvas (WebGL)
│   ├── Control Panel (Zoom, Layout, Fullscreen)
│   ├── Node Info Panel (Hover Details)
│   └── Mode Indicator (Relationship Creation)
├── EntityEditorModal (CRUD Operations)
├── RelationshipEditorModal (CRUD Operations)
└── CreateRelationshipModal (Creation Wizard)
Component: KnowledgeGraphPage (KnowledgeGraphPage.js)
Responsibilities:
Data fetching and caching
Global state management
API orchestration
Error handling
Key Pattern: Container/Presenter Separation
const KnowledgeGraphPage = () => {
  // STATE: Centralized data management
  const [graphData, setGraphData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // LIFECYCLE: Fetch on mount
  useEffect(() => {
    const fetchGraph = async () => {
      setLoading(true);
      try {
        const response = await knowledgeGraphAPI.getGraph();
        setGraphData(response.data);  // Triggers child re-render
      } catch (error) {
        setError('Failed to load knowledge graph');
      } finally {
        setLoading(false);
      }
    };
    fetchGraph();
  }, []);
  
  // CALLBACK: Child components trigger refresh
  const handleGraphUpdate = async () => {
    const response = await knowledgeGraphAPI.getGraph();
    setGraphData(response.data);
  };
  
  return (
    <Layout>
      <QueryForm onSubmit={handleQuerySubmit} />
      <Graph data={graphData} onUpdate={handleGraphUpdate} />
    </Layout>
  );
};
Component: Graph (Graph.js)
Responsibilities:
Sigma.js lifecycle management
Event handling
Layout animation
User interaction modes
Critical Implementation Details:
1. Dual State Management (State + Ref)
// WHY? Event handlers are closures - they capture state at creation time
// Using refs ensures event handlers always access current values

// STATE: For React re-renders
const [addRelationshipMode, setAddRelationshipMode] = useState(false);
const [selectedSource, setSelectedSource] = useState(null);

// REF: For immediate access in event handlers
const addRelationshipModeRef = useRef(false);
const selectedSourceRef = useRef(null);

// SYNC BOTH on update
const toggleMode = (newMode) => {
  setAddRelationshipMode(newMode);      // Triggers re-render
  addRelationshipModeRef.current = newMode;  // Event handlers see this
};
2. Graph Initialization with Cleanup
useEffect(() => {
  // CLEANUP: Destroy previous instance
  if (sigmaRef.current) {
    sigmaRef.current.kill();  // Remove event listeners, free memory
    sigmaRef.current = null;
  }
  
  // CREATE: New graph instance
  const graph = new Graph();
  graphRef.current = graph;
  
  // POPULATE: Add nodes and edges
  // (transformation logic here)
  
  // LAYOUT: Initial positions
  circular.assign(graph);
  
  // RENDER: Attach to DOM
  const renderer = new Sigma(graph, containerRef.current, {
    renderEdgeLabels: true,
    defaultNodeColor: '#999',
    labelSize: 12,
    enableEdgeEvents: true  // Important for edge clicking!
  });
  
  sigmaRef.current = renderer;
  
  // EVENTS: Attach handlers
  setupEventHandlers(renderer, graph);
  
  // ANIMATION: Start layout
  startLayout();
  
  // CLEANUP: On unmount or data change
  return () => {
    stopLayout();
    if (sigmaRef.current) {
      sigmaRef.current.kill();
    }
  };
}, [data]);  // Re-run when data changes
4. GRAPH INITIALIZATION PROCESS {#initialization}
Step-by-Step Initialization
Step 1: Node Transformation
// RAW API DATA:
{
  id: "person_123",
  label: "Albert Einstein",
  type: "\"Person\"",  // Often has extra quotes from backend
  description: "Physicist"
}

// TRANSFORMATION:
const nodeType = (node.type || 'unknown')
  .replace(/"/g, '')      // Remove quotes
  .toLowerCase();          // Normalize case

graph.addNode(node.id, {
  label: node.label,
  nodeType: nodeType,     // Renamed to avoid Sigma.js conflict
  description: node.description || '',
  size: 10,               // Fixed size (could be degree-based)
  color: nodeColors[nodeType] || nodeColors.unknown,
  x: Math.random() * 100, // Random for initial scatter
  y: Math.random() * 100
});

// RESULT IN GRAPHOLOGY:
// Node stored with all attributes accessible via:
// graph.getNodeAttributes(nodeId)
Step 2: Edge Transformation
// RAW API DATA:
{
  source: "person_123",
  target: "org_456",
  label: "worked_at",
  weight: 5  // Relationship strength
}

// VALIDATION & TRANSFORMATION:
if (graph.hasNode(edge.source) && graph.hasNode(edge.target)) {
  try {
    graph.addEdge(edge.source, edge.target, {
      label: edge.label || '',
      weight: edge.weight || 1,
      size: Math.sqrt(edge.weight || 1)  // Visual thickness
    });
  } catch (e) {
    console.warn('Duplicate edge:', e);
  }
}

// IMPORTANT: Always validate node existence before adding edge!
// Graphology throws error if source/target doesn't exist
Step 3: Initial Layout
// WHY CIRCULAR FIRST?
// - Random positions create visual chaos
// - Circular layout provides predictable starting point
// - ForceAtlas2 converges faster from organized start

import { circular } from 'graphology-layout';
circular.assign(graph);

// RESULT: Nodes arranged in a circle
// Now ready for force-directed animation
Step 4: Renderer Configuration
const renderer = new Sigma(graph, containerRef.current, {
  // LABELS
  renderEdgeLabels: true,        // Show relationship types
  labelSize: 12,                 // Node label font size
  labelWeight: 'bold',           // Make labels prominent
  labelColor: { color: '#fff' }, // White on dark background
  
  // EDGES
  defaultEdgeColor: '#ccc',      // Gray edges
  edgeLabelSize: 10,             // Smaller than node labels
  edgeLabelColor: { color: '#999' },
  enableEdgeEvents: true,        // CRITICAL for edge clicks!
  
  // NODES
  defaultNodeColor: '#999'       // Fallback color
});

// GOTCHA: Without enableEdgeEvents, clickEdge won't fire!
5. INTERACTIVE FEATURES IMPLEMENTATION {#interactions}
A. Hover Effect with Neighbor Highlighting
Visual Feedback Pattern:
renderer.on('enterNode', ({ node }) => {
  setHoveredNode(node);  // Show info panel
  
  // HIGHLIGHT STRATEGY: Show node + 1-hop neighbors
  graph.forEachNode((n) => {
    const isTarget = n === node;
    const isNeighbor = graph.hasEdge(node, n) || graph.hasEdge(n, node);
    
    if (isTarget || isNeighbor) {
      graph.setNodeAttribute(n, 'highlighted', true);
    } else {
      graph.setNodeAttribute(n, 'highlighted', false);
    }
  });
  
  renderer.refresh();  // Apply visual changes
});

renderer.on('leaveNode', () => {
  setHoveredNode(null);
  
  // CLEAR ALL HIGHLIGHTS
  graph.forEachNode((n) => {
    graph.setNodeAttribute(n, 'highlighted', false);
  });
  
  renderer.refresh();
});
CSS Styling for Highlighted State:
/* Sigma.js uses these attributes for rendering */
node[highlighted="true"] {
  stroke: #fff;
  stroke-width: 2px;
}
B. Multi-Mode Click Handling
Pattern: State Machine for Click Behavior
// MODE 1: EDIT MODE (default)
// MODE 2: RELATIONSHIP CREATION MODE

renderer.on('clickNode', ({ node }) => {
  const nodeData = graph.getNodeAttributes(node);
  
  if (addRelationshipModeRef.current) {
    // === RELATIONSHIP CREATION MODE ===
    
    if (!selectedSourceRef.current) {
      // STEP 1: Select source node
      console.log('Source selected:', node);
      setSelectedSource(node);
      selectedSourceRef.current = node;
      
      // VISUAL FEEDBACK: Highlight source
      graph.setNodeAttribute(node, 'color', '#ff6b6b');
      renderer.refresh();
      
    } else if (selectedSourceRef.current !== node) {
      // STEP 2: Select target node
      console.log('Target selected:', node);
      setSelectedTarget(node);
      setShowCreateRelModal(true);  // Open modal
      
      // RESET MODE
      resetRelationshipMode();
    }
  } else {
    // === EDIT MODE ===
    setSelectedEntity(node);
    setShowEntityModal(true);
  }
});
State Diagram:
[Normal Mode]
    │
    │ Click "Add Relationship" Button
    ▼
[Relationship Mode: Step 1]
    │
    │ Click Node A
    ▼
[Relationship Mode: Step 2]
    │ (Node A highlighted)
    │
    │ Click Node B
    ▼
[Modal Opens]
    │ (Define relationship details)
    │
    │ Submit
    ▼
[API Call] → [Refresh Graph] → [Normal Mode]
C. Edge Click for Editing
renderer.on('clickEdge', ({ edge }) => {
  const edgeData = graph.getEdgeAttributes(edge);
  const source = graph.source(edge);  // Get source node ID
  const target = graph.target(edge);  // Get target node ID
  
  setSelectedRelationship({
    source,
    target,
    description: edgeData.label || '',
    keywords: edgeData.keywords || '',
    weight: edgeData.weight || 1.0
  });
  
  setShowRelationshipModal(true);
});

// IMPORTANT: Requires enableEdgeEvents: true in Sigma config!
D. Camera Controls (Pan, Zoom, Reset)
// GET CAMERA: Sigma manages camera internally
const camera = sigmaRef.current.getCamera();

// ZOOM IN (Animated)
const handleZoomIn = () => {
  camera.animatedZoom({ duration: 300 });
};

// ZOOM OUT (Animated)
const handleZoomOut = () => {
  camera.animatedUnzoom({ duration: 300 });
};

// RESET VIEW (Center + Default Zoom)
const handleResetZoom = () => {
  camera.animatedReset({ duration: 500 });
};

// TRACK ZOOM LEVEL (for display)
camera.on('updated', () => {
  const ratio = camera.ratio;
  setZoomLevel(1 / ratio);  // Convert to percentage
});
Display Zoom Percentage:
<div className="zoom-level">
  {Math.round(zoomLevel * 100)}%
</div>
E. Fullscreen Toggle (Cross-Browser)
const toggleFullScreen = () => {
  const elem = graphContainerRef.current;
  
  if (!document.fullscreenElement) {
    // ENTER FULLSCREEN (with vendor prefixes)
    if (elem.requestFullscreen) {
      elem.requestFullscreen();
    } else if (elem.webkitRequestFullscreen) {  // Safari
      elem.webkitRequestFullscreen();
    } else if (elem.mozRequestFullScreen) {  // Firefox
      elem.mozRequestFullScreen();
    } else if (elem.msRequestFullscreen) {  // IE11
      elem.msRequestFullscreen();
    }
  } else {
    // EXIT FULLSCREEN
    if (document.exitFullscreen) {
      document.exitFullscreen();
    } else if (document.webkitExitFullscreen) {
      document.webkitExitFullscreen();
    } else if (document.mozCancelFullScreen) {
      document.mozCancelFullScreen();
    } else if (document.msExitFullscreen) {
      document.msExitFullscreen();
    }
  }
};

// LISTEN FOR FULLSCREEN CHANGES
useEffect(() => {
  const handleFullScreenChange = () => {
    setIsFullScreen(!!document.fullscreenElement);
  };
  
  document.addEventListener('fullscreenchange', handleFullScreenChange);
  document.addEventListener('webkitfullscreenchange', handleFullScreenChange);
  document.addEventListener('mozfullscreenchange', handleFullScreenChange);
  
  return () => {
    document.removeEventListener('fullscreenchange', handleFullScreenChange);
    document.removeEventListener('webkitfullscreenchange', handleFullScreenChange);
    document.removeEventListener('mozfullscreenchange', handleFullScreenChange);
  };
}, []);
6. LAYOUT ALGORITHMS & ANIMATION {#layout}
ForceAtlas2: Physics-Based Layout
Concept: Nodes repel each other (like charged particles), edges attract connected nodes (like springs). Implementation:
import forceAtlas2 from 'graphology-layout-forceatlas2';

const startLayout = () => {
  if (!graphRef.current || layoutWorkerRef.current) return;
  
  // INFER SETTINGS: Auto-calculate based on graph size
  const sensibleSettings = forceAtlas2.inferSettings(graphRef.current);
  
  layoutWorkerRef.current = true;
  setIsLayoutRunning(true);
  
  let iterations = 0;
  const maxIterations = 500;  // Prevent infinite loops
  
  const iterate = () => {
    // STOP CONDITIONS
    if (!layoutWorkerRef.current || iterations >= maxIterations) {
      stopLayout();
      return;
    }
    
    // COMPUTE ONE STEP
    forceAtlas2.assign(graphRef.current, {
      iterations: 1,  // One step per frame for smooth animation
      settings: {
        ...sensibleSettings,
        gravity: 1,        // Pull towards center (prevents scatter)
        scalingRatio: 10,  // Node repulsion strength
        slowDown: 5        // Damping factor (higher = slower convergence)
      }
    });
    
    // REFRESH RENDER
    if (sigmaRef.current) {
      sigmaRef.current.refresh();
    }
    
    iterations++;
    
    // NEXT FRAME
    requestAnimationFrame(iterate);  // ~60 FPS animation
  };
  
  iterate();
};
Key Parameters Explained:
Parameter	Effect	Typical Range
gravity	Pulls nodes toward center	0.1 - 10
scalingRatio	Node repulsion strength	1 - 100
slowDown	Damping/friction	1 - 10
iterations	Steps per call	1 (for smooth), 50 (for fast)
Animation Strategy:
Smooth Animation: 1 iteration per frame (60 FPS)
Fast Convergence: 50 iterations per frame (jumpy but quick)
Auto-stop: Max 500 iterations prevents infinite computation
7. STATE MANAGEMENT PATTERNS {#state}
Critical Pattern: Ref + State Synchronization
Problem:
// ❌ BROKEN: Event handler captures stale state
const [mode, setMode] = useState(false);

useEffect(() => {
  renderer.on('clickNode', () => {
    console.log(mode);  // Always logs initial value!
  });
}, []);  // Event handler created once, captures initial `mode`
Solution:
// ✅ WORKING: Ref provides current value
const [mode, setMode] = useState(false);
const modeRef = useRef(false);

const updateMode = (newMode) => {
  setMode(newMode);          // For React re-renders
  modeRef.current = newMode; // For event handlers
};

useEffect(() => {
  renderer.on('clickNode', () => {
    console.log(modeRef.current);  // Always current!
  });
}, []);
Modal State Management
Pattern: Controlled Modals with Callbacks
// PARENT COMPONENT
const [selectedEntity, setSelectedEntity] = useState(null);
const [showModal, setShowModal] = useState(false);

const handleModalUpdate = async () => {
  // REFRESH GRAPH after edit
  const response = await knowledgeGraphAPI.getGraph();
  setGraphData(response.data);
};

// RENDER
<EntityEditorModal
  open={showModal}
  onClose={() => {
    setShowModal(false);
    setSelectedEntity(null);
  }}
  entityName={selectedEntity}
  onUpdate={handleModalUpdate}  // Callback for refresh
/>
Flow:
User clicks node
  → setSelectedEntity(nodeId)
  → setShowModal(true)
  → Modal renders
  → User edits entity
  → Modal calls API
  → Modal calls onUpdate()
  → Parent fetches fresh data
  → Graph re-renders with new data
8. PERFORMANCE OPTIMIZATIONS {#performance}
Implemented Optimizations:
1. Entity Caching (CreateRelationshipModal)
const entityCache = useRef(new Map());
const CACHE_LIMIT = 20;

const fetchEntities = async (search) => {
  const cacheKey = search || '__all__';
  
  // CHECK CACHE
  if (entityCache.current.has(cacheKey)) {
    setEntities(entityCache.current.get(cacheKey));
    return;
  }
  
  // FETCH
  const response = await knowledgeGraphAPI.getAllEntities(search);
  
  // STORE IN CACHE
  entityCache.current.set(cacheKey, response.data.entities);
  
  // LIMIT CACHE SIZE
  if (entityCache.current.size > CACHE_LIMIT) {
    const firstKey = entityCache.current.keys().next().value;
    entityCache.current.delete(firstKey);
  }
  
  setEntities(response.data.entities);
};
2. Debounced Search
const debounceTimer = useRef(null);

const handleSearchChange = (value) => {
  setSearch(value);
  
  // CANCEL PREVIOUS TIMER
  if (debounceTimer.current) {
    clearTimeout(debounceTimer.current);
  }
  
  // SET NEW TIMER
  debounceTimer.current = setTimeout(() => {
    fetchEntities(value);
  }, 300);  // Wait 300ms after user stops typing
};
3. RequestAnimationFrame for Layout
// SYNC WITH BROWSER REFRESH (60 FPS)
const iterate = () => {
  // ... compute layout step ...
  requestAnimationFrame(iterate);  // Next frame
};
4. WebGL Rendering (Sigma.js)
GPU-accelerated rendering
Handles 10,000+ nodes smoothly
Automatic viewport culling (only render visible nodes)
Potential Future Optimizations:
1. Virtual Scrolling for Tables
// Current: Renders all rows (pagination helps)
// Better: Only render visible rows

import { FixedSizeList } from 'react-window';

<FixedSizeList
  height={600}
  itemCount={entities.length}
  itemSize={50}
>
  {({ index }) => <EntityRow entity={entities[index]} />}
</FixedSizeList>
2. Web Workers for Layout
// Current: Layout runs on main thread
// Better: Offload to worker thread

const layoutWorker = new Worker('layout-worker.js');

layoutWorker.postMessage({ graph: graphData });

layoutWorker.onmessage = (e) => {
  applyPositions(e.data.positions);
};
3. Level-of-Detail Rendering
// Show less detail when zoomed out
camera.on('updated', () => {
  const zoomLevel = camera.ratio;
  
  if (zoomLevel < 0.5) {
    // Hide labels, simplify nodes
    renderer.setSetting('renderLabels', false);
  } else {
    renderer.setSetting('renderLabels', true);
  }
});
9. FRAMEWORK-AGNOSTIC PRINCIPLES {#principles}
Universal Patterns for Any Graph Visualization
1. Separation of Data and Rendering
Graph Data Structure (Graphology)
    ↕️
Graph Renderer (Sigma.js)
Benefits:
Change renderer without changing data logic
Test data transformations independently
Easier to debug
Alternatives:
D3.js: Manual DOM manipulation
Cytoscape.js: Integrated data + rendering
React Flow: React-based nodes
2. Event-Driven Architecture
// PUBLISHER: Renderer emits events
renderer.on('clickNode', ({ node }) => {
  // SUBSCRIBER: React component responds
  handleNodeClick(node);
});
Pattern: Observer/Publisher-Subscriber Benefits: Loose coupling, easy to extend
3. Progressive Enhancement of Nodes/Edges
Start Simple:
graph.addNode(id, { x: 0, y: 0 });
Add Visual Properties:
graph.addNode(id, {
  x: 0, y: 0,
  size: 10,
  color: '#4e79a7'
});
Add Semantic Data:
graph.addNode(id, {
  x: 0, y: 0,
  size: 10,
  color: '#4e79a7',
  type: 'person',
  description: 'Details...',
  metadata: { ... }
});
4. Layered Rendering
Layer 1: Canvas/WebGL (Background, high-performance)
Nodes
Edges
Layer 2: HTML/SVG Overlay (Foreground, rich interaction)
Control panel
Info panels
Modals
Layer 3: DOM (Top, UI components)
Forms
Buttons
Tooltips
5. Atomic State Updates
// ❌ BAD: Multiple setState calls
setLoading(true);
setError(null);
setData(null);

// ✅ GOOD: Single atomic update
setState({
  loading: true,
  error: null,
  data: null
});
Building with Another Framework:
D3.js Implementation:
// SELECT CONTAINER
const svg = d3.select('#graph')
  .append('svg')
  .attr('width', width)
  .attr('height', height);

// BIND DATA
const nodes = svg.selectAll('.node')
  .data(graphData.nodes)
  .enter()
  .append('circle')
  .attr('class', 'node')
  .attr('r', 5)
  .attr('fill', d => colorScale(d.type));

// FORCE SIMULATION
const simulation = d3.forceSimulation(graphData.nodes)
  .force('link', d3.forceLink(graphData.edges).id(d => d.id))
  .force('charge', d3.forceManyBody().strength(-100))
  .force('center', d3.forceCenter(width / 2, height / 2));

// UPDATE POSITIONS
simulation.on('tick', () => {
  nodes
    .attr('cx', d => d.x)
    .attr('cy', d => d.y);
});
Cytoscape.js Implementation:
const cy = cytoscape({
  container: document.getElementById('graph'),
  
  elements: {
    nodes: graphData.nodes.map(n => ({
      data: { id: n.id, label: n.label, type: n.type }
    })),
    edges: graphData.edges.map(e => ({
      data: { source: e.source, target: e.target, label: e.label }
    }))
  },
  
  style: [
    {
      selector: 'node',
      style: {
        'background-color': (ele) => colorByType[ele.data('type')],
        'label': 'data(label)'
      }
    }
  ],
  
  layout: {
    name: 'cose',  // Force-directed
    animate: true
  }
});

// EVENT HANDLING
cy.on('tap', 'node', (evt) => {
  const node = evt.target;
  console.log('Clicked:', node.data());
});
SUMMARY: KEY TAKEAWAYS
Architecture Principles:
Separate data structure from rendering
Use event-driven communication
Manage state with refs for immediate access
Implement cleanup in React useEffect
Cache frequently accessed data
Debounce user input
Use requestAnimationFrame for animations
Performance Tips:
Limit layout iterations (cap at 500)
Render labels conditionally (hide when zoomed out)
Use WebGL renderers for large graphs
Paginate entity lists
Cache API responses
User Experience:
Provide visual feedback (hover, click, mode changes)
Show loading states
Handle errors gracefully
Offer multiple interaction modes (edit, create relationships)
Support fullscreen for deep work
Display zoom level and controls
Code Quality:
Validate data before adding to graph
Normalize backend data (remove quotes, lowercase types)
Use TypeScript/PropTypes for type safety
Write cleanup functions
Log errors for debugging
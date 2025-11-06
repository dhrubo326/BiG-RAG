# BiG-RAG Graph Visualization Reconstruction Plan

**Date:** January 2025
**Status:** Blueprint for Implementation
**Purpose:** Complete redesign of graph visualization to properly display BiG-RAG's hypergraph/bipartite structure

---

## Executive Summary

### Problem Statement

The current graph visualization only displays entity nodes as circles, completely missing the **relation nodes** which are the core semantic connectors in BiG-RAG's hypergraph. This creates a fundamental misrepresentation of the knowledge graph structure.

**Current State:**
- Only shows 8,108 entity circles
- Missing 7,277 relation nodes (bipartite edges)
- No visual representation of entity → relation → entity paths
- Poor UX with basic circular layout

**Target State:**
- Display both entities AND relations as distinct visual elements
- Show bipartite graph structure clearly
- Interactive exploration of entity → relation → entity paths
- Professional UX with multiple layout algorithms
- Handle 15K+ nodes with smooth performance

---

## Part 1: Data Structure Analysis

### 1.1 BiG-RAG Hypergraph Architecture

BiG-RAG uses a **true bipartite graph** (hypergraph) where relations are first-class nodes, NOT edges.

```
Traditional Graph (PathRAG):
  Entity A --[edge with label]--> Entity B

BiG-RAG Hypergraph:
  Entity A --[edge]--> Relation Node --[edge]--> Entity B
                       (contains semantic description)
```

**Key Insight:** In BiG-RAG, the semantic meaning is stored in RELATION NODES, not edge labels.

### 1.2 Storage Format

**Location:** `expr/SingleTopic/graph_chunk_entity_relation.graphml` (8.6MB)

**Format:** NetworkX GraphML (XML-based graph format)

**Node Types:**
1. **Entity Nodes** (8,108 nodes)
   - `role`: "entity"
   - `name`: Entity name (e.g., "FALLEN BULLET KIN")
   - `description`: Optional description
   - `entity_type`: Type of entity (concept, person, location, etc.)
   - `weight`: Importance score (0.0 - 1.0)
   - `source_id`: Origin document chunk ID

2. **Relation Nodes** (7,277 nodes) - **THIS IS WHAT'S MISSING**
   - `role`: "bipartite_edge"
   - `weight`: Aggregated importance (sum of occurrences)
   - `source_id`: Source chunk IDs (can be multiple, separated by `<SEP>`)
   - **Node ID**: The actual semantic relationship text
     - Example: `<bipartite_edge>"Fallen Bullet Kin walk towards the player, firing spreads of 3 fire-shaped bullets."`

**Edge Structure:**
- Total: 10,709 edges
- Connect: Relation Nodes ↔ Entity Nodes
- Attributes:
  - `weight`: Edge weight
  - `source_id`: Origin chunk ID

**Example Data Structure:**

```python
# Entity Node
{
  'id': 'FALLEN BULLET KIN',
  'role': 'entity',
  'name': 'FALLEN BULLET KIN',
  'entity_type': 'concept',
  'weight': 0.85,
  'source_id': 'chunk-abc123'
}

# Relation Node (bipartite_edge)
{
  'id': '<bipartite_edge>"Fallen Bullet Kin walk towards the player, firing spreads of 3 fire-shaped bullets."',
  'role': 'bipartite_edge',
  'weight': 26.0,  # Appears 26 times across documents
  'source_id': 'chunk-1a1f8772...<SEP>chunk-21cf9846...'
}

# Edge connecting them
{
  'source': '<bipartite_edge>"Fallen Bullet Kin walk..."',
  'target': 'FALLEN BULLET KIN',
  'weight': 170.0,
  'source_id': 'chunk-xyz789'
}
```

### 1.3 Bipartite Graph Properties

**Partitioning:**
- **Set A:** Entity nodes (entities extracted from text)
- **Set B:** Relation nodes (semantic descriptions)
- **Edges:** Only connect A ↔ B (no A-A or B-B connections)

**Semantic Interpretation:**
```
Query: "What do Fallen Bullet Kin do?"

Graph Path:
  "FALLEN BULLET KIN" (entity)
    → <bipartite_edge>"Fallen Bullet Kin walk towards the player, firing spreads of 3 fire-shaped bullets."
    → "PLAYER" (entity)
    → "FIRE-SHAPED BULLETS" (entity)
```

**Why This Matters:**
- Relations contain the actual knowledge (verbs, actions, properties)
- Entities are just named objects
- Visualizing only entities loses all semantic meaning

---

## Part 2: Framework Selection

### 2.1 Framework Comparison

| Feature | Cytoscape.js | Sigma.js + Graphology |
|---------|-------------|----------------------|
| **Rendering** | Canvas + WebGL | WebGL only |
| **Graph Model** | Built-in | Graphology (separate) |
| **Bipartite Layouts** | ✅ Cose-Bilkent, Dagre | ⚠️ ForceAtlas2 (need custom) |
| **Node Styling** | ✅ CSS-like selectors | ✅ GLSL shaders |
| **Performance (15K nodes)** | ✅ Good (2-3s render) | ✅ Excellent (1-2s render) |
| **Interactive Features** | ✅ Rich API | ✅ Rich API |
| **Bundle Size** | ~500KB | ~200KB (Sigma) + ~100KB (Graphology) |
| **Documentation** | ✅ Excellent | ✅ Good |
| **Community** | ✅ Large (10K+ stars) | ✅ Medium (5K+ stars) |
| **Bipartite-specific** | ✅ Native support | ⚠️ Manual implementation |

### 2.2 Recommendation: **Cytoscape.js**

**Rationale:**

1. **Native Bipartite Support:** Cytoscape.js has layout algorithms specifically designed for bipartite graphs:
   - **Cose-Bilkent:** Compound Spring Embedder, excellent for bipartite
   - **Dagre:** Hierarchical layout, perfect for entity → relation → entity flows

2. **CSS-like Styling:** Easy to visually distinguish entities vs relations:
   ```javascript
   cy.style()
     .selector('node[type="entity"]')
     .style({ 'background-color': '#3B82F6', 'shape': 'ellipse' })
     .selector('node[type="relation"]')
     .style({ 'background-color': '#EF4444', 'shape': 'rectangle' })
   ```

3. **Already Integrated:** Current codebase already uses Cytoscape.js (version 3.33.0), reducing migration risk.

4. **Extension Ecosystem:** Rich plugins for advanced features:
   - `cytoscape-cola`: Force-directed layout
   - `cytoscape-expand-collapse`: Node grouping
   - `cytoscape-context-menus`: Right-click menus
   - `cytoscape-navigator`: Mini-map

**When to Consider Sigma.js:**
- If performance becomes critical (>50K nodes)
- If you need custom WebGL shaders for visual effects
- If you want lighter bundle size

**For BiG-RAG's hypergraph structure, Cytoscape.js is the better choice.**

---

## Part 3: Visualization Strategy

### 3.1 Visual Language

**Node Representation:**

| Node Type | Shape | Color | Size | Label |
|-----------|-------|-------|------|-------|
| **Entity** | Circle/Ellipse | Blue (#3B82F6) | By weight (20-60px) | Entity name |
| **Relation** | Rectangle/Diamond | Red (#EF4444) | By weight (15-50px) | Truncated description (first 30 chars) |

**Edge Representation:**

| Property | Styling |
|----------|---------|
| **Width** | Proportional to weight (1-5px) |
| **Color** | Gray (#94A3B8) |
| **Opacity** | 0.6 (increase on hover) |
| **Curve** | Bezier curve for clarity |

**Interaction States:**

| State | Entity Style | Relation Style |
|-------|-------------|----------------|
| **Default** | Blue circle | Red rectangle |
| **Hover** | Gold border (#FFD700) | Gold border |
| **Selected** | Orange (#FFA500) | Orange |
| **Connected** | Highlight blue | Highlight red |
| **Dimmed** | 30% opacity | 30% opacity |

### 3.2 Layout Algorithms

**Primary Layout: Cose-Bilkent** (Bipartite-optimized)
- Separates entities and relations into visual layers
- Minimizes edge crossings
- Fast for 1000-5000 nodes

**Configuration:**
```javascript
{
  name: 'cose-bilkent',
  nodeDimensionsIncludeLabels: true,
  idealEdgeLength: 150,
  nodeRepulsion: 8000,
  gravity: 0.25,
  numIter: 2500,
  tile: true,
  tilingPaddingVertical: 20,
  tilingPaddingHorizontal: 20
}
```

**Secondary Layouts:**

1. **Dagre** (Hierarchical)
   - Good for exploring entity → relation → entity paths
   - Top-to-bottom or left-to-right flow
   - Best for subgraphs (< 500 nodes)

2. **Cola** (Force-directed)
   - Natural clustering of related entities
   - Good for exploration
   - Slower for large graphs

3. **Concentric** (Radial)
   - Center important nodes (high weight)
   - Rings for less important nodes
   - Good for focused exploration

4. **Breadthfirst** (Tree)
   - Start from selected node
   - Show neighbors in expanding circles
   - Good for path exploration

### 3.3 Progressive Rendering Strategy

**Problem:** 15,385 nodes will overwhelm browser even with WebGL

**Solution:** Multi-level loading

**Level 1: Initial Load** (1000 nodes)
- Top 1000 nodes by weight
- Sample strategy: `top_weighted`
- Render in 2-3 seconds

**Level 2: On-demand Expansion**
- User clicks "Show more nodes" → load next 1000
- User double-clicks node → load its neighbors
- User searches → load matching subgraph

**Level 3: Full Graph**
- Only load if user explicitly requests
- Show warning: "Loading 15K nodes may freeze browser"
- Implement viewport culling (only render visible nodes)

**Implementation:**
```typescript
interface GraphLoadState {
  current: 'initial' | 'partial' | 'full';
  nodesLoaded: number;
  totalNodes: number;
  canLoadMore: boolean;
}

const loadMoreNodes = async (currentCount: number) => {
  const nextBatch = await getGraphData(dataset, {
    limit: 1000,
    offset: currentCount, // NEW: pagination support
    sampleStrategy: 'top_weighted'
  });

  // Append to existing graph
  cy.add(transformToCytoscape(nextBatch));
};
```

---

## Part 4: Technical Architecture

### 4.1 Backend Modifications

**Current Endpoint:** `GET /graph/export`

**Add New Parameters:**

```python
@app.get("/graph/export")
async def export_graph(
    data_source: str,
    limit: int = 1000,
    offset: int = 0,  # NEW: for pagination
    node_types: Optional[str] = None,  # "entity,relation,chunk"
    min_weight: float = 0.0,
    sample_strategy: str = "top_weighted",
    include_relation_nodes: bool = True  # NEW: critical flag
):
    """
    Export graph data with proper support for relation nodes.

    NEW: include_relation_nodes=True ensures bipartite_edge nodes are returned
    """
    G = nx.read_graphml(graph_file)

    all_nodes = []
    for node_id, attrs in G.nodes(data=True):
        role = attrs.get("role", "")

        # Map role to type
        if role == "entity":
            node_type = "entity"
        elif role == "bipartite_edge":
            node_type = "relation"  # NEW: expose as "relation"

            # Extract clean description from node ID
            # ID format: <bipartite_edge>"Description text"
            description = node_id.replace('<bipartite_edge>"', '').replace('"', '')
            attrs['description'] = description
            attrs['label'] = description[:50] + '...' if len(description) > 50 else description
        else:
            node_type = "chunk"

        all_nodes.append({
            "id": node_id,
            "label": attrs.get("label", attrs.get("name", node_id)),
            "name": attrs.get("name", ""),
            "type": node_type,
            "description": attrs.get("description", ""),
            "weight": float(attrs.get("weight", 0.5)),
            "source_id": attrs.get("source_id", ""),
            "metadata": {
                "role": role,
                "entity_type": attrs.get("entity_type", "")
            }
        })

    # Apply filters
    if node_types:
        allowed_types = set(node_types.split(','))
        all_nodes = [n for n in all_nodes if n['type'] in allowed_types]

    if min_weight > 0:
        all_nodes = [n for n in all_nodes if n['weight'] >= min_weight]

    # Apply sampling
    if len(all_nodes) > limit:
        if sample_strategy == "top_weighted":
            all_nodes = sorted(all_nodes, key=lambda x: x['weight'], reverse=True)
        elif sample_strategy == "diverse":
            # Ensure balanced sampling of entities and relations
            entities = [n for n in all_nodes if n['type'] == 'entity']
            relations = [n for n in all_nodes if n['type'] == 'relation']

            entity_limit = int(limit * 0.6)  # 60% entities
            relation_limit = int(limit * 0.4)  # 40% relations

            sampled_entities = sorted(entities, key=lambda x: x['weight'], reverse=True)[:entity_limit]
            sampled_relations = sorted(relations, key=lambda x: x['weight'], reverse=True)[:relation_limit]

            all_nodes = sampled_entities + sampled_relations

        all_nodes = all_nodes[offset:offset + limit]

    # Get edges for sampled nodes
    sampled_node_ids = {n['id'] for n in all_nodes}
    edges = []
    for source, target, attrs in G.edges(data=True):
        if source in sampled_node_ids and target in sampled_node_ids:
            edges.append({
                "id": f"{source}_{target}",
                "source": source,
                "target": target,
                "label": attrs.get("label", ""),
                "weight": float(attrs.get("weight", 1.0)),
                "type": "semantic"
            })

    return {
        "success": True,
        "dataset": data_source,
        "nodes": all_nodes,
        "edges": edges,
        "stats": {
            "totalNodes": len(G.nodes()),
            "totalEdges": len(G.edges()),
            "entities": len([n for n in G.nodes(data=True) if n[1].get('role') == 'entity']),
            "relations": len([n for n in G.nodes(data=True) if n[1].get('role') == 'bipartite_edge']),
            "nodesReturned": len(all_nodes),
            "edgesReturned": len(edges)
        },
        "sampling_info": {
            "sampling_applied": len(G.nodes()) > limit,
            "strategy": sample_strategy,
            "offset": offset,
            "limit": limit
        }
    }
```

**New Endpoint: Get Node Details**

```python
@app.get("/graph/node/{node_id}")
async def get_node_details(node_id: str, data_source: str):
    """
    Get detailed information about a specific node.
    Includes all neighbors and connected edges.
    """
    G = nx.read_graphml(graph_file)

    if node_id not in G:
        raise HTTPException(status_code=404, detail="Node not found")

    node_attrs = G.nodes[node_id]
    neighbors = list(G.neighbors(node_id))

    # Get connected edges with details
    edges = []
    for neighbor in neighbors:
        edge_attrs = G.edges[node_id, neighbor] if G.has_edge(node_id, neighbor) else G.edges[neighbor, node_id]
        edges.append({
            "source": node_id,
            "target": neighbor,
            "weight": edge_attrs.get("weight", 1.0)
        })

    return {
        "node": {
            "id": node_id,
            "attributes": dict(node_attrs),
            "degree": G.degree(node_id),
            "neighbors_count": len(neighbors)
        },
        "neighbors": neighbors,
        "edges": edges
    }
```

### 4.2 Frontend Architecture

**Component Structure:**

```
src/components/graph/
├── GraphCanvas.tsx              # Main Cytoscape canvas wrapper
├── GraphToolbar.tsx             # Top toolbar (layout, filters, search)
├── GraphDetailsPanel.tsx        # Right panel (node details)
├── GraphLegend.tsx              # Color legend (entities vs relations)
├── GraphControls.tsx            # Zoom, fit, center controls
├── GraphContextMenu.tsx         # Right-click menu
└── GraphLoadingState.tsx        # Loading skeleton
```

**Data Flow:**

```
1. User loads graph page
   → GraphViz.tsx
   → useGraph.loadGraph('SingleTopic', { limit: 1000, sampleStrategy: 'diverse' })
   → graph.ts service calls backend
   → Backend samples 600 entities + 400 relations = 1000 nodes
   → Returns JSON with nodes, edges, stats

2. Graph renders
   → GraphCanvas receives nodes/edges
   → Transforms to Cytoscape format
   → Applies cose-bilkent layout
   → Renders with color-coded nodes (blue circles, red rectangles)

3. User interactions
   → Click node → GraphDetailsPanel shows details
   → Hover node → Highlight connected nodes
   → Double-click node → Load neighbors
   → Search → Filter visible nodes
   → Change layout → Re-render with new algorithm
```

**State Management (Zustand):**

```typescript
// src/store/graph.ts
interface GraphStore {
  // Data
  nodes: CytoscapeNode[];
  edges: CytoscapeEdge[];
  stats: GraphStats | null;

  // UI State
  selectedNode: string | null;
  highlightedNodes: Set<string>;
  layout: 'cose-bilkent' | 'dagre' | 'cola' | 'concentric';
  filters: {
    showEntities: boolean;
    showRelations: boolean;
    showChunks: boolean;
    minWeight: number;
  };

  // Loading State
  loadState: 'initial' | 'partial' | 'full';
  nodesLoaded: number;
  totalNodes: number;

  // Actions
  loadGraph: (dataset: string, options: GraphLoadOptions) => Promise<void>;
  loadMoreNodes: () => Promise<void>;
  selectNode: (nodeId: string | null) => void;
  setLayout: (layout: string) => void;
  applyFilters: (filters: GraphFilters) => void;
  highlightPath: (startNode: string, endNode: string) => void;
}
```

### 4.3 Cytoscape Integration

**Initialize Cytoscape:**

```typescript
// src/components/graph/GraphCanvas.tsx
import cytoscape, { Core, EdgeSingular, NodeSingular } from 'cytoscape';
import coseBilkent from 'cytoscape-cose-bilkent';
import cola from 'cytoscape-cola';
import dagre from 'cytoscape-dagre';

// Register layout algorithms
cytoscape.use(coseBilkent);
cytoscape.use(cola);
cytoscape.use(dagre);

const GraphCanvas: React.FC = () => {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<Core | null>(null);
  const { nodes, edges, layout, selectedNode } = useGraphStore();

  useEffect(() => {
    if (!containerRef.current) return;

    // Initialize Cytoscape
    const cy = cytoscape({
      container: containerRef.current,

      // Style
      style: [
        // Entity nodes
        {
          selector: 'node[type="entity"]',
          style: {
            'background-color': '#3B82F6',
            'shape': 'ellipse',
            'width': 'data(size)',
            'height': 'data(size)',
            'label': 'data(label)',
            'font-size': '12px',
            'text-valign': 'center',
            'text-halign': 'center',
            'text-wrap': 'wrap',
            'text-max-width': '100px',
            'border-width': 2,
            'border-color': '#1E40AF',
            'color': '#FFFFFF'
          }
        },

        // Relation nodes (bipartite edges)
        {
          selector: 'node[type="relation"]',
          style: {
            'background-color': '#EF4444',
            'shape': 'rectangle',
            'width': 'data(size)',
            'height': 'data(size)',
            'label': 'data(label)',
            'font-size': '10px',
            'text-valign': 'center',
            'text-halign': 'center',
            'text-wrap': 'wrap',
            'text-max-width': '120px',
            'border-width': 2,
            'border-color': '#991B1B',
            'color': '#FFFFFF'
          }
        },

        // Edges
        {
          selector: 'edge',
          style: {
            'width': 'data(weight)',
            'line-color': '#94A3B8',
            'opacity': 0.6,
            'curve-style': 'bezier',
            'target-arrow-shape': 'none'
          }
        },

        // Hover state
        {
          selector: 'node:active',
          style: {
            'border-color': '#FFD700',
            'border-width': 4
          }
        },

        // Selected state
        {
          selector: 'node.selected',
          style: {
            'background-color': '#FFA500',
            'border-color': '#EA580C',
            'border-width': 4
          }
        },

        // Connected nodes (when node is selected)
        {
          selector: 'node.connected',
          style: {
            'opacity': 1
          }
        },

        // Dimmed nodes (when filtering)
        {
          selector: 'node.dimmed',
          style: {
            'opacity': 0.3
          }
        }
      ],

      // Initial empty elements
      elements: [],

      // Interaction options
      minZoom: 0.1,
      maxZoom: 3,
      wheelSensitivity: 0.2,

      // Performance
      pixelRatio: 'auto',
      motionBlur: true,
      textureOnViewport: true,
      hideEdgesOnViewport: true
    });

    cyRef.current = cy;

    // Event listeners
    cy.on('tap', 'node', (evt) => {
      const node = evt.target;
      selectNode(node.id());
    });

    cy.on('mouseover', 'node', (evt) => {
      const node = evt.target;
      node.style('cursor', 'pointer');

      // Show tooltip
      showTooltip(node.data());
    });

    cy.on('mouseout', 'node', () => {
      hideTooltip();
    });

    return () => {
      cy.destroy();
    };
  }, []);

  // Update graph when nodes/edges change
  useEffect(() => {
    if (!cyRef.current) return;

    const cy = cyRef.current;

    // Transform nodes
    const cytoscapeNodes = nodes.map(node => ({
      data: {
        id: node.data.id,
        label: node.data.label,
        type: node.data.type,
        weight: node.data.weight,
        description: node.data.description,
        size: calculateNodeSize(node.data.weight, node.data.type)
      }
    }));

    // Transform edges
    const cytoscapeEdges = edges.map(edge => ({
      data: {
        id: edge.data.id,
        source: edge.data.source,
        target: edge.data.target,
        weight: calculateEdgeWidth(edge.data.weight)
      }
    }));

    // Update graph
    cy.elements().remove();
    cy.add([...cytoscapeNodes, ...cytoscapeEdges]);

    // Apply layout
    applyLayout(layout);
  }, [nodes, edges, layout]);

  const applyLayout = (layoutName: string) => {
    if (!cyRef.current) return;

    const layoutOptions = getLayoutOptions(layoutName);
    const layout = cyRef.current.layout(layoutOptions);
    layout.run();
  };

  return (
    <div className="relative w-full h-full">
      <div ref={containerRef} className="w-full h-full bg-gray-50" />
    </div>
  );
};
```

**Helper Functions:**

```typescript
// Calculate node size based on weight and type
const calculateNodeSize = (weight: number, type: string): number => {
  const baseSize = type === 'entity' ? 30 : 25; // Relations slightly smaller
  const maxSize = type === 'entity' ? 60 : 50;

  return baseSize + (weight * (maxSize - baseSize));
};

// Calculate edge width based on weight
const calculateEdgeWidth = (weight: number): number => {
  return 1 + (weight / 100) * 4; // 1-5px range
};

// Get layout configuration
const getLayoutOptions = (layoutName: string) => {
  const layouts = {
    'cose-bilkent': {
      name: 'cose-bilkent',
      nodeDimensionsIncludeLabels: true,
      idealEdgeLength: 150,
      nodeRepulsion: 8000,
      gravity: 0.25,
      numIter: 2500,
      animate: true,
      animationDuration: 1000
    },
    'dagre': {
      name: 'dagre',
      rankDir: 'TB', // Top to bottom
      nodeSep: 100,
      edgeSep: 50,
      rankSep: 150,
      animate: true
    },
    'cola': {
      name: 'cola',
      nodeSpacing: 50,
      edgeLength: 150,
      animate: true,
      randomize: false,
      maxSimulationTime: 4000
    },
    'concentric': {
      name: 'concentric',
      concentric: (node: NodeSingular) => node.data('weight'),
      levelWidth: () => 2,
      animate: true
    }
  };

  return layouts[layoutName as keyof typeof layouts] || layouts['cose-bilkent'];
};
```

---

## Part 5: Interactive Features

### 5.1 Node Selection & Details

**Behavior:**
1. Click entity → Show entity details in right panel
2. Click relation → Show relation description and connected entities
3. Highlight connected nodes and edges

**Implementation:**

```typescript
// GraphDetailsPanel.tsx
const GraphDetailsPanel: React.FC = () => {
  const { selectedNode, nodes, edges } = useGraphStore();
  const [nodeDetails, setNodeDetails] = useState<NodeDetails | null>(null);

  useEffect(() => {
    if (!selectedNode) {
      setNodeDetails(null);
      return;
    }

    // Find node
    const node = nodes.find(n => n.data.id === selectedNode);
    if (!node) return;

    // Find connected edges
    const connectedEdges = edges.filter(
      e => e.data.source === selectedNode || e.data.target === selectedNode
    );

    // Find neighbors
    const neighbors = new Set<string>();
    connectedEdges.forEach(edge => {
      const neighborId = edge.data.source === selectedNode ? edge.data.target : edge.data.source;
      neighbors.add(neighborId);
    });

    const neighborNodes = nodes.filter(n => neighbors.has(n.data.id));

    setNodeDetails({
      node: node.data,
      neighbors: neighborNodes.map(n => n.data),
      connectedEdges: connectedEdges.length,
      degree: neighbors.size
    });
  }, [selectedNode, nodes, edges]);

  if (!nodeDetails) {
    return (
      <div className="w-80 bg-white border-l p-6">
        <p className="text-gray-500">Select a node to view details</p>
      </div>
    );
  }

  return (
    <div className="w-80 bg-white border-l p-6 overflow-y-auto">
      <h2 className="text-xl font-bold mb-4">Node Details</h2>

      {/* Node Type Badge */}
      <div className="mb-4">
        <span className={`px-3 py-1 rounded-full text-sm font-medium ${
          nodeDetails.node.type === 'entity'
            ? 'bg-blue-100 text-blue-800'
            : 'bg-red-100 text-red-800'
        }`}>
          {nodeDetails.node.type.toUpperCase()}
        </span>
      </div>

      {/* Node Label/Name */}
      <div className="mb-4">
        <h3 className="text-lg font-semibold">{nodeDetails.node.label}</h3>
      </div>

      {/* Description (for relations) */}
      {nodeDetails.node.description && (
        <div className="mb-4">
          <p className="text-sm text-gray-600">{nodeDetails.node.description}</p>
        </div>
      )}

      {/* Metadata */}
      <div className="mb-4 space-y-2">
        <div className="flex justify-between text-sm">
          <span className="text-gray-600">Weight:</span>
          <span className="font-medium">{nodeDetails.node.weight.toFixed(2)}</span>
        </div>
        <div className="flex justify-between text-sm">
          <span className="text-gray-600">Connections:</span>
          <span className="font-medium">{nodeDetails.degree}</span>
        </div>
        <div className="flex justify-between text-sm">
          <span className="text-gray-600">Edges:</span>
          <span className="font-medium">{nodeDetails.connectedEdges}</span>
        </div>
      </div>

      {/* Connected Nodes */}
      <div className="mb-4">
        <h4 className="font-semibold mb-2">Connected Nodes ({nodeDetails.neighbors.length})</h4>
        <div className="space-y-2 max-h-64 overflow-y-auto">
          {nodeDetails.neighbors.map(neighbor => (
            <button
              key={neighbor.id}
              className="w-full text-left p-2 hover:bg-gray-100 rounded flex items-center gap-2"
              onClick={() => selectNode(neighbor.id)}
            >
              <div className={`w-3 h-3 rounded-full ${
                neighbor.type === 'entity' ? 'bg-blue-500' : 'bg-red-500'
              }`} />
              <span className="text-sm truncate">{neighbor.label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Actions */}
      <div className="space-y-2">
        <button
          className="w-full px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
          onClick={() => expandNeighbors(selectedNode)}
        >
          Expand Neighbors
        </button>
        <button
          className="w-full px-4 py-2 bg-gray-200 rounded hover:bg-gray-300"
          onClick={() => highlightPath(selectedNode)}
        >
          Highlight Paths
        </button>
      </div>
    </div>
  );
};
```

### 5.2 Node Expansion (Load Neighbors)

**Behavior:**
Double-click a node → Load its immediate neighbors from backend

**Implementation:**

```typescript
const expandNeighbors = async (nodeId: string) => {
  try {
    const response = await api.get(`/graph/node/${nodeId}`, {
      params: { data_source: 'SingleTopic' }
    });

    const { neighbors, edges } = response.data;

    // Transform to Cytoscape format
    const newNodes = neighbors.map((neighborId: string) => {
      // Fetch node details
      const nodeData = await getNodeDetails(neighborId);
      return {
        data: {
          id: neighborId,
          label: nodeData.label,
          type: nodeData.type,
          weight: nodeData.weight,
          size: calculateNodeSize(nodeData.weight, nodeData.type)
        }
      };
    });

    const newEdges = edges.map((edge: any) => ({
      data: {
        id: `${edge.source}_${edge.target}`,
        source: edge.source,
        target: edge.target,
        weight: calculateEdgeWidth(edge.weight)
      }
    }));

    // Add to graph
    if (cyRef.current) {
      cyRef.current.add([...newNodes, ...newEdges]);

      // Re-run layout
      const layout = cyRef.current.layout(getLayoutOptions(currentLayout));
      layout.run();
    }

    toast.success(`Loaded ${newNodes.length} neighbors`);
  } catch (error) {
    toast.error('Failed to load neighbors');
  }
};
```

### 5.3 Path Highlighting

**Behavior:**
Click "Highlight Paths" → Show all paths from selected node

**Implementation:**

```typescript
const highlightPath = (startNodeId: string) => {
  if (!cyRef.current) return;

  const cy = cyRef.current;
  const startNode = cy.getElementById(startNodeId);

  // Get all neighbors (1-hop)
  const neighbors = startNode.neighborhood();

  // Dim all nodes except start node and neighbors
  cy.nodes().addClass('dimmed');
  startNode.removeClass('dimmed').addClass('selected');
  neighbors.removeClass('dimmed').addClass('connected');

  // Highlight edges
  cy.edges().style('opacity', 0.1);
  neighbors.connectedEdges().style('opacity', 1);
};

const clearHighlight = () => {
  if (!cyRef.current) return;

  const cy = cyRef.current;
  cy.nodes().removeClass('dimmed selected connected');
  cy.edges().style('opacity', 0.6);
};
```

### 5.4 Search & Filter

**Toolbar Component:**

```typescript
// GraphToolbar.tsx
const GraphToolbar: React.FC = () => {
  const { filters, setFilters, layout, setLayout } = useGraphStore();
  const [searchQuery, setSearchQuery] = useState('');

  const handleSearch = async (query: string) => {
    if (!query.trim()) {
      clearHighlight();
      return;
    }

    try {
      const results = await searchNodes(query, 20);

      if (results.length === 0) {
        toast.info('No matching nodes found');
        return;
      }

      // Highlight matching nodes
      if (cyRef.current) {
        const cy = cyRef.current;
        const matchIds = new Set(results.map(n => n.data.id));

        cy.nodes().forEach(node => {
          if (matchIds.has(node.id())) {
            node.addClass('connected');
          } else {
            node.addClass('dimmed');
          }
        });
      }

      toast.success(`Found ${results.length} matching nodes`);
    } catch (error) {
      toast.error('Search failed');
    }
  };

  return (
    <div className="flex items-center gap-4 p-4 bg-white border-b">
      {/* Search */}
      <div className="flex-1 max-w-md">
        <input
          type="text"
          placeholder="Search nodes..."
          className="w-full px-4 py-2 border rounded-lg"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') handleSearch(searchQuery);
          }}
        />
      </div>

      {/* Layout Selector */}
      <select
        value={layout}
        onChange={(e) => setLayout(e.target.value)}
        className="px-4 py-2 border rounded-lg"
      >
        <option value="cose-bilkent">Cose-Bilkent (Bipartite)</option>
        <option value="dagre">Dagre (Hierarchical)</option>
        <option value="cola">Cola (Force-directed)</option>
        <option value="concentric">Concentric (Radial)</option>
      </select>

      {/* Filters */}
      <div className="flex gap-2">
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={filters.showEntities}
            onChange={(e) => setFilters({ ...filters, showEntities: e.target.checked })}
          />
          <span className="text-sm">Entities</span>
        </label>
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={filters.showRelations}
            onChange={(e) => setFilters({ ...filters, showRelations: e.target.checked })}
          />
          <span className="text-sm">Relations</span>
        </label>
      </div>

      {/* Weight Filter */}
      <div className="flex items-center gap-2">
        <label className="text-sm">Min Weight:</label>
        <input
          type="range"
          min="0"
          max="1"
          step="0.1"
          value={filters.minWeight}
          onChange={(e) => setFilters({ ...filters, minWeight: parseFloat(e.target.value) })}
          className="w-24"
        />
        <span className="text-sm">{filters.minWeight.toFixed(1)}</span>
      </div>
    </div>
  );
};
```

### 5.5 Context Menu (Right-Click)

**Implementation:**

```typescript
// Install: npm install cytoscape-context-menus
import contextMenus from 'cytoscape-context-menus';
import 'cytoscape-context-menus/cytoscape-context-menus.css';

cytoscape.use(contextMenus);

// In GraphCanvas.tsx
useEffect(() => {
  if (!cyRef.current) return;

  const cy = cyRef.current;

  cy.contextMenus({
    menuItems: [
      {
        id: 'expand',
        content: 'Expand Neighbors',
        selector: 'node',
        onClickFunction: (event: any) => {
          const node = event.target;
          expandNeighbors(node.id());
        }
      },
      {
        id: 'highlight',
        content: 'Highlight Paths',
        selector: 'node',
        onClickFunction: (event: any) => {
          const node = event.target;
          highlightPath(node.id());
        }
      },
      {
        id: 'hide',
        content: 'Hide Node',
        selector: 'node',
        onClickFunction: (event: any) => {
          const node = event.target;
          node.style('display', 'none');
        }
      },
      {
        id: 'details',
        content: 'View Details',
        selector: 'node',
        onClickFunction: (event: any) => {
          const node = event.target;
          selectNode(node.id());
        }
      }
    ]
  });
}, []);
```

---

## Part 6: Performance Optimization

### 6.1 Viewport Culling

**Problem:** Rendering 15K nodes even when only 500 are visible

**Solution:** Only render nodes within viewport bounds

```typescript
// Enable in Cytoscape config
const cy = cytoscape({
  // ... other config

  // Performance optimizations
  hideEdgesOnViewport: true,  // Hide edges when panning/zooming
  textureOnViewport: true,    // Use texture for faster rendering
  motionBlur: true,           // Smooth animations
  pixelRatio: 'auto',         // Handle retina displays

  // Viewport culling (custom implementation)
});

// Implement custom culling
cy.on('viewport', debounce(() => {
  const extent = cy.extent();

  cy.nodes().forEach(node => {
    const pos = node.position();
    const isVisible = (
      pos.x >= extent.x1 && pos.x <= extent.x2 &&
      pos.y >= extent.y1 && pos.y <= extent.y2
    );

    if (isVisible) {
      node.style('display', 'element');
    } else {
      node.style('display', 'none');
    }
  });
}, 100));
```

### 6.2 Lazy Loading

**Strategy:** Load graph in chunks as user explores

```typescript
interface LoadingStrategy {
  initial: 1000,      // Load 1000 nodes on page load
  perExpansion: 500,  // Load 500 more when user clicks "Load More"
  perNode: 50         // Load 50 neighbors when user expands a node
}

const loadInitialGraph = async () => {
  const data = await getGraphData('SingleTopic', {
    limit: 1000,
    sampleStrategy: 'diverse'  // Balanced entities + relations
  });

  renderGraph(data);
};

const loadMoreNodes = async () => {
  const currentCount = nodes.length;

  const data = await getGraphData('SingleTopic', {
    limit: 500,
    offset: currentCount,
    sampleStrategy: 'top_weighted'
  });

  appendToGraph(data);
};
```

### 6.3 Web Workers for Layout

**Problem:** Layout calculations block main thread

**Solution:** Offload to web worker

```typescript
// graph-layout.worker.ts
import cytoscape from 'cytoscape';
import coseBilkent from 'cytoscape-cose-bilkent';

cytoscape.use(coseBilkent);

self.onmessage = (e) => {
  const { nodes, edges, layoutName } = e.data;

  // Create headless Cytoscape instance
  const cy = cytoscape({
    headless: true,
    elements: [...nodes, ...edges]
  });

  // Run layout
  const layout = cy.layout(getLayoutOptions(layoutName));
  layout.run();

  // Extract positions
  const positions = cy.nodes().map(node => ({
    id: node.id(),
    x: node.position('x'),
    y: node.position('y')
  }));

  // Send back to main thread
  self.postMessage({ positions });
};

// In GraphCanvas.tsx
const layoutWorker = new Worker(new URL('./graph-layout.worker.ts', import.meta.url));

const applyLayoutAsync = (layoutName: string) => {
  const nodes = cy.nodes().map(n => n.json());
  const edges = cy.edges().map(e => e.json());

  layoutWorker.postMessage({ nodes, edges, layoutName });
};

layoutWorker.onmessage = (e) => {
  const { positions } = e.data;

  positions.forEach(({ id, x, y }) => {
    cy.getElementById(id).position({ x, y });
  });
};
```

### 6.4 Caching

**Backend:** Cache graph transformations

```python
from functools import lru_cache

@lru_cache(maxsize=10)
def load_graph_cached(data_source: str):
    """Cache loaded graphs in memory"""
    graph_file = f"expr/{data_source}/graph_chunk_entity_relation.graphml"
    return nx.read_graphml(graph_file)
```

**Frontend:** Cache API responses

```typescript
// src/services/graph.ts
const graphCache = new Map<string, GraphData>();

export const getGraphData = async (
  dataSource: string,
  options: GraphLoadOptions = {}
): Promise<GraphData> => {
  const cacheKey = `${dataSource}_${JSON.stringify(options)}`;

  if (graphCache.has(cacheKey)) {
    console.log('[Graph Service] Using cached data');
    return graphCache.get(cacheKey)!;
  }

  const data = await fetchGraphData(dataSource, options);
  graphCache.set(cacheKey, data);

  return data;
};
```

---

## Part 7: UX Design

### 7.1 Visual Hierarchy

**Node Importance:**
- Size proportional to weight
- Higher weight = larger node
- Color saturation for emphasis

**Entity vs Relation Distinction:**
- Entities: Blue circles (recognizable objects)
- Relations: Red rectangles (connecting descriptions)
- Clear visual separation

**Edge Styling:**
- Thin edges for low weight
- Thick edges for high weight
- Bezier curves for clarity
- Gray color (non-intrusive)

### 7.2 Tooltips

**On Hover:**
```typescript
const showTooltip = (nodeData: NodeData) => {
  const tooltip = document.createElement('div');
  tooltip.className = 'graph-tooltip';
  tooltip.innerHTML = `
    <div class="font-bold">${nodeData.label}</div>
    <div class="text-sm text-gray-600">${nodeData.type}</div>
    ${nodeData.description ? `<div class="text-xs mt-2">${nodeData.description}</div>` : ''}
    <div class="text-xs mt-1">Weight: ${nodeData.weight.toFixed(2)}</div>
  `;

  document.body.appendChild(tooltip);

  // Position near cursor
  tooltip.style.position = 'absolute';
  tooltip.style.left = `${event.clientX + 10}px`;
  tooltip.style.top = `${event.clientY + 10}px`;
};
```

### 7.3 Legend

```typescript
const GraphLegend: React.FC = () => {
  return (
    <div className="absolute bottom-4 left-4 bg-white p-4 rounded-lg shadow-lg">
      <h3 className="font-semibold mb-2">Legend</h3>
      <div className="space-y-2">
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded-full bg-blue-500" />
          <span className="text-sm">Entity</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-red-500" />
          <span className="text-sm">Relation</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded-full bg-green-500" />
          <span className="text-sm">Chunk</span>
        </div>
      </div>
    </div>
  );
};
```

### 7.4 Loading States

```typescript
const GraphLoadingState: React.FC = () => {
  return (
    <div className="flex flex-col items-center justify-center h-full">
      <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mb-4" />
      <p className="text-gray-600">Loading knowledge graph...</p>
      <p className="text-sm text-gray-500 mt-2">
        This may take a few seconds for large graphs
      </p>
    </div>
  );
};
```

---

## Part 8: Implementation Phases

### Phase 1: Foundation 

**Goal:** Basic hypergraph rendering with entities and relations

**Tasks:**
1. Backend: Update `/graph/export` endpoint
   - Add `include_relation_nodes` parameter
   - Extract relation descriptions from node IDs
   - Return balanced sample (60% entities, 40% relations)

2. Frontend: Update GraphCanvas component
   - Add relation node styling (red rectangles)
   - Implement cose-bilkent layout
   - Add basic click handlers

3. Testing:
   - Verify both entities and relations render
   - Check bipartite structure is visible
   - Confirm performance with 1000 nodes

**Success Criteria:**
- Graph shows blue circles (entities) AND red rectangles (relations)
- Layout separates entities and relations visually
- Loads in < 3 seconds

### Phase 2: Interactivity 

**Goal:** Add interactive features for exploration

**Tasks:**
1. Node selection:
   - Click to select
   - Highlight connected nodes
   - Show details in right panel

2. Path highlighting:
   - Dim unrelated nodes
   - Brighten selected path
   - Clear highlight button

3. Search:
   - Search bar in toolbar
   - Highlight matching nodes
   - Jump to first match

4. Context menu:
   - Right-click options
   - Expand neighbors
   - Hide node
   - View details

**Success Criteria:**
- Can explore entity → relation → entity paths
- Search finds and highlights nodes
- Context menu works smoothly

### Phase 3: Advanced Visualization 

**Goal:** Multiple layouts and advanced styling

**Tasks:**
1. Layout algorithms:
   - Dagre (hierarchical)
   - Cola (force-directed)
   - Concentric (radial)
   - Layout switcher in toolbar

2. Advanced styling:
   - Node size by weight
   - Edge width by weight
   - Color gradients
   - Animation transitions

3. Export:
   - PNG screenshot
   - JSON export
   - GraphML export

**Success Criteria:**
- All 4 layouts work smoothly
- Visual hierarchy is clear
- Export functions work

### Phase 4: Performance & Polish 

**Goal:** Optimize for large graphs and polish UX

**Tasks:**
1. Performance:
   - Viewport culling
   - Web worker layouts
   - Lazy loading
   - Caching

2. UX polish:
   - Smooth animations
   - Tooltips
   - Loading states
   - Error handling

3. Documentation:
   - User guide
   - API documentation
   - Code comments

**Success Criteria:**
- Handles 5000+ nodes smoothly
- Professional look and feel
- Complete documentation

---

## Part 9: Testing Strategy

### 9.1 Unit Tests

**Backend:**
```python
# test_graph_export.py
def test_graph_export_includes_relations():
    response = client.get("/graph/export?data_source=SingleTopic&limit=100")
    data = response.json()

    # Check both entities and relations are present
    node_types = {n['type'] for n in data['nodes']}
    assert 'entity' in node_types
    assert 'relation' in node_types

    # Check bipartite structure
    relation_nodes = [n for n in data['nodes'] if n['type'] == 'relation']
    assert len(relation_nodes) > 0
```

**Frontend:**
```typescript
// GraphCanvas.test.tsx
describe('GraphCanvas', () => {
  it('renders both entities and relations', () => {
    const nodes = [
      { data: { id: '1', type: 'entity', label: 'Entity 1' } },
      { data: { id: '2', type: 'relation', label: 'Relation 1' } }
    ];

    render(<GraphCanvas nodes={nodes} edges={[]} />);

    // Check Cytoscape instance
    expect(cyRef.current.nodes('[type="entity"]').length).toBe(1);
    expect(cyRef.current.nodes('[type="relation"]').length).toBe(1);
  });
});
```

### 9.2 Integration Tests

**Full Flow:**
```typescript
// integration.test.tsx
describe('Graph Visualization Integration', () => {
  it('loads and displays graph correctly', async () => {
    render(<GraphViz />);

    // Wait for graph to load
    await waitFor(() => {
      expect(screen.queryByText('Loading graph...')).not.toBeInTheDocument();
    });

    // Check graph is rendered
    const canvas = screen.getByTestId('graph-canvas');
    expect(canvas).toBeInTheDocument();

    // Check stats are displayed
    expect(screen.getByText(/Entities:/)).toBeInTheDocument();
    expect(screen.getByText(/Relations:/)).toBeInTheDocument();
  });
});
```

### 9.3 Performance Tests

**Benchmarks:**
```typescript
// performance.test.ts
describe('Graph Performance', () => {
  it('loads 1000 nodes in < 3 seconds', async () => {
    const startTime = performance.now();

    await loadGraph('SingleTopic', { limit: 1000 });

    const endTime = performance.now();
    const duration = endTime - startTime;

    expect(duration).toBeLessThan(3000);
  });

  it('applies layout in < 2 seconds', async () => {
    await loadGraph('SingleTopic', { limit: 1000 });

    const startTime = performance.now();

    applyLayout('cose-bilkent');

    const endTime = performance.now();
    const duration = endTime - startTime;

    expect(duration).toBeLessThan(2000);
  });
});
```

---

## Part 10: Success Metrics

### 10.1 Performance Targets

| Metric | Target | Stretch Goal |
|--------|--------|--------------|
| **Initial Load Time** | < 3 seconds | < 2 seconds |
| **Layout Calculation** | < 2 seconds | < 1 second |
| **Search Response** | < 500ms | < 200ms |
| **Node Selection** | < 100ms | < 50ms |
| **Memory Usage** | < 500MB | < 300MB |
| **Max Nodes (Smooth)** | 5000 nodes | 10000 nodes |

### 10.2 UX Targets

| Feature | Target |
|---------|--------|
| **Visual Clarity** | Entities and relations clearly distinguishable |
| **Path Visibility** | Can trace entity → relation → entity paths |
| **Interaction** | All interactions respond within 100ms |
| **Search** | Find nodes in < 500ms |
| **Layout Switching** | Change layouts smoothly with animation |
| **Tooltips** | Show on hover within 200ms |

### 10.3 Functional Requirements

- ✅ Display entities (blue circles) and relations (red rectangles)
- ✅ Show bipartite graph structure clearly
- ✅ Support 4+ layout algorithms
- ✅ Interactive node selection and details
- ✅ Search and filter nodes
- ✅ Expand node neighbors on demand
- ✅ Highlight paths between nodes
- ✅ Export graph (PNG, JSON, GraphML)
- ✅ Handle 1000-5000 nodes smoothly
- ✅ Progressive loading for large graphs

---

## Part 11: Comparison with PathRAG

### 11.1 Key Differences

| Aspect | PathRAG | BiG-RAG |
|--------|---------|---------|
| **Graph Type** | Normal graph (entities + edges) | Hypergraph (entities + relation nodes) |
| **Relations** | Edge labels | First-class nodes |
| **Framework** | Sigma.js + Graphology | Cytoscape.js |
| **Renderer** | WebGL only | Canvas + WebGL |
| **Layout** | ForceAtlas2 | Cose-Bilkent (bipartite-optimized) |
| **Node Types** | 1 type (entities) | 2 types (entities + relations) |
| **Visual Design** | Single color scheme | Dual color scheme (blue + red) |
| **Storage** | Graph database | NetworkX GraphML |

### 11.2 Lessons from PathRAG

**What to Adopt:**
1. **Progressive loading** - Load in chunks as user explores
2. **Search with highlighting** - Visual feedback for search results
3. **Mini-map navigation** - Overview of full graph
4. **Export functionality** - PNG, JSON, GraphML

**What to Modify:**
1. **Layout algorithm** - Use bipartite-specific layouts (not ForceAtlas2)
2. **Node styling** - Distinguish entities vs relations (not single color)
3. **Interaction model** - Focus on entity → relation → entity paths

**What's Unique to BiG-RAG:**
1. **Relation nodes** - Relations are nodes, not edges
2. **Bipartite structure** - Two distinct node types
3. **Semantic descriptions** - Relations contain full text descriptions
4. **Weight aggregation** - Relation weights = sum of occurrences

---

## Part 12: Risk Mitigation

### 12.1 Technical Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Browser freeze with large graphs** | HIGH | Implement viewport culling, lazy loading |
| **Layout calculation too slow** | MEDIUM | Use web workers, pre-calculate positions |
| **Memory leaks** | HIGH | Proper cleanup, use React refs |
| **Cytoscape version conflicts** | LOW | Lock versions in package.json |
| **Backend timeout** | MEDIUM | Increase timeout to 120s, add retry logic |

### 12.2 UX Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Confusing bipartite structure** | HIGH | Clear legend, color coding, tooltips |
| **Too many nodes visible** | HIGH | Default to 1000 nodes, show warning |
| **Slow search** | MEDIUM | Debounce input, limit results |
| **Layout switching jarring** | LOW | Add smooth animations |
| **Missing context** | MEDIUM | Comprehensive tooltips and details panel |

---

## Part 13: Future Enhancements

### 13.1 Short-term (3 months)

1. **Subgraph extraction**
   - Select multiple nodes
   - Extract subgraph between them
   - Export subgraph

2. **Path finding**
   - Find shortest path between two entities
   - Visualize all paths
   - Path length distribution

3. **Node clustering**
   - Detect communities
   - Color by cluster
   - Cluster statistics

### 13.2 Long-term (6+ months)

1. **Real-time updates**
   - WebSocket connection
   - Live graph updates when documents are added
   - Animation for new nodes

2. **3D visualization**
   - 3D force-directed layout
   - WebGL renderer
   - VR support

3. **Graph analytics**
   - Centrality measures
   - PageRank
   - Betweenness centrality
   - Export analytics to CSV

4. **Collaborative features**
   - Share graph views
   - Annotations
   - Comments on nodes

---

## Appendix A: File Checklist

### Backend Files to Modify

- [ ] `backend/server.py`
  - [ ] Update `/graph/export` endpoint (add relation node support)
  - [ ] Add `/graph/node/{node_id}` endpoint
  - [ ] Add diverse sampling strategy

### Frontend Files to Create

- [ ] `src/components/graph/GraphCanvas.tsx` (new component)
- [ ] `src/components/graph/GraphToolbar.tsx` (new component)
- [ ] `src/components/graph/GraphDetailsPanel.tsx` (new component)
- [ ] `src/components/graph/GraphLegend.tsx` (new component)
- [ ] `src/components/graph/GraphControls.tsx` (new component)
- [ ] `src/components/graph/GraphContextMenu.tsx` (new component)
- [ ] `src/components/graph/GraphLoadingState.tsx` (new component)

### Frontend Files to Modify

- [ ] `src/pages/GraphViz.tsx` (restructure)
- [ ] `src/services/graph.ts` (add relation support)
- [ ] `src/store/graph.ts` (update state)
- [ ] `src/hooks/useGraph.ts` (update hook)
- [ ] `src/types/graph.ts` (add relation types)

### Dependencies to Install

```bash
npm install cytoscape-context-menus
npm install cytoscape-cola
npm install cytoscape-dagre
npm install cytoscape-expand-collapse
npm install cytoscape-navigator
```

---

## Appendix B: Code Snippets

### B.1 Complete Cytoscape Styling

```typescript
const cytoscapeStyles = [
  // Entity nodes
  {
    selector: 'node[type="entity"]',
    style: {
      'background-color': '#3B82F6',
      'shape': 'ellipse',
      'width': 'data(size)',
      'height': 'data(size)',
      'label': 'data(label)',
      'font-size': '12px',
      'font-weight': 'bold',
      'text-valign': 'center',
      'text-halign': 'center',
      'text-wrap': 'wrap',
      'text-max-width': '100px',
      'border-width': 2,
      'border-color': '#1E40AF',
      'color': '#FFFFFF',
      'text-outline-color': '#1E40AF',
      'text-outline-width': 2
    }
  },

  // Relation nodes
  {
    selector: 'node[type="relation"]',
    style: {
      'background-color': '#EF4444',
      'shape': 'rectangle',
      'width': 'data(size)',
      'height': 'data(size)',
      'label': 'data(label)',
      'font-size': '10px',
      'font-weight': 'normal',
      'text-valign': 'center',
      'text-halign': 'center',
      'text-wrap': 'wrap',
      'text-max-width': '120px',
      'border-width': 2,
      'border-color': '#991B1B',
      'color': '#FFFFFF',
      'text-outline-color': '#991B1B',
      'text-outline-width': 1
    }
  },

  // Chunk nodes
  {
    selector: 'node[type="chunk"]',
    style: {
      'background-color': '#10B981',
      'shape': 'round-rectangle',
      'width': 'data(size)',
      'height': 'data(size)',
      'label': 'data(label)',
      'font-size': '10px',
      'text-valign': 'center',
      'text-halign': 'center',
      'text-wrap': 'wrap',
      'text-max-width': '100px',
      'border-width': 2,
      'border-color': '#047857',
      'color': '#FFFFFF'
    }
  },

  // Edges
  {
    selector: 'edge',
    style: {
      'width': 'data(width)',
      'line-color': '#94A3B8',
      'opacity': 0.6,
      'curve-style': 'bezier',
      'target-arrow-shape': 'none',
      'source-arrow-shape': 'none'
    }
  },

  // Hover state
  {
    selector: 'node:active',
    style: {
      'border-color': '#FFD700',
      'border-width': 4,
      'overlay-opacity': 0.2,
      'overlay-color': '#FFD700'
    }
  },

  // Selected state
  {
    selector: 'node.selected',
    style: {
      'background-color': '#FFA500',
      'border-color': '#EA580C',
      'border-width': 5,
      'z-index': 999
    }
  },

  // Connected state
  {
    selector: 'node.connected',
    style: {
      'opacity': 1,
      'border-width': 3,
      'border-color': '#22C55E'
    }
  },

  // Dimmed state
  {
    selector: 'node.dimmed',
    style: {
      'opacity': 0.2
    }
  },

  // Highlighted edges
  {
    selector: 'edge.highlighted',
    style: {
      'opacity': 1,
      'line-color': '#22C55E',
      'width': 'calc(data(width) * 1.5)'
    }
  }
];
```

---

## Summary

This plan provides a **complete blueprint** for rebuilding BiG-RAG's graph visualization to properly display its unique hypergraph/bipartite structure.

**Key Takeaways:**

1. **BiG-RAG is fundamentally different** - Relations are nodes, not edges
2. **Cytoscape.js is the right choice** - Native bipartite layout support
3. **Visual distinction is critical** - Blue circles (entities) vs Red rectangles (relations)
4. **Performance requires smart sampling** - 1000 node default, progressive loading
5. **UX must support exploration** - Click, expand, search, highlight paths
6. **Implementation is phased** - Foundation → Interactivity → Advanced → Polish

**Next Steps:**

1. Review and approve this plan
2. Start Phase 1: Foundation (backend + basic rendering)
3. Test with SingleTopic dataset
4. Iterate based on user feedback

**Success will be measured by:**
- Can clearly see entity → relation → entity paths
- Smooth performance with 1000-5000 nodes
- Intuitive UX for knowledge graph exploration

---

**End of Plan**

*Ready for implementation. This document will serve as the authoritative blueprint for BiG-RAG graph visualization reconstruction.*

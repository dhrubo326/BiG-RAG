# Graph Visualization Redesign Plan

## Current Issues Identified

###  1. **Terminology Confusion (CRITICAL)**
- Old code uses "bipartite_edge" terminology
- Node IDs changed: `rel-{hash}` format for relation nodes
- Need to use `d1` (content) field for relation descriptions, not `d5`

### 2. **Orphan Nodes Not Displayed**
- Current implementation may filter out nodes without connections
- Need to show ALL nodes for KG inspection

### 3. **Poor Search UX**
- Search is in toolbar (crowded)
- Results not displayed properly
- Input area too small

### 4. **Data Structure Misunderstanding**
Based on `expr/demo_test/graph_chunk_entity_relation.graphml`:

**Entity Nodes:**
```xml
<node id="&quot;LIONEL MESSI&quot;">
  <data key="d0">entity</data>          <!-- role -->
  <data key="d4">person</data>          <!-- entity_type -->
  <data key="d5">Description here</data> <!-- description -->
  <data key="d3">chunk-xxx</data>       <!-- source_id -->
  <data key="d2">90.0</data>            <!-- weight -->
</node>
```

**Relation Nodes:**
```xml
<node id="rel-79a06bae5060c9ecba31015810f133a4">
  <data key="d0">relation</data>        <!-- role -->
  <data key="d1">Content here</data>    <!-- content (MAIN DATA) -->
  <data key="d2">9.0</data>             <!-- weight -->
  <data key="d3">chunk-xxx</data>       <!-- source_id -->
</node>
```

**Key Differences:**
- Entity nodes: `id` = entity name (quoted), use `d5` for description
- Relation nodes: `id` = hash (`rel-{md5}`), use `d1` for content
- Both have `d0` (role), `d2` (weight), `d3` (source_id)

---

## Redesign Plan

### Phase 1: Fix Backend API Response

**File:** `backend/api/services/graph_export.py`

**Current Issues:**
- May not return orphan nodes
- May use wrong field for relation content

**Required Changes:**
1. Ensure ALL nodes are returned (no filtering by connection count)
2. For relation nodes: use `d1` (content) field as description
3. For entity nodes: use `d5` (description) field
4. Add `node_type` field to distinguish: "entity" | "relation" | "chunk"

**New Response Format:**
```typescript
{
  nodes: [
    {
      id: "\"LIONEL MESSI\"",
      label: "LIONEL MESSI",
      type: "entity",
      entityType: "person",
      description: "Full description from d5",
      weight: 90.0,
      sourceId: "chunk-xxx",
      connections: 5  // Graph edge count
    },
    {
      id: "rel-79a06bae...",
      label: "Lionel Messi, widely regarded...", // First 50 chars of content
      type: "relation",
      content: "Full content from d1",  // MAIN DATA
      weight: 9.0,
      sourceId: "chunk-xxx",
      connections: 2
    }
  ],
  edges: [
    {
      id: "edge-1",
      source: "\"LIONEL MESSI\"",
      target: "rel-79a06bae...",
      weight: 1.0
    }
  ],
  stats: {
    totalNodes: 197,
    totalEdges: 130,
    entities: 115,
    relations: 82,
    chunks: 0,
    orphanNodes: 15
  }
}
```

---

### Phase 2: Redesign Frontend UI

**File:** `frontend/src/pages/GraphViz.tsx`

#### 2.1 **Move Search to Top Bar** (Next to Dataset Selector)

**Old Layout:**
```
[Dataset Selector]
[Toolbar with search inside]
[Graph Canvas]
```

**New Layout:**
```
[Dataset Selector] [Search Input (wide)] [Stats Badge]
[Toolbar: Layout controls, filters, export]
[Graph Canvas]
```

#### 2.2 **Improved Search UI**

```tsx
{/* Top Bar: Dataset + Search + Quick Stats */}
<div className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 px-4 py-3">
  <div className="flex items-center gap-4">
    {/* Dataset Selector */}
    <div className="flex items-center gap-2">
      <label className="text-sm font-medium text-gray-700 dark:text-gray-300 whitespace-nowrap">
        Dataset:
      </label>
      <select className="...">
        {/* options */}
      </select>
    </div>

    {/* Search Bar (Wide, prominent) */}
    <div className="flex-1 max-w-2xl">
      <div className="relative">
        <input
          type="text"
          placeholder="Search entities, relations, or chunks (min 3 characters)..."
          value={searchQuery}
          onChange={(e) => searchNodes(e.target.value)}
          className="w-full px-4 py-2 pl-10 pr-10 text-sm border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all"
        />
        <SearchIcon className="absolute left-3 top-2.5 w-5 h-5 text-gray-400" />
        {searchQuery && (
          <button
            onClick={() => searchNodes('')}
            className="absolute right-3 top-2.5 text-gray-400 hover:text-gray-600"
          >
            <XIcon className="w-4 h-4" />
          </button>
        )}
      </div>

      {/* Search Results Dropdown */}
      {searchResults.length > 0 && (
        <div className="absolute z-50 mt-1 w-full max-w-2xl bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg shadow-xl max-h-96 overflow-y-auto">
          <div className="p-2 text-xs text-gray-500 dark:text-gray-400 border-b">
            Found {searchResults.length} results
          </div>
          {searchResults.map((result) => (
            <button
              key={result.id}
              onClick={() => selectAndZoomToNode(result.id)}
              className="w-full px-4 py-3 text-left hover:bg-gray-100 dark:hover:bg-gray-700 border-b border-gray-100 dark:border-gray-700 transition-colors"
            >
              <div className="flex items-start gap-3">
                {/* Node Type Indicator */}
                <div className={`flex-shrink-0 w-6 h-6 rounded-full flex items-center justify-center ${
                  result.type === 'entity' ? 'bg-blue-500' :
                  result.type === 'relation' ? 'bg-red-500' : 'bg-green-500'
                }`}>
                  {result.type === 'entity' ? 'E' : result.type === 'relation' ? 'R' : 'C'}
                </div>

                {/* Node Info */}
                <div className="flex-1 min-w-0">
                  <div className="font-medium text-sm text-gray-900 dark:text-gray-100 truncate">
                    {result.label}
                  </div>
                  <div className="text-xs text-gray-500 dark:text-gray-400 truncate mt-0.5">
                    {result.type === 'entity'
                      ? `Type: ${result.entityType} • Weight: ${result.weight}`
                      : `Weight: ${result.weight} • ${result.connections} connections`
                    }
                  </div>
                </div>
              </div>
            </button>
          ))}
        </div>
      )}
    </div>

    {/* Quick Stats */}
    <div className="flex items-center gap-4 text-sm">
      <div className="flex items-center gap-2">
        <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
        <span className="font-semibold text-gray-900 dark:text-gray-100">
          {stats.entities}
        </span>
      </div>
      <div className="flex items-center gap-2">
        <div className="w-3 h-3 bg-red-500 transform rotate-45"></div>
        <span className="font-semibold text-gray-900 dark:text-gray-100">
          {stats.relations}
        </span>
      </div>
      <div className="text-xs text-gray-500 dark:text-gray-400">
        {stats.orphanNodes > 0 && `${stats.orphanNodes} orphan nodes`}
      </div>
    </div>
  </div>
</div>
```

#### 2.3 **Node Info Panel Updates**

**File:** `frontend/src/components/graph/NodeInfoPanel.tsx`

Update to show correct fields:
- Entity nodes: Show `description` (from d5)
- Relation nodes: Show `content` (from d1) - this is the main data
- Add "Orphan Node" badge if no connections

```tsx
{/* Node Content */}
<div className="space-y-4">
  {/* Node Type Badge */}
  <div className="flex items-center gap-2">
    <span className={`px-3 py-1 rounded-full text-xs font-semibold ${
      node.data.type === 'entity' ? 'bg-blue-100 text-blue-700' :
      node.data.type === 'relation' ? 'bg-red-100 text-red-700' :
      'bg-green-100 text-green-700'
    }`}>
      {node.data.type.toUpperCase()}
    </span>

    {node.data.connections === 0 && (
      <span className="px-3 py-1 rounded-full text-xs font-semibold bg-yellow-100 text-yellow-700">
        ORPHAN NODE
      </span>
    )}
  </div>

  {/* Entity-specific fields */}
  {node.data.type === 'entity' && (
    <>
      <div>
        <label className="text-xs font-semibold text-gray-500 uppercase">Entity Type</label>
        <p className="text-sm text-gray-900 dark:text-gray-100 mt-1">{node.data.entityType}</p>
      </div>
      <div>
        <label className="text-xs font-semibold text-gray-500 uppercase">Description</label>
        <p className="text-sm text-gray-900 dark:text-gray-100 mt-1">{node.data.description}</p>
      </div>
    </>
  )}

  {/* Relation-specific fields */}
  {node.data.type === 'relation' && (
    <div>
      <label className="text-xs font-semibold text-gray-500 uppercase">Content (Knowledge Fragment)</label>
      <p className="text-sm text-gray-900 dark:text-gray-100 mt-1 p-3 bg-red-50 dark:bg-red-900/20 rounded-lg border border-red-200 dark:border-red-800">
        {node.data.content}
      </p>
    </div>
  )}

  {/* Common fields */}
  <div className="grid grid-cols-2 gap-4">
    <div>
      <label className="text-xs font-semibold text-gray-500 uppercase">Weight</label>
      <p className="text-sm font-mono text-gray-900 dark:text-gray-100 mt-1">{node.data.weight}</p>
    </div>
    <div>
      <label className="text-xs font-semibold text-gray-500 uppercase">Connections</label>
      <p className="text-sm font-mono text-gray-900 dark:text-gray-100 mt-1">{node.data.connections}</p>
    </div>
  </div>
</div>
```

---

### Phase 3: Cytoscape Styling Updates

**File:** `frontend/src/components/graph/GraphCanvas.tsx`

Update node styles to distinguish entity/relation properly:

```typescript
const cytoscapeStylesheet: StylesheetStyle[] = [
  // Entity nodes
  {
    selector: 'node[type="entity"]',
    style: {
      'background-color': '#3B82F6', // Blue
      'shape': 'ellipse',
      'width': (ele) => Math.max(30, Math.min(80, ele.data('weight') * 0.5)),
      'height': (ele) => Math.max(30, Math.min(80, ele.data('weight') * 0.5)),
      'label': 'data(label)',
      'font-size': '12px',
      'text-valign': 'center',
      'text-halign': 'center',
      'color': '#FFFFFF',
      'text-outline-color': '#1E40AF',
      'text-outline-width': 2,
      'border-width': 2,
      'border-color': '#1E40AF',
    },
  },

  // Relation nodes
  {
    selector: 'node[type="relation"]',
    style: {
      'background-color': '#EF4444', // Red
      'shape': 'diamond', // Changed from rectangle
      'width': (ele) => Math.max(25, Math.min(60, ele.data('weight') * 3)),
      'height': (ele) => Math.max(25, Math.min(60, ele.data('weight') * 3)),
      'label': 'data(label)', // First 50 chars of content
      'font-size': '10px',
      'text-valign': 'center',
      'text-halign': 'center',
      'color': '#FFFFFF',
      'text-outline-color': '#991B1B',
      'text-outline-width': 2,
      'border-width': 2,
      'border-color': '#991B1B',
    },
  },

  // Orphan nodes (highlighted)
  {
    selector: 'node[connections = 0]',
    style: {
      'border-width': 3,
      'border-color': '#FBBF24', // Yellow border
      'border-style': 'dashed',
    },
  },

  // ... rest of styles
];
```

---

### Phase 4: Backend API Fix

**File:** `backend/api/services/graph_export.py`

Key changes needed:

```python
async def export_graph_data(
    working_dir: str,
    data_source: str,
    limit: int = 1000,
    offset: int = 0,
    node_types: Optional[List[str]] = None,
    min_weight: float = 0.0,
    sample_strategy: str = "top",
) -> Dict[str, Any]:
    """
    Export graph data for visualization.

    IMPORTANT: Returns ALL nodes (including orphan nodes) for inspection.
    """
    # ... load GraphML ...

    all_nodes = []
    entity_count = 0
    relation_count = 0
    chunk_count = 0
    orphan_count = 0

    for node_id, node_attrs in G.nodes(data=True):
        role = node_attrs.get("role", "unknown")
        degree = G.degree(node_id)  # Connection count

        if degree == 0:
            orphan_count += 1

        # Base node data
        node_data = {
            "id": node_id,
            "type": role,
            "weight": float(node_attrs.get("weight", 0)),
            "sourceId": node_attrs.get("source_id", ""),
            "connections": degree,
        }

        # Entity-specific fields
        if role == "entity":
            entity_count += 1
            node_data.update({
                "label": node_id.strip('"'),  # Remove quotes
                "entityType": node_attrs.get("entity_type", "unknown"),
                "description": node_attrs.get("description", ""),  # d5 field
            })

        # Relation-specific fields
        elif role == "relation":
            relation_count += 1
            content = node_attrs.get("content", "")  # d1 field (MAIN DATA)
            node_data.update({
                "label": content[:50] + "..." if len(content) > 50 else content,
                "content": content,  # Full content from d1
            })

        # Chunk-specific fields
        elif role == "chunk":
            chunk_count += 1
            node_data.update({
                "label": node_id,
                "content": node_attrs.get("content", ""),
            })

        all_nodes.append(node_data)

    # Extract edges
    all_edges = []
    for source, target, edge_attrs in G.edges(data=True):
        all_edges.append({
            "id": f"{source}-{target}",
            "source": source,
            "target": target,
            "weight": float(edge_attrs.get("weight", 1.0)),
        })

    return {
        "nodes": all_nodes,
        "edges": all_edges,
        "stats": {
            "totalNodes": len(all_nodes),
            "totalEdges": len(all_edges),
            "entities": entity_count,
            "relations": relation_count,
            "chunks": chunk_count,
            "orphanNodes": orphan_count,
        }
    }
```

---

## Implementation Checklist

### Backend (Priority 1)
- [ ] Fix `graph_export.py`: Return ALL nodes (no filtering)
- [ ] Use `d1` (content) for relation nodes
- [ ] Use `d5` (description) for entity nodes
- [ ] Add `connections` field (degree count)
- [ ] Add `orphanNodes` to stats
- [ ] Test with `demo_test` dataset

### Frontend (Priority 2)
- [ ] Move search to top bar (next to dataset selector)
- [ ] Make search input wider (flex-1, max-w-2xl)
- [ ] Add search results dropdown
- [ ] Update `NodeInfoPanel` to show correct fields
- [ ] Add "Orphan Node" badge
- [ ] Update Cytoscape styles (diamond shape for relations)
- [ ] Highlight orphan nodes with dashed yellow border

### Testing (Priority 3)
- [ ] Verify orphan nodes are displayed
- [ ] Verify relation nodes show content (d1), not description (d5)
- [ ] Verify entity nodes show description (d5)
- [ ] Test search functionality
- [ ] Test with demo_test dataset (197 nodes, 130 edges)

---

## Expected Result

After redesign:
1. ✅ All 197 nodes visible (including orphan nodes)
2. ✅ Entity nodes (115) show proper descriptions from d5
3. ✅ Relation nodes (82) show content from d1
4. ✅ Search is prominent and usable
5. ✅ Clear visual distinction between node types
6. ✅ Orphan nodes are highlighted for inspection

---

## Questions to Resolve

1. Should we remove chunk nodes from visualization (or show separately)?
2. Should orphan nodes be positioned differently (e.g., bottom-right cluster)?
3. Do we want to filter by connection count (show only nodes with degree >= 1)?
4. Should search be real-time (on every keystroke) or triggered by Enter/button?

---

Generated: 2025-11-10
Dataset Reference: `expr/demo_test` (1 document, 115 entities, 82 relations)

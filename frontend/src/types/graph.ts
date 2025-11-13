// Graph-related type definitions

export interface GraphNode {
  id: string;
  label: string;
  type: 'entity' | 'relation' | 'chunk' | 'document';
  description?: string;
  content?: string;          // ✅ NEW: Relation content (d1 field)
  entityType?: string;        // ✅ NEW: Entity type (person, organization, etc.)
  weight?: number;
  connections?: number;       // ✅ NEW: Number of connections (for orphan detection)
  source_id?: string;
  sourceId?: string;          // Alternative naming from backend
  metadata?: Record<string, any>;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  label?: string;
  weight?: number;
  type?: string;
}

export interface CytoscapeNode {
  data: {
    id: string;
    label: string;
    type: 'entity' | 'relation' | 'chunk' | 'document';
    description?: string;
    content?: string;          // ✅ NEW: Relation content (d1 field)
    entityType?: string;        // ✅ NEW: Entity type (person, organization, etc.)
    weight?: number;
    connections?: number;       // ✅ NEW: Number of connections (for orphan detection)
    source_id?: string;
    sourceId?: string;          // Alternative naming from backend
    metadata?: Record<string, any>;
  };
  position?: { x: number; y: number };
  selected?: boolean;
  selectable?: boolean;
  locked?: boolean;
  grabbable?: boolean;
}

export interface CytoscapeEdge {
  data: {
    id: string;
    source: string;
    target: string;
    label?: string;
    weight?: number;
    type?: string;
  };
}

export type GraphLayout =
  | 'cose-bilkent'
  | 'dagre'
  | 'fcose'
  | 'grid'
  | 'circle'
  | 'concentric'
  | 'breadthfirst';

export interface GraphFilters {
  showEntities: boolean;
  showRelations: boolean;
  showChunks: boolean;
  minWeight: number;
  sourceDocument: string | null;
  showOrphans: boolean;         // ✅ NEW: Show/hide orphan nodes
}

export interface GraphStats {
  totalNodes: number;
  totalEdges: number;
  entities: number;
  relations: number;
  chunks: number;
  documents: number;
  orphanNodes?: number;       // ✅ NEW: Nodes with no connections
}

// ✅ NEW: Orphan node breakdown by type
export interface OrphanBreakdown {
  total: number;
  entities: number;
  relations: number;
  chunks: number;
  included_in_response: number;
  include_all_orphans_mode: boolean;
}

export interface GraphExportOptions {
  format: 'png' | 'json' | 'graphml';
  quality?: number;
  background?: string;
}
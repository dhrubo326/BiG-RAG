// Graph-related type definitions

export interface GraphNode {
  id: string;
  label: string;
  type: 'entity' | 'relation' | 'chunk' | 'document';
  description?: string;
  weight?: number;
  source_id?: string;
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
    weight?: number;
    source_id?: string;
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
}

export interface GraphStats {
  totalNodes: number;
  totalEdges: number;
  entities: number;
  relations: number;
  chunks: number;
  documents: number;
}

export interface GraphExportOptions {
  format: 'png' | 'json' | 'graphml';
  quality?: number;
  background?: string;
}
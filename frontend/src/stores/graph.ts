import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import type {
  CytoscapeNode,
  CytoscapeEdge,
  GraphLayout,
  GraphFilters,
  GraphStats,
  OrphanBreakdown,
} from '../types';
import { DEFAULT_CONFIG } from '../utils/constants';

interface GraphState {
  // Data
  nodes: CytoscapeNode[];
  edges: CytoscapeEdge[];
  selectedNode: CytoscapeNode | null;
  hoveredNode: CytoscapeNode | null;

  // Stats
  stats: GraphStats | null;
  orphanBreakdown: OrphanBreakdown | null;  // ✅ NEW

  // UI State
  layout: GraphLayout;
  filters: GraphFilters;
  isLoading: boolean;
  error: string | null;
  searchQuery: string;

  // ✅ PHASE 4.1: Progressive loading state
  currentDataset: string | null;
  currentOffset: number;
  canLoadMore: boolean;

  // Actions
  setNodes: (nodes: CytoscapeNode[]) => void;
  setEdges: (edges: CytoscapeEdge[]) => void;
  appendNodes: (nodes: CytoscapeNode[]) => void; // ✅ PHASE 4.1: Append instead of replace
  appendEdges: (edges: CytoscapeEdge[]) => void; // ✅ PHASE 4.1: Append instead of replace
  selectNode: (node: CytoscapeNode | null) => void;
  hoverNode: (node: CytoscapeNode | null) => void;
  setLayout: (layout: GraphLayout) => void;
  updateFilters: (filters: Partial<GraphFilters>) => void;
  setSearchQuery: (query: string) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setStats: (stats: GraphStats | null) => void;
  setOrphanBreakdown: (breakdown: OrphanBreakdown | null) => void; // ✅ NEW
  setCurrentDataset: (dataset: string | null) => void; // ✅ PHASE 4.1
  setCurrentOffset: (offset: number) => void; // ✅ PHASE 4.1
  setCanLoadMore: (canLoadMore: boolean) => void; // ✅ PHASE 4.1
  clearGraph: () => void;

  // Computed
  getFilteredNodes: () => CytoscapeNode[];
  getFilteredEdges: () => CytoscapeEdge[];
}

const useGraphStore = create<GraphState>()(
  devtools(
    (set, get) => ({
      // Initial state
      nodes: [],
      edges: [],
      selectedNode: null,
      hoveredNode: null,
      stats: null,
      orphanBreakdown: null,  // ✅ NEW
      layout: DEFAULT_CONFIG.GRAPH_LAYOUT,
      filters: {
        showEntities: true,
        showRelations: true,
        showChunks: true,
        minWeight: DEFAULT_CONFIG.GRAPH_MIN_WEIGHT,
        sourceDocument: null,
        showOrphans: true,  // ✅ NEW: Show orphans by default
      },
      isLoading: false,
      error: null,
      searchQuery: '',

      // ✅ PHASE 4.1: Progressive loading initial state
      currentDataset: null,
      currentOffset: 0,
      canLoadMore: false,

      // Actions
      setNodes: (nodes) => set({ nodes }),
      setEdges: (edges) => set({ edges }),

      // ✅ PHASE 4.1: Append actions for progressive loading
      appendNodes: (newNodes) =>
        set((state) => {
          // Filter out duplicate nodes
          const existingIds = new Set(state.nodes.map((n) => n.data.id));
          const uniqueNewNodes = newNodes.filter((n) => !existingIds.has(n.data.id));
          return { nodes: [...state.nodes, ...uniqueNewNodes] };
        }),

      appendEdges: (newEdges) =>
        set((state) => {
          // Filter out duplicate edges
          const existingIds = new Set(state.edges.map((e) => e.data.id));
          const uniqueNewEdges = newEdges.filter((e) => !existingIds.has(e.data.id));
          return { edges: [...state.edges, ...uniqueNewEdges] };
        }),

      selectNode: (node) => set({ selectedNode: node }),
      hoverNode: (node) => set({ hoveredNode: node }),

      setLayout: (layout) => set({ layout }),

      updateFilters: (filters) =>
        set((state) => ({
          filters: { ...state.filters, ...filters },
        })),

      setSearchQuery: (query) => set({ searchQuery: query }),

      setLoading: (loading) => set({ isLoading: loading }),
      setError: (error) => set({ error }),
      setStats: (stats) => set({ stats }),
      setOrphanBreakdown: (breakdown) => set({ orphanBreakdown: breakdown }), // ✅ NEW

      // ✅ PHASE 4.1: Progressive loading setters
      setCurrentDataset: (dataset) => set({ currentDataset: dataset }),
      setCurrentOffset: (offset) => set({ currentOffset: offset }),
      setCanLoadMore: (canLoadMore) => set({ canLoadMore }),

      clearGraph: () =>
        set({
          nodes: [],
          edges: [],
          selectedNode: null,
          hoveredNode: null,
          stats: null,
          error: null,
          currentOffset: 0,
          canLoadMore: false,
        }),

      // Computed getters
      getFilteredNodes: () => {
        const state = get();
        const { nodes, filters, searchQuery } = state;

        let filtered = nodes;

        // Filter by type
        filtered = filtered.filter((node) => {
          const type = node.data.type;
          if (type === 'entity' && !filters.showEntities) return false;
          if (type === 'relation' && !filters.showRelations) return false;
          if (type === 'chunk' && !filters.showChunks) return false;
          return true;
        });

        // ✅ NOTE: Orphan filtering is handled in Cytoscape (visual hide/show)
        // NOT in the store, so all orphan nodes remain in the data

        // Filter by weight
        if (filters.minWeight > 0) {
          filtered = filtered.filter(
            (node) => (node.data.weight || 0) >= filters.minWeight
          );
        }

        // Filter by source document
        if (filters.sourceDocument) {
          filtered = filtered.filter(
            (node) => node.data.source_id === filters.sourceDocument
          );
        }

        // Filter by search query
        if (searchQuery) {
          const query = searchQuery.toLowerCase();
          filtered = filtered.filter((node) => {
            const label = node.data.label?.toLowerCase() || '';
            const description = node.data.description?.toLowerCase() || '';
            return label.includes(query) || description.includes(query);
          });
        }

        return filtered;
      },

      getFilteredEdges: () => {
        const state = get();
        const filteredNodes = state.getFilteredNodes();
        const nodeIds = new Set(filteredNodes.map((n) => n.data.id));

        // Only include edges where both source and target are in filtered nodes
        return state.edges.filter(
          (edge) =>
            nodeIds.has(edge.data.source) && nodeIds.has(edge.data.target)
        );
      },
    }),
    {
      name: 'graph-store',
    }
  )
);

export default useGraphStore;
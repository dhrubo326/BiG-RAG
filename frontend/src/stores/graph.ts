import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import type {
  CytoscapeNode,
  CytoscapeEdge,
  GraphLayout,
  GraphFilters,
  GraphStats,
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

  // UI State
  layout: GraphLayout;
  filters: GraphFilters;
  isLoading: boolean;
  error: string | null;
  searchQuery: string;

  // Actions
  setNodes: (nodes: CytoscapeNode[]) => void;
  setEdges: (edges: CytoscapeEdge[]) => void;
  selectNode: (node: CytoscapeNode | null) => void;
  hoverNode: (node: CytoscapeNode | null) => void;
  setLayout: (layout: GraphLayout) => void;
  updateFilters: (filters: Partial<GraphFilters>) => void;
  setSearchQuery: (query: string) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setStats: (stats: GraphStats | null) => void;
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
      layout: DEFAULT_CONFIG.GRAPH_LAYOUT,
      filters: {
        showEntities: true,
        showRelations: true,
        showChunks: true,
        minWeight: DEFAULT_CONFIG.GRAPH_MIN_WEIGHT,
        sourceDocument: null,
      },
      isLoading: false,
      error: null,
      searchQuery: '',

      // Actions
      setNodes: (nodes) => set({ nodes }),
      setEdges: (edges) => set({ edges }),

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

      clearGraph: () =>
        set({
          nodes: [],
          edges: [],
          selectedNode: null,
          hoveredNode: null,
          stats: null,
          error: null,
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
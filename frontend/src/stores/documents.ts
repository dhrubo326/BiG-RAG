import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import type { Document } from '../types';

interface DocumentsState {
  // Data
  documents: Document[];
  selectedDocument: Document | null;
  selectedDocumentIds: Set<string>;

  // UI State
  isLoading: boolean;
  isUploading: boolean;
  error: string | null;
  searchQuery: string;
  filters: {
    type: string[];
    source: string[];
    tags: string[];
  };
  sortBy: 'date' | 'title' | 'entities' | 'chunks';
  sortOrder: 'asc' | 'desc';

  // Pagination
  currentPage: number;
  pageSize: number;

  // Actions
  setDocuments: (documents: Document[]) => void;
  addDocument: (document: Document) => void;
  updateDocument: (id: string, updates: Partial<Document>) => void;
  removeDocument: (id: string) => void;
  selectDocument: (document: Document | null) => void;
  toggleDocumentSelection: (id: string) => void;
  selectAllDocuments: () => void;
  clearSelection: () => void;
  setSearchQuery: (query: string) => void;
  updateFilters: (filters: Partial<DocumentsState['filters']>) => void;
  setSortBy: (sortBy: DocumentsState['sortBy']) => void;
  setSortOrder: (order: DocumentsState['sortOrder']) => void;
  setCurrentPage: (page: number) => void;
  setPageSize: (size: number) => void;
  setLoading: (loading: boolean) => void;
  setUploading: (uploading: boolean) => void;
  setError: (error: string | null) => void;

  // Computed
  getFilteredDocuments: () => Document[];
  getSortedDocuments: () => Document[];
  getPaginatedDocuments: () => Document[];
  getTotalPages: () => number;
}

const useDocumentsStore = create<DocumentsState>()(
  devtools(
    (set, get) => ({
      // Initial state
      documents: [],
      selectedDocument: null,
      selectedDocumentIds: new Set(),
      isLoading: false,
      isUploading: false,
      error: null,
      searchQuery: '',
      filters: {
        type: [],
        source: [],
        tags: [],
      },
      sortBy: 'date',
      sortOrder: 'desc',
      currentPage: 1,
      pageSize: 20,

      // Actions
      setDocuments: (documents) =>
        set({
          documents,
          selectedDocument: null,
          selectedDocumentIds: new Set(),
        }),

      addDocument: (document) =>
        set((state) => ({
          documents: [document, ...state.documents],
        })),

      updateDocument: (id, updates) =>
        set((state) => ({
          documents: state.documents.map((doc) =>
            doc.id === id ? { ...doc, ...updates } : doc
          ),
          selectedDocument:
            state.selectedDocument?.id === id
              ? { ...state.selectedDocument, ...updates }
              : state.selectedDocument,
        })),

      removeDocument: (id) =>
        set((state) => ({
          documents: state.documents.filter((doc) => doc.id !== id),
          selectedDocument:
            state.selectedDocument?.id === id ? null : state.selectedDocument,
          selectedDocumentIds: new Set(
            Array.from(state.selectedDocumentIds).filter((docId) => docId !== id)
          ),
        })),

      selectDocument: (document) => set({ selectedDocument: document }),

      toggleDocumentSelection: (id) =>
        set((state) => {
          const newSet = new Set(state.selectedDocumentIds);
          if (newSet.has(id)) {
            newSet.delete(id);
          } else {
            newSet.add(id);
          }
          return { selectedDocumentIds: newSet };
        }),

      selectAllDocuments: () =>
        set((state) => ({
          selectedDocumentIds: new Set(state.documents.map((doc) => doc.id)),
        })),

      clearSelection: () =>
        set({ selectedDocumentIds: new Set(), selectedDocument: null }),

      setSearchQuery: (query) => set({ searchQuery: query, currentPage: 1 }),

      updateFilters: (filters) =>
        set((state) => ({
          filters: { ...state.filters, ...filters },
          currentPage: 1,
        })),

      setSortBy: (sortBy) => set({ sortBy }),

      setSortOrder: (order) => set({ sortOrder: order }),

      setCurrentPage: (page) => set({ currentPage: page }),

      setPageSize: (size) => set({ pageSize: size, currentPage: 1 }),

      setLoading: (loading) => set({ isLoading: loading }),

      setUploading: (uploading) => set({ isUploading: uploading }),

      setError: (error) => set({ error }),

      // Computed getters
      getFilteredDocuments: () => {
        const state = get();
        const { documents, searchQuery, filters } = state;

        let filtered = documents;

        // Filter by search query
        if (searchQuery) {
          const query = searchQuery.toLowerCase();
          filtered = filtered.filter((doc) => {
            const title = doc.title?.toLowerCase() || '';
            const content = doc.content?.toLowerCase() || '';
            const tags = doc.metadata?.tags?.join(' ').toLowerCase() || '';
            return (
              title.includes(query) ||
              content.includes(query) ||
              tags.includes(query)
            );
          });
        }

        // Filter by type (file extension)
        if (filters.type.length > 0) {
          // Implementation depends on how document types are stored
        }

        // Filter by source
        if (filters.source.length > 0) {
          filtered = filtered.filter(
            (doc) =>
              doc.metadata?.source &&
              filters.source.includes(doc.metadata.source)
          );
        }

        // Filter by tags
        if (filters.tags.length > 0) {
          filtered = filtered.filter((doc) =>
            doc.metadata?.tags?.some((tag) => filters.tags.includes(tag))
          );
        }

        return filtered;
      },

      getSortedDocuments: () => {
        const state = get();
        const filtered = state.getFilteredDocuments();
        const { sortBy, sortOrder } = state;

        const sorted = [...filtered].sort((a, b) => {
          let comparison = 0;

          switch (sortBy) {
            case 'title':
              comparison = (a.title || '').localeCompare(b.title || '');
              break;
            case 'date':
              comparison =
                new Date(a.created_at || 0).getTime() -
                new Date(b.created_at || 0).getTime();
              break;
            case 'entities':
              comparison = (a.entities?.length || 0) - (b.entities?.length || 0);
              break;
            case 'chunks':
              comparison = (a.chunks?.length || 0) - (b.chunks?.length || 0);
              break;
          }

          return sortOrder === 'asc' ? comparison : -comparison;
        });

        return sorted;
      },

      getPaginatedDocuments: () => {
        const state = get();
        const sorted = state.getSortedDocuments();
        const { currentPage, pageSize } = state;

        const startIndex = (currentPage - 1) * pageSize;
        const endIndex = startIndex + pageSize;

        return sorted.slice(startIndex, endIndex);
      },

      getTotalPages: () => {
        const state = get();
        const filtered = state.getFilteredDocuments();
        return Math.ceil(filtered.length / state.pageSize);
      },
    }),
    {
      name: 'documents-store',
    }
  )
);

export default useDocumentsStore;
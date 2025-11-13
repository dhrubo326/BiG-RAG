import { useCallback, useEffect, useState } from 'react';
import useDocumentsStore from '../stores/documents';
import {
  getDocuments,
  getDocumentById,
  uploadDocument,
  deleteDocument,
  deleteMultipleDocuments,
} from '../services/documents';
import type { Document, DocumentMetadata } from '../types';
import { toast } from 'sonner';
import { UI_CONFIG } from '../utils/constants';

export const useDocuments = () => {
  const {
    documents,
    selectedDocument,
    selectedDocumentIds,
    isLoading,
    isUploading,
    error,
    searchQuery,
    filters,
    sortBy,
    sortOrder,
    currentPage,
    pageSize,
    setDocuments,
    addDocument,
    updateDocument,
    removeDocument,
    selectDocument,
    toggleDocumentSelection,
    selectAllDocuments,
    clearSelection,
    setSearchQuery,
    updateFilters,
    setSortBy,
    setSortOrder,
    setCurrentPage,
    setPageSize,
    setLoading,
    setUploading,
    setError,
    getFilteredDocuments,
    getSortedDocuments,
    getPaginatedDocuments,
    getTotalPages,
  } = useDocumentsStore();

  const [uploadProgress, setUploadProgress] = useState<number>(0);

  // Load all documents
  const loadDocuments = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const data = await getDocuments();
      setDocuments(data);
      toast.success(`Loaded ${data.length} documents`);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load documents';
      setError(message);
      toast.error(message);
    } finally {
      setLoading(false);
    }
  }, [setDocuments, setLoading, setError]);

  // Load a single document by ID
  const loadDocumentById = useCallback(
    async (id: string) => {
      setLoading(true);
      setError(null);

      try {
        const document = await getDocumentById(id);
        updateDocument(id, document);
        selectDocument(document);
        return document;
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to load document';
        setError(message);
        toast.error(message);
        return null;
      } finally {
        setLoading(false);
      }
    },
    [updateDocument, selectDocument, setLoading, setError]
  );

  // Upload a new document
  const handleUploadDocument = useCallback(
    async (file: File, metadata?: DocumentMetadata) => {
      // Validate file
      if (file.size > UI_CONFIG.MAX_FILE_SIZE) {
        toast.error(`File size exceeds ${UI_CONFIG.MAX_FILE_SIZE / (1024 * 1024)}MB limit`);
        return null;
      }

      const fileExtension = `.${file.name.split('.').pop()?.toLowerCase()}`;
      if (!UI_CONFIG.ALLOWED_FILE_TYPES.includes(fileExtension as any)) {
        toast.error(
          `File type not allowed. Allowed types: ${UI_CONFIG.ALLOWED_FILE_TYPES.join(', ')}`
        );
        return null;
      }

      setUploading(true);
      setUploadProgress(0);
      setError(null);

      try {
        // Upload the document
        const newDocument = await uploadDocument(file, metadata || {}, (progress) => {
          setUploadProgress(progress);
        });

        // Add to store
        addDocument(newDocument);
        toast.success(`Document "${newDocument.title}" uploaded successfully`);
        return newDocument;
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to upload document';
        setError(message);
        toast.error(message);
        return null;
      } finally {
        setUploading(false);
        setUploadProgress(0);
      }
    },
    [addDocument, setUploading, setError]
  );

  // Delete a single document
  const handleDeleteDocument = useCallback(
    async (id: string, hardDelete: boolean = false) => {
      const document = documents.find((doc) => doc.id === id);
      if (!document) {
        toast.error('Document not found');
        return false;
      }

      // Show confirmation with appropriate message
      const deleteType = hardDelete ? 'permanently delete' : 'soft delete';
      const warningMessage = hardDelete
        ? `\n\nWARNING: This will permanently remove:\n- All knowledge graph data (chunks, entities, relations)\n- Document from corpus (prevents resurrection on rebuild)\n- All vector embeddings\n\nThis action cannot be undone!`
        : '\n\nThe document will be marked as deleted but data will be preserved.';

      const confirmed = window.confirm(
        `Are you sure you want to ${deleteType} "${document.title}"?${warningMessage}`
      );
      if (!confirmed) return false;

      setLoading(true);
      setError(null);

      try {
        await deleteDocument(id, hardDelete);
        removeDocument(id);
        toast.success(
          `Document "${document.title}" ${hardDelete ? 'permanently deleted' : 'soft deleted'}`
        );
        return true;
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to delete document';
        setError(message);
        toast.error(message);
        return false;
      } finally {
        setLoading(false);
      }
    },
    [documents, removeDocument, setLoading, setError]
  );

  // Delete multiple documents
  const handleDeleteMultiple = useCallback(
    async (hardDelete: boolean = false) => {
      const ids = Array.from(selectedDocumentIds);
      if (ids.length === 0) {
        toast.warning('No documents selected');
        return false;
      }

      const deleteType = hardDelete ? 'permanently delete' : 'soft delete';
      const warningMessage = hardDelete
        ? `\n\nWARNING: This will permanently remove from ALL ${ids.length} document(s):\n- All knowledge graph data\n- Documents from corpus\n- All vector embeddings\n\nThis action cannot be undone!`
        : '\n\nDocuments will be marked as deleted but data will be preserved.';

      const confirmed = window.confirm(
        `Are you sure you want to ${deleteType} ${ids.length} document(s)?${warningMessage}`
      );
      if (!confirmed) return false;

      setLoading(true);
      setError(null);

      try {
        // Delete all documents with the same hardDelete setting
        await Promise.all(ids.map((id) => deleteDocument(id, hardDelete)));
        ids.forEach((id) => removeDocument(id));
        clearSelection();
        toast.success(
          `${hardDelete ? 'Permanently deleted' : 'Soft deleted'} ${ids.length} document(s)`
        );
        return true;
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to delete documents';
        setError(message);
        toast.error(message);
        return false;
      } finally {
        setLoading(false);
      }
    },
    [selectedDocumentIds, removeDocument, clearSelection, setLoading, setError]
  );

  // Export documents
  const exportDocuments = useCallback(
    (format: 'json' | 'csv' = 'json') => {
      const documentsToExport = selectedDocumentIds.size > 0
        ? documents.filter((doc) => selectedDocumentIds.has(doc.id))
        : documents;

      if (documentsToExport.length === 0) {
        toast.warning('No documents to export');
        return;
      }

      try {
        if (format === 'json') {
          const json = JSON.stringify(documentsToExport, null, 2);
          const blob = new Blob([json], { type: 'application/json' });
          const url = URL.createObjectURL(blob);

          const link = document.createElement('a');
          link.href = url;
          link.download = `documents-export-${Date.now()}.json`;
          link.click();

          URL.revokeObjectURL(url);
        } else if (format === 'csv') {
          // Convert to CSV
          const headers = ['ID', 'Title', 'Entities', 'Chunks', 'Created', 'Tags'];
          const rows = documentsToExport.map((doc) => [
            doc.id,
            doc.title,
            doc.entities?.length || 0,
            doc.chunks?.length || 0,
            doc.created_at || '',
            doc.metadata?.tags?.join(', ') || '',
          ]);

          const csv = [
            headers.join(','),
            ...rows.map((row) => row.map((cell) => `"${cell}"`).join(',')),
          ].join('\n');

          const blob = new Blob([csv], { type: 'text/csv' });
          const url = URL.createObjectURL(blob);

          const link = document.createElement('a');
          link.href = url;
          link.download = `documents-export-${Date.now()}.csv`;
          link.click();

          URL.revokeObjectURL(url);
        }

        toast.success(`Exported ${documentsToExport.length} document(s)`);
      } catch (err) {
        toast.error('Failed to export documents');
      }
    },
    [documents, selectedDocumentIds]
  );

  // Import documents
  const importDocuments = useCallback(
    async (file: File) => {
      const reader = new FileReader();

      reader.onload = async (e) => {
        try {
          const data = JSON.parse(e.target?.result as string);

          if (!Array.isArray(data)) {
            throw new Error('Invalid format: expected array of documents');
          }

          setLoading(true);

          // Upload each document
          for (const doc of data) {
            if (doc.content) {
              // Create a file from content
              const blob = new Blob([doc.content], { type: 'text/plain' });
              const file = new File([blob], doc.title || 'imported.txt', {
                type: 'text/plain',
              });

              await handleUploadDocument(file, doc.metadata);
            }
          }

          toast.success(`Imported ${data.length} document(s)`);
          await loadDocuments();
        } catch (err) {
          toast.error('Failed to import documents');
        } finally {
          setLoading(false);
        }
      };

      reader.readAsText(file);
    },
    [handleUploadDocument, loadDocuments, setLoading]
  );

  // Get unique tags from all documents
  const getAllTags = useCallback(() => {
    const tagsSet = new Set<string>();
    documents.forEach((doc) => {
      doc.metadata?.tags?.forEach((tag) => tagsSet.add(tag));
    });
    return Array.from(tagsSet).sort();
  }, [documents]);

  // Get unique sources from all documents
  const getAllSources = useCallback(() => {
    const sourcesSet = new Set<string>();
    documents.forEach((doc) => {
      if (doc.metadata?.source) {
        sourcesSet.add(doc.metadata.source);
      }
    });
    return Array.from(sourcesSet).sort();
  }, [documents]);

  // Auto-load documents on mount
  useEffect(() => {
    if (documents.length === 0 && !isLoading && !error) {
      loadDocuments();
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  return {
    // State
    documents: getPaginatedDocuments(),
    allDocuments: documents,
    selectedDocument,
    selectedDocumentIds,
    isLoading,
    isUploading,
    uploadProgress,
    error,
    searchQuery,
    filters,
    sortBy,
    sortOrder,
    currentPage,
    pageSize,
    totalPages: getTotalPages(),

    // Actions
    loadDocuments,
    loadDocumentById,
    uploadDocument: handleUploadDocument,
    deleteDocument: handleDeleteDocument,
    deleteMultiple: handleDeleteMultiple,
    selectDocument,
    toggleDocumentSelection,
    selectAllDocuments,
    clearSelection,
    setSearchQuery,
    updateFilters,
    setSortBy,
    setSortOrder,
    setCurrentPage,
    setPageSize,
    exportDocuments,
    importDocuments,

    // Computed
    getAllTags,
    getAllSources,
    filteredCount: getFilteredDocuments().length,
    totalCount: documents.length,
  };
};
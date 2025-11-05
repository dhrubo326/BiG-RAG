/**
 * Documents Page
 *
 * Document management interface:
 * - View all indexed documents
 * - Upload new documents
 * - Delete documents (with cascade cleanup)
 * - View document details and extracted entities
 * - Search and filter documents
 */

import { useState, useEffect } from 'react';
import { Plus, Search, Filter, Download, Upload, Trash2, RefreshCw, ChevronLeft, ChevronRight } from 'lucide-react';
import { useDocuments } from '../hooks/useDocuments';
import DocumentList from '../components/documents/DocumentList';
import DocumentCard from '../components/documents/DocumentCard';
import UploadDialog from '../components/documents/UploadDialog';
import { toast } from 'sonner';

export function Documents() {
  const {
    documents,
    // allDocuments,
    selectedDocument,
    selectedDocumentIds,
    isLoading,
    // isUploading,
    error,
    searchQuery,
    filters,
    sortBy,
    sortOrder,
    currentPage,
    pageSize,
    totalPages,
    loadDocuments,
    uploadDocument,
    deleteDocument,
    deleteMultiple,
    selectDocument,
    toggleDocumentSelection,
    selectAllDocuments,
    clearSelection,
    setSearchQuery,
    updateFilters,
    setSortBy,
    setSortOrder,
    setCurrentPage,
    exportDocuments,
    importDocuments,
    getAllTags,
    getAllSources,
    filteredCount,
    totalCount,
  } = useDocuments();

  const [showUploadDialog, setShowUploadDialog] = useState(false);
  const [showFilters, setShowFilters] = useState(false);
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  const [selectedSources, setSelectedSources] = useState<string[]>([]);

  const allTags = getAllTags();
  const allSources = getAllSources();

  // Apply filters when tags/sources change
  useEffect(() => {
    updateFilters({
      tags: selectedTags,
      source: selectedSources,
    });
  }, [selectedTags, selectedSources]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleUpload = async (file: File, metadata: any) => {
    const result = await uploadDocument(file, metadata);
    if (result) {
      setShowUploadDialog(false);
      toast.success('Document uploaded successfully');
    }
  };

  const handleImportDocuments = () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.onchange = (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (file) {
        importDocuments(file);
      }
    };
    input.click();
  };

  const handleDeleteSelected = async () => {
    if (selectedDocumentIds.size === 0) {
      toast.warning('No documents selected');
      return;
    }
    await deleteMultiple();
  };

  const handleSortChange = (field: string) => {
    if (sortBy === field) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(field as any);
      setSortOrder('asc');
    }
  };

  const handleViewDocument = (doc: typeof documents[0]) => {
    selectDocument(doc);
  };

  const handleDownloadDocument = (doc: typeof documents[0]) => {
    // Create JSON export of single document
    const json = JSON.stringify(doc, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${doc.title.replace(/[^a-z0-9]/gi, '_')}.json`;
    link.click();
    URL.revokeObjectURL(url);
    toast.success('Document downloaded');
  };

  const handleViewInGraph = (doc: typeof documents[0]) => {
    // Navigate to graph page with document filter
    window.location.href = `/graph?doc=${doc.id}`;
  };

  return (
    <div className="container mx-auto p-6 max-w-7xl">
      {/* Header */}
      <div className="flex justify-between items-center mb-6">
        <div>
          <h1 className="text-3xl font-bold">Documents</h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
            {filteredCount} of {totalCount} documents
            {selectedDocumentIds.size > 0 && ` • ${selectedDocumentIds.size} selected`}
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => loadDocuments()}
            className="p-2 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
            title="Refresh"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
          <button
            onClick={() => setShowUploadDialog(true)}
            className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors flex items-center gap-2"
          >
            <Plus className="w-4 h-4" />
            Upload
          </button>
        </div>
      </div>

      {/* Search and Filters Bar */}
      <div className="mb-6 flex gap-3">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
          <input
            type="text"
            placeholder="Search documents..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-10 pr-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
          />
        </div>

        <button
          onClick={() => setShowFilters(!showFilters)}
          className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors flex items-center gap-2"
        >
          <Filter className="w-4 h-4" />
          Filters
          {(selectedTags.length > 0 || selectedSources.length > 0) && (
            <span className="px-2 py-0.5 bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300 text-xs rounded-full">
              {selectedTags.length + selectedSources.length}
            </span>
          )}
        </button>

        {selectedDocumentIds.size > 0 && (
          <>
            <button
              onClick={() => exportDocuments('json')}
              className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors flex items-center gap-2"
            >
              <Download className="w-4 h-4" />
              Export
            </button>
            <button
              onClick={handleDeleteSelected}
              className="px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 transition-colors flex items-center gap-2"
            >
              <Trash2 className="w-4 h-4" />
              Delete
            </button>
          </>
        )}

        <button
          onClick={handleImportDocuments}
          className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors flex items-center gap-2"
        >
          <Upload className="w-4 h-4" />
          Import
        </button>
      </div>

      {/* Filters Panel */}
      {showFilters && (
        <div className="mb-6 p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <h3 className="font-medium mb-2">Tags</h3>
              <div className="space-y-1 max-h-32 overflow-y-auto">
                {allTags.length === 0 ? (
                  <p className="text-sm text-gray-500">No tags available</p>
                ) : (
                  allTags.map((tag) => (
                    <label key={tag} className="flex items-center gap-2 text-sm">
                      <input
                        type="checkbox"
                        checked={selectedTags.includes(tag)}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setSelectedTags([...selectedTags, tag]);
                          } else {
                            setSelectedTags(selectedTags.filter((t) => t !== tag));
                          }
                        }}
                        className="rounded border-gray-300 text-blue-500 focus:ring-blue-500"
                      />
                      <span>{tag}</span>
                    </label>
                  ))
                )}
              </div>
            </div>

            <div>
              <h3 className="font-medium mb-2">Sources</h3>
              <div className="space-y-1 max-h-32 overflow-y-auto">
                {allSources.length === 0 ? (
                  <p className="text-sm text-gray-500">No sources available</p>
                ) : (
                  allSources.map((source) => (
                    <label key={source} className="flex items-center gap-2 text-sm">
                      <input
                        type="checkbox"
                        checked={selectedSources.includes(source)}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setSelectedSources([...selectedSources, source]);
                          } else {
                            setSelectedSources(selectedSources.filter((s) => s !== source));
                          }
                        }}
                        className="rounded border-gray-300 text-blue-500 focus:ring-blue-500"
                      />
                      <span>{source}</span>
                    </label>
                  ))
                )}
              </div>
            </div>
          </div>

          {(selectedTags.length > 0 || selectedSources.length > 0) && (
            <button
              onClick={() => {
                setSelectedTags([]);
                setSelectedSources([]);
              }}
              className="mt-3 text-sm text-blue-500 hover:text-blue-600"
            >
              Clear all filters
            </button>
          )}
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="mb-6 p-3 bg-red-100 dark:bg-red-900/50 text-red-800 dark:text-red-200 rounded-lg">
          {error}
        </div>
      )}

      {/* Documents List */}
      {isLoading ? (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-12 text-center">
          <div className="inline-flex items-center gap-2 text-gray-500">
            <RefreshCw className="w-5 h-5 animate-spin" />
            <span>Loading documents...</span>
          </div>
        </div>
      ) : (
        <DocumentList
          documents={documents}
          selectedIds={selectedDocumentIds}
          sortBy={sortBy}
          sortOrder={sortOrder}
          onSelectDocument={handleViewDocument}
          onToggleSelect={toggleDocumentSelection}
          onSelectAll={selectAllDocuments}
          onSortChange={handleSortChange}
          onDeleteDocument={deleteDocument}
          onViewDocument={handleViewDocument}
          onDownloadDocument={handleDownloadDocument}
        />
      )}

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="mt-6 flex items-center justify-between">
          <p className="text-sm text-gray-500">
            Page {currentPage} of {totalPages}
          </p>
          <div className="flex gap-2">
            <button
              onClick={() => setCurrentPage(currentPage - 1)}
              disabled={currentPage === 1}
              className="p-2 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
              const page = Math.max(1, Math.min(currentPage - 2, totalPages - 4)) + i;
              if (page > totalPages) return null;
              return (
                <button
                  key={page}
                  onClick={() => setCurrentPage(page)}
                  className={`px-3 py-1 rounded-lg transition-colors ${
                    page === currentPage
                      ? 'bg-blue-500 text-white'
                      : 'border border-gray-300 dark:border-gray-600 hover:bg-gray-100 dark:hover:bg-gray-700'
                  }`}
                >
                  {page}
                </button>
              );
            })}
            <button
              onClick={() => setCurrentPage(currentPage + 1)}
              disabled={currentPage === totalPages}
              className="p-2 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}

      {/* Upload Dialog */}
      <UploadDialog
        isOpen={showUploadDialog}
        onClose={() => setShowUploadDialog(false)}
        onUpload={handleUpload}
      />

      {/* Document Preview Card */}
      {selectedDocument && (
        <DocumentCard
          document={selectedDocument}
          onClose={() => selectDocument(null)}
          onDelete={deleteDocument}
          onDownload={handleDownloadDocument}
          onViewInGraph={handleViewInGraph}
        />
      )}
    </div>
  );
}

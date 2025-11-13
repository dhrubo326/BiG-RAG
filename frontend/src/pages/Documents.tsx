/**
 * Documents Page - Performance Optimized with Pagination & Filters
 *
 * Features:
 * - Dataset filtering with dropdown
 * - Pagination/Load More for performance
 * - Pure black title for visibility
 * - Clear document ID background
 * - All buttons always visible
 */

import { useState, useEffect } from 'react';
import {
  Upload,
  Trash2,
  Download,
  RefreshCw,
  Search,
  FileText,
  Calendar,
  Database,
  Tag,
  AlertCircle,
  Copy,
  Eye,
  Layers,
  CheckSquare,
  Square,
  Filter,
  ChevronDown,
} from 'lucide-react';
import { getDocuments, deleteDocument as deleteDocumentAPI, uploadDocument as uploadDocumentAPI } from '../services/documents';
import UploadDialog from '../components/documents/UploadDialog';
import DocumentDetailsModal from '../components/documents/DocumentDetailsModal';
import { toast } from 'sonner';
import type { Document } from '../types';

const ITEMS_PER_PAGE = 50;

export function Documents() {
  const [allDocuments, setAllDocuments] = useState<Document[]>([]);
  const [displayedDocuments, setDisplayedDocuments] = useState<Document[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedDataset, setSelectedDataset] = useState<string>('all');
  const [availableDatasets, setAvailableDatasets] = useState<string[]>([]);
  const [showUploadDialog, setShowUploadDialog] = useState(false);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [currentPage, setCurrentPage] = useState(1);
  const [totalDocuments, setTotalDocuments] = useState(0);
  const [hasMore, setHasMore] = useState(true);

  // Document details modal
  const [detailsModalOpen, setDetailsModalOpen] = useState(false);
  const [selectedDocForDetails, setSelectedDocForDetails] = useState<{ id: string; title: string } | null>(null);

  // Load documents with pagination
  const loadDocuments = async (page: number = 1, append: boolean = false) => {
    setIsLoading(true);
    setError(null);

    try {
      const result = await getDocuments({
        page,
        limit: ITEMS_PER_PAGE,
        dataset: selectedDataset === 'all' ? undefined : selectedDataset,
      });

      const newDocs = result.documents || [];
      setTotalDocuments(result.total);

      if (append) {
        setDisplayedDocuments((prev) => [...prev, ...newDocs]);
      } else {
        setDisplayedDocuments(newDocs);
      }

      setHasMore(newDocs.length === ITEMS_PER_PAGE);

      // Extract unique datasets
      const datasets = new Set<string>();
      newDocs.forEach((doc: any) => {
        if (doc.dataset) datasets.add(doc.dataset);
      });
      if (datasets.size > 0) {
        setAvailableDatasets(['all', ...Array.from(datasets).sort()]);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load documents';
      setError(message);
      toast.error(message);
    } finally {
      setIsLoading(false);
    }
  };

  // Load more documents
  const loadMore = () => {
    const nextPage = currentPage + 1;
    setCurrentPage(nextPage);
    loadDocuments(nextPage, true);
  };

  // Open details modal for a document
  const handleViewDetails = (docId: string, docTitle: string) => {
    setSelectedDocForDetails({ id: docId, title: docTitle });
    setDetailsModalOpen(true);
  };

  // Close details modal
  const handleCloseDetails = () => {
    setDetailsModalOpen(false);
    setSelectedDocForDetails(null);
  };

  // Initial load
  useEffect(() => {
    loadDocuments();
  }, [selectedDataset]);

  // Filter documents based on search
  const filteredDocuments = displayedDocuments.filter(
    (doc: any) =>
      doc.title?.toLowerCase().includes(searchQuery.toLowerCase()) ||
      doc.document_id?.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const handleUpload = async (file: File, metadata: any) => {
    try {
      const result = await uploadDocumentAPI(file, metadata);
      if (result) {
        setShowUploadDialog(false);
        toast.success('Document uploaded successfully!');
        setCurrentPage(1);
        await loadDocuments(1, false);
      }
    } catch (err) {
      toast.error('Upload failed');
    }
  };

  const handleDelete = async (id: string, hard: boolean, title: string) => {
    const deleteType = hard ? 'permanently delete' : 'soft delete';
    const warning = hard
      ? '\n\nWARNING: This will permanently remove:\n- All knowledge graph data\n- Document from corpus\n- All vector embeddings\n\nThis cannot be undone!'
      : '\n\nThe document will be marked as deleted but data will be preserved.';

    if (confirm(`Are you sure you want to ${deleteType} "${title}"?${warning}`)) {
      try {
        await deleteDocumentAPI(id, hard);
        setSelectedIds((prev) => {
          const newSet = new Set(prev);
          newSet.delete(id);
          return newSet;
        });
        toast.success('Document deleted');
        setCurrentPage(1);
        await loadDocuments(1, false);
      } catch (err) {
        toast.error('Delete failed');
      }
    }
  };

  const handleDeleteSelected = async (hard: boolean) => {
    if (selectedIds.size === 0) {
      toast.warning('No documents selected');
      return;
    }

    const deleteType = hard ? 'permanently delete' : 'soft delete';
    const warning = hard
      ? `\n\nWARNING: This will permanently remove ALL data from ${selectedIds.size} documents!\nThis cannot be undone!`
      : '\n\nDocuments will be marked as deleted but data will be preserved.';

    if (confirm(`${deleteType.toUpperCase()} ${selectedIds.size} documents?${warning}`)) {
      try {
        for (const id of Array.from(selectedIds)) {
          await deleteDocumentAPI(id, hard);
        }
        setSelectedIds(new Set());
        toast.success(`Deleted ${selectedIds.size} documents`);
        setCurrentPage(1);
        await loadDocuments(1, false);
      } catch (err) {
        toast.error('Delete failed');
      }
    }
  };

  const toggleSelection = (id: string) => {
    setSelectedIds((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(id)) {
        newSet.delete(id);
      } else {
        newSet.add(id);
      }
      return newSet;
    });
  };

  const toggleSelectAll = () => {
    if (selectedIds.size === filteredDocuments.length && filteredDocuments.length > 0) {
      setSelectedIds(new Set());
    } else {
      setSelectedIds(new Set(filteredDocuments.map((doc: any) => doc.document_id)));
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    toast.success('Copied to clipboard!');
  };

  const handleDownloadDoc = (doc: any) => {
    const json = JSON.stringify(doc, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${doc.title?.replace(/[^a-z0-9]/gi, '_') || doc.document_id}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    toast.success('Document downloaded!');
  };

  const exportDocuments = (format: string) => {
    const json = JSON.stringify(filteredDocuments, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `documents-export-${Date.now()}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    toast.success('Documents exported!');
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">Documents</h1>
          <p className="text-gray-600 dark:text-gray-400">
            Showing {filteredDocuments.length} of {totalDocuments} document{totalDocuments !== 1 ? 's' : ''}
            {selectedIds.size > 0 && ` • ${selectedIds.size} selected`}
          </p>
        </div>

        {/* Toolbar */}
        <div className="mb-6 bg-white dark:bg-gray-800 rounded-lg p-4 shadow">
          <div className="flex flex-wrap gap-3 mb-3">
            {/* Search */}
            <div className="flex-1 min-w-[250px] relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
              <input
                type="text"
                placeholder="Search by title or ID..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-10 pr-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
              />
            </div>

            {/* Dataset Filter */}
            <div className="relative">
              <Filter className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 pointer-events-none" />
              <select
                value={selectedDataset}
                onChange={(e) => {
                  setSelectedDataset(e.target.value);
                  setCurrentPage(1);
                }}
                className="pl-10 pr-8 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white appearance-none cursor-pointer min-w-[150px]"
              >
                <option value="all">All Datasets</option>
                {availableDatasets.filter((d) => d !== 'all').map((dataset) => (
                  <option key={dataset} value={dataset}>
                    {dataset}
                  </option>
                ))}
              </select>
              <ChevronDown className="absolute right-2 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 pointer-events-none" />
            </div>

            {/* Actions */}
            <button
              onClick={() => {
                setCurrentPage(1);
                loadDocuments(1, false);
              }}
              disabled={isLoading}
              className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 flex items-center gap-2"
            >
              <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
              Refresh
            </button>

            <button
              onClick={() => setShowUploadDialog(true)}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center gap-2"
            >
              <Upload className="w-4 h-4" />
              Upload
            </button>
          </div>

          {/* Bulk Actions */}
          {selectedIds.size > 0 && (
            <div className="pt-3 border-t border-gray-200 dark:border-gray-700 flex items-center gap-3">
              <button
                onClick={toggleSelectAll}
                className="text-sm text-blue-600 dark:text-blue-400 hover:underline"
              >
                {selectedIds.size === filteredDocuments.length ? 'Deselect All' : 'Select All'}
              </button>

              <div className="flex-1"></div>

              <button
                onClick={() => exportDocuments('json')}
                className="px-3 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded hover:bg-gray-50 dark:hover:bg-gray-700 flex items-center gap-1"
              >
                <Download className="w-3 h-3" />
                Export
              </button>

              <button
                onClick={() => handleDeleteSelected(false)}
                className="px-3 py-1.5 text-sm bg-orange-500 text-white rounded hover:bg-orange-600 flex items-center gap-1"
              >
                <Trash2 className="w-3 h-3" />
                Soft Delete
              </button>

              <button
                onClick={() => handleDeleteSelected(true)}
                className="px-3 py-1.5 text-sm bg-red-600 text-white rounded hover:bg-red-700 flex items-center gap-1"
              >
                <Trash2 className="w-3 h-3" />
                Hard Delete
              </button>
            </div>
          )}
        </div>

        {/* Error */}
        {error && (
          <div className="mb-6 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg flex items-center gap-3">
            <AlertCircle className="w-5 h-5 text-red-600" />
            <span className="text-red-800 dark:text-red-200">{error}</span>
          </div>
        )}

        {/* Loading */}
        {isLoading && displayedDocuments.length === 0 ? (
          <div className="text-center py-20">
            <RefreshCw className="w-8 h-8 text-blue-500 animate-spin mx-auto mb-4" />
            <p className="text-gray-600 dark:text-gray-400">Loading documents...</p>
          </div>
        ) : filteredDocuments.length === 0 ? (
          <div className="text-center py-20 bg-white dark:bg-gray-800 rounded-lg">
            <FileText className="w-16 h-16 text-gray-400 mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
              No documents found
            </h3>
            <p className="text-gray-600 dark:text-gray-400 mb-6">
              {searchQuery ? 'Try a different search term' : 'Upload your first document to get started'}
            </p>
            {!searchQuery && (
              <button
                onClick={() => setShowUploadDialog(true)}
                className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 inline-flex items-center gap-2"
              >
                <Upload className="w-5 h-5" />
                Upload Document
              </button>
            )}
          </div>
        ) : (
          <>
            {/* Documents List */}
            <div className="space-y-3">
              {filteredDocuments.map((doc: any) => (
                <DocumentRow
                  key={doc.document_id}
                  document={doc}
                  isSelected={selectedIds.has(doc.document_id)}
                  onToggleSelect={() => toggleSelection(doc.document_id)}
                  onViewDetails={() => handleViewDetails(doc.document_id, doc.title || doc.filename || 'Untitled')}
                  onDelete={handleDelete}
                  onDownload={handleDownloadDoc}
                  onCopyId={copyToClipboard}
                />
              ))}
            </div>

            {/* Load More Button */}
            {hasMore && !searchQuery && (
              <div className="mt-6 text-center">
                <button
                  onClick={loadMore}
                  disabled={isLoading}
                  className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 inline-flex items-center gap-2"
                >
                  {isLoading ? (
                    <>
                      <RefreshCw className="w-4 h-4 animate-spin" />
                      Loading...
                    </>
                  ) : (
                    <>
                      <Download className="w-4 h-4" />
                      Load More
                    </>
                  )}
                </button>
                <p className="text-sm text-gray-500 dark:text-gray-400 mt-2">
                  Showing {displayedDocuments.length} of {totalDocuments}
                </p>
              </div>
            )}
          </>
        )}
      </div>

      {/* Upload Dialog */}
      <UploadDialog
        isOpen={showUploadDialog}
        onClose={() => setShowUploadDialog(false)}
        onUpload={handleUpload}
      />

      {/* Document Details Modal */}
      {selectedDocForDetails && (
        <DocumentDetailsModal
          isOpen={detailsModalOpen}
          onClose={handleCloseDetails}
          documentId={selectedDocForDetails.id}
          documentTitle={selectedDocForDetails.title}
        />
      )}
    </div>
  );
}

/* Document Row Component */
interface DocumentRowProps {
  document: any;
  isSelected: boolean;
  onToggleSelect: () => void;
  onViewDetails: () => void;
  onDelete: (id: string, hard: boolean, title: string) => void;
  onDownload: (doc: any) => void;
  onCopyId: (id: string) => void;
}

function DocumentRow({
  document,
  isSelected,
  onToggleSelect,
  onViewDetails,
  onDelete,
  onDownload,
  onCopyId,
}: DocumentRowProps) {
  const metadata = document.metadata || {};
  const uploadDate = document.upload_date || document.created_at || document.indexed_date;
  const dataset = document.dataset || metadata.dataset || 'unknown';

  return (
    <div
      className={`bg-white dark:bg-gray-800 rounded-lg border transition-all ${
        isSelected ? 'border-blue-500' : 'border-gray-200 dark:border-gray-700'
      }`}
    >
      {/* Compact Row */}
      <div className="p-3">
        <div className="flex items-center gap-3">
          {/* Checkbox */}
          <div>
            {isSelected ? (
              <CheckSquare
                className="w-4 h-4 text-blue-600 cursor-pointer"
                onClick={onToggleSelect}
              />
            ) : (
              <Square
                className="w-4 h-4 text-gray-400 cursor-pointer hover:text-blue-600"
                onClick={onToggleSelect}
              />
            )}
          </div>

          {/* Content */}
          <div className="flex-1 min-w-0">
            {/* Title & Dataset Badge */}
            <div className="mb-1 flex items-center gap-2 flex-wrap">
              <h3
                className="text-base font-bold"
                style={{ color: '#111827' }}
              >
                {document.title || document.filename || 'Untitled'}
              </h3>
              <span className="px-2 py-0.5 bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 text-xs font-semibold rounded">
                {dataset}
              </span>
            </div>

            {/* Document ID with better visibility */}
            <div className="flex items-center gap-2 mb-1">
              <code
                className="text-xs font-mono border border-yellow-600 px-2 py-0.5 rounded font-medium"
                style={{ backgroundColor: '#fef3c7', color: '#1f2937' }}
              >
                {document.document_id}
              </code>
              <button
                onClick={() => onCopyId(document.document_id)}
                className="p-0.5 hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
                title="Copy ID"
              >
                <Copy className="w-3 h-3 text-gray-600 dark:text-gray-400" />
              </button>

              {/* Meta Info inline */}
              {uploadDate && (
                <>
                  <span className="text-gray-300 dark:text-gray-600">•</span>
                  <div className="flex items-center gap-1">
                    <Calendar className="w-3 h-3 text-blue-500" />
                    <span className="text-xs text-gray-600 dark:text-gray-400">
                      {new Date(uploadDate).toLocaleDateString()}
                    </span>
                  </div>
                </>
              )}

              {document.status && (
                <>
                  <span className="text-gray-300 dark:text-gray-600">•</span>
                  <div className="flex items-center gap-1">
                    <div
                      className={`w-2 h-2 rounded-full ${
                        document.status === 'indexed' || document.status === 'completed'
                          ? 'bg-green-500'
                          : document.status === 'processing'
                          ? 'bg-yellow-500'
                          : 'bg-gray-400'
                      }`}
                    />
                    <span className="text-xs text-gray-600 dark:text-gray-400 capitalize">
                      {document.status}
                    </span>
                  </div>
                </>
              )}
            </div>

            {/* Tags & Category */}
            {(metadata.category || (metadata.tags && metadata.tags.length > 0)) && (
              <div className="flex flex-wrap gap-1.5">
                {metadata.category && (
                  <span className="px-1.5 py-0.5 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 text-xs rounded">
                    {metadata.category}
                  </span>
                )}
                {metadata.tags?.slice(0, 3).map((tag: string) => (
                  <span
                    key={tag}
                    className="px-1.5 py-0.5 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 text-xs rounded flex items-center gap-0.5"
                  >
                    <Tag className="w-2.5 h-2.5" />
                    {tag}
                  </span>
                ))}
                {metadata.tags && metadata.tags.length > 3 && (
                  <span className="text-xs text-gray-500">+{metadata.tags.length - 3}</span>
                )}
              </div>
            )}
          </div>

          {/* Action Buttons - Horizontal */}
          <div className="flex items-center gap-2">
            <button
              onClick={onViewDetails}
              className="px-3 py-1.5 text-xs bg-blue-600 text-white rounded hover:bg-blue-700 flex items-center gap-1"
              title="View detailed information"
            >
              <Eye className="w-3 h-3" />
              Details
            </button>

            <button
              onClick={() => onDownload(document)}
              className="px-3 py-1.5 text-xs border border-gray-300 dark:border-gray-600 rounded hover:bg-gray-50 dark:hover:bg-gray-700 flex items-center gap-1"
              title="Download JSON"
            >
              <Download className="w-3 h-3" />
              JSON
            </button>

            <button
              onClick={() => onDelete(document.document_id, false, document.title)}
              className="px-3 py-1.5 text-xs bg-orange-500 text-white rounded hover:bg-orange-600 flex items-center gap-1"
              title="Soft delete"
            >
              <Trash2 className="w-3 h-3" />
              Soft
            </button>

            <button
              onClick={() => onDelete(document.document_id, true, document.title)}
              className="px-3 py-1.5 text-xs bg-red-600 text-white rounded hover:bg-red-700 flex items-center gap-1"
              title="Hard delete (permanent)"
            >
              <Trash2 className="w-3 h-3" />
              Hard
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

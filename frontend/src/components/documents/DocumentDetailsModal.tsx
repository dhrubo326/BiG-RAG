/**
 * Document Details Modal
 * Shows detailed information fetched from /documents/{id}?include_entities=true&include_related=true
 */

import { useEffect, useState } from 'react';
import {
  X,
  RefreshCw,
  AlertCircle,
  Database,
  Layers,
  FileText,
  Calendar,
  Tag,
  Copy,
} from 'lucide-react';
import { getDocumentById } from '../../services/documents';
import { toast } from 'sonner';

interface DocumentDetailsModalProps {
  isOpen: boolean;
  onClose: () => void;
  documentId: string;
  documentTitle: string;
}

export default function DocumentDetailsModal({
  isOpen,
  onClose,
  documentId,
  documentTitle,
}: DocumentDetailsModalProps) {
  const [details, setDetails] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen && documentId) {
      loadDetails();
    }
  }, [isOpen, documentId]);

  const loadDetails = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const data = await getDocumentById(documentId, true, true);
      setDetails(data);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load document details';
      setError(message);
      toast.error(message);
    } finally {
      setIsLoading(false);
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    toast.success('Copied to clipboard!');
  };

  if (!isOpen) return null;

  const stats = details?.stats || {};
  const metadata = details?.metadata || {};

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm">
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-2xl max-w-4xl w-full max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-start justify-between p-6 border-b border-gray-200 dark:border-gray-700">
          <div className="flex-1 min-w-0 mr-4">
            <h2
              className="text-2xl font-bold mb-2"
              style={{ color: '#111827' }}
            >
              {documentTitle}
            </h2>
            <div className="flex items-center gap-2">
              <code
                className="text-xs font-mono border border-yellow-600 px-2.5 py-1 rounded font-semibold"
                style={{ backgroundColor: '#fef3c7', color: '#1f2937' }}
              >
                {documentId}
              </code>
              <button
                onClick={() => copyToClipboard(documentId)}
                className="p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
                title="Copy ID"
              >
                <Copy className="w-3 h-3 text-gray-700 dark:text-gray-300" />
              </button>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
          {isLoading ? (
            <div className="text-center py-12">
              <RefreshCw className="w-8 h-8 text-blue-500 animate-spin mx-auto mb-4" />
              <p className="text-gray-600 dark:text-gray-400">Loading document details...</p>
            </div>
          ) : error ? (
            <div className="text-center py-12">
              <AlertCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
              <p className="text-red-600 dark:text-red-400">{error}</p>
              <button
                onClick={loadDetails}
                className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
              >
                Retry
              </button>
            </div>
          ) : details ? (
            <div className="space-y-6">
              {/* Stats Summary */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg border border-purple-200 dark:border-purple-800">
                  <div className="flex items-center gap-2 mb-2">
                    <Database className="w-5 h-5 text-purple-600 dark:text-purple-400" />
                    <h3 className="font-semibold" style={{ color: '#111827' }}>Entities</h3>
                  </div>
                  <p className="text-3xl font-bold text-purple-600 dark:text-purple-400">
                    {stats.entities || 0}
                  </p>
                </div>

                <div className="bg-orange-50 dark:bg-orange-900/20 p-4 rounded-lg border border-orange-200 dark:border-orange-800">
                  <div className="flex items-center gap-2 mb-2">
                    <Layers className="w-5 h-5 text-orange-600 dark:text-orange-400" />
                    <h3 className="font-semibold" style={{ color: '#111827' }}>Relations</h3>
                  </div>
                  <p className="text-3xl font-bold text-orange-600 dark:text-orange-400">
                    {stats.relations || stats.edges || 0}
                  </p>
                </div>

                <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border border-green-200 dark:border-green-800">
                  <div className="flex items-center gap-2 mb-2">
                    <FileText className="w-5 h-5 text-green-600 dark:text-green-400" />
                    <h3 className="font-semibold" style={{ color: '#111827' }}>Chunks</h3>
                  </div>
                  <p className="text-3xl font-bold text-green-600 dark:text-green-400">
                    {stats.chunks || 0}
                  </p>
                </div>
              </div>

              {/* Metadata */}
              {(metadata.category || metadata.tags || details.upload_date || details.created_at) && (
                <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-lg">
                  <h3 className="font-semibold mb-3" style={{ color: '#111827' }}>Metadata</h3>
                  <div className="space-y-2">
                    {(details.upload_date || details.created_at) && (
                      <div className="flex items-center gap-2">
                        <Calendar className="w-4 h-4 text-gray-500" />
                        <span className="text-sm text-gray-600 dark:text-gray-400">
                          {new Date(details.upload_date || details.created_at).toLocaleString()}
                        </span>
                      </div>
                    )}
                    {metadata.category && (
                      <div className="flex items-center gap-2">
                        <Tag className="w-4 h-4 text-gray-500" />
                        <span className="text-sm px-2 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 rounded">
                          {metadata.category}
                        </span>
                      </div>
                    )}
                    {metadata.tags && metadata.tags.length > 0 && (
                      <div className="flex items-start gap-2">
                        <Tag className="w-4 h-4 text-gray-500 mt-1" />
                        <div className="flex flex-wrap gap-2">
                          {metadata.tags.map((tag: string) => (
                            <span
                              key={tag}
                              className="text-xs px-2 py-1 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded"
                            >
                              {tag}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Content Preview */}
              {details.content_preview && (
                <div>
                  <h3 className="font-semibold mb-3" style={{ color: '#111827' }}>
                    Content Preview
                  </h3>
                  <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-lg border border-gray-200 dark:border-gray-700">
                    <p className="text-sm text-gray-700 dark:text-gray-300 whitespace-pre-wrap">
                      {details.content_preview}
                    </p>
                  </div>
                </div>
              )}

              {/* Top Entities */}
              {details.top_entities && details.top_entities.length > 0 && (
                <div>
                  <h3 className="font-semibold mb-3" style={{ color: '#111827' }}>
                    Top Entities ({details.top_entities.length})
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                    {details.top_entities.map((entity: any, idx: number) => (
                      <div
                        key={idx}
                        className="flex items-center justify-between bg-gray-50 dark:bg-gray-900 p-3 rounded-lg border border-gray-200 dark:border-gray-700"
                      >
                        <div className="flex-1 min-w-0">
                          <p className="font-mono text-sm truncate" style={{ color: '#111827' }}>
                            {entity.name}
                          </p>
                          {entity.type && (
                            <p className="text-xs text-gray-500 dark:text-gray-400">{entity.type}</p>
                          )}
                        </div>
                        <span className="ml-2 text-xs font-semibold text-purple-600 dark:text-purple-400">
                          {entity.weight}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Related Documents */}
              {details.related_documents && details.related_documents.length > 0 && (
                <div>
                  <h3 className="font-semibold mb-3" style={{ color: '#111827' }}>
                    Related Documents ({details.related_documents.length})
                  </h3>
                  <div className="space-y-2">
                    {details.related_documents.map((related: any) => (
                      <div
                        key={related.id}
                        className="flex items-center justify-between bg-gray-50 dark:bg-gray-900 p-3 rounded-lg border border-gray-200 dark:border-gray-700"
                      >
                        <span className="text-sm" style={{ color: '#111827' }}>
                          {related.title}
                        </span>
                        <span className="ml-2 text-xs font-semibold text-blue-600 dark:text-blue-400">
                          {(related.similarity * 100).toFixed(0)}% similar
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : null}
        </div>

        {/* Footer */}
        <div className="flex justify-end gap-3 p-6 border-t border-gray-200 dark:border-gray-700">
          <button
            onClick={onClose}
            className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}

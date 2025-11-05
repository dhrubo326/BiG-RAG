import { useEffect, useRef } from 'react';
import { FileText, MoreVertical, Eye, Trash2, Download, ChevronUp, ChevronDown } from 'lucide-react';
import type { Document } from '../../types';
import { formatRelativeTime, formatLargeNumber } from '../../utils/formatters';

interface DocumentListProps {
  documents: Document[];
  selectedIds: Set<string>;
  sortBy: string;
  sortOrder: 'asc' | 'desc';
  onSelectDocument: (doc: Document) => void;
  onToggleSelect: (id: string) => void;
  onSelectAll: () => void;
  onSortChange: (field: string) => void;
  onDeleteDocument: (id: string) => void;
  onViewDocument: (doc: Document) => void;
  onDownloadDocument: (doc: Document) => void;
}

const DocumentList: React.FC<DocumentListProps> = ({
  documents,
  selectedIds,
  sortBy,
  sortOrder,
  onSelectDocument,
  onToggleSelect,
  onSelectAll,
  onSortChange,
  onDeleteDocument,
  onViewDocument,
  onDownloadDocument,
}) => {
  const allSelected = documents.length > 0 && documents.every(doc => selectedIds.has(doc.id));
  const someSelected = documents.some(doc => selectedIds.has(doc.id));
  const checkboxRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (checkboxRef.current) {
      checkboxRef.current.indeterminate = !allSelected && someSelected;
    }
  }, [allSelected, someSelected]);

  const getSortIcon = (field: string) => {
    if (sortBy !== field) return null;
    return sortOrder === 'asc' ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />;
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow overflow-hidden">
      {documents.length === 0 ? (
        <div className="text-center py-12">
          <FileText className="w-12 h-12 mx-auto text-gray-400 mb-3" />
          <p className="text-gray-500 dark:text-gray-400">No documents found</p>
          <p className="text-sm text-gray-400 dark:text-gray-500 mt-1">
            Upload your first document to get started
          </p>
        </div>
      ) : (
        <table className="w-full">
          <thead>
            <tr className="border-b border-gray-200 dark:border-gray-700">
              <th className="p-4 text-left">
                <input
                  ref={checkboxRef}
                  type="checkbox"
                  checked={allSelected}
                  onChange={onSelectAll}
                  className="rounded border-gray-300 text-blue-500 focus:ring-blue-500"
                />
              </th>
              <th className="p-4 text-left">
                <button
                  onClick={() => onSortChange('title')}
                  className="flex items-center gap-1 text-sm font-medium text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-gray-100"
                >
                  Title {getSortIcon('title')}
                </button>
              </th>
              <th className="p-4 text-left">
                <button
                  onClick={() => onSortChange('entities')}
                  className="flex items-center gap-1 text-sm font-medium text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-gray-100"
                >
                  Entities {getSortIcon('entities')}
                </button>
              </th>
              <th className="p-4 text-left">
                <button
                  onClick={() => onSortChange('chunks')}
                  className="flex items-center gap-1 text-sm font-medium text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-gray-100"
                >
                  Chunks {getSortIcon('chunks')}
                </button>
              </th>
              <th className="p-4 text-left">
                <button
                  onClick={() => onSortChange('date')}
                  className="flex items-center gap-1 text-sm font-medium text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-gray-100"
                >
                  Created {getSortIcon('date')}
                </button>
              </th>
              <th className="p-4 text-left">
                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  Actions
                </span>
              </th>
            </tr>
          </thead>
          <tbody>
            {documents.map((doc) => (
              <tr
                key={doc.id}
                className="border-b border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors"
              >
                <td className="p-4">
                  <input
                    type="checkbox"
                    checked={selectedIds.has(doc.id)}
                    onChange={() => onToggleSelect(doc.id)}
                    className="rounded border-gray-300 text-blue-500 focus:ring-blue-500"
                  />
                </td>
                <td className="p-4">
                  <button
                    onClick={() => onSelectDocument(doc)}
                    className="text-left hover:text-blue-500 transition-colors"
                  >
                    <div>
                      <p className="font-medium text-gray-900 dark:text-gray-100">
                        {doc.title}
                      </p>
                      {doc.metadata?.tags && doc.metadata.tags.length > 0 && (
                        <div className="flex flex-wrap gap-1 mt-1">
                          {doc.metadata.tags.slice(0, 3).map((tag) => (
                            <span
                              key={tag}
                              className="px-2 py-0.5 bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400 text-xs rounded"
                            >
                              {tag}
                            </span>
                          ))}
                          {doc.metadata.tags.length > 3 && (
                            <span className="text-xs text-gray-500">
                              +{doc.metadata.tags.length - 3} more
                            </span>
                          )}
                        </div>
                      )}
                    </div>
                  </button>
                </td>
                <td className="p-4 text-sm text-gray-600 dark:text-gray-400">
                  {formatLargeNumber(doc.entities?.length || 0)}
                </td>
                <td className="p-4 text-sm text-gray-600 dark:text-gray-400">
                  {formatLargeNumber(doc.chunks?.length || 0)}
                </td>
                <td className="p-4 text-sm text-gray-600 dark:text-gray-400">
                  {doc.created_at ? formatRelativeTime(doc.created_at) : '-'}
                </td>
                <td className="p-4">
                  <div className="relative group">
                    <button className="p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors">
                      <MoreVertical className="w-4 h-4" />
                    </button>

                    <div className="absolute right-0 top-full mt-1 w-40 bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 z-10">
                      <button
                        onClick={() => onViewDocument(doc)}
                        className="w-full px-4 py-2 text-left text-sm hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors flex items-center gap-2"
                      >
                        <Eye className="w-3 h-3" />
                        View
                      </button>
                      <button
                        onClick={() => onDownloadDocument(doc)}
                        className="w-full px-4 py-2 text-left text-sm hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors flex items-center gap-2"
                      >
                        <Download className="w-3 h-3" />
                        Download
                      </button>
                      <button
                        onClick={() => onDeleteDocument(doc.id)}
                        className="w-full px-4 py-2 text-left text-sm hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors flex items-center gap-2 text-red-500"
                      >
                        <Trash2 className="w-3 h-3" />
                        Delete
                      </button>
                    </div>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
};

export default DocumentList;
import { X, FileText, Tag, Calendar, User, Link, Download, Edit, Trash2, ChevronDown, ChevronRight } from 'lucide-react';
import { useState } from 'react';
import type { Document, EntityInfo, RelationInfo } from '../../types';
import { formatDate, truncateText, formatLargeNumber } from '../../utils/formatters';

interface DocumentCardProps {
  document: Document | null;
  onClose: () => void;
  onEdit?: (doc: Document) => void;
  onDelete?: (id: string) => void;
  onDownload?: (doc: Document) => void;
  onViewInGraph?: (doc: Document) => void;
}

const DocumentCard: React.FC<DocumentCardProps> = ({
  document,
  onClose,
  onEdit,
  onDelete,
  onDownload,
  onViewInGraph,
}) => {
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set(['content']));

  if (!document) return null;

  const toggleSection = (section: string) => {
    const newExpanded = new Set(expandedSections);
    if (newExpanded.has(section)) {
      newExpanded.delete(section);
    } else {
      newExpanded.add(section);
    }
    setExpandedSections(newExpanded);
  };

  const renderEntities = (entities: EntityInfo[]) => {
    const top5 = entities.slice(0, 5);
    return (
      <div className="space-y-2">
        {top5.map((entity, index) => (
          <div key={index} className="flex items-start gap-2 p-2 bg-gray-50 dark:bg-gray-700 rounded">
            <span className="px-2 py-0.5 bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300 text-xs rounded">
              {entity.type || 'Entity'}
            </span>
            <div className="flex-1">
              <p className="font-medium text-sm">{entity.name}</p>
              {entity.description && (
                <p className="text-xs text-gray-600 dark:text-gray-400 mt-0.5">
                  {truncateText(entity.description, 100)}
                </p>
              )}
            </div>
            {entity.weight !== undefined && (
              <span className="text-xs text-gray-500">w: {entity.weight.toFixed(2)}</span>
            )}
          </div>
        ))}
        {entities.length > 5 && (
          <p className="text-sm text-gray-500 dark:text-gray-400 text-center">
            +{entities.length - 5} more entities
          </p>
        )}
      </div>
    );
  };

  const renderRelations = (relations: RelationInfo[]) => {
    const top5 = relations.slice(0, 5);
    return (
      <div className="space-y-2">
        {top5.map((relation, index) => (
          <div key={index} className="flex items-center gap-2 p-2 bg-gray-50 dark:bg-gray-700 rounded text-sm">
            <span className="font-medium">{relation.subject}</span>
            <span className="px-2 py-0.5 bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300 text-xs rounded">
              {relation.predicate}
            </span>
            <span className="font-medium">{relation.object}</span>
          </div>
        ))}
        {relations.length > 5 && (
          <p className="text-sm text-gray-500 dark:text-gray-400 text-center">
            +{relations.length - 5} more relations
          </p>
        )}
      </div>
    );
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-white dark:bg-gray-800 rounded-lg w-full max-w-4xl max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3">
            <FileText className="w-6 h-6 text-gray-500" />
            <div>
              <h2 className="text-xl font-semibold">{document.title}</h2>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                ID: {document.id}
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {/* Metadata */}
          {document.metadata && (
            <div>
              <h3 className="font-medium mb-3">Metadata</h3>
              <div className="grid grid-cols-2 gap-4 text-sm">
                {document.metadata.source && (
                  <div className="flex items-center gap-2 text-gray-600 dark:text-gray-400">
                    <Link className="w-4 h-4" />
                    <span>Source: {document.metadata.source}</span>
                  </div>
                )}
                {document.metadata.category && (
                  <div className="flex items-center gap-2 text-gray-600 dark:text-gray-400">
                    <Tag className="w-4 h-4" />
                    <span>Category: {document.metadata.category}</span>
                  </div>
                )}
                {document.metadata.author && (
                  <div className="flex items-center gap-2 text-gray-600 dark:text-gray-400">
                    <User className="w-4 h-4" />
                    <span>Author: {document.metadata.author}</span>
                  </div>
                )}
                {document.created_at && (
                  <div className="flex items-center gap-2 text-gray-600 dark:text-gray-400">
                    <Calendar className="w-4 h-4" />
                    <span>Created: {formatDate(document.created_at)}</span>
                  </div>
                )}
              </div>
              {document.metadata.tags && document.metadata.tags.length > 0 && (
                <div className="mt-3">
                  <div className="flex flex-wrap gap-2">
                    {document.metadata.tags.map((tag) => (
                      <span
                        key={tag}
                        className="px-2 py-1 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 text-sm rounded"
                      >
                        #{tag}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Statistics */}
          <div>
            <h3 className="font-medium mb-3">Statistics</h3>
            <div className="grid grid-cols-3 gap-4">
              <div className="bg-blue-50 dark:bg-blue-900/20 p-3 rounded-lg">
                <p className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                  {formatLargeNumber(document.entities?.length || 0)}
                </p>
                <p className="text-sm text-gray-600 dark:text-gray-400">Entities</p>
              </div>
              <div className="bg-red-50 dark:bg-red-900/20 p-3 rounded-lg">
                <p className="text-2xl font-bold text-red-600 dark:text-red-400">
                  {formatLargeNumber(document.relations?.length || 0)}
                </p>
                <p className="text-sm text-gray-600 dark:text-gray-400">Relations</p>
              </div>
              <div className="bg-green-50 dark:bg-green-900/20 p-3 rounded-lg">
                <p className="text-2xl font-bold text-green-600 dark:text-green-400">
                  {formatLargeNumber(document.chunks?.length || 0)}
                </p>
                <p className="text-sm text-gray-600 dark:text-gray-400">Chunks</p>
              </div>
            </div>
          </div>

          {/* Content */}
          {document.content && (
            <div>
              <button
                onClick={() => toggleSection('content')}
                className="flex items-center gap-2 font-medium mb-3 hover:text-blue-500 transition-colors"
              >
                {expandedSections.has('content') ? (
                  <ChevronDown className="w-4 h-4" />
                ) : (
                  <ChevronRight className="w-4 h-4" />
                )}
                Content Preview
              </button>
              {expandedSections.has('content') && (
                <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-lg">
                  <p className="text-sm text-gray-700 dark:text-gray-300 whitespace-pre-wrap">
                    {truncateText(document.content, 1000)}
                  </p>
                </div>
              )}
            </div>
          )}

          {/* Entities */}
          {document.entities && document.entities.length > 0 && (
            <div>
              <button
                onClick={() => toggleSection('entities')}
                className="flex items-center gap-2 font-medium mb-3 hover:text-blue-500 transition-colors"
              >
                {expandedSections.has('entities') ? (
                  <ChevronDown className="w-4 h-4" />
                ) : (
                  <ChevronRight className="w-4 h-4" />
                )}
                Extracted Entities ({document.entities.length})
              </button>
              {expandedSections.has('entities') && renderEntities(document.entities)}
            </div>
          )}

          {/* Relations */}
          {document.relations && document.relations.length > 0 && (
            <div>
              <button
                onClick={() => toggleSection('relations')}
                className="flex items-center gap-2 font-medium mb-3 hover:text-blue-500 transition-colors"
              >
                {expandedSections.has('relations') ? (
                  <ChevronDown className="w-4 h-4" />
                ) : (
                  <ChevronRight className="w-4 h-4" />
                )}
                Extracted Relations ({document.relations.length})
              </button>
              {expandedSections.has('relations') && renderRelations(document.relations)}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between p-6 border-t border-gray-200 dark:border-gray-700">
          <div className="flex gap-2">
            {onViewInGraph && (
              <button
                onClick={() => onViewInGraph(document)}
                className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
              >
                View in Graph
              </button>
            )}
            {onDownload && (
              <button
                onClick={() => onDownload(document)}
                className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors flex items-center gap-2"
              >
                <Download className="w-4 h-4" />
                Download
              </button>
            )}
          </div>
          <div className="flex gap-2">
            {onEdit && (
              <button
                onClick={() => onEdit(document)}
                className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors flex items-center gap-2"
              >
                <Edit className="w-4 h-4" />
                Edit
              </button>
            )}
            {onDelete && (
              <button
                onClick={() => onDelete(document.id)}
                className="px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 transition-colors flex items-center gap-2"
              >
                <Trash2 className="w-4 h-4" />
                Delete
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default DocumentCard;
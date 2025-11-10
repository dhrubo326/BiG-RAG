import React from 'react';
import {
  Download,
  Filter,
  Maximize,
  RefreshCw,
  Settings,
  Info,
  ZoomIn,
  ZoomOut,
} from 'lucide-react';
import { GRAPH_LAYOUTS } from '../../utils/constants';
import type { GraphLayout, GraphFilters } from '../../types';

interface GraphToolbarProps {
  layout: GraphLayout;
  filters: GraphFilters;
  onLayoutChange: (layout: GraphLayout) => void;
  onFiltersChange: (filters: Partial<GraphFilters>) => void;
  onExport: (format: 'png' | 'json' | 'graphml') => void;
  onFit: () => void;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onRefresh: () => void;
  onHelp: () => void;
}

const GraphToolbar: React.FC<GraphToolbarProps> = ({
  layout,
  filters,
  onLayoutChange,
  onFiltersChange,
  onExport,
  onFit,
  onZoomIn,
  onZoomOut,
  onRefresh,
  onHelp,
}) => {

  return (
    <div className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 p-3">
      <div className="flex items-center justify-between gap-4">
        {/* Layout Selector */}
        <div className="flex items-center gap-2">
          <Settings className="w-4 h-4 text-gray-500" />
          <select
            value={layout}
            onChange={(e) => onLayoutChange(e.target.value as GraphLayout)}
            className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
          >
            {GRAPH_LAYOUTS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>

        {/* Filters Dropdown */}
        <div className="relative group">
          <button
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors flex items-center gap-2"
            title="Filters"
          >
            <Filter className="w-4 h-4" />
            <span className="text-sm">Filters</span>
          </button>

          {/* Filters Dropdown Menu */}
          <div className="absolute top-full right-0 mt-2 w-64 bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 z-10">
            <div className="p-4 space-y-3">
              <div className="space-y-2">
                <label className="text-sm font-medium">Node Types</label>
                <div className="space-y-1">
                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      checked={filters.showEntities}
                      onChange={(e) => onFiltersChange({ showEntities: e.target.checked })}
                      className="mr-2"
                    />
                    <span className="text-sm">Entities</span>
                  </label>
                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      checked={filters.showRelations}
                      onChange={(e) => onFiltersChange({ showRelations: e.target.checked })}
                      className="mr-2"
                    />
                    <span className="text-sm">Relations</span>
                  </label>
                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      checked={filters.showChunks}
                      onChange={(e) => onFiltersChange({ showChunks: e.target.checked })}
                      className="mr-2"
                    />
                    <span className="text-sm">Chunks</span>
                  </label>
                </div>
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium">
                  Min Weight: {filters.minWeight.toFixed(2)}
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={filters.minWeight}
                  onChange={(e) =>
                    onFiltersChange({ minWeight: parseFloat(e.target.value) })
                  }
                  className="w-full"
                />
              </div>
            </div>
          </div>
        </div>

        {/* Export Dropdown */}
        <div className="relative group">
          <button
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors flex items-center gap-2"
            title="Export"
          >
            <Download className="w-4 h-4" />
            <span className="text-sm">Export</span>
          </button>

          <div className="absolute top-full right-0 mt-2 w-40 bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 z-10">
            <button
              onClick={() => onExport('png')}
              className="w-full px-4 py-2 text-left hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors text-sm"
            >
              Export as PNG
            </button>
            <button
              onClick={() => onExport('json')}
              className="w-full px-4 py-2 text-left hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors text-sm"
            >
              Export as JSON
            </button>
            <button
              onClick={() => onExport('graphml')}
              className="w-full px-4 py-2 text-left hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors text-sm"
            >
              Export as GraphML
            </button>
          </div>
        </div>

        {/* Zoom Controls */}
        <div className="flex items-center gap-1">
          <button
            onClick={onZoomIn}
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
            title="Zoom In"
          >
            <ZoomIn className="w-4 h-4" />
          </button>
          <button
            onClick={onZoomOut}
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
            title="Zoom Out"
          >
            <ZoomOut className="w-4 h-4" />
          </button>
          <button
            onClick={onFit}
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
            title="Fit to View"
          >
            <Maximize className="w-4 h-4" />
          </button>
        </div>

        {/* Additional Controls */}
        <div className="flex items-center gap-1">
          <button
            onClick={onRefresh}
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
            title="Refresh"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
          <button
            onClick={onHelp}
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
            title="Help"
          >
            <Info className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
};

export default GraphToolbar;
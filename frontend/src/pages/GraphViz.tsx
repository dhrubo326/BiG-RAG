/**
 * Graph Visualization Page
 *
 * Interactive bipartite graph visualization using Cytoscape.js:
 * - Entity nodes (blue)
 * - Relation nodes (red)
 * - Document chunk nodes (green)
 * - Interactive exploration (zoom, pan, click)
 * - Layout controls
 */
export function GraphViz() {
  return (
    <div className="container mx-auto p-6 h-full flex flex-col">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold">Knowledge Graph</h1>
        <div className="flex gap-2">
          <button className="px-4 py-2 border dark:border-gray-700 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800" disabled>
            Layout
          </button>
          <button className="px-4 py-2 border dark:border-gray-700 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800" disabled>
            Filter
          </button>
          <button className="px-4 py-2 border dark:border-gray-700 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800" disabled>
            Export
          </button>
        </div>
      </div>

      <div className="flex-1 bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <div className="h-full flex items-center justify-center border-2 border-dashed border-gray-300 dark:border-gray-700 rounded-lg">
          <div className="text-center">
            <p className="text-gray-500 dark:text-gray-400 text-lg mb-2">Graph Visualization</p>
            <p className="text-sm text-gray-400 dark:text-gray-500">
              Cytoscape.js integration will be implemented in Phase 2
            </p>
          </div>
        </div>
      </div>

      <div className="mt-4 bg-white dark:bg-gray-800 rounded-lg shadow p-4">
        <div className="flex gap-6 text-sm">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full bg-blue-500"></div>
            <span>Entities</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full bg-red-500"></div>
            <span>Relations</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full bg-green-500"></div>
            <span>Chunks</span>
          </div>
        </div>
      </div>
    </div>
  )
}

/**
 * Dashboard Page
 *
 * Main landing page showing:
 * - System health and stats
 * - Knowledge graph overview
 * - Recent queries
 * - Quick actions
 */
export function Dashboard() {
  return (
    <div className="container mx-auto p-6">
      <h1 className="text-3xl font-bold mb-6">Dashboard</h1>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
          <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400">Total Documents</h3>
          <p className="text-2xl font-bold mt-2">-</p>
        </div>

        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
          <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400">Total Entities</h3>
          <p className="text-2xl font-bold mt-2">-</p>
        </div>

        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
          <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400">Total Relations</h3>
          <p className="text-2xl font-bold mt-2">-</p>
        </div>

        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
          <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400">API Status</h3>
          <p className="text-2xl font-bold mt-2 text-yellow-600">Loading...</p>
        </div>
      </div>

      <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
        <h2 className="text-xl font-semibold mb-4">Welcome to BiG-RAG</h2>
        <p className="text-gray-600 dark:text-gray-300">
          Bipartite Graph Retrieval-Augmented Generation system is ready.
          Use the navigation to explore the knowledge graph, query documents, and manage your data.
        </p>
      </div>
    </div>
  )
}

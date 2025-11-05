/**
 * Evaluation Page
 *
 * Dataset evaluation and benchmarking:
 * - Run evaluations on QA datasets
 * - View EM/F1 scores
 * - Compare retrieval modes (hybrid, local, global, naive)
 * - View detailed results per query
 * - Export evaluation reports
 */
export function Evaluation() {
  return (
    <div className="container mx-auto p-6">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold">Evaluation</h1>
        <button className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700" disabled>
          Run Evaluation
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
          <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400">Exact Match (EM)</h3>
          <p className="text-2xl font-bold mt-2">-</p>
        </div>

        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
          <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400">F1 Score</h3>
          <p className="text-2xl font-bold mt-2">-</p>
        </div>

        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
          <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400">Total Queries</h3>
          <p className="text-2xl font-bold mt-2">-</p>
        </div>
      </div>

      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold mb-4">Evaluation Results</h2>
        <div className="text-center text-gray-500 dark:text-gray-400 py-12">
          <p className="text-lg mb-2">No evaluation results yet</p>
          <p className="text-sm">
            Run an evaluation to see detailed metrics and performance analysis
          </p>
        </div>
      </div>
    </div>
  )
}

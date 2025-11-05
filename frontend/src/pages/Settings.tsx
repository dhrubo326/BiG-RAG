/**
 * Settings Page
 *
 * Application configuration:
 * - API endpoint configuration
 * - Retrieval mode settings (hybrid, local, global, naive)
 * - Reranking toggle
 * - Theme preferences (light/dark)
 * - Language selection
 * - top_k and other query parameters
 */
export function Settings() {
  return (
    <div className="container mx-auto p-6 max-w-4xl">
      <h1 className="text-3xl font-bold mb-6">Settings</h1>

      <div className="space-y-6">
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <h2 className="text-xl font-semibold mb-4">API Configuration</h2>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">API Endpoint</label>
              <input
                type="text"
                defaultValue="http://localhost:8001"
                className="w-full px-4 py-2 border dark:border-gray-700 rounded-lg bg-white dark:bg-gray-900"
                disabled
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-2">Dataset</label>
              <select className="w-full px-4 py-2 border dark:border-gray-700 rounded-lg bg-white dark:bg-gray-900" disabled>
                <option>SingleTopic</option>
              </select>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <h2 className="text-xl font-semibold mb-4">Retrieval Settings</h2>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">Retrieval Mode</label>
              <select className="w-full px-4 py-2 border dark:border-gray-700 rounded-lg bg-white dark:bg-gray-900" disabled>
                <option>hybrid</option>
                <option>local</option>
                <option>global</option>
                <option>naive</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium mb-2">Top K Results</label>
              <input
                type="number"
                defaultValue="10"
                className="w-full px-4 py-2 border dark:border-gray-700 rounded-lg bg-white dark:bg-gray-900"
                disabled
              />
            </div>
            <div className="flex items-center gap-2">
              <input type="checkbox" id="reranking" className="rounded" disabled />
              <label htmlFor="reranking" className="text-sm font-medium">Enable Semantic Reranking</label>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <h2 className="text-xl font-semibold mb-4">Appearance</h2>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">Theme</label>
              <select className="w-full px-4 py-2 border dark:border-gray-700 rounded-lg bg-white dark:bg-gray-900" disabled>
                <option>Light</option>
                <option>Dark</option>
                <option>System</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium mb-2">Language</label>
              <select className="w-full px-4 py-2 border dark:border-gray-700 rounded-lg bg-white dark:bg-gray-900" disabled>
                <option>English</option>
                <option>中文</option>
              </select>
            </div>
          </div>
        </div>

        <p className="text-sm text-gray-500 dark:text-gray-400 text-center">
          Settings functionality will be implemented in Phase 2
        </p>
      </div>
    </div>
  )
}

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
export function Documents() {
  return (
    <div className="container mx-auto p-6">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold">Documents</h1>
        <button className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700" disabled>
          Upload Document
        </button>
      </div>

      <div className="mb-6">
        <input
          type="text"
          placeholder="Search documents..."
          className="w-full px-4 py-2 border dark:border-gray-700 rounded-lg bg-white dark:bg-gray-900"
          disabled
        />
      </div>

      <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
        <div className="p-6">
          <div className="text-center text-gray-500 dark:text-gray-400 py-12">
            <p className="text-lg mb-2">No documents loaded</p>
            <p className="text-sm">
              Document management interface will be implemented in Phase 2
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

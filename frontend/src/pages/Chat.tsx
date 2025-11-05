/**
 * Chat Page
 *
 * Interactive chat interface for:
 * - Querying the knowledge graph
 * - Real-time retrieval with BiG-RAG
 * - Viewing retrieved context
 * - Conversation history
 */
export function Chat() {
  return (
    <div className="container mx-auto p-6 h-full flex flex-col">
      <h1 className="text-3xl font-bold mb-6">Chat with BiG-RAG</h1>

      <div className="flex-1 bg-white dark:bg-gray-800 rounded-lg shadow p-6 flex flex-col">
        <div className="flex-1 overflow-y-auto mb-4 space-y-4">
          <div className="text-gray-500 dark:text-gray-400 text-center py-8">
            No messages yet. Start a conversation by typing a question below.
          </div>
        </div>

        <div className="border-t dark:border-gray-700 pt-4">
          <div className="flex gap-2">
            <input
              type="text"
              placeholder="Ask a question about your knowledge graph..."
              className="flex-1 px-4 py-2 border dark:border-gray-700 rounded-lg bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100"
              disabled
            />
            <button
              className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
              disabled
            >
              Send
            </button>
          </div>
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
            Chat functionality will be implemented in Phase 2
          </p>
        </div>
      </div>
    </div>
  )
}

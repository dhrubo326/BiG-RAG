/**
 * Footer Component
 *
 * Simple footer with:
 * - Copyright/credits
 * - Version info
 * - Links to docs/GitHub
 */
export function Footer() {
  const currentYear = new Date().getFullYear()

  return (
    <footer className="bg-white dark:bg-gray-800 border-t dark:border-gray-700 py-4 mt-auto">
      <div className="container mx-auto px-4">
        <div className="flex flex-col md:flex-row items-center justify-between gap-4 text-sm text-gray-600 dark:text-gray-400">
          <div>
            <p>
              BiG-RAG v1.0.0 - Bipartite Graph Retrieval-Augmented Generation
            </p>
            <p className="text-xs mt-1">
              Copyright {currentYear}. Built with React 19 + TypeScript + Tailwind CSS v4
            </p>
          </div>

          <div className="flex gap-4">
            <a
              href="https://github.com/dhrubo326/BiG-RAG"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
            >
              GitHub
            </a>
            <a
              href="https://github.com/dhrubo326/BiG-RAG/issues"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
            >
              Issues
            </a>
            <a
              href="https://github.com/dhrubo326/BiG-RAG/discussions"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
            >
              Discussions
            </a>
          </div>
        </div>
      </div>
    </footer>
  )
}

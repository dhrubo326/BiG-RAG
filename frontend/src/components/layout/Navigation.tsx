import { Link, useLocation } from 'react-router'
import { Home, MessageSquare, Brain, Network, FileText, BarChart3, Settings } from 'lucide-react'

/**
 * Navigation Component
 *
 * Top navigation bar with:
 * - BiG-RAG logo/title
 * - Page navigation links
 * - Active route highlighting
 */
export function Navigation() {
  const location = useLocation()

  const isActive = (path: string) => {
    return location.pathname === path
  }

  const navItems = [
    { path: '/', label: 'Dashboard', icon: Home },
    { path: '/chat', label: 'Chat', icon: MessageSquare },
    { path: '/agent', label: 'Agent', icon: Brain },
    { path: '/graph', label: 'Graph', icon: Network },
    { path: '/documents', label: 'Documents', icon: FileText },
    { path: '/evaluation', label: 'Evaluation', icon: BarChart3 },
    { path: '/settings', label: 'Settings', icon: Settings },
  ]

  return (
    <nav className="bg-white dark:bg-gray-800 border-b dark:border-gray-700 sticky top-0 z-50">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Logo/Title */}
          <Link to="/" className="flex items-center gap-2">
            <Network className="w-8 h-8 text-blue-600" />
            <span className="text-xl font-bold">BiG-RAG</span>
          </Link>

          {/* Navigation Links */}
          <div className="flex items-center gap-1">
            {navItems.map((item) => {
              const Icon = item.icon
              const active = isActive(item.path)

              return (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`
                    flex items-center gap-2 px-4 py-2 rounded-lg transition-colors
                    ${
                      active
                        ? 'bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300'
                        : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'
                    }
                  `}
                >
                  <Icon className="w-4 h-4" />
                  <span className="text-sm font-medium">{item.label}</span>
                </Link>
              )
            })}
          </div>
        </div>
      </div>
    </nav>
  )
}

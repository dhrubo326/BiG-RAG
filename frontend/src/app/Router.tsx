import { BrowserRouter, Routes, Route, Navigate } from 'react-router'
import { Dashboard } from '../pages/Dashboard'
import { Chat } from '../pages/Chat'
import { Agent } from '../pages/Agent'
import { GraphViz } from '../pages/GraphViz'
import { Documents } from '../pages/Documents'
import { Evaluation } from '../pages/Evaluation'
import { Settings } from '../pages/Settings'
import { Navigation } from '../components/layout/Navigation'
import { Footer } from '../components/layout/Footer'
import { ErrorBoundary } from '../components/ErrorBoundary'

/**
 * Router Component
 *
 * Application routing using React Router 7:
 * - / → Dashboard (default)
 * - /chat → Chat interface
 * - /agent → Multi-hop reasoning agent
 * - /graph → Graph visualization
 * - /documents → Document management
 * - /evaluation → Evaluation dashboard
 * - /settings → Settings page
 *
 * All routes wrapped with Navigation and Footer layout
 */
export function Router() {
  return (
    <BrowserRouter>
      <div className="min-h-screen flex flex-col bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-gray-100">
        <Navigation />

        <main className="flex-1 overflow-auto">
          <ErrorBoundary>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/chat" element={<Chat />} />
              <Route path="/agent" element={<Agent />} />
              <Route path="/graph" element={<GraphViz />} />
              <Route path="/documents" element={<Documents />} />
              <Route path="/evaluation" element={<Evaluation />} />
              <Route path="/settings" element={<Settings />} />

              {/* Redirect unknown routes to dashboard */}
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </ErrorBoundary>
        </main>

        <Footer />
      </div>
    </BrowserRouter>
  )
}

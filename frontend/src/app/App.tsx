import { useEffect } from 'react'
import { Router } from './Router'
import { Toaster } from 'sonner'

/**
 * App Component
 *
 * Root application component that:
 * - Sets up routing
 * - Provides toast notifications (Sonner)
 * - Handles global error boundaries
 * - Manages app-wide state initialization
 */
export function App() {
  useEffect(() => {
    // Log app initialization
    console.log('[BiG-RAG] Application initialized')

    // Check API connectivity on startup
    checkAPIConnection()
  }, [])

  const checkAPIConnection = async () => {
    try {
      const response = await fetch('/api/health')
      if (response.ok) {
        console.log('[BiG-RAG] Backend API connected')
      } else {
        console.warn('[BiG-RAG] Backend API returned non-OK status:', response.status)
      }
    } catch (error) {
      console.warn('[BiG-RAG] Backend API not reachable:', error)
    }
  }

  return (
    <>
      <Router />
      <Toaster position="top-right" richColors />
    </>
  )
}

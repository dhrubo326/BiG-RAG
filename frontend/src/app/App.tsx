import { useEffect } from 'react'
import { Router } from './Router'
import { Toaster } from 'sonner'
import { logger } from '@/utils/logger'

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
    logger.info('Application initialized')

    // Check API connectivity on startup
    checkAPIConnection()
  }, [])

  const checkAPIConnection = async () => {
    try {
      const response = await fetch('/api/health')
      if (response.ok) {
        logger.info('Backend API connected')
      } else {
        logger.warn('Backend API returned non-OK status', { status: response.status })
      }
    } catch (error) {
      logger.warn('Backend API not reachable', error)
    }
  }

  return (
    <>
      <Router />
      <Toaster position="top-right" richColors />
    </>
  )
}

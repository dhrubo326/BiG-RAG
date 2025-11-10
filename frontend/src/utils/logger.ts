/**
 * BiG-RAG Frontend Logger
 *
 * Provides structured console logging for the React frontend.
 * Logs appear in the browser's DevTools console.
 *
 * Usage:
 *   import { logger } from '@/utils/logger';
 *
 *   logger.info('User logged in', { userId: '123' });
 *   logger.error('Failed to fetch data', error);
 *   logger.debug('API response', response);
 */

export enum LogLevel {
  DEBUG = 0,
  INFO = 1,
  WARN = 2,
  ERROR = 3,
  NONE = 4
}

class Logger {
  private level: LogLevel;
  private prefix: string;

  constructor(prefix: string = '[BiG-RAG]', level: LogLevel = LogLevel.INFO) {
    this.prefix = prefix;
    this.level = level;

    // Set level from environment variable if available
    const envLevel = import.meta.env.VITE_LOG_LEVEL?.toUpperCase();
    if (envLevel && envLevel in LogLevel) {
      this.level = LogLevel[envLevel as keyof typeof LogLevel];
    }
  }

  /**
   * Set minimum log level
   */
  setLevel(level: LogLevel) {
    this.level = level;
  }

  /**
   * Format log message with timestamp and context
   */
  private format(level: string, message: string, context?: any): string {
    const timestamp = new Date().toISOString().split('T')[1].split('.')[0];
    return `${this.prefix} ${timestamp} [${level}] ${message}`;
  }

  /**
   * Debug-level logging (verbose)
   */
  debug(message: string, context?: any) {
    if (this.level <= LogLevel.DEBUG) {
      console.debug(this.format('DEBUG', message), context || '');
    }
  }

  /**
   * Info-level logging (general info)
   */
  info(message: string, context?: any) {
    if (this.level <= LogLevel.INFO) {
      console.log(this.format('INFO', message), context || '');
    }
  }

  /**
   * Warning-level logging
   */
  warn(message: string, context?: any) {
    if (this.level <= LogLevel.WARN) {
      console.warn(this.format('WARN', message), context || '');
    }
  }

  /**
   * Error-level logging
   */
  error(message: string, error?: Error | any) {
    if (this.level <= LogLevel.ERROR) {
      console.error(this.format('ERROR', message), error || '');
    }
  }

  /**
   * Create a child logger with custom prefix
   */
  child(childPrefix: string): Logger {
    return new Logger(`${this.prefix}:${childPrefix}`, this.level);
  }
}

// Export singleton instance
export const logger = new Logger();

// Export child loggers for specific modules
export const apiLogger = logger.child('API');
export const graphLogger = logger.child('Graph');
export const chatLogger = logger.child('Chat');
export const documentLogger = logger.child('Document');

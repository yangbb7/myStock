import React from 'react';
import { message, notification } from 'antd';
import { ApiError } from '../../services/types';

// Error severity levels
export type ErrorSeverity = 'low' | 'medium' | 'high' | 'critical';

// Error categories
export type ErrorCategory = 'network' | 'api' | 'validation' | 'system' | 'user';

// Enhanced error interface
export interface EnhancedError extends Error {
  code?: string;
  severity?: ErrorSeverity;
  category?: ErrorCategory;
  details?: any;
  timestamp?: string;
  requestId?: string;
  retryable?: boolean;
}

// Error logging service
class ErrorLogger {
  private static instance: ErrorLogger;
  private errorQueue: EnhancedError[] = [];
  private maxQueueSize = 100;

  static getInstance(): ErrorLogger {
    if (!ErrorLogger.instance) {
      ErrorLogger.instance = new ErrorLogger();
    }
    return ErrorLogger.instance;
  }

  log(error: EnhancedError): void {
    // Add timestamp if not present
    if (!error.timestamp) {
      error.timestamp = new Date().toISOString();
    }

    // Add to queue
    this.errorQueue.push(error);
    
    // Maintain queue size
    if (this.errorQueue.length > this.maxQueueSize) {
      this.errorQueue.shift();
    }

    // Log to console in development
    if (process.env.NODE_ENV === 'development') {
      console.error('[Error Logger]', error);
    }

    // Send to monitoring service in production
    if (process.env.NODE_ENV === 'production') {
      this.sendToMonitoring(error);
    }
  }

  private async sendToMonitoring(error: EnhancedError): Promise<void> {
    try {
      // In a real application, this would send to a monitoring service
      // like Sentry, LogRocket, or custom error tracking
      await fetch('/api/errors', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          error: {
            message: error.message,
            stack: error.stack,
            code: error.code,
            severity: error.severity,
            category: error.category,
            details: error.details,
            timestamp: error.timestamp,
            requestId: error.requestId,
            url: window.location.href,
            userAgent: navigator.userAgent,
          },
        }),
      });
    } catch (monitoringError) {
      console.error('Failed to send error to monitoring:', monitoringError);
    }
  }

  getErrors(): EnhancedError[] {
    return [...this.errorQueue];
  }

  clearErrors(): void {
    this.errorQueue = [];
  }
}

// Error handler utility functions
export class ErrorHandler {
  private static logger = ErrorLogger.getInstance();

  // Handle API errors
  static handleApiError(error: ApiError | any, context?: string): void {
    const enhancedError: EnhancedError = {
      name: 'ApiError',
      message: error.message || 'API request failed',
      code: error.code,
      severity: ErrorHandler.determineSeverity(error),
      category: 'api',
      details: error.details,
      timestamp: error.timestamp || new Date().toISOString(),
      retryable: ErrorHandler.isRetryable(error),
    };

    ErrorHandler.logger.log(enhancedError);
    ErrorHandler.showUserNotification(enhancedError, context);
  }

  // Handle network errors
  static handleNetworkError(error: Error, context?: string): void {
    const enhancedError: EnhancedError = {
      ...error,
      severity: 'high',
      category: 'network',
      timestamp: new Date().toISOString(),
      retryable: true,
    };

    ErrorHandler.logger.log(enhancedError);
    ErrorHandler.showUserNotification(enhancedError, context);
  }

  // Handle validation errors
  static handleValidationError(error: Error, context?: string): void {
    const enhancedError: EnhancedError = {
      ...error,
      severity: 'low',
      category: 'validation',
      timestamp: new Date().toISOString(),
      retryable: false,
    };

    ErrorHandler.logger.log(enhancedError);
    ErrorHandler.showUserNotification(enhancedError, context);
  }

  // Handle system errors
  static handleSystemError(error: Error, context?: string): void {
    const enhancedError: EnhancedError = {
      ...error,
      severity: 'critical',
      category: 'system',
      timestamp: new Date().toISOString(),
      retryable: false,
    };

    ErrorHandler.logger.log(enhancedError);
    ErrorHandler.showUserNotification(enhancedError, context);
  }

  // Determine error severity
  private static determineSeverity(error: any): ErrorSeverity {
    if (error.code?.startsWith('HTTP_5')) return 'critical';
    if (error.code?.startsWith('HTTP_4')) return 'medium';
    if (error.code === 'TIMEOUT') return 'high';
    if (error.code === 'NETWORK_ERROR') return 'high';
    return 'low';
  }

  // Check if error is retryable
  private static isRetryable(error: any): boolean {
    const retryableCodes = ['TIMEOUT', 'NETWORK_ERROR', 'HTTP_500', 'HTTP_502', 'HTTP_503', 'HTTP_504'];
    return retryableCodes.includes(error.code);
  }

  // Show user notification based on error severity
  private static showUserNotification(error: EnhancedError, context?: string): void {
    const contextMessage = context ? `[${context}] ` : '';
    
    switch (error.severity) {
      case 'critical':
        notification.error({
          message: '系统错误',
          description: `${contextMessage}${error.message}`,
          duration: 0, // Don't auto-close
          key: error.code || 'critical-error',
        });
        break;
        
      case 'high':
        notification.warning({
          message: '连接错误',
          description: `${contextMessage}${error.message}`,
          duration: 8,
          key: error.code || 'high-error',
        });
        break;
        
      case 'medium':
        message.warning(`${contextMessage}${error.message}`, 5);
        break;
        
      case 'low':
      default:
        message.error(`${contextMessage}${error.message}`, 3);
        break;
    }
  }

  // Get error statistics
  static getErrorStats(): {
    total: number;
    byCategory: Record<ErrorCategory, number>;
    bySeverity: Record<ErrorSeverity, number>;
    recent: EnhancedError[];
  } {
    const errors = ErrorHandler.logger.getErrors();
    const byCategory: Record<ErrorCategory, number> = {
      network: 0,
      api: 0,
      validation: 0,
      system: 0,
      user: 0,
    };
    const bySeverity: Record<ErrorSeverity, number> = {
      low: 0,
      medium: 0,
      high: 0,
      critical: 0,
    };

    errors.forEach(error => {
      if (error.category) byCategory[error.category]++;
      if (error.severity) bySeverity[error.severity]++;
    });

    return {
      total: errors.length,
      byCategory,
      bySeverity,
      recent: errors.slice(-10), // Last 10 errors
    };
  }

  // Clear all errors
  static clearErrors(): void {
    ErrorHandler.logger.clearErrors();
  }
}

// React hook for error handling
export const useErrorHandler = () => {
  const handleError = React.useCallback((error: any, context?: string) => {
    if (error.code) {
      ErrorHandler.handleApiError(error, context);
    } else if (error.name === 'NetworkError' || error.message?.includes('network')) {
      ErrorHandler.handleNetworkError(error, context);
    } else {
      ErrorHandler.handleSystemError(error, context);
    }
  }, []);

  const handleValidationError = React.useCallback((error: Error, context?: string) => {
    ErrorHandler.handleValidationError(error, context);
  }, []);

  const getErrorStats = React.useCallback(() => {
    return ErrorHandler.getErrorStats();
  }, []);

  const clearErrors = React.useCallback(() => {
    ErrorHandler.clearErrors();
  }, []);

  return {
    handleError,
    handleValidationError,
    getErrorStats,
    clearErrors,
  };
};

export default ErrorHandler;
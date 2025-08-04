import { message, notification } from 'antd';
import { ErrorHandler } from '../components/Common/ErrorHandler';
import { errorReporting } from '../services/errorReporting';

// Global error handling configuration
export interface ErrorHandlingConfig {
  enableErrorReporting?: boolean;
  enableGlobalHandlers?: boolean;
  enableConsoleLogging?: boolean;
  reportingEndpoint?: string;
  maxErrorReports?: number;
  notificationDuration?: number;
  userId?: string;
}

// Default configuration
const defaultConfig: ErrorHandlingConfig = {
  enableErrorReporting: true,
  enableGlobalHandlers: true,
  enableConsoleLogging: process.env.NODE_ENV === 'development',
  maxErrorReports: 1000,
  notificationDuration: 5,
};

// Setup global error handling
export const setupErrorHandling = (config: ErrorHandlingConfig = {}) => {
  const mergedConfig = { ...defaultConfig, ...config };

  // Configure error reporting
  if (mergedConfig.enableErrorReporting) {
    if (mergedConfig.userId) {
      errorReporting.setUserId(mergedConfig.userId);
    }
  }

  // Configure Ant Design message and notification
  message.config({
    duration: mergedConfig.notificationDuration,
    maxCount: 3,
  });

  notification.config({
    duration: mergedConfig.notificationDuration,
    maxCount: 5,
    placement: 'topRight',
  });

  // Setup global error handlers
  if (mergedConfig.enableGlobalHandlers) {
    setupGlobalErrorHandlers(mergedConfig);
  }

  // Setup console logging
  if (mergedConfig.enableConsoleLogging) {
    setupConsoleLogging();
  }

  console.log('Error handling system initialized', mergedConfig);
};

// Setup global error handlers
const setupGlobalErrorHandlers = (config: ErrorHandlingConfig) => {
  // Handle unhandled JavaScript errors
  window.addEventListener('error', (event) => {
    const error = event.error || new Error(event.message);
    
    ErrorHandler.handleSystemError(error, 'global-error');
    
    if (config.enableConsoleLogging) {
      console.error('Global error caught:', {
        message: event.message,
        filename: event.filename,
        lineno: event.lineno,
        colno: event.colno,
        error: event.error,
      });
    }
  });

  // Handle unhandled promise rejections
  window.addEventListener('unhandledrejection', (event) => {
    const error = new Error(event.reason?.message || 'Unhandled promise rejection');
    error.name = 'UnhandledPromiseRejection';
    
    ErrorHandler.handleSystemError(error, 'unhandled-promise');
    
    if (config.enableConsoleLogging) {
      console.error('Unhandled promise rejection:', event.reason);
    }
    
    // Prevent the default browser behavior
    event.preventDefault();
  });

  // Handle resource loading errors
  window.addEventListener('error', (event) => {
    if (event.target && event.target !== window) {
      const target = event.target as HTMLElement;
      const error = new Error(`Resource loading failed: ${target.tagName}`);
      error.name = 'ResourceLoadError';
      
      ErrorHandler.handleSystemError(error, 'resource-load');
      
      if (config.enableConsoleLogging) {
        console.error('Resource loading error:', {
          element: target.tagName,
          src: (target as any).src || (target as any).href,
          target,
        });
      }
    }
  }, true);

  // Handle network errors (disabled notifications to reduce popup noise)
  window.addEventListener('offline', () => {
    // Just log the offline status, no notification
    if (config.enableConsoleLogging) {
      console.warn('Network offline detected');
    }
  });

  window.addEventListener('online', () => {
    // Just log the online status, no notification
    if (config.enableConsoleLogging) {
      console.log('Network online detected');
    }
    
    // Retry failed error reports
    errorReporting.retryFailedReports().catch(console.warn);
  });
};

// Setup console logging enhancements
const setupConsoleLogging = () => {
  // Enhance console.error to include stack traces
  const originalError = console.error;
  console.error = (...args: any[]) => {
    originalError.apply(console, args);
    
    // Add stack trace for better debugging
    if (args[0] instanceof Error) {
      originalError('Stack trace:', args[0].stack);
    }
  };

  // Add performance monitoring
  const originalFetch = window.fetch;
  window.fetch = async (...args) => {
    const startTime = performance.now();
    
    try {
      const response = await originalFetch(...args);
      const endTime = performance.now();
      
      console.log(`[Fetch] ${args[0]} - ${response.status} (${Math.round(endTime - startTime)}ms)`);
      
      return response;
    } catch (error) {
      const endTime = performance.now();
      console.error(`[Fetch Error] ${args[0]} - ${error} (${Math.round(endTime - startTime)}ms)`);
      throw error;
    }
  };
};

// Error handling utilities
export const errorHandlingUtils = {
  // Test error handling system
  testErrorHandling: () => {
    console.log('Testing error handling system...');
    
    // Test JavaScript error
    setTimeout(() => {
      throw new Error('Test JavaScript error');
    }, 100);
    
    // Test promise rejection
    setTimeout(() => {
      Promise.reject(new Error('Test promise rejection'));
    }, 200);
    
    // Test API error
    setTimeout(() => {
      ErrorHandler.handleApiError({
        code: 'TEST_ERROR',
        message: 'Test API error',
        details: { test: true },
        timestamp: new Date().toISOString(),
      }, 'test');
    }, 300);
  },

  // Get error statistics
  getErrorStats: () => {
    return errorReporting.getErrorStats();
  },

  // Clear all errors
  clearAllErrors: () => {
    ErrorHandler.clearErrors();
    errorReporting.clearReports();
    notification.destroy();
    message.destroy();
  },

  // Export error reports
  exportErrorReports: () => {
    const reports = errorReporting.exportReports();
    const blob = new Blob([reports], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = `error-reports-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    
    URL.revokeObjectURL(url);
  },

  // Health check
  healthCheck: () => {
    const stats = errorReporting.getErrorStats();
    const criticalErrors = stats.bySeverity.critical || 0;
    const recentErrors = stats.recent.filter(
      report => Date.now() - new Date(report.timestamp).getTime() < 5 * 60 * 1000
    ).length;

    return {
      healthy: criticalErrors === 0 && recentErrors < 5,
      criticalErrors,
      recentErrors,
      totalErrors: stats.total,
      lastError: stats.recent[stats.recent.length - 1],
    };
  },
};

// React context for error handling configuration
import React, { createContext, useContext, ReactNode } from 'react';

interface ErrorHandlingContextType {
  config: ErrorHandlingConfig;
  utils: typeof errorHandlingUtils;
}

const ErrorHandlingContext = createContext<ErrorHandlingContextType | null>(null);

export const ErrorHandlingProvider: React.FC<{
  config?: ErrorHandlingConfig;
  children: ReactNode;
}> = ({ config = {}, children }) => {
  React.useEffect(() => {
    setupErrorHandling(config);
  }, [config]);

  const contextValue = {
    config: { ...defaultConfig, ...config },
    utils: errorHandlingUtils,
  };

  return (
    <ErrorHandlingContext.Provider value={contextValue}>
      {children}
    </ErrorHandlingContext.Provider>
  );
};

export const useErrorHandlingContext = () => {
  const context = useContext(ErrorHandlingContext);
  if (!context) {
    throw new Error('useErrorHandlingContext must be used within ErrorHandlingProvider');
  }
  return context;
};

export default setupErrorHandling;
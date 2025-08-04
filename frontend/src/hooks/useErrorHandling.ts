import { useState, useCallback, useEffect } from 'react';
import { message, notification } from 'antd';
import { ErrorHandler } from '../components/Common/ErrorHandler';
import { errorReporting } from '../services/errorReporting';
import { ApiError } from '../services/types';

// Error handling options
interface ErrorHandlingOptions {
  showNotification?: boolean;
  showMessage?: boolean;
  reportError?: boolean;
  context?: string;
  component?: string;
  retryable?: boolean;
  onError?: (error: Error) => void;
  onRetry?: () => void;
}

// Error state interface
interface ErrorState {
  error: Error | null;
  hasError: boolean;
  errorId?: string;
  retryCount: number;
  isRetrying: boolean;
}

// Main error handling hook
export const useErrorHandling = (options: ErrorHandlingOptions = {}) => {
  const [errorState, setErrorState] = useState<ErrorState>({
    error: null,
    hasError: false,
    retryCount: 0,
    isRetrying: false,
  });

  // Handle error function
  const handleError = useCallback((
    error: Error | ApiError,
    customOptions?: Partial<ErrorHandlingOptions>
  ) => {
    const mergedOptions = { ...options, ...customOptions };
    
    // Update error state
    const errorId = mergedOptions.reportError !== false 
      ? errorReporting.reportError(error, {
          component: mergedOptions.component,
          action: mergedOptions.context,
        })
      : undefined;

    setErrorState(prev => ({
      ...prev,
      error,
      hasError: true,
      errorId,
    }));

    // Show user notifications
    if (mergedOptions.showNotification !== false) {
      ErrorHandler.handleApiError(error, mergedOptions.context);
    }

    if (mergedOptions.showMessage) {
      const errorMessage = typeof error === 'string' ? error : error.message;
      message.error(errorMessage);
    }

    // Call custom error handler
    if (mergedOptions.onError) {
      mergedOptions.onError(error);
    }
  }, [options]);

  // Clear error function
  const clearError = useCallback(() => {
    setErrorState({
      error: null,
      hasError: false,
      retryCount: 0,
      isRetrying: false,
    });
  }, []);

  // Retry function
  const retry = useCallback(async () => {
    if (!options.onRetry) return;

    setErrorState(prev => ({
      ...prev,
      isRetrying: true,
      retryCount: prev.retryCount + 1,
    }));

    try {
      await options.onRetry();
      clearError();
    } catch (error) {
      handleError(error as Error, { context: 'retry' });
    } finally {
      setErrorState(prev => ({
        ...prev,
        isRetrying: false,
      }));
    }
  }, [options.onRetry, handleError, clearError]);

  return {
    ...errorState,
    handleError,
    clearError,
    retry,
  };
};

// Hook for API error handling
export const useApiErrorHandling = (context?: string) => {
  const { handleError, ...rest } = useErrorHandling({
    showNotification: true,
    reportError: true,
    context,
    component: 'api',
  });

  const handleApiError = useCallback((error: ApiError | Error) => {
    // Enhanced API error handling
    if ('code' in error) {
      const apiError = error as ApiError;
      
      // Handle specific API error codes
      switch (apiError.code) {
        case 'HTTP_401':
          notification.error({
            message: '认证失败',
            description: '请重新登录',
            key: 'auth-error',
          });
          // Could trigger logout here
          break;
          
        case 'HTTP_403':
          notification.error({
            message: '权限不足',
            description: '您没有权限执行此操作',
            key: 'permission-error',
          });
          break;
          
        case 'HTTP_429':
          notification.warning({
            message: '请求过于频繁',
            description: '请稍后再试',
            key: 'rate-limit-error',
          });
          break;
          
        case 'NETWORK_ERROR':
          notification.error({
            message: '网络连接失败',
            description: '请检查网络连接',
            key: 'network-error',
          });
          break;
          
        default:
          handleError(error);
      }
    } else {
      handleError(error);
    }
  }, [handleError]);

  return {
    ...rest,
    handleError: handleApiError,
  };
};

// Hook for form validation errors
export const useValidationErrorHandling = () => {
  const [validationErrors, setValidationErrors] = useState<Record<string, string[]>>({});

  const handleValidationError = useCallback((errors: Record<string, string[]>) => {
    setValidationErrors(errors);
    
    // Show first error as message
    const firstError = Object.values(errors)[0]?.[0];
    if (firstError) {
      message.error(firstError);
    }
  }, []);

  const clearValidationErrors = useCallback(() => {
    setValidationErrors({});
  }, []);

  const getFieldError = useCallback((field: string) => {
    return validationErrors[field]?.[0];
  }, [validationErrors]);

  const hasFieldError = useCallback((field: string) => {
    return !!validationErrors[field]?.length;
  }, [validationErrors]);

  return {
    validationErrors,
    handleValidationError,
    clearValidationErrors,
    getFieldError,
    hasFieldError,
  };
};

// Hook for async operation error handling
export const useAsyncErrorHandling = <T,>(
  asyncFn: () => Promise<T>,
  options: ErrorHandlingOptions & {
    onSuccess?: (data: T) => void;
    immediate?: boolean;
  } = {}
) => {
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<T | null>(null);
  const { handleError, ...errorState } = useErrorHandling(options);

  const execute = useCallback(async () => {
    setLoading(true);
    
    try {
      const result = await asyncFn();
      setData(result);
      
      if (options.onSuccess) {
        options.onSuccess(result);
      }
      
      return result;
    } catch (error) {
      handleError(error as Error);
      throw error;
    } finally {
      setLoading(false);
    }
  }, [asyncFn, handleError, options]);

  // Execute immediately if requested
  useEffect(() => {
    if (options.immediate) {
      execute();
    }
  }, [execute, options.immediate]);

  return {
    loading,
    data,
    execute,
    ...errorState,
  };
};

// Hook for component-level error boundary
export const useComponentErrorHandling = (componentName: string) => {
  const { handleError, ...rest } = useErrorHandling({
    showNotification: true,
    reportError: true,
    component: componentName,
  });

  // Wrap component methods with error handling
  const withErrorHandling = useCallback(<T extends any[], R>(
    fn: (...args: T) => R,
    context?: string
  ) => {
    return (...args: T): R | undefined => {
      try {
        return fn(...args);
      } catch (error) {
        handleError(error as Error, { context });
        return undefined;
      }
    };
  }, [handleError]);

  // Wrap async component methods
  const withAsyncErrorHandling = useCallback(<T extends any[], R>(
    fn: (...args: T) => Promise<R>,
    context?: string
  ) => {
    return async (...args: T): Promise<R | undefined> => {
      try {
        return await fn(...args);
      } catch (error) {
        handleError(error as Error, { context });
        return undefined;
      }
    };
  }, [handleError]);

  return {
    ...rest,
    handleError,
    withErrorHandling,
    withAsyncErrorHandling,
  };
};

// Global error handler setup
export const setupGlobalErrorHandling = () => {
  // This would be called in the main app component
  const errorHandler = ErrorHandler;
  
  // Setup global handlers
  window.addEventListener('error', (event) => {
    errorHandler.handleSystemError(event.error, 'global');
  });

  window.addEventListener('unhandledrejection', (event) => {
    errorHandler.handleSystemError(new Error(event.reason), 'unhandled-promise');
  });
};

export default useErrorHandling;
import { useEffect, useCallback, useMemo, useRef, useState } from 'react';
import { usePerformanceMonitor } from '../utils/performanceMonitor';
import { apiCache, chartDataCache } from '../utils/cacheManager';
import { debounce, throttle } from 'lodash';

interface PerformanceOptimizationOptions {
  componentName: string;
  enableCaching?: boolean;
  enableDebouncing?: boolean;
  enableThrottling?: boolean;
  debounceDelay?: number;
  throttleDelay?: number;
  enableMemoryMonitoring?: boolean;
  maxMemoryUsage?: number; // in MB
}

export const usePerformanceOptimization = (options: PerformanceOptimizationOptions) => {
  const {
    componentName,
    enableCaching = true,
    enableDebouncing = false,
    enableThrottling = false,
    debounceDelay = 300,
    throttleDelay = 100,
    enableMemoryMonitoring = true,
    maxMemoryUsage = 50
  } = options;

  const performanceMonitor = usePerformanceMonitor(componentName);
  const memoryCheckInterval = useRef<NodeJS.Timeout | null>(null);

  // Memory monitoring
  useEffect(() => {
    if (!enableMemoryMonitoring) return;

    const checkMemory = () => {
      const metrics = performanceMonitor.getMetrics();
      if (metrics.memoryUsage && metrics.memoryUsage.used > maxMemoryUsage * 1024 * 1024) {
        console.warn(`High memory usage detected in ${componentName}:`, {
          used: `${(metrics.memoryUsage.used / 1024 / 1024).toFixed(2)}MB`,
          percentage: `${metrics.memoryUsage.usagePercentage.toFixed(2)}%`
        });
        
        // Trigger cache cleanup
        if (enableCaching) {
          apiCache.clear();
          chartDataCache.clear();
        }
      }
    };

    memoryCheckInterval.current = setInterval(checkMemory, 30000); // Check every 30 seconds

    return () => {
      if (memoryCheckInterval.current) {
        clearInterval(memoryCheckInterval.current);
      }
    };
  }, [componentName, enableMemoryMonitoring, maxMemoryUsage, enableCaching, performanceMonitor]);

  // Optimized debounce function
  const createOptimizedDebounce = useCallback(<T extends (...args: any[]) => any>(
    func: T,
    delay: number = debounceDelay
  ): T => {
    if (!enableDebouncing) return func;
    
    const debouncedFn = debounce(func, delay, {
      leading: false,
      trailing: true,
      maxWait: delay * 2
    });

    return debouncedFn as T;
  }, [enableDebouncing, debounceDelay]);

  // Optimized throttle function
  const createOptimizedThrottle = useCallback(<T extends (...args: any[]) => any>(
    func: T,
    delay: number = throttleDelay
  ): T => {
    if (!enableThrottling) return func;
    
    const throttledFn = throttle(func, delay, {
      leading: true,
      trailing: true
    });

    return throttledFn as T;
  }, [enableThrottling, throttleDelay]);

  // Memoized cache operations
  const cacheOperations = useMemo(() => ({
    get: (key: string) => {
      if (!enableCaching) return null;
      const result = apiCache.get(key);
      if (result) {
        performanceMonitor.recordCacheHit();
      } else {
        performanceMonitor.recordCacheMiss();
      }
      return result;
    },
    set: (key: string, value: any, ttl?: number) => {
      if (!enableCaching) return;
      apiCache.set(key, value, ttl);
    },
    clear: () => {
      if (!enableCaching) return;
      apiCache.clear();
      chartDataCache.clear();
    }
  }), [enableCaching, performanceMonitor]);

  // Performance metrics
  const getPerformanceMetrics = useCallback(() => {
    return performanceMonitor.getReport();
  }, [performanceMonitor]);

  // Cleanup function
  const cleanup = useCallback(() => {
    if (memoryCheckInterval.current) {
      clearInterval(memoryCheckInterval.current);
    }
    if (enableCaching) {
      // Don't clear all cache, just this component's data
      // apiCache.clear();
    }
  }, [enableCaching]);

  return {
    // Optimization functions
    debounce: createOptimizedDebounce,
    throttle: createOptimizedThrottle,
    
    // Cache operations
    cache: cacheOperations,
    
    // Performance monitoring
    getMetrics: getPerformanceMetrics,
    recordNetworkRequest: performanceMonitor.recordNetworkRequest,
    
    // Cleanup
    cleanup
  };
};

// Hook for component-level performance optimization
export const useComponentOptimization = (componentName: string) => {
  const renderCount = useRef(0);
  const lastRenderTime = useRef(Date.now());
  const performanceMonitor = usePerformanceMonitor(componentName);

  useEffect(() => {
    renderCount.current++;
    const now = Date.now();
    const timeSinceLastRender = now - lastRenderTime.current;
    lastRenderTime.current = now;

    // Warn about frequent re-renders
    if (timeSinceLastRender < 16 && renderCount.current > 5) {
      console.warn(`Frequent re-renders detected in ${componentName}:`, {
        renderCount: renderCount.current,
        timeSinceLastRender
      });
    }
  });

  return {
    renderCount: renderCount.current,
    getMetrics: performanceMonitor.getMetrics
  };
};

// Hook for optimizing large lists
export const useListOptimization = <T>(
  items: T[],
  options: {
    pageSize?: number;
    enableVirtualization?: boolean;
    virtualizationThreshold?: number;
  } = {}
) => {
  const {
    pageSize = 50,
    enableVirtualization = true,
    virtualizationThreshold = 100
  } = options;

  // Determine if virtualization should be enabled
  const shouldVirtualize = useMemo(() => {
    return enableVirtualization && items.length > virtualizationThreshold;
  }, [enableVirtualization, items.length, virtualizationThreshold]);

  // Paginated data for non-virtualized lists
  const [currentPage, setCurrentPage] = useState(1);
  const paginatedItems = useMemo(() => {
    if (shouldVirtualize) return items;
    
    const startIndex = (currentPage - 1) * pageSize;
    const endIndex = startIndex + pageSize;
    return items.slice(startIndex, endIndex);
  }, [items, currentPage, pageSize, shouldVirtualize]);

  const totalPages = Math.ceil(items.length / pageSize);

  return {
    items: paginatedItems,
    shouldVirtualize,
    currentPage,
    totalPages,
    setCurrentPage,
    hasNextPage: currentPage < totalPages,
    hasPrevPage: currentPage > 1
  };
};

// Hook for optimizing API calls
export const useApiOptimization = () => {
  const requestQueue = useRef<Array<() => Promise<any>>>([]);
  const isProcessing = useRef(false);
  const maxConcurrentRequests = 5;
  const requestDelay = 50; // ms between requests

  const processQueue = useCallback(async () => {
    if (isProcessing.current || requestQueue.current.length === 0) return;

    isProcessing.current = true;
    const batch = requestQueue.current.splice(0, maxConcurrentRequests);
    
    try {
      await Promise.all(batch.map(request => request()));
    } catch (error) {
      console.error('Batch request error:', error);
    }

    // Small delay before processing next batch
    setTimeout(() => {
      isProcessing.current = false;
      processQueue();
    }, requestDelay);
  }, []);

  const queueRequest = useCallback((request: () => Promise<any>) => {
    requestQueue.current.push(request);
    processQueue();
  }, [processQueue]);

  return {
    queueRequest,
    queueLength: requestQueue.current.length,
    isProcessing: isProcessing.current
  };
};
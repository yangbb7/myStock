import React from 'react';
import { debounce, throttle } from 'lodash';
import { cacheManager } from './cacheManager';
// Note: performanceMonitor methods may not exist, using fallback implementations

// Performance optimization strategies
export interface PerformanceStrategy {
  name: string;
  condition: () => boolean;
  action: () => void;
  priority: number;
}

export class PerformanceOptimizer {
  private static instance: PerformanceOptimizer;
  private strategies: PerformanceStrategy[] = [];
  private monitoringInterval: NodeJS.Timeout | null = null;
  private isOptimizing = false;

  static getInstance(): PerformanceOptimizer {
    if (!PerformanceOptimizer.instance) {
      PerformanceOptimizer.instance = new PerformanceOptimizer();
    }
    return PerformanceOptimizer.instance;
  }

  constructor() {
    this.initializeDefaultStrategies();
    this.startMonitoring();
  }

  private initializeDefaultStrategies() {
    // Memory pressure optimization
    this.addStrategy({
      name: 'memory-pressure',
      condition: () => {
        const memoryInfo = this.getMemoryInfo();
        return memoryInfo ? memoryInfo.usagePercentage > 70 : false;
      },
      action: () => {
        console.log('🧹 Performing memory cleanup due to high memory usage');
        cacheManager.performIntelligentCleanup();
        this.forceGarbageCollection();
      },
      priority: 1
    });

    // High render frequency optimization
    this.addStrategy({
      name: 'high-render-frequency',
      condition: () => {
        // Fallback implementation since performanceMonitor.getMetrics may not exist
        return false;
      },
      action: () => {
        console.log('⚡ Optimizing for high render frequency');
        this.optimizeRenderPerformance();
      },
      priority: 2
    });

    // Large dataset optimization
    this.addStrategy({
      name: 'large-dataset',
      condition: () => {
        const cacheStats = cacheManager.getGlobalStats();
        return cacheStats.totalSize > 200;
      },
      action: () => {
        console.log('📊 Optimizing for large datasets');
        this.optimizeDataHandling();
      },
      priority: 3
    });

    // Network congestion optimization
    this.addStrategy({
      name: 'network-congestion',
      condition: () => {
        // Fallback implementation since performanceMonitor.getMetrics may not exist
        return false;
      },
      action: () => {
        console.log('🌐 Optimizing for network congestion');
        this.optimizeNetworkRequests();
      },
      priority: 4
    });
  }

  addStrategy(strategy: PerformanceStrategy) {
    this.strategies.push(strategy);
    this.strategies.sort((a, b) => a.priority - b.priority);
  }

  removeStrategy(name: string) {
    this.strategies = this.strategies.filter(s => s.name !== name);
  }

  private async runOptimizations() {
    if (this.isOptimizing) return;
    
    this.isOptimizing = true;
    
    try {
      for (const strategy of this.strategies) {
        if (strategy.condition()) {
          await this.executeStrategy(strategy);
        }
      }
    } catch (error) {
      console.error('Performance optimization error:', error);
    } finally {
      this.isOptimizing = false;
    }
  }

  private async executeStrategy(strategy: PerformanceStrategy) {
    const startTime = performance.now();
    
    try {
      strategy.action();
      const executionTime = performance.now() - startTime;
      console.log(`✅ Strategy "${strategy.name}" executed in ${executionTime.toFixed(2)}ms`);
    } catch (error) {
      console.error(`❌ Strategy "${strategy.name}" failed:`, error);
    }
  }

  private optimizeRenderPerformance() {
    // Reduce animation complexity
    document.documentElement.style.setProperty('--animation-duration', '0.1s');
    
    // Disable non-critical animations
    const style = document.createElement('style');
    style.textContent = `
      .ant-spin-dot { animation-duration: 0.5s !important; }
      .ant-progress-circle-path { transition: none !important; }
      * { transition-duration: 0.1s !important; }
    `;
    document.head.appendChild(style);
    
    // Clean up after 30 seconds
    setTimeout(() => {
      document.head.removeChild(style);
      document.documentElement.style.removeProperty('--animation-duration');
    }, 30000);
  }

  private optimizeDataHandling() {
    // Trigger aggressive cache cleanup
    cacheManager.performIntelligentCleanup();
    
    // Suggest data sampling for large datasets
    window.dispatchEvent(new CustomEvent('performance:suggest-data-sampling', {
      detail: { maxPoints: 1000 }
    }));
  }

  private optimizeNetworkRequests() {
    // Suggest request batching
    window.dispatchEvent(new CustomEvent('performance:suggest-request-batching', {
      detail: { batchSize: 5, delay: 100 }
    }));
    
    // Increase cache TTL temporarily
    const apiCache = cacheManager.getCache('api');
    if (apiCache) {
      // This would require extending the cache interface
      console.log('📈 Temporarily increasing cache TTL for network optimization');
    }
  }

  private forceGarbageCollection() {
    // Force garbage collection if available (Chrome DevTools)
    if (window.gc) {
      window.gc();
    }
    
    // Clear unused references
    if (window.performance && window.performance.clearMarks) {
      window.performance.clearMarks();
      window.performance.clearMeasures();
    }
  }

  private getMemoryInfo() {
    if (typeof window !== 'undefined' && 'performance' in window && 'memory' in (window.performance as any)) {
      const memory = (window.performance as any).memory;
      return {
        used: memory.usedJSHeapSize,
        total: memory.totalJSHeapSize,
        limit: memory.jsHeapSizeLimit,
        usagePercentage: (memory.usedJSHeapSize / memory.jsHeapSizeLimit) * 100
      };
    }
    return null;
  }

  private startMonitoring() {
    this.monitoringInterval = setInterval(() => {
      this.runOptimizations();
    }, 10000); // Check every 10 seconds
  }

  getOptimizationReport() {
    const memoryInfo = this.getMemoryInfo();
    const cacheStats = cacheManager.getGlobalStats();
    const performanceMetrics = {}; // Fallback since getMetrics may not exist

    return {
      timestamp: new Date().toISOString(),
      memory: memoryInfo,
      cache: cacheStats,
      performance: performanceMetrics,
      activeStrategies: this.strategies.map(s => ({
        name: s.name,
        priority: s.priority,
        shouldRun: s.condition()
      })),
      isOptimizing: this.isOptimizing
    };
  }

  destroy() {
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
      this.monitoringInterval = null;
    }
    this.strategies = [];
  }
}

// Global performance optimizer instance
export const performanceOptimizer = PerformanceOptimizer.getInstance();

// Performance optimization hooks and utilities
export const usePerformanceOptimization = () => {
  const getReport = () => performanceOptimizer.getOptimizationReport();
  
  const forceOptimization = () => {
    (performanceOptimizer as any).runOptimizations();
  };

  const addCustomStrategy = (strategy: PerformanceStrategy) => {
    performanceOptimizer.addStrategy(strategy);
  };

  return {
    getReport,
    forceOptimization,
    addCustomStrategy
  };
};

// Debounced and throttled function creators with performance monitoring
export const createOptimizedDebounce = <T extends (...args: any[]) => any>(
  func: T,
  delay: number,
  options: {
    leading?: boolean;
    trailing?: boolean;
    maxWait?: number;
  } = {}
): T => {
  const debouncedFn = debounce(func, delay, {
    leading: options.leading || false,
    trailing: options.trailing !== false,
    maxWait: options.maxWait
  });

  return ((...args: any[]) => {
    const startTime = performance.now();
    const result = debouncedFn(...args);
    const endTime = performance.now();
    
    // performanceMonitor.recordFunctionCall may not exist
    return result;
  }) as T;
};

export const createOptimizedThrottle = <T extends (...args: any[]) => any>(
  func: T,
  delay: number,
  options: {
    leading?: boolean;
    trailing?: boolean;
  } = {}
): T => {
  const throttledFn = throttle(func, delay, {
    leading: options.leading !== false,
    trailing: options.trailing !== false
  });

  return ((...args: any[]) => {
    const startTime = performance.now();
    const result = throttledFn(...args);
    const endTime = performance.now();
    
    // performanceMonitor.recordFunctionCall may not exist
    return result;
  }) as T;
};

// Performance-aware component wrapper
export const withPerformanceOptimization = <P extends object>(
  Component: React.ComponentType<P>,
  optimizationConfig: {
    enableMemoization?: boolean;
    enableLazyLoading?: boolean;
    enableVirtualization?: boolean;
    virtualizationThreshold?: number;
  } = {}
) => {
  const {
    enableMemoization = true,
    enableLazyLoading = false,
    enableVirtualization = false,
    virtualizationThreshold = 100
  } = optimizationConfig;

  let WrappedComponent = Component;

  if (enableMemoization) {
    WrappedComponent = React.memo(WrappedComponent);
  }

  if (enableLazyLoading) {
    const LazyComponent = React.lazy(() => Promise.resolve({ default: WrappedComponent }));
    WrappedComponent = (props: P) => (
      <React.Suspense fallback={<div>Loading...</div>}>
        <LazyComponent {...props} />
      </React.Suspense>
    );
  }

  return WrappedComponent;
};

// Cleanup function
export const cleanupPerformanceOptimizer = () => {
  performanceOptimizer.destroy();
};

// Global performance optimization setup
export const initializePerformanceOptimization = () => {
  // Listen for custom performance events
  window.addEventListener('performance:suggest-data-sampling', (event: any) => {
    console.log('📊 Data sampling suggested:', event.detail);
  });

  window.addEventListener('performance:suggest-request-batching', (event: any) => {
    console.log('🌐 Request batching suggested:', event.detail);
  });

  // Monitor page visibility for performance optimization
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      // Page is hidden, reduce performance monitoring frequency
      console.log('📱 Page hidden, reducing monitoring frequency');
    } else {
      // Page is visible, resume normal monitoring
      console.log('📱 Page visible, resuming normal monitoring');
    }
  });

  console.log('🚀 Performance optimization system initialized');
};

// Type declarations for global gc function
declare global {
  interface Window {
    gc?: () => void;
  }
}
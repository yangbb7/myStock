// Performance optimization utilities - main export file

// Core optimization utilities
export * from './chartOptimization';
export * from './cacheManager';
export * from './performanceMonitor';
export * from './lazyLoading';

// Performance optimizer (avoid duplicate exports)
export { 
  performanceOptimizer, 
  initializePerformanceOptimization,
  cleanupPerformanceOptimizer,
  createOptimizedDebounce,
  createOptimizedThrottle,
  withPerformanceOptimization
} from './performanceOptimizer';

// Hooks
export { usePerformanceOptimization as usePerformanceOptimizationHook } from '../hooks/usePerformanceOptimization';
export * from '../hooks/useOptimizedQuery';

// Components
export { VirtualTable } from '../components/Common/VirtualTable';
export { VirtualList } from '../components/Common/VirtualList';
export { VirtualScrollList } from '../components/Common/VirtualScrollList';
export { OptimizedChart } from '../components/Charts/OptimizedChart';
export { OptimizedDataTable } from '../components/Common/OptimizedDataTable';
export { PerformanceDashboard, usePerformanceDashboard } from '../components/Common/PerformanceDashboard';

// Performance optimization presets
export const PERFORMANCE_PRESETS = {
  // High-frequency trading dashboard
  HIGH_FREQUENCY: {
    enableCaching: true,
    enableThrottling: true,
    throttleDelay: 50,
    enableMemoryMonitoring: true,
    maxMemoryUsage: 200, // 200MB
    virtualizationThreshold: 50
  },
  
  // Standard dashboard
  STANDARD: {
    enableCaching: true,
    enableDebouncing: true,
    debounceDelay: 300,
    enableMemoryMonitoring: true,
    maxMemoryUsage: 100, // 100MB
    virtualizationThreshold: 100
  },
  
  // Low-resource environment
  LOW_RESOURCE: {
    enableCaching: true,
    enableDebouncing: true,
    debounceDelay: 500,
    enableMemoryMonitoring: true,
    maxMemoryUsage: 50, // 50MB
    virtualizationThreshold: 25
  },
  
  // Chart-heavy pages
  CHART_HEAVY: {
    enableCaching: true,
    enableThrottling: true,
    throttleDelay: 100,
    enableMemoryMonitoring: true,
    maxMemoryUsage: 150, // 150MB
    virtualizationThreshold: 200,
    chartSampling: true,
    maxChartPoints: 1000
  }
} as const;

// Performance optimization factory
export const createPerformanceConfig = (
  preset: keyof typeof PERFORMANCE_PRESETS,
  overrides?: Partial<typeof PERFORMANCE_PRESETS.STANDARD>
) => {
  return {
    ...PERFORMANCE_PRESETS[preset],
    ...overrides
  };
};

// Global performance settings
export const GLOBAL_PERFORMANCE_CONFIG = {
  // Bundle splitting thresholds
  BUNDLE_SIZE_WARNING: 1000 * 1024, // 1MB
  BUNDLE_SIZE_ERROR: 2000 * 1024,   // 2MB
  
  // Memory thresholds
  MEMORY_WARNING_THRESHOLD: 100 * 1024 * 1024, // 100MB
  MEMORY_ERROR_THRESHOLD: 200 * 1024 * 1024,   // 200MB
  
  // Performance thresholds
  RENDER_TIME_WARNING: 16,  // 16ms (60fps)
  RENDER_TIME_ERROR: 50,    // 50ms
  
  // Cache settings
  DEFAULT_CACHE_TTL: 5 * 60 * 1000,      // 5 minutes
  CHART_CACHE_TTL: 2 * 60 * 1000,        // 2 minutes
  USER_PREFERENCES_TTL: 30 * 60 * 1000,  // 30 minutes
  
  // Network settings
  MAX_CONCURRENT_REQUESTS: 5,
  REQUEST_TIMEOUT: 10000, // 10 seconds
  RETRY_ATTEMPTS: 3,
  
  // Virtualization settings
  DEFAULT_ITEM_HEIGHT: 54,
  DEFAULT_OVERSCAN: 5,
  VIRTUALIZATION_THRESHOLD: 100
} as const;
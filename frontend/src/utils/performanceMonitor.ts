// Performance monitoring and optimization utilities

export interface PerformanceMetrics {
  renderTime: number;
  componentCount: number;
  memoryUsage?: {
    used: number;
    total: number;
    limit: number;
    usagePercentage: number;
  };
  networkRequests: number;
  cacheHitRate: number;
  timestamp: number;
}

export class PerformanceMonitor {
  private static instance: PerformanceMonitor;
  private metrics: PerformanceMetrics[] = [];
  private observers: PerformanceObserver[] = [];
  private renderStartTimes = new Map<string, number>();
  private networkRequestCount = 0;
  private cacheHits = 0;
  private cacheMisses = 0;

  static getInstance(): PerformanceMonitor {
    if (!PerformanceMonitor.instance) {
      PerformanceMonitor.instance = new PerformanceMonitor();
    }
    return PerformanceMonitor.instance;
  }

  constructor() {
    this.initializeObservers();
  }

  private initializeObservers() {
    if (typeof window === 'undefined' || !window.PerformanceObserver) {
      return;
    }

    // Observe navigation timing
    try {
      const navObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        entries.forEach((entry) => {
          if (entry.entryType === 'navigation') {
            this.recordNavigationMetrics(entry as PerformanceNavigationTiming);
          }
        });
      });
      navObserver.observe({ entryTypes: ['navigation'] });
      this.observers.push(navObserver);
    } catch (error) {
      console.warn('Navigation timing observer not supported:', error);
    }

    // Observe resource timing
    try {
      const resourceObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        entries.forEach((entry) => {
          if (entry.entryType === 'resource') {
            this.recordResourceMetrics(entry as PerformanceResourceTiming);
          }
        });
      });
      resourceObserver.observe({ entryTypes: ['resource'] });
      this.observers.push(resourceObserver);
    } catch (error) {
      console.warn('Resource timing observer not supported:', error);
    }

    // Observe long tasks
    try {
      const longTaskObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        entries.forEach((entry) => {
          if (entry.duration > 50) { // Tasks longer than 50ms
            console.warn('Long task detected:', {
              duration: entry.duration,
              startTime: entry.startTime,
              name: entry.name
            });
          }
        });
      });
      longTaskObserver.observe({ entryTypes: ['longtask'] });
      this.observers.push(longTaskObserver);
    } catch (error) {
      console.warn('Long task observer not supported:', error);
    }
  }

  // Record component render start
  startRender(componentName: string): void {
    this.renderStartTimes.set(componentName, performance.now());
  }

  // Record component render end
  endRender(componentName: string): number {
    const startTime = this.renderStartTimes.get(componentName);
    if (startTime) {
      const renderTime = performance.now() - startTime;
      this.renderStartTimes.delete(componentName);
      return renderTime;
    }
    return 0;
  }

  // Record network request
  recordNetworkRequest(): void {
    this.networkRequestCount++;
  }

  // Record cache hit/miss
  recordCacheHit(): void {
    this.cacheHits++;
  }

  recordCacheMiss(): void {
    this.cacheMisses++;
  }

  // Get current performance metrics
  getCurrentMetrics(): PerformanceMetrics {
    const memoryUsage = this.getMemoryUsage();
    const cacheHitRate = this.cacheHits + this.cacheMisses > 0 
      ? this.cacheHits / (this.cacheHits + this.cacheMisses) 
      : 0;

    return {
      renderTime: this.getAverageRenderTime(),
      componentCount: this.renderStartTimes.size,
      memoryUsage,
      networkRequests: this.networkRequestCount,
      cacheHitRate,
      timestamp: Date.now()
    };
  }

  // Get memory usage information
  private getMemoryUsage() {
    if (typeof window !== 'undefined' && 'performance' in window && 'memory' in (window.performance as any)) {
      const memory = (window.performance as any).memory;
      return {
        used: memory.usedJSHeapSize,
        total: memory.totalJSHeapSize,
        limit: memory.jsHeapSizeLimit,
        usagePercentage: (memory.usedJSHeapSize / memory.jsHeapSizeLimit) * 100
      };
    }
    return undefined;
  }

  // Get average render time
  private getAverageRenderTime(): number {
    if (this.metrics.length === 0) return 0;
    const totalRenderTime = this.metrics.reduce((sum, metric) => sum + metric.renderTime, 0);
    return totalRenderTime / this.metrics.length;
  }

  // Record navigation metrics
  private recordNavigationMetrics(entry: PerformanceNavigationTiming) {
    console.log('Navigation metrics:', {
      domContentLoaded: entry.domContentLoadedEventEnd - entry.domContentLoadedEventStart,
      loadComplete: entry.loadEventEnd - entry.loadEventStart,
      totalTime: entry.loadEventEnd - entry.fetchStart
    });
  }

  // Record resource metrics
  private recordResourceMetrics(entry: PerformanceResourceTiming) {
    if (entry.name.includes('/api/')) {
      this.recordNetworkRequest();
    }
  }

  // Get performance report
  getPerformanceReport() {
    const currentMetrics = this.getCurrentMetrics();
    const recentMetrics = this.metrics.slice(-10); // Last 10 measurements

    return {
      current: currentMetrics,
      history: recentMetrics,
      averages: {
        renderTime: recentMetrics.length > 0 
          ? recentMetrics.reduce((sum, m) => sum + m.renderTime, 0) / recentMetrics.length 
          : 0,
        networkRequests: recentMetrics.length > 0 
          ? recentMetrics.reduce((sum, m) => sum + m.networkRequests, 0) / recentMetrics.length 
          : 0,
        cacheHitRate: recentMetrics.length > 0 
          ? recentMetrics.reduce((sum, m) => sum + m.cacheHitRate, 0) / recentMetrics.length 
          : 0
      },
      recommendations: this.getOptimizationRecommendations(currentMetrics)
    };
  }

  // Get optimization recommendations
  private getOptimizationRecommendations(metrics: PerformanceMetrics): string[] {
    const recommendations: string[] = [];

    if (metrics.renderTime > 16) { // 60fps = 16.67ms per frame
      recommendations.push('考虑优化组件渲染性能，使用 React.memo 或 useMemo');
    }

    if (metrics.memoryUsage && metrics.memoryUsage.usagePercentage > 80) {
      recommendations.push('内存使用率过高，考虑清理缓存或优化数据结构');
    }

    if (metrics.cacheHitRate < 0.7) {
      recommendations.push('缓存命中率较低，考虑调整缓存策略');
    }

    if (metrics.networkRequests > 10) {
      recommendations.push('网络请求过多，考虑使用批量请求或缓存');
    }

    return recommendations;
  }

  // Start continuous monitoring
  startMonitoring(interval: number = 5000): void {
    setInterval(() => {
      const metrics = this.getCurrentMetrics();
      this.metrics.push(metrics);
      
      // Keep only last 100 measurements
      if (this.metrics.length > 100) {
        this.metrics.shift();
      }
    }, interval);
  }

  // Cleanup
  destroy(): void {
    this.observers.forEach(observer => observer.disconnect());
    this.observers = [];
    this.metrics = [];
    this.renderStartTimes.clear();
  }
}

import React from 'react';

// React hook for performance monitoring
export const usePerformanceMonitor = (componentName: string) => {
  const monitor = PerformanceMonitor.getInstance();

  React.useEffect(() => {
    monitor.startRender(componentName);
    return () => {
      const renderTime = monitor.endRender(componentName);
      if (renderTime > 16) {
        console.warn(`Slow render detected in ${componentName}: ${renderTime.toFixed(2)}ms`);
      }
    };
  });

  return {
    recordCacheHit: () => monitor.recordCacheHit(),
    recordCacheMiss: () => monitor.recordCacheMiss(),
    recordNetworkRequest: () => monitor.recordNetworkRequest(),
    getMetrics: () => monitor.getCurrentMetrics(),
    getReport: () => monitor.getPerformanceReport()
  };
};

// Performance measurement decorator
export const measurePerformance = <T extends (...args: any[]) => any>(
  fn: T,
  name: string
): T => {
  return ((...args: any[]) => {
    const start = performance.now();
    const result = fn(...args);
    const end = performance.now();
    
    if (end - start > 10) { // Log if function takes more than 10ms
      console.log(`Performance: ${name} took ${(end - start).toFixed(2)}ms`);
    }
    
    return result;
  }) as T;
};

// Bundle size analyzer
export const analyzeBundleSize = () => {
  if (typeof window === 'undefined') return;

  const scripts = Array.from(document.querySelectorAll('script[src]'));
  const styles = Array.from(document.querySelectorAll('link[rel="stylesheet"]'));
  
  const bundleInfo = {
    scripts: scripts.map(script => ({
      src: (script as HTMLScriptElement).src,
      async: (script as HTMLScriptElement).async,
      defer: (script as HTMLScriptElement).defer
    })),
    styles: styles.map(style => ({
      href: (style as HTMLLinkElement).href
    })),
    totalScripts: scripts.length,
    totalStyles: styles.length
  };

  console.log('Bundle analysis:', bundleInfo);
  return bundleInfo;
};

// Global performance monitor instance
export const performanceMonitor = PerformanceMonitor.getInstance();
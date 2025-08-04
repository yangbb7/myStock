// Advanced caching and memory management utilities

export interface CacheItem<T> {
  data: T;
  timestamp: number;
  ttl: number;
  accessCount: number;
  lastAccessed: number;
}

export interface CacheOptions {
  maxSize?: number;
  defaultTTL?: number;
  cleanupInterval?: number;
  enableLRU?: boolean;
}

// LRU Cache implementation with TTL support
export class LRUCache<T> {
  private cache = new Map<string, CacheItem<T>>();
  private maxSize: number;
  private defaultTTL: number;
  private cleanupInterval: number;
  private cleanupTimer: NodeJS.Timeout | null = null;
  private enableLRU: boolean;

  constructor(options: CacheOptions = {}) {
    this.maxSize = options.maxSize || 100;
    this.defaultTTL = options.defaultTTL || 5 * 60 * 1000; // 5 minutes
    this.cleanupInterval = options.cleanupInterval || 60 * 1000; // 1 minute
    this.enableLRU = options.enableLRU !== false;

    this.startCleanup();
  }

  set(key: string, value: T, ttl?: number): void {
    const now = Date.now();
    const item: CacheItem<T> = {
      data: value,
      timestamp: now,
      ttl: ttl || this.defaultTTL,
      accessCount: 0,
      lastAccessed: now
    };

    // Remove oldest item if cache is full
    if (this.cache.size >= this.maxSize && !this.cache.has(key)) {
      this.evictLRU();
    }

    this.cache.set(key, item);
  }

  get(key: string): T | null {
    const item = this.cache.get(key);
    if (!item) return null;

    const now = Date.now();
    
    // Check if item has expired
    if (now - item.timestamp > item.ttl) {
      this.cache.delete(key);
      return null;
    }

    // Update access statistics
    item.accessCount++;
    item.lastAccessed = now;

    return item.data;
  }

  has(key: string): boolean {
    return this.get(key) !== null;
  }

  delete(key: string): boolean {
    return this.cache.delete(key);
  }

  clear(): void {
    this.cache.clear();
  }

  size(): number {
    return this.cache.size;
  }

  protected getCacheEntries(): IterableIterator<[string, CacheItem<T>]> {
    return this.cache.entries();
  }

  // Get cache statistics
  getStats() {
    const items = Array.from(this.cache.values());
    return {
      size: this.cache.size,
      maxSize: this.maxSize,
      totalAccesses: items.reduce((sum, item) => sum + item.accessCount, 0),
      averageAge: items.length > 0 
        ? items.reduce((sum, item) => sum + (Date.now() - item.timestamp), 0) / items.length
        : 0
    };
  }

  protected evictLRU(): void {
    if (!this.enableLRU || this.cache.size === 0) return;

    let oldestKey = '';
    let oldestTime = Date.now();

    for (const [key, item] of this.cache.entries()) {
      if (item.lastAccessed < oldestTime) {
        oldestTime = item.lastAccessed;
        oldestKey = key;
      }
    }

    if (oldestKey) {
      this.cache.delete(oldestKey);
    }
  }

  private cleanup(): void {
    const now = Date.now();
    const keysToDelete: string[] = [];

    for (const [key, item] of this.cache.entries()) {
      if (now - item.timestamp > item.ttl) {
        keysToDelete.push(key);
      }
    }

    keysToDelete.forEach(key => this.cache.delete(key));
  }

  private startCleanup(): void {
    this.cleanupTimer = setInterval(() => {
      this.cleanup();
    }, this.cleanupInterval);
  }

  destroy(): void {
    if (this.cleanupTimer) {
      clearInterval(this.cleanupTimer);
      this.cleanupTimer = null;
    }
    this.clear();
  }
}

// Memory-aware cache manager
export class MemoryAwareCache<T> extends LRUCache<T> {
  private memoryThreshold: number;
  private memoryCheckInterval: number;
  private memoryTimer: NodeJS.Timeout | null = null;

  constructor(options: CacheOptions & { memoryThreshold?: number; memoryCheckInterval?: number } = {}) {
    super(options);
    this.memoryThreshold = options.memoryThreshold || 50 * 1024 * 1024; // 50MB
    this.memoryCheckInterval = options.memoryCheckInterval || 30 * 1000; // 30 seconds

    this.startMemoryMonitoring();
  }

  private getMemoryUsage(): number {
    if (typeof window !== 'undefined' && 'performance' in window && 'memory' in (window.performance as any)) {
      return (window.performance as any).memory.usedJSHeapSize;
    }
    return 0;
  }

  private startMemoryMonitoring(): void {
    this.memoryTimer = setInterval(() => {
      const memoryUsage = this.getMemoryUsage();
      if (memoryUsage > this.memoryThreshold) {
        // Aggressively clean cache when memory usage is high
        const targetSize = Math.floor(this.size() * 0.5);
        while (this.size() > targetSize) {
          this.evictLRU();
        }
      }
    }, this.memoryCheckInterval);
  }

  destroy(): void {
    super.destroy();
    if (this.memoryTimer) {
      clearInterval(this.memoryTimer);
      this.memoryTimer = null;
    }
  }

  protected evictLRU(): void {
    if (this.size() === 0) return;

    let lruKey = '';
    let lruScore = Infinity;

    for (const [key, item] of this.getCacheEntries()) {
      // Score based on access frequency and recency
      const score = item.accessCount / (Date.now() - item.lastAccessed + 1);
      if (score < lruScore) {
        lruScore = score;
        lruKey = key;
      }
    }

    if (lruKey) {
      this.delete(lruKey);
    }
  }
}

// Tiered cache system for different data types
export const apiCache = new MemoryAwareCache<any>({
  maxSize: 300,
  defaultTTL: 5 * 60 * 1000, // 5 minutes
  memoryThreshold: 50 * 1024 * 1024, // 50MB
  cleanupInterval: 30 * 1000 // 30 seconds
});

export const chartDataCache = new MemoryAwareCache<any>({
  maxSize: 100,
  defaultTTL: 2 * 60 * 1000, // 2 minutes
  memoryThreshold: 30 * 1024 * 1024, // 30MB
  cleanupInterval: 15 * 1000 // 15 seconds
});

export const realTimeDataCache = new MemoryAwareCache<any>({
  maxSize: 500,
  defaultTTL: 30 * 1000, // 30 seconds for real-time data
  memoryThreshold: 20 * 1024 * 1024, // 20MB
  cleanupInterval: 10 * 1000 // 10 seconds
});

export const userPreferencesCache = new LRUCache<any>({
  maxSize: 50,
  defaultTTL: 30 * 60 * 1000, // 30 minutes
  cleanupInterval: 5 * 60 * 1000 // 5 minutes
});

export const staticDataCache = new LRUCache<any>({
  maxSize: 100,
  defaultTTL: 60 * 60 * 1000, // 1 hour for static data
  cleanupInterval: 10 * 60 * 1000 // 10 minutes
});

// Cache key generators
export const generateCacheKey = (...parts: (string | number | boolean)[]): string => {
  return parts.map(part => String(part)).join(':');
};

export const generateApiCacheKey = (endpoint: string, params?: Record<string, any>): string => {
  const paramString = params ? JSON.stringify(params) : '';
  return generateCacheKey('api', endpoint, paramString);
};

// Cache decorators for functions
export const withCache = <T extends (...args: any[]) => any>(
  fn: T,
  cache: LRUCache<ReturnType<T>>,
  keyGenerator: (...args: Parameters<T>) => string,
  ttl?: number
): T => {
  return ((...args: Parameters<T>) => {
    const key = keyGenerator(...args);
    const cached = cache.get(key);
    
    if (cached !== null) {
      return cached;
    }

    const result = fn(...args);
    cache.set(key, result, ttl);
    return result;
  }) as T;
};

// Memory monitoring utilities
export const getMemoryInfo = () => {
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
};

export const formatBytes = (bytes: number): string => {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

// Advanced cache management
export class CacheManager {
  private static instance: CacheManager;
  private caches: Map<string, LRUCache<any> | MemoryAwareCache<any>> = new Map();
  private memoryMonitorInterval: NodeJS.Timeout | null = null;

  static getInstance(): CacheManager {
    if (!CacheManager.instance) {
      CacheManager.instance = new CacheManager();
    }
    return CacheManager.instance;
  }

  constructor() {
    this.registerCache('api', apiCache);
    this.registerCache('chartData', chartDataCache);
    this.registerCache('realTimeData', realTimeDataCache);
    this.registerCache('userPreferences', userPreferencesCache);
    this.registerCache('staticData', staticDataCache);
    
    this.startMemoryMonitoring();
  }

  registerCache(name: string, cache: LRUCache<any> | MemoryAwareCache<any>) {
    this.caches.set(name, cache);
  }

  getCache(name: string) {
    return this.caches.get(name);
  }

  // Get global cache statistics
  getGlobalStats() {
    const stats: Record<string, any> = {};
    let totalSize = 0;
    let totalAccesses = 0;

    this.caches.forEach((cache, name) => {
      const cacheStats = cache.getStats();
      stats[name] = cacheStats;
      totalSize += cacheStats.size;
      totalAccesses += cacheStats.totalAccesses;
    });

    return {
      caches: stats,
      totalSize,
      totalAccesses,
      memoryInfo: getMemoryInfo()
    };
  }

  // Intelligent cache cleanup based on memory pressure
  performIntelligentCleanup() {
    const memoryInfo = getMemoryInfo();
    if (!memoryInfo) return;

    const memoryPressure = memoryInfo.usagePercentage;
    
    if (memoryPressure > 80) {
      // High memory pressure - aggressive cleanup
      this.caches.forEach((cache, name) => {
        if (name === 'realTimeData') {
          cache.clear(); // Clear all real-time data
        } else if (name === 'chartData') {
          // Keep only 25% of chart data
          const stats = cache.getStats();
          const targetSize = Math.floor(stats.size * 0.25);
          while (cache.size() > targetSize) {
            (cache as any).evictLRU?.();
          }
        }
      });
    } else if (memoryPressure > 60) {
      // Medium memory pressure - moderate cleanup
      this.caches.forEach((cache, name) => {
        if (name === 'realTimeData') {
          // Keep only recent real-time data
          const stats = cache.getStats();
          const targetSize = Math.floor(stats.size * 0.5);
          while (cache.size() > targetSize) {
            (cache as any).evictLRU?.();
          }
        }
      });
    }
  }

  private startMemoryMonitoring() {
    this.memoryMonitorInterval = setInterval(() => {
      this.performIntelligentCleanup();
    }, 30000); // Check every 30 seconds
  }

  // Cleanup all caches
  cleanup() {
    this.caches.forEach(cache => {
      cache.destroy();
    });
    this.caches.clear();
    
    if (this.memoryMonitorInterval) {
      clearInterval(this.memoryMonitorInterval);
      this.memoryMonitorInterval = null;
    }
  }
}

// Global cache manager instance
export const cacheManager = CacheManager.getInstance();

// Cleanup function for application shutdown
export const cleanupCaches = () => {
  cacheManager.cleanup();
};
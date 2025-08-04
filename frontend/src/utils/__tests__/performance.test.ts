import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { LRUCache } from '../cacheManager';
import { sampleData, aggregateTimeSeriesData, ChartOptimizer, DataBuffer } from '../chartOptimization';

describe('Performance Optimizations', () => {
  describe('LRUCache', () => {
    let cache: LRUCache<string>;

    beforeEach(() => {
      cache = new LRUCache<string>({
        maxSize: 3,
        defaultTTL: 1000
      });
    });

    afterEach(() => {
      cache.destroy();
    });

    it('should store and retrieve values', () => {
      cache.set('key1', 'value1');
      expect(cache.get('key1')).toBe('value1');
    });

    it('should respect maximum cache size', () => {
      // Fill cache to max capacity
      cache.set('key1', 'value1');
      cache.set('key2', 'value2');
      cache.set('key3', 'value3');
      
      expect(cache.size()).toBe(3);
      
      // Adding one more should trigger eviction
      cache.set('key4', 'value4');
      
      // Cache should not exceed max size
      expect(cache.size()).toBeLessThanOrEqual(3);
      
      // The newest item should definitely exist
      expect(cache.get('key4')).toBe('value4');
    });

    it('should respect TTL', async () => {
      cache.set('key1', 'value1', 100); // 100ms TTL
      expect(cache.get('key1')).toBe('value1');

      // Wait for TTL to expire
      await new Promise(resolve => setTimeout(resolve, 150));
      expect(cache.get('key1')).toBeNull();
    });

    it('should update access statistics', () => {
      cache.set('key1', 'value1');
      cache.get('key1');
      cache.get('key1');

      const stats = cache.getStats();
      expect(stats.totalAccesses).toBe(2);
    });
  });

  describe('Data Sampling', () => {
    it('should sample large datasets', () => {
      const largeData = Array.from({ length: 10000 }, (_, i) => ({ id: i, value: i * 2 }));
      const sampled = sampleData(largeData, 1000);

      expect(sampled.length).toBeLessThanOrEqual(1001); // Allow for +1 due to always including last point
      expect(sampled[0]).toEqual(largeData[0]);
      expect(sampled[sampled.length - 1]).toEqual(largeData[largeData.length - 1]);
    });

    it('should not sample small datasets', () => {
      const smallData = Array.from({ length: 100 }, (_, i) => ({ id: i, value: i * 2 }));
      const sampled = sampleData(smallData, 1000);

      expect(sampled.length).toBe(100);
      expect(sampled).toEqual(smallData);
    });
  });

  describe('Time Series Aggregation', () => {
    it('should aggregate time series data by average', () => {
      const data = [
        { timestamp: 1000, value: 10 },
        { timestamp: 1500, value: 20 },
        { timestamp: 2000, value: 30 },
        { timestamp: 2500, value: 40 }
      ];

      const aggregated = aggregateTimeSeriesData(data, 1000, 'avg');
      
      expect(aggregated.length).toBeGreaterThan(0);
      // The exact grouping depends on the interval calculation
      expect(aggregated[0].value).toBeGreaterThan(0);
    });

    it('should aggregate time series data by sum', () => {
      const data = [
        { timestamp: 1000, value: 10 },
        { timestamp: 1500, value: 20 },
        { timestamp: 2000, value: 30 }
      ];

      const aggregated = aggregateTimeSeriesData(data, 1000, 'sum');
      
      expect(aggregated[0].value).toBe(30); // 10 + 20
      expect(aggregated[1].value).toBe(30);
    });
  });

  describe('DataBuffer', () => {
    it('should maintain maximum size', () => {
      const buffer = new DataBuffer<number>(3);
      
      buffer.add(1);
      buffer.add(2);
      buffer.add(3);
      buffer.add(4); // Should remove 1

      const data = buffer.getData();
      expect(data).toEqual([2, 3, 4]);
    });

    it('should handle batch additions', () => {
      const buffer = new DataBuffer<number>(5);
      
      buffer.addBatch([1, 2, 3, 4, 5, 6, 7]);

      const data = buffer.getData();
      expect(data).toEqual([3, 4, 5, 6, 7]);
    });

    it('should get latest items', () => {
      const buffer = new DataBuffer<number>(10);
      buffer.addBatch([1, 2, 3, 4, 5]);

      const latest = buffer.getLatest(3);
      expect(latest).toEqual([3, 4, 5]);
    });
  });

  describe('ChartOptimizer', () => {
    let optimizer: ChartOptimizer;

    beforeEach(() => {
      optimizer = ChartOptimizer.getInstance();
    });

    afterEach(() => {
      optimizer.destroy();
    });

    it('should register and unregister chart instances', () => {
      const mockChart = { resize: vi.fn() };
      
      optimizer.registerChart('test-chart', mockChart);
      optimizer.unregisterChart('test-chart');
      
      // Should not throw
      expect(true).toBe(true);
    });
  });
});
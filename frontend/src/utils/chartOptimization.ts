import { debounce, throttle } from 'lodash';

// Chart performance optimization utilities
export class ChartOptimizer {
  private static instance: ChartOptimizer;
  private resizeObserver: ResizeObserver | null = null;
  private chartInstances = new Map<string, any>();

  static getInstance(): ChartOptimizer {
    if (!ChartOptimizer.instance) {
      ChartOptimizer.instance = new ChartOptimizer();
    }
    return ChartOptimizer.instance;
  }

  // Register chart instance for optimization
  registerChart(id: string, chartInstance: any) {
    this.chartInstances.set(id, chartInstance);
  }

  // Unregister chart instance
  unregisterChart(id: string) {
    this.chartInstances.delete(id);
  }

  // Optimized resize handler
  handleResize = debounce((entries: ResizeObserverEntry[]) => {
    entries.forEach((entry) => {
      const chartId = entry.target.getAttribute('data-chart-id');
      if (chartId && this.chartInstances.has(chartId)) {
        const chart = this.chartInstances.get(chartId);
        if (chart && chart.resize) {
          chart.resize();
        }
      }
    });
  }, 100);

  // Initialize resize observer
  initResizeObserver() {
    if (!this.resizeObserver) {
      this.resizeObserver = new ResizeObserver(this.handleResize);
    }
    return this.resizeObserver;
  }

  // Cleanup
  destroy() {
    if (this.resizeObserver) {
      this.resizeObserver.disconnect();
      this.resizeObserver = null;
    }
    this.chartInstances.clear();
  }
}

// Data sampling for large datasets
export const sampleData = <T>(data: T[], maxPoints: number = 1000): T[] => {
  if (data.length <= maxPoints) {
    return data;
  }

  const step = Math.ceil(data.length / maxPoints);
  const sampled: T[] = [];
  
  for (let i = 0; i < data.length; i += step) {
    sampled.push(data[i]);
  }
  
  // Always include the last point
  if (sampled[sampled.length - 1] !== data[data.length - 1]) {
    sampled.push(data[data.length - 1]);
  }
  
  return sampled;
};

// Adaptive sampling based on zoom level
export const adaptiveSample = <T>(
  data: T[], 
  zoomLevel: number = 1, 
  baseMaxPoints: number = 1000
): T[] => {
  const maxPoints = Math.floor(baseMaxPoints * zoomLevel);
  return sampleData(data, maxPoints);
};

// Data aggregation for time series
export interface TimeSeriesPoint {
  timestamp: number;
  value: number;
  [key: string]: any;
}

export const aggregateTimeSeriesData = (
  data: TimeSeriesPoint[],
  interval: number, // in milliseconds
  aggregationType: 'avg' | 'sum' | 'max' | 'min' | 'last' = 'avg'
): TimeSeriesPoint[] => {
  if (data.length === 0) return [];

  const aggregated: TimeSeriesPoint[] = [];
  const groups = new Map<number, TimeSeriesPoint[]>();

  // Group data by time intervals
  data.forEach(point => {
    const intervalKey = Math.floor(point.timestamp / interval) * interval;
    if (!groups.has(intervalKey)) {
      groups.set(intervalKey, []);
    }
    groups.get(intervalKey)!.push(point);
  });

  // Aggregate each group
  groups.forEach((points, timestamp) => {
    let aggregatedValue: number;
    
    switch (aggregationType) {
      case 'sum':
        aggregatedValue = points.reduce((sum, p) => sum + p.value, 0);
        break;
      case 'max':
        aggregatedValue = Math.max(...points.map(p => p.value));
        break;
      case 'min':
        aggregatedValue = Math.min(...points.map(p => p.value));
        break;
      case 'last':
        aggregatedValue = points[points.length - 1].value;
        break;
      case 'avg':
      default:
        aggregatedValue = points.reduce((sum, p) => sum + p.value, 0) / points.length;
        break;
    }

    const result: TimeSeriesPoint = {
      timestamp,
      value: aggregatedValue,
      count: points.length
    };
    
    // Include other properties from first point (excluding timestamp and value)
    const { timestamp: _, value: __, ...otherProps } = points[0];
    aggregated.push({ ...result, ...otherProps });
  });

  return aggregated.sort((a, b) => a.timestamp - b.timestamp);
};

// Throttled update function for real-time charts
export const createThrottledUpdate = (
  updateFn: (data: any) => void,
  delay: number = 100
) => {
  return throttle(updateFn, delay, { leading: true, trailing: true });
};

// Memory-efficient data buffer for streaming data
export class DataBuffer<T> {
  private buffer: T[] = [];
  private maxSize: number;

  constructor(maxSize: number = 10000) {
    this.maxSize = maxSize;
  }

  add(item: T): void {
    this.buffer.push(item);
    if (this.buffer.length > this.maxSize) {
      this.buffer.shift();
    }
  }

  addBatch(items: T[]): void {
    this.buffer.push(...items);
    if (this.buffer.length > this.maxSize) {
      this.buffer.splice(0, this.buffer.length - this.maxSize);
    }
  }

  getData(): T[] {
    return [...this.buffer];
  }

  getLatest(count: number): T[] {
    return this.buffer.slice(-count);
  }

  clear(): void {
    this.buffer = [];
  }

  size(): number {
    return this.buffer.length;
  }
}

// Chart configuration optimization
export const getOptimizedChartConfig = (dataLength: number, chartType: 'line' | 'candlestick' | 'bar' | 'scatter' = 'line') => {
  const isLargeDataset = dataLength > 5000;
  const isVeryLargeDataset = dataLength > 20000;
  
  const baseConfig = {
    animation: !isLargeDataset,
    progressive: isLargeDataset ? Math.min(dataLength / 10, 1000) : 0,
    progressiveThreshold: isLargeDataset ? 3000 : 0,
    useUTC: true,
    lazyUpdate: true,
    silent: isLargeDataset,
  };

  // Chart-specific optimizations
  const chartSpecificConfig = {
    line: {
      ...baseConfig,
      sampling: isVeryLargeDataset ? 'lttb' : undefined, // Largest-Triangle-Three-Buckets sampling
      large: isVeryLargeDataset,
      largeThreshold: 2000,
    },
    candlestick: {
      ...baseConfig,
      // Candlestick charts need all data points, so we use progressive rendering
      progressive: isLargeDataset ? Math.min(dataLength / 20, 500) : 0,
      progressiveThreshold: isLargeDataset ? 1000 : 0,
    },
    bar: {
      ...baseConfig,
      large: isVeryLargeDataset,
      largeThreshold: 1000,
    },
    scatter: {
      ...baseConfig,
      large: isVeryLargeDataset,
      largeThreshold: 5000,
      sampling: isVeryLargeDataset ? 'lttb' : undefined,
    }
  };

  return chartSpecificConfig[chartType] || baseConfig;
};

// Advanced data sampling strategies
export const SAMPLING_STRATEGIES = {
  // Largest-Triangle-Three-Buckets algorithm for line charts
  lttb: <T extends { x: number; y: number }>(data: T[], threshold: number): T[] => {
    if (data.length <= threshold) return data;
    if (threshold <= 2) return [data[0], data[data.length - 1]];

    const sampled: T[] = [];
    const bucketSize = (data.length - 2) / (threshold - 2);
    
    sampled.push(data[0]); // Always include first point

    for (let i = 0; i < threshold - 2; i++) {
      const avgRangeStart = Math.floor((i + 1) * bucketSize) + 1;
      const avgRangeEnd = Math.floor((i + 2) * bucketSize) + 1;
      const avgRangeLength = avgRangeEnd - avgRangeStart;

      let avgX = 0, avgY = 0;
      for (let j = avgRangeStart; j < avgRangeEnd; j++) {
        avgX += data[j].x;
        avgY += data[j].y;
      }
      avgX /= avgRangeLength;
      avgY /= avgRangeLength;

      const rangeOffs = Math.floor(i * bucketSize) + 1;
      const rangeTo = Math.floor((i + 1) * bucketSize) + 1;

      let maxArea = -1;
      let maxAreaPoint = data[rangeOffs];

      for (let j = rangeOffs; j < rangeTo; j++) {
        const area = Math.abs(
          (sampled[sampled.length - 1].x - avgX) * (data[j].y - sampled[sampled.length - 1].y) -
          (sampled[sampled.length - 1].x - data[j].x) * (avgY - sampled[sampled.length - 1].y)
        ) * 0.5;

        if (area > maxArea) {
          maxArea = area;
          maxAreaPoint = data[j];
        }
      }

      sampled.push(maxAreaPoint);
    }

    sampled.push(data[data.length - 1]); // Always include last point
    return sampled;
  },

  // Simple uniform sampling
  uniform: <T>(data: T[], maxPoints: number): T[] => {
    return sampleData(data, maxPoints);
  },

  // Peak-preserving sampling for financial data
  peakPreserving: <T extends { high?: number; low?: number; value?: number }>(
    data: T[], 
    maxPoints: number
  ): T[] => {
    if (data.length <= maxPoints) return data;

    const sampled: T[] = [];
    const step = Math.ceil(data.length / maxPoints);
    
    for (let i = 0; i < data.length; i += step) {
      const chunk = data.slice(i, Math.min(i + step, data.length));
      
      // Find peak (highest high or value) in chunk
      let peak = chunk[0];
      let valley = chunk[0];
      
      chunk.forEach(point => {
        const peakValue = point.high ?? point.value ?? 0;
        const valleyValue = point.low ?? point.value ?? 0;
        const currentPeakValue = peak.high ?? peak.value ?? 0;
        const currentValleyValue = valley.low ?? valley.value ?? 0;
        
        if (peakValue > currentPeakValue) peak = point;
        if (valleyValue < currentValleyValue) valley = point;
      });
      
      // Add both peak and valley if they're different
      if (peak !== valley) {
        sampled.push(peak, valley);
      } else {
        sampled.push(peak);
      }
    }
    
    return sampled.slice(0, maxPoints);
  }
};

// Smart sampling based on data characteristics
export const smartSample = <T extends Record<string, any>>(
  data: T[],
  maxPoints: number,
  dataType: 'timeseries' | 'financial' | 'scatter' = 'timeseries'
): T[] => {
  if (data.length <= maxPoints) return data;

  switch (dataType) {
    case 'financial':
      return SAMPLING_STRATEGIES.peakPreserving(data, maxPoints);
    case 'scatter':
      // For scatter plots, try to preserve outliers
      return SAMPLING_STRATEGIES.uniform(data, maxPoints);
    case 'timeseries':
    default:
      // Use LTTB for time series data if x,y structure is available
      if (data.length > 0 && 'x' in data[0] && 'y' in data[0]) {
        return SAMPLING_STRATEGIES.lttb(data as any, maxPoints) as T[];
      }
      return SAMPLING_STRATEGIES.uniform(data, maxPoints);
  }
};
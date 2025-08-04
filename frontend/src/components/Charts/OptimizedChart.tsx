import React, { useEffect, useRef, useCallback, useMemo, useState } from 'react';
import ReactECharts from 'echarts-for-react';
import { 
  ChartOptimizer, 
  sampleData, 
  getOptimizedChartConfig,
  createThrottledUpdate,
  DataBuffer,
  smartSample,
  SAMPLING_STRATEGIES
} from '../../utils/chartOptimization';
import { chartDataCache } from '../../utils/cacheManager';
import { performanceMonitor } from '../../utils/performanceMonitor';
import { Card, Tooltip, Button } from 'antd';
import { FullscreenOutlined, DownloadOutlined } from '@ant-design/icons';

interface OptimizedChartProps {
  data: any[];
  option: any;
  title?: string;
  height?: number;
  maxDataPoints?: number;
  enableSampling?: boolean;
  enableThrottling?: boolean;
  enableCaching?: boolean;
  enableRealTimeOptimization?: boolean;
  samplingStrategy?: 'uniform' | 'lttb' | 'peakPreserving' | 'smart';
  dataType?: 'timeseries' | 'financial' | 'scatter';
  throttleDelay?: number;
  onChartReady?: (chart: any) => void;
  onDataSampled?: (originalLength: number, sampledLength: number) => void;
  showPerformanceInfo?: boolean;
  enableExport?: boolean;
}

export const OptimizedChart: React.FC<OptimizedChartProps> = ({
  data,
  option,
  title,
  height = 400,
  maxDataPoints = 1000,
  enableSampling = true,
  enableThrottling = true,
  enableCaching = true,
  enableRealTimeOptimization = false,
  samplingStrategy = 'smart',
  dataType = 'timeseries',
  throttleDelay = 100,
  onChartReady,
  onDataSampled,
  showPerformanceInfo = false,
  enableExport = false
}) => {
  const chartRef = useRef<ReactECharts>(null);
  const chartId = useRef(`chart-${Date.now()}-${Math.random()}`);
  const dataBuffer = useRef(new DataBuffer(maxDataPoints * 2));
  const optimizer = ChartOptimizer.getInstance();
  const [renderCount, setRenderCount] = useState(0);
  const [performanceInfo, setPerformanceInfo] = useState<{
    renderTime: number;
    dataPoints: number;
    sampledPoints: number;
  } | null>(null);
  const lastRenderTime = useRef(Date.now());

  // Generate cache key
  const getCacheKey = useCallback(() => {
    const dataHash = data.length > 0 ? 
      `${data.length}-${JSON.stringify(data[0])}-${JSON.stringify(data[data.length - 1])}` : 
      'empty';
    return `chart-${dataHash}-${maxDataPoints}-${samplingStrategy}`;
  }, [data, maxDataPoints, samplingStrategy]);

  // Process data with advanced sampling
  const processedData = useMemo(() => {
    const startTime = performance.now();
    
    if (!enableSampling || data.length <= maxDataPoints) {
      return data;
    }

    // Try cache first
    const cacheKey = getCacheKey();
    if (enableCaching) {
      const cached = chartDataCache.get(cacheKey);
      if (cached) {
        return cached;
      }
    }

    let sampledData;
    switch (samplingStrategy) {
      case 'uniform':
        sampledData = SAMPLING_STRATEGIES.uniform(data, maxDataPoints);
        break;
      case 'lttb':
        sampledData = SAMPLING_STRATEGIES.lttb(data, maxDataPoints);
        break;
      case 'peakPreserving':
        sampledData = SAMPLING_STRATEGIES.peakPreserving(data, maxDataPoints);
        break;
      case 'smart':
      default:
        sampledData = smartSample(data, maxDataPoints, dataType);
        break;
    }

    // Cache the result
    if (enableCaching) {
      chartDataCache.set(cacheKey, sampledData, 2 * 60 * 1000); // 2 minutes
    }

    const processingTime = performance.now() - startTime;
    console.log(`Data sampling completed in ${processingTime.toFixed(2)}ms`);
    
    // Notify about sampling
    onDataSampled?.(data.length, sampledData.length);
    
    return sampledData;
  }, [data, maxDataPoints, enableSampling, samplingStrategy, dataType, enableCaching, getCacheKey, onDataSampled]);

  // Optimized chart option with performance monitoring
  const optimizedOption = useMemo(() => {
    const startTime = performance.now();
    
    // Determine chart type for optimization
    const chartType = option.series?.[0]?.type || 'line';
    const baseConfig = getOptimizedChartConfig(processedData.length, chartType);
    
    const result = {
      ...option,
      ...baseConfig,
      series: Array.isArray(option.series) 
        ? option.series.map((series: any) => ({
            ...series,
            ...baseConfig,
            data: series.data || processedData
          }))
        : {
            ...option.series,
            ...baseConfig,
            data: option.series?.data || processedData
          },
      // Add data zoom for large datasets
      ...(processedData.length > 5000 && {
        dataZoom: [
          {
            type: 'inside',
            start: 90,
            end: 100
          },
          {
            show: true,
            type: 'slider',
            start: 90,
            end: 100,
            height: 20
          }
        ]
      })
    };

    const processingTime = performance.now() - startTime;
    
    // Update performance info
    setPerformanceInfo({
      renderTime: processingTime,
      dataPoints: data.length,
      sampledPoints: processedData.length
    });

    // Record render metrics
    const now = Date.now();
    const timeSinceLastRender = now - lastRenderTime.current;
    lastRenderTime.current = now;
    setRenderCount(prev => prev + 1);
    
    performanceMonitor.recordRenderTime(timeSinceLastRender);
    
    return result;
  }, [option, processedData, data.length]);

  // Throttled update function with performance monitoring
  const throttledUpdate = useMemo(() => {
    if (!enableThrottling) return null;
    
    return createThrottledUpdate((newData: any) => {
      const chart = chartRef.current?.getEchartsInstance();
      if (chart) {
        const updateStart = performance.now();
        
        chart.setOption({
          ...optimizedOption,
          series: Array.isArray(optimizedOption.series)
            ? optimizedOption.series.map((series: any) => ({
                ...series,
                data: newData
              }))
            : {
                ...optimizedOption.series,
                data: newData
              }
        }, false, true);
        
        const updateTime = performance.now() - updateStart;
        performanceMonitor.recordRenderTime(updateTime);
      }
    }, throttleDelay);
  }, [optimizedOption, enableThrottling, throttleDelay]);

  // Handle chart ready with performance setup
  const handleChartReady = useCallback((chart: any) => {
    optimizer.registerChart(chartId.current, chart);
    
    // Setup real-time optimizations
    if (enableRealTimeOptimization) {
      chart.setOption({
        animation: false,
        animationDuration: 0
      });
    }
    
    // Setup resize observer
    const resizeObserver = optimizer.initResizeObserver();
    const chartElement = chartRef.current?.ele;
    if (chartElement && resizeObserver) {
      chartElement.setAttribute('data-chart-id', chartId.current);
      resizeObserver.observe(chartElement);
    }
    
    onChartReady?.(chart);
  }, [optimizer, onChartReady, enableRealTimeOptimization]);

  // Export chart functionality
  const handleExport = useCallback(() => {
    const chart = chartRef.current?.getEchartsInstance();
    if (chart) {
      const dataURL = chart.getDataURL({
        type: 'png',
        pixelRatio: 2,
        backgroundColor: '#fff'
      });
      
      const link = document.createElement('a');
      link.download = `chart-${chartId.current}.png`;
      link.href = dataURL;
      link.click();
    }
  }, []);

  // Fullscreen functionality
  const handleFullscreen = useCallback(() => {
    const chartElement = chartRef.current?.ele;
    if (chartElement) {
      if (chartElement.requestFullscreen) {
        chartElement.requestFullscreen();
      }
    }
  }, []);

  // Update data buffer and chart
  useEffect(() => {
    dataBuffer.current.addBatch(processedData);
    
    if (throttledUpdate && enableThrottling) {
      throttledUpdate(processedData);
    }
  }, [processedData, throttledUpdate, enableThrottling]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      optimizer.unregisterChart(chartId.current);
      dataBuffer.current.clear();
    };
  }, [optimizer]);

  // Performance monitoring effect
  useEffect(() => {
    if (showPerformanceInfo && process.env.NODE_ENV === 'development') {
      const interval = setInterval(() => {
        const metrics = performanceMonitor.getMetrics();
        console.log(`Chart ${chartId.current} performance:`, {
          renders: renderCount,
          avgRenderTime: metrics.averageRenderTime,
          dataPoints: performanceInfo?.dataPoints,
          sampledPoints: performanceInfo?.sampledPoints,
          memoryUsage: metrics.memoryUsage
        });
      }, 30000);

      return () => clearInterval(interval);
    }
  }, [showPerformanceInfo, renderCount, performanceInfo]);

  const cardExtra = (
    <div style={{ display: 'flex', gap: 8 }}>
      {showPerformanceInfo && performanceInfo && (
        <Tooltip title={`原始数据: ${performanceInfo.dataPoints} 点, 采样后: ${performanceInfo.sampledPoints} 点, 渲染时间: ${performanceInfo.renderTime.toFixed(2)}ms`}>
          <Button size="small" type="text">
            📊 {performanceInfo.sampledPoints}/{performanceInfo.dataPoints}
          </Button>
        </Tooltip>
      )}
      {enableExport && (
        <Tooltip title="导出图表">
          <Button size="small" icon={<DownloadOutlined />} onClick={handleExport} />
        </Tooltip>
      )}
      <Tooltip title="全屏显示">
        <Button size="small" icon={<FullscreenOutlined />} onClick={handleFullscreen} />
      </Tooltip>
    </div>
  );

  return (
    <Card 
      title={title} 
      extra={cardExtra}
      style={{ height: height + 100 }}
    >
      <div 
        data-chart-id={chartId.current}
        style={{ height }}
      >
        <ReactECharts
          ref={chartRef}
          option={optimizedOption}
          style={{ height: '100%', width: '100%' }}
          notMerge={true}
          lazyUpdate={true}
          onChartReady={handleChartReady}
          opts={{
            renderer: processedData.length > 10000 ? 'canvas' : 'svg', // Dynamic renderer selection
            useDirtyRect: true, // Enable dirty rectangle optimization
            useCoarsePointer: processedData.length > 5000, // Optimize for large datasets
          }}
        />
      </div>
    </Card>
  );
};
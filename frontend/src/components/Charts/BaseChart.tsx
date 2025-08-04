import React, { useRef, useEffect, useState } from 'react';
import * as echarts from 'echarts';
import { Card } from 'antd';
import { LoadingState } from '../Common/LoadingState';

export interface BaseChartProps {
  option: echarts.EChartsOption;
  loading?: boolean;
  error?: Error | string | null;
  height?: number | string;
  width?: number | string;
  theme?: string;
  notMerge?: boolean;
  lazyUpdate?: boolean;
  showLoading?: boolean;
  loadingOption?: object;
  onChartReady?: (chart: echarts.ECharts) => void;
  onEvents?: Record<string, (params: any) => void>;
  className?: string;
  style?: React.CSSProperties;
  title?: string;
  extra?: React.ReactNode;
  bordered?: boolean;
  size?: 'default' | 'small';
}

export const BaseChart: React.FC<BaseChartProps> = ({
  option,
  loading = false,
  error,
  height = 400,
  width = '100%',
  theme,
  notMerge = false,
  lazyUpdate = false,
  showLoading = false,
  loadingOption,
  onChartReady,
  onEvents = {},
  className,
  style,
  title,
  extra,
  bordered = true,
  size = 'default',
}) => {
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<echarts.ECharts | null>(null);
  const [isReady, setIsReady] = useState(false);

  // Initialize chart
  useEffect(() => {
    if (!chartRef.current) return;

    // Dispose existing chart
    if (chartInstance.current) {
      chartInstance.current.dispose();
    }

    // Create new chart instance
    chartInstance.current = echarts.init(chartRef.current, theme);
    setIsReady(true);

    // Register events
    Object.entries(onEvents).forEach(([eventName, handler]) => {
      chartInstance.current?.on(eventName, handler);
    });

    // Notify parent component
    if (onChartReady && chartInstance.current) {
      onChartReady(chartInstance.current);
    }

    // Handle resize
    const handleResize = () => {
      chartInstance.current?.resize();
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      if (chartInstance.current) {
        chartInstance.current.dispose();
        chartInstance.current = null;
      }
    };
  }, [theme, onChartReady]);

  // Update chart option
  useEffect(() => {
    if (!chartInstance.current || !isReady) return;

    chartInstance.current.setOption(option, notMerge, lazyUpdate);
  }, [option, notMerge, lazyUpdate, isReady]);

  // Handle loading state
  useEffect(() => {
    if (!chartInstance.current) return;

    if (showLoading || loading) {
      chartInstance.current.showLoading(loadingOption);
    } else {
      chartInstance.current.hideLoading();
    }
  }, [showLoading, loading, loadingOption]);

  // Handle resize when dimensions change
  useEffect(() => {
    const timer = setTimeout(() => {
      chartInstance.current?.resize();
    }, 100);

    return () => clearTimeout(timer);
  }, [height, width]);

  const chartContent = (
    <div
      ref={chartRef}
      style={{
        height: typeof height === 'number' ? `${height}px` : height,
        width: typeof width === 'number' ? `${width}px` : width,
        ...style,
      }}
      className={className}
    />
  );

  if (title || extra || bordered) {
    return (
      <LoadingState loading={loading && !isReady} error={error}>
        <Card
          title={title}
          extra={extra}
          bordered={bordered}
          size={size}
          bodyStyle={{ padding: 0 }}
        >
          {chartContent}
        </Card>
      </LoadingState>
    );
  }

  return (
    <LoadingState loading={loading && !isReady} error={error}>
      {chartContent}
    </LoadingState>
  );
};

export default BaseChart;
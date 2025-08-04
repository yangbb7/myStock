import React, { useState, useEffect, useRef, useMemo } from 'react';
import { BaseChart, BaseChartProps } from './BaseChart';
import dayjs from 'dayjs';

export interface RealTimeDataPoint {
  timestamp: string | number | Date;
  value: number;
  label?: string;
}

export interface RealTimeSeriesConfig {
  name: string;
  color?: string;
  type?: 'line' | 'bar' | 'scatter';
  smooth?: boolean;
  showSymbol?: boolean;
  lineWidth?: number;
  areaStyle?: boolean;
}

interface RealTimeChartProps extends Omit<BaseChartProps, 'option'> {
  data: RealTimeDataPoint[];
  series?: RealTimeSeriesConfig[];
  maxDataPoints?: number;
  updateInterval?: number;
  timeFormat?: string;
  valueFormat?: (value: number) => string;
  showGrid?: boolean;
  showLegend?: boolean;
  showTooltip?: boolean;
  animationDuration?: number;
  yAxisMin?: number | 'dataMin';
  yAxisMax?: number | 'dataMax';
  onDataUpdate?: (data: RealTimeDataPoint[]) => void;
}

export const RealTimeChart: React.FC<RealTimeChartProps> = ({
  data,
  series = [{ name: '实时数据', color: '#1890ff' }],
  maxDataPoints = 100,
  updateInterval = 1000,
  timeFormat = 'HH:mm:ss',
  valueFormat = (value: number) => value.toFixed(2),
  showGrid = true,
  showLegend = true,
  showTooltip = true,
  animationDuration = 300,
  yAxisMin = 'dataMin',
  yAxisMax = 'dataMax',
  onDataUpdate,
  ...chartProps
}) => {
  const [chartData, setChartData] = useState<RealTimeDataPoint[]>([]);
  const dataRef = useRef<RealTimeDataPoint[]>([]);

  // Update internal data when props change
  useEffect(() => {
    const newData = data.slice(-maxDataPoints);
    dataRef.current = newData;
    setChartData(newData);
    
    if (onDataUpdate) {
      onDataUpdate(newData);
    }
  }, [data, maxDataPoints, onDataUpdate]);

  // Auto-update mechanism (if needed for simulated real-time updates)
  useEffect(() => {
    if (updateInterval <= 0) return;

    const interval = setInterval(() => {
      // This would typically be used for simulated data
      // In real applications, data updates come from props
      setChartData(prev => [...prev]);
    }, updateInterval);

    return () => clearInterval(interval);
  }, [updateInterval]);

  const option = useMemo(() => {
    if (!chartData || chartData.length === 0) {
      return {
        title: {
          text: '暂无数据',
          left: 'center',
          top: 'middle',
          textStyle: {
            color: '#999',
            fontSize: 14,
          },
        },
      };
    }

    // Process data for chart
    const timestamps = chartData.map(item => 
      dayjs(item.timestamp).format(timeFormat)
    );
    const values = chartData.map(item => item.value);

    // Create series based on configuration
    const chartSeries = series.map((seriesConfig, index) => ({
      name: seriesConfig.name,
      type: seriesConfig.type || 'line',
      data: values,
      smooth: seriesConfig.smooth !== false,
      showSymbol: seriesConfig.showSymbol || false,
      lineStyle: {
        color: seriesConfig.color || `hsl(${index * 60}, 70%, 50%)`,
        width: seriesConfig.lineWidth || 2,
      },
      itemStyle: {
        color: seriesConfig.color || `hsl(${index * 60}, 70%, 50%)`,
      },
      areaStyle: seriesConfig.areaStyle ? {
        color: {
          type: 'linear',
          x: 0,
          y: 0,
          x2: 0,
          y2: 1,
          colorStops: [
            { offset: 0, color: seriesConfig.color || `hsl(${index * 60}, 70%, 50%)` },
            { offset: 1, color: 'transparent' },
          ],
        },
      } : undefined,
      animationDuration,
    }));

    return {
      tooltip: showTooltip ? {
        trigger: 'axis',
        formatter: (params: any) => {
          const dataIndex = params[0]?.dataIndex;
          if (dataIndex === undefined) return '';

          const item = chartData[dataIndex];
          if (!item) return '';

          return `
            <div style="margin-bottom: 4px; font-weight: bold;">
              ${dayjs(item.timestamp).format('YYYY-MM-DD HH:mm:ss')}
            </div>
            <div>
              ${item.label || '值'}: ${valueFormat(item.value)}
            </div>
          `;
        },
      } : undefined,
      legend: showLegend ? {
        data: series.map(s => s.name),
        top: 10,
      } : undefined,
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        top: showLegend ? '15%' : '3%',
        containLabel: true,
        show: showGrid,
        borderColor: '#f0f0f0',
      },
      xAxis: {
        type: 'category' as const,
        data: timestamps,
        boundaryGap: false,
        axisLine: {
          lineStyle: {
            color: '#d9d9d9',
          },
        },
        axisTick: {
          show: false,
        },
        axisLabel: {
          color: '#666',
          fontSize: 12,
        },
        splitLine: {
          show: showGrid,
          lineStyle: {
            color: '#f0f0f0',
            type: 'dashed',
          },
        },
      },
      yAxis: {
        type: 'value',
        min: yAxisMin,
        max: yAxisMax,
        axisLine: {
          lineStyle: {
            color: '#d9d9d9',
          },
        },
        axisTick: {
          show: false,
        },
        axisLabel: {
          color: '#666',
          fontSize: 12,
          formatter: valueFormat,
        },
        splitLine: {
          show: showGrid,
          lineStyle: {
            color: '#f0f0f0',
            type: 'dashed',
          },
        },
      },
      series: chartSeries,
      animation: true,
      animationDuration,
      animationEasing: 'cubicOut',
    };
  }, [
    chartData,
    series,
    timeFormat,
    valueFormat,
    showGrid,
    showLegend,
    showTooltip,
    animationDuration,
    yAxisMin,
    yAxisMax,
  ]);

  return (
    <BaseChart
      option={option}
      notMerge={false}
      lazyUpdate={true}
      {...chartProps}
    />
  );
};

export default RealTimeChart;
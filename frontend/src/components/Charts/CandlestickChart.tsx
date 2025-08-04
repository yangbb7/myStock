import React, { useMemo } from 'react';
import { BaseChart, BaseChartProps } from './BaseChart';
import dayjs from 'dayjs';

export interface CandlestickData {
  timestamp: string | number | Date;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

export interface TechnicalIndicator {
  name: string;
  data: Array<{ timestamp: string | number | Date; value: number }>;
  color?: string;
  type?: 'line' | 'bar';
}

interface CandlestickChartProps extends Omit<BaseChartProps, 'option'> {
  data: CandlestickData[];
  symbol?: string;
  indicators?: TechnicalIndicator[];
  showVolume?: boolean;
  showMA?: boolean;
  maParams?: number[];
  dateFormat?: string;
  priceFormat?: (value: number) => string;
  volumeFormat?: (value: number) => string;
  onDataZoom?: (params: any) => void;
  onBrush?: (params: any) => void;
}

export const CandlestickChart: React.FC<CandlestickChartProps> = ({
  data,
  symbol = '',
  indicators = [],
  showVolume = true,
  showMA = true,
  maParams = [5, 10, 20, 30],
  dateFormat = 'MM-DD',
  priceFormat = (value: number) => `¥${value.toFixed(2)}`,
  volumeFormat = (value: number) => value.toLocaleString(),
  onDataZoom,
  onBrush,
  ...chartProps
}) => {
  const option = useMemo(() => {
    if (!data || data.length === 0) {
      return {};
    }

    // Process data
    const processedData = data.map(item => ({
      timestamp: dayjs(item.timestamp).format(dateFormat),
      values: [item.open, item.close, item.low, item.high],
      volume: item.volume || 0,
    }));

    const dates = processedData.map(item => item.timestamp);
    const candlestickData = processedData.map(item => item.values);
    const volumeData = processedData.map(item => item.volume);

    // Calculate moving averages
    const calculateMA = (period: number) => {
      const result: (number | null)[] = [];
      for (let i = 0; i < data.length; i++) {
        if (i < period - 1) {
          result.push(null);
        } else {
          const sum = data.slice(i - period + 1, i + 1)
            .reduce((acc, curr) => acc + curr.close, 0);
          result.push(sum / period);
        }
      }
      return result;
    };

    const series: any[] = [
      {
        name: 'K线',
        type: 'candlestick',
        data: candlestickData,
        itemStyle: {
          color: '#ef232a',
          color0: '#14b143',
          borderColor: '#ef232a',
          borderColor0: '#14b143',
        },
        emphasis: {
          itemStyle: {
            color: '#ef232a',
            color0: '#14b143',
            borderColor: '#ef232a',
            borderColor0: '#14b143',
          },
        },
      },
    ];

    // Add moving averages
    if (showMA) {
      const maColors = ['#1890ff', '#52c41a', '#faad14', '#f5222d'];
      maParams.forEach((period, index) => {
        series.push({
          name: `MA${period}`,
          type: 'line',
          data: calculateMA(period),
          smooth: true,
          lineStyle: {
            color: maColors[index % maColors.length],
            width: 1,
          },
          showSymbol: false,
        });
      });
    }

    // Add technical indicators
    indicators.forEach((indicator, index) => {
      const indicatorData = indicator.data.map(item => {
        const dateIndex = dates.findIndex(date => 
          dayjs(item.timestamp).format(dateFormat) === date
        );
        return dateIndex >= 0 ? item.value : null;
      });

      series.push({
        name: indicator.name,
        type: indicator.type || 'line',
        data: indicatorData,
        smooth: true,
        lineStyle: {
          color: indicator.color || `hsl(${index * 60}, 70%, 50%)`,
          width: 1,
        },
        showSymbol: false,
      });
    });

    // Add volume series if enabled
    if (showVolume) {
      series.push({
        name: '成交量',
        type: 'bar',
        yAxisIndex: 1,
        data: volumeData,
        itemStyle: {
          color: (params: any) => {
            const dataIndex = params.dataIndex;
            const currentClose = data[dataIndex]?.close || 0;
            const prevClose = dataIndex > 0 ? data[dataIndex - 1]?.close || 0 : currentClose;
            return currentClose >= prevClose ? '#ef232a' : '#14b143';
          },
        },
      });
    }

    const yAxes: any[] = [
      {
        type: 'value',
        scale: true,
        position: 'right',
        axisLabel: {
          formatter: priceFormat,
        },
        splitLine: {
          show: true,
          lineStyle: {
            color: '#f0f0f0',
          },
        },
      },
    ];

    if (showVolume) {
      yAxes.push({
        type: 'value',
        scale: true,
        position: 'left',
        max: (value: any) => value.max * 4,
        axisLabel: {
          formatter: volumeFormat,
        },
        splitLine: {
          show: false,
        },
      });
    }

    return {
      title: {
        text: symbol ? `${symbol} K线图` : 'K线图',
        left: 'center',
        textStyle: {
          fontSize: 16,
          fontWeight: 'normal' as any,
        },
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'cross',
        },
        formatter: (params: any) => {
          const dataIndex = params[0]?.dataIndex;
          if (dataIndex === undefined) return '';

          const item = data[dataIndex];
          if (!item) return '';

          let tooltip = `
            <div style="margin-bottom: 8px; font-weight: bold;">${dates[dataIndex]}</div>
            <div>开盘: ${priceFormat(item.open)}</div>
            <div>收盘: ${priceFormat(item.close)}</div>
            <div>最高: ${priceFormat(item.high)}</div>
            <div>最低: ${priceFormat(item.low)}</div>
          `;

          if (showVolume && item.volume) {
            tooltip += `<div>成交量: ${volumeFormat(item.volume)}</div>`;
          }

          return tooltip;
        },
      },
      legend: {
        data: [
          'K线',
          ...(showMA ? maParams.map(p => `MA${p}`) : []),
          ...indicators.map(i => i.name),
          ...(showVolume ? ['成交量'] : []),
        ],
        top: 30,
      },
      grid: {
        left: '10%',
        right: '10%',
        top: showVolume ? '15%' : '20%',
        bottom: showVolume ? '25%' : '15%',
      },
      xAxis: {
        type: 'category',
        data: dates,
        scale: true,
        boundaryGap: false,
        axisLine: { onZero: false },
        splitLine: { show: false },
        min: 'dataMin',
        max: 'dataMax',
      },
      yAxis: yAxes,
      series,
      dataZoom: [
        {
          type: 'inside',
          xAxisIndex: [0],
          start: 80,
          end: 100,
        },
        {
          show: true,
          xAxisIndex: [0],
          type: 'slider',
          top: '90%',
          start: 80,
          end: 100,
        },
      ],
      brush: {
        xAxisIndex: 'all',
        brushLink: 'all',
        outOfBrush: {
          colorAlpha: 0.1,
        },
      },
    };
  }, [data, symbol, indicators, showVolume, showMA, maParams, dateFormat, priceFormat, volumeFormat]);

  const events = useMemo(() => {
    const eventHandlers: Record<string, (params: any) => void> = {};
    if (onDataZoom) eventHandlers.datazoom = onDataZoom;
    if (onBrush) eventHandlers.brush = onBrush;
    return eventHandlers;
  }, [onDataZoom, onBrush]);

  return (
    <BaseChart
      option={option}
      onEvents={events}
      height={500}
      {...chartProps}
    />
  );
};

export default CandlestickChart;
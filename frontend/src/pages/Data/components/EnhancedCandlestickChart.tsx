import React, { useState, useEffect, useMemo, useCallback } from 'react';
import {
  Card,
  Select,
  Button,
  Space,
  Row,
  Col,
  DatePicker,
  Checkbox,
  InputNumber,
  Tooltip,
  Tag,
  message,
} from 'antd';
import {
  FullscreenOutlined,
  DownloadOutlined,
  SettingOutlined,
  ReloadOutlined,
  CompressOutlined,
} from '@ant-design/icons';
import { useQuery } from '@tanstack/react-query';
import dayjs, { Dayjs } from 'dayjs';
import { CandlestickChart, CandlestickData, TechnicalIndicator } from '../../../components/Charts';
import { api } from '../../../services/api';
import { MarketDataRecord } from '../../../services/types';

const { Option } = Select;
const { RangePicker } = DatePicker;

interface EnhancedCandlestickChartProps {
  defaultSymbol?: string;
  height?: number;
  showControls?: boolean;
}

interface ChartSettings {
  showVolume: boolean;
  showMA: boolean;
  maParams: number[];
  showIndicators: string[];
  timeframe: string;
  compareSymbols: string[];
}

const AVAILABLE_SYMBOLS = [
  { value: '000001.SZ', label: '平安银行' },
  { value: '000002.SZ', label: '万科A' },
  { value: '600000.SH', label: '浦发银行' },
  { value: '600036.SH', label: '招商银行' },
  { value: '600519.SH', label: '贵州茅台' },
  { value: '000858.SZ', label: '五粮液' },
  { value: '002415.SZ', label: '海康威视' },
  { value: '300059.SZ', label: '东方财富' },
];

const TIMEFRAMES = [
  { value: '1m', label: '1分钟' },
  { value: '5m', label: '5分钟' },
  { value: '15m', label: '15分钟' },
  { value: '30m', label: '30分钟' },
  { value: '1h', label: '1小时' },
  { value: '1d', label: '日线' },
  { value: '1w', label: '周线' },
  { value: '1M', label: '月线' },
];

const TECHNICAL_INDICATORS = [
  { value: 'RSI', label: 'RSI相对强弱指标' },
  { value: 'MACD', label: 'MACD指标' },
  { value: 'BOLL', label: '布林带' },
  { value: 'KDJ', label: 'KDJ指标' },
  { value: 'WR', label: '威廉指标' },
];

const EnhancedCandlestickChart: React.FC<EnhancedCandlestickChartProps> = ({
  defaultSymbol = '000001.SZ',
  height = 600,
  showControls = true,
}) => {
  const [selectedSymbol, setSelectedSymbol] = useState(defaultSymbol);
  const [dateRange, setDateRange] = useState<[Dayjs, Dayjs]>([
    dayjs().subtract(30, 'day'),
    dayjs(),
  ]);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [settings, setSettings] = useState<ChartSettings>({
    showVolume: true,
    showMA: true,
    maParams: [5, 10, 20, 30],
    showIndicators: [],
    timeframe: '1d',
    compareSymbols: [],
  });

  // Fetch market data
  const { data: marketData, isLoading, error, refetch } = useQuery({
    queryKey: ['marketData', selectedSymbol, settings.timeframe, dateRange],
    queryFn: async () => {
      const startDate = dateRange[0].format('YYYY-MM-DD');
      const endDate = dateRange[1].format('YYYY-MM-DD');
      
      const response = await api.data.getMarketData(
        selectedSymbol,
        settings.timeframe,
        startDate,
        endDate
      );
      
      return response;
    },
    enabled: !!selectedSymbol,
  });

  // Fetch comparison data
  const { data: comparisonData } = useQuery({
    queryKey: ['comparisonData', settings.compareSymbols, settings.timeframe, dateRange],
    queryFn: async () => {
      if (settings.compareSymbols.length === 0) return {};
      
      const startDate = dateRange[0].format('YYYY-MM-DD');
      const endDate = dateRange[1].format('YYYY-MM-DD');
      
      const promises = settings.compareSymbols.map(async (symbol) => {
        const response = await api.data.getMarketData(
          symbol,
          settings.timeframe,
          startDate,
          endDate
        );
        return { symbol, data: response };
      });
      
      const results = await Promise.all(promises);
      return results.reduce((acc, { symbol, data }) => {
        acc[symbol] = data;
        return acc;
      }, {} as Record<string, any>);
    },
    enabled: settings.compareSymbols.length > 0,
  });

  // Transform data for chart
  const chartData = useMemo((): CandlestickData[] => {
    if (!marketData?.records) return [];
    
    return marketData.records.map((record: MarketDataRecord) => ({
      timestamp: record.datetime,
      open: record.open,
      high: record.high,
      low: record.low,
      close: record.close,
      volume: record.volume,
    }));
  }, [marketData]);

  // Calculate technical indicators
  const technicalIndicators = useMemo((): TechnicalIndicator[] => {
    if (!chartData.length || settings.showIndicators.length === 0) return [];
    
    const indicators: TechnicalIndicator[] = [];
    
    settings.showIndicators.forEach((indicatorType) => {
      switch (indicatorType) {
        case 'RSI':
          indicators.push(calculateRSI(chartData));
          break;
        case 'MACD':
          indicators.push(...calculateMACD(chartData));
          break;
        case 'BOLL':
          indicators.push(...calculateBollingerBands(chartData));
          break;
        case 'KDJ':
          indicators.push(...calculateKDJ(chartData));
          break;
        case 'WR':
          indicators.push(calculateWilliamsR(chartData));
          break;
      }
    });
    
    return indicators;
  }, [chartData, settings.showIndicators]);

  // Handle symbol change
  const handleSymbolChange = useCallback((symbol: string) => {
    setSelectedSymbol(symbol);
  }, []);

  // Handle date range change
  const handleDateRangeChange = useCallback((dates: [Dayjs, Dayjs] | null) => {
    if (dates) {
      setDateRange(dates);
    }
  }, []);

  // Handle settings change
  const handleSettingsChange = useCallback((key: keyof ChartSettings, value: any) => {
    setSettings(prev => ({
      ...prev,
      [key]: value,
    }));
  }, []);

  // Handle fullscreen toggle
  const handleFullscreenToggle = useCallback(() => {
    setIsFullscreen(prev => !prev);
  }, []);

  // Handle export
  const handleExport = useCallback(() => {
    if (!chartData.length) {
      message.warning('没有数据可导出');
      return;
    }
    
    const csvContent = [
      ['日期', '开盘价', '最高价', '最低价', '收盘价', '成交量'].join(','),
      ...chartData.map(item => [
        dayjs(item.timestamp).format('YYYY-MM-DD'),
        item.open.toFixed(2),
        item.high.toFixed(2),
        item.low.toFixed(2),
        item.close.toFixed(2),
        item.volume?.toString() || '0',
      ].join(','))
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', `${selectedSymbol}_${settings.timeframe}_${dayjs().format('YYYY-MM-DD')}.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    message.success('数据导出成功');
  }, [chartData, selectedSymbol, settings.timeframe]);

  // Technical indicator calculation functions
  function calculateRSI(data: CandlestickData[], period: number = 14): TechnicalIndicator {
    const rsiData: Array<{ timestamp: string | number | Date; value: number }> = [];
    
    for (let i = period; i < data.length; i++) {
      let gains = 0;
      let losses = 0;
      
      for (let j = i - period + 1; j <= i; j++) {
        const change = data[j].close - data[j - 1].close;
        if (change > 0) gains += change;
        else losses -= change;
      }
      
      const avgGain = gains / period;
      const avgLoss = losses / period;
      const rs = avgGain / avgLoss;
      const rsi = 100 - (100 / (1 + rs));
      
      rsiData.push({
        timestamp: data[i].timestamp,
        value: rsi,
      });
    }
    
    return {
      name: 'RSI',
      data: rsiData,
      color: '#722ed1',
    };
  }

  function calculateMACD(data: CandlestickData[]): TechnicalIndicator[] {
    const ema12 = calculateEMA(data, 12);
    const ema26 = calculateEMA(data, 26);
    const macdLine: Array<{ timestamp: string | number | Date; value: number }> = [];
    
    for (let i = 0; i < data.length; i++) {
      if (ema12[i] !== null && ema26[i] !== null) {
        macdLine.push({
          timestamp: data[i].timestamp,
          value: ema12[i]! - ema26[i]!,
        });
      }
    }
    
    return [
      {
        name: 'MACD',
        data: macdLine,
        color: '#1890ff',
      },
    ];
  }

  function calculateBollingerBands(data: CandlestickData[], period: number = 20): TechnicalIndicator[] {
    const upperBand: Array<{ timestamp: string | number | Date; value: number }> = [];
    const lowerBand: Array<{ timestamp: string | number | Date; value: number }> = [];
    
    for (let i = period - 1; i < data.length; i++) {
      const slice = data.slice(i - period + 1, i + 1);
      const avg = slice.reduce((sum, item) => sum + item.close, 0) / period;
      const variance = slice.reduce((sum, item) => sum + Math.pow(item.close - avg, 2), 0) / period;
      const stdDev = Math.sqrt(variance);
      
      upperBand.push({
        timestamp: data[i].timestamp,
        value: avg + 2 * stdDev,
      });
      
      lowerBand.push({
        timestamp: data[i].timestamp,
        value: avg - 2 * stdDev,
      });
    }
    
    return [
      {
        name: '布林上轨',
        data: upperBand,
        color: '#ff4d4f',
      },
      {
        name: '布林下轨',
        data: lowerBand,
        color: '#52c41a',
      },
    ];
  }

  function calculateKDJ(data: CandlestickData[]): TechnicalIndicator[] {
    // Simplified KDJ calculation
    return [];
  }

  function calculateWilliamsR(data: CandlestickData[], period: number = 14): TechnicalIndicator {
    const wrData: Array<{ timestamp: string | number | Date; value: number }> = [];
    
    for (let i = period - 1; i < data.length; i++) {
      const slice = data.slice(i - period + 1, i + 1);
      const highest = Math.max(...slice.map(item => item.high));
      const lowest = Math.min(...slice.map(item => item.low));
      const close = data[i].close;
      
      const wr = ((highest - close) / (highest - lowest)) * -100;
      
      wrData.push({
        timestamp: data[i].timestamp,
        value: wr,
      });
    }
    
    return {
      name: 'Williams %R',
      data: wrData,
      color: '#fa8c16',
    };
  }

  function calculateEMA(data: CandlestickData[], period: number): (number | null)[] {
    const ema: (number | null)[] = [];
    const multiplier = 2 / (period + 1);
    
    for (let i = 0; i < data.length; i++) {
      if (i === 0) {
        ema.push(data[i].close);
      } else if (ema[i - 1] !== null) {
        ema.push((data[i].close - ema[i - 1]!) * multiplier + ema[i - 1]!);
      } else {
        ema.push(null);
      }
    }
    
    return ema;
  }

  const cardStyle = isFullscreen ? {
    position: 'fixed' as const,
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    zIndex: 1000,
    margin: 0,
    borderRadius: 0,
  } : {};

  return (
    <Card
      title={
        <Space>
          <span>K线图分析</span>
          {selectedSymbol && (
            <Tag color="blue">
              {AVAILABLE_SYMBOLS.find(s => s.value === selectedSymbol)?.label || selectedSymbol}
            </Tag>
          )}
          {isLoading && <Tag color="processing">加载中...</Tag>}
        </Space>
      }
      extra={
        showControls && (
          <Space>
            <Button
              icon={<ReloadOutlined />}
              onClick={() => refetch()}
              loading={isLoading}
              size="small"
            >
              刷新
            </Button>
            <Button
              icon={<DownloadOutlined />}
              onClick={handleExport}
              disabled={!chartData.length}
              size="small"
            >
              导出
            </Button>
            <Button
              icon={isFullscreen ? <CompressOutlined /> : <FullscreenOutlined />}
              onClick={handleFullscreenToggle}
              size="small"
            >
              {isFullscreen ? '退出全屏' : '全屏'}
            </Button>
          </Space>
        )
      }
      style={cardStyle}
      bodyStyle={{ padding: isFullscreen ? 24 : 16 }}
    >
      {showControls && (
        <Row gutter={16} style={{ marginBottom: 16 }}>
          <Col span={4}>
            <Select
              value={selectedSymbol}
              onChange={handleSymbolChange}
              style={{ width: '100%' }}
              placeholder="选择股票"
            >
              {AVAILABLE_SYMBOLS.map(symbol => (
                <Option key={symbol.value} value={symbol.value}>
                  {symbol.label}
                </Option>
              ))}
            </Select>
          </Col>
          <Col span={4}>
            <Select
              value={settings.timeframe}
              onChange={(value) => handleSettingsChange('timeframe', value)}
              style={{ width: '100%' }}
            >
              {TIMEFRAMES.map(tf => (
                <Option key={tf.value} value={tf.value}>
                  {tf.label}
                </Option>
              ))}
            </Select>
          </Col>
          <Col span={6}>
            <RangePicker
              value={dateRange}
              onChange={handleDateRangeChange}
              style={{ width: '100%' }}
              format="YYYY-MM-DD"
            />
          </Col>
          <Col span={4}>
            <Select
              mode="multiple"
              placeholder="技术指标"
              value={settings.showIndicators}
              onChange={(value) => handleSettingsChange('showIndicators', value)}
              style={{ width: '100%' }}
              maxTagCount={1}
            >
              {TECHNICAL_INDICATORS.map(indicator => (
                <Option key={indicator.value} value={indicator.value}>
                  {indicator.label}
                </Option>
              ))}
            </Select>
          </Col>
          <Col span={6}>
            <Space>
              <Checkbox
                checked={settings.showVolume}
                onChange={(e) => handleSettingsChange('showVolume', e.target.checked)}
              >
                成交量
              </Checkbox>
              <Checkbox
                checked={settings.showMA}
                onChange={(e) => handleSettingsChange('showMA', e.target.checked)}
              >
                均线
              </Checkbox>
            </Space>
          </Col>
        </Row>
      )}

      {error && (
        <div style={{ textAlign: 'center', padding: 40 }}>
          <p>加载数据失败: {error.message}</p>
          <Button onClick={() => refetch()}>重试</Button>
        </div>
      )}

      <CandlestickChart
        data={chartData}
        symbol={selectedSymbol}
        indicators={technicalIndicators}
        showVolume={settings.showVolume}
        showMA={settings.showMA}
        maParams={settings.maParams}
        height={isFullscreen ? window.innerHeight - 200 : height}
        loading={isLoading}
      />
    </Card>
  );
};

export { EnhancedCandlestickChart };
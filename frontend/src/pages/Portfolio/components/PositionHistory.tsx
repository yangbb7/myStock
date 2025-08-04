import React, { useMemo } from 'react';
import { Card, Row, Col, Statistic, Button, Space, Tag } from 'antd';
import { CloseOutlined, RiseOutlined, ArrowDownOutlined } from '@ant-design/icons';
import { useQuery } from '@tanstack/react-query';
import dayjs from 'dayjs';
import { BaseChart } from '../../../components/Charts/BaseChart';
import { LoadingState } from '../../../components/Common/LoadingState';
import { api } from '../../../services/api';
import type { EChartsOption } from 'echarts';

interface PositionHistoryProps {
  symbol: string;
  onClose: () => void;
  dateRange?: [dayjs.Dayjs, dayjs.Dayjs] | null;
}

export const PositionHistory: React.FC<PositionHistoryProps> = ({
  symbol,
  onClose,
  dateRange,
}) => {
  // Fetch market data for the symbol
  const {
    data: marketData,
    isLoading: marketLoading,
    error: marketError,
  } = useQuery({
    queryKey: ['marketData', symbol, dateRange],
    queryFn: () => api.data.getMarketData(
      symbol,
      '1d',
      dateRange?.[0]?.format('YYYY-MM-DD'),
      dateRange?.[1]?.format('YYYY-MM-DD')
    ),
    enabled: !!symbol,
  });

  // Fetch portfolio history for position tracking
  const {
    data: portfolioHistory,
    isLoading: historyLoading,
  } = useQuery({
    queryKey: ['portfolioHistory', symbol, dateRange],
    queryFn: () => api.portfolio.getHistory({
      startDate: dateRange?.[0]?.format('YYYY-MM-DD'),
      endDate: dateRange?.[1]?.format('YYYY-MM-DD'),
    }),
    enabled: !!symbol,
  });

  // Process chart data
  const chartData = useMemo(() => {
    if (!marketData?.records) return null;

    const records = marketData.records;
    const dates = records.map(record => dayjs(record.datetime).format('MM-DD'));
    const prices = records.map(record => record.close);
    const volumes = records.map(record => record.volume);

    // Calculate price change
    const priceChanges = prices.map((price, index) => {
      if (index === 0) return 0;
      return ((price - prices[index - 1]) / prices[index - 1]) * 100;
    });

    return {
      dates,
      prices,
      volumes,
      priceChanges,
      records,
    };
  }, [marketData]);

  // Calculate statistics
  const statistics = useMemo(() => {
    if (!chartData) return null;

    const { prices, priceChanges } = chartData;
    const currentPrice = prices[prices.length - 1];
    const firstPrice = prices[0];
    const totalReturn = ((currentPrice - firstPrice) / firstPrice) * 100;
    const maxPrice = Math.max(...prices);
    const minPrice = Math.min(...prices);
    const avgVolume = chartData.volumes.reduce((sum, vol) => sum + vol, 0) / chartData.volumes.length;
    
    // Calculate volatility (standard deviation of price changes)
    const avgChange = priceChanges.reduce((sum, change) => sum + change, 0) / priceChanges.length;
    const variance = priceChanges.reduce((sum, change) => sum + Math.pow(change - avgChange, 2), 0) / priceChanges.length;
    const volatility = Math.sqrt(variance);

    return {
      currentPrice,
      totalReturn,
      maxPrice,
      minPrice,
      avgVolume,
      volatility,
      priceRange: maxPrice - minPrice,
      priceRangePercent: ((maxPrice - minPrice) / minPrice) * 100,
    };
  }, [chartData]);

  // Chart option
  const chartOption = useMemo((): EChartsOption => {
    if (!chartData) return {};

    return {
      title: {
        text: `${symbol} 价格走势`,
        left: 'center',
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'cross',
        },
        formatter: (params: any) => {
          const priceData = params.find((p: any) => p.seriesName === '价格');
          const volumeData = params.find((p: any) => p.seriesName === '成交量');
          
          if (!priceData) return '';
          
          const record = chartData.records[priceData.dataIndex];
          return `
            <div>
              <strong>${dayjs(record.datetime).format('YYYY-MM-DD')}</strong><br/>
              开盘: ¥${record.open.toFixed(2)}<br/>
              最高: ¥${record.high.toFixed(2)}<br/>
              最低: ¥${record.low.toFixed(2)}<br/>
              收盘: ¥${record.close.toFixed(2)}<br/>
              成交量: ${record.volume.toLocaleString()}
            </div>
          `;
        },
      },
      legend: {
        data: ['价格', '成交量'],
        top: 30,
      },
      grid: [
        {
          left: '10%',
          right: '8%',
          height: '50%',
        },
        {
          left: '10%',
          right: '8%',
          top: '70%',
          height: '20%',
        },
      ],
      xAxis: [
        {
          type: 'category',
          data: chartData.dates,
          gridIndex: 0,
        },
        {
          type: 'category',
          data: chartData.dates,
          gridIndex: 1,
        },
      ],
      yAxis: [
        {
          type: 'value',
          name: '价格 (¥)',
          gridIndex: 0,
          axisLabel: {
            formatter: (value: number) => `¥${value.toFixed(2)}`,
          },
        },
        {
          type: 'value',
          name: '成交量',
          gridIndex: 1,
          axisLabel: {
            formatter: (value: number) => {
              if (value >= 10000) {
                return `${(value / 10000).toFixed(1)}万`;
              }
              return value.toLocaleString();
            },
          },
        },
      ],
      series: [
        {
          name: '价格',
          type: 'line',
          data: chartData.prices,
          smooth: true,
          symbol: 'none',
          lineStyle: {
            width: 2,
            color: '#1890ff',
          },
          areaStyle: {
            color: {
              type: 'linear',
              x: 0,
              y: 0,
              x2: 0,
              y2: 1,
              colorStops: [
                { offset: 0, color: 'rgba(24, 144, 255, 0.3)' },
                { offset: 1, color: 'rgba(24, 144, 255, 0.1)' },
              ],
            },
          },
          xAxisIndex: 0,
          yAxisIndex: 0,
        },
        {
          name: '成交量',
          type: 'bar',
          data: chartData.volumes,
          itemStyle: {
            color: '#52c41a',
            opacity: 0.7,
          },
          xAxisIndex: 1,
          yAxisIndex: 1,
        },
      ],
    };
  }, [chartData, symbol]);

  return (
    <Card
      title={
        <Space>
          <span>{symbol} 持仓历史</span>
          <Tag color={statistics?.totalReturn >= 0 ? 'green' : 'red'}>
            {statistics?.totalReturn >= 0 ? '+' : ''}{statistics?.totalReturn.toFixed(2)}%
          </Tag>
        </Space>
      }
      extra={
        <Button
          type="text"
          icon={<CloseOutlined />}
          onClick={onClose}
        />
      }
      size="small"
    >
      <LoadingState loading={marketLoading || historyLoading} error={marketError}>
        {statistics && (
          <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
            <Col xs={12} sm={8} md={6}>
              <Statistic
                title="当前价格"
                value={statistics.currentPrice}
                precision={2}
                prefix="¥"
                valueStyle={{ fontSize: '16px' }}
              />
            </Col>
            <Col xs={12} sm={8} md={6}>
              <Statistic
                title="总收益率"
                value={statistics.totalReturn}
                precision={2}
                suffix="%"
                prefix={statistics.totalReturn >= 0 ? <RiseOutlined /> : <ArrowDownOutlined />}
                valueStyle={{ 
                  color: statistics.totalReturn >= 0 ? '#3f8600' : '#cf1322',
                  fontSize: '16px',
                }}
              />
            </Col>
            <Col xs={12} sm={8} md={6}>
              <Statistic
                title="价格区间"
                value={`${statistics.minPrice.toFixed(2)} - ${statistics.maxPrice.toFixed(2)}`}
                valueStyle={{ fontSize: '14px' }}
              />
            </Col>
            <Col xs={12} sm={8} md={6}>
              <Statistic
                title="波动率"
                value={statistics.volatility}
                precision={2}
                suffix="%"
                valueStyle={{ fontSize: '16px' }}
              />
            </Col>
          </Row>
        )}

        <BaseChart
          option={chartOption}
          loading={marketLoading}
          height={400}
        />
      </LoadingState>
    </Card>
  );
};

export default PositionHistory;
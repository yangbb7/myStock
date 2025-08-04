import React, { useState, useMemo } from 'react';
import {
  Card,
  Row,
  Col,
  Statistic,
  Select,
  DatePicker,
  Button,
  Space,
  Table,
  Tag,
  Tooltip,
  Progress,
} from 'antd';
import type { ColumnsType } from 'antd/es/table';
import {
  ReloadOutlined,
  RiseOutlined,
  ArrowDownOutlined,
  InfoCircleOutlined,
} from '@ant-design/icons';
import { useQuery } from '@tanstack/react-query';
import dayjs from 'dayjs';
import { api } from '../../../services/api';
import type { PerformanceMetrics, PortfolioSummary } from '../../../services/types';
import { StatisticCard } from '../../../components/Common/StatisticCard';
import { LoadingState } from '../../../components/Common/LoadingState';
import { BaseChart } from '../../../components/Charts/BaseChart';
import type { EChartsOption } from 'echarts';

const { RangePicker } = DatePicker;

interface BenchmarkData {
  name: string;
  symbol: string;
  return: number;
  volatility: number;
  sharpeRatio: number;
}

interface PerformanceContribution {
  symbol: string;
  contribution: number;
  contributionPercent: number;
  weight: number;
  return: number;
}

export const PerformanceAnalysis: React.FC = () => {
  const [dateRange, setDateRange] = useState<[dayjs.Dayjs, dayjs.Dayjs] | null>([
    dayjs().subtract(1, 'year'),
    dayjs(),
  ]);
  const [benchmarkSymbol, setBenchmarkSymbol] = useState<string>('000300.SH'); // 沪深300
  const [timeframe, setTimeframe] = useState<string>('1d');

  // Fetch portfolio performance
  const {
    data: performance,
    isLoading: performanceLoading,
    error: performanceError,
    refetch: refetchPerformance,
  } = useQuery({
    queryKey: ['portfolioPerformance', dateRange],
    queryFn: () => api.portfolio.getPerformance({
      startDate: dateRange?.[0]?.format('YYYY-MM-DD'),
      endDate: dateRange?.[1]?.format('YYYY-MM-DD'),
    }),
    enabled: !!dateRange,
  });

  // Fetch analytics performance for detailed metrics
  const {
    data: analyticsPerformance,
    isLoading: analyticsLoading,
  } = useQuery({
    queryKey: ['analyticsPerformance', dateRange],
    queryFn: () => api.analytics.getPerformance({
      startDate: dateRange?.[0]?.format('YYYY-MM-DD'),
      endDate: dateRange?.[1]?.format('YYYY-MM-DD'),
    }),
    enabled: !!dateRange,
  });

  // Fetch benchmark data
  const {
    data: benchmarkData,
    isLoading: benchmarkLoading,
  } = useQuery({
    queryKey: ['benchmarkData', benchmarkSymbol, dateRange],
    queryFn: () => api.data.getMarketData(
      benchmarkSymbol,
      timeframe,
      dateRange?.[0]?.format('YYYY-MM-DD'),
      dateRange?.[1]?.format('YYYY-MM-DD')
    ),
    enabled: !!benchmarkSymbol && !!dateRange,
  });

  // Fetch current portfolio for contribution analysis
  const {
    data: portfolioSummary,
    isLoading: portfolioLoading,
  } = useQuery({
    queryKey: ['portfolioSummary'],
    queryFn: () => api.portfolio.getSummary(),
    refetchInterval: 30000, // Refresh every 30 seconds
  });

  // Calculate benchmark performance
  const benchmarkPerformance = useMemo(() => {
    if (!benchmarkData?.records || benchmarkData.records.length < 2) return null;

    const records = benchmarkData.records;
    const firstPrice = records[0].close;
    const lastPrice = records[records.length - 1].close;
    const totalReturn = ((lastPrice - firstPrice) / firstPrice) * 100;

    // Calculate daily returns for volatility
    const dailyReturns = records.slice(1).map((record, index) => {
      const prevPrice = records[index].close;
      return ((record.close - prevPrice) / prevPrice) * 100;
    });

    const avgReturn = dailyReturns.reduce((sum, ret) => sum + ret, 0) / dailyReturns.length;
    const variance = dailyReturns.reduce((sum, ret) => sum + Math.pow(ret - avgReturn, 2), 0) / dailyReturns.length;
    const volatility = Math.sqrt(variance) * Math.sqrt(252); // Annualized volatility

    return {
      totalReturn,
      volatility,
      sharpeRatio: volatility > 0 ? (totalReturn / volatility) : 0,
      records: records.map(record => ({
        date: record.datetime,
        value: record.close,
      })),
    };
  }, [benchmarkData]);

  // Calculate performance contribution by position
  const performanceContribution = useMemo((): PerformanceContribution[] => {
    if (!portfolioSummary?.positions) return [];

    const totalValue = portfolioSummary.totalValue;
    return Object.entries(portfolioSummary.positions).map(([symbol, position]) => {
      const weight = totalValue > 0 ? (position.quantity * position.currentPrice) / totalValue : 0;
      const positionReturn = position.averagePrice > 0 ? 
        ((position.currentPrice - position.averagePrice) / position.averagePrice) * 100 : 0;
      const contribution = weight * positionReturn;
      const contributionPercent = totalValue > 0 ? (contribution / 100) * 100 : 0;

      return {
        symbol,
        contribution,
        contributionPercent,
        weight: weight * 100,
        return: positionReturn,
      };
    }).sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
  }, [portfolioSummary]);

  // Performance chart option
  const performanceChartOption = useMemo((): EChartsOption => {
    if (!performance && !benchmarkPerformance) {
      return {
        title: {
          text: '暂无数据',
          left: 'center',
          top: 'middle',
          textStyle: { color: '#999', fontSize: 14 },
        },
      };
    }

    // Use real portfolio performance data from API
    const portfolioData = performance?.data?.performanceHistory || 
                         analyticsPerformance?.data?.performanceHistory || 
                         [];

    const benchmarkCurve = benchmarkPerformance?.records.map((record) => {
      return ((record.value - benchmarkPerformance.records[0].value) / benchmarkPerformance.records[0].value) * 100;
    }) || [];

    return {
      title: {
        text: '收益率曲线对比',
        left: 'center',
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'cross',
        },
        formatter: (params: any) => {
          const date = params[0]?.axisValue;
          let content = `<strong>${date}</strong><br/>`;
          params.forEach((param: any) => {
            content += `${param.seriesName}: ${param.value >= 0 ? '+' : ''}${param.value.toFixed(2)}%<br/>`;
          });
          return content;
        },
      },
      legend: {
        data: ['投资组合', '基准指数'],
        top: 30,
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        containLabel: true,
      },
      xAxis: {
        type: 'category',
        data: portfolioData.map(item => dayjs(item.date).format('MM-DD')),
        boundaryGap: false,
      },
      yAxis: {
        type: 'value',
        name: '收益率 (%)',
        axisLabel: {
          formatter: (value: number) => `${value.toFixed(1)}%`,
        },
      },
      series: [
        {
          name: '投资组合',
          type: 'line',
          data: portfolioData.map(item => item.value),
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
        },
        {
          name: '基准指数',
          type: 'line',
          data: benchmarkCurve,
          smooth: true,
          symbol: 'none',
          lineStyle: {
            width: 2,
            color: '#52c41a',
            type: 'dashed',
          },
        },
      ],
    };
  }, [performance, benchmarkPerformance]);

  // Performance contribution table columns
  const contributionColumns: ColumnsType<PerformanceContribution> = [
    {
      title: '股票代码',
      dataIndex: 'symbol',
      key: 'symbol',
      width: 120,
      render: (symbol: string) => <strong>{symbol}</strong>,
    },
    {
      title: '权重',
      dataIndex: 'weight',
      key: 'weight',
      width: 100,
      align: 'right',
      render: (weight: number) => (
        <Space direction="vertical" size={0}>
          <span>{weight.toFixed(2)}%</span>
          <Progress
            percent={weight}
            size="small"
            showInfo={false}
            strokeColor={weight > 20 ? '#ff4d4f' : weight > 10 ? '#faad14' : '#52c41a'}
          />
        </Space>
      ),
      sorter: (a, b) => a.weight - b.weight,
    },
    {
      title: '个股收益率',
      dataIndex: 'return',
      key: 'return',
      width: 120,
      align: 'right',
      render: (returnValue: number) => (
        <span style={{ color: returnValue >= 0 ? '#3f8600' : '#cf1322' }}>
          {returnValue >= 0 ? '+' : ''}{returnValue.toFixed(2)}%
        </span>
      ),
      sorter: (a, b) => a.return - b.return,
    },
    {
      title: '收益贡献',
      dataIndex: 'contribution',
      key: 'contribution',
      width: 120,
      align: 'right',
      render: (contribution: number) => (
        <span style={{ color: contribution >= 0 ? '#3f8600' : '#cf1322' }}>
          {contribution >= 0 ? '+' : ''}{contribution.toFixed(2)}%
        </span>
      ),
      sorter: (a, b) => a.contribution - b.contribution,
    },
    {
      title: '贡献占比',
      dataIndex: 'contributionPercent',
      key: 'contributionPercent',
      width: 100,
      align: 'right',
      render: (percent: number) => (
        <Tag color={percent >= 0 ? 'green' : 'red'}>
          {percent >= 0 ? '+' : ''}{percent.toFixed(1)}%
        </Tag>
      ),
    },
  ];

  // Use real performance metrics from API
  const realMetrics = useMemo(() => {
    if (performance?.data) {
      return {
        totalReturn: performance.data.totalReturn || 0,
        annualizedReturn: performance.data.annualizedReturn || 0,
        volatility: performance.data.volatility || 0,
        sharpeRatio: performance.data.sharpeRatio || 0,
        maxDrawdown: performance.data.maxDrawdown || 0,
        calmarRatio: performance.data.calmarRatio || 0,
        winRate: performance.data.winRate || 0,
        profitFactor: performance.data.profitFactor || 0,
      };
    }
    if (analyticsPerformance?.data) {
      return {
        totalReturn: analyticsPerformance.data.totalReturn || 0,
        annualizedReturn: analyticsPerformance.data.annualizedReturn || 0,
        volatility: analyticsPerformance.data.volatility || 0,
        sharpeRatio: analyticsPerformance.data.sharpeRatio || 0,
        maxDrawdown: analyticsPerformance.data.maxDrawdown || 0,
        calmarRatio: analyticsPerformance.data.calmarRatio || 0,
        winRate: analyticsPerformance.data.winRate || 0,
        profitFactor: analyticsPerformance.data.profitFactor || 0,
      };
    }
    return {
      totalReturn: 0,
      annualizedReturn: 0,
      volatility: 0,
      sharpeRatio: 0,
      maxDrawdown: 0,
      calmarRatio: 0,
      winRate: 0,
      profitFactor: 0,
    };
  }, [performance, analyticsPerformance]);

  return (
    <LoadingState loading={performanceLoading || analyticsLoading} error={performanceError}>
      <div style={{ padding: '0 0 24px 0' }}>
        {/* Controls */}
        <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
          <Col xs={24} sm={12} md={8}>
            <RangePicker
              value={dateRange}
              onChange={setDateRange}
              style={{ width: '100%' }}
              placeholder={['开始日期', '结束日期']}
            />
          </Col>
          <Col xs={24} sm={12} md={8}>
            <Select
              value={benchmarkSymbol}
              onChange={setBenchmarkSymbol}
              style={{ width: '100%' }}
              placeholder="选择基准指数"
            >
              <Select.Option value="000300.SH">沪深300</Select.Option>
              <Select.Option value="000905.SH">中证500</Select.Option>
              <Select.Option value="000001.SH">上证指数</Select.Option>
              <Select.Option value="399001.SZ">深证成指</Select.Option>
            </Select>
          </Col>
          <Col xs={24} sm={24} md={8}>
            <Space style={{ width: '100%', justifyContent: 'flex-end' }}>
              <Button
                icon={<ReloadOutlined />}
                onClick={() => refetchPerformance()}
                loading={performanceLoading}
              >
                刷新
              </Button>
            </Space>
          </Col>
        </Row>

        {/* Performance Metrics */}
        <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
          <Col xs={12} sm={8} md={6}>
            <StatisticCard
              title={
                <Space>
                  总收益率
                  <Tooltip title="投资组合在选定时间段内的总收益率">
                    <InfoCircleOutlined style={{ color: '#999' }} />
                  </Tooltip>
                </Space>
              }
              value={realMetrics.totalReturn}
              precision={2}
              suffix="%"
              prefix={realMetrics.totalReturn >= 0 ? <RiseOutlined /> : <ArrowDownOutlined />}
              valueStyle={{ color: realMetrics.totalReturn >= 0 ? '#3f8600' : '#cf1322' }}
              trend={realMetrics.totalReturn >= 0 ? 'up' : 'down'}
            />
          </Col>
          <Col xs={12} sm={8} md={6}>
            <StatisticCard
              title={
                <Space>
                  年化收益率
                  <Tooltip title="投资组合的年化收益率">
                    <InfoCircleOutlined style={{ color: '#999' }} />
                  </Tooltip>
                </Space>
              }
              value={realMetrics.annualizedReturn}
              precision={2}
              suffix="%"
              valueStyle={{ color: '#1890ff' }}
            />
          </Col>
          <Col xs={12} sm={8} md={6}>
            <StatisticCard
              title={
                <Space>
                  夏普比率
                  <Tooltip title="风险调整后的收益率指标，数值越高越好">
                    <InfoCircleOutlined style={{ color: '#999' }} />
                  </Tooltip>
                </Space>
              }
              value={realMetrics.sharpeRatio}
              precision={2}
              valueStyle={{ 
                color: realMetrics.sharpeRatio > 1 ? '#3f8600' : realMetrics.sharpeRatio > 0.5 ? '#faad14' : '#cf1322' 
              }}
            />
          </Col>
          <Col xs={12} sm={8} md={6}>
            <StatisticCard
              title={
                <Space>
                  最大回撤
                  <Tooltip title="投资组合从峰值到谷值的最大跌幅">
                    <InfoCircleOutlined style={{ color: '#999' }} />
                  </Tooltip>
                </Space>
              }
              value={Math.abs(realMetrics.maxDrawdown)}
              precision={2}
              suffix="%"
              valueStyle={{ color: '#cf1322' }}
              trend="down"
            />
          </Col>
        </Row>

        <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
          <Col xs={12} sm={8} md={6}>
            <StatisticCard
              title="波动率"
              value={realMetrics.volatility}
              precision={2}
              suffix="%"
              valueStyle={{ color: '#722ed1' }}
            />
          </Col>
          <Col xs={12} sm={8} md={6}>
            <StatisticCard
              title="卡玛比率"
              value={realMetrics.calmarRatio}
              precision={2}
              valueStyle={{ color: '#13c2c2' }}
            />
          </Col>
          <Col xs={12} sm={8} md={6}>
            <StatisticCard
              title="胜率"
              value={realMetrics.winRate}
              precision={1}
              suffix="%"
              valueStyle={{ color: '#52c41a' }}
            />
          </Col>
          <Col xs={12} sm={8} md={6}>
            <StatisticCard
              title="盈亏比"
              value={realMetrics.profitFactor}
              precision={2}
              valueStyle={{ color: '#fa8c16' }}
            />
          </Col>
        </Row>

        {/* Performance Chart */}
        <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
          <Col span={24}>
            <BaseChart
              option={performanceChartOption}
              loading={benchmarkLoading}
              height={400}
              title="收益率曲线对比"
            />
          </Col>
        </Row>

        {/* Performance Contribution Analysis */}
        <Row gutter={[16, 16]}>
          <Col span={24}>
            <Card title="收益贡献分析" size="small">
              <Table
                columns={contributionColumns}
                dataSource={performanceContribution}
                rowKey="symbol"
                pagination={{
                  pageSize: 10,
                  showSizeChanger: true,
                  showTotal: (total) => `共 ${total} 只股票`,
                }}
                size="small"
                loading={portfolioLoading}
                summary={(data) => {
                  const totalContribution = data.reduce((sum, item) => sum + item.contribution, 0);
                  return (
                    <Table.Summary.Row>
                      <Table.Summary.Cell index={0}>
                        <strong>合计</strong>
                      </Table.Summary.Cell>
                      <Table.Summary.Cell index={1} />
                      <Table.Summary.Cell index={2} />
                      <Table.Summary.Cell index={3}>
                        <strong style={{ color: totalContribution >= 0 ? '#3f8600' : '#cf1322' }}>
                          {totalContribution >= 0 ? '+' : ''}{totalContribution.toFixed(2)}%
                        </strong>
                      </Table.Summary.Cell>
                      <Table.Summary.Cell index={4} />
                    </Table.Summary.Row>
                  );
                }}
              />
            </Card>
          </Col>
        </Row>
      </div>
    </LoadingState>
  );
};

export default PerformanceAnalysis;
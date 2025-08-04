import React, { useState, useMemo } from 'react';
import {
  Card,
  Table,
  Row,
  Col,
  Statistic,
  Progress,
  Tag,
  Space,
  Button,
  Tooltip,
  Select,
  DatePicker,
  Input,
} from 'antd';
import type { ColumnsType } from 'antd/es/table';
import {
  SearchOutlined,
  ReloadOutlined,
  BarChartOutlined,
  PieChartOutlined,
} from '@ant-design/icons';
import { useQuery } from '@tanstack/react-query';
import dayjs from 'dayjs';
import { api } from '../../../services/api';
import type { Position, PortfolioSummary } from '../../../services/types';
import { StatisticCard } from '../../../components/Common/StatisticCard';
import { LoadingState } from '../../../components/Common/LoadingState';
import { PositionChart } from './PositionChart';
import { PositionHistory } from './PositionHistory';

const { RangePicker } = DatePicker;
const { Search } = Input;

interface PositionWithMetrics extends Position {
  key: string;
  marketValue: number;
  weight: number;
  dailyChange: number;
  dailyChangePercent: number;
  totalReturn: number;
  totalReturnPercent: number;
}

export const PositionManagement: React.FC = () => {
  const [searchText, setSearchText] = useState('');
  const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null);
  const [dateRange, setDateRange] = useState<[dayjs.Dayjs, dayjs.Dayjs] | null>(null);
  const [chartType, setChartType] = useState<'pie' | 'bar'>('pie');

  // Fetch portfolio summary
  const {
    data: portfolioSummary,
    isLoading: portfolioLoading,
    error: portfolioError,
    refetch: refetchPortfolio,
  } = useQuery({
    queryKey: ['portfolioSummary'],
    queryFn: () => api.portfolio.getSummary(),
    refetchInterval: 5000, // Refresh every 5 seconds
  });

  // Fetch portfolio history for selected date range
  const {
    data: portfolioHistory,
    isLoading: historyLoading,
  } = useQuery({
    queryKey: ['portfolioHistory', dateRange],
    queryFn: () => api.portfolio.getHistory({
      startDate: dateRange?.[0]?.format('YYYY-MM-DD'),
      endDate: dateRange?.[1]?.format('YYYY-MM-DD'),
    }),
    enabled: !!dateRange,
  });

  // Process positions data
  const positionsData = useMemo(() => {
    if (!portfolioSummary?.positions) return [];

    return Object.entries(portfolioSummary.positions).map(([symbol, position]) => {
      const marketValue = position.quantity * position.currentPrice;
      const weight = portfolioSummary.totalValue > 0 ? (marketValue / portfolioSummary.totalValue) * 100 : 0;
      const totalReturn = (position.currentPrice - position.averagePrice) * position.quantity;
      const totalReturnPercent = position.averagePrice > 0 ? ((position.currentPrice - position.averagePrice) / position.averagePrice) * 100 : 0;
      
      // Mock daily change data (in real implementation, this would come from API)
      const dailyChange = Math.random() * 200 - 100; // Random change for demo
      const dailyChangePercent = Math.random() * 10 - 5; // Random percentage for demo

      return {
        key: symbol,
        symbol,
        quantity: position.quantity,
        averagePrice: position.averagePrice,
        currentPrice: position.currentPrice,
        marketValue,
        weight,
        unrealizedPnl: position.unrealizedPnl,
        dailyChange,
        dailyChangePercent,
        totalReturn,
        totalReturnPercent,
      } as PositionWithMetrics;
    });
  }, [portfolioSummary]);

  // Filter positions based on search text
  const filteredPositions = useMemo(() => {
    if (!searchText) return positionsData;
    return positionsData.filter(position =>
      position.symbol.toLowerCase().includes(searchText.toLowerCase())
    );
  }, [positionsData, searchText]);

  // Calculate summary statistics
  const summaryStats = useMemo(() => {
    const totalMarketValue = positionsData.reduce((sum, pos) => sum + pos.marketValue, 0);
    const totalUnrealizedPnl = positionsData.reduce((sum, pos) => sum + pos.unrealizedPnl, 0);
    const totalDailyChange = positionsData.reduce((sum, pos) => sum + pos.dailyChange, 0);
    const avgWeight = positionsData.length > 0 ? positionsData.reduce((sum, pos) => sum + pos.weight, 0) / positionsData.length : 0;

    return {
      totalMarketValue,
      totalUnrealizedPnl,
      totalDailyChange,
      avgWeight,
      positionCount: positionsData.length,
    };
  }, [positionsData]);

  // Table columns configuration
  const columns: ColumnsType<PositionWithMetrics> = [
    {
      title: '股票代码',
      dataIndex: 'symbol',
      key: 'symbol',
      fixed: 'left',
      width: 120,
      render: (symbol: string) => (
        <Button
          type="link"
          onClick={() => setSelectedSymbol(symbol)}
          style={{ padding: 0, fontWeight: 'bold' }}
        >
          {symbol}
        </Button>
      ),
    },
    {
      title: '持仓数量',
      dataIndex: 'quantity',
      key: 'quantity',
      width: 100,
      align: 'right',
      render: (quantity: number) => quantity.toLocaleString(),
    },
    {
      title: '成本价',
      dataIndex: 'averagePrice',
      key: 'averagePrice',
      width: 100,
      align: 'right',
      render: (price: number) => `¥${price.toFixed(2)}`,
    },
    {
      title: '现价',
      dataIndex: 'currentPrice',
      key: 'currentPrice',
      width: 100,
      align: 'right',
      render: (price: number) => `¥${price.toFixed(2)}`,
    },
    {
      title: '市值',
      dataIndex: 'marketValue',
      key: 'marketValue',
      width: 120,
      align: 'right',
      render: (value: number) => `¥${value.toLocaleString(undefined, { minimumFractionDigits: 2 })}`,
      sorter: (a, b) => a.marketValue - b.marketValue,
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
      title: '日涨跌',
      dataIndex: 'dailyChange',
      key: 'dailyChange',
      width: 120,
      align: 'right',
      render: (change: number, record) => (
        <Space direction="vertical" size={0}>
          <span style={{ color: change >= 0 ? '#3f8600' : '#cf1322' }}>
            {change >= 0 ? '+' : ''}¥{change.toFixed(2)}
          </span>
          <span style={{ color: record.dailyChangePercent >= 0 ? '#3f8600' : '#cf1322', fontSize: '12px' }}>
            {record.dailyChangePercent >= 0 ? '+' : ''}{record.dailyChangePercent.toFixed(2)}%
          </span>
        </Space>
      ),
      sorter: (a, b) => a.dailyChange - b.dailyChange,
    },
    {
      title: '未实现盈亏',
      dataIndex: 'unrealizedPnl',
      key: 'unrealizedPnl',
      width: 120,
      align: 'right',
      render: (pnl: number, record) => (
        <Space direction="vertical" size={0}>
          <span style={{ color: pnl >= 0 ? '#3f8600' : '#cf1322' }}>
            {pnl >= 0 ? '+' : ''}¥{pnl.toFixed(2)}
          </span>
          <span style={{ color: record.totalReturnPercent >= 0 ? '#3f8600' : '#cf1322', fontSize: '12px' }}>
            {record.totalReturnPercent >= 0 ? '+' : ''}{record.totalReturnPercent.toFixed(2)}%
          </span>
        </Space>
      ),
      sorter: (a, b) => a.unrealizedPnl - b.unrealizedPnl,
    },
    {
      title: '操作',
      key: 'actions',
      width: 100,
      fixed: 'right',
      render: (_, record) => (
        <Space>
          <Tooltip title="查看详情">
            <Button
              type="text"
              size="small"
              icon={<BarChartOutlined />}
              onClick={() => setSelectedSymbol(record.symbol)}
            />
          </Tooltip>
        </Space>
      ),
    },
  ];

  return (
    <LoadingState loading={portfolioLoading} error={portfolioError}>
      <div style={{ padding: '0 0 24px 0' }}>
        {/* Summary Statistics */}
        <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
          <Col xs={24} sm={12} md={6}>
            <StatisticCard
              title="总市值"
              value={summaryStats.totalMarketValue}
              precision={2}
              prefix="¥"
              valueStyle={{ color: '#1890ff' }}
            />
          </Col>
          <Col xs={24} sm={12} md={6}>
            <StatisticCard
              title="未实现盈亏"
              value={summaryStats.totalUnrealizedPnl}
              precision={2}
              prefix="¥"
              valueStyle={{ 
                color: summaryStats.totalUnrealizedPnl >= 0 ? '#3f8600' : '#cf1322' 
              }}
              trend={summaryStats.totalUnrealizedPnl >= 0 ? 'up' : 'down'}
            />
          </Col>
          <Col xs={24} sm={12} md={6}>
            <StatisticCard
              title="日盈亏"
              value={summaryStats.totalDailyChange}
              precision={2}
              prefix="¥"
              valueStyle={{ 
                color: summaryStats.totalDailyChange >= 0 ? '#3f8600' : '#cf1322' 
              }}
              trend={summaryStats.totalDailyChange >= 0 ? 'up' : 'down'}
            />
          </Col>
          <Col xs={24} sm={12} md={6}>
            <StatisticCard
              title="持仓数量"
              value={summaryStats.positionCount}
              suffix="只"
              valueStyle={{ color: '#722ed1' }}
            />
          </Col>
        </Row>

        {/* Controls */}
        <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
          <Col xs={24} sm={12} md={8}>
            <Search
              placeholder="搜索股票代码"
              allowClear
              value={searchText}
              onChange={(e) => setSearchText(e.target.value)}
              style={{ width: '100%' }}
            />
          </Col>
          <Col xs={24} sm={12} md={8}>
            <RangePicker
              value={dateRange}
              onChange={setDateRange}
              style={{ width: '100%' }}
              placeholder={['开始日期', '结束日期']}
            />
          </Col>
          <Col xs={24} sm={24} md={8}>
            <Space style={{ width: '100%', justifyContent: 'flex-end' }}>
              <Select
                value={chartType}
                onChange={setChartType}
                style={{ width: 120 }}
              >
                <Select.Option value="pie">饼图</Select.Option>
                <Select.Option value="bar">柱状图</Select.Option>
              </Select>
              <Button
                icon={<ReloadOutlined />}
                onClick={() => refetchPortfolio()}
                loading={portfolioLoading}
              >
                刷新
              </Button>
            </Space>
          </Col>
        </Row>

        {/* Main Content */}
        <Row gutter={[16, 16]}>
          {/* Positions Table */}
          <Col xs={24} xl={16}>
            <Card title="持仓明细" size="small">
              <Table
                columns={columns}
                dataSource={filteredPositions}
                pagination={{
                  pageSize: 10,
                  showSizeChanger: true,
                  showQuickJumper: true,
                  showTotal: (total) => `共 ${total} 只股票`,
                }}
                scroll={{ x: 1000 }}
                size="small"
                loading={portfolioLoading}
              />
            </Card>
          </Col>

          {/* Position Distribution Chart */}
          <Col xs={24} xl={8}>
            <PositionChart
              data={positionsData}
              type={chartType}
              loading={portfolioLoading}
            />
          </Col>
        </Row>

        {/* Position History */}
        {selectedSymbol && (
          <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
            <Col span={24}>
              <PositionHistory
                symbol={selectedSymbol}
                onClose={() => setSelectedSymbol(null)}
                dateRange={dateRange}
              />
            </Col>
          </Row>
        )}
      </div>
    </LoadingState>
  );
};

export default PositionManagement;
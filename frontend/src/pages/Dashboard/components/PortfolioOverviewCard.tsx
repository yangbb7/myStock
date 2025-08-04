import React, { useMemo } from 'react';
import { 
  Card, 
  Row, 
  Col, 
  Statistic, 
  Progress, 
  Typography, 
  Space, 
  Tooltip,
  Alert,
  Spin,
  Tag
} from 'antd';
import { 
  DollarCircleOutlined,
  RiseOutlined,
  FallOutlined,
  PieChartOutlined,
  BarChartOutlined,
  InfoCircleOutlined,
  WalletOutlined,
  StockOutlined
} from '@ant-design/icons';
import { Line } from '@ant-design/charts';
import { usePortfolioSummary, usePortfolioHistory } from '../../../hooks/useApi';
import { formatCurrency, formatPercent } from '../../../utils/format';
import { Position } from '../../../services/types';

const { Text, Title } = Typography;

interface PortfolioTrendData {
  date: string;
  value: number;
  pnl: number;
}

const PortfolioOverviewCard: React.FC = () => {
  const { data: portfolioData, isLoading, error, refetch } = usePortfolioSummary();
  const { data: historyData } = usePortfolioHistory({
    startDate: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString().split('T')[0], // Last 7 days
    endDate: new Date().toISOString().split('T')[0]
  });

  // Calculate derived metrics
  const metrics = useMemo(() => {
    if (!portfolioData) return null;

    const totalValue = portfolioData.totalValue || 0;
    const cashBalance = portfolioData.cashBalance || 0;
    const unrealizedPnl = portfolioData.unrealizedPnl || 0;
    const realizedPnl = portfolioData.realizedPnl || 0;
    const totalPnl = unrealizedPnl + realizedPnl;
    const totalReturn = portfolioData.totalReturn || 0;
    const totalReturnPercent = portfolioData.totalReturnPercent || 0;
    
    // Calculate position metrics
    const positions = Object.values(portfolioData.positions || {}) as Position[];
    const totalPositionValue = positions.reduce((sum, pos) => sum + (pos.quantity * pos.currentPrice), 0);
    const cashRatio = totalValue > 0 ? (cashBalance / totalValue) * 100 : 0;
    const positionRatio = totalValue > 0 ? (totalPositionValue / totalValue) * 100 : 0;
    
    // Find top positions
    const topPositions = positions
      .sort((a, b) => (b.quantity * b.currentPrice) - (a.quantity * a.currentPrice))
      .slice(0, 5);
    
    // Calculate position distribution
    const positionDistribution = positions.map(pos => ({
      symbol: pos.symbol,
      value: pos.quantity * pos.currentPrice,
      weight: totalPositionValue > 0 ? ((pos.quantity * pos.currentPrice) / totalPositionValue) * 100 : 0,
      pnl: pos.unrealizedPnl || 0,
      pnlPercent: pos.averagePrice > 0 ? ((pos.currentPrice - pos.averagePrice) / pos.averagePrice) * 100 : 0
    }));

    return {
      totalValue,
      cashBalance,
      unrealizedPnl,
      realizedPnl,
      totalPnl,
      totalReturn,
      totalReturnPercent,
      cashRatio,
      positionRatio,
      topPositions,
      positionDistribution,
      positionsCount: positions.length
    };
  }, [portfolioData]);

  // Prepare trend chart data
  const trendData = useMemo(() => {
    if (!historyData || !Array.isArray(historyData)) return [];
    
    return historyData.map((item: any) => ({
      date: new Date(item.timestamp).toLocaleDateString(),
      value: item.totalValue || 0,
      pnl: item.totalPnl || 0
    }));
  }, [historyData]);

  const chartConfig = {
    data: trendData,
    xField: 'date',
    yField: 'value',
    height: 200,
    smooth: true,
    point: {
      size: 3,
      shape: 'circle',
    },
    line: {
      color: (metrics?.totalPnl || 0) >= 0 ? '#52c41a' : '#ff4d4f',
    },
    tooltip: {
      formatter: (datum: PortfolioTrendData) => ({
        name: '组合价值',
        value: formatCurrency(datum.value),
      }),
    },
    yAxis: {
      label: {
        formatter: (value: number) => formatCurrency(value, '¥', 0),
      },
    },
  };

  if (isLoading) {
    return (
      <Card title="投资组合概览" loading>
        <div style={{ height: '400px', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
          <Spin size="large" />
        </div>
      </Card>
    );
  }

  if (error) {
    return (
      <Card title="投资组合概览">
        <Alert
          message="数据加载失败"
          description={error.message}
          type="error"
          showIcon
          action={
            <Text 
              style={{ cursor: 'pointer', color: '#1890ff' }} 
              onClick={() => refetch()}
            >
              重试
            </Text>
          }
        />
      </Card>
    );
  }

  if (!metrics) {
    return (
      <Card title="投资组合概览">
        <Alert
          message="暂无数据"
          description="投资组合数据暂不可用"
          type="warning"
          showIcon
        />
      </Card>
    );
  }

  return (
    <Card 
      title={
        <Space>
          <PieChartOutlined />
          <Title level={4} style={{ margin: 0 }}>投资组合概览</Title>
        </Space>
      }
      extra={
        <Tooltip title="点击刷新数据">
          <Text 
            style={{ cursor: 'pointer', color: '#1890ff' }} 
            onClick={() => refetch()}
          >
            刷新
          </Text>
        </Tooltip>
      }
    >
      <Space direction="vertical" size="large" style={{ width: '100%' }}>
        {/* Main Portfolio Statistics */}
        <Row gutter={16}>
          <Col span={12}>
            <Statistic
              title="总资产价值"
              value={metrics.totalValue}
              precision={2}
              prefix={<DollarCircleOutlined />}
              formatter={(value) => formatCurrency(Number(value))}
              valueStyle={{ 
                fontSize: '24px', 
                fontWeight: 'bold',
                color: '#1890ff'
              }}
            />
          </Col>
          <Col span={12}>
            <Statistic
              title="总盈亏"
              value={metrics.totalPnl}
              precision={2}
              prefix={metrics.totalPnl >= 0 ? <RiseOutlined /> : <FallOutlined />}
              formatter={(value) => formatCurrency(Number(value))}
              valueStyle={{ 
                fontSize: '20px',
                color: metrics.totalPnl >= 0 ? '#52c41a' : '#ff4d4f'
              }}
            />
          </Col>
        </Row>

        {/* Secondary Statistics */}
        <Row gutter={16}>
          <Col span={8}>
            <Statistic
              title="现金余额"
              value={metrics.cashBalance}
              precision={2}
              prefix={<WalletOutlined />}
              formatter={(value) => formatCurrency(Number(value))}
              valueStyle={{ fontSize: '16px' }}
            />
          </Col>
          <Col span={8}>
            <Statistic
              title="持仓数量"
              value={metrics.positionsCount}
              suffix="只"
              prefix={<StockOutlined />}
              valueStyle={{ fontSize: '16px' }}
            />
          </Col>
          <Col span={8}>
            <Statistic
              title="总收益率"
              value={metrics.totalReturnPercent}
              precision={2}
              suffix="%"
              prefix={metrics.totalReturnPercent >= 0 ? <RiseOutlined /> : <FallOutlined />}
              valueStyle={{ 
                fontSize: '16px',
                color: metrics.totalReturnPercent >= 0 ? '#52c41a' : '#ff4d4f'
              }}
            />
          </Col>
        </Row>

        {/* Asset Allocation */}
        <div>
          <Title level={5}>
            <Space>
              <BarChartOutlined />
              资产配置
            </Space>
          </Title>
          <Row gutter={16}>
            <Col span={12}>
              <div style={{ marginBottom: '8px' }}>
                <Text>现金比例</Text>
                <Text style={{ float: 'right' }}>{formatPercent(metrics.cashRatio)}</Text>
              </div>
              <Progress 
                percent={metrics.cashRatio} 
                strokeColor="#1890ff"
                showInfo={false}
              />
            </Col>
            <Col span={12}>
              <div style={{ marginBottom: '8px' }}>
                <Text>持仓比例</Text>
                <Text style={{ float: 'right' }}>{formatPercent(metrics.positionRatio)}</Text>
              </div>
              <Progress 
                percent={metrics.positionRatio} 
                strokeColor="#52c41a"
                showInfo={false}
              />
            </Col>
          </Row>
        </div>

        {/* Portfolio Trend Chart */}
        {trendData.length > 0 && (
          <div>
            <Title level={5}>价值趋势 (近7天)</Title>
            <Line {...chartConfig} />
          </div>
        )}

        {/* Top Positions */}
        {metrics.topPositions.length > 0 && (
          <div>
            <Title level={5}>
              <Space>
                <StockOutlined />
                主要持仓
                <Tooltip title="按市值排序的前5大持仓">
                  <InfoCircleOutlined style={{ color: '#8c8c8c' }} />
                </Tooltip>
              </Space>
            </Title>
            <Space direction="vertical" size="small" style={{ width: '100%' }}>
              {metrics.positionDistribution.slice(0, 5).map((position, index) => (
                <div key={position.symbol} style={{ 
                  display: 'flex', 
                  justifyContent: 'space-between', 
                  alignItems: 'center',
                  padding: '8px 0',
                  borderBottom: index < 4 ? '1px solid #f0f0f0' : 'none'
                }}>
                  <Space>
                    <Text strong>{position.symbol}</Text>
                    <Tag color={position.pnl >= 0 ? 'green' : 'red'}>
                      {formatPercent(position.pnlPercent)}
                    </Tag>
                  </Space>
                  <Space>
                    <Text>{formatCurrency(position.value)}</Text>
                    <Text type="secondary">({formatPercent(position.weight)})</Text>
                  </Space>
                </div>
              ))}
            </Space>
          </div>
        )}

        {/* Performance Summary */}
        <Row gutter={16}>
          <Col span={8}>
            <div style={{ textAlign: 'center', padding: '16px', backgroundColor: '#f6ffed', borderRadius: '6px' }}>
              <Text type="secondary">未实现盈亏</Text>
              <div style={{ fontSize: '18px', fontWeight: 'bold', color: metrics.unrealizedPnl >= 0 ? '#52c41a' : '#ff4d4f' }}>
                {formatCurrency(metrics.unrealizedPnl)}
              </div>
            </div>
          </Col>
          <Col span={8}>
            <div style={{ textAlign: 'center', padding: '16px', backgroundColor: '#f0f5ff', borderRadius: '6px' }}>
              <Text type="secondary">已实现盈亏</Text>
              <div style={{ fontSize: '18px', fontWeight: 'bold', color: metrics.realizedPnl >= 0 ? '#52c41a' : '#ff4d4f' }}>
                {formatCurrency(metrics.realizedPnl)}
              </div>
            </div>
          </Col>
          <Col span={8}>
            <div style={{ textAlign: 'center', padding: '16px', backgroundColor: '#fff7e6', borderRadius: '6px' }}>
              <Text type="secondary">总收益</Text>
              <div style={{ fontSize: '18px', fontWeight: 'bold', color: metrics.totalReturn >= 0 ? '#52c41a' : '#ff4d4f' }}>
                {formatCurrency(metrics.totalReturn)}
              </div>
            </div>
          </Col>
        </Row>
      </Space>
    </Card>
  );
};

export default PortfolioOverviewCard;
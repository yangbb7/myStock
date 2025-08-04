import React, { useState, useEffect, useMemo } from 'react';
import {
  Card,
  Row,
  Col,
  Button,
  Space,
  Typography,
  Select,
  InputNumber,
  Table,
  Statistic,
  Progress,
  Alert,
  Tag,
  Divider,
  Modal,
  Form,
  Switch,
  Tooltip,
  message,
  Badge,
} from 'antd';
import {
  PlayCircleOutlined,
  PauseCircleOutlined,
  StopOutlined,
  ReloadOutlined,
  SettingOutlined,
  TrophyOutlined,
  LineChartOutlined,
  BookOutlined,
  BulbOutlined,
  ExclamationCircleOutlined,
} from '@ant-design/icons';
import ReactECharts from 'echarts-for-react';
import dayjs from 'dayjs';
import { useQuery, useMutation } from '@tanstack/react-query';
import { api } from '../../services/api';
import { useWebSocket } from '../../hooks/useWebSocket';
import type { 
  StrategyConfig, 
  OrderStatus, 
  PortfolioSummary, 
  MarketData,
  PerformanceMetrics 
} from '../../services/types';

const { Title, Text } = Typography;
const { Option } = Select;

interface SimulationConfig {
  strategyName: string;
  symbols: string[];
  virtualCapital: number;
  riskLevel: 'conservative' | 'moderate' | 'aggressive';
  autoTrade: boolean;
  maxPositions: number;
  stopLoss?: number;
  takeProfit?: number;
}

interface SimulationState {
  status: 'idle' | 'running' | 'paused' | 'stopped';
  startTime?: string;
  duration: number;
  totalTrades: number;
  successfulTrades: number;
  currentPnl: number;
  currentPnlPercent: number;
}

interface LearningInsight {
  type: 'success' | 'warning' | 'error' | 'info';
  title: string;
  description: string;
  suggestion?: string;
  timestamp: string;
}

export const SimulationTrading: React.FC<{ className?: string }> = ({ className }) => {
  const [simulationConfig, setSimulationConfig] = useState<SimulationConfig>({
    strategyName: '',
    symbols: [],
    virtualCapital: 100000,
    riskLevel: 'moderate',
    autoTrade: true,
    maxPositions: 5,
  });
  
  const [simulationState, setSimulationState] = useState<SimulationState>({
    status: 'idle',
    duration: 0,
    totalTrades: 0,
    successfulTrades: 0,
    currentPnl: 0,
    currentPnlPercent: 0,
  });

  const [virtualPortfolio, setVirtualPortfolio] = useState<PortfolioSummary | null>(null);
  const [simulationOrders, setSimulationOrders] = useState<OrderStatus[]>([]);
  const [learningInsights, setLearningInsights] = useState<LearningInsight[]>([]);
  const [showConfigModal, setShowConfigModal] = useState(false);
  const [equityHistory, setEquityHistory] = useState<Array<{ timestamp: string; equity: number }>>([]);

  // Fetch available strategies
  const { data: strategies } = useQuery({
    queryKey: ['strategies'],
    queryFn: () => api.strategy.getStrategies(),
  });

  // Fetch available symbols
  const { data: symbols } = useQuery({
    queryKey: ['symbols'],
    queryFn: () => api.data.getSymbols(),
  });

  // WebSocket for real-time market data
  const { isConnected } = useWebSocket('/ws/market-data', {
    onMessage: (event) => {
      if (simulationState.status === 'running') {
        const marketData: MarketData = JSON.parse(event.data);
        handleMarketDataUpdate(marketData);
      }
    },
  });

  // Simulation timer
  useEffect(() => {
    let interval: NodeJS.Timeout;
    
    if (simulationState.status === 'running') {
      interval = setInterval(() => {
        setSimulationState(prev => ({
          ...prev,
          duration: prev.duration + 1,
        }));
      }, 1000);
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [simulationState.status]);

  // Initialize virtual portfolio
  useEffect(() => {
    if (simulationState.status === 'running' && !virtualPortfolio) {
      setVirtualPortfolio({
        totalValue: simulationConfig.virtualCapital,
        cashBalance: simulationConfig.virtualCapital,
        positions: {},
        unrealizedPnl: 0,
        positionsCount: 0,
      });
      
      setEquityHistory([{
        timestamp: new Date().toISOString(),
        equity: simulationConfig.virtualCapital,
      }]);
    }
  }, [simulationState.status, simulationConfig.virtualCapital, virtualPortfolio]);

  const handleMarketDataUpdate = (marketData: MarketData) => {
    // Simulate strategy signals and trading logic
    if (simulationConfig.autoTrade && virtualPortfolio) {
      // Simple simulation logic - this would be replaced with actual strategy logic
      const shouldTrade = Math.random() < 0.1; // 10% chance to generate signal
      
      if (shouldTrade && Object.keys(virtualPortfolio.positions).length < simulationConfig.maxPositions) {
        const side = Math.random() < 0.6 ? 'BUY' : 'SELL'; // 60% buy bias
        const quantity = Math.floor(virtualPortfolio.cashBalance * 0.1 / marketData.price);
        
        if (quantity > 0) {
          simulateOrder({
            symbol: marketData.symbol,
            side,
            quantity,
            price: marketData.price,
          });
        }
      }
    }

    // Update portfolio value based on current prices
    if (virtualPortfolio) {
      updatePortfolioValue(marketData);
    }
  };

  const simulateOrder = (orderData: {
    symbol: string;
    side: 'BUY' | 'SELL';
    quantity: number;
    price: number;
  }) => {
    const orderId = `sim_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const order: OrderStatus = {
      orderId,
      symbol: orderData.symbol,
      side: orderData.side,
      quantity: orderData.quantity,
      price: orderData.price,
      orderType: 'MARKET',
      status: 'FILLED',
      timestamp: new Date().toISOString(),
      executedPrice: orderData.price * (1 + (Math.random() - 0.5) * 0.002), // Small slippage
      executedQuantity: orderData.quantity,
    };

    setSimulationOrders(prev => [order, ...prev]);
    
    // Update portfolio
    if (virtualPortfolio) {
      const cost = order.executedPrice! * order.executedQuantity!;
      const newPortfolio = { ...virtualPortfolio };
      
      if (order.side === 'BUY') {
        newPortfolio.cashBalance -= cost;
        if (newPortfolio.positions[order.symbol]) {
          const existing = newPortfolio.positions[order.symbol];
          const totalQuantity = existing.quantity + order.executedQuantity!;
          const totalCost = existing.averagePrice * existing.quantity + cost;
          newPortfolio.positions[order.symbol] = {
            ...existing,
            quantity: totalQuantity,
            averagePrice: totalCost / totalQuantity,
          };
        } else {
          newPortfolio.positions[order.symbol] = {
            symbol: order.symbol,
            quantity: order.executedQuantity!,
            averagePrice: order.executedPrice!,
            currentPrice: order.executedPrice!,
            unrealizedPnl: 0,
          };
          newPortfolio.positionsCount += 1;
        }
      } else {
        // SELL logic
        newPortfolio.cashBalance += cost;
        if (newPortfolio.positions[order.symbol]) {
          const existing = newPortfolio.positions[order.symbol];
          if (existing.quantity >= order.executedQuantity!) {
            const remainingQuantity = existing.quantity - order.executedQuantity!;
            if (remainingQuantity > 0) {
              newPortfolio.positions[order.symbol] = {
                ...existing,
                quantity: remainingQuantity,
              };
            } else {
              delete newPortfolio.positions[order.symbol];
              newPortfolio.positionsCount -= 1;
            }
          }
        }
      }
      
      setVirtualPortfolio(newPortfolio);
    }

    // Update simulation state
    setSimulationState(prev => ({
      ...prev,
      totalTrades: prev.totalTrades + 1,
      successfulTrades: prev.successfulTrades + (Math.random() < 0.65 ? 1 : 0), // 65% success rate
    }));

    // Generate learning insights
    generateLearningInsight(order);
  };

  const updatePortfolioValue = (marketData: MarketData) => {
    if (!virtualPortfolio || !virtualPortfolio.positions[marketData.symbol]) return;

    const newPortfolio = { ...virtualPortfolio };
    const position = newPortfolio.positions[marketData.symbol];
    
    position.currentPrice = marketData.price;
    position.unrealizedPnl = (marketData.price - position.averagePrice) * position.quantity;
    
    // Calculate total portfolio value
    const totalPositionValue = Object.values(newPortfolio.positions).reduce(
      (sum, pos) => sum + pos.currentPrice * pos.quantity, 0
    );
    const totalUnrealizedPnl = Object.values(newPortfolio.positions).reduce(
      (sum, pos) => sum + pos.unrealizedPnl, 0
    );
    
    newPortfolio.totalValue = newPortfolio.cashBalance + totalPositionValue;
    newPortfolio.unrealizedPnl = totalUnrealizedPnl;
    
    setVirtualPortfolio(newPortfolio);

    // Update equity history
    setEquityHistory(prev => [
      ...prev.slice(-100), // Keep last 100 points
      {
        timestamp: new Date().toISOString(),
        equity: newPortfolio.totalValue,
      },
    ]);

    // Update simulation state
    const pnl = newPortfolio.totalValue - simulationConfig.virtualCapital;
    const pnlPercent = (pnl / simulationConfig.virtualCapital) * 100;
    
    setSimulationState(prev => ({
      ...prev,
      currentPnl: pnl,
      currentPnlPercent: pnlPercent,
    }));
  };

  const generateLearningInsight = (order: OrderStatus) => {
    const insights: LearningInsight[] = [];
    
    // Risk management insights
    if (virtualPortfolio && virtualPortfolio.cashBalance < simulationConfig.virtualCapital * 0.1) {
      insights.push({
        type: 'warning',
        title: '现金余额过低',
        description: '当前现金余额不足初始资金的10%，建议控制仓位。',
        suggestion: '考虑减少新开仓位或平仓部分持仓。',
        timestamp: new Date().toISOString(),
      });
    }

    // Position concentration
    if (virtualPortfolio && virtualPortfolio.positionsCount >= simulationConfig.maxPositions) {
      insights.push({
        type: 'info',
        title: '达到最大持仓数量',
        description: `当前持仓数量已达到设定的最大值${simulationConfig.maxPositions}。`,
        suggestion: '考虑平仓部分表现较差的持仓以释放资金。',
        timestamp: new Date().toISOString(),
      });
    }

    // Performance insights
    if (simulationState.totalTrades >= 10) {
      const winRate = (simulationState.successfulTrades / simulationState.totalTrades) * 100;
      if (winRate < 40) {
        insights.push({
          type: 'error',
          title: '胜率偏低',
          description: `当前胜率为${winRate.toFixed(1)}%，低于预期水平。`,
          suggestion: '建议检查策略参数或市场环境是否发生变化。',
          timestamp: new Date().toISOString(),
        });
      } else if (winRate > 70) {
        insights.push({
          type: 'success',
          title: '表现优秀',
          description: `当前胜率为${winRate.toFixed(1)}%，表现良好。`,
          suggestion: '可以考虑适当增加仓位或扩大交易范围。',
          timestamp: new Date().toISOString(),
        });
      }
    }

    if (insights.length > 0) {
      setLearningInsights(prev => [...insights, ...prev.slice(0, 19)]); // Keep last 20 insights
    }
  };

  const handleStartSimulation = () => {
    if (!simulationConfig.strategyName || simulationConfig.symbols.length === 0) {
      message.error('请完成模拟交易配置');
      return;
    }

    setSimulationState({
      status: 'running',
      startTime: new Date().toISOString(),
      duration: 0,
      totalTrades: 0,
      successfulTrades: 0,
      currentPnl: 0,
      currentPnlPercent: 0,
    });

    message.success('模拟交易已开始');
  };

  const handlePauseSimulation = () => {
    setSimulationState(prev => ({ ...prev, status: 'paused' }));
    message.info('模拟交易已暂停');
  };

  const handleResumeSimulation = () => {
    setSimulationState(prev => ({ ...prev, status: 'running' }));
    message.info('模拟交易已恢复');
  };

  const handleStopSimulation = () => {
    setSimulationState(prev => ({ ...prev, status: 'stopped' }));
    message.success('模拟交易已停止');
  };

  const handleResetSimulation = () => {
    setSimulationState({
      status: 'idle',
      duration: 0,
      totalTrades: 0,
      successfulTrades: 0,
      currentPnl: 0,
      currentPnlPercent: 0,
    });
    setVirtualPortfolio(null);
    setSimulationOrders([]);
    setLearningInsights([]);
    setEquityHistory([]);
    message.success('模拟交易已重置');
  };

  // Equity curve chart options
  const equityChartOptions = useMemo(() => {
    if (equityHistory.length === 0) return {};

    const data = equityHistory.map(point => [
      dayjs(point.timestamp).valueOf(),
      point.equity,
    ]);

    return {
      title: {
        text: '模拟资金曲线',
        left: 'center',
      },
      tooltip: {
        trigger: 'axis',
        formatter: (params: any) => {
          const point = params[0];
          return `
            <div>
              <div>时间: ${dayjs(point.data[0]).format('HH:mm:ss')}</div>
              <div>资金: ¥${point.data[1].toLocaleString()}</div>
            </div>
          `;
        },
      },
      xAxis: {
        type: 'time',
        axisLabel: {
          formatter: (value: number) => dayjs(value).format('HH:mm'),
        },
      },
      yAxis: {
        type: 'value',
        axisLabel: {
          formatter: (value: number) => `¥${(value / 1000).toFixed(0)}k`,
        },
      },
      series: [
        {
          name: '资金',
          type: 'line',
          data,
          smooth: true,
          lineStyle: {
            color: simulationState.currentPnl >= 0 ? '#52c41a' : '#ff4d4f',
            width: 2,
          },
        },
      ],
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        containLabel: true,
      },
    };
  }, [equityHistory, simulationState.currentPnl]);

  // Order table columns
  const orderColumns = [
    {
      title: '时间',
      dataIndex: 'timestamp',
      key: 'timestamp',
      width: 100,
      render: (value: string) => dayjs(value).format('HH:mm:ss'),
    },
    {
      title: '股票',
      dataIndex: 'symbol',
      key: 'symbol',
      width: 100,
    },
    {
      title: '方向',
      dataIndex: 'side',
      key: 'side',
      width: 80,
      render: (side: string) => (
        <Tag color={side === 'BUY' ? 'green' : 'red'}>
          {side === 'BUY' ? '买入' : '卖出'}
        </Tag>
      ),
    },
    {
      title: '数量',
      dataIndex: 'executedQuantity',
      key: 'executedQuantity',
      width: 80,
    },
    {
      title: '价格',
      dataIndex: 'executedPrice',
      key: 'executedPrice',
      width: 100,
      render: (value: number) => `¥${value?.toFixed(2)}`,
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 80,
      render: (status: string) => (
        <Tag color="green">已成交</Tag>
      ),
    },
  ];

  const formatDuration = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className={className}>
      <Row gutter={[16, 16]}>
        {/* Control Panel */}
        <Col span={24}>
          <Card>
            <Row justify="space-between" align="middle">
              <Col>
                <Space>
                  <Badge
                    status={
                      simulationState.status === 'running' ? 'processing' :
                      simulationState.status === 'paused' ? 'warning' :
                      simulationState.status === 'stopped' ? 'error' : 'default'
                    }
                    text={
                      simulationState.status === 'running' ? '运行中' :
                      simulationState.status === 'paused' ? '已暂停' :
                      simulationState.status === 'stopped' ? '已停止' : '未开始'
                    }
                  />
                  {simulationState.status !== 'idle' && (
                    <Text>运行时长: {formatDuration(simulationState.duration)}</Text>
                  )}
                  {!isConnected && (
                    <Tag color="red">WebSocket未连接</Tag>
                  )}
                </Space>
              </Col>
              <Col>
                <Space>
                  <Button
                    icon={<SettingOutlined />}
                    onClick={() => setShowConfigModal(true)}
                    disabled={simulationState.status === 'running'}
                  >
                    配置
                  </Button>
                  
                  {simulationState.status === 'idle' && (
                    <Button
                      type="primary"
                      icon={<PlayCircleOutlined />}
                      onClick={handleStartSimulation}
                    >
                      开始模拟
                    </Button>
                  )}
                  
                  {simulationState.status === 'running' && (
                    <Button
                      icon={<PauseCircleOutlined />}
                      onClick={handlePauseSimulation}
                    >
                      暂停
                    </Button>
                  )}
                  
                  {simulationState.status === 'paused' && (
                    <Button
                      type="primary"
                      icon={<PlayCircleOutlined />}
                      onClick={handleResumeSimulation}
                    >
                      继续
                    </Button>
                  )}
                  
                  {(simulationState.status === 'running' || simulationState.status === 'paused') && (
                    <Button
                      danger
                      icon={<StopOutlined />}
                      onClick={handleStopSimulation}
                    >
                      停止
                    </Button>
                  )}
                  
                  <Button
                    icon={<ReloadOutlined />}
                    onClick={handleResetSimulation}
                    disabled={simulationState.status === 'running'}
                  >
                    重置
                  </Button>
                </Space>
              </Col>
            </Row>
          </Card>
        </Col>

        {/* Performance Overview */}
        <Col span={24}>
          <Row gutter={16}>
            <Col span={6}>
              <Card>
                <Statistic
                  title="虚拟资金"
                  value={virtualPortfolio?.totalValue || simulationConfig.virtualCapital}
                  precision={2}
                  prefix="¥"
                  valueStyle={{ color: '#1890ff' }}
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="盈亏"
                  value={simulationState.currentPnl}
                  precision={2}
                  prefix="¥"
                  valueStyle={{ 
                    color: simulationState.currentPnl >= 0 ? '#3f8600' : '#cf1322' 
                  }}
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="盈亏率"
                  value={simulationState.currentPnlPercent}
                  precision={2}
                  suffix="%"
                  valueStyle={{ 
                    color: simulationState.currentPnlPercent >= 0 ? '#3f8600' : '#cf1322' 
                  }}
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="胜率"
                  value={simulationState.totalTrades > 0 ? 
                    (simulationState.successfulTrades / simulationState.totalTrades) * 100 : 0}
                  precision={1}
                  suffix="%"
                  valueStyle={{ color: '#722ed1' }}
                />
              </Card>
            </Col>
          </Row>
        </Col>

        {/* Charts and Data */}
        <Col span={16}>
          <Card title="实时监控">
            {equityHistory.length > 0 ? (
              <ReactECharts
                option={equityChartOptions}
                style={{ height: '400px' }}
                notMerge={true}
                lazyUpdate={true}
              />
            ) : (
              <div style={{ height: '400px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <Text type="secondary">开始模拟交易后将显示资金曲线</Text>
              </div>
            )}
          </Card>
        </Col>

        <Col span={8}>
          <Card title="学习建议" style={{ height: '460px', overflow: 'auto' }}>
            {learningInsights.length === 0 ? (
              <div style={{ textAlign: 'center', padding: '40px 0' }}>
                <BulbOutlined style={{ fontSize: '48px', color: '#d9d9d9' }} />
                <div style={{ marginTop: '16px' }}>
                  <Text type="secondary">开始交易后将显示学习建议</Text>
                </div>
              </div>
            ) : (
              <Space direction="vertical" style={{ width: '100%' }}>
                {learningInsights.map((insight, index) => (
                  <Alert
                    key={index}
                    type={insight.type}
                    message={insight.title}
                    description={
                      <div>
                        <div>{insight.description}</div>
                        {insight.suggestion && (
                          <div style={{ marginTop: '8px', fontStyle: 'italic' }}>
                            💡 {insight.suggestion}
                          </div>
                        )}
                      </div>
                    }
                    showIcon
                    style={{ marginBottom: '8px' }}
                  />
                ))}
              </Space>
            )}
          </Card>
        </Col>

        {/* Orders Table */}
        <Col span={24}>
          <Card title="交易记录">
            <Table
              columns={orderColumns}
              dataSource={simulationOrders}
              rowKey="orderId"
              pagination={{ pageSize: 10 }}
              scroll={{ y: 300 }}
            />
          </Card>
        </Col>
      </Row>

      {/* Configuration Modal */}
      <Modal
        title="模拟交易配置"
        open={showConfigModal}
        onCancel={() => setShowConfigModal(false)}
        onOk={() => {
          setShowConfigModal(false);
          message.success('配置已保存');
        }}
        width={600}
      >
        <Form layout="vertical">
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item label="策略名称">
                <Select
                  value={simulationConfig.strategyName}
                  onChange={(value) => setSimulationConfig(prev => ({ ...prev, strategyName: value }))}
                  placeholder="选择策略"
                >
                  {strategies?.map(strategy => (
                    <Option key={strategy} value={strategy}>
                      {strategy}
                    </Option>
                  ))}
                </Select>
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item label="股票代码">
                <Select
                  mode="multiple"
                  value={simulationConfig.symbols}
                  onChange={(value) => setSimulationConfig(prev => ({ ...prev, symbols: value }))}
                  placeholder="选择股票"
                >
                  {symbols?.map(symbol => (
                    <Option key={symbol} value={symbol}>
                      {symbol}
                    </Option>
                  ))}
                </Select>
              </Form.Item>
            </Col>
          </Row>

          <Row gutter={16}>
            <Col span={12}>
              <Form.Item label="虚拟资金">
                <InputNumber
                  style={{ width: '100%' }}
                  value={simulationConfig.virtualCapital}
                  onChange={(value) => setSimulationConfig(prev => ({ ...prev, virtualCapital: value || 100000 }))}
                  min={10000}
                  max={10000000}
                  step={10000}
                  formatter={(value) => `¥ ${value}`.replace(/\B(?=(\d{3})+(?!\d))/g, ',')}
                  parser={(value) => value!.replace(/¥\s?|(,*)/g, '')}
                />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item label="风险等级">
                <Select
                  value={simulationConfig.riskLevel}
                  onChange={(value) => setSimulationConfig(prev => ({ ...prev, riskLevel: value }))}
                >
                  <Option value="conservative">保守</Option>
                  <Option value="moderate">稳健</Option>
                  <Option value="aggressive">激进</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>

          <Row gutter={16}>
            <Col span={12}>
              <Form.Item label="最大持仓数">
                <InputNumber
                  style={{ width: '100%' }}
                  value={simulationConfig.maxPositions}
                  onChange={(value) => setSimulationConfig(prev => ({ ...prev, maxPositions: value || 5 }))}
                  min={1}
                  max={20}
                />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item label="自动交易">
                <Switch
                  checked={simulationConfig.autoTrade}
                  onChange={(checked) => setSimulationConfig(prev => ({ ...prev, autoTrade: checked }))}
                />
              </Form.Item>
            </Col>
          </Row>
        </Form>
      </Modal>
    </div>
  );
};

export default SimulationTrading;
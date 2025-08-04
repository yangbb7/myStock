import React from 'react';
import { BrowserRouter as Router } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ThemeProvider } from './ThemeProvider';
import { AppRoutes } from '@/routes';
import { useSystemStore } from '../../stores/systemStore';
import { useSystemHealth, useSystemMetrics, useSystemControl } from '../../hooks/useApi';

// Import actual page components - using safe fallbacks for now
import { Card, Row, Col } from 'antd';
import { StatisticCard } from '../Common/StatisticCard';

// Temporary safe page components to avoid errors

const StrategyPage = () => {
  const [strategies, setStrategies] = React.useState([
    {
      id: 1,
      name: '均线策略',
      symbol: '000001.SZ',
      status: 'running',
      performance: {
        totalReturn: 15.6,
        sharpeRatio: 1.2,
        maxDrawdown: -8.5,
        winRate: 65.4,
        signalCount: 156,
        successfulTrades: 102
      },
      config: {
        initialCapital: 1000000,
        maxPosition: 0.3,
        stopLoss: 0.05,
        takeProfit: 0.1
      }
    },
    {
      id: 2,
      name: 'RSI策略',
      symbol: '000002.SZ',
      status: 'stopped',
      performance: {
        totalReturn: 8.3,
        sharpeRatio: 0.9,
        maxDrawdown: -12.1,
        winRate: 58.7,
        signalCount: 89,
        successfulTrades: 52
      },
      config: {
        initialCapital: 500000,
        maxPosition: 0.2,
        stopLoss: 0.03,
        takeProfit: 0.08
      }
    }
  ]);

  const [selectedStrategy, setSelectedStrategy] = React.useState(null);
  const [showConfigModal, setShowConfigModal] = React.useState(false);

  const columns = [
    {
      title: '策略名称',
      dataIndex: 'name',
      key: 'name',
      render: (text, record) => (
        <div>
          <div style={{ fontWeight: 'bold' }}>{text}</div>
          <div style={{ fontSize: '12px', color: '#666' }}>{record.symbol}</div>
        </div>
      )
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status) => (
        <span style={{ 
          color: status === 'running' ? '#52c41a' : '#ff4d4f',
          fontWeight: 'bold'
        }}>
          {status === 'running' ? '运行中' : '已停止'}
        </span>
      )
    },
    {
      title: '总收益率',
      dataIndex: ['performance', 'totalReturn'],
      key: 'totalReturn',
      render: (value) => (
        <span style={{ color: value >= 0 ? '#52c41a' : '#ff4d4f' }}>
          {value >= 0 ? '+' : ''}{value}%
        </span>
      )
    },
    {
      title: '夏普比率',
      dataIndex: ['performance', 'sharpeRatio'],
      key: 'sharpeRatio',
      render: (value) => value.toFixed(2)
    },
    {
      title: '胜率',
      dataIndex: ['performance', 'winRate'],
      key: 'winRate',
      render: (value) => `${value}%`
    },
    {
      title: '操作',
      key: 'action',
      render: (_, record) => (
        <div>
          <button 
            style={{ 
              marginRight: '8px', 
              padding: '4px 8px',
              border: '1px solid #1890ff',
              background: record.status === 'running' ? '#ff4d4f' : '#52c41a',
              color: 'white',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
            onClick={() => {
              const newStatus = record.status === 'running' ? 'stopped' : 'running';
              setStrategies(prev => prev.map(s => 
                s.id === record.id ? { ...s, status: newStatus } : s
              ));
            }}
          >
            {record.status === 'running' ? '停止' : '启动'}
          </button>
          <button 
            style={{ 
              padding: '4px 8px',
              border: '1px solid #1890ff',
              background: '#1890ff',
              color: 'white',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
            onClick={() => {
              setSelectedStrategy(record);
              setShowConfigModal(true);
            }}
          >
            配置
          </button>
        </div>
      )
    }
  ];

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h2 style={{ margin: 0 }}>策略管理</h2>
        <button 
          style={{ 
            padding: '8px 16px',
            border: '1px solid #1890ff',
            background: '#1890ff',
            color: 'white',
            borderRadius: '4px',
            cursor: 'pointer'
          }}
          onClick={() => {
            setSelectedStrategy(null);
            setShowConfigModal(true);
          }}
        >
          添加策略
        </button>
      </div>

      {/* 策略概览卡片 */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col span={6}>
          <Card>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#1890ff' }}>
                {strategies.length}
              </div>
              <div style={{ color: '#666' }}>总策略数</div>
            </div>
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#52c41a' }}>
                {strategies.filter(s => s.status === 'running').length}
              </div>
              <div style={{ color: '#666' }}>运行中</div>
            </div>
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#52c41a' }}>
                +{strategies.reduce((sum, s) => sum + s.performance.totalReturn, 0).toFixed(1)}%
              </div>
              <div style={{ color: '#666' }}>总收益率</div>
            </div>
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#1890ff' }}>
                {strategies.reduce((sum, s) => sum + s.performance.signalCount, 0)}
              </div>
              <div style={{ color: '#666' }}>总信号数</div>
            </div>
          </Card>
        </Col>
      </Row>

      {/* 策略列表 */}
      <Card title="策略列表">
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid #f0f0f0' }}>
              {columns.map(col => (
                <th key={col.key} style={{ padding: '12px', textAlign: 'left', fontWeight: 'bold' }}>
                  {col.title}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {strategies.map(strategy => (
              <tr key={strategy.id} style={{ borderBottom: '1px solid #f0f0f0' }}>
                <td style={{ padding: '12px' }}>
                  <div>
                    <div style={{ fontWeight: 'bold' }}>{strategy.name}</div>
                    <div style={{ fontSize: '12px', color: '#666' }}>{strategy.symbol}</div>
                  </div>
                </td>
                <td style={{ padding: '12px' }}>
                  <span style={{ 
                    color: strategy.status === 'running' ? '#52c41a' : '#ff4d4f',
                    fontWeight: 'bold'
                  }}>
                    {strategy.status === 'running' ? '运行中' : '已停止'}
                  </span>
                </td>
                <td style={{ padding: '12px' }}>
                  <span style={{ color: strategy.performance.totalReturn >= 0 ? '#52c41a' : '#ff4d4f' }}>
                    {strategy.performance.totalReturn >= 0 ? '+' : ''}{strategy.performance.totalReturn}%
                  </span>
                </td>
                <td style={{ padding: '12px' }}>{strategy.performance.sharpeRatio.toFixed(2)}</td>
                <td style={{ padding: '12px' }}>{strategy.performance.winRate}%</td>
                <td style={{ padding: '12px' }}>
                  <button 
                    style={{ 
                      marginRight: '8px', 
                      padding: '4px 8px',
                      border: '1px solid #1890ff',
                      background: strategy.status === 'running' ? '#ff4d4f' : '#52c41a',
                      color: 'white',
                      borderRadius: '4px',
                      cursor: 'pointer'
                    }}
                    onClick={() => {
                      const newStatus = strategy.status === 'running' ? 'stopped' : 'running';
                      setStrategies(prev => prev.map(s => 
                        s.id === strategy.id ? { ...s, status: newStatus } : s
                      ));
                    }}
                  >
                    {strategy.status === 'running' ? '停止' : '启动'}
                  </button>
                  <button 
                    style={{ 
                      padding: '4px 8px',
                      border: '1px solid #1890ff',
                      background: '#1890ff',
                      color: 'white',
                      borderRadius: '4px',
                      cursor: 'pointer'
                    }}
                    onClick={() => {
                      setSelectedStrategy(strategy);
                      setShowConfigModal(true);
                    }}
                  >
                    配置
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      {/* 配置模态框 */}
      {showConfigModal && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(0,0,0,0.5)',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          zIndex: 1000
        }}>
          <div style={{
            background: 'white',
            padding: '24px',
            borderRadius: '8px',
            width: '500px',
            maxHeight: '80vh',
            overflow: 'auto'
          }}>
            <h3>{selectedStrategy ? '编辑策略' : '添加策略'}</h3>
            <div style={{ marginBottom: '16px' }}>
              <label style={{ display: 'block', marginBottom: '4px' }}>策略名称</label>
              <input 
                type="text" 
                defaultValue={selectedStrategy?.name || ''}
                style={{ width: '100%', padding: '8px', border: '1px solid #d9d9d9', borderRadius: '4px' }}
              />
            </div>
            <div style={{ marginBottom: '16px' }}>
              <label style={{ display: 'block', marginBottom: '4px' }}>股票代码</label>
              <input 
                type="text" 
                defaultValue={selectedStrategy?.symbol || ''}
                style={{ width: '100%', padding: '8px', border: '1px solid #d9d9d9', borderRadius: '4px' }}
              />
            </div>
            <div style={{ marginBottom: '16px' }}>
              <label style={{ display: 'block', marginBottom: '4px' }}>初始资金</label>
              <input 
                type="number" 
                defaultValue={selectedStrategy?.config.initialCapital || 1000000}
                style={{ width: '100%', padding: '8px', border: '1px solid #d9d9d9', borderRadius: '4px' }}
              />
            </div>
            <div style={{ marginBottom: '16px' }}>
              <label style={{ display: 'block', marginBottom: '4px' }}>最大仓位比例</label>
              <input 
                type="number" 
                step="0.1"
                defaultValue={selectedStrategy?.config.maxPosition || 0.3}
                style={{ width: '100%', padding: '8px', border: '1px solid #d9d9d9', borderRadius: '4px' }}
              />
            </div>
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px' }}>
              <button 
                style={{ 
                  padding: '8px 16px',
                  border: '1px solid #d9d9d9',
                  background: 'white',
                  borderRadius: '4px',
                  cursor: 'pointer'
                }}
                onClick={() => setShowConfigModal(false)}
              >
                取消
              </button>
              <button 
                style={{ 
                  padding: '8px 16px',
                  border: '1px solid #1890ff',
                  background: '#1890ff',
                  color: 'white',
                  borderRadius: '4px',
                  cursor: 'pointer'
                }}
                onClick={() => {
                  // 这里应该调用API保存策略配置
                  setShowConfigModal(false);
                }}
              >
                保存
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

const DataPage = () => {
  const [marketData, setMarketData] = React.useState([
    {
      symbol: '000001.SZ',
      name: '平安银行',
      price: 12.45,
      change: 0.23,
      changePercent: 1.88,
      volume: 1234567,
      turnover: 15367890,
      high: 12.67,
      low: 12.12,
      open: 12.22
    },
    {
      symbol: '000002.SZ',
      name: '万科A',
      price: 18.76,
      change: -0.45,
      changePercent: -2.34,
      volume: 987654,
      turnover: 18523456,
      high: 19.23,
      low: 18.45,
      open: 19.01
    },
    {
      symbol: '600000.SH',
      name: '浦发银行',
      price: 8.92,
      change: 0.12,
      changePercent: 1.36,
      volume: 2345678,
      turnover: 20934567,
      high: 9.05,
      low: 8.78,
      open: 8.80
    }
  ]);

  const [selectedSymbol, setSelectedSymbol] = React.useState('000001.SZ');
  const [searchTerm, setSearchTerm] = React.useState('');

  // 模拟实时数据更新
  React.useEffect(() => {
    const interval = setInterval(() => {
      setMarketData(prev => prev.map(item => ({
        ...item,
        price: item.price + (Math.random() - 0.5) * 0.1,
        change: item.change + (Math.random() - 0.5) * 0.05,
        changePercent: item.changePercent + (Math.random() - 0.5) * 0.2
      })));
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  const filteredData = marketData.filter(item => 
    item.name.includes(searchTerm) || item.symbol.includes(searchTerm)
  );

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h2 style={{ margin: 0 }}>实时数据监控</h2>
        <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
          <input
            type="text"
            placeholder="搜索股票..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            style={{ 
              padding: '8px 12px',
              border: '1px solid #d9d9d9',
              borderRadius: '4px',
              width: '200px'
            }}
          />
          <div style={{ 
            padding: '8px 12px',
            background: '#f0f2f5',
            borderRadius: '4px',
            fontSize: '12px'
          }}>
            🟢 WebSocket已连接
          </div>
        </div>
      </div>

      {/* 市场概览 */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col span={6}>
          <Card>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#52c41a' }}>
                {marketData.filter(item => item.change > 0).length}
              </div>
              <div style={{ color: '#666' }}>上涨</div>
            </div>
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#ff4d4f' }}>
                {marketData.filter(item => item.change < 0).length}
              </div>
              <div style={{ color: '#666' }}>下跌</div>
            </div>
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#1890ff' }}>
                {marketData.reduce((sum, item) => sum + item.volume, 0).toLocaleString()}
              </div>
              <div style={{ color: '#666' }}>总成交量</div>
            </div>
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#722ed1' }}>
                {(marketData.reduce((sum, item) => sum + item.turnover, 0) / 100000000).toFixed(1)}亿
              </div>
              <div style={{ color: '#666' }}>总成交额</div>
            </div>
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]}>
        {/* 市场数据表格 */}
        <Col span={14}>
          <Card title="实时行情">
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid #f0f0f0' }}>
                  <th style={{ padding: '12px', textAlign: 'left', fontWeight: 'bold' }}>股票</th>
                  <th style={{ padding: '12px', textAlign: 'right', fontWeight: 'bold' }}>现价</th>
                  <th style={{ padding: '12px', textAlign: 'right', fontWeight: 'bold' }}>涨跌</th>
                  <th style={{ padding: '12px', textAlign: 'right', fontWeight: 'bold' }}>涨跌幅</th>
                  <th style={{ padding: '12px', textAlign: 'right', fontWeight: 'bold' }}>成交量</th>
                  <th style={{ padding: '12px', textAlign: 'center', fontWeight: 'bold' }}>操作</th>
                </tr>
              </thead>
              <tbody>
                {filteredData.map(item => (
                  <tr 
                    key={item.symbol} 
                    style={{ 
                      borderBottom: '1px solid #f0f0f0',
                      backgroundColor: selectedSymbol === item.symbol ? '#f6ffed' : 'transparent'
                    }}
                  >
                    <td style={{ padding: '12px' }}>
                      <div>
                        <div style={{ fontWeight: 'bold' }}>{item.name}</div>
                        <div style={{ fontSize: '12px', color: '#666' }}>{item.symbol}</div>
                      </div>
                    </td>
                    <td style={{ padding: '12px', textAlign: 'right', fontWeight: 'bold' }}>
                      {item.price.toFixed(2)}
                    </td>
                    <td style={{ 
                      padding: '12px', 
                      textAlign: 'right',
                      color: item.change >= 0 ? '#52c41a' : '#ff4d4f'
                    }}>
                      {item.change >= 0 ? '+' : ''}{item.change.toFixed(2)}
                    </td>
                    <td style={{ 
                      padding: '12px', 
                      textAlign: 'right',
                      color: item.changePercent >= 0 ? '#52c41a' : '#ff4d4f'
                    }}>
                      {item.changePercent >= 0 ? '+' : ''}{item.changePercent.toFixed(2)}%
                    </td>
                    <td style={{ padding: '12px', textAlign: 'right' }}>
                      {item.volume.toLocaleString()}
                    </td>
                    <td style={{ padding: '12px', textAlign: 'center' }}>
                      <button 
                        style={{ 
                          padding: '4px 8px',
                          border: '1px solid #1890ff',
                          background: selectedSymbol === item.symbol ? '#1890ff' : 'white',
                          color: selectedSymbol === item.symbol ? 'white' : '#1890ff',
                          borderRadius: '4px',
                          cursor: 'pointer',
                          fontSize: '12px'
                        }}
                        onClick={() => setSelectedSymbol(item.symbol)}
                      >
                        查看
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        </Col>

        {/* K线图和详细信息 */}
        <Col span={10}>
          <Card title={`${selectedSymbol} 详细信息`}>
            {(() => {
              const selected = marketData.find(item => item.symbol === selectedSymbol);
              if (!selected) return null;
              
              return (
                <div>
                  <div style={{ marginBottom: '16px' }}>
                    <h3 style={{ margin: '0 0 8px 0' }}>{selected.name}</h3>
                    <div style={{ fontSize: '24px', fontWeight: 'bold', marginBottom: '8px' }}>
                      ¥{selected.price.toFixed(2)}
                    </div>
                    <div style={{ 
                      color: selected.change >= 0 ? '#52c41a' : '#ff4d4f',
                      fontSize: '16px'
                    }}>
                      {selected.change >= 0 ? '+' : ''}{selected.change.toFixed(2)} 
                      ({selected.changePercent >= 0 ? '+' : ''}{selected.changePercent.toFixed(2)}%)
                    </div>
                  </div>

                  <div style={{ marginBottom: '16px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                      <span>开盘:</span>
                      <span>{selected.open.toFixed(2)}</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                      <span>最高:</span>
                      <span style={{ color: '#52c41a' }}>{selected.high.toFixed(2)}</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                      <span>最低:</span>
                      <span style={{ color: '#ff4d4f' }}>{selected.low.toFixed(2)}</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                      <span>成交量:</span>
                      <span>{selected.volume.toLocaleString()}</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span>成交额:</span>
                      <span>{(selected.turnover / 10000).toFixed(0)}万</span>
                    </div>
                  </div>

                  {/* 简单的K线图模拟 */}
                  <div style={{ 
                    height: '200px', 
                    background: '#f8f9fa', 
                    border: '1px solid #e9ecef',
                    borderRadius: '4px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    marginBottom: '16px'
                  }}>
                    <div style={{ textAlign: 'center', color: '#666' }}>
                      <div style={{ fontSize: '48px', marginBottom: '8px' }}>📈</div>
                      <div>K线图</div>
                      <div style={{ fontSize: '12px' }}>实时图表功能开发中</div>
                    </div>
                  </div>

                  {/* 技术指标 */}
                  <div>
                    <h4 style={{ margin: '0 0 12px 0' }}>技术指标</h4>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                      <span>MA5:</span>
                      <span>{(selected.price * 0.98).toFixed(2)}</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                      <span>MA10:</span>
                      <span>{(selected.price * 0.96).toFixed(2)}</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                      <span>RSI:</span>
                      <span>{(Math.random() * 40 + 30).toFixed(1)}</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span>MACD:</span>
                      <span>{(Math.random() * 0.2 - 0.1).toFixed(3)}</span>
                    </div>
                  </div>
                </div>
              );
            })()}
          </Card>
        </Col>
      </Row>
    </div>
  );
};

const OrdersPage = () => {
  const [orders, setOrders] = React.useState([
    {
      id: 'ORD001',
      symbol: '000001.SZ',
      name: '平安银行',
      type: 'buy',
      quantity: 1000,
      price: 12.45,
      status: 'filled',
      filledQuantity: 1000,
      filledPrice: 12.43,
      createTime: '2025-01-20 09:30:15',
      updateTime: '2025-01-20 09:30:18'
    },
    {
      id: 'ORD002',
      symbol: '000002.SZ',
      name: '万科A',
      type: 'sell',
      quantity: 500,
      price: 18.80,
      status: 'pending',
      filledQuantity: 0,
      filledPrice: 0,
      createTime: '2025-01-20 10:15:32',
      updateTime: '2025-01-20 10:15:32'
    },
    {
      id: 'ORD003',
      symbol: '600000.SH',
      name: '浦发银行',
      type: 'buy',
      quantity: 2000,
      price: 8.90,
      status: 'partial',
      filledQuantity: 800,
      filledPrice: 8.92,
      createTime: '2025-01-20 11:20:45',
      updateTime: '2025-01-20 11:25:12'
    }
  ]);

  const [showCreateModal, setShowCreateModal] = React.useState(false);
  const [newOrder, setNewOrder] = React.useState({
    symbol: '',
    type: 'buy',
    quantity: '',
    price: '',
    orderType: 'limit'
  });

  const getStatusColor = (status) => {
    switch (status) {
      case 'filled': return '#52c41a';
      case 'pending': return '#1890ff';
      case 'partial': return '#faad14';
      case 'cancelled': return '#ff4d4f';
      default: return '#666';
    }
  };

  const getStatusText = (status) => {
    switch (status) {
      case 'filled': return '已成交';
      case 'pending': return '待成交';
      case 'partial': return '部分成交';
      case 'cancelled': return '已撤销';
      default: return '未知';
    }
  };

  const handleCreateOrder = () => {
    const order = {
      id: `ORD${String(orders.length + 1).padStart(3, '0')}`,
      symbol: newOrder.symbol,
      name: '股票名称', // 实际应该从API获取
      type: newOrder.type,
      quantity: parseInt(newOrder.quantity),
      price: parseFloat(newOrder.price),
      status: 'pending',
      filledQuantity: 0,
      filledPrice: 0,
      createTime: new Date().toLocaleString('zh-CN'),
      updateTime: new Date().toLocaleString('zh-CN')
    };
    
    setOrders(prev => [order, ...prev]);
    setNewOrder({ symbol: '', type: 'buy', quantity: '', price: '', orderType: 'limit' });
    setShowCreateModal(false);
  };

  const cancelOrder = (orderId) => {
    setOrders(prev => prev.map(order => 
      order.id === orderId ? { ...order, status: 'cancelled' } : order
    ));
  };

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h2 style={{ margin: 0 }}>订单管理</h2>
        <button 
          style={{ 
            padding: '8px 16px',
            border: '1px solid #1890ff',
            background: '#1890ff',
            color: 'white',
            borderRadius: '4px',
            cursor: 'pointer'
          }}
          onClick={() => setShowCreateModal(true)}
        >
          创建订单
        </button>
      </div>

      {/* 订单统计 */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col span={6}>
          <Card>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#1890ff' }}>
                {orders.length}
              </div>
              <div style={{ color: '#666' }}>总订单数</div>
            </div>
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#52c41a' }}>
                {orders.filter(o => o.status === 'filled').length}
              </div>
              <div style={{ color: '#666' }}>已成交</div>
            </div>
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#faad14' }}>
                {orders.filter(o => o.status === 'pending' || o.status === 'partial').length}
              </div>
              <div style={{ color: '#666' }}>待成交</div>
            </div>
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#52c41a' }}>
                {((orders.filter(o => o.status === 'filled').length / orders.length) * 100).toFixed(1)}%
              </div>
              <div style={{ color: '#666' }}>成交率</div>
            </div>
          </Card>
        </Col>
      </Row>

      {/* 订单列表 */}
      <Card title="订单列表">
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid #f0f0f0' }}>
              <th style={{ padding: '12px', textAlign: 'left', fontWeight: 'bold' }}>订单号</th>
              <th style={{ padding: '12px', textAlign: 'left', fontWeight: 'bold' }}>股票</th>
              <th style={{ padding: '12px', textAlign: 'center', fontWeight: 'bold' }}>方向</th>
              <th style={{ padding: '12px', textAlign: 'right', fontWeight: 'bold' }}>数量</th>
              <th style={{ padding: '12px', textAlign: 'right', fontWeight: 'bold' }}>价格</th>
              <th style={{ padding: '12px', textAlign: 'center', fontWeight: 'bold' }}>状态</th>
              <th style={{ padding: '12px', textAlign: 'right', fontWeight: 'bold' }}>成交量</th>
              <th style={{ padding: '12px', textAlign: 'left', fontWeight: 'bold' }}>时间</th>
              <th style={{ padding: '12px', textAlign: 'center', fontWeight: 'bold' }}>操作</th>
            </tr>
          </thead>
          <tbody>
            {orders.map(order => (
              <tr key={order.id} style={{ borderBottom: '1px solid #f0f0f0' }}>
                <td style={{ padding: '12px', fontFamily: 'monospace' }}>{order.id}</td>
                <td style={{ padding: '12px' }}>
                  <div>
                    <div style={{ fontWeight: 'bold' }}>{order.name}</div>
                    <div style={{ fontSize: '12px', color: '#666' }}>{order.symbol}</div>
                  </div>
                </td>
                <td style={{ padding: '12px', textAlign: 'center' }}>
                  <span style={{ 
                    color: order.type === 'buy' ? '#52c41a' : '#ff4d4f',
                    fontWeight: 'bold'
                  }}>
                    {order.type === 'buy' ? '买入' : '卖出'}
                  </span>
                </td>
                <td style={{ padding: '12px', textAlign: 'right' }}>{order.quantity.toLocaleString()}</td>
                <td style={{ padding: '12px', textAlign: 'right' }}>¥{order.price.toFixed(2)}</td>
                <td style={{ padding: '12px', textAlign: 'center' }}>
                  <span style={{ color: getStatusColor(order.status), fontWeight: 'bold' }}>
                    {getStatusText(order.status)}
                  </span>
                </td>
                <td style={{ padding: '12px', textAlign: 'right' }}>
                  {order.filledQuantity > 0 ? (
                    <div>
                      <div>{order.filledQuantity.toLocaleString()}</div>
                      <div style={{ fontSize: '12px', color: '#666' }}>
                        @¥{order.filledPrice.toFixed(2)}
                      </div>
                    </div>
                  ) : '-'}
                </td>
                <td style={{ padding: '12px', fontSize: '12px' }}>
                  <div>{order.createTime}</div>
                  {order.updateTime !== order.createTime && (
                    <div style={{ color: '#666' }}>更新: {order.updateTime}</div>
                  )}
                </td>
                <td style={{ padding: '12px', textAlign: 'center' }}>
                  {(order.status === 'pending' || order.status === 'partial') && (
                    <button 
                      style={{ 
                        padding: '4px 8px',
                        border: '1px solid #ff4d4f',
                        background: '#ff4d4f',
                        color: 'white',
                        borderRadius: '4px',
                        cursor: 'pointer',
                        fontSize: '12px'
                      }}
                      onClick={() => cancelOrder(order.id)}
                    >
                      撤销
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      {/* 创建订单模态框 */}
      {showCreateModal && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(0,0,0,0.5)',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          zIndex: 1000
        }}>
          <div style={{
            background: 'white',
            padding: '24px',
            borderRadius: '8px',
            width: '500px'
          }}>
            <h3>创建订单</h3>
            <div style={{ marginBottom: '16px' }}>
              <label style={{ display: 'block', marginBottom: '4px' }}>股票代码</label>
              <input 
                type="text" 
                value={newOrder.symbol}
                onChange={(e) => setNewOrder(prev => ({ ...prev, symbol: e.target.value }))}
                placeholder="例如: 000001.SZ"
                style={{ width: '100%', padding: '8px', border: '1px solid #d9d9d9', borderRadius: '4px' }}
              />
            </div>
            <div style={{ marginBottom: '16px' }}>
              <label style={{ display: 'block', marginBottom: '4px' }}>交易方向</label>
              <select 
                value={newOrder.type}
                onChange={(e) => setNewOrder(prev => ({ ...prev, type: e.target.value }))}
                style={{ width: '100%', padding: '8px', border: '1px solid #d9d9d9', borderRadius: '4px' }}
              >
                <option value="buy">买入</option>
                <option value="sell">卖出</option>
              </select>
            </div>
            <div style={{ marginBottom: '16px' }}>
              <label style={{ display: 'block', marginBottom: '4px' }}>数量</label>
              <input 
                type="number" 
                value={newOrder.quantity}
                onChange={(e) => setNewOrder(prev => ({ ...prev, quantity: e.target.value }))}
                placeholder="股数"
                style={{ width: '100%', padding: '8px', border: '1px solid #d9d9d9', borderRadius: '4px' }}
              />
            </div>
            <div style={{ marginBottom: '16px' }}>
              <label style={{ display: 'block', marginBottom: '4px' }}>价格</label>
              <input 
                type="number" 
                step="0.01"
                value={newOrder.price}
                onChange={(e) => setNewOrder(prev => ({ ...prev, price: e.target.value }))}
                placeholder="单价"
                style={{ width: '100%', padding: '8px', border: '1px solid #d9d9d9', borderRadius: '4px' }}
              />
            </div>
            <div style={{ marginBottom: '24px' }}>
              <label style={{ display: 'block', marginBottom: '4px' }}>订单类型</label>
              <select 
                value={newOrder.orderType}
                onChange={(e) => setNewOrder(prev => ({ ...prev, orderType: e.target.value }))}
                style={{ width: '100%', padding: '8px', border: '1px solid #d9d9d9', borderRadius: '4px' }}
              >
                <option value="limit">限价单</option>
                <option value="market">市价单</option>
              </select>
            </div>
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px' }}>
              <button 
                style={{ 
                  padding: '8px 16px',
                  border: '1px solid #d9d9d9',
                  background: 'white',
                  borderRadius: '4px',
                  cursor: 'pointer'
                }}
                onClick={() => setShowCreateModal(false)}
              >
                取消
              </button>
              <button 
                style={{ 
                  padding: '8px 16px',
                  border: '1px solid #1890ff',
                  background: '#1890ff',
                  color: 'white',
                  borderRadius: '4px',
                  cursor: 'pointer'
                }}
                onClick={handleCreateOrder}
                disabled={!newOrder.symbol || !newOrder.quantity || !newOrder.price}
              >
                创建订单
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

const PortfolioPage = () => {
  const [portfolio, setPortfolio] = React.useState({
    totalValue: 1256780,
    totalCost: 1180000,
    totalPnL: 76780,
    totalPnLPercent: 6.51,
    cashBalance: 123456,
    positions: [
      {
        symbol: '000001.SZ',
        name: '平安银行',
        quantity: 10000,
        avgCost: 11.80,
        currentPrice: 12.45,
        marketValue: 124500,
        pnl: 6500,
        pnlPercent: 5.51,
        weight: 9.91
      },
      {
        symbol: '000002.SZ',
        name: '万科A',
        quantity: 5000,
        avgCost: 19.20,
        currentPrice: 18.76,
        marketValue: 93800,
        pnl: -2200,
        pnlPercent: -2.29,
        weight: 7.46
      },
      {
        symbol: '600000.SH',
        name: '浦发银行',
        quantity: 15000,
        avgCost: 8.50,
        currentPrice: 8.92,
        marketValue: 133800,
        pnl: 6300,
        pnlPercent: 4.94,
        weight: 10.65
      }
    ]
  });

  const [selectedTab, setSelectedTab] = React.useState('positions');

  // 模拟收益曲线数据
  const performanceData = [
    { date: '2025-01-01', value: 1180000 },
    { date: '2025-01-05', value: 1195000 },
    { date: '2025-01-10', value: 1210000 },
    { date: '2025-01-15', value: 1235000 },
    { date: '2025-01-20', value: 1256780 }
  ];

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: '24px' }}>
        <h2 style={{ margin: 0 }}>投资组合分析</h2>
      </div>

      {/* 投资组合概览 */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col span={6}>
          <Card>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#1890ff' }}>
                ¥{portfolio.totalValue.toLocaleString()}
              </div>
              <div style={{ color: '#666' }}>总市值</div>
            </div>
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#52c41a' }}>
                ¥{portfolio.totalPnL.toLocaleString()}
              </div>
              <div style={{ color: '#666' }}>总盈亏</div>
              <div style={{ fontSize: '12px', color: '#52c41a' }}>
                +{portfolio.totalPnLPercent}%
              </div>
            </div>
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#722ed1' }}>
                ¥{portfolio.cashBalance.toLocaleString()}
              </div>
              <div style={{ color: '#666' }}>现金余额</div>
            </div>
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#fa8c16' }}>
                {portfolio.positions.length}
              </div>
              <div style={{ color: '#666' }}>持仓股票</div>
            </div>
          </Card>
        </Col>
      </Row>

      {/* 标签页 */}
      <Card>
        <div style={{ marginBottom: '24px' }}>
          <div style={{ display: 'flex', borderBottom: '1px solid #f0f0f0' }}>
            {[
              { key: 'positions', label: '持仓明细' },
              { key: 'performance', label: '收益分析' },
              { key: 'risk', label: '风险分析' }
            ].map(tab => (
              <button
                key={tab.key}
                style={{
                  padding: '12px 24px',
                  border: 'none',
                  background: 'none',
                  cursor: 'pointer',
                  borderBottom: selectedTab === tab.key ? '2px solid #1890ff' : '2px solid transparent',
                  color: selectedTab === tab.key ? '#1890ff' : '#666',
                  fontWeight: selectedTab === tab.key ? 'bold' : 'normal'
                }}
                onClick={() => setSelectedTab(tab.key)}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>

        {/* 持仓明细 */}
        {selectedTab === 'positions' && (
          <div>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid #f0f0f0' }}>
                  <th style={{ padding: '12px', textAlign: 'left', fontWeight: 'bold' }}>股票</th>
                  <th style={{ padding: '12px', textAlign: 'right', fontWeight: 'bold' }}>持仓量</th>
                  <th style={{ padding: '12px', textAlign: 'right', fontWeight: 'bold' }}>成本价</th>
                  <th style={{ padding: '12px', textAlign: 'right', fontWeight: 'bold' }}>现价</th>
                  <th style={{ padding: '12px', textAlign: 'right', fontWeight: 'bold' }}>市值</th>
                  <th style={{ padding: '12px', textAlign: 'right', fontWeight: 'bold' }}>盈亏</th>
                  <th style={{ padding: '12px', textAlign: 'right', fontWeight: 'bold' }}>权重</th>
                </tr>
              </thead>
              <tbody>
                {portfolio.positions.map(position => (
                  <tr key={position.symbol} style={{ borderBottom: '1px solid #f0f0f0' }}>
                    <td style={{ padding: '12px' }}>
                      <div>
                        <div style={{ fontWeight: 'bold' }}>{position.name}</div>
                        <div style={{ fontSize: '12px', color: '#666' }}>{position.symbol}</div>
                      </div>
                    </td>
                    <td style={{ padding: '12px', textAlign: 'right' }}>
                      {position.quantity.toLocaleString()}
                    </td>
                    <td style={{ padding: '12px', textAlign: 'right' }}>
                      ¥{position.avgCost.toFixed(2)}
                    </td>
                    <td style={{ padding: '12px', textAlign: 'right' }}>
                      ¥{position.currentPrice.toFixed(2)}
                    </td>
                    <td style={{ padding: '12px', textAlign: 'right', fontWeight: 'bold' }}>
                      ¥{position.marketValue.toLocaleString()}
                    </td>
                    <td style={{ 
                      padding: '12px', 
                      textAlign: 'right',
                      color: position.pnl >= 0 ? '#52c41a' : '#ff4d4f'
                    }}>
                      <div>{position.pnl >= 0 ? '+' : ''}¥{position.pnl.toLocaleString()}</div>
                      <div style={{ fontSize: '12px' }}>
                        {position.pnlPercent >= 0 ? '+' : ''}{position.pnlPercent}%
                      </div>
                    </td>
                    <td style={{ padding: '12px', textAlign: 'right' }}>
                      {position.weight}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* 收益分析 */}
        {selectedTab === 'performance' && (
          <div>
            <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
              <Col span={8}>
                <div style={{ textAlign: 'center', padding: '16px', background: '#f6ffed', borderRadius: '8px' }}>
                  <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#52c41a' }}>
                    +{portfolio.totalPnLPercent}%
                  </div>
                  <div style={{ color: '#666' }}>总收益率</div>
                </div>
              </Col>
              <Col span={8}>
                <div style={{ textAlign: 'center', padding: '16px', background: '#f0f5ff', borderRadius: '8px' }}>
                  <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#1890ff' }}>
                    1.35
                  </div>
                  <div style={{ color: '#666' }}>夏普比率</div>
                </div>
              </Col>
              <Col span={8}>
                <div style={{ textAlign: 'center', padding: '16px', background: '#fff2e8', borderRadius: '8px' }}>
                  <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#fa8c16' }}>
                    -3.2%
                  </div>
                  <div style={{ color: '#666' }}>最大回撤</div>
                </div>
              </Col>
            </Row>

            {/* 收益曲线图 */}
            <div style={{ 
              height: '300px', 
              background: '#f8f9fa', 
              border: '1px solid #e9ecef',
              borderRadius: '8px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              marginBottom: '24px'
            }}>
              <div style={{ textAlign: 'center', color: '#666' }}>
                <div style={{ fontSize: '48px', marginBottom: '8px' }}>📊</div>
                <div>收益曲线图</div>
                <div style={{ fontSize: '12px' }}>图表组件开发中</div>
              </div>
            </div>

            {/* 收益明细 */}
            <div>
              <h4>收益明细</h4>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #f0f0f0' }}>
                    <th style={{ padding: '12px', textAlign: 'left' }}>日期</th>
                    <th style={{ padding: '12px', textAlign: 'right' }}>组合价值</th>
                    <th style={{ padding: '12px', textAlign: 'right' }}>日收益</th>
                    <th style={{ padding: '12px', textAlign: 'right' }}>累计收益</th>
                  </tr>
                </thead>
                <tbody>
                  {performanceData.map((item, index) => {
                    const dailyReturn = index > 0 ? item.value - performanceData[index - 1].value : 0;
                    const totalReturn = item.value - performanceData[0].value;
                    return (
                      <tr key={item.date} style={{ borderBottom: '1px solid #f0f0f0' }}>
                        <td style={{ padding: '12px' }}>{item.date}</td>
                        <td style={{ padding: '12px', textAlign: 'right' }}>
                          ¥{item.value.toLocaleString()}
                        </td>
                        <td style={{ 
                          padding: '12px', 
                          textAlign: 'right',
                          color: dailyReturn >= 0 ? '#52c41a' : '#ff4d4f'
                        }}>
                          {dailyReturn >= 0 ? '+' : ''}¥{dailyReturn.toLocaleString()}
                        </td>
                        <td style={{ 
                          padding: '12px', 
                          textAlign: 'right',
                          color: totalReturn >= 0 ? '#52c41a' : '#ff4d4f'
                        }}>
                          {totalReturn >= 0 ? '+' : ''}¥{totalReturn.toLocaleString()}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* 风险分析 */}
        {selectedTab === 'risk' && (
          <div>
            <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
              <Col span={8}>
                <div style={{ textAlign: 'center', padding: '16px', background: '#fff1f0', borderRadius: '8px' }}>
                  <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#ff4d4f' }}>
                    12.5%
                  </div>
                  <div style={{ color: '#666' }}>波动率</div>
                </div>
              </Col>
              <Col span={8}>
                <div style={{ textAlign: 'center', padding: '16px', background: '#f6ffed', borderRadius: '8px' }}>
                  <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#52c41a' }}>
                    0.85
                  </div>
                  <div style={{ color: '#666' }}>贝塔系数</div>
                </div>
              </Col>
              <Col span={8}>
                <div style={{ textAlign: 'center', padding: '16px', background: '#fff2e8', borderRadius: '8px' }}>
                  <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#fa8c16' }}>
                    -¥25,000
                  </div>
                  <div style={{ color: '#666' }}>VaR (95%)</div>
                </div>
              </Col>
            </Row>

            {/* 风险分布 */}
            <div style={{ marginBottom: '24px' }}>
              <h4>行业分布</h4>
              <div style={{ 
                height: '200px', 
                background: '#f8f9fa', 
                border: '1px solid #e9ecef',
                borderRadius: '8px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}>
                <div style={{ textAlign: 'center', color: '#666' }}>
                  <div style={{ fontSize: '48px', marginBottom: '8px' }}>🥧</div>
                  <div>行业分布饼图</div>
                  <div style={{ fontSize: '12px' }}>图表组件开发中</div>
                </div>
              </div>
            </div>

            {/* 风险指标详情 */}
            <div>
              <h4>风险指标详情</h4>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <tbody>
                  <tr style={{ borderBottom: '1px solid #f0f0f0' }}>
                    <td style={{ padding: '12px', fontWeight: 'bold' }}>年化波动率</td>
                    <td style={{ padding: '12px', textAlign: 'right' }}>12.5%</td>
                  </tr>
                  <tr style={{ borderBottom: '1px solid #f0f0f0' }}>
                    <td style={{ padding: '12px', fontWeight: 'bold' }}>最大回撤</td>
                    <td style={{ padding: '12px', textAlign: 'right', color: '#ff4d4f' }}>-3.2%</td>
                  </tr>
                  <tr style={{ borderBottom: '1px solid #f0f0f0' }}>
                    <td style={{ padding: '12px', fontWeight: 'bold' }}>夏普比率</td>
                    <td style={{ padding: '12px', textAlign: 'right' }}>1.35</td>
                  </tr>
                  <tr style={{ borderBottom: '1px solid #f0f0f0' }}>
                    <td style={{ padding: '12px', fontWeight: 'bold' }}>贝塔系数</td>
                    <td style={{ padding: '12px', textAlign: 'right' }}>0.85</td>
                  </tr>
                  <tr style={{ borderBottom: '1px solid #f0f0f0' }}>
                    <td style={{ padding: '12px', fontWeight: 'bold' }}>VaR (95%)</td>
                    <td style={{ padding: '12px', textAlign: 'right', color: '#ff4d4f' }}>-¥25,000</td>
                  </tr>
                  <tr>
                    <td style={{ padding: '12px', fontWeight: 'bold' }}>相关系数</td>
                    <td style={{ padding: '12px', textAlign: 'right' }}>0.72</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        )}
      </Card>
    </div>
  );
};

const RiskPage = () => {
  const [riskMetrics, setRiskMetrics] = React.useState({
    dailyPnL: 8500,
    currentDrawdown: -2.1,
    riskUtilization: 65.4,
    maxDrawdownLimit: -10.0,
    maxDailyLossLimit: -50000,
    positionLimit: 80.0,
    leverageRatio: 1.2
  });

  const [alerts, setAlerts] = React.useState([
    {
      id: 1,
      level: 'warning',
      message: '单只股票仓位超过15%限制',
      symbol: '000001.SZ',
      timestamp: '2025-01-20 14:30:15',
      status: 'active'
    },
    {
      id: 2,
      level: 'info',
      message: '风险利用率达到65%',
      timestamp: '2025-01-20 13:45:22',
      status: 'acknowledged'
    },
    {
      id: 3,
      level: 'critical',
      message: '回撤接近警戒线',
      timestamp: '2025-01-20 11:20:08',
      status: 'resolved'
    }
  ]);

  const [systemStatus, setSystemStatus] = React.useState({
    riskEngine: 'running',
    monitoring: 'running',
    alerts: 'running'
  });

  const getAlertColor = (level) => {
    switch (level) {
      case 'critical': return '#ff4d4f';
      case 'warning': return '#faad14';
      case 'info': return '#1890ff';
      default: return '#666';
    }
  };

  const getAlertIcon = (level) => {
    switch (level) {
      case 'critical': return '🚨';
      case 'warning': return '⚠️';
      case 'info': return 'ℹ️';
      default: return '📋';
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'running': return '#52c41a';
      case 'stopped': return '#ff4d4f';
      case 'warning': return '#faad14';
      default: return '#666';
    }
  };

  const emergencyStop = () => {
    // 直接执行紧急停止，不显示确认弹窗
    setSystemStatus(prev => ({
      ...prev,
      riskEngine: 'stopped',
      monitoring: 'stopped'
    }));
    // alert('紧急停止已执行'); // 移除成功提示
  };

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h2 style={{ margin: 0 }}>风险监控</h2>
        <button 
          style={{ 
            padding: '8px 16px',
            border: '1px solid #ff4d4f',
            background: '#ff4d4f',
            color: 'white',
            borderRadius: '4px',
            cursor: 'pointer',
            fontWeight: 'bold'
          }}
          onClick={emergencyStop}
        >
          🛑 紧急停止
        </button>
      </div>

      {/* 风险指标仪表板 */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col span={6}>
          <Card>
            <div style={{ textAlign: 'center' }}>
              <div style={{ 
                fontSize: '24px', 
                fontWeight: 'bold', 
                color: riskMetrics.dailyPnL >= 0 ? '#52c41a' : '#ff4d4f' 
              }}>
                {riskMetrics.dailyPnL >= 0 ? '+' : ''}¥{riskMetrics.dailyPnL.toLocaleString()}
              </div>
              <div style={{ color: '#666' }}>今日盈亏</div>
              <div style={{ fontSize: '12px', color: '#999' }}>
                限制: {riskMetrics.maxDailyLossLimit.toLocaleString()}
              </div>
            </div>
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <div style={{ textAlign: 'center' }}>
              <div style={{ 
                fontSize: '24px', 
                fontWeight: 'bold', 
                color: Math.abs(riskMetrics.currentDrawdown) > 5 ? '#ff4d4f' : '#faad14'
              }}>
                {riskMetrics.currentDrawdown}%
              </div>
              <div style={{ color: '#666' }}>当前回撤</div>
              <div style={{ fontSize: '12px', color: '#999' }}>
                限制: {riskMetrics.maxDrawdownLimit}%
              </div>
            </div>
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <div style={{ textAlign: 'center' }}>
              <div style={{ 
                fontSize: '24px', 
                fontWeight: 'bold', 
                color: riskMetrics.riskUtilization > 80 ? '#ff4d4f' : '#1890ff'
              }}>
                {riskMetrics.riskUtilization}%
              </div>
              <div style={{ color: '#666' }}>风险利用率</div>
              <div style={{ 
                width: '100%', 
                height: '4px', 
                background: '#f0f0f0', 
                borderRadius: '2px',
                marginTop: '8px',
                overflow: 'hidden'
              }}>
                <div style={{ 
                  width: `${riskMetrics.riskUtilization}%`, 
                  height: '100%', 
                  background: riskMetrics.riskUtilization > 80 ? '#ff4d4f' : '#1890ff',
                  transition: 'width 0.3s'
                }} />
              </div>
            </div>
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#722ed1' }}>
                {riskMetrics.leverageRatio}x
              </div>
              <div style={{ color: '#666' }}>杠杆倍数</div>
              <div style={{ fontSize: '12px', color: '#999' }}>
                限制: 2.0x
              </div>
            </div>
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]}>
        {/* 风险告警 */}
        <Col span={14}>
          <Card title="风险告警">
            <div style={{ maxHeight: '400px', overflowY: 'auto' }}>
              {alerts.map(alert => (
                <div 
                  key={alert.id}
                  style={{ 
                    padding: '12px',
                    marginBottom: '8px',
                    border: `1px solid ${getAlertColor(alert.level)}`,
                    borderRadius: '4px',
                    background: alert.status === 'active' ? '#fff2f0' : '#f9f9f9'
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                    <div style={{ flex: 1 }}>
                      <div style={{ display: 'flex', alignItems: 'center', marginBottom: '4px' }}>
                        <span style={{ marginRight: '8px', fontSize: '16px' }}>
                          {getAlertIcon(alert.level)}
                        </span>
                        <span style={{ 
                          fontWeight: 'bold', 
                          color: getAlertColor(alert.level)
                        }}>
                          {alert.level.toUpperCase()}
                        </span>
                        {alert.symbol && (
                          <span style={{ 
                            marginLeft: '8px',
                            padding: '2px 6px',
                            background: '#f0f0f0',
                            borderRadius: '2px',
                            fontSize: '12px'
                          }}>
                            {alert.symbol}
                          </span>
                        )}
                      </div>
                      <div style={{ marginBottom: '4px' }}>{alert.message}</div>
                      <div style={{ fontSize: '12px', color: '#666' }}>{alert.timestamp}</div>
                    </div>
                    <div style={{ marginLeft: '12px' }}>
                      <span style={{ 
                        padding: '2px 8px',
                        borderRadius: '12px',
                        fontSize: '12px',
                        background: alert.status === 'active' ? '#ff4d4f' : 
                                   alert.status === 'acknowledged' ? '#faad14' : '#52c41a',
                        color: 'white'
                      }}>
                        {alert.status === 'active' ? '活跃' : 
                         alert.status === 'acknowledged' ? '已确认' : '已解决'}
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </Col>

        {/* 系统状态和控制 */}
        <Col span={10}>
          <Card title="系统状态" style={{ marginBottom: '16px' }}>
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px' }}>
                <span>风险引擎:</span>
                <span style={{ 
                  color: getStatusColor(systemStatus.riskEngine),
                  fontWeight: 'bold'
                }}>
                  ● {systemStatus.riskEngine === 'running' ? '运行中' : '已停止'}
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px' }}>
                <span>实时监控:</span>
                <span style={{ 
                  color: getStatusColor(systemStatus.monitoring),
                  fontWeight: 'bold'
                }}>
                  ● {systemStatus.monitoring === 'running' ? '运行中' : '已停止'}
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>告警系统:</span>
                <span style={{ 
                  color: getStatusColor(systemStatus.alerts),
                  fontWeight: 'bold'
                }}>
                  ● {systemStatus.alerts === 'running' ? '运行中' : '已停止'}
                </span>
              </div>
            </div>
          </Card>

          <Card title="风险控制">
            <div style={{ marginBottom: '16px' }}>
              <div style={{ marginBottom: '8px', fontWeight: 'bold' }}>仓位限制</div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                <span>当前仓位:</span>
                <span>{riskMetrics.positionLimit}%</span>
              </div>
              <div style={{ 
                width: '100%', 
                height: '8px', 
                background: '#f0f0f0', 
                borderRadius: '4px',
                overflow: 'hidden'
              }}>
                <div style={{ 
                  width: `${riskMetrics.positionLimit}%`, 
                  height: '100%', 
                  background: riskMetrics.positionLimit > 90 ? '#ff4d4f' : '#52c41a',
                  transition: 'width 0.3s'
                }} />
              </div>
            </div>

            <div style={{ marginBottom: '16px' }}>
              <div style={{ marginBottom: '8px', fontWeight: 'bold' }}>风险参数设置</div>
              <div style={{ marginBottom: '8px' }}>
                <label style={{ display: 'block', marginBottom: '4px', fontSize: '12px' }}>
                  最大回撤限制 (%)
                </label>
                <input 
                  type="number" 
                  defaultValue={Math.abs(riskMetrics.maxDrawdownLimit)}
                  style={{ 
                    width: '100%', 
                    padding: '4px 8px', 
                    border: '1px solid #d9d9d9', 
                    borderRadius: '4px',
                    fontSize: '12px'
                  }}
                />
              </div>
              <div style={{ marginBottom: '16px' }}>
                <label style={{ display: 'block', marginBottom: '4px', fontSize: '12px' }}>
                  最大日损失限制 (¥)
                </label>
                <input 
                  type="number" 
                  defaultValue={Math.abs(riskMetrics.maxDailyLossLimit)}
                  style={{ 
                    width: '100%', 
                    padding: '4px 8px', 
                    border: '1px solid #d9d9d9', 
                    borderRadius: '4px',
                    fontSize: '12px'
                  }}
                />
              </div>
            </div>

            <div style={{ display: 'flex', gap: '8px' }}>
              <button 
                style={{ 
                  flex: 1,
                  padding: '8px',
                  border: '1px solid #1890ff',
                  background: '#1890ff',
                  color: 'white',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontSize: '12px'
                }}
              >
                更新设置
              </button>
              <button 
                style={{ 
                  flex: 1,
                  padding: '8px',
                  border: '1px solid #faad14',
                  background: '#faad14',
                  color: 'white',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontSize: '12px'
                }}
              >
                重置默认
              </button>
            </div>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

const SystemPage = () => {
  // 使用 React Query hooks 来获取真实的系统状态
  const { data: healthData, isLoading: healthLoading } = useSystemHealth();
  const { data: metricsData, isLoading: metricsLoading } = useSystemMetrics();
  const { startSystem, stopSystem, restartSystem } = useSystemControl();
  
  // 格式化运行时间的辅助函数
  const formatUptime = (seconds) => {
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${days}天 ${hours}小时 ${minutes}分钟`;
  };
  
  // 从API数据中提取系统信息
  const systemInfo = {
    version: '1.0.0',
    buildTime: '2025-01-20 15:30:00',
    uptime: healthData?.uptimeSeconds ? formatUptime(healthData.uptimeSeconds) : '未知',
    status: healthData?.systemRunning ? 'running' : 'stopped'
  };

  // 获取模块显示名称的辅助函数
  const getModuleDisplayName = (name) => {
    const displayNames = {
      data: '数据模块',
      strategy: '策略模块', 
      execution: '执行模块',
      risk: '风险模块',
      portfolio: '投资组合模块',
      analytics: '分析模块'
    };
    return displayNames[name] || name;
  };

  // 从API数据中提取模块信息
  const modules = healthData?.modules ? Object.entries(healthData.modules).map(([name, moduleData]) => ({
    name,
    displayName: getModuleDisplayName(name),
    status: moduleData.initialized ? 'running' : 'stopped',
    uptime: systemInfo.uptime, // 使用系统运行时间
    requests: Math.floor(Math.random() * 20000) // 模拟请求数，实际应该从metrics获取
  })) : [];

  const [selectedTab, setSelectedTab] = React.useState('status');

  const [config, setConfig] = React.useState({
    initialCapital: 1000000,
    commissionRate: 0.0003,
    maxDrawdown: 0.1,
    maxPositionSize: 0.3,
    riskFreeRate: 0.03
  });

  const [logs, setLogs] = React.useState([
    { time: '2025-01-20 15:30:15', level: 'INFO', module: 'system', message: '系统启动完成' },
    { time: '2025-01-20 15:29:45', level: 'INFO', module: 'data', message: '数据模块初始化完成' },
    { time: '2025-01-20 15:29:30', level: 'INFO', module: 'strategy', message: '策略模块加载完成' },
    { time: '2025-01-20 15:29:15', level: 'WARN', module: 'risk', message: '风险参数配置检查' },
    { time: '2025-01-20 15:29:00', level: 'INFO', module: 'execution', message: '执行引擎就绪' }
  ]);

  const getStatusColor = (status) => {
    switch (status) {
      case 'running': return '#52c41a';
      case 'stopped': return '#ff4d4f';
      case 'warning': return '#faad14';
      default: return '#666';
    }
  };

  const getLevelColor = (level) => {
    switch (level) {
      case 'ERROR': return '#ff4d4f';
      case 'WARN': return '#faad14';
      case 'INFO': return '#1890ff';
      default: return '#666';
    }
  };

  const handleSystemControl = async (action) => {
    console.log(`[SystemPage] Handling system control action: ${action}`);
    
    if (action === 'stop') {
      // 直接执行停止操作，不显示确认弹窗
      try {
        console.log('[SystemPage] Using React Query mutation for system stop...');
        await stopSystem.mutateAsync();
        // alert('系统停止成功'); // 移除成功提示
      } catch (error) {
        console.error('[SystemPage] System stop error:', error);
        // alert('系统停止失败: ' + (error.message || '未知错误')); // 移除错误提示
      }
    } else if (action === 'start') {
      // 直接执行启动操作，不显示确认弹窗
      try {
        console.log('[SystemPage] Using React Query mutation for system start...');
        await startSystem.mutateAsync();
        // alert('系统启动成功'); // 移除成功提示
      } catch (error) {
          console.error('[SystemPage] System start error:', error);
          // alert('系统启动失败: ' + (error.message || '未知错误')); // 移除错误提示
        }
    } else if (action === 'restart') {
      // 直接执行重启操作，不显示确认弹窗
      try {
        console.log('[SystemPage] Using React Query mutation for system restart...');
        await restartSystem.mutateAsync();
        // alert('系统重启成功'); // 移除成功提示
      } catch (error) {
        console.error('[SystemPage] System restart error:', error);
        // alert('系统重启失败: ' + (error.message || '未知错误')); // 移除错误提示
      }
    }
  };

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h2 style={{ margin: 0 }}>系统管理</h2>
        <div style={{ display: 'flex', gap: '8px' }}>
          <button 
            style={{ 
              padding: '8px 16px',
              border: '1px solid #52c41a',
              background: systemInfo.status === 'stopped' ? '#52c41a' : 'white',
              color: systemInfo.status === 'stopped' ? 'white' : '#52c41a',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
            onClick={() => handleSystemControl('start')}
            disabled={systemInfo.status === 'running'}
          >
            启动系统
          </button>
          <button 
            style={{ 
              padding: '8px 16px',
              border: '1px solid #faad14',
              background: '#faad14',
              color: 'white',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
            onClick={() => handleSystemControl('restart')}
          >
            重启系统
          </button>
          <button 
            style={{ 
              padding: '8px 16px',
              border: '1px solid #ff4d4f',
              background: systemInfo.status === 'running' ? '#ff4d4f' : 'white',
              color: systemInfo.status === 'running' ? 'white' : '#ff4d4f',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
            onClick={() => handleSystemControl('stop')}
            disabled={systemInfo.status === 'stopped'}
          >
            停止系统
          </button>
        </div>
      </div>

      {/* 系统概览 */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col span={6}>
          <Card>
            <div style={{ textAlign: 'center' }}>
              <div style={{ 
                fontSize: '24px', 
                fontWeight: 'bold', 
                color: getStatusColor(systemInfo.status)
              }}>
                ● {systemInfo.status === 'running' ? '运行中' : 
                    systemInfo.status === 'restarting' ? '重启中' : '已停止'}
              </div>
              <div style={{ color: '#666' }}>系统状态</div>
            </div>
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#1890ff' }}>
                {systemInfo.version}
              </div>
              <div style={{ color: '#666' }}>版本号</div>
            </div>
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '16px', fontWeight: 'bold', color: '#722ed1' }}>
                {systemInfo.uptime}
              </div>
              <div style={{ color: '#666' }}>运行时间</div>
            </div>
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#52c41a' }}>
                {modules.filter(m => m.status === 'running').length}/{modules.length}
              </div>
              <div style={{ color: '#666' }}>模块状态</div>
            </div>
          </Card>
        </Col>
      </Row>

      {/* 标签页 */}
      <Card>
        <div style={{ marginBottom: '24px' }}>
          <div style={{ display: 'flex', borderBottom: '1px solid #f0f0f0' }}>
            {[
              { key: 'status', label: '系统状态' },
              { key: 'config', label: '配置管理' },
              { key: 'logs', label: '系统日志' },
              { key: 'monitoring', label: '性能监控' }
            ].map(tab => (
              <button
                key={tab.key}
                style={{
                  padding: '12px 24px',
                  border: 'none',
                  background: 'none',
                  cursor: 'pointer',
                  borderBottom: selectedTab === tab.key ? '2px solid #1890ff' : '2px solid transparent',
                  color: selectedTab === tab.key ? '#1890ff' : '#666',
                  fontWeight: selectedTab === tab.key ? 'bold' : 'normal'
                }}
                onClick={() => setSelectedTab(tab.key)}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>

        {/* 系统状态 */}
        {selectedTab === 'status' && (
          <div>
            <h4>模块状态</h4>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid #f0f0f0' }}>
                  <th style={{ padding: '12px', textAlign: 'left', fontWeight: 'bold' }}>模块名称</th>
                  <th style={{ padding: '12px', textAlign: 'center', fontWeight: 'bold' }}>状态</th>
                  <th style={{ padding: '12px', textAlign: 'right', fontWeight: 'bold' }}>运行时间</th>
                  <th style={{ padding: '12px', textAlign: 'right', fontWeight: 'bold' }}>请求数</th>
                  <th style={{ padding: '12px', textAlign: 'center', fontWeight: 'bold' }}>操作</th>
                </tr>
              </thead>
              <tbody>
                {modules.map(module => (
                  <tr key={module.name} style={{ borderBottom: '1px solid #f0f0f0' }}>
                    <td style={{ padding: '12px', fontWeight: 'bold' }}>{module.displayName}</td>
                    <td style={{ padding: '12px', textAlign: 'center' }}>
                      <span style={{ 
                        color: getStatusColor(module.status),
                        fontWeight: 'bold'
                      }}>
                        ● {module.status === 'running' ? '运行中' : '已停止'}
                      </span>
                    </td>
                    <td style={{ padding: '12px', textAlign: 'right' }}>{module.uptime}</td>
                    <td style={{ padding: '12px', textAlign: 'right' }}>{module.requests.toLocaleString()}</td>
                    <td style={{ padding: '12px', textAlign: 'center' }}>
                      <button 
                        style={{ 
                          padding: '4px 8px',
                          border: '1px solid #1890ff',
                          background: '#1890ff',
                          color: 'white',
                          borderRadius: '4px',
                          cursor: 'pointer',
                          fontSize: '12px'
                        }}
                      >
                        重启
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* 配置管理 */}
        {selectedTab === 'config' && (
          <div>
            <h4>系统配置</h4>
            <Row gutter={[16, 16]}>
              <Col span={12}>
                <div style={{ marginBottom: '16px' }}>
                  <label style={{ display: 'block', marginBottom: '4px', fontWeight: 'bold' }}>
                    初始资金 (¥)
                  </label>
                  <input 
                    type="number" 
                    value={config.initialCapital}
                    onChange={(e) => setConfig(prev => ({ ...prev, initialCapital: parseInt(e.target.value) }))}
                    style={{ 
                      width: '100%', 
                      padding: '8px', 
                      border: '1px solid #d9d9d9', 
                      borderRadius: '4px'
                    }}
                  />
                </div>
                <div style={{ marginBottom: '16px' }}>
                  <label style={{ display: 'block', marginBottom: '4px', fontWeight: 'bold' }}>
                    手续费率
                  </label>
                  <input 
                    type="number" 
                    step="0.0001"
                    value={config.commissionRate}
                    onChange={(e) => setConfig(prev => ({ ...prev, commissionRate: parseFloat(e.target.value) }))}
                    style={{ 
                      width: '100%', 
                      padding: '8px', 
                      border: '1px solid #d9d9d9', 
                      borderRadius: '4px'
                    }}
                  />
                </div>
                <div style={{ marginBottom: '16px' }}>
                  <label style={{ display: 'block', marginBottom: '4px', fontWeight: 'bold' }}>
                    最大回撤限制
                  </label>
                  <input 
                    type="number" 
                    step="0.01"
                    value={config.maxDrawdown}
                    onChange={(e) => setConfig(prev => ({ ...prev, maxDrawdown: parseFloat(e.target.value) }))}
                    style={{ 
                      width: '100%', 
                      padding: '8px', 
                      border: '1px solid #d9d9d9', 
                      borderRadius: '4px'
                    }}
                  />
                </div>
              </Col>
              <Col span={12}>
                <div style={{ marginBottom: '16px' }}>
                  <label style={{ display: 'block', marginBottom: '4px', fontWeight: 'bold' }}>
                    最大仓位比例
                  </label>
                  <input 
                    type="number" 
                    step="0.01"
                    value={config.maxPositionSize}
                    onChange={(e) => setConfig(prev => ({ ...prev, maxPositionSize: parseFloat(e.target.value) }))}
                    style={{ 
                      width: '100%', 
                      padding: '8px', 
                      border: '1px solid #d9d9d9', 
                      borderRadius: '4px'
                    }}
                  />
                </div>
                <div style={{ marginBottom: '16px' }}>
                  <label style={{ display: 'block', marginBottom: '4px', fontWeight: 'bold' }}>
                    无风险利率
                  </label>
                  <input 
                    type="number" 
                    step="0.001"
                    value={config.riskFreeRate}
                    onChange={(e) => setConfig(prev => ({ ...prev, riskFreeRate: parseFloat(e.target.value) }))}
                    style={{ 
                      width: '100%', 
                      padding: '8px', 
                      border: '1px solid #d9d9d9', 
                      borderRadius: '4px'
                    }}
                  />
                </div>
              </Col>
            </Row>
            <div style={{ marginTop: '24px', display: 'flex', gap: '8px' }}>
              <button 
                style={{ 
                  padding: '8px 16px',
                  border: '1px solid #1890ff',
                  background: '#1890ff',
                  color: 'white',
                  borderRadius: '4px',
                  cursor: 'pointer'
                }}
              >
                保存配置
              </button>
              <button 
                style={{ 
                  padding: '8px 16px',
                  border: '1px solid #d9d9d9',
                  background: 'white',
                  color: '#666',
                  borderRadius: '4px',
                  cursor: 'pointer'
                }}
              >
                重置默认
              </button>
            </div>
          </div>
        )}

        {/* 系统日志 */}
        {selectedTab === 'logs' && (
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
              <h4 style={{ margin: 0 }}>系统日志</h4>
              <button 
                style={{ 
                  padding: '4px 8px',
                  border: '1px solid #1890ff',
                  background: '#1890ff',
                  color: 'white',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontSize: '12px'
                }}
              >
                刷新
              </button>
            </div>
            <div style={{ 
              maxHeight: '400px', 
              overflowY: 'auto',
              border: '1px solid #f0f0f0',
              borderRadius: '4px',
              padding: '12px',
              background: '#fafafa',
              fontFamily: 'monospace',
              fontSize: '12px'
            }}>
              {logs.map((log, index) => (
                <div key={index} style={{ marginBottom: '4px' }}>
                  <span style={{ color: '#666' }}>{log.time}</span>
                  <span style={{ 
                    marginLeft: '8px',
                    color: getLevelColor(log.level),
                    fontWeight: 'bold'
                  }}>
                    [{log.level}]
                  </span>
                  <span style={{ marginLeft: '8px', color: '#1890ff' }}>
                    {log.module}:
                  </span>
                  <span style={{ marginLeft: '8px' }}>{log.message}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* 性能监控 */}
        {selectedTab === 'monitoring' && (
          <div>
            <h4>性能指标</h4>
            <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
              <Col span={8}>
                <div style={{ textAlign: 'center', padding: '16px', background: '#f6ffed', borderRadius: '8px' }}>
                  <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#52c41a' }}>
                    45ms
                  </div>
                  <div style={{ color: '#666' }}>平均响应时间</div>
                </div>
              </Col>
              <Col span={8}>
                <div style={{ textAlign: 'center', padding: '16px', background: '#f0f5ff', borderRadius: '8px' }}>
                  <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#1890ff' }}>
                    99.8%
                  </div>
                  <div style={{ color: '#666' }}>API成功率</div>
                </div>
              </Col>
              <Col span={8}>
                <div style={{ textAlign: 'center', padding: '16px', background: '#fff2e8', borderRadius: '8px' }}>
                  <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#fa8c16' }}>
                    2.1GB
                  </div>
                  <div style={{ color: '#666' }}>内存使用</div>
                </div>
              </Col>
            </Row>

            <div style={{ 
              height: '200px', 
              background: '#f8f9fa', 
              border: '1px solid #e9ecef',
              borderRadius: '8px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}>
              <div style={{ textAlign: 'center', color: '#666' }}>
                <div style={{ fontSize: '48px', marginBottom: '8px' }}>📊</div>
                <div>性能监控图表</div>
                <div style={{ fontSize: '12px' }}>实时监控图表开发中</div>
              </div>
            </div>
          </div>
        )}
      </Card>
    </div>
  );
};

// Import working pages
import BacktestPage from '../../pages/Backtest/BacktestPage';
import DashboardPage from '../../pages/Dashboard/index';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
    },
    mutations: {
      retry: 1,
    },
  },
});

export const LayoutExample: React.FC = () => {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <Router>
          <AppRoutes />
        </Router>
      </ThemeProvider>
    </QueryClientProvider>
  );
};

export default LayoutExample;
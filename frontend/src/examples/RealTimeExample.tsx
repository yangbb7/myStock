import React from 'react';
import { Card, Row, Col, Typography, Space, Button } from 'antd';
import { useMarketData, useSystemStatus, useRiskAlerts } from '../hooks/useRealTime';

const { Title, Text } = Typography;

const RealTimeExample: React.FC = () => {
  // Example 1: Market Data Subscription
  const marketData = useMarketData({
    symbols: ['000001.SZ', '600000.SH', '000002.SZ'],
    throttle: 500, // Update every 500ms
    autoSubscribe: true,
  });

  // Example 2: System Status Monitoring
  const systemStatus = useSystemStatus({
    autoSubscribe: true,
  });

  // Example 3: Risk Alerts
  const riskAlerts = useRiskAlerts({
    autoSubscribe: true,
    maxAlertsSize: 20,
  });

  return (
    <div style={{ padding: '24px' }}>
      <Title level={2}>实时数据通信示例</Title>
      
      <Row gutter={[16, 16]}>
        {/* Market Data Example */}
        <Col span={8}>
          <Card 
            title="市场数据实时推送" 
            extra={
              <Text type={marketData.isConnected ? 'success' : 'danger'}>
                {marketData.isConnected ? '已连接' : '连接断开'}
              </Text>
            }
          >
            <Space direction="vertical" style={{ width: '100%' }}>
              {Object.entries(marketData.data).map(([symbol, data]) => (
                <div key={symbol} style={{ 
                  padding: '8px', 
                  border: '1px solid #f0f0f0', 
                  borderRadius: '4px' 
                }}>
                  <Text strong>{symbol}</Text>
                  <br />
                  <Text>价格: ¥{data.price?.toFixed(2) || 'N/A'}</Text>
                  <br />
                  <Text type={data.change >= 0 ? 'success' : 'danger'}>
                    涨跌: {data.change >= 0 ? '+' : ''}¥{data.change?.toFixed(2) || 'N/A'}
                  </Text>
                  <br />
                  <Text type="secondary" style={{ fontSize: '12px' }}>
                    更新: {data.timestamp ? new Date(data.timestamp).toLocaleTimeString() : 'N/A'}
                  </Text>
                </div>
              ))}
              
              {Object.keys(marketData.data).length === 0 && (
                <Text type="secondary">等待市场数据...</Text>
              )}
              
              <Button 
                size="small" 
                onClick={() => marketData.subscribe(['600036.SH'])}
              >
                订阅新股票
              </Button>
            </Space>
          </Card>
        </Col>

        {/* System Status Example */}
        <Col span={8}>
          <Card 
            title="系统状态监控" 
            extra={
              <Text type={systemStatus.isConnected ? 'success' : 'danger'}>
                {systemStatus.isConnected ? '已连接' : '连接断开'}
              </Text>
            }
          >
            <Space direction="vertical" style={{ width: '100%' }}>
              {systemStatus.data ? (
                <>
                  <div>
                    <Text strong>系统状态: </Text>
                    <Text type={systemStatus.data.systemRunning ? 'success' : 'danger'}>
                      {systemStatus.data.systemRunning ? '运行中' : '已停止'}
                    </Text>
                  </div>
                  
                  <div>
                    <Text strong>运行时间: </Text>
                    <Text>{Math.floor(systemStatus.data.uptimeSeconds / 60)} 分钟</Text>
                  </div>
                  
                  <div>
                    <Text strong>模块状态:</Text>
                    {Object.entries(systemStatus.data.modules).map(([name, module]) => (
                      <div key={name} style={{ marginLeft: '16px' }}>
                        <Text>{name}: </Text>
                        <Text type={module.initialized ? 'success' : 'danger'}>
                          {module.initialized ? '正常' : '异常'}
                        </Text>
                      </div>
                    ))}
                  </div>
                  
                  {systemStatus.lastUpdated && (
                    <Text type="secondary" style={{ fontSize: '12px' }}>
                      最后更新: {systemStatus.lastUpdated.toLocaleTimeString()}
                    </Text>
                  )}
                </>
              ) : (
                <Text type="secondary">等待系统状态数据...</Text>
              )}
            </Space>
          </Card>
        </Col>

        {/* Risk Alerts Example */}
        <Col span={8}>
          <Card 
            title={`风险告警 (${riskAlerts.alerts.length})`}
            extra={
              <Space>
                <Text type={riskAlerts.isConnected ? 'success' : 'danger'}>
                  {riskAlerts.isConnected ? '已连接' : '连接断开'}
                </Text>
                <Button 
                  size="small" 
                  onClick={riskAlerts.clearAlerts}
                  disabled={riskAlerts.alerts.length === 0}
                >
                  清空
                </Button>
              </Space>
            }
          >
            <Space direction="vertical" style={{ width: '100%' }}>
              {riskAlerts.latestAlert && (
                <div style={{ 
                  padding: '8px', 
                  backgroundColor: '#fff2f0', 
                  border: '1px solid #ffccc7',
                  borderRadius: '4px' 
                }}>
                  <Text strong type="danger">最新告警</Text>
                  <br />
                  <Text>{riskAlerts.latestAlert.message}</Text>
                  <br />
                  <Text type="secondary" style={{ fontSize: '12px' }}>
                    {new Date(riskAlerts.latestAlert.timestamp).toLocaleString()}
                  </Text>
                </div>
              )}
              
              {riskAlerts.alerts.slice(0, 3).map((alert, index) => (
                <div key={index} style={{ 
                  padding: '6px', 
                  border: '1px solid #f0f0f0', 
                  borderRadius: '4px' 
                }}>
                  <Text strong>{alert.level.toUpperCase()}</Text>
                  <br />
                  <Text style={{ fontSize: '12px' }}>{alert.message}</Text>
                  <br />
                  <Text type="secondary" style={{ fontSize: '10px' }}>
                    {new Date(alert.timestamp).toLocaleTimeString()}
                  </Text>
                </div>
              ))}
              
              {riskAlerts.alerts.length === 0 && (
                <Text type="secondary">暂无风险告警</Text>
              )}
            </Space>
          </Card>
        </Col>
      </Row>

      {/* Connection Status */}
      <Card style={{ marginTop: '16px' }} size="small">
        <Text strong>连接状态总览: </Text>
        <Text type={marketData.isConnected ? 'success' : 'danger'}>
          市场数据: {marketData.isConnected ? '✓' : '✗'}
        </Text>
        <Text style={{ margin: '0 16px' }} type={systemStatus.isConnected ? 'success' : 'danger'}>
          系统状态: {systemStatus.isConnected ? '✓' : '✗'}
        </Text>
        <Text type={riskAlerts.isConnected ? 'success' : 'danger'}>
          风险告警: {riskAlerts.isConnected ? '✓' : '✗'}
        </Text>
      </Card>
    </div>
  );
};

export default RealTimeExample;
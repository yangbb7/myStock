import React, { useState, useEffect } from 'react';
import { Card, Button, Badge, Typography, Space, List, Alert } from 'antd';
import { useWebSocket } from '../hooks/useWebSocket';
import { useSystemStatus, useMarketData } from '../hooks/useRealTime';

const { Title, Text } = Typography;

const WebSocketTest: React.FC = () => {
  const [logs, setLogs] = useState<string[]>([]);
  const [testSymbols] = useState(['000001.SZ', '000002.SZ', '600000.SH']);

  const { isConnected, connectionState, error } = useWebSocket();
  const { data: systemStatus, lastUpdated } = useSystemStatus({ autoSubscribe: true });
  const { data: marketData, subscribe: subscribeMarketData } = useMarketData({ 
    autoSubscribe: false 
  });

  const addLog = (message: string) => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs(prev => [`[${timestamp}] ${message}`, ...prev.slice(0, 19)]);
  };

  useEffect(() => {
    addLog(`WebSocket状态变更: ${connectionState}`);
  }, [connectionState]);

  useEffect(() => {
    if (systemStatus) {
      addLog(`收到系统状态更新: ${systemStatus.status || 'unknown'}`);
    }
  }, [systemStatus]);

  useEffect(() => {
    const symbolCount = Object.keys(marketData).length;
    if (symbolCount > 0) {
      addLog(`收到市场数据更新: ${symbolCount}个股票`);
    }
  }, [marketData]);

  const handleSubscribeMarketData = () => {
    subscribeMarketData(testSymbols);
    addLog(`订阅市场数据: ${testSymbols.join(', ')}`);
  };

  const getConnectionBadge = () => {
    switch (connectionState) {
      case 'connected':
        return <Badge status="success" text="已连接" />;
      case 'connecting':
        return <Badge status="processing" text="连接中" />;
      case 'disconnected':
        return <Badge status="default" text="未连接" />;
      case 'error':
        return <Badge status="error" text="连接错误" />;
      default:
        return <Badge status="default" text="未知状态" />;
    }
  };

  return (
    <div style={{ padding: '24px' }}>
      <Title level={2}>WebSocket 连接测试</Title>
      
      <Space direction="vertical" size="large" style={{ width: '100%' }}>
        {/* 连接状态 */}
        <Card title="连接状态" size="small">
          <Space direction="vertical">
            <div>
              <Text strong>状态: </Text>
              {getConnectionBadge()}
            </div>
            <div>
              <Text strong>URL: </Text>
              <Text code>{import.meta.env.VITE_WS_BASE_URL || 'http://localhost:8000'}</Text>
            </div>
            {error && (
              <Alert
                message="连接错误"
                description={error.message}
                type="error"
                showIcon
              />
            )}
          </Space>
        </Card>

        {/* 系统状态 */}
        <Card title="系统状态" size="small">
          <Space direction="vertical">
            <div>
              <Text strong>状态: </Text>
              <Text>{systemStatus?.status || '未知'}</Text>
            </div>
            <div>
              <Text strong>最后更新: </Text>
              <Text>{lastUpdated?.toLocaleTimeString() || '无'}</Text>
            </div>
            <div>
              <Text strong>内存使用: </Text>
              <Text>{systemStatus?.memory_usage?.toFixed(2) || '0'}%</Text>
            </div>
            <div>
              <Text strong>CPU使用: </Text>
              <Text>{systemStatus?.cpu_usage?.toFixed(2) || '0'}%</Text>
            </div>
          </Space>
        </Card>

        {/* 市场数据测试 */}
        <Card 
          title="市场数据测试" 
          size="small"
          extra={
            <Button 
              type="primary" 
              onClick={handleSubscribeMarketData}
              disabled={!isConnected}
            >
              订阅测试数据
            </Button>
          }
        >
          <div>
            <Text strong>已订阅股票数量: </Text>
            <Text>{Object.keys(marketData).length}</Text>
          </div>
          {Object.entries(marketData).map(([symbol, data]) => (
            <div key={symbol} style={{ marginTop: '8px' }}>
              <Text strong>{symbol}: </Text>
              <Text>¥{data.price?.toFixed(2) || '0.00'}</Text>
              <Text type={data.change && data.change >= 0 ? 'success' : 'danger'}>
                {' '}({data.change_percent?.toFixed(2) || '0.00'}%)
              </Text>
            </div>
          ))}
        </Card>

        {/* 日志 */}
        <Card title="连接日志" size="small">
          <List
            size="small"
            dataSource={logs}
            renderItem={(item) => (
              <List.Item>
                <Text code style={{ fontSize: '12px' }}>{item}</Text>
              </List.Item>
            )}
            style={{ maxHeight: '300px', overflow: 'auto' }}
          />
        </Card>
      </Space>
    </div>
  );
};

export default WebSocketTest;
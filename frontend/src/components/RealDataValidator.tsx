import React, { useState, useEffect, useRef } from 'react';
import {
  Card,
  Table,
  Button,
  Space,
  Tag,
  Typography,
  Row,
  Col,
  Statistic,
  Alert,
  Spin,
  Progress,
  message,
  Timeline,
} from 'antd';
import {
  CheckCircleOutlined,
  CloseCircleOutlined,
  ReloadOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
} from '@ant-design/icons';
import { api } from '../services/api';

const { Title, Text } = Typography;

interface StockData {
  symbol: string;
  name: string;
  current_price: number;
  timestamp: string;
  source: string;
}

interface DataSourceStatus {
  name: string;
  status: 'success' | 'error' | 'loading';
  data?: StockData;
  error?: string;
  latency?: number;
}

const RealDataValidator: React.FC = () => {
  const [testSymbols] = useState(['000001.SZ', '000002.SZ', '600000.SH', '600036.SH']);
  const [dataStatus, setDataStatus] = useState<Record<string, DataSourceStatus>>({});
  const [isRunning, setIsRunning] = useState(false);
  const [logs, setLogs] = useState<Array<{ time: string; message: string; type: 'success' | 'error' | 'info' }>>([]);
  const intervalRef = useRef<NodeJS.Timeout>();

  const addLog = (message: string, type: 'success' | 'error' | 'info' = 'info') => {
    const time = new Date().toLocaleTimeString();
    setLogs(prev => [{ time, message, type }, ...prev.slice(0, 19)]); // Keep last 20 logs
  };

  const testSingleStock = async (symbol: string): Promise<DataSourceStatus> => {
    const startTime = Date.now();
    
    try {
      console.log(`🔍 [RealDataValidator] Testing ${symbol}...`);
      setDataStatus(prev => ({
        ...prev,
        [symbol]: { name: symbol, status: 'loading' }
      }));

      console.log(`📡 [RealDataValidator] Calling api.data.getRealTimePrice(${symbol})`);
      const response = await api.data.getRealTimePrice(symbol);
      const latency = Date.now() - startTime;
      
      console.log(`📨 [RealDataValidator] Response for ${symbol}:`, response);
      console.log(`⏱️ [RealDataValidator] Latency: ${latency}ms`);
      
      if (response && response.success !== false) {
        const stockData = response;
        let current_price = 0;
        
        console.log(`🔍 [RealDataValidator] Processing response data:`, {
          hasData: !!stockData.data,
          hasCurrentPrice: !!(stockData.data && stockData.data.current_price),
          success: stockData.success
        });
        
        // 直接从实时API响应中获取价格
        if (stockData.data && stockData.data.current_price) {
          current_price = stockData.data.current_price;
          console.log(`💰 [RealDataValidator] Found price for ${symbol}: ¥${current_price}`);
        } else {
          console.error(`❌ [RealDataValidator] No price data found for ${symbol}`, stockData);
        }

        const result: DataSourceStatus = {
          name: symbol,
          status: 'success',
          data: {
            symbol,
            name: stockData.name || symbol,
            current_price,
            timestamp: new Date().toISOString(),
            source: 'EastMoney API'
          },
          latency
        };

        addLog(`✅ ${symbol}: ¥${current_price.toFixed(2)} (${latency}ms)`, 'success');
        console.log(`✅ [RealDataValidator] Success for ${symbol}: ¥${current_price.toFixed(2)}`);
        return result;
      } else {
        const errorMsg = response?.message || 'No data received';
        console.error(`❌ [RealDataValidator] Invalid response for ${symbol}:`, response);
        throw new Error(errorMsg);
      }
    } catch (error) {
      const latency = Date.now() - startTime;
      console.error(`💥 [RealDataValidator] Exception for ${symbol}:`, error);
      
      const result: DataSourceStatus = {
        name: symbol,
        status: 'error',
        error: error instanceof Error ? error.message : 'Unknown error',
        latency
      };

      addLog(`❌ ${symbol}: ${result.error} (${latency}ms)`, 'error');
      console.error(`📝 [RealDataValidator] Error result for ${symbol}:`, result);
      return result;
    }
  };

  const runDataTest = async () => {
    addLog('🚀 开始实时数据验证测试', 'info');
    
    const results = await Promise.all(
      testSymbols.map(symbol => testSingleStock(symbol))
    );

    const statusMap: Record<string, DataSourceStatus> = {};
    results.forEach(result => {
      statusMap[result.name] = result;
    });

    setDataStatus(statusMap);

    const successCount = results.filter(r => r.status === 'success').length;
    const totalCount = results.length;
    
    if (successCount === totalCount) {
      addLog(`🎉 测试完成: ${successCount}/${totalCount} 股票数据获取成功`, 'success');
      message.success('所有股票数据获取成功！');
    } else {
      addLog(`⚠️ 测试完成: ${successCount}/${totalCount} 股票数据获取成功`, 'error');
      message.warning(`部分数据获取失败: ${successCount}/${totalCount}`);
    }
  };

  const startContinuousTest = () => {
    setIsRunning(true);
    addLog('🔄 启动连续数据监测', 'info');
    
    // 立即运行一次
    runDataTest();
    
    // 每5秒运行一次
    intervalRef.current = setInterval(() => {
      runDataTest();
    }, 5000);
  };

  const stopContinuousTest = () => {
    setIsRunning(false);
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = undefined;
    }
    addLog('⏹️ 停止连续数据监测', 'info');
  };

  useEffect(() => {
    // Auto-run test on component mount
    console.log('🚀 [RealDataValidator] Component mounted, running auto-test...');
    runDataTest();
    
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);

  const columns = [
    {
      title: '股票代码',
      dataIndex: 'symbol',
      key: 'symbol',
      render: (symbol: string) => (
        <Text strong>{symbol}</Text>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string, record: DataSourceStatus) => {
        if (status === 'loading') {
          return <Spin size="small" />;
        }
        return (
          <Tag 
            color={status === 'success' ? 'green' : 'red'}
            icon={status === 'success' ? <CheckCircleOutlined /> : <CloseCircleOutlined />}
          >
            {status === 'success' ? '正常' : '异常'}
          </Tag>
        );
      },
    },
    {
      title: '当前价格',
      dataIndex: ['data', 'current_price'],
      key: 'price',
      render: (price: number, record: DataSourceStatus) => {
        if (record.status !== 'success' || !price) {
          return <Text type="secondary">-</Text>;
        }
        return (
          <Statistic
            value={price}
            precision={2}
            prefix="¥"
            valueStyle={{ fontSize: '14px' }}
          />
        );
      },
    },
    {
      title: '响应时间',
      dataIndex: 'latency',
      key: 'latency',
      render: (latency: number) => {
        if (!latency) return <Text type="secondary">-</Text>;
        
        let color = 'green';
        if (latency > 1000) color = 'red';
        else if (latency > 500) color = 'orange';
        
        return <Text style={{ color }}>{latency}ms</Text>;
      },
    },
    {
      title: '数据源',
      dataIndex: ['data', 'source'],
      key: 'source',
      render: (source: string) => source || <Text type="secondary">-</Text>,
    },
    {
      title: '更新时间',
      dataIndex: ['data', 'timestamp'],
      key: 'timestamp',
      render: (timestamp: string) => {
        if (!timestamp) return <Text type="secondary">-</Text>;
        return <Text>{new Date(timestamp).toLocaleTimeString()}</Text>;
      },
    },
  ];

  const tableData = testSymbols.map(symbol => ({
    key: symbol,
    symbol,
    ...dataStatus[symbol],
  }));

  const successCount = Object.values(dataStatus).filter(s => s.status === 'success').length;
  const errorCount = Object.values(dataStatus).filter(s => s.status === 'error').length;
  const totalCount = testSymbols.length;

  return (
    <Card title="🔍 实时数据验证器" style={{ marginBottom: 24 }}>
      <Space direction="vertical" size="large" style={{ width: '100%' }}>
        {/* 统计信息 */}
        <Row gutter={16}>
          <Col span={6}>
            <Statistic
              title="总测试股票"
              value={totalCount}
              prefix="🎯"
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="成功获取"
              value={successCount}
              valueStyle={{ color: '#3f8600' }}
              prefix="✅"
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="获取失败"
              value={errorCount}
              valueStyle={{ color: '#cf1322' }}
              prefix="❌"
            />
          </Col>
          <Col span={6}>
            <Progress
              type="circle"
              size={60}
              percent={totalCount > 0 ? Math.round((successCount / totalCount) * 100) : 0}
              format={percent => `${percent}%`}
            />
          </Col>
        </Row>

        {/* 控制按钮 */}
        <Space>
          <Button
            type="primary"
            icon={<ReloadOutlined />}
            onClick={runDataTest}
            loading={Object.values(dataStatus).some(s => s.status === 'loading')}
          >
            单次测试
          </Button>
          
          {!isRunning ? (
            <Button
              type="default"
              icon={<PlayCircleOutlined />}
              onClick={startContinuousTest}
            >
              连续监测
            </Button>
          ) : (
            <Button
              type="default"
              icon={<PauseCircleOutlined />}
              onClick={stopContinuousTest}
            >
              停止监测
            </Button>
          )}
        </Space>

        {/* 状态提示 */}
        {successCount === totalCount && totalCount > 0 && (
          <Alert
            message="数据获取状态正常"
            description="所有测试股票都能成功获取到真实的市场数据，EastMoney API工作正常。"
            type="success"
            showIcon
          />
        )}

        {errorCount > 0 && (
          <Alert
            message="部分数据获取异常"
            description={`${errorCount}个股票无法获取数据，请检查网络连接或数据源配置。`}
            type="warning"
            showIcon
          />
        )}

        {/* 数据表格 */}
        <Table
          columns={columns}
          dataSource={tableData}
          pagination={false}
          size="small"
          loading={Object.values(dataStatus).some(s => s.status === 'loading')}
        />

        {/* 日志 */}
        {logs.length > 0 && (
          <Card title="📋 测试日志" size="small">
            <Timeline
              style={{ maxHeight: 200, overflowY: 'auto' }}
              items={logs.slice(0, 10).map(log => ({
                color: log.type === 'success' ? 'green' : log.type === 'error' ? 'red' : 'blue',
                children: (
                  <Space>
                    <Text type="secondary">{log.time}</Text>
                    <Text>{log.message}</Text>
                  </Space>
                ),
              }))}
            />
          </Card>
        )}
      </Space>
    </Card>
  );
};

export default RealDataValidator;
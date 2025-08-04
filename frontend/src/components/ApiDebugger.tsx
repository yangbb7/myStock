import React, { useState } from 'react';
import { Card, Button, Space, message, Typography } from 'antd';
import { api } from '../services/api';

const { Text, Paragraph } = Typography;

const ApiDebugger: React.FC = () => {
  const [loading, setLoading] = useState<{ [key: string]: boolean }>({});
  const [results, setResults] = useState<{ [key: string]: any }>({});

  const testApi = async (apiName: string, apiCall: () => Promise<any>) => {
    setLoading(prev => ({ ...prev, [apiName]: true }));
    
    try {
      console.log(`🔥 [API_DEBUG] Testing ${apiName}...`);
      const result = await apiCall();
      console.log(`🔥 [API_DEBUG] ${apiName} success:`, result);
      
      setResults(prev => ({ ...prev, [apiName]: result }));
      message.success(`${apiName} API调用成功`);
    } catch (error) {
      console.error(`🔥 [API_DEBUG] ${apiName} error:`, error);
      setResults(prev => ({ ...prev, [apiName]: { error: error.message } }));
      message.error(`${apiName} API调用失败: ${error.message}`);
    } finally {
      setLoading(prev => ({ ...prev, [apiName]: false }));
    }
  };

  const renderResult = (apiName: string) => {
    const result = results[apiName];
    if (!result) return null;

    return (
      <Card size="small" style={{ marginTop: 8 }}>
        <Text strong>{apiName} 结果:</Text>
        <Paragraph style={{ marginTop: 8, fontSize: '12px', fontFamily: 'monospace' }}>
          {JSON.stringify(result, null, 2)}
        </Paragraph>
      </Card>
    );
  };

  return (
    <Card title="🔧 API 调试器" style={{ margin: '16px' }}>
      <Space direction="vertical" size="middle" style={{ width: '100%' }}>
        <Space wrap>
          <Button 
            type="primary"
            loading={loading.health}
            onClick={() => testApi('系统健康', () => api.system.getHealth())}
          >
            测试系统健康 API
          </Button>
          
          <Button 
            type="primary"
            loading={loading.metrics}
            onClick={() => testApi('系统指标', () => api.system.getMetrics())}
          >
            测试系统指标 API
          </Button>
          
          <Button 
            type="primary"
            loading={loading.portfolio}
            onClick={() => testApi('投资组合', () => api.portfolio.getSummary())}
          >
            测试投资组合 API
          </Button>
          
          <Button 
            type="primary"
            loading={loading.risk}
            onClick={() => testApi('风险指标', () => api.risk.getMetrics())}
          >
            测试风险指标 API
          </Button>
        </Space>

        <Space wrap>
          <Button 
            onClick={() => {
              setResults({});
              message.info('结果已清空');
            }}
          >
            清空结果
          </Button>
          
          <Button 
            onClick={() => {
              console.log('🔥 [API_DEBUG] All results:', results);
              message.info('结果已输出到控制台');
            }}
          >
            输出到控制台
          </Button>
        </Space>

        {/* 显示结果 */}
        {Object.keys(results).map(apiName => renderResult(apiName))}
      </Space>
    </Card>
  );
};

export default ApiDebugger;
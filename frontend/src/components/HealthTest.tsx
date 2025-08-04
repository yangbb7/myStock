import React, { useEffect, useState } from 'react';
import { Card, Button, Space, Typography } from 'antd';
import { ReloadOutlined } from '@ant-design/icons';

const { Title, Text } = Typography;

const HealthTest: React.FC = () => {
  const [healthData, setHealthData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchHealth = async () => {
    console.log('[HealthTest] Starting health fetch...');
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('http://localhost:8000/health');
      console.log('[HealthTest] Response status:', response.status);
      
      const data = await response.json();
      console.log('[HealthTest] Raw response data:', data);
      
      setHealthData(data);
    } catch (err) {
      console.error('[HealthTest] Error:', err);
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHealth();
  }, []);

  return (
    <Card title="Health API Test" style={{ margin: '20px' }}>
      <Space direction="vertical" style={{ width: '100%' }}>
        <Button 
          icon={<ReloadOutlined />} 
          onClick={fetchHealth}
          loading={loading}
        >
          Refresh Health Data
        </Button>
        
        {error && (
          <div style={{ color: 'red' }}>
            <Title level={5}>Error:</Title>
            <Text>{error}</Text>
          </div>
        )}
        
        {healthData && (
          <div>
            <Title level={5}>Health Data:</Title>
            <pre style={{ background: '#f5f5f5', padding: '10px', borderRadius: '4px' }}>
              {JSON.stringify(healthData, null, 2)}
            </pre>
          </div>
        )}
      </Space>
    </Card>
  );
};

export default HealthTest;
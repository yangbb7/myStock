import React from 'react';
import { Button, Space, message } from 'antd';
import { api } from '../services/api';

const SystemControlTest: React.FC = () => {
  const testHealthCheck = async () => {
    try {
      console.log('[Test] Testing health check...');
      const result = await api.system.getHealth();
      console.log('[Test] Health check result:', result);
      message.success('健康检查成功');
    } catch (error) {
      console.error('[Test] Health check failed:', error);
      message.error(`健康检查失败: ${error}`);
    }
  };

  const testSystemStart = async () => {
    try {
      console.log('[Test] Testing system start...');
      const result = await api.system.startSystem();
      console.log('[Test] System start result:', result);
      message.success('系统启动测试成功');
    } catch (error) {
      console.error('[Test] System start failed:', error);
      message.error(`系统启动测试失败: ${error}`);
    }
  };

  const testSystemStop = async () => {
    try {
      console.log('[Test] Testing system stop...');
      const result = await api.system.stopSystem();
      console.log('[Test] System stop result:', result);
      message.success('系统停止测试成功');
    } catch (error) {
      console.error('[Test] System stop failed:', error);
      message.error(`系统停止测试失败: ${error}`);
    }
  };

  const testDirectApiCall = async () => {
    try {
      console.log('[Test] Testing direct API call...');
      const response = await fetch('http://localhost:8000/system/start', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({}),
      });
      
      console.log('[Test] Direct API response status:', response.status);
      const data = await response.json();
      console.log('[Test] Direct API response data:', data);
      
      if (response.ok) {
        message.success('直接API调用成功');
      } else {
        message.error(`直接API调用失败: ${response.status}`);
      }
    } catch (error) {
      console.error('[Test] Direct API call failed:', error);
      message.error(`直接API调用失败: ${error}`);
    }
  };

  return (
    <div style={{ padding: '20px', border: '1px solid #d9d9d9', margin: '20px' }}>
      <h3>系统控制API测试</h3>
      <Space>
        <Button onClick={testHealthCheck}>
          测试健康检查
        </Button>
        <Button onClick={testSystemStart} type="primary">
          测试系统启动
        </Button>
        <Button onClick={testSystemStop} danger>
          测试系统停止
        </Button>
        <Button onClick={testDirectApiCall}>
          直接API调用测试
        </Button>
      </Space>
      <div style={{ marginTop: '10px', fontSize: '12px', color: '#666' }}>
        请打开浏览器控制台查看详细日志
      </div>
    </div>
  );
};

export default SystemControlTest;
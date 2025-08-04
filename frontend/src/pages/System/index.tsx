import React, { useState } from 'react';
import { Tabs, Card } from 'antd';
import {
  DashboardOutlined,
  SettingOutlined,
  ControlOutlined,
  MonitorOutlined,
} from '@ant-design/icons';
import SystemStatusMonitor from './SystemStatusMonitor';
import SystemConfigManager from './SystemConfigManager';
import SystemControlPanel from './SystemControlPanel';
import SystemMonitoringDashboard from './SystemMonitoringDashboard';

// const { TabPane } = Tabs; // Not used with new Tabs API

const SystemManagementPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState('status');

  const tabItems = [
    {
      key: 'status',
      label: (
        <span>
          <DashboardOutlined />
          系统状态
        </span>
      ),
      children: <SystemStatusMonitor />,
    },
    {
      key: 'control',
      label: (
        <span>
          <ControlOutlined />
          系统控制
        </span>
      ),
      children: <SystemControlPanel />,
    },
    {
      key: 'config',
      label: (
        <span>
          <SettingOutlined />
          配置管理
        </span>
      ),
      children: <SystemConfigManager />,
    },
    {
      key: 'monitoring',
      label: (
        <span>
          <MonitorOutlined />
          监控仪表板
        </span>
      ),
      children: <SystemMonitoringDashboard />,
    },
  ];

  return (
    <div style={{ padding: 24, background: '#f0f2f5', minHeight: '100vh' }}>
      <Card>
        <Tabs
          activeKey={activeTab}
          onChange={setActiveTab}
          size="large"
          tabPosition="top"
          items={tabItems}
        />
      </Card>
    </div>
  );
};

export default SystemManagementPage;
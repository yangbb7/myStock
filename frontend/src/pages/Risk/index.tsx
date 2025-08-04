import React from 'react';
import { Typography } from 'antd';
import RiskDashboard from './RiskDashboard';
import RiskAlertsSystem from './RiskAlertsSystem';
import RiskControlPanel from './RiskControlPanel';

const { Title } = Typography;

const RiskMonitoringPage: React.FC = () => {
  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>风险监控</Title>
      </div>
      
      <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
        {/* Risk Indicators Dashboard */}
        <RiskDashboard />
        
        {/* Risk Alerts System */}
        <RiskAlertsSystem />
        
        {/* Risk Control Panel */}
        <RiskControlPanel />
      </div>
    </div>
  );
};

export default RiskMonitoringPage;
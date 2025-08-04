import React from 'react';
import { Row, Col, Space } from 'antd';
import { 
  SystemStatusCard, 
  PortfolioOverviewCard, 
  RiskAlertsCard, 
  SystemControlPanel 
} from './components';
import ApiTestingPanel from '../../components/ApiTestingPanel';
import ApiDebugger from '../../components/ApiDebugger';
import HealthTest from '../../components/HealthTest';
import RealDataValidator from '../../components/RealDataValidator';

const DashboardPage: React.FC = () => {
  return (
    <div style={{ padding: '24px' }}>
      <Space direction="vertical" size="large" style={{ width: '100%' }}>
        {/* Real Data Validator - Shows live stock data status */}
        <RealDataValidator />
        
        {/* Health Test Component - For debugging */}
        <HealthTest />
        
        {/* API Debugger - Added for troubleshooting */}
        <ApiDebugger />
        
        {/* Integrated API Testing Panel - Production Ready */}
        <ApiTestingPanel />
        
        {/* System Control Panel */}
        <SystemControlPanel />
        
        {/* Main Dashboard Grid */}
        <Row gutter={[24, 24]}>
          {/* System Status - Full Width */}
          <Col span={24}>
            <SystemStatusCard />
          </Col>
          
          {/* Portfolio Overview - Left Half */}
          <Col span={12}>
            <PortfolioOverviewCard />
          </Col>
          
          {/* Risk Alerts - Right Half */}
          <Col span={12}>
            <RiskAlertsCard />
          </Col>
        </Row>
      </Space>
    </div>
  );
};

export default DashboardPage;
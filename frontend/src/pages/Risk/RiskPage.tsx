import React from 'react';
import { Tabs, Card } from 'antd';
import { DashboardOutlined, AlertOutlined, SafetyOutlined, BarChartOutlined } from '@ant-design/icons';
import RiskDashboard from './RiskDashboard';
import RiskAlertsSystem from './RiskAlertsSystem';
import RiskControlPanel from './RiskControlPanel';
import IntelligentRiskControl from '@/components/RiskControl/IntelligentRiskControl';

const { TabPane } = Tabs;

const RiskPage: React.FC = () => {
  return (
    <div className="risk-page">
      <Card>
        <Tabs defaultActiveKey="intelligent">
          <TabPane
            tab={
              <span>
                <SafetyOutlined />
                智能风控
              </span>
            }
            key="intelligent"
          >
            <IntelligentRiskControl />
          </TabPane>
          
          <TabPane
            tab={
              <span>
                <DashboardOutlined />
                风险概览
              </span>
            }
            key="dashboard"
          >
            <RiskDashboard />
          </TabPane>
          
          <TabPane
            tab={
              <span>
                <AlertOutlined />
                警报系统
              </span>
            }
            key="alerts"
          >
            <RiskAlertsSystem />
          </TabPane>
          
          <TabPane
            tab={
              <span>
                <BarChartOutlined />
                风控面板
              </span>
            }
            key="control"
          >
            <RiskControlPanel />
          </TabPane>
        </Tabs>
      </Card>
    </div>
  );
};

export default RiskPage;
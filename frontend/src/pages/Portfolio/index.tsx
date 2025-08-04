import React from 'react';
import { Row, Col, Tabs } from 'antd';
import type { TabsProps } from 'antd';
import { PositionManagement } from './components/PositionManagement';
import { PerformanceAnalysis } from './components/PerformanceAnalysis';
import { RiskAnalysis } from './components/RiskAnalysis';
import { ReportExport } from './components/ReportExport';

const { TabPane } = Tabs;

export const PortfolioPage: React.FC = () => {
  const tabItems: TabsProps['items'] = [
    {
      key: 'positions',
      label: '持仓管理',
      children: <PositionManagement />,
    },
    {
      key: 'performance',
      label: '收益分析',
      children: <PerformanceAnalysis />,
    },
    {
      key: 'risk',
      label: '风险分析',
      children: <RiskAnalysis />,
    },
    {
      key: 'reports',
      label: '报告导出',
      children: <ReportExport />,
    },
  ];

  return (
    <div style={{ padding: '24px' }}>
      <Row gutter={[16, 16]}>
        <Col span={24}>
          <Tabs
            defaultActiveKey="positions"
            items={tabItems}
            size="large"
            style={{ minHeight: '600px' }}
          />
        </Col>
      </Row>
    </div>
  );
};

export default PortfolioPage;
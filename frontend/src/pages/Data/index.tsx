import React from 'react';
import { Layout, Row, Col, Card, Space } from 'antd';
import { MarketDataTable } from './components/MarketDataTable';
import { DataProcessingStatus } from './components/DataProcessingStatus';
import { EnhancedCandlestickChart } from './components/EnhancedCandlestickChart';

const { Content } = Layout;

const DataMonitoringPage: React.FC = () => {
  return (
    <Layout>
      <Content style={{ padding: '24px' }}>
        <Space direction="vertical" size="large" style={{ width: '100%' }}>
          <Row gutter={[16, 16]}>
            <Col span={24}>
              <Card title="实时数据监控" size="small">
                <p>监控市场数据实时更新、数据处理状态和历史数据查询</p>
              </Card>
            </Col>
          </Row>

          <Row gutter={[16, 16]}>
            <Col span={24}>
              <MarketDataTable />
            </Col>
          </Row>

          <Row gutter={[16, 16]}>
            <Col span={16}>
              <EnhancedCandlestickChart />
            </Col>
            <Col span={8}>
              <DataProcessingStatus />
            </Col>
          </Row>
        </Space>
      </Content>
    </Layout>
  );
};

export default DataMonitoringPage;
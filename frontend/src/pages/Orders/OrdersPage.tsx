import React, { useState } from 'react';
import {
  Tabs,
  Button,
  Modal,
  Space,
  Typography,
  Card,
} from 'antd';
import {
  PlusOutlined,
  UnorderedListOutlined,
  BarChartOutlined,
  StockOutlined,
} from '@ant-design/icons';
import OrderCreateForm from '../../components/Orders/OrderCreateForm';
import OrderStatusMonitor from '../../components/Orders/OrderStatusMonitor';
import OrderAnalytics from '../../components/Orders/OrderAnalytics';

const { TabPane } = Tabs;
const { Title } = Typography;

const OrdersPage: React.FC = () => {
  const [createOrderModalVisible, setCreateOrderModalVisible] = useState(false);
  const [activeTab, setActiveTab] = useState('monitor');

  const handleOrderCreated = (orderId: string) => {
    setCreateOrderModalVisible(false);
    // Switch to monitor tab to see the new order
    setActiveTab('monitor');
  };

  const tabItems = [
    {
      key: 'monitor',
      label: (
        <Space>
          <UnorderedListOutlined />
          订单监控
        </Space>
      ),
      children: (
        <OrderStatusMonitor
          refreshInterval={5000}
          showFilters={true}
          showStats={true}
        />
      ),
    },
    {
      key: 'analytics',
      label: (
        <Space>
          <BarChartOutlined />
          统计分析
        </Space>
      ),
      children: <OrderAnalytics />,
    },
  ];

  return (
    <div style={{ padding: '24px' }}>
      {/* Page Header */}
        <div style={{ marginBottom: 24 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <Title level={2} style={{ margin: 0, display: 'flex', alignItems: 'center' }}>
                <StockOutlined style={{ marginRight: 12, color: '#1890ff' }} />
                订单管理
              </Title>
              <div style={{ marginTop: 8, color: '#666' }}>
                管理和监控所有交易订单，查看执行状态和性能分析
              </div>
            </div>
            <Button
              type="primary"
              size="large"
              icon={<PlusOutlined />}
              onClick={() => setCreateOrderModalVisible(true)}
            >
              创建订单
            </Button>
          </div>
        </div>

        {/* Main Content */}
        <Card style={{ minHeight: 'calc(100vh - 200px)' }}>
          <Tabs
            activeKey={activeTab}
            onChange={setActiveTab}
            items={tabItems}
            size="large"
            tabBarStyle={{ marginBottom: 24 }}
          />
        </Card>

        {/* Create Order Modal */}
        <Modal
          title="创建新订单"
          open={createOrderModalVisible}
          onCancel={() => setCreateOrderModalVisible(false)}
          footer={null}
          width={800}
          destroyOnClose
        >
          <OrderCreateForm
            onSuccess={handleOrderCreated}
            onCancel={() => setCreateOrderModalVisible(false)}
          />
        </Modal>
    </div>
  );
};

export default OrdersPage;
import React from 'react';
import { Skeleton, Card, Row, Col, Space } from 'antd';

// Dashboard skeleton components
export const DashboardSkeleton: React.FC = () => {
  return (
    <div className="dashboard-skeleton">
      <Row gutter={[16, 16]}>
        {/* System status cards */}
        <Col span={24}>
          <Row gutter={16}>
            {[1, 2, 3, 4].map(i => (
              <Col span={6} key={i}>
                <Card>
                  <Skeleton active paragraph={{ rows: 2 }} />
                </Card>
              </Col>
            ))}
          </Row>
        </Col>
        
        {/* Main content area */}
        <Col span={16}>
          <Card title={<Skeleton.Input style={{ width: 200 }} active />}>
            <Skeleton active paragraph={{ rows: 8 }} />
          </Card>
        </Col>
        
        {/* Side panel */}
        <Col span={8}>
          <Space direction="vertical" style={{ width: '100%' }} size="middle">
            <Card title={<Skeleton.Input style={{ width: 150 }} active />}>
              <Skeleton active paragraph={{ rows: 4 }} />
            </Card>
            <Card title={<Skeleton.Input style={{ width: 150 }} active />}>
              <Skeleton active paragraph={{ rows: 3 }} />
            </Card>
          </Space>
        </Col>
      </Row>
    </div>
  );
};

// Table skeleton component
export const TableSkeleton: React.FC<{ rows?: number; columns?: number }> = ({ 
  rows = 5, 
  columns = 4 
}) => {
  return (
    <div className="table-skeleton">
      {/* Table header */}
      <Row gutter={16} style={{ marginBottom: 16 }}>
        {Array.from({ length: columns }).map((_, i) => (
          <Col span={24 / columns} key={i}>
            <Skeleton.Input style={{ width: '100%' }} active />
          </Col>
        ))}
      </Row>
      
      {/* Table rows */}
      {Array.from({ length: rows }).map((_, rowIndex) => (
        <Row gutter={16} style={{ marginBottom: 8 }} key={rowIndex}>
          {Array.from({ length: columns }).map((_, colIndex) => (
            <Col span={24 / columns} key={colIndex}>
              <Skeleton.Input 
                style={{ width: '100%', height: 32 }} 
                active 
                size="small"
              />
            </Col>
          ))}
        </Row>
      ))}
    </div>
  );
};

// Chart skeleton component
export const ChartSkeleton: React.FC<{ height?: number }> = ({ height = 400 }) => {
  return (
    <div className="chart-skeleton" style={{ height }}>
      <Skeleton.Node 
        active 
        style={{ 
          width: '100%', 
          height: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center'
        }}
      >
        <div style={{ 
          width: '80%', 
          height: '80%', 
          background: 'linear-gradient(90deg, #f0f0f0 25%, #e6e6e6 50%, #f0f0f0 75%)',
          backgroundSize: '200% 100%',
          animation: 'loading 1.5s infinite',
          borderRadius: 4
        }} />
      </Skeleton.Node>
    </div>
  );
};

// Form skeleton component
export const FormSkeleton: React.FC<{ fields?: number }> = ({ fields = 6 }) => {
  return (
    <div className="form-skeleton">
      <Row gutter={[16, 16]}>
        {Array.from({ length: fields }).map((_, i) => (
          <Col span={i % 2 === 0 ? 12 : 12} key={i}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Skeleton.Input style={{ width: 100 }} active size="small" />
              <Skeleton.Input style={{ width: '100%' }} active />
            </Space>
          </Col>
        ))}
        <Col span={24}>
          <Skeleton.Button active style={{ width: 100 }} />
        </Col>
      </Row>
    </div>
  );
};

// Strategy list skeleton
export const StrategyListSkeleton: React.FC = () => {
  return (
    <div className="strategy-list-skeleton">
      <Space direction="vertical" style={{ width: '100%' }} size="middle">
        {[1, 2, 3].map(i => (
          <Card key={i}>
            <Row gutter={16} align="middle">
              <Col span={6}>
                <Skeleton.Input style={{ width: '100%' }} active />
              </Col>
              <Col span={4}>
                <Skeleton.Input style={{ width: '100%' }} active size="small" />
              </Col>
              <Col span={4}>
                <Skeleton.Input style={{ width: '100%' }} active size="small" />
              </Col>
              <Col span={4}>
                <Skeleton.Input style={{ width: '100%' }} active size="small" />
              </Col>
              <Col span={6}>
                <Space>
                  <Skeleton.Button active size="small" />
                  <Skeleton.Button active size="small" />
                </Space>
              </Col>
            </Row>
          </Card>
        ))}
      </Space>
    </div>
  );
};

// Portfolio skeleton
export const PortfolioSkeleton: React.FC = () => {
  return (
    <div className="portfolio-skeleton">
      <Row gutter={[16, 16]}>
        {/* Summary cards */}
        <Col span={24}>
          <Row gutter={16}>
            {[1, 2, 3, 4].map(i => (
              <Col span={6} key={i}>
                <Card>
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <Skeleton.Input style={{ width: 80 }} active size="small" />
                    <Skeleton.Input style={{ width: 120 }} active />
                  </Space>
                </Card>
              </Col>
            ))}
          </Row>
        </Col>
        
        {/* Chart area */}
        <Col span={16}>
          <Card title={<Skeleton.Input style={{ width: 150 }} active />}>
            <ChartSkeleton height={300} />
          </Card>
        </Col>
        
        {/* Holdings table */}
        <Col span={8}>
          <Card title={<Skeleton.Input style={{ width: 100 }} active />}>
            <TableSkeleton rows={6} columns={2} />
          </Card>
        </Col>
      </Row>
    </div>
  );
};

// Order management skeleton
export const OrderSkeleton: React.FC = () => {
  return (
    <div className="order-skeleton">
      <Row gutter={[16, 16]}>
        {/* Order form */}
        <Col span={8}>
          <Card title={<Skeleton.Input style={{ width: 100 }} active />}>
            <FormSkeleton fields={4} />
          </Card>
        </Col>
        
        {/* Order list */}
        <Col span={16}>
          <Card title={<Skeleton.Input style={{ width: 120 }} active />}>
            <TableSkeleton rows={8} columns={6} />
          </Card>
        </Col>
      </Row>
    </div>
  );
};

// Risk monitoring skeleton
export const RiskSkeleton: React.FC = () => {
  return (
    <div className="risk-skeleton">
      <Row gutter={[16, 16]}>
        {/* Risk metrics */}
        <Col span={24}>
          <Row gutter={16}>
            {[1, 2, 3, 4, 5, 6].map(i => (
              <Col span={4} key={i}>
                <Card>
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <Skeleton.Input style={{ width: 60 }} active size="small" />
                    <Skeleton.Input style={{ width: 80 }} active />
                  </Space>
                </Card>
              </Col>
            ))}
          </Row>
        </Col>
        
        {/* Risk chart */}
        <Col span={16}>
          <Card title={<Skeleton.Input style={{ width: 120 }} active />}>
            <ChartSkeleton height={350} />
          </Card>
        </Col>
        
        {/* Risk alerts */}
        <Col span={8}>
          <Card title={<Skeleton.Input style={{ width: 100 }} active />}>
            <Space direction="vertical" style={{ width: '100%' }} size="middle">
              {[1, 2, 3, 4].map(i => (
                <div key={i} style={{ padding: 12, border: '1px solid #f0f0f0', borderRadius: 4 }}>
                  <Skeleton active paragraph={{ rows: 2 }} />
                </div>
              ))}
            </Space>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

// Market data skeleton
export const MarketDataSkeleton: React.FC = () => {
  return (
    <div className="market-data-skeleton">
      <Row gutter={[16, 16]}>
        {/* Market overview */}
        <Col span={24}>
          <Card title={<Skeleton.Input style={{ width: 120 }} active />}>
            <TableSkeleton rows={10} columns={6} />
          </Card>
        </Col>
        
        {/* Chart section */}
        <Col span={16}>
          <Card title={<Skeleton.Input style={{ width: 100 }} active />}>
            <ChartSkeleton height={400} />
          </Card>
        </Col>
        
        {/* Market stats */}
        <Col span={8}>
          <Space direction="vertical" style={{ width: '100%' }} size="middle">
            <Card title={<Skeleton.Input style={{ width: 80 }} active />}>
              <Skeleton active paragraph={{ rows: 3 }} />
            </Card>
            <Card title={<Skeleton.Input style={{ width: 80 }} active />}>
              <Skeleton active paragraph={{ rows: 4 }} />
            </Card>
          </Space>
        </Col>
      </Row>
    </div>
  );
};

// Generic page skeleton
export const PageSkeleton: React.FC<{ 
  title?: boolean;
  breadcrumb?: boolean;
  content?: 'dashboard' | 'table' | 'form' | 'chart' | 'mixed';
}> = ({ 
  title = true, 
  breadcrumb = true, 
  content = 'mixed' 
}) => {
  const renderContent = () => {
    switch (content) {
      case 'dashboard':
        return <DashboardSkeleton />;
      case 'table':
        return <TableSkeleton rows={10} columns={6} />;
      case 'form':
        return <FormSkeleton fields={8} />;
      case 'chart':
        return <ChartSkeleton height={500} />;
      case 'mixed':
      default:
        return (
          <Row gutter={[16, 16]}>
            <Col span={16}>
              <Card>
                <ChartSkeleton height={300} />
              </Card>
            </Col>
            <Col span={8}>
              <Card>
                <TableSkeleton rows={6} columns={2} />
              </Card>
            </Col>
          </Row>
        );
    }
  };

  return (
    <div className="page-skeleton">
      {breadcrumb && (
        <div style={{ marginBottom: 16 }}>
          <Skeleton.Input style={{ width: 200 }} active size="small" />
        </div>
      )}
      
      {title && (
        <div style={{ marginBottom: 24 }}>
          <Skeleton.Input style={{ width: 300 }} active />
        </div>
      )}
      
      {renderContent()}
    </div>
  );
};

// Add CSS animations
const skeletonStyles = `
@keyframes loading {
  0% {
    background-position: 200% 0;
  }
  100% {
    background-position: -200% 0;
  }
}

.chart-skeleton .ant-skeleton-element {
  width: 100% !important;
  height: 100% !important;
}
`;

// Inject styles
if (typeof document !== 'undefined') {
  const styleElement = document.createElement('style');
  styleElement.textContent = skeletonStyles;
  document.head.appendChild(styleElement);
}

export default {
  DashboardSkeleton,
  TableSkeleton,
  ChartSkeleton,
  FormSkeleton,
  StrategyListSkeleton,
  PortfolioSkeleton,
  OrderSkeleton,
  RiskSkeleton,
  MarketDataSkeleton,
  PageSkeleton,
};
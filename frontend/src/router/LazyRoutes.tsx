import React from 'react';
import { lazyImport, LazyLoadingSpinner } from '../utils/lazyLoading';

// Lazy load all page components for better performance
export const DashboardPage = lazyImport(
  () => import('../pages/Dashboard/DashboardPage'),
  3,
  1000
);

export const StrategyPage = lazyImport(
  () => import('../pages/Strategy/StrategyPage'),
  3,
  1000
);

export const DataPage = lazyImport(
  () => import('../pages/Data/DataPage'),
  3,
  1000
);

export const OrdersPage = lazyImport(
  () => import('../pages/Orders/OrdersPage'),
  3,
  1000
);

export const PortfolioPage = lazyImport(
  () => import('../pages/Portfolio/PortfolioPage'),
  3,
  1000
);

export const RiskPage = lazyImport(
  () => import('../pages/Risk/RiskPage'),
  3,
  1000
);

export const BacktestPage = lazyImport(
  () => import('../pages/Backtest/BacktestPage'),
  3,
  1000
);

export const SystemPage = lazyImport(
  () => import('../pages/System/SystemPage'),
  3,
  1000
);

// Lazy load heavy components
export const CandlestickChart = lazyImport(
  () => import('../components/Charts/CandlestickChart'),
  2,
  500
);

export const PerformanceChart = lazyImport(
  () => import('../components/Charts/PerformanceChart'),
  2,
  500
);

export const RiskMetricsChart = lazyImport(
  () => import('../components/Charts/RiskMetricsChart'),
  2,
  500
);

// Custom loading components for different contexts
export const ChartLoadingSpinner: React.FC = () => (
  <div style={{ 
    display: 'flex', 
    justifyContent: 'center', 
    alignItems: 'center', 
    minHeight: '300px',
    background: '#fafafa',
    borderRadius: '6px'
  }}>
    <LazyLoadingSpinner tip="加载图表中..." />
  </div>
);

export const PageLoadingSpinner: React.FC = () => (
  <div style={{ 
    display: 'flex', 
    justifyContent: 'center', 
    alignItems: 'center', 
    minHeight: '60vh'
  }}>
    <LazyLoadingSpinner tip="加载页面中..." />
  </div>
);
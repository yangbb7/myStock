import React, { lazy, Suspense } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { Spin } from 'antd';
import ProtectedRoute from '@/components/auth/ProtectedRoute';
import { MainLayout } from '@/components/Layout/MainLayout';

// Auth pages (not lazy loaded for better UX)
import Login from '@/pages/auth/Login';
import Register from '@/pages/auth/Register';

// Lazy load protected pages
const DashboardPage = lazy(() => import('@/pages/Dashboard'));
const StrategyPage = lazy(() => import('@/pages/Strategy'));
const DataPage = lazy(() => import('@/pages/Data'));
const OrdersPage = lazy(() => import('@/pages/Orders'));
const PortfolioPage = lazy(() => import('@/pages/Portfolio'));
const RiskPage = lazy(() => import('@/pages/Risk'));
const BacktestPage = lazy(() => import('@/pages/Backtest'));
const SystemPage = lazy(() => import('@/pages/System'));

// Loading component
const PageLoading = () => (
  <div style={{ 
    display: 'flex', 
    justifyContent: 'center', 
    alignItems: 'center', 
    height: '100vh' 
  }}>
    <Spin size="large" tip="加载中..." />
  </div>
);

export const AppRoutes: React.FC = () => {
  return (
    <Routes>
      {/* Public routes */}
      <Route path="/login" element={<Login />} />
      <Route path="/register" element={<Register />} />
      
      {/* Protected routes */}
      <Route
        path="/*"
        element={
          <ProtectedRoute>
            <MainLayout>
              <Suspense fallback={<PageLoading />}>
                <Routes>
                  <Route path="/" element={<Navigate to="/dashboard" replace />} />
                  <Route path="/dashboard" element={<DashboardPage />} />
                  <Route path="/strategy" element={<StrategyPage />} />
                  <Route path="/data" element={<DataPage />} />
                  <Route path="/orders" element={<OrdersPage />} />
                  <Route path="/portfolio" element={<PortfolioPage />} />
                  <Route path="/risk" element={<RiskPage />} />
                  <Route path="/backtest" element={<BacktestPage />} />
                  <Route path="/system" element={<SystemPage />} />
                  
                  {/* 404 */}
                  <Route path="*" element={<Navigate to="/dashboard" replace />} />
                </Routes>
              </Suspense>
            </MainLayout>
          </ProtectedRoute>
        }
      />
    </Routes>
  );
};
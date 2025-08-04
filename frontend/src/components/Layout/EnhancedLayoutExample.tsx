import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MainLayout } from './MainLayout';
import { ThemeProvider } from './ThemeProvider';

// Import enhanced page components with API integration
import EnhancedDashboardPage from '../Pages/EnhancedDashboardPage';
import EnhancedStrategyPage from '../Pages/EnhancedStrategyPage';
import EnhancedDataPage from '../Pages/EnhancedDataPage';
import EnhancedOrdersPage from '../Pages/EnhancedOrdersPage';
import EnhancedPortfolioPage from '../Pages/EnhancedPortfolioPage';
import EnhancedRiskPage from '../Pages/EnhancedRiskPage';
import EnhancedSystemPage from '../Pages/EnhancedSystemPage';
import BacktestPage from '../../pages/Backtest/BacktestPage';
import WebSocketTest from '../WebSocketTest';

// Create React Query client with optimized settings
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30000, // 30 seconds
      cacheTime: 300000, // 5 minutes
      refetchOnWindowFocus: false,
      retry: (failureCount, error: any) => {
        // Don't retry on business logic errors
        if (error?.code === 'BUSINESS_LOGIC_ERROR' || error?.code === 'SERVICE_UNAVAILABLE') {
          return false;
        }
        return failureCount < 3;
      },
      retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
    },
    mutations: {
      retry: 1,
      retryDelay: 1000,
    },
  },
});

export const EnhancedLayoutExample: React.FC = () => {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <Router>
          <MainLayout>
            <Routes>
              <Route path="/" element={<EnhancedDashboardPage />} />
              <Route path="/dashboard" element={<EnhancedDashboardPage />} />
              <Route path="/strategy" element={<EnhancedStrategyPage />} />
              <Route path="/data" element={<EnhancedDataPage />} />
              <Route path="/orders" element={<EnhancedOrdersPage />} />
              <Route path="/portfolio" element={<EnhancedPortfolioPage />} />
              <Route path="/risk" element={<EnhancedRiskPage />} />
              <Route path="/backtest" element={<BacktestPage />} />
              <Route path="/system" element={<EnhancedSystemPage />} />
              <Route path="/websocket-test" element={<WebSocketTest />} />
            </Routes>
          </MainLayout>
        </Router>
      </ThemeProvider>
    </QueryClientProvider>
  );
};

export default EnhancedLayoutExample;
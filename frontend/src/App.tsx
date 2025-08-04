
import React, { useEffect, Suspense } from 'react';
import { LayoutExample } from './components/Layout/LayoutExample';
import { ErrorBoundary } from './components/Common/ErrorBoundary';
import { ErrorHandlingProvider } from './utils/errorHandlingSetup';
import { NetworkStatusIndicator } from './components/Common/NetworkMonitor';
import { usePerformanceDashboard } from './components/Common/PerformanceDashboard';
import { performanceMonitor } from './utils/performanceMonitor';
import { usePerformanceOptimization } from './hooks/usePerformanceOptimization';
import { performanceOptimizer, initializePerformanceOptimization } from './utils/performanceOptimizer';
import { LazyLoadingSpinner } from './utils/lazyLoading';
import './App.css';

function App() {
  // Initialize performance optimization for the main app
  const optimization = usePerformanceOptimization({
    componentName: 'App',
    enableCaching: true,
    enableMemoryMonitoring: true,
    maxMemoryUsage: 100 // 100MB threshold for main app
  });

  // Performance dashboard
  const { PerformanceDashboard } = usePerformanceDashboard();

  useEffect(() => {
    // Initialize performance optimization system
    initializePerformanceOptimization();
    
    // Start performance monitoring
    performanceMonitor.startMonitoring(10000); // Monitor every 10 seconds

    // Log performance metrics in development
    if (process.env.NODE_ENV === 'development') {
      const interval = setInterval(() => {
        const appReport = optimization.getMetrics();
        const optimizerReport = performanceOptimizer.getOptimizationReport();
        
        console.group('🚀 Performance Report');
        console.log('App Metrics:', appReport);
        console.log('Optimizer Report:', optimizerReport);
        console.groupEnd();
      }, 30000); // Log every 30 seconds

      return () => clearInterval(interval);
    }

    // Cleanup on unmount
    return () => {
      optimization.cleanup();
    };
  }, [optimization]);

  return (
    <ErrorHandlingProvider
      config={{
        enableErrorReporting: true,
        enableGlobalHandlers: true,
        enableConsoleLogging: process.env.NODE_ENV === 'development',
        notificationDuration: 5,
      }}
    >
      <ErrorBoundary
        onError={(error, errorInfo) => {
          console.error('App Error Boundary:', error, errorInfo);
        }}
      >
        <div className="App">
          <Suspense fallback={<LazyLoadingSpinner tip="加载应用中..." />}>
            <LayoutExample />
          </Suspense>
          <NetworkStatusIndicator position="bottomRight" />
          {process.env.NODE_ENV === 'development' && <PerformanceDashboard />}
        </div>
      </ErrorBoundary>
    </ErrorHandlingProvider>
  );
}

export default App;

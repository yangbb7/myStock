import React, { Suspense, ComponentType } from 'react';
import { Spin } from 'antd';

// Loading component for lazy-loaded components
export const LazyLoadingSpinner: React.FC<{ tip?: string }> = ({ tip = '加载中...' }) => (
  <div style={{ 
    display: 'flex', 
    justifyContent: 'center', 
    alignItems: 'center', 
    minHeight: '200px' 
  }}>
    <Spin size="large" tip={tip} />
  </div>
);

// Higher-order component for lazy loading with error boundary
export const withLazyLoading = <P extends object>(
  Component: ComponentType<P>,
  fallback?: React.ReactNode
) => {
  const LazyComponent = React.lazy(() => Promise.resolve({ default: Component }));
  
  return React.forwardRef<any, P>((props, ref) => (
    <Suspense fallback={fallback || <LazyLoadingSpinner />}>
      <LazyComponent {...props} ref={ref} />
    </Suspense>
  ));
};

// Utility for dynamic imports with retry logic
export const lazyImport = <T extends ComponentType<any>>(
  importFn: () => Promise<{ default: T }>,
  retries = 3,
  delay = 1000
): React.LazyExoticComponent<T> => {
  return React.lazy(async () => {
    let lastError: Error | null = null;
    
    for (let i = 0; i < retries; i++) {
      try {
        return await importFn();
      } catch (error) {
        lastError = error as Error;
        if (i < retries - 1) {
          await new Promise(resolve => setTimeout(resolve, delay * (i + 1)));
        }
      }
    }
    
    throw lastError;
  });
};

// Preload utility for critical routes
export const preloadRoute = (importFn: () => Promise<any>) => {
  const componentImport = importFn();
  return componentImport;
};

// Route-based code splitting helper
export const createLazyRoute = (
  importFn: () => Promise<{ default: ComponentType<any> }>,
  fallback?: React.ReactNode
) => {
  const LazyComponent = lazyImport(importFn);
  
  return (props: any) => (
    <Suspense fallback={fallback || <LazyLoadingSpinner />}>
      <LazyComponent {...props} />
    </Suspense>
  );
};
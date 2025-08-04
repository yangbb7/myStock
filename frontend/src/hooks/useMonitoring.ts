/**
 * 监控服务React Hook
 * 提供便捷的监控功能集成
 */

import { useEffect, useCallback, useRef } from 'react';
import { 
  monitoring, 
  trackEvent, 
  trackError, 
  trackUserAction, 
  trackPerformance,
  trackBusiness,
  logger 
} from '@/services/monitoring';

export interface UseMonitoringOptions {
  trackPageViews?: boolean;
  trackUserInteractions?: boolean;
  trackPerformanceMetrics?: boolean;
  trackErrors?: boolean;
}

export const useMonitoring = (options: UseMonitoringOptions = {}) => {
  const {
    trackPageViews = true,
    trackUserInteractions = true,
    trackPerformanceMetrics = true,
    trackErrors = true,
  } = options;

  const pageStartTime = useRef<number>(Date.now());
  const interactionCount = useRef<number>(0);

  // 页面浏览跟踪
  useEffect(() => {
    if (!trackPageViews) return;

    const startTime = Date.now();
    pageStartTime.current = startTime;

    // 跟踪页面进入
    trackEvent('page_enter', {
      url: window.location.href,
      title: document.title,
      referrer: document.referrer,
    });

    logger.info('Page entered', {
      url: window.location.href,
      title: document.title,
    });

    // 页面离开时跟踪
    return () => {
      const duration = Date.now() - startTime;
      trackEvent('page_leave', {
        url: window.location.href,
        duration,
        interactionCount: interactionCount.current,
      });

      trackPerformance('page_duration', duration, 'ms', {
        url: window.location.href,
        interactionCount: interactionCount.current,
      });

      logger.info('Page left', {
        url: window.location.href,
        duration,
        interactionCount: interactionCount.current,
      });
    };
  }, [trackPageViews]);

  // 错误跟踪
  useEffect(() => {
    if (!trackErrors) return;

    const handleError = (event: ErrorEvent) => {
      trackError(new Error(event.message), {
        filename: event.filename,
        lineno: event.lineno,
        colno: event.colno,
      });
    };

    const handleUnhandledRejection = (event: PromiseRejectionEvent) => {
      trackError(new Error(`Unhandled Promise Rejection: ${event.reason}`), {
        reason: event.reason,
      });
    };

    window.addEventListener('error', handleError);
    window.addEventListener('unhandledrejection', handleUnhandledRejection);

    return () => {
      window.removeEventListener('error', handleError);
      window.removeEventListener('unhandledrejection', handleUnhandledRejection);
    };
  }, [trackErrors]);

  // 性能指标跟踪
  useEffect(() => {
    if (!trackPerformanceMetrics) return;

    // 跟踪页面加载性能
    const trackPageLoadPerformance = () => {
      if (document.readyState === 'complete') {
        const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
        if (navigation) {
          trackPerformance('dom_content_loaded', navigation.domContentLoadedEventEnd - navigation.domContentLoadedEventStart);
          trackPerformance('load_complete', navigation.loadEventEnd - navigation.loadEventStart);
          trackPerformance('first_byte', navigation.responseStart - navigation.requestStart);
        }
      }
    };

    if (document.readyState === 'complete') {
      trackPageLoadPerformance();
    } else {
      window.addEventListener('load', trackPageLoadPerformance);
    }

    return () => {
      window.removeEventListener('load', trackPageLoadPerformance);
    };
  }, [trackPerformanceMetrics]);

  // 用户交互跟踪
  useEffect(() => {
    if (!trackUserInteractions) return;

    const handleClick = (event: MouseEvent) => {
      interactionCount.current++;
      const target = event.target as HTMLElement;
      
      trackUserAction('click', target.tagName.toLowerCase(), {
        elementId: target.id,
        elementClass: target.className,
        x: event.clientX,
        y: event.clientY,
      });
    };

    const handleKeydown = (event: KeyboardEvent) => {
      if (event.key === 'Enter' || event.key === ' ') {
        interactionCount.current++;
        trackUserAction('keydown', event.key, {
          code: event.code,
          ctrlKey: event.ctrlKey,
          shiftKey: event.shiftKey,
          altKey: event.altKey,
        });
      }
    };

    document.addEventListener('click', handleClick);
    document.addEventListener('keydown', handleKeydown);

    return () => {
      document.removeEventListener('click', handleClick);
      document.removeEventListener('keydown', handleKeydown);
    };
  }, [trackUserInteractions]);

  // 返回监控方法
  return {
    // 事件跟踪
    trackEvent: useCallback((eventName: string, properties?: Record<string, any>) => {
      trackEvent(eventName, properties);
    }, []),

    // 错误跟踪
    trackError: useCallback((error: Error, context?: Record<string, any>) => {
      trackError(error, context);
    }, []),

    // 用户操作跟踪
    trackUserAction: useCallback((action: string, target?: string, context?: Record<string, any>) => {
      interactionCount.current++;
      trackUserAction(action, target, context);
    }, []),

    // 性能跟踪
    trackPerformance: useCallback((metric: string, value: number, unit?: string, context?: Record<string, any>) => {
      trackPerformance(metric, value, unit, context);
    }, []),

    // 业务事件跟踪
    trackBusiness: useCallback((event: string, context?: Record<string, any>) => {
      trackBusiness(event, context);
    }, []),

    // 计时器
    startTimer: useCallback((name: string) => {
      const startTime = Date.now();
      return {
        end: (context?: Record<string, any>) => {
          const duration = Date.now() - startTime;
          trackPerformance(name, duration, 'ms', context);
          return duration;
        },
      };
    }, []),

    // 异步操作跟踪
    trackAsync: useCallback(async <T>(
      name: string,
      asyncFn: () => Promise<T>,
      context?: Record<string, any>
    ): Promise<T> => {
      const startTime = Date.now();
      
      try {
        trackEvent(`${name}_start`, context);
        const result = await asyncFn();
        const duration = Date.now() - startTime;
        
        trackEvent(`${name}_success`, { ...context, duration });
        trackPerformance(name, duration, 'ms', { ...context, success: true });
        
        return result;
      } catch (error) {
        const duration = Date.now() - startTime;
        
        trackEvent(`${name}_error`, { ...context, duration, error: error.message });
        trackPerformance(name, duration, 'ms', { ...context, success: false });
        trackError(error as Error, { ...context, operation: name });
        
        throw error;
      }
    }, []),

    // 手动上报
    report: useCallback(async () => {
      await monitoring.reportAll();
    }, []),

    // 获取统计信息
    getStats: useCallback(() => {
      return monitoring.getStats();
    }, []),
  };
};

// 页面级监控Hook
export const usePageMonitoring = (pageName: string, options: UseMonitoringOptions = {}) => {
  const monitoring = useMonitoring(options);

  useEffect(() => {
    // 跟踪页面访问
    monitoring.trackEvent('page_view', {
      pageName,
      url: window.location.href,
      title: document.title,
    });

    logger.info(`Page view: ${pageName}`, {
      url: window.location.href,
      title: document.title,
    });
  }, [pageName, monitoring]);

  return monitoring;
};

// 组件级监控Hook
export const useComponentMonitoring = (componentName: string) => {
  const monitoring = useMonitoring({ trackPageViews: false });
  const renderCount = useRef<number>(0);
  const mountTime = useRef<number>(Date.now());

  useEffect(() => {
    renderCount.current++;
    
    // 首次渲染
    if (renderCount.current === 1) {
      monitoring.trackEvent('component_mount', {
        componentName,
        mountTime: mountTime.current,
      });

      logger.debug(`Component mounted: ${componentName}`);
    }

    // 组件卸载
    return () => {
      const lifetime = Date.now() - mountTime.current;
      
      monitoring.trackEvent('component_unmount', {
        componentName,
        lifetime,
        renderCount: renderCount.current,
      });

      monitoring.trackPerformance(`${componentName}_lifetime`, lifetime, 'ms', {
        renderCount: renderCount.current,
      });

      logger.debug(`Component unmounted: ${componentName}`, {
        lifetime,
        renderCount: renderCount.current,
      });
    };
  }, [componentName, monitoring]);

  return {
    ...monitoring,
    trackRender: useCallback((props?: Record<string, any>) => {
      monitoring.trackEvent('component_render', {
        componentName,
        renderCount: renderCount.current,
        ...props,
      });
    }, [componentName, monitoring]),
  };
};

// API调用监控Hook
export const useApiMonitoring = () => {
  const monitoring = useMonitoring({ trackPageViews: false });

  return useCallback(async <T>(
    apiCall: () => Promise<T>,
    config: {
      name: string;
      method: string;
      url: string;
      context?: Record<string, any>;
    }
  ): Promise<T> => {
    const requestId = `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    const startTime = Date.now();

    try {
      // 开始跟踪
      monitoring.trackEvent('api_request_start', {
        name: config.name,
        method: config.method,
        url: config.url,
        requestId,
        ...config.context,
      });

      const result = await apiCall();
      const duration = Date.now() - startTime;

      // 成功跟踪
      monitoring.trackEvent('api_request_success', {
        name: config.name,
        method: config.method,
        url: config.url,
        requestId,
        duration,
        ...config.context,
      });

      monitoring.trackPerformance(`api_${config.name}`, duration, 'ms', {
        method: config.method,
        url: config.url,
        success: true,
        ...config.context,
      });

      return result;
    } catch (error) {
      const duration = Date.now() - startTime;

      // 错误跟踪
      monitoring.trackEvent('api_request_error', {
        name: config.name,
        method: config.method,
        url: config.url,
        requestId,
        duration,
        error: error.message,
        ...config.context,
      });

      monitoring.trackPerformance(`api_${config.name}`, duration, 'ms', {
        method: config.method,
        url: config.url,
        success: false,
        ...config.context,
      });

      monitoring.trackError(error as Error, {
        apiName: config.name,
        method: config.method,
        url: config.url,
        requestId,
        ...config.context,
      });

      throw error;
    }
  }, [monitoring]);
};

export default useMonitoring;
/**
 * 监控服务初始化配置
 * 在应用启动时初始化所有监控服务
 */

import { 
  monitoring, 
  enhancedErrorReporting,
  behaviorAnalytics,
  logAnalytics,
  type MonitoringConfig 
} from '@/services/monitoring';

/**
 * 获取监控配置
 */
function getMonitoringConfig(): MonitoringConfig {
  const isDevelopment = import.meta.env.DEV;
  const isProduction = import.meta.env.PROD;

  return {
    enabled: import.meta.env.VITE_ENABLE_MONITORING !== 'false',
    userId: undefined, // 将在用户登录后设置
    
    apiEndpoints: {
      errors: import.meta.env.VITE_ERROR_MONITORING_ENDPOINT || '/api/monitoring/errors',
      performance: import.meta.env.VITE_PERFORMANCE_MONITORING_ENDPOINT || '/api/monitoring/performance',
      analytics: import.meta.env.VITE_ANALYTICS_ENDPOINT || '/api/monitoring/analytics',
      logs: import.meta.env.VITE_LOGS_ENDPOINT || '/api/monitoring/logs',
    },

    errorMonitoring: {
      enabled: import.meta.env.VITE_ERROR_MONITORING_ENABLED !== 'false',
      maxQueueSize: parseInt(import.meta.env.VITE_ERROR_QUEUE_SIZE || '100'),
      flushInterval: parseInt(import.meta.env.VITE_ERROR_FLUSH_INTERVAL || '30000'),
    },

    performanceMonitoring: {
      enabled: import.meta.env.VITE_PERFORMANCE_MONITORING_ENABLED !== 'false',
      maxQueueSize: parseInt(import.meta.env.VITE_PERFORMANCE_QUEUE_SIZE || '50'),
      flushInterval: parseInt(import.meta.env.VITE_PERFORMANCE_FLUSH_INTERVAL || '60000'),
    },

    userAnalytics: {
      enabled: import.meta.env.VITE_USER_ANALYTICS_ENABLED !== 'false',
      maxQueueSize: parseInt(import.meta.env.VITE_ANALYTICS_QUEUE_SIZE || '100'),
      flushInterval: parseInt(import.meta.env.VITE_ANALYTICS_FLUSH_INTERVAL || '30000'),
    },

    logging: {
      level: (import.meta.env.VITE_LOG_LEVEL as any) || (isDevelopment ? 'debug' : 'info'),
      enableConsole: import.meta.env.VITE_CONSOLE_LOGGING !== 'false',
      enableRemote: isProduction && import.meta.env.VITE_REMOTE_LOGGING !== 'false',
      maxQueueSize: parseInt(import.meta.env.VITE_LOG_QUEUE_SIZE || '1000'),
      flushInterval: parseInt(import.meta.env.VITE_LOG_FLUSH_INTERVAL || '30000'),
    },
  };
}

/**
 * 初始化监控服务
 */
export function initMonitoring(): void {
  try {
    const config = getMonitoringConfig();
    
    // 在开发环境中输出配置信息
    if (import.meta.env.DEV) {
      console.log('Monitoring configuration:', config);
    }

    monitoring.init(config);

    // 初始化增强错误报告
    enhancedErrorReporting.setEnabled(config.errorMonitoring?.enabled !== false);

    // 初始化行为分析
    behaviorAnalytics.setEnabled(config.userAnalytics?.enabled !== false);

    // 设置全局错误处理
    setupGlobalErrorHandling();

    // 设置性能监控
    setupPerformanceMonitoring();

    // 设置页面可见性监控
    setupVisibilityMonitoring();

    // 设置增强监控
    setupEnhancedMonitoring();

    console.log('Enhanced monitoring services initialized successfully');
  } catch (error) {
    console.error('Failed to initialize monitoring services:', error);
  }
}

/**
 * 设置全局错误处理
 */
function setupGlobalErrorHandling(): void {
  // React错误边界无法捕获的错误
  window.addEventListener('error', (event) => {
    console.error('Global error caught:', event.error);
    
    // 使用增强错误报告
    enhancedErrorReporting.captureEnhancedError(event.error, {
      type: 'global_error',
      filename: event.filename,
      lineno: event.lineno,
      colno: event.colno,
    });
  });

  window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled promise rejection:', event.reason);
    
    // 使用增强错误报告
    enhancedErrorReporting.captureEnhancedError(
      new Error(`Unhandled Promise Rejection: ${event.reason}`),
      {
        type: 'unhandled_rejection',
        reason: event.reason,
      }
    );
  });
}

/**
 * 设置性能监控
 */
function setupPerformanceMonitoring(): void {
  // 监控长任务
  if ('PerformanceObserver' in window) {
    try {
      const observer = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        entries.forEach((entry) => {
          if (entry.entryType === 'longtask') {
            console.warn('Long task detected:', entry);
          }
        });
      });

      observer.observe({ entryTypes: ['longtask'] });
    } catch (error) {
      console.warn('Failed to setup long task observer:', error);
    }
  }

  // 监控内存使用
  if ('memory' in performance) {
    const checkMemory = () => {
      const memory = (performance as any).memory;
      const usedMB = Math.round(memory.usedJSHeapSize / 1024 / 1024);
      const totalMB = Math.round(memory.totalJSHeapSize / 1024 / 1024);
      const limitMB = Math.round(memory.jsHeapSizeLimit / 1024 / 1024);

      // 如果内存使用超过80%，发出警告
      if (usedMB / limitMB > 0.8) {
        console.warn(`High memory usage: ${usedMB}MB / ${limitMB}MB (${Math.round(usedMB / limitMB * 100)}%)`);
      }
    };

    // 每分钟检查一次内存使用
    setInterval(checkMemory, 60000);
  }
}

/**
 * 设置页面可见性监控
 */
function setupVisibilityMonitoring(): void {
  let hiddenTime: number | null = null;

  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      hiddenTime = Date.now();
    } else if (hiddenTime) {
      const hiddenDuration = Date.now() - hiddenTime;
      console.log(`Page was hidden for ${hiddenDuration}ms`);
      hiddenTime = null;
    }
  });
}

/**
 * 设置用户ID（在用户登录后调用）
 */
export function setMonitoringUserId(userId: string): void {
  try {
    monitoring.setUserId(userId);
    console.log('Monitoring user ID set:', userId);
  } catch (error) {
    console.error('Failed to set monitoring user ID:', error);
  }
}

/**
 * 手动上报所有监控数据
 */
export async function reportMonitoringData(): Promise<void> {
  try {
    await monitoring.reportAll();
    console.log('All monitoring data reported successfully');
  } catch (error) {
    console.error('Failed to report monitoring data:', error);
  }
}

/**
 * 获取监控统计信息
 */
export function getMonitoringStats() {
  try {
    return monitoring.getStats();
  } catch (error) {
    console.error('Failed to get monitoring stats:', error);
    return null;
  }
}

/**
 * 销毁监控服务（在应用卸载时调用）
 */
export function destroyMonitoring(): void {
  try {
    monitoring.destroy();
    console.log('Monitoring services destroyed');
  } catch (error) {
    console.error('Failed to destroy monitoring services:', error);
  }
}

// 在页面卸载时自动上报数据
window.addEventListener('beforeunload', () => {
  // 使用sendBeacon确保数据能够发送
  if (navigator.sendBeacon) {
    reportMonitoringData().catch(console.error);
  }
});

// 在页面隐藏时上报数据
document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    reportMonitoringData().catch(console.error);
  }
});

export default initMonitoring;
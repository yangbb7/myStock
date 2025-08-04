/**
 * 监控服务集成
 * 统一管理所有监控服务的初始化和配置
 */

import { errorMonitoring } from './errorMonitoring';
import { performanceMonitoring } from './performanceMonitoring';
import { userAnalytics } from './userAnalytics';
import { logger } from './logger';

export interface MonitoringConfig {
  enabled: boolean;
  userId?: string;
  apiEndpoints: {
    errors: string;
    performance: string;
    analytics: string;
    logs: string;
  };
  errorMonitoring?: {
    enabled: boolean;
    maxQueueSize?: number;
    flushInterval?: number;
  };
  performanceMonitoring?: {
    enabled: boolean;
    maxQueueSize?: number;
    flushInterval?: number;
  };
  userAnalytics?: {
    enabled: boolean;
    maxQueueSize?: number;
    flushInterval?: number;
  };
  logging?: {
    level: 'debug' | 'info' | 'warn' | 'error';
    enableConsole: boolean;
    enableRemote: boolean;
    maxQueueSize?: number;
    flushInterval?: number;
  };
}

class MonitoringService {
  private isInitialized: boolean = false;
  private config?: MonitoringConfig;

  /**
   * 初始化所有监控服务
   */
  init(config: MonitoringConfig) {
    if (this.isInitialized) {
      console.warn('Monitoring service already initialized');
      return;
    }

    this.config = config;

    if (!config.enabled) {
      console.log('Monitoring disabled');
      return;
    }

    try {
      // 初始化错误监控
      if (config.errorMonitoring?.enabled !== false) {
        errorMonitoring.init({
          enabled: true,
          apiEndpoint: config.apiEndpoints.errors,
          userId: config.userId,
          maxQueueSize: config.errorMonitoring?.maxQueueSize,
          flushInterval: config.errorMonitoring?.flushInterval,
        });
      }

      // 初始化性能监控
      if (config.performanceMonitoring?.enabled !== false) {
        performanceMonitoring.init({
          enabled: true,
          apiEndpoint: config.apiEndpoints.performance,
          userId: config.userId,
          maxQueueSize: config.performanceMonitoring?.maxQueueSize,
          flushInterval: config.performanceMonitoring?.flushInterval,
        });
      }

      // 初始化用户分析
      if (config.userAnalytics?.enabled !== false) {
        userAnalytics.init({
          enabled: true,
          apiEndpoint: config.apiEndpoints.analytics,
          userId: config.userId,
          maxQueueSize: config.userAnalytics?.maxQueueSize,
          flushInterval: config.userAnalytics?.flushInterval,
        });
      }

      // 配置日志服务
      if (config.logging?.enableRemote !== false) {
        logger.setRemoteEnabled(true);
        logger.setRemoteEndpoint(config.apiEndpoints.logs);
        
        if (config.logging?.level) {
          logger.setLevel(config.logging.level);
        }
        
        if (config.logging?.enableConsole !== undefined) {
          logger.setConsoleEnabled(config.logging.enableConsole);
        }
      }

      // 设置用户ID
      if (config.userId) {
        this.setUserId(config.userId);
      }

      this.isInitialized = true;
      logger.info('Monitoring service initialized successfully', {
        errorMonitoring: config.errorMonitoring?.enabled !== false,
        performanceMonitoring: config.performanceMonitoring?.enabled !== false,
        userAnalytics: config.userAnalytics?.enabled !== false,
        logging: config.logging?.enableRemote !== false,
      });

    } catch (error) {
      console.error('Failed to initialize monitoring service:', error);
      logger.error('Monitoring initialization failed', { error: error.message });
    }
  }

  /**
   * 设置用户ID
   */
  setUserId(userId: string) {
    if (!this.isInitialized) {
      console.warn('Monitoring service not initialized');
      return;
    }

    errorMonitoring.setUserId(userId);
    performanceMonitoring.setUserId(userId);
    userAnalytics.setUserId(userId);
    logger.setUserId(userId);

    logger.info('User ID updated', { userId });
  }

  /**
   * 手动上报所有数据
   */
  async reportAll() {
    if (!this.isInitialized) {
      console.warn('Monitoring service not initialized');
      return;
    }

    try {
      await Promise.all([
        errorMonitoring.report(),
        performanceMonitoring.report(),
        userAnalytics.report(),
        logger.report(),
      ]);

      logger.info('All monitoring data reported successfully');
    } catch (error) {
      logger.error('Failed to report monitoring data', { error: error.message });
    }
  }

  /**
   * 获取监控统计信息
   */
  getStats() {
    if (!this.isInitialized) {
      return null;
    }

    return {
      logging: logger.getStats(),
      timestamp: Date.now(),
    };
  }

  /**
   * 销毁所有监控服务
   */
  destroy() {
    if (!this.isInitialized) {
      return;
    }

    try {
      errorMonitoring.destroy();
      performanceMonitoring.destroy();
      userAnalytics.destroy();
      logger.destroy();

      this.isInitialized = false;
      console.log('Monitoring service destroyed');
    } catch (error) {
      console.error('Failed to destroy monitoring service:', error);
    }
  }
}

// 创建全局监控服务实例
export const monitoring = new MonitoringService();

// 导出增强监控服务
export { enhancedErrorReporting } from './enhancedErrorReporting';
export { behaviorAnalytics } from './behaviorAnalytics';
export { logAnalytics } from './logAnalytics';

// 导出所有监控服务
export {
  errorMonitoring,
  performanceMonitoring,
  userAnalytics,
  logger,
};

// 导出类型
export type {
  MonitoringConfig,
  ErrorInfo,
  NetworkError,
} from './errorMonitoring';

export type {
  PerformanceMetrics,
  ResourceTiming,
  ApiTiming,
  UserInteraction,
} from './performanceMonitoring';

export type {
  UserEvent,
  PageView,
  UserSession,
  ConversionFunnel,
} from './userAnalytics';

export type {
  LogLevel,
  LogEntry,
  LoggerConfig,
} from './logger';

// 便捷方法
export const trackEvent = (eventName: string, properties?: Record<string, any>) => {
  userAnalytics.trackEvent(eventName, properties);
};

export const trackError = (error: Error, context?: Record<string, any>) => {
  errorMonitoring.captureError({
    message: error.message,
    stack: error.stack,
    level: 'error',
    category: 'javascript',
    extra: context,
  });
  logger.exception(error, context);
};

export const trackApiRequest = (config: {
  url: string;
  method: string;
  requestId: string;
}) => {
  performanceMonitoring.trackApiRequest(config);
  logger.apiRequest({
    method: config.method,
    url: config.url,
  });
};

export const completeApiRequest = (config: {
  requestId: string;
  status: number;
  duration: number;
  responseSize?: number;
  error?: string;
}) => {
  performanceMonitoring.completeApiRequest(config);
  logger.apiRequest({
    method: 'unknown',
    url: 'unknown',
    status: config.status,
    duration: config.duration,
    responseSize: config.responseSize,
    error: config.error,
  });
};

export const trackUserAction = (action: string, target?: string, context?: Record<string, any>) => {
  userAnalytics.trackEvent(action, { target, ...context }, 'interaction');
  logger.userAction(action, target, context);
};

export const trackPerformance = (metric: string, value: number, unit?: string, context?: Record<string, any>) => {
  logger.performance(metric, value, unit, context);
};

export const trackBusiness = (event: string, context?: Record<string, any>) => {
  userAnalytics.trackEvent(event, context, 'business');
  logger.business(event, context);
};

export default MonitoringService;
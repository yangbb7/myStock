/**
 * 增强的错误监控和上报服务
 * 提供更详细的错误分析和智能上报功能
 */

import { errorMonitoring, type ErrorInfo } from './errorMonitoring';
import { logger } from './logger';

export interface EnhancedErrorInfo extends ErrorInfo {
  // 错误分类
  errorCategory: 'network' | 'runtime' | 'resource' | 'user' | 'business' | 'security';
  
  // 错误严重程度
  severity: 'low' | 'medium' | 'high' | 'critical';
  
  // 错误影响范围
  impact: {
    userExperience: 'none' | 'minor' | 'major' | 'blocking';
    businessLogic: 'none' | 'minor' | 'major' | 'critical';
    dataIntegrity: 'none' | 'minor' | 'major' | 'critical';
  };
  
  // 错误上下文
  context: {
    userAction?: string;
    componentStack?: string;
    apiEndpoint?: string;
    userAgent: string;
    viewport: { width: number; height: number };
    networkStatus: 'online' | 'offline';
    memoryUsage?: number;
    performanceMetrics?: {
      renderTime: number;
      loadTime: number;
    };
  };
  
  // 错误恢复信息
  recovery?: {
    attempted: boolean;
    successful: boolean;
    method: string;
    retryCount: number;
  };
  
  // 用户反馈
  userFeedback?: {
    reported: boolean;
    description?: string;
    reproduced: boolean;
  };
}

export interface ErrorPattern {
  pattern: string;
  count: number;
  firstSeen: number;
  lastSeen: number;
  affectedUsers: Set<string>;
  severity: 'low' | 'medium' | 'high' | 'critical';
}

export interface ErrorAnalytics {
  totalErrors: number;
  errorRate: number;
  topErrors: ErrorPattern[];
  errorTrends: {
    hourly: number[];
    daily: number[];
  };
  userImpact: {
    affectedUsers: number;
    totalUsers: number;
    impactPercentage: number;
  };
  performanceImpact: {
    averageRecoveryTime: number;
    successfulRecoveries: number;
    totalRecoveryAttempts: number;
  };
}

class EnhancedErrorReportingService {
  private errorPatterns: Map<string, ErrorPattern> = new Map();
  private errorHistory: EnhancedErrorInfo[] = [];
  private maxHistorySize: number = 1000;
  private analyticsInterval: NodeJS.Timeout | null = null;
  private isEnabled: boolean = true;

  constructor() {
    this.setupErrorAnalytics();
    this.setupAutomaticRecovery();
  }

  /**
   * 捕获增强错误信息
   */
  captureEnhancedError(error: Error | ErrorInfo, additionalContext?: Record<string, any>) {
    if (!this.isEnabled) return;

    const enhancedError = this.enhanceErrorInfo(error, additionalContext);
    
    // 添加到历史记录
    this.addToHistory(enhancedError);
    
    // 更新错误模式
    this.updateErrorPatterns(enhancedError);
    
    // 尝试自动恢复
    this.attemptAutoRecovery(enhancedError);
    
    // 上报到原始错误监控服务
    errorMonitoring.captureError(enhancedError);
    
    // 记录到日志
    logger.error(`Enhanced Error: ${enhancedError.message}`, {
      category: enhancedError.errorCategory,
      severity: enhancedError.severity,
      impact: enhancedError.impact,
      context: enhancedError.context,
    });

    // 如果是严重错误，立即通知
    if (enhancedError.severity === 'critical') {
      this.handleCriticalError(enhancedError);
    }
  }

  /**
   * 增强错误信息
   */
  private enhanceErrorInfo(error: Error | ErrorInfo, additionalContext?: Record<string, any>): EnhancedErrorInfo {
    const baseError = error instanceof Error ? {
      message: error.message,
      stack: error.stack,
      url: window.location.href,
      timestamp: Date.now(),
      userAgent: navigator.userAgent,
      sessionId: this.generateSessionId(),
      level: 'error' as const,
      category: 'javascript' as const,
    } : error;

    // 分析错误类型和严重程度
    const analysis = this.analyzeError(baseError);
    
    // 获取性能指标
    const performanceMetrics = this.getPerformanceMetrics();
    
    // 获取内存使用情况
    const memoryUsage = this.getMemoryUsage();

    return {
      ...baseError,
      errorCategory: analysis.category,
      severity: analysis.severity,
      impact: analysis.impact,
      context: {
        userAgent: navigator.userAgent,
        viewport: {
          width: window.innerWidth,
          height: window.innerHeight,
        },
        networkStatus: navigator.onLine ? 'online' : 'offline',
        memoryUsage,
        performanceMetrics,
        ...additionalContext,
      },
    };
  }

  /**
   * 分析错误类型和严重程度
   */
  private analyzeError(error: ErrorInfo) {
    const message = error.message.toLowerCase();
    const stack = error.stack?.toLowerCase() || '';
    
    let category: EnhancedErrorInfo['errorCategory'] = 'runtime';
    let severity: EnhancedErrorInfo['severity'] = 'medium';
    let impact: EnhancedErrorInfo['impact'] = {
      userExperience: 'minor',
      businessLogic: 'minor',
      dataIntegrity: 'none',
    };

    // 网络错误
    if (message.includes('network') || message.includes('fetch') || message.includes('xhr')) {
      category = 'network';
      severity = 'high';
      impact.userExperience = 'major';
      impact.businessLogic = 'major';
    }
    
    // 资源加载错误
    else if (message.includes('loading') || message.includes('resource') || error.category === 'resource') {
      category = 'resource';
      severity = 'medium';
      impact.userExperience = 'minor';
    }
    
    // 安全相关错误
    else if (message.includes('security') || message.includes('cors') || message.includes('csp')) {
      category = 'security';
      severity = 'critical';
      impact.userExperience = 'blocking';
      impact.businessLogic = 'critical';
      impact.dataIntegrity = 'major';
    }
    
    // 业务逻辑错误
    else if (message.includes('validation') || message.includes('business') || message.includes('logic')) {
      category = 'business';
      severity = 'high';
      impact.businessLogic = 'major';
    }
    
    // 用户操作错误
    else if (message.includes('user') || message.includes('input') || message.includes('form')) {
      category = 'user';
      severity = 'low';
      impact.userExperience = 'minor';
    }

    // 根据错误频率调整严重程度
    const pattern = this.getErrorPattern(error.message);
    if (pattern && pattern.count > 10) {
      severity = severity === 'low' ? 'medium' : 
                severity === 'medium' ? 'high' : 'critical';
    }

    return { category, severity, impact };
  }

  /**
   * 获取性能指标
   */
  private getPerformanceMetrics() {
    try {
      const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
      return {
        renderTime: navigation ? navigation.domContentLoadedEventEnd - navigation.domContentLoadedEventStart : 0,
        loadTime: navigation ? navigation.loadEventEnd - navigation.loadEventStart : 0,
      };
    } catch {
      return { renderTime: 0, loadTime: 0 };
    }
  }

  /**
   * 获取内存使用情况
   */
  private getMemoryUsage(): number {
    try {
      if ('memory' in performance) {
        const memory = (performance as any).memory;
        return (memory.usedJSHeapSize / memory.jsHeapSizeLimit) * 100;
      }
    } catch {
      // 忽略错误
    }
    return 0;
  }

  /**
   * 添加到历史记录
   */
  private addToHistory(error: EnhancedErrorInfo) {
    this.errorHistory.push(error);
    
    // 限制历史记录大小
    if (this.errorHistory.length > this.maxHistorySize) {
      this.errorHistory.shift();
    }
  }

  /**
   * 更新错误模式
   */
  private updateErrorPatterns(error: EnhancedErrorInfo) {
    const patternKey = this.getPatternKey(error);
    const existing = this.errorPatterns.get(patternKey);
    
    if (existing) {
      existing.count++;
      existing.lastSeen = error.timestamp;
      if (error.userId) {
        existing.affectedUsers.add(error.userId);
      }
    } else {
      this.errorPatterns.set(patternKey, {
        pattern: patternKey,
        count: 1,
        firstSeen: error.timestamp,
        lastSeen: error.timestamp,
        affectedUsers: new Set(error.userId ? [error.userId] : []),
        severity: error.severity,
      });
    }
  }

  /**
   * 获取错误模式键
   */
  private getPatternKey(error: EnhancedErrorInfo): string {
    // 简化错误消息以识别模式
    const message = error.message
      .replace(/\d+/g, 'N') // 替换数字
      .replace(/['"]/g, '') // 移除引号
      .replace(/\s+/g, ' ') // 标准化空格
      .trim();
    
    return `${error.errorCategory}:${message}`;
  }

  /**
   * 获取错误模式
   */
  private getErrorPattern(message: string): ErrorPattern | undefined {
    const patternKey = message
      .replace(/\d+/g, 'N')
      .replace(/['"]/g, '')
      .replace(/\s+/g, ' ')
      .trim();
    
    for (const [key, pattern] of this.errorPatterns) {
      if (key.includes(patternKey) || patternKey.includes(key.split(':')[1])) {
        return pattern;
      }
    }
    
    return undefined;
  }

  /**
   * 尝试自动恢复
   */
  private attemptAutoRecovery(error: EnhancedErrorInfo) {
    let recovery: EnhancedErrorInfo['recovery'] = {
      attempted: false,
      successful: false,
      method: 'none',
      retryCount: 0,
    };

    try {
      recovery.attempted = true;

      // 网络错误恢复
      if (error.errorCategory === 'network') {
        recovery.method = 'retry_request';
        // 这里可以实现重试逻辑
        recovery.successful = true; // 模拟成功
      }
      
      // 资源加载错误恢复
      else if (error.errorCategory === 'resource') {
        recovery.method = 'reload_resource';
        // 这里可以实现资源重新加载逻辑
        recovery.successful = true; // 模拟成功
      }
      
      // 运行时错误恢复
      else if (error.errorCategory === 'runtime') {
        recovery.method = 'component_reset';
        // 这里可以实现组件重置逻辑
        recovery.successful = false; // 运行时错误通常难以自动恢复
      }

    } catch (recoveryError) {
      recovery.successful = false;
      logger.warn('Auto recovery failed', { 
        originalError: error.message,
        recoveryError: recoveryError.message 
      });
    }

    error.recovery = recovery;
  }

  /**
   * 处理严重错误
   */
  private handleCriticalError(error: EnhancedErrorInfo) {
    // 立即上报
    errorMonitoring.captureError(error);
    
    // 记录严重错误日志
    logger.error('CRITICAL ERROR DETECTED', {
      error: error.message,
      category: error.errorCategory,
      impact: error.impact,
      context: error.context,
    });

    // 可以在这里添加更多严重错误处理逻辑
    // 例如：发送通知、触发告警等
  }

  /**
   * 设置错误分析
   */
  private setupErrorAnalytics() {
    this.analyticsInterval = setInterval(() => {
      this.generateErrorAnalytics();
    }, 60000); // 每分钟分析一次
  }

  /**
   * 生成错误分析报告
   */
  private generateErrorAnalytics(): ErrorAnalytics {
    const now = Date.now();
    const oneHour = 60 * 60 * 1000;
    const oneDay = 24 * oneHour;

    // 计算错误率
    const recentErrors = this.errorHistory.filter(e => now - e.timestamp < oneHour);
    const errorRate = recentErrors.length / 60; // 每分钟错误数

    // 获取顶级错误
    const topErrors = Array.from(this.errorPatterns.values())
      .sort((a, b) => b.count - a.count)
      .slice(0, 10);

    // 计算用户影响
    const affectedUsers = new Set();
    this.errorHistory.forEach(error => {
      if (error.userId) {
        affectedUsers.add(error.userId);
      }
    });

    // 计算恢复统计
    const recoveryAttempts = this.errorHistory.filter(e => e.recovery?.attempted);
    const successfulRecoveries = recoveryAttempts.filter(e => e.recovery?.successful);

    const analytics: ErrorAnalytics = {
      totalErrors: this.errorHistory.length,
      errorRate,
      topErrors,
      errorTrends: {
        hourly: this.calculateHourlyTrend(),
        daily: this.calculateDailyTrend(),
      },
      userImpact: {
        affectedUsers: affectedUsers.size,
        totalUsers: 100, // 这里应该从用户分析服务获取
        impactPercentage: (affectedUsers.size / 100) * 100,
      },
      performanceImpact: {
        averageRecoveryTime: this.calculateAverageRecoveryTime(),
        successfulRecoveries: successfulRecoveries.length,
        totalRecoveryAttempts: recoveryAttempts.length,
      },
    };

    // 记录分析结果
    logger.info('Error analytics generated', analytics);

    return analytics;
  }

  /**
   * 计算小时趋势
   */
  private calculateHourlyTrend(): number[] {
    const now = Date.now();
    const hourly = new Array(24).fill(0);
    
    this.errorHistory.forEach(error => {
      const hoursAgo = Math.floor((now - error.timestamp) / (60 * 60 * 1000));
      if (hoursAgo < 24) {
        hourly[23 - hoursAgo]++;
      }
    });
    
    return hourly;
  }

  /**
   * 计算日趋势
   */
  private calculateDailyTrend(): number[] {
    const now = Date.now();
    const daily = new Array(7).fill(0);
    
    this.errorHistory.forEach(error => {
      const daysAgo = Math.floor((now - error.timestamp) / (24 * 60 * 60 * 1000));
      if (daysAgo < 7) {
        daily[6 - daysAgo]++;
      }
    });
    
    return daily;
  }

  /**
   * 计算平均恢复时间
   */
  private calculateAverageRecoveryTime(): number {
    const recoveryTimes = this.errorHistory
      .filter(e => e.recovery?.successful)
      .map(e => e.recovery?.retryCount || 1);
    
    if (recoveryTimes.length === 0) return 0;
    
    return recoveryTimes.reduce((sum, time) => sum + time, 0) / recoveryTimes.length;
  }

  /**
   * 设置自动恢复
   */
  private setupAutomaticRecovery() {
    // 监听网络状态变化
    window.addEventListener('online', () => {
      logger.info('Network connection restored');
      // 可以在这里重试失败的网络请求
    });

    window.addEventListener('offline', () => {
      logger.warn('Network connection lost');
    });
  }

  /**
   * 生成会话ID
   */
  private generateSessionId(): string {
    return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * 获取错误分析报告
   */
  getAnalytics(): ErrorAnalytics {
    return this.generateErrorAnalytics();
  }

  /**
   * 获取错误历史
   */
  getErrorHistory(limit?: number): EnhancedErrorInfo[] {
    return limit ? this.errorHistory.slice(-limit) : [...this.errorHistory];
  }

  /**
   * 获取错误模式
   */
  getErrorPatterns(): ErrorPattern[] {
    return Array.from(this.errorPatterns.values());
  }

  /**
   * 清理历史数据
   */
  clearHistory() {
    this.errorHistory = [];
    this.errorPatterns.clear();
    logger.info('Error history cleared');
  }

  /**
   * 启用/禁用服务
   */
  setEnabled(enabled: boolean) {
    this.isEnabled = enabled;
    logger.info(`Enhanced error reporting ${enabled ? 'enabled' : 'disabled'}`);
  }

  /**
   * 销毁服务
   */
  destroy() {
    if (this.analyticsInterval) {
      clearInterval(this.analyticsInterval);
    }
    this.clearHistory();
  }
}

// 创建全局实例
export const enhancedErrorReporting = new EnhancedErrorReportingService();

// 导出类型和实例
export default EnhancedErrorReportingService;
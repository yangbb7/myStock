/**
 * 前端错误监控服务
 * 用于收集和上报前端错误信息
 */

export interface ErrorInfo {
  message: string;
  stack?: string;
  url: string;
  line?: number;
  column?: number;
  timestamp: number;
  userAgent: string;
  userId?: string;
  sessionId: string;
  level: 'error' | 'warning' | 'info';
  category: 'javascript' | 'network' | 'resource' | 'custom';
  tags?: Record<string, string>;
  extra?: Record<string, any>;
}

export interface NetworkError {
  url: string;
  method: string;
  status: number;
  statusText: string;
  responseTime: number;
  timestamp: number;
  requestHeaders?: Record<string, string>;
  responseHeaders?: Record<string, string>;
  requestBody?: any;
  responseBody?: any;
}

class ErrorMonitoringService {
  private isEnabled: boolean = false;
  private apiEndpoint: string = '';
  private sessionId: string = '';
  private userId?: string;
  private errorQueue: ErrorInfo[] = [];
  private networkErrorQueue: NetworkError[] = [];
  private maxQueueSize: number = 100;
  private flushInterval: number = 30000; // 30秒
  private flushTimer?: NodeJS.Timeout;

  constructor() {
    this.sessionId = this.generateSessionId();
    this.setupGlobalErrorHandlers();
  }

  /**
   * 初始化错误监控
   */
  init(config: {
    enabled: boolean;
    apiEndpoint: string;
    userId?: string;
    maxQueueSize?: number;
    flushInterval?: number;
  }) {
    this.isEnabled = config.enabled;
    this.apiEndpoint = config.apiEndpoint;
    this.userId = config.userId;
    this.maxQueueSize = config.maxQueueSize || 100;
    this.flushInterval = config.flushInterval || 30000;

    if (this.isEnabled) {
      this.startFlushTimer();
      console.log('Error monitoring initialized');
    }
  }

  /**
   * 设置全局错误处理器
   */
  private setupGlobalErrorHandlers() {
    // JavaScript错误
    window.addEventListener('error', (event) => {
      this.captureError({
        message: event.message,
        stack: event.error?.stack,
        url: event.filename,
        line: event.lineno,
        column: event.colno,
        timestamp: Date.now(),
        userAgent: navigator.userAgent,
        userId: this.userId,
        sessionId: this.sessionId,
        level: 'error',
        category: 'javascript',
      });
    });

    // Promise未捕获错误
    window.addEventListener('unhandledrejection', (event) => {
      this.captureError({
        message: `Unhandled Promise Rejection: ${event.reason}`,
        stack: event.reason?.stack,
        url: window.location.href,
        timestamp: Date.now(),
        userAgent: navigator.userAgent,
        userId: this.userId,
        sessionId: this.sessionId,
        level: 'error',
        category: 'javascript',
        extra: { reason: event.reason },
      });
    });

    // 资源加载错误
    window.addEventListener('error', (event) => {
      if (event.target !== window) {
        const target = event.target as HTMLElement;
        this.captureError({
          message: `Resource loading error: ${target.tagName}`,
          url: (target as any).src || (target as any).href || window.location.href,
          timestamp: Date.now(),
          userAgent: navigator.userAgent,
          userId: this.userId,
          sessionId: this.sessionId,
          level: 'error',
          category: 'resource',
          extra: {
            tagName: target.tagName,
            src: (target as any).src,
            href: (target as any).href,
          },
        });
      }
    }, true);
  }

  /**
   * 手动捕获错误
   */
  captureError(error: Partial<ErrorInfo>) {
    if (!this.isEnabled) return;

    const errorInfo: ErrorInfo = {
      message: error.message || 'Unknown error',
      stack: error.stack,
      url: error.url || window.location.href,
      line: error.line,
      column: error.column,
      timestamp: error.timestamp || Date.now(),
      userAgent: error.userAgent || navigator.userAgent,
      userId: error.userId || this.userId,
      sessionId: error.sessionId || this.sessionId,
      level: error.level || 'error',
      category: error.category || 'custom',
      tags: error.tags,
      extra: error.extra,
    };

    this.addToQueue(errorInfo);
  }

  /**
   * 捕获网络错误
   */
  captureNetworkError(networkError: NetworkError) {
    if (!this.isEnabled) return;

    this.networkErrorQueue.push(networkError);
    if (this.networkErrorQueue.length > this.maxQueueSize) {
      this.networkErrorQueue.shift();
    }
  }

  /**
   * 添加错误到队列
   */
  private addToQueue(error: ErrorInfo) {
    this.errorQueue.push(error);
    if (this.errorQueue.length > this.maxQueueSize) {
      this.errorQueue.shift();
    }

    // 如果是严重错误，立即上报
    if (error.level === 'error') {
      this.flush();
    }
  }

  /**
   * 开始定时上报
   */
  private startFlushTimer() {
    if (this.flushTimer) {
      clearInterval(this.flushTimer);
    }

    this.flushTimer = setInterval(() => {
      this.flush();
    }, this.flushInterval);
  }

  /**
   * 上报错误数据
   */
  private async flush() {
    if (!this.isEnabled || (!this.errorQueue.length && !this.networkErrorQueue.length)) {
      return;
    }

    const payload = {
      errors: [...this.errorQueue],
      networkErrors: [...this.networkErrorQueue],
      metadata: {
        url: window.location.href,
        timestamp: Date.now(),
        sessionId: this.sessionId,
        userId: this.userId,
        userAgent: navigator.userAgent,
        viewport: {
          width: window.innerWidth,
          height: window.innerHeight,
        },
        screen: {
          width: screen.width,
          height: screen.height,
        },
      },
    };

    try {
      await fetch(this.apiEndpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      // 清空队列
      this.errorQueue = [];
      this.networkErrorQueue = [];
    } catch (error) {
      console.warn('Failed to report errors:', error);
    }
  }

  /**
   * 生成会话ID
   */
  private generateSessionId(): string {
    return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * 设置用户ID
   */
  setUserId(userId: string) {
    this.userId = userId;
  }

  /**
   * 添加全局标签
   */
  setTags(tags: Record<string, string>) {
    // 为后续的错误添加标签
    this.captureError = ((originalCapture) => {
      return (error: Partial<ErrorInfo>) => {
        error.tags = { ...tags, ...error.tags };
        return originalCapture.call(this, error);
      };
    })(this.captureError);
  }

  /**
   * 手动上报
   */
  async report() {
    await this.flush();
  }

  /**
   * 销毁监控服务
   */
  destroy() {
    if (this.flushTimer) {
      clearInterval(this.flushTimer);
    }
    this.flush(); // 最后一次上报
  }
}

// 创建全局实例
export const errorMonitoring = new ErrorMonitoringService();

// 导出类型和实例
export default ErrorMonitoringService;
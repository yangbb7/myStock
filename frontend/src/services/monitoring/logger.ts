/**
 * 前端日志服务
 * 提供结构化日志记录和远程日志收集功能
 */

export type LogLevel = 'debug' | 'info' | 'warn' | 'error';

export interface LogEntry {
  level: LogLevel;
  message: string;
  timestamp: number;
  sessionId: string;
  userId?: string;
  context?: Record<string, any>;
  stack?: string;
  tags?: string[];
  source: {
    file?: string;
    function?: string;
    line?: number;
    column?: number;
  };
  metadata: {
    url: string;
    userAgent: string;
    viewport: { width: number; height: number };
  };
}

export interface LoggerConfig {
  level: LogLevel;
  enableConsole: boolean;
  enableRemote: boolean;
  remoteEndpoint?: string;
  maxQueueSize: number;
  flushInterval: number;
  enableStackTrace: boolean;
  enableSourceMap: boolean;
}

class LoggerService {
  private config: LoggerConfig;
  private sessionId: string;
  private userId?: string;
  private logQueue: LogEntry[] = [];
  private flushTimer?: NodeJS.Timeout;
  private readonly levelPriority: Record<LogLevel, number> = {
    debug: 0,
    info: 1,
    warn: 2,
    error: 3,
  };

  constructor(config: Partial<LoggerConfig> = {}) {
    this.config = {
      level: 'info',
      enableConsole: true,
      enableRemote: false,
      maxQueueSize: 1000,
      flushInterval: 30000, // 30秒
      enableStackTrace: true,
      enableSourceMap: false,
      ...config,
    };

    this.sessionId = this.generateSessionId();
    this.setupRemoteLogging();
  }

  /**
   * 设置远程日志
   */
  private setupRemoteLogging() {
    if (this.config.enableRemote && this.config.remoteEndpoint) {
      this.startFlushTimer();
    }
  }

  /**
   * 记录调试日志
   */
  debug(message: string, context?: Record<string, any>, tags?: string[]) {
    this.log('debug', message, context, tags);
  }

  /**
   * 记录信息日志
   */
  info(message: string, context?: Record<string, any>, tags?: string[]) {
    this.log('info', message, context, tags);
  }

  /**
   * 记录警告日志
   */
  warn(message: string, context?: Record<string, any>, tags?: string[]) {
    this.log('warn', message, context, tags);
  }

  /**
   * 记录错误日志
   */
  error(message: string, context?: Record<string, any>, tags?: string[]) {
    this.log('error', message, context, tags);
  }

  /**
   * 记录异常
   */
  exception(error: Error, context?: Record<string, any>, tags?: string[]) {
    const errorContext = {
      name: error.name,
      message: error.message,
      stack: error.stack,
      ...context,
    };

    this.log('error', `Exception: ${error.message}`, errorContext, tags);
  }

  /**
   * 记录API请求
   */
  apiRequest(config: {
    method: string;
    url: string;
    status?: number;
    duration?: number;
    requestSize?: number;
    responseSize?: number;
    error?: string;
  }) {
    const level: LogLevel = config.error ? 'error' : config.status && config.status >= 400 ? 'warn' : 'info';
    const message = `API ${config.method} ${config.url} - ${config.status || 'pending'}`;
    
    this.log(level, message, {
      type: 'api_request',
      method: config.method,
      url: config.url,
      status: config.status,
      duration: config.duration,
      requestSize: config.requestSize,
      responseSize: config.responseSize,
      error: config.error,
    }, ['api']);
  }

  /**
   * 记录用户操作
   */
  userAction(action: string, target?: string, context?: Record<string, any>) {
    this.log('info', `User action: ${action}`, {
      type: 'user_action',
      action,
      target,
      ...context,
    }, ['user', 'interaction']);
  }

  /**
   * 记录性能指标
   */
  performance(metric: string, value: number, unit: string = 'ms', context?: Record<string, any>) {
    this.log('info', `Performance: ${metric} = ${value}${unit}`, {
      type: 'performance',
      metric,
      value,
      unit,
      ...context,
    }, ['performance']);
  }

  /**
   * 记录业务事件
   */
  business(event: string, context?: Record<string, any>) {
    this.log('info', `Business event: ${event}`, {
      type: 'business',
      event,
      ...context,
    }, ['business']);
  }

  /**
   * 核心日志记录方法
   */
  private log(level: LogLevel, message: string, context?: Record<string, any>, tags?: string[]) {
    // 检查日志级别
    if (this.levelPriority[level] < this.levelPriority[this.config.level]) {
      return;
    }

    const logEntry: LogEntry = {
      level,
      message,
      timestamp: Date.now(),
      sessionId: this.sessionId,
      userId: this.userId,
      context,
      tags,
      source: this.getSourceInfo(),
      metadata: this.getMetadata(),
    };

    // 添加堆栈跟踪
    if (this.config.enableStackTrace && (level === 'error' || level === 'warn')) {
      logEntry.stack = this.getStackTrace();
    }

    // 控制台输出
    if (this.config.enableConsole) {
      this.logToConsole(logEntry);
    }

    // 远程日志
    if (this.config.enableRemote) {
      this.addToQueue(logEntry);
    }
  }

  /**
   * 控制台输出
   */
  private logToConsole(entry: LogEntry) {
    const timestamp = new Date(entry.timestamp).toISOString();
    const prefix = `[${timestamp}] [${entry.level.toUpperCase()}]`;
    const message = `${prefix} ${entry.message}`;

    switch (entry.level) {
      case 'debug':
        console.debug(message, entry.context);
        break;
      case 'info':
        console.info(message, entry.context);
        break;
      case 'warn':
        console.warn(message, entry.context);
        break;
      case 'error':
        console.error(message, entry.context, entry.stack);
        break;
    }
  }

  /**
   * 添加到队列
   */
  private addToQueue(entry: LogEntry) {
    this.logQueue.push(entry);
    
    // 限制队列大小
    if (this.logQueue.length > this.config.maxQueueSize) {
      this.logQueue.shift();
    }

    // 错误日志立即上报
    if (entry.level === 'error') {
      this.flush();
    }
  }

  /**
   * 获取源码信息
   */
  private getSourceInfo() {
    if (!this.config.enableStackTrace) {
      return {};
    }

    try {
      const stack = new Error().stack;
      if (!stack) return {};

      const lines = stack.split('\n');
      // 跳过当前函数和log函数的调用栈
      const callerLine = lines[4] || lines[3] || lines[2];
      
      if (callerLine) {
        const match = callerLine.match(/at\s+(.+?)\s+\((.+?):(\d+):(\d+)\)/);
        if (match) {
          return {
            function: match[1],
            file: match[2],
            line: parseInt(match[3]),
            column: parseInt(match[4]),
          };
        }
      }
    } catch (error) {
      // 忽略错误
    }

    return {};
  }

  /**
   * 获取堆栈跟踪
   */
  private getStackTrace(): string {
    try {
      const stack = new Error().stack;
      if (!stack) return '';

      const lines = stack.split('\n');
      // 移除当前函数的调用栈
      return lines.slice(3).join('\n');
    } catch (error) {
      return '';
    }
  }

  /**
   * 获取元数据
   */
  private getMetadata() {
    return {
      url: window.location.href,
      userAgent: navigator.userAgent,
      viewport: {
        width: window.innerWidth,
        height: window.innerHeight,
      },
    };
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
    }, this.config.flushInterval);
  }

  /**
   * 上报日志
   */
  private async flush() {
    if (!this.config.enableRemote || !this.config.remoteEndpoint || !this.logQueue.length) {
      return;
    }

    const logs = [...this.logQueue];
    this.logQueue = [];

    try {
      await fetch(this.config.remoteEndpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          logs,
          metadata: {
            timestamp: Date.now(),
            sessionId: this.sessionId,
            userId: this.userId,
          },
        }),
      });
    } catch (error) {
      // 上报失败，重新加入队列
      this.logQueue.unshift(...logs);
      console.warn('Failed to send logs to remote endpoint:', error);
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
   * 设置日志级别
   */
  setLevel(level: LogLevel) {
    this.config.level = level;
  }

  /**
   * 启用/禁用控制台输出
   */
  setConsoleEnabled(enabled: boolean) {
    this.config.enableConsole = enabled;
  }

  /**
   * 启用/禁用远程日志
   */
  setRemoteEnabled(enabled: boolean) {
    this.config.enableRemote = enabled;
    
    if (enabled && this.config.remoteEndpoint) {
      this.startFlushTimer();
    } else if (this.flushTimer) {
      clearInterval(this.flushTimer);
    }
  }

  /**
   * 设置远程端点
   */
  setRemoteEndpoint(endpoint: string) {
    this.config.remoteEndpoint = endpoint;
    
    if (this.config.enableRemote) {
      this.startFlushTimer();
    }
  }

  /**
   * 获取日志统计
   */
  getStats() {
    const stats = {
      debug: 0,
      info: 0,
      warn: 0,
      error: 0,
    };

    this.logQueue.forEach(entry => {
      stats[entry.level]++;
    });

    return {
      queueSize: this.logQueue.length,
      maxQueueSize: this.config.maxQueueSize,
      levelCounts: stats,
      sessionId: this.sessionId,
      userId: this.userId,
    };
  }

  /**
   * 清空日志队列
   */
  clear() {
    this.logQueue = [];
  }

  /**
   * 手动上报
   */
  async report() {
    await this.flush();
  }

  /**
   * 销毁日志服务
   */
  destroy() {
    if (this.flushTimer) {
      clearInterval(this.flushTimer);
    }
    this.flush(); // 最后一次上报
  }
}

// 创建默认日志实例
export const logger = new LoggerService({
  level: import.meta.env.DEV ? 'debug' : 'info',
  enableConsole: true,
  enableRemote: import.meta.env.PROD,
  remoteEndpoint: import.meta.env.VITE_LOG_ENDPOINT,
});

// 创建特定用途的日志器
export const apiLogger = new LoggerService({
  level: 'info',
  enableConsole: import.meta.env.DEV,
  enableRemote: true,
  remoteEndpoint: import.meta.env.VITE_API_LOG_ENDPOINT,
});

export const errorLogger = new LoggerService({
  level: 'error',
  enableConsole: true,
  enableRemote: true,
  remoteEndpoint: import.meta.env.VITE_ERROR_LOG_ENDPOINT,
  enableStackTrace: true,
});

// 导出类型和实例
export default LoggerService;
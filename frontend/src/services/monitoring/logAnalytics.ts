/**
 * 日志分析服务
 * 提供高级日志收集、分析和可视化功能
 */

import { logger, type LogEntry, type LogLevel } from './logger';

export interface LogPattern {
  patternId: string;
  pattern: string;
  frequency: number;
  level: LogLevel;
  firstSeen: number;
  lastSeen: number;
  affectedSessions: Set<string>;
  category: 'error' | 'performance' | 'user_action' | 'api' | 'business' | 'system';
  severity: 'low' | 'medium' | 'high' | 'critical';
}

export interface LogAnalytics {
  // 日志统计
  totalLogs: number;
  logsByLevel: Record<LogLevel, number>;
  logsByCategory: Record<string, number>;
  
  // 时间趋势
  trends: {
    hourly: { timestamp: number; count: number; errors: number }[];
    daily: { timestamp: number; count: number; errors: number }[];
  };
  
  // 错误分析
  errorAnalysis: {
    topErrors: LogPattern[];
    errorRate: number;
    criticalErrors: number;
    errorTrend: 'increasing' | 'decreasing' | 'stable';
  };
  
  // 性能分析
  performanceAnalysis: {
    slowOperations: { operation: string; averageTime: number; count: number }[];
    performanceTrend: 'improving' | 'degrading' | 'stable';
    bottlenecks: string[];
  };
  
  // 用户行为分析
  userBehaviorAnalysis: {
    topUserActions: { action: string; count: number; uniqueUsers: number }[];
    userEngagement: number;
    conversionFunnel: { step: string; count: number; dropOffRate: number }[];
  };
  
  // 系统健康度
  systemHealth: {
    healthScore: number;
    alerts: { level: 'warning' | 'critical'; message: string; count: number }[];
    recommendations: string[];
  };
}

export interface LogAlert {
  alertId: string;
  level: 'info' | 'warning' | 'error' | 'critical';
  title: string;
  message: string;
  timestamp: number;
  count: number;
  pattern?: LogPattern;
  acknowledged: boolean;
  actions: string[];
}

export interface LogQuery {
  level?: LogLevel[];
  category?: string[];
  timeRange?: { start: number; end: number };
  searchText?: string;
  sessionId?: string;
  userId?: string;
  limit?: number;
  offset?: number;
}

class LogAnalyticsService {
  private logHistory: LogEntry[] = [];
  private patterns: Map<string, LogPattern> = new Map();
  private alerts: Map<string, LogAlert> = new Map();
  private maxHistorySize: number = 50000;
  private analysisInterval: NodeJS.Timeout | null = null;
  private alertThresholds = {
    errorRate: 0.05, // 5% error rate threshold
    criticalErrors: 10, // 10 critical errors threshold
    performanceDegradation: 0.2, // 20% performance degradation
  };

  constructor() {
    this.setupLogCollection();
    this.startAnalysis();
  }

  /**
   * 设置日志收集
   */
  private setupLogCollection() {
    // 拦截原始日志服务的输出
    const originalLog = logger.log;
    logger.log = (level: LogLevel, message: string, context?: Record<string, any>, tags?: string[]) => {
      // 调用原始日志方法
      originalLog.call(logger, level, message, context, tags);
      
      // 收集日志用于分析
      this.collectLog({
        level,
        message,
        timestamp: Date.now(),
        sessionId: this.generateSessionId(),
        context,
        tags,
        source: this.getSourceInfo(),
        metadata: this.getMetadata(),
      });
    };
  }

  /**
   * 收集日志
   */
  private collectLog(logEntry: LogEntry) {
    // 添加到历史记录
    this.logHistory.push(logEntry);
    
    // 限制历史记录大小
    if (this.logHistory.length > this.maxHistorySize) {
      this.logHistory.shift();
    }

    // 更新模式
    this.updatePatterns(logEntry);
    
    // 检查告警
    this.checkAlerts(logEntry);
  }

  /**
   * 更新日志模式
   */
  private updatePatterns(logEntry: LogEntry) {
    const patternKey = this.extractPattern(logEntry);
    const existing = this.patterns.get(patternKey);
    
    if (existing) {
      existing.frequency++;
      existing.lastSeen = logEntry.timestamp;
      existing.affectedSessions.add(logEntry.sessionId);
    } else {
      this.patterns.set(patternKey, {
        patternId: patternKey,
        pattern: this.simplifyMessage(logEntry.message),
        frequency: 1,
        level: logEntry.level,
        firstSeen: logEntry.timestamp,
        lastSeen: logEntry.timestamp,
        affectedSessions: new Set([logEntry.sessionId]),
        category: this.categorizeLog(logEntry),
        severity: this.assessSeverity(logEntry),
      });
    }
  }

  /**
   * 提取日志模式
   */
  private extractPattern(logEntry: LogEntry): string {
    const message = this.simplifyMessage(logEntry.message);
    const category = this.categorizeLog(logEntry);
    return `${logEntry.level}:${category}:${message}`;
  }

  /**
   * 简化消息
   */
  private simplifyMessage(message: string): string {
    return message
      .replace(/\d+/g, 'N') // 替换数字
      .replace(/['"]/g, '') // 移除引号
      .replace(/\s+/g, ' ') // 标准化空格
      .replace(/https?:\/\/[^\s]+/g, 'URL') // 替换URL
      .trim()
      .slice(0, 100); // 限制长度
  }

  /**
   * 分类日志
   */
  private categorizeLog(logEntry: LogEntry): LogPattern['category'] {
    const message = logEntry.message.toLowerCase();
    const context = logEntry.context || {};
    
    if (message.includes('error') || message.includes('exception') || logEntry.level === 'error') {
      return 'error';
    }
    if (message.includes('performance') || message.includes('slow') || context.type === 'performance') {
      return 'performance';
    }
    if (message.includes('user') || message.includes('click') || context.type === 'user_action') {
      return 'user_action';
    }
    if (message.includes('api') || message.includes('request') || context.type === 'api_request') {
      return 'api';
    }
    if (message.includes('business') || context.type === 'business') {
      return 'business';
    }
    
    return 'system';
  }

  /**
   * 评估严重程度
   */
  private assessSeverity(logEntry: LogEntry): LogPattern['severity'] {
    if (logEntry.level === 'error') {
      const message = logEntry.message.toLowerCase();
      if (message.includes('critical') || message.includes('fatal') || message.includes('crash')) {
        return 'critical';
      }
      return 'high';
    }
    if (logEntry.level === 'warn') {
      return 'medium';
    }
    return 'low';
  }

  /**
   * 检查告警
   */
  private checkAlerts(logEntry: LogEntry) {
    // 错误率告警
    this.checkErrorRateAlert();
    
    // 关键错误告警
    this.checkCriticalErrorAlert(logEntry);
    
    // 性能告警
    this.checkPerformanceAlert(logEntry);
    
    // 模式告警
    this.checkPatternAlert(logEntry);
  }

  /**
   * 检查错误率告警
   */
  private checkErrorRateAlert() {
    const recentLogs = this.getRecentLogs(300000); // 最近5分钟
    const errorLogs = recentLogs.filter(log => log.level === 'error');
    const errorRate = recentLogs.length > 0 ? errorLogs.length / recentLogs.length : 0;
    
    if (errorRate > this.alertThresholds.errorRate) {
      this.createAlert({
        level: 'warning',
        title: '错误率过高',
        message: `最近5分钟错误率达到 ${(errorRate * 100).toFixed(2)}%，超过阈值 ${(this.alertThresholds.errorRate * 100)}%`,
        count: errorLogs.length,
        actions: ['检查系统状态', '查看错误详情', '联系技术支持'],
      });
    }
  }

  /**
   * 检查关键错误告警
   */
  private checkCriticalErrorAlert(logEntry: LogEntry) {
    if (logEntry.level === 'error') {
      const message = logEntry.message.toLowerCase();
      if (message.includes('critical') || message.includes('fatal')) {
        this.createAlert({
          level: 'critical',
          title: '关键错误',
          message: `检测到关键错误: ${logEntry.message}`,
          count: 1,
          actions: ['立即处理', '通知管理员', '系统诊断'],
        });
      }
    }
  }

  /**
   * 检查性能告警
   */
  private checkPerformanceAlert(logEntry: LogEntry) {
    if (logEntry.context?.type === 'performance') {
      const duration = logEntry.context.duration || logEntry.context.value;
      if (duration && duration > 5000) { // 超过5秒
        this.createAlert({
          level: 'warning',
          title: '性能问题',
          message: `检测到慢操作: ${logEntry.message}，耗时 ${duration}ms`,
          count: 1,
          actions: ['性能分析', '优化建议', '监控趋势'],
        });
      }
    }
  }

  /**
   * 检查模式告警
   */
  private checkPatternAlert(logEntry: LogEntry) {
    const patternKey = this.extractPattern(logEntry);
    const pattern = this.patterns.get(patternKey);
    
    if (pattern && pattern.frequency > 10 && pattern.severity === 'high') {
      this.createAlert({
        level: 'warning',
        title: '重复错误模式',
        message: `错误模式 "${pattern.pattern}" 已出现 ${pattern.frequency} 次`,
        count: pattern.frequency,
        pattern,
        actions: ['分析根因', '修复问题', '监控趋势'],
      });
    }
  }

  /**
   * 创建告警
   */
  private createAlert(alertData: Partial<LogAlert>) {
    const alertId = this.generateAlertId(alertData.title || 'Unknown');
    const existing = this.alerts.get(alertId);
    
    if (existing) {
      existing.count += alertData.count || 1;
      existing.timestamp = Date.now();
    } else {
      this.alerts.set(alertId, {
        alertId,
        level: alertData.level || 'info',
        title: alertData.title || 'Unknown Alert',
        message: alertData.message || '',
        timestamp: Date.now(),
        count: alertData.count || 1,
        pattern: alertData.pattern,
        acknowledged: false,
        actions: alertData.actions || [],
      });
    }
  }

  /**
   * 开始分析
   */
  private startAnalysis() {
    this.analysisInterval = setInterval(() => {
      this.performAnalysis();
    }, 60000); // 每分钟分析一次
  }

  /**
   * 执行分析
   */
  private performAnalysis() {
    const analytics = this.generateAnalytics();
    
    // 记录分析结果
    logger.info('Log analytics completed', {
      totalLogs: analytics.totalLogs,
      errorRate: analytics.errorAnalysis.errorRate,
      healthScore: analytics.systemHealth.healthScore,
    });
    
    // 检查系统健康度
    if (analytics.systemHealth.healthScore < 70) {
      this.createAlert({
        level: 'warning',
        title: '系统健康度下降',
        message: `系统健康度为 ${analytics.systemHealth.healthScore}%，建议检查系统状态`,
        count: 1,
        actions: ['系统诊断', '性能优化', '错误修复'],
      });
    }
  }

  /**
   * 生成分析报告
   */
  generateAnalytics(): LogAnalytics {
    const now = Date.now();
    const oneHour = 60 * 60 * 1000;
    const oneDay = 24 * oneHour;

    // 基础统计
    const totalLogs = this.logHistory.length;
    const logsByLevel = this.calculateLogsByLevel();
    const logsByCategory = this.calculateLogsByCategory();

    // 时间趋势
    const trends = this.calculateTrends();

    // 错误分析
    const errorAnalysis = this.analyzeErrors();

    // 性能分析
    const performanceAnalysis = this.analyzePerformance();

    // 用户行为分析
    const userBehaviorAnalysis = this.analyzeUserBehavior();

    // 系统健康度
    const systemHealth = this.calculateSystemHealth();

    return {
      totalLogs,
      logsByLevel,
      logsByCategory,
      trends,
      errorAnalysis,
      performanceAnalysis,
      userBehaviorAnalysis,
      systemHealth,
    };
  }

  /**
   * 计算各级别日志数量
   */
  private calculateLogsByLevel(): Record<LogLevel, number> {
    const counts: Record<LogLevel, number> = {
      debug: 0,
      info: 0,
      warn: 0,
      error: 0,
    };

    this.logHistory.forEach(log => {
      counts[log.level]++;
    });

    return counts;
  }

  /**
   * 计算各类别日志数量
   */
  private calculateLogsByCategory(): Record<string, number> {
    const counts: Record<string, number> = {};

    this.patterns.forEach(pattern => {
      counts[pattern.category] = (counts[pattern.category] || 0) + pattern.frequency;
    });

    return counts;
  }

  /**
   * 计算时间趋势
   */
  private calculateTrends() {
    const now = Date.now();
    const oneHour = 60 * 60 * 1000;
    const oneDay = 24 * oneHour;

    // 小时趋势
    const hourly = [];
    for (let i = 23; i >= 0; i--) {
      const timestamp = now - (i * oneHour);
      const logs = this.logHistory.filter(log => 
        log.timestamp >= timestamp - oneHour && log.timestamp < timestamp
      );
      const errors = logs.filter(log => log.level === 'error').length;
      
      hourly.push({
        timestamp,
        count: logs.length,
        errors,
      });
    }

    // 日趋势
    const daily = [];
    for (let i = 6; i >= 0; i--) {
      const timestamp = now - (i * oneDay);
      const logs = this.logHistory.filter(log => 
        log.timestamp >= timestamp - oneDay && log.timestamp < timestamp
      );
      const errors = logs.filter(log => log.level === 'error').length;
      
      daily.push({
        timestamp,
        count: logs.length,
        errors,
      });
    }

    return { hourly, daily };
  }

  /**
   * 分析错误
   */
  private analyzeErrors() {
    const errorPatterns = Array.from(this.patterns.values())
      .filter(pattern => pattern.category === 'error')
      .sort((a, b) => b.frequency - a.frequency)
      .slice(0, 10);

    const totalLogs = this.logHistory.length;
    const errorLogs = this.logHistory.filter(log => log.level === 'error');
    const errorRate = totalLogs > 0 ? errorLogs.length / totalLogs : 0;

    const criticalErrors = errorPatterns
      .filter(pattern => pattern.severity === 'critical')
      .reduce((sum, pattern) => sum + pattern.frequency, 0);

    // 计算错误趋势
    const recentErrors = this.getRecentLogs(3600000).filter(log => log.level === 'error').length;
    const previousErrors = this.logHistory
      .filter(log => log.timestamp >= Date.now() - 7200000 && log.timestamp < Date.now() - 3600000)
      .filter(log => log.level === 'error').length;

    let errorTrend: 'increasing' | 'decreasing' | 'stable' = 'stable';
    if (recentErrors > previousErrors * 1.2) {
      errorTrend = 'increasing';
    } else if (recentErrors < previousErrors * 0.8) {
      errorTrend = 'decreasing';
    }

    return {
      topErrors: errorPatterns,
      errorRate,
      criticalErrors,
      errorTrend,
    };
  }

  /**
   * 分析性能
   */
  private analyzePerformance() {
    const performanceLogs = this.logHistory.filter(log => 
      log.context?.type === 'performance' || log.message.toLowerCase().includes('performance')
    );

    const operationTimes = new Map<string, { total: number; count: number }>();
    
    performanceLogs.forEach(log => {
      const operation = log.context?.metric || log.message;
      const duration = log.context?.value || log.context?.duration || 0;
      
      if (operation && duration) {
        const existing = operationTimes.get(operation) || { total: 0, count: 0 };
        existing.total += duration;
        existing.count++;
        operationTimes.set(operation, existing);
      }
    });

    const slowOperations = Array.from(operationTimes.entries())
      .map(([operation, data]) => ({
        operation,
        averageTime: data.total / data.count,
        count: data.count,
      }))
      .filter(op => op.averageTime > 1000) // 超过1秒的操作
      .sort((a, b) => b.averageTime - a.averageTime)
      .slice(0, 10);

    // 性能趋势分析（简化）
    const performanceTrend: 'improving' | 'degrading' | 'stable' = 'stable';

    // 瓶颈识别
    const bottlenecks = slowOperations
      .filter(op => op.averageTime > 3000) // 超过3秒
      .map(op => op.operation);

    return {
      slowOperations,
      performanceTrend,
      bottlenecks,
    };
  }

  /**
   * 分析用户行为
   */
  private analyzeUserBehavior() {
    const userActionLogs = this.logHistory.filter(log => 
      log.context?.type === 'user_action' || log.message.toLowerCase().includes('user')
    );

    const actionCounts = new Map<string, { count: number; users: Set<string> }>();
    
    userActionLogs.forEach(log => {
      const action = log.context?.action || log.message;
      const userId = log.userId || 'anonymous';
      
      if (action) {
        const existing = actionCounts.get(action) || { count: 0, users: new Set() };
        existing.count++;
        existing.users.add(userId);
        actionCounts.set(action, existing);
      }
    });

    const topUserActions = Array.from(actionCounts.entries())
      .map(([action, data]) => ({
        action,
        count: data.count,
        uniqueUsers: data.users.size,
      }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 10);

    // 用户参与度（简化计算）
    const uniqueUsers = new Set(userActionLogs.map(log => log.userId).filter(Boolean)).size;
    const userEngagement = userActionLogs.length > 0 ? uniqueUsers / userActionLogs.length * 100 : 0;

    // 转化漏斗（模拟数据）
    const conversionFunnel = [
      { step: '页面访问', count: 100, dropOffRate: 0 },
      { step: '功能使用', count: 80, dropOffRate: 20 },
      { step: '操作完成', count: 60, dropOffRate: 25 },
      { step: '结果确认', count: 50, dropOffRate: 16.7 },
    ];

    return {
      topUserActions,
      userEngagement,
      conversionFunnel,
    };
  }

  /**
   * 计算系统健康度
   */
  private calculateSystemHealth() {
    let healthScore = 100;
    const alerts: LogAlert[] = [];
    const recommendations: string[] = [];

    // 错误率影响
    const errorRate = this.calculateLogsByLevel().error / this.logHistory.length;
    if (errorRate > 0.1) {
      healthScore -= 30;
      alerts.push({
        alertId: 'high-error-rate',
        level: 'critical',
        title: '高错误率',
        message: `错误率达到 ${(errorRate * 100).toFixed(2)}%`,
        timestamp: Date.now(),
        count: 1,
        acknowledged: false,
        actions: [],
      });
      recommendations.push('降低系统错误率');
    } else if (errorRate > 0.05) {
      healthScore -= 15;
      alerts.push({
        alertId: 'moderate-error-rate',
        level: 'warning',
        title: '中等错误率',
        message: `错误率为 ${(errorRate * 100).toFixed(2)}%`,
        timestamp: Date.now(),
        count: 1,
        acknowledged: false,
        actions: [],
      });
      recommendations.push('监控并优化错误处理');
    }

    // 性能影响
    const slowOperations = this.analyzePerformance().slowOperations;
    if (slowOperations.length > 5) {
      healthScore -= 20;
      recommendations.push('优化慢操作性能');
    } else if (slowOperations.length > 2) {
      healthScore -= 10;
      recommendations.push('关注性能瓶颈');
    }

    // 告警数量影响
    const activeAlerts = Array.from(this.alerts.values()).filter(alert => !alert.acknowledged);
    if (activeAlerts.length > 10) {
      healthScore -= 15;
      recommendations.push('处理未确认的告警');
    }

    return {
      healthScore: Math.max(0, healthScore),
      alerts,
      recommendations,
    };
  }

  /**
   * 查询日志
   */
  queryLogs(query: LogQuery): LogEntry[] {
    let filteredLogs = [...this.logHistory];

    // 按级别过滤
    if (query.level && query.level.length > 0) {
      filteredLogs = filteredLogs.filter(log => query.level!.includes(log.level));
    }

    // 按类别过滤
    if (query.category && query.category.length > 0) {
      filteredLogs = filteredLogs.filter(log => {
        const category = this.categorizeLog(log);
        return query.category!.includes(category);
      });
    }

    // 按时间范围过滤
    if (query.timeRange) {
      filteredLogs = filteredLogs.filter(log => 
        log.timestamp >= query.timeRange!.start && log.timestamp <= query.timeRange!.end
      );
    }

    // 按搜索文本过滤
    if (query.searchText) {
      const searchText = query.searchText.toLowerCase();
      filteredLogs = filteredLogs.filter(log => 
        log.message.toLowerCase().includes(searchText) ||
        JSON.stringify(log.context || {}).toLowerCase().includes(searchText)
      );
    }

    // 按会话ID过滤
    if (query.sessionId) {
      filteredLogs = filteredLogs.filter(log => log.sessionId === query.sessionId);
    }

    // 按用户ID过滤
    if (query.userId) {
      filteredLogs = filteredLogs.filter(log => log.userId === query.userId);
    }

    // 分页
    const offset = query.offset || 0;
    const limit = query.limit || 100;
    
    return filteredLogs
      .sort((a, b) => b.timestamp - a.timestamp) // 按时间倒序
      .slice(offset, offset + limit);
  }

  /**
   * 获取最近日志
   */
  private getRecentLogs(timeWindow: number): LogEntry[] {
    const cutoff = Date.now() - timeWindow;
    return this.logHistory.filter(log => log.timestamp >= cutoff);
  }

  /**
   * 获取源码信息
   */
  private getSourceInfo() {
    return {
      file: 'logAnalytics.ts',
      function: 'collectLog',
      line: 0,
      column: 0,
    };
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
   * 生成会话ID
   */
  private generateSessionId(): string {
    return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * 生成告警ID
   */
  private generateAlertId(title: string): string {
    return title.toLowerCase().replace(/[^a-z0-9]/g, '_');
  }

  /**
   * 获取日志模式
   */
  getLogPatterns(): LogPattern[] {
    return Array.from(this.patterns.values())
      .sort((a, b) => b.frequency - a.frequency);
  }

  /**
   * 获取告警
   */
  getAlerts(): LogAlert[] {
    return Array.from(this.alerts.values())
      .sort((a, b) => b.timestamp - a.timestamp);
  }

  /**
   * 确认告警
   */
  acknowledgeAlert(alertId: string) {
    const alert = this.alerts.get(alertId);
    if (alert) {
      alert.acknowledged = true;
      logger.info('Alert acknowledged', { alertId, title: alert.title });
    }
  }

  /**
   * 清理历史数据
   */
  clearHistory() {
    this.logHistory = [];
    this.patterns.clear();
    this.alerts.clear();
    logger.info('Log analytics history cleared');
  }

  /**
   * 导出分析报告
   */
  exportReport(): string {
    const analytics = this.generateAnalytics();
    const report = {
      timestamp: new Date().toISOString(),
      analytics,
      patterns: this.getLogPatterns(),
      alerts: this.getAlerts(),
      summary: {
        totalLogs: analytics.totalLogs,
        errorRate: analytics.errorAnalysis.errorRate,
        healthScore: analytics.systemHealth.healthScore,
        topIssues: analytics.errorAnalysis.topErrors.slice(0, 5),
      },
    };

    return JSON.stringify(report, null, 2);
  }

  /**
   * 销毁服务
   */
  destroy() {
    if (this.analysisInterval) {
      clearInterval(this.analysisInterval);
    }
    this.clearHistory();
  }
}

// 创建全局实例
export const logAnalytics = new LogAnalyticsService();

export default LogAnalyticsService;
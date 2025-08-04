/**
 * 用户行为分析服务
 * 提供详细的用户行为跟踪和分析功能
 */

import { userAnalytics } from './userAnalytics';
import { logger } from './logger';

export interface UserBehaviorEvent {
  eventId: string;
  eventType: 'page_view' | 'click' | 'scroll' | 'form_interaction' | 'api_call' | 'error' | 'custom';
  eventName: string;
  timestamp: number;
  sessionId: string;
  userId?: string;
  
  // 事件属性
  properties: {
    page?: {
      url: string;
      title: string;
      referrer: string;
      loadTime: number;
    };
    element?: {
      tagName: string;
      id?: string;
      className?: string;
      text?: string;
      position: { x: number; y: number };
    };
    form?: {
      formId: string;
      fieldName: string;
      fieldType: string;
      value?: string;
      isValid: boolean;
    };
    api?: {
      endpoint: string;
      method: string;
      status: number;
      duration: number;
      success: boolean;
    };
    custom?: Record<string, any>;
  };
  
  // 上下文信息
  context: {
    viewport: { width: number; height: number };
    scroll: { x: number; y: number };
    mousePosition: { x: number; y: number };
    deviceInfo: {
      type: 'desktop' | 'tablet' | 'mobile';
      os: string;
      browser: string;
    };
  };
}

export interface UserSession {
  sessionId: string;
  userId?: string;
  startTime: number;
  endTime?: number;
  duration?: number;
  
  // 会话统计
  stats: {
    pageViews: number;
    clicks: number;
    scrolls: number;
    formInteractions: number;
    apiCalls: number;
    errors: number;
    customEvents: number;
  };
  
  // 用户路径
  userJourney: {
    pages: string[];
    actions: string[];
    timeSpent: Record<string, number>;
  };
  
  // 转化事件
  conversions: {
    eventName: string;
    timestamp: number;
    value?: number;
  }[];
  
  // 会话质量
  quality: {
    engagementScore: number; // 0-100
    satisfactionScore: number; // 0-100
    completionRate: number; // 0-100
    errorRate: number; // 0-100
  };
}

export interface BehaviorInsights {
  // 用户细分
  userSegments: {
    segmentName: string;
    userCount: number;
    characteristics: string[];
    behaviorPatterns: string[];
  }[];
  
  // 热门路径
  popularPaths: {
    path: string[];
    frequency: number;
    conversionRate: number;
    averageDuration: number;
  }[];
  
  // 流失点分析
  dropOffPoints: {
    page: string;
    action: string;
    dropOffRate: number;
    impact: 'low' | 'medium' | 'high';
  }[];
  
  // 功能使用情况
  featureUsage: {
    featureName: string;
    usageCount: number;
    uniqueUsers: number;
    averageUsageTime: number;
    satisfactionScore: number;
  }[];
}

class BehaviorAnalyticsService {
  private events: UserBehaviorEvent[] = [];
  private sessions: Map<string, UserSession> = new Map();
  private currentSession?: UserSession;
  private maxEventsSize: number = 10000;
  private maxSessionsSize: number = 1000;
  private isEnabled: boolean = true;

  // 鼠标和键盘状态跟踪
  private mousePosition = { x: 0, y: 0 };

  constructor() {
    this.setupEventListeners();
    this.startSession();
  }

  /**
   * 设置事件监听器
   */
  private setupEventListeners() {
    // 鼠标位置跟踪
    document.addEventListener('mousemove', (event) => {
      this.mousePosition = { x: event.clientX, y: event.clientY };
    });

    // 页面可见性变化
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) {
        this.trackEvent('page_hidden', 'page_view', {});
      } else {
        this.trackEvent('page_visible', 'page_view', {});
      }
    });

    // 页面卸载
    window.addEventListener('beforeunload', () => {
      this.endSession();
    });

    // 点击事件详细跟踪
    document.addEventListener('click', (event) => {
      this.trackClickEvent(event);
    }, true);

    // 表单交互跟踪
    document.addEventListener('input', (event) => {
      this.trackFormInteraction(event);
    }, true);

    // 滚动事件跟踪
    let scrollTimeout: NodeJS.Timeout;
    document.addEventListener('scroll', () => {
      clearTimeout(scrollTimeout);
      scrollTimeout = setTimeout(() => {
        this.trackScrollEvent();
      }, 100);
    });
  }

  /**
   * 跟踪点击事件
   */
  private trackClickEvent(event: MouseEvent) {
    const target = event.target as HTMLElement;
    if (!target) return;

    const element = {
      tagName: target.tagName.toLowerCase(),
      id: target.id,
      className: target.className,
      text: target.textContent?.slice(0, 100),
      position: { x: event.clientX, y: event.clientY },
    };

    this.trackEvent('element_click', 'click', {
      element,
      custom: {
        button: event.button,
        detail: event.detail,
      },
    });
  }

  /**
   * 跟踪表单交互
   */
  private trackFormInteraction(event: Event) {
    const target = event.target as HTMLInputElement;
    if (!target || !target.form) return;

    const form = {
      formId: target.form.id || 'unnamed',
      fieldName: target.name || target.id || 'unnamed',
      fieldType: target.type || 'unknown',
      value: target.type === 'password' ? '[REDACTED]' : target.value?.slice(0, 50),
      isValid: target.validity?.valid ?? true,
    };

    this.trackEvent('form_interaction', 'form_interaction', {
      form,
      custom: {
        eventType: event.type,
        required: target.required,
        disabled: target.disabled,
      },
    });
  }

  /**
   * 跟踪滚动事件
   */
  private trackScrollEvent() {
    const scrollPercent = Math.round(
      (window.scrollY / (document.documentElement.scrollHeight - window.innerHeight)) * 100
    );

    this.trackEvent('page_scroll', 'scroll', {
      custom: {
        scrollPercent,
        scrollY: window.scrollY,
        scrollX: window.scrollX,
      },
    });
  }

  /**
   * 跟踪事件
   */
  trackEvent(eventName: string, eventType: UserBehaviorEvent['eventType'], properties: Partial<UserBehaviorEvent['properties']>) {
    if (!this.isEnabled || !this.currentSession) return;

    const event: UserBehaviorEvent = {
      eventId: this.generateEventId(),
      eventType,
      eventName,
      timestamp: Date.now(),
      sessionId: this.currentSession.sessionId,
      userId: this.currentSession.userId,
      properties,
      context: this.getEventContext(),
    };

    // 添加到事件列表
    this.events.push(event);
    if (this.events.length > this.maxEventsSize) {
      this.events.shift();
    }

    // 更新会话统计
    this.updateSessionStats(event);

    // 更新用户路径
    this.updateUserJourney(event);

    // 检查转化事件
    this.checkConversion(event);

    // 记录到日志
    logger.info(`Behavior event: ${eventName}`, {
      eventType,
      sessionId: event.sessionId,
      userId: event.userId,
      properties,
    });

    // 同时发送到用户分析服务
    userAnalytics.trackEvent(eventName, properties, eventType);
  }

  /**
   * 跟踪页面浏览
   */
  trackPageView(url?: string, title?: string, loadTime?: number) {
    const page = {
      url: url || window.location.href,
      title: title || document.title,
      referrer: document.referrer,
      loadTime: loadTime || 0,
    };

    this.trackEvent('page_view', 'page_view', { page });
  }

  /**
   * 跟踪API调用
   */
  trackApiCall(endpoint: string, method: string, status: number, duration: number, success: boolean) {
    const api = {
      endpoint,
      method,
      status,
      duration,
      success,
    };

    this.trackEvent('api_call', 'api_call', { api });
  }

  /**
   * 跟踪错误
   */
  trackError(error: Error, context?: Record<string, any>) {
    this.trackEvent('error_occurred', 'error', {
      custom: {
        message: error.message,
        stack: error.stack,
        name: error.name,
        ...context,
      },
    });
  }

  /**
   * 跟踪转化事件
   */
  trackConversion(eventName: string, value?: number) {
    if (!this.currentSession) return;

    const conversion = {
      eventName,
      timestamp: Date.now(),
      value,
    };

    this.currentSession.conversions.push(conversion);
    this.trackEvent(`conversion_${eventName}`, 'custom', {
      custom: { conversionValue: value },
    });
  }

  /**
   * 开始新会话
   */
  private startSession(userId?: string) {
    const sessionId = this.generateSessionId();
    
    this.currentSession = {
      sessionId,
      userId,
      startTime: Date.now(),
      stats: {
        pageViews: 0,
        clicks: 0,
        scrolls: 0,
        formInteractions: 0,
        apiCalls: 0,
        errors: 0,
        customEvents: 0,
      },
      userJourney: {
        pages: [],
        actions: [],
        timeSpent: {},
      },
      conversions: [],
      quality: {
        engagementScore: 0,
        satisfactionScore: 0,
        completionRate: 0,
        errorRate: 0,
      },
    };

    this.sessions.set(sessionId, this.currentSession);
    
    // 限制会话数量
    if (this.sessions.size > this.maxSessionsSize) {
      const oldestSessionId = this.sessions.keys().next().value;
      this.sessions.delete(oldestSessionId);
    }

    logger.info('New behavior session started', { sessionId, userId });
  }

  /**
   * 结束会话
   */
  private endSession() {
    if (!this.currentSession) return;

    this.currentSession.endTime = Date.now();
    this.currentSession.duration = this.currentSession.endTime - this.currentSession.startTime;

    // 计算会话质量
    this.calculateSessionQuality(this.currentSession);

    logger.info('Behavior session ended', {
      sessionId: this.currentSession.sessionId,
      duration: this.currentSession.duration,
      stats: this.currentSession.stats,
      quality: this.currentSession.quality,
    });

    this.currentSession = undefined;
  }

  /**
   * 更新会话统计
   */
  private updateSessionStats(event: UserBehaviorEvent) {
    if (!this.currentSession) return;

    switch (event.eventType) {
      case 'page_view':
        this.currentSession.stats.pageViews++;
        break;
      case 'click':
        this.currentSession.stats.clicks++;
        break;
      case 'scroll':
        this.currentSession.stats.scrolls++;
        break;
      case 'form_interaction':
        this.currentSession.stats.formInteractions++;
        break;
      case 'api_call':
        this.currentSession.stats.apiCalls++;
        break;
      case 'error':
        this.currentSession.stats.errors++;
        break;
      case 'custom':
        this.currentSession.stats.customEvents++;
        break;
    }
  }

  /**
   * 更新用户路径
   */
  private updateUserJourney(event: UserBehaviorEvent) {
    if (!this.currentSession) return;

    const journey = this.currentSession.userJourney;

    // 记录页面访问
    if (event.eventType === 'page_view' && event.properties.page) {
      const page = event.properties.page.url;
      if (!journey.pages.includes(page)) {
        journey.pages.push(page);
      }
      
      // 记录页面停留时间
      const lastPageTime = journey.timeSpent[page] || 0;
      journey.timeSpent[page] = lastPageTime + 1000; // 简化计算
    }

    // 记录用户操作
    if (event.eventType !== 'page_view') {
      journey.actions.push(`${event.eventType}:${event.eventName}`);
    }
  }

  /**
   * 检查转化事件
   */
  private checkConversion(event: UserBehaviorEvent) {
    // 定义转化事件规则
    const conversionRules = [
      {
        name: 'strategy_created',
        condition: (e: UserBehaviorEvent) => 
          e.eventName === 'form_submit' && 
          e.properties.form?.formId?.includes('strategy'),
      },
      {
        name: 'order_placed',
        condition: (e: UserBehaviorEvent) => 
          e.eventName === 'api_call' && 
          e.properties.api?.endpoint?.includes('/order/create') &&
          e.properties.api?.success,
      },
      {
        name: 'report_exported',
        condition: (e: UserBehaviorEvent) => 
          e.eventName === 'button_click' && 
          e.properties.element?.text?.includes('导出'),
      },
    ];

    conversionRules.forEach(rule => {
      if (rule.condition(event)) {
        this.trackConversion(rule.name);
      }
    });
  }

  /**
   * 计算会话质量
   */
  private calculateSessionQuality(session: UserSession) {
    const stats = session.stats;
    const duration = session.duration || 0;

    // 参与度评分 (基于交互次数和时长)
    const totalInteractions = stats.clicks + stats.scrolls + stats.formInteractions;
    const engagementScore = Math.min(100, (totalInteractions / (duration / 60000)) * 10);

    // 满意度评分 (基于错误率和完成度)
    const errorRate = stats.errors / Math.max(1, totalInteractions);
    const satisfactionScore = Math.max(0, 100 - (errorRate * 100));

    // 完成率评分 (基于转化事件)
    const completionRate = (session.conversions.length / Math.max(1, stats.pageViews)) * 100;

    session.quality = {
      engagementScore: Math.round(engagementScore),
      satisfactionScore: Math.round(satisfactionScore),
      completionRate: Math.round(completionRate),
      errorRate: Math.round(errorRate * 100),
    };
  }

  /**
   * 获取事件上下文
   */
  private getEventContext(): UserBehaviorEvent['context'] {
    return {
      viewport: {
        width: window.innerWidth,
        height: window.innerHeight,
      },
      scroll: {
        x: window.scrollX,
        y: window.scrollY,
      },
      mousePosition: this.mousePosition,
      deviceInfo: this.getDeviceInfo(),
    };
  }

  /**
   * 获取设备信息
   */
  private getDeviceInfo() {
    const userAgent = navigator.userAgent;
    let deviceType: 'desktop' | 'tablet' | 'mobile' = 'desktop';
    
    if (/Mobile|Android|iPhone/.test(userAgent)) {
      deviceType = 'mobile';
    } else if (/iPad|Tablet/.test(userAgent)) {
      deviceType = 'tablet';
    }

    return {
      type: deviceType,
      os: this.getOS(userAgent),
      browser: this.getBrowser(userAgent),
    };
  }

  /**
   * 获取操作系统
   */
  private getOS(userAgent: string): string {
    if (userAgent.includes('Windows')) return 'Windows';
    if (userAgent.includes('Mac')) return 'macOS';
    if (userAgent.includes('Linux')) return 'Linux';
    if (userAgent.includes('Android')) return 'Android';
    if (userAgent.includes('iOS')) return 'iOS';
    return 'Unknown';
  }

  /**
   * 获取浏览器
   */
  private getBrowser(userAgent: string): string {
    if (userAgent.includes('Chrome')) return 'Chrome';
    if (userAgent.includes('Firefox')) return 'Firefox';
    if (userAgent.includes('Safari')) return 'Safari';
    if (userAgent.includes('Edge')) return 'Edge';
    return 'Unknown';
  }

  /**
   * 生成行为洞察
   */
  generateInsights(): BehaviorInsights {
    const insights: BehaviorInsights = {
      userSegments: this.generateUserSegments(),
      popularPaths: this.generatePopularPaths(),
      dropOffPoints: this.generateDropOffPoints(),
      featureUsage: this.generateFeatureUsage(),
    };

    logger.info('Behavior insights generated', insights);
    return insights;
  }

  /**
   * 生成用户细分
   */
  private generateUserSegments() {
    return [
      {
        segmentName: '高活跃用户',
        userCount: Math.floor(this.sessions.size * 0.2),
        characteristics: ['高交互频率', '长会话时长', '多页面访问'],
        behaviorPatterns: ['深度浏览', '功能探索', '数据分析'],
      },
      {
        segmentName: '普通用户',
        userCount: Math.floor(this.sessions.size * 0.6),
        characteristics: ['中等交互频率', '标准会话时长', '目标明确'],
        behaviorPatterns: ['任务导向', '快速操作', '结果关注'],
      },
      {
        segmentName: '新用户',
        userCount: Math.floor(this.sessions.size * 0.2),
        characteristics: ['低交互频率', '短会话时长', '探索性行为'],
        behaviorPatterns: ['界面熟悉', '功能试用', '帮助查看'],
      },
    ];
  }

  /**
   * 生成热门路径
   */
  private generatePopularPaths() {
    const pathFrequency = new Map<string, { count: number; conversions: number; totalDuration: number }>();

    this.sessions.forEach(session => {
      const path = session.userJourney.pages.slice(0, 3);
      if (path.length >= 2) {
        const pathKey = path.join(' -> ');
        const existing = pathFrequency.get(pathKey) || { count: 0, conversions: 0, totalDuration: 0 };
        existing.count++;
        existing.conversions += session.conversions.length;
        existing.totalDuration += session.duration || 0;
        pathFrequency.set(pathKey, existing);
      }
    });

    return Array.from(pathFrequency.entries())
      .map(([pathKey, data]) => ({
        path: pathKey.split(' -> '),
        frequency: data.count,
        conversionRate: data.count > 0 ? (data.conversions / data.count) * 100 : 0,
        averageDuration: data.count > 0 ? data.totalDuration / data.count : 0,
      }))
      .sort((a, b) => b.frequency - a.frequency)
      .slice(0, 10);
  }

  /**
   * 生成流失点分析
   */
  private generateDropOffPoints() {
    const dropOffData = new Map<string, { total: number; dropOff: number }>();

    this.sessions.forEach(session => {
      session.userJourney.pages.forEach((page, index) => {
        const existing = dropOffData.get(page) || { total: 0, dropOff: 0 };
        existing.total++;
        
        if (index === session.userJourney.pages.length - 1 && session.conversions.length === 0) {
          existing.dropOff++;
        }
        
        dropOffData.set(page, existing);
      });
    });

    return Array.from(dropOffData.entries())
      .map(([page, data]) => ({
        page,
        action: 'page_exit',
        dropOffRate: data.total > 0 ? (data.dropOff / data.total) * 100 : 0,
        impact: data.dropOff > 5 ? 'high' : data.dropOff > 2 ? 'medium' : 'low' as const,
      }))
      .filter(item => item.dropOffRate > 10)
      .sort((a, b) => b.dropOffRate - a.dropOffRate)
      .slice(0, 10);
  }

  /**
   * 生成功能使用情况
   */
  private generateFeatureUsage() {
    const featureUsage = new Map<string, { count: number; users: Set<string>; totalTime: number }>();

    this.events.forEach(event => {
      if (event.eventType === 'click' && event.properties.element) {
        const feature = this.identifyFeature(event.properties.element);
        if (feature) {
          const existing = featureUsage.get(feature) || { count: 0, users: new Set(), totalTime: 0 };
          existing.count++;
          if (event.userId) {
            existing.users.add(event.userId);
          }
          existing.totalTime += 1000;
          featureUsage.set(feature, existing);
        }
      }
    });

    return Array.from(featureUsage.entries())
      .map(([featureName, data]) => ({
        featureName,
        usageCount: data.count,
        uniqueUsers: data.users.size,
        averageUsageTime: data.count > 0 ? data.totalTime / data.count : 0,
        satisfactionScore: Math.random() * 100,
      }))
      .sort((a, b) => b.usageCount - a.usageCount)
      .slice(0, 20);
  }

  /**
   * 识别功能
   */
  private identifyFeature(element: UserBehaviorEvent['properties']['element']): string | null {
    if (!element) return null;

    const text = element.text?.toLowerCase() || '';
    const className = element.className?.toLowerCase() || '';
    const id = element.id?.toLowerCase() || '';

    if (text.includes('策略') || className.includes('strategy') || id.includes('strategy')) {
      return '策略管理';
    }
    if (text.includes('订单') || className.includes('order') || id.includes('order')) {
      return '订单管理';
    }
    if (text.includes('投资组合') || className.includes('portfolio') || id.includes('portfolio')) {
      return '投资组合';
    }
    if (text.includes('风险') || className.includes('risk') || id.includes('risk')) {
      return '风险管理';
    }
    if (text.includes('数据') || className.includes('data') || id.includes('data')) {
      return '数据监控';
    }
    if (text.includes('回测') || className.includes('backtest') || id.includes('backtest')) {
      return '回测分析';
    }
    if (text.includes('系统') || className.includes('system') || id.includes('system')) {
      return '系统管理';
    }

    return null;
  }

  /**
   * 生成事件ID
   */
  private generateEventId(): string {
    return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
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
    if (this.currentSession) {
      this.currentSession.userId = userId;
    }
    logger.info('User ID set for behavior analytics', { userId });
  }

  /**
   * 获取当前会话
   */
  getCurrentSession(): UserSession | undefined {
    return this.currentSession;
  }

  /**
   * 获取所有会话
   */
  getAllSessions(): UserSession[] {
    return Array.from(this.sessions.values());
  }

  /**
   * 获取事件历史
   */
  getEventHistory(limit?: number): UserBehaviorEvent[] {
    return limit ? this.events.slice(-limit) : [...this.events];
  }

  /**
   * 清理数据
   */
  clearData() {
    this.events = [];
    this.sessions.clear();
    logger.info('Behavior analytics data cleared');
  }

  /**
   * 启用/禁用服务
   */
  setEnabled(enabled: boolean) {
    this.isEnabled = enabled;
    logger.info(`Behavior analytics ${enabled ? 'enabled' : 'disabled'}`);
  }

  /**
   * 销毁服务
   */
  destroy() {
    this.endSession();
    this.clearData();
  }
}

// 创建全局实例
export const behaviorAnalytics = new BehaviorAnalyticsService();

export default BehaviorAnalyticsService;
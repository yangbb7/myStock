/**
 * 用户行为分析服务
 * 用于收集和分析用户行为数据
 */

export interface UserEvent {
  eventType: string;
  eventName: string;
  timestamp: number;
  sessionId: string;
  userId?: string;
  properties: Record<string, any>;
  context: {
    url: string;
    referrer: string;
    userAgent: string;
    viewport: { width: number; height: number };
    timezone: string;
    language: string;
  };
}

export interface PageView {
  url: string;
  title: string;
  referrer: string;
  timestamp: number;
  sessionId: string;
  userId?: string;
  duration?: number;
  scrollDepth?: number;
  exitType?: 'navigation' | 'close' | 'refresh';
}

export interface UserSession {
  sessionId: string;
  userId?: string;
  startTime: number;
  endTime?: number;
  duration?: number;
  pageViews: number;
  events: number;
  device: {
    type: 'desktop' | 'tablet' | 'mobile';
    os: string;
    browser: string;
  };
  location?: {
    country?: string;
    city?: string;
    timezone: string;
  };
}

export interface ConversionFunnel {
  funnelName: string;
  steps: {
    stepName: string;
    eventType: string;
    timestamp: number;
    completed: boolean;
  }[];
  sessionId: string;
  userId?: string;
  totalDuration: number;
  conversionRate: number;
}

class UserAnalyticsService {
  private isEnabled: boolean = false;
  private apiEndpoint: string = '';
  private sessionId: string = '';
  private userId?: string;
  private currentSession: UserSession;
  private eventQueue: UserEvent[] = [];
  private pageViewQueue: PageView[] = [];
  private currentPageView?: PageView;
  private funnels: Map<string, ConversionFunnel> = new Map();
  private maxQueueSize: number = 100;
  private flushInterval: number = 30000; // 30秒
  private flushTimer?: NodeJS.Timeout;
  private sessionTimer?: NodeJS.Timeout;
  private scrollDepthTimer?: NodeJS.Timeout;
  private maxScrollDepth: number = 0;

  constructor() {
    this.sessionId = this.generateSessionId();
    this.currentSession = this.createSession();
    this.setupEventListeners();
  }

  /**
   * 初始化用户分析
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
      this.trackPageView();
      console.log('User analytics initialized');
    }
  }

  /**
   * 设置事件监听器
   */
  private setupEventListeners() {
    // 页面可见性变化
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) {
        this.handlePageHide();
      } else {
        this.handlePageShow();
      }
    });

    // 页面卸载
    window.addEventListener('beforeunload', () => {
      this.handlePageUnload();
    });

    // 滚动深度跟踪
    window.addEventListener('scroll', this.throttle(() => {
      this.trackScrollDepth();
    }, 100));

    // 点击事件跟踪
    document.addEventListener('click', (event) => {
      this.trackClick(event);
    });

    // 表单提交跟踪
    document.addEventListener('submit', (event) => {
      this.trackFormSubmit(event);
    });

    // 错误跟踪
    window.addEventListener('error', (event) => {
      this.trackError(event);
    });
  }

  /**
   * 创建会话
   */
  private createSession(): UserSession {
    return {
      sessionId: this.sessionId,
      userId: this.userId,
      startTime: Date.now(),
      pageViews: 0,
      events: 0,
      device: this.getDeviceInfo(),
      location: {
        timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
      },
    };
  }

  /**
   * 跟踪页面浏览
   */
  trackPageView(url?: string, title?: string) {
    if (!this.isEnabled) return;

    // 结束当前页面浏览
    if (this.currentPageView) {
      this.endPageView();
    }

    // 开始新的页面浏览
    this.currentPageView = {
      url: url || window.location.href,
      title: title || document.title,
      referrer: document.referrer,
      timestamp: Date.now(),
      sessionId: this.sessionId,
      userId: this.userId,
      scrollDepth: 0,
    };

    this.currentSession.pageViews++;
    this.maxScrollDepth = 0;
  }

  /**
   * 结束页面浏览
   */
  private endPageView(exitType: 'navigation' | 'close' | 'refresh' = 'navigation') {
    if (!this.currentPageView) return;

    this.currentPageView.duration = Date.now() - this.currentPageView.timestamp;
    this.currentPageView.scrollDepth = this.maxScrollDepth;
    this.currentPageView.exitType = exitType;

    this.pageViewQueue.push({ ...this.currentPageView });
    if (this.pageViewQueue.length > this.maxQueueSize) {
      this.pageViewQueue.shift();
    }

    this.currentPageView = undefined;
  }

  /**
   * 跟踪事件
   */
  trackEvent(eventName: string, properties: Record<string, any> = {}, eventType: string = 'custom') {
    if (!this.isEnabled) return;

    const event: UserEvent = {
      eventType,
      eventName,
      timestamp: Date.now(),
      sessionId: this.sessionId,
      userId: this.userId,
      properties,
      context: this.getContext(),
    };

    this.eventQueue.push(event);
    if (this.eventQueue.length > this.maxQueueSize) {
      this.eventQueue.shift();
    }

    this.currentSession.events++;

    // 检查转化漏斗
    this.checkFunnelProgress(eventName, properties);
  }

  /**
   * 跟踪点击事件
   */
  private trackClick(event: MouseEvent) {
    const target = event.target as HTMLElement;
    if (!target) return;

    const properties = {
      elementType: target.tagName.toLowerCase(),
      elementId: target.id,
      elementClass: target.className,
      elementText: target.textContent?.slice(0, 100),
      x: event.clientX,
      y: event.clientY,
      button: event.button,
    };

    this.trackEvent('click', properties, 'interaction');
  }

  /**
   * 跟踪表单提交
   */
  private trackFormSubmit(event: SubmitEvent) {
    const form = event.target as HTMLFormElement;
    if (!form) return;

    const formData = new FormData(form);
    const fields = Array.from(formData.keys());

    const properties = {
      formId: form.id,
      formClass: form.className,
      formAction: form.action,
      formMethod: form.method,
      fieldCount: fields.length,
      fields: fields,
    };

    this.trackEvent('form_submit', properties, 'conversion');
  }

  /**
   * 跟踪错误
   */
  private trackError(event: ErrorEvent) {
    const properties = {
      message: event.message,
      filename: event.filename,
      lineno: event.lineno,
      colno: event.colno,
      stack: event.error?.stack,
    };

    this.trackEvent('javascript_error', properties, 'error');
  }

  /**
   * 跟踪滚动深度
   */
  private trackScrollDepth() {
    const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
    const documentHeight = document.documentElement.scrollHeight - window.innerHeight;
    const scrollPercent = Math.round((scrollTop / documentHeight) * 100);

    if (scrollPercent > this.maxScrollDepth) {
      this.maxScrollDepth = scrollPercent;

      // 跟踪滚动里程碑
      const milestones = [25, 50, 75, 90, 100];
      const milestone = milestones.find(m => scrollPercent >= m && this.maxScrollDepth < m);
      
      if (milestone) {
        this.trackEvent('scroll_depth', { depth: milestone }, 'engagement');
      }
    }
  }

  /**
   * 定义转化漏斗
   */
  defineFunnel(funnelName: string, steps: string[]) {
    const funnel: ConversionFunnel = {
      funnelName,
      steps: steps.map(step => ({
        stepName: step,
        eventType: 'custom',
        timestamp: 0,
        completed: false,
      })),
      sessionId: this.sessionId,
      userId: this.userId,
      totalDuration: 0,
      conversionRate: 0,
    };

    this.funnels.set(funnelName, funnel);
  }

  /**
   * 检查漏斗进度
   */
  private checkFunnelProgress(eventName: string, properties: Record<string, any>) {
    this.funnels.forEach((funnel, funnelName) => {
      const currentStep = funnel.steps.find(step => 
        step.stepName === eventName && !step.completed
      );

      if (currentStep) {
        currentStep.completed = true;
        currentStep.timestamp = Date.now();

        // 计算转化率
        const completedSteps = funnel.steps.filter(step => step.completed).length;
        funnel.conversionRate = (completedSteps / funnel.steps.length) * 100;

        // 如果完成了所有步骤
        if (completedSteps === funnel.steps.length) {
          funnel.totalDuration = Date.now() - funnel.steps[0].timestamp;
          this.trackEvent('funnel_completed', {
            funnelName,
            duration: funnel.totalDuration,
            conversionRate: funnel.conversionRate,
          }, 'conversion');
        }
      }
    });
  }

  /**
   * 跟踪自定义转化事件
   */
  trackConversion(conversionName: string, value?: number, properties: Record<string, any> = {}) {
    this.trackEvent('conversion', {
      conversionName,
      value,
      ...properties,
    }, 'conversion');
  }

  /**
   * 跟踪用户属性
   */
  setUserProperties(properties: Record<string, any>) {
    this.trackEvent('user_properties_updated', properties, 'user');
  }

  /**
   * 处理页面隐藏
   */
  private handlePageHide() {
    this.trackEvent('page_hidden', {}, 'engagement');
    this.flush(); // 立即上报数据
  }

  /**
   * 处理页面显示
   */
  private handlePageShow() {
    this.trackEvent('page_visible', {}, 'engagement');
  }

  /**
   * 处理页面卸载
   */
  private handlePageUnload() {
    this.endPageView('close');
    this.endSession();
    
    // 使用sendBeacon确保数据发送
    if (navigator.sendBeacon && this.apiEndpoint) {
      const payload = this.createPayload();
      navigator.sendBeacon(this.apiEndpoint, JSON.stringify(payload));
    }
  }

  /**
   * 结束会话
   */
  private endSession() {
    this.currentSession.endTime = Date.now();
    this.currentSession.duration = this.currentSession.endTime - this.currentSession.startTime;
  }

  /**
   * 获取设备信息
   */
  private getDeviceInfo() {
    const userAgent = navigator.userAgent;
    let deviceType: 'desktop' | 'tablet' | 'mobile' = 'desktop';
    
    if (/Mobile|Android|iPhone|iPad/.test(userAgent)) {
      deviceType = /iPad/.test(userAgent) ? 'tablet' : 'mobile';
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
   * 获取上下文信息
   */
  private getContext() {
    return {
      url: window.location.href,
      referrer: document.referrer,
      userAgent: navigator.userAgent,
      viewport: {
        width: window.innerWidth,
        height: window.innerHeight,
      },
      timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
      language: navigator.language,
    };
  }

  /**
   * 创建上报数据
   */
  private createPayload() {
    return {
      events: [...this.eventQueue],
      pageViews: [...this.pageViewQueue],
      session: this.currentSession,
      funnels: Array.from(this.funnels.values()),
      metadata: {
        timestamp: Date.now(),
        version: '1.0.0',
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
    }, this.flushInterval);
  }

  /**
   * 上报数据
   */
  private async flush() {
    if (!this.isEnabled || (!this.eventQueue.length && !this.pageViewQueue.length)) {
      return;
    }

    const payload = this.createPayload();

    try {
      await fetch(this.apiEndpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      // 清空队列
      this.eventQueue = [];
      this.pageViewQueue = [];
    } catch (error) {
      console.warn('Failed to report analytics data:', error);
    }
  }

  /**
   * 节流函数
   */
  private throttle<T extends (...args: any[]) => any>(func: T, delay: number): T {
    let timeoutId: NodeJS.Timeout | null = null;
    let lastExecTime = 0;
    
    return ((...args: any[]) => {
      const currentTime = Date.now();
      
      if (currentTime - lastExecTime > delay) {
        func(...args);
        lastExecTime = currentTime;
      } else {
        if (timeoutId) clearTimeout(timeoutId);
        timeoutId = setTimeout(() => {
          func(...args);
          lastExecTime = Date.now();
        }, delay - (currentTime - lastExecTime));
      }
    }) as T;
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
    this.currentSession.userId = userId;
  }

  /**
   * 手动上报
   */
  async report() {
    await this.flush();
  }

  /**
   * 销毁分析服务
   */
  destroy() {
    if (this.flushTimer) {
      clearInterval(this.flushTimer);
    }
    if (this.sessionTimer) {
      clearInterval(this.sessionTimer);
    }
    if (this.scrollDepthTimer) {
      clearInterval(this.scrollDepthTimer);
    }
    
    this.endPageView('close');
    this.endSession();
    this.flush(); // 最后一次上报
  }
}

// 创建全局实例
export const userAnalytics = new UserAnalyticsService();

// 导出类型和实例
export default UserAnalyticsService;
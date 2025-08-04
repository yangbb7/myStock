/**
 * 前端性能监控服务
 * 用于收集和分析前端性能数据
 */

export interface PerformanceMetrics {
  // 页面加载性能
  pageLoad: {
    domContentLoaded: number;
    loadComplete: number;
    firstPaint: number;
    firstContentfulPaint: number;
    largestContentfulPaint: number;
    firstInputDelay: number;
    cumulativeLayoutShift: number;
  };
  
  // 资源加载性能
  resources: ResourceTiming[];
  
  // API请求性能
  apiRequests: ApiTiming[];
  
  // 内存使用情况
  memory?: {
    usedJSHeapSize: number;
    totalJSHeapSize: number;
    jsHeapSizeLimit: number;
  };
  
  // 设备信息
  device: {
    userAgent: string;
    viewport: { width: number; height: number };
    screen: { width: number; height: number };
    connection?: {
      effectiveType: string;
      downlink: number;
      rtt: number;
    };
  };
  
  // 元数据
  metadata: {
    url: string;
    timestamp: number;
    sessionId: string;
    userId?: string;
  };
}

export interface ResourceTiming {
  name: string;
  type: string;
  startTime: number;
  duration: number;
  size: number;
  cached: boolean;
}

export interface ApiTiming {
  url: string;
  method: string;
  startTime: number;
  duration: number;
  status: number;
  size: number;
}

export interface UserInteraction {
  type: 'click' | 'scroll' | 'input' | 'navigation';
  target: string;
  timestamp: number;
  duration?: number;
  metadata?: Record<string, any>;
}

class PerformanceMonitoringService {
  private isEnabled: boolean = false;
  private apiEndpoint: string = '';
  private sessionId: string = '';
  private userId?: string;
  private metricsQueue: PerformanceMetrics[] = [];
  private interactionQueue: UserInteraction[] = [];
  private apiTimings: Map<string, { startTime: number; url: string; method: string }> = new Map();
  private observer?: PerformanceObserver;
  private maxQueueSize: number = 50;
  private flushInterval: number = 60000; // 60秒
  private flushTimer?: NodeJS.Timeout;

  constructor() {
    this.sessionId = this.generateSessionId();
  }

  /**
   * 初始化性能监控
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
    this.maxQueueSize = config.maxQueueSize || 50;
    this.flushInterval = config.flushInterval || 60000;

    if (this.isEnabled) {
      this.setupPerformanceObserver();
      this.setupUserInteractionTracking();
      this.startFlushTimer();
      this.collectInitialMetrics();
      console.log('Performance monitoring initialized');
    }
  }

  /**
   * 设置性能观察器
   */
  private setupPerformanceObserver() {
    if (!('PerformanceObserver' in window)) {
      console.warn('PerformanceObserver not supported');
      return;
    }

    try {
      this.observer = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        entries.forEach((entry) => {
          this.processPerformanceEntry(entry);
        });
      });

      // 观察各种性能指标
      this.observer.observe({ entryTypes: ['navigation', 'resource', 'paint', 'largest-contentful-paint', 'first-input', 'layout-shift'] });
    } catch (error) {
      console.warn('Failed to setup PerformanceObserver:', error);
    }
  }

  /**
   * 处理性能条目
   */
  private processPerformanceEntry(entry: PerformanceEntry) {
    switch (entry.entryType) {
      case 'navigation':
        this.processNavigationEntry(entry as PerformanceNavigationTiming);
        break;
      case 'resource':
        this.processResourceEntry(entry as PerformanceResourceTiming);
        break;
      case 'paint':
        this.processPaintEntry(entry);
        break;
      case 'largest-contentful-paint':
        this.processLCPEntry(entry);
        break;
      case 'first-input':
        this.processFIDEntry(entry);
        break;
      case 'layout-shift':
        this.processCLSEntry(entry);
        break;
    }
  }

  /**
   * 处理导航性能
   */
  private processNavigationEntry(entry: PerformanceNavigationTiming) {
    const metrics: PerformanceMetrics = {
      pageLoad: {
        domContentLoaded: entry.domContentLoadedEventEnd - entry.domContentLoadedEventStart,
        loadComplete: entry.loadEventEnd - entry.loadEventStart,
        firstPaint: 0,
        firstContentfulPaint: 0,
        largestContentfulPaint: 0,
        firstInputDelay: 0,
        cumulativeLayoutShift: 0,
      },
      resources: [],
      apiRequests: [],
      memory: this.getMemoryInfo(),
      device: this.getDeviceInfo(),
      metadata: {
        url: window.location.href,
        timestamp: Date.now(),
        sessionId: this.sessionId,
        userId: this.userId,
      },
    };

    this.addToQueue(metrics);
  }

  /**
   * 处理资源加载性能
   */
  private processResourceEntry(entry: PerformanceResourceTiming) {
    const resource: ResourceTiming = {
      name: entry.name,
      type: this.getResourceType(entry.name),
      startTime: entry.startTime,
      duration: entry.duration,
      size: entry.transferSize || 0,
      cached: entry.transferSize === 0 && entry.decodedBodySize > 0,
    };

    // 添加到当前指标中
    // 这里可以根据需要优化存储策略
  }

  /**
   * 处理绘制性能
   */
  private processPaintEntry(entry: PerformanceEntry) {
    if (entry.name === 'first-paint') {
      // 更新FP指标
    } else if (entry.name === 'first-contentful-paint') {
      // 更新FCP指标
    }
  }

  /**
   * 处理LCP性能
   */
  private processLCPEntry(entry: PerformanceEntry) {
    // 更新LCP指标
  }

  /**
   * 处理FID性能
   */
  private processFIDEntry(entry: PerformanceEntry) {
    // 更新FID指标
  }

  /**
   * 处理CLS性能
   */
  private processCLSEntry(entry: any) {
    // 更新CLS指标
  }

  /**
   * 设置用户交互跟踪
   */
  private setupUserInteractionTracking() {
    // 点击事件
    document.addEventListener('click', (event) => {
      this.trackInteraction({
        type: 'click',
        target: this.getElementSelector(event.target as Element),
        timestamp: Date.now(),
        metadata: {
          x: event.clientX,
          y: event.clientY,
          button: event.button,
        },
      });
    });

    // 滚动事件（防抖）
    let scrollTimer: NodeJS.Timeout;
    document.addEventListener('scroll', () => {
      clearTimeout(scrollTimer);
      scrollTimer = setTimeout(() => {
        this.trackInteraction({
          type: 'scroll',
          target: 'window',
          timestamp: Date.now(),
          metadata: {
            scrollY: window.scrollY,
            scrollX: window.scrollX,
          },
        });
      }, 100);
    });

    // 输入事件（防抖）
    let inputTimer: NodeJS.Timeout;
    document.addEventListener('input', (event) => {
      clearTimeout(inputTimer);
      inputTimer = setTimeout(() => {
        this.trackInteraction({
          type: 'input',
          target: this.getElementSelector(event.target as Element),
          timestamp: Date.now(),
          metadata: {
            inputType: (event.target as HTMLInputElement).type,
            valueLength: (event.target as HTMLInputElement).value?.length || 0,
          },
        });
      }, 300);
    });
  }

  /**
   * 跟踪用户交互
   */
  private trackInteraction(interaction: UserInteraction) {
    if (!this.isEnabled) return;

    this.interactionQueue.push(interaction);
    if (this.interactionQueue.length > this.maxQueueSize) {
      this.interactionQueue.shift();
    }
  }

  /**
   * 跟踪API请求性能
   */
  trackApiRequest(config: {
    url: string;
    method: string;
    requestId: string;
  }) {
    if (!this.isEnabled) return;

    this.apiTimings.set(config.requestId, {
      startTime: performance.now(),
      url: config.url,
      method: config.method,
    });
  }

  /**
   * 完成API请求跟踪
   */
  completeApiRequest(config: {
    requestId: string;
    status: number;
    responseSize?: number;
  }) {
    if (!this.isEnabled) return;

    const timing = this.apiTimings.get(config.requestId);
    if (!timing) return;

    const apiTiming: ApiTiming = {
      url: timing.url,
      method: timing.method,
      startTime: timing.startTime,
      duration: performance.now() - timing.startTime,
      status: config.status,
      size: config.responseSize || 0,
    };

    // 添加到队列或当前指标中
    this.apiTimings.delete(config.requestId);
  }

  /**
   * 收集初始性能指标
   */
  private collectInitialMetrics() {
    // 等待页面加载完成后收集指标
    if (document.readyState === 'complete') {
      this.collectPageLoadMetrics();
    } else {
      window.addEventListener('load', () => {
        setTimeout(() => this.collectPageLoadMetrics(), 1000);
      });
    }
  }

  /**
   * 收集页面加载指标
   */
  private collectPageLoadMetrics() {
    const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
    const paint = performance.getEntriesByType('paint');

    if (!navigation) return;

    const metrics: PerformanceMetrics = {
      pageLoad: {
        domContentLoaded: navigation.domContentLoadedEventEnd - navigation.domContentLoadedEventStart,
        loadComplete: navigation.loadEventEnd - navigation.loadEventStart,
        firstPaint: paint.find(p => p.name === 'first-paint')?.startTime || 0,
        firstContentfulPaint: paint.find(p => p.name === 'first-contentful-paint')?.startTime || 0,
        largestContentfulPaint: 0, // 需要通过observer获取
        firstInputDelay: 0, // 需要通过observer获取
        cumulativeLayoutShift: 0, // 需要通过observer获取
      },
      resources: this.collectResourceMetrics(),
      apiRequests: [],
      memory: this.getMemoryInfo(),
      device: this.getDeviceInfo(),
      metadata: {
        url: window.location.href,
        timestamp: Date.now(),
        sessionId: this.sessionId,
        userId: this.userId,
      },
    };

    this.addToQueue(metrics);
  }

  /**
   * 收集资源指标
   */
  private collectResourceMetrics(): ResourceTiming[] {
    const resources = performance.getEntriesByType('resource') as PerformanceResourceTiming[];
    
    return resources.map(resource => ({
      name: resource.name,
      type: this.getResourceType(resource.name),
      startTime: resource.startTime,
      duration: resource.duration,
      size: resource.transferSize || 0,
      cached: resource.transferSize === 0 && resource.decodedBodySize > 0,
    }));
  }

  /**
   * 获取资源类型
   */
  private getResourceType(url: string): string {
    if (url.includes('.js')) return 'script';
    if (url.includes('.css')) return 'stylesheet';
    if (url.match(/\.(png|jpg|jpeg|gif|svg|webp)$/)) return 'image';
    if (url.includes('/api/')) return 'api';
    return 'other';
  }

  /**
   * 获取内存信息
   */
  private getMemoryInfo() {
    if ('memory' in performance) {
      const memory = (performance as any).memory;
      return {
        usedJSHeapSize: memory.usedJSHeapSize,
        totalJSHeapSize: memory.totalJSHeapSize,
        jsHeapSizeLimit: memory.jsHeapSizeLimit,
      };
    }
    return undefined;
  }

  /**
   * 获取设备信息
   */
  private getDeviceInfo() {
    const connection = (navigator as any).connection || (navigator as any).mozConnection || (navigator as any).webkitConnection;
    
    return {
      userAgent: navigator.userAgent,
      viewport: {
        width: window.innerWidth,
        height: window.innerHeight,
      },
      screen: {
        width: screen.width,
        height: screen.height,
      },
      connection: connection ? {
        effectiveType: connection.effectiveType,
        downlink: connection.downlink,
        rtt: connection.rtt,
      } : undefined,
    };
  }

  /**
   * 获取元素选择器
   */
  private getElementSelector(element: Element): string {
    if (!element) return 'unknown';
    
    if (element.id) return `#${element.id}`;
    if (element.className) return `.${element.className.split(' ')[0]}`;
    return element.tagName.toLowerCase();
  }

  /**
   * 添加到队列
   */
  private addToQueue(metrics: PerformanceMetrics) {
    this.metricsQueue.push(metrics);
    if (this.metricsQueue.length > this.maxQueueSize) {
      this.metricsQueue.shift();
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
   * 上报性能数据
   */
  private async flush() {
    if (!this.isEnabled || (!this.metricsQueue.length && !this.interactionQueue.length)) {
      return;
    }

    const payload = {
      metrics: [...this.metricsQueue],
      interactions: [...this.interactionQueue],
      metadata: {
        timestamp: Date.now(),
        sessionId: this.sessionId,
        userId: this.userId,
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
      this.metricsQueue = [];
      this.interactionQueue = [];
    } catch (error) {
      console.warn('Failed to report performance metrics:', error);
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
   * 手动上报
   */
  async report() {
    await this.flush();
  }

  /**
   * 销毁监控服务
   */
  destroy() {
    if (this.observer) {
      this.observer.disconnect();
    }
    if (this.flushTimer) {
      clearInterval(this.flushTimer);
    }
    this.flush(); // 最后一次上报
  }
}

// 创建全局实例
export const performanceMonitoring = new PerformanceMonitoringService();

// 导出类型和实例
export default PerformanceMonitoringService;
import { ApiError } from './types';

// Error report interface
export interface ErrorReport {
  id: string;
  timestamp: string;
  error: {
    name: string;
    message: string;
    stack?: string;
    code?: string;
  };
  context: {
    url: string;
    userAgent: string;
    userId?: string;
    sessionId: string;
    component?: string;
    action?: string;
    additionalData?: Record<string, any>;
  };
  severity: 'low' | 'medium' | 'high' | 'critical';
  category: 'network' | 'api' | 'validation' | 'system' | 'user';
  fingerprint: string;
  tags: string[];
}

// Error reporting service
class ErrorReportingService {
  private static instance: ErrorReportingService;
  private reports: ErrorReport[] = [];
  private sessionId: string;
  private userId?: string;
  private maxReports = 1000;
  private reportingEndpoint = '/api/errors';

  constructor() {
    this.sessionId = this.generateSessionId();
    this.setupGlobalErrorHandlers();
  }

  static getInstance(): ErrorReportingService {
    if (!ErrorReportingService.instance) {
      ErrorReportingService.instance = new ErrorReportingService();
    }
    return ErrorReportingService.instance;
  }

  // Generate unique session ID
  private generateSessionId(): string {
    return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  // Generate error fingerprint for deduplication
  private generateFingerprint(error: Error | ApiError, context?: string): string {
    const name = 'name' in error ? error.name : 'ApiError';
    const key = `${name}_${error.message}_${context || 'unknown'}`;
    return btoa(key).replace(/[^a-zA-Z0-9]/g, '').substr(0, 32);
  }

  // Setup global error handlers
  private setupGlobalErrorHandlers(): void {
    // Handle unhandled JavaScript errors
    window.addEventListener('error', (event) => {
      this.reportError(event.error, {
        component: 'global',
        action: 'unhandled_error',
        additionalData: {
          filename: event.filename,
          lineno: event.lineno,
          colno: event.colno,
        },
      });
    });

    // Handle unhandled promise rejections
    window.addEventListener('unhandledrejection', (event) => {
      this.reportError(new Error(event.reason), {
        component: 'global',
        action: 'unhandled_rejection',
        additionalData: {
          reason: event.reason,
        },
      });
    });

    // Handle resource loading errors
    window.addEventListener('error', (event) => {
      if (event.target !== window) {
        this.reportError(new Error('Resource loading failed'), {
          component: 'resource',
          action: 'load_error',
          additionalData: {
            element: (event.target as HTMLElement)?.tagName,
            source: (event.target as any)?.src || (event.target as any)?.href,
          },
        });
      }
    }, true);
  }

  // Set user ID for error reports
  setUserId(userId: string): void {
    this.userId = userId;
  }

  // Report an error
  reportError(
    error: Error | ApiError,
    context?: {
      component?: string;
      action?: string;
      additionalData?: Record<string, any>;
    }
  ): string {
    const reportId = `error_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    const report: ErrorReport = {
      id: reportId,
      timestamp: new Date().toISOString(),
      error: {
        name: 'name' in error ? error.name : 'ApiError',
        message: error.message,
        stack: 'stack' in error ? error.stack : undefined,
        code: (error as ApiError).code,
      },
      context: {
        url: window.location.href,
        userAgent: navigator.userAgent,
        userId: this.userId,
        sessionId: this.sessionId,
        component: context?.component,
        action: context?.action,
        additionalData: context?.additionalData,
      },
      severity: this.determineSeverity(error),
      category: this.determineCategory(error),
      fingerprint: this.generateFingerprint(error, context?.component),
      tags: this.generateTags(error, context),
    };

    // Add to local storage
    this.reports.push(report);
    
    // Maintain max reports limit
    if (this.reports.length > this.maxReports) {
      this.reports = this.reports.slice(-this.maxReports);
    }

    // Send to server (async, don't block)
    this.sendReport(report).catch(err => {
      console.warn('Failed to send error report:', err);
    });

    return reportId;
  }

  // Determine error severity
  private determineSeverity(error: Error | ApiError): ErrorReport['severity'] {
    const apiError = error as ApiError;
    
    if (apiError.code?.startsWith('HTTP_5')) return 'critical';
    if (apiError.code?.startsWith('HTTP_4')) return 'medium';
    if (apiError.code === 'TIMEOUT') return 'high';
    if (apiError.code === 'NETWORK_ERROR') return 'high';
    
    const errorName = 'name' in error ? error.name : 'ApiError';
    if (errorName === 'TypeError') return 'medium';
    if (errorName === 'ReferenceError') return 'high';
    if (errorName === 'SyntaxError') return 'critical';
    
    return 'low';
  }

  // Determine error category
  private determineCategory(error: Error | ApiError): ErrorReport['category'] {
    const apiError = error as ApiError;
    
    if (apiError.code) return 'api';
    if (error.message?.includes('network') || error.message?.includes('fetch')) return 'network';
    
    const errorName = 'name' in error ? error.name : 'ApiError';
    if (errorName === 'ValidationError') return 'validation';
    if (errorName === 'TypeError' || errorName === 'ReferenceError') return 'system';
    
    return 'user';
  }

  // Generate tags for error
  private generateTags(
    error: Error | ApiError, 
    context?: { component?: string; action?: string }
  ): string[] {
    const tags: string[] = [];
    
    const errorName = 'name' in error ? error.name : 'ApiError';
    tags.push(`error:${errorName}`);
    
    if (context?.component) {
      tags.push(`component:${context.component}`);
    }
    
    if (context?.action) {
      tags.push(`action:${context.action}`);
    }
    
    const apiError = error as ApiError;
    if (apiError.code) {
      tags.push(`code:${apiError.code}`);
    }
    
    // Browser info
    tags.push(`browser:${this.getBrowserName()}`);
    tags.push(`platform:${navigator.platform}`);
    
    return tags;
  }

  // Get browser name
  private getBrowserName(): string {
    const userAgent = navigator.userAgent;
    
    if (userAgent.includes('Chrome')) return 'chrome';
    if (userAgent.includes('Firefox')) return 'firefox';
    if (userAgent.includes('Safari')) return 'safari';
    if (userAgent.includes('Edge')) return 'edge';
    
    return 'unknown';
  }

  // Send error report to server
  private async sendReport(report: ErrorReport): Promise<void> {
    try {
      await fetch(this.reportingEndpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(report),
      });
    } catch (error) {
      // Store failed reports for retry
      this.storeFailedReport(report);
    }
  }

  // Store failed reports in localStorage for retry
  private storeFailedReport(report: ErrorReport): void {
    try {
      const failedReports = JSON.parse(
        localStorage.getItem('failed_error_reports') || '[]'
      );
      failedReports.push(report);
      
      // Keep only last 50 failed reports
      const limitedReports = failedReports.slice(-50);
      localStorage.setItem('failed_error_reports', JSON.stringify(limitedReports));
    } catch (error) {
      console.warn('Failed to store error report in localStorage:', error);
    }
  }

  // Retry failed reports
  async retryFailedReports(): Promise<void> {
    try {
      const failedReports = JSON.parse(
        localStorage.getItem('failed_error_reports') || '[]'
      );
      
      if (failedReports.length === 0) return;
      
      const retryPromises = failedReports.map((report: ErrorReport) => 
        this.sendReport(report)
      );
      
      await Promise.allSettled(retryPromises);
      
      // Clear failed reports after retry
      localStorage.removeItem('failed_error_reports');
    } catch (error) {
      console.warn('Failed to retry error reports:', error);
    }
  }

  // Get error statistics
  getErrorStats(): {
    total: number;
    bySeverity: Record<string, number>;
    byCategory: Record<string, number>;
    byComponent: Record<string, number>;
    recent: ErrorReport[];
  } {
    const bySeverity: Record<string, number> = {};
    const byCategory: Record<string, number> = {};
    const byComponent: Record<string, number> = {};

    this.reports.forEach(report => {
      bySeverity[report.severity] = (bySeverity[report.severity] || 0) + 1;
      byCategory[report.category] = (byCategory[report.category] || 0) + 1;
      
      if (report.context.component) {
        byComponent[report.context.component] = (byComponent[report.context.component] || 0) + 1;
      }
    });

    return {
      total: this.reports.length,
      bySeverity,
      byCategory,
      byComponent,
      recent: this.reports.slice(-10),
    };
  }

  // Get reports by fingerprint (for deduplication)
  getReportsByFingerprint(fingerprint: string): ErrorReport[] {
    return this.reports.filter(report => report.fingerprint === fingerprint);
  }

  // Clear all reports
  clearReports(): void {
    this.reports = [];
  }

  // Export reports for debugging
  exportReports(): string {
    return JSON.stringify(this.reports, null, 2);
  }

  // Import reports (for testing)
  importReports(reportsJson: string): void {
    try {
      const reports = JSON.parse(reportsJson);
      this.reports = reports;
    } catch (error) {
      console.error('Failed to import reports:', error);
    }
  }
}

// Create singleton instance
export const errorReporting = ErrorReportingService.getInstance();

// React hook for error reporting
export const useErrorReporting = () => {
  const reportError = (
    error: Error | ApiError,
    context?: {
      component?: string;
      action?: string;
      additionalData?: Record<string, any>;
    }
  ) => {
    return errorReporting.reportError(error, context);
  };

  const getStats = () => {
    return errorReporting.getErrorStats();
  };

  const clearReports = () => {
    errorReporting.clearReports();
  };

  const retryFailedReports = () => {
    return errorReporting.retryFailedReports();
  };

  return {
    reportError,
    getStats,
    clearReports,
    retryFailedReports,
  };
};

export default errorReporting;
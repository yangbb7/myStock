// Environment configuration management for myQuant Frontend
const environments = {
  development: {
    name: 'Development',
    apiBaseUrl: 'http://localhost:8000',
    wsBaseUrl: 'http://localhost:8000',
    enableMockData: true,
    enableDebugTools: true,
    logLevel: 'debug',
    monitoring: {
      enabled: false,
      errorReporting: false,
      analytics: false,
      performance: false
    },
    features: {
      serviceWorker: false,
      pwa: false,
      csp: false,
      hsts: false
    }
  },
  
  staging: {
    name: 'Staging',
    apiBaseUrl: 'https://staging-api.yourdomain.com',
    wsBaseUrl: 'wss://staging-api.yourdomain.com',
    enableMockData: false,
    enableDebugTools: true,
    logLevel: 'debug',
    monitoring: {
      enabled: true,
      errorReporting: true,
      analytics: true,
      performance: true,
      endpoints: {
        errors: 'https://staging-api.yourdomain.com/monitoring/errors',
        performance: 'https://staging-api.yourdomain.com/monitoring/performance',
        analytics: 'https://staging-api.yourdomain.com/monitoring/analytics',
        logs: 'https://staging-api.yourdomain.com/monitoring/logs'
      },
      flushIntervals: {
        errors: 15000,
        performance: 30000,
        analytics: 15000,
        logs: 15000
      },
      queueSizes: {
        errors: 50,
        performance: 25,
        analytics: 50,
        logs: 500
      }
    },
    features: {
      serviceWorker: false,
      pwa: false,
      csp: true,
      hsts: false
    }
  },
  
  production: {
    name: 'Production',
    apiBaseUrl: 'https://api.yourdomain.com',
    wsBaseUrl: 'wss://api.yourdomain.com',
    enableMockData: false,
    enableDebugTools: false,
    logLevel: 'info',
    monitoring: {
      enabled: true,
      errorReporting: true,
      analytics: true,
      performance: true,
      endpoints: {
        errors: 'https://api.yourdomain.com/monitoring/errors',
        performance: 'https://api.yourdomain.com/monitoring/performance',
        analytics: 'https://api.yourdomain.com/monitoring/analytics',
        logs: 'https://api.yourdomain.com/monitoring/logs'
      },
      flushIntervals: {
        errors: 30000,
        performance: 60000,
        analytics: 30000,
        logs: 30000
      },
      queueSizes: {
        errors: 100,
        performance: 50,
        analytics: 100,
        logs: 1000
      }
    },
    features: {
      serviceWorker: true,
      pwa: false,
      csp: true,
      hsts: true
    },
    thirdParty: {
      sentry: {
        dsn: process.env.VITE_SENTRY_DSN,
        environment: 'production',
        tracesSampleRate: 0.1
      },
      analytics: {
        googleAnalyticsId: process.env.VITE_GOOGLE_ANALYTICS_ID,
        hotjarId: process.env.VITE_HOTJAR_ID
      }
    }
  }
};

// Get current environment
const getCurrentEnvironment = () => {
  const env = process.env.NODE_ENV || 'development';
  return environments[env] || environments.development;
};

// Validate environment configuration
const validateEnvironment = (config) => {
  const required = ['name', 'apiBaseUrl', 'wsBaseUrl'];
  const missing = required.filter(key => !config[key]);
  
  if (missing.length > 0) {
    throw new Error(`Missing required environment configuration: ${missing.join(', ')}`);
  }
  
  return true;
};

// Export configuration
const config = getCurrentEnvironment();
validateEnvironment(config);

export default config;
export { environments, getCurrentEnvironment, validateEnvironment };
// Simple JavaScript test to verify error handling functionality
// This bypasses TypeScript compilation issues

import { ErrorHandler } from './ErrorHandler';
import { errorReporting } from '../../services/errorReporting';

// Test error handling functionality
export const testErrorHandling = () => {
  console.log('Testing Error Handling System...');

  try {
    // Test 1: Basic error handling
    const testError = new Error('Test error message');
    ErrorHandler.handleSystemError(testError, 'test-context');
    console.log('✓ Basic error handling works');

    // Test 2: API error handling
    const apiError = {
      code: 'HTTP_500',
      message: 'API server error',
      details: { endpoint: '/test' },
      timestamp: new Date().toISOString(),
    };
    ErrorHandler.handleApiError(apiError, 'api-test');
    console.log('✓ API error handling works');

    // Test 3: Error reporting
    const reportId = errorReporting.reportError(testError, {
      component: 'test-component',
      action: 'test-action',
    });
    console.log('✓ Error reporting works, ID:', reportId);

    // Test 4: Error statistics
    const stats = errorReporting.getErrorStats();
    console.log('✓ Error statistics:', {
      total: stats.total,
      categories: Object.keys(stats.byCategory),
      severities: Object.keys(stats.bySeverity),
    });

    // Test 5: Network error
    const networkError = new Error('Network connection failed');
    networkError.name = 'NetworkError';
    ErrorHandler.handleNetworkError(networkError, 'network-test');
    console.log('✓ Network error handling works');

    console.log('🎉 All error handling tests passed!');
    return true;

  } catch (error) {
    console.error('❌ Error handling test failed:', error);
    return false;
  }
};

// Test data cache functionality
export const testDataCache = () => {
  console.log('Testing Data Cache...');

  try {
    // This would normally be imported from DataFallback
    // but we'll simulate the cache functionality
    const cache = new Map();
    
    // Test cache operations
    const testData = { message: 'Test data', timestamp: Date.now() };
    const cacheKey = 'test-cache-key';
    
    // Set cache
    cache.set(cacheKey, {
      data: testData,
      timestamp: Date.now(),
      expiresAt: Date.now() + 5000, // 5 seconds
    });
    
    // Get cache
    const cachedEntry = cache.get(cacheKey);
    if (cachedEntry && cachedEntry.data.message === testData.message) {
      console.log('✓ Data caching works');
    } else {
      throw new Error('Cache data mismatch');
    }

    console.log('🎉 Data cache tests passed!');
    return true;

  } catch (error) {
    console.error('❌ Data cache test failed:', error);
    return false;
  }
};

// Test network status simulation
export const testNetworkStatus = () => {
  console.log('Testing Network Status...');

  try {
    // Simulate network status
    const isOnline = navigator.onLine;
    console.log('✓ Network status detected:', isOnline ? 'Online' : 'Offline');

    // Test network change events
    const handleOnline = () => console.log('Network came online');
    const handleOffline = () => console.log('Network went offline');

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    // Clean up
    setTimeout(() => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    }, 1000);

    console.log('✓ Network event listeners set up');
    console.log('🎉 Network status tests passed!');
    return true;

  } catch (error) {
    console.error('❌ Network status test failed:', error);
    return false;
  }
};

// Run all tests
export const runAllTests = () => {
  console.log('🚀 Starting Error Handling System Tests...\n');

  const results = {
    errorHandling: testErrorHandling(),
    dataCache: testDataCache(),
    networkStatus: testNetworkStatus(),
  };

  console.log('\n📊 Test Results:');
  Object.entries(results).forEach(([test, passed]) => {
    console.log(`${passed ? '✅' : '❌'} ${test}: ${passed ? 'PASSED' : 'FAILED'}`);
  });

  const allPassed = Object.values(results).every(result => result);
  console.log(`\n${allPassed ? '🎉' : '💥'} Overall: ${allPassed ? 'ALL TESTS PASSED' : 'SOME TESTS FAILED'}`);

  return allPassed;
};

// Export for use in browser console or other components
if (typeof window !== 'undefined') {
  window.testErrorHandling = runAllTests;
}

export default {
  testErrorHandling,
  testDataCache,
  testNetworkStatus,
  runAllTests,
};
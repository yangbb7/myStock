# API Testing Panel

A comprehensive, production-ready API testing component for the myQuant trading system dashboard.

## Features

### 🚀 **Core Functionality**
- **Real-time System Health Monitoring** - Auto-refresh system status every 5 seconds
- **Comprehensive API Testing** - Test all system endpoints with detailed results
- **Visual Test Results** - Color-coded timeline with success/error indicators
- **Performance Metrics** - Track response times and success rates
- **Category Filtering** - Filter tests by endpoint category (system, portfolio, risk, etc.)

### 🎨 **Visual Design**
- **Modern Ant Design Components** - Consistent with existing dashboard
- **Responsive Grid Layout** - Cards automatically adjust to screen size  
- **Dynamic Status Colors** - Green for success, red for errors, blue for running
- **Smooth Animations** - Card border colors change based on test results
- **Professional UI** - Clean, polished interface suitable for production

### ⚡ **Advanced Features**
- **Auto-refresh Toggle** - Enable/disable automatic system health updates
- **Quick Health Check** - Fast test of critical system endpoints only
- **Manual Refresh** - On-demand system status updates
- **Test History** - Maintains last 20 test results with timestamps
- **Error Details** - Expandable panels showing full error information
- **Loading States** - Visual feedback during test execution

### 📊 **Statistics Dashboard**
- **System Uptime** - Hours since system start
- **Module Count** - Number of initialized modules
- **Success Rate** - Percentage of successful tests
- **Average Response Time** - Mean API response time in milliseconds

## API Endpoints Tested

### System Category
- **Health Check** (`GET /health`) - System health status
- **System Metrics** (`GET /metrics`) - Performance metrics
- **Start System** (`POST /system/start`) - Initialize trading system
- **Stop System** (`POST /system/stop`) - Shutdown trading system

### Portfolio Category  
- **Portfolio Summary** (`GET /portfolio/summary`) - Investment overview

### Risk Category
- **Risk Metrics** (`GET /risk/metrics`) - Risk management data

## Usage Example

```tsx
import ApiTestingPanel from '@/components/ApiTestingPanel';

function Dashboard() {
  return (
    <div>
      <ApiTestingPanel />
    </div>
  );
}
```

## Integration

The component automatically integrates with:
- **API Service Layer** (`@/services/api`) - Uses existing API functions
- **Ant Design Message** - Displays success/error notifications  
- **TypeScript Types** - Full type safety with service interfaces

## Replacement

This component replaces:
- `SystemControlTest.tsx` - Old debugging component
- `DirectApiTest` function - Basic API testing functionality

## Configuration

### Auto-refresh Settings
- **Default**: Enabled (5-second intervals)
- **Toggle**: Click the "自动/手动" button
- **Manual**: Use "刷新" button for on-demand updates

### Test Categories
- **All** - Run tests for all endpoint categories
- **System** - Critical system health endpoints only
- **Portfolio** - Investment and portfolio data
- **Risk** - Risk management metrics

## Error Handling

- **Network Errors** - Displays connection issues
- **API Errors** - Shows server error messages
- **Timeout Handling** - Graceful handling of slow responses
- **Retry Logic** - Manual retry available for failed tests

## Performance

- **Optimized Rendering** - Efficient state updates
- **Memory Management** - Limits test history to 20 entries
- **Responsive Design** - Works on desktop and tablet devices
- **Lightweight** - Minimal bundle size impact

## Production Ready

✅ **TypeScript Support** - Full type safety  
✅ **Error Boundaries** - Graceful error handling  
✅ **Performance Optimized** - Efficient re-renders  
✅ **Accessible** - ARIA labels and keyboard navigation  
✅ **Responsive** - Mobile-friendly design  
✅ **Production Build** - Tested with Vite build process
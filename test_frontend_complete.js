#!/usr/bin/env node
/**
 * 完整的前端API调用模拟测试
 * 模拟axios请求拦截器和前端组件的完整调用流程
 */

// 使用内置的fetch模拟axios行为
const fetch = require('node-fetch');

const API_BASE_URL = 'http://localhost:8000';

// 模拟前端请求拦截器的逻辑
function mockRequestInterceptor(url) {
  console.log('[Mock Request] Intercepting request:', url);
  
  // 公共API endpoints不需要认证
  const publicEndpoints = ['/health', '/data/realtime', '/data/market', '/data/symbols'];
  const isPublicEndpoint = publicEndpoints.some(endpoint => url.includes(endpoint));
  
  if (isPublicEndpoint) {
    console.log('[Mock Request] Public endpoint - skipping auth:', url);
    return { requiresAuth: false };
  }
  
  return { requiresAuth: true };
}

// 模拟前端API服务的getRealTimePrice方法
async function mockGetRealTimePrice(symbol) {
  const url = `${API_BASE_URL}/data/realtime/${symbol}`;
  
  // 应用请求拦截器逻辑
  const interceptResult = mockRequestInterceptor(url);
  
  try {
    console.log(`[Mock API] Calling getRealTimePrice for ${symbol}`);
    console.log(`[Mock API] Auth required: ${interceptResult.requiresAuth}`);
    
    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      timeout: 5000
    });
    
    console.log(`[Mock API] Response status: ${response.status}`);
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    const data = await response.json();
    console.log(`[Mock API] Response data:`, JSON.stringify(data, null, 2));
    
    return data;
    
  } catch (error) {
    console.error(`[Mock API] Error for ${symbol}:`, error.message);
    throw error;
  }
}

// 模拟RealDataValidator组件的testSingleStock方法
async function mockTestSingleStock(symbol) {
  const startTime = Date.now();
  
  try {
    console.log(`\n🔍 [Mock Component] Testing ${symbol}...`);
    
    // 模拟组件状态更新
    console.log(`[Mock Component] Setting ${symbol} status to 'loading'`);
    
    // 调用API
    const response = await mockGetRealTimePrice(symbol);
    const latency = Date.now() - startTime;
    
    console.log(`[Mock Component] API call completed in ${latency}ms`);
    
    // 模拟前端组件的响应处理逻辑
    if (response && response.success !== false) {
      const stockData = response;
      let current_price = 0;
      
      console.log(`[Mock Component] Processing response for ${symbol}`);
      console.log(`[Mock Component] Response structure:`, {
        hasData: !!stockData.data,
        hasCurrentPrice: !!(stockData.data && stockData.data.current_price),
        success: stockData.success
      });
      
      // 直接从实时API响应中获取价格
      if (stockData.data && stockData.data.current_price) {
        current_price = stockData.data.current_price;
        console.log(`[Mock Component] ✅ ${symbol}: ¥${current_price.toFixed(2)} (${latency}ms)`);
        
        return {
          name: symbol,
          status: 'success',
          data: {
            symbol,
            name: stockData.data.name || symbol,
            current_price,
            timestamp: new Date().toISOString(),
            source: 'EastMoney API'
          },
          latency
        };
      } else {
        console.log(`[Mock Component] ❌ ${symbol}: 价格数据缺失`);
        console.log(`[Mock Component] Data structure:`, stockData);
        return {
          name: symbol,
          status: 'error',
          error: 'Missing price data',
          latency
        };
      }
    } else {
      const errorMsg = response?.message || 'No data received';
      console.log(`[Mock Component] ❌ ${symbol}: ${errorMsg}`);
      return {
        name: symbol,
        status: 'error',
        error: errorMsg,
        latency
      };
    }
    
  } catch (error) {
    const latency = Date.now() - startTime;
    console.log(`[Mock Component] ❌ ${symbol}: Exception - ${error.message} (${latency}ms)`);
    
    return {
      name: symbol,
      status: 'error',
      error: error.message,
      latency
    };
  }
}

async function main() {
  console.log('=' .repeat(80));
  console.log('🔧 完整前端API调用模拟测试');
  console.log('=' .repeat(80));
  console.log(`测试时间: ${new Date().toLocaleString()}`);
  console.log(`模拟环境: 前端组件 -> API服务 -> 后端`);
  console.log();
  
  const testSymbols = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH'];
  const results = [];
  
  // 逐个测试股票
  for (const symbol of testSymbols) {
    const result = await mockTestSingleStock(symbol);
    results.push(result);
  }
  
  console.log('\n📋 测试总结');
  console.log('=' .repeat(60));
  
  const successCount = results.filter(r => r.status === 'success').length;
  const errorCount = results.filter(r => r.status === 'error').length;
  
  console.log(`✅ 成功: ${successCount}/${results.length}`);
  console.log(`❌ 失败: ${errorCount}/${results.length}`);
  console.log();
  
  if (successCount === results.length) {
    console.log('🎉 所有测试通过！前端应该能正常显示股票价格');
    console.log('   如果前端仍显示异常，可能是以下原因：');
    console.log('   1. 浏览器缓存问题 - 请硬刷新页面 (Ctrl+Shift+R)');
    console.log('   2. WebSocket连接问题 - 检查WebSocket状态');
    console.log('   3. 组件状态更新问题 - 检查React状态管理');
  } else {
    console.log('⚠️ 部分测试失败，分析失败原因：');
    results.filter(r => r.status === 'error').forEach(result => {
      console.log(`   ${result.name}: ${result.error}`);
    });
  }
  
  console.log('\n🔍 下一步调试建议：');
  console.log('   1. 打开浏览器开发者工具 (F12)');
  console.log('   2. 查看Console标签页的错误信息');
  console.log('   3. 查看Network标签页的API请求详情');
  console.log('   4. 在RealDataValidator组件中添加console.log调试');
}

// 检查node-fetch是否可用
try {
  main().catch(error => {
    console.error('测试执行失败:', error.message);
    console.log('\n💡 如果遇到模块找不到的错误，请运行:');
    console.log('   npm install node-fetch');
    console.log('\n或者直接在浏览器中访问以下URL测试:');
    console.log('   http://localhost:8000/data/realtime/000001.SZ');
  });
} catch (error) {
  console.error('模块加载失败:', error.message);
  console.log('\n使用curl命令测试API:');
  console.log('   curl http://localhost:8000/data/realtime/000001.SZ');
}
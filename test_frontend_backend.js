#!/usr/bin/env node
/**
 * 测试前端到后端的API连接
 * 模拟前端API调用并验证返回数据
 */

const axios = require('axios');

// 后端API基础URL
const API_BASE_URL = 'http://localhost:8000';

// 测试股票列表
const testSymbols = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH'];

// 模拟前端API请求
async function testRealTimePrice(symbol) {
  try {
    console.log(`🔍 测试股票 ${symbol} 实时价格...`);
    
    const response = await axios.get(`${API_BASE_URL}/data/realtime/${symbol}`, {
      timeout: 5000,
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      }
    });
    
    const data = response.data;
    
    if (data && data.success && data.data) {
      const stockData = data.data;
      console.log(`✅ ${symbol}: ${stockData.name} - ¥${stockData.current_price} (${stockData.source})`);
      console.log(`   时间: ${stockData.timestamp}`);
      return {
        symbol,
        success: true,
        price: stockData.current_price,
        name: stockData.name,
        source: stockData.source
      };
    } else {
      console.log(`❌ ${symbol}: API响应格式异常`);
      console.log('   响应数据:', JSON.stringify(data, null, 2));
      return { symbol, success: false, error: 'Invalid API response format' };
    }
    
  } catch (error) {
    console.log(`❌ ${symbol}: 请求失败`);
    if (error.response) {
      console.log(`   状态码: ${error.response.status}`);
      console.log(`   错误信息: ${error.response.data?.message || error.message}`);
    } else {
      console.log(`   网络错误: ${error.message}`);
    }
    return { 
      symbol, 
      success: false, 
      error: error.response?.data?.message || error.message 
    };
  }
}

async function testHealthCheck() {
  try {
    console.log('🏥 测试健康检查API...');
    const response = await axios.get(`${API_BASE_URL}/health`, { timeout: 3000 });
    
    if (response.data && response.data.success) {
      console.log('✅ 后端服务健康状态正常');
      return true;
    } else {
      console.log('⚠️ 后端服务响应异常');
      return false;
    }
  } catch (error) {
    console.log('❌ 后端服务连接失败:', error.message);
    return false;
  }
}

async function main() {
  console.log('=' * 60);
  console.log('🔧 前端到后端API连接测试');
  console.log('=' * 60);
  console.log(`测试时间: ${new Date().toLocaleString()}`);
  console.log(`后端地址: ${API_BASE_URL}`);
  console.log();
  
  // 1. 测试健康检查
  const healthOk = await testHealthCheck();
  if (!healthOk) {
    console.log('\n❌ 后端服务不可用，测试终止');
    process.exit(1);
  }
  
  console.log();
  
  // 2. 测试实时价格API
  console.log('📊 测试实时价格API...');
  console.log('-' * 40);
  
  const results = [];
  for (const symbol of testSymbols) {
    const result = await testRealTimePrice(symbol);
    results.push(result);
    console.log(); // 空行分隔
  }
  
  // 3. 生成测试报告
  console.log('📋 测试报告');
  console.log('=' * 40);
  
  const successCount = results.filter(r => r.success).length;
  const totalCount = results.length;
  const successRate = (successCount / totalCount * 100).toFixed(1);
  
  console.log(`✅ 成功: ${successCount}/${totalCount} (${successRate}%)`);
  console.log(`❌ 失败: ${totalCount - successCount}/${totalCount}`);
  
  if (successRate >= 80) {
    console.log('\n🎉 前端API连接测试通过！');
    console.log('   ✅ 后端服务正常运行');
    console.log('   ✅ 实时价格API响应正常');
    console.log('   ✅ 数据格式符合前端预期');
  } else {
    console.log('\n⚠️ 前端API连接存在问题');
    console.log('   请检查后端服务状态或API实现');
  }
  
  // 失败详情
  const failedResults = results.filter(r => !r.success);
  if (failedResults.length > 0) {
    console.log('\n📝 失败详情:');
    failedResults.forEach(result => {
      console.log(`   ${result.symbol}: ${result.error}`);
    });
  }
  
  console.log('\n🔗 下一步操作:');
  console.log('   1. 访问 http://localhost:3001/dashboard');
  console.log('   2. 查看"实时数据验证器"组件');
  console.log('   3. 点击"单次测试"按钮验证前端功能');
}

// 运行测试
main().catch(error => {
  console.error('测试脚本执行失败:', error);
  process.exit(1);
});
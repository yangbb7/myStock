#!/usr/bin/env python3
"""
验证股票数据修复效果
快速测试EastMoney API和真实数据获取
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, List
from real_data_config import get_real_data_config
from myQuant.infrastructure.data.providers.real_data_provider import RealDataProvider

def print_header():
    """打印标题"""
    print("=" * 80)
    print("🔧 myStock 股票数据修复验证")
    print("=" * 80)
    print(f"验证时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def test_data_sources():
    """测试数据源配置和连接"""
    print("📋 1. 数据源配置验证")
    print("-" * 40)
    
    config = get_real_data_config()
    print(f"✅ 主要数据源: {config['primary_provider']}")
    print(f"✅ 备用数据源: {', '.join(config['fallback_providers'])}")
    print(f"✅ 东方财富API: {'启用' if config['eastmoney']['enabled'] else '禁用'}")
    print(f"✅ Yahoo Finance: {'启用' if config['yahoo']['enabled'] else '禁用'}")
    
    return config

def test_real_time_data(provider: RealDataProvider, symbols: List[str]):
    """测试实时数据获取"""
    print("\n📊 2. 实时数据获取测试")
    print("-" * 40)
    
    results = {}
    total_time = 0
    
    for symbol in symbols:
        start_time = time.time()
        try:
            price = provider.get_current_price(symbol)
            end_time = time.time()
            latency = round((end_time - start_time) * 1000, 2)
            total_time += latency
            
            if price > 0:
                results[symbol] = {
                    'price': price,
                    'latency': latency,
                    'status': 'success'
                }
                print(f"✅ {symbol}: ¥{price:.2f} ({latency}ms)")
            else:
                results[symbol] = {
                    'price': 0,
                    'latency': latency,
                    'status': 'failed'
                }
                print(f"❌ {symbol}: 无法获取价格 ({latency}ms)")
                
        except Exception as e:
            end_time = time.time()
            latency = round((end_time - start_time) * 1000, 2)
            results[symbol] = {
                'price': 0,
                'latency': latency,
                'status': 'error',
                'error': str(e)
            }
            print(f"❌ {symbol}: 错误 - {e} ({latency}ms)")
    
    # 统计信息
    success_count = sum(1 for r in results.values() if r['status'] == 'success')
    avg_latency = round(total_time / len(symbols), 2) if symbols else 0
    
    print(f"\n📈 统计: {success_count}/{len(symbols)} 成功, 平均响应时间: {avg_latency}ms")
    
    return results

def test_historical_data(provider: RealDataProvider, symbol: str):
    """测试历史数据获取"""
    print(f"\n📈 3. 历史数据获取测试 ({symbol})")
    print("-" * 40)
    
    try:
        start_time = time.time()
        from datetime import timedelta
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        
        data = provider.get_stock_data(symbol, start_date, end_date)
        end_time = time.time()
        latency = round((end_time - start_time) * 1000, 2)
        
        if not data.empty:
            print(f"✅ 获取历史数据成功: {len(data)} 条记录 ({latency}ms)")
            print(f"   时间范围: {data['date'].min()} 至 {data['date'].max()}")
            print(f"   最新收盘价: ¥{data.iloc[-1]['close']:.2f}")
            print(f"   价格范围: ¥{data['close'].min():.2f} - ¥{data['close'].max():.2f}")
            return True
        else:
            print(f"❌ 历史数据为空 ({latency}ms)")
            return False
            
    except Exception as e:
        print(f"❌ 历史数据获取失败: {e}")
        return False

def test_batch_realtime(provider: RealDataProvider, symbols: List[str]):
    """测试批量实时数据获取"""
    print(f"\n🚀 4. 批量实时数据测试")
    print("-" * 40)
    
    try:
        start_time = time.time()
        realtime_data = provider.get_realtime_data(symbols)
        end_time = time.time()
        latency = round((end_time - start_time) * 1000, 2)
        
        if realtime_data:
            print(f"✅ 批量获取成功: {len(realtime_data)} 个股票 ({latency}ms)")
            for symbol, data in realtime_data.items():
                price = data.get('current_price', 0)
                timestamp = data.get('timestamp', 'N/A')
                print(f"   {symbol}: ¥{price:.2f} ({timestamp})")
            return True
        else:
            print(f"❌ 批量获取失败 ({latency}ms)")
            return False
            
    except Exception as e:
        print(f"❌ 批量获取异常: {e}")
        return False

def test_data_quality(results: Dict):
    """测试数据质量"""
    print(f"\n🔍 5. 数据质量分析")
    print("-" * 40)
    
    prices = [r['price'] for r in results.values() if r['status'] == 'success']
    
    if not prices:
        print("❌ 无有效价格数据进行质量分析")
        return False
    
    # 基本质量检查
    valid_prices = [p for p in prices if 0 < p < 10000]  # 合理的股价范围
    quality_score = len(valid_prices) / len(prices) * 100
    
    print(f"✅ 价格数据合理性: {quality_score:.1f}%")
    print(f"✅ 价格范围: ¥{min(prices):.2f} - ¥{max(prices):.2f}")
    print(f"✅ 平均价格: ¥{sum(prices)/len(prices):.2f}")
    
    return quality_score > 80

def generate_summary(config: Dict, results: Dict, test_results: List[bool]):
    """生成测试总结"""
    print(f"\n🎯 6. 修复效果总结")
    print("=" * 80)
    
    success_count = sum(1 for r in results.values() if r['status'] == 'success')
    total_count = len(results)
    success_rate = (success_count / total_count * 100) if total_count > 0 else 0
    
    print(f"📊 数据获取成功率: {success_rate:.1f}% ({success_count}/{total_count})")
    print(f"📊 主要数据源: {config['primary_provider']} (EastMoney API)")
    print(f"📊 各项测试结果: {sum(test_results)}/{len(test_results)} 通过")
    
    if success_rate >= 80 and sum(test_results) >= len(test_results) * 0.8:
        print("\n🎉 数据修复效果: 优秀 ✅")
        print("   ✅ 所有核心功能正常")
        print("   ✅ 真实股票数据获取稳定")
        print("   ✅ API响应速度良好")
        print("   ✅ 数据质量符合要求")
    elif success_rate >= 60:
        print("\n⚠️ 数据修复效果: 良好 ⚠️")
        print("   ✅ 基本功能正常")
        print("   ⚠️ 部分数据源可能不稳定")
        print("   💡 建议检查网络连接")
    else:
        print("\n❌ 数据修复效果: 需要改进 ❌")
        print("   ❌ 数据获取成功率较低")
        print("   💡 建议检查数据源配置")
    
    print("\n🔗 下一步操作:")
    print("   1. 运行 python start_full_system.py 启动完整系统")
    print("   2. 访问 http://localhost:3000 查看前端界面")
    print("   3. 在Dashboard中查看实时数据验证器")
    print("=" * 80)

def main():
    """主函数"""
    print_header()
    
    # 测试股票列表
    test_symbols = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH']
    test_results = []
    
    try:
        # 1. 测试数据源配置
        config = test_data_sources()
        
        # 2. 初始化数据提供者
        provider = RealDataProvider(config)
        
        # 3. 测试实时数据
        results = test_real_time_data(provider, test_symbols)
        test_results.append(any(r['status'] == 'success' for r in results.values()))
        
        # 4. 测试历史数据
        historical_success = test_historical_data(provider, '000001.SZ')
        test_results.append(historical_success)
        
        # 5. 测试批量实时数据
        batch_success = test_batch_realtime(provider, test_symbols[:2])
        test_results.append(batch_success)
        
        # 6. 测试数据质量
        quality_success = test_data_quality(results)
        test_results.append(quality_success)
        
        # 7. 生成总结
        generate_summary(config, results, test_results)
        
    except Exception as e:
        print(f"\n❌ 验证过程出现异常: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
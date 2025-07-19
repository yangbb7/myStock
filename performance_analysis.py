#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
性能分析脚本 - 分析myStock系统的性能瓶颈
"""

import sys
import time
import cProfile
import pstats
import io
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd
import numpy as np

sys.path.insert(0, 'myQuant')

from myQuant import create_default_config, MAStrategy
from myQuant.core.trading_system import TradingSystem

def profile_system_initialization():
    """分析系统初始化性能"""
    print("=== 分析系统初始化性能 ===")
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    # 初始化系统
    config = create_default_config()
    trading_system = TradingSystem(config)
    
    # 添加策略
    strategy = MAStrategy(
        name="PerfTestStrategy",
        symbols=["000001.SZ", "000002.SZ"],
        params={"short_window": 5, "long_window": 20}
    )
    trading_system.add_strategy(strategy)
    
    profiler.disable()
    
    # 分析结果
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    
    print("初始化性能分析 (前20个最耗时函数):")
    print(s.getvalue())
    
    return trading_system

def profile_tick_processing(trading_system: TradingSystem, num_ticks: int = 1000):
    """分析tick处理性能"""
    print(f"\n=== 分析tick处理性能 (处理{num_ticks}个tick) ===")
    
    # 准备测试数据
    test_ticks = []
    symbols = ["000001.SZ", "000002.SZ"]
    
    for i in range(num_ticks):
        for symbol in symbols:
            test_ticks.append({
                "datetime": datetime.now(),
                "symbol": symbol,
                "close": 10 + (i % 100) * 0.1,  # 模拟价格波动
                "volume": 100000 + (i % 50) * 1000,
                "open": 10 + (i % 100) * 0.1,
                "high": 10 + (i % 100) * 0.1 + 0.5,
                "low": 10 + (i % 100) * 0.1 - 0.5,
            })
    
    # 性能测试
    profiler = cProfile.Profile()
    start_time = time.time()
    
    profiler.enable()
    
    for tick in test_ticks:
        trading_system.process_market_tick(tick)
    
    profiler.disable()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"总处理时间: {total_time:.4f}秒")
    print(f"平均每tick处理时间: {total_time/len(test_ticks)*1000:.2f}毫秒")
    print(f"每秒处理tick数: {len(test_ticks)/total_time:.0f}")
    
    # 分析结果
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    
    print("\ntick处理性能分析 (前20个最耗时函数):")
    print(s.getvalue())
    
    return {
        "total_time": total_time,
        "avg_tick_time": total_time/len(test_ticks)*1000,
        "ticks_per_second": len(test_ticks)/total_time
    }

def profile_memory_usage(trading_system: TradingSystem):
    """分析内存使用"""
    print("\n=== 分析内存使用 ===")
    
    try:
        import psutil
        process = psutil.Process()
        
        # 获取初始内存
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"初始内存使用: {initial_memory:.2f} MB")
        
        # 处理大量数据后的内存
        for i in range(10000):
            tick = {
                "datetime": datetime.now(),
                "symbol": "000001.SZ",
                "close": 10 + i * 0.001,
                "volume": 100000,
                "open": 10 + i * 0.001,
                "high": 10 + i * 0.001 + 0.1,
                "low": 10 + i * 0.001 - 0.1,
            }
            trading_system.process_market_tick(tick)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"处理10000个tick后内存使用: {final_memory:.2f} MB")
        print(f"内存增长: {final_memory - initial_memory:.2f} MB")
        
        return {
            "initial_memory": initial_memory,
            "final_memory": final_memory,
            "memory_growth": final_memory - initial_memory
        }
        
    except ImportError:
        print("psutil未安装，无法分析内存使用")
        return None

def analyze_bottlenecks(trading_system: TradingSystem):
    """分析系统瓶颈"""
    print("\n=== 系统瓶颈分析 ===")
    
    # 1. 组件性能测试
    components = {
        "data_manager": trading_system.data_manager,
        "strategy_engine": trading_system.strategy_engine,
        "portfolio_manager": trading_system.portfolio_manager,
        "order_manager": trading_system.order_manager,
        "execution_engine": trading_system.execution_engine,
        "risk_manager": trading_system.risk_manager
    }
    
    # 测试每个组件的响应时间
    test_tick = {
        "datetime": datetime.now(),
        "symbol": "000001.SZ",
        "close": 15.0,
        "volume": 100000,
        "open": 15.0,
        "high": 15.5,
        "low": 14.5,
    }
    
    component_times = {}
    
    # 测试数据管理器
    start = time.time()
    for _ in range(1000):
        trading_system.data_manager.process_bar(test_tick)
    component_times["data_manager"] = (time.time() - start) / 1000 * 1000  # ms
    
    # 测试策略引擎
    start = time.time()
    for _ in range(1000):
        signals = trading_system.strategy_engine.process_bar_data(test_tick)
    component_times["strategy_engine"] = (time.time() - start) / 1000 * 1000  # ms
    
    # 测试投资组合管理器
    start = time.time()
    for _ in range(1000):
        trading_system.portfolio_manager.calculate_total_value()
    component_times["portfolio_manager"] = (time.time() - start) / 1000 * 1000  # ms
    
    print("组件平均响应时间 (毫秒):")
    for component, time_ms in sorted(component_times.items(), key=lambda x: x[1], reverse=True):
        print(f"  {component}: {time_ms:.3f}ms")
    
    return component_times

def generate_performance_report(results: Dict[str, Any]):
    """生成性能报告"""
    print("\n" + "="*60)
    print("myStock 性能分析报告")
    print("="*60)
    
    if "tick_performance" in results:
        perf = results["tick_performance"]
        print(f"\n📊 Tick处理性能:")
        print(f"  平均处理时间: {perf['avg_tick_time']:.2f}ms")
        print(f"  吞吐量: {perf['ticks_per_second']:.0f} ticks/秒")
        
        # 性能评级
        if perf['avg_tick_time'] < 1.0:
            print("  性能评级: 优秀 ✅")
        elif perf['avg_tick_time'] < 5.0:
            print("  性能评级: 良好 ✅")
        elif perf['avg_tick_time'] < 10.0:
            print("  性能评级: 一般 ⚠️")
        else:
            print("  性能评级: 需要优化 ❌")
    
    if "memory_usage" in results and results["memory_usage"]:
        mem = results["memory_usage"]
        print(f"\n💾 内存使用:")
        print(f"  初始内存: {mem['initial_memory']:.2f} MB")
        print(f"  内存增长: {mem['memory_growth']:.2f} MB")
        
        if mem['memory_growth'] < 10:
            print("  内存管理: 优秀 ✅")
        elif mem['memory_growth'] < 50:
            print("  内存管理: 良好 ✅")
        elif mem['memory_growth'] < 100:
            print("  内存管理: 一般 ⚠️")
        else:
            print("  内存管理: 需要优化 ❌")
    
    if "component_times" in results:
        times = results["component_times"]
        print(f"\n⚙️  组件性能分析:")
        slowest = max(times.items(), key=lambda x: x[1])
        print(f"  最慢组件: {slowest[0]} ({slowest[1]:.3f}ms)")
        
        # 优化建议
        print(f"\n🔧 优化建议:")
        if slowest[1] > 5.0:
            print(f"  - 优化 {slowest[0]} 组件性能")
        if results.get("tick_performance", {}).get("avg_tick_time", 0) > 5.0:
            print("  - 考虑使用多线程处理tick数据")
        if results.get("memory_usage", {}).get("memory_growth", 0) > 50:
            print("  - 实现数据清理机制，定期清理历史数据")
            print("  - 考虑使用内存池或对象池")

def main():
    """主函数"""
    print("🚀 myStock 性能分析工具")
    print("=" * 60)
    
    results = {}
    
    # 1. 系统初始化性能
    trading_system = profile_system_initialization()
    
    # 2. Tick处理性能
    tick_perf = profile_tick_processing(trading_system, num_ticks=1000)
    results["tick_performance"] = tick_perf
    
    # 3. 内存使用分析
    memory_usage = profile_memory_usage(trading_system)
    results["memory_usage"] = memory_usage
    
    # 4. 系统瓶颈分析
    component_times = analyze_bottlenecks(trading_system)
    results["component_times"] = component_times
    
    # 5. 生成性能报告
    generate_performance_report(results)
    
    # 6. 保存结果
    import json
    with open('performance_report.json', 'w', encoding='utf-8') as f:
        # 转换datetime对象为字符串
        json_results = {}
        for k, v in results.items():
            if isinstance(v, dict):
                json_results[k] = {str(k2): str(v2) for k2, v2 in v.items()}
            else:
                json_results[k] = str(v)
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 详细报告已保存到 performance_report.json")

if __name__ == "__main__":
    main()
"""
股票活跃度分析演示
演示如何使用活跃度分析器筛选活跃股票
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 直接导入活跃度分析器
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'myQuant', 'core', 'analytics'))
from activity_analyzer import ActivityAnalyzer


def generate_sample_stock_data(symbol: str, days: int = 100, 
                             base_price: float = 50.0, 
                             activity_level: str = "medium") -> pd.DataFrame:
    """
    生成样本股票数据，模拟不同活跃度
    
    Args:
        symbol: 股票代码
        days: 天数
        base_price: 基础价格
        activity_level: 活跃度级别 ("high", "medium", "low")
    """
    np.random.seed(hash(symbol) % 1000)  # 基于symbol的固定种子
    
    # 根据活跃度级别设置参数
    if activity_level == "high":
        volatility = 0.035      # 高波动
        volume_base = 5000000   # 高成交量
        volume_var = 0.8        # 高成交量变化
    elif activity_level == "medium":
        volatility = 0.025      # 中等波动
        volume_base = 2000000   # 中等成交量
        volume_var = 0.5        # 中等成交量变化
    else:  # low
        volatility = 0.015      # 低波动
        volume_base = 500000    # 低成交量
        volume_var = 0.3        # 低成交量变化
    
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=days),
        end=datetime.now(),
        freq='D'
    )
    
    data = []
    current_price = base_price
    
    for date in dates:
        # 价格变化
        daily_return = np.random.normal(0, volatility)
        current_price = current_price * (1 + daily_return)
        current_price = max(0.1, current_price)  # 避免负价格
        
        # OHLC数据
        open_price = current_price * (1 + np.random.normal(0, 0.005))
        high_price = current_price * (1 + abs(np.random.normal(0, 0.01)))
        low_price = current_price * (1 - abs(np.random.normal(0, 0.01)))
        close_price = current_price
        
        # 成交量
        volume = int(volume_base * (1 + np.random.normal(0, volume_var)))
        volume = max(1000, volume)  # 最小成交量
        
        data.append({
            'date': date,
            'open': round(open_price, 2),
            'high': round(max(open_price, high_price, close_price), 2),
            'low': round(min(open_price, low_price, close_price), 2),
            'close': round(close_price, 2),
            'volume': volume,
            'symbol': symbol
        })
    
    return pd.DataFrame(data)


def main():
    """主函数"""
    print("🎯 股票活跃度分析演示")
    print("=" * 50)
    
    # 创建活跃度分析器
    analyzer = ActivityAnalyzer()
    
    # 生成不同活跃度的样本股票数据
    sample_stocks = {
        "000001.SZ": ("平安银行", "high"),
        "000002.SZ": ("万科A", "medium"),
        "600000.SH": ("浦发银行", "medium"),
        "600036.SH": ("招商银行", "high"),
        "000858.SZ": ("五粮液", "low"),
        "600519.SH": ("贵州茅台", "high"),
        "002415.SZ": ("海康威视", "medium"),
        "000568.SZ": ("泸州老窖", "low"),
    }
    
    print("\n📊 生成样本数据并计算活跃度...")
    stocks_data = {}
    activity_results = []
    
    for symbol, (name, activity_level) in sample_stocks.items():
        # 生成样本数据
        data = generate_sample_stock_data(
            symbol, 
            days=100, 
            base_price=np.random.uniform(20, 100),
            activity_level=activity_level
        )
        stocks_data[symbol] = data
        
        # 计算活跃度
        score = analyzer.calculate_activity_score(data)
        activity_results.append((symbol, name, score, activity_level))
        
        print(f"  {symbol} ({name}): {score:.1f}分 [预期: {activity_level}]")
    
    # 按活跃度排序
    activity_results.sort(key=lambda x: x[2], reverse=True)
    
    print("\n🏆 活跃度排行榜:")
    print("-" * 70)
    print(f"{'排名':<4} {'股票代码':<10} {'股票名称':<8} {'活跃度评分':<10} {'等级':<4} {'预期级别':<8}")
    print("-" * 70)
    
    for i, (symbol, name, score, expected_level) in enumerate(activity_results, 1):
        grade = analyzer._get_activity_grade(score)
        print(f"{i:<4} {symbol:<10} {name:<8} {score:<10.1f} {grade:<4} {expected_level:<8}")
    
    # 筛选高活跃度股票
    print("\n🔍 筛选高活跃度股票 (评分 >= 60):")
    print("-" * 50)
    
    active_stocks = analyzer.filter_active_stocks(stocks_data, min_activity_score=60)
    
    if active_stocks:
        for symbol, score in active_stocks:
            name = next(info[0] for sym, info in sample_stocks.items() if sym == symbol)
            print(f"  ✅ {symbol} ({name}): {score:.1f}分")
    else:
        print("  ❌ 没有找到符合条件的高活跃度股票")
    
    # 生成详细报告
    print("\n📋 详细活跃度报告:")
    print("=" * 80)
    
    # 选择前3只股票生成详细报告
    for symbol, _, _, _ in activity_results[:3]:
        name = sample_stocks[symbol][0]
        data = stocks_data[symbol]
        
        report = analyzer.get_activity_report(symbol, data)
        
        print(f"\n📈 {symbol} ({name}) - 活跃度等级: {report['grade']}")
        print(f"   总评分: {report['total_score']}")
        print(f"   子评分:")
        for key, value in report['sub_scores'].items():
            print(f"     {key}: {value}")
        print(f"   关键指标:")
        for key, value in report['metrics'].items():
            if 'amount' in key or 'volume' in key:
                print(f"     {key}: {value:,}")
            else:
                print(f"     {key}: {value}")
    
    # 显示活跃度标准
    print("\n📖 活跃度评级标准:")
    print("-" * 60)
    
    criteria = analyzer.get_activity_criteria()
    for level, info in criteria.items():
        print(f"\n{info['description']} (评分 >= {info['min_score']}):")
        for criterion in info['criteria']:
            print(f"  - {criterion}")
    
    print(f"\n✅ 活跃度分析完成！")
    print(f"💡 建议关注活跃度评分 >= 60 的股票进行量化交易")


if __name__ == "__main__":
    main()
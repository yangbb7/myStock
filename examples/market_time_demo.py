#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
市场时间管理演示脚本
"""

import sys
sys.path.insert(0, 'myQuant')

from datetime import datetime, timedelta
from myQuant.core.market_time_manager import MarketTimeManager, MarketStatus

def demo_market_time_management():
    """演示市场时间管理功能"""
    print("🕐 市场时间管理演示")
    print("="*60)
    
    # 创建市场时间管理器
    market_manager = MarketTimeManager("A_SHARE")
    
    # 当前时间状态
    current_time = datetime.now()
    print(f"📅 当前时间: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    status = market_manager.get_market_status(current_time)
    print(f"📊 市场状态: {status.value}")
    print(f"🔓 是否开盘: {'是' if market_manager.is_market_open(current_time) else '否'}")
    print(f"💰 允许交易: {'是' if market_manager.is_trading_allowed(current_time) else '否'}")
    
    # 获取详细时段信息
    session_info = market_manager.get_current_session_info(current_time)
    print(f"\n📋 当前时段信息:")
    print(f"  交易日: {'是' if session_info['is_trading_day'] else '否'}")
    print(f"  市场状态: {session_info['market_status']}")
    print(f"  允许交易: {'是' if session_info['trading_allowed'] else '否'}")
    
    if session_info.get('next_trading_time'):
        print(f"  下次交易时间: {session_info['next_trading_time']}")
    
    # 演示不同时间的状态
    print(f"\n🕐 各时段状态演示:")
    
    test_times = [
        "09:15",  # 集合竞价
        "09:30",  # 开盘
        "10:30",  # 上午交易
        "11:30",  # 上午收盘
        "12:00",  # 午休
        "13:00",  # 下午开盘
        "14:30",  # 下午交易
        "15:00",  # 收盘
        "20:00",  # 晚上
    ]
    
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    for time_str in test_times:
        hour, minute = map(int, time_str.split(':'))
        test_time = today.replace(hour=hour, minute=minute)
        
        status = market_manager.get_market_status(test_time)
        trading_allowed = market_manager.is_trading_allowed(test_time)
        
        status_icon = "🟢" if status == MarketStatus.OPEN else "🟡" if status == MarketStatus.PRE_MARKET else "🔴"
        
        print(f"  {status_icon} {time_str} - {status.value} {'(可交易)' if trading_allowed else ''}")
    
    # 市场时间信息
    market_info = market_manager.get_market_hours_info()
    print(f"\n⏰ 市场时间配置:")
    print(f"  市场类型: {market_info['market_type']}")
    print(f"  集合竞价: {market_info['market_hours']['pre_market_start']} - {market_info['market_hours']['pre_market_end']}")
    print(f"  上午交易: {market_info['market_hours']['morning_open']} - {market_info['market_hours']['morning_close']}")
    print(f"  下午交易: {market_info['market_hours']['afternoon_open']} - {market_info['market_hours']['afternoon_close']}")
    print(f"  盘后时间: {market_info['market_hours']['post_market_start']} - {market_info['market_hours']['post_market_end']}")
    print(f"  工作日: {[['周一','周二','周三','周四','周五','周六','周日'][day] for day in market_info['trading_days']]}")
    print(f"  节假日数量: {market_info['holidays_count']}")
    
    # 交易日历
    print(f"\n📅 本周交易日历:")
    start_of_week = datetime.now() - timedelta(days=datetime.now().weekday())
    week_days = [start_of_week + timedelta(days=i) for i in range(7)]
    
    for day in week_days:
        is_trading_day = market_manager.is_trading_day(day)
        day_name = ['周一','周二','周三','周四','周五','周六','周日'][day.weekday()]
        icon = "📈" if is_trading_day else "📵"
        
        print(f"  {icon} {day.strftime('%m-%d')} {day_name} {'交易日' if is_trading_day else '休市'}")

def demo_different_markets():
    """演示不同市场的时间"""
    print(f"\n🌍 不同市场时间对比:")
    print("="*60)
    
    markets = {
        "A_SHARE": "中国A股",
        "US_STOCK": "美股",
        "HK_STOCK": "港股"
    }
    
    current_time = datetime.now().replace(hour=10, minute=0)  # 上午10点
    
    for market_code, market_name in markets.items():
        manager = MarketTimeManager(market_code)
        status = manager.get_market_status(current_time)
        is_open = manager.is_market_open(current_time)
        
        print(f"\n📊 {market_name} ({market_code}):")
        print(f"  状态: {status.value}")
        print(f"  是否开盘: {'是' if is_open else '否'}")
        
        market_info = manager.get_market_hours_info()
        print(f"  交易时间: {market_info['market_hours']}")

def demo_trading_calendar():
    """演示交易日历功能"""
    print(f"\n📅 交易日历演示:")
    print("="*60)
    
    manager = MarketTimeManager("A_SHARE")
    
    # 获取本月交易日
    start_date = datetime.now().replace(day=1)
    end_date = (start_date + timedelta(days=32)).replace(day=1) - timedelta(days=1)
    
    trading_days = manager.get_trading_calendar(start_date, end_date)
    
    print(f"📊 {start_date.strftime('%Y年%m月')} 交易日历:")
    print(f"  总天数: {(end_date - start_date).days + 1}")
    print(f"  交易日数: {len(trading_days)}")
    print(f"  休市日数: {(end_date - start_date).days + 1 - len(trading_days)}")
    
    print(f"\n📈 交易日列表:")
    for i, day in enumerate(trading_days[:10]):  # 显示前10个交易日
        day_name = ['周一','周二','周三','周四','周五','周六','周日'][day.weekday()]
        print(f"  {i+1:2d}. {day.strftime('%m-%d')} {day_name}")
    
    if len(trading_days) > 10:
        print(f"  ... 还有 {len(trading_days) - 10} 个交易日")

def main():
    """主函数"""
    demo_market_time_management()
    demo_different_markets()
    demo_trading_calendar()
    
    print(f"\n✅ 市场时间管理演示完成!")
    print(f"\n💡 系统优势:")
    print(f"  ✓ 准确的交易时间控制")
    print(f"  ✓ 多市场支持")
    print(f"  ✓ 节假日管理")
    print(f"  ✓ 实时状态监控")
    print(f"  ✓ 自动交易时间判断")

if __name__ == "__main__":
    main()
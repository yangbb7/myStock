#!/usr/bin/env python3
"""
真实数据源配置和测试
"""

import os
from typing import Dict, Any
from myQuant.infrastructure.data.providers.real_data_provider import RealDataProvider

def get_real_data_config() -> Dict[str, Any]:
    """获取真实数据源配置"""
    
    # 优先从环境变量获取配置
    tushare_token = os.getenv('TUSHARE_TOKEN')
    
    config = {
        "primary_provider": os.getenv('PRIMARY_DATA_PROVIDER', 'eastmoney'),  # 默认使用东方财富
        "fallback_providers": os.getenv('FALLBACK_DATA_PROVIDERS', 'yahoo').split(','),
        
        # Tushare配置
        "tushare": {
            "enabled": bool(tushare_token),
            "token": tushare_token
        },
        
        # Yahoo Finance配置
        "yahoo": {
            "enabled": os.getenv('YAHOO_FINANCE_ENABLED', 'true').lower() == 'true'
        },
        
        # 东方财富配置
        "eastmoney": {
            "enabled": os.getenv('EASTMONEY_ENABLED', 'true').lower() == 'true'
        }
    }
    
    # 如果有Tushare token，优先使用Tushare
    if tushare_token:
        config["primary_provider"] = "tushare"
        config["fallback_providers"] = ["eastmoney", "yahoo"]
    else:
        # 没有Tushare时，优先使用东方财富（对中国股票支持更好）
        config["primary_provider"] = "eastmoney"
        config["fallback_providers"] = ["yahoo"]
    
    return config

def test_real_data_providers():
    """测试真实数据提供者连接"""
    config = get_real_data_config()
    print("=== 数据源配置测试 ===")
    print(f"主要数据源: {config['primary_provider']}")
    print(f"备用数据源: {config['fallback_providers']}")
    
    # 初始化数据提供者
    provider = RealDataProvider(config)
    
    # 测试连接
    print("\n=== 测试数据源连接 ===")
    results = provider.test_connection()
    
    for name, status in results.items():
        status_text = "✅ 连接成功" if status else "❌ 连接失败"
        print(f"{name}: {status_text}")
    
    # 测试获取实时数据
    test_symbols = ["000001.SZ", "000002.SZ", "600000.SH"]
    print(f"\n=== 测试实时数据获取 ===")
    print(f"测试股票: {test_symbols}")
    
    realtime_data = provider.get_realtime_data(test_symbols)
    
    if realtime_data:
        for symbol, data in realtime_data.items():
            print(f"{symbol}: ¥{data['current_price']:.2f} ({data['timestamp']})")
    else:
        print("❌ 未能获取实时数据")
    
    return provider

if __name__ == "__main__":
    provider = test_real_data_providers()
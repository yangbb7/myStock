#!/usr/bin/env python3
"""
启动 myStock 系统（集成真实数据）
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# 设置环境变量
os.environ.setdefault('PRIMARY_DATA_PROVIDER', 'yahoo')
os.environ.setdefault('YAHOO_FINANCE_ENABLED', 'true')
os.environ.setdefault('EASTMONEY_ENABLED', 'true')
os.environ.setdefault('LOG_LEVEL', 'INFO')

async def main():
    """主启动函数"""
    print("🚀 启动 myStock 量化交易系统（真实数据版本）")
    print("=" * 60)
    
    # 检查依赖
    print("📦 检查依赖...")
    try:
        import yfinance
        print("✅ yfinance 已安装")
    except ImportError:
        print("❌ yfinance 未安装，请运行: pip install yfinance")
        sys.exit(1)
    
    # 测试数据连接
    print("🔗 测试数据连接...")
    from real_data_config import get_real_data_config
    from myQuant.infrastructure.data.providers.real_data_provider import RealDataProvider
    
    config = get_real_data_config()
    provider = RealDataProvider(config)
    
    # 快速测试
    test_data = provider.get_realtime_data(["000001.SZ"])
    if test_data:
        price = test_data["000001.SZ"]["current_price"]
        print(f"✅ 数据连接成功 - 平安银行: ¥{price:.2f}")
    else:
        print("❌ 数据连接失败")
        sys.exit(1)
    
    print("\n📋 系统配置:")
    print(f"  - 主要数据源: {config['primary_provider']}")
    print(f"  - 备用数据源: {', '.join(config['fallback_providers'])}")
    print(f"  - 更新频率: 每3秒")
    print(f"  - 监控股票: 000001.SZ, 000002.SZ, 600000.SH, 600036.SH")
    
    print("\n🌟 系统特性:")
    print("  ✅ 真实股票价格（Yahoo Finance）")
    print("  ✅ 修复的涨跌幅计算")
    print("  ✅ 实时价格更新")
    print("  ✅ 自动故障转移")
    
    print("\n🎯 接下来可以:")
    print("  1. 启动后端API服务")
    print("  2. 启动前端界面")
    print("  3. 查看实时股票数据")
    
    print(f"\n" + "=" * 60)
    print("🎉 myStock 系统已准备就绪！")
    
    return True

if __name__ == "__main__":
    asyncio.run(main())
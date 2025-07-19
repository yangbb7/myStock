# -*- coding: utf-8 -*-
"""
真实数据源配置
"""

import os
from typing import Any, Dict

# 从环境变量获取API密钥
TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN", "")

# 数据源配置
DATA_SOURCES_CONFIG = {
    # 主要数据提供者
    "primary_provider": "tushare",
    # 备用数据提供者
    "fallback_providers": ["yahoo", "eastmoney"],
    # Tushare配置
    "tushare": {
        "token": TUSHARE_TOKEN,
        "enabled": bool(TUSHARE_TOKEN),
        "rate_limit": 200,  # 每分钟请求限制
        "timeout": 30,
    },
    # Yahoo Finance配置
    "yahoo": {"enabled": True, "timeout": 30, "max_retries": 3},
    # 东方财富配置
    "eastmoney": {"enabled": True, "timeout": 30, "max_retries": 3},
    # 数据缓存配置
    "cache": {
        "enabled": True,
        "ttl": 300,  # 缓存时间（秒）
        "max_size": 1000,  # 最大缓存条目数
    },
    # 数据质量检查
    "quality_check": {
        "enabled": True,
        "min_trading_days": 200,  # 最少交易日数据
        "max_price_change": 0.2,  # 最大日价格变化（20%）
        "min_volume": 1000,  # 最小成交量
    },
}

# 测试配置（当没有API密钥时使用）
TEST_DATA_CONFIG = {
    "primary_provider": "yahoo",
    "fallback_providers": ["eastmoney"],
    "tushare": {"enabled": False},
    "yahoo": {"enabled": True},
    "eastmoney": {"enabled": True},
    "cache": {"enabled": False},
    "quality_check": {"enabled": False},
}


def get_data_config() -> Dict[str, Any]:
    """获取数据配置"""
    if TUSHARE_TOKEN:
        return DATA_SOURCES_CONFIG
    else:
        print("警告: 未设置TUSHARE_TOKEN环境变量，将使用测试配置")
        return TEST_DATA_CONFIG


# 常用股票代码映射
STOCK_SYMBOLS = {
    # A股主要指数
    "SHCI": "000001.SH",  # 上证综指
    "SZCI": "399001.SZ",  # 深证成指
    "CSI300": "000300.SH",  # 沪深300
    "CSI500": "000905.SH",  # 中证500
    # 热门股票
    "PING_AN": "000001.SZ",  # 平安银行
    "WANKE_A": "000002.SZ",  # 万科A
    "YZGL": "000858.SZ",  # 五粮液
    "MOUTAI": "600519.SH",  # 贵州茅台
    "TCBANK": "600036.SH",  # 招商银行
}

# 行业板块映射
SECTOR_SYMBOLS = {
    "bank": ["000001.SZ", "600036.SH", "601318.SH"],  # 银行
    "liquor": ["600519.SH", "000858.SZ", "600809.SH"],  # 白酒
    "tech": ["000002.SZ", "300059.SZ", "002415.SZ"],  # 科技
    "pharma": ["000001.SZ", "600276.SH", "300003.SZ"],  # 医药
}


def get_test_symbols() -> list:
    """获取测试用股票代码"""
    return list(STOCK_SYMBOLS.values())[:5]  # 返回前5个常用股票

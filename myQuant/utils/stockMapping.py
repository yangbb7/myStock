# -*- coding: utf-8 -*-
"""
Stock mapping utility for Chinese stock market
"""

from typing import Dict, List, Optional

# Stock symbol to name mapping - same as frontend
STOCK_SYMBOL_MAP: Dict[str, str] = {
    # 主要银行股
    '000001.SZ': '平安银行',
    '600000.SH': '浦发银行',
    '600036.SH': '招商银行',
    '601166.SH': '兴业银行',
    '601318.SH': '中国平安',
    '601328.SH': '交通银行',
    '601398.SH': '工商银行',
    '601939.SH': '建设银行',
    '601988.SH': '中国银行',
    
    # 地产股
    '000002.SZ': '万科A',
    '001979.SZ': '招商蛇口',
    '600048.SH': '保利发展',
    '600340.SH': '华夏幸福',
    
    # 白酒股
    '000858.SZ': '五粮液',
    '000596.SZ': '古井贡酒',
    '000799.SZ': '酒鬼酒',
    '002304.SZ': '洋河股份',
    '600519.SH': '贵州茅台',
    '600779.SH': '水井坊',
    '603589.SH': '口子窖',
    
    # 科技股
    '000063.SZ': '中兴通讯',
    '000725.SZ': '京东方A',
    '002415.SZ': '海康威视',
    '002594.SZ': '比亚迪',
    '300059.SZ': '东方财富',
    '300750.SZ': '宁德时代',
    '600893.SH': '航发动力',
    
    # 医药股
    '000538.SZ': '云南白药',
    '002422.SZ': '科伦药业',
    '300015.SZ': '爱尔眼科',
    '600276.SH': '恒瑞医药',
    '600436.SH': '片仔癀',
    
    # 消费股
    '000895.SZ': '双汇发展',
    '600887.SH': '伊利股份',
    
    # 指数ETF
    '510050.SH': '50ETF',
    '510300.SH': '300ETF',
    '510500.SH': '500ETF',
    
    # Common test symbols
    '601890.SH': '亚星锚链',
    '000632.SZ': '三木集团',
    '002589.SZ': '瑞康医药',
    '300316.SZ': '晶盛机电',
}

def getStockName(symbol: str) -> str:
    """获取股票名称"""
    return STOCK_SYMBOL_MAP.get(symbol, symbol)

def getStockDisplayName(symbol: str) -> str:
    """获取格式化显示名称"""
    name = STOCK_SYMBOL_MAP.get(symbol)
    return f"{name}({symbol})" if name else symbol

def getStockInfo(symbol: str) -> Dict[str, str]:
    """获取股票信息"""
    name = getStockName(symbol)
    exchange = 'SH' if symbol.endswith('.SH') else 'SZ'
    
    return {
        'symbol': symbol,
        'name': name,
        'exchange': exchange,
    }

def searchStocks(query: str) -> List[Dict[str, str]]:
    """搜索股票"""
    if not query.strip():
        return [getStockInfo(symbol) for symbol in STOCK_SYMBOL_MAP.keys()]
    
    query_lower = query.lower()
    results = []
    
    for symbol, name in STOCK_SYMBOL_MAP.items():
        if query_lower in symbol.lower() or query in name:
            results.append(getStockInfo(symbol))
    
    return results
"""
市场数据模型

包含MarketData、RealTimeQuote和TechnicalIndicators等核心数据结构
"""

import re
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
from pydantic import BaseModel, validator


class MarketData(BaseModel):
    """市场数据模型"""
    
    symbol: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    turnover: Decimal
    
    @validator('symbol')
    def validate_symbol_format(cls, v):
        """验证股票代码格式"""
        if not re.match(r'^[0-9]{6}\.(SZ|SH)$', v):
            raise ValueError(f'Invalid symbol format: {v}')
        return v
    
    @validator('volume')
    def validate_volume(cls, v):
        """验证成交量"""
        if v < 0:
            raise ValueError('Volume must be non-negative')
        return v
    
    @validator('turnover')
    def validate_turnover(cls, v):
        """验证成交额"""
        if v < 0:
            raise ValueError('Turnover must be non-negative')
        return v
    
    def __init__(self, **data):
        super().__init__(**data)
        self._validate_price_relationships()
        
    def _validate_price_relationships(self):
        """验证价格关系"""
        # 验证价格关系
        if self.high < max(self.open, self.close):
            raise ValueError('High price must be >= max(open, close)')
        if self.low > min(self.open, self.close):
            raise ValueError('Low price must be <= min(open, close)')
        if self.high < self.low:
            raise ValueError('High price must be >= low price')
    
    def calculate_change_amount(self, previous_close: Decimal) -> Decimal:
        """计算价格变动额"""
        return self.close - previous_close
    
    def calculate_change_percent(self, previous_close: Decimal) -> Decimal:
        """计算价格变动百分比"""
        if previous_close == 0:
            return Decimal('0')
        return ((self.close - previous_close) / previous_close * 100).quantize(Decimal('0.01'))
    
    class Config:
        """Pydantic配置"""
        arbitrary_types_allowed = True
        json_encoders = {
            Decimal: lambda v: float(v),
            datetime: lambda v: v.isoformat()
        }


class RealTimeQuote(BaseModel):
    """实时行情模型"""
    
    symbol: str
    current_price: Decimal
    change_amount: Optional[Decimal] = None
    change_percent: Optional[Decimal] = None
    volume: Optional[int] = None
    turnover: Optional[Decimal] = None
    
    # 五档买卖盘
    bid_price_1: Optional[Decimal] = None
    bid_volume_1: Optional[int] = None
    bid_price_2: Optional[Decimal] = None
    bid_volume_2: Optional[int] = None
    bid_price_3: Optional[Decimal] = None
    bid_volume_3: Optional[int] = None
    bid_price_4: Optional[Decimal] = None
    bid_volume_4: Optional[int] = None
    bid_price_5: Optional[Decimal] = None
    bid_volume_5: Optional[int] = None
    
    ask_price_1: Optional[Decimal] = None
    ask_volume_1: Optional[int] = None
    ask_price_2: Optional[Decimal] = None
    ask_volume_2: Optional[int] = None
    ask_price_3: Optional[Decimal] = None
    ask_volume_3: Optional[int] = None
    ask_price_4: Optional[Decimal] = None
    ask_volume_4: Optional[int] = None
    ask_price_5: Optional[Decimal] = None
    ask_volume_5: Optional[int] = None
    
    timestamp: datetime = None
    
    def __init__(self, **data):
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now(timezone.utc)
        super().__init__(**data)
    
    @validator('symbol')
    def validate_symbol_format(cls, v):
        """验证股票代码格式"""
        if not re.match(r'^[0-9]{6}\.(SZ|SH)$', v):
            raise ValueError(f'Invalid symbol format: {v}')
        return v
    
    def get_spread(self) -> Optional[Decimal]:
        """获取买卖价差"""
        if self.bid_price_1 and self.ask_price_1:
            return self.ask_price_1 - self.bid_price_1
        return None
    
    def get_mid_price(self) -> Optional[Decimal]:
        """获取中间价"""
        if self.bid_price_1 and self.ask_price_1:
            return (self.bid_price_1 + self.ask_price_1) / 2
        return None
    
    class Config:
        """Pydantic配置"""
        arbitrary_types_allowed = True
        json_encoders = {
            Decimal: lambda v: float(v),
            datetime: lambda v: v.isoformat()
        }


# KlineData 是 MarketData 的别名，保持向后兼容性
KlineData = MarketData


class TechnicalIndicators:
    """技术指标计算工具类"""
    
    @staticmethod
    def calculate_ma(prices: List[Decimal], period: int) -> List[Decimal]:
        """计算移动平均线"""
        if len(prices) < period:
            return []
        
        ma_values = []
        for i in range(period - 1, len(prices)):
            ma_value = sum(prices[i - period + 1:i + 1]) / period
            ma_values.append(ma_value.quantize(Decimal('0.01')))
        
        return ma_values
    
    @staticmethod
    def calculate_rsi(prices: List[Decimal], period: int = 14) -> List[Decimal]:
        """计算RSI指标"""
        if len(prices) < period + 1:
            return [Decimal('50')]  # 如果数据不足，返回中性值50
        
        # 计算价格变化
        price_changes = []
        for i in range(1, len(prices)):
            change = prices[i] - prices[i - 1]
            price_changes.append(change)
        
        rsi_values = []
        
        for i in range(period - 1, len(price_changes)):
            gains = []
            losses = []
            
            for j in range(i - period + 1, i + 1):
                if price_changes[j] > 0:
                    gains.append(price_changes[j])
                    losses.append(Decimal('0'))
                else:
                    gains.append(Decimal('0'))
                    losses.append(abs(price_changes[j]))
            
            avg_gain = sum(gains) / period
            avg_loss = sum(losses) / period
            
            if avg_loss == 0:
                rsi = Decimal('100')
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            rsi_values.append(rsi.quantize(Decimal('0.01')))
        
        return rsi_values
    
    @staticmethod
    def calculate_macd(prices: List[Decimal], fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Dict[str, List[Decimal]]:
        """计算MACD指标"""
        if len(prices) < slow_period:
            return {"macd": [], "signal": [], "histogram": []}
        
        # 转换为numpy数组便于计算
        price_array = np.array([float(p) for p in prices])
        
        # 计算EMA
        def calculate_ema(data, period):
            alpha = 2 / (period + 1)
            ema = np.zeros_like(data)
            ema[0] = data[0]
            for i in range(1, len(data)):
                ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
            return ema
        
        fast_ema = calculate_ema(price_array, fast_period)
        slow_ema = calculate_ema(price_array, slow_period)
        
        # MACD线
        macd_line = fast_ema - slow_ema
        
        # 信号线
        signal_line = calculate_ema(macd_line, signal_period)
        
        # 柱状图
        histogram = macd_line - signal_line
        
        # 转换回Decimal
        macd_values = [Decimal(str(v)).quantize(Decimal('0.0001')) for v in macd_line[slow_period - 1:]]
        signal_values = [Decimal(str(v)).quantize(Decimal('0.0001')) for v in signal_line[slow_period - 1:]]
        histogram_values = [Decimal(str(v)).quantize(Decimal('0.0001')) for v in histogram[slow_period - 1:]]
        
        return {
            "macd": macd_values,
            "signal": signal_values,
            "histogram": histogram_values
        }
    
    @staticmethod
    def calculate_bollinger_bands(prices: List[Decimal], period: int = 20, std_dev: float = 2.0) -> Dict[str, List[Decimal]]:
        """计算布林带"""
        if len(prices) < period:
            return {"upper": [], "middle": [], "lower": []}
        
        middle_band = TechnicalIndicators.calculate_ma(prices, period)
        
        upper_band = []
        lower_band = []
        
        for i in range(period - 1, len(prices)):
            # 计算标准差
            price_slice = prices[i - period + 1:i + 1]
            mean_price = sum(price_slice) / period
            variance = sum([(p - mean_price) ** 2 for p in price_slice]) / period
            std = variance ** Decimal('0.5')
            
            upper = mean_price + Decimal(str(std_dev)) * std
            lower = mean_price - Decimal(str(std_dev)) * std
            
            upper_band.append(upper.quantize(Decimal('0.01')))
            lower_band.append(lower.quantize(Decimal('0.01')))
        
        return {
            "upper": upper_band,
            "middle": middle_band,
            "lower": lower_band
        }
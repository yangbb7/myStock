"""
技术指标计算器

实现各种技术指标的计算，包括移动平均线、MACD、RSI、布林带等
"""

import asyncio
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any, Union, Tuple
from decimal import Decimal
import logging

from myQuant.core.models.market_data import KlineData, MarketData


class TechnicalIndicators:
    """技术指标计算类（简化版）"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_rsi(self, prices: List[Union[float, Decimal]], period: int = 14) -> Optional[float]:
        """计算RSI（相对强弱指标）"""
        if len(prices) < period + 1:
            return None
        
        # Convert to float for calculations
        float_prices = [float(p) for p in prices]
        
        # 计算价格变化
        price_changes = [float_prices[i] - float_prices[i-1] for i in range(1, len(float_prices))]
        
        # 分离涨跌
        gains = [change if change > 0 else 0 for change in price_changes]
        losses = [-change if change < 0 else 0 for change in price_changes]
        
        # 计算平均涨跌
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_ma(self, prices: List[Union[float, Decimal]], period: int) -> Optional[float]:
        """计算简单移动平均线"""
        if len(prices) < period:
            return None
        
        # Convert to float for calculations
        float_prices = [float(p) for p in prices]
        return sum(float_prices[-period:]) / period
    
    def calculate_ema(self, prices: List[Union[float, Decimal]], period: int) -> Optional[float]:
        """计算指数移动平均线"""
        if len(prices) < period:
            return None
        
        # Convert to float for calculations
        float_prices = [float(p) for p in prices]
        multiplier = 2 / (period + 1)
        ema = float_prices[0]
        
        for price in float_prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def calculate_macd(self, prices: List[Union[float, Decimal]], 
                      fast_period: int = 12, 
                      slow_period: int = 26, 
                      signal_period: int = 9) -> Dict[str, Optional[float]]:
        """计算MACD指标"""
        if len(prices) < slow_period:
            return {'macd': None, 'signal': None, 'histogram': None}
        
        # Convert to float for calculations
        float_prices = [float(p) for p in prices]
        
        # 计算快慢EMA
        fast_ema = self._calculate_ema_series(float_prices, fast_period)
        slow_ema = self._calculate_ema_series(float_prices, slow_period)
        
        if not fast_ema or not slow_ema:
            return {'macd': None, 'signal': None, 'histogram': None}
        
        # 计算MACD线
        macd_line = [fast_ema[i] - slow_ema[i] for i in range(len(float_prices))]
        
        # 计算信号线
        signal_line = self._calculate_ema_series(macd_line, signal_period)
        
        if not signal_line:
            return {'macd': macd_line[-1], 'signal': None, 'histogram': None}
        
        # 计算柱状图
        histogram = macd_line[-1] - signal_line[-1]
        
        return {
            'macd': macd_line[-1],
            'signal': signal_line[-1],
            'histogram': histogram
        }
    
    def calculate_bollinger_bands(self, prices: List[Union[float, Decimal]], 
                                 period: int = 20, 
                                 std_dev: float = 2.0) -> Dict[str, Optional[float]]:
        """计算布林带"""
        if len(prices) < period:
            return {'upper': None, 'middle': None, 'lower': None}
        
        # Convert to float for calculations
        float_prices = [float(p) for p in prices]
        
        # 计算中轨（SMA）
        middle = self.calculate_ma(float_prices, period)
        if middle is None:
            return {'upper': None, 'middle': None, 'lower': None}
        
        # 计算标准差
        recent_prices = float_prices[-period:]
        std = np.std(recent_prices)
        
        # 计算上下轨
        upper = middle + (std_dev * float(std))
        lower = middle - (std_dev * float(std))
        
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower
        }
    
    def calculate_kdj(self, highs: List[float], lows: List[float], closes: List[float],
                     n: int = 9, m1: int = 3, m2: int = 3) -> Dict[str, Optional[float]]:
        """计算KDJ指标"""
        if len(highs) < n or len(lows) < n or len(closes) < n:
            return {'k': None, 'd': None, 'j': None}
        
        # 计算RSV
        lowest_low = min(lows[-n:])
        highest_high = max(highs[-n:])
        
        if highest_high == lowest_low:
            rsv = 50
        else:
            rsv = (closes[-1] - lowest_low) / (highest_high - lowest_low) * 100
        
        # 简化计算，假设前值
        k = rsv  # 实际应该是加权平均
        d = k    # 实际应该是K的加权平均
        j = 3 * k - 2 * d
        
        return {'k': k, 'd': d, 'j': j}
    
    def calculate_volume_ratio(self, volumes: List[float], period: int = 5) -> Optional[float]:
        """计算量比"""
        if len(volumes) < period + 1:
            return None
        
        current_volume = volumes[-1]
        avg_volume = sum(volumes[-period-1:-1]) / period
        
        if avg_volume == 0:
            return None
        
        return current_volume / avg_volume
    
    def calculate_obv(self, prices: List[float], volumes: List[float]) -> Optional[float]:
        """计算能量潮（OBV）"""
        if len(prices) < 2 or len(volumes) < 2:
            return None
        
        obv = 0
        for i in range(1, len(prices)):
            if prices[i] > prices[i-1]:
                obv += volumes[i]
            elif prices[i] < prices[i-1]:
                obv -= volumes[i]
            # 价格不变时OBV不变
        
        return obv
    
    def calculate_atr(self, highs: List[float], lows: List[float], closes: List[float], 
                     period: int = 14) -> Optional[float]:
        """计算平均真实波幅（ATR）"""
        if len(highs) < period + 1 or len(lows) < period + 1 or len(closes) < period + 1:
            return None
        
        true_ranges = []
        for i in range(1, len(highs)):
            high_low = highs[i] - lows[i]
            high_close = abs(highs[i] - closes[i-1])
            low_close = abs(lows[i] - closes[i-1])
            true_range = max(high_low, high_close, low_close)
            true_ranges.append(true_range)
        
        if len(true_ranges) < period:
            return None
        
        return sum(true_ranges[-period:]) / period
    
    def calculate_cci(self, highs: List[float], lows: List[float], closes: List[float], 
                     period: int = 20) -> Optional[float]:
        """计算商品通道指数（CCI）"""
        if len(highs) < period or len(lows) < period or len(closes) < period:
            return None
        
        # 计算典型价格
        typical_prices = [(highs[i] + lows[i] + closes[i]) / 3 for i in range(len(closes))]
        
        # 计算移动平均
        ma = sum(typical_prices[-period:]) / period
        
        # 计算平均偏差
        deviations = [abs(tp - ma) for tp in typical_prices[-period:]]
        mean_deviation = sum(deviations) / period
        
        if mean_deviation == 0:
            return 0
        
        # 计算CCI
        cci = (typical_prices[-1] - ma) / (0.015 * mean_deviation)
        
        return cci
    
    def calculate_williams_r(self, highs: List[float], lows: List[float], closes: List[float], 
                           period: int = 14) -> Optional[float]:
        """计算威廉指标（Williams %R）"""
        if len(highs) < period or len(lows) < period or len(closes) < period:
            return None
        
        highest_high = max(highs[-period:])
        lowest_low = min(lows[-period:])
        
        if highest_high == lowest_low:
            return -50
        
        williams_r = (highest_high - closes[-1]) / (highest_high - lowest_low) * -100
        
        return williams_r
    
    def _calculate_ema_series(self, prices: List[Union[float, Decimal]], period: int) -> List[float]:
        """计算EMA序列（内部方法）"""
        if len(prices) < period:
            return []
        
        multiplier = 2 / (period + 1)
        # Convert to float for calculations
        float_prices = [float(p) for p in prices]
        ema_values = [float_prices[0]]
        
        for i in range(1, len(float_prices)):
            ema = (float_prices[i] * multiplier) + (ema_values[-1] * (1 - multiplier))
            ema_values.append(ema)
        
        return ema_values
    
    def identify_patterns(self, prices: List[float], volumes: List[float]) -> Dict[str, bool]:
        """识别价格形态"""
        patterns = {
            'golden_cross': False,  # 金叉
            'death_cross': False,   # 死叉
            'volume_breakout': False,  # 放量突破
            'price_breakout': False,   # 价格突破
            'support_test': False,     # 支撑位测试
            'resistance_test': False   # 阻力位测试
        }
        
        if len(prices) < 60:  # 需要足够的数据
            return patterns
        
        # 计算短期和长期均线
        ma5 = self.calculate_ma(prices, 5)
        ma20 = self.calculate_ma(prices, 20)
        ma60 = self.calculate_ma(prices, 60)
        
        # 检测金叉/死叉
        if ma5 and ma20:
            # 获取前一天的均线
            prev_ma5 = self.calculate_ma(prices[:-1], 5)
            prev_ma20 = self.calculate_ma(prices[:-1], 20)
            
            if prev_ma5 and prev_ma20:
                # 金叉：短期均线从下向上穿过长期均线
                if prev_ma5 <= prev_ma20 and ma5 > ma20:
                    patterns['golden_cross'] = True
                # 死叉：短期均线从上向下穿过长期均线
                elif prev_ma5 >= prev_ma20 and ma5 < ma20:
                    patterns['death_cross'] = True
        
        # 检测放量
        if len(volumes) >= 20:
            avg_volume = sum(volumes[-20:]) / 20
            if volumes[-1] > avg_volume * 1.5:  # 成交量是平均值的1.5倍以上
                patterns['volume_breakout'] = True
        
        # 检测价格突破
        if len(prices) >= 20:
            recent_high = max(prices[-20:-1])  # 不包括当前价格
            recent_low = min(prices[-20:-1])
            current_price = prices[-1]
            
            if current_price > recent_high:
                patterns['price_breakout'] = True
            elif current_price < recent_low * 1.02:  # 接近最低点（2%范围内）
                patterns['support_test'] = True
            elif current_price > recent_high * 0.98:  # 接近最高点（2%范围内）
                patterns['resistance_test'] = True
        
        return patterns


class TechnicalIndicatorCalculator(TechnicalIndicators):
    """技术指标计算器（保持向后兼容）"""
    
    async def calculate_sma(self, prices: List[Union[float, Decimal]], period: int) -> List[Optional[float]]:
        """计算简单移动平均线 (SMA) - 异步版本"""
        if len(prices) < period:
            return [None] * len(prices)
        
        # Convert to float for calculations
        float_prices = [float(p) for p in prices]
        sma_values = []
        for i in range(len(float_prices)):
            if i < period - 1:
                sma_values.append(None)
            else:
                avg = sum(float_prices[i-period+1:i+1]) / period
                sma_values.append(avg)
        
        return sma_values
    
    async def calculate_ema(self, prices: List[Union[float, Decimal]], period: int) -> List[Optional[float]]:
        """计算指数移动平均线 (EMA) - 异步版本"""
        if len(prices) == 0:
            return []
        
        multiplier = 2 / (period + 1)
        # Convert to float for calculations
        float_prices = [float(p) for p in prices]
        ema_values = [float_prices[0]]  # 第一个EMA值等于第一个价格
        
        for i in range(1, len(float_prices)):
            ema = (float_prices[i] * multiplier) + (ema_values[i-1] * (1 - multiplier))
            ema_values.append(ema)
        
        return ema_values
    
    async def calculate_rsi_series(self, prices: List[Union[float, Decimal]], period: int = 14) -> List[Optional[float]]:
        """计算RSI序列 - 异步版本"""
        if len(prices) < period + 1:
            return [None] * len(prices)
        
        rsi_values = [None] * period
        
        for i in range(period, len(prices)):
            price_slice = prices[:i+1]
            # Convert to synchronous call since base class method is not async
            rsi = super().calculate_rsi(price_slice, period)
            rsi_values.append(rsi)
        
        return rsi_values
    
    async def calculate_macd_series(self, prices: List[Union[float, Decimal]], 
                                  fast_period: int = 12, 
                                  slow_period: int = 26, 
                                  signal_period: int = 9) -> Dict[str, List[Optional[float]]]:
        """计算MACD序列 - 异步版本"""
        if len(prices) < slow_period:
            return {
                'macd': [None] * len(prices),
                'signal': [None] * len(prices),
                'histogram': [None] * len(prices)
            }
        
        # 计算快慢EMA
        fast_ema = await self.calculate_ema(prices, fast_period)
        slow_ema = await self.calculate_ema(prices, slow_period)
        
        # 计算MACD线
        macd_line = []
        for i in range(len(prices)):
            if i < slow_period - 1:
                macd_line.append(None)
            else:
                macd_line.append(fast_ema[i] - slow_ema[i])
        
        # 计算信号线
        macd_values = [v for v in macd_line if v is not None]
        if len(macd_values) >= signal_period:
            signal_ema = await self.calculate_ema(macd_values, signal_period)
            signal_line = [None] * (len(prices) - len(signal_ema)) + signal_ema
        else:
            signal_line = [None] * len(prices)
        
        # 计算柱状图
        histogram = []
        for i in range(len(prices)):
            if macd_line[i] is not None and signal_line[i] is not None:
                histogram.append(macd_line[i] - signal_line[i])
            else:
                histogram.append(None)
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    async def calculate_bollinger_bands_series(self, prices: List[Union[float, Decimal]], 
                                             period: int = 20, 
                                             std_dev: float = 2.0) -> Dict[str, List[Optional[float]]]:
        """计算布林带序列 - 异步版本"""
        if len(prices) < period:
            return {
                'upper': [None] * len(prices),
                'middle': [None] * len(prices),
                'lower': [None] * len(prices)
            }
        
        upper_band = []
        middle_band = []
        lower_band = []
        
        for i in range(len(prices)):
            if i < period - 1:
                upper_band.append(None)
                middle_band.append(None)
                lower_band.append(None)
            else:
                # Convert to float for calculations
                float_slice = [float(p) for p in prices[i-period+1:i+1]]
                
                # 计算中轨（SMA）
                middle = sum(float_slice) / period
                middle_band.append(middle)
                
                # 计算标准差
                std = np.std(float_slice)
                
                # 计算上下轨
                upper_band.append(middle + (std_dev * float(std)))
                lower_band.append(middle - (std_dev * float(std)))
        
        return {
            'upper': upper_band,
            'middle': middle_band,
            'lower': lower_band
        }
    
    async def calculate_rsi(self, prices: List[Union[float, Decimal]], period: int = 14) -> List[Optional[float]]:
        """计算RSI（相对强弱指标） - 异步版本返回列表"""
        # Use the existing rsi_series method
        return await self.calculate_rsi_series(prices, period)
    
    async def calculate_macd(self, prices: List[Union[float, Decimal]], 
                           fast_period: int = 12, 
                           slow_period: int = 26, 
                           signal_period: int = 9) -> Dict[str, List[Optional[float]]]:
        """计算MACD指标 - 异步版本返回序列"""
        # Use the existing macd_series method
        return await self.calculate_macd_series(prices, fast_period, slow_period, signal_period)
    
    async def calculate_bollinger_bands(self, prices: List[Union[float, Decimal]], 
                                       period: int = 20, 
                                       std_dev: float = 2.0) -> Dict[str, List[Optional[float]]]:
        """计算布林带 - 异步版本返回序列"""
        # Use the existing bollinger_bands_series method
        return await self.calculate_bollinger_bands_series(prices, period, std_dev)
    
    async def calculate_stochastic(self, market_data: List[MarketData], 
                                 period: int = 14, 
                                 k_smoothing: int = 3, 
                                 d_smoothing: int = 3) -> Dict[str, List[Optional[float]]]:
        """计算随机震荡指标 (Stochastic Oscillator)"""
        if len(market_data) < period:
            return {
                'k_percent': [None] * len(market_data), 
                'd_percent': [None] * len(market_data)
            }
        
        # 提取价格数据
        highs = [float(data.high) for data in market_data]
        lows = [float(data.low) for data in market_data]
        closes = [float(data.close) for data in market_data]
        
        k_values = []
        d_values = []
        
        # 计算每个点的KDJ值
        for i in range(len(market_data)):
            if i < period - 1:
                k_values.append(None)
                d_values.append(None)
            else:
                # 计算RSV
                period_highs = highs[i-period+1:i+1]
                period_lows = lows[i-period+1:i+1]
                
                highest_high = max(period_highs)
                lowest_low = min(period_lows)
                
                if highest_high == lowest_low:
                    rsv = 50
                else:
                    rsv = (closes[i] - lowest_low) / (highest_high - lowest_low) * 100
                
                # 简化计算K和D
                k_values.append(rsv)  # 实际应该是加权平均
                d_values.append(rsv)  # 实际应该是K的加权平均
        
        return {
            'k_percent': k_values,
            'd_percent': d_values
        }
    
    async def calculate_atr(self, market_data: List[MarketData], period: int = 14) -> List[Optional[float]]:
        """计算平均真实波幅（ATR） - 异步版本返回序列"""
        if len(market_data) < period + 1:
            return [None] * len(market_data)
        
        # 提取价格数据
        highs = [float(data.high) for data in market_data]
        lows = [float(data.low) for data in market_data]
        closes = [float(data.close) for data in market_data]
        
        atr_values = []
        
        for i in range(len(market_data)):
            if i < period:
                atr_values.append(None)
            else:
                # 计算从第1个到当前的ATR
                slice_highs = highs[:i+1]
                slice_lows = lows[:i+1]
                slice_closes = closes[:i+1]
                
                atr = super().calculate_atr(slice_highs, slice_lows, slice_closes, period)
                atr_values.append(atr)
        
        return atr_values
    
    async def calculate_volume_indicators(self, market_data: List[MarketData]) -> Dict[str, List[Optional[float]]]:
        """计算成交量指标返回序列"""
        if len(market_data) < 1:
            return {
                'obv': [],
                'vwap': [],
                'volume_ma': []
            }
        
        # 提取数据
        volumes = [float(data.volume) for data in market_data]
        prices = [float(data.close) for data in market_data]
        highs = [float(data.high) for data in market_data]
        lows = [float(data.low) for data in market_data]
        turnovers = [float(data.turnover) for data in market_data]
        
        # 计算OBV序列
        obv_values = []
        obv = 0
        for i in range(len(market_data)):
            if i == 0:
                obv_values.append(obv)
            else:
                if prices[i] > prices[i-1]:
                    obv += volumes[i]
                elif prices[i] < prices[i-1]:
                    obv -= volumes[i]
                obv_values.append(obv)
        
        # 计算VWAP序列
        vwap_values = []
        for i in range(len(market_data)):
            if i == 0:
                typical_price = (highs[i] + lows[i] + prices[i]) / 3
                vwap_values.append(typical_price)
            else:
                # 简化VWAP计算
                cumulative_volume = sum(volumes[:i+1])
                cumulative_pv = sum((highs[j] + lows[j] + prices[j]) / 3 * volumes[j] for j in range(i+1))
                vwap = cumulative_pv / cumulative_volume if cumulative_volume > 0 else prices[i]
                vwap_values.append(vwap)
        
        # 计算成交量移动平均
        volume_ma_values = []
        period = 5
        for i in range(len(market_data)):
            if i < period - 1:
                volume_ma_values.append(None)
            else:
                ma = sum(volumes[i-period+1:i+1]) / period
                volume_ma_values.append(ma)
        
        return {
            'obv': obv_values,
            'vwap': vwap_values,
            'volume_ma': volume_ma_values
        }
    
    async def batch_calculate(self, market_data: List[MarketData], 
                            indicators_config: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """批量计算指标"""
        results = {}
        
        # 提取价格数据
        prices = [float(data.close) for data in market_data]
        
        for indicator_name, config in indicators_config.items():
            indicator_type = config.get('type')
            
            try:
                if indicator_type == 'sma':
                    period = config.get('period', 20)
                    result = await self.calculate_sma(prices, period)
                    results[indicator_name] = result
                    
                elif indicator_type == 'ema':
                    period = config.get('period', 12)
                    result = await self.calculate_ema(prices, period)
                    results[indicator_name] = result
                    
                elif indicator_type == 'rsi':
                    period = config.get('period', 14)
                    result = await self.calculate_rsi(prices, period)
                    results[indicator_name] = result
                    
                elif indicator_type == 'macd':
                    fast = config.get('fast', 12)
                    slow = config.get('slow', 26)
                    signal = config.get('signal', 9)
                    result = await self.calculate_macd(prices, fast, slow, signal)
                    results[indicator_name] = result
                    
                elif indicator_type == 'bollinger_bands':
                    period = config.get('period', 20)
                    std_dev = config.get('std_dev', 2.0)
                    result = await self.calculate_bollinger_bands(prices, period, std_dev)
                    results[indicator_name] = result
                    
                else:
                    results[indicator_name] = None
                    
            except Exception as e:
                self.logger.error(f"Error calculating {indicator_name}: {e}")
                results[indicator_name] = None
        
        return results
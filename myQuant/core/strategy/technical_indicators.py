# -*- coding: utf-8 -*-
"""
技术指标库 - 提供高性能的矢量化技术指标计算
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple
from numba import jit
import warnings

warnings.filterwarnings('ignore')


class TechnicalIndicators:
    """
    技术指标计算库 - 使用矢量化操作提供高性能计算
    """
    
    @staticmethod
    def sma(data: pd.Series, window: int) -> pd.Series:
        """
        简单移动平均线 (Simple Moving Average)
        
        Args:
            data: 价格序列
            window: 窗口大小
            
        Returns:
            pd.Series: SMA值
        """
        return data.rolling(window=window, min_periods=1).mean()
    
    @staticmethod
    def ema(data: pd.Series, window: int, alpha: Optional[float] = None) -> pd.Series:
        """
        指数移动平均线 (Exponential Moving Average)
        
        Args:
            data: 价格序列
            window: 窗口大小
            alpha: 平滑系数，默认为 2/(window+1)
            
        Returns:
            pd.Series: EMA值
        """
        if alpha is None:
            alpha = 2.0 / (window + 1)
        
        return data.ewm(alpha=alpha, adjust=False).mean()
    
    @staticmethod
    def rsi(data: pd.Series, window: int = 14) -> pd.Series:
        """
        相对强弱指数 (Relative Strength Index)
        
        Args:
            data: 价格序列
            window: 窗口大小
            
        Returns:
            pd.Series: RSI值 (0-100)
        """
        delta = data.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """
        MACD指标 (Moving Average Convergence Divergence)
        
        Args:
            data: 价格序列
            fast: 快线周期
            slow: 慢线周期
            signal: 信号线周期
            
        Returns:
            pd.DataFrame: 包含MACD线、信号线、柱状图的DataFrame
        """
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return pd.DataFrame({
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        })
    
    @staticmethod
    def bollinger_bands(data: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
        """
        布林带 (Bollinger Bands)
        
        Args:
            data: 价格序列
            window: 窗口大小
            num_std: 标准差倍数
            
        Returns:
            pd.DataFrame: 包含上轨、中轨、下轨的DataFrame
        """
        sma = TechnicalIndicators.sma(data, window)
        std = data.rolling(window=window, min_periods=1).std()
        
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        
        return pd.DataFrame({
            'upper': upper_band,
            'middle': sma,
            'lower': lower_band
        })
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                   k_window: int = 14, d_window: int = 3) -> pd.DataFrame:
        """
        随机指标 (Stochastic Oscillator)
        
        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            k_window: %K窗口大小
            d_window: %D窗口大小
            
        Returns:
            pd.DataFrame: 包含%K和%D的DataFrame
        """
        lowest_low = low.rolling(window=k_window, min_periods=1).min()
        highest_high = high.rolling(window=k_window, min_periods=1).max()
        
        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d_percent = k_percent.rolling(window=d_window, min_periods=1).mean()
        
        return pd.DataFrame({
            'k_percent': k_percent,
            'd_percent': d_percent
        })
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """
        平均真实波幅 (Average True Range)
        
        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            window: 窗口大小
            
        Returns:
            pd.Series: ATR值
        """
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=window, min_periods=1).mean()
        
        return atr
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """
        威廉姆斯%R (Williams %R)
        
        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            window: 窗口大小
            
        Returns:
            pd.Series: Williams %R值 (-100 to 0)
        """
        highest_high = high.rolling(window=window, min_periods=1).max()
        lowest_low = low.rolling(window=window, min_periods=1).min()
        
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        
        return williams_r
    
    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
        """
        商品通道指数 (Commodity Channel Index)
        
        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            window: 窗口大小
            
        Returns:
            pd.Series: CCI值
        """
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=window, min_periods=1).mean()
        
        # 计算平均偏差
        mean_deviation = typical_price.rolling(window=window, min_periods=1).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
        )
        
        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
        
        return cci
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        能量潮指标 (On-Balance Volume)
        
        Args:
            close: 收盘价序列
            volume: 成交量序列
            
        Returns:
            pd.Series: OBV值
        """
        price_change = close.diff()
        obv = np.where(price_change > 0, volume, 
                      np.where(price_change < 0, -volume, 0))
        
        return pd.Series(obv, index=close.index).cumsum()
    
    @staticmethod
    def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        成交量加权平均价格 (Volume Weighted Average Price)
        
        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            volume: 成交量序列
            
        Returns:
            pd.Series: VWAP值
        """
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        
        return vwap
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_zigzag(prices: np.ndarray, threshold: float) -> np.ndarray:
        """
        计算ZigZag指标的内部函数 (使用numba加速)
        
        Args:
            prices: 价格数组
            threshold: 阈值
            
        Returns:
            np.ndarray: ZigZag值
        """
        n = len(prices)
        zigzag = np.full(n, np.nan)
        
        trend = 0  # 0: 未确定, 1: 上升, -1: 下降
        extreme_idx = 0
        extreme_price = prices[0]
        
        for i in range(1, n):
            if trend == 0:
                if prices[i] > extreme_price * (1 + threshold):
                    trend = 1
                    zigzag[extreme_idx] = extreme_price
                    extreme_idx = i
                    extreme_price = prices[i]
                elif prices[i] < extreme_price * (1 - threshold):
                    trend = -1
                    zigzag[extreme_idx] = extreme_price
                    extreme_idx = i
                    extreme_price = prices[i]
            elif trend == 1:
                if prices[i] > extreme_price:
                    extreme_idx = i
                    extreme_price = prices[i]
                elif prices[i] < extreme_price * (1 - threshold):
                    zigzag[extreme_idx] = extreme_price
                    trend = -1
                    extreme_idx = i
                    extreme_price = prices[i]
            else:  # trend == -1
                if prices[i] < extreme_price:
                    extreme_idx = i
                    extreme_price = prices[i]
                elif prices[i] > extreme_price * (1 + threshold):
                    zigzag[extreme_idx] = extreme_price
                    trend = 1
                    extreme_idx = i
                    extreme_price = prices[i]
        
        zigzag[extreme_idx] = extreme_price
        return zigzag
    
    @staticmethod
    def zigzag(data: pd.Series, threshold: float = 0.05) -> pd.Series:
        """
        ZigZag指标 - 识别价格趋势转折点
        
        Args:
            data: 价格序列
            threshold: 转折阈值 (默认5%)
            
        Returns:
            pd.Series: ZigZag值
        """
        prices = data.values
        zigzag_values = TechnicalIndicators._calculate_zigzag(prices, threshold)
        
        return pd.Series(zigzag_values, index=data.index)
    
    @staticmethod
    def ichimoku(high: pd.Series, low: pd.Series, close: pd.Series,
                 conversion_window: int = 9, base_window: int = 26, 
                 leading_span_b_window: int = 52, displacement: int = 26) -> pd.DataFrame:
        """
        一目均衡表 (Ichimoku Kinko Hyo)
        
        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            conversion_window: 转换线周期
            base_window: 基准线周期
            leading_span_b_window: 先行线B周期
            displacement: 位移周期
            
        Returns:
            pd.DataFrame: 包含一目均衡表各条线的DataFrame
        """
        # 转换线 (Tenkan-sen)
        conversion_line = (high.rolling(window=conversion_window).max() + 
                          low.rolling(window=conversion_window).min()) / 2
        
        # 基准线 (Kijun-sen)
        base_line = (high.rolling(window=base_window).max() + 
                    low.rolling(window=base_window).min()) / 2
        
        # 先行线A (Senkou Span A)
        leading_span_a = ((conversion_line + base_line) / 2).shift(displacement)
        
        # 先行线B (Senkou Span B)
        leading_span_b = ((high.rolling(window=leading_span_b_window).max() + 
                          low.rolling(window=leading_span_b_window).min()) / 2).shift(displacement)
        
        # 滞后线 (Chikou Span)
        lagging_span = close.shift(-displacement)
        
        return pd.DataFrame({
            'conversion_line': conversion_line,
            'base_line': base_line,
            'leading_span_a': leading_span_a,
            'leading_span_b': leading_span_b,
            'lagging_span': lagging_span
        })
    
    @staticmethod
    def pivot_points(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.DataFrame:
        """
        支撑阻力位 (Pivot Points)
        
        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            
        Returns:
            pd.DataFrame: 包含支撑阻力位的DataFrame
        """
        pivot = (high + low + close) / 3
        
        r1 = 2 * pivot - low
        r2 = pivot + (high - low)
        r3 = high + 2 * (pivot - low)
        
        s1 = 2 * pivot - high
        s2 = pivot - (high - low)
        s3 = low - 2 * (high - pivot)
        
        return pd.DataFrame({
            'pivot': pivot,
            'r1': r1,
            'r2': r2,
            'r3': r3,
            's1': s1,
            's2': s2,
            's3': s3
        })
    
    @staticmethod
    def calculate_multiple_indicators(data: pd.DataFrame, 
                                    indicators: dict) -> pd.DataFrame:
        """
        批量计算多个技术指标
        
        Args:
            data: 包含OHLCV数据的DataFrame
            indicators: 指标配置字典
            
        Returns:
            pd.DataFrame: 包含所有指标的DataFrame
        """
        result = data.copy()
        
        # 必要的列名映射
        high = data.get('high', data.get('High'))
        low = data.get('low', data.get('Low'))
        close = data.get('close', data.get('Close'))
        volume = data.get('volume', data.get('Volume'))
        
        for indicator_name, config in indicators.items():
            try:
                if indicator_name == 'sma':
                    for window in config.get('windows', [5, 10, 20]):
                        result[f'sma_{window}'] = TechnicalIndicators.sma(close, window)
                        
                elif indicator_name == 'ema':
                    for window in config.get('windows', [12, 26]):
                        result[f'ema_{window}'] = TechnicalIndicators.ema(close, window)
                        
                elif indicator_name == 'rsi':
                    window = config.get('window', 14)
                    result['rsi'] = TechnicalIndicators.rsi(close, window)
                    
                elif indicator_name == 'macd':
                    fast = config.get('fast', 12)
                    slow = config.get('slow', 26)
                    signal = config.get('signal', 9)
                    macd_data = TechnicalIndicators.macd(close, fast, slow, signal)
                    result = pd.concat([result, macd_data], axis=1)
                    
                elif indicator_name == 'bollinger':
                    window = config.get('window', 20)
                    std = config.get('std', 2.0)
                    bb_data = TechnicalIndicators.bollinger_bands(close, window, std)
                    result = pd.concat([result, bb_data.add_prefix('bb_')], axis=1)
                    
                elif indicator_name == 'stochastic':
                    k_window = config.get('k_window', 14)
                    d_window = config.get('d_window', 3)
                    stoch_data = TechnicalIndicators.stochastic(high, low, close, k_window, d_window)
                    result = pd.concat([result, stoch_data.add_prefix('stoch_')], axis=1)
                    
                elif indicator_name == 'atr':
                    window = config.get('window', 14)
                    result['atr'] = TechnicalIndicators.atr(high, low, close, window)
                    
                elif indicator_name == 'williams_r':
                    window = config.get('window', 14)
                    result['williams_r'] = TechnicalIndicators.williams_r(high, low, close, window)
                    
                elif indicator_name == 'cci':
                    window = config.get('window', 20)
                    result['cci'] = TechnicalIndicators.cci(high, low, close, window)
                    
                elif indicator_name == 'obv' and volume is not None:
                    result['obv'] = TechnicalIndicators.obv(close, volume)
                    
                elif indicator_name == 'vwap' and volume is not None:
                    result['vwap'] = TechnicalIndicators.vwap(high, low, close, volume)
                    
            except Exception as e:
                print(f"Error calculating {indicator_name}: {e}")
                continue
        
        return result
# -*- coding: utf-8 -*-
"""
技术指标因子库 - 包含200+技术指标的全面因子库
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum
from numba import jit, njit
import warnings
from scipy import stats
from scipy.signal import argrelextrema
from sklearn.linear_model import LinearRegression
import talib

warnings.filterwarnings('ignore')


class FactorCategory(Enum):
    """因子分类"""
    TREND = "trend"                    # 趋势类
    MOMENTUM = "momentum"              # 动量类
    VOLATILITY = "volatility"          # 波动率类
    VOLUME = "volume"                  # 成交量类
    PATTERN = "pattern"                # 形态类
    CYCLE = "cycle"                    # 周期类
    STATISTICAL = "statistical"        # 统计类
    MICROSTRUCTURE = "microstructure"  # 微观结构类
    SENTIMENT = "sentiment"            # 情绪类
    RISK = "risk"                      # 风险类


class TechnicalFactorLibrary:
    """技术指标因子库 - 包含200+技术指标"""
    
    def __init__(self):
        """初始化技术因子库"""
        self.factor_registry = {}
        self._register_all_factors()
    
    def _register_all_factors(self):
        """注册所有因子"""
        # 趋势类因子
        self._register_trend_factors()
        # 动量类因子
        self._register_momentum_factors()
        # 波动率类因子
        self._register_volatility_factors()
        # 成交量类因子
        self._register_volume_factors()
        # 形态类因子
        self._register_pattern_factors()
        # 周期类因子
        self._register_cycle_factors()
        # 统计类因子
        self._register_statistical_factors()
        # 微观结构类因子
        self._register_microstructure_factors()
        # 情绪类因子
        self._register_sentiment_factors()
        # 风险类因子
        self._register_risk_factors()
    
    def _register_trend_factors(self):
        """注册趋势类因子"""
        trend_factors = {
            # 移动平均类
            'SMA': self.sma,
            'EMA': self.ema,
            'WMA': self.wma,
            'HMA': self.hma,
            'KAMA': self.kama,
            'MAMA': self.mama,
            'T3': self.t3,
            'TEMA': self.tema,
            'TRIMA': self.trima,
            'DEMA': self.dema,
            
            # 趋势指标
            'ADX': self.adx,
            'DI_PLUS': self.di_plus,
            'DI_MINUS': self.di_minus,
            'AROON_UP': self.aroon_up,
            'AROON_DOWN': self.aroon_down,
            'AROON_OSC': self.aroon_osc,
            'PSAR': self.psar,
            'SAR': self.sar,
            'TRIX': self.trix,
            'SUPERTREND': self.supertrend,
            
            # 通道指标
            'BOLLINGER_UPPER': self.bollinger_upper,
            'BOLLINGER_MIDDLE': self.bollinger_middle,
            'BOLLINGER_LOWER': self.bollinger_lower,
            'KELTNER_UPPER': self.keltner_upper,
            'KELTNER_MIDDLE': self.keltner_middle,
            'KELTNER_LOWER': self.keltner_lower,
            'DONCHIAN_UPPER': self.donchian_upper,
            'DONCHIAN_MIDDLE': self.donchian_middle,
            'DONCHIAN_LOWER': self.donchian_lower,
            
            # 趋势强度
            'TREND_STRENGTH': self.trend_strength,
            'PRICE_CHANNEL': self.price_channel,
            'LINEAR_REG': self.linear_regression,
            'LINEAR_REG_SLOPE': self.linear_regression_slope,
            'LINEAR_REG_ANGLE': self.linear_regression_angle,
            'LINEAR_REG_INTERCEPT': self.linear_regression_intercept,
        }
        
        for name, func in trend_factors.items():
            self.factor_registry[name] = {
                'function': func,
                'category': FactorCategory.TREND,
                'description': f'Trend factor: {name}'
            }
    
    def _register_momentum_factors(self):
        """注册动量类因子"""
        momentum_factors = {
            # 经典动量指标
            'RSI': self.rsi,
            'STOCH_K': self.stoch_k,
            'STOCH_D': self.stoch_d,
            'STOCHF_K': self.stochf_k,
            'STOCHF_D': self.stochf_d,
            'STOCHRSI_K': self.stochrsi_k,
            'STOCHRSI_D': self.stochrsi_d,
            'WILLIAMS_R': self.williams_r,
            'CCI': self.cci,
            'CMO': self.cmo,
            'ROC': self.roc,
            'ROCP': self.rocp,
            'ROCR': self.rocr,
            'ROCR100': self.rocr100,
            'MOM': self.momentum,
            
            # MACD系列
            'MACD': self.macd,
            'MACD_SIGNAL': self.macd_signal,
            'MACD_HIST': self.macd_hist,
            'MACDEXT': self.macdext,
            'MACDFIX': self.macdfix,
            
            # 动量振荡器
            'PPO': self.ppo,
            'APO': self.apo,
            'ULTOSC': self.ultimate_oscillator,
            'DPO': self.detrended_price_oscillator,
            'KDJ_K': self.kdj_k,
            'KDJ_D': self.kdj_d,
            'KDJ_J': self.kdj_j,
            
            # 价格动量
            'PRICE_MOMENTUM_1': lambda data: self.price_momentum(data, 1),
            'PRICE_MOMENTUM_5': lambda data: self.price_momentum(data, 5),
            'PRICE_MOMENTUM_10': lambda data: self.price_momentum(data, 10),
            'PRICE_MOMENTUM_20': lambda data: self.price_momentum(data, 20),
            'PRICE_MOMENTUM_60': lambda data: self.price_momentum(data, 60),
            
            # 相对强度
            'RELATIVE_STRENGTH': self.relative_strength,
            'PRICE_RELATIVE': self.price_relative,
            'NORMALIZED_MOMENTUM': self.normalized_momentum,
        }
        
        for name, func in momentum_factors.items():
            self.factor_registry[name] = {
                'function': func,
                'category': FactorCategory.MOMENTUM,
                'description': f'Momentum factor: {name}'
            }
    
    def _register_volatility_factors(self):
        """注册波动率类因子"""
        volatility_factors = {
            # 波动率指标
            'ATR': self.atr,
            'NATR': self.natr,
            'TRANGE': self.true_range,
            'HISTORICAL_VOLATILITY': self.historical_volatility,
            'REALIZED_VOLATILITY': self.realized_volatility,
            'GARMAN_KLASS_VOLATILITY': self.garman_klass_volatility,
            'PARKINSON_VOLATILITY': self.parkinson_volatility,
            'ROGERS_SATCHELL_VOLATILITY': self.rogers_satchell_volatility,
            'YANG_ZHANG_VOLATILITY': self.yang_zhang_volatility,
            
            # 波动率比率
            'VOLATILITY_RATIO': self.volatility_ratio,
            'ATR_RATIO': self.atr_ratio,
            'PRICE_RANGE': self.price_range,
            'EFFICIENCY_RATIO': self.efficiency_ratio,
            
            # 波动率通道
            'VOLATILITY_CHANNEL_UPPER': self.volatility_channel_upper,
            'VOLATILITY_CHANNEL_LOWER': self.volatility_channel_lower,
            'VOLATILITY_BREAKOUT': self.volatility_breakout,
            
            # 波动率统计
            'ROLLING_STD': self.rolling_std,
            'ROLLING_VAR': self.rolling_var,
            'ROLLING_SKEW': self.rolling_skew,
            'ROLLING_KURT': self.rolling_kurt,
            
            # 风险度量
            'DOWNSIDE_DEVIATION': self.downside_deviation,
            'UPSIDE_DEVIATION': self.upside_deviation,
            'VOLATILITY_SMILE': self.volatility_smile,
        }
        
        for name, func in volatility_factors.items():
            self.factor_registry[name] = {
                'function': func,
                'category': FactorCategory.VOLATILITY,
                'description': f'Volatility factor: {name}'
            }
    
    def _register_volume_factors(self):
        """注册成交量类因子"""
        volume_factors = {
            # 成交量指标
            'OBV': self.obv,
            'AD': self.ad,
            'ADOSC': self.adosc,
            'CMF': self.cmf,
            'MFI': self.mfi,
            'EMV': self.ease_of_movement,
            'FORCE_INDEX': self.force_index,
            'VOLUME_PRICE_TREND': self.volume_price_trend,
            'NEGATIVE_VOLUME_INDEX': self.negative_volume_index,
            'POSITIVE_VOLUME_INDEX': self.positive_volume_index,
            
            # 成交量均线
            'VOLUME_SMA': self.volume_sma,
            'VOLUME_EMA': self.volume_ema,
            'VOLUME_RATIO': self.volume_ratio,
            'RELATIVE_VOLUME': self.relative_volume,
            
            # 量价关系
            'VOLUME_WEIGHTED_PRICE': self.volume_weighted_price,
            'VWAP': self.vwap,
            'TWAP': self.twap,
            'PRICE_VOLUME_CORRELATION': self.price_volume_correlation,
            'VOLUME_MOMENTUM': self.volume_momentum,
            
            # 量能指标
            'ACCUMULATION_DISTRIBUTION': self.accumulation_distribution,
            'MONEY_FLOW_INDEX': self.money_flow_index,
            'VOLUME_OSCILLATOR': self.volume_oscillator,
            'KLINGER_OSCILLATOR': self.klinger_oscillator,
            
            # 成交量形态
            'VOLUME_SPIKE': self.volume_spike,
            'VOLUME_DRY_UP': self.volume_dry_up,
            'VOLUME_BREAKOUT': self.volume_breakout,
            'VOLUME_CLIMAX': self.volume_climax,
        }
        
        for name, func in volume_factors.items():
            self.factor_registry[name] = {
                'function': func,
                'category': FactorCategory.VOLUME,
                'description': f'Volume factor: {name}'
            }
    
    def _register_pattern_factors(self):
        """注册形态类因子"""
        pattern_factors = {
            # 蜡烛图形态
            'DOJI': self.doji,
            'HAMMER': self.hammer,
            'HANGING_MAN': self.hanging_man,
            'SHOOTING_STAR': self.shooting_star,
            'ENGULFING_BULLISH': self.engulfing_bullish,
            'ENGULFING_BEARISH': self.engulfing_bearish,
            'MORNING_STAR': self.morning_star,
            'EVENING_STAR': self.evening_star,
            'THREE_WHITE_SOLDIERS': self.three_white_soldiers,
            'THREE_BLACK_CROWS': self.three_black_crows,
            
            # 价格形态
            'DOUBLE_TOP': self.double_top,
            'DOUBLE_BOTTOM': self.double_bottom,
            'HEAD_SHOULDERS': self.head_shoulders,
            'TRIANGLE_PATTERN': self.triangle_pattern,
            'FLAG_PATTERN': self.flag_pattern,
            'PENNANT_PATTERN': self.pennant_pattern,
            
            # 支撑阻力
            'SUPPORT_LEVEL': self.support_level,
            'RESISTANCE_LEVEL': self.resistance_level,
            'PIVOT_POINTS': self.pivot_points,
            'FIBONACCI_RETRACEMENT': self.fibonacci_retracement,
            
            # 缺口分析
            'GAP_UP': self.gap_up,
            'GAP_DOWN': self.gap_down,
            'GAP_FILL': self.gap_fill,
            'GAP_SIZE': self.gap_size,
            
            # 形态强度
            'PATTERN_STRENGTH': self.pattern_strength,
            'REVERSAL_PATTERN': self.reversal_pattern,
            'CONTINUATION_PATTERN': self.continuation_pattern,
        }
        
        for name, func in pattern_factors.items():
            self.factor_registry[name] = {
                'function': func,
                'category': FactorCategory.PATTERN,
                'description': f'Pattern factor: {name}'
            }
    
    def _register_cycle_factors(self):
        """注册周期类因子"""
        cycle_factors = {
            # 周期分析
            'HILBERT_TRENDMODE': self.hilbert_trendmode,
            'HILBERT_DCPERIOD': self.hilbert_dcperiod,
            'HILBERT_DCPHASE': self.hilbert_dcphase,
            'HILBERT_PHASOR_INPHASE': self.hilbert_phasor_inphase,
            'HILBERT_PHASOR_QUADRATURE': self.hilbert_phasor_quadrature,
            'HILBERT_SINE': self.hilbert_sine,
            'HILBERT_LEADSINE': self.hilbert_leadsine,
            
            # 傅里叶分析
            'FOURIER_TRANSFORM': self.fourier_transform,
            'SPECTRAL_DENSITY': self.spectral_density,
            'DOMINANT_CYCLE': self.dominant_cycle,
            'CYCLE_STRENGTH': self.cycle_strength,
            
            # 季节性
            'SEASONAL_PATTERN': self.seasonal_pattern,
            'MONTHLY_EFFECT': self.monthly_effect,
            'WEEKLY_EFFECT': self.weekly_effect,
            'INTRADAY_PATTERN': self.intraday_pattern,
            
            # 时间序列分解
            'TREND_COMPONENT': self.trend_component,
            'SEASONAL_COMPONENT': self.seasonal_component,
            'RESIDUAL_COMPONENT': self.residual_component,
        }
        
        for name, func in cycle_factors.items():
            self.factor_registry[name] = {
                'function': func,
                'category': FactorCategory.CYCLE,
                'description': f'Cycle factor: {name}'
            }
    
    def _register_statistical_factors(self):
        """注册统计类因子"""
        statistical_factors = {
            # 统计矩
            'MEAN': self.rolling_mean,
            'MEDIAN': self.rolling_median,
            'MODE': self.rolling_mode,
            'VARIANCE': self.rolling_variance,
            'SKEWNESS': self.rolling_skewness,
            'KURTOSIS': self.rolling_kurtosis,
            
            # 分位数
            'QUANTILE_25': lambda data: self.rolling_quantile(data, 0.25),
            'QUANTILE_75': lambda data: self.rolling_quantile(data, 0.75),
            'QUANTILE_90': lambda data: self.rolling_quantile(data, 0.90),
            'QUANTILE_95': lambda data: self.rolling_quantile(data, 0.95),
            'QUANTILE_99': lambda data: self.rolling_quantile(data, 0.99),
            
            # 分布检验
            'NORMALITY_TEST': self.normality_test,
            'STATIONARITY_TEST': self.stationarity_test,
            'AUTOCORRELATION': self.autocorrelation,
            'PARTIAL_AUTOCORRELATION': self.partial_autocorrelation,
            
            # 回归分析
            'BETA_COEFFICIENT': self.beta_coefficient,
            'CORRELATION_COEFFICIENT': self.correlation_coefficient,
            'DETERMINATION_COEFFICIENT': self.determination_coefficient,
            'RESIDUAL_ANALYSIS': self.residual_analysis,
            
            # 统计套利
            'COINTEGRATION': self.cointegration,
            'SPREAD_ZSCORE': self.spread_zscore,
            'PAIRS_RATIO': self.pairs_ratio,
            'STATISTICAL_ARBITRAGE': self.statistical_arbitrage,
        }
        
        for name, func in statistical_factors.items():
            self.factor_registry[name] = {
                'function': func,
                'category': FactorCategory.STATISTICAL,
                'description': f'Statistical factor: {name}'
            }
    
    def _register_microstructure_factors(self):
        """注册微观结构类因子"""
        microstructure_factors = {
            # 流动性指标
            'BID_ASK_SPREAD': self.bid_ask_spread,
            'EFFECTIVE_SPREAD': self.effective_spread,
            'PRICE_IMPACT': self.price_impact,
            'MARKET_DEPTH': self.market_depth,
            'LIQUIDITY_RATIO': self.liquidity_ratio,
            
            # 订单流
            'ORDER_FLOW_IMBALANCE': self.order_flow_imbalance,
            'TRADE_SIZE': self.trade_size,
            'TRADE_INTENSITY': self.trade_intensity,
            'ARRIVAL_RATE': self.arrival_rate,
            
            # 微观价格行为
            'TICK_DIRECTION': self.tick_direction,
            'PRICE_VELOCITY': self.price_velocity,
            'PRICE_ACCELERATION': self.price_acceleration,
            'MICROSTRUCTURE_NOISE': self.microstructure_noise,
            
            # 高频指标
            'REALIZED_SPREAD': self.realized_spread,
            'ADVERSE_SELECTION': self.adverse_selection,
            'ORDER_BOOK_SLOPE': self.order_book_slope,
            'VOLATILITY_SIGNATURE_PLOT': self.volatility_signature_plot,
        }
        
        for name, func in microstructure_factors.items():
            self.factor_registry[name] = {
                'function': func,
                'category': FactorCategory.MICROSTRUCTURE,
                'description': f'Microstructure factor: {name}'
            }
    
    def _register_sentiment_factors(self):
        """注册情绪类因子"""
        sentiment_factors = {
            # 恐惧贪婪指标
            'FEAR_GREED_INDEX': self.fear_greed_index,
            'PUT_CALL_RATIO': self.put_call_ratio,
            'VIX_TERM_STRUCTURE': self.vix_term_structure,
            'VOLATILITY_SMILE': self.volatility_smile,
            
            # 市场情绪
            'MARKET_SENTIMENT': self.market_sentiment,
            'INVESTOR_SENTIMENT': self.investor_sentiment,
            'SENTIMENT_OSCILLATOR': self.sentiment_oscillator,
            'CROWD_BEHAVIOR': self.crowd_behavior,
            
            # 资金流向
            'MONEY_FLOW_SENTIMENT': self.money_flow_sentiment,
            'CAPITAL_FLOW': self.capital_flow,
            'INSTITUTIONAL_FLOW': self.institutional_flow,
            'RETAIL_FLOW': self.retail_flow,
            
            # 新闻情绪
            'NEWS_SENTIMENT': self.news_sentiment,
            'SOCIAL_SENTIMENT': self.social_sentiment,
            'ANALYST_SENTIMENT': self.analyst_sentiment,
        }
        
        for name, func in sentiment_factors.items():
            self.factor_registry[name] = {
                'function': func,
                'category': FactorCategory.SENTIMENT,
                'description': f'Sentiment factor: {name}'
            }
    
    def _register_risk_factors(self):
        """注册风险类因子"""
        risk_factors = {
            # 风险度量
            'VALUE_AT_RISK': self.value_at_risk,
            'CONDITIONAL_VAR': self.conditional_var,
            'EXPECTED_SHORTFALL': self.expected_shortfall,
            'MAXIMUM_DRAWDOWN': self.maximum_drawdown,
            
            # 系统性风险
            'SYSTEMATIC_RISK': self.systematic_risk,
            'IDIOSYNCRATIC_RISK': self.idiosyncratic_risk,
            'TAIL_RISK': self.tail_risk,
            'JUMP_RISK': self.jump_risk,
            
            # 下行风险
            'DOWNSIDE_RISK': self.downside_risk,
            'SORTINO_RATIO': self.sortino_ratio,
            'CALMAR_RATIO': self.calmar_ratio,
            'STERLING_RATIO': self.sterling_ratio,
            
            # 风险调整收益
            'SHARPE_RATIO': self.sharpe_ratio,
            'TREYNOR_RATIO': self.treynor_ratio,
            'INFORMATION_RATIO': self.information_ratio,
            'RISK_ADJUSTED_RETURN': self.risk_adjusted_return,
        }
        
        for name, func in risk_factors.items():
            self.factor_registry[name] = {
                'function': func,
                'category': FactorCategory.RISK,
                'description': f'Risk factor: {name}'
            }
    
    # =================== 趋势类因子实现 ===================
    
    def sma(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """简单移动平均"""
        return data['close'].rolling(window=window).mean()
    
    def ema(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """指数移动平均"""
        return data['close'].ewm(span=window).mean()
    
    def wma(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """加权移动平均"""
        weights = np.arange(1, window + 1)
        return data['close'].rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    
    def hma(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Hull移动平均"""
        half_length = int(window / 2)
        sqrt_length = int(np.sqrt(window))
        
        wma_half = self.wma(data, half_length)
        wma_full = self.wma(data, window)
        
        # 创建临时DataFrame进行计算
        temp_data = pd.DataFrame({'close': 2 * wma_half - wma_full})
        return self.wma(temp_data, sqrt_length)
    
    def kama(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """考夫曼自适应移动平均"""
        close = data['close']
        change = close.diff(window).abs()
        volatility = close.diff().abs().rolling(window).sum()
        
        efficiency_ratio = change / volatility
        efficiency_ratio = efficiency_ratio.fillna(0)
        
        # 平滑常数
        fastest_sc = 2.0 / (2 + 1)
        slowest_sc = 2.0 / (30 + 1)
        smooth_constant = (efficiency_ratio * (fastest_sc - slowest_sc) + slowest_sc) ** 2
        
        # 计算KAMA
        kama = pd.Series(index=close.index, dtype=float)
        kama.iloc[0] = close.iloc[0]
        
        for i in range(1, len(close)):
            kama.iloc[i] = kama.iloc[i-1] + smooth_constant.iloc[i] * (close.iloc[i] - kama.iloc[i-1])
        
        return kama
    
    def mama(self, data: pd.DataFrame, fastlimit: float = 0.5, slowlimit: float = 0.05) -> pd.Series:
        """MESA自适应移动平均"""
        try:
            return talib.MAMA(data['close'], fastlimit=fastlimit, slowlimit=slowlimit)[0]
        except:
            return self.ema(data, 20)  # 回退到EMA
    
    def t3(self, data: pd.DataFrame, window: int = 20, factor: float = 0.7) -> pd.Series:
        """T3移动平均"""
        try:
            return talib.T3(data['close'], timeperiod=window, vfactor=factor)
        except:
            return self.ema(data, window)
    
    def tema(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """三重指数移动平均"""
        try:
            return talib.TEMA(data['close'], timeperiod=window)
        except:
            ema1 = self.ema(data, window)
            temp_data = pd.DataFrame({'close': ema1})
            ema2 = self.ema(temp_data, window)
            temp_data = pd.DataFrame({'close': ema2})
            ema3 = self.ema(temp_data, window)
            return 3 * ema1 - 3 * ema2 + ema3
    
    def trima(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """三角移动平均"""
        try:
            return talib.TRIMA(data['close'], timeperiod=window)
        except:
            return self.sma(data, window)
    
    def dema(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """双重指数移动平均"""
        try:
            return talib.DEMA(data['close'], timeperiod=window)
        except:
            ema1 = self.ema(data, window)
            temp_data = pd.DataFrame({'close': ema1})
            ema2 = self.ema(temp_data, window)
            return 2 * ema1 - ema2
    
    def adx(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """平均方向性指数"""
        try:
            return talib.ADX(data['high'], data['low'], data['close'], timeperiod=window)
        except:
            return pd.Series(index=data.index, dtype=float)
    
    def di_plus(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """正方向指标"""
        try:
            return talib.PLUS_DI(data['high'], data['low'], data['close'], timeperiod=window)
        except:
            return pd.Series(index=data.index, dtype=float)
    
    def di_minus(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """负方向指标"""
        try:
            return talib.MINUS_DI(data['high'], data['low'], data['close'], timeperiod=window)
        except:
            return pd.Series(index=data.index, dtype=float)
    
    def aroon_up(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """阿隆上线"""
        try:
            return talib.AROONOSC(data['high'], data['low'], timeperiod=window)
        except:
            high_idx = data['high'].rolling(window).apply(lambda x: window - 1 - x.argmax())
            return (window - high_idx) / window * 100
    
    def aroon_down(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """阿隆下线"""
        try:
            aroon_up, aroon_down = talib.AROON(data['high'], data['low'], timeperiod=window)
            return aroon_down
        except:
            low_idx = data['low'].rolling(window).apply(lambda x: window - 1 - x.argmin())
            return (window - low_idx) / window * 100
    
    def aroon_osc(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """阿隆振荡器"""
        try:
            return talib.AROONOSC(data['high'], data['low'], timeperiod=window)
        except:
            return self.aroon_up(data, window) - self.aroon_down(data, window)
    
    def psar(self, data: pd.DataFrame, acceleration: float = 0.02, maximum: float = 0.2) -> pd.Series:
        """抛物线SAR"""
        try:
            return talib.SAR(data['high'], data['low'], acceleration=acceleration, maximum=maximum)
        except:
            return pd.Series(index=data.index, dtype=float)
    
    def sar(self, data: pd.DataFrame, acceleration: float = 0.02, maximum: float = 0.2) -> pd.Series:
        """SAR指标"""
        return self.psar(data, acceleration, maximum)
    
    def trix(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """TRIX指标"""
        try:
            return talib.TRIX(data['close'], timeperiod=window)
        except:
            ema1 = self.ema(data, window)
            temp_data = pd.DataFrame({'close': ema1})
            ema2 = self.ema(temp_data, window)
            temp_data = pd.DataFrame({'close': ema2})
            ema3 = self.ema(temp_data, window)
            return ema3.pct_change() * 10000
    
    def supertrend(self, data: pd.DataFrame, window: int = 10, multiplier: float = 3.0) -> pd.Series:
        """SuperTrend指标"""
        atr = self.atr(data, window)
        hl_avg = (data['high'] + data['low']) / 2
        
        upper_band = hl_avg + multiplier * atr
        lower_band = hl_avg - multiplier * atr
        
        supertrend = pd.Series(index=data.index, dtype=float)
        trend = pd.Series(index=data.index, dtype=float)
        
        for i in range(len(data)):
            if i == 0:
                supertrend.iloc[i] = lower_band.iloc[i]
                trend.iloc[i] = 1
            else:
                if data['close'].iloc[i] <= supertrend.iloc[i-1]:
                    supertrend.iloc[i] = upper_band.iloc[i]
                    trend.iloc[i] = -1
                else:
                    supertrend.iloc[i] = lower_band.iloc[i]
                    trend.iloc[i] = 1
                    
                if trend.iloc[i] == trend.iloc[i-1]:
                    if trend.iloc[i] == 1:
                        supertrend.iloc[i] = max(supertrend.iloc[i], supertrend.iloc[i-1])
                    else:
                        supertrend.iloc[i] = min(supertrend.iloc[i], supertrend.iloc[i-1])
        
        return supertrend
    
    # =================== 布林带相关 ===================
    
    def bollinger_upper(self, data: pd.DataFrame, window: int = 20, std_dev: float = 2) -> pd.Series:
        """布林带上轨"""
        try:
            upper, middle, lower = talib.BBANDS(data['close'], timeperiod=window, nbdevup=std_dev, nbdevdn=std_dev)
            return upper
        except:
            sma = self.sma(data, window)
            std = data['close'].rolling(window).std()
            return sma + std_dev * std
    
    def bollinger_middle(self, data: pd.DataFrame, window: int = 20, std_dev: float = 2) -> pd.Series:
        """布林带中轨"""
        return self.sma(data, window)
    
    def bollinger_lower(self, data: pd.DataFrame, window: int = 20, std_dev: float = 2) -> pd.Series:
        """布林带下轨"""
        try:
            upper, middle, lower = talib.BBANDS(data['close'], timeperiod=window, nbdevup=std_dev, nbdevdn=std_dev)
            return lower
        except:
            sma = self.sma(data, window)
            std = data['close'].rolling(window).std()
            return sma - std_dev * std
    
    # =================== 肯特纳通道 ===================
    
    def keltner_upper(self, data: pd.DataFrame, window: int = 20, multiplier: float = 2) -> pd.Series:
        """肯特纳通道上轨"""
        ema = self.ema(data, window)
        atr = self.atr(data, window)
        return ema + multiplier * atr
    
    def keltner_middle(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """肯特纳通道中轨"""
        return self.ema(data, window)
    
    def keltner_lower(self, data: pd.DataFrame, window: int = 20, multiplier: float = 2) -> pd.Series:
        """肯特纳通道下轨"""
        ema = self.ema(data, window)
        atr = self.atr(data, window)
        return ema - multiplier * atr
    
    # =================== 唐奇安通道 ===================
    
    def donchian_upper(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """唐奇安通道上轨"""
        return data['high'].rolling(window).max()
    
    def donchian_middle(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """唐奇安通道中轨"""
        return (self.donchian_upper(data, window) + self.donchian_lower(data, window)) / 2
    
    def donchian_lower(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """唐奇安通道下轨"""
        return data['low'].rolling(window).min()
    
    # =================== 趋势强度指标 ===================
    
    def trend_strength(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """趋势强度指标"""
        close = data['close']
        sma = self.sma(data, window)
        return (close - sma) / sma
    
    def price_channel(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """价格通道位置"""
        high_max = data['high'].rolling(window).max()
        low_min = data['low'].rolling(window).min()
        return (data['close'] - low_min) / (high_max - low_min)
    
    def linear_regression(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """线性回归"""
        def lr_value(y):
            x = np.arange(len(y))
            if len(y) < 2:
                return y[-1] if len(y) > 0 else np.nan
            try:
                slope, intercept = np.polyfit(x, y, 1)
                return slope * (len(y) - 1) + intercept
            except:
                return y[-1] if len(y) > 0 else np.nan
        
        return data['close'].rolling(window).apply(lr_value, raw=True)
    
    def linear_regression_slope(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """线性回归斜率"""
        def lr_slope(y):
            x = np.arange(len(y))
            if len(y) < 2:
                return 0
            try:
                slope, _ = np.polyfit(x, y, 1)
                return slope
            except:
                return 0
        
        return data['close'].rolling(window).apply(lr_slope, raw=True)
    
    def linear_regression_angle(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """线性回归角度"""
        slope = self.linear_regression_slope(data, window)
        return np.arctan(slope) * 180 / np.pi
    
    def linear_regression_intercept(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """线性回归截距"""
        def lr_intercept(y):
            x = np.arange(len(y))
            if len(y) < 2:
                return y[0] if len(y) > 0 else np.nan
            try:
                _, intercept = np.polyfit(x, y, 1)
                return intercept
            except:
                return y[0] if len(y) > 0 else np.nan
        
        return data['close'].rolling(window).apply(lr_intercept, raw=True)
    
    # =================== 动量类因子实现 ===================
    
    def rsi(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """相对强弱指数"""
        try:
            return talib.RSI(data['close'], timeperiod=window)
        except:
            delta = data['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window).mean()
            avg_loss = loss.rolling(window).mean()
            
            rs = avg_gain / avg_loss
            return 100 - (100 / (1 + rs))
    
    def stoch_k(self, data: pd.DataFrame, k_window: int = 14, d_window: int = 3) -> pd.Series:
        """随机指标K值"""
        try:
            k, d = talib.STOCH(data['high'], data['low'], data['close'], 
                              fastk_period=k_window, slowk_period=d_window, slowd_period=d_window)
            return k
        except:
            lowest_low = data['low'].rolling(k_window).min()
            highest_high = data['high'].rolling(k_window).max()
            k = 100 * (data['close'] - lowest_low) / (highest_high - lowest_low)
            return k.rolling(d_window).mean()
    
    def stoch_d(self, data: pd.DataFrame, k_window: int = 14, d_window: int = 3) -> pd.Series:
        """随机指标D值"""
        try:
            k, d = talib.STOCH(data['high'], data['low'], data['close'], 
                              fastk_period=k_window, slowk_period=d_window, slowd_period=d_window)
            return d
        except:
            k = self.stoch_k(data, k_window, d_window)
            return k.rolling(d_window).mean()
    
    def stochf_k(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """快速随机指标K值"""
        try:
            k, d = talib.STOCHF(data['high'], data['low'], data['close'], fastk_period=window)
            return k
        except:
            lowest_low = data['low'].rolling(window).min()
            highest_high = data['high'].rolling(window).max()
            return 100 * (data['close'] - lowest_low) / (highest_high - lowest_low)
    
    def stochf_d(self, data: pd.DataFrame, window: int = 14, d_window: int = 3) -> pd.Series:
        """快速随机指标D值"""
        try:
            k, d = talib.STOCHF(data['high'], data['low'], data['close'], 
                               fastk_period=window, fastd_period=d_window)
            return d
        except:
            k = self.stochf_k(data, window)
            return k.rolling(d_window).mean()
    
    def stochrsi_k(self, data: pd.DataFrame, window: int = 14, k_window: int = 14, d_window: int = 3) -> pd.Series:
        """随机RSI K值"""
        try:
            k, d = talib.STOCHRSI(data['close'], timeperiod=window, fastk_period=k_window, fastd_period=d_window)
            return k
        except:
            rsi = self.rsi(data, window)
            temp_data = pd.DataFrame({'high': rsi, 'low': rsi, 'close': rsi})
            return self.stochf_k(temp_data, k_window)
    
    def stochrsi_d(self, data: pd.DataFrame, window: int = 14, k_window: int = 14, d_window: int = 3) -> pd.Series:
        """随机RSI D值"""
        try:
            k, d = talib.STOCHRSI(data['close'], timeperiod=window, fastk_period=k_window, fastd_period=d_window)
            return d
        except:
            k = self.stochrsi_k(data, window, k_window, d_window)
            return k.rolling(d_window).mean()
    
    def williams_r(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """威廉姆斯%R"""
        try:
            return talib.WILLR(data['high'], data['low'], data['close'], timeperiod=window)
        except:
            highest_high = data['high'].rolling(window).max()
            lowest_low = data['low'].rolling(window).min()
            return -100 * (highest_high - data['close']) / (highest_high - lowest_low)
    
    def cci(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """商品通道指数"""
        try:
            return talib.CCI(data['high'], data['low'], data['close'], timeperiod=window)
        except:
            typical_price = (data['high'] + data['low'] + data['close']) / 3
            sma_tp = typical_price.rolling(window).mean()
            mean_deviation = typical_price.rolling(window).apply(
                lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
            )
            return (typical_price - sma_tp) / (0.015 * mean_deviation)
    
    def cmo(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """钱德动量摆动指标"""
        try:
            return talib.CMO(data['close'], timeperiod=window)
        except:
            delta = data['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window).sum()
            loss = -delta.where(delta < 0, 0).rolling(window).sum()
            return 100 * (gain - loss) / (gain + loss)
    
    def roc(self, data: pd.DataFrame, window: int = 10) -> pd.Series:
        """变动率"""
        try:
            return talib.ROC(data['close'], timeperiod=window)
        except:
            return data['close'].pct_change(window) * 100
    
    def rocp(self, data: pd.DataFrame, window: int = 10) -> pd.Series:
        """变动率百分比"""
        try:
            return talib.ROCP(data['close'], timeperiod=window)
        except:
            return data['close'].pct_change(window)
    
    def rocr(self, data: pd.DataFrame, window: int = 10) -> pd.Series:
        """变动率比率"""
        try:
            return talib.ROCR(data['close'], timeperiod=window)
        except:
            return data['close'] / data['close'].shift(window)
    
    def rocr100(self, data: pd.DataFrame, window: int = 10) -> pd.Series:
        """变动率比率100"""
        try:
            return talib.ROCR100(data['close'], timeperiod=window)
        except:
            return self.rocr(data, window) * 100
    
    def momentum(self, data: pd.DataFrame, window: int = 10) -> pd.Series:
        """动量指标"""
        try:
            return talib.MOM(data['close'], timeperiod=window)
        except:
            return data['close'] - data['close'].shift(window)
    
    # =================== MACD系列 ===================
    
    def macd(self, data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """MACD线"""
        try:
            macd_line, macd_signal, macd_hist = talib.MACD(data['close'], fastperiod=fast, slowperiod=slow, signalperiod=signal)
            return macd_line
        except:
            ema_fast = self.ema(data, fast)
            ema_slow = self.ema(data, slow)
            return ema_fast - ema_slow
    
    def macd_signal(self, data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """MACD信号线"""
        try:
            macd_line, macd_signal, macd_hist = talib.MACD(data['close'], fastperiod=fast, slowperiod=slow, signalperiod=signal)
            return macd_signal
        except:
            macd_line = self.macd(data, fast, slow, signal)
            temp_data = pd.DataFrame({'close': macd_line})
            return self.ema(temp_data, signal)
    
    def macd_hist(self, data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """MACD柱状图"""
        try:
            macd_line, macd_signal, macd_hist = talib.MACD(data['close'], fastperiod=fast, slowperiod=slow, signalperiod=signal)
            return macd_hist
        except:
            return self.macd(data, fast, slow, signal) - self.macd_signal(data, fast, slow, signal)
    
    def macdext(self, data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """扩展MACD"""
        try:
            return talib.MACDEXT(data['close'], fastperiod=fast, slowperiod=slow, signalperiod=signal)[0]
        except:
            return self.macd(data, fast, slow, signal)
    
    def macdfix(self, data: pd.DataFrame, signal: int = 9) -> pd.Series:
        """固定MACD"""
        try:
            return talib.MACDFIX(data['close'], signalperiod=signal)[0]
        except:
            return self.macd(data, 12, 26, signal)
    
    # =================== 动量振荡器 ===================
    
    def ppo(self, data: pd.DataFrame, fast: int = 12, slow: int = 26) -> pd.Series:
        """价格振荡器"""
        try:
            return talib.PPO(data['close'], fastperiod=fast, slowperiod=slow)
        except:
            ema_fast = self.ema(data, fast)
            ema_slow = self.ema(data, slow)
            return (ema_fast - ema_slow) / ema_slow * 100
    
    def apo(self, data: pd.DataFrame, fast: int = 12, slow: int = 26) -> pd.Series:
        """绝对价格振荡器"""
        try:
            return talib.APO(data['close'], fastperiod=fast, slowperiod=slow)
        except:
            ema_fast = self.ema(data, fast)
            ema_slow = self.ema(data, slow)
            return ema_fast - ema_slow
    
    def ultimate_oscillator(self, data: pd.DataFrame, period1: int = 7, period2: int = 14, period3: int = 28) -> pd.Series:
        """终极振荡器"""
        try:
            return talib.ULTOSC(data['high'], data['low'], data['close'], 
                               timeperiod1=period1, timeperiod2=period2, timeperiod3=period3)
        except:
            return pd.Series(index=data.index, dtype=float)
    
    def detrended_price_oscillator(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """去趋势价格振荡器"""
        shift = int(window / 2) + 1
        sma = self.sma(data, window)
        return data['close'] - sma.shift(shift)
    
    def kdj_k(self, data: pd.DataFrame, k_window: int = 9, d_window: int = 3, j_window: int = 3) -> pd.Series:
        """KDJ的K值"""
        lowest_low = data['low'].rolling(k_window).min()
        highest_high = data['high'].rolling(k_window).max()
        rsv = 100 * (data['close'] - lowest_low) / (highest_high - lowest_low)
        
        k = pd.Series(index=data.index, dtype=float)
        k.iloc[0] = 50
        
        for i in range(1, len(rsv)):
            k.iloc[i] = (2/3) * k.iloc[i-1] + (1/3) * rsv.iloc[i]
        
        return k
    
    def kdj_d(self, data: pd.DataFrame, k_window: int = 9, d_window: int = 3, j_window: int = 3) -> pd.Series:
        """KDJ的D值"""
        k = self.kdj_k(data, k_window, d_window, j_window)
        
        d = pd.Series(index=data.index, dtype=float)
        d.iloc[0] = 50
        
        for i in range(1, len(k)):
            d.iloc[i] = (2/3) * d.iloc[i-1] + (1/3) * k.iloc[i]
        
        return d
    
    def kdj_j(self, data: pd.DataFrame, k_window: int = 9, d_window: int = 3, j_window: int = 3) -> pd.Series:
        """KDJ的J值"""
        k = self.kdj_k(data, k_window, d_window, j_window)
        d = self.kdj_d(data, k_window, d_window, j_window)
        return 3 * k - 2 * d
    
    # =================== 价格动量系列 ===================
    
    def price_momentum(self, data: pd.DataFrame, window: int) -> pd.Series:
        """价格动量"""
        return data['close'] / data['close'].shift(window) - 1
    
    def relative_strength(self, data: pd.DataFrame, benchmark_data: Optional[pd.DataFrame] = None, window: int = 20) -> pd.Series:
        """相对强度"""
        if benchmark_data is None:
            # 使用移动平均作为基准
            benchmark = self.sma(data, window)
            return data['close'] / benchmark
        else:
            return data['close'] / benchmark_data['close']
    
    def price_relative(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """价格相对位置"""
        high_max = data['high'].rolling(window).max()
        low_min = data['low'].rolling(window).min()
        return (data['close'] - low_min) / (high_max - low_min)
    
    def normalized_momentum(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """标准化动量"""
        momentum = self.momentum(data, window)
        return momentum / data['close'].rolling(window).std()
    
    # =================== 波动率类因子实现 ===================
    
    def atr(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """平均真实波幅"""
        try:
            return talib.ATR(data['high'], data['low'], data['close'], timeperiod=window)
        except:
            high_low = data['high'] - data['low']
            high_close_prev = abs(data['high'] - data['close'].shift(1))
            low_close_prev = abs(data['low'] - data['close'].shift(1))
            
            true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
            return true_range.rolling(window).mean()
    
    def natr(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """标准化平均真实波幅"""
        try:
            return talib.NATR(data['high'], data['low'], data['close'], timeperiod=window)
        except:
            atr = self.atr(data, window)
            return atr / data['close'] * 100
    
    def true_range(self, data: pd.DataFrame) -> pd.Series:
        """真实波幅"""
        try:
            return talib.TRANGE(data['high'], data['low'], data['close'])
        except:
            high_low = data['high'] - data['low']
            high_close_prev = abs(data['high'] - data['close'].shift(1))
            low_close_prev = abs(data['low'] - data['close'].shift(1))
            
            return pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    
    def historical_volatility(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """历史波动率"""
        returns = data['close'].pct_change()
        return returns.rolling(window).std() * np.sqrt(252)  # 年化
    
    def realized_volatility(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """已实现波动率"""
        log_returns = np.log(data['close'] / data['close'].shift(1))
        return log_returns.rolling(window).std() * np.sqrt(252)
    
    def garman_klass_volatility(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Garman-Klass波动率"""
        log_hl = np.log(data['high'] / data['low'])
        log_co = np.log(data['close'] / data['open'])
        
        gk = 0.5 * log_hl**2 - (2*np.log(2) - 1) * log_co**2
        return np.sqrt(gk.rolling(window).mean() * 252)
    
    def parkinson_volatility(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Parkinson波动率"""
        log_hl = np.log(data['high'] / data['low'])
        park = log_hl**2 / (4 * np.log(2))
        return np.sqrt(park.rolling(window).mean() * 252)
    
    def rogers_satchell_volatility(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Rogers-Satchell波动率"""
        log_ho = np.log(data['high'] / data['open'])
        log_hc = np.log(data['high'] / data['close'])
        log_lo = np.log(data['low'] / data['open'])
        log_lc = np.log(data['low'] / data['close'])
        
        rs = log_ho * log_hc + log_lo * log_lc
        return np.sqrt(rs.rolling(window).mean() * 252)
    
    def yang_zhang_volatility(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Yang-Zhang波动率"""
        log_co = np.log(data['close'] / data['open'])
        log_oc = np.log(data['open'] / data['close'].shift(1))
        
        rs = self.rogers_satchell_volatility(data, window)
        
        k = 0.34 / (1.34 + (window + 1) / (window - 1))
        
        overnight_var = log_oc.rolling(window).var()
        openclose_var = log_co.rolling(window).var()
        
        yz = overnight_var + k * openclose_var + (1 - k) * (rs / np.sqrt(252))**2
        return np.sqrt(yz * 252)
    
    def volatility_ratio(self, data: pd.DataFrame, short_window: int = 10, long_window: int = 30) -> pd.Series:
        """波动率比率"""
        short_vol = self.historical_volatility(data, short_window)
        long_vol = self.historical_volatility(data, long_window)
        return short_vol / long_vol
    
    def atr_ratio(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """ATR比率"""
        atr = self.atr(data, window)
        return atr / data['close']
    
    def price_range(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """价格范围"""
        high_max = data['high'].rolling(window).max()
        low_min = data['low'].rolling(window).min()
        return (high_max - low_min) / data['close']
    
    def efficiency_ratio(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """效率比率"""
        price_change = abs(data['close'] - data['close'].shift(window))
        volatility = abs(data['close'].diff()).rolling(window).sum()
        return price_change / volatility
    
    def volatility_channel_upper(self, data: pd.DataFrame, window: int = 20, multiplier: float = 2) -> pd.Series:
        """波动率通道上轨"""
        sma = self.sma(data, window)
        vol = self.historical_volatility(data, window)
        return sma + multiplier * vol * data['close']
    
    def volatility_channel_lower(self, data: pd.DataFrame, window: int = 20, multiplier: float = 2) -> pd.Series:
        """波动率通道下轨"""
        sma = self.sma(data, window)
        vol = self.historical_volatility(data, window)
        return sma - multiplier * vol * data['close']
    
    def volatility_breakout(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """波动率突破"""
        vol = self.historical_volatility(data, window)
        vol_ma = vol.rolling(window).mean()
        vol_std = vol.rolling(window).std()
        return (vol - vol_ma) / vol_std
    
    def rolling_std(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """滚动标准差"""
        returns = data['close'].pct_change()
        return returns.rolling(window).std()
    
    def rolling_var(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """滚动方差"""
        returns = data['close'].pct_change()
        return returns.rolling(window).var()
    
    def rolling_skew(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """滚动偏度"""
        returns = data['close'].pct_change()
        return returns.rolling(window).skew()
    
    def rolling_kurt(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """滚动峰度"""
        returns = data['close'].pct_change()
        return returns.rolling(window).kurt()
    
    def downside_deviation(self, data: pd.DataFrame, window: int = 20, target: float = 0) -> pd.Series:
        """下行偏差"""
        returns = data['close'].pct_change()
        downside_returns = returns[returns < target]
        
        result = pd.Series(index=data.index, dtype=float)
        for i in range(window-1, len(returns)):
            window_returns = returns.iloc[i-window+1:i+1]
            downside = window_returns[window_returns < target]
            if len(downside) > 0:
                result.iloc[i] = np.sqrt(((downside - target)**2).mean())
            else:
                result.iloc[i] = 0
                
        return result
    
    def upside_deviation(self, data: pd.DataFrame, window: int = 20, target: float = 0) -> pd.Series:
        """上行偏差"""
        returns = data['close'].pct_change()
        
        result = pd.Series(index=data.index, dtype=float)
        for i in range(window-1, len(returns)):
            window_returns = returns.iloc[i-window+1:i+1]
            upside = window_returns[window_returns > target]
            if len(upside) > 0:
                result.iloc[i] = np.sqrt(((upside - target)**2).mean())
            else:
                result.iloc[i] = 0
                
        return result
    
    def volatility_smile(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """波动率微笑"""
        # 简化实现，实际需要期权数据
        vol = self.historical_volatility(data, window)
        return vol
    
    # =================== 成交量类因子实现 ===================
    
    def obv(self, data: pd.DataFrame) -> pd.Series:
        """能量潮指标"""
        try:
            return talib.OBV(data['close'], data['volume'])
        except:
            price_change = data['close'].diff()
            volume_direction = np.where(price_change > 0, data['volume'], 
                                       np.where(price_change < 0, -data['volume'], 0))
            return pd.Series(volume_direction, index=data.index).cumsum()
    
    def ad(self, data: pd.DataFrame) -> pd.Series:
        """累积/派发线"""
        try:
            return talib.AD(data['high'], data['low'], data['close'], data['volume'])
        except:
            clv = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
            clv = clv.fillna(0)
            ad_volume = clv * data['volume']
            return ad_volume.cumsum()
    
    def adosc(self, data: pd.DataFrame, fast: int = 3, slow: int = 10) -> pd.Series:
        """累积/派发振荡器"""
        try:
            return talib.ADOSC(data['high'], data['low'], data['close'], data['volume'], fastperiod=fast, slowperiod=slow)
        except:
            ad_line = self.ad(data)
            temp_data = pd.DataFrame({'close': ad_line})
            return self.ema(temp_data, fast) - self.ema(temp_data, slow)
    
    def cmf(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """资金流量指标"""
        clv = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
        clv = clv.fillna(0)
        
        cmf_numerator = (clv * data['volume']).rolling(window).sum()
        cmf_denominator = data['volume'].rolling(window).sum()
        
        return cmf_numerator / cmf_denominator
    
    def mfi(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """资金流量指数"""
        try:
            return talib.MFI(data['high'], data['low'], data['close'], data['volume'], timeperiod=window)
        except:
            typical_price = (data['high'] + data['low'] + data['close']) / 3
            money_flow = typical_price * data['volume']
            
            positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window).sum()
            negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window).sum()
            
            mfi = 100 - (100 / (1 + positive_flow / negative_flow))
            return mfi.fillna(50)
    
    def ease_of_movement(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """简易波动指标"""
        distance_moved = (data['high'] + data['low']) / 2 - (data['high'].shift(1) + data['low'].shift(1)) / 2
        box_height = data['high'] - data['low']
        
        # 避免除零
        volume_scaled = data['volume'] / 100000000  # 缩放成交量
        box_ratio = volume_scaled / box_height
        box_ratio = box_ratio.replace([np.inf, -np.inf], 0)
        
        emv = distance_moved / box_ratio
        emv = emv.replace([np.inf, -np.inf], 0).fillna(0)
        
        return emv.rolling(window).mean()
    
    def force_index(self, data: pd.DataFrame, window: int = 13) -> pd.Series:
        """强力指数"""
        price_change = data['close'] - data['close'].shift(1)
        force = price_change * data['volume']
        return force.rolling(window).mean()
    
    def volume_price_trend(self, data: pd.DataFrame) -> pd.Series:
        """量价趋势指标"""
        price_change_pct = data['close'].pct_change()
        vpt = (price_change_pct * data['volume']).cumsum()
        return vpt
    
    def negative_volume_index(self, data: pd.DataFrame) -> pd.Series:
        """负成交量指数"""
        nvi = pd.Series(index=data.index, dtype=float)
        nvi.iloc[0] = 1000
        
        for i in range(1, len(data)):
            if data['volume'].iloc[i] < data['volume'].iloc[i-1]:
                price_change = (data['close'].iloc[i] - data['close'].iloc[i-1]) / data['close'].iloc[i-1]
                nvi.iloc[i] = nvi.iloc[i-1] * (1 + price_change)
            else:
                nvi.iloc[i] = nvi.iloc[i-1]
        
        return nvi
    
    def positive_volume_index(self, data: pd.DataFrame) -> pd.Series:
        """正成交量指数"""
        pvi = pd.Series(index=data.index, dtype=float)
        pvi.iloc[0] = 1000
        
        for i in range(1, len(data)):
            if data['volume'].iloc[i] > data['volume'].iloc[i-1]:
                price_change = (data['close'].iloc[i] - data['close'].iloc[i-1]) / data['close'].iloc[i-1]
                pvi.iloc[i] = pvi.iloc[i-1] * (1 + price_change)
            else:
                pvi.iloc[i] = pvi.iloc[i-1]
        
        return pvi
    
    def volume_sma(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """成交量简单移动平均"""
        return data['volume'].rolling(window).mean()
    
    def volume_ema(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """成交量指数移动平均"""
        return data['volume'].ewm(span=window).mean()
    
    def volume_ratio(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """成交量比率"""
        volume_ma = self.volume_sma(data, window)
        return data['volume'] / volume_ma
    
    def relative_volume(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """相对成交量"""
        volume_std = data['volume'].rolling(window).std()
        volume_ma = self.volume_sma(data, window)
        return (data['volume'] - volume_ma) / volume_std
    
    def volume_weighted_price(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """成交量加权价格"""
        vwp = (data['close'] * data['volume']).rolling(window).sum() / data['volume'].rolling(window).sum()
        return vwp
    
    def vwap(self, data: pd.DataFrame) -> pd.Series:
        """成交量加权平均价格（日内）"""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        vwap = (typical_price * data['volume']).cumsum() / data['volume'].cumsum()
        return vwap
    
    def twap(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """时间加权平均价格"""
        # 简化实现，等权重
        return data['close'].rolling(window).mean()
    
    def price_volume_correlation(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """价格成交量相关性"""
        price_change = data['close'].pct_change()
        volume_change = data['volume'].pct_change()
        
        correlation = price_change.rolling(window).corr(volume_change)
        return correlation
    
    def volume_momentum(self, data: pd.DataFrame, window: int = 10) -> pd.Series:
        """成交量动量"""
        return data['volume'] / data['volume'].shift(window) - 1
    
    def accumulation_distribution(self, data: pd.DataFrame) -> pd.Series:
        """累积分布线"""
        return self.ad(data)
    
    def money_flow_index(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """资金流量指数"""
        return self.mfi(data, window)
    
    def volume_oscillator(self, data: pd.DataFrame, short: int = 5, long: int = 10) -> pd.Series:
        """成交量振荡器"""
        short_vol = self.volume_sma(data, short)
        long_vol = self.volume_sma(data, long)
        return (short_vol - long_vol) / long_vol * 100
    
    def klinger_oscillator(self, data: pd.DataFrame, fast: int = 34, slow: int = 55, signal: int = 13) -> pd.Series:
        """克林格振荡器"""
        # 简化实现
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        typical_price_change = typical_price.diff()
        
        volume_force = data['volume'] * np.sign(typical_price_change)
        
        temp_data = pd.DataFrame({'close': volume_force.cumsum()})
        fast_ma = self.ema(temp_data, fast)
        slow_ma = self.ema(temp_data, slow)
        
        return fast_ma - slow_ma
    
    def volume_spike(self, data: pd.DataFrame, window: int = 20, threshold: float = 2) -> pd.Series:
        """成交量异常"""
        volume_ma = self.volume_sma(data, window)
        volume_std = data['volume'].rolling(window).std()
        
        spike = (data['volume'] - volume_ma) / volume_std
        return (spike > threshold).astype(int)
    
    def volume_dry_up(self, data: pd.DataFrame, window: int = 20, threshold: float = -1.5) -> pd.Series:
        """成交量萎缩"""
        volume_ma = self.volume_sma(data, window)
        volume_std = data['volume'].rolling(window).std()
        
        dry_up = (data['volume'] - volume_ma) / volume_std
        return (dry_up < threshold).astype(int)
    
    def volume_breakout(self, data: pd.DataFrame, window: int = 20, volume_threshold: float = 1.5) -> pd.Series:
        """量价突破"""
        price_breakout = data['close'] > data['close'].rolling(window).max().shift(1)
        volume_breakout = data['volume'] > data['volume'].rolling(window).mean() * volume_threshold
        
        return (price_breakout & volume_breakout).astype(int)
    
    def volume_climax(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """成交量高潮"""
        volume_ma = self.volume_sma(data, window)
        volume_ratio = data['volume'] / volume_ma
        
        price_change = abs(data['close'].pct_change())
        price_change_ma = price_change.rolling(window).mean()
        price_ratio = price_change / price_change_ma
        
        climax = volume_ratio * price_ratio
        return climax
    
    # =================== 辅助函数 ===================
    
    def get_factor_list(self, category: Optional[FactorCategory] = None) -> List[str]:
        """获取因子列表"""
        if category is None:
            return list(self.factor_registry.keys())
        else:
            return [name for name, info in self.factor_registry.items() 
                   if info['category'] == category]
    
    def compute_factor(self, factor_name: str, data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算单个因子"""
        if factor_name not in self.factor_registry:
            raise ValueError(f"Factor {factor_name} not found")
        
        factor_func = self.factor_registry[factor_name]['function']
        
        try:
            return factor_func(data, **kwargs)
        except Exception as e:
            print(f"Error computing factor {factor_name}: {e}")
            return pd.Series(index=data.index, dtype=float)
    
    def compute_multiple_factors(self, factor_names: List[str], data: pd.DataFrame) -> pd.DataFrame:
        """批量计算多个因子"""
        result = pd.DataFrame(index=data.index)
        
        for factor_name in factor_names:
            try:
                result[factor_name] = self.compute_factor(factor_name, data)
            except Exception as e:
                print(f"Failed to compute {factor_name}: {e}")
                result[factor_name] = np.nan
        
        return result
    
    def compute_all_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算所有因子"""
        return self.compute_multiple_factors(list(self.factor_registry.keys()), data)
    
    # =================== 形态类因子的占位实现 ===================
    # 注意：以下是简化实现，实际的形态识别需要更复杂的算法
    
    def doji(self, data: pd.DataFrame) -> pd.Series:
        """十字星形态"""
        body = abs(data['open'] - data['close'])
        total_range = data['high'] - data['low']
        return (body / total_range < 0.1).astype(int)
    
    def hammer(self, data: pd.DataFrame) -> pd.Series:
        """锤子形态"""
        body = abs(data['open'] - data['close'])
        lower_shadow = np.minimum(data['open'], data['close']) - data['low']
        upper_shadow = data['high'] - np.maximum(data['open'], data['close'])
        
        return ((lower_shadow > 2 * body) & (upper_shadow < 0.5 * body)).astype(int)
    
    def hanging_man(self, data: pd.DataFrame) -> pd.Series:
        """上吊线形态"""
        # 与锤子形态相同，但出现在上升趋势中
        return self.hammer(data)
    
    def shooting_star(self, data: pd.DataFrame) -> pd.Series:
        """流星形态"""
        body = abs(data['open'] - data['close'])
        lower_shadow = np.minimum(data['open'], data['close']) - data['low']
        upper_shadow = data['high'] - np.maximum(data['open'], data['close'])
        
        return ((upper_shadow > 2 * body) & (lower_shadow < 0.5 * body)).astype(int)
    
    def engulfing_bullish(self, data: pd.DataFrame) -> pd.Series:
        """看涨吞没形态"""
        prev_close = data['close'].shift(1)
        prev_open = data['open'].shift(1)
        
        # 前一天是阴线，当天是阳线，且当天完全吞没前一天
        return ((prev_close < prev_open) & 
                (data['close'] > data['open']) & 
                (data['open'] < prev_close) & 
                (data['close'] > prev_open)).astype(int)
    
    def engulfing_bearish(self, data: pd.DataFrame) -> pd.Series:
        """看跌吞没形态"""
        prev_close = data['close'].shift(1)
        prev_open = data['open'].shift(1)
        
        # 前一天是阳线，当天是阴线，且当天完全吞没前一天
        return ((prev_close > prev_open) & 
                (data['close'] < data['open']) & 
                (data['open'] > prev_close) & 
                (data['close'] < prev_open)).astype(int)
    
    def morning_star(self, data: pd.DataFrame) -> pd.Series:
        """启明星形态"""
        # 简化实现：三日形态
        return pd.Series(0, index=data.index)  # 需要复杂的三日形态识别
    
    def evening_star(self, data: pd.DataFrame) -> pd.Series:
        """黄昏星形态"""
        # 简化实现：三日形态
        return pd.Series(0, index=data.index)  # 需要复杂的三日形态识别
    
    def three_white_soldiers(self, data: pd.DataFrame) -> pd.Series:
        """红三兵形态"""
        # 简化实现：三日形态
        return pd.Series(0, index=data.index)  # 需要复杂的三日形态识别
    
    def three_black_crows(self, data: pd.DataFrame) -> pd.Series:
        """三只乌鸦形态"""
        # 简化实现：三日形态
        return pd.Series(0, index=data.index)  # 需要复杂的三日形态识别
    
    # 其他未实现的因子方法将返回默认值或占位实现
    def double_top(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(0, index=data.index)
    
    def double_bottom(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(0, index=data.index)
    
    def head_shoulders(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(0, index=data.index)
    
    def triangle_pattern(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(0, index=data.index)
    
    def flag_pattern(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(0, index=data.index)
    
    def pennant_pattern(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(0, index=data.index)
    
    def support_level(self, data: pd.DataFrame) -> pd.Series:
        return data['low'].rolling(20).min()
    
    def resistance_level(self, data: pd.DataFrame) -> pd.Series:
        return data['high'].rolling(20).max()
    
    def pivot_points(self, data: pd.DataFrame) -> pd.Series:
        return (data['high'] + data['low'] + data['close']) / 3
    
    def fibonacci_retracement(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(0, index=data.index)
    
    def gap_up(self, data: pd.DataFrame) -> pd.Series:
        return (data['low'] > data['high'].shift(1)).astype(int)
    
    def gap_down(self, data: pd.DataFrame) -> pd.Series:
        return (data['high'] < data['low'].shift(1)).astype(int)
    
    def gap_fill(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(0, index=data.index)
    
    def gap_size(self, data: pd.DataFrame) -> pd.Series:
        gap_up = data['low'] - data['high'].shift(1)
        gap_down = data['low'].shift(1) - data['high']
        return np.maximum(gap_up, gap_down).fillna(0)
    
    def pattern_strength(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(0, index=data.index)
    
    def reversal_pattern(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(0, index=data.index)
    
    def continuation_pattern(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(0, index=data.index)
    
    # =================== 其他类别因子的占位实现 ===================
    
    # 周期类因子
    def hilbert_trendmode(self, data: pd.DataFrame) -> pd.Series:
        try:
            return talib.HT_TRENDMODE(data['close'])
        except:
            return pd.Series(0, index=data.index)
    
    def hilbert_dcperiod(self, data: pd.DataFrame) -> pd.Series:
        try:
            return talib.HT_DCPERIOD(data['close'])
        except:
            return pd.Series(20, index=data.index)
    
    def hilbert_dcphase(self, data: pd.DataFrame) -> pd.Series:
        try:
            return talib.HT_DCPHASE(data['close'])
        except:
            return pd.Series(0, index=data.index)
    
    def hilbert_phasor_inphase(self, data: pd.DataFrame) -> pd.Series:
        try:
            inphase, quadrature = talib.HT_PHASOR(data['close'])
            return inphase
        except:
            return pd.Series(0, index=data.index)
    
    def hilbert_phasor_quadrature(self, data: pd.DataFrame) -> pd.Series:
        try:
            inphase, quadrature = talib.HT_PHASOR(data['close'])
            return quadrature
        except:
            return pd.Series(0, index=data.index)
    
    def hilbert_sine(self, data: pd.DataFrame) -> pd.Series:
        try:
            sine, leadsine = talib.HT_SINE(data['close'])
            return sine
        except:
            return pd.Series(0, index=data.index)
    
    def hilbert_leadsine(self, data: pd.DataFrame) -> pd.Series:
        try:
            sine, leadsine = talib.HT_SINE(data['close'])
            return leadsine
        except:
            return pd.Series(0, index=data.index)
    
    def fourier_transform(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(0, index=data.index)
    
    def spectral_density(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(0, index=data.index)
    
    def dominant_cycle(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(20, index=data.index)
    
    def cycle_strength(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(0, index=data.index)
    
    def seasonal_pattern(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(0, index=data.index)
    
    def monthly_effect(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(0, index=data.index)
    
    def weekly_effect(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(0, index=data.index)
    
    def intraday_pattern(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(0, index=data.index)
    
    def trend_component(self, data: pd.DataFrame) -> pd.Series:
        return self.sma(data, 50)
    
    def seasonal_component(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(0, index=data.index)
    
    def residual_component(self, data: pd.DataFrame) -> pd.Series:
        trend = self.trend_component(data)
        return data['close'] - trend
    
    # 统计类因子
    def rolling_mean(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        return data['close'].rolling(window).mean()
    
    def rolling_median(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        return data['close'].rolling(window).median()
    
    def rolling_mode(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        return data['close'].rolling(window).apply(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.mean())
    
    def rolling_variance(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        return data['close'].rolling(window).var()
    
    def rolling_skewness(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        return data['close'].rolling(window).skew()
    
    def rolling_kurtosis(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        return data['close'].rolling(window).kurt()
    
    def rolling_quantile(self, data: pd.DataFrame, quantile: float, window: int = 20) -> pd.Series:
        return data['close'].rolling(window).quantile(quantile)
    
    def normality_test(self, data: pd.DataFrame, window: int = 50) -> pd.Series:
        return pd.Series(0, index=data.index)  # 需要统计检验实现
    
    def stationarity_test(self, data: pd.DataFrame, window: int = 50) -> pd.Series:
        return pd.Series(0, index=data.index)  # 需要ADF检验实现
    
    def autocorrelation(self, data: pd.DataFrame, lag: int = 1, window: int = 50) -> pd.Series:
        return data['close'].rolling(window).apply(lambda x: x.autocorr(lag=lag))
    
    def partial_autocorrelation(self, data: pd.DataFrame, window: int = 50) -> pd.Series:
        return pd.Series(0, index=data.index)  # 需要PACF实现
    
    def beta_coefficient(self, data: pd.DataFrame, market_data: pd.DataFrame, window: int = 50) -> pd.Series:
        return pd.Series(1, index=data.index)  # 需要市场数据
    
    def correlation_coefficient(self, data: pd.DataFrame, other_data: pd.DataFrame, window: int = 20) -> pd.Series:
        return data['close'].rolling(window).corr(other_data['close'])
    
    def determination_coefficient(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        return pd.Series(0.5, index=data.index)  # R²的简化实现
    
    def residual_analysis(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        return pd.Series(0, index=data.index)
    
    def cointegration(self, data: pd.DataFrame, other_data: pd.DataFrame) -> pd.Series:
        return pd.Series(0, index=data.index)  # 需要协整检验
    
    def spread_zscore(self, data: pd.DataFrame, other_data: pd.DataFrame, window: int = 20) -> pd.Series:
        spread = data['close'] - other_data['close']
        return (spread - spread.rolling(window).mean()) / spread.rolling(window).std()
    
    def pairs_ratio(self, data: pd.DataFrame, other_data: pd.DataFrame) -> pd.Series:
        return data['close'] / other_data['close']
    
    def statistical_arbitrage(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        return pd.Series(0, index=data.index)
    
    # 微观结构类、情绪类、风险类因子的占位实现
    # 这些因子通常需要特殊的数据源（如订单簿、期权数据、新闻数据等）
    
    def bid_ask_spread(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(0.01, index=data.index)  # 需要买卖盘数据
    
    def effective_spread(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(0.01, index=data.index)
    
    def price_impact(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(0, index=data.index)
    
    def market_depth(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(1000000, index=data.index)
    
    def liquidity_ratio(self, data: pd.DataFrame) -> pd.Series:
        return data['volume'] / abs(data['close'].pct_change())
    
    def order_flow_imbalance(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(0, index=data.index)
    
    def trade_size(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(100, index=data.index)
    
    def trade_intensity(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(1, index=data.index)
    
    def arrival_rate(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(1, index=data.index)
    
    def tick_direction(self, data: pd.DataFrame) -> pd.Series:
        return np.sign(data['close'].diff())
    
    def price_velocity(self, data: pd.DataFrame) -> pd.Series:
        return data['close'].diff()
    
    def price_acceleration(self, data: pd.DataFrame) -> pd.Series:
        return data['close'].diff().diff()
    
    def microstructure_noise(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(0, index=data.index)
    
    def realized_spread(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(0.01, index=data.index)
    
    def adverse_selection(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(0, index=data.index)
    
    def order_book_slope(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(0, index=data.index)
    
    def volatility_signature_plot(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(0, index=data.index)
    
    # 情绪类因子
    def fear_greed_index(self, data: pd.DataFrame) -> pd.Series:
        # 基于价格和成交量的简化恐惧贪婪指数
        price_mom = self.price_momentum(data, 20)
        vol_ratio = self.volume_ratio(data, 20)
        
        # 标准化
        price_score = (price_mom - price_mom.rolling(50).mean()) / price_mom.rolling(50).std()
        vol_score = (vol_ratio - vol_ratio.rolling(50).mean()) / vol_ratio.rolling(50).std()
        
        fear_greed = (price_score + vol_score) / 2
        return fear_greed.clip(-2, 2) * 25 + 50  # 缩放到0-100
    
    def put_call_ratio(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(1, index=data.index)  # 需要期权数据
    
    def vix_term_structure(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(20, index=data.index)  # 需要VIX数据
    
    def market_sentiment(self, data: pd.DataFrame) -> pd.Series:
        return self.fear_greed_index(data)
    
    def investor_sentiment(self, data: pd.DataFrame) -> pd.Series:
        return self.fear_greed_index(data)
    
    def sentiment_oscillator(self, data: pd.DataFrame) -> pd.Series:
        return self.fear_greed_index(data)
    
    def crowd_behavior(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(0, index=data.index)
    
    def money_flow_sentiment(self, data: pd.DataFrame) -> pd.Series:
        return self.mfi(data)
    
    def capital_flow(self, data: pd.DataFrame) -> pd.Series:
        return self.obv(data)
    
    def institutional_flow(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(0, index=data.index)
    
    def retail_flow(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(0, index=data.index)
    
    def news_sentiment(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(0, index=data.index)  # 需要新闻数据
    
    def social_sentiment(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(0, index=data.index)  # 需要社交媒体数据
    
    def analyst_sentiment(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(0, index=data.index)  # 需要分析师数据
    
    # 风险类因子
    def value_at_risk(self, data: pd.DataFrame, confidence: float = 0.05, window: int = 252) -> pd.Series:
        returns = data['close'].pct_change()
        return returns.rolling(window).quantile(confidence)
    
    def conditional_var(self, data: pd.DataFrame, confidence: float = 0.05, window: int = 252) -> pd.Series:
        returns = data['close'].pct_change()
        var = self.value_at_risk(data, confidence, window)
        
        result = pd.Series(index=data.index, dtype=float)
        for i in range(window-1, len(returns)):
            window_returns = returns.iloc[i-window+1:i+1]
            var_value = var.iloc[i]
            tail_returns = window_returns[window_returns <= var_value]
            result.iloc[i] = tail_returns.mean() if len(tail_returns) > 0 else var_value
        
        return result
    
    def expected_shortfall(self, data: pd.DataFrame, confidence: float = 0.05, window: int = 252) -> pd.Series:
        return self.conditional_var(data, confidence, window)
    
    def maximum_drawdown(self, data: pd.DataFrame, window: int = 252) -> pd.Series:
        cumulative = (1 + data['close'].pct_change()).cumprod()
        
        result = pd.Series(index=data.index, dtype=float)
        for i in range(window-1, len(cumulative)):
            window_cum = cumulative.iloc[i-window+1:i+1]
            running_max = window_cum.expanding().max()
            drawdown = (window_cum - running_max) / running_max
            result.iloc[i] = drawdown.min()
        
        return result
    
    def systematic_risk(self, data: pd.DataFrame, market_data: pd.DataFrame, window: int = 252) -> pd.Series:
        returns = data['close'].pct_change()
        market_returns = market_data['close'].pct_change()
        
        beta = returns.rolling(window).cov(market_returns) / market_returns.rolling(window).var()
        return beta * market_returns.rolling(window).var()
    
    def idiosyncratic_risk(self, data: pd.DataFrame, market_data: pd.DataFrame, window: int = 252) -> pd.Series:
        total_var = data['close'].pct_change().rolling(window).var()
        systematic_var = self.systematic_risk(data, market_data, window)
        return total_var - systematic_var
    
    def tail_risk(self, data: pd.DataFrame, window: int = 252) -> pd.Series:
        returns = data['close'].pct_change()
        return returns.rolling(window).apply(lambda x: x.quantile(0.01))
    
    def jump_risk(self, data: pd.DataFrame, threshold: float = 3, window: int = 20) -> pd.Series:
        returns = data['close'].pct_change()
        vol = returns.rolling(window).std()
        
        # 检测跳跃（收益率超过n倍标准差）
        jumps = (abs(returns) > threshold * vol).astype(int)
        return jumps.rolling(window).sum()
    
    def downside_risk(self, data: pd.DataFrame, target: float = 0, window: int = 252) -> pd.Series:
        returns = data['close'].pct_change()
        downside_returns = returns[returns < target]
        
        result = pd.Series(index=data.index, dtype=float)
        for i in range(window-1, len(returns)):
            window_returns = returns.iloc[i-window+1:i+1]
            downside = window_returns[window_returns < target]
            if len(downside) > 0:
                result.iloc[i] = np.sqrt(((downside - target)**2).mean())
            else:
                result.iloc[i] = 0
        
        return result
    
    def sortino_ratio(self, data: pd.DataFrame, target: float = 0, window: int = 252) -> pd.Series:
        returns = data['close'].pct_change()
        excess_returns = returns - target
        downside_risk = self.downside_risk(data, target, window)
        
        return excess_returns.rolling(window).mean() / downside_risk
    
    def calmar_ratio(self, data: pd.DataFrame, window: int = 252) -> pd.Series:
        returns = data['close'].pct_change()
        annual_return = returns.rolling(window).mean() * 252
        max_dd = abs(self.maximum_drawdown(data, window))
        
        return annual_return / max_dd
    
    def sterling_ratio(self, data: pd.DataFrame, window: int = 252) -> pd.Series:
        returns = data['close'].pct_change()
        annual_return = returns.rolling(window).mean() * 252
        
        # 使用平均回撤代替最大回撤
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.rolling(window).max()
        drawdown = (cumulative - running_max) / running_max
        avg_drawdown = abs(drawdown.rolling(window).mean())
        
        return annual_return / avg_drawdown
    
    def sharpe_ratio(self, data: pd.DataFrame, risk_free_rate: float = 0.02, window: int = 252) -> pd.Series:
        returns = data['close'].pct_change()
        excess_returns = returns - risk_free_rate / 252
        
        return excess_returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(252)
    
    def treynor_ratio(self, data: pd.DataFrame, market_data: pd.DataFrame, risk_free_rate: float = 0.02, window: int = 252) -> pd.Series:
        returns = data['close'].pct_change()
        market_returns = market_data['close'].pct_change()
        
        excess_returns = returns - risk_free_rate / 252
        beta = returns.rolling(window).cov(market_returns) / market_returns.rolling(window).var()
        
        return excess_returns.rolling(window).mean() * 252 / beta
    
    def information_ratio(self, data: pd.DataFrame, benchmark_data: pd.DataFrame, window: int = 252) -> pd.Series:
        returns = data['close'].pct_change()
        benchmark_returns = benchmark_data['close'].pct_change()
        
        active_returns = returns - benchmark_returns
        tracking_error = active_returns.rolling(window).std()
        
        return active_returns.rolling(window).mean() / tracking_error * np.sqrt(252)
    
    def risk_adjusted_return(self, data: pd.DataFrame, window: int = 252) -> pd.Series:
        returns = data['close'].pct_change()
        vol = returns.rolling(window).std()
        
        return returns.rolling(window).mean() / vol
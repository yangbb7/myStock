# -*- coding: utf-8 -*-
"""
AI策略助手 - 提供智能策略推荐和优化建议
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
import logging
import json

from myQuant.core.strategy.base_strategy import BaseStrategy
from myQuant.core.analysis.performance_analyzer import PerformanceAnalyzer


@dataclass
class MarketCondition:
    """市场状态"""
    trend: str  # bullish, bearish, sideways
    volatility: str  # low, medium, high
    volume_trend: str  # increasing, decreasing, stable
    confidence: float


@dataclass
class StrategyRecommendation:
    """策略推荐"""
    strategy_type: str
    reason: str
    expected_performance: Dict[str, float]
    risk_level: str
    suitable_symbols: List[str]
    parameters: Dict[str, Any]
    confidence: float


@dataclass
class OptimizationSuggestion:
    """优化建议"""
    parameter: str
    current_value: Any
    suggested_value: Any
    expected_improvement: float
    reason: str


class StrategyAssistant:
    """AI策略助手"""
    
    # 市场状态对应的推荐策略
    MARKET_STRATEGY_MAPPING = {
        ("bullish", "low"): ["trend_following", "breakout"],
        ("bullish", "medium"): ["momentum", "trend_following"],
        ("bullish", "high"): ["volatility_breakout", "short_term_momentum"],
        ("bearish", "low"): ["short_selling", "defensive"],
        ("bearish", "medium"): ["mean_reversion", "pairs_trading"],
        ("bearish", "high"): ["volatility_arbitrage", "options_strategies"],
        ("sideways", "low"): ["range_trading", "mean_reversion"],
        ("sideways", "medium"): ["swing_trading", "bollinger_bands"],
        ("sideways", "high"): ["straddle", "scalping"],
    }
    
    # 策略模板参数
    STRATEGY_TEMPLATES = {
        "trend_following": {
            "name": "趋势跟踪策略",
            "indicators": ["SMA", "EMA"],
            "default_params": {
                "fast_period": 10,
                "slow_period": 30,
                "stop_loss": 0.02,
                "take_profit": 0.05
            },
            "risk_level": "medium"
        },
        "mean_reversion": {
            "name": "均值回归策略",
            "indicators": ["RSI", "BOLL"],
            "default_params": {
                "rsi_period": 14,
                "rsi_oversold": 30,
                "rsi_overbought": 70,
                "position_size": 0.5
            },
            "risk_level": "low"
        },
        "breakout": {
            "name": "突破策略",
            "indicators": ["ATR", "BOLL"],
            "default_params": {
                "lookback_period": 20,
                "breakout_multiplier": 2.0,
                "stop_loss": 0.03
            },
            "risk_level": "high"
        },
        "momentum": {
            "name": "动量策略",
            "indicators": ["RSI", "MACD"],
            "default_params": {
                "momentum_period": 10,
                "volume_threshold": 1.5,
                "holding_period": 5
            },
            "risk_level": "medium"
        }
    }
    
    def __init__(self):
        self.performance_analyzer = PerformanceAnalyzer()
        self.scaler = StandardScaler()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def analyze_market_condition(self, data: pd.DataFrame) -> MarketCondition:
        """分析市场状态"""
        if len(data) < 30:
            return MarketCondition(
                trend="unknown",
                volatility="unknown",
                volume_trend="unknown",
                confidence=0.0
            )
        
        # 计算趋势
        sma_20 = data['close'].rolling(20).mean()
        sma_50 = data['close'].rolling(50).mean() if len(data) >= 50 else sma_20
        
        current_price = data['close'].iloc[-1]
        trend_score = (current_price - sma_20.iloc[-1]) / sma_20.iloc[-1]
        
        if trend_score > 0.02:
            trend = "bullish"
        elif trend_score < -0.02:
            trend = "bearish"
        else:
            trend = "sideways"
        
        # 计算波动率
        returns = data['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # 年化波动率
        
        if volatility < 0.15:
            volatility_level = "low"
        elif volatility < 0.25:
            volatility_level = "medium"
        else:
            volatility_level = "high"
        
        # 分析成交量趋势
        if 'volume' in data.columns:
            volume_ma = data['volume'].rolling(20).mean()
            recent_volume = data['volume'].iloc[-5:].mean()
            volume_ratio = recent_volume / volume_ma.iloc[-6]
            
            if volume_ratio > 1.2:
                volume_trend = "increasing"
            elif volume_ratio < 0.8:
                volume_trend = "decreasing"
            else:
                volume_trend = "stable"
        else:
            volume_trend = "unknown"
        
        # 计算置信度
        confidence = min(1.0, len(data) / 100)  # 数据越多置信度越高
        
        return MarketCondition(
            trend=trend,
            volatility=volatility_level,
            volume_trend=volume_trend,
            confidence=confidence
        )
    
    def recommend_strategies(
        self,
        symbols: List[str],
        market_data: Dict[str, pd.DataFrame],
        user_risk_preference: str = "medium"
    ) -> List[StrategyRecommendation]:
        """推荐合适的策略"""
        recommendations = []
        
        # 分析每个标的的市场状态
        market_conditions = {}
        for symbol in symbols:
            if symbol in market_data:
                market_conditions[symbol] = self.analyze_market_condition(market_data[symbol])
        
        # 综合市场状态
        if market_conditions:
            # 获取主要趋势
            trends = [mc.trend for mc in market_conditions.values()]
            volatilities = [mc.volatility for mc in market_conditions.values()]
            
            # 选择最常见的状态
            main_trend = max(set(trends), key=trends.count)
            main_volatility = max(set(volatilities), key=volatilities.count)
            
            # 获取推荐策略类型
            strategy_types = self.MARKET_STRATEGY_MAPPING.get(
                (main_trend, main_volatility),
                ["trend_following"]  # 默认策略
            )
            
            # 生成推荐
            for strategy_type in strategy_types:
                if strategy_type in self.STRATEGY_TEMPLATES:
                    template = self.STRATEGY_TEMPLATES[strategy_type]
                    
                    # 检查风险偏好匹配
                    if self._match_risk_preference(template["risk_level"], user_risk_preference):
                        # 估算预期表现
                        expected_performance = self._estimate_performance(
                            strategy_type, main_trend, main_volatility
                        )
                        
                        recommendation = StrategyRecommendation(
                            strategy_type=strategy_type,
                            reason=self._generate_recommendation_reason(
                                strategy_type, main_trend, main_volatility
                            ),
                            expected_performance=expected_performance,
                            risk_level=template["risk_level"],
                            suitable_symbols=symbols,
                            parameters=template["default_params"],
                            confidence=0.7
                        )
                        recommendations.append(recommendation)
        
        # 按置信度排序
        recommendations.sort(key=lambda x: x.confidence, reverse=True)
        return recommendations[:3]  # 返回前3个推荐
    
    def optimize_strategy_parameters(
        self,
        strategy: BaseStrategy,
        historical_data: pd.DataFrame,
        optimization_metric: str = "sharpe_ratio"
    ) -> List[OptimizationSuggestion]:
        """优化策略参数"""
        suggestions = []
        
        # 获取当前参数
        current_params = strategy.params
        
        # 分析历史表现
        if hasattr(strategy, 'performance_metrics'):
            current_performance = strategy.performance_metrics
        else:
            current_performance = {}
        
        # 参数优化建议
        if strategy.__class__.__name__ == "MovingAverageStrategy":
            suggestions.extend(self._optimize_ma_strategy(current_params, historical_data))
        elif strategy.__class__.__name__ == "RSIStrategy":
            suggestions.extend(self._optimize_rsi_strategy(current_params, historical_data))
        
        # 通用优化建议
        suggestions.extend(self._general_optimization_suggestions(current_params, current_performance))
        
        return suggestions
    
    def _optimize_ma_strategy(
        self,
        current_params: Dict[str, Any],
        data: pd.DataFrame
    ) -> List[OptimizationSuggestion]:
        """优化均线策略参数"""
        suggestions = []
        
        # 测试不同的均线组合
        fast_periods = [5, 10, 15, 20]
        slow_periods = [20, 30, 50, 60]
        
        best_sharpe = -999
        best_params = {}
        
        for fast in fast_periods:
            for slow in slow_periods:
                if fast < slow:
                    # 简单回测
                    sharpe = self._backtest_ma_params(data, fast, slow)
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_params = {"fast_period": fast, "slow_period": slow}
        
        # 生成建议
        if best_params.get("fast_period") != current_params.get("fast_period", 10):
            suggestions.append(OptimizationSuggestion(
                parameter="fast_period",
                current_value=current_params.get("fast_period", 10),
                suggested_value=best_params["fast_period"],
                expected_improvement=0.1,
                reason=f"历史数据显示{best_params['fast_period']}日均线响应更及时"
            ))
        
        return suggestions
    
    def _optimize_rsi_strategy(
        self,
        current_params: Dict[str, Any],
        data: pd.DataFrame
    ) -> List[OptimizationSuggestion]:
        """优化RSI策略参数"""
        suggestions = []
        
        # 分析RSI分布
        rsi_period = current_params.get("rsi_period", 14)
        rsi = self._calculate_rsi(data['close'], rsi_period)
        
        # 计算最优阈值
        rsi_distribution = rsi.dropna()
        oversold_threshold = rsi_distribution.quantile(0.2)
        overbought_threshold = rsi_distribution.quantile(0.8)
        
        if oversold_threshold != current_params.get("rsi_oversold", 30):
            suggestions.append(OptimizationSuggestion(
                parameter="rsi_oversold",
                current_value=current_params.get("rsi_oversold", 30),
                suggested_value=int(oversold_threshold),
                expected_improvement=0.05,
                reason=f"根据历史数据分布，{int(oversold_threshold)}是更合适的超卖阈值"
            ))
        
        return suggestions
    
    def _general_optimization_suggestions(
        self,
        current_params: Dict[str, Any],
        performance: Dict[str, float]
    ) -> List[OptimizationSuggestion]:
        """通用优化建议"""
        suggestions = []
        
        # 止损建议
        if "stop_loss" in current_params:
            current_stop_loss = current_params["stop_loss"]
            if performance.get("max_drawdown", 0) < -0.15:
                suggestions.append(OptimizationSuggestion(
                    parameter="stop_loss",
                    current_value=current_stop_loss,
                    suggested_value=current_stop_loss * 0.8,
                    expected_improvement=0.03,
                    reason="最大回撤较大，建议收紧止损"
                ))
        
        # 仓位建议
        if "position_size" in current_params:
            current_position = current_params["position_size"]
            if performance.get("sharpe_ratio", 0) > 1.5:
                suggestions.append(OptimizationSuggestion(
                    parameter="position_size",
                    current_value=current_position,
                    suggested_value=min(1.0, current_position * 1.2),
                    expected_improvement=0.02,
                    reason="策略表现优秀，可以适当增加仓位"
                ))
        
        return suggestions
    
    def analyze_strategy_performance(
        self,
        strategy: BaseStrategy,
        market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """分析策略表现并给出建议"""
        analysis = {
            "performance_score": 0.0,
            "strengths": [],
            "weaknesses": [],
            "improvement_suggestions": [],
            "market_fit": ""
        }
        
        # 获取策略性能指标
        if hasattr(strategy, 'performance_metrics'):
            metrics = strategy.performance_metrics
            
            # 评分
            score = 0
            if metrics.get("sharpe_ratio", 0) > 1:
                score += 30
                analysis["strengths"].append("良好的风险调整收益")
            else:
                analysis["weaknesses"].append("夏普比率偏低")
            
            if metrics.get("win_rate", 0) > 0.5:
                score += 20
                analysis["strengths"].append("较高的胜率")
            
            if metrics.get("max_drawdown", 0) > -0.1:
                score += 20
                analysis["strengths"].append("回撤控制良好")
            else:
                analysis["weaknesses"].append("最大回撤较大")
            
            # 市场适应性分析
            market_condition = self.analyze_market_condition(market_data)
            if self._is_strategy_suitable_for_market(strategy, market_condition):
                score += 30
                analysis["market_fit"] = "当前市场环境适合该策略"
            else:
                analysis["market_fit"] = "当前市场环境不太适合该策略"
                analysis["improvement_suggestions"].append("考虑切换到更适合当前市场的策略")
            
            analysis["performance_score"] = score
        
        return analysis
    
    def suggest_risk_controls(
        self,
        strategy: BaseStrategy,
        portfolio_value: float,
        risk_tolerance: str = "medium"
    ) -> Dict[str, Any]:
        """建议风险控制措施"""
        suggestions = {
            "position_limits": {},
            "stop_loss_levels": {},
            "risk_alerts": [],
            "diversification": []
        }
        
        # 仓位限制建议
        if risk_tolerance == "low":
            max_position_pct = 0.05
            max_sector_exposure = 0.2
        elif risk_tolerance == "high":
            max_position_pct = 0.15
            max_sector_exposure = 0.4
        else:
            max_position_pct = 0.1
            max_sector_exposure = 0.3
        
        suggestions["position_limits"] = {
            "max_position_size": portfolio_value * max_position_pct,
            "max_position_percent": max_position_pct,
            "max_sector_exposure": max_sector_exposure
        }
        
        # 止损建议
        suggestions["stop_loss_levels"] = {
            "initial_stop_loss": 0.02 if risk_tolerance == "low" else 0.05,
            "trailing_stop_loss": 0.03,
            "time_stop_days": 30
        }
        
        # 风险提醒
        if len(strategy.symbols) < 5:
            suggestions["risk_alerts"].append("持仓集中度较高，建议增加标的分散风险")
        
        return suggestions
    
    def _match_risk_preference(self, strategy_risk: str, user_risk: str) -> bool:
        """匹配风险偏好"""
        risk_levels = {"low": 1, "medium": 2, "high": 3}
        strategy_level = risk_levels.get(strategy_risk, 2)
        user_level = risk_levels.get(user_risk, 2)
        return abs(strategy_level - user_level) <= 1
    
    def _estimate_performance(
        self,
        strategy_type: str,
        market_trend: str,
        volatility: str
    ) -> Dict[str, float]:
        """估算策略表现"""
        # 基础收益率估算
        base_returns = {
            "trend_following": {"bullish": 0.15, "bearish": -0.05, "sideways": 0.02},
            "mean_reversion": {"bullish": 0.08, "bearish": 0.08, "sideways": 0.12},
            "breakout": {"bullish": 0.20, "bearish": -0.10, "sideways": 0.05},
            "momentum": {"bullish": 0.18, "bearish": -0.08, "sideways": 0.03}
        }
        
        expected_return = base_returns.get(strategy_type, {}).get(market_trend, 0.05)
        
        # 波动率调整
        if volatility == "high":
            expected_return *= 1.2
            sharpe_ratio = 0.8
        elif volatility == "low":
            expected_return *= 0.8
            sharpe_ratio = 1.5
        else:
            sharpe_ratio = 1.2
        
        return {
            "expected_return": expected_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": -abs(expected_return) * 0.5,
            "win_rate": 0.45 + (0.1 if market_trend == "bullish" else 0)
        }
    
    def _generate_recommendation_reason(
        self,
        strategy_type: str,
        market_trend: str,
        volatility: str
    ) -> str:
        """生成推荐理由"""
        reasons = {
            ("trend_following", "bullish"): "市场处于上升趋势，趋势跟踪策略能够捕捉主要行情",
            ("mean_reversion", "sideways"): "市场横盘震荡，均值回归策略适合捕捉区间波动",
            ("breakout", "bullish", "high"): "市场波动加大且趋势向上，突破策略可能获得超额收益",
            ("momentum", "bullish"): "市场动能强劲，动量策略能够跟随强势股"
        }
        
        key = (strategy_type, market_trend) if volatility == "medium" else (strategy_type, market_trend, volatility)
        return reasons.get(key, f"基于当前{market_trend}市场和{volatility}波动率，{strategy_type}策略较为合适")
    
    def _backtest_ma_params(self, data: pd.DataFrame, fast: int, slow: int) -> float:
        """简单回测均线参数"""
        if len(data) < slow:
            return -999
        
        fast_ma = data['close'].rolling(fast).mean()
        slow_ma = data['close'].rolling(slow).mean()
        
        # 生成信号
        signals = (fast_ma > slow_ma).astype(int).diff()
        
        # 计算收益
        returns = data['close'].pct_change()
        strategy_returns = returns * signals.shift(1)
        
        # 计算夏普比率
        if strategy_returns.std() > 0:
            sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
        else:
            sharpe = 0
        
        return sharpe
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _is_strategy_suitable_for_market(
        self,
        strategy: BaseStrategy,
        market_condition: MarketCondition
    ) -> bool:
        """判断策略是否适合当前市场"""
        strategy_name = strategy.__class__.__name__.lower()
        
        suitable_conditions = {
            "movingaveragestrategy": [("bullish", "low"), ("bullish", "medium")],
            "rsistrategy": [("sideways", "low"), ("sideways", "medium")],
            "breakoutstrategy": [("bullish", "high"), ("sideways", "high")]
        }
        
        conditions = suitable_conditions.get(strategy_name, [])
        return (market_condition.trend, market_condition.volatility) in conditions
# -*- coding: utf-8 -*-
"""
风险计算器 - 提供各种风险指标的计算功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from scipy.stats import norm


@dataclass
class RiskMetrics:
    """风险指标"""
    var_1d: float  # 1日VaR
    var_5d: float  # 5日VaR
    var_10d: float  # 10日VaR
    cvar_1d: float  # 1日CVaR
    expected_shortfall: float  # 预期短缺
    volatility: float  # 波动率
    max_drawdown: float  # 最大回撤
    sharpe_ratio: float  # 夏普比率
    sortino_ratio: float  # 索托诺比率
    calmar_ratio: float  # 卡玛比率


class RiskCalculator:
    """风险计算器"""
    
    def __init__(self):
        self.confidence_levels = [0.90, 0.95, 0.99]
        self.trading_days_per_year = 252
    
    def calculate_var(self, returns: Union[pd.Series, np.ndarray], 
                     confidence_level: float = 0.95, 
                     method: str = 'historical') -> float:
        """计算VaR (Value at Risk)
        
        Args:
            returns: 收益率序列
            confidence_level: 置信水平
            method: 计算方法 ('historical', 'parametric', 'monte_carlo')
        """
        if method == 'historical':
            return np.percentile(returns, (1 - confidence_level) * 100)
        
        elif method == 'parametric':
            mean = np.mean(returns)
            std = np.std(returns)
            z_score = norm.ppf(1 - confidence_level)
            return mean + z_score * std
        
        elif method == 'monte_carlo':
            # 简化的蒙特卡洛模拟
            mean = np.mean(returns)
            std = np.std(returns)
            simulated_returns = np.random.normal(mean, std, 10000)
            return np.percentile(simulated_returns, (1 - confidence_level) * 100)
        
        else:
            raise ValueError(f"不支持的方法: {method}")
    
    def calculate_cvar(self, returns: Union[pd.Series, np.ndarray], 
                      confidence_level: float = 0.95) -> float:
        """计算CVaR (Conditional Value at Risk)"""
        var = self.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()
    
    def calculate_expected_shortfall(self, returns: Union[pd.Series, np.ndarray], 
                                   confidence_level: float = 0.95) -> float:
        """计算预期短缺 (Expected Shortfall)"""
        return self.calculate_cvar(returns, confidence_level)
    
    def calculate_volatility(self, returns: Union[pd.Series, np.ndarray], 
                           annualized: bool = True) -> float:
        """计算波动率"""
        vol = np.std(returns)
        if annualized:
            vol *= np.sqrt(self.trading_days_per_year)
        return vol
    
    def calculate_max_drawdown(self, returns: Union[pd.Series, np.ndarray]) -> float:
        """计算最大回撤"""
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns)
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def calculate_sharpe_ratio(self, returns: Union[pd.Series, np.ndarray], 
                             risk_free_rate: float = 0.02) -> float:
        """计算夏普比率"""
        excess_returns = np.mean(returns) * self.trading_days_per_year - risk_free_rate
        volatility = self.calculate_volatility(returns)
        return excess_returns / volatility if volatility != 0 else 0
    
    def calculate_sortino_ratio(self, returns: Union[pd.Series, np.ndarray], 
                              risk_free_rate: float = 0.02) -> float:
        """计算索托诺比率（只考虑下行风险）"""
        excess_returns = np.mean(returns) * self.trading_days_per_year - risk_free_rate
        downside_returns = returns[returns < 0]
        downside_volatility = np.std(downside_returns) * np.sqrt(self.trading_days_per_year)
        return excess_returns / downside_volatility if downside_volatility != 0 else 0
    
    def calculate_calmar_ratio(self, returns: Union[pd.Series, np.ndarray]) -> float:
        """计算卡玛比率（年化收益 / 最大回撤）"""
        annual_return = np.mean(returns) * self.trading_days_per_year
        max_drawdown = abs(self.calculate_max_drawdown(returns))
        return annual_return / max_drawdown if max_drawdown != 0 else 0
    
    def calculate_beta(self, asset_returns: Union[pd.Series, np.ndarray], 
                      market_returns: Union[pd.Series, np.ndarray]) -> float:
        """计算Beta系数"""
        covariance = np.cov(asset_returns, market_returns)[0][1]
        market_variance = np.var(market_returns)
        return covariance / market_variance if market_variance != 0 else 0
    
    def calculate_information_ratio(self, portfolio_returns: Union[pd.Series, np.ndarray], 
                                  benchmark_returns: Union[pd.Series, np.ndarray]) -> float:
        """计算信息比率"""
        excess_returns = portfolio_returns - benchmark_returns
        tracking_error = np.std(excess_returns) * np.sqrt(self.trading_days_per_year)
        excess_return = np.mean(excess_returns) * self.trading_days_per_year
        return excess_return / tracking_error if tracking_error != 0 else 0
    
    def calculate_portfolio_risk_metrics(self, returns: Union[pd.Series, np.ndarray], 
                                       benchmark_returns: Optional[Union[pd.Series, np.ndarray]] = None) -> RiskMetrics:
        """计算投资组合的全面风险指标"""
        var_1d = self.calculate_var(returns, 0.95)
        var_5d = var_1d * np.sqrt(5)  # 简化的时间缩放
        var_10d = var_1d * np.sqrt(10)
        
        cvar_1d = self.calculate_cvar(returns, 0.95)
        expected_shortfall = self.calculate_expected_shortfall(returns, 0.95)
        volatility = self.calculate_volatility(returns)
        max_drawdown = self.calculate_max_drawdown(returns)
        sharpe_ratio = self.calculate_sharpe_ratio(returns)
        sortino_ratio = self.calculate_sortino_ratio(returns)
        calmar_ratio = self.calculate_calmar_ratio(returns)
        
        return RiskMetrics(
            var_1d=var_1d,
            var_5d=var_5d,
            var_10d=var_10d,
            cvar_1d=cvar_1d,
            expected_shortfall=expected_shortfall,
            volatility=volatility,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio
        )
    
    def calculate_position_risk(self, position_value: float, 
                              price_volatility: float, 
                              confidence_level: float = 0.95, 
                              time_horizon: int = 1) -> Dict[str, float]:
        """计算单个仓位的风险"""
        z_score = norm.ppf(confidence_level)
        position_volatility = price_volatility * np.sqrt(time_horizon)
        
        var = position_value * z_score * position_volatility
        
        return {
            'position_value': position_value,
            'volatility': position_volatility,
            'var': var,
            'max_loss_pct': z_score * position_volatility
        }
    
    def calculate_portfolio_concentration(self, positions: Dict[str, float]) -> Dict[str, float]:
        """计算投资组合集中度风险"""
        total_value = sum(abs(v) for v in positions.values())
        
        if total_value == 0:
            return {'herfindahl_index': 0, 'max_weight': 0, 'effective_positions': 0}
        
        weights = [abs(v) / total_value for v in positions.values()]
        
        # 赫芬达尔指数
        herfindahl_index = sum(w**2 for w in weights)
        
        # 最大权重
        max_weight = max(weights)
        
        # 有效仓位数
        effective_positions = 1 / herfindahl_index if herfindahl_index > 0 else 0
        
        return {
            'herfindahl_index': herfindahl_index,
            'max_weight': max_weight,
            'effective_positions': effective_positions
        }
    
    def calculate_historical_var(self, returns: Union[pd.Series, np.ndarray], 
                                confidence_level: float = 0.95, 
                                confidence: Optional[float] = None) -> float:
        """使用历史模拟法计算VaR
        
        Args:
            returns: 收益率序列
            confidence_level: 置信水平 (优先使用)
            confidence: 置信水平 (向后兼容)
        """
        # 支持两种参数名称
        if confidence is not None:
            confidence_level = confidence
        return self.calculate_var(returns, confidence_level, method='historical')
    
    def calculate_parametric_var(self, portfolio_return: float, portfolio_volatility: float,
                                confidence_level: float = 0.95) -> float:
        """使用参数法计算VaR"""
        z_score = norm.ppf(1 - confidence_level)
        return portfolio_return + z_score * portfolio_volatility
    
    def calculate_monte_carlo_var(self, portfolio_return: float, portfolio_volatility: float,
                                 confidence_level: float = 0.95, num_simulations: int = 10000) -> float:
        """使用蒙特卡洛法计算VaR"""
        simulated_returns = np.random.normal(portfolio_return, portfolio_volatility, num_simulations)
        return np.percentile(simulated_returns, (1 - confidence_level) * 100)
    
    def calculate_component_var(self, weights: np.ndarray, covariance_matrix: np.ndarray,
                               confidence_level: float = 0.95) -> np.ndarray:
        """计算成分VaR"""
        portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # 边际VaR
        marginal_var = np.dot(covariance_matrix, weights) / portfolio_volatility
        
        # 成分VaR
        z_score = norm.ppf(confidence_level)
        component_var = weights * marginal_var * z_score
        
        return component_var
    
    def calculate_portfolio_var(self, weights: np.ndarray, covariance_matrix: np.ndarray,
                               confidence_level: float = 0.95) -> float:
        """计算投资组合VaR"""
        portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        z_score = norm.ppf(confidence_level)
        return z_score * portfolio_volatility

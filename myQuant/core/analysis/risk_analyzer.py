# -*- coding: utf-8 -*-
"""
风险分析器 - 提供投资组合风险分析功能
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
    var_95: float  # 95% VaR
    var_99: float  # 99% VaR 
    cvar_95: float  # 95% CVaR
    cvar_99: float  # 99% CVaR
    volatility: float  # 波动率
    sharpe_ratio: float  # 夏普比率
    max_drawdown: float  # 最大回撤
    beta: Optional[float] = None  # Beta值
    tracking_error: Optional[float] = None  # 跟踪误差


class RiskAnalyzer:
    """风险分析器"""
    
    def __init__(self):
        self.confidence_levels = [0.95, 0.99]
        
    def calculate_var(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """计算VaR (Value at Risk)"""
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    def calculate_cvar(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """计算CVaR (Conditional Value at Risk)"""
        var = self.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()
    
    def calculate_volatility(self, returns: pd.Series, annualized: bool = True) -> float:
        """计算波动率"""
        vol = returns.std()
        if annualized:
            vol *= np.sqrt(252)  # 年化
        return vol
    
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """计算夏普比率"""
        excess_returns = returns.mean() * 252 - risk_free_rate
        volatility = self.calculate_volatility(returns)
        return excess_returns / volatility if volatility != 0 else 0
    
    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """计算最大回撤"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def calculate_beta(self, asset_returns: pd.Series, market_returns: pd.Series) -> float:
        """计算Beta值"""
        covariance = np.cov(asset_returns, market_returns)[0][1]
        market_variance = np.var(market_returns)
        return covariance / market_variance if market_variance != 0 else 0
    
    def calculate_tracking_error(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """计算跟踪误差"""
        return (portfolio_returns - benchmark_returns).std() * np.sqrt(252)
    
    def analyze_portfolio_risk(self, returns: pd.Series, benchmark_returns: Optional[pd.Series] = None) -> RiskMetrics:
        """分析投资组合风险"""
        var_95 = self.calculate_var(returns, 0.95)
        var_99 = self.calculate_var(returns, 0.99)
        cvar_95 = self.calculate_cvar(returns, 0.95)
        cvar_99 = self.calculate_cvar(returns, 0.99)
        volatility = self.calculate_volatility(returns)
        sharpe_ratio = self.calculate_sharpe_ratio(returns)
        max_drawdown = self.calculate_max_drawdown(returns)
        
        beta = None
        tracking_error = None
        
        if benchmark_returns is not None:
            beta = self.calculate_beta(returns, benchmark_returns)
            tracking_error = self.calculate_tracking_error(returns, benchmark_returns)
        
        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            beta=beta,
            tracking_error=tracking_error
        )
    
    def calculate_concentration_risk(self, positions: Union[List[Dict], Dict[str, Dict]]) -> Dict[str, float]:
        """计算集中度风险
        
        Args:
            positions: 持仓列表或字典
                - 列表格式: [{'weight': 0.1, 'value': 100000}, ...]
                - 字典格式: {'AAPL': {'weight': 0.1, 'value': 100000}, ...}
        """
        # 处理不同的输入格式
        if isinstance(positions, dict):
            # 字典格式，转换为列表
            positions_list = list(positions.values())
        else:
            positions_list = positions
            
        weights = [pos['weight'] for pos in positions_list]
        weights_array = np.array(weights)
        
        # 最大单一持仓权重
        largest_position_weight = max(weights)
        
        # 前5大持仓集中度
        sorted_weights = sorted(weights, reverse=True)
        top5_concentration = sum(sorted_weights[:5])
        
        # 赫芬达尔指数 (Herfindahl Index)
        herfindahl_index = sum(w**2 for w in weights)
        
        # 计算集中度风险评分 (0-1, 越高风险越大)
        concentration_score = (
            largest_position_weight * 0.4 +  # 最大持仓权重的贡献
            top5_concentration * 0.3 +       # 前5大持仓集中度的贡献
            herfindahl_index * 0.3           # 赫芬达尔指数的贡献
        )
        
        return concentration_score
    
    def calculate_sector_concentration(self, positions: Union[List[Dict], Dict[str, Dict]]) -> Dict[str, float]:
        """计算行业集中度
        
        Args:
            positions: 持仓列表或字典
        """
        # 处理不同的输入格式
        if isinstance(positions, dict):
            positions_list = list(positions.values())
        else:
            positions_list = positions
            
        sector_weights = {}
        
        for pos in positions_list:
            sector = pos.get('sector', 'Unknown')
            weight = pos.get('weight', 0)
            
            if sector in sector_weights:
                sector_weights[sector] += weight
            else:
                sector_weights[sector] = weight
        
        return sector_weights
    
    def calculate_portfolio_var(self, weights: np.ndarray, volatilities: np.ndarray, 
                              correlation_matrix: np.ndarray, confidence_level: float = 0.95) -> float:
        """计算投资组合VaR"""
        # 计算协方差矩阵
        vol_matrix = np.outer(volatilities, volatilities)
        cov_matrix = vol_matrix * correlation_matrix
        
        # 计算投资组合方差
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
        portfolio_std = np.sqrt(portfolio_variance)
        
        # 假设正态分布，计算VaR
        from scipy.stats import norm
        var = norm.ppf(1 - confidence_level) * portfolio_std
        
        return var
    
    def run_stress_tests(self, positions: List[Dict], stress_scenarios: Dict) -> Dict:
        """运行压力测试"""
        results = {}
        
        for scenario_name, scenario_params in stress_scenarios.items():
            scenario_result = self._apply_stress_scenario(positions, scenario_params)
            results[scenario_name] = scenario_result
        
        return results
    
    def _apply_stress_scenario(self, positions: List[Dict], scenario_params: Dict) -> Dict:
        """应用压力测试场景"""
        portfolio_impact = 0.0
        position_impacts = {}
        
        if 'market_factor' in scenario_params:
            # 市场整体影响
            market_factor = scenario_params['market_factor']
            for pos in positions:
                impact = pos['weight'] * market_factor
                portfolio_impact += impact
                position_impacts[pos['symbol']] = impact
        
        if 'sector_factors' in scenario_params:
            # 行业因子影响
            sector_factors = scenario_params['sector_factors']
            for pos in positions:
                sector = pos['sector']
                if sector in sector_factors:
                    impact = pos['weight'] * sector_factors[sector]
                    portfolio_impact += impact
                    if pos['symbol'] in position_impacts:
                        position_impacts[pos['symbol']] += impact
                    else:
                        position_impacts[pos['symbol']] = impact
        
        return {
            'portfolio_impact': portfolio_impact,
            'position_impacts': position_impacts
        }
    
    def calculate_risk_budgets(self, positions: List[Dict]) -> Dict[str, float]:
        """计算风险预算"""
        # 简化的风险预算计算，基于权重和假设的风险贡献
        total_risk = 0.0
        risk_contributions = {}
        
        # 计算每个持仓的风险贡献（简化为权重的平方）
        for pos in positions:
            symbol = pos['symbol']
            weight = pos['weight']
            risk_contrib = weight ** 2  # 简化的风险贡献计算
            risk_contributions[symbol] = risk_contrib
            total_risk += risk_contrib
        
        # 标准化为风险预算（总和为1）
        risk_budgets = {}
        if total_risk > 0:
            for symbol, risk_contrib in risk_contributions.items():
                risk_budgets[symbol] = risk_contrib / total_risk
        
        return risk_budgets

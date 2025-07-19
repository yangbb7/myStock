# -*- coding: utf-8 -*-
"""
绩效指标计算模块
"""

from typing import Any, Dict

import numpy as np
import pandas as pd


class PerformanceMetrics:
    """绩效指标计算器"""

    def __init__(self, risk_free_rate: float = 0.03, trading_days_per_year: int = 252):
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = trading_days_per_year

    def calculate_returns(
        self, portfolio_values: pd.Series, method: str = "simple"
    ) -> pd.Series:
        """计算收益率"""
        if method == "simple":
            return portfolio_values.pct_change().dropna()
        elif method == "log":
            return np.log(portfolio_values / portfolio_values.shift(1)).dropna()
        else:
            raise ValueError("method must be 'simple' or 'log'")

    def calculate_cumulative_returns(self, returns: pd.Series) -> pd.Series:
        """计算累计收益率"""
        return (1 + returns).cumprod() - 1

    def calculate_annualized_return(self, returns: pd.Series) -> float:
        """计算年化收益率"""
        if len(returns) == 0:
            return 0.0

        total_return = (1 + returns).prod() - 1
        periods = len(returns) / self.trading_days_per_year

        if periods <= 0:
            return 0.0

        return (1 + total_return) ** (1 / periods) - 1

    def calculate_volatility(
        self, returns: pd.Series, annualize: bool = False, handle_nan: bool = False
    ) -> float:
        """计算波动率"""
        if len(returns) == 0:
            return 0.0

        if handle_nan:
            returns = returns.dropna()

        vol = returns.std()

        if annualize:
            vol *= np.sqrt(self.trading_days_per_year)

        return vol

    def calculate_sharpe_ratio(
        self, returns: pd.Series, risk_free_rate: float = None
    ) -> float:
        """计算夏普比率"""
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate

        excess_returns = returns - risk_free_rate / self.trading_days_per_year

        if len(excess_returns) == 0 or excess_returns.std() == 0:
            return 0.0

        return (
            np.sqrt(self.trading_days_per_year)
            * excess_returns.mean()
            / excess_returns.std()
        )

    def calculate_sortino_ratio(
        self, returns: pd.Series, target_return: float = 0, risk_free_rate: float = None
    ) -> float:
        """计算索提诺比率"""
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate

        excess_returns = returns - risk_free_rate / self.trading_days_per_year
        downside_returns = excess_returns[excess_returns < target_return]

        if len(downside_returns) == 0:
            return float("inf") if excess_returns.mean() > 0 else 0.0

        downside_deviation = np.sqrt((downside_returns**2).mean())

        if downside_deviation == 0:
            return 0.0

        return (
            np.sqrt(self.trading_days_per_year)
            * excess_returns.mean()
            / downside_deviation
        )

    def calculate_calmar_ratio(self, portfolio_values: pd.Series) -> float:
        """计算卡尔玛比率"""
        from .risk_metrics import RiskMetrics

        returns = self.calculate_returns(portfolio_values)
        annual_return = self.calculate_annualized_return(returns)

        risk_metrics = RiskMetrics()
        max_dd = risk_metrics.calculate_max_drawdown(portfolio_values)

        if max_dd["max_drawdown"] == 0:
            return float("inf") if annual_return > 0 else 0.0

        return annual_return / abs(max_dd["max_drawdown"])

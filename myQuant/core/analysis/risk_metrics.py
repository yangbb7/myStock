# -*- coding: utf-8 -*-
"""
风险指标计算模块
"""

import math
from typing import Any, Dict

import numpy as np
import pandas as pd

from ..exceptions import DataException


class RiskMetrics:
    """风险指标计算器"""

    def __init__(self, trading_days_per_year: int = 252):
        self.trading_days_per_year = trading_days_per_year

    def _norm_ppf(self, p: float) -> float:
        """标准正态分布逆累积分布函数的近似实现"""
        # 使用Beasley-Springer-Moro算法的简化版本
        if p <= 0.5:
            # 对于p <= 0.5，使用对称性
            return -self._norm_ppf(1 - p)

        # 对于p > 0.5，使用近似公式
        t = math.sqrt(-2 * math.log(1 - p))
        c0 = 2.515517
        c1 = 0.802853
        c2 = 0.010328
        d1 = 1.432788
        d2 = 0.189269
        d3 = 0.001308

        return t - (c0 + c1 * t + c2 * t * t) / (
            1 + d1 * t + d2 * t * t + d3 * t * t * t
        )

    def calculate_max_drawdown(self, portfolio_values: pd.Series) -> Dict[str, Any]:
        """计算最大回撤"""
        if len(portfolio_values) == 0:
            return {
                "max_drawdown": 0.0,
                "max_drawdown_duration": 0,
                "start_date": None,
                "end_date": None,
                "recovery_date": None,
            }

        # 计算累计最高值
        cumulative_max = portfolio_values.expanding().max()

        # 计算回撤
        drawdowns = (portfolio_values - cumulative_max) / cumulative_max

        # 找到最大回撤
        max_drawdown = drawdowns.min()
        max_drawdown_idx = drawdowns.idxmin()

        # 找到最大回撤的开始时间
        start_idx = None
        for i in range(len(portfolio_values)):
            if portfolio_values.index[i] > max_drawdown_idx:
                break
            if (
                portfolio_values.iloc[i]
                == cumulative_max.iloc[drawdowns.index.get_loc(max_drawdown_idx)]
            ):
                start_idx = portfolio_values.index[i]

        # 找到恢复时间
        recovery_idx = None
        if start_idx and max_drawdown_idx:
            peak_value = portfolio_values.loc[start_idx]
            for i in range(len(portfolio_values)):
                if portfolio_values.index[i] <= max_drawdown_idx:
                    continue
                if portfolio_values.iloc[i] >= peak_value:
                    recovery_idx = portfolio_values.index[i]
                    break

        # 计算持续时间
        duration = 0
        if start_idx and max_drawdown_idx:
            # 检查索引是否为datetime类型
            if hasattr(max_drawdown_idx, 'days') and hasattr(start_idx, 'days'):
                duration = (max_drawdown_idx - start_idx).days
            else:
                # 如果不是datetime类型，计算索引差值
                try:
                    start_pos = portfolio_values.index.get_loc(start_idx)
                    end_pos = portfolio_values.index.get_loc(max_drawdown_idx)
                    duration = end_pos - start_pos
                except (KeyError, ValueError):
                    duration = 0

        return {
            "max_drawdown": max_drawdown,
            "max_drawdown_duration": duration,
            "start_date": start_idx,
            "end_date": max_drawdown_idx,
            "recovery_date": recovery_idx,
        }

    def calculate_var(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95,
        method: str = "historical",
    ) -> float:
        """计算在险价值(VaR)"""
        if len(returns) == 0:
            return 0.0

        if method == "historical":
            return np.percentile(returns, (1 - confidence_level) * 100)
        elif method == "parametric":
            mean = returns.mean()
            std = returns.std()
            # 使用逆标准正态分布近似
            z_score = self._norm_ppf(1 - confidence_level)
            return mean + z_score * std
        else:
            raise DataException("method must be 'historical' or 'parametric'")

    def calculate_cvar(
        self, returns: pd.Series, confidence_level: float = 0.95
    ) -> float:
        """计算条件在险价值(CVaR)"""
        if len(returns) == 0:
            return 0.0

        var = self.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()

    def calculate_downside_deviation(
        self, returns: pd.Series, target_return: float = 0
    ) -> float:
        """计算下行标准差"""
        downside_returns = returns[returns < target_return]
        if len(downside_returns) == 0:
            return 0.0
        return np.sqrt((downside_returns**2).mean())

    def calculate_tail_ratio(
        self, returns: pd.Series, tail_percentile: float = 0.05
    ) -> float:
        """计算尾部比率"""
        if len(returns) == 0:
            return 0.0

        upper_tail = np.percentile(returns, (1 - tail_percentile) * 100)
        lower_tail = np.percentile(returns, tail_percentile * 100)

        if lower_tail == 0:
            return float("inf") if upper_tail > 0 else 0.0

        return abs(upper_tail / lower_tail)

    def calculate_skewness(self, returns: pd.Series) -> float:
        """计算偏度"""
        # 使用numpy计算偏度
        n = len(returns)
        if n < 3:
            return 0.0
        mean = returns.mean()
        std = returns.std()
        if std == 0:
            return 0.0

        m3 = np.sum((returns - mean) ** 3) / n
        return m3 / (std**3)

    def calculate_kurtosis(self, returns: pd.Series) -> float:
        """计算峰度"""
        # 使用numpy计算峰度
        n = len(returns)
        if n < 4:
            return 0.0
        mean = returns.mean()
        std = returns.std()
        if std == 0:
            return 0.0

        m4 = np.sum((returns - mean) ** 4) / n
        return m4 / (std**4) - 3  # 减去3是超额峰度

    def calculate_value_at_risk_components(
        self, returns: pd.Series, confidence_levels: list = None
    ) -> Dict[str, float]:
        """计算多个置信水平的VaR"""
        if confidence_levels is None:
            confidence_levels = [0.90, 0.95, 0.99]

        var_components = {}
        for level in confidence_levels:
            var_components[f"VaR_{int(level*100)}%"] = self.calculate_var(
                returns, level
            )
            var_components[f"CVaR_{int(level*100)}%"] = self.calculate_cvar(
                returns, level
            )

        return var_components

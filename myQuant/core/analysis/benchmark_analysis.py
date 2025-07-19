# -*- coding: utf-8 -*-
"""
基准分析模块
"""

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


class BenchmarkAnalysis:
    """基准分析器"""

    def __init__(self, trading_days_per_year: int = 252):
        self.trading_days_per_year = trading_days_per_year

    def calculate_beta(
        self, portfolio_returns: pd.Series, benchmark_returns: pd.Series
    ) -> float:
        """计算贝塔系数"""
        if len(portfolio_returns) == 0 or len(benchmark_returns) == 0:
            return 0.0

        # 对齐数据
        aligned_data = pd.concat(
            [portfolio_returns, benchmark_returns], axis=1
        ).dropna()
        if len(aligned_data) == 0:
            return 0.0

        portfolio_aligned = aligned_data.iloc[:, 0]
        benchmark_aligned = aligned_data.iloc[:, 1]

        covariance = np.cov(portfolio_aligned, benchmark_aligned)[0, 1]
        variance = np.var(benchmark_aligned)

        if variance == 0:
            return 0.0

        return covariance / variance

    def calculate_alpha(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        risk_free_rate: float = 0.03,
    ) -> float:
        """计算阿尔法系数"""
        beta = self.calculate_beta(portfolio_returns, benchmark_returns)

        # 对齐数据
        aligned_data = pd.concat(
            [portfolio_returns, benchmark_returns], axis=1
        ).dropna()
        if len(aligned_data) == 0:
            return 0.0

        portfolio_aligned = aligned_data.iloc[:, 0]
        benchmark_aligned = aligned_data.iloc[:, 1]

        # 计算超额收益
        risk_free_daily = risk_free_rate / self.trading_days_per_year
        portfolio_excess = portfolio_aligned.mean() - risk_free_daily
        benchmark_excess = benchmark_aligned.mean() - risk_free_daily

        # 年化阿尔法
        alpha = (
            portfolio_excess - beta * benchmark_excess
        ) * self.trading_days_per_year

        return alpha

    def calculate_alpha_beta(
        self, portfolio_returns: pd.Series, benchmark_returns: pd.Series
    ) -> Tuple[float, float]:
        """同时计算阿尔法和贝塔"""
        beta = self.calculate_beta(portfolio_returns, benchmark_returns)
        alpha = self.calculate_alpha(portfolio_returns, benchmark_returns)
        return alpha, beta

    def calculate_tracking_error(
        self, portfolio_returns: pd.Series, benchmark_returns: pd.Series
    ) -> float:
        """计算跟踪误差"""
        # 对齐数据
        aligned_data = pd.concat(
            [portfolio_returns, benchmark_returns], axis=1
        ).dropna()
        if len(aligned_data) == 0:
            return 0.0

        portfolio_aligned = aligned_data.iloc[:, 0]
        benchmark_aligned = aligned_data.iloc[:, 1]

        # 计算超额收益的标准差
        excess_returns = portfolio_aligned - benchmark_aligned
        tracking_error = excess_returns.std() * np.sqrt(self.trading_days_per_year)

        return tracking_error

    def calculate_information_ratio(
        self, portfolio_returns: pd.Series, benchmark_returns: pd.Series
    ) -> float:
        """计算信息比率"""
        # 对齐数据
        aligned_data = pd.concat(
            [portfolio_returns, benchmark_returns], axis=1
        ).dropna()
        if len(aligned_data) == 0:
            return 0.0

        portfolio_aligned = aligned_data.iloc[:, 0]
        benchmark_aligned = aligned_data.iloc[:, 1]

        # 计算超额收益
        excess_returns = portfolio_aligned - benchmark_aligned

        if excess_returns.std() == 0:
            return 0.0

        # 年化信息比率
        ir = (excess_returns.mean() * self.trading_days_per_year) / (
            excess_returns.std() * np.sqrt(self.trading_days_per_year)
        )

        return ir

    def calculate_treynor_ratio(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        risk_free_rate: float = 0.03,
    ) -> float:
        """计算特雷诺比率"""
        beta = self.calculate_beta(portfolio_returns, benchmark_returns)

        if beta == 0:
            return 0.0

        # 对齐数据
        aligned_data = pd.concat(
            [portfolio_returns, benchmark_returns], axis=1
        ).dropna()
        if len(aligned_data) == 0:
            return 0.0

        portfolio_aligned = aligned_data.iloc[:, 0]

        # 计算年化收益率
        annual_return = (1 + portfolio_aligned.mean()) ** self.trading_days_per_year - 1

        return (annual_return - risk_free_rate) / beta

    def calculate_capture_ratios(
        self, portfolio_returns: pd.Series, benchmark_returns: pd.Series
    ) -> Dict[str, float]:
        """计算上行/下行捕获比率"""
        # 对齐数据
        aligned_data = pd.concat(
            [portfolio_returns, benchmark_returns], axis=1
        ).dropna()
        if len(aligned_data) == 0:
            return {"upside_capture": 0.0, "downside_capture": 0.0}

        portfolio_aligned = aligned_data.iloc[:, 0]
        benchmark_aligned = aligned_data.iloc[:, 1]

        # 分离上行和下行市场
        upside_mask = benchmark_aligned > 0
        downside_mask = benchmark_aligned < 0

        upside_portfolio = portfolio_aligned[upside_mask]
        upside_benchmark = benchmark_aligned[upside_mask]
        downside_portfolio = portfolio_aligned[downside_mask]
        downside_benchmark = benchmark_aligned[downside_mask]

        # 计算捕获比率
        upside_capture = 0.0
        if len(upside_portfolio) > 0 and upside_benchmark.mean() != 0:
            upside_capture = upside_portfolio.mean() / upside_benchmark.mean()

        downside_capture = 0.0
        if len(downside_portfolio) > 0 and downside_benchmark.mean() != 0:
            downside_capture = downside_portfolio.mean() / downside_benchmark.mean()

        return {"upside_capture": upside_capture, "downside_capture": downside_capture}

    def calculate_correlation(
        self, portfolio_returns: pd.Series, benchmark_returns: pd.Series
    ) -> float:
        """计算相关系数"""
        # 对齐数据
        aligned_data = pd.concat(
            [portfolio_returns, benchmark_returns], axis=1
        ).dropna()
        if len(aligned_data) < 2:
            return 0.0

        portfolio_aligned = aligned_data.iloc[:, 0]
        benchmark_aligned = aligned_data.iloc[:, 1]

        correlation = np.corrcoef(portfolio_aligned, benchmark_aligned)[0, 1]

        return correlation if not np.isnan(correlation) else 0.0

    def calculate_relative_strength(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        window: int = 252,
    ) -> pd.Series:
        """计算相对强度"""
        # 对齐数据
        aligned_data = pd.concat(
            [portfolio_returns, benchmark_returns], axis=1
        ).dropna()
        if len(aligned_data) == 0:
            return pd.Series()

        portfolio_aligned = aligned_data.iloc[:, 0]
        benchmark_aligned = aligned_data.iloc[:, 1]

        # 计算累计收益
        portfolio_cumret = (1 + portfolio_aligned).cumprod()
        benchmark_cumret = (1 + benchmark_aligned).cumprod()

        # 计算相对强度
        relative_strength = portfolio_cumret / benchmark_cumret

        return relative_strength

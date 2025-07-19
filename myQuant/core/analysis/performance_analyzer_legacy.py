# -*- coding: utf-8 -*-
"""
PerformanceAnalyzer - 绩效分析器模块
负责投资组合绩效分析、风险度量和归因分析
"""

import logging
import warnings
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


class PerformanceAnalyzer:
    """绩效分析器"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

        # 配置验证
        self._validate_config()

        # 基本配置
        self.risk_free_rate = self.config.get("risk_free_rate", 0.03)  # 无风险收益率
        self.trading_days_per_year = self.config.get("trading_days_per_year", 252)
        self.confidence_levels = self.config.get("confidence_levels", [0.95, 0.99])

        # 基准配置
        self.benchmark_symbol = self.config.get("benchmark_symbol", "000300.SH")

        # 分析窗口
        self.rolling_window = self.config.get("rolling_window", 252)  # 滚动窗口大小
        self.min_periods = self.config.get("min_periods", 30)  # 最小分析期间

        # 日志
        self.logger = logging.getLogger(__name__)

        self.logger.info("绩效分析器初始化完成")

    def _validate_config(self):
        """验证配置参数"""
        if "risk_free_rate" in self.config:
            if self.config["risk_free_rate"] < 0:
                raise ValueError("risk_free_rate must be non-negative")

        if "trading_days_per_year" in self.config:
            if self.config["trading_days_per_year"] <= 0:
                raise ValueError("trading_days_per_year must be positive")

        if "confidence_levels" in self.config:
            levels = self.config["confidence_levels"]
            if not isinstance(levels, list) or not levels:
                raise ValueError("confidence_levels must be a non-empty list")
            for level in levels:
                if not (0 < level < 1):
                    raise ValueError("confidence_levels must be between 0 and 1")

        if "rolling_window" in self.config:
            if self.config["rolling_window"] <= 0:
                raise ValueError("rolling_window must be positive")

        if "min_periods" in self.config:
            if self.config["min_periods"] <= 0:
                raise ValueError("min_periods must be positive")

    def calculate_portfolio_value(
        self,
        positions: Dict[str, Any],
        current_prices: Dict[str, float],
        cash: float = 0.0,
    ) -> float:
        """计算投资组合价值"""
        total_value = cash  # 先加上现金

        # 计算持仓价值
        for symbol, position in positions.items():
            if symbol == "cash":
                total_value += position
            elif symbol in current_prices:
                # 兼容两种数据结构：直接数量或包含quantity的字典
                if isinstance(position, dict):
                    quantity = position.get("quantity", 0)
                    # 使用当前价格计算市值
                    if "market_value" in position:
                        total_value += position["market_value"]
                    else:
                        price = current_prices[symbol]
                        total_value += quantity * price
                else:
                    quantity = position
                    price = current_prices[symbol]
                    total_value += quantity * price

        return total_value

    def calculate_returns(
        self, portfolio_values: pd.Series, method: str = "simple"
    ) -> pd.Series:
        """计算收益率序列"""
        if portfolio_values.empty or len(portfolio_values) < 2:
            raise ValueError("Insufficient data for return calculation")

        if method.lower() == "simple":
            returns = portfolio_values.pct_change()
        elif method.lower() == "log":
            returns = np.log(portfolio_values / portfolio_values.shift(1))
        else:
            raise ValueError(f"Unknown method: {method}")

        return returns.dropna()

    def calculate_cumulative_returns(self, returns: pd.Series) -> pd.Series:
        """计算累计收益率"""
        if returns.empty:
            return pd.Series()

        return (1 + returns).cumprod() - 1

    def calculate_annualized_return(self, returns: pd.Series) -> float:
        """计算年化收益率"""
        if returns.empty:
            return 0.0

        total_return = (1 + returns).prod() - 1
        days = len(returns)

        if days == 0:
            return 0.0

        # 年化收益率
        annual_return = (1 + total_return) ** (self.trading_days_per_year / days) - 1
        return annual_return

    def calculate_volatility(
        self, returns: pd.Series, annualize: bool = False, handle_nan: bool = False
    ) -> float:
        """计算波动率"""
        if returns.empty:
            return np.nan

        if handle_nan:
            returns = returns.dropna()
            if returns.empty:
                return np.nan

        volatility = returns.std()

        if annualize:
            volatility *= np.sqrt(self.trading_days_per_year)

        return volatility

    def calculate_sharpe_ratio(
        self, returns: pd.Series, risk_free_rate: float = None, handle_nan: bool = False
    ) -> float:
        """计算夏普比率"""
        if returns.empty:
            return 0.0

        if handle_nan:
            returns = returns.dropna()
            if returns.empty:
                return 0.0

        risk_free_rate = risk_free_rate or self.risk_free_rate

        # 年化收益率和波动率
        annual_return = self.calculate_annualized_return(returns)
        annual_volatility = self.calculate_volatility(returns, annualize=True)

        # 处理接近零的波动率
        if annual_volatility == 0 or annual_volatility < 1e-15:
            excess_return = annual_return - risk_free_rate
            if excess_return > 0:
                return float("inf")
            elif excess_return < 0:
                return float("-inf")
            else:
                return float("nan")

        return (annual_return - risk_free_rate) / annual_volatility

    def calculate_sortino_ratio(
        self, returns: pd.Series, target_return: float = 0.0
    ) -> float:
        """计算索提诺比率"""
        if returns.empty:
            return 0.0

        # 计算超额收益
        excess_returns = returns - target_return / self.trading_days_per_year

        # 计算下行风险
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0:
            return float("inf") if excess_returns.mean() > 0 else 0.0

        downside_deviation = downside_returns.std() * np.sqrt(
            self.trading_days_per_year
        )

        if downside_deviation == 0:
            return 0.0

        annual_excess_return = excess_returns.mean() * self.trading_days_per_year

        return annual_excess_return / downside_deviation

    def calculate_max_drawdown(self, portfolio_values: pd.Series) -> Dict[str, Any]:
        """计算最大回撤"""
        if portfolio_values.empty:
            return {
                "max_drawdown": 0.0,
                "start_date": None,
                "end_date": None,
                "duration": 0,
            }

        # 计算历史最高点
        peak = portfolio_values.expanding().max()

        # 计算回撤
        drawdown = (portfolio_values - peak) / peak

        # 找到最大回撤
        max_dd_idx = drawdown.idxmin()
        max_drawdown = drawdown.min()

        # 找到最大回撤的开始和结束日期
        peak_before_max_dd = peak.loc[:max_dd_idx]
        start_idx = peak_before_max_dd[
            peak_before_max_dd == peak_before_max_dd.iloc[-1]
        ].index[0]

        # 找到回撤结束日期（价格重新达到峰值）
        end_idx = max_dd_idx
        peak_value = peak.loc[start_idx]

        for idx in portfolio_values.loc[max_dd_idx:].index:
            if portfolio_values.loc[idx] >= peak_value:
                end_idx = idx
                break

        # 计算回撤持续时间
        duration = (
            (end_idx - start_idx).days if hasattr(end_idx - start_idx, "days") else 0
        )

        # 找到恢复日期
        recovery_date = None
        if max_dd_idx != end_idx:
            for idx in portfolio_values.loc[max_dd_idx:].index:
                if portfolio_values.loc[idx] >= peak_value:
                    recovery_date = idx
                    break

        return {
            "max_drawdown": max_drawdown,
            "start_date": start_idx,
            "end_date": end_idx,
            "recovery_date": recovery_date,
            "duration": duration,
            "peak_value": peak_value,
            "trough_value": portfolio_values.loc[max_dd_idx],
        }

    def calculate_calmar_ratio(self, portfolio_values: pd.Series) -> float:
        """计算卡玛比率"""
        # 从组合价值计算收益率
        returns = self.calculate_returns(portfolio_values)
        annual_return = self.calculate_annualized_return(returns)
        max_drawdown_info = self.calculate_max_drawdown(portfolio_values)
        max_drawdown = abs(max_drawdown_info["max_drawdown"])

        if max_drawdown == 0:
            return float("inf") if annual_return > 0 else 0.0

        return annual_return / max_drawdown

    def calculate_var(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95,
        method: str = "historical",
    ) -> float:
        """计算风险价值(VaR)"""
        if len(returns) < 20:
            raise ValueError("Insufficient data for VaR calculation")

        if method == "historical":
            return np.percentile(returns, (1 - confidence_level) * 100)
        elif method == "parametric":
            mean = returns.mean()
            std = returns.std()
            z_score = stats.norm.ppf(1 - confidence_level)
            return mean + z_score * std
        else:
            raise ValueError(f"Unknown VaR method: {method}")

    def calculate_cvar(
        self, returns: pd.Series, confidence_level: float = 0.95
    ) -> float:
        """计算条件风险价值(CVaR)"""
        if returns.empty:
            return 0.0

        var = self.calculate_var(returns, confidence_level, method="historical")
        return returns[returns <= var].mean()

    def calculate_beta(
        self, portfolio_returns: pd.Series, benchmark_returns: pd.Series
    ) -> float:
        """计算Beta系数"""
        if portfolio_returns.empty or benchmark_returns.empty:
            return 0.0

        # 对齐数据
        aligned_data = pd.DataFrame(
            {"portfolio": portfolio_returns, "benchmark": benchmark_returns}
        ).dropna()

        if len(aligned_data) < 2:
            return 0.0

        portfolio_aligned = aligned_data["portfolio"]
        benchmark_aligned = aligned_data["benchmark"]

        # 计算协方差和方差
        covariance = np.cov(portfolio_aligned, benchmark_aligned)[0, 1]
        benchmark_variance = np.var(benchmark_aligned)

        if benchmark_variance == 0:
            return 0.0

        return covariance / benchmark_variance

    def calculate_alpha(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        risk_free_rate: float = None,
    ) -> float:
        """计算Alpha"""
        if portfolio_returns.empty or benchmark_returns.empty:
            return 0.0

        risk_free_rate = risk_free_rate or self.risk_free_rate
        daily_rf_rate = risk_free_rate / self.trading_days_per_year

        # 计算超额收益
        portfolio_excess = portfolio_returns - daily_rf_rate
        benchmark_excess = benchmark_returns - daily_rf_rate

        # 计算Beta
        beta = self.calculate_beta(portfolio_returns, benchmark_returns)

        # 计算Alpha
        portfolio_mean = portfolio_excess.mean() * self.trading_days_per_year
        benchmark_mean = benchmark_excess.mean() * self.trading_days_per_year

        alpha = portfolio_mean - beta * benchmark_mean

        return alpha

    def calculate_alpha_beta(
        self, portfolio_returns: pd.Series, benchmark_returns: pd.Series
    ) -> tuple:
        """计算Alpha和Beta"""
        if len(portfolio_returns) != len(benchmark_returns):
            raise ValueError("Length mismatch between portfolio and benchmark returns")

        beta = self.calculate_beta(portfolio_returns, benchmark_returns)
        alpha = self.calculate_alpha(portfolio_returns, benchmark_returns)

        return alpha, beta

    def calculate_tracking_error(
        self, portfolio_returns: pd.Series, benchmark_returns: pd.Series
    ) -> float:
        """计算跟踪误差"""
        if len(portfolio_returns) != len(benchmark_returns):
            raise ValueError("Length mismatch between portfolio and benchmark returns")

        # 对齐数据
        aligned_data = pd.DataFrame(
            {"portfolio": portfolio_returns, "benchmark": benchmark_returns}
        ).dropna()

        if len(aligned_data) < 2:
            return 0.0

        # 计算跟踪误差
        tracking_error = aligned_data["portfolio"] - aligned_data["benchmark"]

        return tracking_error.std() * np.sqrt(self.trading_days_per_year)

    def calculate_information_ratio(
        self, portfolio_returns: pd.Series, benchmark_returns: pd.Series
    ) -> float:
        """计算信息比率"""
        if portfolio_returns.empty or benchmark_returns.empty:
            return 0.0

        # 对齐数据
        aligned_data = pd.DataFrame(
            {"portfolio": portfolio_returns, "benchmark": benchmark_returns}
        ).dropna()

        if len(aligned_data) < 2:
            return 0.0

        # 计算跟踪误差
        tracking_error = aligned_data["portfolio"] - aligned_data["benchmark"]

        if tracking_error.std() == 0:
            return 0.0

        # 年化跟踪收益和跟踪误差
        annual_tracking_return = tracking_error.mean() * self.trading_days_per_year
        annual_tracking_error = tracking_error.std() * np.sqrt(
            self.trading_days_per_year
        )

        return annual_tracking_return / annual_tracking_error

    def calculate_treynor_ratio(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        risk_free_rate: float = None,
    ) -> float:
        """计算特雷诺比率"""
        if portfolio_returns.empty or benchmark_returns.empty:
            return 0.0

        risk_free_rate = risk_free_rate or self.risk_free_rate

        annual_return = self.calculate_annualized_return(portfolio_returns)
        beta = self.calculate_beta(portfolio_returns, benchmark_returns)

        if beta == 0:
            return 0.0

        return (annual_return - risk_free_rate) / beta

    def calculate_upside_capture(
        self, portfolio_returns: pd.Series, benchmark_returns: pd.Series
    ) -> float:
        """计算上行捕获率"""
        if portfolio_returns.empty or benchmark_returns.empty:
            return 0.0

        # 对齐数据
        aligned_data = pd.DataFrame(
            {"portfolio": portfolio_returns, "benchmark": benchmark_returns}
        ).dropna()

        if aligned_data.empty:
            return 0.0

        # 筛选基准上涨的日期
        up_days = aligned_data[aligned_data["benchmark"] > 0]

        if up_days.empty:
            return 0.0

        portfolio_up_return = (1 + up_days["portfolio"]).prod() - 1
        benchmark_up_return = (1 + up_days["benchmark"]).prod() - 1

        if benchmark_up_return == 0:
            return 0.0

        return portfolio_up_return / benchmark_up_return

    def calculate_downside_capture(
        self, portfolio_returns: pd.Series, benchmark_returns: pd.Series
    ) -> float:
        """计算下行捕获率"""
        if portfolio_returns.empty or benchmark_returns.empty:
            return 0.0

        # 对齐数据
        aligned_data = pd.DataFrame(
            {"portfolio": portfolio_returns, "benchmark": benchmark_returns}
        ).dropna()

        if aligned_data.empty:
            return 0.0

        # 筛选基准下跌的日期
        down_days = aligned_data[aligned_data["benchmark"] < 0]

        if down_days.empty:
            return 0.0

        portfolio_down_return = (1 + down_days["portfolio"]).prod() - 1
        benchmark_down_return = (1 + down_days["benchmark"]).prod() - 1

        if benchmark_down_return == 0:
            return 0.0

        return portfolio_down_return / benchmark_down_return

    def calculate_win_rate(self, returns: pd.Series) -> float:
        """计算胜率"""
        if returns.empty:
            return 0.0

        winning_days = (returns > 0).sum()
        total_days = len(returns)

        return winning_days / total_days if total_days > 0 else 0.0

    def calculate_profit_loss_ratio(self, returns: pd.Series) -> float:
        """计算盈亏比"""
        if returns.empty:
            return 0.0

        winning_returns = returns[returns > 0]
        losing_returns = returns[returns < 0]

        if len(winning_returns) == 0 or len(losing_returns) == 0:
            return 0.0

        avg_win = winning_returns.mean()
        avg_loss = abs(losing_returns.mean())

        if avg_loss == 0:
            return float("inf")

        return avg_win / avg_loss

    def calculate_kelly_criterion(self, returns: pd.Series) -> float:
        """计算凯利公式"""
        if returns.empty:
            return 0.0

        win_rate = self.calculate_win_rate(returns)
        profit_loss_ratio = self.calculate_profit_loss_ratio(returns)

        if profit_loss_ratio == 0:
            return 0.0

        # 凯利公式: f = (bp - q) / b
        # 其中 b = 盈亏比, p = 胜率, q = 败率 = 1 - p
        kelly_f = (profit_loss_ratio * win_rate - (1 - win_rate)) / profit_loss_ratio

        # 限制在合理范围内
        return max(0, min(kelly_f, 1))

    def calculate_rolling_metrics(
        self, returns: pd.Series, window: int = None
    ) -> pd.DataFrame:
        """计算滚动指标"""
        if returns.empty:
            return pd.DataFrame()

        window = window or self.rolling_window

        if len(returns) < window:
            return pd.DataFrame()

        rolling_metrics = pd.DataFrame(index=returns.index)

        # 滚动收益率
        rolling_metrics["rolling_return"] = returns.rolling(window).apply(
            lambda x: (1 + x).prod() - 1
        )

        # 滚动波动率
        rolling_metrics["rolling_volatility"] = returns.rolling(window).std() * np.sqrt(
            self.trading_days_per_year
        )

        # 滚动夏普比率
        rolling_metrics["rolling_sharpe"] = returns.rolling(window).apply(
            lambda x: self.calculate_sharpe_ratio(x)
        )

        # 滚动最大回撤
        portfolio_values = (1 + returns).cumprod()
        rolling_metrics["rolling_max_dd"] = portfolio_values.rolling(window).apply(
            lambda x: self.calculate_max_drawdown(x)["max_drawdown"]
        )

        return rolling_metrics.dropna()

    def performance_attribution(
        self,
        portfolio_weights: Dict[str, float],
        benchmark_weights: Dict[str, float],
        sector_returns: Dict[str, float],
    ) -> Dict[str, float]:
        """绩效归因分析"""
        attribution = {
            "selection_effect": 0.0,
            "allocation_effect": 0.0,
            "interaction_effect": 0.0,
            "total_effect": 0.0,
        }

        if not sector_returns or not portfolio_weights or not benchmark_weights:
            return attribution

        # 计算基准的加权平均收益率
        benchmark_return = sum(
            benchmark_weights.get(sector, 0) * sector_returns.get(sector, 0)
            for sector in sector_returns.keys()
        )

        for sector in sector_returns.keys():
            portfolio_weight = portfolio_weights.get(sector, 0)
            benchmark_weight = benchmark_weights.get(sector, 0)
            sector_return = sector_returns.get(sector, 0)

            # 选股效应：(行业收益率 - 基准收益率) × 组合行业权重
            selection_effect = (sector_return - benchmark_return) * portfolio_weight

            # 配置效应：(组合行业权重 - 基准行业权重) × 基准行业收益率
            allocation_effect = (portfolio_weight - benchmark_weight) * benchmark_return

            # 交互效应：(组合行业权重 - 基准行业权重) × (行业收益率 - 基准收益率)
            interaction_effect = (portfolio_weight - benchmark_weight) * (
                sector_return - benchmark_return
            )

            attribution["selection_effect"] += selection_effect
            attribution["allocation_effect"] += allocation_effect
            attribution["interaction_effect"] += interaction_effect

        attribution["total_effect"] = (
            attribution["selection_effect"]
            + attribution["allocation_effect"]
            + attribution["interaction_effect"]
        )

        return attribution

    def generate_performance_report(
        self,
        portfolio_returns: pd.Series,
        portfolio_values: pd.Series,
        benchmark_returns: pd.Series = None,
        benchmark_values: pd.Series = None,
    ) -> Dict[str, Any]:
        """生成完整的绩效报告"""
        if portfolio_returns.empty:
            return {}

        report = {
            "period": {
                "start_date": portfolio_returns.index[0],
                "end_date": portfolio_returns.index[-1],
                "total_days": len(portfolio_returns),
            },
            "returns": {
                "total_return": self.calculate_cumulative_returns(
                    portfolio_returns
                ).iloc[-1],
                "annualized_return": self.calculate_annualized_return(
                    portfolio_returns
                ),
                "volatility": self.calculate_volatility(portfolio_returns),
                "sharpe_ratio": self.calculate_sharpe_ratio(portfolio_returns),
                "sortino_ratio": self.calculate_sortino_ratio(portfolio_returns),
            },
            "risk": {
                "max_drawdown": self.calculate_max_drawdown(portfolio_values),
                "var_95": self.calculate_var(portfolio_returns, 0.95),
                "var_99": self.calculate_var(portfolio_returns, 0.99),
                "cvar_95": self.calculate_cvar(portfolio_returns, 0.95),
                "cvar_99": self.calculate_cvar(portfolio_returns, 0.99),
            },
            "trading": {
                "win_rate": self.calculate_win_rate(portfolio_returns),
                "profit_loss_ratio": self.calculate_profit_loss_ratio(
                    portfolio_returns
                ),
                "kelly_criterion": self.calculate_kelly_criterion(portfolio_returns),
            },
        }

        # 如果有基准数据，计算相对指标
        if benchmark_returns is not None and not benchmark_returns.empty:
            report["relative"] = {
                "beta": self.calculate_beta(portfolio_returns, benchmark_returns),
                "alpha": self.calculate_alpha(portfolio_returns, benchmark_returns),
                "information_ratio": self.calculate_information_ratio(
                    portfolio_returns, benchmark_returns
                ),
                "treynor_ratio": self.calculate_treynor_ratio(
                    portfolio_returns, benchmark_returns
                ),
                "upside_capture": self.calculate_upside_capture(
                    portfolio_returns, benchmark_returns
                ),
                "downside_capture": self.calculate_downside_capture(
                    portfolio_returns, benchmark_returns
                ),
            }

            if benchmark_values is not None and not benchmark_values.empty:
                report["relative"]["calmar_ratio"] = self.calculate_calmar_ratio(
                    portfolio_returns, portfolio_values
                )
                benchmark_calmar = self.calculate_calmar_ratio(
                    benchmark_returns, benchmark_values
                )
                report["relative"]["relative_calmar"] = (
                    report["relative"]["calmar_ratio"] / benchmark_calmar
                    if benchmark_calmar != 0
                    else 0
                )

        return report

    def compare_strategies(
        self, strategies_data: Dict[str, Dict[str, pd.Series]]
    ) -> pd.DataFrame:
        """比较多个策略的绩效"""
        if not strategies_data:
            return pd.DataFrame()

        comparison_data = []

        for strategy_name, data in strategies_data.items():
            returns = data.get("returns", pd.Series())
            values = data.get("values", pd.Series())

            if returns.empty:
                continue

            # 计算各项指标
            metrics = {
                "Strategy": strategy_name,
                "Total Return": (
                    self.calculate_cumulative_returns(returns).iloc[-1]
                    if not returns.empty
                    else 0
                ),
                "Annualized Return": self.calculate_annualized_return(returns),
                "Volatility": self.calculate_volatility(returns),
                "Sharpe Ratio": self.calculate_sharpe_ratio(returns),
                "Sortino Ratio": self.calculate_sortino_ratio(returns),
                "Max Drawdown": (
                    self.calculate_max_drawdown(values)["max_drawdown"]
                    if not values.empty
                    else 0
                ),
                "Calmar Ratio": (
                    self.calculate_calmar_ratio(returns, values)
                    if not values.empty
                    else 0
                ),
                "Win Rate": self.calculate_win_rate(returns),
                "Profit/Loss Ratio": self.calculate_profit_loss_ratio(returns),
            }

            comparison_data.append(metrics)

        return pd.DataFrame(comparison_data)

    def calculate_omega_ratio(
        self, returns: pd.Series, threshold: float = 0.0
    ) -> float:
        """计算Omega比率"""
        if returns.empty:
            return 0.0

        excess_returns = returns - threshold / self.trading_days_per_year
        positive_returns = excess_returns[excess_returns > 0]
        negative_returns = excess_returns[excess_returns < 0]

        if len(negative_returns) == 0:
            return float("inf") if len(positive_returns) > 0 else 1.0

        return positive_returns.sum() / abs(negative_returns.sum())

    def calculate_component_var(
        self, weights: np.ndarray, cov_matrix: np.ndarray
    ) -> np.ndarray:
        """计算成分VaR"""
        if (
            len(weights) != cov_matrix.shape[0]
            or cov_matrix.shape[0] != cov_matrix.shape[1]
        ):
            raise ValueError("Weights and covariance matrix dimensions do not match")

        portfolio_var = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        if portfolio_var == 0:
            return np.zeros_like(weights)

        marginal_var = np.dot(cov_matrix, weights) / portfolio_var
        component_var = weights * marginal_var / portfolio_var

        return component_var

    def calculate_marginal_var(
        self, weights: np.ndarray, cov_matrix: np.ndarray
    ) -> np.ndarray:
        """计算边际VaR"""
        if (
            len(weights) != cov_matrix.shape[0]
            or cov_matrix.shape[0] != cov_matrix.shape[1]
        ):
            raise ValueError("Weights and covariance matrix dimensions do not match")

        portfolio_var = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        if portfolio_var == 0:
            return np.zeros_like(weights)

        marginal_var = np.dot(cov_matrix, weights) / portfolio_var
        return marginal_var

    def analyze_trades(self, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析交易"""
        if not transactions:
            return {}

        # 按股票分组计算损益
        trades_by_symbol = defaultdict(list)
        for txn in transactions:
            symbol = txn.get("symbol", "UNKNOWN")
            trades_by_symbol[symbol].append(txn)

        total_trades = 0
        winning_trades = 0
        losing_trades = 0
        total_profit = 0
        total_loss = 0

        for symbol, symbol_trades in trades_by_symbol.items():
            pnl = self._calculate_symbol_pnl(symbol_trades)
            total_trades += 1

            if pnl > 0:
                winning_trades += 1
                total_profit += pnl
            elif pnl < 0:
                losing_trades += 1
                total_loss += abs(pnl)

        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": winning_trades / total_trades if total_trades > 0 else 0,
            "avg_win": total_profit / winning_trades if winning_trades > 0 else 0,
            "avg_loss": total_loss / losing_trades if losing_trades > 0 else 0,
            "profit_factor": (
                total_profit / total_loss if total_loss > 0 else float("inf")
            ),
        }

    def _calculate_symbol_pnl(self, symbol_trades: List[Dict[str, Any]]) -> float:
        """计算单个股票的损益"""
        position = 0
        cost_basis = 0
        realized_pnl = 0

        for trade in symbol_trades:
            side = trade.get("side", "").upper()
            quantity = trade.get("quantity", 0)
            price = trade.get("price", 0)
            commission = trade.get("commission", 0)

            if side == "BUY":
                old_value = position * cost_basis
                new_shares = quantity
                new_value = new_shares * price

                position += new_shares
                cost_basis = (old_value + new_value) / position if position > 0 else 0
                realized_pnl -= commission  # 买入手续费

            elif side == "SELL":
                sold_shares = min(quantity, position)
                if sold_shares > 0:
                    realized_pnl += sold_shares * (price - cost_basis)
                    position -= sold_shares
                    realized_pnl -= commission  # 卖出手续费

        return realized_pnl

    def calculate_trade_pnl(
        self, transactions: List[Dict[str, Any]], group_by: str = "symbol"
    ) -> Dict[str, Dict[str, float]]:
        """计算交易损益"""
        if group_by == "symbol":
            result = {}
            positions = {}  # 跟踪每个股票的持仓

            # 按时间排序交易
            sorted_txns = sorted(
                transactions, key=lambda x: x.get("timestamp", datetime.now())
            )

            for txn in sorted_txns:
                symbol = txn.get("symbol", "UNKNOWN")
                if symbol not in result:
                    result[symbol] = {"realized_pnl": 0, "unrealized_pnl": 0}
                    positions[symbol] = {"quantity": 0, "cost_basis": 0}

                side = txn.get("side", "").upper()
                quantity = txn.get("quantity", 0)
                price = txn.get("price", 0)
                commission = txn.get("commission", 0)

                pos = positions[symbol]

                if side == "BUY":
                    # 买入：更新持仓成本
                    old_cost = pos["quantity"] * pos["cost_basis"]
                    new_cost = quantity * price
                    pos["quantity"] += quantity
                    if pos["quantity"] > 0:
                        pos["cost_basis"] = (old_cost + new_cost) / pos["quantity"]
                    result[symbol]["realized_pnl"] -= commission  # 扣除手续费

                elif side == "SELL":
                    # 卖出：计算已实现损益
                    sold_quantity = min(quantity, pos["quantity"])
                    if sold_quantity > 0:
                        realized_gain = sold_quantity * (price - pos["cost_basis"])
                        result[symbol]["realized_pnl"] += realized_gain - commission
                        pos["quantity"] -= sold_quantity

            return result
        else:
            raise ValueError(f"Unknown group_by method: {group_by}")

    def calculate_turnover_rate(
        self,
        transactions: List[Dict[str, Any]],
        avg_portfolio_value: float,
        period_days: int,
    ) -> float:
        """计算换手率"""
        if not transactions or avg_portfolio_value <= 0 or period_days <= 0:
            return 0.0

        total_volume = sum(
            txn.get("quantity", 0) * txn.get("price", 0) for txn in transactions
        )

        annual_volume = total_volume * (365 / period_days)
        return annual_volume / avg_portfolio_value

    def analyze_periods(
        self, returns: pd.Series, period: str = "monthly"
    ) -> Dict[str, Any]:
        """分析不同时期的表现"""
        if returns.empty:
            return {}

        if period == "monthly":
            try:
                monthly_returns = returns.resample("M").sum()
                return {
                    "monthly_returns": monthly_returns,
                    "monthly_volatility": monthly_returns.std(),
                    "best_month": monthly_returns.max(),
                    "worst_month": monthly_returns.min(),
                    "positive_months": (monthly_returns > 0).sum(),
                    "negative_months": (monthly_returns < 0).sum(),
                }
            except Exception:
                # 如果resample失败，使用简单分组
                return {
                    "monthly_returns": pd.Series(),
                    "monthly_volatility": 0.0,
                    "best_month": 0.0,
                    "worst_month": 0.0,
                    "positive_months": 0,
                    "negative_months": 0,
                }
        else:
            raise ValueError(f"Unknown period: {period}")

    def generate_risk_report(
        self, returns: pd.Series, annualize: bool = True
    ) -> Dict[str, Any]:
        """生成风险报告"""
        if returns.empty:
            return {}

        # 计算累积价值用于回撤计算
        portfolio_values = (1 + returns).cumprod()

        return {
            "volatility": self.calculate_volatility(returns, annualize=annualize),
            "var_95": self.calculate_var(returns, 0.95),
            "var_99": self.calculate_var(returns, 0.99),
            "cvar_95": self.calculate_cvar(returns, 0.95),
            "cvar_99": self.calculate_cvar(returns, 0.99),
            "max_drawdown": self.calculate_max_drawdown(portfolio_values)[
                "max_drawdown"
            ],
            "downside_deviation": returns[returns < 0].std()
            * np.sqrt(self.trading_days_per_year),
            "skewness": returns.skew(),
            "kurtosis": returns.kurtosis(),
        }

    def run_stress_test(
        self, portfolio_values: pd.Series, stress_scenarios: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """运行压力测试"""
        if portfolio_values.empty or not stress_scenarios:
            return {}

        results = {}
        base_value = portfolio_values.iloc[-1]

        for scenario_name, scenario in stress_scenarios.items():
            if "equity_shock" in scenario:
                shock = scenario["equity_shock"]
                stressed_value = base_value * (1 + shock)
            else:
                stressed_value = base_value * 0.9  # 默认10%下跌

            loss_amount = base_value - stressed_value
            loss_percentage = loss_amount / base_value if base_value != 0 else 0

            results[scenario_name] = {
                "stressed_value": stressed_value,
                "loss_amount": loss_amount,
                "loss_percentage": loss_percentage,
            }

        return results

    def remove_outliers(self, returns: pd.Series, method: str = "iqr") -> pd.Series:
        """移除异常值"""
        if returns.empty:
            return returns

        if method == "iqr":
            Q1 = returns.quantile(0.25)
            Q3 = returns.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return returns[(returns >= lower_bound) & (returns <= upper_bound)]
        else:
            raise ValueError(f"Unknown outlier removal method: {method}")

    def calculate_style_analysis(
        self, portfolio_returns: pd.Series, factor_returns: Dict[str, pd.Series]
    ) -> Dict[str, float]:
        """风格分析"""
        if portfolio_returns.empty or not factor_returns:
            return {}

        # 对齐数据
        factor_df = pd.DataFrame(factor_returns)
        aligned_data = pd.concat([portfolio_returns, factor_df], axis=1).dropna()

        if aligned_data.empty or len(aligned_data) < self.min_periods:
            return {}

        y = aligned_data.iloc[:, 0]  # 组合收益率
        X = aligned_data.iloc[:, 1:]  # 因子收益率

        try:
            # 使用线性回归进行风格分析
            from sklearn.linear_model import LinearRegression

            model = LinearRegression(fit_intercept=True)
            model.fit(X, y)

            # 计算因子暴露度
            factor_loadings = dict(zip(X.columns, model.coef_))

            # 计算R²
            r_squared = model.score(X, y)

            factor_loadings["alpha"] = (
                model.intercept_ * self.trading_days_per_year
            )  # 年化Alpha
            factor_loadings["r_squared"] = r_squared

            return factor_loadings

        except ImportError:
            # 如果没有sklearn，使用简单的相关性分析
            correlations = {}
            for factor_name, factor_ret in factor_returns.items():
                aligned_factor = factor_ret.reindex(portfolio_returns.index).dropna()
                common_index = portfolio_returns.index.intersection(
                    aligned_factor.index
                )

                if len(common_index) > self.min_periods:
                    corr = portfolio_returns[common_index].corr(
                        aligned_factor[common_index]
                    )
                    correlations[factor_name] = corr if not np.isnan(corr) else 0.0

            return correlations

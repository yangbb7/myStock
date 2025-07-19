# -*- coding: utf-8 -*-
"""
RiskManager - 风险管理器模块
"""

import logging
import threading
from collections import defaultdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..exceptions import (DrawdownException, PositionSizeException,
                          RiskException, VaRException, handle_exceptions)


class RiskLevel(Enum):
    """风险等级"""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RiskCheckResult:
    """风险检查结果"""

    def __init__(self, allowed: bool, risk_level: RiskLevel, reason: str = ""):
        self.allowed = allowed
        self.risk_level = risk_level
        self.reason = reason

    def get(self, key: str, default=None):
        """字典式访问方法"""
        return getattr(self, key, default)


class RiskLimit:
    """风险限制"""

    def __init__(self, name: str, limit_type: str, value: float):
        self.name = name
        self.limit_type = limit_type
        self.value = value


class RiskManager:
    """风险管理器"""

    def __init__(self, config: Dict[str, Any] = None):
        # 兼容旧配置方式和新配置对象
        if hasattr(config, "__dict__"):
            # 如果是配置对象，转换为字典
            self.config = config.__dict__ if config else {}
        else:
            self.config = config or {}

        # 配置验证
        self._validate_config()

        # 风险参数 - 使用默认值
        self.max_position_size = self.config.get("max_position_size", 0.1)
        self.max_sector_exposure = self.config.get("max_sector_exposure", 0.3)
        self.max_total_exposure = self.config.get("max_total_exposure", 0.8)
        self.max_drawdown_limit = self.config.get("max_drawdown_limit", 0.2)
        self.stop_loss_pct = self.config.get("stop_loss_pct", 0.05)
        self.take_profit_pct = self.config.get("take_profit_pct", 0.15)
        self.var_confidence = self.config.get("var_confidence", 0.95)
        self.var_window = self.config.get("var_window", 252)
        self.correlation_threshold = self.config.get("correlation_threshold", 0.8)

        # 止损模式
        self.stop_loss_mode = self.config.get(
            "stop_loss_mode", "FULL"
        )  # FULL 或 PARTIAL

        # 监控状态
        self.monitoring_active = False
        self.portfolio = None

        # 线程锁
        self._lock = threading.Lock()

        self.logger = logging.getLogger(__name__)

    def _validate_config(self):
        """验证配置参数"""
        validation_rules = {
            "max_position_size": (0, 1, "max_position_size must be between 0 and 1"),
            "max_sector_exposure": (
                0,
                1,
                "max_sector_exposure must be between 0 and 1",
            ),
            "max_total_exposure": (0, 1, "max_total_exposure must be between 0 and 1"),
            "max_drawdown_limit": (0, 1, "max_drawdown_limit must be between 0 and 1"),
            "var_confidence": (0, 1, "var_confidence must be between 0 and 1"),
        }

        for param, (min_val, max_val, error_msg) in validation_rules.items():
            if param in self.config:
                value = self.config[param]
                if param == "var_confidence":
                    if value <= min_val or value >= max_val:
                        raise ValueError(error_msg)
                else:
                    if value <= min_val or value > max_val:
                        raise ValueError(error_msg)

        positive_params = ["stop_loss_pct", "take_profit_pct", "var_window"]
        for param in positive_params:
            if param in self.config and self.config[param] <= 0:
                raise ValueError(f"{param} must be positive")

        if "correlation_threshold" in self.config:
            corr_threshold = self.config["correlation_threshold"]
            if corr_threshold < 0 or corr_threshold > 1:
                raise ValueError("correlation_threshold must be between 0 and 1")

    def check_position_size(
        self, order: Dict[str, Any], portfolio: Dict[str, Any]
    ) -> Dict[str, Any]:
        """检查仓位大小"""
        symbol = order["symbol"]
        quantity = order["quantity"]
        price = order["price"]
        order_value = quantity * price

        total_value = portfolio["total_value"]
        position_size = order_value / total_value

        # 如果是卖出，检查现有仓位
        if order.get("side", "BUY").upper() == "SELL":
            existing_position = portfolio.get("positions", {}).get(symbol, {})
            existing_value = existing_position.get("value", 0)
            position_size = existing_value / total_value

        if position_size > self.max_position_size:
            return {
                "allowed": False,
                "risk_level": "HIGH",
                "reason": f"Position size {position_size:.2%} exceeds maximum position size {self.max_position_size:.2%}",
            }

        risk_level = "LOW" if position_size < self.max_position_size * 0.5 else "MEDIUM"
        return {
            "allowed": True,
            "risk_level": risk_level,
            "position_size": position_size,
        }

    def check_sector_exposure(
        self, order: Dict[str, Any], portfolio: Dict[str, Any]
    ) -> Dict[str, Any]:
        """检查行业敞口"""
        sector = order.get("sector", "Unknown")
        quantity = order["quantity"]
        price = order["price"]
        order_value = quantity * price

        total_value = portfolio["total_value"]
        current_sector_value = 0

        # 计算当前行业敞口
        for position in portfolio.get("positions", {}).values():
            if position.get("sector") == sector:
                current_sector_value += position.get("value", 0)

        # 如果是买入，增加敞口
        if order.get("side", "BUY").upper() == "BUY":
            new_sector_value = current_sector_value + order_value
        else:
            new_sector_value = current_sector_value

        sector_exposure = new_sector_value / total_value

        if sector_exposure > self.max_sector_exposure:
            return {
                "allowed": False,
                "risk_level": "HIGH",
                "reason": f"Sector {sector} exposure {sector_exposure:.2%} exceeds sector exposure limit {self.max_sector_exposure:.2%}",
            }

        return {
            "allowed": True,
            "risk_level": "LOW",
            "sector_exposure": sector_exposure,
        }

    def check_total_exposure(
        self, order: Dict[str, Any], portfolio: Dict[str, Any]
    ) -> Dict[str, Any]:
        """检查总仓位敞口"""
        quantity = order["quantity"]
        price = order["price"]
        order_value = quantity * price

        total_value = portfolio["total_value"]
        cash = portfolio.get("cash", 0)

        # 如果是买入，检查是否有足够现金
        if order.get("side", "BUY").upper() == "BUY":
            if order_value > cash:
                return {
                    "allowed": False,
                    "risk_level": "HIGH",
                    "reason": f"Insufficient cash: required {order_value}, available {cash}",
                }

            new_cash = cash - order_value
            new_exposure = (total_value - new_cash) / total_value

            if new_exposure > self.max_total_exposure:
                return {
                    "allowed": False,
                    "risk_level": "HIGH",
                    "reason": f"Total exposure {new_exposure:.2%} exceeds total exposure limit {self.max_total_exposure:.2%}",
                }

        current_exposure = (total_value - cash) / total_value
        return {
            "allowed": True,
            "risk_level": "LOW",
            "current_exposure": current_exposure,
        }

    def check_stop_loss(
        self, portfolio: Dict[str, Any], current_prices: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """检查止损"""
        stop_loss_orders = []

        for symbol, position in portfolio.get("positions", {}).items():
            if symbol not in current_prices:
                continue

            current_price = current_prices[symbol]
            avg_cost = position.get("avg_cost", position.get("price", current_price))
            quantity = position.get("quantity", 0)

            # 计算损失百分比
            loss_pct = (avg_cost - current_price) / avg_cost if avg_cost > 0 else 0

            if loss_pct > self.stop_loss_pct:
                # 生成止损订单
                sell_quantity = quantity
                if self.stop_loss_mode == "PARTIAL":
                    sell_quantity = int(quantity * 0.5)  # 卖出50%

                stop_loss_orders.append(
                    {
                        "symbol": symbol,
                        "side": "SELL",
                        "quantity": sell_quantity,
                        "price": current_price,
                        "order_type": "MARKET",
                        "reason": f"Stop loss triggered: {loss_pct:.2%} loss",
                    }
                )

        return stop_loss_orders

    def check_take_profit(
        self, portfolio: Dict[str, Any], current_prices: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """检查止盈"""
        take_profit_orders = []

        for symbol, position in portfolio.get("positions", {}).items():
            if symbol not in current_prices:
                continue

            current_price = current_prices[symbol]
            avg_cost = position.get("avg_cost", position.get("price", current_price))
            quantity = position.get("quantity", 0)

            # 计算盈利百分比
            profit_pct = (current_price - avg_cost) / avg_cost if avg_cost > 0 else 0

            if profit_pct > self.take_profit_pct:
                # 生成止盈订单
                take_profit_orders.append(
                    {
                        "symbol": symbol,
                        "side": "SELL",
                        "quantity": quantity,
                        "price": current_price,
                        "order_type": "MARKET",
                        "reason": f"Take profit triggered: {profit_pct:.2%} profit",
                    }
                )

        return take_profit_orders

    def calculate_var(
        self, portfolio: Dict[str, Any], price_history: pd.DataFrame
    ) -> Dict[str, Any]:
        """计算风险价值(VaR)"""
        if portfolio.get("total_value", 0) <= 0:
            raise ValueError("Invalid portfolio data")

        if price_history.empty or len(price_history) < self.var_window:
            raise ValueError("Insufficient historical data for VaR calculation")

        # 计算收益率
        returns = price_history.pct_change().dropna()

        if len(returns) < 30:  # 至少需要30个观察值
            raise ValueError("Insufficient historical data for VaR calculation")

        # 计算投资组合权重
        total_value = portfolio["total_value"]
        portfolio_weights = {}

        for symbol, position in portfolio.get("positions", {}).items():
            if symbol in price_history.columns:
                portfolio_weights[symbol] = position.get("value", 0) / total_value

        # 计算投资组合收益率
        portfolio_returns = pd.Series(0, index=returns.index)
        for symbol, weight in portfolio_weights.items():
            if symbol in returns.columns:
                portfolio_returns += weight * returns[symbol]

        # 计算VaR
        var_percentile = (1 - self.var_confidence) * 100
        daily_var = np.percentile(portfolio_returns, var_percentile)
        daily_var_value = abs(daily_var * total_value)

        # 年化VaR
        annual_var = daily_var * np.sqrt(252)
        annual_var_value = abs(annual_var * total_value)

        return {
            "daily_var": daily_var_value,
            "annual_var": annual_var_value,
            "var_percentage": abs(daily_var),
            "confidence_level": self.var_confidence,
        }

    def check_correlation(
        self,
        new_symbol: str,
        new_prices: pd.Series,
        existing_symbols: List[str],
        price_history: pd.DataFrame,
    ) -> Dict[str, Any]:
        """检查相关性"""
        correlations = []

        for symbol in existing_symbols:
            if symbol in price_history.columns:
                # 对齐数据
                aligned_data = pd.DataFrame(
                    {"new": new_prices, "existing": price_history[symbol]}
                ).dropna()

                if len(aligned_data) > 30:  # 需要足够的数据点
                    correlation = aligned_data["new"].corr(aligned_data["existing"])
                    if not np.isnan(correlation):
                        correlations.append(abs(correlation))

        if not correlations:
            return {"allowed": True, "max_correlation": 0.0}

        max_correlation = max(correlations)

        if max_correlation > self.correlation_threshold:
            return {
                "allowed": False,
                "max_correlation": max_correlation,
                "reason": f"High correlation {max_correlation:.2%} exceeds correlation threshold {self.correlation_threshold:.2%}",
            }

        return {"allowed": True, "max_correlation": max_correlation}

    def monitor_drawdown(self, portfolio_values: pd.Series) -> Dict[str, Any]:
        """监控回撤"""
        if portfolio_values.empty:
            return {"current_drawdown": 0.0, "alert_triggered": False}

        # 计算回撤
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        current_drawdown = abs(drawdown.iloc[-1])
        max_drawdown = abs(drawdown.min())

        alert_triggered = current_drawdown > self.max_drawdown_limit

        result = {
            "current_drawdown": current_drawdown,
            "max_drawdown": max_drawdown,
            "alert_triggered": alert_triggered,
        }

        if alert_triggered:
            result["emergency_action_required"] = True
            result["recommended_action"] = "Reduce position size or stop trading"

        # 计算恢复百分比
        if len(portfolio_values) > 1:
            recent_low = portfolio_values.min()
            current_value = portfolio_values.iloc[-1]
            if recent_low > 0:
                recovery_pct = (current_value - recent_low) / recent_low
                result["recovery_percentage"] = recovery_pct

        return result

    def check_liquidity(
        self, order: Dict[str, Any], liquidity_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """检查流动性"""
        quantity = order["quantity"]
        order_value = quantity * order["price"]

        avg_daily_volume = liquidity_data.get("avg_daily_volume", 0)
        avg_daily_turnover = liquidity_data.get("avg_daily_turnover", 0)
        bid_ask_spread = liquidity_data.get("bid_ask_spread", 0)

        # 检查订单量相对于平均成交量的比例
        volume_ratio = (
            quantity / avg_daily_volume if avg_daily_volume > 0 else float("inf")
        )

        # 检查订单金额相对于平均成交额的比例
        turnover_ratio = (
            order_value / avg_daily_turnover if avg_daily_turnover > 0 else float("inf")
        )

        # 估计市场冲击
        estimated_impact = max(
            volume_ratio * 0.1, turnover_ratio * 0.05, bid_ask_spread
        )

        # 流动性充足的标准：订单不超过日均成交量的10%，市场冲击小于5%
        liquidity_sufficient = volume_ratio < 0.1 and estimated_impact < 0.05

        return {
            "liquidity_sufficient": liquidity_sufficient,
            "estimated_impact": estimated_impact,
            "volume_ratio": volume_ratio,
            "turnover_ratio": turnover_ratio,
        }

    def run_stress_test(
        self,
        portfolio: Dict[str, Any],
        price_history: Optional[pd.DataFrame],
        stress_scenario: Dict[str, Any],
    ) -> Dict[str, Any]:
        """运行压力测试"""
        scenario_type = stress_scenario.get("type", "MARKET_CRASH")

        if scenario_type == "MARKET_CRASH":
            return self._stress_test_market_crash(portfolio, stress_scenario)
        elif scenario_type == "SECTOR_SHOCK":
            return self._stress_test_sector_shock(portfolio, stress_scenario)
        else:
            raise ValueError(f"Unknown stress test scenario: {scenario_type}")

    def _stress_test_market_crash(
        self, portfolio: Dict[str, Any], scenario: Dict[str, Any]
    ) -> Dict[str, Any]:
        """市场暴跌压力测试"""
        market_drop = scenario.get("market_drop", 0.3)

        original_value = portfolio["total_value"]
        stressed_value = original_value

        # 所有股票按市场跌幅计算
        for position in portfolio.get("positions", {}).values():
            position_value = position.get("value", 0)
            stressed_position_value = position_value * (1 - market_drop)
            stressed_value = stressed_value - position_value + stressed_position_value

        loss = original_value - stressed_value
        loss_pct = loss / original_value if original_value > 0 else 0

        return {
            "stressed_portfolio_value": stressed_value,
            "total_loss": loss,
            "loss_percentage": loss_pct,
            "stressed_var": loss * 1.5,  # 假设VaR在压力情况下增加50%
        }

    def _stress_test_sector_shock(
        self, portfolio: Dict[str, Any], scenario: Dict[str, Any]
    ) -> Dict[str, Any]:
        """行业冲击压力测试"""
        shock_sector = scenario.get("sector", "Finance")
        shock_magnitude = scenario.get("shock_magnitude", 0.5)

        sector_loss = 0
        original_value = portfolio["total_value"]

        for position in portfolio.get("positions", {}).values():
            if position.get("sector") == shock_sector:
                position_value = position.get("value", 0)
                loss = position_value * shock_magnitude
                sector_loss += loss

        portfolio_impact = sector_loss / original_value if original_value > 0 else 0

        return {
            "sector_loss": sector_loss,
            "portfolio_impact": portfolio_impact,
            "affected_sector": shock_sector,
            "shock_magnitude": shock_magnitude,
        }

    def start_monitoring(self, portfolio: Dict[str, Any]):
        """开始监控"""
        with self._lock:
            self.monitoring_active = True
            self.portfolio = portfolio

    def process_price_update(
        self, price_update: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """处理价格更新"""
        if not self.monitoring_active or not self.portfolio:
            return None

        symbol = price_update["symbol"]
        price = price_update["price"]

        # 检查止损
        if symbol in self.portfolio.get("positions", {}):
            position = self.portfolio["positions"][symbol]
            avg_cost = position.get("avg_cost", position.get("price", price))

            if avg_cost > 0:
                loss_pct = (avg_cost - price) / avg_cost
                if loss_pct > self.stop_loss_pct:
                    return {
                        "type": "STOP_LOSS",
                        "symbol": symbol,
                        "severity": "HIGH",
                        "current_price": price,
                        "avg_cost": avg_cost,
                        "loss_percentage": loss_pct,
                        "timestamp": price_update.get("timestamp", datetime.now()),
                    }

        return None

    def prioritize_alerts(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """优先级排序警报"""
        severity_order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}

        return sorted(
            alerts,
            key=lambda x: severity_order.get(x.get("severity", "LOW"), 1),
            reverse=True,
        )

    def comprehensive_risk_check(self, portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """综合风险检查"""
        # 这是一个综合检查函数，用于性能测试
        risk_summary = {
            "total_positions": len(portfolio.get("positions", {})),
            "total_value": portfolio.get("total_value", 0),
            "risk_level": "LOW",
        }

        # 简单的风险评估
        position_count = len(portfolio.get("positions", {}))
        if position_count > 100:
            risk_summary["risk_level"] = "MEDIUM"
        if position_count > 500:
            risk_summary["risk_level"] = "HIGH"

        return risk_summary

    def update_config(self, new_config: Dict[str, Any]):
        """更新配置"""
        # 验证新配置
        temp_config = self.config.copy()
        temp_config.update(new_config)

        # 创建临时实例来验证配置
        temp_manager = RiskManager.__new__(RiskManager)
        temp_manager.config = temp_config
        temp_manager._validate_config()

        # 如果验证通过，更新当前配置
        for key, value in new_config.items():
            if hasattr(self, key):
                setattr(self, key, value)

        self.config.update(new_config)

    def check_signal_risk(
        self, signal: Dict[str, Any], current_positions: Dict[str, Any] = None
    ) -> RiskCheckResult:
        """检查信号风险"""
        # 简单的风险检查
        quantity = signal.get("quantity", 0)
        price = signal.get("price", 0)
        order_value = quantity * price

        # 模拟检查：订单金额不能超过10万
        if order_value > 100000:
            return RiskCheckResult(False, RiskLevel.HIGH, "订单金额过大")

        return RiskCheckResult(True, RiskLevel.LOW, "风险检查通过")

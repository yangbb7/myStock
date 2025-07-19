# -*- coding: utf-8 -*-
"""
PortfolioManager - 投资组合管理器模块
"""

import logging
import threading
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class PositionSide(Enum):
    """持仓方向"""

    LONG = "LONG"
    SHORT = "SHORT"


class RebalanceFrequency(Enum):
    """再平衡频率"""

    DAILY = "DAILY"
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"


class Position:
    """持仓类"""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.quantity = 0
        self.avg_cost = 0.0
        self.current_price = 0.0
        self.market_value = 0.0
        self.unrealized_pnl = 0.0


class Transaction:
    """交易记录"""

    def __init__(self, symbol: str, side: str, quantity: int, price: float):
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.price = price
        self.timestamp = datetime.now()


class PortfolioManager:
    """投资组合管理器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # 配置验证
        self._validate_config()

        self.initial_capital = config["initial_capital"]
        self.current_cash = self.initial_capital
        self.base_currency = config.get("base_currency", "CNY")
        self.commission_rate = config.get("commission_rate", 0.0003)
        self.min_commission = config.get("min_commission", 5.0)
        self.min_position_value = config.get("min_position_value", 1000)
        self.max_positions = config.get("max_positions", 50)
        self.cash_buffer = config.get("cash_buffer", 0.05)
        self.rebalance_frequency = config.get("rebalance_frequency", "monthly")
        self.rebalance_threshold = config.get("rebalance_threshold", 0.05)

        self.positions = {}
        self.transaction_history = []
        self.target_weights = config.get("target_weights", {})

        # 线程锁用于并发安全
        self._lock = threading.Lock()

        self.logger = logging.getLogger(__name__)

        # 计算总价值的属性
        self.total_value = self.initial_capital

        # 外部组件依赖
        self.data_manager = None

    def _validate_config(self):
        """验证配置参数"""
        if "initial_capital" not in self.config:
            raise KeyError("initial_capital is required in config")

        if self.config["initial_capital"] <= 0:
            raise ValueError("Initial capital must be positive")

        if "commission_rate" in self.config:
            if self.config["commission_rate"] < 0:
                raise ValueError("commission_rate must be non-negative")

        if "max_positions" in self.config:
            if self.config["max_positions"] <= 0:
                raise ValueError("max_positions must be positive")

        if "cash_buffer" in self.config:
            if not (0 <= self.config["cash_buffer"] <= 1):
                raise ValueError("cash_buffer must be between 0 and 1")

        if "min_position_value" in self.config:
            if self.config["min_position_value"] <= 0:
                raise ValueError("min_position_value must be positive")

    def validate_order(self, order: Dict[str, Any]):
        """验证订单"""
        if not order.get("symbol") or order["symbol"].strip() == "":
            raise ValueError("Symbol cannot be empty")

        if order.get("side") not in ["BUY", "SELL"]:
            raise ValueError("Invalid order side")

        if order.get("quantity", 0) <= 0:
            raise ValueError("Quantity must be positive")

        if order.get("price", 0) <= 0:
            raise ValueError("Price must be positive")

    def update_position(self, execution_result: Dict[str, Any]):
        """更新持仓"""
        with self._lock:
            symbol = execution_result["symbol"]
            quantity = execution_result["quantity"]
            price = execution_result["price"]
            commission = execution_result.get("commission", 0)
            side = execution_result.get("side", "BUY")

            if symbol not in self.positions:
                if side.upper() == "SELL":
                    raise ValueError("Position not found")

                self.positions[symbol] = {
                    "quantity": 0,
                    "avg_cost": 0.0,
                    "current_price": price,
                    "market_value": 0.0,
                    "unrealized_pnl": 0.0,
                    "sector": execution_result.get("sector", "Unknown"),
                    "weight": 0.0,
                    "last_updated": datetime.now(),
                }

            position = self.positions[symbol]

            if side.upper() == "BUY":
                # 买入
                total_cost = quantity * price + commission
                if total_cost > self.current_cash:
                    raise ValueError("Insufficient cash")

                if position["quantity"] == 0:
                    position["avg_cost"] = price
                else:
                    total_value = (
                        position["avg_cost"] * position["quantity"] + price * quantity
                    )
                    total_quantity = position["quantity"] + quantity
                    position["avg_cost"] = total_value / total_quantity

                position["quantity"] += quantity
                position["current_price"] = price
                position["market_value"] = position["quantity"] * price
                position["unrealized_pnl"] = (price - position["avg_cost"]) * position[
                    "quantity"
                ]
                position["last_updated"] = datetime.now()

                self.current_cash -= total_cost

            elif side.upper() == "SELL":
                # 卖出
                if position["quantity"] < quantity:
                    raise ValueError("Insufficient shares")

                position["quantity"] -= quantity
                total_proceeds = quantity * price - commission
                self.current_cash += total_proceeds

                if position["quantity"] == 0:
                    del self.positions[symbol]
                else:
                    position["current_price"] = price
                    position["market_value"] = position["quantity"] * price
                    position["unrealized_pnl"] = (
                        price - position["avg_cost"]
                    ) * position["quantity"]
                    position["last_updated"] = datetime.now()

            # 记录交易
            self.record_transaction(
                {
                    "timestamp": datetime.now(),
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "price": price,
                    "commission": commission,
                    "total_value": (
                        quantity * price + commission
                        if side.upper() == "BUY"
                        else quantity * price - commission
                    ),
                }
            )

            # 更新权重
            self.calculate_weights()

    def update_prices(self, price_data: Dict[str, float]):
        """更新价格"""
        with self._lock:
            for symbol, position in self.positions.items():
                if symbol in price_data:
                    new_price = price_data[symbol]

                    # 验证价格有效性
                    if new_price > 0 and not np.isnan(new_price):
                        position["current_price"] = new_price
                        position["market_value"] = position["quantity"] * new_price
                        position["unrealized_pnl"] = (
                            new_price - position["avg_cost"]
                        ) * position["quantity"]
                        position["last_updated"] = datetime.now()

            # 更新权重
            self.calculate_weights()

    def calculate_total_value(self) -> float:
        """计算总价值"""
        position_value = sum(pos["market_value"] for pos in self.positions.values())
        self.total_value = self.current_cash + position_value
        return self.total_value

    def calculate_unrealized_pnl(self) -> float:
        """计算未实现盈亏"""
        return sum(pos["unrealized_pnl"] for pos in self.positions.values())

    def calculate_weights(self):
        """计算权重"""
        total_value = self.calculate_total_value()
        if total_value > 0:
            for position in self.positions.values():
                position["weight"] = position["market_value"] / total_value

    def check_rebalance_needed(
        self, target_weights: Dict[str, float], tolerance: float = 0.02
    ) -> bool:
        """检查是否需要再平衡"""
        self.calculate_weights()

        for symbol, target_weight in target_weights.items():
            current_weight = self.positions.get(symbol, {}).get("weight", 0.0)
            if abs(current_weight - target_weight) > tolerance:
                return True

        return False

    def generate_rebalance_orders(
        self, target_weights: Dict[str, float], price_data: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """生成再平衡订单"""
        orders = []
        total_value = self.calculate_total_value()

        for symbol, target_weight in target_weights.items():
            target_value = total_value * target_weight
            current_value = self.positions.get(symbol, {}).get("market_value", 0.0)
            difference = target_value - current_value

            if symbol in price_data and abs(difference) > self.min_position_value:
                price = price_data[symbol]
                quantity = abs(int(difference / price))

                if quantity > 0:
                    side = "BUY" if difference > 0 else "SELL"

                    # 检查现金约束
                    if side == "BUY":
                        required_cash = quantity * price * (1 + self.commission_rate)
                        if required_cash > self.current_cash:
                            # 现金不足，减少购买数量
                            quantity = int(
                                self.current_cash / (price * (1 + self.commission_rate))
                            )
                            if quantity == 0:
                                continue

                    orders.append(
                        {
                            "symbol": symbol,
                            "side": side,
                            "quantity": quantity,
                            "price": price,
                            "order_type": "MARKET",
                        }
                    )

        return orders

    def calculate_portfolio_beta(self, stock_betas: Dict[str, float]) -> float:
        """计算投资组合Beta"""
        total_value = sum(pos["market_value"] for pos in self.positions.values())
        if total_value == 0:
            return 0.0

        weighted_beta = 0.0
        for symbol, position in self.positions.items():
            if symbol in stock_betas:
                weight = position["market_value"] / total_value
                weighted_beta += weight * stock_betas[symbol]

        return weighted_beta

    def calculate_sector_exposure(self) -> Dict[str, float]:
        """计算行业敞口"""
        sector_exposure = {}
        for position in self.positions.values():
            sector = position.get("sector", "Unknown")
            if sector not in sector_exposure:
                sector_exposure[sector] = 0.0
            sector_exposure[sector] += position["market_value"]

        return sector_exposure

    def calculate_concentration_risk(self) -> Dict[str, Any]:
        """计算集中度风险"""
        total_value = sum(pos["market_value"] for pos in self.positions.values())
        if total_value == 0:
            return {
                "herfindahl_index": 0.0,
                "top_holdings_percentage": 0.0,
                "number_of_holdings": 0,
            }

        weights = [pos["market_value"] / total_value for pos in self.positions.values()]
        weights.sort(reverse=True)

        # 赫芬达尔指数
        herfindahl_index = sum(w**2 for w in weights)

        # 前十大持仓占比
        top_holdings_percentage = (
            sum(weights[:10]) if len(weights) >= 10 else sum(weights)
        )

        return {
            "herfindahl_index": herfindahl_index,
            "top_holdings_percentage": top_holdings_percentage,
            "number_of_holdings": len(self.positions),
        }

    def calculate_commission(self, order_value: float) -> float:
        """计算佣金"""
        commission = order_value * self.commission_rate
        return max(commission, self.min_commission)

    def calculate_total_trading_costs(
        self, executed_orders: List[Dict[str, Any]]
    ) -> float:
        """计算总交易成本"""
        total_costs = 0.0
        for order in executed_orders:
            total_costs += order.get("commission", 0.0) + order.get("slippage", 0.0)
        return total_costs

    def check_cash_sufficiency(self, order: Dict[str, Any]) -> bool:
        """检查现金充足性"""
        if order["side"].upper() == "BUY":
            required_cash = order["quantity"] * order["price"]
            commission = self.calculate_commission(required_cash)
            total_required = required_cash + commission
            return self.current_cash >= total_required
        return True

    def optimize_cash_allocation(
        self, opportunities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """优化现金分配"""
        # 简单的贪心算法：按预期收益率排序
        sorted_opportunities = sorted(
            opportunities, key=lambda x: x["expected_return"], reverse=True
        )

        allocations = []
        remaining_cash = self.current_cash

        for opp in sorted_opportunities:
            if remaining_cash >= opp["required_cash"]:
                allocations.append(
                    {"symbol": opp["symbol"], "amount": opp["required_cash"]}
                )
                remaining_cash -= opp["required_cash"]

        return allocations

    def calculate_returns(self, historical_values: pd.Series) -> pd.Series:
        """计算收益率"""
        return historical_values.pct_change().dropna()

    def calculate_sharpe_ratio(
        self, returns: pd.Series, risk_free_rate: float
    ) -> float:
        """计算夏普比率"""
        if returns.empty or returns.std() == 0:
            return 0.0

        excess_returns = returns - risk_free_rate / 252  # 日化无风险收益率
        return excess_returns.mean() / returns.std() * np.sqrt(252)  # 年化夏普比率

    def calculate_max_drawdown(self, portfolio_values: pd.Series) -> float:
        """计算最大回撤"""
        if portfolio_values.empty:
            return 0.0

        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        return drawdown.min()

    def record_transaction(self, transaction: Dict[str, Any]):
        """记录交易"""
        self.transaction_history.append(transaction)

    def get_transaction_history(
        self, symbol: str = None, start_date: datetime = None, end_date: datetime = None
    ) -> List[Dict[str, Any]]:
        """获取交易历史"""
        filtered_history = self.transaction_history

        if symbol:
            filtered_history = [t for t in filtered_history if t["symbol"] == symbol]

        if start_date:
            filtered_history = [
                t for t in filtered_history if t["timestamp"] >= start_date
            ]

        if end_date:
            filtered_history = [
                t for t in filtered_history if t["timestamp"] <= end_date
            ]

        return filtered_history

    def save_state(self) -> Dict[str, Any]:
        """保存投资组合状态"""
        return {
            "positions": self.positions.copy(),
            "current_cash": self.current_cash,
            "timestamp": datetime.now().isoformat(),
        }

    def load_state(self, state_data: Dict[str, Any]):
        """加载投资组合状态"""
        self.positions = state_data.get("positions", {})
        self.current_cash = state_data.get("current_cash", self.initial_capital)
        self.calculate_weights()

    def get_current_positions(self) -> Dict[str, Any]:
        """获取当前持仓"""
        return self.positions

    def process_signal(self, signal: Dict[str, Any]):
        """处理信号"""
        order = {
            "symbol": signal["symbol"],
            "side": signal["signal_type"],
            "quantity": signal["quantity"],
            "price": signal["price"],
        }
        try:
            self.update_position(order)
            return order  # 返回创建的订单
        except Exception as e:
            self.logger.error(f"Error processing signal: {e}")
            return None

    def resolve_signal_conflicts(
        self, signals: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """解决信号冲突"""
        # 简单处理：返回原信号
        return signals

    def create_order_from_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """从信号创建订单"""
        return {
            "symbol": signal["symbol"],
            "side": signal["signal_type"],
            "quantity": signal["quantity"],
            "price": signal["price"],
            "order_type": "MARKET",
        }

    def get_positions(self) -> Dict[str, Any]:
        """获取持仓信息"""
        return self.positions

    def get_positions_with_fallback(self) -> Dict[str, Any]:
        """获取持仓信息（带回退机制）"""
        try:
            # 尝试从数据管理器获取最新价格
            if self.data_manager:
                for symbol in self.positions:
                    try:
                        current_price = self.data_manager.get_current_price(symbol)
                        if current_price:
                            self.positions[symbol]["current_price"] = current_price
                    except Exception:
                        # 数据管理器故障时使用缓存价格
                        self.logger.warning(
                            f"Failed to get current price for {symbol}, using cached price"
                        )
                        continue
        except Exception as e:
            self.logger.warning(
                f"Data manager error in get_positions_with_fallback: {e}"
            )

        # 无论如何都返回当前持仓
        return self.positions

    def get_position_price(self, symbol: str) -> float:
        """获取持仓价格"""
        if symbol in self.positions:
            return self.positions[symbol]["current_price"]
        return 0.0

    def sync_risk_config(self, risk_manager):
        """同步风险配置"""
        self.max_position_size = getattr(risk_manager, "max_position_size", 0.1)

    def get_position(self, symbol: str) -> Dict[str, Any]:
        """获取指定股票的持仓信息"""
        if symbol in self.positions:
            return self.positions[symbol]
        return {"quantity": 0, "avg_cost": 0.0, "market_value": 0.0}

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

from .exceptions import (ConfigurationException, InsufficientFundsException,
                          PortfolioException, PositionException,
                          handle_exceptions)


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
        # 兼容旧配置方式和新配置对象
        if hasattr(config, "__dict__"):
            # 如果是配置对象，转换为字典
            self.config = config.__dict__ if config else {}
        else:
            self.config = config or {}

        # 初始化日志 (在配置验证之前)
        self.logger = logging.getLogger(__name__)

        # 配置验证
        self._validate_config()

        self.initial_capital = self.config["initial_capital"]
        self.current_cash = self.initial_capital
        self.base_currency = self.config.get("base_currency", "CNY")
        self.commission_rate = self.config.get("commission_rate", 0.0003)
        self.min_commission = self.config.get("min_commission", 5.0)
        self.min_position_value = self.config.get("min_position_value", 1000)
        self.max_positions = self.config.get("max_positions", 50)
        self.cash_buffer = self.config.get("cash_buffer", 0.05)
        self.rebalance_frequency = self.config.get("rebalance_frequency", "monthly")
        self.rebalance_threshold = self.config.get("rebalance_threshold", 0.05)

        self.positions = {}
        self.transaction_history = []
        self.target_weights = self.config.get("target_weights", {})

        # 线程锁用于并发安全
        self._lock = threading.RLock()

        # 计算总价值的属性
        self.total_value = self.initial_capital

        # 外部组件依赖
        self.data_manager = None

    def _validate_config(self):
        """验证配置参数"""
        # 为initial_capital提供默认值
        if "initial_capital" not in self.config:
            self.config["initial_capital"] = 1000000  # 默认100万
            self.logger.warning("initial_capital not provided, using default value: 1000000")

        if self.config["initial_capital"] <= 0:
            raise ConfigurationException(
                "Initial capital must be positive", config_key="initial_capital"
            )

        if "commission_rate" in self.config:
            if self.config["commission_rate"] < 0:
                raise ConfigurationException(
                    "commission_rate must be non-negative", config_key="commission_rate"
                )

        if "max_positions" in self.config:
            if self.config["max_positions"] <= 0:
                raise ConfigurationException(
                    "max_positions must be positive", config_key="max_positions"
                )

        if "cash_buffer" in self.config:
            if not (0 <= self.config["cash_buffer"] <= 1):
                raise ConfigurationException(
                    "cash_buffer must be between 0 and 1", config_key="cash_buffer"
                )

        if "min_position_value" in self.config:
            if self.config["min_position_value"] <= 0:
                raise ConfigurationException(
                    "min_position_value must be positive",
                    config_key="min_position_value",
                )

    def validate_order(self, order: Dict[str, Any]):
        """验证订单"""
        if not order.get("symbol") or order["symbol"].strip() == "":
            raise PortfolioException(
                "Symbol cannot be empty", operation="validate_order"
            )

        if order.get("side") not in ["BUY", "SELL"]:
            raise PortfolioException("Invalid order side", operation="validate_order")

        if order.get("quantity", 0) <= 0:
            raise PortfolioException(
                "Quantity must be positive", operation="validate_order"
            )

        if order.get("price", 0) <= 0:
            raise PortfolioException(
                "Price must be positive", operation="validate_order"
            )

    @handle_exceptions(reraise_types=(PortfolioException, PositionException, ValueError, TypeError))
    def update_position(self, execution_result: Dict[str, Any]):
        """更新持仓"""
        with self._lock:
            symbol = execution_result["symbol"]
            quantity = execution_result["quantity"]
            price = execution_result["price"]
            commission = execution_result.get("commission", 0)
            side = execution_result.get("side", "BUY")
            
            # 输入验证
            if not symbol or not isinstance(symbol, str):
                raise ValueError("Symbol must be a non-empty string")
            if not isinstance(quantity, (int, float)) or quantity <= 0:
                raise ValueError("Quantity must be a positive number")
            if not isinstance(price, (int, float)) or price <= 0:
                raise ValueError("Price must be a positive number")
            if side.upper() not in ["BUY", "SELL"]:
                raise ValueError("Side must be either 'BUY' or 'SELL'")

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
                    raise PortfolioException("Insufficient shares", operation="update_position")

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
    
    def update_position_from_execution(self, execution_result: Dict[str, Any]):
        """从执行结果更新持仓"""
        symbol = execution_result['symbol']
        side = execution_result['side']
        quantity = execution_result['executed_quantity']
        price = execution_result['executed_price']
        commission = execution_result['commission']
        
        with self._lock:
            if symbol not in self.positions:
                self.positions[symbol] = {
                    'quantity': 0,
                    'avg_cost': 0.0,
                    'current_price': price,
                    'market_value': 0.0,
                    'unrealized_pnl': 0.0,
                    'last_updated': datetime.now()
                }
            
            position = self.positions[symbol]
            
            if side.upper() == 'BUY':
                # 买入逻辑
                old_quantity = position['quantity']
                old_cost = position['avg_cost']
                
                total_cost = quantity * price + commission
                new_quantity = old_quantity + quantity
                
                if new_quantity > 0:
                    # 计算新的平均成本
                    position['avg_cost'] = (old_quantity * old_cost + total_cost) / new_quantity
                    position['quantity'] = new_quantity
                    
                # 扣除现金
                self.current_cash -= total_cost
                
            elif side.upper() == 'SELL':
                # 卖出逻辑
                position['quantity'] -= quantity
                proceeds = quantity * price - commission
                self.current_cash += proceeds
                
                if position['quantity'] <= 0:
                    del self.positions[symbol]
                    return
            
            # 更新市价和盈亏
            position['current_price'] = price
            position['market_value'] = position['quantity'] * price
            position['unrealized_pnl'] = (price - position['avg_cost']) * position['quantity']
            position['last_updated'] = datetime.now()
    
    def update_market_price(self, symbol: str, price: float):
        """更新市场价格"""
        if symbol in self.positions:
            position = self.positions[symbol]
            position['current_price'] = price
            position['market_value'] = position['quantity'] * price
            position['unrealized_pnl'] = (price - position['avg_cost']) * position['quantity']
            position['last_updated'] = datetime.now()

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

    def _calculate_portfolio_value_unlocked(self, current_prices: Dict[str, float] = None) -> float:
        """计算投资组合价值（内部无锁版本）
        
        Args:
            current_prices: 当前价格字典，如果提供则使用这些价格更新持仓
            
        Returns:
            float: 投资组合总价值（现金 + 持仓市值）
        """
        total_position_value = 0.0
        
        for symbol, position in self.positions.items():
            if current_prices and symbol in current_prices:
                # 使用提供的最新价格
                current_price = current_prices[symbol]
                if current_price > 0 and not np.isnan(current_price):
                    # 更新持仓价格和市值
                    position["current_price"] = current_price
                    position["market_value"] = position["quantity"] * current_price
                    position["unrealized_pnl"] = (current_price - position["avg_cost"]) * position["quantity"]
                    position["last_updated"] = datetime.now()
            
            # 累加持仓市值
            total_position_value += position["market_value"]
        
        # 总价值 = 现金 + 持仓市值
        total_value = self.current_cash + total_position_value
        self.total_value = total_value
        
        # 更新权重
        if total_value > 0:
            for position in self.positions.values():
                position["weight"] = position["market_value"] / total_value
        
        return total_value

    def calculate_portfolio_value(self, current_prices: Dict[str, float] = None) -> float:
        """计算投资组合价值
        
        Args:
            current_prices: 当前价格字典，如果提供则使用这些价格更新持仓
            
        Returns:
            float: 投资组合总价值（现金 + 持仓市值）
        """
        with self._lock:
            return self._calculate_portfolio_value_unlocked(current_prices)

    def validate_position_size(self, symbol: str, intended_value: float, current_prices: Dict[str, float] = None) -> bool:
        """验证仓位大小是否符合风险控制要求
        
        Args:
            symbol: 股票代码
            intended_value: 预期投资金额
            current_prices: 当前价格字典（可选）
            
        Returns:
            bool: True表示通过验证，False表示超出限制
        """
        with self._lock:
            # 获取当前投资组合价值（使用内部无锁版本）
            total_portfolio_value = self._calculate_portfolio_value_unlocked(current_prices)
            
            if total_portfolio_value <= 0:
                self.logger.warning("Portfolio value is zero or negative, rejecting position")
                return False
            
            # 计算预期仓位占比
            position_percentage = intended_value / total_portfolio_value
            
            # 检查单个仓位限制（默认最大10%）
            max_single_position = self.config.get('max_single_position', 0.1)
            if position_percentage > max_single_position:
                self.logger.warning(
                    f"Position size {position_percentage:.2%} exceeds maximum single position limit {max_single_position:.2%} for {symbol}"
                )
                return False
            
            # 检查最小仓位价值
            if intended_value < self.min_position_value:
                self.logger.warning(
                    f"Position value {intended_value} is below minimum position value {self.min_position_value} for {symbol}"
                )
                return False
            
            # 检查现有仓位 + 新仓位的总大小
            current_position_value = 0.0
            if symbol in self.positions:
                current_position_value = self.positions[symbol]["market_value"]
            
            total_position_value = current_position_value + intended_value
            total_position_percentage = total_position_value / total_portfolio_value
            
            if total_position_percentage > max_single_position:
                self.logger.warning(
                    f"Total position size {total_position_percentage:.2%} would exceed maximum limit {max_single_position:.2%} for {symbol}"
                )
                return False
            
            # 检查总仓位数量限制
            if symbol not in self.positions and len(self.positions) >= self.max_positions:
                self.logger.warning(
                    f"Cannot add new position for {symbol}: maximum positions limit {self.max_positions} reached"
                )
                return False
            
            # 检查行业集中度（如果有行业信息）
            if hasattr(self, '_check_sector_concentration'):
                if not self._check_sector_concentration(symbol, intended_value, total_portfolio_value):
                    return False
            
            # 检查现金充足性（对于买入操作）
            commission = self.calculate_commission(intended_value)
            total_cost = intended_value + commission
            
            # 保留现金缓冲
            required_cash_buffer = total_portfolio_value * self.cash_buffer
            available_cash = self.current_cash - required_cash_buffer
            
            if total_cost > available_cash:
                self.logger.warning(
                    f"Insufficient cash: need {total_cost}, available {available_cash} (after {self.cash_buffer:.1%} buffer) for {symbol}"
                )
                return False
            
            self.logger.info(f"Position validation passed for {symbol}: {position_percentage:.2%} of portfolio")
            return True

    def _check_sector_concentration(self, symbol: str, intended_value: float, total_portfolio_value: float) -> bool:
        """检查行业集中度限制（辅助方法）"""
        # 获取股票的行业信息（这里简化处理，实际应该从数据源获取）
        sector = self._get_symbol_sector(symbol)
        if not sector or sector == "Unknown":
            return True  # 如果没有行业信息，跳过检查
        
        # 计算当前行业敞口
        sector_exposure = self.calculate_sector_exposure()
        current_sector_value = sector_exposure.get(sector, 0.0)
        new_sector_value = current_sector_value + intended_value
        new_sector_percentage = new_sector_value / total_portfolio_value
        
        # 行业集中度限制（默认最大30%）
        max_sector_concentration = self.config.get('max_sector_concentration', 0.3)
        
        if new_sector_percentage > max_sector_concentration:
            self.logger.warning(
                f"Sector concentration {new_sector_percentage:.2%} would exceed maximum limit {max_sector_concentration:.2%} for sector {sector}"
            )
            return False
        
        return True
    
    def _get_symbol_sector(self, symbol: str) -> str:
        """获取股票所属行业（简化实现）"""
        # 这里可以从现有仓位或外部数据源获取行业信息
        if symbol in self.positions:
            return self.positions[symbol].get("sector", "Unknown")
        
        # 简化的行业分类逻辑（实际应该从数据源获取）
        if symbol.startswith("60"):  # 上交所主板
            return "Traditional"
        elif symbol.startswith("00"):  # 深交所主板/中小板
            return "Growth"
        elif symbol.startswith("30"):  # 创业板
            return "Technology"
        else:
            return "Unknown"

    def rebalance_portfolio(self, target_weights: Dict[str, float], current_prices: Dict[str, float], 
                           tolerance: float = None) -> List[Dict[str, Any]]:
        """执行投资组合再平衡
        
        Args:
            target_weights: 目标权重字典 {symbol: weight}
            current_prices: 当前价格字典 {symbol: price}
            tolerance: 偏差容忍度，超过此值才执行再平衡
            
        Returns:
            List[Dict]: 再平衡订单列表
        """
        if tolerance is None:
            tolerance = self.rebalance_threshold
        
        with self._lock:
            # 更新投资组合价值和权重
            total_value = self._calculate_portfolio_value_unlocked(current_prices)
            
            if total_value <= 0:
                self.logger.warning("Portfolio value is zero or negative, cannot rebalance")
                return []
            
            # 检查目标权重是否合理
            total_target_weight = sum(target_weights.values())
            if abs(total_target_weight - 1.0) > 0.001:  # 允许小的浮点误差
                self.logger.warning(f"Target weights sum to {total_target_weight:.3f}, not 1.0")
                # 归一化权重
                target_weights = {symbol: weight/total_target_weight for symbol, weight in target_weights.items()}
            
            rebalance_orders = []
            total_deviation = 0.0
            
            # 计算所有股票的偏差
            deviations = {}
            for symbol, target_weight in target_weights.items():
                current_weight = self.positions.get(symbol, {}).get("weight", 0.0)
                deviation = abs(target_weight - current_weight)
                deviations[symbol] = {
                    'current_weight': current_weight,
                    'target_weight': target_weight,
                    'deviation': deviation,
                    'needs_rebalance': deviation > tolerance
                }
                total_deviation += deviation
            
            # 如果总偏差小于容忍度，不需要再平衡
            if total_deviation < tolerance:
                self.logger.info(f"Portfolio deviation {total_deviation:.3f} below tolerance {tolerance:.3f}, no rebalancing needed")
                return []
            
            self.logger.info(f"Starting portfolio rebalance with total deviation {total_deviation:.3f}")
            
            # 预留现金缓冲
            available_value = total_value * (1 - self.cash_buffer)
            
            # 生成再平衡订单
            for symbol, target_weight in target_weights.items():
                if symbol not in current_prices:
                    self.logger.warning(f"No price available for {symbol}, skipping rebalancing")
                    continue
                
                deviation_info = deviations[symbol]
                if not deviation_info['needs_rebalance']:
                    continue
                
                current_price = current_prices[symbol]
                target_value = available_value * target_weight
                current_value = self.positions.get(symbol, {}).get("market_value", 0.0)
                value_difference = target_value - current_value
                
                # 计算需要的股票数量变化
                if abs(value_difference) < self.min_position_value:
                    continue  # 变化太小，跳过
                
                if value_difference > 0:
                    # 需要买入
                    quantity_to_buy = int(value_difference / current_price)
                    
                    if quantity_to_buy > 0:
                        order_value = quantity_to_buy * current_price
                        commission = self.calculate_commission(order_value)
                        total_cost = order_value + commission
                        
                        # 验证现金充足性
                        if total_cost <= self.current_cash:
                            # 验证仓位大小
                            if self.validate_position_size(symbol, order_value, current_prices):
                                rebalance_orders.append({
                                    'symbol': symbol,
                                    'side': 'BUY',
                                    'quantity': quantity_to_buy,
                                    'price': current_price,
                                    'order_type': 'MARKET',
                                    'reason': 'REBALANCE',
                                    'target_weight': target_weight,
                                    'current_weight': deviation_info['current_weight'],
                                    'estimated_cost': total_cost
                                })
                                # 扣除预期现金使用，用于后续订单计算
                                self.current_cash -= total_cost
                            else:
                                self.logger.warning(f"Position size validation failed for {symbol}, skipping buy order")
                        else:
                            self.logger.warning(f"Insufficient cash for {symbol}: need {total_cost}, have {self.current_cash}")
                
                else:
                    # 需要卖出
                    current_quantity = self.positions.get(symbol, {}).get("quantity", 0)
                    if current_quantity > 0:
                        quantity_to_sell = min(int(abs(value_difference) / current_price), current_quantity)
                        
                        if quantity_to_sell > 0:
                            order_value = quantity_to_sell * current_price
                            commission = self.calculate_commission(order_value)
                            proceeds = order_value - commission
                            
                            rebalance_orders.append({
                                'symbol': symbol,
                                'side': 'SELL',
                                'quantity': quantity_to_sell,
                                'price': current_price,
                                'order_type': 'MARKET',
                                'reason': 'REBALANCE',
                                'target_weight': target_weight,
                                'current_weight': deviation_info['current_weight'],
                                'estimated_proceeds': proceeds
                            })
                            # 增加预期现金收入
                            self.current_cash += proceeds
            
            # 处理需要清仓的股票（目标权重为0但当前有持仓）
            for symbol, position in self.positions.items():
                if symbol not in target_weights and position["quantity"] > 0:
                    if symbol in current_prices:
                        current_price = current_prices[symbol]
                        quantity_to_sell = position["quantity"]
                        order_value = quantity_to_sell * current_price
                        commission = self.calculate_commission(order_value)
                        proceeds = order_value - commission
                        
                        rebalance_orders.append({
                            'symbol': symbol,
                            'side': 'SELL',
                            'quantity': quantity_to_sell,
                            'price': current_price,
                            'order_type': 'MARKET',
                            'reason': 'LIQUIDATE',
                            'target_weight': 0.0,
                            'current_weight': position.get("weight", 0.0),
                            'estimated_proceeds': proceeds
                        })
            
            # 恢复现金到实际值（上面的扣除只是为了计算）
            self.current_cash = total_value - sum(pos["market_value"] for pos in self.positions.values())
            
            self.logger.info(f"Generated {len(rebalance_orders)} rebalancing orders")
            
            return rebalance_orders

    def calculate_performance_metrics(self, current_prices: Dict[str, float] = None) -> Dict[str, Any]:
        """计算基础绩效指标
        
        Note: 这个方法提供基础绩效指标。复杂的分析（如夏普比率、最大回撤等）
              应该使用专门的PerformanceAnalyzer组件。
        
        Args:
            current_prices: 当前价格字典（可选）
            
        Returns:
            Dict: 基础绩效指标字典
        """
        with self._lock:
            # 计算当前投资组合价值
            current_value = self._calculate_portfolio_value_unlocked(current_prices)
            
            if self.initial_capital <= 0:
                return {
                    'error': 'Invalid initial capital',
                    'current_value': current_value,
                    'total_return': 0.0
                }
            
            # 基础收益指标
            total_return = (current_value - self.initial_capital) / self.initial_capital
            total_invested = self.initial_capital - self.current_cash
            investment_ratio = total_invested / self.initial_capital if self.initial_capital > 0 else 0.0
            
            # 持仓相关指标
            position_count = len(self.positions)
            total_position_value = sum(pos["market_value"] for pos in self.positions.values())
            cash_ratio = self.current_cash / current_value if current_value > 0 else 1.0
            
            # 计算已实现和未实现盈亏
            total_unrealized_pnl = self.calculate_unrealized_pnl()
            
            # 从交易历史计算已实现盈亏
            realized_pnl = 0.0
            total_commission = 0.0
            for transaction in self.transaction_history:
                if transaction.get('side', '').upper() == 'SELL':
                    # 简化的已实现盈亏计算
                    symbol = transaction['symbol']
                    sell_price = transaction['price']
                    quantity = transaction['quantity']
                    commission = transaction.get('commission', 0.0)
                    
                    # 获取该股票的平均成本（注意：这是简化计算）
                    if symbol in self.positions:
                        avg_cost = self.positions[symbol]['avg_cost']
                        realized_pnl += (sell_price - avg_cost) * quantity - commission
                
                total_commission += transaction.get('commission', 0.0)
            
            # 计算集中度指标
            concentration_risk = self.calculate_concentration_risk()
            
            # 计算权重相关指标
            weights = [pos.get("weight", 0.0) for pos in self.positions.values()]
            max_position_weight = max(weights) if weights else 0.0
            avg_position_weight = sum(weights) / len(weights) if weights else 0.0
            
            # 资金使用效率
            capital_efficiency = total_position_value / self.initial_capital if self.initial_capital > 0 else 0.0
            
            # 基础风险指标
            position_diversity = len(self.positions)  # 持仓多样性（简单的数量指标）
            
            # 交易活动指标
            total_trades = len(self.transaction_history)
            buy_trades = sum(1 for t in self.transaction_history if t.get('side', '').upper() == 'BUY')
            sell_trades = total_trades - buy_trades
            
            # 构造绩效指标字典
            performance_metrics = {
                # 基础收益指标
                'current_value': current_value,
                'initial_capital': self.initial_capital,
                'total_return': total_return,
                'total_return_pct': total_return * 100,
                'absolute_return': current_value - self.initial_capital,
                
                # 现金和投资状况
                'current_cash': self.current_cash,
                'cash_ratio': cash_ratio,
                'cash_ratio_pct': cash_ratio * 100,
                'total_invested': total_invested,
                'investment_ratio': investment_ratio,
                'investment_ratio_pct': investment_ratio * 100,
                'capital_efficiency': capital_efficiency,
                
                # 持仓指标
                'position_count': position_count,
                'total_position_value': total_position_value,
                'max_position_weight': max_position_weight,
                'max_position_weight_pct': max_position_weight * 100,
                'avg_position_weight': avg_position_weight,
                'avg_position_weight_pct': avg_position_weight * 100,
                
                # 盈亏指标
                'unrealized_pnl': total_unrealized_pnl,
                'realized_pnl': realized_pnl,
                'total_pnl': total_unrealized_pnl + realized_pnl,
                'unrealized_return': total_unrealized_pnl / self.initial_capital if self.initial_capital > 0 else 0.0,
                
                # 成本指标
                'total_commission': total_commission,
                'commission_ratio': total_commission / self.initial_capital if self.initial_capital > 0 else 0.0,
                
                # 风险指标（基础）
                'concentration_risk': concentration_risk,
                'herfindahl_index': concentration_risk.get('herfindahl_index', 0.0),
                'top_holdings_pct': concentration_risk.get('top_holdings_percentage', 0.0) * 100,
                'position_diversity': position_diversity,
                
                # 交易活动
                'total_trades': total_trades,
                'buy_trades': buy_trades,
                'sell_trades': sell_trades,
                'trade_balance': buy_trades - sell_trades,
                
                # 元数据
                'calculation_timestamp': datetime.now().isoformat(),
                'note': 'Basic metrics only. Use PerformanceAnalyzer for advanced metrics like Sharpe ratio, drawdown, etc.'
            }
            
            # 添加行业敞口信息
            sector_exposure = self.calculate_sector_exposure()
            if sector_exposure:
                performance_metrics['sector_exposure'] = sector_exposure
                performance_metrics['sector_count'] = len(sector_exposure)
                # 最大行业敞口
                max_sector_exposure = max(sector_exposure.values()) if sector_exposure else 0.0
                performance_metrics['max_sector_exposure'] = max_sector_exposure
                performance_metrics['max_sector_exposure_pct'] = (max_sector_exposure / current_value * 100) if current_value > 0 else 0.0
            
            self.logger.info(f"Calculated basic performance metrics: {total_return:.2%} return, {position_count} positions")
            
            return performance_metrics

    def calculate_diversification_metrics(self, current_prices: Dict[str, float] = None) -> Dict[str, Any]:
        """计算多元化指标
        
        Note: 这个方法提供多元化相关的原始数据和基础指标。
              复杂的多元化分析应该委托给专门的风险管理或分析组件。
        
        Args:
            current_prices: 当前价格字典（可选）
            
        Returns:
            Dict: 多元化指标和原始数据
        """
        with self._lock:
            # 更新投资组合价值
            total_value = self._calculate_portfolio_value_unlocked(current_prices)
            
            if not self.positions or total_value <= 0:
                return {
                    'diversification_score': 0.0,
                    'position_count': 0,
                    'sector_count': 0,
                    'concentration_risk': 'N/A',
                    'note': 'No positions or zero portfolio value'
                }
            
            # 基础多元化指标
            position_count = len(self.positions)
            
            # 获取权重数据
            weights = [pos.get("weight", 0.0) for pos in self.positions.values()]
            position_values = [pos.get("market_value", 0.0) for pos in self.positions.values()]
            
            # 计算集中度指标
            concentration_metrics = self.calculate_concentration_risk()
            herfindahl_index = concentration_metrics.get('herfindahl_index', 0.0)
            
            # 行业多元化分析
            sector_exposure = self.calculate_sector_exposure()
            sector_count = len(sector_exposure)
            sector_weights = {}
            sector_concentration = 0.0
            
            if sector_exposure and total_value > 0:
                for sector, exposure_value in sector_exposure.items():
                    sector_weight = exposure_value / total_value
                    sector_weights[sector] = sector_weight
                    sector_concentration += sector_weight ** 2
            
            # 计算等权重基准的偏差
            equal_weight = 1.0 / position_count if position_count > 0 else 0.0
            weight_deviations = [abs(w - equal_weight) for w in weights]
            avg_weight_deviation = sum(weight_deviations) / len(weight_deviations) if weight_deviations else 0.0
            
            # 有效股票数量（1 / 赫芬达尔指数）
            effective_number_of_stocks = 1.0 / herfindahl_index if herfindahl_index > 0 else 0.0
            
            # 多元化比率（有效股票数 / 实际股票数）
            diversification_ratio = effective_number_of_stocks / position_count if position_count > 0 else 0.0
            
            # 最大和最小仓位权重
            max_position_weight = max(weights) if weights else 0.0
            min_position_weight = min(weights) if weights else 0.0
            
            # 权重分布统计
            if len(weights) > 1:
                weight_std = np.std(weights)
                weight_variance = np.var(weights)
                weight_range = max_position_weight - min_position_weight
            else:
                weight_std = 0.0
                weight_variance = 0.0
                weight_range = 0.0
            
            # 前N大持仓占比
            sorted_weights = sorted(weights, reverse=True)
            top_5_concentration = sum(sorted_weights[:5]) if len(sorted_weights) >= 5 else sum(sorted_weights)
            top_3_concentration = sum(sorted_weights[:3]) if len(sorted_weights) >= 3 else sum(sorted_weights)
            
            # 多元化评分（综合指标，0-100分）
            # 基于有效股票数、行业分散度和权重均匀度
            position_score = min(effective_number_of_stocks / max(position_count, 1) * 100, 100)
            sector_score = min(sector_count / max(position_count, 1) * 100, 100) if sector_count > 0 else 0
            weight_uniformity_score = max(0, 100 - avg_weight_deviation * 1000)  # 权重均匀度评分
            
            diversification_score = (position_score * 0.4 + sector_score * 0.3 + weight_uniformity_score * 0.3)
            
            # 风险集中度分类
            if herfindahl_index < 0.15:
                concentration_level = "Low"
            elif herfindahl_index < 0.25:
                concentration_level = "Moderate"
            else:
                concentration_level = "High"
            
            # 构造多元化指标字典
            diversification_metrics = {
                # 综合评分
                'diversification_score': round(diversification_score, 2),
                'concentration_level': concentration_level,
                
                # 基础指标
                'position_count': position_count,
                'effective_number_of_stocks': round(effective_number_of_stocks, 2),
                'diversification_ratio': round(diversification_ratio, 3),
                
                # 集中度指标
                'herfindahl_index': round(herfindahl_index, 4),
                'top_3_concentration': round(top_3_concentration, 3),
                'top_5_concentration': round(top_5_concentration, 3),
                'max_position_weight': round(max_position_weight, 3),
                'min_position_weight': round(min_position_weight, 3),
                
                # 权重分布
                'weight_std': round(weight_std, 4),
                'weight_variance': round(weight_variance, 4),
                'weight_range': round(weight_range, 3),
                'avg_weight_deviation': round(avg_weight_deviation, 4),
                'equal_weight_benchmark': round(equal_weight, 3),
                
                # 行业多元化
                'sector_count': sector_count,
                'sector_concentration_index': round(sector_concentration, 4),
                'sector_weights': sector_weights,
                'sector_exposure_values': sector_exposure,
                
                # 原始数据（供其他组件使用）
                'position_weights': {symbol: pos.get("weight", 0.0) for symbol, pos in self.positions.items()},
                'position_values': {symbol: pos.get("market_value", 0.0) for symbol, pos in self.positions.items()},
                'total_portfolio_value': total_value,
                
                # 元数据
                'calculation_timestamp': datetime.now().isoformat(),
                'note': 'Basic diversification metrics. Use specialized risk analysis tools for advanced correlation analysis.',
                
                # 建议和警告
                'recommendations': self._get_diversification_recommendations(
                    position_count, herfindahl_index, sector_count, max_position_weight
                )
            }
            
            self.logger.info(
                f"Diversification analysis: {diversification_score:.1f}/100 score, "
                f"{position_count} positions, {sector_count} sectors, "
                f"{concentration_level.lower()} concentration"
            )
            
            return diversification_metrics

    def _get_diversification_recommendations(self, position_count: int, herfindahl_index: float, 
                                           sector_count: int, max_position_weight: float) -> List[str]:
        """生成多元化建议"""
        recommendations = []
        
        if position_count < 10:
            recommendations.append(f"Consider increasing position count from {position_count} to 10+ for better diversification")
        
        if herfindahl_index > 0.25:
            recommendations.append(f"High concentration risk (HHI: {herfindahl_index:.3f}). Consider rebalancing positions")
        
        if max_position_weight > 0.15:
            recommendations.append(f"Largest position ({max_position_weight:.1%}) exceeds recommended 15% limit")
        
        if sector_count < 5 and position_count > 5:
            recommendations.append(f"Consider diversifying across more sectors (currently {sector_count})")
        
        if not recommendations:
            recommendations.append("Portfolio diversification appears adequate")
        
        return recommendations

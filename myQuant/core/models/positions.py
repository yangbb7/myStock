# -*- coding: utf-8 -*-
"""
Positions - 持仓相关数据模型
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class PositionSide(Enum):
    """持仓方向"""

    LONG = "LONG"  # 多头
    SHORT = "SHORT"  # 空头


class Position:
    """持仓基础类"""

    def __init__(self, symbol: str, side: PositionSide = PositionSide.LONG):
        self.symbol = symbol
        self.side = side

        # 数量和成本
        self.quantity = 0
        self.avg_cost = 0.0
        self.total_cost = 0.0

        # 当前价格和市值
        self.current_price = 0.0
        self.market_value = 0.0

        # 盈亏
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.total_pnl = 0.0

        # 时间信息
        self.first_trade_time = None
        self.last_trade_time = None
        self.last_update_time = datetime.now()

        # 统计信息
        self.trade_count = 0
        self.total_commission = 0.0

        # 额外信息
        self.sector = "Unknown"
        self.weight = 0.0
        self.metadata: Dict[str, Any] = {}

    def add_trade(
        self,
        quantity: int,
        price: float,
        commission: float = 0.0,
        trade_time: datetime = None,
    ):
        """添加交易记录"""
        trade_time = trade_time or datetime.now()

        if quantity == 0:
            return

        # 记录第一次交易时间
        if self.first_trade_time is None:
            self.first_trade_time = trade_time

        self.last_trade_time = trade_time
        self.last_update_time = trade_time
        self.trade_count += 1
        self.total_commission += commission

        if quantity > 0:  # 买入
            self._add_long_position(quantity, price, commission)
        else:  # 卖出
            self._reduce_position(-quantity, price, commission)

    def _add_long_position(self, quantity: int, price: float, commission: float):
        """增加多头持仓"""
        if self.quantity == 0:
            # 新开仓
            self.quantity = quantity
            self.avg_cost = price
            self.total_cost = quantity * price + commission
        else:
            # 加仓
            new_total_cost = self.total_cost + quantity * price + commission
            new_quantity = self.quantity + quantity
            self.avg_cost = new_total_cost / new_quantity
            self.quantity = new_quantity
            self.total_cost = new_total_cost

    def _reduce_position(self, quantity: int, price: float, commission: float):
        """减少持仓（卖出）"""
        if quantity > self.quantity:
            raise ValueError(
                f"Cannot sell {quantity} shares, only have {self.quantity}"
            )

        # 计算实现盈亏
        realized_gain = (price - self.avg_cost) * quantity - commission
        self.realized_pnl += realized_gain

        # 更新持仓
        remaining_quantity = self.quantity - quantity
        if remaining_quantity == 0:
            # 全部卖出
            self.quantity = 0
            self.avg_cost = 0.0
            self.total_cost = 0.0
        else:
            # 部分卖出，成本价不变，总成本按比例减少
            self.quantity = remaining_quantity
            self.total_cost = self.avg_cost * remaining_quantity

    def update_price(self, price: float):
        """更新当前价格"""
        self.current_price = price
        self.market_value = self.quantity * price

        if self.quantity > 0:
            self.unrealized_pnl = (price - self.avg_cost) * self.quantity
        else:
            self.unrealized_pnl = 0.0

        self.total_pnl = self.realized_pnl + self.unrealized_pnl
        self.last_update_time = datetime.now()

    def calculate_return(self) -> float:
        """计算收益率"""
        if self.total_cost == 0:
            return 0.0
        return self.total_pnl / self.total_cost

    def calculate_return_rate(self) -> float:
        """计算收益率（百分比）"""
        return self.calculate_return() * 100

    def is_profitable(self) -> bool:
        """是否盈利"""
        return self.total_pnl > 0

    def is_empty(self) -> bool:
        """是否空仓"""
        return self.quantity == 0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "avg_cost": self.avg_cost,
            "total_cost": self.total_cost,
            "current_price": self.current_price,
            "market_value": self.market_value,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "total_pnl": self.total_pnl,
            "return_rate": self.calculate_return_rate(),
            "first_trade_time": (
                self.first_trade_time.isoformat() if self.first_trade_time else None
            ),
            "last_trade_time": (
                self.last_trade_time.isoformat() if self.last_trade_time else None
            ),
            "last_update_time": self.last_update_time.isoformat(),
            "trade_count": self.trade_count,
            "total_commission": self.total_commission,
            "sector": self.sector,
            "weight": self.weight,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Position":
        """从字典创建持仓对象"""
        position = cls(
            symbol=data["symbol"], side=PositionSide(data.get("side", "LONG"))
        )

        # 恢复基本信息
        position.quantity = data.get("quantity", 0)
        position.avg_cost = data.get("avg_cost", 0.0)
        position.total_cost = data.get("total_cost", 0.0)
        position.current_price = data.get("current_price", 0.0)
        position.market_value = data.get("market_value", 0.0)
        position.unrealized_pnl = data.get("unrealized_pnl", 0.0)
        position.realized_pnl = data.get("realized_pnl", 0.0)
        position.total_pnl = data.get("total_pnl", 0.0)

        # 恢复时间信息
        if data.get("first_trade_time"):
            position.first_trade_time = datetime.fromisoformat(data["first_trade_time"])
        if data.get("last_trade_time"):
            position.last_trade_time = datetime.fromisoformat(data["last_trade_time"])
        if data.get("last_update_time"):
            position.last_update_time = datetime.fromisoformat(data["last_update_time"])

        # 恢复统计信息
        position.trade_count = data.get("trade_count", 0)
        position.total_commission = data.get("total_commission", 0.0)
        position.sector = data.get("sector", "Unknown")
        position.weight = data.get("weight", 0.0)
        position.metadata = data.get("metadata", {})

        return position


class Portfolio:
    """投资组合持仓集合"""

    def __init__(self, initial_cash: float = 0.0):
        self.positions: Dict[str, Position] = {}
        self.cash = initial_cash
        self.initial_cash = initial_cash

        # 统计信息
        self.total_value = initial_cash
        self.total_pnl = 0.0
        self.total_commission = 0.0

        # 时间信息
        self.created_time = datetime.now()
        self.last_update_time = datetime.now()

    def add_position(self, position: Position):
        """添加持仓"""
        self.positions[position.symbol] = position
        self._update_totals()

    def get_position(self, symbol: str) -> Optional[Position]:
        """获取持仓"""
        return self.positions.get(symbol)

    def has_position(self, symbol: str) -> bool:
        """检查是否有持仓"""
        position = self.positions.get(symbol)
        return position is not None and not position.is_empty()

    def remove_position(self, symbol: str):
        """移除持仓"""
        if symbol in self.positions:
            del self.positions[symbol]
            self._update_totals()

    def update_prices(self, price_data: Dict[str, float]):
        """批量更新价格"""
        for symbol, price in price_data.items():
            if symbol in self.positions:
                self.positions[symbol].update_price(price)

        self._update_totals()

    def get_total_value(self) -> float:
        """获取总价值"""
        return self.total_value

    def get_position_value(self) -> float:
        """获取持仓总价值"""
        return sum(pos.market_value for pos in self.positions.values())

    def get_cash_ratio(self) -> float:
        """获取现金比例"""
        if self.total_value == 0:
            return 1.0
        return self.cash / self.total_value

    def get_position_count(self) -> int:
        """获取持仓数量"""
        return len([pos for pos in self.positions.values() if not pos.is_empty()])

    def get_profitable_positions(self) -> Dict[str, Position]:
        """获取盈利持仓"""
        return {
            symbol: pos
            for symbol, pos in self.positions.items()
            if pos.is_profitable() and not pos.is_empty()
        }

    def get_losing_positions(self) -> Dict[str, Position]:
        """获取亏损持仓"""
        return {
            symbol: pos
            for symbol, pos in self.positions.items()
            if not pos.is_profitable() and not pos.is_empty()
        }

    def _update_totals(self):
        """更新总计信息"""
        position_value = sum(pos.market_value for pos in self.positions.values())
        self.total_value = self.cash + position_value

        self.total_pnl = sum(pos.total_pnl for pos in self.positions.values())
        self.total_commission = sum(
            pos.total_commission for pos in self.positions.values()
        )

        self.last_update_time = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "positions": {
                symbol: pos.to_dict() for symbol, pos in self.positions.items()
            },
            "cash": self.cash,
            "initial_cash": self.initial_cash,
            "total_value": self.total_value,
            "total_pnl": self.total_pnl,
            "total_commission": self.total_commission,
            "created_time": self.created_time.isoformat(),
            "last_update_time": self.last_update_time.isoformat(),
            "position_count": self.get_position_count(),
            "cash_ratio": self.get_cash_ratio(),
        }

# -*- coding: utf-8 -*-
"""
Positions - 持仓相关数据模型
"""

import re
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, Optional
from dataclasses import dataclass


class PositionSide(Enum):
    """持仓方向"""

    LONG = "LONG"  # 多头
    SHORT = "SHORT"  # 空头


class PositionStatus(str, Enum):
    """持仓状态"""
    ACTIVE = "ACTIVE"      # 活跃
    CLOSED = "CLOSED"      # 已平仓
    SUSPENDED = "SUSPENDED"  # 暂停


class Position:
    """持仓基础类"""

    def __init__(
        self, 
        user_id: int,
        symbol: str, 
        quantity: int,
        average_price: Optional[Decimal] = None,
        avg_cost: Optional[Decimal] = None,  # 兼容参数
        side: PositionSide = PositionSide.LONG,
        status: PositionStatus = PositionStatus.ACTIVE
    ):
        self.user_id = user_id
        self.symbol = symbol
        self.quantity = quantity
        # 支持两种参数名
        self.average_price = average_price or avg_cost or Decimal('0')
        self.avg_cost = self.average_price  # 别名，兼容旧代码
        self.side = side
        self.status = status
        self.realized_pnl = Decimal('0')
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
        
        # 验证数据
        self._validate_position()
    
    def _validate_position(self):
        """持仓验证"""
        # 验证股票代码格式
        if not re.match(r'^[0-9]{6}\.(SZ|SH)$', self.symbol):
            raise ValueError(f'Invalid symbol format: {self.symbol}')
        
        # 验证数量
        if self.quantity < 0:
            raise ValueError('Quantity cannot be negative')
        
        # 验证平均价格
        if self.average_price <= 0:
            raise ValueError('Average price must be positive')
    
    def calculate_market_value(self, current_price: Decimal) -> Decimal:
        """计算市值
        
        Args:
            current_price: 当前价格
            
        Returns:
            Decimal: 市值
        """
        return current_price * self.quantity
    
    def calculate_unrealized_pnl(self, current_price: Decimal) -> Decimal:
        """计算未实现盈亏
        
        Args:
            current_price: 当前价格
            
        Returns:
            Decimal: 未实现盈亏
        """
        return (current_price - self.average_price) * self.quantity
    
    def calculate_percentage(self, current_price: Decimal, total_portfolio_value: Decimal) -> Decimal:
        """计算持仓比例
        
        Args:
            current_price: 当前价格
            total_portfolio_value: 投资组合总价值
            
        Returns:
            Decimal: 持仓比例（百分比）
        """
        if total_portfolio_value == 0:
            return Decimal('0')
        
        market_value = self.calculate_market_value(current_price)
        return (market_value / total_portfolio_value * 100).quantize(Decimal('0.01'))
    
    def get_market_value(self) -> Decimal:
        """获取市值（使用平均价格）
        
        Returns:
            Decimal: 市值
        """
        return self.calculate_market_value(self.average_price)
    
    def update_position(self, quantity_change: int, price: Decimal) -> bool:
        """更新持仓
        
        Args:
            quantity_change: 数量变化（正数为增加，负数为减少）
            price: 交易价格
            
        Returns:
            bool: 更新是否成功
        """
        if quantity_change == 0:
            return True
        
        new_quantity = self.quantity + quantity_change
        
        # 检查是否会导致负数量
        if new_quantity < 0:
            return False
        
        # 如果是增加持仓，更新平均成本
        if quantity_change > 0:
            total_cost = self.average_price * self.quantity + price * quantity_change
            self.quantity = new_quantity
            self.average_price = total_cost / self.quantity
        else:
            # 如果是减少持仓，不更新平均成本，但记录实现盈亏
            realized_pnl_change = (price - self.average_price) * abs(quantity_change)
            self.realized_pnl += realized_pnl_change
            self.quantity = new_quantity
        
        # 如果数量为0，标记为已平仓
        if self.quantity == 0:
            self.status = PositionStatus.CLOSED
        
        self.updated_at = datetime.now(timezone.utc)
        return True
    
    def close(self, close_price: Decimal) -> bool:
        """平仓
        
        Args:
            close_price: 平仓价格
            
        Returns:
            bool: 平仓是否成功
        """
        if self.status != PositionStatus.ACTIVE or self.quantity == 0:
            return False
        
        # 计算实现盈亏
        self.realized_pnl = (close_price - self.average_price) * self.quantity
        
        # 更新状态
        self.quantity = 0
        self.status = PositionStatus.CLOSED
        self.updated_at = datetime.now(timezone.utc)
        
        return True
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'user_id': self.user_id,
            'symbol': self.symbol,
            'quantity': self.quantity,
            'average_price': float(self.average_price),
            'status': self.status.value,
            'realized_pnl': float(self.realized_pnl),
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

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

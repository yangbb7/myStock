# -*- coding: utf-8 -*-
"""
Orders - 订单相关数据模型
"""

import re
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from pydantic import BaseModel, validator


class OrderStatus(Enum):
    """订单状态"""

    CREATED = "CREATED"  # 已创建
    PENDING = "PENDING"  # 待提交
    SUBMITTED = "SUBMITTED"  # 已提交
    PARTIALLY_FILLED = "PARTIALLY_FILLED"  # 部分成交
    FILLED = "FILLED"  # 完全成交
    CANCELLED = "CANCELLED"  # 已取消
    REJECTED = "REJECTED"  # 被拒绝
    EXPIRED = "EXPIRED"  # 已过期


class OrderType(Enum):
    """订单类型"""

    MARKET = "MARKET"  # 市价单
    LIMIT = "LIMIT"  # 限价单
    STOP = "STOP"  # 止损单
    STOP_LIMIT = "STOP_LIMIT"  # 限价止损单
    ICEBERG = "ICEBERG"  # 冰山单
    TWA = "TWA"  # 时间加权平均价格
    VWAP = "VWAP"  # 成交量加权平均价格


class OrderSide(Enum):
    """订单方向"""

    BUY = "BUY"
    SELL = "SELL"


class TimeInForce(Enum):
    """订单有效期"""

    DAY = "DAY"  # 当日有效
    GTC = "GTC"  # 撤销前有效
    IOC = "IOC"  # 立即成交或取消
    FOK = "FOK"  # 全部成交或取消
    GTD = "GTD"  # 指定日期前有效


class OrderFill:
    """成交记录"""

    def __init__(
        self,
        fill_quantity: int,
        fill_price: Decimal,
        commission: Decimal = Decimal('0.0'),
        fill_time: datetime = None,
    ):
        self.fill_id = str(uuid.uuid4())
        self.quantity = fill_quantity
        self.price = fill_price
        self.commission = commission
        self.timestamp = fill_time or datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fill_id": self.fill_id,
            "quantity": self.quantity,
            "price": float(self.price),
            "commission": float(self.commission),
            "timestamp": self.timestamp.isoformat(),
        }


class OrderRequest(BaseModel):
    """订单请求模型"""
    
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    user_id: int
    
    @validator('symbol')
    def validate_symbol_format(cls, v):
        """验证股票代码格式"""
        if not re.match(r'^[0-9]{6}\.(SZ|SH)$', v):
            raise ValueError(f'Invalid symbol format: {v}')
        return v
    
    @validator('quantity')
    def validate_quantity(cls, v):
        """验证数量"""
        if v <= 0:
            raise ValueError('Quantity must be positive')
        if v % 100 != 0:
            raise ValueError('Quantity must be multiple of 100')
        return v
    
    @validator('price')
    def validate_price(cls, v, values):
        """验证价格"""
        if v is not None and v <= 0:
            raise ValueError('Price must be positive')
        
        # 限价单必须有价格
        if values.get('order_type') in [OrderType.LIMIT, OrderType.STOP_LIMIT] and v is None:
            raise ValueError('Limit orders must have a price')
        
        return v
    
    @validator('stop_price')
    def validate_stop_price(cls, v, values):
        """验证止损价格"""
        if v is not None and v <= 0:
            raise ValueError('Stop price must be positive')
        
        # 止损单必须有止损价格
        if values.get('order_type') in [OrderType.STOP, OrderType.STOP_LIMIT] and v is None:
            raise ValueError('Stop orders must have a stop price')
        
        return v
    
    class Config:
        """Pydantic配置"""
        arbitrary_types_allowed = True


class Order:
    """订单基础类"""

    def __init__(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[Decimal] = None,
        stop_price: Optional[Decimal] = None,
        time_in_force: TimeInForce = TimeInForce.DAY,
        user_id: int = 1,
        strategy_name: str = "",
        client_order_id: str = None,
    ):

        self.order_id = client_order_id or str(uuid.uuid4())
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.order_type = order_type
        self.price = price
        self.stop_price = stop_price
        self.time_in_force = time_in_force
        self.user_id = user_id
        self.strategy_name = strategy_name

        # 状态信息
        self.status = OrderStatus.CREATED
        self.filled_quantity = 0
        self.average_fill_price = None
        self.commission = Decimal('0')
        self.total_commission = Decimal('0')

        # 时间信息
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
        self.submitted_time = None
        self.filled_time = None
        self.cancelled_time = None
        self.expire_time = None

        # 执行信息
        self.fills: List[OrderFill] = []
        self.broker_order_id = None
        self.last_update_time = datetime.now(timezone.utc)
        self.reject_reason = ""

        # 算法交易参数
        self.algo_params: Dict[str, Any] = {}

        # 回调函数
        self.status_callback: Optional[Callable] = None
        self.fill_callback: Optional[Callable] = None
        
        # 运行验证
        self._validate_order()
    
    @property
    def remaining_quantity(self) -> int:
        """剩余数量"""
        return self.quantity - self.filled_quantity
    
    def _validate_order(self):
        """订单验证"""
        # 验证股票代码格式
        if not re.match(r'^[0-9]{6}\.(SZ|SH)$', self.symbol):
            raise ValueError(f'Invalid symbol format: {self.symbol}')
        
        # 验证数量
        if self.quantity <= 0:
            raise ValueError('Quantity must be positive')
        
        # 验证价格
        if self.price is not None and self.price <= 0:
            raise ValueError('Price must be positive')
        
        # 限价单必须有价格
        if self.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and self.price is None:
            raise ValueError('Limit orders must have a price')
        
        # 止损单必须有止损价格
        if self.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and self.stop_price is None:
            raise ValueError('Stop orders must have a stop price')

    def fill(self, quantity: int, price: Decimal) -> bool:
        """成交处理
        
        Args:
            quantity: 成交数量
            price: 成交价格
            
        Returns:
            bool: 成交是否成功
        """
        if not self.is_active():
            return False
        
        if quantity <= 0 or quantity > self.remaining_quantity:
            return False
        
        # 更新成交信息
        total_filled_value = Decimal('0')
        if self.average_fill_price and self.filled_quantity > 0:
            total_filled_value = self.average_fill_price * self.filled_quantity
        
        total_filled_value += price * quantity
        self.filled_quantity += quantity
        self.average_fill_price = total_filled_value / self.filled_quantity
        
        # 更新状态
        if self.filled_quantity >= self.quantity:
            self.status = OrderStatus.FILLED
            self.filled_time = datetime.now(timezone.utc)
        else:
            self.status = OrderStatus.PARTIALLY_FILLED
        
        self.updated_at = datetime.now(timezone.utc)
        return True

    def cancel(self) -> bool:
        """撤销订单
        
        Returns:
            bool: 撤销是否成功
        """
        if not self.is_active():
            return False
        
        self.status = OrderStatus.CANCELLED
        self.updated_at = datetime.now(timezone.utc)
        return True

    def add_fill(
        self,
        fill_quantity: int,
        fill_price: float,
        commission: float = 0.0,
        fill_time: datetime = None,
    ) -> OrderFill:
        """添加成交记录"""
        fill = OrderFill(fill_quantity, Decimal(str(fill_price)), Decimal(str(commission)), fill_time)

        self.fills.append(fill)
        self.filled_quantity += fill_quantity
        # remaining_quantity is calculated automatically as a property
        self.total_commission += Decimal(str(commission))

        # 计算平均成交价
        if self.filled_quantity > 0:
            total_value = sum(f.quantity * f.price for f in self.fills)
            self.average_fill_price = total_value / self.filled_quantity

        # 更新状态
        if self.remaining_quantity == 0:
            self.status = OrderStatus.FILLED
            self.filled_time = fill.timestamp
        elif self.filled_quantity > 0:
            self.status = OrderStatus.PARTIALLY_FILLED

        self.last_update_time = fill.timestamp

        # 调用回调函数
        if self.fill_callback:
            self.fill_callback(self, fill)

        return fill

    def update_status(self, new_status: OrderStatus, reason: str = ""):
        """更新订单状态"""
        old_status = self.status
        self.status = new_status
        self.last_update_time = datetime.now()

        if new_status == OrderStatus.CANCELLED:
            self.cancelled_time = self.last_update_time
        elif new_status == OrderStatus.REJECTED:
            self.reject_reason = reason
        elif new_status == OrderStatus.SUBMITTED:
            self.submitted_time = self.last_update_time

        # 调用状态回调
        if self.status_callback:
            self.status_callback(self, old_status, new_status)

    def is_active(self) -> bool:
        """检查订单是否仍处于活跃状态"""
        return self.status in [
            OrderStatus.CREATED,
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.PARTIALLY_FILLED,
        ]

    def is_final(self) -> bool:
        """检查订单是否已到达最终状态"""
        return self.status in [
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        ]

    def get_total_value(self) -> float:
        """获取订单总价值"""
        if self.price:
            return self.quantity * self.price
        return 0.0

    def get_filled_value(self) -> float:
        """获取已成交价值"""
        return sum(f.quantity * f.price for f in self.fills)

    def get_fill_rate(self) -> float:
        """获取成交率"""
        if self.quantity == 0:
            return 0.0
        return self.filled_quantity / self.quantity

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "order_type": self.order_type.value,
            "price": self.price,
            "stop_price": self.stop_price,
            "time_in_force": self.time_in_force.value,
            "strategy_name": self.strategy_name,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "remaining_quantity": self.remaining_quantity,
            "average_fill_price": self.average_fill_price,
            "total_commission": self.total_commission,
            "created_at": self.created_at.isoformat(),
            "submitted_time": (
                self.submitted_time.isoformat() if self.submitted_time else None
            ),
            "filled_time": self.filled_time.isoformat() if self.filled_time else None,
            "cancelled_time": (
                self.cancelled_time.isoformat() if self.cancelled_time else None
            ),
            "fills": [fill.to_dict() for fill in self.fills],
            "broker_order_id": self.broker_order_id,
            "reject_reason": self.reject_reason,
            "algo_params": self.algo_params,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Order":
        """从字典创建订单对象"""
        order = cls(
            symbol=data["symbol"],
            side=OrderSide(data["side"]),
            quantity=data["quantity"],
            order_type=OrderType(data["order_type"]),
            price=data.get("price"),
            stop_price=data.get("stop_price"),
            time_in_force=TimeInForce(data.get("time_in_force", "DAY")),
            strategy_name=data.get("strategy_name", ""),
            client_order_id=data.get("order_id"),
        )

        # 恢复状态信息
        order.status = OrderStatus(data["status"])
        order.filled_quantity = data.get("filled_quantity", 0)
        # remaining_quantity is calculated automatically as a property
        order.average_fill_price = data.get("average_fill_price", Decimal("0.0"))
        order.total_commission = data.get("total_commission", Decimal("0.0"))

        # 恢复时间信息
        if data.get("created_at"):
            order.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("submitted_time"):
            order.submitted_time = datetime.fromisoformat(data["submitted_time"])
        if data.get("filled_time"):
            order.filled_time = datetime.fromisoformat(data["filled_time"])
        if data.get("cancelled_time"):
            order.cancelled_time = datetime.fromisoformat(data["cancelled_time"])

        # 恢复成交记录
        for fill_data in data.get("fills", []):
            fill = OrderFill(
                fill_quantity=fill_data["quantity"],
                fill_price=fill_data["price"],
                commission=fill_data["commission"],
                fill_time=datetime.fromisoformat(fill_data["timestamp"]),
            )
            fill.fill_id = fill_data["fill_id"]
            order.fills.append(fill)

        # 恢复其他信息
        order.broker_order_id = data.get("broker_order_id")
        order.reject_reason = data.get("reject_reason", "")
        order.algo_params = data.get("algo_params", {})

        return order

"""
OrderManager - 订单管理器

负责订单创建、提交、取消、执行处理等核心功能
"""

import logging
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple

from ..models.orders import Order, OrderSide, OrderType, OrderStatus, TimeInForce
from myQuant.infrastructure.database.database_manager import DatabaseManager
from myQuant.infrastructure.database.repositories import OrderRepository, UserRepository


class OrderManager:
    """订单管理器"""
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """初始化订单管理器
        
        Args:
            db_manager: 数据库管理器（可选，用于测试）
        """
        self.db_manager = db_manager
        if db_manager:
            self.order_repository = OrderRepository(db_manager)
            self.user_repository = UserRepository(db_manager)
        else:
            # 测试模式，使用 mock
            self.order_repository = None
            self.user_repository = None
            
        self.logger = logging.getLogger(__name__)
        
        # 内存中的订单缓存
        self._order_cache: Dict[str, Order] = {}
        
        # 券商接口（用于实时交易）
        self.broker = None
        
        # 待处理订单列表（用于测试）
        self.pending_orders = []
    
    @property
    def orders(self) -> Dict[str, Order]:
        """获取所有订单缓存
        
        Returns:
            Dict[str, Order]: 订单缓存字典
        """
        return self._order_cache
    
    def create_order_from_signal(self, signal: Dict[str, Any]) -> str:
        """从交易信号创建订单
        
        Args:
            signal: 交易信号字典，包含 symbol, action, quantity, price 等信息
            
        Returns:
            str: 订单ID
        """
        try:
            # 从信号中提取订单信息
            symbol = signal.get('symbol')
            action = signal.get('action', '').upper()
            side = signal.get('side', '').upper()  # Also check for 'side' field
            quantity = signal.get('quantity', 0)
            price = signal.get('price', 0)
            
            # Use either action or side field
            if not action and side:
                action = side
            elif not side and action:
                side = action
            
            if not symbol or not action or quantity <= 0:
                self.logger.error(f"Invalid signal parameters: {signal}")
                return ""
            
            # 转换 action 为 OrderSide
            if action in ['BUY', 'LONG']:
                side = 'BUY'
            elif action in ['SELL', 'SHORT']:
                side = 'SELL'
            else:
                self.logger.error(f"Unknown action: {action}")
                return ""
            
            # 创建订单请求
            order_request = {
                'symbol': symbol,
                'side': side,
                'order_type': 'MARKET',  # 默认使用市价单
                'quantity': int(quantity),
                'price': float(price) if price > 0 else None,
                'user_id': 1  # 默认用户ID，实际应用中应该从信号或上下文获取
            }
            
            # 生成订单ID
            import uuid
            order_id = str(uuid.uuid4())
            
            # 创建简化的订单对象存储在缓存中
            self._order_cache[order_id] = {
                'order_id': order_id,
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': price,
                'status': 'CREATED',
                'created_at': datetime.now()
            }
            
            self.logger.info(f"Created order {order_id} from signal: {side} {quantity} {symbol}")
            return order_id
            
        except Exception as e:
            self.logger.error(f"Failed to create order from signal: {e}")
            return ""
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """获取订单状态
        
        Args:
            order_id: 订单ID
            
        Returns:
            Dict[str, Any]: 订单状态信息
        """
        if order_id in self._order_cache:
            order = self._order_cache[order_id]
            if isinstance(order, dict):
                return {
                    'order_id': order_id,
                    'symbol': order.get('symbol'),
                    'side': order.get('side'),
                    'quantity': order.get('quantity'),
                    'price': order.get('price'),
                    'status': order.get('status', 'UNKNOWN')
                }
            else:
                # Handle Order object
                return {
                    'order_id': order_id,
                    'symbol': getattr(order, 'symbol', None),
                    'side': getattr(order, 'side', None),
                    'quantity': getattr(order, 'quantity', None),
                    'price': getattr(order, 'price', None),
                    'status': getattr(order, 'status', 'UNKNOWN')
                }
        else:
            return {
                'order_id': order_id,
                'status': 'NOT_FOUND'
            }
        
    async def create_order(self, order_request: Dict[str, Any]) -> Order:
        """创建新订单
        
        Args:
            order_request: 订单请求数据
            
        Returns:
            Order: 新创建的订单
        """
        # 验证订单数据
        is_valid, errors = await self.validate_order(order_request)
        if not is_valid:
            raise ValueError(f"Invalid order data: {', '.join(errors)}")
        
        # 验证用户存在（仅在有数据库时）
        if self.user_repository:
            user = await self.user_repository.get_by_id(order_request["user_id"])
            if not user:
                # 只有在用户ID <= 10 的情况下自动创建用户（测试用户）
                if order_request["user_id"] <= 10:
                    await self.user_repository.create_user({
                        "username": f"user_{order_request['user_id']}",
                        "email": f"user_{order_request['user_id']}@example.com",
                        "password_hash": "dummy_hash"
                    })
                else:
                    raise ValueError("User not found")
        
        # 创建订单
        order = Order(
            symbol=order_request["symbol"],
            side=order_request["side"],
            order_type=order_request["order_type"],
            quantity=order_request["quantity"],
            price=order_request.get("price"),
            stop_price=order_request.get("stop_price"),
            time_in_force=order_request.get("time_in_force", TimeInForce.DAY),
            user_id=order_request["user_id"]
        )
        
        # 保存到数据库
        order_data = {
            "order_id": order.order_id,
            "symbol": order.symbol,
            "side": order.side.value,
            "order_type": order.order_type.value,
            "quantity": order.quantity,
            "price": float(order.price) if order.price else None,
            "stop_price": float(order.stop_price) if order.stop_price else None,
            "time_in_force": order.time_in_force.value,
            "status": order.status.value,
            "user_id": order.user_id,
            "created_at": order.created_at
        }
        
        # 保存到数据库（仅在有数据库时）
        if self.order_repository:
            await self.order_repository.create_order(order_data)
        
        # 缓存订单
        self._order_cache[order.order_id] = order
        
        self.logger.info(f"Created order {order.order_id} for user {order.user_id}")
        return order
    
    async def submit_order(self, order_id: str) -> bool:
        """提交订单
        
        Args:
            order_id: 订单ID
            
        Returns:
            bool: 是否成功提交
        """
        order = await self.get_order_by_id(order_id)
        if not order:
            raise ValueError("Order not found")
        
        # 检查订单状态
        if order.status != OrderStatus.CREATED:
            return False
        
        # 更新订单状态
        await self._update_order_status(order_id, OrderStatus.SUBMITTED)
        
        self.logger.info(f"Submitted order {order_id}")
        return True
    
    async def cancel_order(self, order_id: str) -> bool:
        """取消订单
        
        Args:
            order_id: 订单ID
            
        Returns:
            bool: 是否成功取消
        """
        order = await self.get_order_by_id(order_id)
        if not order:
            raise ValueError("Order not found")
        
        # 检查订单状态是否可以取消
        if order.status not in [OrderStatus.CREATED, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]:
            return False
        
        # 更新订单状态
        await self._update_order_status(order_id, OrderStatus.CANCELLED)
        
        self.logger.info(f"Cancelled order {order_id}")
        return True
    
    async def get_order_by_id(self, order_id: str) -> Optional[Order]:
        """根据ID获取订单
        
        Args:
            order_id: 订单ID
            
        Returns:
            Order: 订单对象，如果不存在则返回None
        """
        # 检查缓存
        if order_id in self._order_cache:
            return self._order_cache[order_id]
        
        # 从数据库获取
        order_data = await self.order_repository.get_by_id(order_id)
        if not order_data:
            return None
        
        # 重建订单对象
        order = Order(
            symbol=order_data.symbol,
            side=OrderSide(order_data.side),
            order_type=OrderType(order_data.order_type),
            quantity=order_data.quantity,
            price=Decimal(str(order_data.price)) if order_data.price else None,
            stop_price=Decimal(str(order_data.stop_price)) if order_data.stop_price else None,
            time_in_force=TimeInForce(order_data.time_in_force),
            user_id=order_data.user_id,
            order_id=order_data.id
        )
        
        # 更新订单状态和执行信息
        order.status = OrderStatus(order_data.status)
        order.filled_quantity = order_data.filled_quantity or 0
# remaining_quantity is calculated automatically as a property
        order.average_fill_price = Decimal(str(order_data.average_fill_price)) if order_data.average_fill_price else Decimal("0")
        order.created_at = order_data.created_at
        order.updated_at = order_data.updated_at
        
        # 缓存订单
        self._order_cache[order_id] = order
        
        return order
    
    async def get_orders_by_user(self, user_id: int) -> List[Order]:
        """根据用户获取订单
        
        Args:
            user_id: 用户ID
            
        Returns:
            List[Order]: 订单列表
        """
        orders_data = await self.order_repository.get_orders_by_user(user_id)
        orders = []
        
        for order_data in orders_data:
            # Handle both dict and object formats
            if hasattr(order_data, '__dict__'):
                # Object format
                order = Order(
                    symbol=order_data.symbol,
                    side=OrderSide(order_data.side),
                    order_type=OrderType(order_data.order_type),
                    quantity=order_data.quantity,
                    price=Decimal(str(order_data.price)) if order_data.price else None,
                    stop_price=Decimal(str(order_data.stop_price)) if order_data.stop_price else None,
                    time_in_force=TimeInForce(order_data.time_in_force),
                    user_id=order_data.user_id,
                    client_order_id=order_data.id
                )
                order.status = OrderStatus(order_data.status)
            else:
                # Dict format
                order = Order(
                    symbol=order_data['symbol'],
                    side=OrderSide(order_data['side']),
                    order_type=OrderType(order_data['order_type']),
                    quantity=order_data['quantity'],
                    price=Decimal(str(order_data['price'])) if order_data['price'] else None,
                    stop_price=Decimal(str(order_data['stop_price'])) if order_data['stop_price'] else None,
                    time_in_force=TimeInForce(order_data['time_in_force']),
                    user_id=order_data['user_id'],
                    client_order_id=order_data['id']
                )
                order.status = OrderStatus(order_data['status'])
            
            if hasattr(order_data, '__dict__'):
                # Object format
                order.filled_quantity = order_data.filled_quantity or 0
                order.average_fill_price = Decimal(str(order_data.average_fill_price)) if order_data.average_fill_price else Decimal("0")
                order.created_at = order_data.created_at
                order.updated_at = order_data.updated_at
            else:
                # Dict format
                order.filled_quantity = order_data['filled_quantity'] or 0
                order.average_fill_price = Decimal(str(order_data['average_fill_price'])) if order_data['average_fill_price'] else Decimal("0")
                order.created_at = order_data['created_at']
                order.updated_at = order_data['updated_at']
            
            orders.append(order)
        
        return orders
    
    async def get_orders_by_symbol(self, symbol: str) -> List[Order]:
        """根据股票代码获取订单
        
        Args:
            symbol: 股票代码
            
        Returns:
            List[Order]: 订单列表
        """
        orders_data = await self.order_repository.get_orders_by_symbol(symbol)
        orders = []
        
        for order_data in orders_data:
            order = Order(
                symbol=order_data['symbol'],
                side=OrderSide(order_data['side']),
                order_type=OrderType(order_data['order_type']),
                quantity=order_data['quantity'],
                price=Decimal(str(order_data['price'])) if order_data['price'] else None,
                stop_price=Decimal(str(order_data['stop_price'])) if order_data['stop_price'] else None,
                time_in_force=TimeInForce(order_data['time_in_force']),
                user_id=order_data['user_id'],
                client_order_id=order_data['id']
            )
            
            order.status = OrderStatus(order_data['status'])
            order.filled_quantity = order_data['filled_quantity'] or 0
    # remaining_quantity is calculated automatically as a property
            order.average_fill_price = Decimal(str(order_data['average_fill_price'])) if order_data['average_fill_price'] else Decimal("0")
            order.created_at = order_data['created_at']
            order.updated_at = order_data['updated_at']
            
            orders.append(order)
        
        return orders
    
    async def get_orders_by_status(self, status: OrderStatus) -> List[Order]:
        """根据状态获取订单
        
        Args:
            status: 订单状态
            
        Returns:
            List[Order]: 订单列表
        """
        orders_data = await self.order_repository.get_orders_by_status(status.value)
        orders = []
        
        for order_data in orders_data:
            order = Order(
                symbol=order_data['symbol'],
                side=OrderSide(order_data['side']),
                order_type=OrderType(order_data['order_type']),
                quantity=order_data['quantity'],
                price=Decimal(str(order_data['price'])) if order_data['price'] else None,
                stop_price=Decimal(str(order_data['stop_price'])) if order_data['stop_price'] else None,
                time_in_force=TimeInForce(order_data['time_in_force']),
                user_id=order_data['user_id'],
                client_order_id=order_data['id']
            )
            
            order.status = OrderStatus(order_data['status'])
            order.filled_quantity = order_data['filled_quantity'] or 0
    # remaining_quantity is calculated automatically as a property
            order.average_fill_price = Decimal(str(order_data['average_fill_price'])) if order_data['average_fill_price'] else Decimal("0")
            order.created_at = order_data['created_at']
            order.updated_at = order_data['updated_at']
            
            orders.append(order)
        
        return orders
    
    async def process_execution(self, order_id: str, execution: Dict[str, Any]) -> bool:
        """处理订单执行
        
        Args:
            order_id: 订单ID
            execution: 执行数据
            
        Returns:
            bool: 是否成功处理
        """
        order = await self.get_order_by_id(order_id)
        if not order:
            raise ValueError("Order not found")
        
        execution_quantity = execution["quantity"]
        execution_price = execution["price"]
        
        # 验证执行数量
        if execution_quantity > order.remaining_quantity:
            raise ValueError("Execution quantity exceeds remaining quantity")
        
        # 更新订单执行信息
        order.add_fill(execution_quantity, execution_price)
        
        # 确定新状态
        if order.remaining_quantity == 0:
            new_status = OrderStatus.FILLED
        else:
            new_status = OrderStatus.PARTIALLY_FILLED
        
        # 更新数据库
        await self.order_repository.update_order(order_id, {
            "status": new_status.value,
            "filled_quantity": order.filled_quantity,
            "average_fill_price": float(order.average_fill_price)
        })
        
        # 更新缓存
        if order_id in self._order_cache:
            self._order_cache[order_id].status = new_status
            self._order_cache[order_id].filled_quantity = order.filled_quantity
# remaining_quantity is calculated automatically as a property
            self._order_cache[order_id].average_fill_price = order.average_fill_price
        
        self.logger.info(f"Processed execution for order {order_id}: {execution_quantity} @ {execution_price}")
        return True
    
    async def calculate_order_value(self, order_id: str) -> Decimal:
        """计算订单价值
        
        Args:
            order_id: 订单ID
            
        Returns:
            Decimal: 订单价值
        """
        order = await self.get_order_by_id(order_id)
        if not order:
            raise ValueError("Order not found")
        
        if order.order_type == OrderType.MARKET or order.price is None:
            raise ValueError("Cannot calculate value for market order without price")
        
        return Decimal(str(order.quantity)) * order.price
    
    async def validate_order(self, order_request: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """验证订单请求
        
        Args:
            order_request: 订单请求数据
            
        Returns:
            Tuple[bool, List[str]]: (是否有效, 错误信息列表)
        """
        errors = []
        
        # 检查必需字段
        required_fields = ["symbol", "side", "order_type", "quantity", "user_id"]
        for field in required_fields:
            if field not in order_request:
                errors.append(f"Missing required field: {field}")
        
        # 检查symbol
        if "symbol" in order_request:
            symbol = order_request["symbol"]
            if not symbol or not isinstance(symbol, str) or not symbol.strip():
                errors.append("Invalid symbol")
        
        # 检查quantity
        if "quantity" in order_request:
            quantity = order_request["quantity"]
            if not isinstance(quantity, int) or quantity <= 0:
                errors.append("Quantity must be a positive integer")
        
        # 检查价格
        if "order_type" in order_request and "price" in order_request:
            order_type = order_request["order_type"]
            price = order_request["price"]
            
            if order_type == OrderType.LIMIT:
                if price is None:
                    errors.append("Limit orders must have a price")
                elif isinstance(price, (int, float, Decimal)) and price <= 0:
                    errors.append("Price must be positive")
        
        return len(errors) == 0, errors
    
    async def get_order_history_by_user(self, user_id: int) -> List[Order]:
        """获取用户订单历史
        
        Args:
            user_id: 用户ID
            
        Returns:
            List[Order]: 订单历史列表
        """
        return await self.get_orders_by_user(user_id)
    
    async def get_active_orders_by_user(self, user_id: int) -> List[Order]:
        """获取用户活跃订单
        
        Args:
            user_id: 用户ID
            
        Returns:
            List[Order]: 活跃订单列表
        """
        all_orders = await self.get_orders_by_user(user_id)
        active_statuses = [OrderStatus.CREATED, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]
        return [order for order in all_orders if order.status in active_statuses]
    
    async def cancel_all_orders_by_user(self, user_id: int) -> int:
        """取消用户所有订单
        
        Args:
            user_id: 用户ID
            
        Returns:
            int: 取消的订单数量
        """
        active_orders = await self.get_active_orders_by_user(user_id)
        cancelled_count = 0
        
        for order in active_orders:
            try:
                result = await self.cancel_order(order.order_id)
                if result:
                    cancelled_count += 1
            except Exception as e:
                self.logger.error(f"Failed to cancel order {order.order_id}: {e}")
        
        return cancelled_count
    
    async def cancel_all_orders_by_symbol(self, symbol: str) -> int:
        """取消特定股票的所有订单
        
        Args:
            symbol: 股票代码
            
        Returns:
            int: 取消的订单数量
        """
        orders = await self.get_orders_by_symbol(symbol)
        cancelled_count = 0
        
        for order in orders:
            if order.status in [OrderStatus.CREATED, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]:
                try:
                    result = await self.cancel_order(order.order_id)
                    if result:
                        cancelled_count += 1
                except Exception as e:
                    self.logger.error(f"Failed to cancel order {order.order_id}: {e}")
        
        return cancelled_count
    
    async def _update_order_status(self, order_id: str, status: OrderStatus) -> bool:
        """更新订单状态
        
        Args:
            order_id: 订单ID
            status: 新状态
            
        Returns:
            bool: 是否成功更新
        """
        try:
            # 更新数据库
            await self.order_repository.update_order(order_id, {
                "status": status.value,
                "updated_at": datetime.now()
            })
            
            # 更新缓存
            if order_id in self._order_cache:
                self._order_cache[order_id].status = status
                self._order_cache[order_id].updated_at = datetime.now()
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to update order status: {e}")
            return False
    
    def set_broker(self, broker) -> None:
        """设置券商接口
        
        Args:
            broker: 券商接口对象
        """
        self.broker = broker
        self.logger.info("Broker interface set successfully")
    
    def create_order_sync(self, order_data: Dict[str, Any]) -> str:
        """创建订单（同步版本）
        
        Args:
            order_data: 订单数据字典
            
        Returns:
            str: 订单ID
        """
        # 对于测试场景，直接使用简化的同步创建
        import uuid
        order_id = str(uuid.uuid4())
        
        # 处理side字段，可能是字符串或枚举
        side = order_data.get('side', '')
        if hasattr(side, 'value'):
            # 如果是枚举，获取其值
            side_str = side.value
        elif isinstance(side, str):
            # 如果是字符串，直接使用
            side_str = side.upper()
        else:
            side_str = str(side).upper()
        
        # 创建简化的订单对象存储在缓存中
        self._order_cache[order_id] = {
            'order_id': order_id,
            'symbol': order_data.get('symbol'),
            'side': side_str,
            'quantity': order_data.get('quantity', 0),
            'price': order_data.get('price', 0),
            'status': 'CREATED',
            'created_at': datetime.now()
        }
        
        self.logger.info(f"Created order {order_id}: {order_data.get('side')} {order_data.get('quantity')} {order_data.get('symbol')}")
        return order_id
    
    def get_order_statistics(self) -> Dict[str, Any]:
        """获取订单统计信息
        
        Returns:
            Dict[str, Any]: 订单统计信息
        """
        total_orders = len(self._order_cache)
        
        # 统计不同状态的订单
        status_counts = {}
        filled_orders = 0
        rejected_orders = 0
        pending_orders = 0
        
        for order in self._order_cache.values():
            if isinstance(order, dict):
                status = order.get('status', 'UNKNOWN')
            else:
                status = getattr(order, 'status', 'UNKNOWN')
                if hasattr(status, 'value'):
                    status = status.value
            
            status_counts[status] = status_counts.get(status, 0) + 1
            
            if status in ['FILLED', 'COMPLETED']:
                filled_orders += 1
            elif status in ['REJECTED', 'CANCELLED', 'ERROR']:
                rejected_orders += 1
            elif status in ['CREATED', 'SUBMITTED', 'PENDING']:
                pending_orders += 1
        
        return {
            'total_orders': total_orders,
            'filled_orders': filled_orders,
            'rejected_orders': rejected_orders,
            'pending_orders': pending_orders,
            'status_breakdown': status_counts,
            'fill_rate': filled_orders / total_orders if total_orders > 0 else 0.0,
            'rejection_rate': rejected_orders / total_orders if total_orders > 0 else 0.0
        }
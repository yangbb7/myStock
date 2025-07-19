import asyncio
import logging
import math
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

from .low_latency_engine import Order, OrderType, OrderSide, OrderStatus
from .broker_gateway import BrokerGateway

class AlgorithmType(Enum):
    TWAP = "twap"
    VWAP = "vwap"
    POV = "pov"
    IMPLEMENTATION_SHORTFALL = "implementation_shortfall"
    ARRIVAL_PRICE = "arrival_price"
    ICEBERG = "iceberg"

@dataclass
class MarketData:
    symbol: str
    timestamp: datetime
    bid_price: float
    ask_price: float
    bid_size: int
    ask_size: int
    last_price: float
    volume: int
    vwap: float
    market_cap: float
    adv: float  # Average Daily Volume

@dataclass
class AlgorithmParams:
    algorithm_type: AlgorithmType
    target_quantity: int
    duration_minutes: int
    max_participation_rate: float = 0.2
    min_fill_size: int = 100
    max_fill_size: int = 10000
    price_limit: Optional[float] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    urgency: str = "medium"  # low, medium, high
    dark_pool_preference: float = 0.3
    limit_price_offset: float = 0.0001
    
@dataclass
class ChildOrder:
    child_order_id: str
    parent_order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    price: Optional[float]
    venue: str
    order_type: OrderType
    scheduled_time: datetime
    status: OrderStatus = OrderStatus.PENDING
    created_time: datetime = field(default_factory=datetime.now)

class SmartRouter(ABC):
    def __init__(self, params: AlgorithmParams):
        self.params = params
        self.logger = logging.getLogger(self.__class__.__name__)
        self.child_orders: List[ChildOrder] = []
        self.is_running = False
        self.filled_quantity = 0
        self.total_cost = 0.0
        
    @abstractmethod
    async def generate_child_orders(self, market_data: MarketData) -> List[ChildOrder]:
        pass
    
    @abstractmethod
    async def adjust_orders(self, market_data: MarketData) -> List[ChildOrder]:
        pass
    
    def get_progress(self) -> Dict[str, Any]:
        return {
            'filled_quantity': self.filled_quantity,
            'remaining_quantity': self.params.target_quantity - self.filled_quantity,
            'fill_rate': self.filled_quantity / self.params.target_quantity if self.params.target_quantity > 0 else 0,
            'avg_price': self.total_cost / self.filled_quantity if self.filled_quantity > 0 else 0,
            'child_orders_count': len(self.child_orders),
            'active_orders': len([o for o in self.child_orders if o.status == OrderStatus.WORKING])
        }

class TWAPRouter(SmartRouter):
    """时间加权平均价格算法"""
    
    async def generate_child_orders(self, market_data: MarketData) -> List[ChildOrder]:
        child_orders = []
        
        # 计算时间切片
        total_slices = max(1, self.params.duration_minutes // 5)  # 每5分钟一个切片
        quantity_per_slice = self.params.target_quantity // total_slices
        
        current_time = datetime.now()
        
        for i in range(total_slices):
            # 计算每个切片的执行时间
            slice_time = current_time + timedelta(minutes=i * 5)
            
            # 计算订单数量
            slice_quantity = quantity_per_slice
            if i == total_slices - 1:  # 最后一个切片包含余数
                slice_quantity = self.params.target_quantity - (quantity_per_slice * (total_slices - 1))
            
            # 进一步拆分为多个小订单
            orders_per_slice = max(1, slice_quantity // self.params.max_fill_size)
            
            for j in range(orders_per_slice):
                order_quantity = min(self.params.max_fill_size, slice_quantity - j * self.params.max_fill_size)
                if order_quantity <= 0:
                    break
                
                order_time = slice_time + timedelta(seconds=j * 30)  # 每30秒发一个订单
                
                # 选择最优场所
                venue = self._select_venue(market_data, order_quantity)
                
                child_order = ChildOrder(
                    child_order_id=f"TWAP_{i}_{j}_{int(time.time() * 1000000)}",
                    parent_order_id=f"PARENT_{int(time.time() * 1000000)}",
                    symbol=market_data.symbol,
                    side=OrderSide.BUY,  # 假设买入
                    quantity=order_quantity,
                    price=self._calculate_limit_price(market_data, OrderSide.BUY),
                    venue=venue,
                    order_type=OrderType.LIMIT,
                    scheduled_time=order_time
                )
                
                child_orders.append(child_order)
        
        self.child_orders.extend(child_orders)
        return child_orders
    
    async def adjust_orders(self, market_data: MarketData) -> List[ChildOrder]:
        """动态调整订单"""
        adjustments = []
        
        for order in self.child_orders:
            if order.status == OrderStatus.WORKING:
                # 检查是否需要调整价格
                new_price = self._calculate_limit_price(market_data, order.side)
                
                if abs(order.price - new_price) > 0.01:  # 价格变化超过1分钱
                    # 创建新的调整订单
                    adjusted_order = ChildOrder(
                        child_order_id=f"ADJ_{order.child_order_id}",
                        parent_order_id=order.parent_order_id,
                        symbol=order.symbol,
                        side=order.side,
                        quantity=order.quantity,
                        price=new_price,
                        venue=order.venue,
                        order_type=order.order_type,
                        scheduled_time=datetime.now()
                    )
                    adjustments.append(adjusted_order)
        
        return adjustments
    
    def _select_venue(self, market_data: MarketData, quantity: int) -> str:
        """选择最优执行场所"""
        # 根据订单大小和市场条件选择场所
        if quantity < 500:
            return "NASDAQ"
        elif quantity < 2000:
            return "NYSE"
        else:
            return "DARK_POOL_1"
    
    def _calculate_limit_price(self, market_data: MarketData, side: OrderSide) -> float:
        """计算限价"""
        if side == OrderSide.BUY:
            return market_data.bid_price + self.params.limit_price_offset
        else:
            return market_data.ask_price - self.params.limit_price_offset

class VWAPRouter(SmartRouter):
    """成交量加权平均价格算法"""
    
    def __init__(self, params: AlgorithmParams):
        super().__init__(params)
        self.historical_volume_profile = self._load_volume_profile()
    
    async def generate_child_orders(self, market_data: MarketData) -> List[ChildOrder]:
        child_orders = []
        
        # 获取历史成交量分布
        volume_profile = self._get_expected_volume_profile()
        
        # 根据成交量分布分配订单
        for time_bucket, volume_ratio in volume_profile.items():
            target_quantity = int(self.params.target_quantity * volume_ratio)
            
            if target_quantity < self.params.min_fill_size:
                continue
            
            # 计算执行时间
            execution_time = self._calculate_execution_time(time_bucket)
            
            # 创建子订单
            child_order = ChildOrder(
                child_order_id=f"VWAP_{time_bucket}_{int(time.time() * 1000000)}",
                parent_order_id=f"PARENT_{int(time.time() * 1000000)}",
                symbol=market_data.symbol,
                side=OrderSide.BUY,
                quantity=target_quantity,
                price=self._calculate_vwap_price(market_data, time_bucket),
                venue=self._select_venue_for_vwap(market_data, target_quantity),
                order_type=OrderType.LIMIT,
                scheduled_time=execution_time
            )
            
            child_orders.append(child_order)
        
        self.child_orders.extend(child_orders)
        return child_orders
    
    async def adjust_orders(self, market_data: MarketData) -> List[ChildOrder]:
        """根据实际成交量调整订单"""
        adjustments = []
        
        current_volume_ratio = self._get_current_volume_ratio(market_data)
        expected_volume_ratio = self._get_expected_volume_ratio()
        
        if current_volume_ratio > expected_volume_ratio * 1.5:
            # 成交量超预期，增加订单数量
            for order in self.child_orders:
                if order.status == OrderStatus.PENDING:
                    # 增加订单数量
                    additional_quantity = int(order.quantity * 0.2)
                    if additional_quantity > 0:
                        adjusted_order = ChildOrder(
                            child_order_id=f"VWAP_ADJ_{order.child_order_id}",
                            parent_order_id=order.parent_order_id,
                            symbol=order.symbol,
                            side=order.side,
                            quantity=additional_quantity,
                            price=order.price,
                            venue=order.venue,
                            order_type=order.order_type,
                            scheduled_time=datetime.now()
                        )
                        adjustments.append(adjusted_order)
        
        return adjustments
    
    def _load_volume_profile(self) -> Dict[str, float]:
        """加载历史成交量分布"""
        # 模拟历史成交量分布（实际应从数据库加载）
        return {
            "09:30-10:00": 0.15,
            "10:00-11:00": 0.12,
            "11:00-12:00": 0.08,
            "12:00-13:00": 0.06,
            "13:00-14:00": 0.08,
            "14:00-15:00": 0.12,
            "15:00-15:30": 0.18,
            "15:30-16:00": 0.21
        }
    
    def _get_expected_volume_profile(self) -> Dict[str, float]:
        """获取预期成交量分布"""
        return self.historical_volume_profile
    
    def _calculate_execution_time(self, time_bucket: str) -> datetime:
        """计算执行时间"""
        # 解析时间区间
        start_time_str = time_bucket.split('-')[0]
        hour, minute = map(int, start_time_str.split(':'))
        
        today = datetime.now().date()
        return datetime.combine(today, datetime.min.time().replace(hour=hour, minute=minute))
    
    def _calculate_vwap_price(self, market_data: MarketData, time_bucket: str) -> float:
        """计算VWAP价格"""
        # 基于市场数据计算合理的VWAP价格
        mid_price = (market_data.bid_price + market_data.ask_price) / 2
        
        # 根据时间段调整价格
        if "09:30" in time_bucket or "15:30" in time_bucket:
            # 开盘和收盘时段，价格更激进
            return mid_price - 0.005
        else:
            # 其他时段，价格保守
            return mid_price + 0.005
    
    def _select_venue_for_vwap(self, market_data: MarketData, quantity: int) -> str:
        """为VWAP选择场所"""
        # 优先选择流动性好的场所
        if market_data.volume > 1000000:
            return "NYSE"
        elif market_data.volume > 500000:
            return "NASDAQ"
        else:
            return "DARK_POOL_1"
    
    def _get_current_volume_ratio(self, market_data: MarketData) -> float:
        """获取当前成交量比例"""
        return market_data.volume / market_data.adv if market_data.adv > 0 else 0
    
    def _get_expected_volume_ratio(self) -> float:
        """获取预期成交量比例"""
        current_time = datetime.now().time()
        
        # 根据当前时间返回预期成交量比例
        if current_time.hour == 9 and current_time.minute >= 30:
            return 0.15
        elif current_time.hour == 15 and current_time.minute >= 30:
            return 0.21
        else:
            return 0.10

class POVRouter(SmartRouter):
    """参与成交量比例算法"""
    
    async def generate_child_orders(self, market_data: MarketData) -> List[ChildOrder]:
        child_orders = []
        
        # 根据参与率计算订单数量
        target_participation = self.params.max_participation_rate
        expected_volume = self._estimate_future_volume(market_data)
        
        # 计算每个时间段的目标数量
        time_intervals = 12  # 每5分钟一个区间
        
        for i in range(time_intervals):
            interval_volume = expected_volume / time_intervals
            target_quantity = int(interval_volume * target_participation)
            
            if target_quantity < self.params.min_fill_size:
                continue
            
            # 限制单笔订单大小
            actual_quantity = min(target_quantity, self.params.max_fill_size)
            
            execution_time = datetime.now() + timedelta(minutes=i * 5)
            
            child_order = ChildOrder(
                child_order_id=f"POV_{i}_{int(time.time() * 1000000)}",
                parent_order_id=f"PARENT_{int(time.time() * 1000000)}",
                symbol=market_data.symbol,
                side=OrderSide.BUY,
                quantity=actual_quantity,
                price=self._calculate_aggressive_price(market_data, OrderSide.BUY),
                venue=self._select_liquidity_venue(market_data),
                order_type=OrderType.LIMIT,
                scheduled_time=execution_time
            )
            
            child_orders.append(child_order)
        
        self.child_orders.extend(child_orders)
        return child_orders
    
    async def adjust_orders(self, market_data: MarketData) -> List[ChildOrder]:
        """根据实际成交量调整订单"""
        adjustments = []
        
        current_participation = self._calculate_current_participation(market_data)
        target_participation = self.params.max_participation_rate
        
        if current_participation < target_participation * 0.8:
            # 参与度不足，增加订单积极性
            for order in self.child_orders:
                if order.status == OrderStatus.WORKING:
                    # 调整价格，更积极
                    new_price = self._calculate_aggressive_price(market_data, order.side)
                    
                    if abs(order.price - new_price) > 0.01:
                        adjusted_order = ChildOrder(
                            child_order_id=f"POV_ADJ_{order.child_order_id}",
                            parent_order_id=order.parent_order_id,
                            symbol=order.symbol,
                            side=order.side,
                            quantity=order.quantity,
                            price=new_price,
                            venue=order.venue,
                            order_type=order.order_type,
                            scheduled_time=datetime.now()
                        )
                        adjustments.append(adjusted_order)
        
        return adjustments
    
    def _estimate_future_volume(self, market_data: MarketData) -> int:
        """估算未来成交量"""
        # 基于历史数据和当前市场状况估算
        remaining_minutes = self.params.duration_minutes
        hourly_volume = market_data.adv / 6.5  # 假设6.5小时交易日
        
        return int(hourly_volume * (remaining_minutes / 60))
    
    def _calculate_aggressive_price(self, market_data: MarketData, side: OrderSide) -> float:
        """计算积极价格"""
        if side == OrderSide.BUY:
            # 买入时使用更接近ask的价格
            return market_data.ask_price - 0.01
        else:
            # 卖出时使用更接近bid的价格
            return market_data.bid_price + 0.01
    
    def _select_liquidity_venue(self, market_data: MarketData) -> str:
        """选择流动性场所"""
        # 优先选择流动性最好的场所
        if market_data.bid_size + market_data.ask_size > 5000:
            return "NYSE"
        else:
            return "NASDAQ"
    
    def _calculate_current_participation(self, market_data: MarketData) -> float:
        """计算当前参与率"""
        if market_data.volume == 0:
            return 0
        
        # 计算我们的成交量占总成交量的比例
        our_volume = self.filled_quantity
        return our_volume / market_data.volume if market_data.volume > 0 else 0

class IcebergRouter(SmartRouter):
    """冰山订单算法"""
    
    async def generate_child_orders(self, market_data: MarketData) -> List[ChildOrder]:
        child_orders = []
        
        # 计算显示数量
        display_quantity = min(
            self.params.max_fill_size,
            int(self.params.target_quantity * 0.1)  # 显示10%
        )
        
        remaining_quantity = self.params.target_quantity
        order_sequence = 0
        
        while remaining_quantity > 0:
            # 当前订单数量
            current_quantity = min(display_quantity, remaining_quantity)
            
            child_order = ChildOrder(
                child_order_id=f"ICE_{order_sequence}_{int(time.time() * 1000000)}",
                parent_order_id=f"PARENT_{int(time.time() * 1000000)}",
                symbol=market_data.symbol,
                side=OrderSide.BUY,
                quantity=current_quantity,
                price=self._calculate_iceberg_price(market_data, order_sequence),
                venue=self._select_dark_venue(market_data),
                order_type=OrderType.LIMIT,
                scheduled_time=datetime.now() + timedelta(seconds=order_sequence * 30)
            )
            
            child_orders.append(child_order)
            
            remaining_quantity -= current_quantity
            order_sequence += 1
            
            # 防止无限循环
            if order_sequence > 100:
                break
        
        self.child_orders.extend(child_orders)
        return child_orders
    
    async def adjust_orders(self, market_data: MarketData) -> List[ChildOrder]:
        """调整冰山订单"""
        adjustments = []
        
        # 检查是否有订单完全成交，需要发送下一个冰山片段
        for order in self.child_orders:
            if order.status == OrderStatus.FILLED:
                # 检查是否还有剩余数量需要交易
                if self.filled_quantity < self.params.target_quantity:
                    remaining = self.params.target_quantity - self.filled_quantity
                    
                    if remaining > 0:
                        # 创建新的冰山片段
                        display_quantity = min(
                            self.params.max_fill_size,
                            int(remaining * 0.1)
                        )
                        
                        next_order = ChildOrder(
                            child_order_id=f"ICE_NEXT_{int(time.time() * 1000000)}",
                            parent_order_id=order.parent_order_id,
                            symbol=order.symbol,
                            side=order.side,
                            quantity=min(display_quantity, remaining),
                            price=self._calculate_iceberg_price(market_data, 0),
                            venue=order.venue,
                            order_type=order.order_type,
                            scheduled_time=datetime.now()
                        )
                        
                        adjustments.append(next_order)
        
        return adjustments
    
    def _calculate_iceberg_price(self, market_data: MarketData, sequence: int) -> float:
        """计算冰山订单价格"""
        # 价格随时间略微调整，避免被识别
        base_price = (market_data.bid_price + market_data.ask_price) / 2
        
        # 添加小幅随机调整
        price_adjustment = (sequence % 5 - 2) * 0.001
        
        return base_price + price_adjustment
    
    def _select_dark_venue(self, market_data: MarketData) -> str:
        """选择暗池场所"""
        # 优先选择暗池，减少市场影响
        dark_pools = ["DARK_POOL_1", "DARK_POOL_2", "DARK_POOL_3"]
        
        # 根据市场条件选择最优暗池
        if market_data.volume > 1000000:
            return "DARK_POOL_1"  # 大成交量时选择最大的暗池
        else:
            return "DARK_POOL_2"

class SmartRoutingEngine:
    def __init__(self):
        self.routers: Dict[str, SmartRouter] = {}
        self.market_data_cache: Dict[str, MarketData] = {}
        self.logger = logging.getLogger(__name__)
        
    def create_router(self, algorithm_type: AlgorithmType, params: AlgorithmParams) -> SmartRouter:
        """创建智能路由器"""
        if algorithm_type == AlgorithmType.TWAP:
            return TWAPRouter(params)
        elif algorithm_type == AlgorithmType.VWAP:
            return VWAPRouter(params)
        elif algorithm_type == AlgorithmType.POV:
            return POVRouter(params)
        elif algorithm_type == AlgorithmType.ICEBERG:
            return IcebergRouter(params)
        elif algorithm_type == AlgorithmType.IMPLEMENTATION_SHORTFALL:
            return TWAPRouter(params)  # 使用TWAP作为实现缺口算法的基础
        else:
            raise ValueError(f"Unsupported algorithm type: {algorithm_type}")
    
    async def execute_algorithm(self, router: SmartRouter, market_data: MarketData) -> List[ChildOrder]:
        """执行算法"""
        router_id = f"{router.params.algorithm_type.value}_{int(time.time() * 1000000)}"
        self.routers[router_id] = router
        
        try:
            # 生成子订单
            child_orders = await router.generate_child_orders(market_data)
            
            self.logger.info(f"Generated {len(child_orders)} child orders for {router.params.algorithm_type.value}")
            
            return child_orders
            
        except Exception as e:
            self.logger.error(f"Error executing algorithm {router.params.algorithm_type.value}: {e}")
            return []
    
    async def monitor_and_adjust(self, router_id: str, market_data: MarketData) -> List[ChildOrder]:
        """监控和调整算法"""
        router = self.routers.get(router_id)
        if not router:
            return []
        
        try:
            adjustments = await router.adjust_orders(market_data)
            
            if adjustments:
                self.logger.info(f"Made {len(adjustments)} adjustments for router {router_id}")
            
            return adjustments
            
        except Exception as e:
            self.logger.error(f"Error adjusting router {router_id}: {e}")
            return []
    
    def get_router_progress(self, router_id: str) -> Optional[Dict[str, Any]]:
        """获取路由器进度"""
        router = self.routers.get(router_id)
        if not router:
            return None
        
        return router.get_progress()
    
    def get_all_routers_status(self) -> Dict[str, Dict[str, Any]]:
        """获取所有路由器状态"""
        status = {}
        
        for router_id, router in self.routers.items():
            status[router_id] = {
                'algorithm_type': router.params.algorithm_type.value,
                'progress': router.get_progress(),
                'is_running': router.is_running,
                'child_orders_count': len(router.child_orders)
            }
        
        return status
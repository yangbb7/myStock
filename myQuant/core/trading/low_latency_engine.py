import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue, Empty
import struct
import socket

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TWAP = "twap"
    VWAP = "vwap"
    POV = "pov"
    ICEBERG = "iceberg"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    PENDING = "pending"
    WORKING = "working"
    PARTIAL_FILL = "partial_fill"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

class VenueType(Enum):
    EXCHANGE = "exchange"
    DARK_POOL = "dark_pool"
    ECN = "ecn"
    BROKER = "broker"

@dataclass
class Order:
    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "DAY"
    venue: Optional[str] = None
    venue_type: Optional[VenueType] = None
    client_order_id: Optional[str] = None
    parent_order_id: Optional[str] = None
    algo_params: Dict[str, Any] = field(default_factory=dict)
    created_time: datetime = field(default_factory=datetime.now)
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    avg_fill_price: float = 0.0
    leaves_quantity: int = 0
    last_fill_time: Optional[datetime] = None
    latency_us: int = 0

@dataclass
class Fill:
    fill_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    price: float
    venue: str
    timestamp: datetime
    execution_id: str
    commission: float = 0.0
    liquidity_flag: str = "A"  # A=Added, R=Removed, H=Hidden

@dataclass
class LatencyMetrics:
    order_to_market_us: int = 0
    market_to_fill_us: int = 0
    total_round_trip_us: int = 0
    network_latency_us: int = 0
    processing_latency_us: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

class LowLatencyTradingEngine:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 高性能组件
        self.order_queue = Queue(maxsize=10000)
        self.fill_queue = Queue(maxsize=10000)
        self.market_data_queue = Queue(maxsize=50000)
        
        # 执行器
        self.executor = ThreadPoolExecutor(max_workers=config.get('max_workers', 4))
        
        # 订单管理
        self.orders: Dict[str, Order] = {}
        self.order_lock = threading.RLock()
        
        # 性能指标
        self.latency_metrics: List[LatencyMetrics] = []
        self.metrics_lock = threading.RLock()
        
        # 运行状态
        self.is_running = False
        self.worker_threads: List[threading.Thread] = []
        
        # 连接管理
        self.broker_connections: Dict[str, Any] = {}
        self.market_data_connections: Dict[str, Any] = {}
        
        # 风险控制
        self.risk_limits = config.get('risk_limits', {})
        self.position_limits = config.get('position_limits', {})
        
    def start(self):
        """启动低延迟交易引擎"""
        if self.is_running:
            return
        
        self.is_running = True
        self.logger.info("Starting Low Latency Trading Engine...")
        
        # 启动工作线程
        self._start_worker_threads()
        
        # 初始化连接
        self._initialize_connections()
        
        self.logger.info("Low Latency Trading Engine started successfully")
    
    def stop(self):
        """停止交易引擎"""
        if not self.is_running:
            return
        
        self.logger.info("Stopping Low Latency Trading Engine...")
        
        self.is_running = False
        
        # 等待工作线程结束
        for thread in self.worker_threads:
            thread.join(timeout=5)
        
        # 关闭连接
        self._cleanup_connections()
        
        self.logger.info("Low Latency Trading Engine stopped")
    
    def submit_order(self, order: Order) -> bool:
        """提交订单到交易引擎"""
        start_time = time.time_ns()
        
        try:
            # 预检查
            if not self._pre_order_check(order):
                return False
            
            # 计算处理延迟
            order.latency_us = (time.time_ns() - start_time) // 1000
            
            # 加入队列
            self.order_queue.put(order, timeout=0.001)
            
            # 记录订单
            with self.order_lock:
                self.orders[order.order_id] = order
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to submit order {order.order_id}: {e}")
            return False
    
    def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        start_time = time.time_ns()
        
        try:
            with self.order_lock:
                if order_id not in self.orders:
                    return False
                
                order = self.orders[order_id]
                if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
                    return False
            
            # 发送取消请求
            cancel_latency = (time.time_ns() - start_time) // 1000
            
            # 模拟取消执行
            self._process_cancel_order(order_id)
            
            self.logger.debug(f"Cancel order {order_id} processed in {cancel_latency}μs")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """获取订单状态"""
        with self.order_lock:
            return self.orders.get(order_id)
    
    def get_latency_stats(self) -> Dict[str, Any]:
        """获取延迟统计"""
        with self.metrics_lock:
            if not self.latency_metrics:
                return {}
            
            recent_metrics = self.latency_metrics[-1000:]  # 最近1000笔
            
            order_to_market = [m.order_to_market_us for m in recent_metrics]
            market_to_fill = [m.market_to_fill_us for m in recent_metrics]
            total_round_trip = [m.total_round_trip_us for m in recent_metrics]
            
            return {
                'order_to_market_us': {
                    'avg': sum(order_to_market) / len(order_to_market),
                    'p50': sorted(order_to_market)[len(order_to_market) // 2],
                    'p95': sorted(order_to_market)[int(len(order_to_market) * 0.95)],
                    'p99': sorted(order_to_market)[int(len(order_to_market) * 0.99)],
                    'max': max(order_to_market),
                    'min': min(order_to_market)
                },
                'market_to_fill_us': {
                    'avg': sum(market_to_fill) / len(market_to_fill),
                    'p50': sorted(market_to_fill)[len(market_to_fill) // 2],
                    'p95': sorted(market_to_fill)[int(len(market_to_fill) * 0.95)],
                    'p99': sorted(market_to_fill)[int(len(market_to_fill) * 0.99)],
                    'max': max(market_to_fill),
                    'min': min(market_to_fill)
                },
                'total_round_trip_us': {
                    'avg': sum(total_round_trip) / len(total_round_trip),
                    'p50': sorted(total_round_trip)[len(total_round_trip) // 2],
                    'p95': sorted(total_round_trip)[int(len(total_round_trip) * 0.95)],
                    'p99': sorted(total_round_trip)[int(len(total_round_trip) * 0.99)],
                    'max': max(total_round_trip),
                    'min': min(total_round_trip)
                },
                'sample_count': len(recent_metrics)
            }
    
    def _start_worker_threads(self):
        """启动工作线程"""
        # 订单处理线程
        order_thread = threading.Thread(
            target=self._order_processing_loop,
            name="OrderProcessor",
            daemon=True
        )
        order_thread.start()
        self.worker_threads.append(order_thread)
        
        # 成交处理线程
        fill_thread = threading.Thread(
            target=self._fill_processing_loop,
            name="FillProcessor",
            daemon=True
        )
        fill_thread.start()
        self.worker_threads.append(fill_thread)
        
        # 市场数据处理线程
        market_data_thread = threading.Thread(
            target=self._market_data_processing_loop,
            name="MarketDataProcessor",
            daemon=True
        )
        market_data_thread.start()
        self.worker_threads.append(market_data_thread)
    
    def _order_processing_loop(self):
        """订单处理循环"""
        while self.is_running:
            try:
                order = self.order_queue.get(timeout=0.1)
                self._process_order(order)
                self.order_queue.task_done()
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in order processing loop: {e}")
    
    def _fill_processing_loop(self):
        """成交处理循环"""
        while self.is_running:
            try:
                fill = self.fill_queue.get(timeout=0.1)
                self._process_fill(fill)
                self.fill_queue.task_done()
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in fill processing loop: {e}")
    
    def _market_data_processing_loop(self):
        """市场数据处理循环"""
        while self.is_running:
            try:
                market_data = self.market_data_queue.get(timeout=0.1)
                self._process_market_data(market_data)
                self.market_data_queue.task_done()
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in market data processing loop: {e}")
    
    def _process_order(self, order: Order):
        """处理订单"""
        start_time = time.time_ns()
        
        try:
            # 更新订单状态
            order.status = OrderStatus.WORKING
            order.leaves_quantity = order.quantity
            
            # 发送到交易所/券商
            self._send_order_to_venue(order)
            
            # 记录延迟
            processing_time = (time.time_ns() - start_time) // 1000
            
            metrics = LatencyMetrics(
                order_to_market_us=processing_time,
                processing_latency_us=processing_time
            )
            
            with self.metrics_lock:
                self.latency_metrics.append(metrics)
            
            self.logger.debug(f"Order {order.order_id} processed in {processing_time}μs")
            
        except Exception as e:
            self.logger.error(f"Error processing order {order.order_id}: {e}")
            order.status = OrderStatus.REJECTED
    
    def _process_fill(self, fill: Fill):
        """处理成交"""
        start_time = time.time_ns()
        
        try:
            with self.order_lock:
                order = self.orders.get(fill.order_id)
                if not order:
                    return
                
                # 更新订单状态
                order.filled_quantity += fill.quantity
                order.leaves_quantity = order.quantity - order.filled_quantity
                order.last_fill_time = fill.timestamp
                
                # 计算平均成交价
                if order.filled_quantity > 0:
                    total_value = order.avg_fill_price * (order.filled_quantity - fill.quantity)
                    total_value += fill.price * fill.quantity
                    order.avg_fill_price = total_value / order.filled_quantity
                
                # 更新状态
                if order.leaves_quantity == 0:
                    order.status = OrderStatus.FILLED
                else:
                    order.status = OrderStatus.PARTIAL_FILL
            
            # 记录延迟
            processing_time = (time.time_ns() - start_time) // 1000
            
            # 更新延迟指标
            with self.metrics_lock:
                if self.latency_metrics:
                    self.latency_metrics[-1].market_to_fill_us = processing_time
                    self.latency_metrics[-1].total_round_trip_us = (
                        self.latency_metrics[-1].order_to_market_us + processing_time
                    )
            
            self.logger.debug(f"Fill {fill.fill_id} processed in {processing_time}μs")
            
        except Exception as e:
            self.logger.error(f"Error processing fill {fill.fill_id}: {e}")
    
    def _process_market_data(self, market_data: Dict[str, Any]):
        """处理市场数据"""
        # 高频市场数据处理
        # 用于算法交易决策
        pass
    
    def _send_order_to_venue(self, order: Order):
        """发送订单到交易场所"""
        # 模拟发送订单
        venue = order.venue or "default_venue"
        
        # 模拟网络延迟
        network_delay = 0.0001  # 100微秒
        time.sleep(network_delay)
        
        # 模拟成交
        if order.order_type == OrderType.MARKET:
            # 市价单立即成交
            fill = Fill(
                fill_id=f"F{int(time.time() * 1000000)}",
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=order.price or 100.0,
                venue=venue,
                timestamp=datetime.now(),
                execution_id=f"E{int(time.time() * 1000000)}"
            )
            
            # 异步处理成交
            self.fill_queue.put(fill)
    
    def _process_cancel_order(self, order_id: str):
        """处理取消订单"""
        with self.order_lock:
            if order_id in self.orders:
                order = self.orders[order_id]
                if order.status == OrderStatus.WORKING:
                    order.status = OrderStatus.CANCELLED
    
    def _pre_order_check(self, order: Order) -> bool:
        """订单预检查"""
        # 风险控制检查
        if not self._check_risk_limits(order):
            return False
        
        # 仓位限制检查
        if not self._check_position_limits(order):
            return False
        
        # 订单合规性检查
        if not self._check_order_compliance(order):
            return False
        
        return True
    
    def _check_risk_limits(self, order: Order) -> bool:
        """检查风险限制"""
        # 单笔订单限制
        max_order_size = self.risk_limits.get('max_order_size', 100000)
        if order.quantity > max_order_size:
            self.logger.warning(f"Order {order.order_id} exceeds max size limit")
            return False
        
        # 单日交易限制
        daily_limit = self.risk_limits.get('daily_trading_limit', 1000000)
        # 这里应该检查当日累计交易量
        
        return True
    
    def _check_position_limits(self, order: Order) -> bool:
        """检查仓位限制"""
        # 单只股票仓位限制
        max_position = self.position_limits.get(order.symbol, 50000)
        # 这里应该检查当前仓位
        
        return True
    
    def _check_order_compliance(self, order: Order) -> bool:
        """检查订单合规性"""
        # 基本合规检查
        if order.quantity <= 0:
            return False
        
        if order.order_type == OrderType.LIMIT and order.price <= 0:
            return False
        
        return True
    
    def _initialize_connections(self):
        """初始化连接"""
        # 初始化券商连接
        for broker_name, config in self.config.get('brokers', {}).items():
            try:
                # 这里应该建立真实的券商连接
                self.broker_connections[broker_name] = {
                    'host': config.get('host'),
                    'port': config.get('port'),
                    'api_key': config.get('api_key'),
                    'connected': True
                }
                self.logger.info(f"Connected to broker: {broker_name}")
            except Exception as e:
                self.logger.error(f"Failed to connect to broker {broker_name}: {e}")
        
        # 初始化市场数据连接
        for feed_name, config in self.config.get('market_data_feeds', {}).items():
            try:
                self.market_data_connections[feed_name] = {
                    'host': config.get('host'),
                    'port': config.get('port'),
                    'connected': True
                }
                self.logger.info(f"Connected to market data feed: {feed_name}")
            except Exception as e:
                self.logger.error(f"Failed to connect to feed {feed_name}: {e}")
    
    def _cleanup_connections(self):
        """清理连接"""
        for broker_name in self.broker_connections:
            try:
                # 关闭券商连接
                self.broker_connections[broker_name]['connected'] = False
                self.logger.info(f"Disconnected from broker: {broker_name}")
            except Exception as e:
                self.logger.error(f"Error disconnecting from broker {broker_name}: {e}")
        
        for feed_name in self.market_data_connections:
            try:
                # 关闭市场数据连接
                self.market_data_connections[feed_name]['connected'] = False
                self.logger.info(f"Disconnected from market data feed: {feed_name}")
            except Exception as e:
                self.logger.error(f"Error disconnecting from feed {feed_name}: {e}")
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """获取引擎统计信息"""
        return {
            'is_running': self.is_running,
            'total_orders': len(self.orders),
            'pending_orders': len([o for o in self.orders.values() if o.status == OrderStatus.PENDING]),
            'working_orders': len([o for o in self.orders.values() if o.status == OrderStatus.WORKING]),
            'filled_orders': len([o for o in self.orders.values() if o.status == OrderStatus.FILLED]),
            'cancelled_orders': len([o for o in self.orders.values() if o.status == OrderStatus.CANCELLED]),
            'order_queue_size': self.order_queue.qsize(),
            'fill_queue_size': self.fill_queue.qsize(),
            'market_data_queue_size': self.market_data_queue.qsize(),
            'broker_connections': len(self.broker_connections),
            'market_data_connections': len(self.market_data_connections),
            'latency_metrics_count': len(self.latency_metrics)
        }
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from scipy.stats import norm, poisson, exponential
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
from collections import deque, defaultdict
import random
import warnings
warnings.filterwarnings('ignore')

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    ICEBERG = "iceberg"
    HIDDEN = "hidden"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    PENDING = "pending"
    PARTIAL_FILL = "partial_fill"
    FILLED = "filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"

class TraderType(Enum):
    MARKET_MAKER = "market_maker"
    INSTITUTIONAL = "institutional"
    RETAIL = "retail"
    HFT = "hft"
    ARBITRAGEUR = "arbitrageur"
    NOISE_TRADER = "noise_trader"

class MarketEvent(Enum):
    ORDER_ARRIVAL = "order_arrival"
    ORDER_CANCEL = "order_cancel"
    ORDER_MODIFY = "order_modify"
    TRADE_EXECUTION = "trade_execution"
    PRICE_UPDATE = "price_update"
    VOLUME_UPDATE = "volume_update"

@dataclass
class Order:
    """订单"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float]
    trader_type: TraderType
    timestamp: datetime
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0
    remaining_quantity: float = 0
    time_in_force: str = "GTC"  # Good Till Cancel
    hidden_quantity: float = 0
    minimum_quantity: float = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.remaining_quantity == 0:
            self.remaining_quantity = self.quantity

@dataclass
class Trade:
    """交易"""
    trade_id: str
    symbol: str
    price: float
    quantity: float
    timestamp: datetime
    buyer_order_id: str
    seller_order_id: str
    buyer_trader_type: TraderType
    seller_trader_type: TraderType
    is_aggressive: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OrderBookLevel:
    """订单簿层级"""
    price: float
    quantity: float
    order_count: int
    orders: List[Order] = field(default_factory=list)

@dataclass
class OrderBook:
    """订单簿"""
    symbol: str
    bids: List[OrderBookLevel] = field(default_factory=list)
    asks: List[OrderBookLevel] = field(default_factory=list)
    last_update: datetime = field(default_factory=datetime.now)
    
    def get_best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None
    
    def get_best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None
    
    def get_spread(self) -> Optional[float]:
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        return best_ask - best_bid if best_bid and best_ask else None
    
    def get_mid_price(self) -> Optional[float]:
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        return (best_bid + best_ask) / 2 if best_bid and best_ask else None

@dataclass
class MarketState:
    """市场状态"""
    symbol: str
    timestamp: datetime
    last_price: float
    volume: float
    order_book: OrderBook
    volatility: float
    liquidity: float
    imbalance: float
    microstructure_noise: float
    market_regime: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TraderBehavior:
    """交易员行为模型"""
    trader_type: TraderType
    arrival_rate: float  # 订单到达率
    order_size_distribution: Dict[str, float]  # 订单大小分布参数
    price_improvement_prob: float  # 价格改善概率
    cancel_rate: float  # 取消率
    latency_distribution: Dict[str, float]  # 延迟分布参数
    risk_aversion: float  # 风险厌恶系数
    information_advantage: float  # 信息优势
    reaction_speed: float  # 反应速度
    market_impact_sensitivity: float  # 市场冲击敏感性
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SimulationResult:
    """模拟结果"""
    symbol: str
    start_time: datetime
    end_time: datetime
    total_trades: int
    total_volume: float
    price_history: List[float]
    spread_history: List[float]
    volume_history: List[float]
    order_book_snapshots: List[OrderBook]
    trade_history: List[Trade]
    market_quality_metrics: Dict[str, float]
    microstructure_metrics: Dict[str, float]
    trader_performance: Dict[TraderType, Dict[str, float]]
    simulation_parameters: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

class MarketMicrostructureSimulator:
    """
    市场微观结构模拟器
    
    模拟订单簿动态、交易员行为、市场冲击、流动性提供等
    市场微观结构特征，用于分析交易策略的市场影响。
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 模拟参数
        self.symbol = config.get('symbol', 'DEFAULT')
        self.tick_size = config.get('tick_size', 0.01)
        self.lot_size = config.get('lot_size', 100)
        self.market_hours = config.get('market_hours', (9, 16))
        
        # 订单簿参数
        self.max_book_depth = config.get('max_book_depth', 10)
        self.initial_spread = config.get('initial_spread', 0.02)
        self.initial_price = config.get('initial_price', 100.0)
        
        # 交易员类型和行为
        self.trader_behaviors = {}
        self._initialize_trader_behaviors()
        
        # 市场状态
        self.current_time = datetime.now()
        self.order_book = OrderBook(symbol=self.symbol)
        self.market_state = MarketState(
            symbol=self.symbol,
            timestamp=self.current_time,
            last_price=self.initial_price,
            volume=0,
            order_book=self.order_book,
            volatility=0.02,
            liquidity=1.0,
            imbalance=0.0,
            microstructure_noise=0.0,
            market_regime="normal"
        )
        
        # 事件队列
        self.event_queue = []
        self.next_order_id = 1
        self.next_trade_id = 1
        
        # 历史数据
        self.trade_history = []
        self.order_history = []
        self.price_history = []
        self.spread_history = []
        self.volume_history = []
        self.order_book_snapshots = []
        
        # 性能指标
        self.market_quality_metrics = {}
        self.microstructure_metrics = {}
        
        # 随机数生成器
        self.rng = np.random.RandomState(config.get('random_seed', 42))
        
    def _initialize_trader_behaviors(self):
        """初始化交易员行为模型"""
        # 市场做市商
        self.trader_behaviors[TraderType.MARKET_MAKER] = TraderBehavior(
            trader_type=TraderType.MARKET_MAKER,
            arrival_rate=50.0,  # 高频率
            order_size_distribution={'mean': 1000, 'std': 200},
            price_improvement_prob=0.8,
            cancel_rate=0.3,
            latency_distribution={'mean': 0.1, 'std': 0.05},
            risk_aversion=0.3,
            information_advantage=0.2,
            reaction_speed=0.9,
            market_impact_sensitivity=0.8
        )
        
        # 机构投资者
        self.trader_behaviors[TraderType.INSTITUTIONAL] = TraderBehavior(
            trader_type=TraderType.INSTITUTIONAL,
            arrival_rate=5.0,
            order_size_distribution={'mean': 10000, 'std': 5000},
            price_improvement_prob=0.6,
            cancel_rate=0.1,
            latency_distribution={'mean': 1.0, 'std': 0.5},
            risk_aversion=0.7,
            information_advantage=0.6,
            reaction_speed=0.5,
            market_impact_sensitivity=0.9
        )
        
        # 零售交易者
        self.trader_behaviors[TraderType.RETAIL] = TraderBehavior(
            trader_type=TraderType.RETAIL,
            arrival_rate=20.0,
            order_size_distribution={'mean': 500, 'std': 200},
            price_improvement_prob=0.3,
            cancel_rate=0.2,
            latency_distribution={'mean': 2.0, 'std': 1.0},
            risk_aversion=0.5,
            information_advantage=0.1,
            reaction_speed=0.3,
            market_impact_sensitivity=0.5
        )
        
        # 高频交易
        self.trader_behaviors[TraderType.HFT] = TraderBehavior(
            trader_type=TraderType.HFT,
            arrival_rate=100.0,
            order_size_distribution={'mean': 300, 'std': 100},
            price_improvement_prob=0.9,
            cancel_rate=0.8,
            latency_distribution={'mean': 0.01, 'std': 0.005},
            risk_aversion=0.1,
            information_advantage=0.3,
            reaction_speed=1.0,
            market_impact_sensitivity=0.95
        )
        
        # 套利交易者
        self.trader_behaviors[TraderType.ARBITRAGEUR] = TraderBehavior(
            trader_type=TraderType.ARBITRAGEUR,
            arrival_rate=10.0,
            order_size_distribution={'mean': 2000, 'std': 500},
            price_improvement_prob=0.4,
            cancel_rate=0.1,
            latency_distribution={'mean': 0.5, 'std': 0.2},
            risk_aversion=0.2,
            information_advantage=0.8,
            reaction_speed=0.8,
            market_impact_sensitivity=0.7
        )
        
        # 噪声交易者
        self.trader_behaviors[TraderType.NOISE_TRADER] = TraderBehavior(
            trader_type=TraderType.NOISE_TRADER,
            arrival_rate=15.0,
            order_size_distribution={'mean': 800, 'std': 300},
            price_improvement_prob=0.2,
            cancel_rate=0.3,
            latency_distribution={'mean': 3.0, 'std': 1.5},
            risk_aversion=0.4,
            information_advantage=0.0,
            reaction_speed=0.2,
            market_impact_sensitivity=0.3
        )
    
    async def run_simulation(self, duration_hours: float = 1.0, 
                           time_step_seconds: float = 1.0) -> SimulationResult:
        """运行市场微观结构模拟"""
        try:
            self.logger.info(f"开始市场微观结构模拟，持续时间: {duration_hours}小时")
            
            # 初始化订单簿
            await self._initialize_order_book()
            
            # 计算模拟参数
            total_seconds = duration_hours * 3600
            num_steps = int(total_seconds / time_step_seconds)
            
            # 生成初始事件
            await self._generate_initial_events(time_step_seconds)
            
            # 主模拟循环
            for step in range(num_steps):
                current_time = self.current_time + timedelta(seconds=step * time_step_seconds)
                
                # 处理当前时间的事件
                await self._process_events(current_time)
                
                # 生成新事件
                await self._generate_events(current_time, time_step_seconds)
                
                # 更新市场状态
                await self._update_market_state(current_time)
                
                # 记录快照
                if step % 60 == 0:  # 每分钟记录一次
                    await self._record_snapshot(current_time)
                
                # 进度报告
                if step % (num_steps // 10) == 0:
                    progress = (step / num_steps) * 100
                    self.logger.info(f"模拟进度: {progress:.1f}%")
            
            # 计算结果指标
            await self._calculate_metrics()
            
            # 构建结果
            result = SimulationResult(
                symbol=self.symbol,
                start_time=self.current_time,
                end_time=current_time,
                total_trades=len(self.trade_history),
                total_volume=sum(trade.quantity for trade in self.trade_history),
                price_history=self.price_history,
                spread_history=self.spread_history,
                volume_history=self.volume_history,
                order_book_snapshots=self.order_book_snapshots,
                trade_history=self.trade_history,
                market_quality_metrics=self.market_quality_metrics,
                microstructure_metrics=self.microstructure_metrics,
                trader_performance=await self._calculate_trader_performance(),
                simulation_parameters=self.config
            )
            
            self.logger.info(f"模拟完成，总交易数: {len(self.trade_history)}")
            return result
            
        except Exception as e:
            self.logger.error(f"市场微观结构模拟失败: {e}")
            raise
    
    async def _initialize_order_book(self):
        """初始化订单簿"""
        mid_price = self.initial_price
        spread = self.initial_spread
        
        # 生成买卖订单
        for i in range(5):
            # 买单
            bid_price = mid_price - spread/2 - i * self.tick_size
            bid_quantity = self.rng.uniform(1000, 5000)
            bid_level = OrderBookLevel(
                price=bid_price,
                quantity=bid_quantity,
                order_count=self.rng.randint(1, 5)
            )
            self.order_book.bids.append(bid_level)
            
            # 卖单
            ask_price = mid_price + spread/2 + i * self.tick_size
            ask_quantity = self.rng.uniform(1000, 5000)
            ask_level = OrderBookLevel(
                price=ask_price,
                quantity=ask_quantity,
                order_count=self.rng.randint(1, 5)
            )
            self.order_book.asks.append(ask_level)
        
        # 排序
        self.order_book.bids.sort(key=lambda x: x.price, reverse=True)
        self.order_book.asks.sort(key=lambda x: x.price)
    
    async def _generate_initial_events(self, time_step: float):
        """生成初始事件"""
        for trader_type, behavior in self.trader_behaviors.items():
            # 计算下一个订单到达时间
            next_arrival = self.rng.exponential(1.0 / behavior.arrival_rate)
            event_time = self.current_time + timedelta(seconds=next_arrival)
            
            self.event_queue.append({
                'type': MarketEvent.ORDER_ARRIVAL,
                'time': event_time,
                'trader_type': trader_type
            })
        
        # 排序事件队列
        self.event_queue.sort(key=lambda x: x['time'])
    
    async def _generate_events(self, current_time: datetime, time_step: float):
        """生成新事件"""
        for trader_type, behavior in self.trader_behaviors.items():
            # 基于到达率生成订单
            if self.rng.random() < behavior.arrival_rate * time_step / 3600:
                self.event_queue.append({
                    'type': MarketEvent.ORDER_ARRIVAL,
                    'time': current_time,
                    'trader_type': trader_type
                })
        
        # 排序事件队列
        self.event_queue.sort(key=lambda x: x['time'])
    
    async def _process_events(self, current_time: datetime):
        """处理事件"""
        # 处理所有当前时间的事件
        events_to_process = [e for e in self.event_queue if e['time'] <= current_time]
        
        for event in events_to_process:
            if event['type'] == MarketEvent.ORDER_ARRIVAL:
                await self._handle_order_arrival(event, current_time)
            elif event['type'] == MarketEvent.ORDER_CANCEL:
                await self._handle_order_cancel(event, current_time)
            elif event['type'] == MarketEvent.ORDER_MODIFY:
                await self._handle_order_modify(event, current_time)
            
            # 从队列中移除已处理的事件
            self.event_queue.remove(event)
    
    async def _handle_order_arrival(self, event: Dict[str, Any], current_time: datetime):
        """处理订单到达"""
        trader_type = event['trader_type']
        behavior = self.trader_behaviors[trader_type]
        
        # 生成订单
        order = await self._generate_order(trader_type, behavior, current_time)
        
        if order:
            # 尝试匹配订单
            await self._match_order(order, current_time)
            
            # 如果订单未完全成交，添加到订单簿
            if order.remaining_quantity > 0:
                await self._add_order_to_book(order)
            
            # 记录订单
            self.order_history.append(order)
            
            # 生成下一个订单到达事件
            next_arrival = self.rng.exponential(1.0 / behavior.arrival_rate)
            next_event_time = current_time + timedelta(seconds=next_arrival)
            
            self.event_queue.append({
                'type': MarketEvent.ORDER_ARRIVAL,
                'time': next_event_time,
                'trader_type': trader_type
            })
    
    async def _generate_order(self, trader_type: TraderType, 
                            behavior: TraderBehavior, 
                            current_time: datetime) -> Optional[Order]:
        """生成订单"""
        try:
            # 决定订单方向
            side = OrderSide.BUY if self.rng.random() < 0.5 else OrderSide.SELL
            
            # 订单大小
            mean_size = behavior.order_size_distribution['mean']
            std_size = behavior.order_size_distribution['std']
            quantity = max(self.lot_size, int(self.rng.normal(mean_size, std_size)))
            
            # 订单类型
            if trader_type == TraderType.MARKET_MAKER:
                order_type = OrderType.LIMIT
            elif trader_type == TraderType.HFT:
                order_type = OrderType.LIMIT if self.rng.random() < 0.8 else OrderType.MARKET
            else:
                order_type = OrderType.MARKET if self.rng.random() < 0.3 else OrderType.LIMIT
            
            # 订单价格
            price = None
            if order_type == OrderType.LIMIT:
                price = await self._determine_limit_price(side, behavior, current_time)
            
            # 创建订单
            order = Order(
                order_id=f"ORDER_{self.next_order_id}",
                symbol=self.symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                trader_type=trader_type,
                timestamp=current_time
            )
            
            self.next_order_id += 1
            return order
            
        except Exception as e:
            self.logger.error(f"生成订单失败: {e}")
            return None
    
    async def _determine_limit_price(self, side: OrderSide, 
                                   behavior: TraderBehavior, 
                                   current_time: datetime) -> float:
        """确定限价订单价格"""
        best_bid = self.order_book.get_best_bid()
        best_ask = self.order_book.get_best_ask()
        
        if not best_bid or not best_ask:
            return self.market_state.last_price
        
        # 基于交易员类型确定价格策略
        if behavior.trader_type == TraderType.MARKET_MAKER:
            # 做市商提供流动性
            if side == OrderSide.BUY:
                # 在最佳买价附近或略高
                price = best_bid + self.rng.uniform(-self.tick_size, self.tick_size)
            else:
                # 在最佳卖价附近或略低
                price = best_ask + self.rng.uniform(-self.tick_size, self.tick_size)
        else:
            # 其他交易员更激进
            if side == OrderSide.BUY:
                if self.rng.random() < behavior.price_improvement_prob:
                    price = best_bid + self.tick_size
                else:
                    price = best_bid - self.rng.uniform(0, 3 * self.tick_size)
            else:
                if self.rng.random() < behavior.price_improvement_prob:
                    price = best_ask - self.tick_size
                else:
                    price = best_ask + self.rng.uniform(0, 3 * self.tick_size)
        
        return max(0.01, price)
    
    async def _match_order(self, order: Order, current_time: datetime):
        """匹配订单"""
        if order.order_type == OrderType.MARKET:
            await self._match_market_order(order, current_time)
        elif order.order_type == OrderType.LIMIT:
            await self._match_limit_order(order, current_time)
    
    async def _match_market_order(self, order: Order, current_time: datetime):
        """匹配市价单"""
        if order.side == OrderSide.BUY:
            # 买入市价单匹配卖单
            for ask_level in self.order_book.asks:
                if order.remaining_quantity <= 0:
                    break
                
                fill_quantity = min(order.remaining_quantity, ask_level.quantity)
                
                # 执行交易
                await self._execute_trade(
                    order, ask_level, fill_quantity, ask_level.price, current_time
                )
                
                # 更新订单簿
                ask_level.quantity -= fill_quantity
                order.remaining_quantity -= fill_quantity
                order.filled_quantity += fill_quantity
                
                if ask_level.quantity <= 0:
                    self.order_book.asks.remove(ask_level)
        
        else:  # SELL
            # 卖出市价单匹配买单
            for bid_level in self.order_book.bids:
                if order.remaining_quantity <= 0:
                    break
                
                fill_quantity = min(order.remaining_quantity, bid_level.quantity)
                
                # 执行交易
                await self._execute_trade(
                    order, bid_level, fill_quantity, bid_level.price, current_time
                )
                
                # 更新订单簿
                bid_level.quantity -= fill_quantity
                order.remaining_quantity -= fill_quantity
                order.filled_quantity += fill_quantity
                
                if bid_level.quantity <= 0:
                    self.order_book.bids.remove(bid_level)
        
        # 更新订单状态
        if order.remaining_quantity <= 0:
            order.status = OrderStatus.FILLED
        elif order.filled_quantity > 0:
            order.status = OrderStatus.PARTIAL_FILL
    
    async def _match_limit_order(self, order: Order, current_time: datetime):
        """匹配限价单"""
        if order.side == OrderSide.BUY:
            # 买入限价单匹配价格不高于限价的卖单
            for ask_level in self.order_book.asks:
                if order.remaining_quantity <= 0 or ask_level.price > order.price:
                    break
                
                fill_quantity = min(order.remaining_quantity, ask_level.quantity)
                
                # 执行交易
                await self._execute_trade(
                    order, ask_level, fill_quantity, ask_level.price, current_time
                )
                
                # 更新订单簿
                ask_level.quantity -= fill_quantity
                order.remaining_quantity -= fill_quantity
                order.filled_quantity += fill_quantity
                
                if ask_level.quantity <= 0:
                    self.order_book.asks.remove(ask_level)
        
        else:  # SELL
            # 卖出限价单匹配价格不低于限价的买单
            for bid_level in self.order_book.bids:
                if order.remaining_quantity <= 0 or bid_level.price < order.price:
                    break
                
                fill_quantity = min(order.remaining_quantity, bid_level.quantity)
                
                # 执行交易
                await self._execute_trade(
                    order, bid_level, fill_quantity, bid_level.price, current_time
                )
                
                # 更新订单簿
                bid_level.quantity -= fill_quantity
                order.remaining_quantity -= fill_quantity
                order.filled_quantity += fill_quantity
                
                if bid_level.quantity <= 0:
                    self.order_book.bids.remove(bid_level)
        
        # 更新订单状态
        if order.remaining_quantity <= 0:
            order.status = OrderStatus.FILLED
        elif order.filled_quantity > 0:
            order.status = OrderStatus.PARTIAL_FILL
    
    async def _execute_trade(self, aggressive_order: Order, 
                           passive_level: OrderBookLevel,
                           quantity: float, price: float, 
                           current_time: datetime):
        """执行交易"""
        # 创建交易记录
        trade = Trade(
            trade_id=f"TRADE_{self.next_trade_id}",
            symbol=self.symbol,
            price=price,
            quantity=quantity,
            timestamp=current_time,
            buyer_order_id=aggressive_order.order_id if aggressive_order.side == OrderSide.BUY else "PASSIVE",
            seller_order_id=aggressive_order.order_id if aggressive_order.side == OrderSide.SELL else "PASSIVE",
            buyer_trader_type=aggressive_order.trader_type if aggressive_order.side == OrderSide.BUY else TraderType.MARKET_MAKER,
            seller_trader_type=aggressive_order.trader_type if aggressive_order.side == OrderSide.SELL else TraderType.MARKET_MAKER,
            is_aggressive=True
        )
        
        self.next_trade_id += 1
        self.trade_history.append(trade)
        
        # 更新市场状态
        self.market_state.last_price = price
        self.market_state.volume += quantity
        
        # 记录价格历史
        self.price_history.append(price)
    
    async def _add_order_to_book(self, order: Order):
        """添加订单到订单簿"""
        if order.side == OrderSide.BUY:
            # 找到合适的位置插入买单
            inserted = False
            for i, bid_level in enumerate(self.order_book.bids):
                if order.price > bid_level.price:
                    # 插入新价格层级
                    new_level = OrderBookLevel(
                        price=order.price,
                        quantity=order.remaining_quantity,
                        order_count=1,
                        orders=[order]
                    )
                    self.order_book.bids.insert(i, new_level)
                    inserted = True
                    break
                elif order.price == bid_level.price:
                    # 加入现有价格层级
                    bid_level.quantity += order.remaining_quantity
                    bid_level.order_count += 1
                    bid_level.orders.append(order)
                    inserted = True
                    break
            
            if not inserted:
                # 添加到末尾
                new_level = OrderBookLevel(
                    price=order.price,
                    quantity=order.remaining_quantity,
                    order_count=1,
                    orders=[order]
                )
                self.order_book.bids.append(new_level)
        
        else:  # SELL
            # 找到合适的位置插入卖单
            inserted = False
            for i, ask_level in enumerate(self.order_book.asks):
                if order.price < ask_level.price:
                    # 插入新价格层级
                    new_level = OrderBookLevel(
                        price=order.price,
                        quantity=order.remaining_quantity,
                        order_count=1,
                        orders=[order]
                    )
                    self.order_book.asks.insert(i, new_level)
                    inserted = True
                    break
                elif order.price == ask_level.price:
                    # 加入现有价格层级
                    ask_level.quantity += order.remaining_quantity
                    ask_level.order_count += 1
                    ask_level.orders.append(order)
                    inserted = True
                    break
            
            if not inserted:
                # 添加到末尾
                new_level = OrderBookLevel(
                    price=order.price,
                    quantity=order.remaining_quantity,
                    order_count=1,
                    orders=[order]
                )
                self.order_book.asks.append(new_level)
    
    async def _handle_order_cancel(self, event: Dict[str, Any], current_time: datetime):
        """处理订单取消"""
        # 简化实现
        pass
    
    async def _handle_order_modify(self, event: Dict[str, Any], current_time: datetime):
        """处理订单修改"""
        # 简化实现
        pass
    
    async def _update_market_state(self, current_time: datetime):
        """更新市场状态"""
        self.market_state.timestamp = current_time
        
        # 计算点差
        spread = self.order_book.get_spread()
        if spread:
            self.spread_history.append(spread)
        
        # 计算订单簿失衡
        total_bid_quantity = sum(level.quantity for level in self.order_book.bids[:5])
        total_ask_quantity = sum(level.quantity for level in self.order_book.asks[:5])
        
        if total_bid_quantity + total_ask_quantity > 0:
            imbalance = (total_bid_quantity - total_ask_quantity) / (total_bid_quantity + total_ask_quantity)
            self.market_state.imbalance = imbalance
        
        # 计算流动性
        if len(self.order_book.bids) > 0 and len(self.order_book.asks) > 0:
            liquidity = min(self.order_book.bids[0].quantity, self.order_book.asks[0].quantity)
            self.market_state.liquidity = liquidity
        
        # 计算波动率（基于最近的价格变化）
        if len(self.price_history) > 10:
            recent_prices = self.price_history[-10:]
            returns = np.diff(np.log(recent_prices))
            volatility = np.std(returns) * np.sqrt(252 * 24 * 3600)  # 年化波动率
            self.market_state.volatility = volatility
    
    async def _record_snapshot(self, current_time: datetime):
        """记录快照"""
        # 深拷贝订单簿
        snapshot = OrderBook(
            symbol=self.symbol,
            bids=[OrderBookLevel(level.price, level.quantity, level.order_count) 
                  for level in self.order_book.bids],
            asks=[OrderBookLevel(level.price, level.quantity, level.order_count) 
                  for level in self.order_book.asks],
            last_update=current_time
        )
        
        self.order_book_snapshots.append(snapshot)
        
        # 记录成交量
        recent_volume = sum(trade.quantity for trade in self.trade_history 
                          if trade.timestamp >= current_time - timedelta(minutes=1))
        self.volume_history.append(recent_volume)
    
    async def _calculate_metrics(self):
        """计算指标"""
        if not self.trade_history:
            return
        
        # 市场质量指标
        prices = [trade.price for trade in self.trade_history]
        spreads = [s for s in self.spread_history if s is not None]
        
        self.market_quality_metrics = {
            'average_spread': np.mean(spreads) if spreads else 0,
            'spread_volatility': np.std(spreads) if spreads else 0,
            'price_volatility': np.std(prices) if prices else 0,
            'trade_count': len(self.trade_history),
            'average_trade_size': np.mean([trade.quantity for trade in self.trade_history]),
            'total_volume': sum(trade.quantity for trade in self.trade_history),
            'price_efficiency': self._calculate_price_efficiency(),
            'market_depth': self._calculate_market_depth(),
            'price_impact': self._calculate_price_impact(),
            'resilience': self._calculate_resilience()
        }
        
        # 微观结构指标
        self.microstructure_metrics = {
            'order_flow_imbalance': self._calculate_order_flow_imbalance(),
            'effective_spread': self._calculate_effective_spread(),
            'realized_spread': self._calculate_realized_spread(),
            'price_improvement': self._calculate_price_improvement(),
            'fill_rate': self._calculate_fill_rate(),
            'adverse_selection': self._calculate_adverse_selection(),
            'inventory_risk': self._calculate_inventory_risk(),
            'tick_size_constraint': self._calculate_tick_size_constraint()
        }
    
    def _calculate_price_efficiency(self) -> float:
        """计算价格效率"""
        if len(self.price_history) < 2:
            return 0
        
        # 基于价格随机游走的偏离度
        returns = np.diff(np.log(self.price_history))
        autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
        
        return 1 - abs(autocorr)
    
    def _calculate_market_depth(self) -> float:
        """计算市场深度"""
        if not self.order_book_snapshots:
            return 0
        
        # 平均市场深度
        depths = []
        for snapshot in self.order_book_snapshots:
            bid_depth = sum(level.quantity for level in snapshot.bids[:5])
            ask_depth = sum(level.quantity for level in snapshot.asks[:5])
            depths.append(bid_depth + ask_depth)
        
        return np.mean(depths) if depths else 0
    
    def _calculate_price_impact(self) -> float:
        """计算价格冲击"""
        if len(self.trade_history) < 2:
            return 0
        
        # 计算大单的价格冲击
        large_trades = [trade for trade in self.trade_history if trade.quantity > 5000]
        impacts = []
        
        for trade in large_trades:
            # 寻找交易前后的价格变化
            before_trades = [t for t in self.trade_history if t.timestamp < trade.timestamp]
            after_trades = [t for t in self.trade_history if t.timestamp > trade.timestamp]
            
            if before_trades and after_trades:
                before_price = before_trades[-1].price
                after_price = after_trades[0].price
                impact = abs(after_price - before_price) / before_price
                impacts.append(impact)
        
        return np.mean(impacts) if impacts else 0
    
    def _calculate_resilience(self) -> float:
        """计算市场弹性"""
        # 简化实现：基于点差的恢复速度
        if len(self.spread_history) < 10:
            return 0
        
        # 计算点差的平均回复时间
        spread_changes = np.diff(self.spread_history)
        resilience = 1 / (1 + np.std(spread_changes))
        
        return resilience
    
    def _calculate_order_flow_imbalance(self) -> float:
        """计算订单流失衡"""
        buy_volume = sum(trade.quantity for trade in self.trade_history 
                        if trade.buyer_trader_type != TraderType.MARKET_MAKER)
        sell_volume = sum(trade.quantity for trade in self.trade_history 
                         if trade.seller_trader_type != TraderType.MARKET_MAKER)
        
        total_volume = buy_volume + sell_volume
        return (buy_volume - sell_volume) / total_volume if total_volume > 0 else 0
    
    def _calculate_effective_spread(self) -> float:
        """计算有效点差"""
        if not self.trade_history:
            return 0
        
        # 简化实现
        return np.mean(self.spread_history) if self.spread_history else 0
    
    def _calculate_realized_spread(self) -> float:
        """计算实现点差"""
        # 简化实现
        return 0
    
    def _calculate_price_improvement(self) -> float:
        """计算价格改善"""
        # 简化实现
        return 0
    
    def _calculate_fill_rate(self) -> float:
        """计算成交率"""
        filled_orders = len([order for order in self.order_history 
                           if order.status == OrderStatus.FILLED])
        total_orders = len(self.order_history)
        
        return filled_orders / total_orders if total_orders > 0 else 0
    
    def _calculate_adverse_selection(self) -> float:
        """计算逆向选择"""
        # 简化实现
        return 0
    
    def _calculate_inventory_risk(self) -> float:
        """计算库存风险"""
        # 简化实现
        return 0
    
    def _calculate_tick_size_constraint(self) -> float:
        """计算tick size约束"""
        # 计算价格改善受tick size限制的比例
        if not self.spread_history:
            return 0
        
        constrained_spreads = len([s for s in self.spread_history if s <= self.tick_size])
        return constrained_spreads / len(self.spread_history)
    
    async def _calculate_trader_performance(self) -> Dict[TraderType, Dict[str, float]]:
        """计算交易员绩效"""
        performance = {}
        
        for trader_type in TraderType:
            trader_orders = [order for order in self.order_history 
                           if order.trader_type == trader_type]
            trader_trades = [trade for trade in self.trade_history 
                           if trade.buyer_trader_type == trader_type or trade.seller_trader_type == trader_type]
            
            if trader_orders:
                performance[trader_type] = {
                    'order_count': len(trader_orders),
                    'trade_count': len(trader_trades),
                    'fill_rate': len([o for o in trader_orders if o.status == OrderStatus.FILLED]) / len(trader_orders),
                    'average_order_size': np.mean([o.quantity for o in trader_orders]),
                    'total_volume': sum(t.quantity for t in trader_trades),
                    'market_share': sum(t.quantity for t in trader_trades) / sum(t.quantity for t in self.trade_history) if self.trade_history else 0
                }
        
        return performance
    
    async def analyze_market_impact(self, order_sizes: List[float]) -> Dict[str, Any]:
        """分析市场冲击"""
        impact_analysis = {}
        
        for size in order_sizes:
            # 模拟不同规模订单的市场冲击
            estimated_impact = self._estimate_market_impact(size)
            impact_analysis[f"size_{size}"] = estimated_impact
        
        return impact_analysis
    
    def _estimate_market_impact(self, order_size: float) -> Dict[str, float]:
        """估算市场冲击"""
        # 基于订单簿深度估算
        if not self.order_book.bids or not self.order_book.asks:
            return {'temporary_impact': 0, 'permanent_impact': 0}
        
        # 临时冲击
        total_depth = sum(level.quantity for level in self.order_book.bids[:5])
        participation_rate = order_size / total_depth if total_depth > 0 else 1
        temporary_impact = 0.001 * np.sqrt(participation_rate)  # 平方根模型
        
        # 永久冲击
        permanent_impact = 0.0001 * participation_rate
        
        return {
            'temporary_impact': temporary_impact,
            'permanent_impact': permanent_impact,
            'total_impact': temporary_impact + permanent_impact
        }
    
    async def generate_simulation_report(self, result: SimulationResult) -> Dict[str, Any]:
        """生成模拟报告"""
        report = {
            'summary': {
                'symbol': result.symbol,
                'duration': str(result.end_time - result.start_time),
                'total_trades': result.total_trades,
                'total_volume': f"{result.total_volume:,.0f}",
                'average_price': f"{np.mean(result.price_history):.2f}" if result.price_history else "N/A",
                'price_volatility': f"{np.std(result.price_history):.4f}" if result.price_history else "N/A"
            },
            'market_quality': result.market_quality_metrics,
            'microstructure': result.microstructure_metrics,
            'trader_performance': result.trader_performance,
            'key_insights': [
                f"平均点差: {result.market_quality_metrics.get('average_spread', 0):.4f}",
                f"价格效率: {result.market_quality_metrics.get('price_efficiency', 0):.2f}",
                f"市场深度: {result.market_quality_metrics.get('market_depth', 0):.0f}",
                f"成交率: {result.microstructure_metrics.get('fill_rate', 0):.2%}"
            ]
        }
        
        return report
    
    async def visualize_simulation(self, result: SimulationResult, save_path: str = None):
        """可视化模拟结果"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 价格走势
        if result.price_history:
            axes[0, 0].plot(result.price_history)
            axes[0, 0].set_title('Price Movement')
            axes[0, 0].set_xlabel('Time Step')
            axes[0, 0].set_ylabel('Price')
        
        # 点差变化
        if result.spread_history:
            axes[0, 1].plot(result.spread_history)
            axes[0, 1].set_title('Spread Evolution')
            axes[0, 1].set_xlabel('Time Step')
            axes[0, 1].set_ylabel('Spread')
        
        # 成交量
        if result.volume_history:
            axes[1, 0].plot(result.volume_history)
            axes[1, 0].set_title('Volume Profile')
            axes[1, 0].set_xlabel('Time Step')
            axes[1, 0].set_ylabel('Volume')
        
        # 交易员市场份额
        if result.trader_performance:
            trader_types = list(result.trader_performance.keys())
            market_shares = [result.trader_performance[t]['market_share'] for t in trader_types]
            
            axes[1, 1].pie(market_shares, labels=[t.value for t in trader_types], autopct='%1.1f%%')
            axes[1, 1].set_title('Market Share by Trader Type')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
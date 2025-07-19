import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import time

from .low_latency_engine import LowLatencyTradingEngine, Order, Fill, OrderStatus, OrderSide, OrderType, VenueType
from .broker_gateway import BrokerGatewayManager, BrokerConfig, BrokerType
from .smart_routing import SmartRoutingEngine, AlgorithmType, AlgorithmParams, MarketData
from .market_impact import MarketImpactMinimizer, ImpactModel
from .dark_pool_connector import DarkPoolManager, DarkPoolConfig, DarkPoolType

class ExecutionStatus(Enum):
    PENDING = "pending"
    ROUTING = "routing"
    EXECUTING = "executing"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"

@dataclass
class ExecutionRequest:
    request_id: str
    symbol: str
    side: OrderSide
    quantity: int
    algorithm_type: AlgorithmType
    algorithm_params: AlgorithmParams
    priority: str = "normal"
    max_duration_minutes: int = 60
    price_limit: Optional[float] = None
    created_time: datetime = field(default_factory=datetime.now)
    status: ExecutionStatus = ExecutionStatus.PENDING
    
@dataclass
class ExecutionReport:
    request_id: str
    symbol: str
    total_quantity: int
    filled_quantity: int
    remaining_quantity: int
    avg_fill_price: float
    total_cost: float
    total_commission: float
    market_impact: float
    implementation_shortfall: float
    execution_time_seconds: float
    fill_rate: float
    venues_used: List[str]
    algorithm_used: str
    child_orders_count: int
    successful_fills: int
    cancelled_orders: int
    rejected_orders: int
    benchmark_performance: Dict[str, float]
    start_time: datetime
    end_time: datetime
    
class ExecutionManagementSystem:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 核心组件
        self.trading_engine = LowLatencyTradingEngine(config.get('trading_engine', {}))
        self.broker_manager = BrokerGatewayManager()
        self.smart_router = SmartRoutingEngine()
        self.impact_minimizer = MarketImpactMinimizer()
        self.dark_pool_manager = DarkPoolManager()
        
        # 执行管理
        self.execution_requests: Dict[str, ExecutionRequest] = {}
        self.execution_reports: Dict[str, ExecutionReport] = {}
        self.active_executions: Dict[str, Dict[str, Any]] = {}
        
        # 性能监控
        self.performance_metrics: Dict[str, Any] = {}
        
        # 事件回调
        self.event_callbacks: Dict[str, List[Callable]] = {
            'order_fill': [],
            'execution_complete': [],
            'execution_failed': [],
            'risk_alert': []
        }
        
        self.is_running = False
        
    async def initialize(self):
        """初始化执行管理系统"""
        try:
            self.logger.info("Initializing Execution Management System...")
            
            # 初始化交易引擎
            self.trading_engine.start()
            
            # 初始化券商连接
            await self._initialize_brokers()
            
            # 初始化暗池连接
            await self._initialize_dark_pools()
            
            # 校准市场影响模型
            await self._calibrate_impact_model()
            
            # 启动监控任务
            asyncio.create_task(self._execution_monitor())
            asyncio.create_task(self._performance_monitor())
            
            self.is_running = True
            self.logger.info("Execution Management System initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Execution Management System: {e}")
            raise
    
    async def shutdown(self):
        """关闭执行管理系统"""
        try:
            self.logger.info("Shutting down Execution Management System...")
            
            self.is_running = False
            
            # 停止所有活跃执行
            for request_id in list(self.active_executions.keys()):
                await self.cancel_execution(request_id)
            
            # 关闭连接
            await self.broker_manager.disconnect_all()
            await self.dark_pool_manager.disconnect_all_pools()
            
            # 停止交易引擎
            self.trading_engine.stop()
            
            self.logger.info("Execution Management System shut down successfully")
            
        except Exception as e:
            self.logger.error(f"Error shutting down Execution Management System: {e}")
    
    async def submit_execution_request(self, request: ExecutionRequest) -> str:
        """提交执行请求"""
        try:
            # 验证请求
            if not self._validate_execution_request(request):
                raise ValueError("Invalid execution request")
            
            # 记录请求
            self.execution_requests[request.request_id] = request
            
            # 创建执行报告
            report = ExecutionReport(
                request_id=request.request_id,
                symbol=request.symbol,
                total_quantity=request.quantity,
                filled_quantity=0,
                remaining_quantity=request.quantity,
                avg_fill_price=0.0,
                total_cost=0.0,
                total_commission=0.0,
                market_impact=0.0,
                implementation_shortfall=0.0,
                execution_time_seconds=0.0,
                fill_rate=0.0,
                venues_used=[],
                algorithm_used=request.algorithm_type.value,
                child_orders_count=0,
                successful_fills=0,
                cancelled_orders=0,
                rejected_orders=0,
                benchmark_performance={},
                start_time=datetime.now(),
                end_time=datetime.now()
            )
            
            self.execution_reports[request.request_id] = report
            
            # 启动执行
            asyncio.create_task(self._execute_request(request))
            
            self.logger.info(f"Submitted execution request {request.request_id} for {request.quantity} shares of {request.symbol}")
            
            return request.request_id
            
        except Exception as e:
            self.logger.error(f"Error submitting execution request: {e}")
            raise
    
    async def cancel_execution(self, request_id: str) -> bool:
        """取消执行请求"""
        try:
            if request_id not in self.execution_requests:
                return False
            
            request = self.execution_requests[request_id]
            
            if request.status in [ExecutionStatus.COMPLETED, ExecutionStatus.CANCELLED, ExecutionStatus.FAILED]:
                return False
            
            # 取消所有子订单
            execution_data = self.active_executions.get(request_id, {})
            child_orders = execution_data.get('child_orders', [])
            
            for child_order in child_orders:
                if child_order.status in [OrderStatus.WORKING, OrderStatus.PENDING]:
                    await self.trading_engine.cancel_order(child_order.child_order_id)
            
            # 更新状态
            request.status = ExecutionStatus.CANCELLED
            
            # 更新报告
            report = self.execution_reports.get(request_id)
            if report:
                report.end_time = datetime.now()
                report.execution_time_seconds = (report.end_time - report.start_time).total_seconds()
            
            # 清理活跃执行
            if request_id in self.active_executions:
                del self.active_executions[request_id]
            
            self.logger.info(f"Cancelled execution request {request_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error cancelling execution request {request_id}: {e}")
            return False
    
    async def get_execution_status(self, request_id: str) -> Optional[ExecutionReport]:
        """获取执行状态"""
        return self.execution_reports.get(request_id)
    
    async def get_all_executions(self) -> List[ExecutionReport]:
        """获取所有执行报告"""
        return list(self.execution_reports.values())
    
    async def get_active_executions(self) -> List[str]:
        """获取活跃执行请求ID"""
        return list(self.active_executions.keys())
    
    def register_event_callback(self, event_type: str, callback: Callable):
        """注册事件回调"""
        if event_type in self.event_callbacks:
            self.event_callbacks[event_type].append(callback)
    
    async def _execute_request(self, request: ExecutionRequest):
        """执行请求"""
        try:
            request.status = ExecutionStatus.ROUTING
            
            # 获取市场数据
            market_data = await self._get_market_data(request.symbol)
            
            if not market_data:
                request.status = ExecutionStatus.FAILED
                return
            
            # 创建智能路由器
            router = self.smart_router.create_router(request.algorithm_type, request.algorithm_params)
            
            # 生成子订单
            child_orders = await self.smart_router.execute_algorithm(router, market_data)
            
            if not child_orders:
                request.status = ExecutionStatus.FAILED
                return
            
            # 记录执行数据
            execution_data = {
                'router': router,
                'child_orders': child_orders,
                'market_data': market_data,
                'start_time': datetime.now(),
                'venues_used': set(),
                'fills': []
            }
            
            self.active_executions[request.request_id] = execution_data
            
            # 更新报告
            report = self.execution_reports[request.request_id]
            report.child_orders_count = len(child_orders)
            
            request.status = ExecutionStatus.EXECUTING
            
            # 执行子订单
            await self._execute_child_orders(request.request_id, child_orders)
            
        except Exception as e:
            self.logger.error(f"Error executing request {request.request_id}: {e}")
            request.status = ExecutionStatus.FAILED
            
            # 触发失败回调
            for callback in self.event_callbacks['execution_failed']:
                try:
                    await callback(request.request_id, str(e))
                except Exception:
                    pass
    
    async def _execute_child_orders(self, request_id: str, child_orders: List):
        """执行子订单"""
        try:
            execution_data = self.active_executions[request_id]
            
            for child_order in child_orders:
                # 选择最优执行场所
                venue = await self._select_optimal_venue(child_order)
                
                if venue:
                    # 创建订单
                    order = Order(
                        order_id=child_order.child_order_id,
                        symbol=child_order.symbol,
                        side=child_order.side,
                        quantity=child_order.quantity,
                        order_type=child_order.order_type,
                        price=child_order.price,
                        venue=venue,
                        parent_order_id=request_id
                    )
                    
                    # 提交订单
                    success = await self.trading_engine.submit_order(order)
                    
                    if success:
                        execution_data['venues_used'].add(venue)
                        
                        # 模拟成交
                        await self._simulate_fill(order, request_id)
                    
                    # 控制执行节奏
                    await asyncio.sleep(0.1)
            
        except Exception as e:
            self.logger.error(f"Error executing child orders for request {request_id}: {e}")
    
    async def _simulate_fill(self, order: Order, request_id: str):
        """模拟成交"""
        try:
            # 模拟成交延迟
            await asyncio.sleep(0.01)
            
            # 创建成交记录
            fill = Fill(
                fill_id=f"FILL_{int(time.time() * 1000000)}",
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=order.price or 150.0,
                venue=order.venue,
                timestamp=datetime.now(),
                execution_id=f"EXEC_{int(time.time() * 1000000)}",
                commission=order.quantity * (order.price or 150.0) * 0.001
            )
            
            # 处理成交
            await self._process_fill(fill, request_id)
            
        except Exception as e:
            self.logger.error(f"Error simulating fill for order {order.order_id}: {e}")
    
    async def _process_fill(self, fill: Fill, request_id: str):
        """处理成交"""
        try:
            execution_data = self.active_executions.get(request_id)
            if not execution_data:
                return
            
            # 记录成交
            execution_data['fills'].append(fill)
            
            # 更新报告
            report = self.execution_reports[request_id]
            report.filled_quantity += fill.quantity
            report.remaining_quantity = report.total_quantity - report.filled_quantity
            report.total_cost += fill.quantity * fill.price
            report.total_commission += fill.commission
            report.successful_fills += 1
            
            # 计算平均成交价
            if report.filled_quantity > 0:
                report.avg_fill_price = report.total_cost / report.filled_quantity
            
            # 计算成交率
            report.fill_rate = report.filled_quantity / report.total_quantity
            
            # 更新使用的场所
            if fill.venue not in report.venues_used:
                report.venues_used.append(fill.venue)
            
            # 触发成交回调
            for callback in self.event_callbacks['order_fill']:
                try:
                    await callback(request_id, fill)
                except Exception:
                    pass
            
            # 检查是否完成
            if report.filled_quantity >= report.total_quantity:
                await self._complete_execution(request_id)
            
        except Exception as e:
            self.logger.error(f"Error processing fill {fill.fill_id}: {e}")
    
    async def _complete_execution(self, request_id: str):
        """完成执行"""
        try:
            request = self.execution_requests[request_id]
            request.status = ExecutionStatus.COMPLETED
            
            # 更新报告
            report = self.execution_reports[request_id]
            report.end_time = datetime.now()
            report.execution_time_seconds = (report.end_time - report.start_time).total_seconds()
            
            # 计算性能指标
            await self._calculate_performance_metrics(request_id)
            
            # 清理活跃执行
            if request_id in self.active_executions:
                del self.active_executions[request_id]
            
            # 触发完成回调
            for callback in self.event_callbacks['execution_complete']:
                try:
                    await callback(request_id, report)
                except Exception:
                    pass
            
            self.logger.info(f"Completed execution request {request_id}")
            
        except Exception as e:
            self.logger.error(f"Error completing execution {request_id}: {e}")
    
    async def _calculate_performance_metrics(self, request_id: str):
        """计算性能指标"""
        try:
            report = self.execution_reports[request_id]
            execution_data = self.active_executions.get(request_id, {})
            
            # 计算市场影响
            market_data = execution_data.get('market_data')
            if market_data:
                impact_prediction = self.impact_minimizer.predict_impact(
                    report.total_quantity, market_data, report.execution_time_seconds / 3600
                )
                
                report.market_impact = impact_prediction.total_impact
                report.implementation_shortfall = impact_prediction.implementation_shortfall
            
            # 计算基准性能
            benchmark_vwap = market_data.vwap if market_data else report.avg_fill_price
            
            report.benchmark_performance = {
                'vs_vwap': (report.avg_fill_price - benchmark_vwap) / benchmark_vwap * 100,
                'vs_arrival_price': (report.avg_fill_price - market_data.last_price) / market_data.last_price * 100 if market_data else 0,
                'slippage_bps': (report.avg_fill_price - market_data.last_price) / market_data.last_price * 10000 if market_data else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics for {request_id}: {e}")
    
    async def _select_optimal_venue(self, child_order) -> Optional[str]:
        """选择最优执行场所"""
        try:
            # 根据订单特征选择场所
            if child_order.quantity > 5000:
                # 大单优先选择暗池
                dark_pool = await self.dark_pool_manager.get_best_dark_pool(
                    child_order.symbol, child_order.quantity, child_order.side
                )
                if dark_pool:
                    return dark_pool
            
            # 选择最优券商
            connected_brokers = self.broker_manager.get_connected_gateways()
            if connected_brokers:
                return connected_brokers[0]  # 简单选择第一个
            
            return "NASDAQ"  # 默认场所
            
        except Exception as e:
            self.logger.error(f"Error selecting optimal venue: {e}")
            return "NASDAQ"
    
    async def _get_market_data(self, symbol: str) -> Optional[MarketData]:
        """获取市场数据"""
        try:
            # 模拟市场数据
            return MarketData(
                symbol=symbol,
                timestamp=datetime.now(),
                bid_price=149.99,
                ask_price=150.01,
                bid_size=1000,
                ask_size=1000,
                last_price=150.0,
                volume=100000,
                vwap=149.95,
                market_cap=2500000000000,
                adv=1000000
            )
            
        except Exception as e:
            self.logger.error(f"Error getting market data for {symbol}: {e}")
            return None
    
    async def _initialize_brokers(self):
        """初始化券商连接"""
        try:
            # 从配置加载券商
            brokers_config = self.config.get('brokers', {})
            
            for broker_name, config in brokers_config.items():
                broker_config = BrokerConfig(
                    name=broker_name,
                    broker_type=BrokerType(config.get('type', 'native')),
                    host=config.get('host', 'localhost'),
                    port=config.get('port', 8080),
                    api_key=config.get('api_key', ''),
                    secret_key=config.get('secret_key', ''),
                    username=config.get('username', ''),
                    password=config.get('password', ''),
                    venue_type=VenueType(config.get('venue_type', 'broker'))
                )
                
                gateway = self.broker_manager.create_gateway(broker_config)
                self.broker_manager.add_gateway(gateway)
            
            # 连接所有券商
            results = await self.broker_manager.connect_all()
            
            connected_count = sum(1 for success in results.values() if success)
            self.logger.info(f"Connected to {connected_count}/{len(results)} brokers")
            
        except Exception as e:
            self.logger.error(f"Error initializing brokers: {e}")
    
    async def _initialize_dark_pools(self):
        """初始化暗池连接"""
        try:
            # 从配置加载暗池
            dark_pools_config = self.config.get('dark_pools', {})
            
            for pool_name, config in dark_pools_config.items():
                dark_pool_config = DarkPoolConfig(
                    name=pool_name,
                    dark_pool_type=DarkPoolType(config.get('type', 'institutional')),
                    min_order_size=config.get('min_order_size', 100),
                    max_order_size=config.get('max_order_size', 100000),
                    matching_algorithm=config.get('matching_algorithm', 'pro_rata'),
                    crossing_times=config.get('crossing_times', ['12:00', '16:00']),
                    commission_rate=config.get('commission_rate', 0.001),
                    access_fee=config.get('access_fee', 0.0001),
                    participation_threshold=config.get('participation_threshold', 0.05),
                    liquidity_indication=config.get('liquidity_indication', True),
                    pre_trade_transparency=config.get('pre_trade_transparency', False),
                    post_trade_transparency=config.get('post_trade_transparency', True),
                    supported_order_types=config.get('supported_order_types', ['limit', 'market']),
                    api_endpoint=config.get('api_endpoint', ''),
                    api_key=config.get('api_key', '')
                )
                
                self.dark_pool_manager.add_dark_pool(dark_pool_config)
            
            # 连接所有暗池
            results = await self.dark_pool_manager.connect_all_pools()
            
            connected_count = sum(1 for success in results.values() if success)
            self.logger.info(f"Connected to {connected_count}/{len(results)} dark pools")
            
        except Exception as e:
            self.logger.error(f"Error initializing dark pools: {e}")
    
    async def _calibrate_impact_model(self):
        """校准市场影响模型"""
        try:
            # 模拟历史数据
            historical_data = [
                {'impact': 0.001, 'volume': 100000, 'spread': 0.01},
                {'impact': 0.002, 'volume': 200000, 'spread': 0.02},
                {'impact': 0.003, 'volume': 300000, 'spread': 0.015}
            ]
            
            self.impact_minimizer.calibrate_model(historical_data)
            
        except Exception as e:
            self.logger.error(f"Error calibrating impact model: {e}")
    
    async def _execution_monitor(self):
        """执行监控"""
        while self.is_running:
            try:
                # 检查超时的执行
                current_time = datetime.now()
                
                for request_id, request in self.execution_requests.items():
                    if (request.status == ExecutionStatus.EXECUTING and 
                        (current_time - request.created_time).total_seconds() > 
                        request.max_duration_minutes * 60):
                        
                        self.logger.warning(f"Execution {request_id} timed out")
                        await self.cancel_execution(request_id)
                
                await asyncio.sleep(30)  # 每30秒检查一次
                
            except Exception as e:
                self.logger.error(f"Error in execution monitor: {e}")
                await asyncio.sleep(60)
    
    async def _performance_monitor(self):
        """性能监控"""
        while self.is_running:
            try:
                # 收集性能指标
                self.performance_metrics = {
                    'total_executions': len(self.execution_requests),
                    'active_executions': len(self.active_executions),
                    'completed_executions': len([r for r in self.execution_requests.values() 
                                                if r.status == ExecutionStatus.COMPLETED]),
                    'cancelled_executions': len([r for r in self.execution_requests.values() 
                                                if r.status == ExecutionStatus.CANCELLED]),
                    'failed_executions': len([r for r in self.execution_requests.values() 
                                            if r.status == ExecutionStatus.FAILED]),
                    'avg_execution_time': self._calculate_avg_execution_time(),
                    'avg_fill_rate': self._calculate_avg_fill_rate(),
                    'trading_engine_stats': self.trading_engine.get_engine_stats(),
                    'broker_health': await self.broker_manager.get_all_health_status(),
                    'dark_pool_status': self.dark_pool_manager.get_all_pool_status()
                }
                
                await asyncio.sleep(60)  # 每分钟更新一次
                
            except Exception as e:
                self.logger.error(f"Error in performance monitor: {e}")
                await asyncio.sleep(60)
    
    def _calculate_avg_execution_time(self) -> float:
        """计算平均执行时间"""
        completed_reports = [r for r in self.execution_reports.values() 
                           if r.execution_time_seconds > 0]
        
        if not completed_reports:
            return 0
        
        return sum(r.execution_time_seconds for r in completed_reports) / len(completed_reports)
    
    def _calculate_avg_fill_rate(self) -> float:
        """计算平均成交率"""
        completed_reports = [r for r in self.execution_reports.values() 
                           if r.fill_rate > 0]
        
        if not completed_reports:
            return 0
        
        return sum(r.fill_rate for r in completed_reports) / len(completed_reports)
    
    def _validate_execution_request(self, request: ExecutionRequest) -> bool:
        """验证执行请求"""
        if request.quantity <= 0:
            return False
        
        if request.max_duration_minutes <= 0:
            return False
        
        if request.price_limit is not None and request.price_limit <= 0:
            return False
        
        return True
    
    def get_system_health(self) -> Dict[str, Any]:
        """获取系统健康状态"""
        return {
            'is_running': self.is_running,
            'performance_metrics': self.performance_metrics,
            'component_health': {
                'trading_engine': self.trading_engine.is_running,
                'connected_brokers': len(self.broker_manager.get_connected_gateways()),
                'connected_dark_pools': len(self.dark_pool_manager.get_connected_pools())
            }
        }
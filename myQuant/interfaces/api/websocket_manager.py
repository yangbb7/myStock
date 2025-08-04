# -*- coding: utf-8 -*-
"""
WebSocket Manager - 实时数据推送管理器
提供WebSocket连接管理和实时数据推送功能
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, asdict

import socketio
from fastapi import FastAPI
from socketio import AsyncServer

from myQuant.core.enhanced_trading_system import EnhancedTradingSystem
from myQuant.infrastructure.data.providers.real_data_provider import RealDataProvider


@dataclass
class MarketDataMessage:
    """市场数据消息"""
    symbol: str
    name: str  # 添加股票名称
    price: float
    volume: int
    timestamp: str
    change: Optional[float] = None
    change_percent: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None


@dataclass
class SystemStatusMessage:
    """系统状态消息"""
    status: str
    uptime: float
    memory_usage: float
    cpu_usage: float
    active_strategies: int
    total_orders: int
    portfolio_value: float
    timestamp: str


@dataclass
class OrderUpdateMessage:
    """订单更新消息"""
    order_id: str
    symbol: str
    side: str
    quantity: int
    price: float
    status: str
    filled_quantity: int
    remaining_quantity: int
    timestamp: str


@dataclass
class RiskAlertMessage:
    """风险告警消息"""
    alert_type: str
    severity: str
    message: str
    symbol: Optional[str] = None
    current_value: Optional[float] = None
    threshold: Optional[float] = None
    timestamp: str = None


class WebSocketManager:
    """WebSocket连接管理器"""
    
    def __init__(self, trading_system: EnhancedTradingSystem):
        self.trading_system = trading_system
        self.logger = logging.getLogger(__name__)
        
        # 创建Socket.IO服务器
        self.sio = AsyncServer(
            cors_allowed_origins="*",
            logger=False,
            engineio_logger=False,
            async_mode='asgi'
        )
        
        # 连接管理
        self.connections: Set[str] = set()
        self.subscriptions: Dict[str, Dict[str, Set[str]]] = {
            'market_data': {},  # symbol -> set of session_ids
            'system_status': set(),  # set of session_ids
            'order_updates': set(),  # set of session_ids
            'risk_alerts': set(),   # set of session_ids
        }
        
        # 客户端管理（用于测试兼容性）
        self.clients: Dict[str, Any] = {}  # client_id -> websocket
        self.client_subscriptions: Dict[str, Dict[str, List[str]]] = {}  # client_id -> {type: [symbols]}
        
        # 数据缓存
        self.market_data_cache: Dict[str, MarketDataMessage] = {}
        
        # 初始化真实数据提供者
        self.real_data_provider = self._init_real_data_provider()
        self.system_status_cache: Optional[SystemStatusMessage] = None
        
        # 定时任务
        self.background_tasks: List[asyncio.Task] = []

    def _init_real_data_provider(self) -> Optional[RealDataProvider]:
        """初始化真实数据提供者"""
        try:
            # 数据源配置
            config = {
                "primary_provider": os.getenv('PRIMARY_DATA_PROVIDER', 'yahoo'),
                "fallback_providers": os.getenv('FALLBACK_DATA_PROVIDERS', 'eastmoney').split(','),
                
                # Tushare配置
                "tushare": {
                    "enabled": bool(os.getenv('TUSHARE_TOKEN')),
                    "token": os.getenv('TUSHARE_TOKEN')
                },
                
                # Yahoo Finance配置
                "yahoo": {
                    "enabled": os.getenv('YAHOO_FINANCE_ENABLED', 'true').lower() == 'true'
                },
                
                # 东方财富配置
                "eastmoney": {
                    "enabled": os.getenv('EASTMONEY_ENABLED', 'true').lower() == 'true'
                }
            }
            
            # 如果有Tushare token，优先使用Tushare
            if os.getenv('TUSHARE_TOKEN'):
                config["primary_provider"] = "tushare"
                config["fallback_providers"] = ["yahoo", "eastmoney"]
            
            provider = RealDataProvider(config)
            self.logger.info("真实数据提供者初始化成功")
            return provider
            
        except Exception as e:
            self.logger.error(f"真实数据提供者初始化失败: {e}")
            return None
        
        # 注册事件处理器
        self._register_event_handlers()
    
    def _register_event_handlers(self):
        """注册Socket.IO事件处理器"""
        
        @self.sio.event
        async def connect(sid, environ):
            """客户端连接"""
            self.connections.add(sid)
            self.logger.info(f"Client connected: {sid}")
            
            # 发送当前系统状态
            if self.system_status_cache:
                await self.sio.emit('system_status', {
                    'type': 'system_status',
                    'data': asdict(self.system_status_cache)
                }, room=sid)
        
        @self.sio.event
        async def disconnect(sid):
            """客户端断开连接"""
            self.connections.discard(sid)
            
            # 清理订阅
            for subscription_type, subscriptions in self.subscriptions.items():
                if subscription_type == 'market_data':
                    for symbol, sids in subscriptions.items():
                        sids.discard(sid)
                else:
                    subscriptions.discard(sid)
            
            self.logger.info(f"Client disconnected: {sid}")
        
        @self.sio.event
        async def subscribe(sid, data):
            """订阅数据"""
            try:
                subscription_type = data.get('type')
                options = data.get('options', {})
                self.logger.info(f"Subscribe request from {sid}: type={subscription_type}, data={data}")
                
                if subscription_type == 'market_data':
                    symbols = options.get('symbols', [])
                    for symbol in symbols:
                        if symbol not in self.subscriptions['market_data']:
                            self.subscriptions['market_data'][symbol] = set()
                        self.subscriptions['market_data'][symbol].add(sid)
                        
                        # 发送缓存的数据
                        if symbol in self.market_data_cache:
                            await self.sio.emit('market_data', {
                                'type': 'market_data',
                                'data': asdict(self.market_data_cache[symbol])
                            }, room=sid)
                
                elif subscription_type == 'system_status':
                    self.subscriptions['system_status'].add(sid)
                    
                    # 发送当前状态
                    if self.system_status_cache:
                        await self.sio.emit('system_status', {
                            'type': 'system_status',
                            'data': asdict(self.system_status_cache)
                        }, room=sid)
                
                elif subscription_type == 'order_update':
                    self.subscriptions['order_updates'].add(sid)
                
                elif subscription_type == 'risk_alert':
                    self.subscriptions['risk_alerts'].add(sid)
                
                self.logger.info(f"Client {sid} subscribed to {subscription_type}")
                
            except Exception as e:
                self.logger.error(f"Subscription error: {e}")
                await self.sio.emit('error', {
                    'message': f'Subscription failed: {str(e)}'
                }, room=sid)
        
        @self.sio.event
        async def unsubscribe(sid, data):
            """取消订阅"""
            try:
                subscription_type = data.get('type')
                
                if subscription_type == 'market_data':
                    for symbol, sids in self.subscriptions['market_data'].items():
                        sids.discard(sid)
                elif subscription_type in self.subscriptions:
                    self.subscriptions[subscription_type].discard(sid)
                
                self.logger.info(f"Client {sid} unsubscribed from {subscription_type}")
                
            except Exception as e:
                self.logger.error(f"Unsubscription error: {e}")
    
    async def start_background_tasks(self):
        """启动后台任务"""
        self.logger.info("Starting WebSocket background tasks...")
        
        # 系统状态推送任务
        self.background_tasks.append(
            asyncio.create_task(self._system_status_task())
        )
        self.logger.info("System status task started")
        
        # 市场数据模拟任务（用于演示）
        self.background_tasks.append(
            asyncio.create_task(self._market_data_simulation_task())
        )
        self.logger.info("Market data simulation task started")
    
    async def stop_background_tasks(self):
        """停止后台任务"""
        for task in self.background_tasks:
            task.cancel()
        
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        self.background_tasks.clear()
    
    async def _system_status_task(self):
        """系统状态推送任务"""
        while True:
            try:
                if self.subscriptions['system_status']:
                    # 获取系统状态
                    status = self.trading_system.get_system_status()
                    health = self.trading_system.get_system_health()
                    
                    status_message = SystemStatusMessage(
                        status=status.get('status', 'unknown'),
                        uptime=status.get('uptime', 0),
                        memory_usage=health.get('memory_usage', 0),
                        cpu_usage=health.get('cpu_usage', 0),
                        active_strategies=status.get('strategies_count', 0),
                        total_orders=status.get('orders_count', 0),
                        portfolio_value=status.get('portfolio_value', 0),
                        timestamp=datetime.now().isoformat()
                    )
                    
                    self.system_status_cache = status_message
                    
                    # 推送给订阅的客户端
                    await self.sio.emit('system_status', {
                        'type': 'system_status',
                        'data': asdict(status_message)
                    }, room=list(self.subscriptions['system_status']))
                
                await asyncio.sleep(5)  # 每5秒推送一次
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"System status task error: {e}")
                await asyncio.sleep(5)
    
    async def _market_data_simulation_task(self):
        """实时市场数据推送任务（支持真实数据）"""
        self.logger.info("Market data simulation task started running")
        
        # 监控股票列表
        symbols = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH', '000858.SZ']
        previous_prices = {}  # 存储上次价格用于计算涨跌
        
        while True:
            try:
                # 获取所有监控股票的当前数据
                subscribed_symbols = []
                for symbol in symbols:
                    if symbol in self.subscriptions['market_data'] and self.subscriptions['market_data'][symbol]:
                        subscribed_symbols.append(symbol)
                
                if not subscribed_symbols:
                    self.logger.debug(f"No subscribed symbols, waiting... subscriptions={self.subscriptions['market_data']}")
                    await asyncio.sleep(5)
                    continue
                
                self.logger.info(f"Processing market data for symbols: {subscribed_symbols}")
                
                # 从真实数据源获取数据
                realtime_data = {}
                if self.real_data_provider:
                    try:
                        realtime_data = self.real_data_provider.get_realtime_data(subscribed_symbols)
                        self.logger.info(f"从真实数据源获取到{len(realtime_data)}个股票数据")
                    except Exception as e:
                        self.logger.error(f"真实数据源获取失败: {e}")
                        # 不使用模拟数据，跳过本次推送
                        await asyncio.sleep(5)
                        continue
                else:
                    self.logger.error("真实数据提供者未初始化，无法获取实时数据")
                    await asyncio.sleep(5)
                    continue
                
                # 如果没有真实数据，跳过本次推送
                if not realtime_data:
                    self.logger.warning("未获取到真实数据，跳过本次推送")
                    await asyncio.sleep(5)
                    continue
                
                for symbol in subscribed_symbols:
                    if symbol in realtime_data:
                        data = realtime_data[symbol]
                        current_price = data['current_price']
                        
                        # 计算涨跌
                        if symbol in previous_prices:
                            previous_price = previous_prices[symbol]
                            change = current_price - previous_price
                            change_percent = (change / previous_price * 100) if previous_price > 0 else 0
                        else:
                            # 首次获取数据，假设涨跌为0
                            change = 0
                            change_percent = 0
                        
                        # 更新上次价格
                        previous_prices[symbol] = current_price
                        
                        # 获取股票名称
                        stock_name = symbol  # 默认使用符号
                        if hasattr(self.trading_system, 'modules') and 'data' in self.trading_system.modules:
                            try:
                                stock_info = await self.trading_system.modules['data'].get_stock_info(symbol)
                                stock_name = stock_info.get('name', symbol) if stock_info else symbol
                            except Exception:
                                pass  # 如果获取失败，使用默认值
                        
                        market_data = MarketDataMessage(
                            symbol=symbol,
                            name=stock_name,
                            price=round(current_price, 2),
                            volume=data.get('volume', 0),  # 使用真实成交量数据
                            timestamp=data.get('timestamp', datetime.now()).isoformat() if hasattr(data.get('timestamp', datetime.now()), 'isoformat') else str(data.get('timestamp', datetime.now())),
                            change=round(change, 2),
                            change_percent=round(change_percent, 2),
                            bid=data.get('bid', current_price * 0.999),  # 优先使用真实买一价
                            ask=data.get('ask', current_price * 1.001),  # 优先使用真实卖一价
                            bid_size=data.get('bid_size', 0),  # 使用真实买一量
                            ask_size=data.get('ask_size', 0)   # 使用真实卖一量
                        )
                        
                        self.market_data_cache[symbol] = market_data
                        
                        # 推送给订阅该股票的客户端
                        await self.sio.emit('market_data', {
                            'type': 'market_data',
                            'data': asdict(market_data)
                        }, room=list(self.subscriptions['market_data'][symbol]))
                
                await asyncio.sleep(3)  # 每3秒更新一次真实数据
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Market data simulation error: {e}")
                await asyncio.sleep(3)
    
    
    async def broadcast_order_update(self, order_data: Dict[str, Any]):
        """广播订单更新"""
        if self.subscriptions['order_updates']:
            order_message = OrderUpdateMessage(
                order_id=order_data.get('order_id', ''),
                symbol=order_data.get('symbol', ''),
                side=order_data.get('side', ''),
                quantity=order_data.get('quantity', 0),
                price=order_data.get('price', 0),
                status=order_data.get('status', ''),
                filled_quantity=order_data.get('filled_quantity', 0),
                remaining_quantity=order_data.get('remaining_quantity', 0),
                timestamp=datetime.now().isoformat()
            )
            
            await self.sio.emit('order_update', {
                'type': 'order_update',
                'data': asdict(order_message)
            }, room=list(self.subscriptions['order_updates']))
    
    async def broadcast_risk_alert(self, alert_data: Dict[str, Any]):
        """广播风险告警"""
        if self.subscriptions['risk_alerts']:
            risk_message = RiskAlertMessage(
                alert_type=alert_data.get('alert_type', ''),
                severity=alert_data.get('severity', 'info'),
                message=alert_data.get('message', ''),
                symbol=alert_data.get('symbol'),
                current_value=alert_data.get('current_value'),
                threshold=alert_data.get('threshold'),
                timestamp=datetime.now().isoformat()
            )
            
            await self.sio.emit('risk_alert', {
                'type': 'risk_alert',
                'data': asdict(risk_message)
            }, room=list(self.subscriptions['risk_alerts']))
    
    async def broadcast_market_data(self, market_data: Dict[str, Any]):
        """广播市场数据"""
        symbol = market_data.get('symbol')
        if symbol and symbol in self.subscriptions['market_data']:
            if self.subscriptions['market_data'][symbol]:
                # 获取股票名称
                stock_name = symbol  # 默认使用符号
                if hasattr(self.trading_system, 'modules') and 'data' in self.trading_system.modules:
                    try:
                        stock_info = await self.trading_system.modules['data'].get_stock_info(symbol)
                        stock_name = stock_info.get('name', symbol) if stock_info else symbol
                    except Exception:
                        pass  # 如果获取失败，使用默认值
                
                message = MarketDataMessage(
                    symbol=symbol,
                    name=stock_name,
                    price=market_data.get('price', 0),
                    volume=market_data.get('volume', 0),
                    timestamp=market_data.get('timestamp', datetime.now().isoformat()),
                    change=market_data.get('change'),
                    change_percent=market_data.get('change_percent'),
                    bid=market_data.get('bid'),
                    ask=market_data.get('ask'),
                    bid_size=market_data.get('bid_size'),
                    ask_size=market_data.get('ask_size')
                )
                
                self.market_data_cache[symbol] = message
                
                await self.sio.emit('market_data', {
                    'type': 'market_data',
                    'data': asdict(message)
                }, room=list(self.subscriptions['market_data'][symbol]))
    
    def get_stats(self) -> Dict[str, Any]:
        """获取WebSocket统计信息"""
        return {
            'total_connections': len(self.connections),
            'market_data_subscriptions': {
                symbol: len(sids) for symbol, sids in self.subscriptions['market_data'].items()
            },
            'system_status_subscriptions': len(self.subscriptions['system_status']),
            'order_update_subscriptions': len(self.subscriptions['order_updates']),
            'risk_alert_subscriptions': len(self.subscriptions['risk_alerts']),
            'cached_symbols': list(self.market_data_cache.keys())
        }
    
    def mount_to_app(self, app: FastAPI):
        """将WebSocket服务挂载到FastAPI应用"""
        # 创建Socket.IO应用
        socket_app = socketio.ASGIApp(self.sio, other_asgi_app=app)
        return socket_app
    
    # Test compatibility methods
    async def add_client(self, websocket) -> str:
        """添加客户端（测试兼容性方法）"""
        client_id = str(uuid.uuid4())
        self.clients[client_id] = websocket
        self.client_subscriptions[client_id] = {}
        return client_id
    
    async def remove_client(self, client_id: str):
        """移除客户端"""
        if client_id in self.clients:
            del self.clients[client_id]
        if client_id in self.client_subscriptions:
            del self.client_subscriptions[client_id]
    
    def get_connected_clients(self) -> List[str]:
        """获取已连接的客户端列表"""
        return list(self.clients.keys())
    
    async def subscribe_client(self, client_id: str, subscription_type: str, symbols: List[str]):
        """为客户端添加订阅"""
        if client_id not in self.client_subscriptions:
            self.client_subscriptions[client_id] = {}
        
        if subscription_type not in self.client_subscriptions[client_id]:
            self.client_subscriptions[client_id][subscription_type] = []
        
        self.client_subscriptions[client_id][subscription_type].extend(symbols)
    
    def get_client_subscriptions(self, client_id: str) -> Dict[str, List[str]]:
        """获取客户端订阅"""
        return self.client_subscriptions.get(client_id, {})
    
    async def broadcast_to_channel(self, channel: str, message: Dict[str, Any]):
        """广播消息到频道"""
        # 这里可以实现向订阅了特定频道的客户端发送消息
        for client_id, websocket in self.clients.items():
            subscriptions = self.client_subscriptions.get(client_id, {})
            if channel in subscriptions:
                # 模拟发送消息
                if hasattr(websocket, 'send_text'):
                    await websocket.send_text(json.dumps(message))
    
    async def authenticate_client(self, websocket, auth_message: Dict[str, Any]) -> bool:
        """客户端认证（测试兼容性方法）"""
        try:
            from myQuant.core.auth.jwt_manager import verify_token
            token = auth_message.get('token')
            if not token:
                return False
            
            user_data = verify_token(token)
            return user_data is not None
        except Exception:
            # 如果认证模块不可用，使用简单的token验证
            return auth_message.get('token') == "valid_jwt_token"
    
    async def handle_client_message(self, client_id: str, message: Dict[str, Any]):
        """处理客户端消息"""
        message_type = message.get('type')
        
        if message_type == 'subscribe':
            channel = message.get('channel')
            symbols = message.get('symbols', [])
            if channel and symbols:
                await self.subscribe_client(client_id, channel, symbols)
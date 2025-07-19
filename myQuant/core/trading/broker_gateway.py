import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import socket
import ssl
import json
import struct
from abc import ABC, abstractmethod

from .low_latency_engine import Order, Fill, OrderStatus, OrderSide, VenueType

class BrokerType(Enum):
    FIX = "fix"
    NATIVE = "native"
    REST = "rest"
    WEBSOCKET = "websocket"

@dataclass
class BrokerConfig:
    name: str
    broker_type: BrokerType
    host: str
    port: int
    api_key: str
    secret_key: str
    username: str
    password: str
    venue_type: VenueType
    ssl_enabled: bool = True
    heartbeat_interval: int = 30
    max_reconnect_attempts: int = 5
    reconnect_delay: int = 5
    timeout: int = 10

class BrokerGateway(ABC):
    def __init__(self, config: BrokerConfig):
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{config.name}")
        self.is_connected = False
        self.session = None
        self.connection_stats = {
            'connect_time': None,
            'last_heartbeat': None,
            'messages_sent': 0,
            'messages_received': 0,
            'errors': 0,
            'reconnect_attempts': 0
        }
        
    @abstractmethod
    async def connect(self) -> bool:
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        pass
    
    @abstractmethod
    async def send_order(self, order: Order) -> bool:
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        pass
    
    @abstractmethod
    async def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        pass
    
    @abstractmethod
    async def get_positions(self) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    async def get_account_info(self) -> Dict[str, Any]:
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        return {
            'broker_name': self.config.name,
            'is_connected': self.is_connected,
            'connection_stats': self.connection_stats
        }

class FIXBrokerGateway(BrokerGateway):
    def __init__(self, config: BrokerConfig):
        super().__init__(config)
        self.socket = None
        self.seq_num = 1
        self.target_comp_id = config.username
        self.sender_comp_id = "MYQUANT"
        
    async def connect(self) -> bool:
        try:
            self.logger.info(f"Connecting to FIX broker {self.config.name}...")
            
            # 创建socket连接
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            
            if self.config.ssl_enabled:
                context = ssl.create_default_context()
                self.socket = context.wrap_socket(self.socket, server_hostname=self.config.host)
            
            self.socket.settimeout(self.config.timeout)
            await asyncio.get_event_loop().run_in_executor(
                None, self.socket.connect, (self.config.host, self.config.port)
            )
            
            # 发送登录消息
            login_msg = self._create_logon_message()
            await self._send_message(login_msg)
            
            # 启动心跳
            asyncio.create_task(self._heartbeat_loop())
            
            self.is_connected = True
            self.connection_stats['connect_time'] = datetime.now()
            self.logger.info(f"Connected to FIX broker {self.config.name}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to FIX broker {self.config.name}: {e}")
            self.connection_stats['errors'] += 1
            return False
    
    async def disconnect(self) -> bool:
        try:
            if self.socket:
                # 发送登出消息
                logout_msg = self._create_logout_message()
                await self._send_message(logout_msg)
                
                self.socket.close()
                self.socket = None
            
            self.is_connected = False
            self.logger.info(f"Disconnected from FIX broker {self.config.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error disconnecting from FIX broker {self.config.name}: {e}")
            return False
    
    async def send_order(self, order: Order) -> bool:
        if not self.is_connected:
            return False
        
        try:
            fix_message = self._create_new_order_message(order)
            await self._send_message(fix_message)
            self.logger.debug(f"Sent order {order.order_id} to FIX broker")
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending order {order.order_id}: {e}")
            return False
    
    async def cancel_order(self, order_id: str) -> bool:
        if not self.is_connected:
            return False
        
        try:
            cancel_msg = self._create_cancel_message(order_id)
            await self._send_message(cancel_msg)
            self.logger.debug(f"Sent cancel request for order {order_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    async def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        # FIX协议通过执行报告获取订单状态
        # 这里返回模拟状态
        return {
            'order_id': order_id,
            'status': 'working',
            'filled_quantity': 0,
            'leaves_quantity': 100
        }
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        # 模拟仓位信息
        return [
            {'symbol': 'AAPL', 'quantity': 1000, 'avg_price': 150.0},
            {'symbol': 'MSFT', 'quantity': 500, 'avg_price': 300.0}
        ]
    
    async def get_account_info(self) -> Dict[str, Any]:
        # 模拟账户信息
        return {
            'account_id': self.config.username,
            'buying_power': 1000000.0,
            'cash_balance': 500000.0,
            'total_equity': 1500000.0
        }
    
    def _create_logon_message(self) -> str:
        """创建FIX登录消息"""
        fields = [
            f"8=FIX.4.4",
            f"35=A",
            f"34={self.seq_num}",
            f"49={self.sender_comp_id}",
            f"56={self.target_comp_id}",
            f"52={datetime.now().strftime('%Y%m%d-%H:%M:%S')}",
            f"98=0",
            f"108=30",
            f"553={self.config.username}",
            f"554={self.config.password}"
        ]
        
        body = "".join(f"{field}\x01" for field in fields[3:])
        body_length = len(body)
        
        header = f"8=FIX.4.4\x019={body_length}\x01"
        message = header + body
        
        # 计算校验和
        checksum = sum(ord(c) for c in message) % 256
        message += f"10={checksum:03d}\x01"
        
        self.seq_num += 1
        return message
    
    def _create_logout_message(self) -> str:
        """创建FIX登出消息"""
        fields = [
            f"8=FIX.4.4",
            f"35=5",
            f"34={self.seq_num}",
            f"49={self.sender_comp_id}",
            f"56={self.target_comp_id}",
            f"52={datetime.now().strftime('%Y%m%d-%H:%M:%S')}"
        ]
        
        body = "".join(f"{field}\x01" for field in fields[3:])
        body_length = len(body)
        
        header = f"8=FIX.4.4\x019={body_length}\x01"
        message = header + body
        
        checksum = sum(ord(c) for c in message) % 256
        message += f"10={checksum:03d}\x01"
        
        self.seq_num += 1
        return message
    
    def _create_new_order_message(self, order: Order) -> str:
        """创建新订单FIX消息"""
        side = "1" if order.side == OrderSide.BUY else "2"
        ord_type = "1" if order.order_type.value == "market" else "2"
        
        fields = [
            f"8=FIX.4.4",
            f"35=D",
            f"34={self.seq_num}",
            f"49={self.sender_comp_id}",
            f"56={self.target_comp_id}",
            f"52={datetime.now().strftime('%Y%m%d-%H:%M:%S')}",
            f"11={order.client_order_id or order.order_id}",
            f"55={order.symbol}",
            f"54={side}",
            f"38={order.quantity}",
            f"40={ord_type}",
            f"59={order.time_in_force}"
        ]
        
        if order.price:
            fields.append(f"44={order.price}")
        
        body = "".join(f"{field}\x01" for field in fields[3:])
        body_length = len(body)
        
        header = f"8=FIX.4.4\x019={body_length}\x01"
        message = header + body
        
        checksum = sum(ord(c) for c in message) % 256
        message += f"10={checksum:03d}\x01"
        
        self.seq_num += 1
        return message
    
    def _create_cancel_message(self, order_id: str) -> str:
        """创建取消订单FIX消息"""
        fields = [
            f"8=FIX.4.4",
            f"35=F",
            f"34={self.seq_num}",
            f"49={self.sender_comp_id}",
            f"56={self.target_comp_id}",
            f"52={datetime.now().strftime('%Y%m%d-%H:%M:%S')}",
            f"11={order_id}_CANCEL",
            f"41={order_id}"
        ]
        
        body = "".join(f"{field}\x01" for field in fields[3:])
        body_length = len(body)
        
        header = f"8=FIX.4.4\x019={body_length}\x01"
        message = header + body
        
        checksum = sum(ord(c) for c in message) % 256
        message += f"10={checksum:03d}\x01"
        
        self.seq_num += 1
        return message
    
    async def _send_message(self, message: str):
        """发送FIX消息"""
        if not self.socket:
            raise Exception("Socket not connected")
        
        await asyncio.get_event_loop().run_in_executor(
            None, self.socket.send, message.encode('utf-8')
        )
        
        self.connection_stats['messages_sent'] += 1
    
    async def _heartbeat_loop(self):
        """心跳循环"""
        while self.is_connected:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)
                
                if self.is_connected:
                    heartbeat_msg = self._create_heartbeat_message()
                    await self._send_message(heartbeat_msg)
                    self.connection_stats['last_heartbeat'] = datetime.now()
                    
            except Exception as e:
                self.logger.error(f"Error in heartbeat loop: {e}")
                break
    
    def _create_heartbeat_message(self) -> str:
        """创建心跳消息"""
        fields = [
            f"8=FIX.4.4",
            f"35=0",
            f"34={self.seq_num}",
            f"49={self.sender_comp_id}",
            f"56={self.target_comp_id}",
            f"52={datetime.now().strftime('%Y%m%d-%H:%M:%S')}"
        ]
        
        body = "".join(f"{field}\x01" for field in fields[3:])
        body_length = len(body)
        
        header = f"8=FIX.4.4\x019={body_length}\x01"
        message = header + body
        
        checksum = sum(ord(c) for c in message) % 256
        message += f"10={checksum:03d}\x01"
        
        self.seq_num += 1
        return message

class NativeBrokerGateway(BrokerGateway):
    """原生券商API网关"""
    
    async def connect(self) -> bool:
        try:
            self.logger.info(f"Connecting to native broker {self.config.name}...")
            
            # 模拟原生API连接
            await asyncio.sleep(0.1)
            
            self.is_connected = True
            self.connection_stats['connect_time'] = datetime.now()
            self.logger.info(f"Connected to native broker {self.config.name}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to native broker {self.config.name}: {e}")
            return False
    
    async def disconnect(self) -> bool:
        self.is_connected = False
        self.logger.info(f"Disconnected from native broker {self.config.name}")
        return True
    
    async def send_order(self, order: Order) -> bool:
        if not self.is_connected:
            return False
        
        try:
            # 模拟原生API订单提交
            await asyncio.sleep(0.001)  # 1ms延迟
            
            self.connection_stats['messages_sent'] += 1
            self.logger.debug(f"Sent order {order.order_id} to native broker")
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending order {order.order_id}: {e}")
            return False
    
    async def cancel_order(self, order_id: str) -> bool:
        if not self.is_connected:
            return False
        
        try:
            # 模拟原生API取消订单
            await asyncio.sleep(0.001)
            
            self.connection_stats['messages_sent'] += 1
            self.logger.debug(f"Sent cancel request for order {order_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    async def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        return {
            'order_id': order_id,
            'status': 'working',
            'filled_quantity': 0,
            'leaves_quantity': 100
        }
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        return [
            {'symbol': 'AAPL', 'quantity': 1000, 'avg_price': 150.0},
            {'symbol': 'MSFT', 'quantity': 500, 'avg_price': 300.0}
        ]
    
    async def get_account_info(self) -> Dict[str, Any]:
        return {
            'account_id': self.config.username,
            'buying_power': 1000000.0,
            'cash_balance': 500000.0,
            'total_equity': 1500000.0
        }

class BrokerGatewayManager:
    def __init__(self):
        self.gateways: Dict[str, BrokerGateway] = {}
        self.logger = logging.getLogger(__name__)
    
    def add_gateway(self, gateway: BrokerGateway):
        """添加券商网关"""
        self.gateways[gateway.config.name] = gateway
        self.logger.info(f"Added broker gateway: {gateway.config.name}")
    
    async def connect_all(self) -> Dict[str, bool]:
        """连接所有券商"""
        results = {}
        
        for name, gateway in self.gateways.items():
            try:
                success = await gateway.connect()
                results[name] = success
            except Exception as e:
                self.logger.error(f"Failed to connect to {name}: {e}")
                results[name] = False
        
        return results
    
    async def disconnect_all(self) -> Dict[str, bool]:
        """断开所有券商连接"""
        results = {}
        
        for name, gateway in self.gateways.items():
            try:
                success = await gateway.disconnect()
                results[name] = success
            except Exception as e:
                self.logger.error(f"Failed to disconnect from {name}: {e}")
                results[name] = False
        
        return results
    
    async def send_order_to_venue(self, order: Order, venue_name: str) -> bool:
        """发送订单到指定券商"""
        gateway = self.gateways.get(venue_name)
        if not gateway:
            self.logger.error(f"Unknown venue: {venue_name}")
            return False
        
        return await gateway.send_order(order)
    
    async def get_all_health_status(self) -> Dict[str, Dict[str, Any]]:
        """获取所有券商健康状态"""
        status = {}
        
        for name, gateway in self.gateways.items():
            try:
                status[name] = await gateway.health_check()
            except Exception as e:
                self.logger.error(f"Failed to get health status for {name}: {e}")
                status[name] = {'error': str(e)}
        
        return status
    
    def get_connected_gateways(self) -> List[str]:
        """获取已连接的券商列表"""
        return [name for name, gateway in self.gateways.items() if gateway.is_connected]
    
    def create_gateway(self, config: BrokerConfig) -> BrokerGateway:
        """创建券商网关"""
        if config.broker_type == BrokerType.FIX:
            return FIXBrokerGateway(config)
        elif config.broker_type == BrokerType.NATIVE:
            return NativeBrokerGateway(config)
        else:
            raise ValueError(f"Unsupported broker type: {config.broker_type}")
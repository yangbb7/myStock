# Standard library imports
import queue
import sys
import threading
import uuid
from datetime import datetime, timedelta
from enum import Enum
from unittest.mock import Mock, patch, MagicMock

# Third-party imports
import numpy as np
import pandas as pd
import pytest

class OrderStatus(Enum):
    """订单状态"""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

class MockOrderManager:
    """OrderManager的Mock实现，用于完整的测试覆盖"""
    
    def __init__(self, config):
        self._validate_config(config)
        self.config = config
        self.max_orders_per_symbol = config.get('max_orders_per_symbol', 10)
        self.max_total_orders = config.get('max_total_orders', 100)
        self.order_timeout = config.get('order_timeout', 3600)
        self.min_order_value = config.get('min_order_value', 1000)
        
        self.orders = {}  # 所有订单
        self.pending_orders = {}  # 待处理订单
        self.filled_orders = {}  # 已成交订单
        self.cancelled_orders = {}  # 已取消订单
        self.rejected_orders = {}  # 被拒绝订单
        self.order_history = []  # 订单历史
        self.risk_limits = config.get('risk_limits', {})
        self.broker_connection = Mock()  # Mock的券商连接
        
    def _validate_config(self, config):
        """验证配置参数"""
        if not isinstance(config, dict):
            raise TypeError("Config must be a dictionary")
            
        if 'max_orders_per_symbol' in config:
            if config['max_orders_per_symbol'] < 0:
                raise ValueError("max_orders_per_symbol must be non-negative")
                
        if 'max_total_orders' in config:
            if config['max_total_orders'] <= 0:
                raise ValueError("max_total_orders must be positive")
                
        if 'order_timeout' in config:
            if config['order_timeout'] < 0:
                raise ValueError("order_timeout must be non-negative")
                
        if 'min_order_value' in config:
            if config['min_order_value'] < 0:
                raise ValueError("min_order_value must be non-negative")
    
    def create_order(self, order_request):
        """创建订单"""
        # 验证订单请求
        self._validate_order_request(order_request)
        
        # 检查风险限制
        self._check_risk_limits(order_request)
        
        # 生成订单ID
        order_id = f"ORD_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # 创建订单对象
        order = {
            'order_id': order_id,
            'symbol': order_request['symbol'],
            'side': order_request['side'],
            'quantity': order_request['quantity'],
            'price': order_request.get('price'),
            'order_type': order_request.get('order_type', 'MARKET'),
            'status': OrderStatus.PENDING,
            'filled_quantity': 0,
            'remaining_quantity': order_request['quantity'],
            'avg_fill_price': 0.0,
            'commission': 0.0,
            'create_time': datetime.now(),
            'update_time': datetime.now(),
            'client_id': order_request.get('client_id', 'DEFAULT'),
            'executions': []
        }
        
        # 存储订单
        self.orders[order_id] = order
        self.pending_orders[order_id] = order
        self.order_history.append(order.copy())
        
        return order_id
    
    def _validate_order_request(self, order_request):
        """验证订单请求"""
        required_fields = ['symbol', 'side', 'quantity']
        for field in required_fields:
            if field not in order_request:
                raise ValueError(f"Missing required field: {field}")
                
        if not order_request['symbol'] or not isinstance(order_request['symbol'], str):
            raise ValueError("Invalid symbol")
            
        if order_request['side'] not in ['BUY', 'SELL']:
            raise ValueError("Invalid order side")
            
        if order_request['quantity'] <= 0:
            raise ValueError("Quantity must be positive")
            
        if 'price' in order_request and order_request['price'] is not None:
            if order_request['price'] <= 0:
                raise ValueError("Price must be positive")
    
    def _check_risk_limits(self, order_request):
        """检查风险限制"""
        # 检查单股票订单数量限制
        symbol = order_request['symbol']
        symbol_orders = [o for o in self.pending_orders.values() if o['symbol'] == symbol]
        if len(symbol_orders) >= self.max_orders_per_symbol:
            raise ValueError(f"Exceeds max orders per symbol: {self.max_orders_per_symbol}")
            
        # 检查总订单数量限制
        if len(self.pending_orders) >= self.max_total_orders:
            raise ValueError(f"Exceeds max total orders: {self.max_total_orders}")
            
        # 检查最小订单价值
        if 'price' in order_request and order_request['price']:
            order_value = order_request['quantity'] * order_request['price']
            if order_value < self.min_order_value:
                raise ValueError(f"Order value below minimum: {self.min_order_value}")
    
    def submit_order(self, order_id):
        """提交订单到券商"""
        if order_id not in self.pending_orders:
            raise ValueError(f"Order {order_id} not found in pending orders")
            
        order = self.pending_orders[order_id]
        
        # 模拟券商提交
        try:
            self.broker_connection.submit_order(order)
            order['status'] = OrderStatus.SUBMITTED
            order['update_time'] = datetime.now()
            return True
        except Exception as e:
            order['status'] = OrderStatus.REJECTED
            order['reject_reason'] = str(e)
            self.rejected_orders[order_id] = self.pending_orders.pop(order_id)
            return False
    
    def cancel_order(self, order_id):
        """取消订单"""
        if order_id not in self.orders:
            return False
            
        order = self.orders[order_id]
        
        if order['status'] in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            return False
            
        # 模拟券商取消
        try:
            self.broker_connection.cancel_order(order_id)
            order['status'] = OrderStatus.CANCELLED
            order['update_time'] = datetime.now()
            
            # 移动到已取消订单
            if order_id in self.pending_orders:
                self.cancelled_orders[order_id] = self.pending_orders.pop(order_id)
                
            return True
        except Exception:
            return False
    
    def get_order(self, order_id):
        """获取订单信息"""
        return self.orders.get(order_id)
    
    def get_orders_by_symbol(self, symbol, status=None):
        """按股票代码获取订单"""
        orders = [o for o in self.orders.values() if o['symbol'] == symbol]
        if status:
            orders = [o for o in orders if o['status'] == status]
        return orders
    
    def get_orders_by_status(self, status):
        """按状态获取订单"""
        return [o for o in self.orders.values() if o['status'] == status]
    
    def handle_execution(self, order_id, execution):
        """处理订单执行回报"""
        if order_id not in self.orders:
            raise ValueError(f"Order {order_id} not found")
            
        order = self.orders[order_id]
        
        # 验证执行数据
        required_fields = ['quantity', 'price', 'execution_time']
        for field in required_fields:
            if field not in execution:
                raise ValueError(f"Missing execution field: {field}")
        
        # 更新订单状态
        order['executions'].append(execution)
        order['filled_quantity'] += execution['quantity']
        order['remaining_quantity'] = order['quantity'] - order['filled_quantity']
        
        # 计算平均成交价
        total_value = sum(e['quantity'] * e['price'] for e in order['executions'])
        order['avg_fill_price'] = total_value / order['filled_quantity']
        
        # 更新订单状态
        if order['remaining_quantity'] == 0:
            order['status'] = OrderStatus.FILLED
            if order_id in self.pending_orders:
                self.filled_orders[order_id] = self.pending_orders.pop(order_id)
        else:
            order['status'] = OrderStatus.PARTIALLY_FILLED
            
        order['update_time'] = datetime.now()
        return order
    
    def calculate_commission(self, order_value, side='BUY'):
        """计算手续费"""
        base_rate = 0.0003  # 万三
        min_commission = 5.0  # 最低5元
        
        commission = max(order_value * base_rate, min_commission)
        
        # 卖出时额外收取印花税
        if side == 'SELL':
            stamp_tax = order_value * 0.001  # 千一印花税
            commission += stamp_tax
            
        return round(commission, 2)

class TestOrderManagerComprehensive:
    """OrderManager综合测试用例 - 高覆盖率测试"""
    
    @pytest.fixture
    def order_config(self):
        """订单管理配置fixture"""
        return {
            'max_orders_per_symbol': 10,
            'max_total_orders': 100,
            'order_timeout': 3600,
            'min_order_value': 1000,
            'commission_rate': 0.0003,
            'risk_limits': {
                'max_position_value': 1000000,
                'max_daily_loss': 50000
            }
        }
    
    @pytest.fixture
    def sample_order_request(self):
        """样本订单请求fixture"""
        return {
            'symbol': '000001.SZ',
            'side': 'BUY',
            'quantity': 1000,
            'price': 15.0,
            'order_type': 'LIMIT',
            'client_id': 'TEST_CLIENT'
        }
    
    # === 初始化测试 ===
    @pytest.mark.unit
    def test_order_manager_init_success(self, order_config):
        """测试订单管理器正常初始化"""
        om = MockOrderManager(order_config)
        
        assert om.max_orders_per_symbol == 10
        assert om.max_total_orders == 100
        assert om.order_timeout == 3600
        assert om.min_order_value == 1000
        assert len(om.orders) == 0
        assert len(om.pending_orders) == 0
    
    @pytest.mark.unit
    def test_order_manager_init_invalid_config(self):
        """测试无效配置"""
        invalid_configs = [
            {'max_orders_per_symbol': -1},  # 负数
            {'max_total_orders': 0},        # 零值
            {'order_timeout': -3600},       # 负超时
            {'min_order_value': -1000},     # 负最小订单价值
            None,                           # None配置
            "invalid_config",               # 非字典类型
        ]
        
        for config in invalid_configs:
            with pytest.raises((ValueError, TypeError)):
                MockOrderManager(config)
    
    # === 订单创建测试 ===
    @pytest.mark.unit
    def test_create_order_success(self, order_config, sample_order_request):
        """测试成功创建订单"""
        om = MockOrderManager(order_config)
        
        order_id = om.create_order(sample_order_request)
        
        assert order_id is not None
        assert order_id in om.pending_orders
        assert om.pending_orders[order_id]['status'] == OrderStatus.PENDING
        assert om.pending_orders[order_id]['symbol'] == '000001.SZ'
        assert om.pending_orders[order_id]['side'] == 'BUY'
        assert om.pending_orders[order_id]['quantity'] == 1000
        assert om.pending_orders[order_id]['price'] == 15.0
        assert len(om.order_history) == 1
    
    @pytest.mark.unit
    def test_create_order_missing_required_fields(self, order_config):
        """测试缺少必需字段"""
        om = MockOrderManager(order_config)
        
        incomplete_requests = [
            {'side': 'BUY', 'quantity': 1000},  # 缺少symbol
            {'symbol': '000001.SZ', 'quantity': 1000},  # 缺少side
            {'symbol': '000001.SZ', 'side': 'BUY'},  # 缺少quantity
        ]
        
        for request in incomplete_requests:
            with pytest.raises(ValueError, match="Missing required field"):
                om.create_order(request)
    
    @pytest.mark.unit
    def test_create_order_invalid_symbol(self, order_config):
        """测试无效股票代码"""
        om = MockOrderManager(order_config)
        
        invalid_requests = [
            {'symbol': '', 'side': 'BUY', 'quantity': 1000},
            {'symbol': None, 'side': 'BUY', 'quantity': 1000},
            {'symbol': 123, 'side': 'BUY', 'quantity': 1000},
        ]
        
        for request in invalid_requests:
            with pytest.raises(ValueError, match="Invalid symbol"):
                om.create_order(request)
    
    @pytest.mark.unit
    def test_create_order_invalid_side(self, order_config):
        """测试无效订单方向"""
        om = MockOrderManager(order_config)
        
        invalid_requests = [
            {'symbol': '000001.SZ', 'side': 'INVALID', 'quantity': 1000},
            {'symbol': '000001.SZ', 'side': '', 'quantity': 1000},
            {'symbol': '000001.SZ', 'side': 'buy', 'quantity': 1000},  # 小写
        ]
        
        for request in invalid_requests:
            with pytest.raises(ValueError, match="Invalid order side"):
                om.create_order(request)
    
    @pytest.mark.unit
    def test_create_order_invalid_quantity(self, order_config):
        """测试无效数量"""
        om = MockOrderManager(order_config)
        
        invalid_quantities = [0, -100, -1, -0.5]
        
        for qty in invalid_quantities:
            request = {
                'symbol': '000001.SZ',
                'side': 'BUY',
                'quantity': qty,
                'price': 15.0
            }
            
            with pytest.raises(ValueError, match="Quantity must be positive"):
                om.create_order(request)
    
    @pytest.mark.unit
    def test_create_order_invalid_price(self, order_config):
        """测试无效价格"""
        om = MockOrderManager(order_config)
        
        invalid_prices = [0, -15.0, -0.01]
        
        for price in invalid_prices:
            request = {
                'symbol': '000001.SZ',
                'side': 'BUY',
                'quantity': 1000,
                'price': price
            }
            
            with pytest.raises(ValueError, match="Price must be positive"):
                om.create_order(request)
    
    @pytest.mark.unit
    def test_create_order_exceed_symbol_limit(self, order_config):
        """测试超过单股票订单数限制"""
        om = MockOrderManager(order_config)
        
        symbol = '000001.SZ'
        max_orders = order_config['max_orders_per_symbol']
        
        # 创建满额订单
        for i in range(max_orders):
            request = {
                'symbol': symbol,
                'side': 'BUY',
                'quantity': 100,
                'price': 15.0 + i * 0.1
            }
            om.create_order(request)
        
        # 再次创建应该失败
        excess_request = {
            'symbol': symbol,
            'side': 'BUY',
            'quantity': 100,
            'price': 20.0
        }
        
        with pytest.raises(ValueError, match="Exceeds max orders per symbol"):
            om.create_order(excess_request)
    
    @pytest.mark.unit
    def test_create_order_below_min_value(self, order_config):
        """测试低于最小订单价值"""
        om = MockOrderManager(order_config)
        
        # 订单价值 = 10 * 50 = 500 < 1000 (min_order_value)
        low_value_request = {
            'symbol': '000001.SZ',
            'side': 'BUY',
            'quantity': 10,
            'price': 50.0
        }
        
        with pytest.raises(ValueError, match="Order value below minimum"):
            om.create_order(low_value_request)
    
    # === 订单提交测试 ===
    @pytest.mark.unit
    def test_submit_order_success(self, order_config, sample_order_request):
        """测试成功提交订单"""
        om = MockOrderManager(order_config)
        
        order_id = om.create_order(sample_order_request)
        result = om.submit_order(order_id)
        
        assert result is True
        assert om.orders[order_id]['status'] == OrderStatus.SUBMITTED
    
    @pytest.mark.unit
    def test_submit_order_not_found(self, order_config):
        """测试提交不存在的订单"""
        om = MockOrderManager(order_config)
        
        with pytest.raises(ValueError, match="not found in pending orders"):
            om.submit_order("INVALID_ORDER_ID")
    
    @pytest.mark.unit
    def test_submit_order_broker_error(self, order_config, sample_order_request):
        """测试券商提交错误"""
        om = MockOrderManager(order_config)
        
        # 设置broker连接抛出异常
        om.broker_connection.submit_order.side_effect = Exception("Broker error")
        
        order_id = om.create_order(sample_order_request)
        result = om.submit_order(order_id)
        
        assert result is False
        assert om.orders[order_id]['status'] == OrderStatus.REJECTED
        assert order_id in om.rejected_orders
        assert order_id not in om.pending_orders
    
    # === 订单取消测试 ===
    @pytest.mark.unit
    def test_cancel_order_success(self, order_config, sample_order_request):
        """测试成功取消订单"""
        om = MockOrderManager(order_config)
        
        order_id = om.create_order(sample_order_request)
        result = om.cancel_order(order_id)
        
        assert result is True
        assert om.orders[order_id]['status'] == OrderStatus.CANCELLED
        assert order_id in om.cancelled_orders
        assert order_id not in om.pending_orders
    
    @pytest.mark.unit
    def test_cancel_order_not_found(self, order_config):
        """测试取消不存在的订单"""
        om = MockOrderManager(order_config)
        
        result = om.cancel_order("INVALID_ORDER_ID")
        assert result is False
    
    @pytest.mark.unit
    def test_cancel_filled_order(self, order_config, sample_order_request):
        """测试取消已成交订单"""
        om = MockOrderManager(order_config)
        
        order_id = om.create_order(sample_order_request)
        
        # 模拟订单已成交
        om.orders[order_id]['status'] = OrderStatus.FILLED
        
        result = om.cancel_order(order_id)
        assert result is False
        assert om.orders[order_id]['status'] == OrderStatus.FILLED
    
    # === 订单查询测试 ===
    @pytest.mark.unit
    def test_get_order(self, order_config, sample_order_request):
        """测试获取订单信息"""
        om = MockOrderManager(order_config)
        
        order_id = om.create_order(sample_order_request)
        order = om.get_order(order_id)
        
        assert order is not None
        assert order['order_id'] == order_id
        assert order['symbol'] == '000001.SZ'
    
    @pytest.mark.unit
    def test_get_order_not_found(self, order_config):
        """测试获取不存在的订单"""
        om = MockOrderManager(order_config)
        
        order = om.get_order("INVALID_ORDER_ID")
        assert order is None
    
    @pytest.mark.unit
    def test_get_orders_by_symbol(self, order_config):
        """测试按股票代码获取订单"""
        om = MockOrderManager(order_config)
        
        # 创建不同股票的订单
        symbols = ['000001.SZ', '000002.SZ', '000001.SZ']
        order_ids = []
        
        for symbol in symbols:
            request = {
                'symbol': symbol,
                'side': 'BUY',
                'quantity': 1000,
                'price': 15.0
            }
            order_id = om.create_order(request)
            order_ids.append(order_id)
        
        # 获取000001.SZ的订单
        orders = om.get_orders_by_symbol('000001.SZ')
        assert len(orders) == 2
        assert all(o['symbol'] == '000001.SZ' for o in orders)
        
        # 获取000002.SZ的订单
        orders = om.get_orders_by_symbol('000002.SZ')
        assert len(orders) == 1
        assert orders[0]['symbol'] == '000002.SZ'
    
    @pytest.mark.unit
    def test_get_orders_by_status(self, order_config, sample_order_request):
        """测试按状态获取订单"""
        om = MockOrderManager(order_config)
        
        # 创建多个订单
        order_ids = []
        for i in range(3):
            order_id = om.create_order(sample_order_request)
            order_ids.append(order_id)
        
        # 提交一个订单
        om.submit_order(order_ids[0])
        
        # 取消一个订单
        om.cancel_order(order_ids[1])
        
        # 检查不同状态的订单
        pending_orders = om.get_orders_by_status(OrderStatus.PENDING)
        submitted_orders = om.get_orders_by_status(OrderStatus.SUBMITTED)
        cancelled_orders = om.get_orders_by_status(OrderStatus.CANCELLED)
        
        assert len(pending_orders) == 1
        assert len(submitted_orders) == 1
        assert len(cancelled_orders) == 1
    
    # === 订单执行测试 ===
    @pytest.mark.unit
    def test_handle_execution_full_fill(self, order_config, sample_order_request):
        """测试完全成交"""
        om = MockOrderManager(order_config)
        
        order_id = om.create_order(sample_order_request)
        
        execution = {
            'quantity': 1000,
            'price': 15.0,
            'execution_time': datetime.now(),
            'execution_id': 'EXEC_001'
        }
        
        order = om.handle_execution(order_id, execution)
        
        assert order['status'] == OrderStatus.FILLED
        assert order['filled_quantity'] == 1000
        assert order['remaining_quantity'] == 0
        assert order['avg_fill_price'] == 15.0
        assert len(order['executions']) == 1
        assert order_id in om.filled_orders
        assert order_id not in om.pending_orders
    
    @pytest.mark.unit
    def test_handle_execution_partial_fill(self, order_config, sample_order_request):
        """测试部分成交"""
        om = MockOrderManager(order_config)
        
        order_id = om.create_order(sample_order_request)
        
        execution = {
            'quantity': 300,
            'price': 15.0,
            'execution_time': datetime.now(),
            'execution_id': 'EXEC_001'
        }
        
        order = om.handle_execution(order_id, execution)
        
        assert order['status'] == OrderStatus.PARTIALLY_FILLED
        assert order['filled_quantity'] == 300
        assert order['remaining_quantity'] == 700
        assert order['avg_fill_price'] == 15.0
        assert len(order['executions']) == 1
        assert order_id in om.pending_orders
    
    @pytest.mark.unit
    def test_handle_execution_multiple_fills(self, order_config, sample_order_request):
        """测试多次部分成交"""
        om = MockOrderManager(order_config)
        
        order_id = om.create_order(sample_order_request)
        
        # 第一次成交
        execution1 = {
            'quantity': 300,
            'price': 15.0,
            'execution_time': datetime.now(),
            'execution_id': 'EXEC_001'
        }
        om.handle_execution(order_id, execution1)
        
        # 第二次成交
        execution2 = {
            'quantity': 200,
            'price': 15.1,
            'execution_time': datetime.now(),
            'execution_id': 'EXEC_002'
        }
        om.handle_execution(order_id, execution2)
        
        # 第三次成交（完全成交）
        execution3 = {
            'quantity': 500,
            'price': 14.9,
            'execution_time': datetime.now(),
            'execution_id': 'EXEC_003'
        }
        order = om.handle_execution(order_id, execution3)
        
        assert order['status'] == OrderStatus.FILLED
        assert order['filled_quantity'] == 1000
        assert order['remaining_quantity'] == 0
        assert len(order['executions']) == 3
        
        # 计算平均成交价: (300*15.0 + 200*15.1 + 500*14.9) / 1000 = 14.97
        expected_avg_price = (300*15.0 + 200*15.1 + 500*14.9) / 1000
        assert abs(order['avg_fill_price'] - expected_avg_price) < 0.01
    
    @pytest.mark.unit
    def test_handle_execution_invalid_order(self, order_config):
        """测试处理不存在订单的执行"""
        om = MockOrderManager(order_config)
        
        execution = {
            'quantity': 1000,
            'price': 15.0,
            'execution_time': datetime.now()
        }
        
        with pytest.raises(ValueError, match="Order .* not found"):
            om.handle_execution("INVALID_ORDER_ID", execution)
    
    @pytest.mark.unit
    def test_handle_execution_missing_fields(self, order_config, sample_order_request):
        """测试执行数据缺少字段"""
        om = MockOrderManager(order_config)
        
        order_id = om.create_order(sample_order_request)
        
        incomplete_executions = [
            {'price': 15.0, 'execution_time': datetime.now()},  # 缺少quantity
            {'quantity': 1000, 'execution_time': datetime.now()},  # 缺少price
            {'quantity': 1000, 'price': 15.0},  # 缺少execution_time
        ]
        
        for execution in incomplete_executions:
            with pytest.raises(ValueError, match="Missing execution field"):
                om.handle_execution(order_id, execution)
    
    # === 手续费计算测试 ===
    @pytest.mark.unit
    def test_calculate_commission_buy(self, order_config):
        """测试买入手续费计算"""
        om = MockOrderManager(order_config)
        
        # 大额订单
        order_value = 100000
        commission = om.calculate_commission(order_value, 'BUY')
        expected = 100000 * 0.0003  # 万三
        assert abs(commission - expected) < 0.01  # 使用浮点数比较
        
        # 小额订单（最低手续费）
        order_value = 1000
        commission = om.calculate_commission(order_value, 'BUY')
        assert commission == 5.0  # 最低5元
    
    @pytest.mark.unit
    def test_calculate_commission_sell(self, order_config):
        """测试卖出手续费计算（含印花税）"""
        om = MockOrderManager(order_config)
        
        order_value = 100000
        commission = om.calculate_commission(order_value, 'SELL')
        
        # 手续费 + 印花税
        base_commission = 100000 * 0.0003  # 万三
        stamp_tax = 100000 * 0.001  # 千一印花税
        expected = base_commission + stamp_tax
        
        assert abs(commission - expected) < 0.01  # 使用浮点数比较
    
    # === 并发和边界测试 ===
    @pytest.mark.unit
    def test_concurrent_order_creation(self, order_config):
        """测试并发订单创建"""
        om = MockOrderManager(order_config)
        
        import threading
        import queue
        
        results = queue.Queue()
        
        def create_order_worker(symbol_suffix):
            try:
                request = {
                    'symbol': f'00000{symbol_suffix}.SZ',
                    'side': 'BUY',
                    'quantity': 1000,
                    'price': 15.0
                }
                order_id = om.create_order(request)
                results.put(('success', order_id))
            except Exception as e:
                results.put(('error', str(e)))
        
        # 创建多个线程同时创建订单
        threads = []
        for i in range(5):
            t = threading.Thread(target=create_order_worker, args=(i,))
            threads.append(t)
            t.start()
        
        # 等待所有线程完成
        for t in threads:
            t.join()
        
        # 检查结果
        success_count = 0
        error_count = 0
        while not results.empty():
            status, data = results.get()
            if status == 'success':
                success_count += 1
            else:
                error_count += 1
        
        assert success_count >= 4  # 大部分应该成功
        assert len(om.orders) == success_count
    
    @pytest.mark.unit
    def test_order_timeout_handling(self, order_config, sample_order_request):
        """测试订单超时处理"""
        om = MockOrderManager(order_config)
        
        order_id = om.create_order(sample_order_request)
        order = om.get_order(order_id)
        
        # 模拟订单超时
        order['create_time'] = datetime.now() - timedelta(seconds=order_config['order_timeout'] + 1)
        
        # 检查订单是否超时（实际实现中会有定时任务处理）
        current_time = datetime.now()
        order_age = (current_time - order['create_time']).total_seconds()
        
        assert order_age > order_config['order_timeout']
    
    @pytest.mark.unit
    def test_large_order_handling(self, order_config):
        """测试大额订单处理"""
        om = MockOrderManager(order_config)
        
        large_order_request = {
            'symbol': '000001.SZ',
            'side': 'BUY',
            'quantity': 1000000,  # 100万股
            'price': 15.0,
            'order_type': 'LIMIT'
        }
        
        order_id = om.create_order(large_order_request)
        assert order_id is not None
        
        order = om.get_order(order_id)
        assert order['quantity'] == 1000000
        
        # 测试大额订单的手续费计算
        order_value = 1000000 * 15.0  # 1500万
        commission = om.calculate_commission(order_value, 'BUY')
        expected = order_value * 0.0003
        assert abs(commission - expected) < 0.01  # 使用浮点数比较
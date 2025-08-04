"""
OrderManager单元测试

测试订单管理器的核心功能
按照TDD原则，先编写完整的单元测试，确保测试全部失败，然后实现功能代码
"""

import pytest
import pytest_asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional
from unittest.mock import AsyncMock, Mock, patch

from myQuant.core.managers.order_manager import OrderManager
from myQuant.core.models.orders import Order, OrderSide, OrderType, OrderStatus, TimeInForce
from myQuant.infrastructure.database.database_manager import DatabaseManager
from myQuant.infrastructure.database.repositories import OrderRepository, UserRepository


@pytest.fixture
def test_db_url():
    """测试数据库URL"""
    return "sqlite:///:memory:"


@pytest_asyncio.fixture
async def database_manager(test_db_url):
    """测试用数据库管理器"""
    manager = DatabaseManager(test_db_url)
    await manager.initialize()
    yield manager
    await manager.close()


@pytest.fixture
def sample_order_requests():
    """示例订单请求数据"""
    return [
        {
            "symbol": "000001.SZ",
            "side": OrderSide.BUY,
            "order_type": OrderType.LIMIT,
            "quantity": 1000,
            "price": Decimal("15.00"),
            "user_id": 1
        },
        {
            "symbol": "000002.SZ", 
            "side": OrderSide.SELL,
            "order_type": OrderType.MARKET,
            "quantity": 500,
            "price": None,
            "user_id": 1
        },
        {
            "symbol": "600000.SH",
            "side": OrderSide.BUY,
            "order_type": OrderType.STOP_LIMIT,
            "quantity": 800,
            "price": Decimal("25.00"),
            "stop_price": Decimal("24.50"),
            "user_id": 2
        }
    ]


@pytest.fixture
def sample_executions():
    """示例执行数据"""
    return [
        {
            "execution_id": "EXEC_001",
            "quantity": 500,
            "price": Decimal("15.05"),
            "timestamp": datetime.now(timezone.utc),
            "commission": Decimal("2.25")
        },
        {
            "execution_id": "EXEC_002", 
            "quantity": 300,
            "price": Decimal("15.10"),
            "timestamp": datetime.now(timezone.utc),
            "commission": Decimal("1.35")
        }
    ]


class TestOrderManager:
    """OrderManager测试"""

    @pytest.mark.asyncio
    async def test_order_manager_initialization(self, database_manager):
        """测试OrderManager初始化"""
        # Arrange & Act
        manager = OrderManager(database_manager)

        # Assert
        assert manager.db_manager == database_manager
        assert manager.order_repository is not None
        assert manager.user_repository is not None

    @pytest.mark.asyncio
    async def test_create_order_success(self, database_manager, sample_order_requests):
        """测试成功创建订单"""
        # Arrange
        manager = OrderManager(database_manager)
        order_request = sample_order_requests[0]

        # Act
        order = await manager.create_order(order_request)

        # Assert
        assert isinstance(order, Order)
        assert order.symbol == "000001.SZ"
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.LIMIT
        assert order.quantity == 1000
        assert order.price == Decimal("15.00")
        assert order.status == OrderStatus.CREATED
        assert order.user_id == 1

    @pytest.mark.asyncio
    async def test_create_order_invalid_user(self, database_manager):
        """测试无效用户ID创建订单"""
        # Arrange
        manager = OrderManager(database_manager)
        invalid_request = {
            "symbol": "000001.SZ",
            "side": OrderSide.BUY,
            "order_type": OrderType.LIMIT,
            "quantity": 1000,
            "price": Decimal("15.00"),
            "user_id": 999  # 不存在的用户
        }

        # Act & Assert
        with pytest.raises(ValueError, match="User not found"):
            await manager.create_order(invalid_request)

    @pytest.mark.asyncio
    async def test_create_order_invalid_data(self, database_manager):
        """测试无效数据创建订单"""
        # Arrange
        manager = OrderManager(database_manager)
        
        invalid_requests = [
            {
                "symbol": "",  # 空symbol
                "side": OrderSide.BUY,
                "order_type": OrderType.LIMIT,
                "quantity": 1000,
                "price": Decimal("15.00"),
                "user_id": 1
            },
            {
                "symbol": "000001.SZ",
                "side": OrderSide.BUY,
                "order_type": OrderType.LIMIT,
                "quantity": 0,  # 无效数量
                "price": Decimal("15.00"),
                "user_id": 1
            },
            {
                "symbol": "000001.SZ",
                "side": OrderSide.BUY,
                "order_type": OrderType.LIMIT,
                "quantity": 1000,
                "price": Decimal("0"),  # 无效价格
                "user_id": 1
            }
        ]

        # Act & Assert
        for invalid_request in invalid_requests:
            with pytest.raises(ValueError):
                await manager.create_order(invalid_request)

    @pytest.mark.asyncio
    async def test_submit_order_success(self, database_manager, sample_order_requests):
        """测试成功提交订单"""
        # Arrange
        manager = OrderManager(database_manager)
        order_request = sample_order_requests[0]
        order = await manager.create_order(order_request)

        # Act
        result = await manager.submit_order(order.order_id)

        # Assert
        assert result is True
        updated_order = await manager.get_order_by_id(order.order_id)
        assert updated_order.status == OrderStatus.SUBMITTED

    @pytest.mark.asyncio
    async def test_submit_order_not_found(self, database_manager):
        """测试提交不存在的订单"""
        # Arrange
        manager = OrderManager(database_manager)

        # Act & Assert
        with pytest.raises(ValueError, match="Order not found"):
            await manager.submit_order("INVALID_ORDER_ID")

    @pytest.mark.asyncio
    async def test_submit_order_invalid_status(self, database_manager, sample_order_requests):
        """测试提交非法状态的订单"""
        # Arrange
        manager = OrderManager(database_manager)
        order_request = sample_order_requests[0]
        order = await manager.create_order(order_request)
        
        # 将订单设置为已成交状态
        await manager._update_order_status(order.order_id, OrderStatus.FILLED)

        # Act & Assert
        result = await manager.submit_order(order.order_id)
        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_order_success(self, database_manager, sample_order_requests):
        """测试成功取消订单"""
        # Arrange
        manager = OrderManager(database_manager)
        order_request = sample_order_requests[0]
        order = await manager.create_order(order_request)
        await manager.submit_order(order.order_id)

        # Act
        result = await manager.cancel_order(order.order_id)

        # Assert
        assert result is True
        updated_order = await manager.get_order_by_id(order.order_id)
        assert updated_order.status == OrderStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_order_not_found(self, database_manager):
        """测试取消不存在的订单"""
        # Arrange
        manager = OrderManager(database_manager)

        # Act & Assert
        with pytest.raises(ValueError, match="Order not found"):
            await manager.cancel_order("INVALID_ORDER_ID")

    @pytest.mark.asyncio
    async def test_cancel_order_invalid_status(self, database_manager, sample_order_requests):
        """测试取消非法状态的订单"""
        # Arrange
        manager = OrderManager(database_manager)
        order_request = sample_order_requests[0]
        order = await manager.create_order(order_request)
        
        # 将订单设置为已成交状态
        await manager._update_order_status(order.order_id, OrderStatus.FILLED)

        # Act
        result = await manager.cancel_order(order.order_id)

        # Assert
        assert result is False

    @pytest.mark.asyncio
    async def test_get_order_by_id(self, database_manager, sample_order_requests):
        """测试根据ID获取订单"""
        # Arrange
        manager = OrderManager(database_manager)
        order_request = sample_order_requests[0]
        order = await manager.create_order(order_request)

        # Act
        retrieved_order = await manager.get_order_by_id(order.order_id)

        # Assert
        assert retrieved_order is not None
        assert retrieved_order.order_id == order.order_id
        assert retrieved_order.symbol == order.symbol
        assert retrieved_order.quantity == order.quantity

    @pytest.mark.asyncio
    async def test_get_order_by_id_not_found(self, database_manager):
        """测试获取不存在的订单"""
        # Arrange
        manager = OrderManager(database_manager)

        # Act
        order = await manager.get_order_by_id("INVALID_ORDER_ID")

        # Assert
        assert order is None

    @pytest.mark.asyncio
    async def test_get_orders_by_user(self, database_manager, sample_order_requests):
        """测试根据用户获取订单"""
        # Arrange
        manager = OrderManager(database_manager)
        
        # 创建不同用户的订单
        order1 = await manager.create_order(sample_order_requests[0])  # user_id = 1
        order2 = await manager.create_order(sample_order_requests[1])  # user_id = 1
        order3 = await manager.create_order(sample_order_requests[2])  # user_id = 2

        # Act
        user1_orders = await manager.get_orders_by_user(1)
        user2_orders = await manager.get_orders_by_user(2)

        # Assert
        assert len(user1_orders) == 2
        assert len(user2_orders) == 1
        assert all(order.user_id == 1 for order in user1_orders)
        assert all(order.user_id == 2 for order in user2_orders)

    @pytest.mark.asyncio
    async def test_get_orders_by_symbol(self, database_manager, sample_order_requests):
        """测试根据股票代码获取订单"""
        # Arrange
        manager = OrderManager(database_manager)
        
        # 创建不同股票的订单
        order1 = await manager.create_order(sample_order_requests[0])  # 000001.SZ
        order2 = await manager.create_order(sample_order_requests[1])  # 000002.SZ
        order3 = await manager.create_order(sample_order_requests[2])  # 600000.SH

        # Act
        orders_000001 = await manager.get_orders_by_symbol("000001.SZ")
        orders_000002 = await manager.get_orders_by_symbol("000002.SZ")

        # Assert
        assert len(orders_000001) == 1
        assert len(orders_000002) == 1
        assert orders_000001[0].symbol == "000001.SZ"
        assert orders_000002[0].symbol == "000002.SZ"

    @pytest.mark.asyncio
    async def test_get_orders_by_status(self, database_manager, sample_order_requests):
        """测试根据状态获取订单"""
        # Arrange
        manager = OrderManager(database_manager)
        
        # 创建订单并设置不同状态
        order1 = await manager.create_order(sample_order_requests[0])
        order2 = await manager.create_order(sample_order_requests[1])
        order3 = await manager.create_order(sample_order_requests[2])
        
        # 提交一些订单
        await manager.submit_order(order1.order_id)
        await manager.submit_order(order2.order_id)

        # Act
        created_orders = await manager.get_orders_by_status(OrderStatus.CREATED)
        submitted_orders = await manager.get_orders_by_status(OrderStatus.SUBMITTED)

        # Assert
        assert len(created_orders) == 1
        assert len(submitted_orders) == 2
        assert created_orders[0].status == OrderStatus.CREATED
        assert all(order.status == OrderStatus.SUBMITTED for order in submitted_orders)

    @pytest.mark.asyncio
    async def test_process_execution_full_fill(self, database_manager, sample_order_requests, sample_executions):
        """测试处理完全成交"""
        # Arrange
        manager = OrderManager(database_manager)
        order_request = sample_order_requests[0]
        order = await manager.create_order(order_request)
        await manager.submit_order(order.order_id)
        
        execution = sample_executions[0].copy()
        execution["quantity"] = 1000  # 完全成交

        # Act
        result = await manager.process_execution(order.order_id, execution)

        # Assert
        assert result is True
        updated_order = await manager.get_order_by_id(order.order_id)
        assert updated_order.status == OrderStatus.FILLED
        assert updated_order.filled_quantity == 1000
        assert updated_order.remaining_quantity == 0

    @pytest.mark.asyncio
    async def test_process_execution_partial_fill(self, database_manager, sample_order_requests, sample_executions):
        """测试处理部分成交"""
        # Arrange
        manager = OrderManager(database_manager)
        order_request = sample_order_requests[0]
        order = await manager.create_order(order_request)
        await manager.submit_order(order.order_id)
        
        execution = sample_executions[0].copy()
        execution["quantity"] = 500  # 部分成交

        # Act
        result = await manager.process_execution(order.order_id, execution)

        # Assert
        assert result is True
        updated_order = await manager.get_order_by_id(order.order_id)
        assert updated_order.status == OrderStatus.PARTIALLY_FILLED
        assert updated_order.filled_quantity == 500
        assert updated_order.remaining_quantity == 500

    @pytest.mark.asyncio
    async def test_process_execution_multiple_fills(self, database_manager, sample_order_requests, sample_executions):
        """测试处理多次成交"""
        # Arrange
        manager = OrderManager(database_manager)
        order_request = sample_order_requests[0]
        order = await manager.create_order(order_request)
        await manager.submit_order(order.order_id)

        # Act - 第一次成交
        execution1 = sample_executions[0].copy()
        execution1["quantity"] = 500
        result1 = await manager.process_execution(order.order_id, execution1)
        
        # Act - 第二次成交
        execution2 = sample_executions[1].copy()
        execution2["quantity"] = 500
        result2 = await manager.process_execution(order.order_id, execution2)

        # Assert
        assert result1 is True
        assert result2 is True
        
        updated_order = await manager.get_order_by_id(order.order_id)
        assert updated_order.status == OrderStatus.FILLED
        assert updated_order.filled_quantity == 1000
        assert updated_order.remaining_quantity == 0

    @pytest.mark.asyncio
    async def test_process_execution_order_not_found(self, database_manager, sample_executions):
        """测试处理不存在订单的执行"""
        # Arrange
        manager = OrderManager(database_manager)
        execution = sample_executions[0]

        # Act & Assert
        with pytest.raises(ValueError, match="Order not found"):
            await manager.process_execution("INVALID_ORDER_ID", execution)

    @pytest.mark.asyncio
    async def test_process_execution_invalid_quantity(self, database_manager, sample_order_requests):
        """测试处理无效执行数量"""
        # Arrange
        manager = OrderManager(database_manager)
        order_request = sample_order_requests[0]
        order = await manager.create_order(order_request)
        await manager.submit_order(order.order_id)
        
        invalid_execution = {
            "execution_id": "EXEC_001",
            "quantity": 2000,  # 超过订单数量
            "price": Decimal("15.05"),
            "timestamp": datetime.now(timezone.utc),
            "commission": Decimal("2.25")
        }

        # Act & Assert
        with pytest.raises(ValueError, match="Execution quantity exceeds remaining quantity"):
            await manager.process_execution(order.order_id, invalid_execution)

    @pytest.mark.asyncio
    async def test_calculate_order_value(self, database_manager, sample_order_requests):
        """测试计算订单价值"""
        # Arrange
        manager = OrderManager(database_manager)
        order_request = sample_order_requests[0]
        order = await manager.create_order(order_request)

        # Act
        value = await manager.calculate_order_value(order.order_id)

        # Assert
        expected_value = Decimal("1000") * Decimal("15.00")  # 1000 * 15.00 = 15000
        assert value == expected_value

    @pytest.mark.asyncio
    async def test_calculate_order_value_market_order(self, database_manager, sample_order_requests):
        """测试计算市价订单价值"""
        # Arrange
        manager = OrderManager(database_manager)
        order_request = sample_order_requests[1]  # 市价订单
        order = await manager.create_order(order_request)

        # Act & Assert
        with pytest.raises(ValueError, match="Cannot calculate value for market order without price"):
            await manager.calculate_order_value(order.order_id)

    @pytest.mark.asyncio
    async def test_validate_order_basic_validation(self, database_manager):
        """测试订单基础验证"""
        # Arrange
        manager = OrderManager(database_manager)
        
        # 验证空symbol
        invalid_request = {
            "symbol": "",
            "side": OrderSide.BUY,
            "order_type": OrderType.LIMIT,
            "quantity": 1000,
            "price": Decimal("15.00"),
            "user_id": 1
        }

        # Act & Assert
        is_valid, errors = await manager.validate_order(invalid_request)
        assert is_valid is False
        assert any("symbol" in error.lower() for error in errors)

    @pytest.mark.asyncio
    async def test_validate_order_limit_price_required(self, database_manager):
        """测试限价订单必须有价格"""
        # Arrange
        manager = OrderManager(database_manager)
        
        invalid_request = {
            "symbol": "000001.SZ",
            "side": OrderSide.BUY,
            "order_type": OrderType.LIMIT,
            "quantity": 1000,
            "price": None,  # 限价订单缺少价格
            "user_id": 1
        }

        # Act & Assert
        is_valid, errors = await manager.validate_order(invalid_request)
        assert is_valid is False
        assert any("price" in error.lower() for error in errors)

    @pytest.mark.asyncio
    async def test_get_order_history_by_user(self, database_manager, sample_order_requests):
        """测试获取用户订单历史"""
        # Arrange
        manager = OrderManager(database_manager)
        
        # 创建订单并执行一些操作
        order1 = await manager.create_order(sample_order_requests[0])
        order2 = await manager.create_order(sample_order_requests[1])
        
        await manager.submit_order(order1.order_id)
        await manager.cancel_order(order1.order_id)

        # Act
        history = await manager.get_order_history_by_user(1)

        # Assert
        assert len(history) == 2
        assert all(order.user_id == 1 for order in history)

    @pytest.mark.asyncio
    async def test_get_active_orders_by_user(self, database_manager, sample_order_requests):
        """测试获取用户活跃订单"""
        # Arrange
        manager = OrderManager(database_manager)
        
        # 创建订单并设置不同状态
        order1 = await manager.create_order(sample_order_requests[0])
        order2 = await manager.create_order(sample_order_requests[1])
        
        await manager.submit_order(order1.order_id)
        await manager.cancel_order(order2.order_id)

        # Act
        active_orders = await manager.get_active_orders_by_user(1)

        # Assert
        assert len(active_orders) == 1
        assert active_orders[0].status == OrderStatus.SUBMITTED

    @pytest.mark.asyncio 
    async def test_cancel_all_orders_by_user(self, database_manager, sample_order_requests):
        """测试取消用户所有订单"""
        # Arrange
        manager = OrderManager(database_manager)
        
        # 创建多个订单
        order1 = await manager.create_order(sample_order_requests[0])
        order2 = await manager.create_order(sample_order_requests[1])
        
        await manager.submit_order(order1.order_id)
        await manager.submit_order(order2.order_id)

        # Act
        cancelled_count = await manager.cancel_all_orders_by_user(1)

        # Assert
        assert cancelled_count == 2
        
        # 验证订单状态
        updated_order1 = await manager.get_order_by_id(order1.order_id)
        updated_order2 = await manager.get_order_by_id(order2.order_id)
        assert updated_order1.status == OrderStatus.CANCELLED
        assert updated_order2.status == OrderStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_all_orders_by_symbol(self, database_manager, sample_order_requests):
        """测试取消特定股票的所有订单"""
        # Arrange
        manager = OrderManager(database_manager)
        
        # 创建不同股票的订单
        order1 = await manager.create_order(sample_order_requests[0])  # 000001.SZ
        order2 = await manager.create_order(sample_order_requests[1])  # 000002.SZ
        
        await manager.submit_order(order1.order_id)
        await manager.submit_order(order2.order_id)

        # Act
        cancelled_count = await manager.cancel_all_orders_by_symbol("000001.SZ")

        # Assert
        assert cancelled_count == 1
        
        # 验证订单状态
        updated_order1 = await manager.get_order_by_id(order1.order_id)
        updated_order2 = await manager.get_order_by_id(order2.order_id)
        assert updated_order1.status == OrderStatus.CANCELLED
        assert updated_order2.status == OrderStatus.SUBMITTED  # 不同股票，未取消


if __name__ == "__main__":
    # 运行测试确保全部失败（TDD第一步）
    pytest.main([__file__, "-v", "--tb=short"])
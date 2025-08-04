"""
PortfolioManager单元测试

测试投资组合管理器的核心功能
按照TDD原则，先编写完整的单元测试，确保测试全部失败，然后实现功能代码
"""

import pytest
import pytest_asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional
from unittest.mock import AsyncMock, Mock, patch

from myQuant.core.managers.portfolio_manager import PortfolioManager
from myQuant.core.models.portfolio import Portfolio
from myQuant.core.models.positions import Position
from myQuant.core.models.orders import Order, OrderSide, OrderType
from myQuant.infrastructure.database.database_manager import DatabaseManager
from myQuant.infrastructure.database.repositories import PositionRepository, UserRepository


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
def mock_position_repository():
    """模拟持仓Repository"""
    return Mock(spec=PositionRepository)


@pytest.fixture
def mock_user_repository():
    """模拟用户Repository"""
    return Mock(spec=UserRepository)


@pytest.fixture
def sample_positions():
    """示例持仓数据"""
    return [
        Position(1, "000001.SZ", 1000, Decimal("12.00")),
        Position(1, "000002.SZ", 500, Decimal("25.00")),
        Position(1, "600000.SH", 800, Decimal("8.50"))
    ]


@pytest.fixture
def sample_prices():
    """示例价格数据"""
    return {
        "000001.SZ": Decimal("12.75"),
        "000002.SZ": Decimal("26.00"),
        "600000.SH": Decimal("8.80")
    }


class TestPortfolioManager:
    """PortfolioManager测试"""

    @pytest.mark.asyncio
    async def test_portfolio_manager_initialization(self, database_manager):
        """测试PortfolioManager初始化"""
        # Arrange & Act
        manager = PortfolioManager(database_manager)

        # Assert
        assert manager.db_manager == database_manager
        assert manager.position_repository is not None
        assert manager.user_repository is not None

    @pytest.mark.asyncio
    async def test_create_portfolio(self, database_manager):
        """测试创建投资组合"""
        # Arrange
        manager = PortfolioManager(database_manager)
        user_id = 1
        initial_capital = Decimal("1000000.00")

        # Act
        portfolio = await manager.create_portfolio(user_id, initial_capital)

        # Assert
        assert isinstance(portfolio, Portfolio)
        assert portfolio.user_id == user_id
        assert portfolio.initial_capital == initial_capital
        assert portfolio.cash_balance == initial_capital
        assert len(portfolio.positions) == 0

    @pytest.mark.asyncio
    async def test_get_portfolio_by_user(self, database_manager, sample_positions):
        """测试根据用户获取投资组合"""
        # Arrange
        manager = PortfolioManager(database_manager)
        user_id = 1
        
        # 预先创建投资组合和持仓
        portfolio = await manager.create_portfolio(user_id, Decimal("1000000.00"))
        
        # Mock持仓数据
        with patch.object(manager.position_repository, 'get_positions_by_user', return_value=sample_positions):
            # Act
            retrieved_portfolio = await manager.get_portfolio_by_user(user_id)

            # Assert
            assert isinstance(retrieved_portfolio, Portfolio)
            assert retrieved_portfolio.user_id == user_id
            assert len(retrieved_portfolio.positions) == 3

    @pytest.mark.asyncio
    async def test_add_position_to_portfolio(self, database_manager):
        """测试向投资组合添加持仓"""
        # Arrange
        manager = PortfolioManager(database_manager)
        user_id = 1
        portfolio = await manager.create_portfolio(user_id, Decimal("1000000.00"))
        
        position_data = {
            "symbol": "000001.SZ",
            "quantity": 1000,
            "average_price": Decimal("12.00")
        }

        # Act
        result = await manager.add_position(user_id, position_data)

        # Assert
        assert result is True
        
        # 验证持仓已添加到数据库
        portfolio = await manager.get_portfolio_by_user(user_id)
        assert "000001.SZ" in [pos.symbol for pos in portfolio.positions.values()]

    @pytest.mark.asyncio
    async def test_update_position_in_portfolio(self, database_manager):
        """测试更新投资组合中的持仓"""
        # Arrange
        manager = PortfolioManager(database_manager)
        user_id = 1
        await manager.create_portfolio(user_id, Decimal("1000000.00"))
        
        # 先添加持仓
        position_data = {
            "symbol": "000001.SZ",
            "quantity": 1000,
            "average_price": Decimal("12.00")
        }
        await manager.add_position(user_id, position_data)

        # Act - 更新持仓
        update_data = {
            "quantity": 1500,
            "average_price": Decimal("12.20")
        }
        result = await manager.update_position_async(user_id, "000001.SZ", update_data)

        # Assert
        assert result is True
        
        # 验证持仓已更新
        portfolio = await manager.get_portfolio_by_user(user_id)
        position = portfolio.positions["000001.SZ"]
        assert position.quantity == 1500
        assert position.average_price == Decimal("12.20")

    @pytest.mark.asyncio
    async def test_remove_position_from_portfolio(self, database_manager):
        """测试从投资组合移除持仓"""
        # Arrange
        manager = PortfolioManager(database_manager)
        user_id = 1
        await manager.create_portfolio(user_id, Decimal("1000000.00"))
        
        # 先添加持仓
        position_data = {
            "symbol": "000001.SZ",
            "quantity": 1000,
            "average_price": Decimal("12.00")
        }
        await manager.add_position(user_id, position_data)

        # Act
        result = await manager.remove_position(user_id, "000001.SZ")

        # Assert
        assert result is True
        
        # 验证持仓已移除
        portfolio = await manager.get_portfolio_by_user(user_id)
        assert "000001.SZ" not in portfolio.positions

    @pytest.mark.asyncio
    async def test_calculate_portfolio_value(self, database_manager, sample_positions, sample_prices):
        """测试计算投资组合总价值"""
        # Arrange
        manager = PortfolioManager(database_manager)
        user_id = 1
        portfolio = await manager.create_portfolio(user_id, Decimal("1000000.00"))
        
        # Mock持仓数据
        with patch.object(manager.position_repository, 'get_positions_by_user', return_value=sample_positions):
            # Act
            total_value = await manager.calculate_portfolio_value(user_id, sample_prices)

            # Assert
            # 000001.SZ: 1000 * 12.75 = 12750
            # 000002.SZ: 500 * 26.00 = 13000  
            # 600000.SH: 800 * 8.80 = 7040
            # 持仓价值: 1000*12.75 + 500*26.00 + 800*8.80 = 12750 + 13000 + 7040 = 32790
            # 现金: 1000000 (因为没有实际购买，现金未减少)
            # 总计: 32790 + 1000000 = 1032790
            expected_total = Decimal("1032790.00")
            assert total_value == expected_total

    @pytest.mark.asyncio
    async def test_calculate_portfolio_pnl(self, database_manager, sample_positions, sample_prices):
        """测试计算投资组合盈亏"""
        # Arrange
        manager = PortfolioManager(database_manager)
        user_id = 1
        await manager.create_portfolio(user_id, Decimal("1000000.00"))
        
        # Mock持仓数据
        with patch.object(manager.position_repository, 'get_positions_by_user', return_value=sample_positions):
            # Act
            total_pnl = await manager.calculate_portfolio_pnl(user_id, sample_prices)

            # Assert
            # 000001.SZ: (12.75 - 12.00) * 1000 = 750
            # 000002.SZ: (26.00 - 25.00) * 500 = 500
            # 600000.SH: (8.80 - 8.50) * 800 = 240
            # 总计: 1490
            expected_pnl = Decimal("1490.00")
            assert total_pnl == expected_pnl

    @pytest.mark.asyncio
    async def test_get_portfolio_summary(self, database_manager, sample_positions, sample_prices):
        """测试获取投资组合摘要"""
        # Arrange
        manager = PortfolioManager(database_manager)
        user_id = 1
        await manager.create_portfolio(user_id, Decimal("1000000.00"))
        
        # Mock持仓数据
        with patch.object(manager.position_repository, 'get_positions_by_user', return_value=sample_positions):
            # Act
            summary = await manager.get_portfolio_summary(user_id, sample_prices)

            # Assert
            assert summary.total_value > 0
            assert summary.position_value > 0
            assert summary.cash_balance > 0
            assert summary.total_pnl > 0
            assert summary.positions_count == 3
            assert summary.total_return > 0

    @pytest.mark.asyncio
    async def test_update_cash_balance(self, database_manager):
        """测试更新现金余额"""
        # Arrange
        manager = PortfolioManager(database_manager)
        user_id = 1
        await manager.create_portfolio(user_id, Decimal("1000000.00"))

        # Act - 减少现金（买入股票）
        result = await manager.update_cash_balance(user_id, Decimal("-24500.00"))

        # Assert
        assert result is True
        
        portfolio = await manager.get_portfolio_by_user(user_id)
        assert portfolio.cash_balance == Decimal("975500.00")

        # Act - 增加现金（卖出股票）
        result = await manager.update_cash_balance(user_id, Decimal("12750.00"))

        # Assert
        assert result is True
        
        portfolio = await manager.get_portfolio_by_user(user_id)
        assert portfolio.cash_balance == Decimal("988250.00")

    @pytest.mark.asyncio
    async def test_update_cash_balance_insufficient_funds(self, database_manager):
        """测试现金余额不足的情况"""
        # Arrange
        manager = PortfolioManager(database_manager)
        user_id = 1
        await manager.create_portfolio(user_id, Decimal("1000000.00"))

        # Act - 尝试减少过多现金
        result = await manager.update_cash_balance(user_id, Decimal("-2000000.00"))

        # Assert
        assert result is False
        
        # 验证余额未改变
        portfolio = await manager.get_portfolio_by_user(user_id)
        assert portfolio.cash_balance == Decimal("1000000.00")

    @pytest.mark.asyncio
    async def test_process_order_execution_buy(self, database_manager):
        """测试处理订单执行 - 买入"""
        # Arrange
        manager = PortfolioManager(database_manager)
        user_id = 1
        await manager.create_portfolio(user_id, Decimal("1000000.00"))

        order = Order(
            symbol="000001.SZ",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1000,
            price=Decimal("12.50"),
            user_id=user_id
        )
        order.fill(1000, Decimal("12.50"))  # 完全成交

        # Act
        result = await manager.process_order_execution(order)

        # Assert
        assert result is True
        
        # 验证持仓已添加
        portfolio = await manager.get_portfolio_by_user(user_id)
        assert "000001.SZ" in [pos.symbol for pos in portfolio.positions.values()]
        
        # 验证现金减少
        expected_cash = Decimal("1000000.00") - Decimal("12500.00")  # 1000 * 12.50
        assert portfolio.cash_balance == expected_cash

    @pytest.mark.asyncio
    async def test_process_order_execution_sell(self, database_manager):
        """测试处理订单执行 - 卖出"""
        # Arrange
        manager = PortfolioManager(database_manager)
        user_id = 1
        await manager.create_portfolio(user_id, Decimal("1000000.00"))
        
        # 先添加持仓
        position_data = {
            "symbol": "000001.SZ",
            "quantity": 2000,
            "average_price": Decimal("12.00")
        }
        await manager.add_position(user_id, position_data)

        order = Order(
            symbol="000001.SZ",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=1000,
            price=Decimal("12.75"),
            user_id=user_id
        )
        order.fill(1000, Decimal("12.75"))  # 完全成交

        # Act
        result = await manager.process_order_execution(order)

        # Assert
        assert result is True
        
        # 验证持仓减少
        portfolio = await manager.get_portfolio_by_user(user_id)
        position = portfolio.positions["000001.SZ"]
        assert position.quantity == 1000  # 2000 - 1000 = 1000
        
        # 验证现金增加
        initial_cost = Decimal("2000") * Decimal("12.00")  # 24000
        remaining_cash = Decimal("1000000.00") - initial_cost
        sale_proceeds = Decimal("1000") * Decimal("12.75")  # 12750
        expected_cash = remaining_cash + sale_proceeds
        assert portfolio.cash_balance == expected_cash

    @pytest.mark.asyncio
    async def test_get_position_allocation(self, database_manager, sample_positions, sample_prices):
        """测试获取持仓配置"""
        # Arrange
        manager = PortfolioManager(database_manager)
        user_id = 1
        await manager.create_portfolio(user_id, Decimal("1000000.00"))
        
        # Mock持仓数据
        with patch.object(manager.position_repository, 'get_positions_by_user', return_value=sample_positions):
            # Act
            allocation = await manager.get_position_allocation(user_id, sample_prices)

            # Assert
            assert isinstance(allocation, dict)
            assert len(allocation) == 3
            assert "000001.SZ" in allocation
            assert "000002.SZ" in allocation
            assert "600000.SH" in allocation
            
            # 验证配置比例总和约为100%（除了现金部分）
            total_allocation = sum(allocation.values())
            assert 0 < total_allocation < 100  # 应该小于100%，因为有现金

    @pytest.mark.asyncio
    async def test_rebalance_portfolio(self, database_manager, sample_positions):
        """测试投资组合再平衡"""
        # Arrange
        manager = PortfolioManager(database_manager)
        user_id = 1
        await manager.create_portfolio(user_id, Decimal("1000000.00"))
        
        target_allocation = {
            "000001.SZ": Decimal("30.0"),  # 30%
            "000002.SZ": Decimal("40.0"),  # 40%
            "600000.SH": Decimal("20.0"),  # 20%
        }
        
        current_prices = {
            "000001.SZ": Decimal("12.75"),
            "000002.SZ": Decimal("26.00"),
            "600000.SH": Decimal("8.80")
        }

        # Act
        rebalance_orders = await manager.rebalance_portfolio(user_id, target_allocation, current_prices)

        # Assert
        assert isinstance(rebalance_orders, list)
        # 验证生成的再平衡订单
        for order in rebalance_orders:
            assert hasattr(order, 'symbol')
            assert hasattr(order, 'side')
            assert hasattr(order, 'quantity')
            assert hasattr(order, 'price')

    @pytest.mark.asyncio
    async def test_portfolio_performance_metrics(self, database_manager, sample_positions, sample_prices):
        """测试投资组合绩效指标"""
        # Arrange
        manager = PortfolioManager(database_manager)
        user_id = 1
        await manager.create_portfolio(user_id, Decimal("1000000.00"))
        
        # Mock持仓数据
        with patch.object(manager.position_repository, 'get_positions_by_user', return_value=sample_positions):
            # Mock历史价值数据
            historical_values = [
                Decimal("1000000.00"),
                Decimal("1005000.00"),
                Decimal("1002000.00"),
                Decimal("1001490.00")
            ]

            # Act
            metrics = await manager.calculate_performance_metrics(user_id, sample_prices, historical_values)

            # Assert
            assert hasattr(metrics, 'total_return')
            assert hasattr(metrics, 'volatility')
            assert hasattr(metrics, 'max_drawdown')
            assert hasattr(metrics, 'sharpe_ratio')
            assert metrics.total_return > 0
            assert metrics.volatility >= 0
            assert metrics.max_drawdown <= 0


if __name__ == "__main__":
    # 运行测试确保全部失败（TDD第一步）
    pytest.main([__file__, "-v", "--tb=short"])
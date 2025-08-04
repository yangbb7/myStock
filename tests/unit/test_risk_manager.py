# -*- coding: utf-8 -*-
"""
RiskManager测试
"""

import pytest
import pytest_asyncio
import asyncio
from decimal import Decimal
from datetime import datetime, timezone
from typing import Dict, List, Any

from myQuant.core.managers.risk_manager import RiskManager
from myQuant.core.models.portfolio import Portfolio, Position
from myQuant.core.models.orders import Order, OrderType, OrderSide, OrderStatus
from myQuant.infrastructure.database.database_manager import DatabaseManager


class TestRiskManager:
    """RiskManager测试"""

    @pytest.mark.asyncio
    async def test_risk_manager_initialization(self, database_manager):
        """测试风险管理器初始化"""
        # Act
        manager = RiskManager(database_manager)
        
        # Assert
        assert manager is not None
        assert manager.db_manager == database_manager
        assert hasattr(manager, 'portfolio_repository')
        assert hasattr(manager, 'order_repository')
        assert hasattr(manager, 'user_repository')

    @pytest.mark.asyncio
    async def test_calculate_portfolio_risk_basic(self, database_manager, sample_positions):
        """测试计算基础投资组合风险"""
        # Arrange
        manager = RiskManager(database_manager)
        portfolio = Portfolio(user_id=1, initial_capital=Decimal("100000"))
        for pos_data in sample_positions:
            position = Position(
                user_id=1,
                symbol=pos_data["symbol"],
                quantity=pos_data["quantity"],
                average_price=Decimal(str(pos_data["average_price"]))
            )
            portfolio.add_position(position)
        
        # Act
        risk_metrics = await manager.calculate_portfolio_risk(portfolio)
        
        # Assert
        assert risk_metrics is not None
        assert "total_value" in risk_metrics
        assert "var_95" in risk_metrics
        assert "max_drawdown" in risk_metrics
        assert "beta" in risk_metrics
        assert "sharpe_ratio" in risk_metrics
        assert risk_metrics["total_value"] > 0

    @pytest.mark.asyncio
    async def test_calculate_portfolio_var_95(self, database_manager, sample_positions):
        """测试计算投资组合95% VaR"""
        # Arrange
        manager = RiskManager(database_manager)
        portfolio = Portfolio(user_id=1, initial_capital=Decimal("100000"))
        for pos_data in sample_positions:
            position = Position(
                user_id=1,
                symbol=pos_data["symbol"],
                quantity=pos_data["quantity"],
                average_price=Decimal(str(pos_data["average_price"]))
            )
            portfolio.add_position(position)
        
        # Act
        var_95 = await manager.calculate_var_95(portfolio)
        
        # Assert
        assert isinstance(var_95, Decimal)
        assert var_95 > 0

    @pytest.mark.asyncio
    async def test_calculate_portfolio_beta(self, database_manager, sample_positions):
        """测试计算投资组合贝塔系数"""
        # Arrange
        manager = RiskManager(database_manager)
        portfolio = Portfolio(user_id=1, initial_capital=Decimal("100000"))
        for pos_data in sample_positions:
            position = Position(
                user_id=1,
                symbol=pos_data["symbol"],
                quantity=pos_data["quantity"],
                average_price=Decimal(str(pos_data["average_price"]))
            )
            portfolio.add_position(position)
        
        # Act
        beta = await manager.calculate_portfolio_beta(portfolio)
        
        # Assert
        assert isinstance(beta, Decimal)
        assert beta >= 0

    @pytest.mark.asyncio
    async def test_validate_order_position_limit(self, database_manager, sample_order_requests):
        """测试订单仓位限制验证"""
        # Arrange
        manager = RiskManager(database_manager)
        order_request = sample_order_requests[0]
        user_id = order_request["user_id"]
        
        # Act
        is_valid, reason = await manager.validate_order_position_limit(order_request, user_id)
        
        # Assert
        assert isinstance(is_valid, bool)
        assert isinstance(reason, str)

    @pytest.mark.asyncio
    async def test_validate_order_risk_limit(self, database_manager, sample_order_requests):
        """测试订单风险限制验证"""
        # Arrange
        manager = RiskManager(database_manager)
        order_request = sample_order_requests[0]
        user_id = order_request["user_id"]
        
        # Act
        is_valid, reason = await manager.validate_order_risk_limit(order_request, user_id)
        
        # Assert
        assert isinstance(is_valid, bool)
        assert isinstance(reason, str)

    @pytest.mark.asyncio
    async def test_validate_order_concentration_limit(self, database_manager, sample_order_requests):
        """测试订单集中度限制验证"""
        # Arrange
        manager = RiskManager(database_manager)
        order_request = sample_order_requests[0]
        user_id = order_request["user_id"]
        
        # Act
        is_valid, reason = await manager.validate_order_concentration_limit(order_request, user_id)
        
        # Assert
        assert isinstance(is_valid, bool)
        assert isinstance(reason, str)

    @pytest.mark.asyncio
    async def test_check_portfolio_risk_limits(self, database_manager, sample_positions):
        """测试检查投资组合风险限制"""
        # Arrange
        manager = RiskManager(database_manager)
        portfolio = Portfolio(user_id=1, initial_capital=Decimal("100000"))
        for pos_data in sample_positions:
            position = Position(
                user_id=1,
                symbol=pos_data["symbol"],
                quantity=pos_data["quantity"],
                average_price=Decimal(str(pos_data["average_price"]))
            )
            portfolio.add_position(position)
        
        # Act
        result = await manager.check_portfolio_risk_limits(portfolio)
        
        # Assert
        assert isinstance(result, dict)
        assert "within_limits" in result
        assert "violations" in result
        assert isinstance(result["within_limits"], bool)
        assert isinstance(result["violations"], list)

    @pytest.mark.asyncio
    async def test_calculate_maximum_position_size(self, database_manager):
        """测试计算最大仓位大小"""
        # Arrange
        manager = RiskManager(database_manager)
        user_id = 1
        symbol = "000001.SZ"
        current_price = Decimal("15.00")
        
        # Act
        max_size = await manager.calculate_maximum_position_size(user_id, symbol, current_price)
        
        # Assert
        assert isinstance(max_size, int)
        assert max_size >= 0

    @pytest.mark.asyncio
    async def test_calculate_stop_loss_price(self, database_manager):
        """测试计算止损价格"""
        # Arrange
        manager = RiskManager(database_manager)
        entry_price = Decimal("15.00")
        side = OrderSide.BUY
        risk_tolerance = Decimal("0.05")  # 5%
        
        # Act
        stop_loss_price = await manager.calculate_stop_loss_price(entry_price, side, risk_tolerance)
        
        # Assert
        assert isinstance(stop_loss_price, Decimal)
        if side == OrderSide.BUY:
            assert stop_loss_price < entry_price
        else:
            assert stop_loss_price > entry_price

    @pytest.mark.asyncio
    async def test_get_user_risk_config(self, database_manager):
        """测试获取用户风险配置"""
        # Arrange
        manager = RiskManager(database_manager)
        user_id = 1
        
        # Act
        config = await manager.get_user_risk_config(user_id)
        
        # Assert
        assert isinstance(config, dict)
        assert "risk_tolerance" in config
        assert "max_position_size" in config
        assert "max_daily_loss" in config
        assert "max_concentration" in config

    @pytest.mark.asyncio
    async def test_update_user_risk_config(self, database_manager):
        """测试更新用户风险配置"""
        # Arrange
        manager = RiskManager(database_manager)
        user_id = 1
        new_config = {
            "risk_tolerance": 0.03,
            "max_position_size": 0.15,
            "max_daily_loss": 0.02
        }
        
        # Act
        result = await manager.update_user_risk_config(user_id, new_config)
        
        # Assert
        assert result is True

    @pytest.mark.asyncio
    async def test_calculate_sharpe_ratio(self, database_manager, sample_positions):
        """测试计算夏普比率"""
        # Arrange
        manager = RiskManager(database_manager)
        portfolio = Portfolio(user_id=1, initial_capital=Decimal("100000"))
        for pos_data in sample_positions:
            position = Position(
                user_id=1,
                symbol=pos_data["symbol"],
                quantity=pos_data["quantity"],
                average_price=Decimal(str(pos_data["average_price"]))
            )
            portfolio.add_position(position)
        
        # Act
        sharpe_ratio = await manager.calculate_sharpe_ratio(portfolio)
        
        # Assert
        assert isinstance(sharpe_ratio, Decimal)

    @pytest.mark.asyncio
    async def test_calculate_maximum_drawdown(self, database_manager, sample_positions):
        """测试计算最大回撤"""
        # Arrange
        manager = RiskManager(database_manager)
        portfolio = Portfolio(user_id=1, initial_capital=Decimal("100000"))
        for pos_data in sample_positions:
            position = Position(
                user_id=1,
                symbol=pos_data["symbol"],
                quantity=pos_data["quantity"],
                average_price=Decimal(str(pos_data["average_price"]))
            )
            portfolio.add_position(position)
        
        # Act
        max_drawdown = await manager.calculate_maximum_drawdown(portfolio)
        
        # Assert
        assert isinstance(max_drawdown, Decimal)
        assert max_drawdown >= 0

    @pytest.mark.asyncio
    async def test_monitor_real_time_risk(self, database_manager):
        """测试实时风险监控"""
        # Arrange
        manager = RiskManager(database_manager)
        user_id = 1
        
        # Act
        result = await manager.monitor_real_time_risk(user_id)
        
        # Assert
        assert isinstance(result, dict)
        assert "alerts" in result
        assert "risk_level" in result
        assert isinstance(result["alerts"], list)
        assert result["risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

    @pytest.mark.asyncio
    async def test_generate_risk_report(self, database_manager):
        """测试生成风险报告"""
        # Arrange
        manager = RiskManager(database_manager)
        user_id = 1
        
        # Act
        report = await manager.generate_risk_report(user_id)
        
        # Assert
        assert isinstance(report, dict)
        assert "portfolio_value" in report
        assert "risk_metrics" in report
        assert "compliance_status" in report
        assert "recommendations" in report

    @pytest.mark.asyncio
    async def test_validate_order_comprehensive(self, database_manager, sample_order_requests):
        """测试综合订单验证"""
        # Arrange
        manager = RiskManager(database_manager)
        order_request = sample_order_requests[0]
        user_id = order_request["user_id"]
        
        # Act
        is_valid, violations = await manager.validate_order_comprehensive(order_request, user_id)
        
        # Assert
        assert isinstance(is_valid, bool)
        assert isinstance(violations, list)

    @pytest.mark.asyncio
    async def test_calculate_stress_test_scenarios(self, database_manager, sample_positions):
        """测试计算压力测试场景"""
        # Arrange
        manager = RiskManager(database_manager)
        portfolio = Portfolio(user_id=1, initial_capital=Decimal("100000"))
        for pos_data in sample_positions:
            position = Position(
                user_id=1,
                symbol=pos_data["symbol"],
                quantity=pos_data["quantity"],
                average_price=Decimal(str(pos_data["average_price"]))
            )
            portfolio.add_position(position)
        
        scenarios = [
            {"market_drop": -0.10},  # 10% 市场下跌
            {"market_drop": -0.20},  # 20% 市场下跌
            {"sector_drop": {"sector": "technology", "drop": -0.15}}  # 科技股下跌15%
        ]
        
        # Act
        results = await manager.calculate_stress_test_scenarios(portfolio, scenarios)
        
        # Assert
        assert isinstance(results, list)
        assert len(results) == len(scenarios)
        for result in results:
            assert "scenario" in result
            assert "portfolio_value_impact" in result
            assert "risk_metrics" in result

    @pytest.mark.asyncio
    async def test_get_risk_alerts(self, database_manager):
        """测试获取风险预警"""
        # Arrange
        manager = RiskManager(database_manager)
        user_id = 1
        
        # Act
        alerts = await manager.get_risk_alerts(user_id)
        
        # Assert
        assert isinstance(alerts, list)
        for alert in alerts:
            assert "type" in alert
            assert "message" in alert
            assert "severity" in alert
            assert "timestamp" in alert

    @pytest.mark.asyncio
    async def test_calculate_correlation_matrix(self, database_manager, sample_positions):
        """测试计算相关性矩阵"""
        # Arrange
        manager = RiskManager(database_manager)
        symbols = [pos["symbol"] for pos in sample_positions]
        
        # Act
        correlation_matrix = await manager.calculate_correlation_matrix(symbols)
        
        # Assert
        assert isinstance(correlation_matrix, dict)
        assert len(correlation_matrix) == len(symbols)
        for symbol in symbols:
            assert symbol in correlation_matrix
            assert isinstance(correlation_matrix[symbol], dict)


# 测试数据fixtures
@pytest_asyncio.fixture
async def database_manager():
    """数据库管理器fixture"""
    db_manager = DatabaseManager("sqlite://:memory:")
    await db_manager.initialize()
    yield db_manager
    await db_manager.close()


@pytest.fixture
def sample_positions():
    """示例持仓数据"""
    return [
        {
            "symbol": "000001.SZ",
            "quantity": 1000,
            "average_price": 15.50
        },
        {
            "symbol": "000002.SZ", 
            "quantity": 2000,
            "average_price": 25.30
        },
        {
            "symbol": "600000.SH",
            "quantity": 500,
            "average_price": 8.90
        }
    ]


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
            "user_id": 1
        }
    ]
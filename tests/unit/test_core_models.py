"""
核心数据模型单元测试

测试核心数据模型：MarketData, Order, Position, Portfolio
按照TDD原则，先编写完整的单元测试，确保测试全部失败，然后实现功能代码
"""

import pytest
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional
from unittest.mock import Mock, patch

# 由于模型还未实现，这些导入将会失败，这是预期的TDD行为
try:
    from myQuant.core.models.market_data import MarketData, RealTimeQuote, TechnicalIndicators
    from myQuant.core.models.orders import Order, OrderRequest, OrderStatus, OrderType, OrderSide
    from myQuant.core.models.positions import Position, PositionStatus
    from myQuant.core.models.portfolio import Portfolio, PortfolioSummary, PerformanceMetrics
except ImportError:
    # TDD阶段，模型还未实现，跳过导入错误
    pass


class TestMarketData:
    """MarketData模型测试"""

    def test_market_data_creation(self):
        """测试MarketData创建"""
        # Arrange
        symbol = "000001.SZ"
        timestamp = datetime.now(timezone.utc)
        open_price = Decimal("12.50")
        high_price = Decimal("12.80")
        low_price = Decimal("12.30")
        close_price = Decimal("12.75")
        volume = 1000000
        turnover = Decimal("12650000.00")

        # Act
        market_data = MarketData(
            symbol=symbol,
            timestamp=timestamp,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume,
            turnover=turnover
        )

        # Assert
        assert market_data.symbol == symbol
        assert market_data.timestamp == timestamp
        assert market_data.open == open_price
        assert market_data.high == high_price
        assert market_data.low == low_price
        assert market_data.close == close_price
        assert market_data.volume == volume
        assert market_data.turnover == turnover

    def test_market_data_validation(self):
        """测试MarketData数据验证"""
        # Test invalid price relationships
        with pytest.raises(ValueError):
            MarketData(
                symbol="000001.SZ",
                timestamp=datetime.now(timezone.utc),
                open=Decimal("12.50"),
                high=Decimal("12.00"),  # high < open
                low=Decimal("12.30"),
                close=Decimal("12.75"),
                volume=1000000,
                turnover=Decimal("12650000.00")
            )

        # Test negative volume
        with pytest.raises(ValueError):
            MarketData(
                symbol="000001.SZ",
                timestamp=datetime.now(timezone.utc),
                open=Decimal("12.50"),
                high=Decimal("12.80"),
                low=Decimal("12.30"),
                close=Decimal("12.75"),
                volume=-1000,  # negative volume
                turnover=Decimal("12650000.00")
            )

        # Test invalid symbol format
        with pytest.raises(ValueError):
            MarketData(
                symbol="INVALID",  # invalid symbol format
                timestamp=datetime.now(timezone.utc),
                open=Decimal("12.50"),
                high=Decimal("12.80"),
                low=Decimal("12.30"),
                close=Decimal("12.75"),
                volume=1000000,
                turnover=Decimal("12650000.00")
            )

    def test_market_data_price_change(self):
        """测试价格变动计算"""
        # Arrange
        current_data = MarketData(
            symbol="000001.SZ",
            timestamp=datetime.now(timezone.utc),
            open=Decimal("12.50"),
            high=Decimal("12.80"),
            low=Decimal("12.30"),
            close=Decimal("12.75"),
            volume=1000000,
            turnover=Decimal("12650000.00")
        )
        
        previous_close = Decimal("12.50")

        # Act
        change_amount = current_data.calculate_change_amount(previous_close)
        change_percent = current_data.calculate_change_percent(previous_close)

        # Assert
        assert change_amount == Decimal("0.25")
        assert change_percent == Decimal("2.00")

    def test_real_time_quote_creation(self):
        """测试实时行情创建"""
        # Arrange
        symbol = "000001.SZ"
        current_price = Decimal("12.75")
        bid_price_1 = Decimal("12.74")
        bid_volume_1 = 50000
        ask_price_1 = Decimal("12.75")
        ask_volume_1 = 30000

        # Act
        quote = RealTimeQuote(
            symbol=symbol,
            current_price=current_price,
            bid_price_1=bid_price_1,
            bid_volume_1=bid_volume_1,
            ask_price_1=ask_price_1,
            ask_volume_1=ask_volume_1,
            timestamp=datetime.now(timezone.utc)
        )

        # Assert
        assert quote.symbol == symbol
        assert quote.current_price == current_price
        assert quote.bid_price_1 == bid_price_1
        assert quote.bid_volume_1 == bid_volume_1
        assert quote.ask_price_1 == ask_price_1
        assert quote.ask_volume_1 == ask_volume_1

    def test_technical_indicators_calculation(self):
        """测试技术指标计算"""
        # Arrange
        prices = [Decimal(str(price)) for price in [10.0, 10.5, 11.0, 10.8, 11.2, 11.5, 11.3, 12.0, 12.2, 11.9]]

        # Act & Assert
        # 移动平均线
        ma_5 = TechnicalIndicators.calculate_ma(prices, 5)
        assert len(ma_5) == 6  # 10个数据点，5日均线有6个有效值
        assert ma_5[-1] == Decimal("11.78")  # 最后5个数据的平均值

        # RSI
        rsi = TechnicalIndicators.calculate_rsi(prices, 14)
        assert 0 <= rsi[-1] <= 100  # RSI值在0-100之间

        # MACD
        macd_data = TechnicalIndicators.calculate_macd(prices)
        assert "macd" in macd_data
        assert "signal" in macd_data
        assert "histogram" in macd_data


class TestOrder:
    """Order模型测试"""

    def test_order_creation(self):
        """测试Order创建"""
        # Arrange
        order_data = {
            "symbol": "000001.SZ",
            "side": OrderSide.BUY,
            "order_type": OrderType.LIMIT,
            "quantity": 1000,
            "price": Decimal("12.50"),
            "user_id": 1
        }

        # Act
        order = Order(**order_data)

        # Assert
        assert order.symbol == "000001.SZ"
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.LIMIT
        assert order.quantity == 1000
        assert order.price == Decimal("12.50")
        assert order.user_id == 1
        assert order.status == OrderStatus.CREATED
        assert order.filled_quantity == 0
        assert order.order_id is not None

    def test_market_order_creation(self):
        """测试市价单创建"""
        # Arrange & Act
        order = Order(
            symbol="000001.SZ",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1000,
            user_id=1
        )

        # Assert
        assert order.order_type == OrderType.MARKET
        assert order.price is None  # 市价单没有价格

    def test_order_validation(self):
        """测试Order数据验证"""
        # Test invalid quantity
        with pytest.raises(ValueError):
            Order(
                symbol="000001.SZ",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=0,  # invalid quantity
                price=Decimal("12.50"),
                user_id=1
            )

        # Test limit order without price
        with pytest.raises(ValueError):
            Order(
                symbol="000001.SZ",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=1000,
                price=None,  # limit order must have price
                user_id=1
            )

        # Test invalid symbol format
        with pytest.raises(ValueError):
            Order(
                symbol="INVALID",  # invalid symbol
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=1000,
                price=Decimal("12.50"),
                user_id=1
            )

    def test_order_fill(self):
        """测试订单成交"""
        # Arrange
        order = Order(
            symbol="000001.SZ",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1000,
            price=Decimal("12.50"),
            user_id=1
        )

        # Act - 部分成交
        fill_result = order.fill(500, Decimal("12.48"))

        # Assert
        assert fill_result is True
        assert order.filled_quantity == 500
        assert order.status == OrderStatus.PARTIALLY_FILLED
        assert order.average_fill_price == Decimal("12.48")

        # Act - 完全成交
        fill_result = order.fill(500, Decimal("12.52"))

        # Assert
        assert fill_result is True
        assert order.filled_quantity == 1000
        assert order.status == OrderStatus.FILLED
        # 加权平均价格
        expected_avg = (Decimal("12.48") * 500 + Decimal("12.52") * 500) / 1000
        assert order.average_fill_price == expected_avg

    def test_order_cancel(self):
        """测试订单撤销"""
        # Arrange
        order = Order(
            symbol="000001.SZ",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1000,
            price=Decimal("12.50"),
            user_id=1
        )

        # Act
        cancel_result = order.cancel()

        # Assert
        assert cancel_result is True
        assert order.status == OrderStatus.CANCELLED

        # Test cannot cancel filled order
        filled_order = Order(
            symbol="000001.SZ",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1000,
            price=Decimal("12.50"),
            user_id=1
        )
        filled_order.fill(1000, Decimal("12.50"))
        
        cancel_result = filled_order.cancel()
        assert cancel_result is False
        assert filled_order.status == OrderStatus.FILLED

    def test_order_remaining_quantity(self):
        """测试订单剩余数量计算"""
        # Arrange
        order = Order(
            symbol="000001.SZ",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1000,
            price=Decimal("12.50"),
            user_id=1
        )

        # Act & Assert
        assert order.remaining_quantity == 1000

        order.fill(300, Decimal("12.48"))
        assert order.remaining_quantity == 700

        order.fill(700, Decimal("12.52"))
        assert order.remaining_quantity == 0


class TestPosition:
    """Position模型测试"""

    def test_position_creation(self):
        """测试Position创建"""
        # Arrange & Act
        position = Position(
            user_id=1,
            symbol="000001.SZ",
            quantity=1000,
            average_price=Decimal("12.00")
        )

        # Assert
        assert position.user_id == 1
        assert position.symbol == "000001.SZ"
        assert position.quantity == 1000
        assert position.average_price == Decimal("12.00")
        assert position.status == PositionStatus.ACTIVE

    def test_position_market_value_calculation(self):
        """测试持仓市值计算"""
        # Arrange
        position = Position(
            user_id=1,
            symbol="000001.SZ",
            quantity=1000,
            average_price=Decimal("12.00")
        )
        current_price = Decimal("12.75")

        # Act
        market_value = position.calculate_market_value(current_price)

        # Assert
        assert market_value == Decimal("12750.00")

    def test_position_unrealized_pnl_calculation(self):
        """测试未实现盈亏计算"""
        # Arrange
        position = Position(
            user_id=1,
            symbol="000001.SZ",
            quantity=1000,
            average_price=Decimal("12.00")
        )
        current_price = Decimal("12.75")

        # Act
        unrealized_pnl = position.calculate_unrealized_pnl(current_price)

        # Assert
        assert unrealized_pnl == Decimal("750.00")

    def test_position_percentage_calculation(self):
        """测试持仓比例计算"""
        # Arrange
        position = Position(
            user_id=1,
            symbol="000001.SZ",
            quantity=1000,
            average_price=Decimal("12.00")
        )
        current_price = Decimal("12.75")
        total_portfolio_value = Decimal("100000.00")

        # Act
        percentage = position.calculate_percentage(current_price, total_portfolio_value)

        # Assert
        assert percentage == Decimal("12.75")  # 12750 / 100000 * 100

    def test_position_update(self):
        """测试持仓更新"""
        # Arrange
        position = Position(
            user_id=1,
            symbol="000001.SZ",
            quantity=1000,
            average_price=Decimal("12.00")
        )

        # Act - 增加持仓
        position.update_position(500, Decimal("13.00"))

        # Assert
        assert position.quantity == 1500
        # 加权平均成本: (1000*12.00 + 500*13.00) / 1500 = 12.33
        expected_avg = (Decimal("12.00") * 1000 + Decimal("13.00") * 500) / 1500
        assert position.average_price == expected_avg

        # Act - 减少持仓
        position.update_position(-500, Decimal("13.50"))

        # Assert
        assert position.quantity == 1000
        # 减仓不改变平均成本
        assert position.average_price == expected_avg

    def test_position_close(self):
        """测试平仓"""
        # Arrange
        position = Position(
            user_id=1,
            symbol="000001.SZ",
            quantity=1000,
            average_price=Decimal("12.00")
        )

        # Act
        close_result = position.close(Decimal("12.75"))

        # Assert
        assert close_result is True
        assert position.quantity == 0
        assert position.status == PositionStatus.CLOSED
        assert position.realized_pnl == Decimal("750.00")


class TestPortfolio:
    """Portfolio模型测试"""

    @pytest.fixture
    def sample_positions(self):
        """示例持仓数据"""
        return [
            Position(1, "000001.SZ", 1000, Decimal("12.00")),
            Position(1, "000002.SZ", 500, Decimal("25.00")),
            Position(1, "600000.SH", 800, Decimal("8.50"))
        ]

    @pytest.fixture
    def sample_prices(self):
        """示例价格数据"""
        return {
            "000001.SZ": Decimal("12.75"),
            "000002.SZ": Decimal("26.00"),
            "600000.SH": Decimal("8.80")
        }

    def test_portfolio_creation(self):
        """测试Portfolio创建"""
        # Arrange & Act
        portfolio = Portfolio(
            user_id=1,
            initial_capital=Decimal("1000000.00")
        )

        # Assert
        assert portfolio.user_id == 1
        assert portfolio.initial_capital == Decimal("1000000.00")
        assert portfolio.cash_balance == Decimal("1000000.00")
        assert len(portfolio.positions) == 0

    def test_portfolio_add_position(self, sample_positions):
        """测试添加持仓"""
        # Arrange
        portfolio = Portfolio(
            user_id=1,
            initial_capital=Decimal("1000000.00")
        )

        # Act
        for position in sample_positions:
            result = portfolio.add_position(position)
            assert result is True

        # Assert
        assert len(portfolio.positions) == 3
        assert "000001.SZ" in portfolio.positions
        assert "000002.SZ" in portfolio.positions
        assert "600000.SH" in portfolio.positions

    def test_portfolio_calculate_total_value(self, sample_positions, sample_prices):
        """测试投资组合总价值计算"""
        # Arrange
        portfolio = Portfolio(
            user_id=1,
            initial_capital=Decimal("1000000.00")
        )
        portfolio.cash_balance = Decimal("975500.00")  # 剩余现金
        
        for position in sample_positions:
            portfolio.add_position(position)

        # Act
        total_value = portfolio.calculate_total_value(sample_prices)

        # Assert
        # 000001.SZ: 1000 * 12.75 = 12750
        # 000002.SZ: 500 * 26.00 = 13000  
        # 600000.SH: 800 * 8.80 = 7040
        # 现金: 975500
        # 总计: 1008290
        expected_total = Decimal("1008290.00")
        assert total_value == expected_total

    def test_portfolio_calculate_total_pnl(self, sample_positions, sample_prices):
        """测试总盈亏计算"""
        # Arrange
        portfolio = Portfolio(
            user_id=1,
            initial_capital=Decimal("1000000.00")
        )
        
        for position in sample_positions:
            portfolio.add_position(position)

        # Act
        total_pnl = portfolio.calculate_total_pnl(sample_prices)

        # Assert
        # 000001.SZ: (12.75 - 12.00) * 1000 = 750
        # 000002.SZ: (26.00 - 25.00) * 500 = 500
        # 600000.SH: (8.80 - 8.50) * 800 = 240
        # 总计: 1490
        expected_pnl = Decimal("1490.00")
        assert total_pnl == expected_pnl

    def test_portfolio_get_summary(self, sample_positions, sample_prices):
        """测试投资组合摘要"""
        # Arrange
        portfolio = Portfolio(
            user_id=1,
            initial_capital=Decimal("1000000.00")
        )
        portfolio.cash_balance = Decimal("975500.00")
        
        for position in sample_positions:
            portfolio.add_position(position)

        # Act
        summary = portfolio.get_summary(sample_prices)

        # Assert
        assert isinstance(summary, PortfolioSummary)
        assert summary.total_value == Decimal("1008290.00")
        assert summary.cash_balance == Decimal("975500.00")
        assert summary.position_value == Decimal("32790.00")
        assert summary.total_pnl == Decimal("1490.00")
        assert summary.total_return == Decimal("0.15")  # 1490/1000000*100 = 0.15%
        assert summary.positions_count == 3

    def test_portfolio_update_cash_balance(self):
        """测试现金余额更新"""
        # Arrange
        portfolio = Portfolio(
            user_id=1,
            initial_capital=Decimal("1000000.00")
        )

        # Act - 减少现金（买入股票）
        result = portfolio.update_cash_balance(Decimal("-24500.00"))

        # Assert
        assert result is True
        assert portfolio.cash_balance == Decimal("975500.00")

        # Act - 增加现金（卖出股票）
        result = portfolio.update_cash_balance(Decimal("12750.00"))

        # Assert
        assert result is True
        assert portfolio.cash_balance == Decimal("988250.00")

        # Test insufficient cash
        result = portfolio.update_cash_balance(Decimal("-2000000.00"))
        assert result is False
        assert portfolio.cash_balance == Decimal("988250.00")  # 余额不变

    def test_portfolio_performance_metrics(self, sample_positions, sample_prices):
        """测试绩效指标计算"""
        # Arrange
        portfolio = Portfolio(
            user_id=1,
            initial_capital=Decimal("1000000.00")
        )
        
        for position in sample_positions:
            portfolio.add_position(position)

        # Mock历史数据
        historical_values = [
            Decimal("1000000.00"),
            Decimal("1005000.00"),
            Decimal("1002000.00"),
            Decimal("1008290.00")
        ]

        # Act
        metrics = portfolio.calculate_performance_metrics(sample_prices, historical_values)

        # Assert
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_return > 0
        assert metrics.volatility >= 0
        assert metrics.max_drawdown <= 0
        assert metrics.sharpe_ratio is not None


if __name__ == "__main__":
    # 运行测试确保全部失败（TDD第一步）
    pytest.main([__file__, "-v", "--tb=short"])
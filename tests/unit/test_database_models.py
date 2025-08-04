"""
数据库模型测试

测试各个数据模型的字段验证、数据类型转换、边界情况等
"""

import pytest
from datetime import datetime
from decimal import Decimal
from dataclasses import fields, asdict
from typing import Optional

from myQuant.infrastructure.database.models import (
    StockTable, KlineDailyTable, RealTimeQuoteTable,
    OrderTable, PositionTable, TransactionTable,
    UserTable, UserConfigTable, StrategyTable,
    AlertTable, RiskMetricTable
)


class TestDatabaseModels:
    """数据库模型测试类"""
    
    def test_stock_table_default_values(self):
        """测试股票表模型默认值"""
        stock = StockTable(symbol="000001.SZ", name="平安银行")
        
        assert stock.symbol == "000001.SZ"
        assert stock.name == "平安银行"
        assert stock.market == "SZ"  # 默认值
        assert stock.sector is None
        assert stock.industry is None
        assert stock.listing_date is None
        assert stock.total_shares is None
        assert stock.float_shares is None
        assert stock.created_at is None
    
    def test_stock_table_full_initialization(self):
        """测试股票表模型完整初始化"""
        created_time = datetime.now()
        stock = StockTable(
            symbol="600036.SH",
            name="招商银行",
            sector="金融",
            industry="银行",
            market="SH",
            listing_date="2002-04-09",
            total_shares=25219845601,
            float_shares=20628944429,
            created_at=created_time
        )
        
        assert stock.symbol == "600036.SH"
        assert stock.name == "招商银行"
        assert stock.sector == "金融"
        assert stock.industry == "银行"
        assert stock.market == "SH"
        assert stock.listing_date == "2002-04-09"
        assert stock.total_shares == 25219845601
        assert stock.float_shares == 20628944429
        assert stock.created_at == created_time
    
    def test_kline_daily_table_default_values(self):
        """测试K线数据表模型默认值"""
        kline = KlineDailyTable()
        
        assert kline.id is None
        assert kline.symbol == ""
        assert kline.trade_date == ""
        assert kline.open_price == 0.0
        assert kline.high_price == 0.0
        assert kline.low_price == 0.0
        assert kline.close_price == 0.0
        assert kline.volume == 0
        assert kline.turnover == 0.0
    
    def test_kline_daily_table_with_data(self):
        """测试K线数据表模型数据验证"""
        kline = KlineDailyTable(
            symbol="000001.SZ",
            trade_date="2024-01-15",
            open_price=10.25,
            high_price=10.88,
            low_price=10.15,
            close_price=10.75,
            volume=125000000,
            turnover=1340000000.50
        )
        
        assert kline.symbol == "000001.SZ"
        assert kline.trade_date == "2024-01-15"
        assert kline.open_price == 10.25
        assert kline.high_price == 10.88
        assert kline.low_price == 10.15
        assert kline.close_price == 10.75
        assert kline.volume == 125000000
        assert kline.turnover == 1340000000.50
    
    def test_real_time_quote_table(self):
        """测试实时行情表模型"""
        last_updated = datetime.now()
        quote = RealTimeQuoteTable(
            symbol="000001.SZ",
            current_price=10.85,
            change_amount=0.10,
            change_percent=0.93,
            volume=85000000,
            turnover=920000000.0,
            bid_price_1=10.84,
            bid_volume_1=12000,
            ask_price_1=10.85,
            ask_volume_1=8500,
            last_updated=last_updated
        )
        
        assert quote.symbol == "000001.SZ"
        assert quote.current_price == 10.85
        assert quote.change_amount == 0.10
        assert quote.change_percent == 0.93
        assert quote.volume == 85000000
        assert quote.turnover == 920000000.0
        assert quote.bid_price_1 == 10.84
        assert quote.bid_volume_1 == 12000
        assert quote.ask_price_1 == 10.85
        assert quote.ask_volume_1 == 8500
        assert quote.last_updated == last_updated
    
    def test_order_table_default_values(self):
        """测试订单表模型默认值"""
        order = OrderTable(
            id="ORD-001",
            user_id=1,
            symbol="000001.SZ",
            order_type="LIMIT",
            side="BUY",
            quantity=1000
        )
        
        assert order.id == "ORD-001"
        assert order.user_id == 1
        assert order.symbol == "000001.SZ"
        assert order.order_type == "LIMIT"
        assert order.side == "BUY"
        assert order.quantity == 1000
        assert order.price is None
        assert order.stop_price is None
        assert order.time_in_force == "DAY"  # 默认值
        assert order.filled_quantity == 0  # 默认值
        assert order.average_fill_price is None
        assert order.status == "PENDING"  # 默认值
        assert order.created_at is None
        assert order.updated_at is None
    
    def test_order_table_full_data(self):
        """测试订单表模型完整数据"""
        created_time = datetime.now()
        order = OrderTable(
            id="ORD-002",
            user_id=1,
            symbol="600036.SH",
            order_type="STOP",
            side="SELL",
            quantity=500,
            price=35.50,
            stop_price=35.00,
            time_in_force="GTC",
            filled_quantity=500,
            average_fill_price=35.48,
            status="FILLED",
            created_at=created_time,
            updated_at=created_time
        )
        
        assert order.price == 35.50
        assert order.stop_price == 35.00
        assert order.time_in_force == "GTC"
        assert order.filled_quantity == 500
        assert order.average_fill_price == 35.48
        assert order.status == "FILLED"
    
    def test_position_table(self):
        """测试持仓表模型"""
        position = PositionTable(
            id=1,
            user_id=1,
            symbol="000001.SZ",
            quantity=2000,
            average_price=10.55
        )
        
        assert position.id == 1
        assert position.user_id == 1
        assert position.symbol == "000001.SZ"
        assert position.quantity == 2000
        assert position.average_price == 10.55
    
    def test_transaction_table(self):
        """测试交易记录表模型"""
        executed_time = datetime.now()
        transaction = TransactionTable(
            id=1,
            order_id="ORD-001",
            user_id=1,
            symbol="000001.SZ",
            side="BUY",
            quantity=1000,
            price=10.50,
            commission=5.25,
            executed_at=executed_time
        )
        
        assert transaction.id == 1
        assert transaction.order_id == "ORD-001"
        assert transaction.user_id == 1
        assert transaction.symbol == "000001.SZ"
        assert transaction.side == "BUY"
        assert transaction.quantity == 1000
        assert transaction.price == 10.50
        assert transaction.commission == 5.25
        assert transaction.executed_at == executed_time
    
    def test_user_table(self):
        """测试用户表模型"""
        created_time = datetime.now()
        user = UserTable(
            id=1,
            username="testuser",
            email="test@example.com",
            password_hash="hashed_password_123",
            created_at=created_time,
            updated_at=created_time,
            is_active=True
        )
        
        assert user.id == 1
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.password_hash == "hashed_password_123"
        assert user.created_at == created_time
        assert user.updated_at == created_time
        assert user.is_active is True
    
    def test_user_config_table(self):
        """测试用户配置表模型"""
        config = UserConfigTable(
            user_id=1,
            risk_tolerance=0.03,
            max_position_size=0.15,
            notification_settings='{"email": true, "sms": false}',
            trading_preferences='{"theme": "dark", "language": "zh-CN"}'
        )
        
        assert config.user_id == 1
        assert config.risk_tolerance == 0.03
        assert config.max_position_size == 0.15
        assert config.notification_settings == '{"email": true, "sms": false}'
        assert config.trading_preferences == '{"theme": "dark", "language": "zh-CN"}'
    
    def test_strategy_table(self):
        """测试策略配置表模型"""
        created_time = datetime.now()
        strategy = StrategyTable(
            id=1,
            user_id=1,
            name="MA Crossover",
            type="TECHNICAL",
            parameters='{"fast_period": 5, "slow_period": 20}',
            is_active=True,
            created_at=created_time,
            updated_at=created_time
        )
        
        assert strategy.id == 1
        assert strategy.user_id == 1
        assert strategy.name == "MA Crossover"
        assert strategy.type == "TECHNICAL"
        assert strategy.parameters == '{"fast_period": 5, "slow_period": 20}'
        assert strategy.is_active is True
    
    def test_alert_table(self):
        """测试提醒设置表模型"""
        created_time = datetime.now()
        triggered_time = datetime.now()
        alert = AlertTable(
            id=1,
            user_id=1,
            symbol="000001.SZ",
            alert_type="PRICE",
            condition_type="ABOVE",
            threshold_value=11.00,
            is_active=True,
            created_at=created_time,
            triggered_at=triggered_time
        )
        
        assert alert.id == 1
        assert alert.user_id == 1
        assert alert.symbol == "000001.SZ"
        assert alert.alert_type == "PRICE"
        assert alert.condition_type == "ABOVE"
        assert alert.threshold_value == 11.00
        assert alert.is_active is True
        assert alert.created_at == created_time
        assert alert.triggered_at == triggered_time
    
    def test_risk_metric_table(self):
        """测试风险管理表模型"""
        calculated_time = datetime.now()
        risk_metric = RiskMetricTable(
            id=1,
            user_id=1,
            date="2024-01-15",
            portfolio_value=1000000.00,
            daily_pnl=5000.00,
            max_drawdown=0.08,
            var_95=25000.00,
            beta=1.15,
            sharpe_ratio=1.85,
            calculated_at=calculated_time
        )
        
        assert risk_metric.id == 1
        assert risk_metric.user_id == 1
        assert risk_metric.date == "2024-01-15"
        assert risk_metric.portfolio_value == 1000000.00
        assert risk_metric.daily_pnl == 5000.00
        assert risk_metric.max_drawdown == 0.08
        assert risk_metric.var_95 == 25000.00
        assert risk_metric.beta == 1.15
        assert risk_metric.sharpe_ratio == 1.85
        assert risk_metric.calculated_at == calculated_time
    
    def test_dataclass_conversion(self):
        """测试数据类转换功能"""
        # 创建一个模型实例
        stock = StockTable(
            symbol="000002.SZ",
            name="万科A",
            sector="房地产",
            market="SZ"
        )
        
        # 转换为字典
        stock_dict = asdict(stock)
        
        assert isinstance(stock_dict, dict)
        assert stock_dict['symbol'] == "000002.SZ"
        assert stock_dict['name'] == "万科A"
        assert stock_dict['sector'] == "房地产"
        assert stock_dict['market'] == "SZ"
        assert stock_dict['industry'] is None
    
    def test_model_field_types(self):
        """测试模型字段类型定义"""
        # 获取OrderTable的字段信息
        order_fields = fields(OrderTable)
        
        # 验证字段类型
        field_types = {field.name: field.type for field in order_fields}
        
        assert field_types['id'] == str
        assert field_types['user_id'] == int
        assert field_types['symbol'] == str
        assert field_types['quantity'] == int
        assert field_types['price'] == Optional[float]
        assert field_types['status'] == str
    
    def test_edge_cases_numeric_values(self):
        """测试数值类型边界情况"""
        # 测试大数值
        kline = KlineDailyTable(
            symbol="BRK-A",
            trade_date="2024-01-15",
            open_price=545000.00,  # 伯克希尔哈撒韦A股价格
            high_price=548000.00,
            low_price=544000.00,
            close_price=547000.00,
            volume=150,  # 交易量很小
            turnover=82050000.00
        )
        
        assert kline.open_price == 545000.00
        assert kline.volume == 150
        
        # 测试小数值
        risk = RiskMetricTable(
            user_id=1,
            date="2024-01-15",
            beta=0.001,  # 极低的贝塔值
            sharpe_ratio=-2.5  # 负的夏普比率
        )
        
        assert risk.beta == 0.001
        assert risk.sharpe_ratio == -2.5
    
    def test_string_field_edge_cases(self):
        """测试字符串字段边界情况"""
        # 测试长字符串
        long_name = "这是一个非常长的股票名称用于测试字段长度限制" * 5
        stock = StockTable(
            symbol="TEST001",
            name=long_name[:100]  # 假设数据库限制100字符
        )
        
        assert len(stock.name) == 100
        
        # 测试特殊字符
        special_params = '{"key": "value with \"quotes\" and \\backslash"}'
        strategy = StrategyTable(
            user_id=1,
            name="Test Strategy",
            type="CUSTOM",
            parameters=special_params
        )
        
        assert strategy.parameters == special_params
    
    def test_default_value_consistency(self):
        """测试默认值的一致性"""
        # 创建多个实例验证默认值一致
        order1 = OrderTable("O1", 1, "A", "MARKET", "BUY", 100)
        order2 = OrderTable("O2", 2, "B", "LIMIT", "SELL", 200)
        
        # 验证默认值相同
        assert order1.status == order2.status == "PENDING"
        assert order1.time_in_force == order2.time_in_force == "DAY"
        assert order1.filled_quantity == order2.filled_quantity == 0
    
    def test_optional_fields_none_handling(self):
        """测试可选字段的None处理"""
        # 创建只有必需字段的实例
        user = UserTable(username="minimal_user", password_hash="hash123")
        
        # 验证可选字段为None或默认值
        assert user.id is None
        assert user.email is None
        assert user.created_at is None
        assert user.updated_at is None
        assert user.is_active is True  # 有默认值
    
    def test_json_field_validation(self):
        """测试JSON字段的有效性"""
        # 有效的JSON字符串
        valid_json = '{"alerts": ["email", "sms"], "frequency": "daily"}'
        config = UserConfigTable(
            user_id=1,
            notification_settings=valid_json
        )
        
        assert config.notification_settings == valid_json
        
        # 验证可以存储复杂的JSON结构
        complex_json = '''
        {
            "strategies": {
                "ma_cross": {"enabled": true, "params": {"fast": 5, "slow": 20}},
                "rsi": {"enabled": false, "params": {"period": 14, "overbought": 70}}
            },
            "risk_rules": [
                {"type": "max_loss", "value": 0.02},
                {"type": "position_size", "value": 0.1}
            ]
        }
        '''
        config.trading_preferences = complex_json.strip()
        assert config.trading_preferences == complex_json.strip()
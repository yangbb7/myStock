# -*- coding: utf-8 -*-
"""
测试异常模块
"""

import pytest
from unittest.mock import Mock, patch
from myQuant.core.exceptions import (
    MyQuantException,
    DataException,
    DataSourceException,
    DataValidationException,
    DataMissingException,
    StrategyException,
    StrategyInitializationException,
    StrategyExecutionException,
    SignalException,
    RiskException,
    PositionSizeException,
    DrawdownException,
    VaRException,
    OrderException,
    OrderExecutionException,
    OrderValidationException,
    PortfolioException,
    InsufficientFundsException,
    PositionException,
    BacktestException,
    ConfigurationException,
    APIException,
    NetworkException,
    ProcessingException,
    MonitoringException,
    ExceptionFactory,
    GlobalExceptionHandler,
    handle_exceptions
)


class TestMyQuantExceptions:
    """测试自定义异常类"""
    
    def test_myquant_exception_base(self):
        """测试基础异常类"""
        error_msg = "Test error message"
        exc = MyQuantException(error_msg)
        
        assert str(exc) == error_msg
        assert isinstance(exc, Exception)
        assert exc.message == error_msg
        assert exc.timestamp is not None
        assert exc.error_code == "MYQUANT_ERROR"
        assert exc.context == {}
    
    def test_myquant_exception_with_context(self):
        """测试带上下文的异常"""
        error_msg = "Test error"
        context = {"symbol": "AAPL", "price": 150.0}
        exc = MyQuantException(error_msg, context=context)
        
        assert exc.context == context
        assert "symbol" in exc.context
        assert exc.context["symbol"] == "AAPL"
    
    def test_data_exception_hierarchy(self):
        """测试数据异常继承关系"""
        exc = DataException("Data error")
        
        assert isinstance(exc, MyQuantException)
        assert isinstance(exc, DataException)
        assert exc.error_code == "DATA_ERROR"
    
    def test_data_source_exception(self):
        """测试数据源异常"""
        exc = DataSourceException("API failed", data_source="Yahoo")
        
        assert isinstance(exc, DataException)
        assert exc.data_source == "Yahoo"
        assert exc.error_code == "DATA_SOURCE_ERROR"
    
    def test_data_validation_exception(self):
        """测试数据验证异常"""
        exc = DataValidationException("Invalid data", field="price", value=-10)
        
        assert isinstance(exc, DataException)
        assert exc.field == "price"
        assert exc.value == -10
        assert exc.error_code == "DATA_VALIDATION_ERROR"
    
    def test_data_missing_exception(self):
        """测试数据缺失异常"""
        exc = DataMissingException("Missing data", symbol="AAPL", date_range="2023-01-01 to 2023-12-31")
        
        assert isinstance(exc, DataException)
        assert exc.symbol == "AAPL"
        assert exc.date_range == "2023-01-01 to 2023-12-31"
        assert exc.error_code == "DATA_MISSING_ERROR"
    
    def test_strategy_exceptions(self):
        """测试策略异常"""
        # 策略异常
        exc = StrategyException("Strategy error")
        assert isinstance(exc, MyQuantException)
        assert exc.error_code == "STRATEGY_ERROR"
        
        # 策略初始化异常
        exc = StrategyInitializationException("Init failed", strategy_name="TestStrategy")
        assert isinstance(exc, StrategyException)
        assert exc.strategy_name == "TestStrategy"
        assert exc.error_code == "STRATEGY_INITIALIZATION_ERROR"
        
        # 策略执行异常
        exc = StrategyExecutionException("Execution failed", strategy_name="TestStrategy", step="signal_generation")
        assert isinstance(exc, StrategyException)
        assert exc.strategy_name == "TestStrategy"
        assert exc.step == "signal_generation"
        assert exc.error_code == "STRATEGY_EXECUTION_ERROR"
        
        # 信号异常
        exc = SignalException("Invalid signal", signal_type="BUY", symbol="AAPL")
        assert isinstance(exc, StrategyException)
        assert exc.signal_type == "BUY"
        assert exc.symbol == "AAPL"
        assert exc.error_code == "SIGNAL_ERROR"
    
    def test_risk_exceptions(self):
        """测试风险异常"""
        # 风险异常
        exc = RiskException("Risk error")
        assert isinstance(exc, MyQuantException)
        assert exc.error_code == "RISK_ERROR"
        
        # 持仓规模异常
        exc = PositionSizeException("Position too large", symbol="AAPL", size=1000000)
        assert isinstance(exc, RiskException)
        assert exc.symbol == "AAPL"
        assert exc.size == 1000000
        assert exc.error_code == "POSITION_SIZE_ERROR"
        
        # 回撤异常
        exc = DrawdownException("Drawdown exceeded", current_drawdown=0.25, max_drawdown=0.20)
        assert isinstance(exc, RiskException)
        assert exc.current_drawdown == 0.25
        assert exc.max_drawdown == 0.20
        assert exc.error_code == "DRAWDOWN_ERROR"
        
        # VaR异常
        exc = VaRException("VaR exceeded", current_var=0.15, max_var=0.10)
        assert isinstance(exc, RiskException)
        assert exc.current_var == 0.15
        assert exc.max_var == 0.10
        assert exc.error_code == "VAR_ERROR"
    
    def test_order_exceptions(self):
        """测试订单异常"""
        # 订单异常
        exc = OrderException("Order error")
        assert isinstance(exc, MyQuantException)
        assert exc.error_code == "ORDER_ERROR"
        
        # 订单执行异常
        exc = OrderExecutionException("Execution failed", order_id="12345", reason="Insufficient funds")
        assert isinstance(exc, OrderException)
        assert exc.order_id == "12345"
        assert exc.reason == "Insufficient funds"
        assert exc.error_code == "ORDER_EXECUTION_ERROR"
        
        # 订单验证异常
        exc = OrderValidationException("Invalid order", field="quantity", value=-10)
        assert isinstance(exc, OrderException)
        assert exc.field == "quantity"
        assert exc.value == -10
        assert exc.error_code == "ORDER_VALIDATION_ERROR"
    
    def test_portfolio_exceptions(self):
        """测试投资组合异常"""
        # 投资组合异常
        exc = PortfolioException("Portfolio error")
        assert isinstance(exc, MyQuantException)
        assert exc.error_code == "PORTFOLIO_ERROR"
        
        # 资金不足异常
        exc = InsufficientFundsException("Not enough funds", required=10000, available=5000)
        assert isinstance(exc, PortfolioException)
        assert exc.required == 10000
        assert exc.available == 5000
        assert exc.error_code == "INSUFFICIENT_FUNDS_ERROR"
        
        # 持仓异常
        exc = PositionException("Position error", symbol="AAPL", position_size=100)
        assert isinstance(exc, PortfolioException)
        assert exc.symbol == "AAPL"
        assert exc.position_size == 100
        assert exc.error_code == "POSITION_ERROR"
    
    def test_other_exceptions(self):
        """测试其他异常"""
        # 回测异常
        exc = BacktestException("Backtest failed", start_date="2023-01-01", end_date="2023-12-31")
        assert isinstance(exc, MyQuantException)
        assert exc.start_date == "2023-01-01"
        assert exc.end_date == "2023-12-31"
        assert exc.error_code == "BACKTEST_ERROR"
        
        # 配置异常
        exc = ConfigurationException("Invalid config", config_key="api_key")
        assert isinstance(exc, MyQuantException)
        assert exc.config_key == "api_key"
        assert exc.error_code == "CONFIGURATION_ERROR"
        
        # API异常
        exc = APIException("API error", endpoint="/data/stock", status_code=500)
        assert isinstance(exc, MyQuantException)
        assert exc.endpoint == "/data/stock"
        assert exc.status_code == 500
        assert exc.error_code == "API_ERROR"
        
        # 网络异常
        exc = NetworkException("Connection failed", url="https://api.example.com", timeout=30)
        assert isinstance(exc, MyQuantException)
        assert exc.url == "https://api.example.com"
        assert exc.timeout == 30
        assert exc.error_code == "NETWORK_ERROR"
        
        # 处理异常
        exc = ProcessingException("Processing failed", operation="data_cleaning", data_size=1000)
        assert isinstance(exc, MyQuantException)
        assert exc.operation == "data_cleaning"
        assert exc.data_size == 1000
        assert exc.error_code == "PROCESSING_ERROR"
        
        # 监控异常
        exc = MonitoringException("Monitoring failed", metric="cpu_usage", value=95.5)
        assert isinstance(exc, MyQuantException)
        assert exc.metric == "cpu_usage"
        assert exc.value == 95.5
        assert exc.error_code == "MONITORING_ERROR"


class TestExceptionFactory:
    """测试异常工厂类"""
    
    def test_create_data_exception(self):
        """测试创建数据异常"""
        exc = ExceptionFactory.create_data_exception("source_error", "API failed", data_source="Yahoo")
        
        assert isinstance(exc, DataSourceException)
        assert exc.data_source == "Yahoo"
        assert str(exc) == "API failed"
    
    def test_create_data_exception_unknown_type(self):
        """测试创建未知类型的数据异常"""
        exc = ExceptionFactory.create_data_exception("unknown_error", "Unknown error")
        
        assert isinstance(exc, DataException)
        assert str(exc) == "Unknown error"
    
    def test_create_strategy_exception(self):
        """测试创建策略异常"""
        exc = ExceptionFactory.create_strategy_exception("execution_error", "Strategy failed", strategy_name="TestStrategy")
        
        assert isinstance(exc, StrategyExecutionException)
        assert exc.strategy_name == "TestStrategy"
        assert str(exc) == "Strategy failed"
    
    def test_create_strategy_exception_unknown_type(self):
        """测试创建未知类型的策略异常"""
        exc = ExceptionFactory.create_strategy_exception("unknown_error", "Unknown error")
        
        assert isinstance(exc, StrategyException)
        assert str(exc) == "Unknown error"
    
    def test_create_risk_exception(self):
        """测试创建风险异常"""
        exc = ExceptionFactory.create_risk_exception("drawdown_error", "Drawdown exceeded", current_drawdown=0.25)
        
        assert isinstance(exc, DrawdownException)
        assert exc.current_drawdown == 0.25
        assert str(exc) == "Drawdown exceeded"
    
    def test_create_risk_exception_unknown_type(self):
        """测试创建未知类型的风险异常"""
        exc = ExceptionFactory.create_risk_exception("unknown_error", "Unknown error")
        
        assert isinstance(exc, RiskException)
        assert str(exc) == "Unknown error"


class TestGlobalExceptionHandler:
    """测试全局异常处理器"""
    
    def test_handle_exception_with_logging(self):
        """测试异常处理并记录日志"""
        handler = GlobalExceptionHandler()
        
        with patch('logging.getLogger') as mock_logger:
            mock_log = Mock()
            mock_logger.return_value = mock_log
            
            exc = ValueError("Test error")
            result = handler.handle_exception(exc, log_error=True)
            
            assert result is False
            mock_log.error.assert_called_once()
    
    def test_handle_exception_without_logging(self):
        """测试异常处理不记录日志"""
        handler = GlobalExceptionHandler()
        
        with patch('logging.getLogger') as mock_logger:
            mock_log = Mock()
            mock_logger.return_value = mock_log
            
            exc = ValueError("Test error")
            result = handler.handle_exception(exc, log_error=False)
            
            assert result is False
            mock_log.error.assert_not_called()
    
    def test_handle_exception_with_context(self):
        """测试异常处理带上下文"""
        handler = GlobalExceptionHandler()
        
        with patch('logging.getLogger') as mock_logger:
            mock_log = Mock()
            mock_logger.return_value = mock_log
            
            exc = DataException("Data error", context={"symbol": "AAPL"})
            result = handler.handle_exception(exc, log_error=True)
            
            assert result is False
            mock_log.error.assert_called_once()
            # 检查日志调用参数是否包含上下文信息
            call_args = mock_log.error.call_args[0][0]
            assert "symbol" in call_args
    
    def test_handle_exception_reraise(self):
        """测试异常处理重新抛出"""
        handler = GlobalExceptionHandler()
        
        exc = ValueError("Test error")
        
        with pytest.raises(ValueError):
            handler.handle_exception(exc, reraise=True)
    
    def test_handle_exception_notify(self):
        """测试异常处理通知"""
        handler = GlobalExceptionHandler()
        
        with patch('logging.getLogger'):
            with patch.object(handler, 'notify_exception') as mock_notify:
                exc = ValueError("Test error")
                handler.handle_exception(exc, notify=True)
                
                mock_notify.assert_called_once_with(exc)
    
    def test_notify_exception(self):
        """测试异常通知"""
        handler = GlobalExceptionHandler()
        
        # 模拟通知逻辑
        with patch.object(handler, '_send_notification') as mock_send:
            exc = ValueError("Test error")
            handler.notify_exception(exc)
            
            mock_send.assert_called_once()
    
    def test_get_exception_details(self):
        """测试获取异常详情"""
        handler = GlobalExceptionHandler()
        
        exc = DataException("Data error", context={"symbol": "AAPL"})
        details = handler.get_exception_details(exc)
        
        assert "exception_type" in details
        assert "message" in details
        assert "timestamp" in details
        assert "context" in details
        assert details["exception_type"] == "DataException"
        assert details["message"] == "Data error"
        assert details["context"]["symbol"] == "AAPL"


class TestHandleExceptionsDecorator:
    """测试异常处理装饰器"""
    
    def test_decorator_success(self):
        """测试装饰器成功执行"""
        @handle_exceptions(default_return="default")
        def test_func():
            return "success"
        
        result = test_func()
        assert result == "success"
    
    def test_decorator_exception_with_default(self):
        """测试装饰器异常处理返回默认值"""
        @handle_exceptions(default_return="default")
        def test_func():
            raise ValueError("Test error")
        
        result = test_func()
        assert result == "default"
    
    def test_decorator_exception_without_default(self):
        """测试装饰器异常处理无默认值"""
        @handle_exceptions()
        def test_func():
            raise ValueError("Test error")
        
        result = test_func()
        assert result is None
    
    def test_decorator_with_logging(self):
        """测试装饰器记录日志"""
        with patch('logging.getLogger') as mock_logger:
            mock_log = Mock()
            mock_logger.return_value = mock_log
            
            @handle_exceptions(log_error=True)
            def test_func():
                raise ValueError("Test error")
            
            result = test_func()
            assert result is None
            mock_log.error.assert_called_once()
    
    def test_decorator_reraise(self):
        """测试装饰器重新抛出异常"""
        @handle_exceptions(reraise=True)
        def test_func():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            test_func()
    
    def test_decorator_with_exception_types(self):
        """测试装饰器指定异常类型"""
        @handle_exceptions(exception_types=(ValueError,), default_return="handled")
        def test_func(should_raise_value_error=True):
            if should_raise_value_error:
                raise ValueError("Value error")
            else:
                raise TypeError("Type error")
        
        # 处理指定类型的异常
        result = test_func(True)
        assert result == "handled"
        
        # 不处理未指定类型的异常
        with pytest.raises(TypeError):
            test_func(False)
    
    def test_decorator_preserve_function_info(self):
        """测试装饰器保留函数信息"""
        @handle_exceptions()
        def test_func():
            """Test function docstring"""
            pass
        
        assert test_func.__name__ == "test_func"
        assert test_func.__doc__ == "Test function docstring"


class TestEdgeCases:
    """测试边界情况"""
    
    def test_empty_error_message(self):
        """测试空错误消息"""
        exc = MyQuantException("")
        assert str(exc) == ""
        assert exc.message == ""
    
    def test_none_error_message(self):
        """测试None错误消息"""
        exc = MyQuantException(None)
        assert exc.message is None
    
    def test_unicode_error_message(self):
        """测试Unicode错误消息"""
        unicode_msg = "Unicode错误消息 🚀"
        exc = MyQuantException(unicode_msg)
        assert str(exc) == unicode_msg
        assert exc.message == unicode_msg
    
    def test_nested_exception_context(self):
        """测试嵌套异常上下文"""
        inner_exc = ValueError("Inner error")
        outer_exc = MyQuantException("Outer error", context={"inner": inner_exc})
        
        assert "inner" in outer_exc.context
        assert isinstance(outer_exc.context["inner"], ValueError)
    
    def test_exception_with_large_context(self):
        """测试大上下文异常"""
        large_context = {f"key_{i}": f"value_{i}" for i in range(1000)}
        exc = MyQuantException("Large context", context=large_context)
        
        assert len(exc.context) == 1000
        assert exc.context["key_0"] == "value_0"
        assert exc.context["key_999"] == "value_999"
    
    def test_exception_serialization(self):
        """测试异常序列化"""
        exc = DataException("Data error", context={"symbol": "AAPL"})
        
        # 测试是否可以转换为字符串
        str_repr = str(exc)
        assert "Data error" in str_repr
        
        # 测试是否可以转换为字典
        dict_repr = exc.to_dict()
        assert dict_repr["message"] == "Data error"
        assert dict_repr["error_code"] == "DATA_ERROR"
        assert dict_repr["context"]["symbol"] == "AAPL"
    
    def test_exception_comparison(self):
        """测试异常比较"""
        exc1 = DataException("Data error")
        exc2 = DataException("Data error")
        exc3 = DataException("Different error")
        
        # 同类型同消息的异常应该被认为相等
        assert exc1 == exc2
        assert exc1 != exc3
        
        # 不同类型的异常不应该相等
        strategy_exc = StrategyException("Data error")
        assert exc1 != strategy_exc
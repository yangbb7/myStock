"""
MyQuant 异常处理机制

提供分层异常体系，支持不同类型的错误分类和处理。
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional


class MyQuantException(Exception):
    """基础异常类"""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "MYQUANT_ERROR"
        self.details = details or {}
        self.cause = cause
        self.context = context or {}
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "cause": str(self.cause) if self.cause else None,
        }
    
    def __eq__(self, other):
        """比较异常对象"""
        if not isinstance(other, self.__class__):
            return False
        return self.message == other.message and self.error_code == other.error_code


class DataException(MyQuantException):
    """数据相关异常"""

    def __init__(
        self,
        message: str,
        data_source: Optional[str] = None,
        symbol: Optional[str] = None,
        **kwargs,
    ):
        # 提取额外的参数
        context = kwargs.pop("context", {})
        error_code = kwargs.pop("error_code", "DATA_ERROR")
        details = kwargs.pop("details", {})
        
        # 处理data_source和symbol
        if data_source:
            details["data_source"] = data_source
            setattr(self, "data_source", data_source)
        if symbol:
            details["symbol"] = symbol
            setattr(self, "symbol", symbol)
        
        # 设置其他属性
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        super().__init__(message, error_code=error_code, details=details, context=context)


class DataSourceException(DataException):
    """数据源异常"""

    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("error_code", "DATA_SOURCE_ERROR")
        super().__init__(message, **kwargs)


class DataValidationException(DataException):
    """数据验证异常"""

    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("error_code", "DATA_VALIDATION_ERROR")
        super().__init__(message, **kwargs)


class DataMissingException(DataException):
    """数据缺失异常"""

    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("error_code", "DATA_MISSING_ERROR")
        super().__init__(message, **kwargs)


class StrategyException(MyQuantException):
    """策略相关异常"""

    def __init__(
        self,
        message: str,
        strategy_name: Optional[str] = None,
        strategy_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        # 提取标准参数
        context = kwargs.pop("context", {})
        error_code = kwargs.pop("error_code", "STRATEGY_ERROR")
        details = kwargs.pop("details", {})
        
        # 设置策略相关属性
        if strategy_name:
            details["strategy_name"] = strategy_name
            setattr(self, "strategy_name", strategy_name)
        if strategy_params:
            details["strategy_params"] = strategy_params
            setattr(self, "strategy_params", strategy_params)
        
        # 设置其他属性
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        super().__init__(message, error_code=error_code, details=details, context=context)


class StrategyInitializationException(StrategyException):
    """策略初始化异常"""

    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("error_code", "STRATEGY_INITIALIZATION_ERROR")
        super().__init__(message, **kwargs)


class StrategyExecutionException(StrategyException):
    """策略执行异常"""

    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("error_code", "STRATEGY_EXECUTION_ERROR")
        super().__init__(message, **kwargs)


class SignalException(StrategyException):
    """信号生成异常"""

    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("error_code", "SIGNAL_ERROR")
        super().__init__(message, **kwargs)


class RiskException(MyQuantException):
    """风险控制异常"""

    def __init__(
        self,
        message: str,
        risk_type: Optional[str] = None,
        current_value: Optional[float] = None,
        limit_value: Optional[float] = None,
        **kwargs,
    ):
        # 提取标准参数
        context = kwargs.pop("context", {})
        error_code = kwargs.pop("error_code", "RISK_ERROR")
        details = kwargs.pop("details", {})
        
        # 设置风险相关属性
        if risk_type:
            details["risk_type"] = risk_type
            setattr(self, "risk_type", risk_type)
        if current_value is not None:
            details["current_value"] = current_value
            setattr(self, "current_value", current_value)
        if limit_value is not None:
            details["limit_value"] = limit_value
            setattr(self, "limit_value", limit_value)
        
        # 设置其他属性
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        super().__init__(message, error_code=error_code, details=details, context=context)


class PositionSizeException(RiskException):
    """仓位规模异常"""

    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("error_code", "POSITION_SIZE_ERROR")
        super().__init__(message, **kwargs)


class DrawdownException(RiskException):
    """回撤异常"""

    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("error_code", "DRAWDOWN_ERROR")
        super().__init__(message, **kwargs)


class VaRException(RiskException):
    """风险价值异常"""

    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("error_code", "VAR_ERROR")
        super().__init__(message, **kwargs)


class OrderException(MyQuantException):
    """订单相关异常"""

    def __init__(
        self,
        message: str,
        order_id: Optional[str] = None,
        symbol: Optional[str] = None,
        order_type: Optional[str] = None,
        **kwargs,
    ):
        # 提取标准参数
        context = kwargs.pop("context", {})
        error_code = kwargs.pop("error_code", "ORDER_ERROR")
        details = kwargs.pop("details", {})
        
        # 设置订单相关属性
        if order_id:
            details["order_id"] = order_id
            setattr(self, "order_id", order_id)
        if symbol:
            details["symbol"] = symbol
            setattr(self, "symbol", symbol)
        if order_type:
            details["order_type"] = order_type
            setattr(self, "order_type", order_type)
        
        # 设置其他属性
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        super().__init__(message, error_code=error_code, details=details, context=context)


class OrderExecutionException(OrderException):
    """订单执行异常"""

    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("error_code", "ORDER_EXECUTION_ERROR")
        super().__init__(message, **kwargs)


class OrderValidationException(OrderException):
    """订单验证异常"""

    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("error_code", "ORDER_VALIDATION_ERROR")
        super().__init__(message, **kwargs)


class PortfolioException(MyQuantException):
    """投资组合异常"""

    def __init__(
        self,
        message: str,
        portfolio_id: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs,
    ):
        # 提取标准参数
        context = kwargs.pop("context", {})
        error_code = kwargs.pop("error_code", "PORTFOLIO_ERROR")
        details = kwargs.pop("details", {})
        
        # 设置投资组合相关属性
        if portfolio_id:
            details["portfolio_id"] = portfolio_id
            setattr(self, "portfolio_id", portfolio_id)
        if operation:
            details["operation"] = operation
            setattr(self, "operation", operation)
        
        # 设置其他属性
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        super().__init__(message, error_code=error_code, details=details, context=context)


class InsufficientFundsException(PortfolioException):
    """资金不足异常"""

    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("error_code", "INSUFFICIENT_FUNDS_ERROR")
        super().__init__(message, **kwargs)


class PositionException(PortfolioException):
    """持仓异常"""

    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("error_code", "POSITION_ERROR")
        super().__init__(message, **kwargs)


class BacktestException(MyQuantException):
    """回测异常"""

    def __init__(
        self,
        message: str,
        backtest_id: Optional[str] = None,
        period: Optional[str] = None,
        **kwargs,
    ):
        # 提取标准参数
        context = kwargs.pop("context", {})
        error_code = kwargs.pop("error_code", "BACKTEST_ERROR")
        details = kwargs.pop("details", {})
        
        # 设置回测相关属性
        if backtest_id:
            details["backtest_id"] = backtest_id
            setattr(self, "backtest_id", backtest_id)
        if period:
            details["period"] = period
            setattr(self, "period", period)
        
        # 设置其他属性
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        super().__init__(message, error_code=error_code, details=details, context=context)


class ConfigurationException(MyQuantException):
    """配置异常"""

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_file: Optional[str] = None,
        **kwargs,
    ):
        # 提取标准参数
        context = kwargs.pop("context", {})
        error_code = kwargs.pop("error_code", "CONFIGURATION_ERROR")
        details = kwargs.pop("details", {})
        
        # 设置配置相关属性
        if config_key:
            details["config_key"] = config_key
            setattr(self, "config_key", config_key)
        if config_file:
            details["config_file"] = config_file
            setattr(self, "config_file", config_file)
        
        # 设置其他属性
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        super().__init__(message, error_code=error_code, details=details, context=context)


class APIException(MyQuantException):
    """API异常"""

    def __init__(
        self,
        message: str,
        api_endpoint: Optional[str] = None,
        status_code: Optional[int] = None,
        response_data: Optional[Any] = None,
        **kwargs,
    ):
        # 提取标准参数
        context = kwargs.pop("context", {})
        error_code = kwargs.pop("error_code", "API_ERROR")
        details = kwargs.pop("details", {})
        
        # 设置API相关属性
        if api_endpoint:
            details["api_endpoint"] = api_endpoint
            setattr(self, "endpoint", api_endpoint)
        if status_code:
            details["status_code"] = status_code
            setattr(self, "status_code", status_code)
        if response_data:
            details["response_data"] = response_data
            setattr(self, "response_data", response_data)
        
        # 设置其他属性
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        super().__init__(message, error_code=error_code, details=details, context=context)


class NetworkException(MyQuantException):
    """网络异常"""

    def __init__(
        self,
        message: str,
        url: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        # 提取标准参数
        context = kwargs.pop("context", {})
        error_code = kwargs.pop("error_code", "NETWORK_ERROR")
        details = kwargs.pop("details", {})
        
        # 设置网络相关属性
        if url:
            details["url"] = url
            setattr(self, "url", url)
        if timeout:
            details["timeout"] = timeout
            setattr(self, "timeout", timeout)
        
        # 设置其他属性
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        super().__init__(message, error_code=error_code, details=details, context=context)


class ProcessingException(MyQuantException):
    """数据处理异常"""

    def __init__(
        self,
        message: str,
        processor: Optional[str] = None,
        data_type: Optional[str] = None,
        **kwargs,
    ):
        # 提取标准参数
        context = kwargs.pop("context", {})
        error_code = kwargs.pop("error_code", "PROCESSING_ERROR")
        details = kwargs.pop("details", {})
        
        # 设置处理相关属性
        if processor:
            details["processor"] = processor
            setattr(self, "processor", processor)
        if data_type:
            details["data_type"] = data_type
            setattr(self, "data_type", data_type)
        
        # 设置其他属性
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        super().__init__(message, error_code=error_code, details=details, context=context)


class MonitoringException(MyQuantException):
    """监控异常"""

    def __init__(
        self,
        message: str,
        component: Optional[str] = None,
        metric: Optional[str] = None,
        **kwargs,
    ):
        # 提取标准参数
        context = kwargs.pop("context", {})
        error_code = kwargs.pop("error_code", "MONITORING_ERROR")
        details = kwargs.pop("details", {})
        
        # 设置监控相关属性
        if component:
            details["component"] = component
            setattr(self, "component", component)
        if metric:
            details["metric"] = metric
            setattr(self, "metric", metric)
        
        # 设置其他属性
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        super().__init__(message, error_code=error_code, details=details, context=context)


# 异常处理装饰器
def handle_exceptions(
    default_return=None, reraise_types: tuple = (), log_level: int = logging.ERROR,
    log_error: bool = True, reraise: bool = False, exception_types: tuple = None
):
    """异常处理装饰器"""
    import functools
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except reraise_types:
                raise
            except MyQuantException as e:
                if log_error:
                    logger = logging.getLogger()
                    logger.error(f"MyQuant异常 in {func.__name__}: {e}")
                if reraise:
                    raise
                if default_return is not None:
                    return default_return
                raise
            except Exception as e:
                # 如果指定了exception_types，仅处理指定类型的异常
                if exception_types and type(e) in exception_types:
                    if log_error:
                        logger = logging.getLogger()
                        logger.error(f"已处理异常 in {func.__name__}: {e}")
                    return default_return
                elif exception_types and type(e) not in exception_types:
                    # 不处理未指定类型的异常
                    raise
                
                if log_error:
                    logger = logging.getLogger()
                    logger.error(f"未处理异常 in {func.__name__}: {e}")
                if reraise:
                    raise
                if default_return is not None:
                    return default_return
                # 如果指定了reraise_types，说明期望异常处理，应该包装并抛出
                # 如果没有指定reraise_types，则返回None
                if reraise_types:
                    raise MyQuantException(
                        f"未处理异常在 {func.__name__}: {str(e)}", cause=e
                    )
                else:
                    return None

        return wrapper

    return decorator


# 异常工厂
class ExceptionFactory:
    """异常工厂类"""

    @staticmethod
    def create_data_exception(error_type: str, message: str, **kwargs) -> DataException:
        """创建数据异常"""
        exception_map = {
            "source_error": DataSourceException,
            "validation_error": DataValidationException,
            "missing_error": DataMissingException,
        }
        exception_class = exception_map.get(error_type, DataException)
        return exception_class(message, **kwargs)

    @staticmethod
    def create_strategy_exception(
        error_type: str, message: str, **kwargs
    ) -> StrategyException:
        """创建策略异常"""
        exception_map = {
            "initialization_error": StrategyInitializationException,
            "execution_error": StrategyExecutionException,
            "signal_error": SignalException,
        }
        exception_class = exception_map.get(error_type, StrategyException)
        return exception_class(message, **kwargs)

    @staticmethod
    def create_risk_exception(error_type: str, message: str, **kwargs) -> RiskException:
        """创建风险异常"""
        exception_map = {
            "position_size_error": PositionSizeException,
            "drawdown_error": DrawdownException,
            "var_error": VaRException,
        }
        exception_class = exception_map.get(error_type, RiskException)
        return exception_class(message, **kwargs)


# 全局异常处理器
class GlobalExceptionHandler:
    """全局异常处理器"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        if logger:
            self.logger = logger
        else:
            # 延迟初始化，这样可以在测试中被正确模拟
            self.logger = None

    def handle_exception(
        self, exception: Exception, context: Optional[Dict[str, Any]] = None,
        log_error: bool = True, reraise: bool = False, notify: bool = False
    ) -> bool:
        """处理异常"""
        context = context or {}
        
        if reraise:
            if log_error:
                logger = self.logger or logging.getLogger(__name__)
                logger.error(f"异常将被重新抛出: {str(exception)}")
            raise exception
        
        if notify:
            self.notify_exception(exception)

        # 获取logger，支持测试中的mock
        logger = self.logger or logging.getLogger(__name__)

        if isinstance(exception, MyQuantException):
            error_info = exception.to_dict()
            error_info.update(context)

            if log_error:
                # 在日志消息中包含上下文信息
                context_str = ""
                if exception.context:
                    context_items = [f"{k}={v}" for k, v in exception.context.items()]
                    context_str = f" [context: {', '.join(context_items)}]"
                logger.error(
                    f"MyQuant异常: {exception.message}{context_str}", extra={"error_info": error_info}
                )
        else:
            error_info = {
                "type": "UnhandledException",
                "message": str(exception),
                "timestamp": datetime.now().isoformat(),
                "context": context,
            }

            if log_error:
                logger.error(
                    f"未处理异常: {str(exception)}", extra={"error_info": error_info}
                )
        
        # 返回False表示处理了异常但未成功解决
        return False
    
    def notify_exception(self, exception: Exception):
        """通知异常"""
        self._send_notification(exception)
    
    def _send_notification(self, exception: Exception):
        """发送异常通知的内部方法"""
        # 在实际实现中，这里会发送邮件、短信等通知
        pass
    
    def get_exception_details(self, exception: Exception) -> Dict[str, Any]:
        """获取异常详情"""
        if isinstance(exception, MyQuantException):
            return {
                "exception_type": exception.__class__.__name__,
                "message": exception.message,
                "error_code": exception.error_code,
                "context": exception.context,
                "details": exception.details,
                "timestamp": exception.timestamp.isoformat(),
            }
        else:
            return {
                "exception_type": exception.__class__.__name__,
                "message": str(exception),
                "timestamp": datetime.now().isoformat(),
                "context": {},
                "details": {},
            }

# -*- coding: utf-8 -*-
"""
依赖注入容器 - 管理应用程序所有组件的依赖关系
"""

import logging

from dependency_injector import containers, providers
from dependency_injector.wiring import Provide, inject

from ..core.analysis.performance_analyzer import PerformanceAnalyzer
from ..core.engines.backtest_engine import BacktestEngine
from ..core.engines.execution_engine import ExecutionEngine
from ..core.engines.strategy_engine import StrategyEngine
from ..core.managers.data_manager import DataManager
from ..core.managers.order_manager import OrderManager
from ..core.managers.portfolio_manager import PortfolioManager
from ..core.managers.risk_manager import RiskManager
from ..core.trading_system import TradingSystem
from ..infrastructure.monitoring.exception_logger import ExceptionLogger
from ..infrastructure.monitoring.logging import configure_logging
from ..infrastructure.monitoring.metrics import MetricsCollector
from .config.settings import ConfigManager, get_config_manager


class ApplicationContainer(containers.DeclarativeContainer):
    """应用程序依赖注入容器"""

    # 配置相关
    config_manager = providers.Singleton(ConfigManager)
    config = providers.Singleton(lambda cm: cm.get_config() or cm.load_config(), cm=config_manager)

    # 日志配置
    logging_config = providers.Resource(configure_logging)

    # 异常日志记录器
    exception_logger = providers.Singleton(ExceptionLogger)

    # 指标收集器
    metrics_collector = providers.Singleton(MetricsCollector)

    # 数据管理器
    data_manager = providers.Singleton(
        DataManager, config=config.provided.data_provider
    )

    # 投资组合管理器
    portfolio_manager = providers.Factory(
        PortfolioManager, config=config.provided.trading
    )

    # 订单管理器
    order_manager = providers.Factory(OrderManager, config=config.provided.trading)

    # 风险管理器
    risk_manager = providers.Factory(RiskManager, config=config.provided.trading)

    # 执行引擎
    execution_engine = providers.Factory(
        ExecutionEngine,
        config=config.provided.trading,
    )

    # 策略引擎
    strategy_engine = providers.Factory(
        StrategyEngine,
        config=config.provided.trading,
    )

    # 回测引擎
    backtest_engine = providers.Factory(
        BacktestEngine,
        config=config.provided.backtest,
    )

    # 绩效分析器
    performance_analyzer = providers.Factory(
        PerformanceAnalyzer, config=config.provided.backtest
    )

    # 交易系统
    trading_system = providers.Singleton(
        TradingSystem, 
        config=config.provided.trading,
        data_manager=data_manager,
        strategy_engine=strategy_engine,
        backtest_engine=backtest_engine,
        risk_manager=risk_manager,
        portfolio_manager=portfolio_manager,
        order_manager=order_manager,
        execution_engine=execution_engine,
        performance_analyzer=performance_analyzer
    )


class ContainerManager:
    """容器管理器"""

    def __init__(self):
        self._container = ApplicationContainer()
        self._wired_modules = []

    def wire_modules(self, modules):
        """连接模块"""
        self._container.wire(modules=modules)
        self._wired_modules.extend(modules)

    def unwire_modules(self, modules=None):
        """断开模块连接"""
        if modules is None:
            modules = self._wired_modules

        # 尝试不同的unwire方法
        try:
            self._container.unwire(modules=modules)
        except TypeError:
            # 如果modules参数不被支持，尝试不传参数
            try:
                self._container.unwire()
            except:
                pass  # 如果unwire失败，继续执行

        for module in modules:
            if module in self._wired_modules:
                self._wired_modules.remove(module)

    def get_container(self):
        """获取容器"""
        return self._container

    def reset(self):
        """重置容器"""
        self.unwire_modules()
        try:
            self._container.reset_last_provided()
        except AttributeError:
            # 如果没有reset_last_provided方法，尝试其他方式
            try:
                self._container.reset_singletons()
            except AttributeError:
                # 如果都不存在，创建新容器
                self._container = ApplicationContainer()

    def shutdown(self):
        """关闭容器"""
        self.unwire_modules()
        self._container.shutdown_resources()


# 全局容器管理器
_container_manager = None


def get_container_manager() -> ContainerManager:
    """获取全局容器管理器"""
    global _container_manager
    if _container_manager is None:
        _container_manager = ContainerManager()
    return _container_manager


def get_container() -> ApplicationContainer:
    """获取应用程序容器"""
    return get_container_manager().get_container()


def wire_container(modules):
    """连接容器到模块"""
    get_container_manager().wire_modules(modules)


def unwire_container(modules=None):
    """断开容器连接"""
    get_container_manager().unwire_modules(modules)


# 便捷函数用于获取组件
def get_data_manager():
    """获取数据管理器"""
    return get_container().data_manager()


def get_portfolio_manager():
    """获取投资组合管理器"""
    return get_container().portfolio_manager()


def get_strategy_engine():
    """获取策略引擎"""
    return get_container().strategy_engine()


def get_trading_system():
    """获取交易系统"""
    return get_container().trading_system()


def get_performance_analyzer():
    """获取绩效分析器"""
    return get_container().performance_analyzer()


def get_backtest_engine():
    """获取回测引擎"""
    return get_container().backtest_engine()


# 用于依赖注入的装饰器
def inject_dependencies(func):
    """依赖注入装饰器"""
    return inject(func)


# 示例用法
if __name__ == "__main__":
    # 获取容器
    container = get_container()

    # 获取组件
    data_manager = container.data_manager()
    portfolio_manager = container.portfolio_manager()
    strategy_engine = container.strategy_engine()

    print("依赖注入容器初始化成功")
    print(f"数据管理器: {data_manager}")
    print(f"投资组合管理器: {portfolio_manager}")
    print(f"策略引擎: {strategy_engine}")

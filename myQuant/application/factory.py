# -*- coding: utf-8 -*-
"""
应用程序工厂 - 创建和配置应用程序实例
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.trading_system import TradingSystem
from ..infrastructure.config.settings import ConfigManager, Environment
from ..infrastructure.container import ApplicationContainer, ContainerManager


class ApplicationFactory:
    """应用程序工厂"""

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.container_manager = ContainerManager()
        self.logger = logging.getLogger(__name__)

    def create_app(
        self,
        environment: Environment = Environment.DEVELOPMENT,
        config_file: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
        wire_modules: Optional[List[str]] = None,
    ) -> TradingSystem:
        """创建应用程序实例"""

        # 1. 初始化容器
        container = self.container_manager.get_container()

        # 2. 配置管理器
        config_manager = container.config_manager()
        config_manager.config_dir = self.config_dir

        # 3. 加载配置
        config = config_manager.load_config(config_file, environment)

        # 4. 应用配置覆盖
        if config_overrides:
            config_manager.update_config(config_overrides)

        # 5. 初始化资源
        self._initialize_resources(container)

        # 6. 连接模块
        if wire_modules:
            self.container_manager.wire_modules(wire_modules)

        # 7. 创建交易系统
        trading_system = container.trading_system()

        self.logger.info(f"应用程序创建成功 - 环境: {environment.value}")
        return trading_system

    def _initialize_resources(self, container: ApplicationContainer):
        """初始化资源"""
        try:
            # 安全地初始化资源
            try:
                container.logging_config()
            except Exception as e:
                self.logger.warning(f"日志配置初始化失败: {e}")

            try:
                container.exception_logger()
            except Exception as e:
                self.logger.warning(f"异常记录器初始化失败: {e}")

            try:
                container.metrics_collector()
            except Exception as e:
                self.logger.warning(f"指标收集器初始化失败: {e}")

            self.logger.info("资源初始化完成")
        except Exception as e:
            self.logger.error(f"资源初始化失败: {e}")
            # 不抛出异常，允许继续执行

    def create_backtest_app(
        self,
        config_file: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
    ):
        """创建回测应用程序"""
        container = self.container_manager.get_container()

        # 加载配置
        config_manager = container.config_manager()
        config_manager.config_dir = self.config_dir

        # 确保配置被加载
        if config_file:
            config_manager.load_config(config_file, Environment.DEVELOPMENT)
        else:
            config_manager.load_config(None, Environment.DEVELOPMENT)

        if config_overrides:
            config_manager.update_config(config_overrides)

        # 初始化资源
        self._initialize_resources(container)

        # 创建回测引擎
        backtest_engine = container.backtest_engine()

        self.logger.info("回测应用程序创建成功")
        return backtest_engine

    def create_analysis_app(
        self,
        config_file: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
    ):
        """创建分析应用程序"""
        container = self.container_manager.get_container()

        # 加载配置
        config_manager = container.config_manager()
        config_manager.config_dir = self.config_dir

        # 确保配置被加载
        if config_file:
            config_manager.load_config(config_file, Environment.DEVELOPMENT)
        else:
            config_manager.load_config(None, Environment.DEVELOPMENT)

        if config_overrides:
            config_manager.update_config(config_overrides)

        # 初始化资源
        self._initialize_resources(container)

        # 创建分析器
        performance_analyzer = container.performance_analyzer()
        data_manager = container.data_manager()

        self.logger.info("分析应用程序创建成功")
        return {
            "performance_analyzer": performance_analyzer,
            "data_manager": data_manager,
        }

    def shutdown(self):
        """关闭应用程序"""
        try:
            self.container_manager.shutdown()
            self.logger.info("应用程序关闭成功")
        except Exception as e:
            self.logger.error(f"应用程序关闭失败: {e}")


class AppBuilder:
    """应用程序构建器"""

    def __init__(self):
        self.factory = ApplicationFactory()
        self.environment = Environment.DEVELOPMENT
        self.config_file = None
        self.config_overrides = {}
        self.wire_modules = []

    def with_environment(self, environment: Environment):
        """设置环境"""
        self.environment = environment
        return self

    def with_config_file(self, config_file: str):
        """设置配置文件"""
        self.config_file = config_file
        return self

    def with_config_override(self, key: str, value: Any):
        """添加配置覆盖"""
        self.config_overrides[key] = value
        return self

    def with_config_overrides(self, overrides: Dict[str, Any]):
        """批量添加配置覆盖"""
        self.config_overrides.update(overrides)
        return self

    def with_wire_modules(self, modules: List[str]):
        """设置连接模块"""
        self.wire_modules = modules
        return self

    def build(self) -> TradingSystem:
        """构建应用程序"""
        return self.factory.create_app(
            environment=self.environment,
            config_file=self.config_file,
            config_overrides=self.config_overrides,
            wire_modules=self.wire_modules,
        )

    def build_backtest(self):
        """构建回测应用程序"""
        return self.factory.create_backtest_app(
            config_file=self.config_file, config_overrides=self.config_overrides
        )

    def build_analysis(self):
        """构建分析应用程序"""
        return self.factory.create_analysis_app(
            config_file=self.config_file, config_overrides=self.config_overrides
        )


# 便捷函数
def create_app(
    environment: Environment = Environment.DEVELOPMENT,
    config_file: Optional[str] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> TradingSystem:
    """创建应用程序快捷方式"""
    factory = ApplicationFactory()
    return factory.create_app(environment, config_file, config_overrides)


def create_backtest_app(
    config_file: Optional[str] = None, config_overrides: Optional[Dict[str, Any]] = None
):
    """创建回测应用程序快捷方式"""
    factory = ApplicationFactory()
    return factory.create_backtest_app(config_file, config_overrides)


def create_analysis_app(
    config_file: Optional[str] = None, config_overrides: Optional[Dict[str, Any]] = None
):
    """创建分析应用程序快捷方式"""
    factory = ApplicationFactory()
    return factory.create_analysis_app(config_file, config_overrides)


# 示例用法
if __name__ == "__main__":
    # 使用构建器模式
    app = (
        AppBuilder()
        .with_environment(Environment.DEVELOPMENT)
        .with_config_override("trading.initial_capital", 2000000)
        .with_config_override("trading.commission_rate", 0.0002)
        .build()
    )

    print(f"应用程序创建成功: {app}")

    # 使用工厂直接创建
    backtest_app = create_backtest_app(
        config_overrides={
            "backtest.start_date": "2020-01-01",
            "backtest.end_date": "2023-12-31",
        }
    )

    print(f"回测应用程序创建成功: {backtest_app}")

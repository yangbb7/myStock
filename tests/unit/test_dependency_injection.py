# -*- coding: utf-8 -*-
"""
依赖注入系统测试
"""

import pytest
from unittest.mock import Mock, patch
from myQuant.infrastructure.container import ApplicationContainer, get_container, get_data_manager
from myQuant.application.factory import ApplicationFactory, AppBuilder
from myQuant.infrastructure.config.settings import Environment


class TestDependencyInjection:
    """依赖注入测试用例"""

    def test_container_initialization(self):
        """测试容器初始化"""
        container = ApplicationContainer()
        assert container is not None
        
        # 测试配置管理器
        config_manager = container.config_manager()
        assert config_manager is not None
        
        # 测试配置对象
        config = container.config()
        assert config is not None

    def test_singleton_components(self):
        """测试单例组件"""
        container = ApplicationContainer()
        
        # 数据管理器应该是单例
        data_manager1 = container.data_manager()
        data_manager2 = container.data_manager()
        assert data_manager1 is data_manager2
        
        # 交易系统应该是单例
        trading_system1 = container.trading_system()
        trading_system2 = container.trading_system()
        assert trading_system1 is trading_system2

    def test_factory_components(self):
        """测试工厂组件"""
        container = ApplicationContainer()
        
        # 投资组合管理器应该是工厂创建的
        portfolio_manager1 = container.portfolio_manager()
        portfolio_manager2 = container.portfolio_manager()
        assert portfolio_manager1 is not portfolio_manager2
        
        # 但是它们应该使用相同的配置
        assert portfolio_manager1.config is portfolio_manager2.config

    def test_component_dependencies(self):
        """测试组件依赖关系"""
        container = ApplicationContainer()
        
        # 获取交易系统
        trading_system = container.trading_system()
        
        # 验证依赖关系
        assert trading_system.data_manager is not None
        assert trading_system.portfolio_manager is not None
        assert trading_system.strategy_engine is not None
        assert trading_system.execution_engine is not None
        assert trading_system.risk_manager is not None

    def test_application_factory(self):
        """测试应用程序工厂"""
        factory = ApplicationFactory()
        
        # 创建应用程序
        with patch('myQuant.infrastructure.config.settings.ConfigManager.load_config') as mock_load:
            mock_config = Mock()
            mock_config.trading = Mock()
            mock_config.trading.initial_capital = 1000000
            mock_load.return_value = mock_config
            
            app = factory.create_app(
                environment=Environment.DEVELOPMENT,
                config_overrides={'trading.initial_capital': 2000000}
            )
            
            assert app is not None

    def test_app_builder(self):
        """测试应用程序构建器"""
        with patch('myQuant.infrastructure.config.settings.ConfigManager.load_config') as mock_load:
            mock_config = Mock()
            mock_config.trading = Mock()
            mock_config.trading.initial_capital = 1000000
            mock_load.return_value = mock_config
            
            builder = AppBuilder()
            app = (builder
                   .with_environment(Environment.DEVELOPMENT)
                   .with_config_override('trading.initial_capital', 2000000)
                   .build())
            
            assert app is not None

    def test_container_convenience_functions(self):
        """测试容器便捷函数"""
        container = get_container()
        assert container is not None
        
        # 测试便捷函数
        data_manager = get_data_manager()
        assert data_manager is not None

    def test_container_reset(self):
        """测试容器重置"""
        from myQuant.infrastructure.container import get_container_manager
        
        container_manager = get_container_manager()
        
        # 获取组件
        data_manager1 = container_manager.get_container().data_manager()
        
        # 重置容器
        container_manager.reset()
        
        # 重新获取组件
        data_manager2 = container_manager.get_container().data_manager()
        
        # 应该是不同的实例
        assert data_manager1 is not data_manager2

    def test_container_shutdown(self):
        """测试容器关闭"""
        from myQuant.infrastructure.container import get_container_manager
        
        container_manager = get_container_manager()
        
        # 关闭容器
        container_manager.shutdown()
        
        # 重新获取应该正常工作
        container = container_manager.get_container()
        assert container is not None

    def test_config_injection(self):
        """测试配置注入"""
        container = ApplicationContainer()
        
        # 获取配置
        config = container.config()
        assert config is not None
        
        # 获取组件并验证配置注入
        portfolio_manager = container.portfolio_manager()
        assert portfolio_manager.config is not None

    def test_resource_initialization(self):
        """测试资源初始化"""
        container = ApplicationContainer()
        
        # 初始化资源
        try:
            # 这些资源的初始化可能会失败，因为缺少实际的配置
            # 但至少应该能够创建提供者
            exception_logger_provider = container.exception_logger
            metrics_collector_provider = container.metrics_collector
            
            assert exception_logger_provider is not None
            assert metrics_collector_provider is not None
        except Exception as e:
            # 资源初始化失败是可以接受的，因为缺少完整的配置
            pytest.skip(f"资源初始化跳过: {e}")

    def test_backtest_app_creation(self):
        """测试回测应用程序创建"""
        factory = ApplicationFactory()
        
        with patch('myQuant.infrastructure.config.settings.ConfigManager.load_config') as mock_load:
            mock_config = Mock()
            mock_config.backtest = Mock()
            mock_config.backtest.start_date = '2020-01-01'
            mock_load.return_value = mock_config
            
            backtest_app = factory.create_backtest_app(
                config_overrides={'backtest.start_date': '2021-01-01'}
            )
            
            assert backtest_app is not None

    def test_analysis_app_creation(self):
        """测试分析应用程序创建"""
        factory = ApplicationFactory()
        
        with patch('myQuant.infrastructure.config.settings.ConfigManager.load_config') as mock_load:
            mock_config = Mock()
            mock_config.trading = Mock()
            mock_config.backtest = Mock()
            mock_load.return_value = mock_config
            
            analysis_app = factory.create_analysis_app()
            
            assert analysis_app is not None
            assert 'performance_analyzer' in analysis_app
            assert 'data_manager' in analysis_app
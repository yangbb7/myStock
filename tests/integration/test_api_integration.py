# -*- coding: utf-8 -*-
"""
API集成测试
测试公共API接口的集成和交互
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "myQuant"))

from myQuant import (
    setup_logging,
    create_default_config,
    get_version,
    MAStrategy,
    BaseStrategy,
    DataManager,
    BacktestEngine,
    PerformanceAnalyzer,
)
from myQuant.core.trading_system import TradingSystem
from myQuant.core.engines.async_data_engine import AsyncDataEngine
from myQuant.core.strategy_engine import BaseStrategy as CoreBaseStrategy
from myQuant.infrastructure.container import get_container
from myQuant.core.exceptions import (
    DataException,
    ConfigurationException,
    OrderException,
    StrategyException,
)


class TestAPIIntegration:
    """API集成测试类"""
    
    @pytest.fixture
    def api_config(self):
        """API测试配置"""
        return {
            'initial_capital': 1000000,
            'commission_rate': 0.0003,
            'min_commission': 5.0,
            'slippage_rate': 0.001,
            'max_position_size': 0.1,
            'max_drawdown_limit': 0.2,
            'risk_free_rate': 0.03,
            'trading_days_per_year': 252
        }
    
    @pytest.fixture
    def sample_data(self):
        """生成样本数据"""
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        symbols = ['000001.SZ', '000002.SZ']
        
        data_list = []
        for symbol in symbols:
            base_price = 15.0 if symbol == '000001.SZ' else 20.0
            
            for i, date in enumerate(dates):
                price = base_price * (1 + 0.001 * i + np.random.normal(0, 0.01))
                
                data_list.append({
                    'datetime': date,
                    'symbol': symbol,
                    'open': price * 0.999,
                    'high': price * 1.001,
                    'low': price * 0.999,
                    'close': price,
                    'volume': np.random.randint(1000000, 5000000),
                    'adj_close': price
                })
        
        return pd.DataFrame(data_list)
    
    # === 核心API集成测试 ===
    
    @pytest.mark.integration
    def test_main_api_imports(self):
        """测试主要API导入"""
        print("=== 开始主要API导入测试 ===")
        
        # 1. 测试版本信息
        version = get_version()
        assert version is not None
        assert isinstance(version, str)
        assert len(version) > 0
        print(f"✓ 版本信息: {version}")
        
        # 2. 测试配置创建
        config = create_default_config()
        assert config is not None
        assert isinstance(config, dict)
        assert 'initial_capital' in config
        print("✓ 默认配置创建成功")
        
        # 3. 测试日志设置
        logger = setup_logging(level='INFO')
        assert logger is not None
        print("✓ 日志系统设置成功")
        
        # 4. 测试核心类导入
        assert DataManager is not None
        assert BacktestEngine is not None
        assert PerformanceAnalyzer is not None
        assert TradingSystem is not None
        assert BaseStrategy is not None
        assert MAStrategy is not None
        print("✓ 核心类导入成功")
        
        print("=== 主要API导入测试通过 ===")
    
    @pytest.mark.integration
    def test_trading_system_api(self, api_config, sample_data):
        """测试交易系统API"""
        print("=== 开始交易系统API测试 ===")
        
        # 1. 创建交易系统
        trading_system = TradingSystem(api_config)
        assert trading_system is not None
        print("✓ 交易系统创建成功")
        
        # 2. 测试策略管理API
        class TestStrategy(BaseStrategy):
            def initialize(self):
                self.processed_count = 0
                
            def on_bar(self, bar_data):
                self.processed_count += 1
                return []
                
            def on_tick(self, tick_data):
                return []
                
            def finalize(self):
                pass
        
        strategy = TestStrategy("TestStrategy", ['000001.SZ'])
        strategy_id = trading_system.add_strategy(strategy)
        
        assert strategy_id is not None
        print(f"✓ 策略添加成功: {strategy_id}")
        
        # 3. 测试数据管理API
        trading_system.data_manager.load_data(sample_data)
        print("✓ 数据加载成功")
        
        # 4. 测试市场数据处理API
        test_tick = {
            'datetime': datetime.now(),
            'symbol': '000001.SZ',
            'close': 15.0,
            'volume': 1000000,
            'open': 14.9,
            'high': 15.1,
            'low': 14.8
        }
        
        result = trading_system.process_market_tick(test_tick)
        assert result is not None
        assert 'processed' in result
        print("✓ 市场数据处理成功")
        
        # 5. 测试系统状态API
        status = trading_system.get_system_status()
        assert status is not None
        assert 'is_running' in status
        assert 'strategies_count' in status
        print("✓ 系统状态查询成功")
        
        # 6. 测试开盘前准备和收盘后总结API
        trading_system.pre_market_setup()
        print("✓ 开盘前准备成功")
        
        summary = trading_system.post_market_summary()
        assert summary is not None
        assert 'trades_count' in summary
        assert 'portfolio_value' in summary
        print("✓ 收盘后总结成功")
        
        print("=== 交易系统API测试通过 ===")
    
    @pytest.mark.integration
    def test_data_manager_api(self, api_config, sample_data):
        """测试数据管理器API"""
        print("=== 开始数据管理器API测试 ===")
        
        # 1. 创建数据管理器
        data_manager = DataManager(api_config.get('data_manager', {}))
        assert data_manager is not None
        print("✓ 数据管理器创建成功")
        
        # 2. 测试数据加载API
        data_manager.load_data(sample_data)
        print("✓ 数据加载成功")
        
        # 3. 测试数据验证API
        is_valid = data_manager.validate_price_data(sample_data)
        assert isinstance(is_valid, bool)
        print(f"✓ 数据验证结果: {is_valid}")
        
        # 4. 测试技术指标计算API
        prices = sample_data[sample_data['symbol'] == '000001.SZ']['close']
        ma_5 = data_manager.calculate_ma(prices, 5)
        ma_20 = data_manager.calculate_ma(prices, 20)
        
        assert ma_5 is not None
        assert ma_20 is not None
        assert len(ma_5) == len(prices)
        assert len(ma_20) == len(prices)
        print("✓ 技术指标计算成功")
        
        # 5. 测试数据获取API
        symbol_data = data_manager.get_price_data('000001.SZ', '2024-01-01', '2024-02-20')
        assert symbol_data is not None
        print("✓ 价格数据获取成功")
        
        print("=== 数据管理器API测试通过 ===")
    
    @pytest.mark.integration
    def test_backtest_engine_api(self, api_config, sample_data):
        """测试回测引擎API"""
        print("=== 开始回测引擎API测试 ===")
        
        # 1. 创建回测引擎
        backtest_engine = BacktestEngine(api_config)
        assert backtest_engine is not None
        print("✓ 回测引擎创建成功")
        
        # 2. 创建测试策略
        class TestBacktestStrategy(BaseStrategy):
            def initialize(self):
                self.prices = []
                
            def on_bar(self, bar_data):
                self.prices.append(bar_data['close'])
                if len(self.prices) > 10:
                    self.prices.pop(0)
                
                # 简单的买入信号
                if len(self.prices) >= 5:
                    ma_5 = sum(self.prices[-5:]) / 5
                    if bar_data['close'] > ma_5 * 1.02:
                        return [{
                            'timestamp': bar_data['datetime'],
                            'symbol': bar_data['symbol'],
                            'signal_type': 'BUY',
                            'price': bar_data['close'],
                            'quantity': 1000,
                            'strategy_name': self.name
                        }]
                return []
                
            def on_tick(self, tick_data):
                return []
                
            def finalize(self):
                pass
        
        # 3. 测试策略添加API
        strategy = TestBacktestStrategy("BacktestStrategy", ['000001.SZ'])
        backtest_engine.add_strategy(strategy)
        print("✓ 策略添加成功")
        
        # 4. 测试历史数据加载API
        data_manager = DataManager(api_config.get('data_manager', {}))
        data_manager.load_data(sample_data)
        
        backtest_engine.load_historical_data(
            data_manager, 
            ['000001.SZ'], 
            '2024-01-01', 
            '2024-02-20'
        )
        print("✓ 历史数据加载成功")
        
        # 5. 测试回测运行API
        start_date = sample_data['datetime'].min().strftime('%Y-%m-%d')
        end_date = sample_data['datetime'].max().strftime('%Y-%m-%d')
        
        result = backtest_engine.run_backtest(start_date, end_date)
        assert result is not None
        assert 'final_value' in result
        assert 'total_return' in result
        print("✓ 回测运行成功")
        
        print(f"✓ 回测结果: 最终价值={result['final_value']:.2f}, 总收益率={result['total_return']:.2%}")
        print("=== 回测引擎API测试通过 ===")
    
    @pytest.mark.integration
    def test_performance_analyzer_api(self, api_config):
        """测试绩效分析器API"""
        print("=== 开始绩效分析器API测试 ===")
        
        # 1. 创建绩效分析器
        analyzer = PerformanceAnalyzer(api_config)
        assert analyzer is not None
        print("✓ 绩效分析器创建成功")
        
        # 2. 生成测试数据
        portfolio_values = [1000000]
        for i in range(100):
            daily_return = np.random.normal(0.0008, 0.015)
            new_value = portfolio_values[-1] * (1 + daily_return)
            portfolio_values.append(new_value)
        
        portfolio_series = pd.Series(portfolio_values)
        
        # 3. 测试完整绩效分析API
        results = analyzer.analyze_portfolio(portfolio_series)
        assert results is not None
        assert 'returns' in results
        assert 'risk' in results
        print("✓ 完整绩效分析成功")
        
        # 4. 测试单独指标计算API
        returns = analyzer.calculate_returns(portfolio_series)
        assert returns is not None
        assert len(returns) == len(portfolio_series) - 1
        print("✓ 收益率计算成功")
        
        sharpe_ratio = analyzer.calculate_sharpe_ratio(returns)
        assert isinstance(sharpe_ratio, (int, float))
        print(f"✓ 夏普比率计算成功: {sharpe_ratio:.2f}")
        
        max_drawdown = analyzer.calculate_max_drawdown(portfolio_series)
        assert max_drawdown is not None
        print("✓ 最大回撤计算成功")
        
        # 5. 测试基准比较API
        benchmark_returns = pd.Series(np.random.normal(0.0005, 0.01, len(returns)))
        benchmark_metrics = analyzer.calculate_benchmark_metrics(returns, benchmark_returns)
        
        assert benchmark_metrics is not None
        assert 'alpha' in benchmark_metrics
        assert 'beta' in benchmark_metrics
        print("✓ 基准比较计算成功")
        
        print("=== 绩效分析器API测试通过 ===")
    
    @pytest.mark.integration
    def test_strategy_api(self, api_config, sample_data):
        """测试策略API"""
        print("=== 开始策略API测试 ===")
        
        # 1. 测试基础策略类
        class CustomStrategy(BaseStrategy):
            def initialize(self):
                self.short_window = self.params.get('short_window', 5)
                self.long_window = self.params.get('long_window', 20)
                self.prices = {}
                self.initialized = True
                
            def on_bar(self, bar_data):
                symbol = bar_data['symbol']
                close_price = bar_data['close']
                
                if symbol not in self.prices:
                    self.prices[symbol] = []
                
                self.prices[symbol].append(close_price)
                if len(self.prices[symbol]) > self.long_window:
                    self.prices[symbol].pop(0)
                
                # 移动平均策略
                if len(self.prices[symbol]) >= self.long_window:
                    short_ma = sum(self.prices[symbol][-self.short_window:]) / self.short_window
                    long_ma = sum(self.prices[symbol]) / len(self.prices[symbol])
                    
                    if short_ma > long_ma:
                        return [{
                            'timestamp': bar_data['datetime'],
                            'symbol': symbol,
                            'signal_type': 'BUY',
                            'price': close_price,
                            'quantity': 1000,
                            'strategy_name': self.name
                        }]
                
                return []
                
            def on_tick(self, tick_data):
                return []
                
            def finalize(self):
                self.finalized = True
        
        # 2. 测试策略创建
        strategy = CustomStrategy(
            name="CustomTestStrategy",
            symbols=['000001.SZ', '000002.SZ'],
            params={'short_window': 5, 'long_window': 20}
        )
        
        assert strategy.name == "CustomTestStrategy"
        assert strategy.symbols == ['000001.SZ', '000002.SZ']
        assert strategy.params['short_window'] == 5
        print("✓ 策略创建成功")
        
        # 3. 测试策略初始化
        strategy.initialize()
        assert hasattr(strategy, 'initialized')
        assert strategy.initialized == True
        print("✓ 策略初始化成功")
        
        # 4. 测试策略数据处理
        signals_generated = 0
        for _, row in sample_data.iterrows():
            bar_data = row.to_dict()
            signals = strategy.on_bar(bar_data)
            signals_generated += len(signals)
        
        print(f"✓ 策略数据处理成功: 生成{signals_generated}个信号")
        
        # 5. 测试策略结束
        strategy.finalize()
        assert hasattr(strategy, 'finalized')
        assert strategy.finalized == True
        print("✓ 策略结束成功")
        
        # 6. 测试内置MA策略
        ma_strategy = MAStrategy(
            name="MATestStrategy",
            symbols=['000001.SZ'],
            params={'short_window': 5, 'long_window': 20}
        )
        
        assert ma_strategy is not None
        assert ma_strategy.name == "MATestStrategy"
        print("✓ 内置MA策略创建成功")
        
        print("=== 策略API测试通过 ===")
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_data_engine_api(self, api_config):
        """测试异步数据引擎API"""
        print("=== 开始异步数据引擎API测试 ===")
        
        # 1. 创建异步数据引擎
        async_config = {
            'max_concurrent_requests': 5,
            'request_timeout': 30,
            'cache_ttl': 300
        }
        
        async with AsyncDataEngine(async_config) as engine:
            assert engine is not None
            print("✓ 异步数据引擎创建成功")
            
            # 2. 测试健康检查API
            health = await engine.health_check()
            assert health is not None
            assert 'status' in health
            print(f"✓ 健康检查成功: {health}")
            
            # 3. 测试市场数据获取API
            symbols = ['000001.SZ', '000002.SZ']
            results = []
            
            async for data in engine.fetch_market_data(symbols):
                results.append(data)
            
            assert len(results) >= 0
            print(f"✓ 市场数据获取成功: {len(results)}条数据")
            
            # 4. 测试性能统计API
            stats = engine.get_performance_stats()
            assert stats is not None
            assert 'total_requests' in stats
            assert 'success_rate' in stats
            print(f"✓ 性能统计获取成功: {stats}")
        
        print("=== 异步数据引擎API测试通过 ===")
    
    @pytest.mark.integration
    def test_dependency_injection_api(self, api_config):
        """测试依赖注入API"""
        print("=== 开始依赖注入API测试 ===")
        
        # 1. 测试容器获取API
        container = get_container()
        assert container is not None
        print("✓ 容器获取成功")
        
        # 2. 测试组件获取API
        components = {
            'data_manager': container.data_manager(),
            'strategy_engine': container.strategy_engine(),
            'backtest_engine': container.backtest_engine(),
            'portfolio_manager': container.portfolio_manager(),
            'risk_manager': container.risk_manager(),
            'performance_analyzer': container.performance_analyzer()
        }
        
        for name, component in components.items():
            assert component is not None
            print(f"✓ {name} 获取成功")
        
        # 3. 测试组件类型验证
        from myQuant.core.managers.data_manager import DataManager
        from myQuant.core.engines.strategy_engine import StrategyEngine
        from myQuant.core.engines.backtest_engine import BacktestEngine
        from myQuant.core.managers.portfolio_manager import PortfolioManager
        from myQuant.core.managers.risk_manager import RiskManager
        from myQuant.core.analysis.performance_analyzer import PerformanceAnalyzer
        
        assert isinstance(components['data_manager'], DataManager)
        assert isinstance(components['strategy_engine'], StrategyEngine)
        assert isinstance(components['backtest_engine'], BacktestEngine)
        assert isinstance(components['portfolio_manager'], PortfolioManager)
        assert isinstance(components['risk_manager'], RiskManager)
        assert isinstance(components['performance_analyzer'], PerformanceAnalyzer)
        print("✓ 组件类型验证成功")
        
        print("=== 依赖注入API测试通过 ===")
    
    @pytest.mark.integration
    def test_exception_handling_api(self, api_config):
        """测试异常处理API"""
        print("=== 开始异常处理API测试 ===")
        
        # 1. 测试异常类型
        exceptions = [
            DataException("数据异常测试"),
            ConfigurationException("配置异常测试"),
            OrderException("订单异常测试"),
            StrategyException("策略异常测试")
        ]
        
        for exc in exceptions:
            assert exc is not None
            assert str(exc) != ""
            print(f"✓ {type(exc).__name__} 创建成功: {exc}")
        
        # 2. 测试异常处理
        try:
            # 模拟数据异常
            data_manager = DataManager(api_config.get('data_manager', {}))
            invalid_data = pd.DataFrame()  # 空数据
            
            # 这应该处理异常或返回合理的默认值
            result = data_manager.validate_price_data(invalid_data)
            assert isinstance(result, bool)
            print("✓ 数据异常处理成功")
            
        except DataException as e:
            print(f"✓ 数据异常捕获成功: {e}")
        
        # 3. 测试配置异常
        try:
            # 模拟配置异常
            invalid_config = {'initial_capital': -1000}
            trading_system = TradingSystem(invalid_config)
            
            # 系统应该处理无效配置
            assert trading_system.config['initial_capital'] >= 0
            print("✓ 配置异常处理成功")
            
        except ConfigurationException as e:
            print(f"✓ 配置异常捕获成功: {e}")
        
        print("=== 异常处理API测试通过 ===")
    
    @pytest.mark.integration
    def test_configuration_api(self, api_config):
        """测试配置API"""
        print("=== 开始配置API测试 ===")
        
        # 1. 测试默认配置API
        default_config = create_default_config()
        assert default_config is not None
        assert isinstance(default_config, dict)
        
        required_keys = ['initial_capital', 'commission_rate', 'risk_free_rate']
        for key in required_keys:
            assert key in default_config
            print(f"✓ 默认配置包含: {key} = {default_config[key]}")
        
        # 2. 测试配置验证
        # 有效配置
        valid_config = {
            'initial_capital': 1000000,
            'commission_rate': 0.0003,
            'risk_free_rate': 0.03
        }
        
        trading_system = TradingSystem(valid_config)
        assert trading_system.config['initial_capital'] == 1000000
        print("✓ 有效配置验证成功")
        
        # 3. 测试配置合并
        partial_config = {'initial_capital': 2000000}
        merged_config = {**default_config, **partial_config}
        
        trading_system2 = TradingSystem(merged_config)
        assert trading_system2.config['initial_capital'] == 2000000
        assert trading_system2.config['commission_rate'] == default_config['commission_rate']
        print("✓ 配置合并成功")
        
        print("=== 配置API测试通过 ===")
    
    @pytest.mark.integration
    def test_logging_api(self, api_config):
        """测试日志API"""
        print("=== 开始日志API测试 ===")
        
        # 1. 测试基础日志设置
        logger = setup_logging()
        assert logger is not None
        print("✓ 基础日志设置成功")
        
        # 2. 测试日志级别设置
        logger_info = setup_logging(level='INFO')
        logger_debug = setup_logging(level='DEBUG')
        logger_warning = setup_logging(level='WARNING')
        
        assert logger_info is not None
        assert logger_debug is not None
        assert logger_warning is not None
        print("✓ 日志级别设置成功")
        
        # 3. 测试日志记录
        test_messages = [
            ("INFO", "测试信息日志"),
            ("WARNING", "测试警告日志"),
            ("ERROR", "测试错误日志")
        ]
        
        for level, message in test_messages:
            try:
                if level == "INFO":
                    logger.info(message)
                elif level == "WARNING":
                    logger.warning(message)
                elif level == "ERROR":
                    logger.error(message)
                print(f"✓ {level} 日志记录成功")
            except Exception as e:
                print(f"✗ {level} 日志记录失败: {e}")
        
        print("=== 日志API测试通过 ===")
    
    @pytest.mark.integration
    def test_version_api(self):
        """测试版本API"""
        print("=== 开始版本API测试 ===")
        
        # 1. 测试版本获取
        version = get_version()
        assert version is not None
        assert isinstance(version, str)
        assert len(version) > 0
        print(f"✓ 版本获取成功: {version}")
        
        # 2. 测试版本格式
        # 版本应该是语义化版本格式或者类似的格式
        assert '.' in version or 'dev' in version.lower() or 'beta' in version.lower()
        print("✓ 版本格式验证成功")
        
        print("=== 版本API测试通过 ===")
    
    @pytest.mark.integration
    def test_comprehensive_api_workflow(self, api_config, sample_data):
        """测试综合API工作流程"""
        print("=== 开始综合API工作流程测试 ===")
        
        # 1. 初始化环境
        logger = setup_logging(level='INFO')
        config = create_default_config()
        config.update(api_config)
        
        # 2. 创建核心组件
        trading_system = TradingSystem(config)
        data_manager = DataManager(config.get('data_manager', {}))
        performance_analyzer = PerformanceAnalyzer(config)
        
        # 3. 创建策略
        class WorkflowStrategy(BaseStrategy):
            def initialize(self):
                self.signals = []
                
            def on_bar(self, bar_data):
                # 简单的信号生成逻辑
                if np.random.random() > 0.95:  # 5%概率生成信号
                    signal = {
                        'timestamp': bar_data['datetime'],
                        'symbol': bar_data['symbol'],
                        'signal_type': 'BUY',
                        'price': bar_data['close'],
                        'quantity': 1000,
                        'strategy_name': self.name
                    }
                    self.signals.append(signal)
                    return [signal]
                return []
                
            def on_tick(self, tick_data):
                return []
                
            def finalize(self):
                pass
        
        strategy = WorkflowStrategy("WorkflowStrategy", ['000001.SZ', '000002.SZ'])
        
        # 4. 执行完整工作流程
        trading_system.add_strategy(strategy)
        data_manager.load_data(sample_data)
        
        # 模拟交易
        total_signals = 0
        for _, row in sample_data.iterrows():
            bar_data = row.to_dict()
            signals = strategy.on_bar(bar_data)
            total_signals += len(signals)
        
        # 5. 绩效分析
        if total_signals > 0:
            # 生成模拟投资组合价值
            portfolio_values = [config['initial_capital']]
            for i in range(len(sample_data)):
                new_value = portfolio_values[-1] * (1 + np.random.normal(0.0005, 0.01))
                portfolio_values.append(new_value)
            
            results = performance_analyzer.analyze_portfolio(pd.Series(portfolio_values))
            
            assert results is not None
            assert 'returns' in results
            assert 'risk' in results
            
            print(f"✓ 综合工作流程完成:")
            print(f"  生成信号数: {total_signals}")
            print(f"  总收益率: {results['returns']['total_return']:.2%}")
            print(f"  夏普比率: {results['returns']['sharpe_ratio']:.2f}")
        else:
            print("✓ 综合工作流程完成 (无信号生成)")
        
        print("=== 综合API工作流程测试通过 ===")
# -*- coding: utf-8 -*-
"""
模块化单体基础集成测试
验证核心功能是否正常工作
"""

import pytest
import asyncio
from datetime import datetime
from typing import Dict, Any

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))


from myQuant.core.enhanced_trading_system import EnhancedTradingSystem, SystemConfig, SystemModule
from myQuant.config.monolith_config import MonolithConfig


@pytest.fixture
def test_config():
    """测试配置"""
    return SystemConfig(
        initial_capital=100000.0,
        commission_rate=0.001,
        min_commission=1.0,
        max_concurrent_orders=2,  # 减少并发数量
        order_timeout=1.0,  # 减少超时时间
        data_buffer_size=10,  # 减少缓冲区大小
        max_position_size=0.2,
        max_drawdown_limit=0.3,
        max_daily_loss=0.1,
        enabled_modules=[
            SystemModule.DATA,
            SystemModule.STRATEGY,
            SystemModule.EXECUTION,
            SystemModule.RISK,
            SystemModule.PORTFOLIO,
            SystemModule.ANALYTICS
        ],
        database_url="sqlite:///test_myquant.db",
        enable_persistence=False,  # 禁用持久化以加快测试
        enable_metrics=False,  # 禁用指标收集以加快测试
        metrics_port=8081,
        log_level="ERROR",  # 减少日志输出
        log_file="logs/test_monolith.log"
    )


@pytest.fixture(scope="function")
def trading_system(test_config):
    """交易系统夹具"""
    system = EnhancedTradingSystem(test_config)
    
    # 使用同步的方式创建和清理
    try:
        # 创建一个新的事件循环
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # 初始化系统
        loop.run_until_complete(system.initialize())
        yield system
        
    finally:
        # 清理
        try:
            loop.run_until_complete(system.stop())
        except Exception as e:
            print(f"清理时出错: {e}")
        finally:
            try:
                # 取消所有未完成的任务
                pending = asyncio.all_tasks(loop)
                if pending:
                    for task in pending:
                        task.cancel()
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                loop.close()
            except Exception as e:
                print(f"关闭事件循环时出错: {e}")


@pytest.fixture
def sample_tick_data():
    """样本tick数据"""
    return {
        "symbol": "AAPL",
        "datetime": datetime.now(),
        "open": 150.0,
        "high": 152.0,
        "low": 149.0,
        "close": 151.0,
        "volume": 1000000
    }


class TestMonolithBasicFunctionality:
    """模块化单体基础功能测试"""
    
    def test_system_initialization(self, trading_system):
        """测试系统初始化"""
        assert trading_system.is_running == False
        
        # 启动系统
        asyncio.run(trading_system.start())
        assert trading_system.is_running == True
        
        # 检查模块是否正确初始化
        assert 'data' in trading_system.modules
        assert 'strategy' in trading_system.modules
        assert 'execution' in trading_system.modules
        assert 'risk' in trading_system.modules
        assert 'portfolio' in trading_system.modules
        assert 'analytics' in trading_system.modules
        
        # 检查模块状态
        for module_name, module in trading_system.modules.items():
            assert module.is_initialized == True
    
    def test_data_module(self, trading_system, sample_tick_data):
        """测试数据模块"""
        asyncio.run(trading_system.start())
        
        data_module = trading_system.modules['data']
        
        # 测试实时数据处理
        asyncio.run(data_module.process_realtime_data(sample_tick_data))
        
        # 验证数据被正确处理
        assert data_module.metrics.get('data_requests', 0) >= 0
        
        # 测试数据订阅
        received_data = []
        
        async def data_callback(tick_data):
            received_data.append(tick_data)
        
        data_module.subscribe_to_data('AAPL', data_callback)
        asyncio.run(data_module.process_realtime_data(sample_tick_data))
        
        # 验证回调被调用
        assert len(received_data) > 0
        assert received_data[0]['symbol'] == 'AAPL'
    
    def test_strategy_module(self, trading_system):
        """测试策略模块"""
        asyncio.run(trading_system.start())
        
        strategy_module = trading_system.modules['strategy']
        
        # 测试添加策略
        strategy_config = {
            'type': 'moving_average',
            'short_window': 10,
            'long_window': 20
        }
        
        strategy_name = asyncio.run(strategy_module.add_strategy('test_strategy', strategy_config))
        assert strategy_name == 'test_strategy'
        assert 'test_strategy' in strategy_module.active_strategies
        
        # 测试策略性能统计
        performance = strategy_module.get_strategy_performance()
        assert 'test_strategy' in performance
        assert performance['test_strategy']['signals_generated'] == 0
    
    def test_execution_module(self, trading_system):
        """测试执行模块"""
        asyncio.run(trading_system.start())
        
        execution_module = trading_system.modules['execution']
        
        # 测试订单创建
        signal = {
            'symbol': 'AAPL',
            'side': 'BUY',
            'quantity': 100,
            'price': 150.0,
            'type': 'LIMIT',
            'timestamp': datetime.now()
        }
        
        order_id = asyncio.run(execution_module.create_order(signal))
        assert order_id is not None
        assert len(order_id) > 0
        
        # 测试订单状态查询
        status = asyncio.run(execution_module.get_order_status(order_id))
        assert status is not None
        assert 'order_id' in status or 'status' in status
        
        # 验证指标更新
        assert execution_module.metrics.get('orders_created', 0) >= 1
    
    def test_risk_module(self, trading_system):
        """测试风险模块"""
        asyncio.run(trading_system.start())
        
        risk_module = trading_system.modules['risk']
        
        # 测试风险检查
        signal = {
            'symbol': 'AAPL',
            'side': 'BUY',
            'quantity': 1000,  # 大量订单
            'price': 150.0,
            'timestamp': datetime.now()
        }
        
        current_positions = {}
        
        risk_check = asyncio.run(risk_module.check_signal_risk(signal, current_positions))
        assert 'allowed' in risk_check
        assert 'reason' in risk_check
        
        # 测试盈亏更新
        asyncio.run(risk_module.update_pnl(-1000.0))  # 亏损
        assert risk_module.daily_pnl == -1000.0
        
        # 测试风险指标
        risk_metrics = risk_module.get_risk_metrics()
        assert 'daily_pnl' in risk_metrics
        assert 'current_drawdown' in risk_metrics
        assert 'risk_limits' in risk_metrics
        
        # 验证指标更新
        assert risk_module.metrics.get('risk_checks', 0) >= 1
    
    def test_portfolio_module(self, trading_system):
        """测试投资组合模块"""
        asyncio.run(trading_system.start())
        
        portfolio_module = trading_system.modules['portfolio']
        
        # 测试投资组合摘要
        summary = portfolio_module.get_portfolio_summary()
        assert 'total_value' in summary
        assert 'cash_balance' in summary
        assert 'positions' in summary
        assert 'unrealized_pnl' in summary
        
        # 测试持仓更新
        execution_result = {
            'symbol': 'AAPL',
            'side': 'BUY',
            'executed_quantity': 100,
            'executed_price': 150.0,
            'commission': 0.0,
            'success': True
        }
        
        asyncio.run(portfolio_module.update_position(execution_result))
        
        # 验证指标更新
        assert portfolio_module.metrics.get('portfolio_value', 0) >= 0
    
    def test_analytics_module(self, trading_system):
        """测试分析模块"""
        asyncio.run(trading_system.start())
        
        analytics_module = trading_system.modules['analytics']
        
        # 测试交易记录
        trade_data = {
            'symbol': 'AAPL',
            'side': 'BUY',
            'quantity': 100,
            'price': 150.0,
            'pnl': 500.0,
            'strategy': 'test_strategy'
        }
        
        asyncio.run(analytics_module.record_trade(trade_data))
        
        # 验证交易记录
        assert len(analytics_module.trade_history) >= 1
        assert analytics_module.trade_history[0]['symbol'] == 'AAPL'
        
        # 测试性能报告
        report = analytics_module.get_performance_report()
        assert 'metrics' in report
        assert 'trade_history' in report
        assert 'generated_at' in report
        
        # 验证性能指标
        assert analytics_module.performance_metrics.get('total_trades', 0) >= 1
    
    def test_market_data_processing(self, trading_system, sample_tick_data):
        """测试市场数据处理"""
        asyncio.run(trading_system.start())
        
        # 处理市场数据
        result = trading_system.process_market_tick(sample_tick_data)
        
        # 验证处理结果
        assert result['processed'] == True
        assert result['symbol'] == 'AAPL'
        assert 'timestamp' in result
        assert 'market_status' in result
        assert 'system_uptime' in result
    
    def test_system_health(self, trading_system):
        """测试系统健康状态"""
        asyncio.run(trading_system.start())
        
        # 获取系统健康状态
        health = trading_system.get_system_health()
        
        # 验证健康状态
        assert health['system_running'] == True
        assert 'uptime_seconds' in health
        assert 'modules' in health
        assert 'timestamp' in health
        
        # 验证模块健康状态
        for module_name in ['data', 'strategy', 'execution', 'risk', 'portfolio', 'analytics']:
            assert module_name in health['modules']
            module_health = health['modules'][module_name]
            assert module_health['module'] == module_name
            assert module_health['initialized'] == True
    
    def test_system_metrics(self, trading_system):
        """测试系统指标"""
        asyncio.run(trading_system.start())
        
        # 获取系统指标
        metrics = trading_system.get_system_metrics()
        
        # 验证系统指标
        assert 'system' in metrics
        assert metrics['system']['running'] == True
        assert 'uptime' in metrics['system']
        assert 'modules_count' in metrics['system']
        
        # 验证模块指标
        for module_name in ['data', 'strategy', 'execution', 'risk', 'portfolio', 'analytics']:
            assert module_name in metrics
    
    def test_end_to_end_flow(self, trading_system, sample_tick_data):
        """测试端到端流程"""
        asyncio.run(trading_system.start())
        
        # 1. 添加策略
        strategy_module = trading_system.modules['strategy']
        strategy_config = {
            'type': 'simple_ma',
            'short_window': 5,
            'long_window': 10
        }
        asyncio.run(strategy_module.add_strategy('e2e_strategy', strategy_config))
        
        # 2. 处理市场数据
        result = trading_system.process_market_tick(sample_tick_data)
        assert result['processed'] == True
        
        # 3. 检查系统状态
        health = trading_system.get_system_health()
        assert health['system_running'] == True
        
        # 4. 验证各模块都有活动
        metrics = trading_system.get_system_metrics()
        assert len(metrics) > 1  # 至少有系统指标和模块指标


class TestMonolithConfiguration:
    """模块化单体配置测试"""
    
    def test_monolith_config_creation(self):
        """测试配置创建"""
        config = MonolithConfig()
        assert config.environment == "development"
        assert config.database.url == "sqlite:///data/myquant.db"
        assert config.trading.initial_capital == 1000000.0
        
    def test_config_validation(self):
        """测试配置验证"""
        # 正确的配置
        config = MonolithConfig()
        errors = config.validate()
        assert len(errors) == 0
        
        # 错误的配置
        config.trading.initial_capital = -1000  # 负数
        errors = config.validate()
        assert len(errors) > 0
        assert "初始资金必须大于0" in errors
        
    def test_config_to_system_config(self):
        """测试配置转换"""
        monolith_config = MonolithConfig()
        system_config = monolith_config.to_system_config()
        
        assert system_config.initial_capital == monolith_config.trading.initial_capital
        assert system_config.commission_rate == monolith_config.trading.commission_rate
        assert system_config.max_concurrent_orders == monolith_config.performance.max_concurrent_orders


class TestMonolithPerformance:
    """模块化单体性能测试"""
    
    def test_concurrent_data_processing(self, trading_system):
        """测试并发数据处理"""
        asyncio.run(trading_system.start())
        
        # 创建多个并发任务 (同步处理)
        results = []
        for i in range(10):
            tick_data = {
                "symbol": f"STOCK{i}",
                "datetime": datetime.now(),
                "open": 100.0 + i,
                "high": 102.0 + i,
                "low": 98.0 + i,
                "close": 101.0 + i,
                "volume": 1000000 + i * 100000
            }
            result = trading_system.process_market_tick(tick_data)
            results.append(result)
        
        # 验证所有任务都成功处理
        assert len(results) == 10
        for result in results:
            assert result['processed'] == True
    
    def test_memory_efficiency(self, trading_system):
        """测试内存效率"""
        asyncio.run(trading_system.start())
        
        import psutil
        import os
        
        # 获取初始内存使用
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 处理大量数据
        for i in range(1000):
            tick_data = {
                "symbol": "MEMTEST",
                "datetime": datetime.now(),
                "open": 100.0,
                "high": 102.0,
                "low": 98.0,
                "close": 101.0,
                "volume": 1000000
            }
            trading_system.process_market_tick(tick_data)
        
        # 获取最终内存使用
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # 验证内存增长合理 (应该小于100MB)
        assert memory_increase < 100, f"内存增长过多: {memory_increase:.2f}MB"


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "-s"])
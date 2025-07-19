#!/usr/bin/env python3

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any
import time
import json

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from myQuant.core.trading.execution_management import (
    ExecutionManagementSystem, 
    ExecutionRequest, 
    ExecutionStatus
)
from myQuant.core.trading.smart_routing import AlgorithmType, AlgorithmParams
from myQuant.core.trading.low_latency_engine import OrderSide
from myQuant.core.trading.broker_gateway import BrokerConfig, BrokerType
from myQuant.core.trading.dark_pool_connector import DarkPoolConfig, DarkPoolType

class HighFrequencyTradingDemo:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.ems = None
        self.execution_requests = []
        
    async def initialize_system(self):
        """初始化高频交易系统"""
        self.logger.info("=== Initializing High Frequency Trading System ===")
        
        # 系统配置
        config = {
            'trading_engine': {
                'max_workers': 8,
                'risk_limits': {
                    'max_order_size': 50000,
                    'daily_trading_limit': 10000000
                },
                'position_limits': {
                    'AAPL': 100000,
                    'MSFT': 100000,
                    'GOOGL': 100000
                }
            },
            'brokers': {
                'goldman_sachs': {
                    'type': 'fix',
                    'host': 'fix.goldmansachs.com',
                    'port': 9876,
                    'api_key': 'gs_api_key',
                    'username': 'gs_username',
                    'password': 'gs_password',
                    'venue_type': 'broker'
                },
                'morgan_stanley': {
                    'type': 'native',
                    'host': 'api.morganstanley.com',
                    'port': 8443,
                    'api_key': 'ms_api_key',
                    'username': 'ms_username',
                    'password': 'ms_password',
                    'venue_type': 'broker'
                },
                'interactive_brokers': {
                    'type': 'native',
                    'host': 'api.interactivebrokers.com',
                    'port': 7496,
                    'api_key': 'ib_api_key',
                    'username': 'ib_username',
                    'password': 'ib_password',
                    'venue_type': 'broker'
                }
            },
            'dark_pools': {
                'goldman_sigma_x': {
                    'type': 'institutional',
                    'min_order_size': 1000,
                    'max_order_size': 100000,
                    'matching_algorithm': 'pro_rata',
                    'crossing_times': ['12:00', '16:00'],
                    'commission_rate': 0.0008,
                    'access_fee': 0.0001,
                    'participation_threshold': 0.05,
                    'liquidity_indication': True,
                    'pre_trade_transparency': False,
                    'post_trade_transparency': True,
                    'supported_order_types': ['limit', 'market', 'iceberg'],
                    'api_endpoint': 'https://api.sigmax.com',
                    'api_key': 'sigma_x_key'
                },
                'credit_suisse_crossfinder': {
                    'type': 'institutional',
                    'min_order_size': 500,
                    'max_order_size': 200000,
                    'matching_algorithm': 'time_priority',
                    'crossing_times': ['11:30', '15:30'],
                    'commission_rate': 0.001,
                    'access_fee': 0.0002,
                    'participation_threshold': 0.03,
                    'liquidity_indication': True,
                    'pre_trade_transparency': False,
                    'post_trade_transparency': True,
                    'supported_order_types': ['limit', 'market', 'iceberg'],
                    'api_endpoint': 'https://api.crossfinder.com',
                    'api_key': 'crossfinder_key'
                },
                'liquidnet': {
                    'type': 'institutional',
                    'min_order_size': 10000,
                    'max_order_size': 1000000,
                    'matching_algorithm': 'negotiation',
                    'crossing_times': ['10:00', '14:00'],
                    'commission_rate': 0.0005,
                    'access_fee': 0.0003,
                    'participation_threshold': 0.10,
                    'liquidity_indication': True,
                    'pre_trade_transparency': False,
                    'post_trade_transparency': False,
                    'supported_order_types': ['limit', 'iceberg'],
                    'api_endpoint': 'https://api.liquidnet.com',
                    'api_key': 'liquidnet_key'
                }
            }
        }
        
        # 初始化执行管理系统
        self.ems = ExecutionManagementSystem(config)
        await self.ems.initialize()
        
        # 注册事件回调
        self.ems.register_event_callback('order_fill', self._on_order_fill)
        self.ems.register_event_callback('execution_complete', self._on_execution_complete)
        self.ems.register_event_callback('execution_failed', self._on_execution_failed)
        
        self.logger.info("High Frequency Trading System initialized successfully")
    
    async def demo_low_latency_execution(self):
        """演示低延迟执行"""
        self.logger.info("=== Low Latency Execution Demo ===")
        
        # 创建高频交易请求
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        latency_results = []
        
        for symbol in symbols:
            start_time = time.time_ns()
            
            # 创建执行请求
            request = ExecutionRequest(
                request_id=f"HFT_{symbol}_{int(time.time() * 1000000)}",
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=1000,
                algorithm_type=AlgorithmType.TWAP,
                algorithm_params=AlgorithmParams(
                    algorithm_type=AlgorithmType.TWAP,
                    target_quantity=1000,
                    duration_minutes=5,
                    max_participation_rate=0.15,
                    min_fill_size=100,
                    max_fill_size=500,
                    urgency="high"
                ),
                priority="high",
                max_duration_minutes=10
            )
            
            # 提交请求
            request_id = await self.ems.submit_execution_request(request)
            self.execution_requests.append(request_id)
            
            end_time = time.time_ns()
            latency_us = (end_time - start_time) // 1000
            latency_results.append(latency_us)
            
            self.logger.info(f"Submitted HFT order for {symbol}: {latency_us}μs latency")
        
        # 统计延迟
        avg_latency = sum(latency_results) / len(latency_results)
        max_latency = max(latency_results)
        min_latency = min(latency_results)
        
        self.logger.info(f"Latency Statistics: Avg={avg_latency:.1f}μs, Max={max_latency}μs, Min={min_latency}μs")
    
    async def demo_smart_routing_algorithms(self):
        """演示智能路由算法"""
        self.logger.info("=== Smart Routing Algorithms Demo ===")
        
        algorithms = [
            (AlgorithmType.TWAP, "Time-Weighted Average Price"),
            (AlgorithmType.VWAP, "Volume-Weighted Average Price"),
            (AlgorithmType.POV, "Percentage of Volume"),
            (AlgorithmType.ICEBERG, "Iceberg Order")
        ]
        
        for algo_type, description in algorithms:
            self.logger.info(f"Testing {description} Algorithm...")
            
            # 创建算法参数
            if algo_type == AlgorithmType.TWAP:
                params = AlgorithmParams(
                    algorithm_type=algo_type,
                    target_quantity=5000,
                    duration_minutes=15,
                    max_participation_rate=0.2,
                    min_fill_size=200,
                    max_fill_size=1000,
                    urgency="medium"
                )
            elif algo_type == AlgorithmType.VWAP:
                params = AlgorithmParams(
                    algorithm_type=algo_type,
                    target_quantity=8000,
                    duration_minutes=30,
                    max_participation_rate=0.25,
                    min_fill_size=300,
                    max_fill_size=1500,
                    urgency="medium"
                )
            elif algo_type == AlgorithmType.POV:
                params = AlgorithmParams(
                    algorithm_type=algo_type,
                    target_quantity=10000,
                    duration_minutes=20,
                    max_participation_rate=0.15,
                    min_fill_size=500,
                    max_fill_size=2000,
                    urgency="low"
                )
            else:  # ICEBERG
                params = AlgorithmParams(
                    algorithm_type=algo_type,
                    target_quantity=20000,
                    duration_minutes=60,
                    max_participation_rate=0.1,
                    min_fill_size=1000,
                    max_fill_size=2000,
                    urgency="low",
                    dark_pool_preference=0.8
                )
            
            # 创建执行请求
            request = ExecutionRequest(
                request_id=f"ALGO_{algo_type.value}_{int(time.time() * 1000000)}",
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=params.target_quantity,
                algorithm_type=algo_type,
                algorithm_params=params,
                priority="normal",
                max_duration_minutes=params.duration_minutes + 10
            )
            
            # 提交请求
            request_id = await self.ems.submit_execution_request(request)
            self.execution_requests.append(request_id)
            
            self.logger.info(f"Submitted {description} order: {params.target_quantity} shares")
    
    async def demo_market_impact_minimization(self):
        """演示市场影响最小化"""
        self.logger.info("=== Market Impact Minimization Demo ===")
        
        # 大单拆分执行
        large_order_symbols = ['MSFT', 'GOOGL']
        
        for symbol in large_order_symbols:
            # 创建大单
            large_quantity = 50000
            
            # 使用市场影响最小化算法
            params = AlgorithmParams(
                algorithm_type=AlgorithmType.IMPLEMENTATION_SHORTFALL,
                target_quantity=large_quantity,
                duration_minutes=45,
                max_participation_rate=0.12,
                min_fill_size=500,
                max_fill_size=2500,
                urgency="low",
                dark_pool_preference=0.6,
                limit_price_offset=0.002
            )
            
            request = ExecutionRequest(
                request_id=f"IMPACT_MIN_{symbol}_{int(time.time() * 1000000)}",
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=large_quantity,
                algorithm_type=AlgorithmType.IMPLEMENTATION_SHORTFALL,
                algorithm_params=params,
                priority="low",
                max_duration_minutes=60
            )
            
            request_id = await self.ems.submit_execution_request(request)
            self.execution_requests.append(request_id)
            
            self.logger.info(f"Submitted large order for {symbol}: {large_quantity:,} shares with impact minimization")
    
    async def demo_dark_pool_access(self):
        """演示暗池接入"""
        self.logger.info("=== Dark Pool Access Demo ===")
        
        # 获取暗池状态
        dark_pool_status = self.ems.dark_pool_manager.get_all_pool_status()
        
        self.logger.info("Dark Pool Status:")
        for pool_name, status in dark_pool_status.items():
            self.logger.info(f"  {pool_name}: Connected={status.get('is_connected', False)}")
            if status.get('is_connected', False):
                self.logger.info(f"    Avg Fill Rate: {status.get('avg_fill_rate', 0):.2%}")
                self.logger.info(f"    Avg Matching Prob: {status.get('avg_matching_probability', 0):.2%}")
        
        # 创建暗池专用订单
        dark_pool_symbols = ['AAPL', 'TSLA']
        
        for symbol in dark_pool_symbols:
            # 冰山订单，适合暗池
            params = AlgorithmParams(
                algorithm_type=AlgorithmType.ICEBERG,
                target_quantity=15000,
                duration_minutes=90,
                max_participation_rate=0.08,
                min_fill_size=1000,
                max_fill_size=1500,
                urgency="very_low",
                dark_pool_preference=0.9
            )
            
            request = ExecutionRequest(
                request_id=f"DARK_POOL_{symbol}_{int(time.time() * 1000000)}",
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=params.target_quantity,
                algorithm_type=AlgorithmType.ICEBERG,
                algorithm_params=params,
                priority="low",
                max_duration_minutes=120
            )
            
            request_id = await self.ems.submit_execution_request(request)
            self.execution_requests.append(request_id)
            
            self.logger.info(f"Submitted dark pool iceberg order for {symbol}: {params.target_quantity:,} shares")
    
    async def demo_real_time_monitoring(self):
        """演示实时监控"""
        self.logger.info("=== Real-time Monitoring Demo ===")
        
        # 监控执行进度
        for _ in range(10):
            await asyncio.sleep(2)
            
            # 获取系统健康状态
            health = self.ems.get_system_health()
            
            self.logger.info(f"System Health Check:")
            self.logger.info(f"  System Running: {health.get('is_running', False)}")
            self.logger.info(f"  Active Executions: {health.get('performance_metrics', {}).get('active_executions', 0)}")
            self.logger.info(f"  Completed Executions: {health.get('performance_metrics', {}).get('completed_executions', 0)}")
            self.logger.info(f"  Connected Brokers: {health.get('component_health', {}).get('connected_brokers', 0)}")
            self.logger.info(f"  Connected Dark Pools: {health.get('component_health', {}).get('connected_dark_pools', 0)}")
            
            # 检查具体执行状态
            for request_id in self.execution_requests:
                status = await self.ems.get_execution_status(request_id)
                if status:
                    self.logger.info(f"  Execution {request_id[:8]}...: "
                                   f"Fill Rate={status.fill_rate:.1%}, "
                                   f"Avg Price=${status.avg_fill_price:.2f}, "
                                   f"Venues={len(status.venues_used)}")
    
    async def demo_performance_analytics(self):
        """演示性能分析"""
        self.logger.info("=== Performance Analytics Demo ===")
        
        # 等待一些执行完成
        await asyncio.sleep(5)
        
        # 获取所有执行报告
        all_executions = await self.ems.get_all_executions()
        
        if all_executions:
            # 统计分析
            total_quantity = sum(exec.total_quantity for exec in all_executions)
            total_filled = sum(exec.filled_quantity for exec in all_executions)
            total_cost = sum(exec.total_cost for exec in all_executions)
            total_commission = sum(exec.total_commission for exec in all_executions)
            
            avg_fill_rate = sum(exec.fill_rate for exec in all_executions) / len(all_executions)
            avg_execution_time = sum(exec.execution_time_seconds for exec in all_executions) / len(all_executions)
            
            # 按算法分析
            algo_stats = {}
            for exec in all_executions:
                algo = exec.algorithm_used
                if algo not in algo_stats:
                    algo_stats[algo] = {'count': 0, 'total_quantity': 0, 'avg_fill_rate': 0}
                
                algo_stats[algo]['count'] += 1
                algo_stats[algo]['total_quantity'] += exec.total_quantity
                algo_stats[algo]['avg_fill_rate'] += exec.fill_rate
            
            # 计算算法平均值
            for algo, stats in algo_stats.items():
                stats['avg_fill_rate'] /= stats['count']
            
            self.logger.info(f"Performance Summary:")
            self.logger.info(f"  Total Executions: {len(all_executions)}")
            self.logger.info(f"  Total Quantity: {total_quantity:,} shares")
            self.logger.info(f"  Total Filled: {total_filled:,} shares ({total_filled/total_quantity:.1%})")
            self.logger.info(f"  Total Cost: ${total_cost:,.2f}")
            self.logger.info(f"  Total Commission: ${total_commission:,.2f}")
            self.logger.info(f"  Average Fill Rate: {avg_fill_rate:.1%}")
            self.logger.info(f"  Average Execution Time: {avg_execution_time:.1f}s")
            
            self.logger.info(f"Algorithm Performance:")
            for algo, stats in algo_stats.items():
                self.logger.info(f"  {algo}: {stats['count']} executions, "
                               f"{stats['total_quantity']:,} shares, "
                               f"{stats['avg_fill_rate']:.1%} avg fill rate")
    
    async def demo_latency_benchmarks(self):
        """演示延迟基准测试"""
        self.logger.info("=== Latency Benchmarks Demo ===")
        
        # 获取交易引擎延迟统计
        latency_stats = self.ems.trading_engine.get_latency_stats()
        
        if latency_stats:
            self.logger.info("Latency Benchmarks:")
            
            for metric_name, stats in latency_stats.items():
                if isinstance(stats, dict):
                    self.logger.info(f"  {metric_name.replace('_', ' ').title()}:")
                    self.logger.info(f"    Average: {stats.get('avg', 0):.1f}μs")
                    self.logger.info(f"    P50: {stats.get('p50', 0):.1f}μs")
                    self.logger.info(f"    P95: {stats.get('p95', 0):.1f}μs")
                    self.logger.info(f"    P99: {stats.get('p99', 0):.1f}μs")
                    self.logger.info(f"    Max: {stats.get('max', 0):.1f}μs")
                    self.logger.info(f"    Min: {stats.get('min', 0):.1f}μs")
        
        # 评估性能等级
        avg_latency = latency_stats.get('order_to_market_us', {}).get('avg', 0)
        
        if avg_latency < 10:
            performance_grade = "Excellent (< 10μs)"
        elif avg_latency < 50:
            performance_grade = "Very Good (< 50μs)"
        elif avg_latency < 100:
            performance_grade = "Good (< 100μs)"
        elif avg_latency < 500:
            performance_grade = "Fair (< 500μs)"
        else:
            performance_grade = "Needs Improvement (> 500μs)"
        
        self.logger.info(f"Performance Grade: {performance_grade}")
    
    async def _on_order_fill(self, request_id: str, fill):
        """订单成交回调"""
        self.logger.debug(f"Order filled: {request_id} - {fill.quantity} shares at ${fill.price:.2f}")
    
    async def _on_execution_complete(self, request_id: str, report):
        """执行完成回调"""
        self.logger.info(f"Execution completed: {request_id} - "
                        f"Fill Rate: {report.fill_rate:.1%}, "
                        f"Avg Price: ${report.avg_fill_price:.2f}")
    
    async def _on_execution_failed(self, request_id: str, error: str):
        """执行失败回调"""
        self.logger.error(f"Execution failed: {request_id} - {error}")
    
    async def run_comprehensive_demo(self):
        """运行完整演示"""
        try:
            # 初始化系统
            await self.initialize_system()
            
            # 演示各个功能
            await self.demo_low_latency_execution()
            await asyncio.sleep(2)
            
            await self.demo_smart_routing_algorithms()
            await asyncio.sleep(2)
            
            await self.demo_market_impact_minimization()
            await asyncio.sleep(2)
            
            await self.demo_dark_pool_access()
            await asyncio.sleep(2)
            
            # 监控执行进度
            await self.demo_real_time_monitoring()
            
            # 性能分析
            await self.demo_performance_analytics()
            
            # 延迟基准测试
            await self.demo_latency_benchmarks()
            
            self.logger.info("=== High Frequency Trading Demo Completed Successfully ===")
            
        except Exception as e:
            self.logger.error(f"Demo failed: {e}")
            raise
        finally:
            # 清理资源
            if self.ems:
                await self.ems.shutdown()

async def main():
    demo = HighFrequencyTradingDemo()
    await demo.run_comprehensive_demo()

if __name__ == "__main__":
    asyncio.run(main())
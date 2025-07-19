#!/usr/bin/env python3

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from myQuant.core.data import (
    MultiSourceDataManager,
    DataType,
    DataRequest,
    DataQualityMonitor,
    ShardedStorageManager
)
from myQuant.core.data.config import DATA_PROVIDER_CONFIGS

class EnhancedDataDemo:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.data_manager = MultiSourceDataManager(DATA_PROVIDER_CONFIGS)
        self.quality_monitor = DataQualityMonitor()
        self.storage_manager = ShardedStorageManager()
        
    async def initialize(self):
        """初始化数据管理器"""
        await self.data_manager.initialize()
        self.logger.info("Enhanced Data System initialized")
    
    async def demo_multi_source_data(self):
        """演示多数据源集成"""
        self.logger.info("=== Multi-Source Data Integration Demo ===")
        
        symbols = ['AAPL', '000001.SZ', 'GOOGL']
        
        for symbol in symbols:
            try:
                request = DataRequest(
                    symbol=symbol,
                    data_type=DataType.TICK,
                    frequency='1min'
                )
                
                # 获取多个数据源的数据
                multi_source_data = await self.data_manager.get_multi_source_data(request)
                
                self.logger.info(f"Symbol: {symbol}")
                for source, response in multi_source_data.items():
                    if response:
                        self.logger.info(f"  {source}: Quality Score = {response.quality_metrics.overall_score:.3f}, "
                                       f"Latency = {response.latency_ms:.1f}ms")
                    else:
                        self.logger.warning(f"  {source}: No data available")
                
            except Exception as e:
                self.logger.error(f"Error fetching data for {symbol}: {e}")
    
    async def demo_realtime_level2(self):
        """演示实时Level-2市场数据"""
        self.logger.info("=== Real-time Level-2 Market Data Demo ===")
        
        level2_provider = None
        for provider_name, provider in self.data_manager.providers.items():
            if provider_name == 'level2':
                level2_provider = provider
                break
        
        if not level2_provider:
            self.logger.warning("Level-2 provider not available")
            return
        
        symbols = ['AAPL', 'MSFT']
        received_count = 0
        
        async def data_callback(response):
            nonlocal received_count
            received_count += 1
            
            if received_count <= 5:  # 只显示前5条数据
                data = response.data
                if 'bids' in data and 'asks' in data:
                    best_bid = data['bids'][0]['price'] if data['bids'] else 0
                    best_ask = data['asks'][0]['price'] if data['asks'] else 0
                    spread = data.get('spread', 0)
                    
                    self.logger.info(f"Level-2 {data['symbol']}: "
                                   f"Bid={best_bid:.2f}, Ask={best_ask:.2f}, "
                                   f"Spread={spread:.4f}, Latency={response.latency_ms:.1f}ms")
        
        # 订阅实时数据
        await self.data_manager.subscribe_realtime(symbols, DataType.ORDERBOOK, data_callback)
        
        # 运行5秒钟
        await asyncio.sleep(5)
        
        # 取消订阅
        for provider_name, provider in self.data_manager.providers.items():
            if provider_name == 'level2':
                await provider.unsubscribe_realtime(symbols, DataType.ORDERBOOK)
        
        self.logger.info(f"Received {received_count} Level-2 updates")
    
    async def demo_fundamental_data(self):
        """演示基本面数据集成"""
        self.logger.info("=== Fundamental Data Integration Demo ===")
        
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        
        for symbol in symbols:
            try:
                request = DataRequest(
                    symbol=symbol,
                    data_type=DataType.FUNDAMENTAL
                )
                
                response = await self.data_manager.get_data(request, preferred_provider='bloomberg')
                
                if response and response.data:
                    data = response.data
                    self.logger.info(f"Fundamental Data for {symbol}:")
                    self.logger.info(f"  Revenue: ${data.get('revenue', 0):,.0f}")
                    self.logger.info(f"  Net Profit: ${data.get('net_profit', 0):,.0f}")
                    self.logger.info(f"  P/E Ratio: {data.get('pe_ratio', 0):.2f}")
                    self.logger.info(f"  Market Cap: ${data.get('market_cap', 0):,.0f}")
                    self.logger.info(f"  Quality Score: {response.quality_metrics.overall_score:.3f}")
                
            except Exception as e:
                self.logger.error(f"Error fetching fundamental data for {symbol}: {e}")
    
    async def demo_alternative_data(self):
        """演示另类数据源"""
        self.logger.info("=== Alternative Data Sources Demo ===")
        
        symbols = ['AAPL', 'TSLA']
        
        for symbol in symbols:
            try:
                request = DataRequest(
                    symbol=symbol,
                    data_type=DataType.ALTERNATIVE
                )
                
                response = await self.data_manager.get_data(request, preferred_provider='alpha')
                
                if response and response.data:
                    data = response.data
                    self.logger.info(f"Alternative Data for {symbol}:")
                    
                    if 'social_sentiment' in data:
                        sentiment = data['social_sentiment']
                        self.logger.info(f"  Social Sentiment: {sentiment.get('sentiment_score', 0):.3f}")
                        self.logger.info(f"  Twitter Mentions: {sentiment.get('twitter_mentions', 0)}")
                    
                    if 'satellite_data' in data:
                        satellite = data['satellite_data']
                        self.logger.info(f"  Foot Traffic: {satellite.get('retail_foot_traffic', 0):.1f}%")
                    
                    if 'patent_activity' in data:
                        patent = data['patent_activity']
                        self.logger.info(f"  Patents Filed: {patent.get('patents_filed', 0)}")
                
            except Exception as e:
                self.logger.error(f"Error fetching alternative data for {symbol}: {e}")
    
    async def demo_data_quality_monitoring(self):
        """演示数据质量监控"""
        self.logger.info("=== Data Quality Monitoring Demo ===")
        
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        
        for symbol in symbols:
            try:
                request = DataRequest(
                    symbol=symbol,
                    data_type=DataType.TICK
                )
                
                response = await self.data_manager.get_data(request)
                
                if response:
                    metrics = response.quality_metrics
                    self.logger.info(f"Quality Metrics for {symbol}:")
                    self.logger.info(f"  Completeness: {metrics.completeness:.3f}")
                    self.logger.info(f"  Accuracy: {metrics.accuracy:.3f}")
                    self.logger.info(f"  Timeliness: {metrics.timeliness:.3f}")
                    self.logger.info(f"  Consistency: {metrics.consistency:.3f}")
                    self.logger.info(f"  Validity: {metrics.validity:.3f}")
                    self.logger.info(f"  Overall Score: {metrics.overall_score:.3f}")
                    self.logger.info(f"  Anomaly Count: {metrics.anomaly_count}")
                
            except Exception as e:
                self.logger.error(f"Error in quality monitoring for {symbol}: {e}")
        
        # 获取整体质量指标
        overall_metrics = await self.quality_monitor.get_overall_metrics()
        self.logger.info(f"Overall System Metrics:")
        self.logger.info(f"  Overall Score: {overall_metrics.get('overall_score', 0):.3f}")
        self.logger.info(f"  Total Alerts: {overall_metrics.get('total_alerts', 0)}")
        self.logger.info(f"  Data Sources Monitored: {overall_metrics.get('data_sources_monitored', 0)}")
    
    async def demo_historical_backfill(self):
        """演示历史数据回填"""
        self.logger.info("=== Historical Data Backfill Demo ===")
        
        symbols = ['AAPL', 'MSFT']
        start_date = datetime.now() - timedelta(days=7)
        end_date = datetime.now()
        
        self.logger.info(f"Starting backfill for {len(symbols)} symbols from {start_date.date()} to {end_date.date()}")
        
        successful_backfills = await self.data_manager.backfill_historical_data(
            symbols, start_date, end_date, DataType.KLINE
        )
        
        self.logger.info(f"Successfully backfilled {successful_backfills}/{len(symbols)} symbols")
    
    async def demo_storage_system(self):
        """演示分片存储系统"""
        self.logger.info("=== Sharded Storage System Demo ===")
        
        # 获取存储统计信息
        stats = await self.storage_manager.get_storage_stats()
        
        self.logger.info(f"Storage Statistics:")
        self.logger.info(f"  Total Shards: {stats.get('total_shards', 0)}")
        self.logger.info(f"  Total Size: {stats.get('total_size_mb', 0):.1f} MB")
        self.logger.info(f"  Oldest Data: {stats.get('oldest_data', 'N/A')}")
        self.logger.info(f"  Newest Data: {stats.get('newest_data', 'N/A')}")
        
        # 数据类型统计
        for data_type, type_stats in stats.get('data_type_stats', {}).items():
            self.logger.info(f"  {data_type}: {type_stats.get('shard_count', 0)} shards, "
                           f"{type_stats.get('size_mb', 0):.1f} MB")
    
    async def demo_system_health(self):
        """演示系统健康状态"""
        self.logger.info("=== System Health Status Demo ===")
        
        health_status = await self.data_manager.get_health_status()
        
        self.logger.info(f"System Health:")
        self.logger.info(f"  Total Providers: {health_status.get('total_providers', 0)}")
        self.logger.info(f"  Active Providers: {health_status.get('active_providers', 0)}")
        
        # 提供商状态
        provider_status = health_status.get('provider_status', {})
        for provider, status in provider_status.items():
            connection_status = "Connected" if status.get('is_connected', False) else "Disconnected"
            self.logger.info(f"  {provider}: {connection_status}")
        
        # 电路断路器状态
        circuit_breakers = health_status.get('circuit_breakers', {})
        for provider, breaker in circuit_breakers.items():
            if breaker.get('is_open', False):
                self.logger.warning(f"  Circuit Breaker OPEN for {provider}")
    
    async def run_all_demos(self):
        """运行所有演示"""
        try:
            await self.initialize()
            
            await self.demo_multi_source_data()
            await asyncio.sleep(1)
            
            await self.demo_realtime_level2()
            await asyncio.sleep(1)
            
            await self.demo_fundamental_data()
            await asyncio.sleep(1)
            
            await self.demo_alternative_data()
            await asyncio.sleep(1)
            
            await self.demo_data_quality_monitoring()
            await asyncio.sleep(1)
            
            await self.demo_historical_backfill()
            await asyncio.sleep(1)
            
            await self.demo_storage_system()
            await asyncio.sleep(1)
            
            await self.demo_system_health()
            
            self.logger.info("=== All Enhanced Data Demos Completed ===")
            
        except Exception as e:
            self.logger.error(f"Error in demo execution: {e}")
            raise

async def main():
    demo = EnhancedDataDemo()
    await demo.run_all_demos()

if __name__ == "__main__":
    asyncio.run(main())
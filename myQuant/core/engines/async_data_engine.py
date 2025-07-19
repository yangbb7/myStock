# -*- coding: utf-8 -*-
"""
异步数据引擎 - 提供高性能的并发数据获取能力
"""

import asyncio
import aiohttp
import logging
import time
from typing import List, Dict, Any, AsyncGenerator, Optional, Callable
from datetime import datetime
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import secrets

from ..exceptions import DataException, NetworkException
from ..monitoring.exception_logger import ExceptionLogger


class AsyncDataEngine:
    """异步数据引擎 - 提供并发数据获取和处理能力"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.exception_logger = ExceptionLogger()

        # 配置参数
        self.max_concurrent_requests = self.config.get(
            "max_concurrent_requests", 10
        )
        self.request_timeout = self.config.get("request_timeout", 30)
        self.retry_attempts = self.config.get("retry_attempts", 3)
        self.retry_delay = self.config.get("retry_delay", 1)
        self.rate_limit_per_second = self.config.get(
            "rate_limit_per_second", 10
        )

        # 会话池
        self._session_pool = []
        self._semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        self._rate_limiter = asyncio.Semaphore(self.rate_limit_per_second)

        # 缓存和统计
        self._cache = {}
        self._cache_ttl = self.config.get("cache_ttl", 300)  # 5分钟缓存
        self._request_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cache_hits": 0,
            "average_response_time": 0,
        }

        self.logger.info("异步数据引擎初始化完成")

    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self._initialize_session_pool()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self._cleanup_session_pool()

    async def _initialize_session_pool(self):
        """初始化会话池"""
        timeout = aiohttp.ClientTimeout(total=self.request_timeout)
        connector = aiohttp.TCPConnector(
            limit=self.max_concurrent_requests,
            limit_per_host=5,
            ttl_dns_cache=300,
            use_dns_cache=True,
        )

        for _ in range(min(3, self.max_concurrent_requests)):
            session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={
                    "User-Agent": "MyQuant-AsyncEngine/1.0",
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip, deflate",
                },
            )
            self._session_pool.append(session)

    async def _cleanup_session_pool(self):
        """清理会话池"""
        for session in self._session_pool:
            await session.close()
        self._session_pool.clear()

    def _get_session(self) -> aiohttp.ClientSession:
        """获取会话对象"""
        if not self._session_pool:
            raise NetworkException(
                "Session pool not initialized", error_code="SESSION_POOL_EMPTY"
            )
        return self._session_pool[0]  # 简化实现，使用第一个会话

    async def fetch_market_data(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        并发获取多个股票的市场数据

        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期

        Yields:
            Dict: 包含股票数据的字典
        """
        if not symbols:
            return

        try:
            # 创建所有请求任务
            tasks = []
            for symbol in symbols:
                task = asyncio.create_task(
                    self._fetch_symbol_data(symbol, start_date, end_date)
                )
                tasks.append(task)

            # 并发执行并逐个返回结果
            for coro in asyncio.as_completed(tasks):
                try:
                    result = await coro
                    if result:
                        yield result
                except Exception as e:
                    self.exception_logger.log_exception(
                        e, {"operation": "fetch_market_data", "symbols": symbols}
                    )
                    yield {"error": str(e), "symbols": symbols}  # type: ignore

        except Exception as e:
            self.exception_logger.log_exception(
                e,
                {"operation": "fetch_market_data_batch", "symbols_count": len(symbols)},
            )
            raise DataException(f"批量获取市场数据失败: {e}", cause=e)

    async def _fetch_symbol_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        获取单个股票的数据

        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            Dict: 股票数据
        """
        cache_key = f"{symbol}_{start_date}_{end_date}"

        # 检查缓存
        if self._is_cache_valid(cache_key):
            self._request_stats["cache_hits"] += 1
            return self._cache[cache_key]["data"]

        async with self._semaphore:  # 限制并发数
            async with self._rate_limiter:  # 限制请求频率
                start_time = time.time()

                try:
                    # 模拟数据获取 - 实际实现中应该调用真实的数据API
                    data = await self._simulate_api_call(symbol, start_date, end_date)

                    # 更新缓存
                    self._cache[cache_key] = {"data": data, "timestamp": time.time()}

                    # 更新统计
                    response_time = time.time() - start_time
                    self._update_request_stats(True, response_time)

                    return data

                except Exception as e:
                    self._update_request_stats(False, time.time() - start_time)
                    raise DataException(f"获取股票 {symbol} 数据失败: {e}", cause=e)

    async def _simulate_api_call(
        self, symbol: str, start_date: Optional[str], end_date: Optional[str]
    ) -> Dict[str, Any]:
        """
        模拟API调用 - 实际实现中应该替换为真实的数据源

        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            Dict: 模拟的股票数据
        """
        # 模拟网络延迟
        delay = 0.1 + secrets.randbelow(50) / 1000  # 0.1-0.15秒随机延迟
        await asyncio.sleep(delay)

        # 生成模拟数据
        days = 30 if not start_date or not end_date else 30
        dates = pd.date_range(start="2023-01-01", periods=days, freq="D")

        # 使用更安全的随机数生成
        base_price = 10 + (secrets.randbelow(4000) / 100)  # 10-50的随机价格
        prices = []

        for i in range(days):
            # 模拟价格波动（使用更安全的随机数）
            change_percent = (secrets.randbelow(401) - 200) / 10000  # -2% to +2%
            base_price = max(0.01, base_price * (1 + change_percent))

            # 生成OHLC数据
            open_factor = 0.98 + (secrets.randbelow(401) / 10000)  # 0.98-1.02
            high_factor = 1.00 + (secrets.randbelow(501) / 10000)  # 1.00-1.05
            low_factor = 0.95 + (secrets.randbelow(501) / 10000)  # 0.95-1.00

            open_price = base_price * open_factor
            high_price = max(open_price, base_price) * high_factor
            low_price = min(open_price, base_price) * low_factor

            prices.append(
                {
                    "datetime": dates[i].strftime("%Y-%m-%d"),
                    "symbol": symbol,
                    "open": round(open_price, 2),
                    "high": round(high_price, 2),
                    "low": round(low_price, 2),
                    "close": round(base_price, 2),
                    "volume": 1000000 + secrets.randbelow(9000001),  # 1M-10M
                    "adj_close": round(base_price, 2),
                }
            )

        return {
            "symbol": symbol,
            "data": prices,
            "metadata": {
                "source": "async_engine_simulation",
                "fetch_time": datetime.now().isoformat(),
                "data_count": len(prices),
            },
        }

    async def fetch_real_time_data(
        self, symbols: List[str]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        获取实时行情数据

        Args:
            symbols: 股票代码列表

        Yields:
            Dict: 实时行情数据
        """
        try:
            while True:
                tasks = [self._fetch_real_time_quote(symbol) for symbol in symbols]

                results = await asyncio.gather(*tasks, return_exceptions=True)

                for result in results:
                    if isinstance(result, Exception):
                        self.logger.warning(f"实时数据获取失败: {result}")
                        continue

                    if result:
                        yield result

                # 等待下一次更新
                await asyncio.sleep(1)

        except asyncio.CancelledError:
            self.logger.info("实时数据流已取消")
        except Exception as e:
            self.exception_logger.log_exception(
                e, {"operation": "fetch_real_time_data", "symbols": symbols}
            )
            raise DataException(f"实时数据获取失败: {e}", cause=e)

    async def _fetch_real_time_quote(self, symbol: str) -> Dict[str, Any]:
        """获取单个股票的实时报价"""
        async with self._rate_limiter:
            # 模拟实时数据
            await asyncio.sleep(0.01)

            # 使用更安全的随机数生成
            base_price = 10 + (secrets.randbelow(4001) / 100)  # 10-50
            change = (secrets.randbelow(401) - 200) / 100  # -2 to +2
            change_percent = (secrets.randbelow(1001) - 500) / 100  # -5% to +5%

            return {
                "symbol": symbol,
                "price": round(base_price, 2),
                "change": round(change, 2),
                "change_percent": round(change_percent, 2),
                "volume": 100000 + secrets.randbelow(900001),  # 100K-1M
                "timestamp": datetime.now().isoformat(),
                "bid": round(base_price * 0.999, 2),
                "ask": round(base_price * 1.001, 2),
                "bid_size": 100 + secrets.randbelow(901),  # 100-1000
                "ask_size": 100 + secrets.randbelow(901),  # 100-1000
            }

    async def fetch_fundamental_data(self, symbols: List[str]) -> Dict[str, Any]:
        """
        批量获取基本面数据

        Args:
            symbols: 股票代码列表

        Returns:
            Dict: 包含所有股票基本面数据的字典
        """
        try:
            tasks = [self._fetch_symbol_fundamentals(symbol) for symbol in symbols]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            fundamental_data = {}
            for symbol, result in zip(symbols, results):
                if isinstance(result, Exception):
                    self.logger.warning(f"获取 {symbol} 基本面数据失败: {result}")
                    fundamental_data[symbol] = None
                else:
                    fundamental_data[symbol] = result

            return fundamental_data

        except Exception as e:
            self.exception_logger.log_exception(
                e, {"operation": "fetch_fundamental_data", "symbols": symbols}
            )
            raise DataException(f"批量获取基本面数据失败: {e}", cause=e)

    async def _fetch_symbol_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """获取单个股票的基本面数据"""
        async with self._semaphore:
            # 模拟基本面数据获取
            await asyncio.sleep(0.2)

            # 使用更安全的随机数生成
            pe_ratio = 5 + (secrets.randbelow(4501) / 100)  # 5-50
            pb_ratio = 0.5 + (secrets.randbelow(451) / 100)  # 0.5-5.0
            roe = 0.05 + (secrets.randbelow(2001) / 10000)  # 0.05-0.25
            debt_ratio = 0.1 + (secrets.randbelow(7001) / 10000)  # 0.1-0.8
            dividend_yield = secrets.randbelow(801) / 10000  # 0-0.08
            market_cap = 1000000000 + secrets.randbelow(99000000001)  # 1B-100B
            revenue_growth = (secrets.randbelow(7001) - 2000) / 10000  # -0.2-0.5
            net_profit_growth = (secrets.randbelow(11001) - 3000) / 10000  # -0.3-0.8

            return {
                "symbol": symbol,
                "pe_ratio": round(pe_ratio, 2),
                "pb_ratio": round(pb_ratio, 2),
                "roe": round(roe, 4),
                "debt_ratio": round(debt_ratio, 4),
                "dividend_yield": round(dividend_yield, 4),
                "market_cap": market_cap,
                "revenue_growth": round(revenue_growth, 4),
                "net_profit_growth": round(net_profit_growth, 4),
                "update_time": datetime.now().isoformat(),
            }

    async def process_data_batch(
        self,
        data_batch: List[Dict[str, Any]],
        processor_func: Callable[[Dict[str, Any]], Any],
    ) -> List[Any]:
        """
        并发处理数据批次

        Args:
            data_batch: 数据批次
            processor_func: 处理函数

        Returns:
            List: 处理结果列表
        """
        try:
            # 使用线程池处理CPU密集型任务
            loop = asyncio.get_event_loop()

            with ThreadPoolExecutor(
                max_workers=self.max_concurrent_requests
            ) as executor:
                tasks = [
                    loop.run_in_executor(executor, processor_func, data_item)
                    for data_item in data_batch
                ]

                results = await asyncio.gather(*tasks, return_exceptions=True)

                processed_results = []
                for result in results:
                    if isinstance(result, Exception):
                        self.logger.warning(f"数据处理失败: {result}")
                        processed_results.append(None)
                    else:
                        processed_results.append(result)

                return processed_results

        except Exception as e:
            self.exception_logger.log_exception(
                e, {"operation": "process_data_batch", "batch_size": len(data_batch)}
            )
            raise DataException(f"批量数据处理失败: {e}", cause=e)

    def _is_cache_valid(self, cache_key: str) -> bool:
        """检查缓存是否有效"""
        if cache_key not in self._cache:
            return False

        cache_entry = self._cache[cache_key]
        return (time.time() - cache_entry["timestamp"]) < self._cache_ttl

    def _update_request_stats(self, success: bool, response_time: float):
        """更新请求统计信息"""
        self._request_stats["total_requests"] += 1

        if success:
            self._request_stats["successful_requests"] += 1
        else:
            self._request_stats["failed_requests"] += 1

        # 更新平均响应时间
        total_requests = self._request_stats["total_requests"]
        current_avg = self._request_stats["average_response_time"]
        new_avg = (current_avg * (total_requests - 1) + response_time) / total_requests
        self._request_stats["average_response_time"] = new_avg

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        stats = self._request_stats.copy()
        stats["cache_size"] = len(self._cache)
        total_requests = max(stats["total_requests"], 1)
        stats["success_rate"] = stats["successful_requests"] / total_requests
        return stats

    def clear_cache(self):
        """清理缓存"""
        self._cache.clear()
        self.logger.info("缓存已清理")

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 测试基本功能
            test_symbol = "000001.SZ"
            start_time = time.time()

            await self._fetch_symbol_data(test_symbol)
            response_time = time.time() - start_time

            return {
                "status": "healthy",
                "response_time": response_time,
                "session_pool_size": len(self._session_pool),
                "cache_size": len(self._cache),
                "last_check": datetime.now().isoformat(),
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "last_check": datetime.now().isoformat(),
            }


# 便利函数
async def fetch_multiple_symbols(
    symbols: List[str], config: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    便利函数：批量获取多个股票数据

    Args:
        symbols: 股票代码列表
        config: 配置参数

    Returns:
        List: 股票数据列表
    """
    async with AsyncDataEngine(config) as engine:
        results = []
        async for data in engine.fetch_market_data(symbols):
            results.append(data)
        return results


async def fetch_real_time_quotes(
    symbols: List[str], duration: int = 60, config: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    便利函数：获取实时行情数据

    Args:
        symbols: 股票代码列表
        duration: 持续时间（秒）
        config: 配置参数

    Returns:
        List: 实时行情数据列表
    """
    async with AsyncDataEngine(config) as engine:
        results = []
        start_time = time.time()

        async for quote in engine.fetch_real_time_data(symbols):
            results.append(quote)

            if time.time() - start_time > duration:
                break

        return results

# -*- coding: utf-8 -*-
"""
并发市场数据处理器 - 实现高性能的市场数据并发处理
"""

import asyncio
import logging
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ..exceptions import DataException, ProcessingException
from ..interfaces.async_data_interface import IAsyncDataProcessor
from ..monitoring.exception_logger import ExceptionLogger


class ProcessingPriority(Enum):
    """处理优先级"""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ProcessingTask:
    """处理任务"""

    task_id: str
    data: Any
    processor_func: Callable
    priority: ProcessingPriority = ProcessingPriority.NORMAL
    metadata: Dict[str, Any] = None
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.metadata is None:
            self.metadata = {}


class ConcurrentMarketProcessor(IAsyncDataProcessor):
    """并发市场数据处理器"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.exception_logger = ExceptionLogger()

        # 配置参数
        self.max_workers = self.config.get("max_workers", min(32, mp.cpu_count() + 4))
        self.thread_pool_size = self.config.get("thread_pool_size", self.max_workers)
        self.process_pool_size = self.config.get("process_pool_size", mp.cpu_count())
        self.queue_size = self.config.get("queue_size", 1000)
        self.batch_size = self.config.get("batch_size", 100)
        self.timeout = self.config.get("timeout", 300)

        # 执行器池
        self._thread_executor = None
        self._process_executor = None

        # 任务队列
        self._task_queue = asyncio.Queue(maxsize=self.queue_size)
        self._priority_queue = asyncio.PriorityQueue()
        self._result_cache = {}

        # 统计信息
        self._stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "avg_processing_time": 0,
            "queue_size": 0,
        }

        # 运行状态
        self._running = False
        self._workers = []

        self.logger.info("并发市场数据处理器初始化完成")

    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.stop()

    async def start(self):
        """启动处理器"""
        if self._running:
            return

        self._running = True

        # 初始化执行器
        self._thread_executor = ThreadPoolExecutor(
            max_workers=self.thread_pool_size, thread_name_prefix="MarketProcessor"
        )
        self._process_executor = ProcessPoolExecutor(max_workers=self.process_pool_size)

        # 启动工作线程
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self._workers.append(worker)

        self.logger.info(f"并发处理器已启动，工作线程数: {self.max_workers}")

    async def stop(self):
        """停止处理器"""
        if not self._running:
            return

        self._running = False

        # 取消所有工作线程
        for worker in self._workers:
            worker.cancel()

        # 等待工作线程结束
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

        # 关闭执行器
        if self._thread_executor:
            self._thread_executor.shutdown(wait=True)
        if self._process_executor:
            self._process_executor.shutdown(wait=True)

        self.logger.info("并发处理器已停止")

    async def _worker(self, worker_name: str):
        """工作线程"""
        self.logger.debug(f"工作线程 {worker_name} 启动")

        try:
            while self._running:
                try:
                    # 从优先级队列获取任务
                    try:
                        priority, task = await asyncio.wait_for(
                            self._priority_queue.get(), timeout=1.0
                        )
                        task: ProcessingTask = task
                    except asyncio.TimeoutError:
                        continue

                    # 处理任务
                    await self._process_task(task, worker_name)

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.exception_logger.log_exception(
                        e, {"worker": worker_name, "operation": "worker_loop"}
                    )

        except asyncio.CancelledError:
            pass
        finally:
            self.logger.debug(f"工作线程 {worker_name} 结束")

    async def _process_task(self, task: ProcessingTask, worker_name: str):
        """处理单个任务"""
        start_time = time.time()
        self._stats["total_tasks"] += 1

        try:
            # 选择执行器类型
            if task.metadata.get("use_process_pool", False):
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self._process_executor, task.processor_func, task.data
                )
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self._thread_executor, task.processor_func, task.data
                )

            # 缓存结果
            self._result_cache[task.task_id] = {
                "result": result,
                "timestamp": time.time(),
                "processing_time": time.time() - start_time,
            }

            self._stats["completed_tasks"] += 1

            # 更新平均处理时间
            processing_time = time.time() - start_time
            total_tasks = self._stats["completed_tasks"]
            current_avg = self._stats["avg_processing_time"]
            self._stats["avg_processing_time"] = (
                current_avg * (total_tasks - 1) + processing_time
            ) / total_tasks

            self.logger.debug(
                f"任务 {task.task_id} 处理完成，耗时: {processing_time:.3f}s"
            )

        except Exception as e:
            self._stats["failed_tasks"] += 1
            self.exception_logger.log_exception(
                e,
                {
                    "task_id": task.task_id,
                    "worker": worker_name,
                    "operation": "process_task",
                },
            )

            # 缓存错误结果
            self._result_cache[task.task_id] = {
                "error": str(e),
                "timestamp": time.time(),
                "processing_time": time.time() - start_time,
            }

    async def submit_task(self, task: ProcessingTask) -> str:
        """
        提交处理任务

        Args:
            task: 处理任务

        Returns:
            str: 任务ID
        """
        if not self._running:
            await self.start()

        try:
            # 添加到优先级队列
            priority_value = task.priority.value
            await self._priority_queue.put((priority_value, task))

            self._stats["queue_size"] = self._priority_queue.qsize()

            self.logger.debug(f"任务 {task.task_id} 已提交到队列")
            return task.task_id

        except Exception as e:
            self.exception_logger.log_exception(
                e, {"task_id": task.task_id, "operation": "submit_task"}
            )
            raise ProcessingException(f"提交任务失败: {e}", cause=e)

    async def get_result(self, task_id: str, timeout: float = None) -> Any:
        """
        获取任务结果

        Args:
            task_id: 任务ID
            timeout: 超时时间

        Returns:
            Any: 任务结果
        """
        timeout = timeout or self.timeout
        start_time = time.time()

        while time.time() - start_time < timeout:
            if task_id in self._result_cache:
                result_data = self._result_cache[task_id]

                if "error" in result_data:
                    raise ProcessingException(f"任务执行失败: {result_data['error']}")

                return result_data["result"]

            await asyncio.sleep(0.1)

        raise ProcessingException(f"获取任务 {task_id} 结果超时")

    async def process_price_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        处理价格数据

        Args:
            data: 原始价格数据

        Returns:
            DataFrame: 处理后的价格数据
        """

        def _process_price_data_sync(df: pd.DataFrame) -> pd.DataFrame:
            """同步处理价格数据"""
            try:
                # 数据清洗
                df = df.copy()
                df = df.dropna()

                # 价格调整
                if "adj_close" not in df.columns and "close" in df.columns:
                    df["adj_close"] = df["close"]

                # 计算收益率
                if len(df) > 1:
                    df["returns"] = df["close"].pct_change()
                    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

                # 价格验证
                for col in ["open", "high", "low", "close"]:
                    if col in df.columns:
                        df.loc[df[col] <= 0, col] = np.nan

                # 逻辑验证
                if all(col in df.columns for col in ["high", "low", "open", "close"]):
                    # 修复不合理的价格关系
                    df.loc[df["high"] < df["low"], "high"] = df["low"]
                    df.loc[df["high"] < df["open"], "high"] = df["open"]
                    df.loc[df["high"] < df["close"], "high"] = df["close"]
                    df.loc[df["low"] > df["open"], "low"] = df["open"]
                    df.loc[df["low"] > df["close"], "low"] = df["close"]

                return df

            except Exception as e:
                raise ProcessingException(f"价格数据处理失败: {e}")

        task = ProcessingTask(
            task_id=f"price_data_{int(time.time() * 1000)}",
            data=data,
            processor_func=_process_price_data_sync,
            priority=ProcessingPriority.NORMAL,
        )

        task_id = await self.submit_task(task)
        return await self.get_result(task_id)

    async def calculate_technical_indicators(
        self, data: pd.DataFrame, indicators: List[str]
    ) -> pd.DataFrame:
        """
        计算技术指标

        Args:
            data: 价格数据
            indicators: 指标列表

        Returns:
            DataFrame: 包含技术指标的数据
        """

        def _calculate_indicators_sync(
            df: pd.DataFrame, indicators_list: List[str]
        ) -> pd.DataFrame:
            """同步计算技术指标"""
            try:
                result_df = df.copy()

                if "close" not in df.columns:
                    raise ValueError("数据缺少close列")

                close_prices = df["close"]

                for indicator in indicators_list:
                    if indicator == "sma_20":
                        result_df["sma_20"] = close_prices.rolling(window=20).mean()
                    elif indicator == "sma_50":
                        result_df["sma_50"] = close_prices.rolling(window=50).mean()
                    elif indicator == "ema_12":
                        result_df["ema_12"] = close_prices.ewm(span=12).mean()
                    elif indicator == "ema_26":
                        result_df["ema_26"] = close_prices.ewm(span=26).mean()
                    elif indicator == "rsi_14":
                        delta = close_prices.diff()
                        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                        rs = gain / loss
                        result_df["rsi_14"] = 100 - (100 / (1 + rs))
                    elif indicator == "macd":
                        ema_12 = close_prices.ewm(span=12).mean()
                        ema_26 = close_prices.ewm(span=26).mean()
                        result_df["macd"] = ema_12 - ema_26
                        result_df["macd_signal"] = result_df["macd"].ewm(span=9).mean()
                        result_df["macd_histogram"] = (
                            result_df["macd"] - result_df["macd_signal"]
                        )
                    elif indicator == "bollinger_bands":
                        sma_20 = close_prices.rolling(window=20).mean()
                        std_20 = close_prices.rolling(window=20).std()
                        result_df["bb_upper"] = sma_20 + (std_20 * 2)
                        result_df["bb_lower"] = sma_20 - (std_20 * 2)
                        result_df["bb_middle"] = sma_20
                    elif indicator == "atr_14":
                        if all(col in df.columns for col in ["high", "low", "close"]):
                            high = df["high"]
                            low = df["low"]
                            close = df["close"]

                            tr1 = high - low
                            tr2 = abs(high - close.shift(1))
                            tr3 = abs(low - close.shift(1))

                            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                            result_df["atr_14"] = true_range.rolling(window=14).mean()

                return result_df

            except Exception as e:
                raise ProcessingException(f"技术指标计算失败: {e}")

        task = ProcessingTask(
            task_id=f"indicators_{int(time.time() * 1000)}",
            data=(data, indicators),
            processor_func=lambda x: _calculate_indicators_sync(x[0], x[1]),
            priority=ProcessingPriority.HIGH,
            metadata={"use_process_pool": len(data) > 1000},  # 大数据集使用进程池
        )

        task_id = await self.submit_task(task)
        return await self.get_result(task_id)

    async def validate_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        验证数据质量

        Args:
            data: 待验证的数据

        Returns:
            Dict: 数据质量报告
        """

        def _validate_data_sync(df: pd.DataFrame) -> Dict[str, Any]:
            """同步验证数据质量"""
            try:
                report = {
                    "is_valid": True,
                    "total_rows": len(df),
                    "issues": [],
                    "quality_score": 100.0,
                    "missing_data": {},
                    "anomalies": [],
                }

                if df.empty:
                    report["is_valid"] = False
                    report["issues"].append("数据为空")
                    report["quality_score"] = 0.0
                    return report

                # 检查缺失值
                missing_data = df.isnull().sum()
                for col, missing_count in missing_data.items():
                    if missing_count > 0:
                        missing_ratio = missing_count / len(df)
                        report["missing_data"][col] = {
                            "count": int(missing_count),
                            "ratio": round(missing_ratio, 4),
                        }

                        if missing_ratio > 0.1:  # 超过10%缺失
                            report["issues"].append(
                                f"列 {col} 缺失值过多: {missing_ratio:.2%}"
                            )
                            report["quality_score"] -= 20

                # 检查价格数据逻辑
                if all(col in df.columns for col in ["high", "low", "open", "close"]):
                    # 检查高低价关系
                    invalid_high_low = (df["high"] < df["low"]).sum()
                    if invalid_high_low > 0:
                        report["issues"].append(
                            f"发现 {invalid_high_low} 行高价小于低价"
                        )
                        report["quality_score"] -= 15

                    # 检查开盘价、收盘价是否在高低价范围内
                    invalid_open = (
                        (df["open"] > df["high"]) | (df["open"] < df["low"])
                    ).sum()
                    invalid_close = (
                        (df["close"] > df["high"]) | (df["close"] < df["low"])
                    ).sum()

                    if invalid_open > 0:
                        report["issues"].append(
                            f"发现 {invalid_open} 行开盘价超出高低价范围"
                        )
                        report["quality_score"] -= 10

                    if invalid_close > 0:
                        report["issues"].append(
                            f"发现 {invalid_close} 行收盘价超出高低价范围"
                        )
                        report["quality_score"] -= 10

                # 检查负价格
                for price_col in ["open", "high", "low", "close"]:
                    if price_col in df.columns:
                        negative_prices = (df[price_col] <= 0).sum()
                        if negative_prices > 0:
                            report["issues"].append(
                                f"发现 {negative_prices} 行负价格或零价格"
                            )
                            report["quality_score"] -= 25

                # 检查成交量
                if "volume" in df.columns:
                    zero_volume = (df["volume"] <= 0).sum()
                    if zero_volume > 0:
                        report["issues"].append(f"发现 {zero_volume} 行零成交量")
                        report["quality_score"] -= 5

                # 检查重复数据
                if "datetime" in df.columns or "date" in df.columns:
                    date_col = "datetime" if "datetime" in df.columns else "date"
                    duplicates = df[date_col].duplicated().sum()
                    if duplicates > 0:
                        report["issues"].append(f"发现 {duplicates} 行重复日期")
                        report["quality_score"] -= 15

                # 最终判断
                report["quality_score"] = max(0.0, report["quality_score"])
                if report["quality_score"] < 70:
                    report["is_valid"] = False

                return report

            except Exception as e:
                return {"is_valid": False, "error": str(e), "quality_score": 0.0}

        task = ProcessingTask(
            task_id=f"validation_{int(time.time() * 1000)}",
            data=data,
            processor_func=_validate_data_sync,
            priority=ProcessingPriority.HIGH,
        )

        task_id = await self.submit_task(task)
        return await self.get_result(task_id)

    async def normalize_data(
        self, data: pd.DataFrame, method: str = "standard"
    ) -> pd.DataFrame:
        """
        数据标准化

        Args:
            data: 原始数据
            method: 标准化方法 ('standard', 'minmax', 'robust')

        Returns:
            DataFrame: 标准化后的数据
        """

        def _normalize_data_sync(df: pd.DataFrame, norm_method: str) -> pd.DataFrame:
            """同步数据标准化"""
            try:
                result_df = df.copy()
                numeric_columns = df.select_dtypes(include=[np.number]).columns

                for col in numeric_columns:
                    if col in ["volume"]:  # 跳过某些列
                        continue

                    if norm_method == "standard":
                        # Z-score标准化
                        mean_val = df[col].mean()
                        std_val = df[col].std()
                        if std_val != 0:
                            result_df[f"{col}_normalized"] = (
                                df[col] - mean_val
                            ) / std_val

                    elif norm_method == "minmax":
                        # Min-Max标准化
                        min_val = df[col].min()
                        max_val = df[col].max()
                        if max_val != min_val:
                            result_df[f"{col}_normalized"] = (df[col] - min_val) / (
                                max_val - min_val
                            )

                    elif norm_method == "robust":
                        # 稳健标准化
                        median_val = df[col].median()
                        mad_val = abs(df[col] - median_val).median()
                        if mad_val != 0:
                            result_df[f"{col}_normalized"] = (
                                df[col] - median_val
                            ) / mad_val

                return result_df

            except Exception as e:
                raise ProcessingException(f"数据标准化失败: {e}")

        task = ProcessingTask(
            task_id=f"normalize_{int(time.time() * 1000)}",
            data=(data, method),
            processor_func=lambda x: _normalize_data_sync(x[0], x[1]),
            priority=ProcessingPriority.NORMAL,
        )

        task_id = await self.submit_task(task)
        return await self.get_result(task_id)

    async def process_batch(
        self, data_list: List[pd.DataFrame], processor_func: Callable
    ) -> List[Any]:
        """
        批量处理数据

        Args:
            data_list: 数据列表
            processor_func: 处理函数

        Returns:
            List: 处理结果列表
        """
        try:
            tasks = []

            for i, data in enumerate(data_list):
                task = ProcessingTask(
                    task_id=f"batch_{int(time.time() * 1000)}_{i}",
                    data=data,
                    processor_func=processor_func,
                    priority=ProcessingPriority.NORMAL,
                )
                task_id = await self.submit_task(task)
                tasks.append(task_id)

            # 等待所有任务完成
            results = []
            for task_id in tasks:
                try:
                    result = await self.get_result(task_id)
                    results.append(result)
                except Exception as e:
                    self.logger.warning(f"批量处理任务 {task_id} 失败: {e}")
                    results.append(None)

            return results

        except Exception as e:
            self.exception_logger.log_exception(
                e, {"operation": "process_batch", "batch_size": len(data_list)}
            )
            raise ProcessingException(f"批量处理失败: {e}", cause=e)

    def get_stats(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        stats = self._stats.copy()
        stats["queue_size"] = self._priority_queue.qsize()
        stats["cache_size"] = len(self._result_cache)
        stats["is_running"] = self._running
        stats["worker_count"] = len(self._workers)
        return stats

    def clear_cache(self):
        """清理结果缓存"""
        self._result_cache.clear()
        self.logger.info("处理结果缓存已清理")

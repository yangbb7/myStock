# -*- coding: utf-8 -*-
"""
异步数据接口 - 定义异步数据操作的标准接口
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import pandas as pd


class IAsyncDataProvider(ABC):
    """异步数据提供者接口"""

    @abstractmethod
    async def get_price_data(
        self, symbol: str, start_date: str = None, end_date: str = None
    ) -> pd.DataFrame:
        """
        异步获取价格数据

        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            DataFrame: 价格数据
        """
        pass

    @abstractmethod
    async def get_price_data_batch(
        self, symbols: List[str], start_date: str = None, end_date: str = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        批量异步获取价格数据

        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期

        Yields:
            Dict: 包含股票代码和价格数据的字典
        """
        pass

    @abstractmethod
    async def get_real_time_quote(self, symbol: str) -> Dict[str, Any]:
        """
        获取实时报价

        Args:
            symbol: 股票代码

        Returns:
            Dict: 实时报价数据
        """
        pass

    @abstractmethod
    async def get_real_time_quotes_batch(
        self, symbols: List[str]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        批量获取实时报价

        Args:
            symbols: 股票代码列表

        Yields:
            Dict: 实时报价数据
        """
        pass

    @abstractmethod
    async def get_fundamental_data(
        self, symbol: str, report_date: str = None
    ) -> Dict[str, Any]:
        """
        获取基本面数据

        Args:
            symbol: 股票代码
            report_date: 报告日期

        Returns:
            Dict: 基本面数据
        """
        pass

    @abstractmethod
    async def get_technical_indicators(
        self, symbol: str, indicators: List[str], period: int = 20
    ) -> Dict[str, Any]:
        """
        获取技术指标

        Args:
            symbol: 股票代码
            indicators: 指标列表
            period: 计算周期

        Returns:
            Dict: 技术指标数据
        """
        pass


class IAsyncDataProcessor(ABC):
    """异步数据处理器接口"""

    @abstractmethod
    async def process_price_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        处理价格数据

        Args:
            data: 原始价格数据

        Returns:
            DataFrame: 处理后的价格数据
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    async def validate_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        验证数据质量

        Args:
            data: 待验证的数据

        Returns:
            Dict: 数据质量报告
        """
        pass

    @abstractmethod
    async def normalize_data(
        self, data: pd.DataFrame, method: str = "standard"
    ) -> pd.DataFrame:
        """
        数据标准化

        Args:
            data: 原始数据
            method: 标准化方法

        Returns:
            DataFrame: 标准化后的数据
        """
        pass


class IAsyncDataCache(ABC):
    """异步数据缓存接口"""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """
        获取缓存数据

        Args:
            key: 缓存键

        Returns:
            Any: 缓存的数据，如果不存在返回None
        """
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """
        设置缓存数据

        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间（秒）

        Returns:
            bool: 是否设置成功
        """
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """
        删除缓存数据

        Args:
            key: 缓存键

        Returns:
            bool: 是否删除成功
        """
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """
        检查缓存是否存在

        Args:
            key: 缓存键

        Returns:
            bool: 是否存在
        """
        pass

    @abstractmethod
    async def clear(self) -> bool:
        """
        清理所有缓存

        Returns:
            bool: 是否清理成功
        """
        pass


class IAsyncDataAggregator(ABC):
    """异步数据聚合器接口"""

    @abstractmethod
    async def aggregate_market_data(
        self,
        symbols: List[str],
        aggregation_func: str = "mean",
        time_window: str = "1D",
    ) -> pd.DataFrame:
        """
        聚合市场数据

        Args:
            symbols: 股票代码列表
            aggregation_func: 聚合函数
            time_window: 时间窗口

        Returns:
            DataFrame: 聚合后的数据
        """
        pass

    @abstractmethod
    async def calculate_portfolio_metrics(
        self, weights: Dict[str, float], data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        计算投资组合指标

        Args:
            weights: 权重字典
            data: 价格数据字典

        Returns:
            Dict: 投资组合指标
        """
        pass

    @abstractmethod
    async def generate_market_summary(
        self, symbols: List[str], date: str = None
    ) -> Dict[str, Any]:
        """
        生成市场总结

        Args:
            symbols: 股票代码列表
            date: 日期

        Returns:
            Dict: 市场总结
        """
        pass


class IAsyncDataNotifier(ABC):
    """异步数据通知器接口"""

    @abstractmethod
    async def subscribe_price_updates(
        self, symbols: List[str]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        订阅价格更新

        Args:
            symbols: 股票代码列表

        Yields:
            Dict: 价格更新数据
        """
        pass

    @abstractmethod
    async def subscribe_market_events(
        self, event_types: List[str]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        订阅市场事件

        Args:
            event_types: 事件类型列表

        Yields:
            Dict: 市场事件数据
        """
        pass

    @abstractmethod
    async def notify_data_update(self, symbol: str, data_type: str, data: Any) -> bool:
        """
        通知数据更新

        Args:
            symbol: 股票代码
            data_type: 数据类型
            data: 数据内容

        Returns:
            bool: 是否通知成功
        """
        pass


class IAsyncDataValidator(ABC):
    """异步数据验证器接口"""

    @abstractmethod
    async def validate_price_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        验证价格数据

        Args:
            data: 价格数据

        Returns:
            Dict: 验证结果
        """
        pass

    @abstractmethod
    async def validate_real_time_data(self, data: Dict[str, Any]) -> bool:
        """
        验证实时数据

        Args:
            data: 实时数据

        Returns:
            bool: 是否有效
        """
        pass

    @abstractmethod
    async def check_data_completeness(
        self, symbol: str, start_date: str, end_date: str
    ) -> Dict[str, Any]:
        """
        检查数据完整性

        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            Dict: 完整性检查结果
        """
        pass

    @abstractmethod
    async def detect_data_anomalies(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        检测数据异常

        Args:
            data: 待检测的数据

        Returns:
            List: 异常列表
        """
        pass


class AsyncDataContext:
    """异步数据上下文管理器"""

    def __init__(
        self,
        provider: IAsyncDataProvider,
        processor: IAsyncDataProcessor = None,
        cache: IAsyncDataCache = None,
        validator: IAsyncDataValidator = None,
    ):
        self.provider = provider
        self.processor = processor
        self.cache = cache
        self.validator = validator

    async def __aenter__(self):
        """进入异步上下文"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """退出异步上下文"""
        if hasattr(self.provider, "close"):
            await self.provider.close()
        if self.cache and hasattr(self.cache, "close"):
            await self.cache.close()

    async def get_validated_data(
        self, symbol: str, start_date: str = None, end_date: str = None
    ) -> pd.DataFrame:
        """
        获取验证后的数据

        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            DataFrame: 验证后的数据
        """
        # 尝试从缓存获取
        if self.cache:
            cache_key = f"{symbol}_{start_date}_{end_date}"
            cached_data = await self.cache.get(cache_key)
            if cached_data:
                return cached_data

        # 从提供者获取数据
        data = await self.provider.get_price_data(symbol, start_date, end_date)

        # 验证数据
        if self.validator:
            validation_result = await self.validator.validate_price_data(data)
            if not validation_result.get("is_valid", True):
                raise ValueError(f"数据验证失败: {validation_result.get('errors', [])}")

        # 处理数据
        if self.processor:
            data = await self.processor.process_price_data(data)

        # 缓存数据
        if self.cache:
            await self.cache.set(cache_key, data, ttl=300)

        return data

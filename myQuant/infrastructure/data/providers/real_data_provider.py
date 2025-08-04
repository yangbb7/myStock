# -*- coding: utf-8 -*-
"""
真实数据提供者 - 集成多种真实数据源
"""

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests


class DataProviderInterface(ABC):
    """数据提供者接口"""

    @abstractmethod
    def get_stock_data(
        self, symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """获取股票历史数据"""
        pass

    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        """获取当前价格"""
        pass

    @abstractmethod
    def get_realtime_data(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """获取实时数据"""
        pass


class TushareProvider(DataProviderInterface):
    """Tushare数据提供者"""

    def __init__(self, token: str):
        self.token = token
        self.base_url = "http://api.tushare.pro"
        self.logger = logging.getLogger(__name__)

        # 尝试导入tushare
        try:
            import tushare as ts

            ts.set_token(token)
            self.ts = ts
            self.pro = ts.pro_api()
            self.logger.info("Tushare初始化成功")
        except ImportError:
            self.logger.warning("Tushare未安装，将使用API接口")
            self.ts = None
            self.pro = None

    def _convert_symbol(self, symbol: str) -> str:
        """转换股票代码格式"""
        if "." in symbol:
            code, market = symbol.split(".")
            if market == "SZ":
                return f"{code}.SZ"
            elif market == "SH":
                return f"{code}.SH"
        return symbol

    def get_stock_data(
        self, symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """获取股票历史数据"""
        try:
            ts_symbol = self._convert_symbol(symbol)

            if self.pro:
                # 使用tushare pro接口
                df = self.pro.daily(
                    ts_code=ts_symbol,
                    start_date=start_date.replace("-", ""),
                    end_date=end_date.replace("-", ""),
                )
                if df is not None and not df.empty:
                    df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
                    df = df.rename(
                        columns={
                            "ts_code": "symbol",
                            "open": "open",
                            "high": "high",
                            "low": "low",
                            "close": "close",
                            "vol": "volume",
                        }
                    )
                    df = df[
                        ["date", "symbol", "open", "high", "low", "close", "volume"]
                    ]
                    df = df.sort_values("date").reset_index(drop=True)
                    return df
            else:
                # 使用HTTP API
                return self._get_data_via_api(ts_symbol, start_date, end_date)

        except Exception as e:
            self.logger.error(f"获取Tushare数据失败: {symbol}, {e}")
            return pd.DataFrame()

    def _get_data_via_api(
        self, symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """通过API获取数据"""
        try:
            params = {
                "api_name": "daily",
                "token": self.token,
                "params": {
                    "ts_code": symbol,
                    "start_date": start_date.replace("-", ""),
                    "end_date": end_date.replace("-", ""),
                },
            }

            response = requests.post(self.base_url, json=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                if data["code"] == 0:
                    df = pd.DataFrame(
                        data["data"]["items"], columns=data["data"]["fields"]
                    )
                    df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
                    df = df.rename(columns={"ts_code": "symbol", "vol": "volume"})
                    return df[
                        ["date", "symbol", "open", "high", "low", "close", "volume"]
                    ]

        except Exception as e:
            self.logger.error(f"API获取数据失败: {e}")

        return pd.DataFrame()

    def get_current_price(self, symbol: str) -> float:
        """获取当前价格"""
        try:
            if self.ts:
                # 获取实时数据
                df = self.ts.get_realtime_quotes(symbol)
                if not df.empty:
                    return float(df.iloc[0]["price"])

            # 备用方案：使用最新交易日数据
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
            df = self.get_stock_data(symbol, start_date, end_date)
            if not df.empty:
                return float(df.iloc[-1]["close"])

        except Exception as e:
            self.logger.error(f"获取当前价格失败: {symbol}, {e}")

        return 0.0

    def get_realtime_data(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """获取实时数据"""
        result = {}
        for symbol in symbols:
            try:
                price = self.get_current_price(symbol)
                if price > 0:
                    result[symbol] = {
                        "current_price": price,
                        "timestamp": datetime.now(),
                        "symbol": symbol,
                    }
            except Exception as e:
                self.logger.error(f"获取实时数据失败: {symbol}, {e}")

        return result


class YahooFinanceProvider(DataProviderInterface):
    """Yahoo Finance数据提供者"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        try:
            import yfinance as yf

            self.yf = yf
            self.logger.info("Yahoo Finance初始化成功")
        except ImportError:
            self.logger.warning("yfinance未安装")
            self.yf = None

    def _convert_symbol(self, symbol: str) -> str:
        """转换股票代码格式为Yahoo Finance格式"""
        if "." in symbol:
            code, market = symbol.split(".")
            if market == "SZ":
                return f"{code}.SZ"
            elif market == "SH":
                return f"{code}.SS"
        return symbol

    def get_stock_data(
        self, symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """获取股票历史数据"""
        if not self.yf:
            return pd.DataFrame()

        try:
            yf_symbol = self._convert_symbol(symbol)
            ticker = self.yf.Ticker(yf_symbol)
            df = ticker.history(start=start_date, end=end_date)

            if not df.empty:
                df = df.reset_index()
                df = df.rename(
                    columns={
                        "Date": "date",
                        "Open": "open",
                        "High": "high",
                        "Low": "low",
                        "Close": "close",
                        "Volume": "volume",
                    }
                )
                df["symbol"] = symbol
                return df[["date", "symbol", "open", "high", "low", "close", "volume"]]

        except Exception as e:
            self.logger.error(f"获取Yahoo Finance数据失败: {symbol}, {e}")

        return pd.DataFrame()

    def get_current_price(self, symbol: str) -> float:
        """获取当前价格"""
        if not self.yf:
            return 0.0

        try:
            yf_symbol = self._convert_symbol(symbol)
            ticker = self.yf.Ticker(yf_symbol)
            info = ticker.info
            return float(info.get("regularMarketPrice", 0))

        except Exception as e:
            self.logger.error(f"获取Yahoo Finance当前价格失败: {symbol}, {e}")

        return 0.0

    def get_realtime_data(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """获取实时数据"""
        result = {}
        for symbol in symbols:
            price = self.get_current_price(symbol)
            if price > 0:
                result[symbol] = {
                    "current_price": price,
                    "timestamp": datetime.now(),
                    "symbol": symbol,
                }
        return result


class EastMoneyProvider(DataProviderInterface):
    """东方财富数据提供者"""

    def __init__(self):
        self.base_url = "http://push2his.eastmoney.com/api/qt/stock/kline/get"
        self.realtime_url = "http://push2.eastmoney.com/api/qt/slist/get"
        self.logger = logging.getLogger(__name__)

    def _convert_symbol(self, symbol: str) -> str:
        """转换股票代码格式"""
        if "." in symbol:
            code, market = symbol.split(".")
            if market == "SZ":
                return f"0.{code}"
            elif market == "SH":
                return f"1.{code}"
        return symbol

    def get_stock_data(
        self, symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """获取股票历史数据"""
        try:
            em_symbol = self._convert_symbol(symbol)

            params = {
                "secid": em_symbol,
                "fields1": "f1,f2,f3,f4,f5,f6",
                "fields2": "f51,f52,f53,f54,f55,f56,f57,f58",
                "klt": "101",  # 日K线
                "fqt": "1",  # 前复权
                "beg": start_date.replace("-", ""),
                "end": end_date.replace("-", ""),
                "ut": "fa5fd1943c7b386f172d6893dbfba10b",
            }

            response = requests.get(self.base_url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                if data.get("data") and data["data"].get("klines"):
                    lines = []
                    for line in data["data"]["klines"]:
                        parts = line.split(",")
                        lines.append(
                            {
                                "date": pd.to_datetime(parts[0]),
                                "symbol": symbol,
                                "open": float(parts[1]),
                                "close": float(parts[2]),
                                "high": float(parts[3]),
                                "low": float(parts[4]),
                                "volume": int(parts[5]),
                            }
                        )

                    return pd.DataFrame(lines)

        except Exception as e:
            self.logger.error(f"获取东方财富数据失败: {symbol}, {e}")

        return pd.DataFrame()

    def get_current_price(self, symbol: str) -> float:
        """获取当前价格"""
        try:
            em_symbol = self._convert_symbol(symbol)

            # 使用更完整的实时数据API
            params = {
                "secids": em_symbol,
                "fields": "f43,f44,f45,f46,f47,f48,f57,f58",  # 更多字段
                "ut": "fa5fd1943c7b386f172d6893dbfba10b",
                "fltt": "2",
                "invt": "2"
            }

            response = requests.get(self.realtime_url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("data") and data["data"].get("diff") and len(data["data"]["diff"]) > 0:
                    stock_data = data["data"]["diff"][0]
                    # f43是当前价格字段
                    current_price = stock_data.get("f43", 0)
                    if current_price and current_price != "-":
                        return float(current_price)

            # 如果实时API失败，使用历史数据的最新价格作为备用
            from datetime import datetime, timedelta
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
            
            hist_data = self.get_stock_data(symbol, start_date, end_date)
            if not hist_data.empty:
                return float(hist_data.iloc[-1]["close"])

        except Exception as e:
            self.logger.error(f"获取东方财富当前价格失败: {symbol}, {e}")

        return 0.0

    def get_realtime_data(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """获取实时数据"""
        result = {}
        for symbol in symbols:
            price = self.get_current_price(symbol)
            if price > 0:
                result[symbol] = {
                    "current_price": price,
                    "timestamp": datetime.now(),
                    "symbol": symbol,
                }
        return result


class RealDataProvider:
    """真实数据提供者管理器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.providers = {}

        # 初始化故障转移管理器
        from ..fallback_manager import FallbackManager
        fallback_config = {
            'fallback_strategy': config.get('fallback_strategy', 'graceful_degradation'),
            'max_retries': config.get('max_retries', 3),
            'quality_threshold': config.get('quality_threshold', 0.8),
            'quality_config': config.get('quality_config', {})
        }
        self.fallback_manager = FallbackManager(fallback_config)

        # 初始化数据提供者
        self._init_providers()

        # 设置主要和备用提供者
        self.primary_provider = config.get("primary_provider", "tushare")
        self.fallback_providers = config.get(
            "fallback_providers", ["yahoo", "eastmoney"]
        )

        # 注册数据源到故障转移管理器
        self._register_data_sources()

    def _init_providers(self):
        """初始化数据提供者"""
        # 券商API集成（优先级最高）
        if "broker_apis" in self.config and self.config["broker_apis"]:
            try:
                from .broker_api_providers import BrokerAPIManager
                self.broker_api_manager = BrokerAPIManager(self.config["broker_apis"])
                self.logger.info("券商API管理器初始化成功")
            except Exception as e:
                self.logger.error(f"券商API管理器初始化失败: {e}")
                self.broker_api_manager = None
        else:
            self.broker_api_manager = None

        # Tushare
        if "tushare" in self.config and self.config["tushare"].get("token"):
            try:
                self.providers["tushare"] = TushareProvider(
                    self.config["tushare"]["token"]
                )
                self.logger.info("Tushare提供者初始化成功")
            except Exception as e:
                self.logger.error(f"Tushare提供者初始化失败: {e}")

        # Yahoo Finance
        if self.config.get("yahoo", {}).get("enabled", True):
            try:
                self.providers["yahoo"] = YahooFinanceProvider()
                self.logger.info("Yahoo Finance提供者初始化成功")
            except Exception as e:
                self.logger.error(f"Yahoo Finance提供者初始化失败: {e}")

        # 东方财富
        if self.config.get("eastmoney", {}).get("enabled", True):
            try:
                self.providers["eastmoney"] = EastMoneyProvider()
                self.logger.info("东方财富提供者初始化成功")
            except Exception as e:
                self.logger.error(f"东方财富提供者初始化失败: {e}")

    def _register_data_sources(self):
        """注册数据源到故障转移管理器"""
        # 注册券商API（优先级最高）
        if self.broker_api_manager:
            self.fallback_manager.register_data_source('broker_api', priority=1)
        
        # 注册传统数据源
        for source_name in self.providers.keys():
            if source_name == 'tushare':
                priority = 2
            elif source_name == 'yahoo':
                priority = 3
            elif source_name == 'eastmoney':
                priority = 4
            else:
                priority = 10
            
            self.fallback_manager.register_data_source(source_name, priority)
        
        self.logger.info(f"已注册{len(self.providers) + (1 if self.broker_api_manager else 0)}个数据源")

    def get_stock_data(
        self, symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """获取股票历史数据（使用故障转移机制）"""
        # 构建数据获取函数字典
        data_fetchers = {}
        
        # 添加券商API获取函数
        if self.broker_api_manager:
            def broker_fetch(symbol, start_date, end_date):
                # 券商API通常不提供历史数据，这里可以扩展
                # 暂时返回空DataFrame
                return pd.DataFrame()
            data_fetchers['broker_api'] = broker_fetch
        
        # 添加传统数据源获取函数
        for provider_name, provider in self.providers.items():
            data_fetchers[provider_name] = provider.get_stock_data
        
        try:
            # 使用故障转移管理器执行数据获取
            data, successful_source, quality_metrics = self.fallback_manager.execute_with_fallback(
                data_fetchers, symbol, start_date, end_date
            )
            
            self.logger.info(
                f"从{successful_source}成功获取{symbol}的数据，"
                f"质量评分: {quality_metrics.overall_score:.2f}"
            )
            
            return data
            
        except Exception as e:
            self.logger.error(f"所有数据源都无法获取{symbol}的数据: {e}")
            # 不再生成Mock数据，直接抛出异常
            raise DataSourceException(
                f"无法从任何数据源获取{symbol}的真实数据",
                symbol=symbol,
                data_source="All",
                cause=e
            )

    def get_current_price(self, symbol: str) -> float:
        """获取当前价格（带备用机制）"""
        providers_to_try = [self.primary_provider] + self.fallback_providers

        for provider_name in providers_to_try:
            if provider_name in self.providers:
                try:
                    price = self.providers[provider_name].get_current_price(symbol)
                    if price > 0:
                        return price
                except Exception as e:
                    self.logger.warning(f"{provider_name}获取当前价格失败: {e}")
                    continue

        self.logger.error(f"所有数据源都无法获取{symbol}的当前价格")
        return 0.0

    def get_realtime_data(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """获取实时数据（优先使用券商API）"""
        result = {}

        # 优先使用券商API获取实时数据
        if self.broker_api_manager:
            try:
                broker_data = self.broker_api_manager.get_realtime_data(symbols)
                if broker_data and len(broker_data) > 0:
                    self.logger.info(f"从券商API获取到{len(broker_data)}个股票的实时数据")
                    result.update(broker_data)
                    # 如果券商API获取了所有数据，直接返回
                    if len(result) == len(symbols):
                        return result
            except Exception as e:
                self.logger.warning(f"券商API获取实时数据失败: {e}")

        # 如果券商API未能获取所有数据，使用传统数据源补充
        missing_symbols = [s for s in symbols if s not in result]
        if not missing_symbols:
            return result

        # 尝试从主要提供者获取缺失数据
        if self.primary_provider in self.providers:
            try:
                provider_data = self.providers[self.primary_provider].get_realtime_data(
                    missing_symbols
                )
                result.update(provider_data)
                missing_symbols = [s for s in missing_symbols if s not in provider_data]
            except Exception as e:
                self.logger.warning(f"{self.primary_provider}获取实时数据失败: {e}")

        # 如果仍有缺失数据，逐个股票尝试备用提供者
        for symbol in missing_symbols:
            price = self.get_current_price(symbol)
            if price > 0:
                result[symbol] = {
                    "current_price": price,
                    "timestamp": datetime.now(),
                    "symbol": symbol,
                }

        return result

    def get_account_positions(self, broker_name: Optional[str] = None) -> Dict[str, Any]:
        """获取券商账户持仓数据"""
        if self.broker_api_manager:
            try:
                return self.broker_api_manager.get_account_positions(broker_name)
            except Exception as e:
                self.logger.error(f"获取券商持仓数据失败: {e}")
                return {}
        else:
            self.logger.warning("券商API管理器未初始化，无法获取持仓数据")
            return {}

    def get_order_status(self, order_id: str, broker_name: Optional[str] = None) -> Dict[str, Any]:
        """获取券商订单状态"""
        if self.broker_api_manager:
            try:
                return self.broker_api_manager.get_order_status(order_id, broker_name)
            except Exception as e:
                self.logger.error(f"获取券商订单状态失败: {e}")
                return {}
        else:
            self.logger.warning("券商API管理器未初始化，无法获取订单状态")
            return {}

    def test_connection(self) -> Dict[str, bool]:
        """测试所有数据源连接"""
        results = {}
        test_symbol = "000001.SZ"  # 使用平安银行测试
        test_start = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        test_end = datetime.now().strftime("%Y-%m-%d")

        for name, provider in self.providers.items():
            try:
                # 测试历史数据获取
                df = provider.get_stock_data(test_symbol, test_start, test_end)
                # 测试当前价格获取
                price = provider.get_current_price(test_symbol)

                results[name] = not df.empty and price > 0

            except Exception as e:
                self.logger.error(f"测试{name}连接失败: {e}")
                results[name] = False

        return results

    def get_health_status(self) -> Dict[str, Any]:
        """获取数据提供者健康状态
        
        Returns:
            Dict[str, Any]: 健康状态报告
        """
        return self.fallback_manager.get_health_report()

    def get_quality_report(self) -> Dict[str, Any]:
        """获取数据质量报告
        
        Returns:
            Dict[str, Any]: 数据质量报告
        """
        return self.fallback_manager.quality_monitor.get_quality_report()

    def get_recommended_sources(self, count: int = 3) -> List[str]:
        """获取推荐的数据源
        
        Args:
            count: 推荐数量
            
        Returns:
            List[str]: 推荐的数据源列表
        """
        return self.fallback_manager.get_recommended_sources(count)

    def reset_source_status(self, source_name: str):
        """重置数据源状态
        
        Args:
            source_name: 数据源名称
        """
        self.fallback_manager.reset_circuit_breaker(source_name)
        self.logger.info(f"数据源{source_name}状态已重置")

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

            params = {
                "secids": em_symbol,
                "fields": "f43",  # 当前价格
                "ut": "fa5fd1943c7b386f172d6893dbfba10b",
            }

            response = requests.get(self.realtime_url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("data") and data["data"].get("diff"):
                    return float(data["data"]["diff"][0]["f43"])

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

        # 初始化数据提供者
        self._init_providers()

        # 设置主要和备用提供者
        self.primary_provider = config.get("primary_provider", "tushare")
        self.fallback_providers = config.get(
            "fallback_providers", ["yahoo", "eastmoney"]
        )

    def _init_providers(self):
        """初始化数据提供者"""
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

    def get_stock_data(
        self, symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """获取股票历史数据（带备用机制）"""
        providers_to_try = [self.primary_provider] + self.fallback_providers

        for provider_name in providers_to_try:
            if provider_name in self.providers:
                try:
                    df = self.providers[provider_name].get_stock_data(
                        symbol, start_date, end_date
                    )
                    if not df.empty:
                        self.logger.info(f"从{provider_name}成功获取{symbol}的数据")
                        return df
                except Exception as e:
                    self.logger.warning(f"{provider_name}获取数据失败: {e}")
                    continue

        self.logger.error(f"所有数据源都无法获取{symbol}的数据")
        return pd.DataFrame()

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
        """获取实时数据"""
        result = {}

        # 尝试从主要提供者获取所有数据
        if self.primary_provider in self.providers:
            try:
                result = self.providers[self.primary_provider].get_realtime_data(
                    symbols
                )
                if len(result) == len(symbols):
                    return result
            except Exception as e:
                self.logger.warning(f"{self.primary_provider}获取实时数据失败: {e}")

        # 如果主要提供者失败，逐个股票尝试备用提供者
        missing_symbols = [s for s in symbols if s not in result]

        for symbol in missing_symbols:
            price = self.get_current_price(symbol)
            if price > 0:
                result[symbol] = {
                    "current_price": price,
                    "timestamp": datetime.now(),
                    "symbol": symbol,
                }

        return result

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

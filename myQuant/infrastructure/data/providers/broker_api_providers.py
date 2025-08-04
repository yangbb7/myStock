# -*- coding: utf-8 -*-
"""
券商API数据提供者 - 集成主流券商API接口
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import requests


class BrokerAPIInterface(ABC):
    """券商API接口基类"""

    @abstractmethod
    def get_realtime_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """获取实时行情数据"""
        pass

    @abstractmethod
    def get_historical_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取历史数据"""
        pass

    @abstractmethod
    def get_account_positions(self) -> Dict[str, Any]:
        """获取账户持仓"""
        pass

    @abstractmethod
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """获取订单状态"""
        pass


class HuataiSecuritiesAPI(BrokerAPIInterface):
    """华泰证券API接口"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.base_url = config.get('base_url', 'https://api.htsc.com.cn')
        self.api_key = config.get('api_key')
        self.secret_key = config.get('secret_key')
        self.account_id = config.get('account_id')
        
        if not all([self.api_key, self.secret_key, self.account_id]):
            self.logger.warning("华泰证券API配置不完整，部分功能可能不可用")

    def get_realtime_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """获取实时行情数据"""
        try:
            # 华泰证券API实时行情接口
            url = f"{self.base_url}/api/v1/market/realtime"
            params = {
                'symbols': ','.join(symbols),
                'fields': 'last_price,volume,bid,ask,bid_size,ask_size,change,change_percent'
            }
            headers = {
                'Authorization': f'Bearer {self._get_access_token()}',
                'Content-Type': 'application/json'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return self._format_realtime_data(data)
            else:
                self.logger.error(f"华泰证券API请求失败: {response.status_code}")
                return {}
                
        except Exception as e:
            self.logger.error(f"华泰证券实时数据获取失败: {e}")
            return {}

    def get_historical_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取历史数据"""
        try:
            url = f"{self.base_url}/api/v1/market/history"
            params = {
                'symbol': symbol,
                'start_date': start_date,
                'end_date': end_date,
                'frequency': 'daily'
            }
            headers = {
                'Authorization': f'Bearer {self._get_access_token()}',
                'Content-Type': 'application/json'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=30)
            if response.status_code == 200:
                data = response.json()
                return self._format_historical_data(data)
            else:
                self.logger.error(f"华泰证券历史数据请求失败: {response.status_code}")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"华泰证券历史数据获取失败: {e}")
            return pd.DataFrame()

    def get_account_positions(self) -> Dict[str, Any]:
        """获取账户持仓"""
        try:
            url = f"{self.base_url}/api/v1/account/{self.account_id}/positions"
            headers = {
                'Authorization': f'Bearer {self._get_access_token()}',
                'Content-Type': 'application/json'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"华泰证券持仓数据请求失败: {response.status_code}")
                return {}
                
        except Exception as e:
            self.logger.error(f"华泰证券持仓数据获取失败: {e}")
            return {}

    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """获取订单状态"""
        try:
            url = f"{self.base_url}/api/v1/orders/{order_id}"
            headers = {
                'Authorization': f'Bearer {self._get_access_token()}',
                'Content-Type': 'application/json'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"华泰证券订单状态请求失败: {response.status_code}")
                return {}
                
        except Exception as e:
            self.logger.error(f"华泰证券订单状态获取失败: {e}")
            return {}

    def _get_access_token(self) -> str:
        """获取访问令牌"""
        # 这里应该实现华泰证券的OAuth认证流程
        # 为了演示，返回配置中的api_key
        return self.api_key

    def _format_realtime_data(self, raw_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """格式化实时数据"""
        formatted_data = {}
        for item in raw_data.get('data', []):
            symbol = item.get('symbol')
            formatted_data[symbol] = {
                'current_price': float(item.get('last_price', 0)),
                'volume': int(item.get('volume', 0)),
                'bid': float(item.get('bid', 0)),
                'ask': float(item.get('ask', 0)),
                'bid_size': int(item.get('bid_size', 0)),
                'ask_size': int(item.get('ask_size', 0)),
                'change': float(item.get('change', 0)),
                'change_percent': float(item.get('change_percent', 0)),
                'timestamp': datetime.now(),
                'symbol': symbol
            }
        return formatted_data

    def _format_historical_data(self, raw_data: Dict[str, Any]) -> pd.DataFrame:
        """格式化历史数据"""
        data_list = []
        for item in raw_data.get('data', []):
            data_list.append({
                'date': pd.to_datetime(item.get('date')),
                'symbol': item.get('symbol'),
                'open': float(item.get('open', 0)),
                'high': float(item.get('high', 0)),
                'low': float(item.get('low', 0)),
                'close': float(item.get('close', 0)),
                'volume': int(item.get('volume', 0)),
                'adj_close': float(item.get('adj_close', item.get('close', 0)))
            })
        return pd.DataFrame(data_list)


class CiticSecuritiesAPI(BrokerAPIInterface):
    """中信证券API接口"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.base_url = config.get('base_url', 'https://api.citics.com')
        self.api_key = config.get('api_key')
        self.secret_key = config.get('secret_key')
        self.account_id = config.get('account_id')

    def get_realtime_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """获取实时行情数据"""
        try:
            # 中信证券API实现
            url = f"{self.base_url}/market/quotes"
            data = {
                'symbols': symbols,
                'api_key': self.api_key
            }
            
            response = requests.post(url, json=data, timeout=10)
            if response.status_code == 200:
                return self._format_realtime_data(response.json())
            else:
                self.logger.error(f"中信证券API请求失败: {response.status_code}")
                return {}
                
        except Exception as e:
            self.logger.error(f"中信证券实时数据获取失败: {e}")
            return {}

    def get_historical_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取历史数据"""
        try:
            url = f"{self.base_url}/market/history"
            data = {
                'symbol': symbol,
                'start_date': start_date,
                'end_date': end_date,
                'api_key': self.api_key
            }
            
            response = requests.post(url, json=data, timeout=30)
            if response.status_code == 200:
                return self._format_historical_data(response.json())
            else:
                self.logger.error(f"中信证券历史数据请求失败: {response.status_code}")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"中信证券历史数据获取失败: {e}")
            return pd.DataFrame()

    def get_account_positions(self) -> Dict[str, Any]:
        """获取账户持仓"""
        try:
            url = f"{self.base_url}/account/positions"
            data = {
                'account_id': self.account_id,
                'api_key': self.api_key
            }
            
            response = requests.post(url, json=data, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"中信证券持仓数据请求失败: {response.status_code}")
                return {}
                
        except Exception as e:
            self.logger.error(f"中信证券持仓数据获取失败: {e}")
            return {}

    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """获取订单状态"""
        try:
            url = f"{self.base_url}/orders/status"
            data = {
                'order_id': order_id,
                'api_key': self.api_key
            }
            
            response = requests.post(url, json=data, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"中信证券订单状态请求失败: {response.status_code}")
                return {}
                
        except Exception as e:
            self.logger.error(f"中信证券订单状态获取失败: {e}")
            return {}

    def _format_realtime_data(self, raw_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """格式化实时数据"""
        formatted_data = {}
        for item in raw_data.get('quotes', []):
            symbol = item.get('code')
            formatted_data[symbol] = {
                'current_price': float(item.get('price', 0)),
                'volume': int(item.get('volume', 0)),
                'bid': float(item.get('bid1', 0)),
                'ask': float(item.get('ask1', 0)),
                'bid_size': int(item.get('bid1_size', 0)),
                'ask_size': int(item.get('ask1_size', 0)),
                'timestamp': datetime.now(),
                'symbol': symbol
            }
        return formatted_data

    def _format_historical_data(self, raw_data: Dict[str, Any]) -> pd.DataFrame:
        """格式化历史数据"""
        data_list = []
        for item in raw_data.get('klines', []):
            data_list.append({
                'date': pd.to_datetime(item.get('date')),
                'symbol': item.get('code'),
                'open': float(item.get('open', 0)),
                'high': float(item.get('high', 0)),
                'low': float(item.get('low', 0)),
                'close': float(item.get('close', 0)),
                'volume': int(item.get('volume', 0)),
                'adj_close': float(item.get('close', 0))
            })
        return pd.DataFrame(data_list)


class GTJASecuritiesAPI(BrokerAPIInterface):
    """国泰君安证券API接口"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.base_url = config.get('base_url', 'https://api.gtja.com')
        self.api_key = config.get('api_key')
        self.secret_key = config.get('secret_key')
        self.account_id = config.get('account_id')

    def get_realtime_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """获取实时行情数据"""
        try:
            # 国泰君安API实现
            url = f"{self.base_url}/v1/quotes/realtime"
            headers = {
                'X-API-KEY': self.api_key,
                'Content-Type': 'application/json'
            }
            data = {'symbols': symbols}
            
            response = requests.post(url, json=data, headers=headers, timeout=10)
            if response.status_code == 200:
                return self._format_realtime_data(response.json())
            else:
                self.logger.error(f"国泰君安API请求失败: {response.status_code}")
                return {}
                
        except Exception as e:
            self.logger.error(f"国泰君安实时数据获取失败: {e}")
            return {}

    def get_historical_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取历史数据"""
        try:
            url = f"{self.base_url}/v1/quotes/history"
            headers = {
                'X-API-KEY': self.api_key,
                'Content-Type': 'application/json'
            }
            data = {
                'symbol': symbol,
                'start_date': start_date,
                'end_date': end_date,
                'period': 'day'
            }
            
            response = requests.post(url, json=data, headers=headers, timeout=30)
            if response.status_code == 200:
                return self._format_historical_data(response.json())
            else:
                self.logger.error(f"国泰君安历史数据请求失败: {response.status_code}")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"国泰君安历史数据获取失败: {e}")
            return pd.DataFrame()

    def get_account_positions(self) -> Dict[str, Any]:
        """获取账户持仓"""
        try:
            url = f"{self.base_url}/v1/account/positions"
            headers = {
                'X-API-KEY': self.api_key,
                'X-ACCOUNT-ID': self.account_id,
                'Content-Type': 'application/json'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"国泰君安持仓数据请求失败: {response.status_code}")
                return {}
                
        except Exception as e:
            self.logger.error(f"国泰君安持仓数据获取失败: {e}")
            return {}

    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """获取订单状态"""
        try:
            url = f"{self.base_url}/v1/orders/{order_id}/status"
            headers = {
                'X-API-KEY': self.api_key,
                'Content-Type': 'application/json'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"国泰君安订单状态请求失败: {response.status_code}")
                return {}
                
        except Exception as e:
            self.logger.error(f"国泰君安订单状态获取失败: {e}")
            return {}

    def _format_realtime_data(self, raw_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """格式化实时数据"""
        formatted_data = {}
        for item in raw_data.get('data', []):
            symbol = item.get('symbol')
            formatted_data[symbol] = {
                'current_price': float(item.get('last_price', 0)),
                'volume': int(item.get('volume', 0)),
                'bid': float(item.get('bid_price', 0)),
                'ask': float(item.get('ask_price', 0)),
                'bid_size': int(item.get('bid_volume', 0)),
                'ask_size': int(item.get('ask_volume', 0)),
                'timestamp': datetime.now(),
                'symbol': symbol
            }
        return formatted_data

    def _format_historical_data(self, raw_data: Dict[str, Any]) -> pd.DataFrame:
        """格式化历史数据"""
        data_list = []
        for item in raw_data.get('data', []):
            data_list.append({
                'date': pd.to_datetime(item.get('trade_date')),
                'symbol': item.get('symbol'),
                'open': float(item.get('open_price', 0)),
                'high': float(item.get('high_price', 0)),
                'low': float(item.get('low_price', 0)),
                'close': float(item.get('close_price', 0)),
                'volume': int(item.get('volume', 0)),
                'adj_close': float(item.get('close_price', 0))
            })
        return pd.DataFrame(data_list)


class BrokerAPIManager:
    """券商API管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.broker_apis = {}
        
        # 初始化券商API
        self._init_broker_apis()
        
        # 设置主要和备用券商
        self.primary_broker = config.get('primary_broker', 'huatai')
        self.fallback_brokers = config.get('fallback_brokers', ['citic', 'gtja'])

    def _init_broker_apis(self):
        """初始化券商API"""
        # 华泰证券
        if 'huatai' in self.config and self.config['huatai'].get('enabled', False):
            try:
                self.broker_apis['huatai'] = HuataiSecuritiesAPI(self.config['huatai'])
                self.logger.info("华泰证券API初始化成功")
            except Exception as e:
                self.logger.error(f"华泰证券API初始化失败: {e}")

        # 中信证券
        if 'citic' in self.config and self.config['citic'].get('enabled', False):
            try:
                self.broker_apis['citic'] = CiticSecuritiesAPI(self.config['citic'])
                self.logger.info("中信证券API初始化成功")
            except Exception as e:
                self.logger.error(f"中信证券API初始化失败: {e}")

        # 国泰君安
        if 'gtja' in self.config and self.config['gtja'].get('enabled', False):
            try:
                self.broker_apis['gtja'] = GTJASecuritiesAPI(self.config['gtja'])
                self.logger.info("国泰君安API初始化成功")
            except Exception as e:
                self.logger.error(f"国泰君安API初始化失败: {e}")

    def get_realtime_data(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """获取实时数据（带备用机制）"""
        brokers_to_try = [self.primary_broker] + self.fallback_brokers
        
        for broker_name in brokers_to_try:
            if broker_name in self.broker_apis:
                try:
                    data = self.broker_apis[broker_name].get_realtime_quotes(symbols)
                    if data:
                        self.logger.debug(f"从{broker_name}成功获取实时数据")
                        return data
                except Exception as e:
                    self.logger.warning(f"{broker_name}获取实时数据失败: {e}")
                    continue
        
        self.logger.error("所有券商API都无法获取实时数据")
        return {}

    def get_account_positions(self, broker_name: Optional[str] = None) -> Dict[str, Any]:
        """获取账户持仓"""
        broker_name = broker_name or self.primary_broker
        
        if broker_name in self.broker_apis:
            try:
                return self.broker_apis[broker_name].get_account_positions()
            except Exception as e:
                self.logger.error(f"{broker_name}获取持仓数据失败: {e}")
        
        return {}

    def get_order_status(self, order_id: str, broker_name: Optional[str] = None) -> Dict[str, Any]:
        """获取订单状态"""
        broker_name = broker_name or self.primary_broker
        
        if broker_name in self.broker_apis:
            try:
                return self.broker_apis[broker_name].get_order_status(order_id)
            except Exception as e:
                self.logger.error(f"{broker_name}获取订单状态失败: {e}")
        
        return {}

    def test_connections(self) -> Dict[str, bool]:
        """测试所有券商API连接"""
        results = {}
        test_symbols = ['000001.SZ']  # 使用平安银行测试
        
        for broker_name, api in self.broker_apis.items():
            try:
                # 测试实时数据获取
                data = api.get_realtime_quotes(test_symbols)
                results[broker_name] = bool(data)
            except Exception as e:
                self.logger.error(f"测试{broker_name}连接失败: {e}")
                results[broker_name] = False
        
        return results
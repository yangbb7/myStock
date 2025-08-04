"""
技术指标计算API

提供RESTful API接口用于计算各种技术指标
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json

from myQuant.core.analysis.technical_indicators import TechnicalIndicatorCalculator
from myQuant.core.analysis.indicator_factory import IndicatorFactory
from myQuant.core.models.market_data import KlineData
from myQuant.infrastructure.database.database_manager import DatabaseManager


class TechnicalIndicatorsAPI:
    """技术指标API"""
    
    def __init__(self, database_manager: DatabaseManager):
        self.db = database_manager
        self.calculator = TechnicalIndicatorCalculator()
        self.factory = IndicatorFactory()
        self.logger = logging.getLogger(__name__)
        self._cache = {}  # 简单内存缓存
    
    async def calculate_indicator(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """计算单个指标"""
        try:
            # 验证请求
            is_valid, errors = await self.validate_request(request_data)
            if not is_valid:
                return {
                    'success': False,
                    'error': f'请求验证失败: {", ".join(errors)}'
                }
            
            symbol = request_data['symbol']
            indicator_type = request_data['indicator']
            parameters = request_data.get('parameters', {})
            
            # 获取数据
            if 'data' in request_data:
                # 使用提供的数据
                kline_data = self._convert_to_kline_data(request_data['data'])
            else:
                # 从数据库获取数据
                kline_data = await self._fetch_kline_data_from_db(request_data)
            
            if not kline_data:
                return {
                    'success': False,
                    'error': '无法获取市场数据'
                }
            
            # 检查数据是否足够
            min_data_required = self._get_min_data_required(indicator_type, parameters)
            if len(kline_data) < min_data_required:
                return {
                    'success': False,
                    'error': f'数据不足，需要至少 {min_data_required} 个数据点，当前只有 {len(kline_data)} 个'
                }
            
            # 计算指标
            result = await self._calculate_single_indicator(kline_data, indicator_type, parameters)
            
            return {
                'success': True,
                'data': {
                    'symbol': symbol,
                    'indicator': indicator_type,
                    'parameters': parameters,
                    'values': result,
                    'data_points': len(kline_data),
                    'calculated_at': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"计算指标失败: {e}")
            return {
                'success': False,
                'error': f'计算指标时发生错误: {str(e)}'
            }
    
    async def calculate_multiple_indicators(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """计算多个指标"""
        try:
            symbol = request_data['symbol']
            indicators_config = request_data['indicators']
            
            # 获取数据
            if 'data' in request_data:
                kline_data = self._convert_to_kline_data(request_data['data'])
            else:
                kline_data = await self._fetch_kline_data_from_db(request_data)
            
            if not kline_data:
                return {
                    'success': False,
                    'error': '无法获取市场数据'
                }
            
            # 批量计算指标
            results = await self.calculator.batch_calculate(kline_data, indicators_config)
            
            return {
                'success': True,
                'data': {
                    'symbol': symbol,
                    'indicators': results,
                    'data_points': len(kline_data),
                    'calculated_at': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"批量计算指标失败: {e}")
            return {
                'success': False,
                'error': f'批量计算指标时发生错误: {str(e)}'
            }
    
    async def get_indicator_from_database(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """从数据库获取历史数据并计算指标"""
        try:
            symbol = request_data['symbol']
            indicator_type = request_data['indicator']
            parameters = request_data.get('parameters', {})
            start_date = request_data.get('start_date')
            end_date = request_data.get('end_date')
            
            # 构建查询
            sql = """
            SELECT symbol, trade_date, open_price, high_price, low_price, close_price, volume, turnover
            FROM kline_daily 
            WHERE symbol = ?
            """
            params = [symbol]
            
            if start_date:
                sql += " AND trade_date >= ?"
                params.append(start_date)
            
            if end_date:
                sql += " AND trade_date <= ?"
                params.append(end_date)
            
            sql += " ORDER BY trade_date"
            
            # 查询数据
            raw_data = await self.db.fetch_all(sql, tuple(params))
            
            if not raw_data:
                return {
                    'success': False,
                    'error': f'未找到股票 {symbol} 的历史数据'
                }
            
            # 转换为KlineData格式
            kline_data = []
            for row in raw_data:
                kline_data.append(KlineData(
                    symbol=row['symbol'],
                    timestamp=datetime.strptime(row['trade_date'], '%Y-%m-%d'),
                    open=float(row['open_price']),
                    high=float(row['high_price']),
                    low=float(row['low_price']),
                    close=float(row['close_price']),
                    volume=int(row['volume']),
                    turnover=float(row['turnover'])
                ))
            
            # 计算指标
            result = await self._calculate_single_indicator(kline_data, indicator_type, parameters)
            
            return {
                'success': True,
                'data': {
                    'symbol': symbol,
                    'indicator': indicator_type,
                    'parameters': parameters,
                    'values': result,
                    'data_points': len(kline_data),
                    'date_range': {
                        'start': raw_data[0]['trade_date'],
                        'end': raw_data[-1]['trade_date']
                    },
                    'calculated_at': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"从数据库计算指标失败: {e}")
            return {
                'success': False,
                'error': f'从数据库计算指标时发生错误: {str(e)}'
            }
    
    async def validate_request(self, request_data: Dict[str, Any]) -> tuple[bool, List[str]]:
        """验证请求数据"""
        errors = []
        
        # 检查必需字段
        if not request_data.get('symbol'):
            errors.append('缺少股票代码 (symbol)')
        
        if not request_data.get('indicator'):
            errors.append('缺少指标类型 (indicator)')
        
        # 验证股票代码格式
        symbol = request_data.get('symbol', '')
        if symbol and not self._is_valid_symbol(symbol):
            errors.append('无效的股票代码格式')
        
        # 验证指标类型
        indicator = request_data.get('indicator', '')
        if indicator and indicator not in self.factory.list_available_indicators():
            errors.append(f'不支持的指标类型: {indicator}')
        
        # 验证参数
        if indicator and 'parameters' in request_data:
            validation_result = self.factory.validate_parameters(indicator, request_data['parameters'])
            if not validation_result['is_valid']:
                errors.extend(validation_result['errors'])
        
        return len(errors) == 0, errors
    
    async def cache_indicator_result(self, cache_key: str, data: Dict[str, Any], ttl: int = 300):
        """缓存指标结果"""
        self._cache[cache_key] = {
            'data': data,
            'timestamp': datetime.now(),
            'ttl': ttl
        }
    
    async def get_cached_indicator_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """获取缓存的指标结果"""
        if cache_key in self._cache:
            cached_item = self._cache[cache_key]
            
            # 检查是否过期
            if datetime.now() - cached_item['timestamp'] <= timedelta(seconds=cached_item['ttl']):
                return cached_item['data']
            else:
                # 删除过期缓存
                del self._cache[cache_key]
        
        return None
    
    def _convert_to_kline_data(self, raw_data: List[Dict[str, Any]]) -> List[KlineData]:
        """将原始数据转换为KlineData格式"""
        kline_data = []
        
        for item in raw_data:
            try:
                kline_data.append(KlineData(
                    symbol=item.get('symbol', ''),
                    timestamp=datetime.fromisoformat(item['timestamp'].replace('Z', '+00:00')) if isinstance(item['timestamp'], str) else item['timestamp'],
                    open=float(item['open']),
                    high=float(item['high']),
                    low=float(item['low']),
                    close=float(item['close']),
                    volume=int(item['volume']),
                    turnover=float(item.get('turnover', 0))
                ))
            except (KeyError, ValueError, TypeError) as e:
                self.logger.warning(f"跳过无效数据项: {e}")
                continue
        
        return kline_data
    
    async def _fetch_kline_data_from_db(self, request_data: Dict[str, Any]) -> List[KlineData]:
        """从数据库获取K线数据"""
        # 这里应该根据请求参数构建查询
        # 暂时返回空列表，实际实现需要根据具体需求
        return []
    
    async def _calculate_single_indicator(self, kline_data: List[KlineData], 
                                        indicator_type: str, parameters: Dict[str, Any]) -> Any:
        """计算单个指标"""
        prices = [data.close for data in kline_data]
        
        if indicator_type == 'sma':
            return await self.calculator.calculate_sma(prices, parameters.get('period', 20))
        elif indicator_type == 'ema':
            return await self.calculator.calculate_ema(prices, parameters.get('period', 12))
        elif indicator_type == 'rsi':
            return await self.calculator.calculate_rsi(prices, parameters.get('period', 14))
        elif indicator_type == 'macd':
            return await self.calculator.calculate_macd(
                prices, 
                parameters.get('fast_period', 12),
                parameters.get('slow_period', 26),
                parameters.get('signal_period', 9)
            )
        elif indicator_type == 'bollinger_bands':
            return await self.calculator.calculate_bollinger_bands(
                prices, 
                parameters.get('period', 20),
                parameters.get('std_dev', 2.0)
            )
        elif indicator_type == 'stochastic':
            return await self.calculator.calculate_stochastic(
                kline_data,
                parameters.get('period', 14),
                parameters.get('k_smoothing', 3),
                parameters.get('d_smoothing', 3)
            )
        elif indicator_type == 'atr':
            return await self.calculator.calculate_atr(kline_data, parameters.get('period', 14))
        else:
            raise ValueError(f"不支持的指标类型: {indicator_type}")
    
    def _get_min_data_required(self, indicator_type: str, parameters: Dict[str, Any]) -> int:
        """获取计算指标所需的最小数据量"""
        if indicator_type == 'sma':
            return parameters.get('period', 20)
        elif indicator_type == 'ema':
            return parameters.get('period', 12)
        elif indicator_type == 'rsi':
            return parameters.get('period', 14) + 1
        elif indicator_type == 'macd':
            return max(parameters.get('slow_period', 26), parameters.get('fast_period', 12))
        elif indicator_type == 'bollinger_bands':
            return parameters.get('period', 20)
        elif indicator_type == 'stochastic':
            return parameters.get('period', 14)
        elif indicator_type == 'atr':
            return parameters.get('period', 14) + 1
        else:
            return 50  # 默认最小数据量
    
    def _is_valid_symbol(self, symbol: str) -> bool:
        """验证股票代码格式"""
        # 简单的A股代码验证
        import re
        pattern = r'^[0-9]{6}\.(SZ|SH)$'
        return bool(re.match(pattern, symbol))
    
    async def get_available_indicators(self) -> Dict[str, Any]:
        """获取可用指标列表"""
        indicators = {}
        
        for indicator_type in self.factory.list_available_indicators():
            indicators[indicator_type] = {
                'name': indicator_type,
                'description': self.factory.get_indicator_description(indicator_type),
                'parameters': self.factory.get_indicator_parameters(indicator_type)
            }
        
        return {
            'success': True,
            'data': {
                'indicators': indicators,
                'total_count': len(indicators)
            }
        }
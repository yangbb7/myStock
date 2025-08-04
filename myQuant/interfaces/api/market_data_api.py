# -*- coding: utf-8 -*-
"""
市场数据API

提供实时行情、历史数据、技术指标等功能
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, date
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
import asyncio
import time
from decimal import Decimal

from myQuant.core.managers.data_manager import DataManager
from myQuant.core.models.market_data import RealTimeQuote, KlineData


class StockQuote(BaseModel):
    """股票报价"""
    symbol: str
    name: str
    current_price: float
    change: float
    change_percent: float
    open_price: float
    high_price: float
    low_price: float
    volume: int
    turnover: float
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    timestamp: datetime


class HistoricalPrice(BaseModel):
    """历史价格数据"""
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int
    adj_close: Optional[float] = None


class TechnicalIndicator(BaseModel):
    """技术指标"""
    name: str
    value: float
    date: date
    parameters: Optional[Dict[str, Any]] = None


class MarketDataAPI:
    """市场数据API"""
    
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
        self.router = APIRouter(prefix="/market-data", tags=["market-data"])
        self._cache = {}
        self._rate_limiter = {"requests_per_second": 100, "last_request_time": 0, "request_count": 0}
        self._setup_routes()
    
    async def get_kline_data(self, symbol: str, period: str, start_date: Optional[str] = None, 
                           end_date: Optional[str] = None, limit: Optional[int] = None) -> Dict:
        """获取K线数据"""
        # Validation
        if not symbol or symbol.strip() == "":
            return {
                'success': False,
                'error': 'Symbol cannot be empty'
            }
        
        if period not in ["1min", "5min", "15min", "30min", "1h", "1d", "1w", "1M"]:
            return {
                'success': False,
                'error': f'Invalid period: {period}'
            }
        
        # Date range validation
        if start_date and end_date:
            if start_date > end_date:
                return {
                    'success': False,
                    'error': 'Start date must be before end date'
                }
            
        # Apply rate limiting
        await self._apply_rate_limit()
        
        query = """
        SELECT symbol, timestamp, open, high, low, close, volume
        FROM kline_data 
        WHERE symbol = ? AND period = ?
        """
        params = [symbol, period]
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
            
        query += " ORDER BY timestamp DESC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
            
        records = await self.data_manager.fetch_all(query, params)
        
        return {
            'success': True,
            'data': {
                'symbol': symbol,
                'period': period,
                'records': records
            }
        }
    
    async def get_realtime_quote(self, symbol: str) -> Dict:
        """获取实时报价"""
        try:
            await self._apply_rate_limit()
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
        
        # Try cache first
        cache_key = f"realtime_{symbol}"
        cached_data = self._cache.get(cache_key)
        if cached_data and time.time() - cached_data['timestamp'] < 1:
            return {
                'success': True,
                'data': cached_data['data']
            }
        
        # Get from data provider
        if hasattr(self.data_manager, 'data_provider'):
            quote = await self.data_manager.data_provider.get_realtime_quote(symbol)
            if quote:
                # Convert RealTimeQuote to dict
                quote_dict = quote.dict() if hasattr(quote, 'dict') else quote
                
                # Add bid_ask info for the test
                if isinstance(quote_dict, dict):
                    quote_dict['bid_ask'] = {
                        'bid_price_1': quote_dict.get('bid_price_1'),
                        'bid_volume_1': quote_dict.get('bid_volume_1'),
                        'ask_price_1': quote_dict.get('ask_price_1'),
                        'ask_volume_1': quote_dict.get('ask_volume_1')
                    }
                
                # Cache the result
                self._cache[cache_key] = {
                    'data': quote_dict,
                    'timestamp': time.time()
                }
                return {
                    'success': True,
                    'data': quote_dict
                }
        
        return {
            'success': False,
            'error': 'Quote not found'
        }
    
    async def get_batch_realtime_quotes(self, symbols: List[str]) -> Dict:
        """获取多个股票的实时报价"""
        await self._apply_rate_limit()
        
        if hasattr(self.data_manager, 'data_provider') and hasattr(self.data_manager.data_provider, 'get_batch_realtime_quotes'):
            quotes = await self.data_manager.data_provider.get_batch_realtime_quotes(symbols)
            # Convert RealTimeQuote objects to dicts
            quote_dicts = []
            for quote in quotes:
                if hasattr(quote, 'dict'):
                    quote_dicts.append(quote.dict())
                elif hasattr(quote, 'model_dump'):
                    quote_dicts.append(quote.model_dump())
                else:
                    quote_dicts.append(quote)
            
            return {
                'success': True,
                'data': {
                    'quotes': quote_dicts
                }
            }
        else:
            # Fallback to individual calls
            quotes = []
            for symbol in symbols:
                quote_result = await self.get_realtime_quote(symbol)
                if quote_result.get('success') and quote_result.get('data'):
                    quotes.append(quote_result['data'])
            return {
                'success': True,
                'data': {
                    'quotes': quotes
                }
            }
    
    async def get_market_depth(self, symbol: str, level: int = 5) -> Dict:
        """获取市场深度数据"""
        await self._apply_rate_limit()
        
        if hasattr(self.data_manager, 'data_provider'):
            depth_data = await self.data_manager.data_provider.get_market_depth(symbol, level)
            return {
                'success': True,
                'data': depth_data
            }
        return {
            'success': False,
            'error': 'Market depth data not available'
        }
    
    async def get_trade_ticks(self, symbol: str, limit: int = 100) -> Dict:
        """获取逐笔交易数据"""
        await self._apply_rate_limit()
        
        if hasattr(self.data_manager, 'data_provider'):
            ticks_data = await self.data_manager.data_provider.get_trade_ticks(symbol, limit)
            return {
                'success': True,
                'data': {
                    'ticks': ticks_data
                }
            }
        return {
            'success': False,
            'error': 'Trade ticks data not available'
        }
    
    async def search_stocks(self, keyword: str, limit: int = 20) -> Dict:
        """搜索股票"""
        if not keyword or keyword.strip() == "":
            return {
                'success': True,
                'data': {
                    'stocks': []
                }
            }
            
        query = """
        SELECT symbol, name, sector 
        FROM stocks 
        WHERE name LIKE ? OR symbol LIKE ?
        LIMIT ?
        """
        search_term = f"%{keyword}%"
        results = await self.data_manager.fetch_all(query, [search_term, search_term, limit])
        return {
            'success': True,
            'data': {
                'stocks': results
            }
        }
    
    async def get_stock_info(self, symbol: str) -> Dict:
        """获取股票基本信息"""
        query = """
        SELECT symbol, name, sector, industry, market, listing_date, 
               total_shares, float_shares
        FROM stocks 
        WHERE symbol = ?
        """
        result = await self.data_manager.fetch_one(query, [symbol])
        if result:
            return {
                'success': True,
                'data': result
            }
        return {
            'success': False,
            'error': 'Stock not found'
        }
    
    async def cache_market_data(self, cache_key: str, data: Dict, ttl: int = 60):
        """缓存市场数据"""
        self._cache[cache_key] = {
            'data': data,
            'timestamp': time.time(),
            'ttl': ttl
        }
    
    async def get_cached_market_data(self, cache_key: str) -> Optional[Dict]:
        """获取缓存的市场数据"""
        cached_data = self._cache.get(cache_key)
        if cached_data:
            if time.time() - cached_data['timestamp'] < cached_data.get('ttl', 60):
                return cached_data['data']
            else:
                # Remove expired cache
                del self._cache[cache_key]
        return None
    
    def set_rate_limit(self, requests_per_second: int):
        """设置请求限流"""
        self._rate_limiter['requests_per_second'] = requests_per_second
    
    async def _apply_rate_limit(self):
        """应用请求限流"""
        current_time = time.time()
        time_since_last = current_time - self._rate_limiter['last_request_time']
        
        if time_since_last >= 1.0:
            # Reset counter every second
            self._rate_limiter['request_count'] = 0
            self._rate_limiter['last_request_time'] = current_time
        
        if self._rate_limiter['request_count'] >= self._rate_limiter['requests_per_second']:
            # Sleep to maintain rate limit
            sleep_time = 1.0 - time_since_last
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
                self._rate_limiter['request_count'] = 0
                self._rate_limiter['last_request_time'] = time.time()
        
        self._rate_limiter['request_count'] += 1
    
    def _setup_routes(self):
        """设置路由"""
        
        @self.router.get("/quote/{symbol}", response_model=StockQuote)
        async def get_stock_quote(symbol: str):
            """获取股票实时报价"""
            try:
                quote_data = await self.data_manager.get_realtime_quote(symbol)
                if not quote_data:
                    raise HTTPException(status_code=404, detail="Stock not found")
                
                return StockQuote(**quote_data)
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.router.get("/quotes", response_model=List[StockQuote])
        async def get_multiple_quotes(symbols: List[str] = Query(...)):
            """获取多只股票实时报价"""
            try:
                quotes = []
                for symbol in symbols:
                    quote_data = await self.data_manager.get_realtime_quote(symbol)
                    if quote_data:
                        quotes.append(StockQuote(**quote_data))
                
                return quotes
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.router.get("/history/{symbol}", response_model=List[HistoricalPrice])
        async def get_historical_data(
            symbol: str,
            start_date: date = Query(...),
            end_date: date = Query(...),
            period: str = Query(default="daily", pattern="^(daily|weekly|monthly)$")
        ):
            """获取历史价格数据"""
            try:
                if start_date > end_date:
                    raise HTTPException(status_code=400, detail="Start date must be before end date")
                
                data = await self.data_manager.get_historical_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    period=period
                )
                
                return [
                    HistoricalPrice(
                        date=row['date'],
                        open=row['open'],
                        high=row['high'],
                        low=row['low'],
                        close=row['close'],
                        volume=row['volume'],
                        adj_close=row.get('adj_close')
                    )
                    for row in data
                ]
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.router.get("/indicators/{symbol}", response_model=List[TechnicalIndicator])
        async def get_technical_indicators(
            symbol: str,
            indicators: List[str] = Query(...),
            start_date: Optional[date] = Query(None),
            end_date: Optional[date] = Query(None)
        ):
            """获取技术指标"""
            try:
                result = []
                
                for indicator_name in indicators:
                    indicator_data = await self.data_manager.calculate_technical_indicator(
                        symbol=symbol,
                        indicator=indicator_name,
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    if indicator_data:
                        for row in indicator_data:
                            result.append(TechnicalIndicator(
                                name=indicator_name,
                                value=row['value'],
                                date=row['date'],
                                parameters=row.get('parameters')
                            ))
                
                return result
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.router.get("/search", response_model=List[Dict[str, str]])
        async def search_stocks(
            keyword: str = Query(..., min_length=1),
            limit: int = Query(default=20, le=100)
        ):
            """搜索股票"""
            try:
                results = await self.data_manager.search_stocks(keyword, limit)
                return results
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.router.get("/market-summary", response_model=Dict[str, Any])
        async def get_market_summary():
            """获取市场概况"""
            try:
                summary = await self.data_manager.get_market_summary()
                return summary
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.router.get("/sector-performance", response_model=List[Dict[str, Any]])
        async def get_sector_performance():
            """获取行业表现"""
            try:
                performance = await self.data_manager.get_sector_performance()
                return performance
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.router.get("/hot-stocks", response_model=List[StockQuote])
        async def get_hot_stocks(
            category: str = Query(default="volume", pattern="^(volume|gainers|losers|active)$"),
            limit: int = Query(default=20, le=100)
        ):
            """获取热门股票"""
            try:
                hot_stocks = await self.data_manager.get_hot_stocks(category, limit)
                return [StockQuote(**stock) for stock in hot_stocks]
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
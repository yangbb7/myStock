"""
股票筛选API

提供股票筛选、过滤、排序、导出等功能
支持基本面、技术面、市场数据等多维度筛选
"""

import csv
import io
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from enum import Enum

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field, validator
import pandas as pd
import numpy as np

from myQuant.infrastructure.database.database_manager import DatabaseManager
from myQuant.infrastructure.database.repositories import StockRepository, KlineRepository
from myQuant.infrastructure.container import get_container
from myQuant.core.analysis.technical_indicators import TechnicalIndicators


router = APIRouter(prefix="/api/v1/screening", tags=["screening"])


class SortField(str, Enum):
    """排序字段"""
    PRICE = "price"
    VOLUME = "volume"
    CHANGE_PERCENT = "change_percent"
    MARKET_CAP = "market_cap"
    PE_RATIO = "pe_ratio"
    PB_RATIO = "pb_ratio"
    ROE = "roe"
    TURNOVER = "turnover"


class SortOrder(str, Enum):
    """排序顺序"""
    ASC = "asc"
    DESC = "desc"


class PriceRange(BaseModel):
    """价格区间"""
    min: Optional[float] = Field(None, ge=0)
    max: Optional[float] = Field(None, ge=0)
    
    @validator('max')
    def validate_max(cls, v, values):
        if v is not None and 'min' in values and values['min'] is not None:
            if v < values['min']:
                raise ValueError('max must be greater than min')
        return v


class VolumeRange(BaseModel):
    """成交量区间"""
    min: Optional[int] = Field(None, ge=0)
    max: Optional[int] = Field(None, ge=0)
    
    @validator('max')
    def validate_max(cls, v, values):
        if v is not None and 'min' in values and values['min'] is not None:
            if v < values['min']:
                raise ValueError('max must be greater than min')
        return v


class MarketCapRange(BaseModel):
    """市值区间"""
    min: Optional[float] = Field(None, ge=0)
    max: Optional[float] = Field(None, ge=0)
    
    @validator('max')
    def validate_max(cls, v, values):
        if v is not None and 'min' in values and values['min'] is not None:
            if v < values['min']:
                raise ValueError('max must be greater than min')
        return v


class RSIRange(BaseModel):
    """RSI区间"""
    min: Optional[int] = Field(None, ge=0, le=100)
    max: Optional[int] = Field(None, ge=0, le=100)
    
    @validator('max')
    def validate_max(cls, v, values):
        if v is not None and 'min' in values and values['min'] is not None:
            if v < values['min']:
                raise ValueError('max must be greater than min')
        return v


class MACondition(BaseModel):
    """均线条件"""
    ma5_above_ma20: Optional[bool] = None
    ma20_above_ma60: Optional[bool] = None
    price_above_ma5: Optional[bool] = None
    price_above_ma20: Optional[bool] = None


class VolumeCondition(BaseModel):
    """成交量条件"""
    above_average: Optional[bool] = None
    multiple: Optional[float] = Field(None, ge=0)  # 成交量是平均值的倍数


class TechnicalCriteria(BaseModel):
    """技术面筛选条件"""
    rsi_range: Optional[RSIRange] = None
    ma_condition: Optional[MACondition] = None
    volume_condition: Optional[VolumeCondition] = None
    macd_signal: Optional[str] = Field(None, pattern="^(golden_cross|death_cross|bullish|bearish)$")
    kdj_range: Optional[Dict[str, RSIRange]] = None  # K, D, J的范围


class PERRange(BaseModel):
    """市盈率区间"""
    min: Optional[float] = Field(None, ge=0)
    max: Optional[float] = Field(None, ge=0)
    
    @validator('max')
    def validate_max(cls, v, values):
        if v is not None and 'min' in values and values['min'] is not None:
            if v < values['min']:
                raise ValueError('max must be greater than min')
        return v


class PBRange(BaseModel):
    """市净率区间"""
    min: Optional[float] = Field(None, ge=0)
    max: Optional[float] = Field(None, ge=0)
    
    @validator('max')
    def validate_max(cls, v, values):
        if v is not None and 'min' in values and values['min'] is not None:
            if v < values['min']:
                raise ValueError('max must be greater than min')
        return v


class FundamentalCriteria(BaseModel):
    """基本面筛选条件"""
    pe_range: Optional[PERRange] = None
    pb_range: Optional[PBRange] = None
    roe_min: Optional[float] = Field(None, ge=0, le=100)
    revenue_growth_min: Optional[float] = Field(None, ge=-100, le=1000)
    profit_growth_min: Optional[float] = Field(None, ge=-100, le=1000)
    debt_ratio_max: Optional[float] = Field(None, ge=0, le=100)


class StockFilterCriteria(BaseModel):
    """股票筛选条件"""
    markets: Optional[List[str]] = Field(None, description="市场列表，如['SH', 'SZ']")
    sectors: Optional[List[str]] = Field(None, description="行业列表")
    industries: Optional[List[str]] = Field(None, description="细分行业列表")
    price_range: Optional[PriceRange] = None
    volume_range: Optional[VolumeRange] = None
    market_cap_range: Optional[MarketCapRange] = None
    technical: Optional[TechnicalCriteria] = None
    fundamental: Optional[FundamentalCriteria] = None


class StockScreeningRequest(BaseModel):
    """股票筛选请求"""
    keyword: Optional[str] = Field(None, description="搜索关键词（股票代码或名称）")
    filters: Optional[StockFilterCriteria] = None
    sort_by: Optional[SortField] = Field(SortField.VOLUME, description="排序字段")
    sort_order: Optional[SortOrder] = Field(SortOrder.DESC, description="排序顺序")
    page: int = Field(1, ge=1, description="页码")
    page_size: int = Field(20, ge=1, le=1000, description="每页数量")
    include_real_time: bool = Field(False, description="是否包含实时数据")
    export_format: Optional[str] = Field(None, pattern="^(csv|excel|json)$", description="导出格式")


class StockInfo(BaseModel):
    """股票信息"""
    symbol: str
    name: str
    market: str
    sector: Optional[str] = None
    industry: Optional[str] = None
    current_price: Optional[float] = None
    change_amount: Optional[float] = None
    change_percent: Optional[float] = None
    volume: Optional[int] = None
    turnover: Optional[float] = None
    market_cap: Optional[float] = None
    technical_indicators: Optional[Dict[str, float]] = None
    fundamental_data: Optional[Dict[str, float]] = None
    last_updated: Optional[datetime] = None


class StockScreeningResponse(BaseModel):
    """股票筛选响应"""
    stocks: List[StockInfo]
    total: int
    page: int
    page_size: int
    has_more: bool


class StockScreeningService:
    """股票筛选服务"""
    
    def __init__(self, db_manager=None):
        if db_manager is None:
            container = get_container()
            self.db_manager = container.db_manager()
        else:
            self.db_manager = db_manager
        self.stock_repo = StockRepository(self.db_manager)
        self.kline_repo = KlineRepository(self.db_manager)
        self.technical_indicators = TechnicalIndicators()
        self._cache = {}  # 简单的内存缓存
    
    async def screen_stocks(self, request: StockScreeningRequest) -> StockScreeningResponse:
        """筛选股票"""
        # 构建缓存键
        cache_key = self._build_cache_key(request)
        
        # 检查缓存
        if cache_key in self._cache:
            cached_data, cached_time = self._cache[cache_key]
            if (datetime.now() - cached_time).seconds < 300:  # 5分钟缓存
                return cached_data
        
        # 获取所有股票
        all_stocks = await self._get_all_stocks()
        
        # 应用关键词搜索
        if request.keyword:
            all_stocks = self._filter_by_keyword(all_stocks, request.keyword)
        
        # 应用筛选条件
        if request.filters:
            all_stocks = await self._apply_filters(all_stocks, request.filters)
        
        # 获取股票的最新数据
        stock_data = await self._enrich_stock_data(
            all_stocks, 
            include_real_time=request.include_real_time
        )
        
        # 排序
        sorted_stocks = self._sort_stocks(stock_data, request.sort_by, request.sort_order)
        
        # 分页
        total = len(sorted_stocks)
        start_index = (request.page - 1) * request.page_size
        end_index = start_index + request.page_size
        page_stocks = sorted_stocks[start_index:end_index]
        
        # 构建响应
        response = StockScreeningResponse(
            stocks=page_stocks,
            total=total,
            page=request.page,
            page_size=request.page_size,
            has_more=end_index < total
        )
        
        # 更新缓存
        self._cache[cache_key] = (response, datetime.now())
        
        return response
    
    async def export_screening_results(
        self, request: StockScreeningRequest, format: str
    ) -> Response:
        """导出筛选结果"""
        # 获取所有结果（不分页）
        request.page_size = 1000  # 设置较大的页大小
        results = await self.screen_stocks(request)
        
        if format == "csv":
            return self._export_to_csv(results.stocks)
        elif format == "excel":
            return self._export_to_excel(results.stocks)
        elif format == "json":
            return self._export_to_json(results.stocks)
        else:
            raise HTTPException(status_code=400, detail="Unsupported export format")
    
    async def _get_all_stocks(self) -> List[Dict[str, Any]]:
        """获取所有股票"""
        return await self.stock_repo.get_all(limit=10000)
    
    def _filter_by_keyword(self, stocks: List[Dict[str, Any]], keyword: str) -> List[Dict[str, Any]]:
        """按关键词过滤"""
        keyword_lower = keyword.lower()
        return [
            stock for stock in stocks
            if keyword_lower in stock['symbol'].lower() or 
               keyword_lower in stock['name'].lower() or
               keyword_lower in stock.get('industry', '').lower()
        ]
    
    async def _apply_filters(
        self, stocks: List[Dict[str, Any]], filters: StockFilterCriteria
    ) -> List[Dict[str, Any]]:
        """应用筛选条件"""
        filtered_stocks = stocks
        
        # 市场筛选
        if filters.markets:
            filtered_stocks = [
                s for s in filtered_stocks 
                if s['market'] in filters.markets
            ]
        
        # 行业筛选
        if filters.sectors:
            filtered_stocks = [
                s for s in filtered_stocks 
                if s.get('sector') in filters.sectors
            ]
        
        # 细分行业筛选
        if filters.industries:
            filtered_stocks = [
                s for s in filtered_stocks 
                if s.get('industry') in filters.industries
            ]
        
        # 获取股票的价格和成交量数据
        if any([filters.price_range, filters.volume_range, filters.technical]):
            stock_symbols = [s['symbol'] for s in filtered_stocks]
            latest_klines = await self._get_latest_klines(stock_symbols)
            
            # 价格筛选
            if filters.price_range:
                filtered_stocks = [
                    s for s in filtered_stocks
                    if s['symbol'] in latest_klines and
                       self._check_price_range(
                           latest_klines[s['symbol']]['close_price'],
                           filters.price_range
                       )
                ]
            
            # 成交量筛选
            if filters.volume_range:
                filtered_stocks = [
                    s for s in filtered_stocks
                    if s['symbol'] in latest_klines and
                       self._check_volume_range(
                           latest_klines[s['symbol']]['volume'],
                           filters.volume_range
                       )
                ]
            
            # 技术指标筛选
            if filters.technical:
                filtered_stocks = await self._apply_technical_filters(
                    filtered_stocks, filters.technical, latest_klines
                )
        
        # 基本面筛选（模拟数据）
        if filters.fundamental:
            filtered_stocks = self._apply_fundamental_filters(
                filtered_stocks, filters.fundamental
            )
        
        return filtered_stocks
    
    async def _get_latest_klines(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """获取股票的最新K线数据"""
        result = {}
        for symbol in symbols:
            klines = await self.kline_repo.get_latest(symbol, limit=1)
            if klines:
                result[symbol] = klines[0]
        return result
    
    def _check_price_range(self, price: float, price_range: PriceRange) -> bool:
        """检查价格是否在范围内"""
        if price_range.min is not None and price < price_range.min:
            return False
        if price_range.max is not None and price > price_range.max:
            return False
        return True
    
    def _check_volume_range(self, volume: int, volume_range: VolumeRange) -> bool:
        """检查成交量是否在范围内"""
        if volume_range.min is not None and volume < volume_range.min:
            return False
        if volume_range.max is not None and volume > volume_range.max:
            return False
        return True
    
    async def _apply_technical_filters(
        self, stocks: List[Dict[str, Any]], 
        technical: TechnicalCriteria,
        latest_klines: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """应用技术指标筛选"""
        filtered_stocks = []
        
        for stock in stocks:
            symbol = stock['symbol']
            if symbol not in latest_klines:
                continue
            
            # 获取历史数据计算技术指标
            klines = await self.kline_repo.get_latest(symbol, limit=60)
            if len(klines) < 20:  # 数据不足
                continue
            
            # 准备数据
            closes = [k['close_price'] for k in reversed(klines)]
            volumes = [k['volume'] for k in reversed(klines)]
            
            # 计算技术指标
            indicators = {}
            
            # RSI
            if technical.rsi_range:
                rsi = self.technical_indicators.calculate_rsi(closes, period=14)
                if rsi is None or not self._check_rsi_range(rsi, technical.rsi_range):
                    continue
                indicators['rsi'] = rsi
            
            # 均线
            if technical.ma_condition:
                ma5 = np.mean(closes[-5:])
                ma20 = np.mean(closes[-20:])
                current_price = closes[-1]
                
                if technical.ma_condition.ma5_above_ma20 is not None:
                    if (ma5 > ma20) != technical.ma_condition.ma5_above_ma20:
                        continue
                
                if technical.ma_condition.price_above_ma5 is not None:
                    if (current_price > ma5) != technical.ma_condition.price_above_ma5:
                        continue
                
                indicators['ma5'] = ma5
                indicators['ma20'] = ma20
            
            # 成交量条件
            if technical.volume_condition:
                avg_volume = np.mean(volumes[-20:])
                current_volume = volumes[-1]
                
                if technical.volume_condition.above_average:
                    if current_volume <= avg_volume:
                        continue
                
                if technical.volume_condition.multiple is not None:
                    if current_volume < avg_volume * technical.volume_condition.multiple:
                        continue
            
            stock['technical_indicators'] = indicators
            filtered_stocks.append(stock)
        
        return filtered_stocks
    
    def _check_rsi_range(self, rsi: float, rsi_range: RSIRange) -> bool:
        """检查RSI是否在范围内"""
        if rsi_range.min is not None and rsi < rsi_range.min:
            return False
        if rsi_range.max is not None and rsi > rsi_range.max:
            return False
        return True
    
    def _apply_fundamental_filters(
        self, stocks: List[Dict[str, Any]], fundamental: FundamentalCriteria
    ) -> List[Dict[str, Any]]:
        """应用基本面筛选（模拟数据）"""
        # 实际应用中，这里应该从财务数据库获取真实数据
        # 这里使用模拟数据进行演示
        filtered_stocks = []
        
        for stock in stocks:
            # 模拟基本面数据
            mock_fundamentals = {
                'pe': np.random.uniform(5, 50),
                'pb': np.random.uniform(0.5, 10),
                'roe': np.random.uniform(5, 30),
                'revenue_growth': np.random.uniform(-20, 50),
                'profit_growth': np.random.uniform(-30, 60),
                'debt_ratio': np.random.uniform(10, 80)
            }
            
            # 检查PE范围
            if fundamental.pe_range:
                if not self._check_value_range(
                    mock_fundamentals['pe'], 
                    fundamental.pe_range.min, 
                    fundamental.pe_range.max
                ):
                    continue
            
            # 检查PB范围
            if fundamental.pb_range:
                if not self._check_value_range(
                    mock_fundamentals['pb'], 
                    fundamental.pb_range.min, 
                    fundamental.pb_range.max
                ):
                    continue
            
            # 检查ROE
            if fundamental.roe_min is not None:
                if mock_fundamentals['roe'] < fundamental.roe_min:
                    continue
            
            # 检查营收增长
            if fundamental.revenue_growth_min is not None:
                if mock_fundamentals['revenue_growth'] < fundamental.revenue_growth_min:
                    continue
            
            # 检查利润增长
            if fundamental.profit_growth_min is not None:
                if mock_fundamentals['profit_growth'] < fundamental.profit_growth_min:
                    continue
            
            # 检查负债率
            if fundamental.debt_ratio_max is not None:
                if mock_fundamentals['debt_ratio'] > fundamental.debt_ratio_max:
                    continue
            
            stock['fundamental_data'] = mock_fundamentals
            filtered_stocks.append(stock)
        
        return filtered_stocks
    
    def _check_value_range(self, value: float, min_val: Optional[float], max_val: Optional[float]) -> bool:
        """检查值是否在范围内"""
        if min_val is not None and value < min_val:
            return False
        if max_val is not None and value > max_val:
            return False
        return True
    
    async def _enrich_stock_data(
        self, stocks: List[Dict[str, Any]], include_real_time: bool = False
    ) -> List[StockInfo]:
        """丰富股票数据"""
        enriched_stocks = []
        
        for stock in stocks:
            # 获取最新K线数据
            klines = await self.kline_repo.get_latest(stock['symbol'], limit=1)
            
            stock_info = StockInfo(
                symbol=stock['symbol'],
                name=stock['name'],
                market=stock['market'],
                sector=stock.get('sector'),
                industry=stock.get('industry'),
                technical_indicators=stock.get('technical_indicators'),
                fundamental_data=stock.get('fundamental_data')
            )
            
            if klines:
                latest_kline = klines[0]
                stock_info.current_price = latest_kline['close_price']
                stock_info.volume = latest_kline['volume']
                stock_info.turnover = latest_kline['turnover']
                
                # 计算涨跌额和涨跌幅
                if len(klines) > 1:
                    prev_close = klines[1]['close_price']
                    stock_info.change_amount = latest_kline['close_price'] - prev_close
                    stock_info.change_percent = (stock_info.change_amount / prev_close) * 100
            
            # 模拟市值数据
            if stock_info.current_price:
                stock_info.market_cap = stock_info.current_price * np.random.uniform(1e8, 1e11)
            
            if include_real_time:
                stock_info.last_updated = datetime.now()
            
            enriched_stocks.append(stock_info)
        
        return enriched_stocks
    
    def _sort_stocks(
        self, stocks: List[StockInfo], sort_by: SortField, sort_order: SortOrder
    ) -> List[StockInfo]:
        """排序股票"""
        # 定义排序键
        sort_keys = {
            SortField.PRICE: lambda s: s.current_price or 0,
            SortField.VOLUME: lambda s: s.volume or 0,
            SortField.CHANGE_PERCENT: lambda s: s.change_percent or 0,
            SortField.MARKET_CAP: lambda s: s.market_cap or 0,
            SortField.PE_RATIO: lambda s: s.fundamental_data.get('pe', 0) if s.fundamental_data else 0,
            SortField.PB_RATIO: lambda s: s.fundamental_data.get('pb', 0) if s.fundamental_data else 0,
            SortField.ROE: lambda s: s.fundamental_data.get('roe', 0) if s.fundamental_data else 0,
            SortField.TURNOVER: lambda s: s.turnover or 0,
        }
        
        key_func = sort_keys.get(sort_by, sort_keys[SortField.VOLUME])
        reverse = sort_order == SortOrder.DESC
        
        return sorted(stocks, key=key_func, reverse=reverse)
    
    def _build_cache_key(self, request: StockScreeningRequest) -> str:
        """构建缓存键"""
        # 简单的缓存键生成，实际应用中可能需要更复杂的逻辑
        key_parts = [
            request.keyword or '',
            str(request.filters) if request.filters else '',
            request.sort_by,
            request.sort_order,
            str(request.page),
            str(request.page_size)
        ]
        return '|'.join(key_parts)
    
    def _export_to_csv(self, stocks: List[StockInfo]) -> Response:
        """导出为CSV"""
        output = io.StringIO()
        writer = csv.writer(output)
        
        # 写入标题行
        headers = [
            'Symbol', 'Name', 'Market', 'Sector', 'Industry',
            'Current Price', 'Change %', 'Volume', 'Turnover', 'Market Cap'
        ]
        writer.writerow(headers)
        
        # 写入数据行
        for stock in stocks:
            row = [
                stock.symbol,
                stock.name,
                stock.market,
                stock.sector or '',
                stock.industry or '',
                stock.current_price or '',
                f"{stock.change_percent:.2f}" if stock.change_percent else '',
                stock.volume or '',
                stock.turnover or '',
                f"{stock.market_cap:.0f}" if stock.market_cap else ''
            ]
            writer.writerow(row)
        
        content = output.getvalue()
        output.close()
        
        return Response(
            content=content,
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=stock_screening_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            }
        )
    
    def _export_to_excel(self, stocks: List[StockInfo]) -> Response:
        """导出为Excel"""
        # 转换为DataFrame
        data = []
        for stock in stocks:
            data.append({
                'Symbol': stock.symbol,
                'Name': stock.name,
                'Market': stock.market,
                'Sector': stock.sector,
                'Industry': stock.industry,
                'Current Price': stock.current_price,
                'Change %': stock.change_percent,
                'Volume': stock.volume,
                'Turnover': stock.turnover,
                'Market Cap': stock.market_cap
            })
        
        df = pd.DataFrame(data)
        
        # 导出为Excel
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Stock Screening', index=False)
        
        content = output.getvalue()
        output.close()
        
        return Response(
            content=content,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": f"attachment; filename=stock_screening_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            }
        )
    
    def _export_to_json(self, stocks: List[StockInfo]) -> Response:
        """导出为JSON"""
        data = [stock.dict() for stock in stocks]
        
        return Response(
            content=json.dumps(data, ensure_ascii=False, indent=2, default=str),
            media_type="application/json",
            headers={
                "Content-Disposition": f"attachment; filename=stock_screening_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            }
        )


# 创建服务实例
screening_service = StockScreeningService()


@router.post("/filter", response_model=StockScreeningResponse)
async def filter_stocks(request: StockScreeningRequest):
    """股票筛选接口"""
    try:
        return await screening_service.screen_stocks(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/export")
async def export_stocks(request: StockScreeningRequest):
    """导出筛选结果"""
    if not request.export_format:
        raise HTTPException(status_code=400, detail="Export format is required")
    
    try:
        return await screening_service.export_screening_results(
            request, request.export_format
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
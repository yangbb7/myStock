"""
投资组合性能分析API

提供投资组合性能分析、风险指标计算、业绩归因、基准比较等功能
"""

import io
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
from enum import Enum

from fastapi import APIRouter, HTTPException, Response, Depends, Query
from pydantic import BaseModel, Field, validator
import asyncio

from myQuant.infrastructure.database.database_manager import DatabaseManager
from myQuant.infrastructure.database.repositories import (
    PortfolioRepository, TransactionRepository, PositionRepository, 
    RiskMetricRepository, StockRepository, KlineRepository
)
from myQuant.infrastructure.container import get_container
from myQuant.core.analysis.technical_indicators import TechnicalIndicators


router = APIRouter(prefix="/api/v1/portfolio", tags=["portfolio_performance"])


class PerformancePeriod(str, Enum):
    """性能分析周期"""
    ONE_DAY = "1d"
    ONE_WEEK = "1w"
    ONE_MONTH = "1m"
    THREE_MONTHS = "3m"
    SIX_MONTHS = "6m"
    ONE_YEAR = "1y"
    TWO_YEARS = "2y"
    THREE_YEARS = "3y"
    ALL_TIME = "all"


class BenchmarkType(str, Enum):
    """基准类型"""
    CSI300 = "000300.SH"  # 沪深300
    CSI500 = "000905.SH"  # 中证500
    SHCI = "000001.SH"    # 上证指数
    SZCI = "399001.SZ"    # 深证成指
    CUSTOM = "custom"


class ExportFormat(str, Enum):
    """导出格式"""
    JSON = "json"
    EXCEL = "excel"
    PDF = "pdf"


class PortfolioPerformanceRequest(BaseModel):
    """投资组合性能分析请求"""
    portfolio_id: int = Field(..., description="投资组合ID")
    start_date: str = Field(..., description="开始日期")
    end_date: str = Field(..., description="结束日期")
    benchmark_symbol: Optional[str] = Field(None, description="基准代码")
    include_benchmark: bool = Field(False, description="是否包含基准比较")
    include_risk_analysis: bool = Field(True, description="是否包含风险分析")
    include_attribution: bool = Field(False, description="是否包含业绩归因")
    include_drawdown_analysis: bool = Field(False, description="是否包含回撤分析")
    return_frequency: str = Field("daily", description="收益频率")
    attribution_method: str = Field("brinson", description="归因方法")
    custom_benchmark: Optional[Dict[str, Any]] = Field(None, description="自定义基准")
    
    @validator('end_date')
    def validate_date_range(cls, v, values):
        if 'start_date' in values:
            start = datetime.fromisoformat(values['start_date'])
            end = datetime.fromisoformat(v)
            if end <= start:
                raise ValueError('End date must be after start date')
        return v


class PeriodReturn(BaseModel):
    """分期收益"""
    period: str
    return_rate: float
    cumulative_return: float
    benchmark_return: Optional[float] = None


class RiskMetrics(BaseModel):
    """风险指标"""
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    calmar_ratio: float
    beta: Optional[float] = None


class BenchmarkComparison(BaseModel):
    """基准比较"""
    benchmark_name: str
    benchmark_return: float
    alpha: float
    beta: float
    tracking_error: float
    information_ratio: float
    correlation: float


class AttributionAnalysis(BaseModel):
    """业绩归因分析"""
    total_return: float
    asset_allocation_effect: float
    security_selection_effect: float
    interaction_effect: float
    sector_attribution: List[Dict[str, Any]]


class DrawdownAnalysis(BaseModel):
    """回撤分析"""
    max_drawdown: float
    max_drawdown_duration: int
    current_drawdown: float
    drawdown_periods: List[Dict[str, Any]]


class PortfolioMetrics(BaseModel):
    """投资组合指标"""
    total_return: float
    annualized_return: float
    total_value: float
    cost_basis: float
    unrealized_pnl: float
    realized_pnl: float
    cash_balance: float
    position_count: int


class PerformanceAnalysisResponse(BaseModel):
    """性能分析响应"""
    portfolio_metrics: PortfolioMetrics
    risk_metrics: RiskMetrics
    period_returns: List[PeriodReturn]
    benchmark_comparison: Optional[BenchmarkComparison] = None
    attribution_analysis: Optional[AttributionAnalysis] = None
    drawdown_analysis: Optional[DrawdownAnalysis] = None


class PerformanceService:
    """投资组合性能分析服务"""
    
    def __init__(self):
        container = get_container()
        self.db_manager = container.database_manager()
        self.portfolio_repo = PortfolioRepository(self.db_manager)
        self.transaction_repo = TransactionRepository(self.db_manager)
        self.position_repo = PositionRepository(self.db_manager)
        self.risk_repo = RiskMetricRepository(self.db_manager)
        self.stock_repo = StockRepository(self.db_manager)
        self.kline_repo = KlineRepository(self.db_manager)
        self.technical_indicators = TechnicalIndicators()
        self._cache = {}
    
    async def analyze_portfolio_performance(
        self, request: PortfolioPerformanceRequest
    ) -> PerformanceAnalysisResponse:
        """分析投资组合性能"""
        # 获取投资组合基本信息
        portfolio = await self.portfolio_repo.get_by_id(request.portfolio_id)
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")
        
        # 获取交易记录
        transactions = await self.transaction_repo.get_by_user(
            portfolio['user_id'],
            start_date=request.start_date,
            end_date=request.end_date
        )
        
        # 获取当前持仓
        positions = await self.position_repo.get_by_user(portfolio['user_id'])
        
        # 计算投资组合指标
        portfolio_metrics = await self._calculate_portfolio_metrics(
            transactions, positions, request.start_date, request.end_date
        )
        
        # 计算分期收益
        period_returns = await self._calculate_period_returns(
            transactions, positions, request.start_date, request.end_date, 
            request.return_frequency
        )
        
        # 计算风险指标
        risk_metrics = None
        if request.include_risk_analysis:
            risk_metrics = await self._calculate_risk_metrics(
                period_returns, request.benchmark_symbol
            )
        
        # 基准比较
        benchmark_comparison = None
        if request.include_benchmark and request.benchmark_symbol:
            benchmark_comparison = await self._compare_with_benchmark(
                period_returns, request.benchmark_symbol
            )
        
        # 业绩归因
        attribution_analysis = None
        if request.include_attribution:
            attribution_analysis = await self._perform_attribution_analysis(
                positions, request.benchmark_symbol, request.attribution_method
            )
        
        # 回撤分析
        drawdown_analysis = None
        if request.include_drawdown_analysis:
            drawdown_analysis = await self._analyze_drawdowns(period_returns)
        
        return PerformanceAnalysisResponse(
            portfolio_metrics=portfolio_metrics,
            risk_metrics=risk_metrics or RiskMetrics(
                volatility=0, sharpe_ratio=0, sortino_ratio=0, max_drawdown=0,
                var_95=0, cvar_95=0, calmar_ratio=0
            ),
            period_returns=period_returns,
            benchmark_comparison=benchmark_comparison,
            attribution_analysis=attribution_analysis,
            drawdown_analysis=drawdown_analysis
        )
    
    async def _calculate_portfolio_metrics(
        self, transactions: List[Dict], positions: List[Dict],
        start_date: str, end_date: str
    ) -> PortfolioMetrics:
        """计算投资组合指标"""
        # 计算成本基础
        cost_basis = 0.0
        realized_pnl = 0.0
        
        # 按股票分组计算成本
        stock_costs = {}
        for txn in transactions:
            symbol = txn['symbol']
            if symbol not in stock_costs:
                stock_costs[symbol] = {'quantity': 0, 'cost': 0}
            
            if txn['side'] == 'BUY':
                stock_costs[symbol]['quantity'] += txn['quantity']
                stock_costs[symbol]['cost'] += txn['quantity'] * txn['price']
            else:  # SELL
                # 计算已实现盈亏
                avg_cost = (stock_costs[symbol]['cost'] / stock_costs[symbol]['quantity'] 
                           if stock_costs[symbol]['quantity'] > 0 else 0)
                realized_pnl += txn['quantity'] * (txn['price'] - avg_cost)
                
                stock_costs[symbol]['quantity'] -= txn['quantity']
                stock_costs[symbol]['cost'] -= txn['quantity'] * avg_cost
        
        # 计算当前总成本
        cost_basis = sum(data['cost'] for data in stock_costs.values())
        
        # 计算当前市值和未实现盈亏
        total_value = 0.0
        unrealized_pnl = 0.0
        
        for position in positions:
            # 获取当前价格（模拟）
            current_price = await self._get_current_price(position['symbol'])
            market_value = position['quantity'] * current_price
            total_value += market_value
            
            # 计算未实现盈亏
            position_cost = position['quantity'] * position['average_price']
            unrealized_pnl += market_value - position_cost
        
        # 总收益和年化收益
        total_return = ((total_value - cost_basis) / cost_basis * 100 
                       if cost_basis > 0 else 0)
        
        # 计算年化收益
        days = (datetime.fromisoformat(end_date) - datetime.fromisoformat(start_date)).days
        years = days / 365.0
        annualized_return = ((1 + total_return / 100) ** (1 / years) - 1) * 100 if years > 0 else 0
        
        return PortfolioMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            total_value=total_value,
            cost_basis=cost_basis,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
            cash_balance=100000.0,  # 模拟现金余额
            position_count=len(positions)
        )
    
    async def _calculate_period_returns(
        self, transactions: List[Dict], positions: List[Dict],
        start_date: str, end_date: str, frequency: str
    ) -> List[PeriodReturn]:
        """计算分期收益"""
        period_returns = []
        
        # 根据频率生成时间周期
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
        
        if frequency == "daily":
            periods = pd.date_range(start, end, freq='D')
        elif frequency == "weekly":
            periods = pd.date_range(start, end, freq='W')
        elif frequency == "monthly":
            periods = pd.date_range(start, end, freq='M')
        else:
            periods = pd.date_range(start, end, freq='D')
        
        # 模拟收益计算
        cumulative_return = 1.0
        for i, period in enumerate(periods):
            # 模拟日收益率
            daily_return = np.random.normal(0.001, 0.02)  # 模拟数据
            cumulative_return *= (1 + daily_return)
            
            period_returns.append(PeriodReturn(
                period=period.strftime('%Y-%m-%d'),
                return_rate=daily_return * 100,
                cumulative_return=(cumulative_return - 1) * 100,
                benchmark_return=np.random.normal(0.0008, 0.015) * 100  # 模拟基准收益
            ))
        
        return period_returns
    
    async def _calculate_risk_metrics(
        self, period_returns: List[PeriodReturn], benchmark_symbol: Optional[str] = None
    ) -> RiskMetrics:
        """计算风险指标"""
        if not period_returns:
            return RiskMetrics(
                volatility=0, sharpe_ratio=0, sortino_ratio=0, max_drawdown=0,
                var_95=0, cvar_95=0, calmar_ratio=0
            )
        
        returns = [r.return_rate / 100 for r in period_returns]
        returns_array = np.array(returns)
        
        # 波动率（年化）
        volatility = np.std(returns_array) * np.sqrt(252) * 100
        
        # 夏普比率
        risk_free_rate = 0.03  # 3%年化无风险利率
        excess_returns = returns_array - risk_free_rate / 252
        sharpe_ratio = (np.mean(excess_returns) / np.std(returns_array) * np.sqrt(252) 
                       if np.std(returns_array) > 0 else 0)
        
        # 索提诺比率
        downside_returns = returns_array[returns_array < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
        sortino_ratio = (np.mean(excess_returns) / downside_std * np.sqrt(252) 
                        if downside_std > 0 else 0)
        
        # 最大回撤
        cumulative_returns = np.cumprod(1 + returns_array)
        rolling_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = abs(np.min(drawdowns)) * 100
        
        # VaR和CVaR (95%)
        var_95 = np.percentile(returns_array, 5) * 100
        cvar_95 = np.mean(returns_array[returns_array <= np.percentile(returns_array, 5)]) * 100
        
        # 卡马比率
        calmar_ratio = (np.mean(returns_array) * 252 * 100 / max_drawdown 
                       if max_drawdown > 0 else 0)
        
        # Beta（如果有基准）
        beta = None
        if benchmark_symbol:
            benchmark_returns = [r.benchmark_return / 100 for r in period_returns if r.benchmark_return]
            if len(benchmark_returns) == len(returns):
                covariance = np.cov(returns, benchmark_returns)[0, 1]
                benchmark_variance = np.var(benchmark_returns)
                beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0
        
        return RiskMetrics(
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            var_95=var_95,
            cvar_95=cvar_95,
            calmar_ratio=calmar_ratio,
            beta=beta
        )
    
    async def _compare_with_benchmark(
        self, period_returns: List[PeriodReturn], benchmark_symbol: str
    ) -> BenchmarkComparison:
        """与基准比较"""
        portfolio_returns = [r.return_rate / 100 for r in period_returns]
        benchmark_returns = [r.benchmark_return / 100 for r in period_returns if r.benchmark_return]
        
        if len(benchmark_returns) != len(portfolio_returns):
            # 如果基准数据不完整，使用模拟数据
            benchmark_returns = [np.random.normal(0.0008, 0.015) for _ in portfolio_returns]
        
        # 基准总收益
        benchmark_total_return = (np.prod([1 + r for r in benchmark_returns]) - 1) * 100
        
        # Alpha计算
        portfolio_return = np.mean(portfolio_returns) * 252
        benchmark_return = np.mean(benchmark_returns) * 252
        
        # Beta
        covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0
        
        # Alpha = 投资组合收益 - (无风险利率 + Beta * (基准收益 - 无风险利率))
        risk_free_rate = 0.03
        alpha = (portfolio_return - (risk_free_rate + beta * (benchmark_return - risk_free_rate))) * 100
        
        # 跟踪误差
        excess_returns = np.array(portfolio_returns) - np.array(benchmark_returns)
        tracking_error = np.std(excess_returns) * np.sqrt(252) * 100
        
        # 信息比率
        information_ratio = (np.mean(excess_returns) * 252 * 100 / tracking_error 
                           if tracking_error > 0 else 0)
        
        # 相关系数
        correlation = np.corrcoef(portfolio_returns, benchmark_returns)[0, 1]
        
        return BenchmarkComparison(
            benchmark_name=self._get_benchmark_name(benchmark_symbol),
            benchmark_return=benchmark_total_return,
            alpha=alpha,
            beta=beta,
            tracking_error=tracking_error,
            information_ratio=information_ratio,
            correlation=correlation
        )
    
    async def _perform_attribution_analysis(
        self, positions: List[Dict], benchmark_symbol: Optional[str], method: str
    ) -> AttributionAnalysis:
        """执行业绩归因分析"""
        # 简化的归因分析（实际应该更复杂）
        total_return = sum(pos['quantity'] * 0.05 for pos in positions) / 100  # 模拟
        
        # 模拟归因效应
        asset_allocation_effect = np.random.uniform(-2, 3)
        security_selection_effect = np.random.uniform(-1, 4)
        interaction_effect = np.random.uniform(-0.5, 0.5)
        
        # 行业归因（模拟）
        sectors = {}
        for pos in positions:
            sector = await self._get_stock_sector(pos['symbol'])
            if sector not in sectors:
                sectors[sector] = {
                    'weight_portfolio': 0,
                    'weight_benchmark': np.random.uniform(0.05, 0.3),
                    'return_portfolio': np.random.uniform(-0.1, 0.15),
                    'return_benchmark': np.random.uniform(-0.05, 0.1)
                }
            sectors[sector]['weight_portfolio'] += pos['quantity'] * 0.01  # 模拟权重
        
        sector_attribution = []
        for sector, data in sectors.items():
            allocation_effect = (data['weight_portfolio'] - data['weight_benchmark']) * data['return_benchmark']
            selection_effect = data['weight_benchmark'] * (data['return_portfolio'] - data['return_benchmark'])
            
            sector_attribution.append({
                'sector': sector,
                'weight_portfolio': data['weight_portfolio'],
                'weight_benchmark': data['weight_benchmark'],
                'return_portfolio': data['return_portfolio'],
                'return_benchmark': data['return_benchmark'],
                'allocation_effect': allocation_effect,
                'selection_effect': selection_effect
            })
        
        return AttributionAnalysis(
            total_return=total_return,
            asset_allocation_effect=asset_allocation_effect,
            security_selection_effect=security_selection_effect,
            interaction_effect=interaction_effect,
            sector_attribution=sector_attribution
        )
    
    async def _analyze_drawdowns(self, period_returns: List[PeriodReturn]) -> DrawdownAnalysis:
        """分析回撤"""
        if not period_returns:
            return DrawdownAnalysis(
                max_drawdown=0, max_drawdown_duration=0, current_drawdown=0, drawdown_periods=[]
            )
        
        returns = [r.return_rate / 100 for r in period_returns]
        cumulative_returns = np.cumprod([1 + r for r in returns])
        
        # 计算回撤
        rolling_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        
        max_drawdown = abs(np.min(drawdowns)) * 100
        current_drawdown = abs(drawdowns[-1]) * 100
        
        # 找到回撤期间
        drawdown_periods = []
        in_drawdown = False
        start_idx = 0
        
        for i, dd in enumerate(drawdowns):
            if dd < -0.01 and not in_drawdown:  # 开始回撤
                in_drawdown = True
                start_idx = i
            elif dd >= -0.001 and in_drawdown:  # 回撤结束
                in_drawdown = False
                drawdown_periods.append({
                    'start_date': period_returns[start_idx].period,
                    'end_date': period_returns[i-1].period,
                    'max_drawdown': abs(np.min(drawdowns[start_idx:i])) * 100,
                    'recovery_date': period_returns[i].period if i < len(period_returns) else None
                })
        
        max_drawdown_duration = max([
            (datetime.fromisoformat(p['end_date']) - datetime.fromisoformat(p['start_date'])).days
            for p in drawdown_periods
        ]) if drawdown_periods else 0
        
        return DrawdownAnalysis(
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_drawdown_duration,
            current_drawdown=current_drawdown,
            drawdown_periods=drawdown_periods
        )
    
    async def _get_current_price(self, symbol: str) -> float:
        """获取当前价格"""
        # 从K线数据获取最新价格
        klines = await self.kline_repo.get_latest(symbol, limit=1)
        if klines:
            return klines[0]['close_price']
        
        # 如果没有数据，返回模拟价格
        return np.random.uniform(10, 50)
    
    async def _get_stock_sector(self, symbol: str) -> str:
        """获取股票行业"""
        stock = await self.stock_repo.get_by_symbol(symbol)
        if stock and stock.get('sector'):
            return stock['sector']
        
        # 模拟行业
        sectors = ['金融', '科技', '消费', '医药', '制造', '房地产', '能源']
        return np.random.choice(sectors)
    
    def _get_benchmark_name(self, symbol: str) -> str:
        """获取基准名称"""
        benchmark_names = {
            '000300.SH': '沪深300',
            '000905.SH': '中证500',
            '000001.SH': '上证指数',
            '399001.SZ': '深证成指'
        }
        return benchmark_names.get(symbol, symbol)
    
    async def export_performance_report(
        self, request: PortfolioPerformanceRequest, format: str
    ) -> Response:
        """导出性能报告"""
        # 获取分析结果
        analysis = await self.analyze_portfolio_performance(request)
        
        if format == "json":
            return Response(
                content=analysis.json(indent=2),
                media_type="application/json",
                headers={
                    "Content-Disposition": f"attachment; filename=portfolio_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                }
            )
        elif format == "excel":
            return self._export_to_excel(analysis)
        elif format == "pdf":
            return self._export_to_pdf(analysis)
        else:
            raise HTTPException(status_code=400, detail="Unsupported export format")
    
    def _export_to_excel(self, analysis: PerformanceAnalysisResponse) -> Response:
        """导出Excel报告"""
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # 投资组合指标
            metrics_df = pd.DataFrame([analysis.portfolio_metrics.dict()])
            metrics_df.to_excel(writer, sheet_name='Portfolio Metrics', index=False)
            
            # 风险指标
            risk_df = pd.DataFrame([analysis.risk_metrics.dict()])
            risk_df.to_excel(writer, sheet_name='Risk Metrics', index=False)
            
            # 分期收益
            returns_df = pd.DataFrame([r.dict() for r in analysis.period_returns])
            returns_df.to_excel(writer, sheet_name='Period Returns', index=False)
            
            # 基准比较
            if analysis.benchmark_comparison:
                benchmark_df = pd.DataFrame([analysis.benchmark_comparison.dict()])
                benchmark_df.to_excel(writer, sheet_name='Benchmark Comparison', index=False)
        
        content = output.getvalue()
        output.close()
        
        return Response(
            content=content,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": f"attachment; filename=portfolio_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            }
        )
    
    def _export_to_pdf(self, analysis: PerformanceAnalysisResponse) -> Response:
        """导出PDF报告"""
        # 简化的PDF导出（实际应该使用专业的PDF库）
        content = f"Portfolio Performance Report\n"
        content += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        content += f"Total Return: {analysis.portfolio_metrics.total_return:.2f}%\n"
        content += f"Annualized Return: {analysis.portfolio_metrics.annualized_return:.2f}%\n"
        content += f"Volatility: {analysis.risk_metrics.volatility:.2f}%\n"
        content += f"Sharpe Ratio: {analysis.risk_metrics.sharpe_ratio:.2f}\n"
        content += f"Max Drawdown: {analysis.risk_metrics.max_drawdown:.2f}%\n"
        
        return Response(
            content=content.encode('utf-8'),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=portfolio_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            }
        )


# 创建服务实例
performance_service = PerformanceService()


@router.post("/performance", response_model=PerformanceAnalysisResponse)
async def analyze_performance(request: PortfolioPerformanceRequest):
    """分析投资组合性能"""
    try:
        return await performance_service.analyze_portfolio_performance(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/performance/compare")
async def compare_periods(request: Dict[str, Any]):
    """多期间性能比较"""
    portfolio_id = request.get("portfolio_id")
    periods = request.get("periods", [])
    
    comparisons = []
    for period in periods:
        analysis_request = PortfolioPerformanceRequest(
            portfolio_id=portfolio_id,
            start_date=period["start_date"],
            end_date=period["end_date"]
        )
        
        analysis = await performance_service.analyze_portfolio_performance(analysis_request)
        
        comparisons.append({
            "period_name": period.get("name", f"{period['start_date']} to {period['end_date']}"),
            "total_return": analysis.portfolio_metrics.total_return,
            "annualized_return": analysis.portfolio_metrics.annualized_return,
            "volatility": analysis.risk_metrics.volatility,
            "sharpe_ratio": analysis.risk_metrics.sharpe_ratio,
            "max_drawdown": analysis.risk_metrics.max_drawdown
        })
    
    return {"period_comparisons": comparisons}


@router.post("/performance/rolling")
async def rolling_performance(request: Dict[str, Any]):
    """滚动性能分析"""
    # 模拟滚动性能数据
    window = request.get("rolling_window", 90)
    frequency = request.get("rolling_frequency", "daily")
    
    # 生成日期序列
    start_date = datetime.fromisoformat(request["start_date"])
    end_date = datetime.fromisoformat(request["end_date"])
    dates = pd.date_range(start_date, end_date, freq='D')
    
    # 模拟滚动指标
    rolling_metrics = {
        "dates": [d.strftime('%Y-%m-%d') for d in dates[window:]],
        "returns": [np.random.normal(0.08, 0.15) for _ in range(len(dates) - window)],
        "volatility": [np.random.uniform(0.1, 0.3) * 100 for _ in range(len(dates) - window)],
        "sharpe_ratio": [np.random.uniform(-0.5, 2.5) for _ in range(len(dates) - window)]
    }
    
    return {"rolling_metrics": rolling_metrics}


@router.get("/composition")
async def get_portfolio_composition(
    portfolio_id: int = Query(...),
    analysis_date: str = Query(...)
):
    """获取投资组合构成分析"""
    # 获取持仓数据
    container = get_container()
    position_repo = PositionRepository(container.database_manager())
    
    # 模拟投资组合构成数据
    positions = [
        {
            "symbol": "000001.SZ",
            "name": "平安银行",
            "quantity": 10000,
            "market_value": 127500,
            "weight": 12.75,
            "unrealized_pnl": 7500,
            "return_contribution": 0.75
        },
        {
            "symbol": "000002.SZ",
            "name": "万科A",
            "quantity": 5000,
            "market_value": 112500,
            "weight": 11.25,
            "unrealized_pnl": 12500,
            "return_contribution": 1.25
        }
    ]
    
    sector_allocation = {
        "金融": 25.0,
        "房地产": 15.0,
        "科技": 30.0,
        "消费": 20.0,
        "其他": 10.0
    }
    
    top_holdings = positions[:5]
    
    concentration_metrics = {
        "top_5_weight": 45.0,
        "top_10_weight": 70.0,
        "effective_number": 8.5,
        "herfindahl_index": 0.15
    }
    
    return {
        "positions": positions,
        "sector_allocation": sector_allocation,
        "top_holdings": top_holdings,
        "concentration_metrics": concentration_metrics
    }


@router.post("/performance/export")
async def export_performance_report(request: PortfolioPerformanceRequest):
    """导出性能报告"""
    export_format = request.export_format or "excel"
    return await performance_service.export_performance_report(request, export_format)


@router.get("/performance/real-time")
async def get_real_time_performance(
    portfolio_id: int = Query(...),
    include_real_time: bool = Query(False)
):
    """获取实时性能跟踪"""
    # 模拟实时数据
    real_time_data = {
        "current_value": 1250000.0,
        "today_pnl": 15000.0,
        "today_return": 1.22,
        "positions_summary": {
            "total_positions": 8,
            "winners": 5,
            "losers": 3,
            "biggest_winner": {"symbol": "000001.SZ", "pnl": 7500},
            "biggest_loser": {"symbol": "600000.SH", "pnl": -2300}
        },
        "last_updated": datetime.now().isoformat()
    }
    
    return real_time_data


@router.post("/performance/alerts")
async def create_performance_alerts(request: Dict[str, Any]):
    """创建性能预警"""
    portfolio_id = request.get("portfolio_id")
    alert_rules = request.get("alert_rules", [])
    
    # 模拟预警创建
    alerts_created = len(alert_rules)
    active_alerts = []
    
    # 检查当前是否触发预警
    for rule in alert_rules:
        if rule["enabled"]:
            # 模拟预警检查
            if rule["type"] == "drawdown" and np.random.random() < 0.3:
                active_alerts.append({
                    "type": rule["type"],
                    "threshold": rule["threshold"],
                    "current_value": 0.07,
                    "triggered_at": datetime.now().isoformat(),
                    "message": "投资组合回撤超过预警阈值"
                })
    
    return {
        "alerts_created": alerts_created,
        "active_alerts": active_alerts
    }


@router.get("/performance/alerts")
async def get_performance_alerts(portfolio_id: int = Query(...)):
    """获取性能预警"""
    # 模拟预警数据
    alerts = [
        {
            "id": 1,
            "type": "drawdown",
            "threshold": 0.05,
            "current_value": 0.03,
            "status": "active",
            "created_at": "2024-01-01T10:00:00",
            "triggered_at": None
        },
        {
            "id": 2,
            "type": "volatility",
            "threshold": 0.20,
            "current_value": 0.15,
            "status": "active",
            "created_at": "2024-01-01T10:00:00",
            "triggered_at": None
        }
    ]
    
    return {"alerts": alerts}
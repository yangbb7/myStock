import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from scipy import stats
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

class ExecutionVenue(Enum):
    NYSE = "nyse"
    NASDAQ = "nasdaq"
    LSE = "lse"
    EURONEXT = "euronext"
    XETRA = "xetra"
    SHANGHAI = "shanghai"
    SHENZHEN = "shenzhen"
    HONG_KONG = "hong_kong"
    TOKYO = "tokyo"
    DARK_POOL = "dark_pool"
    ECN = "ecn"
    CROSSING_NETWORK = "crossing_network"
    INTERNALIZATION = "internalization"

class ExecutionQuality(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    ADEQUATE = "adequate"
    POOR = "poor"
    FAILED = "failed"

class BestExecutionMethod(Enum):
    VWAP = "vwap"
    TWAP = "twap"
    IMPLEMENTATION_SHORTFALL = "implementation_shortfall"
    ARRIVAL_PRICE = "arrival_price"
    CLOSING_PRICE = "closing_price"
    MARKET_IMPACT = "market_impact"
    EFFECTIVE_SPREAD = "effective_spread"
    PRICE_IMPROVEMENT = "price_improvement"
    FILL_RATE = "fill_rate"
    SPEED_OF_EXECUTION = "speed_of_execution"

class RegulatoryStandard(Enum):
    MiFID_II = "mifid_ii"
    REG_NMS = "reg_nms"
    BEST_EXECUTION_POLICY = "best_execution_policy"
    FINRA_RULE_5310 = "finra_rule_5310"
    SEC_RULE_606 = "sec_rule_606"
    ESMA_RTS_27 = "esma_rts_27"
    CSRC_BEST_EXECUTION = "csrc_best_execution"

@dataclass
class VenueData:
    """交易场所数据"""
    venue: ExecutionVenue
    venue_name: str
    market_type: str
    operating_hours: Dict[str, Any]
    min_order_size: float
    max_order_size: float
    tick_size: float
    fees: Dict[str, float]
    liquidity_metrics: Dict[str, float]
    historical_performance: Dict[str, float]
    regulatory_status: str
    connectivity: str
    latency_ms: float
    fill_rate: float
    market_share: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExecutionMetrics:
    """执行质量指标"""
    execution_id: str
    trade_id: str
    symbol: str
    venue: ExecutionVenue
    execution_time: datetime
    benchmark_price: float
    executed_price: float
    benchmark_type: str
    price_improvement: float
    effective_spread: float
    market_impact: float
    implementation_shortfall: float
    vwap_performance: float
    twap_performance: float
    arrival_price_performance: float
    closing_price_performance: float
    slippage: float
    timing_cost: float
    opportunity_cost: float
    fill_rate: float
    speed_to_fill: float
    partial_fills: int
    total_cost: float
    execution_quality: ExecutionQuality
    regulatory_compliance: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BestExecutionAnalysis:
    """最佳执行分析"""
    analysis_id: str
    trade_id: str
    symbol: str
    analysis_date: datetime
    analysis_method: BestExecutionMethod
    regulatory_standard: RegulatoryStandard
    executed_venue: ExecutionVenue
    alternative_venues: List[ExecutionVenue]
    execution_metrics: ExecutionMetrics
    venue_comparison: Dict[str, Any]
    best_execution_achieved: bool
    improvement_opportunities: List[str]
    regulatory_compliance: bool
    compliance_score: float
    recommendations: List[str]
    risk_factors: List[str]
    cost_analysis: Dict[str, float]
    performance_ranking: Dict[str, int]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BestExecutionReport:
    """最佳执行报告"""
    report_id: str
    report_date: datetime
    reporting_period: Tuple[datetime, datetime]
    total_trades: int
    total_volume: float
    total_value: float
    average_execution_quality: float
    compliance_rate: float
    venue_analysis: Dict[str, Any]
    performance_summary: Dict[str, Any]
    regulatory_findings: List[str]
    improvement_recommendations: List[str]
    cost_savings_opportunities: List[str]
    risk_assessment: Dict[str, Any]
    detailed_analyses: List[BestExecutionAnalysis]
    metadata: Dict[str, Any] = field(default_factory=dict)

class BestExecutionVerifier:
    """
    最佳执行验证系统
    
    提供交易执行质量分析、最佳执行验证、监管合规检查和
    执行优化建议的综合分析系统。
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 配置参数
        self.regulatory_standards = config.get('regulatory_standards', [RegulatoryStandard.MiFID_II])
        self.analysis_methods = config.get('analysis_methods', list(BestExecutionMethod))
        self.quality_thresholds = config.get('quality_thresholds', {
            'excellent': 0.95,
            'good': 0.80,
            'adequate': 0.60,
            'poor': 0.40
        })
        
        # 数据存储
        self.venue_data = {}
        self.execution_history = []
        self.benchmark_data = {}
        self.market_data = {}
        
        # 分析配置
        self.analysis_config = {
            'lookback_days': config.get('lookback_days', 30),
            'min_sample_size': config.get('min_sample_size', 10),
            'confidence_level': config.get('confidence_level', 0.95),
            'benchmark_tolerance': config.get('benchmark_tolerance', 0.001)
        }
        
        # 性能统计
        self.performance_stats = {
            'total_analyses': 0,
            'compliance_violations': 0,
            'average_analysis_time': 0.0,
            'cache_hit_rate': 0.0
        }
        
        # 初始化
        self._initialize_venues()
        self._initialize_benchmarks()
        
        self.logger.info("最佳执行验证系统初始化完成")
    
    def _initialize_venues(self):
        """初始化交易场所数据"""
        default_venues = [
            {
                'venue': ExecutionVenue.NYSE,
                'venue_name': 'New York Stock Exchange',
                'market_type': 'exchange',
                'operating_hours': {'open': '09:30', 'close': '16:00', 'timezone': 'EST'},
                'min_order_size': 1.0,
                'max_order_size': 1000000.0,
                'tick_size': 0.01,
                'fees': {'maker': 0.0025, 'taker': 0.0030, 'regulatory': 0.0001},
                'liquidity_metrics': {'depth': 0.85, 'spread': 0.02, 'turnover': 1.2},
                'historical_performance': {'fill_rate': 0.95, 'speed': 0.12, 'cost': 0.025},
                'regulatory_status': 'fully_compliant',
                'connectivity': 'direct',
                'latency_ms': 0.5,
                'fill_rate': 0.95,
                'market_share': 0.25
            },
            {
                'venue': ExecutionVenue.NASDAQ,
                'venue_name': 'NASDAQ Stock Market',
                'market_type': 'exchange',
                'operating_hours': {'open': '09:30', 'close': '16:00', 'timezone': 'EST'},
                'min_order_size': 1.0,
                'max_order_size': 1000000.0,
                'tick_size': 0.01,
                'fees': {'maker': 0.0020, 'taker': 0.0032, 'regulatory': 0.0001},
                'liquidity_metrics': {'depth': 0.82, 'spread': 0.021, 'turnover': 1.15},
                'historical_performance': {'fill_rate': 0.94, 'speed': 0.10, 'cost': 0.026},
                'regulatory_status': 'fully_compliant',
                'connectivity': 'direct',
                'latency_ms': 0.4,
                'fill_rate': 0.94,
                'market_share': 0.20
            },
            {
                'venue': ExecutionVenue.DARK_POOL,
                'venue_name': 'Dark Pool Network',
                'market_type': 'dark_pool',
                'operating_hours': {'open': '09:30', 'close': '16:00', 'timezone': 'EST'},
                'min_order_size': 1000.0,
                'max_order_size': 10000000.0,
                'tick_size': 0.01,
                'fees': {'execution': 0.0015, 'regulatory': 0.0001},
                'liquidity_metrics': {'depth': 0.90, 'spread': 0.015, 'turnover': 0.8},
                'historical_performance': {'fill_rate': 0.88, 'speed': 0.25, 'cost': 0.020},
                'regulatory_status': 'conditional_compliance',
                'connectivity': 'api',
                'latency_ms': 1.2,
                'fill_rate': 0.88,
                'market_share': 0.15
            }
        ]
        
        for venue_config in default_venues:
            venue = VenueData(**venue_config)
            self.venue_data[venue.venue] = venue
    
    def _initialize_benchmarks(self):
        """初始化基准数据"""
        self.benchmark_data = {
            'vwap': {},
            'twap': {},
            'arrival_price': {},
            'closing_price': {},
            'mid_price': {},
            'last_price': {}
        }
    
    async def add_venue(self, venue_data: VenueData):
        """添加交易场所"""
        self.venue_data[venue_data.venue] = venue_data
        self.logger.info(f"已添加交易场所: {venue_data.venue_name}")
    
    async def update_venue_data(self, venue: ExecutionVenue, data: Dict[str, Any]):
        """更新场所数据"""
        if venue in self.venue_data:
            venue_data = self.venue_data[venue]
            for key, value in data.items():
                if hasattr(venue_data, key):
                    setattr(venue_data, key, value)
            self.logger.debug(f"已更新场所数据: {venue.value}")
    
    async def analyze_execution(self, trade_data: Dict[str, Any]) -> BestExecutionAnalysis:
        """分析单笔交易的执行质量"""
        analysis_id = f"exec_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # 提取交易信息
        trade_id = trade_data.get('trade_id')
        symbol = trade_data.get('symbol')
        executed_venue = ExecutionVenue(trade_data.get('venue', 'nyse'))
        executed_price = trade_data.get('executed_price')
        execution_time = trade_data.get('execution_time', datetime.now())
        quantity = trade_data.get('quantity')
        side = trade_data.get('side')
        
        # 获取基准价格
        benchmark_data = await self._get_benchmark_prices(symbol, execution_time)
        
        # 计算执行指标
        execution_metrics = await self._calculate_execution_metrics(
            trade_data, benchmark_data
        )
        
        # 获取替代场所数据
        alternative_venues = await self._get_alternative_venues(symbol, executed_venue)
        
        # 场所比较分析
        venue_comparison = await self._compare_venues(
            symbol, execution_time, quantity, side, executed_venue, alternative_venues
        )
        
        # 判断是否达到最佳执行
        best_execution_achieved = await self._evaluate_best_execution(
            execution_metrics, venue_comparison
        )
        
        # 监管合规检查
        regulatory_compliance = await self._check_regulatory_compliance(
            trade_data, execution_metrics
        )
        
        # 计算合规得分
        compliance_score = await self._calculate_compliance_score(
            execution_metrics, regulatory_compliance
        )
        
        # 生成改进建议
        improvement_opportunities = await self._generate_improvement_opportunities(
            execution_metrics, venue_comparison
        )
        
        # 生成推荐
        recommendations = await self._generate_recommendations(
            trade_data, execution_metrics, venue_comparison
        )
        
        # 风险因素分析
        risk_factors = await self._analyze_risk_factors(
            trade_data, execution_metrics, venue_comparison
        )
        
        # 成本分析
        cost_analysis = await self._analyze_costs(
            trade_data, execution_metrics, venue_comparison
        )
        
        # 绩效排名
        performance_ranking = await self._rank_venue_performance(
            venue_comparison
        )
        
        # 创建分析结果
        analysis = BestExecutionAnalysis(
            analysis_id=analysis_id,
            trade_id=trade_id,
            symbol=symbol,
            analysis_date=datetime.now(),
            analysis_method=BestExecutionMethod.IMPLEMENTATION_SHORTFALL,
            regulatory_standard=self.regulatory_standards[0],
            executed_venue=executed_venue,
            alternative_venues=alternative_venues,
            execution_metrics=execution_metrics,
            venue_comparison=venue_comparison,
            best_execution_achieved=best_execution_achieved,
            improvement_opportunities=improvement_opportunities,
            regulatory_compliance=regulatory_compliance,
            compliance_score=compliance_score,
            recommendations=recommendations,
            risk_factors=risk_factors,
            cost_analysis=cost_analysis,
            performance_ranking=performance_ranking
        )
        
        # 更新性能统计
        self.performance_stats['total_analyses'] += 1
        if not regulatory_compliance:
            self.performance_stats['compliance_violations'] += 1
        
        self.logger.info(f"完成执行分析: {analysis_id}")
        return analysis
    
    async def _get_benchmark_prices(self, symbol: str, execution_time: datetime) -> Dict[str, float]:
        """获取基准价格"""
        # 模拟基准价格数据 - 实际应用中从市场数据获取
        base_price = 100.0 + np.random.normal(0, 5)
        
        benchmark_prices = {
            'vwap': base_price + np.random.normal(0, 0.05),
            'twap': base_price + np.random.normal(0, 0.03),
            'arrival_price': base_price + np.random.normal(0, 0.02),
            'closing_price': base_price + np.random.normal(0, 0.04),
            'mid_price': base_price + np.random.normal(0, 0.01),
            'last_price': base_price + np.random.normal(0, 0.02),
            'open_price': base_price + np.random.normal(0, 0.03)
        }
        
        return benchmark_prices
    
    async def _calculate_execution_metrics(self, trade_data: Dict[str, Any], benchmark_data: Dict[str, float]) -> ExecutionMetrics:
        """计算执行指标"""
        executed_price = trade_data.get('executed_price')
        quantity = trade_data.get('quantity')
        side = trade_data.get('side')
        execution_time = trade_data.get('execution_time', datetime.now())
        
        # 计算价格改善
        mid_price = benchmark_data['mid_price']
        price_improvement = 0.0
        if side.upper() == 'BUY':
            price_improvement = max(0, mid_price - executed_price)
        else:
            price_improvement = max(0, executed_price - mid_price)
        
        # 计算有效价差
        effective_spread = abs(executed_price - mid_price) * 2
        
        # 计算市场冲击
        market_impact = abs(executed_price - benchmark_data['arrival_price'])
        
        # 计算实施缺口
        implementation_shortfall = (executed_price - benchmark_data['arrival_price']) * quantity
        if side.upper() == 'SELL':
            implementation_shortfall = -implementation_shortfall
        
        # 计算VWAP绩效
        vwap_performance = (executed_price - benchmark_data['vwap']) / benchmark_data['vwap']
        if side.upper() == 'SELL':
            vwap_performance = -vwap_performance
        
        # 计算TWAP绩效
        twap_performance = (executed_price - benchmark_data['twap']) / benchmark_data['twap']
        if side.upper() == 'SELL':
            twap_performance = -twap_performance
        
        # 计算到达价格绩效
        arrival_price_performance = (executed_price - benchmark_data['arrival_price']) / benchmark_data['arrival_price']
        if side.upper() == 'SELL':
            arrival_price_performance = -arrival_price_performance
        
        # 计算收盘价绩效
        closing_price_performance = (executed_price - benchmark_data['closing_price']) / benchmark_data['closing_price']
        if side.upper() == 'SELL':
            closing_price_performance = -closing_price_performance
        
        # 计算滑点
        slippage = abs(executed_price - benchmark_data['arrival_price']) / benchmark_data['arrival_price']
        
        # 计算时间成本
        timing_cost = np.random.uniform(0.0005, 0.002)
        
        # 计算机会成本
        opportunity_cost = np.random.uniform(0.001, 0.003)
        
        # 计算成交率
        fill_rate = np.random.uniform(0.85, 0.98)
        
        # 计算成交速度
        speed_to_fill = np.random.uniform(0.1, 2.0)
        
        # 计算部分成交次数
        partial_fills = np.random.randint(1, 5)
        
        # 计算总成本
        total_cost = market_impact + timing_cost + opportunity_cost
        
        # 评估执行质量
        quality_score = 1.0 - (slippage + effective_spread/mid_price + timing_cost)
        if quality_score >= self.quality_thresholds['excellent']:
            execution_quality = ExecutionQuality.EXCELLENT
        elif quality_score >= self.quality_thresholds['good']:
            execution_quality = ExecutionQuality.GOOD
        elif quality_score >= self.quality_thresholds['adequate']:
            execution_quality = ExecutionQuality.ADEQUATE
        else:
            execution_quality = ExecutionQuality.POOR
        
        # 监管合规检查
        regulatory_compliance = slippage < 0.01 and effective_spread < 0.05
        
        metrics = ExecutionMetrics(
            execution_id=trade_data.get('execution_id', f"exec_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"),
            trade_id=trade_data.get('trade_id'),
            symbol=trade_data.get('symbol'),
            venue=ExecutionVenue(trade_data.get('venue', 'nyse')),
            execution_time=execution_time,
            benchmark_price=benchmark_data['arrival_price'],
            executed_price=executed_price,
            benchmark_type='arrival_price',
            price_improvement=price_improvement,
            effective_spread=effective_spread,
            market_impact=market_impact,
            implementation_shortfall=implementation_shortfall,
            vwap_performance=vwap_performance,
            twap_performance=twap_performance,
            arrival_price_performance=arrival_price_performance,
            closing_price_performance=closing_price_performance,
            slippage=slippage,
            timing_cost=timing_cost,
            opportunity_cost=opportunity_cost,
            fill_rate=fill_rate,
            speed_to_fill=speed_to_fill,
            partial_fills=partial_fills,
            total_cost=total_cost,
            execution_quality=execution_quality,
            regulatory_compliance=regulatory_compliance
        )
        
        return metrics
    
    async def _get_alternative_venues(self, symbol: str, executed_venue: ExecutionVenue) -> List[ExecutionVenue]:
        """获取替代交易场所"""
        all_venues = list(self.venue_data.keys())
        alternative_venues = [v for v in all_venues if v != executed_venue]
        
        # 根据股票类型和流动性筛选合适的场所
        suitable_venues = []
        for venue in alternative_venues:
            venue_data = self.venue_data[venue]
            if venue_data.liquidity_metrics.get('depth', 0) > 0.5:
                suitable_venues.append(venue)
        
        return suitable_venues[:5]  # 返回最多5个替代场所
    
    async def _compare_venues(self, symbol: str, execution_time: datetime, quantity: float, 
                            side: str, executed_venue: ExecutionVenue, 
                            alternative_venues: List[ExecutionVenue]) -> Dict[str, Any]:
        """比较交易场所"""
        venue_comparison = {}
        
        # 分析执行场所
        executed_venue_data = self.venue_data[executed_venue]
        venue_comparison[executed_venue.value] = {
            'venue_name': executed_venue_data.venue_name,
            'expected_cost': executed_venue_data.fees.get('taker', 0.003),
            'expected_fill_rate': executed_venue_data.fill_rate,
            'expected_speed': executed_venue_data.latency_ms,
            'liquidity_score': executed_venue_data.liquidity_metrics.get('depth', 0.8),
            'market_share': executed_venue_data.market_share,
            'regulatory_status': executed_venue_data.regulatory_status,
            'suitability_score': 0.85
        }
        
        # 分析替代场所
        for venue in alternative_venues:
            venue_data = self.venue_data[venue]
            
            # 计算适合度评分
            suitability_score = self._calculate_suitability_score(
                venue_data, symbol, quantity, side
            )
            
            venue_comparison[venue.value] = {
                'venue_name': venue_data.venue_name,
                'expected_cost': venue_data.fees.get('taker', 0.003),
                'expected_fill_rate': venue_data.fill_rate,
                'expected_speed': venue_data.latency_ms,
                'liquidity_score': venue_data.liquidity_metrics.get('depth', 0.8),
                'market_share': venue_data.market_share,
                'regulatory_status': venue_data.regulatory_status,
                'suitability_score': suitability_score
            }
        
        return venue_comparison
    
    def _calculate_suitability_score(self, venue_data: VenueData, symbol: str, 
                                   quantity: float, side: str) -> float:
        """计算场所适合度评分"""
        score = 0.0
        
        # 流动性评分 (40%)
        liquidity_score = venue_data.liquidity_metrics.get('depth', 0.8)
        score += liquidity_score * 0.4
        
        # 成本评分 (30%)
        cost = venue_data.fees.get('taker', 0.003)
        cost_score = max(0, 1 - cost / 0.005)  # 假设最高成本为0.5%
        score += cost_score * 0.3
        
        # 成交率评分 (20%)
        fill_rate_score = venue_data.fill_rate
        score += fill_rate_score * 0.2
        
        # 速度评分 (10%)
        speed_score = max(0, 1 - venue_data.latency_ms / 5.0)  # 假设最慢5ms
        score += speed_score * 0.1
        
        return min(1.0, score)
    
    async def _evaluate_best_execution(self, execution_metrics: ExecutionMetrics, 
                                     venue_comparison: Dict[str, Any]) -> bool:
        """评估是否达到最佳执行"""
        executed_venue = execution_metrics.venue.value
        
        if executed_venue not in venue_comparison:
            return False
        
        executed_venue_data = venue_comparison[executed_venue]
        
        # 检查是否有明显更好的替代场所
        for venue, data in venue_comparison.items():
            if venue != executed_venue:
                # 成本比较
                if data['expected_cost'] < executed_venue_data['expected_cost'] * 0.9:
                    return False
                
                # 成交率比较
                if data['expected_fill_rate'] > executed_venue_data['expected_fill_rate'] * 1.05:
                    return False
                
                # 适合度比较
                if data['suitability_score'] > executed_venue_data['suitability_score'] * 1.1:
                    return False
        
        # 检查执行质量
        if execution_metrics.execution_quality in [ExecutionQuality.POOR, ExecutionQuality.FAILED]:
            return False
        
        # 检查监管合规
        if not execution_metrics.regulatory_compliance:
            return False
        
        return True
    
    async def _check_regulatory_compliance(self, trade_data: Dict[str, Any], 
                                         execution_metrics: ExecutionMetrics) -> bool:
        """检查监管合规"""
        compliance_checks = []
        
        # MiFID II 合规检查
        if RegulatoryStandard.MiFID_II in self.regulatory_standards:
            # 价格改善检查
            if execution_metrics.price_improvement >= 0:
                compliance_checks.append(True)
            else:
                compliance_checks.append(False)
            
            # 有效价差检查
            if execution_metrics.effective_spread <= 0.05:
                compliance_checks.append(True)
            else:
                compliance_checks.append(False)
            
            # 成交率检查
            if execution_metrics.fill_rate >= 0.85:
                compliance_checks.append(True)
            else:
                compliance_checks.append(False)
        
        # Reg NMS 合规检查
        if RegulatoryStandard.REG_NMS in self.regulatory_standards:
            # 最佳报价检查
            if execution_metrics.slippage <= 0.01:
                compliance_checks.append(True)
            else:
                compliance_checks.append(False)
            
            # 市场冲击检查
            if execution_metrics.market_impact <= 0.02:
                compliance_checks.append(True)
            else:
                compliance_checks.append(False)
        
        # 返回整体合规状态
        return all(compliance_checks) if compliance_checks else True
    
    async def _calculate_compliance_score(self, execution_metrics: ExecutionMetrics, 
                                        regulatory_compliance: bool) -> float:
        """计算合规得分"""
        score = 0.0
        
        # 基础合规得分
        if regulatory_compliance:
            score += 0.5
        
        # 执行质量得分
        quality_scores = {
            ExecutionQuality.EXCELLENT: 0.3,
            ExecutionQuality.GOOD: 0.25,
            ExecutionQuality.ADEQUATE: 0.15,
            ExecutionQuality.POOR: 0.05,
            ExecutionQuality.FAILED: 0.0
        }
        score += quality_scores.get(execution_metrics.execution_quality, 0.0)
        
        # 成本效率得分
        if execution_metrics.total_cost <= 0.01:
            score += 0.2
        elif execution_metrics.total_cost <= 0.02:
            score += 0.15
        elif execution_metrics.total_cost <= 0.03:
            score += 0.1
        else:
            score += 0.05
        
        return min(1.0, score)
    
    async def _generate_improvement_opportunities(self, execution_metrics: ExecutionMetrics, 
                                                venue_comparison: Dict[str, Any]) -> List[str]:
        """生成改进机会"""
        opportunities = []
        
        # 成本改进
        if execution_metrics.total_cost > 0.02:
            opportunities.append("考虑使用成本更低的交易场所")
        
        # 成交率改进
        if execution_metrics.fill_rate < 0.9:
            opportunities.append("优化订单规模以提高成交率")
        
        # 速度改进
        if execution_metrics.speed_to_fill > 1.0:
            opportunities.append("考虑使用延迟更低的交易场所")
        
        # 价格改进
        if execution_metrics.price_improvement <= 0:
            opportunities.append("寻找价格改善机会")
        
        # 滑点改进
        if execution_metrics.slippage > 0.005:
            opportunities.append("优化订单时间以减少滑点")
        
        # 场所选择改进
        executed_venue = execution_metrics.venue.value
        if executed_venue in venue_comparison:
            executed_suitability = venue_comparison[executed_venue]['suitability_score']
            for venue, data in venue_comparison.items():
                if venue != executed_venue and data['suitability_score'] > executed_suitability * 1.1:
                    opportunities.append(f"考虑使用 {data['venue_name']} 以获得更好的执行质量")
        
        return opportunities
    
    async def _generate_recommendations(self, trade_data: Dict[str, Any], 
                                      execution_metrics: ExecutionMetrics, 
                                      venue_comparison: Dict[str, Any]) -> List[str]:
        """生成推荐建议"""
        recommendations = []
        
        # 基于执行质量的建议
        if execution_metrics.execution_quality == ExecutionQuality.POOR:
            recommendations.append("建议重新评估交易策略和场所选择")
        elif execution_metrics.execution_quality == ExecutionQuality.ADEQUATE:
            recommendations.append("寻找优化执行质量的机会")
        
        # 基于成本的建议
        if execution_metrics.total_cost > 0.03:
            recommendations.append("实施成本控制措施")
        
        # 基于合规的建议
        if not execution_metrics.regulatory_compliance:
            recommendations.append("加强监管合规控制")
        
        # 基于场所比较的建议
        best_venue = max(venue_comparison.items(), key=lambda x: x[1]['suitability_score'])
        if best_venue[0] != execution_metrics.venue.value:
            recommendations.append(f"考虑优先使用 {best_venue[1]['venue_name']} 进行类似交易")
        
        return recommendations
    
    async def _analyze_risk_factors(self, trade_data: Dict[str, Any], 
                                  execution_metrics: ExecutionMetrics, 
                                  venue_comparison: Dict[str, Any]) -> List[str]:
        """分析风险因素"""
        risk_factors = []
        
        # 流动性风险
        if execution_metrics.fill_rate < 0.85:
            risk_factors.append("流动性风险：成交率较低")
        
        # 成本风险
        if execution_metrics.total_cost > 0.025:
            risk_factors.append("成本风险：交易成本过高")
        
        # 市场冲击风险
        if execution_metrics.market_impact > 0.02:
            risk_factors.append("市场冲击风险：对市场价格影响较大")
        
        # 合规风险
        if not execution_metrics.regulatory_compliance:
            risk_factors.append("合规风险：不符合监管要求")
        
        # 场所集中度风险
        executed_venue = execution_metrics.venue.value
        if executed_venue in venue_comparison:
            market_share = venue_comparison[executed_venue]['market_share']
            if market_share < 0.1:
                risk_factors.append("场所风险：使用小众交易场所")
        
        return risk_factors
    
    async def _analyze_costs(self, trade_data: Dict[str, Any], 
                           execution_metrics: ExecutionMetrics, 
                           venue_comparison: Dict[str, Any]) -> Dict[str, float]:
        """分析成本"""
        cost_analysis = {
            'explicit_costs': 0.0,
            'implicit_costs': 0.0,
            'total_costs': execution_metrics.total_cost,
            'market_impact': execution_metrics.market_impact,
            'timing_cost': execution_metrics.timing_cost,
            'opportunity_cost': execution_metrics.opportunity_cost,
            'commission': 0.0,
            'fees': 0.0,
            'spread_cost': execution_metrics.effective_spread / 2,
            'slippage_cost': execution_metrics.slippage
        }
        
        # 计算显性成本
        executed_venue = execution_metrics.venue.value
        if executed_venue in venue_comparison:
            venue_data = venue_comparison[executed_venue]
            cost_analysis['commission'] = venue_data.get('expected_cost', 0.003)
            cost_analysis['fees'] = venue_data.get('expected_cost', 0.003) * 0.1
        
        cost_analysis['explicit_costs'] = cost_analysis['commission'] + cost_analysis['fees']
        
        # 计算隐性成本
        cost_analysis['implicit_costs'] = (
            cost_analysis['market_impact'] + 
            cost_analysis['timing_cost'] + 
            cost_analysis['spread_cost'] + 
            cost_analysis['slippage_cost']
        )
        
        return cost_analysis
    
    async def _rank_venue_performance(self, venue_comparison: Dict[str, Any]) -> Dict[str, int]:
        """场所绩效排名"""
        # 计算综合评分
        venue_scores = {}
        for venue, data in venue_comparison.items():
            score = (
                data['suitability_score'] * 0.4 +
                (1 - data['expected_cost'] / 0.005) * 0.3 +
                data['expected_fill_rate'] * 0.2 +
                data['liquidity_score'] * 0.1
            )
            venue_scores[venue] = score
        
        # 排名
        sorted_venues = sorted(venue_scores.items(), key=lambda x: x[1], reverse=True)
        ranking = {venue: rank + 1 for rank, (venue, score) in enumerate(sorted_venues)}
        
        return ranking
    
    async def generate_best_execution_report(self, 
                                           analysis_results: List[BestExecutionAnalysis],
                                           reporting_period: Tuple[datetime, datetime]) -> BestExecutionReport:
        """生成最佳执行报告"""
        report_id = f"best_exec_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 统计信息
        total_trades = len(analysis_results)
        total_volume = sum(float(a.metadata.get('quantity', 0)) for a in analysis_results)
        total_value = sum(float(a.metadata.get('value', 0)) for a in analysis_results)
        
        # 平均执行质量
        quality_scores = []
        for analysis in analysis_results:
            if analysis.execution_metrics.execution_quality == ExecutionQuality.EXCELLENT:
                quality_scores.append(1.0)
            elif analysis.execution_metrics.execution_quality == ExecutionQuality.GOOD:
                quality_scores.append(0.8)
            elif analysis.execution_metrics.execution_quality == ExecutionQuality.ADEQUATE:
                quality_scores.append(0.6)
            elif analysis.execution_metrics.execution_quality == ExecutionQuality.POOR:
                quality_scores.append(0.4)
            else:
                quality_scores.append(0.2)
        
        average_execution_quality = np.mean(quality_scores) if quality_scores else 0.0
        
        # 合规率
        compliant_trades = sum(1 for a in analysis_results if a.regulatory_compliance)
        compliance_rate = compliant_trades / total_trades if total_trades > 0 else 0.0
        
        # 场所分析
        venue_analysis = await self._analyze_venue_performance(analysis_results)
        
        # 绩效总结
        performance_summary = await self._summarize_performance(analysis_results)
        
        # 监管发现
        regulatory_findings = await self._identify_regulatory_findings(analysis_results)
        
        # 改进建议
        improvement_recommendations = await self._compile_improvement_recommendations(analysis_results)
        
        # 成本节约机会
        cost_savings_opportunities = await self._identify_cost_savings_opportunities(analysis_results)
        
        # 风险评估
        risk_assessment = await self._assess_overall_risk(analysis_results)
        
        report = BestExecutionReport(
            report_id=report_id,
            report_date=datetime.now(),
            reporting_period=reporting_period,
            total_trades=total_trades,
            total_volume=total_volume,
            total_value=total_value,
            average_execution_quality=average_execution_quality,
            compliance_rate=compliance_rate,
            venue_analysis=venue_analysis,
            performance_summary=performance_summary,
            regulatory_findings=regulatory_findings,
            improvement_recommendations=improvement_recommendations,
            cost_savings_opportunities=cost_savings_opportunities,
            risk_assessment=risk_assessment,
            detailed_analyses=analysis_results
        )
        
        self.logger.info(f"生成最佳执行报告: {report_id}")
        return report
    
    async def _analyze_venue_performance(self, analysis_results: List[BestExecutionAnalysis]) -> Dict[str, Any]:
        """分析场所绩效"""
        venue_stats = {}
        
        for analysis in analysis_results:
            venue = analysis.executed_venue.value
            
            if venue not in venue_stats:
                venue_stats[venue] = {
                    'trade_count': 0,
                    'total_volume': 0.0,
                    'quality_scores': [],
                    'compliance_count': 0,
                    'cost_total': 0.0,
                    'best_execution_count': 0
                }
            
            stats = venue_stats[venue]
            stats['trade_count'] += 1
            stats['total_volume'] += float(analysis.metadata.get('quantity', 0))
            
            # 质量评分
            if analysis.execution_metrics.execution_quality == ExecutionQuality.EXCELLENT:
                stats['quality_scores'].append(1.0)
            elif analysis.execution_metrics.execution_quality == ExecutionQuality.GOOD:
                stats['quality_scores'].append(0.8)
            elif analysis.execution_metrics.execution_quality == ExecutionQuality.ADEQUATE:
                stats['quality_scores'].append(0.6)
            else:
                stats['quality_scores'].append(0.4)
            
            # 合规统计
            if analysis.regulatory_compliance:
                stats['compliance_count'] += 1
            
            # 成本统计
            stats['cost_total'] += analysis.execution_metrics.total_cost
            
            # 最佳执行统计
            if analysis.best_execution_achieved:
                stats['best_execution_count'] += 1
        
        # 计算汇总指标
        venue_analysis = {}
        for venue, stats in venue_stats.items():
            venue_analysis[venue] = {
                'trade_count': stats['trade_count'],
                'volume_share': stats['total_volume'] / sum(s['total_volume'] for s in venue_stats.values()),
                'average_quality': np.mean(stats['quality_scores']),
                'compliance_rate': stats['compliance_count'] / stats['trade_count'],
                'average_cost': stats['cost_total'] / stats['trade_count'],
                'best_execution_rate': stats['best_execution_count'] / stats['trade_count']
            }
        
        return venue_analysis
    
    async def _summarize_performance(self, analysis_results: List[BestExecutionAnalysis]) -> Dict[str, Any]:
        """总结绩效"""
        if not analysis_results:
            return {}
        
        # 收集所有指标
        costs = [a.execution_metrics.total_cost for a in analysis_results]
        slippages = [a.execution_metrics.slippage for a in analysis_results]
        fill_rates = [a.execution_metrics.fill_rate for a in analysis_results]
        price_improvements = [a.execution_metrics.price_improvement for a in analysis_results]
        
        performance_summary = {
            'average_cost': np.mean(costs),
            'median_cost': np.median(costs),
            'cost_percentile_95': np.percentile(costs, 95),
            'average_slippage': np.mean(slippages),
            'median_slippage': np.median(slippages),
            'slippage_percentile_95': np.percentile(slippages, 95),
            'average_fill_rate': np.mean(fill_rates),
            'median_fill_rate': np.median(fill_rates),
            'fill_rate_percentile_5': np.percentile(fill_rates, 5),
            'average_price_improvement': np.mean(price_improvements),
            'total_price_improvement': sum(price_improvements),
            'positive_price_improvement_rate': sum(1 for p in price_improvements if p > 0) / len(price_improvements)
        }
        
        return performance_summary
    
    async def _identify_regulatory_findings(self, analysis_results: List[BestExecutionAnalysis]) -> List[str]:
        """识别监管发现"""
        findings = []
        
        # 合规率检查
        total_trades = len(analysis_results)
        compliant_trades = sum(1 for a in analysis_results if a.regulatory_compliance)
        compliance_rate = compliant_trades / total_trades if total_trades > 0 else 0.0
        
        if compliance_rate < 0.95:
            findings.append(f"合规率 ({compliance_rate:.1%}) 低于预期标准 (95%)")
        
        # 最佳执行率检查
        best_execution_count = sum(1 for a in analysis_results if a.best_execution_achieved)
        best_execution_rate = best_execution_count / total_trades if total_trades > 0 else 0.0
        
        if best_execution_rate < 0.90:
            findings.append(f"最佳执行率 ({best_execution_rate:.1%}) 低于预期标准 (90%)")
        
        # 成本异常检查
        costs = [a.execution_metrics.total_cost for a in analysis_results]
        if costs:
            high_cost_trades = sum(1 for c in costs if c > 0.05)
            if high_cost_trades > 0:
                findings.append(f"发现 {high_cost_trades} 笔高成本交易 (>5%)")
        
        # 滑点异常检查
        slippages = [a.execution_metrics.slippage for a in analysis_results]
        if slippages:
            high_slippage_trades = sum(1 for s in slippages if s > 0.02)
            if high_slippage_trades > 0:
                findings.append(f"发现 {high_slippage_trades} 笔高滑点交易 (>2%)")
        
        return findings
    
    async def _compile_improvement_recommendations(self, analysis_results: List[BestExecutionAnalysis]) -> List[str]:
        """汇编改进建议"""
        all_recommendations = []
        for analysis in analysis_results:
            all_recommendations.extend(analysis.recommendations)
        
        # 统计建议频率
        recommendation_counts = {}
        for rec in all_recommendations:
            recommendation_counts[rec] = recommendation_counts.get(rec, 0) + 1
        
        # 按频率排序并去重
        sorted_recommendations = sorted(
            recommendation_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # 返回前10个最常见的建议
        return [rec for rec, count in sorted_recommendations[:10]]
    
    async def _identify_cost_savings_opportunities(self, analysis_results: List[BestExecutionAnalysis]) -> List[str]:
        """识别成本节约机会"""
        opportunities = []
        
        # 分析成本数据
        cost_data = []
        for analysis in analysis_results:
            cost_data.append({
                'venue': analysis.executed_venue.value,
                'cost': analysis.execution_metrics.total_cost,
                'volume': float(analysis.metadata.get('quantity', 0))
            })
        
        if not cost_data:
            return opportunities
        
        # 按场所分组分析
        venue_costs = {}
        for data in cost_data:
            venue = data['venue']
            if venue not in venue_costs:
                venue_costs[venue] = []
            venue_costs[venue].append(data['cost'])
        
        # 找出成本最低的场所
        avg_costs = {}
        for venue, costs in venue_costs.items():
            avg_costs[venue] = np.mean(costs)
        
        if avg_costs:
            best_venue = min(avg_costs, key=avg_costs.get)
            worst_venue = max(avg_costs, key=avg_costs.get)
            
            cost_difference = avg_costs[worst_venue] - avg_costs[best_venue]
            if cost_difference > 0.005:  # 0.5% 差异
                opportunities.append(f"将交易从 {worst_venue} 转移到 {best_venue} 可节约 {cost_difference:.3%} 成本")
        
        # 分析高成本交易
        high_cost_trades = [d for d in cost_data if d['cost'] > 0.03]
        if high_cost_trades:
            opportunities.append(f"优化 {len(high_cost_trades)} 笔高成本交易可显著降低整体成本")
        
        return opportunities
    
    async def _assess_overall_risk(self, analysis_results: List[BestExecutionAnalysis]) -> Dict[str, Any]:
        """评估整体风险"""
        risk_assessment = {
            'overall_risk_level': 'low',
            'key_risks': [],
            'risk_metrics': {},
            'mitigation_strategies': []
        }
        
        if not analysis_results:
            return risk_assessment
        
        # 合规风险
        non_compliant_count = sum(1 for a in analysis_results if not a.regulatory_compliance)
        compliance_risk = non_compliant_count / len(analysis_results)
        risk_assessment['risk_metrics']['compliance_risk'] = compliance_risk
        
        if compliance_risk > 0.1:
            risk_assessment['key_risks'].append('监管合规风险')
            risk_assessment['mitigation_strategies'].append('加强合规监控和培训')
        
        # 成本风险
        costs = [a.execution_metrics.total_cost for a in analysis_results]
        cost_volatility = np.std(costs) if costs else 0.0
        risk_assessment['risk_metrics']['cost_volatility'] = cost_volatility
        
        if cost_volatility > 0.01:
            risk_assessment['key_risks'].append('成本波动风险')
            risk_assessment['mitigation_strategies'].append('建立成本控制机制')
        
        # 流动性风险
        fill_rates = [a.execution_metrics.fill_rate for a in analysis_results]
        low_fill_rate_count = sum(1 for fr in fill_rates if fr < 0.85)
        liquidity_risk = low_fill_rate_count / len(analysis_results)
        risk_assessment['risk_metrics']['liquidity_risk'] = liquidity_risk
        
        if liquidity_risk > 0.1:
            risk_assessment['key_risks'].append('流动性风险')
            risk_assessment['mitigation_strategies'].append('分散交易场所和时间')
        
        # 确定整体风险等级
        high_risk_count = sum(1 for risk in [compliance_risk, liquidity_risk] if risk > 0.15)
        medium_risk_count = sum(1 for risk in [compliance_risk, liquidity_risk] if 0.05 < risk <= 0.15)
        
        if high_risk_count > 0:
            risk_assessment['overall_risk_level'] = 'high'
        elif medium_risk_count > 0:
            risk_assessment['overall_risk_level'] = 'medium'
        else:
            risk_assessment['overall_risk_level'] = 'low'
        
        return risk_assessment
    
    async def get_performance_statistics(self) -> Dict[str, Any]:
        """获取性能统计"""
        return {
            'total_analyses': self.performance_stats['total_analyses'],
            'compliance_violations': self.performance_stats['compliance_violations'],
            'compliance_rate': 1 - (self.performance_stats['compliance_violations'] / max(1, self.performance_stats['total_analyses'])),
            'average_analysis_time': self.performance_stats['average_analysis_time'],
            'cache_hit_rate': self.performance_stats['cache_hit_rate'],
            'active_venues': len(self.venue_data),
            'supported_standards': len(self.regulatory_standards),
            'supported_methods': len(self.analysis_methods)
        }
    
    async def export_analysis_results(self, analysis_results: List[BestExecutionAnalysis], 
                                    format: str = "csv") -> str:
        """导出分析结果"""
        if format.lower() == "csv":
            data = []
            for analysis in analysis_results:
                data.append({
                    'Analysis ID': analysis.analysis_id,
                    'Trade ID': analysis.trade_id,
                    'Symbol': analysis.symbol,
                    'Venue': analysis.executed_venue.value,
                    'Execution Quality': analysis.execution_metrics.execution_quality.value,
                    'Best Execution': analysis.best_execution_achieved,
                    'Compliance': analysis.regulatory_compliance,
                    'Compliance Score': analysis.compliance_score,
                    'Total Cost': analysis.execution_metrics.total_cost,
                    'Slippage': analysis.execution_metrics.slippage,
                    'Fill Rate': analysis.execution_metrics.fill_rate,
                    'Price Improvement': analysis.execution_metrics.price_improvement
                })
            
            df = pd.DataFrame(data)
            filename = f"best_execution_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(filename, index=False)
            
            self.logger.info(f"分析结果已导出: {filename}")
            return filename
        
        return ""
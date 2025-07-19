import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, lognorm
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

class CostComponent(Enum):
    COMMISSION = "commission"
    SLIPPAGE = "slippage"
    MARKET_IMPACT = "market_impact"
    SPREAD = "spread"
    TIMING_COST = "timing_cost"
    OPPORTUNITY_COST = "opportunity_cost"
    FINANCING_COST = "financing_cost"
    CUSTODY_COST = "custody_cost"
    REGULATORY_COST = "regulatory_cost"
    TAXES = "taxes"

class MarketRegime(Enum):
    NORMAL = "normal"
    HIGH_VOLATILITY = "high_volatility"
    LOW_LIQUIDITY = "low_liquidity"
    CRISIS = "crisis"
    TRENDING = "trending"
    RANGING = "ranging"

class TradingVenue(Enum):
    PRIMARY_EXCHANGE = "primary_exchange"
    DARK_POOL = "dark_pool"
    ECN = "ecn"
    OTC = "otc"
    CROSSING_NETWORK = "crossing_network"
    ALTERNATIVE_VENUE = "alternative_venue"

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    ICEBERG = "iceberg"
    TWAP = "twap"
    VWAP = "vwap"
    IMPLEMENTATION_SHORTFALL = "implementation_shortfall"
    PARTICIPATION_RATE = "participation_rate"

@dataclass
class MarketData:
    """市场数据"""
    symbol: str
    timestamp: datetime
    price: float
    bid: float
    ask: float
    volume: float
    volatility: float
    adv: float  # Average Daily Volume
    market_cap: float
    liquidity_score: float
    spread: float
    depth: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TradeInstruction:
    """交易指令"""
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: float
    urgency: str  # 'LOW', 'MEDIUM', 'HIGH'
    time_horizon: int  # minutes
    participation_rate: float
    order_type: OrderType
    venue_preference: List[TradingVenue]
    risk_tolerance: float
    benchmark_price: float
    constraints: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CostBreakdown:
    """成本分解"""
    commission: float
    slippage: float
    market_impact: float
    spread_cost: float
    timing_cost: float
    opportunity_cost: float
    financing_cost: float
    total_cost: float
    cost_basis_points: float
    explanation: Dict[str, str] = field(default_factory=dict)

@dataclass
class TransactionCostEstimate:
    """交易成本估算"""
    symbol: str
    instruction: TradeInstruction
    expected_cost: float
    cost_range: Tuple[float, float]
    confidence_interval: float
    cost_breakdown: CostBreakdown
    optimal_strategy: str
    execution_time: int
    market_regime: MarketRegime
    venue_analysis: Dict[TradingVenue, float]
    risk_metrics: Dict[str, float]
    sensitivity_analysis: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExecutionResult:
    """执行结果"""
    instruction: TradeInstruction
    actual_cost: float
    predicted_cost: float
    cost_difference: float
    execution_quality: float
    fills: List[Dict[str, Any]]
    timing_analysis: Dict[str, float]
    venue_performance: Dict[TradingVenue, Dict[str, float]]
    lessons_learned: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

class TransactionCostModel:
    """
    先进的交易成本模型
    
    提供多层次的交易成本分析，包括显性成本（佣金、税费）、
    隐性成本（滑点、市场冲击）和机会成本等。
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 模型参数
        self.calibration_period = config.get('calibration_period', 252)
        self.confidence_level = config.get('confidence_level', 0.95)
        self.rebalance_frequency = config.get('rebalance_frequency', 'daily')
        
        # 成本参数
        self.commission_rates = config.get('commission_rates', {})
        self.tax_rates = config.get('tax_rates', {})
        self.financing_rates = config.get('financing_rates', {})
        
        # 市场冲击参数
        self.impact_models = {
            'linear': self._linear_impact_model,
            'square_root': self._square_root_impact_model,
            'concave': self._concave_impact_model,
            'almgren_chriss': self._almgren_chriss_model,
            'kis': self._kis_model
        }
        
        # 滑点模型
        self.slippage_models = {
            'fixed': self._fixed_slippage_model,
            'proportional': self._proportional_slippage_model,
            'adaptive': self._adaptive_slippage_model
        }
        
        # 模型校准数据
        self.calibration_data = {}
        self.model_parameters = {}
        
        # 执行历史
        self.execution_history = []
        self.performance_metrics = {}
        
    async def estimate_transaction_cost(self, 
                                      instruction: TradeInstruction,
                                      market_data: MarketData,
                                      historical_data: pd.DataFrame = None) -> TransactionCostEstimate:
        """估算交易成本"""
        try:
            # 检测市场制度
            market_regime = await self._detect_market_regime(market_data, historical_data)
            
            # 估算各成本组件
            cost_breakdown = await self._estimate_cost_components(
                instruction, market_data, market_regime
            )
            
            # 计算总成本
            total_cost = (
                cost_breakdown.commission +
                cost_breakdown.slippage +
                cost_breakdown.market_impact +
                cost_breakdown.spread_cost +
                cost_breakdown.timing_cost +
                cost_breakdown.opportunity_cost +
                cost_breakdown.financing_cost
            )
            
            # 基点成本
            notional_value = instruction.quantity * market_data.price
            cost_basis_points = (total_cost / notional_value) * 10000 if notional_value > 0 else 0
            
            cost_breakdown.total_cost = total_cost
            cost_breakdown.cost_basis_points = cost_basis_points
            
            # 成本区间估算
            cost_range = await self._estimate_cost_range(
                total_cost, instruction, market_data, market_regime
            )
            
            # 最优执行策略
            optimal_strategy = await self._determine_optimal_strategy(
                instruction, market_data, market_regime
            )
            
            # 交易场所分析
            venue_analysis = await self._analyze_venues(
                instruction, market_data, market_regime
            )
            
            # 风险指标
            risk_metrics = await self._calculate_risk_metrics(
                instruction, market_data, cost_breakdown
            )
            
            # 敏感性分析
            sensitivity_analysis = await self._perform_sensitivity_analysis(
                instruction, market_data, market_regime
            )
            
            return TransactionCostEstimate(
                symbol=instruction.symbol,
                instruction=instruction,
                expected_cost=total_cost,
                cost_range=cost_range,
                confidence_interval=self.confidence_level,
                cost_breakdown=cost_breakdown,
                optimal_strategy=optimal_strategy,
                execution_time=self._estimate_execution_time(instruction, market_data),
                market_regime=market_regime,
                venue_analysis=venue_analysis,
                risk_metrics=risk_metrics,
                sensitivity_analysis=sensitivity_analysis
            )
            
        except Exception as e:
            self.logger.error(f"交易成本估算失败: {e}")
            raise
    
    async def _detect_market_regime(self, market_data: MarketData, 
                                  historical_data: pd.DataFrame = None) -> MarketRegime:
        """检测市场制度"""
        try:
            # 基于波动率的制度检测
            if market_data.volatility > 0.30:
                return MarketRegime.HIGH_VOLATILITY
            elif market_data.liquidity_score < 0.3:
                return MarketRegime.LOW_LIQUIDITY
            elif market_data.spread > market_data.price * 0.01:
                return MarketRegime.CRISIS
            else:
                return MarketRegime.NORMAL
                
        except Exception as e:
            self.logger.warning(f"市场制度检测失败: {e}")
            return MarketRegime.NORMAL
    
    async def _estimate_cost_components(self, 
                                      instruction: TradeInstruction,
                                      market_data: MarketData,
                                      market_regime: MarketRegime) -> CostBreakdown:
        """估算成本组件"""
        try:
            notional_value = instruction.quantity * market_data.price
            
            # 佣金成本
            commission = await self._calculate_commission(instruction, market_data)
            
            # 滑点成本
            slippage = await self._calculate_slippage(instruction, market_data, market_regime)
            
            # 市场冲击成本
            market_impact = await self._calculate_market_impact(instruction, market_data, market_regime)
            
            # 点差成本
            spread_cost = await self._calculate_spread_cost(instruction, market_data)
            
            # 时机成本
            timing_cost = await self._calculate_timing_cost(instruction, market_data)
            
            # 机会成本
            opportunity_cost = await self._calculate_opportunity_cost(instruction, market_data)
            
            # 融资成本
            financing_cost = await self._calculate_financing_cost(instruction, market_data)
            
            return CostBreakdown(
                commission=commission,
                slippage=slippage,
                market_impact=market_impact,
                spread_cost=spread_cost,
                timing_cost=timing_cost,
                opportunity_cost=opportunity_cost,
                financing_cost=financing_cost,
                total_cost=0,  # 稍后计算
                cost_basis_points=0,  # 稍后计算
                explanation={
                    'commission': f'基于{commission/notional_value*10000:.1f}bp佣金率',
                    'slippage': f'基于{market_data.volatility:.1%}波动率的滑点',
                    'market_impact': f'基于{instruction.quantity/market_data.adv:.1%}参与率的市场冲击',
                    'spread_cost': f'基于{market_data.spread/market_data.price*10000:.1f}bp点差',
                    'timing_cost': f'基于{instruction.time_horizon}分钟执行时间',
                    'opportunity_cost': f'基于{instruction.urgency}紧急程度',
                    'financing_cost': f'基于{self.financing_rates.get("default", 0.03):.1%}融资成本率'
                }
            )
            
        except Exception as e:
            self.logger.error(f"成本组件估算失败: {e}")
            raise
    
    async def _calculate_commission(self, instruction: TradeInstruction, 
                                  market_data: MarketData) -> float:
        """计算佣金"""
        notional_value = instruction.quantity * market_data.price
        
        # 默认佣金率
        commission_rate = self.commission_rates.get(instruction.symbol, 0.0005)
        
        # 基于交易量的佣金层级
        if notional_value > 1000000:  # 大额交易
            commission_rate *= 0.8
        elif notional_value > 100000:  # 中额交易
            commission_rate *= 0.9
        
        # 最小佣金
        min_commission = 5.0
        
        commission = max(notional_value * commission_rate, min_commission)
        
        return commission
    
    async def _calculate_slippage(self, instruction: TradeInstruction,
                                market_data: MarketData,
                                market_regime: MarketRegime) -> float:
        """计算滑点"""
        base_slippage = market_data.spread / 2  # 半个点差
        
        # 市场制度调整
        regime_multiplier = {
            MarketRegime.NORMAL: 1.0,
            MarketRegime.HIGH_VOLATILITY: 1.5,
            MarketRegime.LOW_LIQUIDITY: 2.0,
            MarketRegime.CRISIS: 3.0,
            MarketRegime.TRENDING: 0.8,
            MarketRegime.RANGING: 1.2
        }.get(market_regime, 1.0)
        
        # 订单类型调整
        order_type_multiplier = {
            OrderType.MARKET: 1.0,
            OrderType.LIMIT: 0.3,
            OrderType.TWAP: 0.6,
            OrderType.VWAP: 0.7,
            OrderType.IMPLEMENTATION_SHORTFALL: 0.5
        }.get(instruction.order_type, 1.0)
        
        # 紧急程度调整
        urgency_multiplier = {
            'LOW': 0.5,
            'MEDIUM': 1.0,
            'HIGH': 2.0
        }.get(instruction.urgency, 1.0)
        
        slippage = base_slippage * regime_multiplier * order_type_multiplier * urgency_multiplier
        
        return slippage * instruction.quantity
    
    async def _calculate_market_impact(self, instruction: TradeInstruction,
                                     market_data: MarketData,
                                     market_regime: MarketRegime) -> float:
        """计算市场冲击"""
        # 参与率
        participation_rate = instruction.quantity / market_data.adv
        
        # 使用平方根模型
        impact_model = self.config.get('impact_model', 'square_root')
        
        if impact_model in self.impact_models:
            impact = self.impact_models[impact_model](
                participation_rate, market_data, market_regime
            )
        else:
            impact = self._square_root_impact_model(
                participation_rate, market_data, market_regime
            )
        
        return impact * instruction.quantity * market_data.price
    
    def _linear_impact_model(self, participation_rate: float, 
                           market_data: MarketData, 
                           market_regime: MarketRegime) -> float:
        """线性冲击模型"""
        base_impact = 0.001  # 0.1%
        
        regime_multiplier = {
            MarketRegime.NORMAL: 1.0,
            MarketRegime.HIGH_VOLATILITY: 1.5,
            MarketRegime.LOW_LIQUIDITY: 2.0,
            MarketRegime.CRISIS: 3.0
        }.get(market_regime, 1.0)
        
        return base_impact * participation_rate * regime_multiplier
    
    def _square_root_impact_model(self, participation_rate: float,
                                market_data: MarketData,
                                market_regime: MarketRegime) -> float:
        """平方根冲击模型"""
        base_impact = 0.001
        
        regime_multiplier = {
            MarketRegime.NORMAL: 1.0,
            MarketRegime.HIGH_VOLATILITY: 1.5,
            MarketRegime.LOW_LIQUIDITY: 2.0,
            MarketRegime.CRISIS: 3.0
        }.get(market_regime, 1.0)
        
        return base_impact * np.sqrt(participation_rate) * regime_multiplier
    
    def _concave_impact_model(self, participation_rate: float,
                            market_data: MarketData,
                            market_regime: MarketRegime) -> float:
        """凹形冲击模型"""
        base_impact = 0.001
        
        regime_multiplier = {
            MarketRegime.NORMAL: 1.0,
            MarketRegime.HIGH_VOLATILITY: 1.5,
            MarketRegime.LOW_LIQUIDITY: 2.0,
            MarketRegime.CRISIS: 3.0
        }.get(market_regime, 1.0)
        
        return base_impact * (participation_rate ** 0.6) * regime_multiplier
    
    def _almgren_chriss_model(self, participation_rate: float,
                            market_data: MarketData,
                            market_regime: MarketRegime) -> float:
        """Almgren-Chriss模型"""
        # 简化版本
        eta = 0.001  # 临时冲击参数
        gamma = 0.0001  # 永久冲击参数
        
        temporary_impact = eta * participation_rate
        permanent_impact = gamma * participation_rate
        
        return temporary_impact + permanent_impact
    
    def _kis_model(self, participation_rate: float,
                 market_data: MarketData,
                 market_regime: MarketRegime) -> float:
        """KIS (Keep It Simple) 模型"""
        # 基于ITG研究的简化模型
        alpha = 0.6
        beta = 0.8
        
        volatility_factor = market_data.volatility
        liquidity_factor = 1.0 / market_data.liquidity_score
        
        impact = alpha * (participation_rate ** beta) * volatility_factor * liquidity_factor
        
        return impact
    
    async def _calculate_spread_cost(self, instruction: TradeInstruction,
                                   market_data: MarketData) -> float:
        """计算点差成本"""
        if instruction.order_type == OrderType.MARKET:
            # 市价单承担半个点差
            spread_cost = market_data.spread / 2
        elif instruction.order_type == OrderType.LIMIT:
            # 限价单可能节省点差
            spread_cost = -market_data.spread / 4
        else:
            # 算法订单的平均点差成本
            spread_cost = market_data.spread / 3
        
        return spread_cost * instruction.quantity
    
    async def _calculate_timing_cost(self, instruction: TradeInstruction,
                                   market_data: MarketData) -> float:
        """计算时机成本"""
        # 基于执行时间窗口的成本
        time_cost_rate = market_data.volatility / np.sqrt(252 * 24 * 60)  # 分钟级波动率
        
        # 时间成本随执行时间增加
        timing_cost = time_cost_rate * np.sqrt(instruction.time_horizon) * market_data.price
        
        return timing_cost * instruction.quantity
    
    async def _calculate_opportunity_cost(self, instruction: TradeInstruction,
                                        market_data: MarketData) -> float:
        """计算机会成本"""
        # 基于紧急程度的机会成本
        urgency_cost = {
            'LOW': 0.0,
            'MEDIUM': market_data.volatility * 0.1,
            'HIGH': market_data.volatility * 0.3
        }.get(instruction.urgency, 0.0)
        
        opportunity_cost = urgency_cost * market_data.price
        
        return opportunity_cost * instruction.quantity
    
    async def _calculate_financing_cost(self, instruction: TradeInstruction,
                                      market_data: MarketData) -> float:
        """计算融资成本"""
        if instruction.side == 'BUY':
            # 买入需要融资
            financing_rate = self.financing_rates.get(instruction.symbol, 0.03)
            financing_days = instruction.time_horizon / (24 * 60)  # 转换为天
            
            financing_cost = (
                instruction.quantity * market_data.price * 
                financing_rate * financing_days / 365
            )
        else:
            # 卖出产生现金
            financing_cost = 0.0
        
        return financing_cost
    
    async def _estimate_cost_range(self, expected_cost: float,
                                 instruction: TradeInstruction,
                                 market_data: MarketData,
                                 market_regime: MarketRegime) -> Tuple[float, float]:
        """估算成本区间"""
        # 基于市场制度的不确定性
        uncertainty_factor = {
            MarketRegime.NORMAL: 0.2,
            MarketRegime.HIGH_VOLATILITY: 0.5,
            MarketRegime.LOW_LIQUIDITY: 0.8,
            MarketRegime.CRISIS: 1.5,
            MarketRegime.TRENDING: 0.3,
            MarketRegime.RANGING: 0.4
        }.get(market_regime, 0.3)
        
        # 置信区间
        z_score = norm.ppf(1 - (1 - self.confidence_level) / 2)
        margin = expected_cost * uncertainty_factor * z_score
        
        lower_bound = max(0, expected_cost - margin)
        upper_bound = expected_cost + margin
        
        return (lower_bound, upper_bound)
    
    async def _determine_optimal_strategy(self, instruction: TradeInstruction,
                                        market_data: MarketData,
                                        market_regime: MarketRegime) -> str:
        """确定最优执行策略"""
        # 基于市场条件和指令特征选择策略
        
        if instruction.urgency == 'HIGH':
            return 'AGGRESSIVE_MARKET_ORDER'
        elif instruction.urgency == 'LOW' and market_regime == MarketRegime.NORMAL:
            return 'PATIENT_LIMIT_ORDER'
        elif instruction.quantity / market_data.adv > 0.1:
            return 'TWAP_ALGORITHM'
        elif market_regime == MarketRegime.HIGH_VOLATILITY:
            return 'IMPLEMENTATION_SHORTFALL'
        else:
            return 'VWAP_ALGORITHM'
    
    async def _analyze_venues(self, instruction: TradeInstruction,
                            market_data: MarketData,
                            market_regime: MarketRegime) -> Dict[TradingVenue, float]:
        """分析交易场所"""
        venue_costs = {}
        
        # 主要交易所
        venue_costs[TradingVenue.PRIMARY_EXCHANGE] = market_data.spread * 0.5
        
        # 暗池
        venue_costs[TradingVenue.DARK_POOL] = market_data.spread * 0.3
        
        # ECN
        venue_costs[TradingVenue.ECN] = market_data.spread * 0.4
        
        # 场外交易
        venue_costs[TradingVenue.OTC] = market_data.spread * 0.8
        
        return venue_costs
    
    async def _calculate_risk_metrics(self, instruction: TradeInstruction,
                                    market_data: MarketData,
                                    cost_breakdown: CostBreakdown) -> Dict[str, float]:
        """计算风险指标"""
        return {
            'execution_risk': cost_breakdown.market_impact / (instruction.quantity * market_data.price),
            'timing_risk': cost_breakdown.timing_cost / (instruction.quantity * market_data.price),
            'liquidity_risk': 1.0 - market_data.liquidity_score,
            'volatility_risk': market_data.volatility,
            'concentration_risk': instruction.quantity / market_data.adv
        }
    
    async def _perform_sensitivity_analysis(self, instruction: TradeInstruction,
                                          market_data: MarketData,
                                          market_regime: MarketRegime) -> Dict[str, float]:
        """执行敏感性分析"""
        sensitivity = {}
        
        # 基准成本
        base_estimate = await self.estimate_transaction_cost(instruction, market_data)
        base_cost = base_estimate.expected_cost
        
        # 波动率敏感性
        high_vol_data = market_data
        high_vol_data.volatility *= 1.5
        high_vol_estimate = await self.estimate_transaction_cost(instruction, high_vol_data)
        sensitivity['volatility_sensitivity'] = (high_vol_estimate.expected_cost - base_cost) / base_cost
        
        # 流动性敏感性
        low_liq_data = market_data
        low_liq_data.liquidity_score *= 0.5
        low_liq_estimate = await self.estimate_transaction_cost(instruction, low_liq_data)
        sensitivity['liquidity_sensitivity'] = (low_liq_estimate.expected_cost - base_cost) / base_cost
        
        # 点差敏感性
        wide_spread_data = market_data
        wide_spread_data.spread *= 2
        wide_spread_estimate = await self.estimate_transaction_cost(instruction, wide_spread_data)
        sensitivity['spread_sensitivity'] = (wide_spread_estimate.expected_cost - base_cost) / base_cost
        
        return sensitivity
    
    def _estimate_execution_time(self, instruction: TradeInstruction,
                               market_data: MarketData) -> int:
        """估算执行时间"""
        # 基于参与率的执行时间估算
        participation_rate = min(instruction.participation_rate, 0.3)  # 最大30%
        
        if participation_rate > 0:
            execution_time = int(1 / participation_rate)  # 简化估算
        else:
            execution_time = instruction.time_horizon
        
        return min(execution_time, instruction.time_horizon)
    
    async def optimize_execution_strategy(self, instruction: TradeInstruction,
                                        market_data: MarketData,
                                        constraints: Dict[str, Any] = None) -> Dict[str, Any]:
        """优化执行策略"""
        try:
            # 定义优化目标和约束
            def objective(params):
                # params: [participation_rate, time_horizon, urgency_weight]
                modified_instruction = instruction
                modified_instruction.participation_rate = params[0]
                modified_instruction.time_horizon = int(params[1])
                
                # 计算成本（同步版本）
                cost_estimate = asyncio.run(self.estimate_transaction_cost(
                    modified_instruction, market_data
                ))
                
                return cost_estimate.expected_cost
            
            # 约束条件
            constraints_list = []
            
            # 参与率约束
            constraints_list.append({
                'type': 'ineq',
                'fun': lambda x: 0.3 - x[0]  # 参与率不超过30%
            })
            
            # 时间约束
            constraints_list.append({
                'type': 'ineq',
                'fun': lambda x: x[1] - 1  # 至少1分钟
            })
            
            # 初始值
            x0 = [0.1, 60, 0.5]  # 10%参与率，60分钟，中等紧急程度
            
            # 边界
            bounds = [(0.01, 0.30), (1, 480), (0.1, 1.0)]
            
            # 优化
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_list
            )
            
            if result.success:
                optimal_params = result.x
                optimal_cost = result.fun
                
                return {
                    'optimal_participation_rate': optimal_params[0],
                    'optimal_time_horizon': int(optimal_params[1]),
                    'optimal_urgency_weight': optimal_params[2],
                    'optimal_cost': optimal_cost,
                    'optimization_success': True,
                    'savings': instruction.quantity * market_data.price * 0.01 - optimal_cost  # 假设基准成本1%
                }
            else:
                return {
                    'optimization_success': False,
                    'error': result.message
                }
                
        except Exception as e:
            self.logger.error(f"执行策略优化失败: {e}")
            return {'optimization_success': False, 'error': str(e)}
    
    async def calibrate_model(self, execution_data: List[ExecutionResult]):
        """校准模型"""
        try:
            # 收集校准数据
            calibration_records = []
            
            for result in execution_data:
                record = {
                    'predicted_cost': result.predicted_cost,
                    'actual_cost': result.actual_cost,
                    'cost_difference': result.cost_difference,
                    'participation_rate': result.instruction.quantity / 1000000,  # 假设ADV
                    'market_regime': 'normal',  # 简化
                    'volatility': 0.2,  # 假设波动率
                    'liquidity_score': 0.8  # 假设流动性分数
                }
                calibration_records.append(record)
            
            calibration_df = pd.DataFrame(calibration_records)
            
            # 模型校准
            if not calibration_df.empty:
                # 计算预测准确性
                mae = np.mean(np.abs(calibration_df['cost_difference']))
                rmse = np.sqrt(np.mean(calibration_df['cost_difference'] ** 2))
                
                # 更新模型参数
                self.model_parameters['mae'] = mae
                self.model_parameters['rmse'] = rmse
                self.model_parameters['calibration_date'] = datetime.now()
                
                # 保存校准数据
                self.calibration_data = calibration_df
                
                self.logger.info(f"模型校准完成: MAE={mae:.4f}, RMSE={rmse:.4f}")
                
                return {
                    'calibration_success': True,
                    'mae': mae,
                    'rmse': rmse,
                    'sample_size': len(calibration_records)
                }
            else:
                return {
                    'calibration_success': False,
                    'error': 'No calibration data available'
                }
                
        except Exception as e:
            self.logger.error(f"模型校准失败: {e}")
            return {'calibration_success': False, 'error': str(e)}
    
    async def generate_cost_report(self, estimate: TransactionCostEstimate) -> Dict[str, Any]:
        """生成成本报告"""
        report = {
            'executive_summary': {
                'symbol': estimate.symbol,
                'expected_cost': f"${estimate.expected_cost:,.2f}",
                'cost_basis_points': f"{estimate.cost_breakdown.cost_basis_points:.1f} bp",
                'cost_range': f"${estimate.cost_range[0]:,.2f} - ${estimate.cost_range[1]:,.2f}",
                'optimal_strategy': estimate.optimal_strategy,
                'execution_time': f"{estimate.execution_time} minutes",
                'market_regime': estimate.market_regime.value
            },
            'cost_breakdown': {
                'commission': f"${estimate.cost_breakdown.commission:,.2f}",
                'slippage': f"${estimate.cost_breakdown.slippage:,.2f}",
                'market_impact': f"${estimate.cost_breakdown.market_impact:,.2f}",
                'spread_cost': f"${estimate.cost_breakdown.spread_cost:,.2f}",
                'timing_cost': f"${estimate.cost_breakdown.timing_cost:,.2f}",
                'opportunity_cost': f"${estimate.cost_breakdown.opportunity_cost:,.2f}",
                'financing_cost': f"${estimate.cost_breakdown.financing_cost:,.2f}"
            },
            'risk_analysis': estimate.risk_metrics,
            'venue_analysis': estimate.venue_analysis,
            'sensitivity_analysis': estimate.sensitivity_analysis,
            'recommendations': [
                f"建议使用{estimate.optimal_strategy}策略",
                f"预计执行时间{estimate.execution_time}分钟",
                f"当前市场制度: {estimate.market_regime.value}"
            ]
        }
        
        return report
    
    async def visualize_cost_analysis(self, estimate: TransactionCostEstimate, 
                                    save_path: str = None):
        """可视化成本分析"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 成本分解饼图
        cost_components = [
            estimate.cost_breakdown.commission,
            estimate.cost_breakdown.slippage,
            estimate.cost_breakdown.market_impact,
            estimate.cost_breakdown.spread_cost,
            estimate.cost_breakdown.timing_cost,
            estimate.cost_breakdown.opportunity_cost,
            estimate.cost_breakdown.financing_cost
        ]
        
        labels = ['Commission', 'Slippage', 'Market Impact', 'Spread Cost', 
                 'Timing Cost', 'Opportunity Cost', 'Financing Cost']
        
        axes[0, 0].pie(cost_components, labels=labels, autopct='%1.1f%%')
        axes[0, 0].set_title('Cost Breakdown')
        
        # 成本区间
        axes[0, 1].bar(['Expected', 'Lower Bound', 'Upper Bound'], 
                      [estimate.expected_cost, estimate.cost_range[0], estimate.cost_range[1]])
        axes[0, 1].set_title('Cost Range')
        axes[0, 1].set_ylabel('Cost ($)')
        
        # 场所分析
        if estimate.venue_analysis:
            venues = list(estimate.venue_analysis.keys())
            costs = list(estimate.venue_analysis.values())
            axes[1, 0].bar([v.value for v in venues], costs)
            axes[1, 0].set_title('Venue Analysis')
            axes[1, 0].set_ylabel('Cost ($)')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 敏感性分析
        if estimate.sensitivity_analysis:
            factors = list(estimate.sensitivity_analysis.keys())
            sensitivities = list(estimate.sensitivity_analysis.values())
            axes[1, 1].bar(factors, sensitivities)
            axes[1, 1].set_title('Sensitivity Analysis')
            axes[1, 1].set_ylabel('Sensitivity')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
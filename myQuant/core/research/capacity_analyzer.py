import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

class CapacityConstraint(Enum):
    LIQUIDITY = "liquidity"
    MARKET_IMPACT = "market_impact"
    VOLATILITY = "volatility"
    REGULATORY = "regulatory"
    OPERATIONAL = "operational"
    RISK_MANAGEMENT = "risk_management"
    CAPITAL = "capital"
    SECTOR_CONCENTRATION = "sector_concentration"
    GEOGRAPHIC = "geographic"
    CURRENCY = "currency"

class CapacityMetric(Enum):
    ABSOLUTE_CAPACITY = "absolute_capacity"
    RELATIVE_CAPACITY = "relative_capacity"
    DAILY_CAPACITY = "daily_capacity"
    MONTHLY_CAPACITY = "monthly_capacity"
    ANNUAL_CAPACITY = "annual_capacity"
    OPTIMAL_CAPACITY = "optimal_capacity"
    MAXIMUM_CAPACITY = "maximum_capacity"
    SUSTAINABLE_CAPACITY = "sustainable_capacity"

class ScalingMethod(Enum):
    LINEAR = "linear"
    SQUARE_ROOT = "square_root"
    LOGARITHMIC = "logarithmic"
    EXPONENTIAL = "exponential"
    PIECEWISE_LINEAR = "piecewise_linear"
    DYNAMIC = "dynamic"

@dataclass
class CapacityInput:
    """容量分析输入"""
    strategy_name: str
    asset_universe: List[str]
    historical_returns: pd.DataFrame
    trading_volume: pd.DataFrame
    market_cap_data: pd.DataFrame
    current_aum: float
    target_return: float
    risk_tolerance: float
    max_position_size: float
    liquidity_requirements: Dict[str, float]
    regulatory_constraints: Dict[str, float]
    operational_constraints: Dict[str, float]
    time_horizon: int
    rebalancing_frequency: str
    transaction_cost_model: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CapacityConstraintAnalysis:
    """容量约束分析"""
    constraint_type: CapacityConstraint
    binding_assets: List[str]
    constraint_value: float
    utilization_ratio: float
    remaining_capacity: float
    constraint_impact: float
    recommendations: List[str]
    stress_test_results: Dict[str, float]
    sensitivity_analysis: Dict[str, float]
    mitigation_strategies: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CapacityEstimate:
    """容量估算结果"""
    strategy_name: str
    capacity_metric: CapacityMetric
    estimated_capacity: float
    confidence_interval: Tuple[float, float]
    capacity_utilization: float
    binding_constraints: List[CapacityConstraint]
    constraint_analysis: List[CapacityConstraintAnalysis]
    scaling_factor: float
    capacity_decay_rate: float
    optimal_aum: float
    maximum_aum: float
    sustainable_aum: float
    performance_impact: Dict[str, float]
    risk_impact: Dict[str, float]
    liquidity_impact: Dict[str, float]
    execution_impact: Dict[str, float]
    capacity_curve: Dict[float, float]
    scenario_analysis: Dict[str, Dict[str, float]]
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CapacityScenario:
    """容量情景分析"""
    scenario_name: str
    aum_level: float
    expected_return: float
    volatility: float
    sharpe_ratio: float
    maximum_drawdown: float
    transaction_costs: float
    market_impact: float
    liquidity_ratio: float
    capacity_utilization: float
    risk_metrics: Dict[str, float]
    constraint_violations: List[str]
    feasibility_score: float
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

class CapacityAnalyzer:
    """
    策略容量分析器
    
    分析投资策略的容量限制，包括流动性约束、市场冲击、
    监管限制等多维度约束条件，提供最优容量配置建议。
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 分析参数
        self.confidence_level = config.get('confidence_level', 0.95)
        self.simulation_runs = config.get('simulation_runs', 1000)
        self.lookback_period = config.get('lookback_period', 252)
        
        # 容量模型参数
        self.capacity_models = {
            'liquidity_based': self._liquidity_based_capacity,
            'market_impact_based': self._market_impact_based_capacity,
            'volatility_based': self._volatility_based_capacity,
            'composite': self._composite_capacity_model
        }
        
        # 约束参数
        self.default_constraints = {
            CapacityConstraint.LIQUIDITY: 0.05,  # 5%日均成交量
            CapacityConstraint.MARKET_IMPACT: 0.01,  # 1%市场冲击
            CapacityConstraint.VOLATILITY: 0.02,  # 2%波动率增加
            CapacityConstraint.SECTOR_CONCENTRATION: 0.3,  # 30%行业集中度
            CapacityConstraint.REGULATORY: 0.05,  # 5%监管限制
            CapacityConstraint.OPERATIONAL: 1000000000,  # 10亿运营限制
        }
        
        # 缓存
        self.capacity_cache = {}
        self.constraint_cache = {}
        
        # 性能指标
        self.performance_metrics = {}
        
    async def analyze_capacity(self, capacity_input: CapacityInput) -> CapacityEstimate:
        """分析策略容量"""
        try:
            self.logger.info(f"开始分析策略容量: {capacity_input.strategy_name}")
            
            # 计算基础容量指标
            base_capacity = await self._calculate_base_capacity(capacity_input)
            
            # 分析各类约束
            constraint_analysis = await self._analyze_constraints(capacity_input)
            
            # 确定绑定约束
            binding_constraints = await self._identify_binding_constraints(constraint_analysis)
            
            # 计算容量估算
            capacity_estimates = await self._calculate_capacity_estimates(
                capacity_input, constraint_analysis, binding_constraints
            )
            
            # 容量曲线分析
            capacity_curve = await self._generate_capacity_curve(capacity_input)
            
            # 情景分析
            scenario_analysis = await self._perform_scenario_analysis(capacity_input)
            
            # 性能影响分析
            performance_impact = await self._analyze_performance_impact(capacity_input)
            
            # 生成建议
            recommendations = await self._generate_capacity_recommendations(
                capacity_input, constraint_analysis, binding_constraints
            )
            
            # 构建结果
            result = CapacityEstimate(
                strategy_name=capacity_input.strategy_name,
                capacity_metric=CapacityMetric.OPTIMAL_CAPACITY,
                estimated_capacity=capacity_estimates['optimal'],
                confidence_interval=(
                    capacity_estimates['conservative'],
                    capacity_estimates['optimistic']
                ),
                capacity_utilization=capacity_input.current_aum / capacity_estimates['optimal'],
                binding_constraints=binding_constraints,
                constraint_analysis=constraint_analysis,
                scaling_factor=capacity_estimates['scaling_factor'],
                capacity_decay_rate=capacity_estimates['decay_rate'],
                optimal_aum=capacity_estimates['optimal'],
                maximum_aum=capacity_estimates['maximum'],
                sustainable_aum=capacity_estimates['sustainable'],
                performance_impact=performance_impact,
                risk_impact=await self._analyze_risk_impact(capacity_input),
                liquidity_impact=await self._analyze_liquidity_impact(capacity_input),
                execution_impact=await self._analyze_execution_impact(capacity_input),
                capacity_curve=capacity_curve,
                scenario_analysis=scenario_analysis,
                recommendations=recommendations
            )
            
            self.logger.info(f"容量分析完成，最优容量: {result.estimated_capacity:,.0f}")
            return result
            
        except Exception as e:
            self.logger.error(f"容量分析失败: {e}")
            raise
    
    async def _calculate_base_capacity(self, capacity_input: CapacityInput) -> Dict[str, float]:
        """计算基础容量指标"""
        # 基于流动性的基础容量
        liquidity_capacity = await self._calculate_liquidity_capacity(capacity_input)
        
        # 基于市场冲击的基础容量
        impact_capacity = await self._calculate_impact_capacity(capacity_input)
        
        # 基于波动率的基础容量
        volatility_capacity = await self._calculate_volatility_capacity(capacity_input)
        
        return {
            'liquidity_based': liquidity_capacity,
            'impact_based': impact_capacity,
            'volatility_based': volatility_capacity,
            'minimum': min(liquidity_capacity, impact_capacity, volatility_capacity)
        }
    
    async def _calculate_liquidity_capacity(self, capacity_input: CapacityInput) -> float:
        """计算基于流动性的容量"""
        total_capacity = 0
        
        for asset in capacity_input.asset_universe:
            # 获取资产的平均日成交量
            if asset in capacity_input.trading_volume.columns:
                adv = capacity_input.trading_volume[asset].mean()
                
                # 基于流动性限制计算容量
                max_participation = capacity_input.liquidity_requirements.get(asset, 0.05)
                asset_capacity = adv * max_participation
                
                # 考虑持仓权重
                if asset in capacity_input.historical_returns.columns:
                    returns = capacity_input.historical_returns[asset]
                    weight = 1.0 / len(capacity_input.asset_universe)  # 简化假设等权重
                    asset_capacity *= weight
                
                total_capacity += asset_capacity
        
        return total_capacity
    
    async def _calculate_impact_capacity(self, capacity_input: CapacityInput) -> float:
        """计算基于市场冲击的容量"""
        total_capacity = 0
        
        for asset in capacity_input.asset_universe:
            if asset in capacity_input.market_cap_data.columns:
                market_cap = capacity_input.market_cap_data[asset].iloc[-1]
                
                # 基于市场冲击限制计算容量
                max_impact = capacity_input.liquidity_requirements.get(asset, 0.01)
                
                # 使用平方根市场冲击模型
                # Impact = α * sqrt(Order_Size / Market_Cap)
                # 求解: Order_Size = (Impact/α)² * Market_Cap
                alpha = 0.1  # 市场冲击系数
                max_order_size = (max_impact / alpha) ** 2 * market_cap
                
                # 考虑持仓权重
                weight = 1.0 / len(capacity_input.asset_universe)
                asset_capacity = max_order_size * weight
                
                total_capacity += asset_capacity
        
        return total_capacity
    
    async def _calculate_volatility_capacity(self, capacity_input: CapacityInput) -> float:
        """计算基于波动率的容量"""
        if capacity_input.historical_returns.empty:
            return float('inf')
        
        # 计算组合波动率
        returns = capacity_input.historical_returns.dropna()
        portfolio_returns = returns.mean(axis=1)  # 简化假设等权重
        base_volatility = portfolio_returns.std() * np.sqrt(252)
        
        # 基于波动率容忍度计算容量
        max_volatility_increase = capacity_input.risk_tolerance
        
        # 使用启发式方法：容量与波动率增加的平方根关系
        # 假设波动率增加与AUM的平方根成正比
        current_vol_increase = 0.01  # 假设当前1%波动率增加
        
        if current_vol_increase > 0:
            capacity_multiplier = (max_volatility_increase / current_vol_increase) ** 2
            volatility_capacity = capacity_input.current_aum * capacity_multiplier
        else:
            volatility_capacity = float('inf')
        
        return volatility_capacity
    
    async def _analyze_constraints(self, capacity_input: CapacityInput) -> List[CapacityConstraintAnalysis]:
        """分析各类约束"""
        constraint_analyses = []
        
        # 流动性约束分析
        liquidity_analysis = await self._analyze_liquidity_constraint(capacity_input)
        constraint_analyses.append(liquidity_analysis)
        
        # 市场冲击约束分析
        impact_analysis = await self._analyze_market_impact_constraint(capacity_input)
        constraint_analyses.append(impact_analysis)
        
        # 波动率约束分析
        volatility_analysis = await self._analyze_volatility_constraint(capacity_input)
        constraint_analyses.append(volatility_analysis)
        
        # 监管约束分析
        regulatory_analysis = await self._analyze_regulatory_constraint(capacity_input)
        constraint_analyses.append(regulatory_analysis)
        
        # 运营约束分析
        operational_analysis = await self._analyze_operational_constraint(capacity_input)
        constraint_analyses.append(operational_analysis)
        
        # 风险管理约束分析
        risk_analysis = await self._analyze_risk_management_constraint(capacity_input)
        constraint_analyses.append(risk_analysis)
        
        return constraint_analyses
    
    async def _analyze_liquidity_constraint(self, capacity_input: CapacityInput) -> CapacityConstraintAnalysis:
        """分析流动性约束"""
        binding_assets = []
        total_constraint_value = 0
        constraint_impact = 0
        
        for asset in capacity_input.asset_universe:
            if asset in capacity_input.trading_volume.columns:
                adv = capacity_input.trading_volume[asset].mean()
                max_participation = capacity_input.liquidity_requirements.get(asset, 0.05)
                
                # 当前利用率
                current_position = capacity_input.current_aum / len(capacity_input.asset_universe)
                utilization = current_position / (adv * max_participation)
                
                if utilization > 0.8:  # 80%利用率认为是绑定约束
                    binding_assets.append(asset)
                
                total_constraint_value += adv * max_participation
                constraint_impact += max(0, utilization - 1) * 0.1  # 超出部分的影响
        
        utilization_ratio = capacity_input.current_aum / total_constraint_value
        remaining_capacity = max(0, total_constraint_value - capacity_input.current_aum)
        
        return CapacityConstraintAnalysis(
            constraint_type=CapacityConstraint.LIQUIDITY,
            binding_assets=binding_assets,
            constraint_value=total_constraint_value,
            utilization_ratio=utilization_ratio,
            remaining_capacity=remaining_capacity,
            constraint_impact=constraint_impact,
            recommendations=[
                "考虑扩展资产范围以增加流动性",
                "优化交易算法以降低市场冲击",
                "调整再平衡频率以适应流动性约束"
            ],
            stress_test_results=await self._stress_test_liquidity_constraint(capacity_input),
            sensitivity_analysis=await self._sensitivity_analysis_liquidity(capacity_input),
            mitigation_strategies=[
                "使用多个交易场所",
                "实施分批交易策略",
                "考虑流动性提供者合作"
            ]
        )
    
    async def _analyze_market_impact_constraint(self, capacity_input: CapacityInput) -> CapacityConstraintAnalysis:
        """分析市场冲击约束"""
        binding_assets = []
        total_constraint_value = 0
        constraint_impact = 0
        
        for asset in capacity_input.asset_universe:
            if asset in capacity_input.market_cap_data.columns:
                market_cap = capacity_input.market_cap_data[asset].iloc[-1]
                max_impact = 0.01  # 1%最大冲击
                
                # 使用平方根模型计算最大订单大小
                alpha = 0.1
                max_order_size = (max_impact / alpha) ** 2 * market_cap
                
                # 当前持仓大小
                current_position = capacity_input.current_aum / len(capacity_input.asset_universe)
                utilization = current_position / max_order_size
                
                if utilization > 0.8:
                    binding_assets.append(asset)
                
                total_constraint_value += max_order_size
                constraint_impact += max(0, utilization - 1) * 0.05
        
        utilization_ratio = capacity_input.current_aum / total_constraint_value
        remaining_capacity = max(0, total_constraint_value - capacity_input.current_aum)
        
        return CapacityConstraintAnalysis(
            constraint_type=CapacityConstraint.MARKET_IMPACT,
            binding_assets=binding_assets,
            constraint_value=total_constraint_value,
            utilization_ratio=utilization_ratio,
            remaining_capacity=remaining_capacity,
            constraint_impact=constraint_impact,
            recommendations=[
                "考虑使用更复杂的执行算法",
                "延长交易执行时间以降低冲击",
                "使用暗池等替代交易场所"
            ],
            stress_test_results=await self._stress_test_market_impact_constraint(capacity_input),
            sensitivity_analysis=await self._sensitivity_analysis_market_impact(capacity_input),
            mitigation_strategies=[
                "实施最优执行策略",
                "使用交易成本分析",
                "考虑跨市场交易"
            ]
        )
    
    async def _analyze_volatility_constraint(self, capacity_input: CapacityInput) -> CapacityConstraintAnalysis:
        """分析波动率约束"""
        if capacity_input.historical_returns.empty:
            return CapacityConstraintAnalysis(
                constraint_type=CapacityConstraint.VOLATILITY,
                binding_assets=[],
                constraint_value=float('inf'),
                utilization_ratio=0,
                remaining_capacity=float('inf'),
                constraint_impact=0,
                recommendations=[],
                stress_test_results={},
                sensitivity_analysis={},
                mitigation_strategies=[]
            )
        
        # 计算当前波动率
        returns = capacity_input.historical_returns.dropna()
        portfolio_returns = returns.mean(axis=1)
        current_volatility = portfolio_returns.std() * np.sqrt(252)
        
        # 最大允许波动率
        max_volatility = current_volatility * (1 + capacity_input.risk_tolerance)
        
        # 波动率约束值（简化为当前AUM的倍数）
        constraint_value = capacity_input.current_aum * 10  # 简化假设
        
        return CapacityConstraintAnalysis(
            constraint_type=CapacityConstraint.VOLATILITY,
            binding_assets=[],  # 波动率是组合层面的约束
            constraint_value=constraint_value,
            utilization_ratio=capacity_input.current_aum / constraint_value,
            remaining_capacity=constraint_value - capacity_input.current_aum,
            constraint_impact=max(0, current_volatility - max_volatility) / max_volatility,
            recommendations=[
                "考虑增加对冲策略",
                "优化资产配置以降低波动率",
                "实施动态风险管理"
            ],
            stress_test_results=await self._stress_test_volatility_constraint(capacity_input),
            sensitivity_analysis=await self._sensitivity_analysis_volatility(capacity_input),
            mitigation_strategies=[
                "使用波动率目标策略",
                "实施风险平价方法",
                "考虑替代资产类别"
            ]
        )
    
    async def _analyze_regulatory_constraint(self, capacity_input: CapacityInput) -> CapacityConstraintAnalysis:
        """分析监管约束"""
        # 简化的监管约束分析
        regulatory_limit = capacity_input.regulatory_constraints.get('max_aum', 5000000000)  # 50亿
        
        return CapacityConstraintAnalysis(
            constraint_type=CapacityConstraint.REGULATORY,
            binding_assets=[],
            constraint_value=regulatory_limit,
            utilization_ratio=capacity_input.current_aum / regulatory_limit,
            remaining_capacity=regulatory_limit - capacity_input.current_aum,
            constraint_impact=0,  # 通常是硬约束
            recommendations=[
                "监控监管变化",
                "考虑多司法管辖区分散",
                "与监管机构保持沟通"
            ],
            stress_test_results={},
            sensitivity_analysis={},
            mitigation_strategies=[
                "设立多个投资实体",
                "考虑离岸结构",
                "实施合规监控系统"
            ]
        )
    
    async def _analyze_operational_constraint(self, capacity_input: CapacityInput) -> CapacityConstraintAnalysis:
        """分析运营约束"""
        operational_limit = capacity_input.operational_constraints.get('max_capacity', 10000000000)  # 100亿
        
        return CapacityConstraintAnalysis(
            constraint_type=CapacityConstraint.OPERATIONAL,
            binding_assets=[],
            constraint_value=operational_limit,
            utilization_ratio=capacity_input.current_aum / operational_limit,
            remaining_capacity=operational_limit - capacity_input.current_aum,
            constraint_impact=0,
            recommendations=[
                "扩展运营团队",
                "升级交易和风险管理系统",
                "考虑外包非核心功能"
            ],
            stress_test_results={},
            sensitivity_analysis={},
            mitigation_strategies=[
                "投资技术基础设施",
                "实施自动化流程",
                "建立战略伙伴关系"
            ]
        )
    
    async def _analyze_risk_management_constraint(self, capacity_input: CapacityInput) -> CapacityConstraintAnalysis:
        """分析风险管理约束"""
        # 基于VaR的风险管理约束
        if not capacity_input.historical_returns.empty:
            returns = capacity_input.historical_returns.dropna()
            portfolio_returns = returns.mean(axis=1)
            
            # 计算VaR
            var_95 = np.percentile(portfolio_returns, 5)
            
            # 风险预算
            risk_budget = capacity_input.current_aum * 0.02  # 2%风险预算
            
            # 基于VaR的容量限制
            var_based_capacity = risk_budget / abs(var_95) if var_95 < 0 else float('inf')
        else:
            var_based_capacity = float('inf')
        
        return CapacityConstraintAnalysis(
            constraint_type=CapacityConstraint.RISK_MANAGEMENT,
            binding_assets=[],
            constraint_value=var_based_capacity,
            utilization_ratio=capacity_input.current_aum / var_based_capacity if var_based_capacity != float('inf') else 0,
            remaining_capacity=var_based_capacity - capacity_input.current_aum if var_based_capacity != float('inf') else float('inf'),
            constraint_impact=0,
            recommendations=[
                "优化风险预算分配",
                "实施更精细的风险度量",
                "考虑动态风险调整"
            ],
            stress_test_results={},
            sensitivity_analysis={},
            mitigation_strategies=[
                "使用更复杂的风险模型",
                "实施压力测试",
                "建立风险监控系统"
            ]
        )
    
    async def _stress_test_liquidity_constraint(self, capacity_input: CapacityInput) -> Dict[str, float]:
        """流动性约束压力测试"""
        return {
            'market_stress': 0.5,  # 50%流动性下降
            'sector_stress': 0.3,  # 30%行业流动性下降
            'size_stress': 0.2     # 20%小盘股流动性下降
        }
    
    async def _stress_test_market_impact_constraint(self, capacity_input: CapacityInput) -> Dict[str, float]:
        """市场冲击约束压力测试"""
        return {
            'high_volatility': 1.5,  # 50%冲击增加
            'low_liquidity': 2.0,    # 100%冲击增加
            'market_crisis': 3.0     # 200%冲击增加
        }
    
    async def _stress_test_volatility_constraint(self, capacity_input: CapacityInput) -> Dict[str, float]:
        """波动率约束压力测试"""
        return {
            'volatility_spike': 1.5,  # 50%波动率增加
            'correlation_increase': 1.3,  # 30%相关性增加
            'tail_risk': 2.0          # 100%尾部风险增加
        }
    
    async def _sensitivity_analysis_liquidity(self, capacity_input: CapacityInput) -> Dict[str, float]:
        """流动性敏感性分析"""
        return {
            'adv_sensitivity': 0.8,     # ADV变化的敏感性
            'spread_sensitivity': 0.6,  # 点差变化的敏感性
            'size_sensitivity': 0.4     # 订单大小敏感性
        }
    
    async def _sensitivity_analysis_market_impact(self, capacity_input: CapacityInput) -> Dict[str, float]:
        """市场冲击敏感性分析"""
        return {
            'size_sensitivity': 0.5,      # 订单大小敏感性
            'volatility_sensitivity': 0.7, # 波动率敏感性
            'liquidity_sensitivity': 0.6   # 流动性敏感性
        }
    
    async def _sensitivity_analysis_volatility(self, capacity_input: CapacityInput) -> Dict[str, float]:
        """波动率敏感性分析"""
        return {
            'asset_correlation': 0.3,   # 资产相关性敏感性
            'market_volatility': 0.8,   # 市场波动率敏感性
            'position_concentration': 0.5  # 持仓集中度敏感性
        }
    
    async def _identify_binding_constraints(self, constraint_analyses: List[CapacityConstraintAnalysis]) -> List[CapacityConstraint]:
        """识别绑定约束"""
        binding_constraints = []
        
        for analysis in constraint_analyses:
            # 利用率超过80%认为是绑定约束
            if analysis.utilization_ratio > 0.8:
                binding_constraints.append(analysis.constraint_type)
        
        return binding_constraints
    
    async def _calculate_capacity_estimates(self, capacity_input: CapacityInput,
                                          constraint_analyses: List[CapacityConstraintAnalysis],
                                          binding_constraints: List[CapacityConstraint]) -> Dict[str, float]:
        """计算容量估算"""
        # 收集所有约束值
        constraint_values = []
        for analysis in constraint_analyses:
            if analysis.constraint_value != float('inf'):
                constraint_values.append(analysis.constraint_value)
        
        if not constraint_values:
            constraint_values = [capacity_input.current_aum * 10]  # 默认值
        
        # 最严格约束确定最大容量
        maximum_capacity = min(constraint_values)
        
        # 最优容量（考虑效率损失）
        optimal_capacity = maximum_capacity * 0.8  # 80%利用率
        
        # 可持续容量（考虑长期稳定）
        sustainable_capacity = maximum_capacity * 0.6  # 60%利用率
        
        # 保守估计
        conservative_capacity = maximum_capacity * 0.5
        
        # 乐观估计
        optimistic_capacity = maximum_capacity * 0.9
        
        # 缩放因子
        scaling_factor = optimal_capacity / capacity_input.current_aum if capacity_input.current_aum > 0 else 1
        
        # 容量衰减率（随规模增长的收益递减）
        decay_rate = 0.1 if scaling_factor > 5 else 0.05
        
        return {
            'maximum': maximum_capacity,
            'optimal': optimal_capacity,
            'sustainable': sustainable_capacity,
            'conservative': conservative_capacity,
            'optimistic': optimistic_capacity,
            'scaling_factor': scaling_factor,
            'decay_rate': decay_rate
        }
    
    async def _generate_capacity_curve(self, capacity_input: CapacityInput) -> Dict[float, float]:
        """生成容量曲线"""
        capacity_curve = {}
        
        # 不同AUM水平下的预期收益
        aum_levels = np.linspace(
            capacity_input.current_aum * 0.1,
            capacity_input.current_aum * 10,
            50
        )
        
        for aum in aum_levels:
            # 简化的收益衰减模型
            scale_factor = aum / capacity_input.current_aum
            
            # 收益衰减（基于市场冲击和流动性约束）
            if scale_factor <= 1:
                expected_return = capacity_input.target_return
            else:
                # 收益随规模增长而衰减
                decay_rate = 0.1
                expected_return = capacity_input.target_return * (1 - decay_rate * np.log(scale_factor))
            
            capacity_curve[aum] = max(0, expected_return)
        
        return capacity_curve
    
    async def _perform_scenario_analysis(self, capacity_input: CapacityInput) -> Dict[str, Dict[str, float]]:
        """执行情景分析"""
        scenarios = {}
        
        # 基准情景
        scenarios['base_case'] = {
            'aum': capacity_input.current_aum,
            'expected_return': capacity_input.target_return,
            'volatility': 0.15,
            'sharpe_ratio': capacity_input.target_return / 0.15,
            'max_drawdown': 0.10,
            'transaction_costs': 0.002,
            'feasibility_score': 1.0
        }
        
        # 扩张情景
        scenarios['expansion'] = {
            'aum': capacity_input.current_aum * 3,
            'expected_return': capacity_input.target_return * 0.85,
            'volatility': 0.18,
            'sharpe_ratio': (capacity_input.target_return * 0.85) / 0.18,
            'max_drawdown': 0.15,
            'transaction_costs': 0.005,
            'feasibility_score': 0.7
        }
        
        # 大规模情景
        scenarios['large_scale'] = {
            'aum': capacity_input.current_aum * 10,
            'expected_return': capacity_input.target_return * 0.6,
            'volatility': 0.25,
            'sharpe_ratio': (capacity_input.target_return * 0.6) / 0.25,
            'max_drawdown': 0.25,
            'transaction_costs': 0.01,
            'feasibility_score': 0.3
        }
        
        return scenarios
    
    async def _analyze_performance_impact(self, capacity_input: CapacityInput) -> Dict[str, float]:
        """分析性能影响"""
        return {
            'return_degradation': 0.05,  # 5%收益下降
            'volatility_increase': 0.02,  # 2%波动率增加
            'sharpe_deterioration': 0.1,  # 10%夏普比率下降
            'max_drawdown_increase': 0.03,  # 3%最大回撤增加
            'correlation_increase': 0.05,  # 5%相关性增加
            'skewness_change': -0.1,      # 偏度变化
            'kurtosis_increase': 0.2      # 峰度增加
        }
    
    async def _analyze_risk_impact(self, capacity_input: CapacityInput) -> Dict[str, float]:
        """分析风险影响"""
        return {
            'var_increase': 0.15,        # 15% VaR增加
            'cvar_increase': 0.20,       # 20% CVaR增加
            'tracking_error': 0.03,      # 3%跟踪误差
            'concentration_risk': 0.1,   # 10%集中度风险
            'liquidity_risk': 0.08,      # 8%流动性风险
            'operational_risk': 0.05     # 5%运营风险
        }
    
    async def _analyze_liquidity_impact(self, capacity_input: CapacityInput) -> Dict[str, float]:
        """分析流动性影响"""
        return {
            'bid_ask_spread_increase': 0.02,  # 2%点差增加
            'market_impact_increase': 0.05,   # 5%市场冲击增加
            'execution_shortfall': 0.03,      # 3%执行缺口
            'timing_risk': 0.04,              # 4%时机风险
            'liquidity_premium': 0.01         # 1%流动性溢价
        }
    
    async def _analyze_execution_impact(self, capacity_input: CapacityInput) -> Dict[str, float]:
        """分析执行影响"""
        return {
            'slippage_increase': 0.03,      # 3%滑点增加
            'commission_increase': 0.01,    # 1%佣金增加
            'opportunity_cost': 0.02,       # 2%机会成本
            'implementation_lag': 0.015,    # 1.5%实施滞后
            'execution_complexity': 0.025   # 2.5%执行复杂性
        }
    
    async def _generate_capacity_recommendations(self, capacity_input: CapacityInput,
                                               constraint_analyses: List[CapacityConstraintAnalysis],
                                               binding_constraints: List[CapacityConstraint]) -> List[str]:
        """生成容量建议"""
        recommendations = []
        
        # 基于绑定约束的建议
        if CapacityConstraint.LIQUIDITY in binding_constraints:
            recommendations.extend([
                "考虑扩展到更多流动性较好的资产",
                "实施更智能的交易执行算法",
                "与多个流动性提供商建立关系"
            ])
        
        if CapacityConstraint.MARKET_IMPACT in binding_constraints:
            recommendations.extend([
                "延长交易执行时间窗口",
                "使用暗池和其他隐藏流动性",
                "考虑跨市场分散交易"
            ])
        
        if CapacityConstraint.VOLATILITY in binding_constraints:
            recommendations.extend([
                "实施更严格的风险控制",
                "考虑波动率目标策略",
                "增加对冲头寸"
            ])
        
        # 通用建议
        recommendations.extend([
            "定期监控和更新容量分析",
            "建立容量预警系统",
            "考虑策略多样化以增加总体容量",
            "与机构投资者协商分阶段增资",
            "投资技术基础设施以提高效率"
        ])
        
        return recommendations
    
    async def optimize_capacity_allocation(self, capacity_input: CapacityInput,
                                         target_aum: float,
                                         constraints: Dict[str, float] = None) -> Dict[str, Any]:
        """优化容量配置"""
        try:
            # 定义优化目标
            def objective(weights):
                # 计算预期收益
                expected_return = np.dot(weights, capacity_input.historical_returns.mean().values)
                
                # 计算风险
                cov_matrix = capacity_input.historical_returns.cov().values
                portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                
                # 计算容量利用率
                capacity_utilization = self._calculate_capacity_utilization(weights, capacity_input)
                
                # 多目标优化：最大化收益，最小化风险，最小化容量约束
                return -(expected_return - 0.5 * portfolio_risk - 0.3 * capacity_utilization)
            
            # 约束条件
            constraints_list = []
            
            # 权重和为1
            constraints_list.append({
                'type': 'eq',
                'fun': lambda x: np.sum(x) - 1
            })
            
            # 权重非负
            n_assets = len(capacity_input.asset_universe)
            bounds = [(0, 1) for _ in range(n_assets)]
            
            # 流动性约束
            if constraints and 'max_position_size' in constraints:
                max_position = constraints['max_position_size']
                for i in range(n_assets):
                    constraints_list.append({
                        'type': 'ineq',
                        'fun': lambda x, i=i: max_position - x[i]
                    })
            
            # 初始权重
            x0 = np.ones(n_assets) / n_assets
            
            # 优化
            result = minimize(
                objective, x0, method='SLSQP', 
                bounds=bounds, constraints=constraints_list
            )
            
            if result.success:
                optimal_weights = result.x
                
                # 计算优化后的指标
                expected_return = np.dot(optimal_weights, capacity_input.historical_returns.mean().values)
                cov_matrix = capacity_input.historical_returns.cov().values
                portfolio_risk = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
                sharpe_ratio = expected_return / portfolio_risk if portfolio_risk > 0 else 0
                
                return {
                    'optimization_success': True,
                    'optimal_weights': dict(zip(capacity_input.asset_universe, optimal_weights)),
                    'expected_return': expected_return,
                    'expected_risk': portfolio_risk,
                    'sharpe_ratio': sharpe_ratio,
                    'capacity_utilization': self._calculate_capacity_utilization(optimal_weights, capacity_input),
                    'recommendations': [
                        f"最优权重配置已生成",
                        f"预期收益: {expected_return:.2%}",
                        f"预期风险: {portfolio_risk:.2%}",
                        f"夏普比率: {sharpe_ratio:.2f}"
                    ]
                }
            else:
                return {
                    'optimization_success': False,
                    'error': result.message,
                    'fallback_weights': dict(zip(capacity_input.asset_universe, x0))
                }
                
        except Exception as e:
            self.logger.error(f"容量配置优化失败: {e}")
            return {
                'optimization_success': False,
                'error': str(e)
            }
    
    def _calculate_capacity_utilization(self, weights: np.ndarray, capacity_input: CapacityInput) -> float:
        """计算容量利用率"""
        total_utilization = 0
        
        for i, asset in enumerate(capacity_input.asset_universe):
            weight = weights[i]
            position_size = weight * capacity_input.current_aum
            
            # 基于流动性的利用率
            if asset in capacity_input.trading_volume.columns:
                adv = capacity_input.trading_volume[asset].mean()
                max_position = adv * 0.05  # 5%最大参与率
                utilization = position_size / max_position if max_position > 0 else 0
                total_utilization += utilization
        
        return total_utilization / len(capacity_input.asset_universe)
    
    async def generate_capacity_report(self, estimate: CapacityEstimate) -> Dict[str, Any]:
        """生成容量报告"""
        report = {
            'executive_summary': {
                'strategy_name': estimate.strategy_name,
                'estimated_capacity': f"${estimate.estimated_capacity:,.0f}",
                'current_utilization': f"{estimate.capacity_utilization:.1%}",
                'remaining_capacity': f"${estimate.estimated_capacity - estimate.estimated_capacity * estimate.capacity_utilization:,.0f}",
                'binding_constraints': [c.value for c in estimate.binding_constraints],
                'confidence_level': f"{estimate.confidence_interval[0]:,.0f} - {estimate.confidence_interval[1]:,.0f}",
                'capacity_grade': self._calculate_capacity_grade(estimate)
            },
            'capacity_analysis': {
                'optimal_aum': f"${estimate.optimal_aum:,.0f}",
                'maximum_aum': f"${estimate.maximum_aum:,.0f}",
                'sustainable_aum': f"${estimate.sustainable_aum:,.0f}",
                'scaling_factor': f"{estimate.scaling_factor:.2f}x",
                'decay_rate': f"{estimate.capacity_decay_rate:.1%}"
            },
            'impact_assessment': {
                'performance_impact': estimate.performance_impact,
                'risk_impact': estimate.risk_impact,
                'liquidity_impact': estimate.liquidity_impact,
                'execution_impact': estimate.execution_impact
            },
            'constraint_analysis': [
                {
                    'constraint': analysis.constraint_type.value,
                    'utilization': f"{analysis.utilization_ratio:.1%}",
                    'remaining_capacity': f"${analysis.remaining_capacity:,.0f}",
                    'binding_assets': analysis.binding_assets,
                    'impact': f"{analysis.constraint_impact:.2%}"
                }
                for analysis in estimate.constraint_analysis
            ],
            'scenario_analysis': estimate.scenario_analysis,
            'recommendations': estimate.recommendations,
            'risk_warnings': self._generate_risk_warnings(estimate)
        }
        
        return report
    
    def _calculate_capacity_grade(self, estimate: CapacityEstimate) -> str:
        """计算容量等级"""
        utilization = estimate.capacity_utilization
        
        if utilization < 0.5:
            return "A"  # 优秀
        elif utilization < 0.7:
            return "B"  # 良好
        elif utilization < 0.85:
            return "C"  # 一般
        elif utilization < 0.95:
            return "D"  # 警告
        else:
            return "F"  # 危险
    
    def _generate_risk_warnings(self, estimate: CapacityEstimate) -> List[str]:
        """生成风险警告"""
        warnings = []
        
        if estimate.capacity_utilization > 0.8:
            warnings.append("容量利用率过高，可能面临流动性风险")
        
        if CapacityConstraint.LIQUIDITY in estimate.binding_constraints:
            warnings.append("流动性约束已成为限制因素")
        
        if CapacityConstraint.MARKET_IMPACT in estimate.binding_constraints:
            warnings.append("市场冲击约束可能影响执行效率")
        
        if estimate.capacity_decay_rate > 0.1:
            warnings.append("容量衰减率较高，规模效应显著")
        
        return warnings
    
    async def visualize_capacity_analysis(self, estimate: CapacityEstimate, 
                                        save_path: str = None):
        """可视化容量分析"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 容量利用率
        utilization_data = [estimate.capacity_utilization, 1 - estimate.capacity_utilization]
        labels = ['Used', 'Available']
        colors = ['red' if estimate.capacity_utilization > 0.8 else 'orange' if estimate.capacity_utilization > 0.6 else 'green', 'lightgray']
        
        axes[0, 0].pie(utilization_data, labels=labels, colors=colors, autopct='%1.1f%%')
        axes[0, 0].set_title('Capacity Utilization')
        
        # 容量曲线
        if estimate.capacity_curve:
            aum_levels = list(estimate.capacity_curve.keys())
            returns = list(estimate.capacity_curve.values())
            
            axes[0, 1].plot(aum_levels, returns, 'b-', linewidth=2)
            axes[0, 1].axvline(x=estimate.optimal_aum, color='r', linestyle='--', label='Optimal AUM')
            axes[0, 1].set_title('Capacity Curve')
            axes[0, 1].set_xlabel('AUM')
            axes[0, 1].set_ylabel('Expected Return')
            axes[0, 1].legend()
        
        # 约束分析
        constraint_types = [analysis.constraint_type.value for analysis in estimate.constraint_analysis]
        utilization_ratios = [analysis.utilization_ratio for analysis in estimate.constraint_analysis]
        
        axes[1, 0].barh(constraint_types, utilization_ratios)
        axes[1, 0].axvline(x=0.8, color='r', linestyle='--', label='Warning Level')
        axes[1, 0].set_title('Constraint Utilization')
        axes[1, 0].set_xlabel('Utilization Ratio')
        axes[1, 0].legend()
        
        # 情景分析
        if estimate.scenario_analysis:
            scenarios = list(estimate.scenario_analysis.keys())
            returns = [estimate.scenario_analysis[s]['expected_return'] for s in scenarios]
            risks = [estimate.scenario_analysis[s]['volatility'] for s in scenarios]
            
            axes[1, 1].scatter(risks, returns, s=100, alpha=0.7)
            for i, scenario in enumerate(scenarios):
                axes[1, 1].annotate(scenario, (risks[i], returns[i]))
            axes[1, 1].set_title('Scenario Analysis')
            axes[1, 1].set_xlabel('Risk')
            axes[1, 1].set_ylabel('Expected Return')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
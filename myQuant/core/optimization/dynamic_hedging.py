import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

class HedgingStrategy(Enum):
    DELTA_HEDGING = "delta_hedging"
    GAMMA_HEDGING = "gamma_hedging"
    VEGA_HEDGING = "vega_hedging"
    THETA_HEDGING = "theta_hedging"
    VOLATILITY_HEDGING = "volatility_hedging"
    CROSS_HEDGING = "cross_hedging"
    DYNAMIC_REPLICATION = "dynamic_replication"
    MINIMUM_VARIANCE_HEDGING = "minimum_variance_hedging"
    OPTIMAL_HEDGING = "optimal_hedging"
    PAIRS_HEDGING = "pairs_hedging"

class HedgingFrequency(Enum):
    CONTINUOUS = "continuous"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    EVENT_DRIVEN = "event_driven"
    THRESHOLD_BASED = "threshold_based"

class RiskMeasure(Enum):
    VARIANCE = "variance"
    VALUE_AT_RISK = "value_at_risk"
    CONDITIONAL_VALUE_AT_RISK = "conditional_value_at_risk"
    EXPECTED_SHORTFALL = "expected_shortfall"
    MAXIMUM_DRAWDOWN = "maximum_drawdown"
    SEMI_VARIANCE = "semi_variance"

@dataclass
class HedgingInstrument:
    """对冲工具"""
    symbol: str
    instrument_type: str  # 'stock', 'option', 'future', 'swap', 'currency'
    expiration: Optional[datetime] = None
    strike: Optional[float] = None
    option_type: Optional[str] = None  # 'call', 'put'
    multiplier: float = 1.0
    liquidity_score: float = 1.0
    transaction_cost: float = 0.0
    margin_requirement: float = 0.0
    correlation_with_target: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HedgeRatio:
    """对冲比率"""
    instrument: HedgingInstrument
    ratio: float
    confidence: float
    hedge_effectiveness: float
    basis_risk: float
    rollover_risk: float
    liquidity_risk: float
    cost_of_carry: float
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class HedgingPosition:
    """对冲头寸"""
    instrument: HedgingInstrument
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0
    entry_time: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HedgingResult:
    """对冲结果"""
    strategy: HedgingStrategy
    target_exposure: float
    hedged_exposure: float
    hedge_effectiveness: float
    basis_risk: float
    residual_risk: float
    hedge_positions: List[HedgingPosition]
    hedge_ratios: List[HedgeRatio]
    total_hedge_cost: float
    portfolio_var_before: float
    portfolio_var_after: float
    risk_reduction: float
    sharpe_improvement: float
    tracking_error: float
    rebalancing_frequency: HedgingFrequency
    backtest_results: Optional[Dict[str, Any]] = None
    optimization_success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

class DynamicHedgingEngine:
    """
    动态对冲引擎
    
    提供多种动态对冲策略，包括Delta对冲、Gamma对冲、波动率对冲、
    交叉对冲等，以及实时风险监控和对冲比率优化。
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 引擎参数
        self.rebalancing_frequency = HedgingFrequency(config.get('rebalancing_frequency', 'daily'))
        self.risk_measure = RiskMeasure(config.get('risk_measure', 'variance'))
        self.hedge_effectiveness_threshold = config.get('hedge_effectiveness_threshold', 0.8)
        self.max_hedge_ratio = config.get('max_hedge_ratio', 2.0)
        self.min_hedge_ratio = config.get('min_hedge_ratio', -2.0)
        
        # 优化参数
        self.optimization_method = config.get('optimization_method', 'SLSQP')
        self.max_iterations = config.get('max_iterations', 1000)
        self.tolerance = config.get('tolerance', 1e-6)
        
        # 交易成本参数
        self.transaction_cost_rate = config.get('transaction_cost_rate', 0.001)
        self.bid_ask_spread = config.get('bid_ask_spread', 0.0001)
        self.market_impact_factor = config.get('market_impact_factor', 0.1)
        
        # 风险参数
        self.confidence_level = config.get('confidence_level', 0.95)
        self.lookback_period = config.get('lookback_period', 252)
        self.volatility_window = config.get('volatility_window', 30)
        
        # 数据存储
        self.hedge_positions: Dict[str, HedgingPosition] = {}
        self.hedge_ratios: Dict[str, HedgeRatio] = {}
        self.hedging_history: List[HedgingResult] = []
        
        # 市场数据缓存
        self.market_data_cache: Dict[str, pd.DataFrame] = {}
        self.volatility_cache: Dict[str, float] = {}
        self.correlation_cache: Dict[str, np.ndarray] = {}
        
        # 执行器
        self.executor = ThreadPoolExecutor(max_workers=config.get('max_workers', 4))
        
        self.logger.info(f"Dynamic hedging engine initialized with {self.rebalancing_frequency.value} frequency")
    
    async def create_hedge_strategy(self, 
                                  target_portfolio: Dict[str, float],
                                  hedge_instruments: List[HedgingInstrument],
                                  strategy: HedgingStrategy = HedgingStrategy.DELTA_HEDGING,
                                  constraints: Optional[Dict[str, Any]] = None,
                                  **kwargs) -> HedgingResult:
        """
        创建动态对冲策略
        """
        try:
            self.logger.info(f"Creating hedge strategy: {strategy.value}")
            
            # 验证输入
            if not self._validate_inputs(target_portfolio, hedge_instruments):
                raise ValueError("Invalid inputs for hedge strategy creation")
            
            # 选择对冲策略
            if strategy == HedgingStrategy.DELTA_HEDGING:
                result = await self._create_delta_hedge(
                    target_portfolio, hedge_instruments, constraints, **kwargs
                )
            elif strategy == HedgingStrategy.GAMMA_HEDGING:
                result = await self._create_gamma_hedge(
                    target_portfolio, hedge_instruments, constraints, **kwargs
                )
            elif strategy == HedgingStrategy.VEGA_HEDGING:
                result = await self._create_vega_hedge(
                    target_portfolio, hedge_instruments, constraints, **kwargs
                )
            elif strategy == HedgingStrategy.VOLATILITY_HEDGING:
                result = await self._create_volatility_hedge(
                    target_portfolio, hedge_instruments, constraints, **kwargs
                )
            elif strategy == HedgingStrategy.CROSS_HEDGING:
                result = await self._create_cross_hedge(
                    target_portfolio, hedge_instruments, constraints, **kwargs
                )
            elif strategy == HedgingStrategy.MINIMUM_VARIANCE_HEDGING:
                result = await self._create_minimum_variance_hedge(
                    target_portfolio, hedge_instruments, constraints, **kwargs
                )
            elif strategy == HedgingStrategy.OPTIMAL_HEDGING:
                result = await self._create_optimal_hedge(
                    target_portfolio, hedge_instruments, constraints, **kwargs
                )
            elif strategy == HedgingStrategy.PAIRS_HEDGING:
                result = await self._create_pairs_hedge(
                    target_portfolio, hedge_instruments, constraints, **kwargs
                )
            else:
                raise ValueError(f"Unsupported hedging strategy: {strategy}")
            
            # 保存结果
            self.hedging_history.append(result)
            
            # 更新头寸记录
            await self._update_hedge_positions(result)
            
            self.logger.info(f"Hedge strategy created successfully with {result.hedge_effectiveness:.2%} effectiveness")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error creating hedge strategy: {e}")
            raise
    
    async def _create_delta_hedge(self, 
                                target_portfolio: Dict[str, float],
                                hedge_instruments: List[HedgingInstrument],
                                constraints: Optional[Dict[str, Any]] = None,
                                **kwargs) -> HedgingResult:
        """
        创建Delta对冲策略
        """
        try:
            # 计算目标组合的Delta
            target_delta = await self._calculate_portfolio_delta(target_portfolio, **kwargs)
            
            # 选择对冲工具
            hedge_instruments_filtered = [
                inst for inst in hedge_instruments 
                if inst.instrument_type in ['option', 'future', 'stock']
            ]
            
            if not hedge_instruments_filtered:
                raise ValueError("No suitable hedge instruments for delta hedging")
            
            # 计算各对冲工具的Delta
            hedge_deltas = []
            for instrument in hedge_instruments_filtered:
                delta = await self._calculate_instrument_delta(instrument, **kwargs)
                hedge_deltas.append(delta)
            
            hedge_deltas = np.array(hedge_deltas)
            
            # 优化对冲比率
            def objective_function(hedge_ratios):
                # 计算对冲后的Delta
                hedged_delta = target_delta + np.dot(hedge_ratios, hedge_deltas)
                
                # 计算对冲成本
                hedge_cost = np.sum(np.abs(hedge_ratios) * np.array([
                    inst.transaction_cost * inst.multiplier 
                    for inst in hedge_instruments_filtered
                ]))
                
                # 目标：最小化剩余Delta + 对冲成本
                return hedged_delta ** 2 + self.transaction_cost_rate * hedge_cost
            
            # 约束条件
            constraints_list = []
            
            # 对冲比率界限
            bounds = [(self.min_hedge_ratio, self.max_hedge_ratio) 
                     for _ in hedge_instruments_filtered]
            
            # 自定义约束
            if constraints:
                # 总对冲金额限制
                if 'max_hedge_notional' in constraints:
                    def hedge_notional_constraint(hedge_ratios):
                        total_notional = np.sum(np.abs(hedge_ratios) * np.array([
                            inst.multiplier for inst in hedge_instruments_filtered
                        ]))
                        return constraints['max_hedge_notional'] - total_notional
                    
                    constraints_list.append({
                        'type': 'ineq',
                        'fun': hedge_notional_constraint
                    })
                
                # 流动性约束
                if 'min_liquidity_score' in constraints:
                    high_liquidity_instruments = [
                        i for i, inst in enumerate(hedge_instruments_filtered)
                        if inst.liquidity_score >= constraints['min_liquidity_score']
                    ]
                    
                    if high_liquidity_instruments:
                        def liquidity_constraint(hedge_ratios):
                            # 高流动性工具的权重应该更高
                            high_liquidity_weight = np.sum(np.abs(hedge_ratios[high_liquidity_instruments]))
                            total_weight = np.sum(np.abs(hedge_ratios))
                            return high_liquidity_weight - 0.7 * total_weight
                        
                        constraints_list.append({
                            'type': 'ineq',
                            'fun': liquidity_constraint
                        })
            
            # 初始对冲比率
            initial_hedge_ratios = np.zeros(len(hedge_instruments_filtered))
            
            # 如果目标Delta不为零，给出合理的初始值
            if abs(target_delta) > 1e-6 and len(hedge_deltas) > 0:
                # 选择Delta最大的工具作为主要对冲工具
                primary_hedge_idx = np.argmax(np.abs(hedge_deltas))
                if abs(hedge_deltas[primary_hedge_idx]) > 1e-6:
                    initial_hedge_ratios[primary_hedge_idx] = -target_delta / hedge_deltas[primary_hedge_idx]
            
            # 优化
            result = minimize(
                objective_function,
                initial_hedge_ratios,
                method=self.optimization_method,
                bounds=bounds,
                constraints=constraints_list,
                options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
            )
            
            optimal_hedge_ratios = result.x
            
            # 计算对冲效果
            hedged_delta = target_delta + np.dot(optimal_hedge_ratios, hedge_deltas)
            hedge_effectiveness = 1 - abs(hedged_delta) / abs(target_delta) if abs(target_delta) > 1e-6 else 1.0
            
            # 创建对冲头寸
            hedge_positions = []
            hedge_ratios = []
            total_hedge_cost = 0.0
            
            for i, (instrument, ratio) in enumerate(zip(hedge_instruments_filtered, optimal_hedge_ratios)):
                if abs(ratio) > 1e-6:  # 只保留非零头寸
                    # 计算头寸大小
                    quantity = ratio * instrument.multiplier
                    
                    # 估算当前价格（这里需要实际市场数据）
                    current_price = kwargs.get('current_prices', {}).get(instrument.symbol, 100.0)
                    
                    # 计算Delta
                    delta = hedge_deltas[i]
                    
                    # 创建头寸
                    position = HedgingPosition(
                        instrument=instrument,
                        quantity=quantity,
                        entry_price=current_price,
                        current_price=current_price,
                        unrealized_pnl=0.0,
                        delta=delta * quantity
                    )
                    
                    hedge_positions.append(position)
                    
                    # 创建对冲比率记录
                    hedge_ratio = HedgeRatio(
                        instrument=instrument,
                        ratio=ratio,
                        confidence=0.95,  # 可以基于历史数据计算
                        hedge_effectiveness=hedge_effectiveness,
                        basis_risk=0.05,  # 基差风险估计
                        rollover_risk=0.02,  # 展期风险估计
                        liquidity_risk=1.0 - instrument.liquidity_score,
                        cost_of_carry=0.03  # 持有成本估计
                    )
                    
                    hedge_ratios.append(hedge_ratio)
                    
                    # 计算对冲成本
                    hedge_cost = abs(quantity) * instrument.transaction_cost
                    total_hedge_cost += hedge_cost
            
            # 计算风险指标
            portfolio_var_before = await self._calculate_portfolio_variance(
                target_portfolio, **kwargs
            )
            
            # 构建对冲后的组合
            hedged_portfolio = target_portfolio.copy()
            for position in hedge_positions:
                symbol = position.instrument.symbol
                if symbol in hedged_portfolio:
                    hedged_portfolio[symbol] += position.quantity
                else:
                    hedged_portfolio[symbol] = position.quantity
            
            portfolio_var_after = await self._calculate_portfolio_variance(
                hedged_portfolio, **kwargs
            )
            
            risk_reduction = (portfolio_var_before - portfolio_var_after) / portfolio_var_before
            
            return HedgingResult(
                strategy=HedgingStrategy.DELTA_HEDGING,
                target_exposure=target_delta,
                hedged_exposure=hedged_delta,
                hedge_effectiveness=hedge_effectiveness,
                basis_risk=np.mean([hr.basis_risk for hr in hedge_ratios]) if hedge_ratios else 0.0,
                residual_risk=abs(hedged_delta),
                hedge_positions=hedge_positions,
                hedge_ratios=hedge_ratios,
                total_hedge_cost=total_hedge_cost,
                portfolio_var_before=portfolio_var_before,
                portfolio_var_after=portfolio_var_after,
                risk_reduction=risk_reduction,
                sharpe_improvement=risk_reduction * 0.1,  # 估算
                tracking_error=np.sqrt(portfolio_var_after),
                rebalancing_frequency=self.rebalancing_frequency,
                optimization_success=result.success
            )
            
        except Exception as e:
            self.logger.error(f"Error in delta hedging: {e}")
            raise
    
    async def _create_gamma_hedge(self, 
                                target_portfolio: Dict[str, float],
                                hedge_instruments: List[HedgingInstrument],
                                constraints: Optional[Dict[str, Any]] = None,
                                **kwargs) -> HedgingResult:
        """
        创建Gamma对冲策略
        """
        try:
            # 计算目标组合的Gamma
            target_gamma = await self._calculate_portfolio_gamma(target_portfolio, **kwargs)
            
            # 选择期权工具进行Gamma对冲
            option_instruments = [
                inst for inst in hedge_instruments 
                if inst.instrument_type == 'option'
            ]
            
            if not option_instruments:
                raise ValueError("No option instruments available for gamma hedging")
            
            # 计算各期权的Gamma
            hedge_gammas = []
            for instrument in option_instruments:
                gamma = await self._calculate_instrument_gamma(instrument, **kwargs)
                hedge_gammas.append(gamma)
            
            hedge_gammas = np.array(hedge_gammas)
            
            # 优化Gamma对冲比率
            def objective_function(hedge_ratios):
                # 计算对冲后的Gamma
                hedged_gamma = target_gamma + np.dot(hedge_ratios, hedge_gammas)
                
                # 计算对冲成本
                hedge_cost = np.sum(np.abs(hedge_ratios) * np.array([
                    inst.transaction_cost * inst.multiplier 
                    for inst in option_instruments
                ]))
                
                # 目标：最小化剩余Gamma + 对冲成本
                return hedged_gamma ** 2 + self.transaction_cost_rate * hedge_cost
            
            # 约束条件
            constraints_list = []
            bounds = [(self.min_hedge_ratio, self.max_hedge_ratio) 
                     for _ in option_instruments]
            
            # 初始对冲比率
            initial_hedge_ratios = np.zeros(len(option_instruments))
            
            # 如果目标Gamma不为零，给出合理的初始值
            if abs(target_gamma) > 1e-6 and len(hedge_gammas) > 0:
                primary_hedge_idx = np.argmax(np.abs(hedge_gammas))
                if abs(hedge_gammas[primary_hedge_idx]) > 1e-6:
                    initial_hedge_ratios[primary_hedge_idx] = -target_gamma / hedge_gammas[primary_hedge_idx]
            
            # 优化
            result = minimize(
                objective_function,
                initial_hedge_ratios,
                method=self.optimization_method,
                bounds=bounds,
                constraints=constraints_list,
                options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
            )
            
            optimal_hedge_ratios = result.x
            
            # 计算对冲效果
            hedged_gamma = target_gamma + np.dot(optimal_hedge_ratios, hedge_gammas)
            hedge_effectiveness = 1 - abs(hedged_gamma) / abs(target_gamma) if abs(target_gamma) > 1e-6 else 1.0
            
            # 创建对冲头寸
            hedge_positions = []
            hedge_ratios = []
            total_hedge_cost = 0.0
            
            for i, (instrument, ratio) in enumerate(zip(option_instruments, optimal_hedge_ratios)):
                if abs(ratio) > 1e-6:
                    quantity = ratio * instrument.multiplier
                    current_price = kwargs.get('current_prices', {}).get(instrument.symbol, 100.0)
                    
                    gamma = hedge_gammas[i]
                    
                    position = HedgingPosition(
                        instrument=instrument,
                        quantity=quantity,
                        entry_price=current_price,
                        current_price=current_price,
                        unrealized_pnl=0.0,
                        gamma=gamma * quantity
                    )
                    
                    hedge_positions.append(position)
                    
                    hedge_ratio = HedgeRatio(
                        instrument=instrument,
                        ratio=ratio,
                        confidence=0.95,
                        hedge_effectiveness=hedge_effectiveness,
                        basis_risk=0.05,
                        rollover_risk=0.02,
                        liquidity_risk=1.0 - instrument.liquidity_score,
                        cost_of_carry=0.03
                    )
                    
                    hedge_ratios.append(hedge_ratio)
                    
                    hedge_cost = abs(quantity) * instrument.transaction_cost
                    total_hedge_cost += hedge_cost
            
            # 计算风险指标
            portfolio_var_before = await self._calculate_portfolio_variance(
                target_portfolio, **kwargs
            )
            
            # 构建对冲后的组合
            hedged_portfolio = target_portfolio.copy()
            for position in hedge_positions:
                symbol = position.instrument.symbol
                if symbol in hedged_portfolio:
                    hedged_portfolio[symbol] += position.quantity
                else:
                    hedged_portfolio[symbol] = position.quantity
            
            portfolio_var_after = await self._calculate_portfolio_variance(
                hedged_portfolio, **kwargs
            )
            
            risk_reduction = (portfolio_var_before - portfolio_var_after) / portfolio_var_before
            
            return HedgingResult(
                strategy=HedgingStrategy.GAMMA_HEDGING,
                target_exposure=target_gamma,
                hedged_exposure=hedged_gamma,
                hedge_effectiveness=hedge_effectiveness,
                basis_risk=np.mean([hr.basis_risk for hr in hedge_ratios]) if hedge_ratios else 0.0,
                residual_risk=abs(hedged_gamma),
                hedge_positions=hedge_positions,
                hedge_ratios=hedge_ratios,
                total_hedge_cost=total_hedge_cost,
                portfolio_var_before=portfolio_var_before,
                portfolio_var_after=portfolio_var_after,
                risk_reduction=risk_reduction,
                sharpe_improvement=risk_reduction * 0.1,
                tracking_error=np.sqrt(portfolio_var_after),
                rebalancing_frequency=self.rebalancing_frequency,
                optimization_success=result.success
            )
            
        except Exception as e:
            self.logger.error(f"Error in gamma hedging: {e}")
            raise
    
    async def _create_volatility_hedge(self, 
                                     target_portfolio: Dict[str, float],
                                     hedge_instruments: List[HedgingInstrument],
                                     constraints: Optional[Dict[str, Any]] = None,
                                     **kwargs) -> HedgingResult:
        """
        创建波动率对冲策略
        """
        try:
            # 计算目标组合的波动率暴露
            target_vega = await self._calculate_portfolio_vega(target_portfolio, **kwargs)
            
            # 选择波动率敏感工具
            volatility_instruments = [
                inst for inst in hedge_instruments 
                if inst.instrument_type in ['option', 'volatility_swap', 'variance_swap']
            ]
            
            if not volatility_instruments:
                raise ValueError("No volatility-sensitive instruments available")
            
            # 计算各工具的Vega
            hedge_vegas = []
            for instrument in volatility_instruments:
                vega = await self._calculate_instrument_vega(instrument, **kwargs)
                hedge_vegas.append(vega)
            
            hedge_vegas = np.array(hedge_vegas)
            
            # 优化波动率对冲比率
            def objective_function(hedge_ratios):
                # 计算对冲后的Vega
                hedged_vega = target_vega + np.dot(hedge_ratios, hedge_vegas)
                
                # 计算对冲成本
                hedge_cost = np.sum(np.abs(hedge_ratios) * np.array([
                    inst.transaction_cost * inst.multiplier 
                    for inst in volatility_instruments
                ]))
                
                # 目标：最小化剩余Vega + 对冲成本
                return hedged_vega ** 2 + self.transaction_cost_rate * hedge_cost
            
            # 约束条件
            constraints_list = []
            bounds = [(self.min_hedge_ratio, self.max_hedge_ratio) 
                     for _ in volatility_instruments]
            
            # 初始对冲比率
            initial_hedge_ratios = np.zeros(len(volatility_instruments))
            
            if abs(target_vega) > 1e-6 and len(hedge_vegas) > 0:
                primary_hedge_idx = np.argmax(np.abs(hedge_vegas))
                if abs(hedge_vegas[primary_hedge_idx]) > 1e-6:
                    initial_hedge_ratios[primary_hedge_idx] = -target_vega / hedge_vegas[primary_hedge_idx]
            
            # 优化
            result = minimize(
                objective_function,
                initial_hedge_ratios,
                method=self.optimization_method,
                bounds=bounds,
                constraints=constraints_list,
                options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
            )
            
            optimal_hedge_ratios = result.x
            
            # 计算对冲效果
            hedged_vega = target_vega + np.dot(optimal_hedge_ratios, hedge_vegas)
            hedge_effectiveness = 1 - abs(hedged_vega) / abs(target_vega) if abs(target_vega) > 1e-6 else 1.0
            
            # 创建对冲头寸
            hedge_positions = []
            hedge_ratios = []
            total_hedge_cost = 0.0
            
            for i, (instrument, ratio) in enumerate(zip(volatility_instruments, optimal_hedge_ratios)):
                if abs(ratio) > 1e-6:
                    quantity = ratio * instrument.multiplier
                    current_price = kwargs.get('current_prices', {}).get(instrument.symbol, 100.0)
                    
                    vega = hedge_vegas[i]
                    
                    position = HedgingPosition(
                        instrument=instrument,
                        quantity=quantity,
                        entry_price=current_price,
                        current_price=current_price,
                        unrealized_pnl=0.0,
                        vega=vega * quantity
                    )
                    
                    hedge_positions.append(position)
                    
                    hedge_ratio = HedgeRatio(
                        instrument=instrument,
                        ratio=ratio,
                        confidence=0.95,
                        hedge_effectiveness=hedge_effectiveness,
                        basis_risk=0.08,  # 波动率基差风险通常较高
                        rollover_risk=0.03,
                        liquidity_risk=1.0 - instrument.liquidity_score,
                        cost_of_carry=0.04
                    )
                    
                    hedge_ratios.append(hedge_ratio)
                    
                    hedge_cost = abs(quantity) * instrument.transaction_cost
                    total_hedge_cost += hedge_cost
            
            # 计算风险指标
            portfolio_var_before = await self._calculate_portfolio_variance(
                target_portfolio, **kwargs
            )
            
            # 构建对冲后的组合
            hedged_portfolio = target_portfolio.copy()
            for position in hedge_positions:
                symbol = position.instrument.symbol
                if symbol in hedged_portfolio:
                    hedged_portfolio[symbol] += position.quantity
                else:
                    hedged_portfolio[symbol] = position.quantity
            
            portfolio_var_after = await self._calculate_portfolio_variance(
                hedged_portfolio, **kwargs
            )
            
            risk_reduction = (portfolio_var_before - portfolio_var_after) / portfolio_var_before
            
            return HedgingResult(
                strategy=HedgingStrategy.VOLATILITY_HEDGING,
                target_exposure=target_vega,
                hedged_exposure=hedged_vega,
                hedge_effectiveness=hedge_effectiveness,
                basis_risk=np.mean([hr.basis_risk for hr in hedge_ratios]) if hedge_ratios else 0.0,
                residual_risk=abs(hedged_vega),
                hedge_positions=hedge_positions,
                hedge_ratios=hedge_ratios,
                total_hedge_cost=total_hedge_cost,
                portfolio_var_before=portfolio_var_before,
                portfolio_var_after=portfolio_var_after,
                risk_reduction=risk_reduction,
                sharpe_improvement=risk_reduction * 0.12,  # 波动率对冲通常有更好的夏普改善
                tracking_error=np.sqrt(portfolio_var_after),
                rebalancing_frequency=self.rebalancing_frequency,
                optimization_success=result.success
            )
            
        except Exception as e:
            self.logger.error(f"Error in volatility hedging: {e}")
            raise
    
    async def _create_minimum_variance_hedge(self, 
                                           target_portfolio: Dict[str, float],
                                           hedge_instruments: List[HedgingInstrument],
                                           constraints: Optional[Dict[str, Any]] = None,
                                           **kwargs) -> HedgingResult:
        """
        创建最小方差对冲策略
        """
        try:
            # 获取或构建协方差矩阵
            covariance_matrix = await self._get_covariance_matrix(
                list(target_portfolio.keys()) + [inst.symbol for inst in hedge_instruments],
                **kwargs
            )
            
            # 构建目标向量（目标组合权重）
            all_symbols = list(target_portfolio.keys()) + [inst.symbol for inst in hedge_instruments]
            target_weights = np.zeros(len(all_symbols))
            
            for i, symbol in enumerate(all_symbols):
                if symbol in target_portfolio:
                    target_weights[i] = target_portfolio[symbol]
            
            n_target = len(target_portfolio)
            n_hedge = len(hedge_instruments)
            
            # 分离目标组合和对冲工具的协方差
            Sigma_11 = covariance_matrix[:n_target, :n_target]  # 目标组合内部协方差
            Sigma_12 = covariance_matrix[:n_target, n_target:]  # 目标组合与对冲工具的协方差
            Sigma_22 = covariance_matrix[n_target:, n_target:]  # 对冲工具内部协方差
            
            # 最小方差对冲比率计算
            # h* = -Sigma_22^(-1) * Sigma_12^T * w_target
            target_weights_only = target_weights[:n_target]
            
            try:
                # 计算对冲比率
                optimal_hedge_ratios = -np.linalg.solve(
                    Sigma_22, 
                    np.dot(Sigma_12.T, target_weights_only)
                )
            except np.linalg.LinAlgError:
                # 如果矩阵奇异，使用伪逆
                optimal_hedge_ratios = -np.dot(
                    np.linalg.pinv(Sigma_22), 
                    np.dot(Sigma_12.T, target_weights_only)
                )
            
            # 应用约束
            if constraints:
                # 对冲比率界限
                min_ratio = constraints.get('min_hedge_ratio', self.min_hedge_ratio)
                max_ratio = constraints.get('max_hedge_ratio', self.max_hedge_ratio)
                optimal_hedge_ratios = np.clip(optimal_hedge_ratios, min_ratio, max_ratio)
            
            # 计算对冲效果
            # 对冲前方差
            portfolio_var_before = np.dot(target_weights_only, np.dot(Sigma_11, target_weights_only))
            
            # 对冲后方差
            hedged_weights = np.concatenate([target_weights_only, optimal_hedge_ratios])
            portfolio_var_after = np.dot(hedged_weights, np.dot(covariance_matrix, hedged_weights))
            
            hedge_effectiveness = (portfolio_var_before - portfolio_var_after) / portfolio_var_before
            
            # 创建对冲头寸
            hedge_positions = []
            hedge_ratios = []
            total_hedge_cost = 0.0
            
            for i, (instrument, ratio) in enumerate(zip(hedge_instruments, optimal_hedge_ratios)):
                if abs(ratio) > 1e-6:
                    quantity = ratio * instrument.multiplier
                    current_price = kwargs.get('current_prices', {}).get(instrument.symbol, 100.0)
                    
                    position = HedgingPosition(
                        instrument=instrument,
                        quantity=quantity,
                        entry_price=current_price,
                        current_price=current_price,
                        unrealized_pnl=0.0
                    )
                    
                    hedge_positions.append(position)
                    
                    hedge_ratio = HedgeRatio(
                        instrument=instrument,
                        ratio=ratio,
                        confidence=0.98,  # 最小方差对冲理论上更可靠
                        hedge_effectiveness=hedge_effectiveness,
                        basis_risk=0.03,
                        rollover_risk=0.02,
                        liquidity_risk=1.0 - instrument.liquidity_score,
                        cost_of_carry=0.03
                    )
                    
                    hedge_ratios.append(hedge_ratio)
                    
                    hedge_cost = abs(quantity) * instrument.transaction_cost
                    total_hedge_cost += hedge_cost
            
            risk_reduction = (portfolio_var_before - portfolio_var_after) / portfolio_var_before
            
            return HedgingResult(
                strategy=HedgingStrategy.MINIMUM_VARIANCE_HEDGING,
                target_exposure=np.sqrt(portfolio_var_before),
                hedged_exposure=np.sqrt(portfolio_var_after),
                hedge_effectiveness=hedge_effectiveness,
                basis_risk=np.mean([hr.basis_risk for hr in hedge_ratios]) if hedge_ratios else 0.0,
                residual_risk=np.sqrt(portfolio_var_after),
                hedge_positions=hedge_positions,
                hedge_ratios=hedge_ratios,
                total_hedge_cost=total_hedge_cost,
                portfolio_var_before=portfolio_var_before,
                portfolio_var_after=portfolio_var_after,
                risk_reduction=risk_reduction,
                sharpe_improvement=risk_reduction * 0.15,  # 最小方差对冲通常有较好的夏普改善
                tracking_error=np.sqrt(portfolio_var_after),
                rebalancing_frequency=self.rebalancing_frequency,
                optimization_success=True
            )
            
        except Exception as e:
            self.logger.error(f"Error in minimum variance hedging: {e}")
            raise
    
    async def _create_cross_hedge(self, 
                                target_portfolio: Dict[str, float],
                                hedge_instruments: List[HedgingInstrument],
                                constraints: Optional[Dict[str, Any]] = None,
                                **kwargs) -> HedgingResult:
        """
        创建交叉对冲策略
        """
        try:
            # 交叉对冲通常用于不同资产类别或市场之间的对冲
            # 计算目标组合与对冲工具之间的相关性
            
            target_symbols = list(target_portfolio.keys())
            hedge_symbols = [inst.symbol for inst in hedge_instruments]
            
            # 获取相关性矩阵
            correlation_matrix = await self._get_correlation_matrix(
                target_symbols + hedge_symbols, **kwargs
            )
            
            n_target = len(target_symbols)
            n_hedge = len(hedge_symbols)
            
            # 分离目标组合和对冲工具的相关性
            target_hedge_corr = correlation_matrix[:n_target, n_target:]
            
            # 计算目标组合的综合暴露
            target_weights = np.array([target_portfolio[symbol] for symbol in target_symbols])
            
            # 计算每个对冲工具与目标组合的相关性
            portfolio_hedge_corr = np.dot(target_weights, target_hedge_corr)
            
            # 基于相关性和流动性选择最佳对冲工具
            hedge_scores = []
            for i, instrument in enumerate(hedge_instruments):
                correlation = abs(portfolio_hedge_corr[i])
                liquidity = instrument.liquidity_score
                cost = 1.0 / (1.0 + instrument.transaction_cost)
                
                # 综合评分
                score = correlation * liquidity * cost
                hedge_scores.append(score)
            
            # 选择评分最高的对冲工具
            best_hedge_indices = np.argsort(hedge_scores)[-min(5, len(hedge_instruments)):]
            selected_instruments = [hedge_instruments[i] for i in best_hedge_indices]
            selected_correlations = [portfolio_hedge_corr[i] for i in best_hedge_indices]
            
            # 优化交叉对冲比率
            def objective_function(hedge_ratios):
                # 计算跟踪误差
                hedge_exposure = np.dot(hedge_ratios, selected_correlations)
                tracking_error = (1.0 - hedge_exposure) ** 2
                
                # 计算对冲成本
                hedge_cost = np.sum(np.abs(hedge_ratios) * np.array([
                    inst.transaction_cost * inst.multiplier 
                    for inst in selected_instruments
                ]))
                
                # 目标：最小化跟踪误差 + 对冲成本
                return tracking_error + self.transaction_cost_rate * hedge_cost
            
            # 约束条件
            constraints_list = []
            bounds = [(self.min_hedge_ratio, self.max_hedge_ratio) 
                     for _ in selected_instruments]
            
            # 初始对冲比率
            initial_hedge_ratios = np.array([
                1.0 / corr if abs(corr) > 1e-6 else 1.0 
                for corr in selected_correlations
            ])
            initial_hedge_ratios = np.clip(initial_hedge_ratios, -1.0, 1.0)
            
            # 优化
            result = minimize(
                objective_function,
                initial_hedge_ratios,
                method=self.optimization_method,
                bounds=bounds,
                constraints=constraints_list,
                options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
            )
            
            optimal_hedge_ratios = result.x
            
            # 计算对冲效果
            hedge_exposure = np.dot(optimal_hedge_ratios, selected_correlations)
            hedge_effectiveness = abs(hedge_exposure)
            
            # 创建对冲头寸
            hedge_positions = []
            hedge_ratios = []
            total_hedge_cost = 0.0
            
            for i, (instrument, ratio) in enumerate(zip(selected_instruments, optimal_hedge_ratios)):
                if abs(ratio) > 1e-6:
                    quantity = ratio * instrument.multiplier
                    current_price = kwargs.get('current_prices', {}).get(instrument.symbol, 100.0)
                    
                    position = HedgingPosition(
                        instrument=instrument,
                        quantity=quantity,
                        entry_price=current_price,
                        current_price=current_price,
                        unrealized_pnl=0.0
                    )
                    
                    hedge_positions.append(position)
                    
                    hedge_ratio = HedgeRatio(
                        instrument=instrument,
                        ratio=ratio,
                        confidence=0.85,  # 交叉对冲置信度通常较低
                        hedge_effectiveness=hedge_effectiveness,
                        basis_risk=0.15,  # 交叉对冲基差风险较高
                        rollover_risk=0.05,
                        liquidity_risk=1.0 - instrument.liquidity_score,
                        cost_of_carry=0.04
                    )
                    
                    hedge_ratios.append(hedge_ratio)
                    
                    hedge_cost = abs(quantity) * instrument.transaction_cost
                    total_hedge_cost += hedge_cost
            
            # 计算风险指标
            portfolio_var_before = await self._calculate_portfolio_variance(
                target_portfolio, **kwargs
            )
            
            # 构建对冲后的组合
            hedged_portfolio = target_portfolio.copy()
            for position in hedge_positions:
                symbol = position.instrument.symbol
                if symbol in hedged_portfolio:
                    hedged_portfolio[symbol] += position.quantity
                else:
                    hedged_portfolio[symbol] = position.quantity
            
            portfolio_var_after = await self._calculate_portfolio_variance(
                hedged_portfolio, **kwargs
            )
            
            risk_reduction = (portfolio_var_before - portfolio_var_after) / portfolio_var_before
            
            return HedgingResult(
                strategy=HedgingStrategy.CROSS_HEDGING,
                target_exposure=1.0,  # 标准化暴露
                hedged_exposure=hedge_exposure,
                hedge_effectiveness=hedge_effectiveness,
                basis_risk=np.mean([hr.basis_risk for hr in hedge_ratios]) if hedge_ratios else 0.0,
                residual_risk=1.0 - hedge_effectiveness,
                hedge_positions=hedge_positions,
                hedge_ratios=hedge_ratios,
                total_hedge_cost=total_hedge_cost,
                portfolio_var_before=portfolio_var_before,
                portfolio_var_after=portfolio_var_after,
                risk_reduction=risk_reduction,
                sharpe_improvement=risk_reduction * 0.08,  # 交叉对冲夏普改善相对较小
                tracking_error=np.sqrt(portfolio_var_after),
                rebalancing_frequency=self.rebalancing_frequency,
                optimization_success=result.success
            )
            
        except Exception as e:
            self.logger.error(f"Error in cross hedging: {e}")
            raise
    
    async def _create_optimal_hedge(self, 
                                  target_portfolio: Dict[str, float],
                                  hedge_instruments: List[HedgingInstrument],
                                  constraints: Optional[Dict[str, Any]] = None,
                                  **kwargs) -> HedgingResult:
        """
        创建最优对冲策略（综合考虑多种因素）
        """
        try:
            # 获取协方差矩阵
            all_symbols = list(target_portfolio.keys()) + [inst.symbol for inst in hedge_instruments]
            covariance_matrix = await self._get_covariance_matrix(all_symbols, **kwargs)
            
            # 构建目标向量
            target_weights = np.zeros(len(all_symbols))
            for i, symbol in enumerate(all_symbols):
                if symbol in target_portfolio:
                    target_weights[i] = target_portfolio[symbol]
            
            n_target = len(target_portfolio)
            n_hedge = len(hedge_instruments)
            
            # 分离协方差矩阵
            Sigma_11 = covariance_matrix[:n_target, :n_target]
            Sigma_12 = covariance_matrix[:n_target, n_target:]
            Sigma_22 = covariance_matrix[n_target:, n_target:]
            
            target_weights_only = target_weights[:n_target]
            
            # 最优对冲函数
            def objective_function(hedge_ratios):
                # 构建完整的权重向量
                full_weights = np.concatenate([target_weights_only, hedge_ratios])
                
                # 计算组合方差
                portfolio_variance = np.dot(full_weights, np.dot(covariance_matrix, full_weights))
                
                # 计算对冲成本
                hedge_cost = 0.0
                for i, (instrument, ratio) in enumerate(zip(hedge_instruments, hedge_ratios)):
                    # 交易成本
                    transaction_cost = abs(ratio) * instrument.transaction_cost * instrument.multiplier
                    
                    # 流动性成本
                    liquidity_cost = abs(ratio) * (1.0 - instrument.liquidity_score) * 0.001
                    
                    # 持有成本
                    holding_cost = abs(ratio) * 0.03 / 252  # 年化3%的持有成本
                    
                    hedge_cost += transaction_cost + liquidity_cost + holding_cost
                
                # 风险预算惩罚
                risk_penalty = 0.0
                if constraints and 'max_risk_budget' in constraints:
                    max_risk = constraints['max_risk_budget']
                    if portfolio_variance > max_risk:
                        risk_penalty = 1000 * (portfolio_variance - max_risk)
                
                # 集中度惩罚
                concentration_penalty = 0.0
                if constraints and 'max_concentration' in constraints:
                    max_conc = constraints['max_concentration']
                    max_weight = np.max(np.abs(hedge_ratios))
                    if max_weight > max_conc:
                        concentration_penalty = 1000 * (max_weight - max_conc)
                
                # 综合目标函数
                return (portfolio_variance + 
                       self.transaction_cost_rate * hedge_cost + 
                       risk_penalty + 
                       concentration_penalty)
            
            # 约束条件
            constraints_list = []
            bounds = [(self.min_hedge_ratio, self.max_hedge_ratio) 
                     for _ in hedge_instruments]
            
            # 对冲预算约束
            if constraints and 'max_hedge_notional' in constraints:
                def hedge_budget_constraint(hedge_ratios):
                    total_notional = np.sum(np.abs(hedge_ratios) * np.array([
                        inst.multiplier for inst in hedge_instruments
                    ]))
                    return constraints['max_hedge_notional'] - total_notional
                
                constraints_list.append({
                    'type': 'ineq',
                    'fun': hedge_budget_constraint
                })
            
            # 风险降低约束
            if constraints and 'min_risk_reduction' in constraints:
                portfolio_var_before = np.dot(target_weights_only, np.dot(Sigma_11, target_weights_only))
                
                def risk_reduction_constraint(hedge_ratios):
                    full_weights = np.concatenate([target_weights_only, hedge_ratios])
                    portfolio_var_after = np.dot(full_weights, np.dot(covariance_matrix, full_weights))
                    risk_reduction = (portfolio_var_before - portfolio_var_after) / portfolio_var_before
                    return risk_reduction - constraints['min_risk_reduction']
                
                constraints_list.append({
                    'type': 'ineq',
                    'fun': risk_reduction_constraint
                })
            
            # 使用最小方差对冲作为初始值
            try:
                initial_hedge_ratios = -np.linalg.solve(
                    Sigma_22 + np.eye(n_hedge) * 1e-6,  # 添加正则化
                    np.dot(Sigma_12.T, target_weights_only)
                )
            except np.linalg.LinAlgError:
                initial_hedge_ratios = np.zeros(n_hedge)
            
            # 限制初始值范围
            initial_hedge_ratios = np.clip(initial_hedge_ratios, -1.0, 1.0)
            
            # 优化
            result = minimize(
                objective_function,
                initial_hedge_ratios,
                method=self.optimization_method,
                bounds=bounds,
                constraints=constraints_list,
                options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
            )
            
            optimal_hedge_ratios = result.x
            
            # 计算对冲效果
            portfolio_var_before = np.dot(target_weights_only, np.dot(Sigma_11, target_weights_only))
            
            hedged_weights = np.concatenate([target_weights_only, optimal_hedge_ratios])
            portfolio_var_after = np.dot(hedged_weights, np.dot(covariance_matrix, hedged_weights))
            
            hedge_effectiveness = (portfolio_var_before - portfolio_var_after) / portfolio_var_before
            
            # 创建对冲头寸
            hedge_positions = []
            hedge_ratios = []
            total_hedge_cost = 0.0
            
            for i, (instrument, ratio) in enumerate(zip(hedge_instruments, optimal_hedge_ratios)):
                if abs(ratio) > 1e-6:
                    quantity = ratio * instrument.multiplier
                    current_price = kwargs.get('current_prices', {}).get(instrument.symbol, 100.0)
                    
                    position = HedgingPosition(
                        instrument=instrument,
                        quantity=quantity,
                        entry_price=current_price,
                        current_price=current_price,
                        unrealized_pnl=0.0
                    )
                    
                    hedge_positions.append(position)
                    
                    hedge_ratio = HedgeRatio(
                        instrument=instrument,
                        ratio=ratio,
                        confidence=0.95,
                        hedge_effectiveness=hedge_effectiveness,
                        basis_risk=0.05,
                        rollover_risk=0.02,
                        liquidity_risk=1.0 - instrument.liquidity_score,
                        cost_of_carry=0.03
                    )
                    
                    hedge_ratios.append(hedge_ratio)
                    
                    hedge_cost = abs(quantity) * instrument.transaction_cost
                    total_hedge_cost += hedge_cost
            
            risk_reduction = (portfolio_var_before - portfolio_var_after) / portfolio_var_before
            
            return HedgingResult(
                strategy=HedgingStrategy.OPTIMAL_HEDGING,
                target_exposure=np.sqrt(portfolio_var_before),
                hedged_exposure=np.sqrt(portfolio_var_after),
                hedge_effectiveness=hedge_effectiveness,
                basis_risk=np.mean([hr.basis_risk for hr in hedge_ratios]) if hedge_ratios else 0.0,
                residual_risk=np.sqrt(portfolio_var_after),
                hedge_positions=hedge_positions,
                hedge_ratios=hedge_ratios,
                total_hedge_cost=total_hedge_cost,
                portfolio_var_before=portfolio_var_before,
                portfolio_var_after=portfolio_var_after,
                risk_reduction=risk_reduction,
                sharpe_improvement=risk_reduction * 0.2,  # 最优对冲应该有最好的夏普改善
                tracking_error=np.sqrt(portfolio_var_after),
                rebalancing_frequency=self.rebalancing_frequency,
                optimization_success=result.success
            )
            
        except Exception as e:
            self.logger.error(f"Error in optimal hedging: {e}")
            raise
    
    async def _create_pairs_hedge(self, 
                                target_portfolio: Dict[str, float],
                                hedge_instruments: List[HedgingInstrument],
                                constraints: Optional[Dict[str, Any]] = None,
                                **kwargs) -> HedgingResult:
        """
        创建配对对冲策略
        """
        try:
            # 配对对冲策略用于对冲相关性较高的资产对
            target_symbols = list(target_portfolio.keys())
            hedge_symbols = [inst.symbol for inst in hedge_instruments]
            
            # 获取相关性矩阵
            correlation_matrix = await self._get_correlation_matrix(
                target_symbols + hedge_symbols, **kwargs
            )
            
            # 找到最佳配对
            best_pairs = []
            n_target = len(target_symbols)
            
            for i, target_symbol in enumerate(target_symbols):
                target_weight = target_portfolio[target_symbol]
                
                # 找到与目标资产相关性最高的对冲工具
                correlations = correlation_matrix[i, n_target:]
                best_hedge_idx = np.argmax(np.abs(correlations))
                best_correlation = correlations[best_hedge_idx]
                
                if abs(best_correlation) > 0.5:  # 相关性阈值
                    best_pairs.append({
                        'target_symbol': target_symbol,
                        'target_weight': target_weight,
                        'hedge_instrument': hedge_instruments[best_hedge_idx],
                        'correlation': best_correlation
                    })
            
            if not best_pairs:
                raise ValueError("No suitable pairs found for pairs hedging")
            
            # 为每个配对计算对冲比率
            hedge_positions = []
            hedge_ratios = []
            total_hedge_cost = 0.0
            
            for pair in best_pairs:
                target_symbol = pair['target_symbol']
                target_weight = pair['target_weight']
                hedge_instrument = pair['hedge_instrument']
                correlation = pair['correlation']
                
                # 获取价格波动率
                target_vol = await self._get_volatility(target_symbol, **kwargs)
                hedge_vol = await self._get_volatility(hedge_instrument.symbol, **kwargs)
                
                # 计算对冲比率
                hedge_ratio = -correlation * (target_vol / hedge_vol) * target_weight
                
                # 应用约束
                hedge_ratio = np.clip(hedge_ratio, self.min_hedge_ratio, self.max_hedge_ratio)
                
                if abs(hedge_ratio) > 1e-6:
                    quantity = hedge_ratio * hedge_instrument.multiplier
                    current_price = kwargs.get('current_prices', {}).get(hedge_instrument.symbol, 100.0)
                    
                    position = HedgingPosition(
                        instrument=hedge_instrument,
                        quantity=quantity,
                        entry_price=current_price,
                        current_price=current_price,
                        unrealized_pnl=0.0
                    )
                    
                    hedge_positions.append(position)
                    
                    hedge_ratio_obj = HedgeRatio(
                        instrument=hedge_instrument,
                        ratio=hedge_ratio,
                        confidence=0.90,
                        hedge_effectiveness=abs(correlation),
                        basis_risk=0.1 * (1 - abs(correlation)),
                        rollover_risk=0.02,
                        liquidity_risk=1.0 - hedge_instrument.liquidity_score,
                        cost_of_carry=0.03
                    )
                    
                    hedge_ratios.append(hedge_ratio_obj)
                    
                    hedge_cost = abs(quantity) * hedge_instrument.transaction_cost
                    total_hedge_cost += hedge_cost
            
            # 计算整体对冲效果
            overall_hedge_effectiveness = np.mean([hr.hedge_effectiveness for hr in hedge_ratios])
            
            # 计算风险指标
            portfolio_var_before = await self._calculate_portfolio_variance(
                target_portfolio, **kwargs
            )
            
            # 构建对冲后的组合
            hedged_portfolio = target_portfolio.copy()
            for position in hedge_positions:
                symbol = position.instrument.symbol
                if symbol in hedged_portfolio:
                    hedged_portfolio[symbol] += position.quantity
                else:
                    hedged_portfolio[symbol] = position.quantity
            
            portfolio_var_after = await self._calculate_portfolio_variance(
                hedged_portfolio, **kwargs
            )
            
            risk_reduction = (portfolio_var_before - portfolio_var_after) / portfolio_var_before
            
            return HedgingResult(
                strategy=HedgingStrategy.PAIRS_HEDGING,
                target_exposure=np.sqrt(portfolio_var_before),
                hedged_exposure=np.sqrt(portfolio_var_after),
                hedge_effectiveness=overall_hedge_effectiveness,
                basis_risk=np.mean([hr.basis_risk for hr in hedge_ratios]) if hedge_ratios else 0.0,
                residual_risk=np.sqrt(portfolio_var_after),
                hedge_positions=hedge_positions,
                hedge_ratios=hedge_ratios,
                total_hedge_cost=total_hedge_cost,
                portfolio_var_before=portfolio_var_before,
                portfolio_var_after=portfolio_var_after,
                risk_reduction=risk_reduction,
                sharpe_improvement=risk_reduction * 0.12,
                tracking_error=np.sqrt(portfolio_var_after),
                rebalancing_frequency=self.rebalancing_frequency,
                optimization_success=True
            )
            
        except Exception as e:
            self.logger.error(f"Error in pairs hedging: {e}")
            raise
    
    async def rebalance_hedge(self, 
                            hedging_result: HedgingResult,
                            new_market_data: Dict[str, Any],
                            **kwargs) -> HedgingResult:
        """
        重新平衡对冲头寸
        """
        try:
            self.logger.info(f"Rebalancing hedge positions for {hedging_result.strategy.value}")
            
            # 更新市场数据
            await self._update_market_data(new_market_data)
            
            # 重新计算对冲比率
            target_portfolio = {}
            for position in hedging_result.hedge_positions:
                symbol = position.instrument.symbol
                target_portfolio[symbol] = position.quantity
            
            # 重新创建对冲策略
            hedge_instruments = [position.instrument for position in hedging_result.hedge_positions]
            
            new_result = await self.create_hedge_strategy(
                target_portfolio=target_portfolio,
                hedge_instruments=hedge_instruments,
                strategy=hedging_result.strategy,
                **kwargs
            )
            
            # 计算调整量
            position_adjustments = []
            for old_pos, new_pos in zip(hedging_result.hedge_positions, new_result.hedge_positions):
                adjustment = new_pos.quantity - old_pos.quantity
                if abs(adjustment) > 1e-6:
                    position_adjustments.append({
                        'instrument': old_pos.instrument,
                        'old_quantity': old_pos.quantity,
                        'new_quantity': new_pos.quantity,
                        'adjustment': adjustment
                    })
            
            new_result.metadata['position_adjustments'] = position_adjustments
            new_result.metadata['rebalance_trigger'] = kwargs.get('rebalance_trigger', 'scheduled')
            
            return new_result
            
        except Exception as e:
            self.logger.error(f"Error rebalancing hedge: {e}")
            raise
    
    async def monitor_hedge_performance(self, 
                                      hedging_result: HedgingResult,
                                      market_data: Dict[str, Any],
                                      **kwargs) -> Dict[str, Any]:
        """
        监控对冲表现
        """
        try:
            # 计算当前P&L
            current_pnl = 0.0
            position_pnls = []
            
            for position in hedging_result.hedge_positions:
                symbol = position.instrument.symbol
                current_price = market_data.get('prices', {}).get(symbol, position.current_price)
                
                # 计算未实现盈亏
                pnl = (current_price - position.entry_price) * position.quantity
                current_pnl += pnl
                
                position_pnls.append({
                    'symbol': symbol,
                    'quantity': position.quantity,
                    'entry_price': position.entry_price,
                    'current_price': current_price,
                    'pnl': pnl,
                    'pnl_percent': pnl / (position.entry_price * abs(position.quantity)) if position.quantity != 0 else 0
                })
            
            # 计算对冲效果
            current_hedge_effectiveness = await self._calculate_current_hedge_effectiveness(
                hedging_result, market_data, **kwargs
            )
            
            # 检查重新平衡触发条件
            rebalance_needed = False
            rebalance_reasons = []
            
            # 对冲效果下降
            if current_hedge_effectiveness < hedging_result.hedge_effectiveness * 0.8:
                rebalance_needed = True
                rebalance_reasons.append("hedge_effectiveness_decline")
            
            # 头寸偏离过大
            max_position_drift = max([
                abs(pos['pnl_percent']) for pos in position_pnls
            ]) if position_pnls else 0
            
            if max_position_drift > 0.1:  # 10%的偏离阈值
                rebalance_needed = True
                rebalance_reasons.append("position_drift")
            
            # 市场波动率变化
            current_volatility = await self._calculate_market_volatility(market_data, **kwargs)
            historical_volatility = hedging_result.metadata.get('historical_volatility', current_volatility)
            
            if abs(current_volatility - historical_volatility) / historical_volatility > 0.2:
                rebalance_needed = True
                rebalance_reasons.append("volatility_change")
            
            # 计算风险指标
            current_var = await self._calculate_current_var(hedging_result, market_data, **kwargs)
            
            performance_metrics = {
                'current_pnl': current_pnl,
                'position_pnls': position_pnls,
                'current_hedge_effectiveness': current_hedge_effectiveness,
                'effectiveness_change': current_hedge_effectiveness - hedging_result.hedge_effectiveness,
                'current_var': current_var,
                'var_change': current_var - hedging_result.portfolio_var_after,
                'rebalance_needed': rebalance_needed,
                'rebalance_reasons': rebalance_reasons,
                'monitoring_timestamp': datetime.now(),
                'market_data_timestamp': market_data.get('timestamp', datetime.now())
            }
            
            return performance_metrics
            
        except Exception as e:
            self.logger.error(f"Error monitoring hedge performance: {e}")
            return {}
    
    # 辅助方法
    async def _calculate_portfolio_delta(self, portfolio: Dict[str, float], **kwargs) -> float:
        """计算组合Delta"""
        try:
            total_delta = 0.0
            for symbol, weight in portfolio.items():
                # 对于股票，Delta = 1
                # 对于期权，需要使用Black-Scholes计算
                if kwargs.get('option_data', {}).get(symbol):
                    option_data = kwargs['option_data'][symbol]
                    delta = self._calculate_black_scholes_delta(option_data)
                else:
                    delta = 1.0  # 假设是股票
                
                total_delta += weight * delta
            
            return total_delta
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio delta: {e}")
            return 0.0
    
    async def _calculate_portfolio_gamma(self, portfolio: Dict[str, float], **kwargs) -> float:
        """计算组合Gamma"""
        try:
            total_gamma = 0.0
            for symbol, weight in portfolio.items():
                if kwargs.get('option_data', {}).get(symbol):
                    option_data = kwargs['option_data'][symbol]
                    gamma = self._calculate_black_scholes_gamma(option_data)
                else:
                    gamma = 0.0  # 股票的Gamma为0
                
                total_gamma += weight * gamma
            
            return total_gamma
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio gamma: {e}")
            return 0.0
    
    async def _calculate_portfolio_vega(self, portfolio: Dict[str, float], **kwargs) -> float:
        """计算组合Vega"""
        try:
            total_vega = 0.0
            for symbol, weight in portfolio.items():
                if kwargs.get('option_data', {}).get(symbol):
                    option_data = kwargs['option_data'][symbol]
                    vega = self._calculate_black_scholes_vega(option_data)
                else:
                    vega = 0.0  # 股票的Vega为0
                
                total_vega += weight * vega
            
            return total_vega
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio vega: {e}")
            return 0.0
    
    async def _calculate_instrument_delta(self, instrument: HedgingInstrument, **kwargs) -> float:
        """计算工具Delta"""
        try:
            if instrument.instrument_type == 'stock':
                return 1.0
            elif instrument.instrument_type == 'option':
                option_data = kwargs.get('option_data', {}).get(instrument.symbol, {})
                return self._calculate_black_scholes_delta(option_data)
            elif instrument.instrument_type == 'future':
                return 1.0
            else:
                return 1.0
                
        except Exception as e:
            self.logger.error(f"Error calculating instrument delta: {e}")
            return 1.0
    
    async def _calculate_instrument_gamma(self, instrument: HedgingInstrument, **kwargs) -> float:
        """计算工具Gamma"""
        try:
            if instrument.instrument_type == 'option':
                option_data = kwargs.get('option_data', {}).get(instrument.symbol, {})
                return self._calculate_black_scholes_gamma(option_data)
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error calculating instrument gamma: {e}")
            return 0.0
    
    async def _calculate_instrument_vega(self, instrument: HedgingInstrument, **kwargs) -> float:
        """计算工具Vega"""
        try:
            if instrument.instrument_type == 'option':
                option_data = kwargs.get('option_data', {}).get(instrument.symbol, {})
                return self._calculate_black_scholes_vega(option_data)
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error calculating instrument vega: {e}")
            return 0.0
    
    def _calculate_black_scholes_delta(self, option_data: Dict[str, Any]) -> float:
        """计算Black-Scholes Delta"""
        try:
            S = option_data.get('underlying_price', 100.0)
            K = option_data.get('strike_price', 100.0)
            T = option_data.get('time_to_expiry', 0.25)
            r = option_data.get('risk_free_rate', 0.05)
            sigma = option_data.get('volatility', 0.2)
            option_type = option_data.get('option_type', 'call')
            
            if T <= 0:
                return 1.0 if option_type == 'call' and S > K else 0.0
            
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            
            if option_type == 'call':
                return norm.cdf(d1)
            else:
                return norm.cdf(d1) - 1.0
                
        except Exception as e:
            self.logger.error(f"Error calculating Black-Scholes delta: {e}")
            return 1.0
    
    def _calculate_black_scholes_gamma(self, option_data: Dict[str, Any]) -> float:
        """计算Black-Scholes Gamma"""
        try:
            S = option_data.get('underlying_price', 100.0)
            K = option_data.get('strike_price', 100.0)
            T = option_data.get('time_to_expiry', 0.25)
            r = option_data.get('risk_free_rate', 0.05)
            sigma = option_data.get('volatility', 0.2)
            
            if T <= 0:
                return 0.0
            
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            
            return norm.pdf(d1) / (S * sigma * np.sqrt(T))
            
        except Exception as e:
            self.logger.error(f"Error calculating Black-Scholes gamma: {e}")
            return 0.0
    
    def _calculate_black_scholes_vega(self, option_data: Dict[str, Any]) -> float:
        """计算Black-Scholes Vega"""
        try:
            S = option_data.get('underlying_price', 100.0)
            K = option_data.get('strike_price', 100.0)
            T = option_data.get('time_to_expiry', 0.25)
            r = option_data.get('risk_free_rate', 0.05)
            sigma = option_data.get('volatility', 0.2)
            
            if T <= 0:
                return 0.0
            
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            
            return S * norm.pdf(d1) * np.sqrt(T)
            
        except Exception as e:
            self.logger.error(f"Error calculating Black-Scholes vega: {e}")
            return 0.0
    
    async def _calculate_portfolio_variance(self, portfolio: Dict[str, float], **kwargs) -> float:
        """计算组合方差"""
        try:
            symbols = list(portfolio.keys())
            weights = np.array(list(portfolio.values()))
            
            # 获取协方差矩阵
            covariance_matrix = await self._get_covariance_matrix(symbols, **kwargs)
            
            # 计算组合方差
            portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
            
            return portfolio_variance
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio variance: {e}")
            return 0.01  # 默认方差
    
    async def _get_covariance_matrix(self, symbols: List[str], **kwargs) -> np.ndarray:
        """获取协方差矩阵"""
        try:
            # 尝试从kwargs中获取
            if 'covariance_matrix' in kwargs:
                return kwargs['covariance_matrix']
            
            # 生成模拟协方差矩阵
            n = len(symbols)
            correlation_matrix = np.random.uniform(0.1, 0.9, (n, n))
            correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
            np.fill_diagonal(correlation_matrix, 1.0)
            
            # 使用随机波动率
            volatilities = np.random.uniform(0.1, 0.3, n)
            
            # 转换为协方差矩阵
            covariance_matrix = np.outer(volatilities, volatilities) * correlation_matrix
            
            return covariance_matrix
            
        except Exception as e:
            self.logger.error(f"Error getting covariance matrix: {e}")
            n = len(symbols)
            return np.eye(n) * 0.04  # 默认协方差矩阵
    
    async def _get_correlation_matrix(self, symbols: List[str], **kwargs) -> np.ndarray:
        """获取相关性矩阵"""
        try:
            # 尝试从kwargs中获取
            if 'correlation_matrix' in kwargs:
                return kwargs['correlation_matrix']
            
            # 生成模拟相关性矩阵
            n = len(symbols)
            correlation_matrix = np.random.uniform(0.1, 0.9, (n, n))
            correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
            np.fill_diagonal(correlation_matrix, 1.0)
            
            return correlation_matrix
            
        except Exception as e:
            self.logger.error(f"Error getting correlation matrix: {e}")
            n = len(symbols)
            return np.eye(n)  # 默认单位矩阵
    
    async def _get_volatility(self, symbol: str, **kwargs) -> float:
        """获取波动率"""
        try:
            if 'volatilities' in kwargs and symbol in kwargs['volatilities']:
                return kwargs['volatilities'][symbol]
            
            # 从缓存中获取
            if symbol in self.volatility_cache:
                return self.volatility_cache[symbol]
            
            # 生成模拟波动率
            volatility = np.random.uniform(0.1, 0.3)
            self.volatility_cache[symbol] = volatility
            
            return volatility
            
        except Exception as e:
            self.logger.error(f"Error getting volatility for {symbol}: {e}")
            return 0.2  # 默认波动率
    
    async def _update_market_data(self, market_data: Dict[str, Any]):
        """更新市场数据"""
        try:
            # 更新价格缓存
            if 'prices' in market_data:
                for symbol, price in market_data['prices'].items():
                    self.market_data_cache[symbol] = price
            
            # 更新波动率缓存
            if 'volatilities' in market_data:
                for symbol, volatility in market_data['volatilities'].items():
                    self.volatility_cache[symbol] = volatility
            
            # 更新相关性缓存
            if 'correlation_matrix' in market_data:
                self.correlation_cache['latest'] = market_data['correlation_matrix']
            
            self.logger.info("Market data updated successfully")
            
        except Exception as e:
            self.logger.error(f"Error updating market data: {e}")
    
    async def _update_hedge_positions(self, hedging_result: HedgingResult):
        """更新对冲头寸记录"""
        try:
            for position in hedging_result.hedge_positions:
                symbol = position.instrument.symbol
                self.hedge_positions[symbol] = position
            
            for hedge_ratio in hedging_result.hedge_ratios:
                symbol = hedge_ratio.instrument.symbol
                self.hedge_ratios[symbol] = hedge_ratio
            
            self.logger.info(f"Updated {len(hedging_result.hedge_positions)} hedge positions")
            
        except Exception as e:
            self.logger.error(f"Error updating hedge positions: {e}")
    
    async def _calculate_current_hedge_effectiveness(self, 
                                                   hedging_result: HedgingResult,
                                                   market_data: Dict[str, Any],
                                                   **kwargs) -> float:
        """计算当前对冲效果"""
        try:
            # 简化的对冲效果计算
            # 实际应用中需要更复杂的计算
            
            # 基于P&L相关性计算
            total_pnl = 0.0
            for position in hedging_result.hedge_positions:
                symbol = position.instrument.symbol
                current_price = market_data.get('prices', {}).get(symbol, position.current_price)
                pnl = (current_price - position.entry_price) * position.quantity
                total_pnl += pnl
            
            # 基于波动率变化调整
            current_volatility = await self._calculate_market_volatility(market_data, **kwargs)
            historical_volatility = hedging_result.metadata.get('historical_volatility', current_volatility)
            
            volatility_adjustment = 1.0 - abs(current_volatility - historical_volatility) / historical_volatility
            
            # 综合对冲效果
            current_effectiveness = hedging_result.hedge_effectiveness * volatility_adjustment
            
            return max(0.0, min(1.0, current_effectiveness))
            
        except Exception as e:
            self.logger.error(f"Error calculating current hedge effectiveness: {e}")
            return hedging_result.hedge_effectiveness
    
    async def _calculate_market_volatility(self, market_data: Dict[str, Any], **kwargs) -> float:
        """计算市场波动率"""
        try:
            if 'market_volatility' in market_data:
                return market_data['market_volatility']
            
            # 从价格数据计算
            if 'price_history' in market_data:
                prices = market_data['price_history']
                returns = np.diff(np.log(prices))
                return np.std(returns) * np.sqrt(252)  # 年化波动率
            
            # 默认波动率
            return 0.2
            
        except Exception as e:
            self.logger.error(f"Error calculating market volatility: {e}")
            return 0.2
    
    async def _calculate_current_var(self, 
                                   hedging_result: HedgingResult,
                                   market_data: Dict[str, Any],
                                   **kwargs) -> float:
        """计算当前VaR"""
        try:
            # 简化的VaR计算
            current_volatility = await self._calculate_market_volatility(market_data, **kwargs)
            
            # 基于正态分布假设
            confidence_level = self.confidence_level
            z_score = norm.ppf(confidence_level)
            
            # 计算当前组合价值
            total_value = 0.0
            for position in hedging_result.hedge_positions:
                symbol = position.instrument.symbol
                current_price = market_data.get('prices', {}).get(symbol, position.current_price)
                value = current_price * abs(position.quantity)
                total_value += value
            
            # VaR计算
            var = total_value * current_volatility * z_score / np.sqrt(252)
            
            return var
            
        except Exception as e:
            self.logger.error(f"Error calculating current VaR: {e}")
            return hedging_result.portfolio_var_after
    
    def _validate_inputs(self, 
                        target_portfolio: Dict[str, float],
                        hedge_instruments: List[HedgingInstrument]) -> bool:
        """验证输入参数"""
        try:
            # 检查目标组合
            if not target_portfolio:
                return False
            
            # 检查权重
            total_weight = sum(abs(weight) for weight in target_portfolio.values())
            if total_weight == 0:
                return False
            
            # 检查对冲工具
            if not hedge_instruments:
                return False
            
            # 检查工具有效性
            for instrument in hedge_instruments:
                if not instrument.symbol:
                    return False
                if instrument.multiplier <= 0:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating inputs: {e}")
            return False
    
    def get_hedge_summary(self) -> Dict[str, Any]:
        """获取对冲摘要"""
        try:
            if not self.hedging_history:
                return {}
            
            recent_hedges = self.hedging_history[-10:]  # 最近10次对冲
            
            summary = {
                'total_hedges': len(self.hedging_history),
                'active_positions': len(self.hedge_positions),
                'strategies_used': list(set(hedge.strategy.value for hedge in recent_hedges)),
                'average_effectiveness': np.mean([hedge.hedge_effectiveness for hedge in recent_hedges]),
                'average_risk_reduction': np.mean([hedge.risk_reduction for hedge in recent_hedges]),
                'total_hedge_cost': sum([hedge.total_hedge_cost for hedge in recent_hedges]),
                'average_sharpe_improvement': np.mean([hedge.sharpe_improvement for hedge in recent_hedges]),
                'rebalancing_frequency': self.rebalancing_frequency.value,
                'last_hedge_date': self.hedging_history[-1].metadata.get('timestamp', datetime.now())
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting hedge summary: {e}")
            return {}
    
    def plot_hedge_performance(self, 
                              hedging_result: HedgingResult,
                              save_path: Optional[str] = None):
        """绘制对冲表现图"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. 对冲效果比较
            categories = ['风险暴露', '对冲成本', '剩余风险']
            before_values = [hedging_result.target_exposure, 0, hedging_result.target_exposure]
            after_values = [hedging_result.hedged_exposure, hedging_result.total_hedge_cost, 
                          hedging_result.residual_risk]
            
            x = np.arange(len(categories))
            width = 0.35
            
            ax1.bar(x - width/2, before_values, width, label='对冲前', alpha=0.7)
            ax1.bar(x + width/2, after_values, width, label='对冲后', alpha=0.7)
            ax1.set_xlabel('指标')
            ax1.set_ylabel('数值')
            ax1.set_title('对冲效果比较')
            ax1.set_xticks(x)
            ax1.set_xticklabels(categories)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. 风险降低效果
            risk_metrics = ['组合方差', '跟踪误差', '风险降低']
            risk_values = [hedging_result.portfolio_var_after, hedging_result.tracking_error, 
                         hedging_result.risk_reduction]
            
            ax2.bar(risk_metrics, risk_values, alpha=0.7, color='green')
            ax2.set_ylabel('数值')
            ax2.set_title('风险指标')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
            
            # 3. 对冲头寸分布
            if hedging_result.hedge_positions:
                instruments = [pos.instrument.symbol for pos in hedging_result.hedge_positions]
                quantities = [abs(pos.quantity) for pos in hedging_result.hedge_positions]
                
                ax3.pie(quantities, labels=instruments, autopct='%1.1f%%', startangle=90)
                ax3.set_title('对冲头寸分布')
            
            # 4. 对冲比率
            if hedging_result.hedge_ratios:
                instruments = [hr.instrument.symbol for hr in hedging_result.hedge_ratios]
                ratios = [hr.ratio for hr in hedging_result.hedge_ratios]
                
                colors = ['red' if r < 0 else 'blue' for r in ratios]
                ax4.bar(instruments, ratios, color=colors, alpha=0.7)
                ax4.set_ylabel('对冲比率')
                ax4.set_title('对冲比率分布')
                ax4.tick_params(axis='x', rotation=45)
                ax4.grid(True, alpha=0.3)
                ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Error plotting hedge performance: {e}")
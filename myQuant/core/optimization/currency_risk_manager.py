import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
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

class CurrencyRiskType(Enum):
    TRANSACTION_RISK = "transaction_risk"
    TRANSLATION_RISK = "translation_risk"
    ECONOMIC_RISK = "economic_risk"
    CONTINGENT_RISK = "contingent_risk"

class HedgingMethod(Enum):
    FORWARD_CONTRACTS = "forward_contracts"
    CURRENCY_OPTIONS = "currency_options"
    CURRENCY_SWAPS = "currency_swaps"
    MONEY_MARKET_HEDGE = "money_market_hedge"
    NATURAL_HEDGING = "natural_hedging"
    NETTING = "netting"
    LEADING_LAGGING = "leading_lagging"
    CURRENCY_DIVERSIFICATION = "currency_diversification"

class RiskMeasure(Enum):
    VALUE_AT_RISK = "value_at_risk"
    CONDITIONAL_VALUE_AT_RISK = "conditional_value_at_risk"
    VOLATILITY = "volatility"
    MAXIMUM_DRAWDOWN = "maximum_drawdown"
    TRACKING_ERROR = "tracking_error"
    DOWNSIDE_DEVIATION = "downside_deviation"

@dataclass
class CurrencyExposure:
    """货币风险暴露"""
    currency: str
    base_currency: str
    exposure_amount: float
    exposure_type: CurrencyRiskType
    maturity_date: Optional[datetime] = None
    confidence_level: float = 0.95
    volatility: float = 0.15
    correlation_with_base: float = 0.0
    liquidity_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CurrencyHedgeInstrument:
    """货币对冲工具"""
    instrument_type: HedgingMethod
    currency_pair: str
    notional_amount: float
    maturity_date: Optional[datetime] = None
    strike_rate: Optional[float] = None
    premium: float = 0.0
    transaction_cost: float = 0.001
    liquidity_score: float = 1.0
    margin_requirement: float = 0.0
    counterparty_risk: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CurrencyHedgePosition:
    """货币对冲头寸"""
    instrument: CurrencyHedgeInstrument
    position_size: float
    entry_rate: float
    current_rate: float
    unrealized_pnl: float
    delta: float = 1.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0
    entry_date: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)

@dataclass
class CurrencyRiskMetrics:
    """货币风险指标"""
    total_exposure: float
    hedged_exposure: float
    residual_exposure: float
    hedge_ratio: float
    hedge_effectiveness: float
    basis_risk: float
    var_95: float
    var_99: float
    cvar_95: float
    volatility: float
    maximum_drawdown: float
    sharpe_ratio: float
    tracking_error: float
    currency_beta: float
    correlation_stability: float

@dataclass
class CurrencyRiskResult:
    """货币风险管理结果"""
    base_currency: str
    total_exposures: List[CurrencyExposure]
    hedge_positions: List[CurrencyHedgePosition]
    risk_metrics: CurrencyRiskMetrics
    hedging_method: HedgingMethod
    total_hedge_cost: float
    net_hedge_benefit: float
    optimization_success: bool
    rebalancing_frequency: str
    backtest_results: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class CurrencyRiskManager:
    """
    货币风险管理器
    
    提供全面的货币风险识别、测量、监控和对冲功能，包括：
    - 交易风险管理
    - 折算风险管理
    - 经济风险管理
    - 多种对冲策略
    - 动态对冲比率优化
    - 风险绩效归因分析
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 基础配置
        self.base_currency = config.get('base_currency', 'USD')
        self.risk_tolerance = config.get('risk_tolerance', 0.05)
        self.confidence_level = config.get('confidence_level', 0.95)
        self.rebalancing_frequency = config.get('rebalancing_frequency', 'monthly')
        
        # 对冲参数
        self.hedge_effectiveness_threshold = config.get('hedge_effectiveness_threshold', 0.8)
        self.max_hedge_ratio = config.get('max_hedge_ratio', 1.0)
        self.min_hedge_ratio = config.get('min_hedge_ratio', 0.0)
        self.hedge_cost_threshold = config.get('hedge_cost_threshold', 0.02)
        
        # 风险参数
        self.lookback_period = config.get('lookback_period', 252)
        self.volatility_window = config.get('volatility_window', 30)
        self.correlation_window = config.get('correlation_window', 60)
        
        # 优化参数
        self.optimization_method = config.get('optimization_method', 'SLSQP')
        self.max_iterations = config.get('max_iterations', 1000)
        self.tolerance = config.get('tolerance', 1e-6)
        
        # 数据存储
        self.currency_exposures: Dict[str, CurrencyExposure] = {}
        self.hedge_positions: Dict[str, CurrencyHedgePosition] = {}
        self.risk_history: List[CurrencyRiskResult] = []
        
        # 市场数据缓存
        self.exchange_rates: Dict[str, float] = {}
        self.volatilities: Dict[str, float] = {}
        self.correlations: Dict[str, Dict[str, float]] = {}
        self.interest_rates: Dict[str, float] = {}
        
        # 执行器
        self.executor = ThreadPoolExecutor(max_workers=config.get('max_workers', 4))
        
        self.logger.info(f"Currency risk manager initialized with base currency: {self.base_currency}")
    
    async def analyze_currency_risk(self, 
                                   exposures: List[CurrencyExposure],
                                   market_data: Dict[str, Any],
                                   **kwargs) -> CurrencyRiskResult:
        """
        分析货币风险
        """
        try:
            self.logger.info(f"Analyzing currency risk for {len(exposures)} exposures")
            
            # 更新市场数据
            await self._update_market_data(market_data)
            
            # 计算风险指标
            risk_metrics = await self._calculate_risk_metrics(exposures, **kwargs)
            
            # 创建基础结果
            result = CurrencyRiskResult(
                base_currency=self.base_currency,
                total_exposures=exposures,
                hedge_positions=[],
                risk_metrics=risk_metrics,
                hedging_method=HedgingMethod.NATURAL_HEDGING,
                total_hedge_cost=0.0,
                net_hedge_benefit=0.0,
                optimization_success=True,
                rebalancing_frequency=self.rebalancing_frequency
            )
            
            # 保存结果
            self.risk_history.append(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing currency risk: {e}")
            raise
    
    async def create_hedge_strategy(self, 
                                   exposures: List[CurrencyExposure],
                                   hedging_method: HedgingMethod,
                                   hedge_instruments: List[CurrencyHedgeInstrument],
                                   constraints: Optional[Dict[str, Any]] = None,
                                   **kwargs) -> CurrencyRiskResult:
        """
        创建货币对冲策略
        """
        try:
            self.logger.info(f"Creating hedge strategy using {hedging_method.value}")
            
            # 分析基础风险
            base_result = await self.analyze_currency_risk(exposures, kwargs.get('market_data', {}))
            
            # 选择对冲方法
            if hedging_method == HedgingMethod.FORWARD_CONTRACTS:
                hedge_result = await self._create_forward_hedge(
                    exposures, hedge_instruments, constraints, **kwargs
                )
            elif hedging_method == HedgingMethod.CURRENCY_OPTIONS:
                hedge_result = await self._create_options_hedge(
                    exposures, hedge_instruments, constraints, **kwargs
                )
            elif hedging_method == HedgingMethod.CURRENCY_SWAPS:
                hedge_result = await self._create_swap_hedge(
                    exposures, hedge_instruments, constraints, **kwargs
                )
            elif hedging_method == HedgingMethod.MONEY_MARKET_HEDGE:
                hedge_result = await self._create_money_market_hedge(
                    exposures, hedge_instruments, constraints, **kwargs
                )
            elif hedging_method == HedgingMethod.NATURAL_HEDGING:
                hedge_result = await self._create_natural_hedge(
                    exposures, hedge_instruments, constraints, **kwargs
                )
            elif hedging_method == HedgingMethod.NETTING:
                hedge_result = await self._create_netting_hedge(
                    exposures, hedge_instruments, constraints, **kwargs
                )
            elif hedging_method == HedgingMethod.CURRENCY_DIVERSIFICATION:
                hedge_result = await self._create_diversification_hedge(
                    exposures, hedge_instruments, constraints, **kwargs
                )
            else:
                raise ValueError(f"Unsupported hedging method: {hedging_method}")
            
            # 计算对冲收益
            net_benefit = (base_result.risk_metrics.var_95 - hedge_result.risk_metrics.var_95) - hedge_result.total_hedge_cost
            hedge_result.net_hedge_benefit = net_benefit
            
            # 保存结果
            self.risk_history.append(hedge_result)
            
            self.logger.info(f"Hedge strategy created with {hedge_result.risk_metrics.hedge_effectiveness:.2%} effectiveness")
            
            return hedge_result
            
        except Exception as e:
            self.logger.error(f"Error creating hedge strategy: {e}")
            raise
    
    async def _create_forward_hedge(self, 
                                   exposures: List[CurrencyExposure],
                                   hedge_instruments: List[CurrencyHedgeInstrument],
                                   constraints: Optional[Dict[str, Any]] = None,
                                   **kwargs) -> CurrencyRiskResult:
        """
        创建远期合约对冲
        """
        try:
            # 筛选远期合约工具
            forward_instruments = [
                inst for inst in hedge_instruments 
                if inst.instrument_type == HedgingMethod.FORWARD_CONTRACTS
            ]
            
            if not forward_instruments:
                raise ValueError("No forward contract instruments available")
            
            # 按货币对分组暴露
            exposure_by_currency = {}
            for exposure in exposures:
                currency_pair = f"{exposure.currency}{exposure.base_currency}"
                if currency_pair not in exposure_by_currency:
                    exposure_by_currency[currency_pair] = []
                exposure_by_currency[currency_pair].append(exposure)
            
            # 为每个货币对创建对冲
            hedge_positions = []
            total_hedge_cost = 0.0
            
            for currency_pair, currency_exposures in exposure_by_currency.items():
                # 计算净暴露
                net_exposure = sum(exp.exposure_amount for exp in currency_exposures)
                
                if abs(net_exposure) < 1e-6:
                    continue  # 暴露已经自然对冲
                
                # 找到匹配的远期合约
                matching_forwards = [
                    inst for inst in forward_instruments 
                    if inst.currency_pair == currency_pair
                ]
                
                if not matching_forwards:
                    continue
                
                # 选择最优的远期合约
                best_forward = max(matching_forwards, key=lambda x: x.liquidity_score)
                
                # 计算对冲头寸大小
                hedge_size = await self._optimize_hedge_size(
                    net_exposure, best_forward, constraints, **kwargs
                )
                
                if abs(hedge_size) > 1e-6:
                    # 获取当前汇率
                    current_rate = self.exchange_rates.get(currency_pair, 1.0)
                    
                    # 创建对冲头寸
                    hedge_position = CurrencyHedgePosition(
                        instrument=best_forward,
                        position_size=hedge_size,
                        entry_rate=current_rate,
                        current_rate=current_rate,
                        unrealized_pnl=0.0,
                        delta=1.0  # 远期合约的Delta为1
                    )
                    
                    hedge_positions.append(hedge_position)
                    
                    # 计算对冲成本
                    hedge_cost = abs(hedge_size) * best_forward.transaction_cost
                    total_hedge_cost += hedge_cost
            
            # 计算对冲后的风险指标
            hedged_exposures = self._calculate_hedged_exposures(exposures, hedge_positions)
            risk_metrics = await self._calculate_risk_metrics(hedged_exposures, **kwargs)
            
            # 计算对冲效果
            original_risk = await self._calculate_risk_metrics(exposures, **kwargs)
            hedge_effectiveness = (original_risk.var_95 - risk_metrics.var_95) / original_risk.var_95
            risk_metrics.hedge_effectiveness = hedge_effectiveness
            
            return CurrencyRiskResult(
                base_currency=self.base_currency,
                total_exposures=exposures,
                hedge_positions=hedge_positions,
                risk_metrics=risk_metrics,
                hedging_method=HedgingMethod.FORWARD_CONTRACTS,
                total_hedge_cost=total_hedge_cost,
                net_hedge_benefit=0.0,
                optimization_success=True,
                rebalancing_frequency=self.rebalancing_frequency
            )
            
        except Exception as e:
            self.logger.error(f"Error creating forward hedge: {e}")
            raise
    
    async def _create_options_hedge(self, 
                                   exposures: List[CurrencyExposure],
                                   hedge_instruments: List[CurrencyHedgeInstrument],
                                   constraints: Optional[Dict[str, Any]] = None,
                                   **kwargs) -> CurrencyRiskResult:
        """
        创建期权对冲
        """
        try:
            # 筛选期权工具
            option_instruments = [
                inst for inst in hedge_instruments 
                if inst.instrument_type == HedgingMethod.CURRENCY_OPTIONS
            ]
            
            if not option_instruments:
                raise ValueError("No currency option instruments available")
            
            # 按货币对分组暴露
            exposure_by_currency = {}
            for exposure in exposures:
                currency_pair = f"{exposure.currency}{exposure.base_currency}"
                if currency_pair not in exposure_by_currency:
                    exposure_by_currency[currency_pair] = []
                exposure_by_currency[currency_pair].append(exposure)
            
            # 为每个货币对创建期权对冲
            hedge_positions = []
            total_hedge_cost = 0.0
            
            for currency_pair, currency_exposures in exposure_by_currency.items():
                # 计算净暴露
                net_exposure = sum(exp.exposure_amount for exp in currency_exposures)
                
                if abs(net_exposure) < 1e-6:
                    continue
                
                # 找到匹配的期权
                matching_options = [
                    inst for inst in option_instruments 
                    if inst.currency_pair == currency_pair
                ]
                
                if not matching_options:
                    continue
                
                # 选择最优期权组合
                optimal_options = await self._optimize_options_portfolio(
                    net_exposure, matching_options, constraints, **kwargs
                )
                
                for option_position in optimal_options:
                    hedge_positions.append(option_position)
                    
                    # 计算期权成本（权利金）
                    option_cost = abs(option_position.position_size) * option_position.instrument.premium
                    total_hedge_cost += option_cost
            
            # 计算对冲后的风险指标
            hedged_exposures = self._calculate_hedged_exposures(exposures, hedge_positions)
            risk_metrics = await self._calculate_risk_metrics(hedged_exposures, **kwargs)
            
            # 计算对冲效果
            original_risk = await self._calculate_risk_metrics(exposures, **kwargs)
            hedge_effectiveness = (original_risk.var_95 - risk_metrics.var_95) / original_risk.var_95
            risk_metrics.hedge_effectiveness = hedge_effectiveness
            
            return CurrencyRiskResult(
                base_currency=self.base_currency,
                total_exposures=exposures,
                hedge_positions=hedge_positions,
                risk_metrics=risk_metrics,
                hedging_method=HedgingMethod.CURRENCY_OPTIONS,
                total_hedge_cost=total_hedge_cost,
                net_hedge_benefit=0.0,
                optimization_success=True,
                rebalancing_frequency=self.rebalancing_frequency
            )
            
        except Exception as e:
            self.logger.error(f"Error creating options hedge: {e}")
            raise
    
    async def _create_natural_hedge(self, 
                                   exposures: List[CurrencyExposure],
                                   hedge_instruments: List[CurrencyHedgeInstrument],
                                   constraints: Optional[Dict[str, Any]] = None,
                                   **kwargs) -> CurrencyRiskResult:
        """
        创建自然对冲
        """
        try:
            # 按货币分组暴露
            exposure_by_currency = {}
            for exposure in exposures:
                if exposure.currency not in exposure_by_currency:
                    exposure_by_currency[exposure.currency] = []
                exposure_by_currency[exposure.currency].append(exposure)
            
            # 寻找自然对冲机会
            natural_hedges = []
            
            for currency, currency_exposures in exposure_by_currency.items():
                # 计算同一货币的正负暴露
                positive_exposures = [exp for exp in currency_exposures if exp.exposure_amount > 0]
                negative_exposures = [exp for exp in currency_exposures if exp.exposure_amount < 0]
                
                positive_total = sum(exp.exposure_amount for exp in positive_exposures)
                negative_total = sum(exp.exposure_amount for exp in negative_exposures)
                
                # 计算自然对冲量
                natural_hedge_amount = min(positive_total, abs(negative_total))
                
                if natural_hedge_amount > 0:
                    natural_hedges.append({
                        'currency': currency,
                        'hedge_amount': natural_hedge_amount,
                        'positive_exposures': positive_exposures,
                        'negative_exposures': negative_exposures
                    })
            
            # 创建自然对冲头寸（概念性的，不需要实际交易）
            hedge_positions = []
            total_hedge_cost = 0.0  # 自然对冲没有直接成本
            
            for hedge_info in natural_hedges:
                # 创建概念性的对冲头寸
                hedge_instrument = CurrencyHedgeInstrument(
                    instrument_type=HedgingMethod.NATURAL_HEDGING,
                    currency_pair=f"{hedge_info['currency']}{self.base_currency}",
                    notional_amount=hedge_info['hedge_amount'],
                    transaction_cost=0.0  # 自然对冲无交易成本
                )
                
                hedge_position = CurrencyHedgePosition(
                    instrument=hedge_instrument,
                    position_size=hedge_info['hedge_amount'],
                    entry_rate=1.0,
                    current_rate=1.0,
                    unrealized_pnl=0.0,
                    delta=1.0
                )
                
                hedge_positions.append(hedge_position)
            
            # 计算对冲后的风险指标
            hedged_exposures = self._calculate_hedged_exposures(exposures, hedge_positions)
            risk_metrics = await self._calculate_risk_metrics(hedged_exposures, **kwargs)
            
            # 计算对冲效果
            original_risk = await self._calculate_risk_metrics(exposures, **kwargs)
            hedge_effectiveness = (original_risk.var_95 - risk_metrics.var_95) / original_risk.var_95
            risk_metrics.hedge_effectiveness = hedge_effectiveness
            
            return CurrencyRiskResult(
                base_currency=self.base_currency,
                total_exposures=exposures,
                hedge_positions=hedge_positions,
                risk_metrics=risk_metrics,
                hedging_method=HedgingMethod.NATURAL_HEDGING,
                total_hedge_cost=total_hedge_cost,
                net_hedge_benefit=original_risk.var_95 - risk_metrics.var_95,
                optimization_success=True,
                rebalancing_frequency=self.rebalancing_frequency
            )
            
        except Exception as e:
            self.logger.error(f"Error creating natural hedge: {e}")
            raise
    
    async def _create_netting_hedge(self, 
                                   exposures: List[CurrencyExposure],
                                   hedge_instruments: List[CurrencyHedgeInstrument],
                                   constraints: Optional[Dict[str, Any]] = None,
                                   **kwargs) -> CurrencyRiskResult:
        """
        创建净额结算对冲
        """
        try:
            # 按货币和到期日分组
            grouped_exposures = {}
            
            for exposure in exposures:
                # 创建分组键
                maturity_key = exposure.maturity_date.strftime('%Y-%m') if exposure.maturity_date else 'spot'
                group_key = f"{exposure.currency}_{maturity_key}"
                
                if group_key not in grouped_exposures:
                    grouped_exposures[group_key] = []
                grouped_exposures[group_key].append(exposure)
            
            # 计算净暴露
            net_exposures = []
            netting_benefits = []
            
            for group_key, group_exposures in grouped_exposures.items():
                # 计算净暴露
                net_amount = sum(exp.exposure_amount for exp in group_exposures)
                
                if abs(net_amount) > 1e-6:
                    # 创建净暴露
                    net_exposure = CurrencyExposure(
                        currency=group_exposures[0].currency,
                        base_currency=group_exposures[0].base_currency,
                        exposure_amount=net_amount,
                        exposure_type=CurrencyRiskType.TRANSACTION_RISK,
                        maturity_date=group_exposures[0].maturity_date
                    )
                    net_exposures.append(net_exposure)
                    
                    # 计算净额结算收益
                    gross_exposure = sum(abs(exp.exposure_amount) for exp in group_exposures)
                    netting_benefit = gross_exposure - abs(net_amount)
                    netting_benefits.append(netting_benefit)
            
            # 创建净额结算头寸
            hedge_positions = []
            total_hedge_cost = 0.0
            
            for i, net_exposure in enumerate(net_exposures):
                # 如果净暴露仍然较大，考虑进一步对冲
                if abs(net_exposure.exposure_amount) > constraints.get('min_hedge_threshold', 10000):
                    # 寻找合适的对冲工具
                    suitable_instruments = [
                        inst for inst in hedge_instruments 
                        if inst.currency_pair == f"{net_exposure.currency}{net_exposure.base_currency}"
                    ]
                    
                    if suitable_instruments:
                        best_instrument = max(suitable_instruments, key=lambda x: x.liquidity_score)
                        
                        # 计算对冲头寸
                        hedge_size = -net_exposure.exposure_amount
                        current_rate = self.exchange_rates.get(best_instrument.currency_pair, 1.0)
                        
                        hedge_position = CurrencyHedgePosition(
                            instrument=best_instrument,
                            position_size=hedge_size,
                            entry_rate=current_rate,
                            current_rate=current_rate,
                            unrealized_pnl=0.0,
                            delta=1.0
                        )
                        
                        hedge_positions.append(hedge_position)
                        
                        # 计算对冲成本
                        hedge_cost = abs(hedge_size) * best_instrument.transaction_cost
                        total_hedge_cost += hedge_cost
            
            # 计算对冲后的风险指标
            hedged_exposures = self._calculate_hedged_exposures(net_exposures, hedge_positions)
            risk_metrics = await self._calculate_risk_metrics(hedged_exposures, **kwargs)
            
            # 计算对冲效果
            original_risk = await self._calculate_risk_metrics(exposures, **kwargs)
            hedge_effectiveness = (original_risk.var_95 - risk_metrics.var_95) / original_risk.var_95
            risk_metrics.hedge_effectiveness = hedge_effectiveness
            
            # 净额结算收益
            total_netting_benefit = sum(netting_benefits)
            
            return CurrencyRiskResult(
                base_currency=self.base_currency,
                total_exposures=exposures,
                hedge_positions=hedge_positions,
                risk_metrics=risk_metrics,
                hedging_method=HedgingMethod.NETTING,
                total_hedge_cost=total_hedge_cost,
                net_hedge_benefit=total_netting_benefit - total_hedge_cost,
                optimization_success=True,
                rebalancing_frequency=self.rebalancing_frequency,
                metadata={'netting_benefit': total_netting_benefit, 'net_exposures': len(net_exposures)}
            )
            
        except Exception as e:
            self.logger.error(f"Error creating netting hedge: {e}")
            raise
    
    async def _create_diversification_hedge(self, 
                                          exposures: List[CurrencyExposure],
                                          hedge_instruments: List[CurrencyHedgeInstrument],
                                          constraints: Optional[Dict[str, Any]] = None,
                                          **kwargs) -> CurrencyRiskResult:
        """
        创建货币分散化对冲
        """
        try:
            # 分析货币集中度
            currency_concentration = self._analyze_currency_concentration(exposures)
            
            # 构建多元化目标
            target_diversification = await self._optimize_currency_diversification(
                exposures, constraints, **kwargs
            )
            
            # 创建分散化头寸
            hedge_positions = []
            total_hedge_cost = 0.0
            
            for target_currency, target_weight in target_diversification.items():
                current_weight = currency_concentration.get(target_currency, 0.0)
                weight_adjustment = target_weight - current_weight
                
                if abs(weight_adjustment) > 0.01:  # 1%的调整阈值
                    # 寻找合适的对冲工具
                    suitable_instruments = [
                        inst for inst in hedge_instruments 
                        if target_currency in inst.currency_pair
                    ]
                    
                    if suitable_instruments:
                        best_instrument = max(suitable_instruments, key=lambda x: x.liquidity_score)
                        
                        # 计算调整头寸
                        total_exposure = sum(abs(exp.exposure_amount) for exp in exposures)
                        adjustment_amount = weight_adjustment * total_exposure
                        
                        current_rate = self.exchange_rates.get(best_instrument.currency_pair, 1.0)
                        
                        hedge_position = CurrencyHedgePosition(
                            instrument=best_instrument,
                            position_size=adjustment_amount,
                            entry_rate=current_rate,
                            current_rate=current_rate,
                            unrealized_pnl=0.0,
                            delta=1.0
                        )
                        
                        hedge_positions.append(hedge_position)
                        
                        # 计算对冲成本
                        hedge_cost = abs(adjustment_amount) * best_instrument.transaction_cost
                        total_hedge_cost += hedge_cost
            
            # 计算对冲后的风险指标
            hedged_exposures = self._calculate_hedged_exposures(exposures, hedge_positions)
            risk_metrics = await self._calculate_risk_metrics(hedged_exposures, **kwargs)
            
            # 计算对冲效果
            original_risk = await self._calculate_risk_metrics(exposures, **kwargs)
            hedge_effectiveness = (original_risk.var_95 - risk_metrics.var_95) / original_risk.var_95
            risk_metrics.hedge_effectiveness = hedge_effectiveness
            
            # 计算分散化收益
            diversification_benefit = self._calculate_diversification_benefit(
                currency_concentration, target_diversification
            )
            
            return CurrencyRiskResult(
                base_currency=self.base_currency,
                total_exposures=exposures,
                hedge_positions=hedge_positions,
                risk_metrics=risk_metrics,
                hedging_method=HedgingMethod.CURRENCY_DIVERSIFICATION,
                total_hedge_cost=total_hedge_cost,
                net_hedge_benefit=diversification_benefit - total_hedge_cost,
                optimization_success=True,
                rebalancing_frequency=self.rebalancing_frequency,
                metadata={
                    'original_concentration': currency_concentration,
                    'target_diversification': target_diversification,
                    'diversification_benefit': diversification_benefit
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error creating diversification hedge: {e}")
            raise
    
    async def monitor_currency_risk(self, 
                                   currency_result: CurrencyRiskResult,
                                   market_data: Dict[str, Any],
                                   **kwargs) -> Dict[str, Any]:
        """
        监控货币风险
        """
        try:
            # 更新市场数据
            await self._update_market_data(market_data)
            
            # 计算当前P&L
            current_pnl = 0.0
            position_pnls = []
            
            for position in currency_result.hedge_positions:
                current_rate = self.exchange_rates.get(position.instrument.currency_pair, position.current_rate)
                
                # 计算未实现盈亏
                rate_change = current_rate - position.entry_rate
                pnl = position.position_size * rate_change
                current_pnl += pnl
                
                position_pnls.append({
                    'currency_pair': position.instrument.currency_pair,
                    'position_size': position.position_size,
                    'entry_rate': position.entry_rate,
                    'current_rate': current_rate,
                    'pnl': pnl,
                    'pnl_percent': pnl / (position.entry_rate * abs(position.position_size)) if position.position_size != 0 else 0
                })
            
            # 重新计算风险指标
            current_risk_metrics = await self._calculate_risk_metrics(
                currency_result.total_exposures, **kwargs
            )
            
            # 检查重新平衡触发条件
            rebalance_needed = False
            rebalance_reasons = []
            
            # 对冲效果下降
            current_hedge_effectiveness = self._calculate_current_hedge_effectiveness(
                currency_result, current_risk_metrics
            )
            
            if current_hedge_effectiveness < currency_result.risk_metrics.hedge_effectiveness * 0.8:
                rebalance_needed = True
                rebalance_reasons.append("hedge_effectiveness_decline")
            
            # 汇率波动过大
            for position in currency_result.hedge_positions:
                rate_change_pct = abs(position.current_rate - position.entry_rate) / position.entry_rate
                if rate_change_pct > 0.05:  # 5%的汇率变动阈值
                    rebalance_needed = True
                    rebalance_reasons.append("exchange_rate_volatility")
                    break
            
            # 风险超限
            if current_risk_metrics.var_95 > currency_result.risk_metrics.var_95 * 1.2:
                rebalance_needed = True
                rebalance_reasons.append("risk_limit_breach")
            
            # 计算绩效指标
            performance_metrics = {
                'current_pnl': current_pnl,
                'position_pnls': position_pnls,
                'current_risk_metrics': current_risk_metrics,
                'current_hedge_effectiveness': current_hedge_effectiveness,
                'effectiveness_change': current_hedge_effectiveness - currency_result.risk_metrics.hedge_effectiveness,
                'risk_change': current_risk_metrics.var_95 - currency_result.risk_metrics.var_95,
                'rebalance_needed': rebalance_needed,
                'rebalance_reasons': rebalance_reasons,
                'monitoring_timestamp': datetime.now(),
                'market_data_timestamp': market_data.get('timestamp', datetime.now())
            }
            
            return performance_metrics
            
        except Exception as e:
            self.logger.error(f"Error monitoring currency risk: {e}")
            return {}
    
    # 辅助方法
    async def _update_market_data(self, market_data: Dict[str, Any]):
        """更新市场数据"""
        try:
            # 更新汇率
            if 'exchange_rates' in market_data:
                self.exchange_rates.update(market_data['exchange_rates'])
            
            # 更新波动率
            if 'volatilities' in market_data:
                self.volatilities.update(market_data['volatilities'])
            
            # 更新相关性
            if 'correlations' in market_data:
                self.correlations.update(market_data['correlations'])
            
            # 更新利率
            if 'interest_rates' in market_data:
                self.interest_rates.update(market_data['interest_rates'])
            
            self.logger.info("Market data updated successfully")
            
        except Exception as e:
            self.logger.error(f"Error updating market data: {e}")
    
    async def _calculate_risk_metrics(self, 
                                    exposures: List[CurrencyExposure],
                                    **kwargs) -> CurrencyRiskMetrics:
        """计算风险指标"""
        try:
            # 计算总暴露
            total_exposure = sum(abs(exp.exposure_amount) for exp in exposures)
            
            # 按货币分组
            currency_exposures = {}
            for exp in exposures:
                if exp.currency not in currency_exposures:
                    currency_exposures[exp.currency] = 0
                currency_exposures[exp.currency] += exp.exposure_amount
            
            # 计算组合波动率
            portfolio_volatility = await self._calculate_portfolio_volatility(currency_exposures)
            
            # 计算VaR
            var_95 = total_exposure * portfolio_volatility * norm.ppf(0.95)
            var_99 = total_exposure * portfolio_volatility * norm.ppf(0.99)
            
            # 计算CVaR
            cvar_95 = total_exposure * portfolio_volatility * norm.pdf(norm.ppf(0.95)) / (1 - 0.95)
            
            # 计算其他指标
            hedge_ratio = 0.0  # 基础分析时为0
            hedge_effectiveness = 0.0
            basis_risk = 0.05  # 默认基差风险
            maximum_drawdown = var_95 * 2  # 估算最大回撤
            sharpe_ratio = 0.0  # 无收益时为0
            tracking_error = portfolio_volatility
            currency_beta = 1.0  # 默认贝塔值
            correlation_stability = 0.8  # 默认相关性稳定性
            
            return CurrencyRiskMetrics(
                total_exposure=total_exposure,
                hedged_exposure=total_exposure,
                residual_exposure=total_exposure,
                hedge_ratio=hedge_ratio,
                hedge_effectiveness=hedge_effectiveness,
                basis_risk=basis_risk,
                var_95=var_95,
                var_99=var_99,
                cvar_95=cvar_95,
                volatility=portfolio_volatility,
                maximum_drawdown=maximum_drawdown,
                sharpe_ratio=sharpe_ratio,
                tracking_error=tracking_error,
                currency_beta=currency_beta,
                correlation_stability=correlation_stability
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")
            # 返回默认指标
            return CurrencyRiskMetrics(
                total_exposure=0.0,
                hedged_exposure=0.0,
                residual_exposure=0.0,
                hedge_ratio=0.0,
                hedge_effectiveness=0.0,
                basis_risk=0.05,
                var_95=0.0,
                var_99=0.0,
                cvar_95=0.0,
                volatility=0.15,
                maximum_drawdown=0.0,
                sharpe_ratio=0.0,
                tracking_error=0.15,
                currency_beta=1.0,
                correlation_stability=0.8
            )
    
    async def _calculate_portfolio_volatility(self, currency_exposures: Dict[str, float]) -> float:
        """计算组合波动率"""
        try:
            currencies = list(currency_exposures.keys())
            weights = np.array(list(currency_exposures.values()))
            
            # 标准化权重
            total_weight = np.sum(np.abs(weights))
            if total_weight > 0:
                weights = weights / total_weight
            
            # 构建协方差矩阵
            n = len(currencies)
            covariance_matrix = np.zeros((n, n))
            
            for i, curr_i in enumerate(currencies):
                vol_i = self.volatilities.get(curr_i, 0.15)
                for j, curr_j in enumerate(currencies):
                    if i == j:
                        covariance_matrix[i, j] = vol_i ** 2
                    else:
                        vol_j = self.volatilities.get(curr_j, 0.15)
                        corr = self.correlations.get(curr_i, {}).get(curr_j, 0.3)
                        covariance_matrix[i, j] = vol_i * vol_j * corr
            
            # 计算组合波动率
            portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
            
            return np.sqrt(max(0, portfolio_variance))
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio volatility: {e}")
            return 0.15  # 默认波动率
    
    async def _optimize_hedge_size(self, 
                                 net_exposure: float,
                                 hedge_instrument: CurrencyHedgeInstrument,
                                 constraints: Optional[Dict[str, Any]] = None,
                                 **kwargs) -> float:
        """优化对冲头寸大小"""
        try:
            # 基本对冲比率
            hedge_ratio = constraints.get('hedge_ratio', 1.0) if constraints else 1.0
            
            # 计算最优对冲头寸
            optimal_hedge_size = -net_exposure * hedge_ratio
            
            # 应用约束
            if constraints:
                # 最大对冲限制
                max_hedge = constraints.get('max_hedge_amount', abs(net_exposure))
                optimal_hedge_size = np.clip(optimal_hedge_size, -max_hedge, max_hedge)
                
                # 最小对冲门槛
                min_hedge_threshold = constraints.get('min_hedge_threshold', 0)
                if abs(optimal_hedge_size) < min_hedge_threshold:
                    optimal_hedge_size = 0
            
            return optimal_hedge_size
            
        except Exception as e:
            self.logger.error(f"Error optimizing hedge size: {e}")
            return -net_exposure  # 默认完全对冲
    
    async def _optimize_options_portfolio(self, 
                                        net_exposure: float,
                                        option_instruments: List[CurrencyHedgeInstrument],
                                        constraints: Optional[Dict[str, Any]] = None,
                                        **kwargs) -> List[CurrencyHedgePosition]:
        """优化期权组合"""
        try:
            # 简化的期权组合优化
            hedge_positions = []
            
            # 按执行价格排序
            sorted_options = sorted(option_instruments, key=lambda x: x.strike_rate or 0)
            
            if len(sorted_options) == 0:
                return hedge_positions
            
            # 选择最优期权
            if net_exposure > 0:
                # 需要对冲长头寸，买入看跌期权
                best_option = min(sorted_options, key=lambda x: x.premium)
            else:
                # 需要对冲短头寸，买入看涨期权
                best_option = min(sorted_options, key=lambda x: x.premium)
            
            # 计算期权头寸大小
            option_size = await self._optimize_hedge_size(net_exposure, best_option, constraints, **kwargs)
            
            if abs(option_size) > 1e-6:
                current_rate = self.exchange_rates.get(best_option.currency_pair, 1.0)
                
                # 计算期权Greeks
                delta = self._calculate_option_delta(best_option, current_rate)
                gamma = self._calculate_option_gamma(best_option, current_rate)
                vega = self._calculate_option_vega(best_option, current_rate)
                theta = self._calculate_option_theta(best_option, current_rate)
                
                hedge_position = CurrencyHedgePosition(
                    instrument=best_option,
                    position_size=option_size,
                    entry_rate=current_rate,
                    current_rate=current_rate,
                    unrealized_pnl=0.0,
                    delta=delta,
                    gamma=gamma,
                    vega=vega,
                    theta=theta
                )
                
                hedge_positions.append(hedge_position)
            
            return hedge_positions
            
        except Exception as e:
            self.logger.error(f"Error optimizing options portfolio: {e}")
            return []
    
    def _calculate_option_delta(self, option: CurrencyHedgeInstrument, current_rate: float) -> float:
        """计算期权Delta"""
        try:
            # 简化的Delta计算
            if option.strike_rate is None:
                return 0.5
            
            # 使用简化的Black-Scholes公式
            moneyness = current_rate / option.strike_rate
            if moneyness > 1.05:
                return 0.8
            elif moneyness < 0.95:
                return 0.2
            else:
                return 0.5
                
        except Exception as e:
            self.logger.error(f"Error calculating option delta: {e}")
            return 0.5
    
    def _calculate_option_gamma(self, option: CurrencyHedgeInstrument, current_rate: float) -> float:
        """计算期权Gamma"""
        try:
            # 简化的Gamma计算
            if option.strike_rate is None:
                return 0.01
            
            # 平值期权具有最大的Gamma
            moneyness = current_rate / option.strike_rate
            if 0.95 <= moneyness <= 1.05:
                return 0.02
            else:
                return 0.005
                
        except Exception as e:
            self.logger.error(f"Error calculating option gamma: {e}")
            return 0.01
    
    def _calculate_option_vega(self, option: CurrencyHedgeInstrument, current_rate: float) -> float:
        """计算期权Vega"""
        try:
            # 简化的Vega计算
            if option.strike_rate is None:
                return 0.1
            
            # 平值期权具有最大的Vega
            moneyness = current_rate / option.strike_rate
            if 0.95 <= moneyness <= 1.05:
                return 0.15
            else:
                return 0.05
                
        except Exception as e:
            self.logger.error(f"Error calculating option vega: {e}")
            return 0.1
    
    def _calculate_option_theta(self, option: CurrencyHedgeInstrument, current_rate: float) -> float:
        """计算期权Theta"""
        try:
            # 简化的Theta计算
            if option.maturity_date is None:
                return -0.01
            
            # 时间衰减
            days_to_expiry = (option.maturity_date - datetime.now()).days
            if days_to_expiry > 0:
                return -option.premium / days_to_expiry
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error calculating option theta: {e}")
            return -0.01
    
    def _calculate_hedged_exposures(self, 
                                   original_exposures: List[CurrencyExposure],
                                   hedge_positions: List[CurrencyHedgePosition]) -> List[CurrencyExposure]:
        """计算对冲后的暴露"""
        try:
            # 复制原始暴露
            hedged_exposures = [exp for exp in original_exposures]
            
            # 应用对冲头寸
            for hedge_position in hedge_positions:
                currency_pair = hedge_position.instrument.currency_pair
                
                # 提取货币对
                if len(currency_pair) == 6:  # 标准货币对格式
                    currency1 = currency_pair[:3]
                    currency2 = currency_pair[3:]
                else:
                    continue
                
                # 创建对冲暴露
                hedge_exposure = CurrencyExposure(
                    currency=currency1,
                    base_currency=currency2,
                    exposure_amount=-hedge_position.position_size,  # 对冲是相反头寸
                    exposure_type=CurrencyRiskType.TRANSACTION_RISK
                )
                
                hedged_exposures.append(hedge_exposure)
            
            return hedged_exposures
            
        except Exception as e:
            self.logger.error(f"Error calculating hedged exposures: {e}")
            return original_exposures
    
    def _analyze_currency_concentration(self, exposures: List[CurrencyExposure]) -> Dict[str, float]:
        """分析货币集中度"""
        try:
            currency_amounts = {}
            total_exposure = 0
            
            for exp in exposures:
                if exp.currency not in currency_amounts:
                    currency_amounts[exp.currency] = 0
                currency_amounts[exp.currency] += abs(exp.exposure_amount)
                total_exposure += abs(exp.exposure_amount)
            
            # 计算权重
            concentration = {}
            for currency, amount in currency_amounts.items():
                concentration[currency] = amount / total_exposure if total_exposure > 0 else 0
            
            return concentration
            
        except Exception as e:
            self.logger.error(f"Error analyzing currency concentration: {e}")
            return {}
    
    async def _optimize_currency_diversification(self, 
                                               exposures: List[CurrencyExposure],
                                               constraints: Optional[Dict[str, Any]] = None,
                                               **kwargs) -> Dict[str, float]:
        """优化货币分散化"""
        try:
            # 获取所有货币
            currencies = list(set(exp.currency for exp in exposures))
            
            if not currencies:
                return {}
            
            # 设置目标权重
            target_diversification = {}
            
            if constraints and 'target_weights' in constraints:
                target_diversification = constraints['target_weights']
            else:
                # 等权重分散
                equal_weight = 1.0 / len(currencies)
                for currency in currencies:
                    target_diversification[currency] = equal_weight
            
            # 确保权重和为1
            total_weight = sum(target_diversification.values())
            if total_weight > 0:
                for currency in target_diversification:
                    target_diversification[currency] /= total_weight
            
            return target_diversification
            
        except Exception as e:
            self.logger.error(f"Error optimizing currency diversification: {e}")
            return {}
    
    def _calculate_diversification_benefit(self, 
                                         current_concentration: Dict[str, float],
                                         target_diversification: Dict[str, float]) -> float:
        """计算分散化收益"""
        try:
            # 计算Herfindahl指数
            current_hhi = sum(weight ** 2 for weight in current_concentration.values())
            target_hhi = sum(weight ** 2 for weight in target_diversification.values())
            
            # 分散化收益
            diversification_benefit = (current_hhi - target_hhi) * 100000  # 假设基础暴露为100,000
            
            return max(0, diversification_benefit)
            
        except Exception as e:
            self.logger.error(f"Error calculating diversification benefit: {e}")
            return 0.0
    
    def _calculate_current_hedge_effectiveness(self, 
                                             currency_result: CurrencyRiskResult,
                                             current_risk_metrics: CurrencyRiskMetrics) -> float:
        """计算当前对冲效果"""
        try:
            original_var = currency_result.risk_metrics.var_95
            current_var = current_risk_metrics.var_95
            
            if original_var > 0:
                return (original_var - current_var) / original_var
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error calculating current hedge effectiveness: {e}")
            return 0.0
    
    def get_currency_risk_summary(self) -> Dict[str, Any]:
        """获取货币风险摘要"""
        try:
            if not self.risk_history:
                return {}
            
            recent_results = self.risk_history[-10:]
            
            summary = {
                'base_currency': self.base_currency,
                'total_risk_assessments': len(self.risk_history),
                'active_hedge_positions': len(self.hedge_positions),
                'hedging_methods_used': list(set(result.hedging_method.value for result in recent_results)),
                'average_hedge_effectiveness': np.mean([result.risk_metrics.hedge_effectiveness for result in recent_results]),
                'average_risk_reduction': np.mean([result.net_hedge_benefit for result in recent_results]),
                'total_hedge_costs': sum([result.total_hedge_cost for result in recent_results]),
                'current_currencies': list(self.exchange_rates.keys()),
                'risk_tolerance': self.risk_tolerance,
                'last_assessment_date': self.risk_history[-1].metadata.get('timestamp', datetime.now())
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting currency risk summary: {e}")
            return {}
    
    def plot_currency_risk_analysis(self, 
                                   currency_result: CurrencyRiskResult,
                                   save_path: Optional[str] = None):
        """绘制货币风险分析图"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. 货币暴露分布
            if currency_result.total_exposures:
                currencies = [exp.currency for exp in currency_result.total_exposures]
                exposures = [abs(exp.exposure_amount) for exp in currency_result.total_exposures]
                
                currency_totals = {}
                for currency, exposure in zip(currencies, exposures):
                    currency_totals[currency] = currency_totals.get(currency, 0) + exposure
                
                ax1.pie(currency_totals.values(), labels=currency_totals.keys(), 
                       autopct='%1.1f%%', startangle=90)
                ax1.set_title('货币暴露分布')
            
            # 2. 风险指标对比
            risk_categories = ['VaR 95%', 'CVaR 95%', '波动率', '最大回撤']
            before_hedge = [
                currency_result.risk_metrics.var_95 * 1.2,  # 假设对冲前风险更高
                currency_result.risk_metrics.cvar_95 * 1.2,
                currency_result.risk_metrics.volatility * 1.2,
                currency_result.risk_metrics.maximum_drawdown * 1.2
            ]
            after_hedge = [
                currency_result.risk_metrics.var_95,
                currency_result.risk_metrics.cvar_95,
                currency_result.risk_metrics.volatility,
                currency_result.risk_metrics.maximum_drawdown
            ]
            
            x = np.arange(len(risk_categories))
            width = 0.35
            
            ax2.bar(x - width/2, before_hedge, width, label='对冲前', alpha=0.7)
            ax2.bar(x + width/2, after_hedge, width, label='对冲后', alpha=0.7)
            ax2.set_xlabel('风险指标')
            ax2.set_ylabel('数值')
            ax2.set_title('风险指标对比')
            ax2.set_xticks(x)
            ax2.set_xticklabels(risk_categories)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. 对冲头寸分布
            if currency_result.hedge_positions:
                hedge_instruments = [pos.instrument.currency_pair for pos in currency_result.hedge_positions]
                hedge_sizes = [abs(pos.position_size) for pos in currency_result.hedge_positions]
                
                ax3.bar(hedge_instruments, hedge_sizes, alpha=0.7, color='orange')
                ax3.set_ylabel('头寸大小')
                ax3.set_title('对冲头寸分布')
                ax3.tick_params(axis='x', rotation=45)
                ax3.grid(True, alpha=0.3)
            
            # 4. 对冲效果分析
            effectiveness_metrics = [
                '对冲效果',
                '基差风险',
                '跟踪误差',
                '相关性稳定性'
            ]
            effectiveness_values = [
                currency_result.risk_metrics.hedge_effectiveness,
                currency_result.risk_metrics.basis_risk,
                currency_result.risk_metrics.tracking_error,
                currency_result.risk_metrics.correlation_stability
            ]
            
            ax4.bar(effectiveness_metrics, effectiveness_values, alpha=0.7, color='green')
            ax4.set_ylabel('数值')
            ax4.set_title('对冲效果分析')
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Error plotting currency risk analysis: {e}")
    
    def export_currency_risk_report(self, 
                                   currency_result: CurrencyRiskResult,
                                   file_path: str):
        """导出货币风险报告"""
        try:
            report_data = {
                'report_metadata': {
                    'base_currency': currency_result.base_currency,
                    'hedging_method': currency_result.hedging_method.value,
                    'generation_date': datetime.now().isoformat(),
                    'rebalancing_frequency': currency_result.rebalancing_frequency
                },
                'risk_metrics': {
                    'total_exposure': currency_result.risk_metrics.total_exposure,
                    'hedged_exposure': currency_result.risk_metrics.hedged_exposure,
                    'residual_exposure': currency_result.risk_metrics.residual_exposure,
                    'hedge_ratio': currency_result.risk_metrics.hedge_ratio,
                    'hedge_effectiveness': currency_result.risk_metrics.hedge_effectiveness,
                    'var_95': currency_result.risk_metrics.var_95,
                    'var_99': currency_result.risk_metrics.var_99,
                    'cvar_95': currency_result.risk_metrics.cvar_95,
                    'volatility': currency_result.risk_metrics.volatility,
                    'maximum_drawdown': currency_result.risk_metrics.maximum_drawdown,
                    'sharpe_ratio': currency_result.risk_metrics.sharpe_ratio
                },
                'exposures': [
                    {
                        'currency': exp.currency,
                        'exposure_amount': exp.exposure_amount,
                        'exposure_type': exp.exposure_type.value,
                        'maturity_date': exp.maturity_date.isoformat() if exp.maturity_date else None
                    } for exp in currency_result.total_exposures
                ],
                'hedge_positions': [
                    {
                        'currency_pair': pos.instrument.currency_pair,
                        'instrument_type': pos.instrument.instrument_type.value,
                        'position_size': pos.position_size,
                        'entry_rate': pos.entry_rate,
                        'current_rate': pos.current_rate,
                        'unrealized_pnl': pos.unrealized_pnl
                    } for pos in currency_result.hedge_positions
                ],
                'cost_benefit_analysis': {
                    'total_hedge_cost': currency_result.total_hedge_cost,
                    'net_hedge_benefit': currency_result.net_hedge_benefit,
                    'cost_benefit_ratio': currency_result.net_hedge_benefit / currency_result.total_hedge_cost if currency_result.total_hedge_cost > 0 else 0
                }
            }
            
            import json
            with open(file_path, 'w') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Currency risk report exported to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error exporting currency risk report: {e}")
            raise
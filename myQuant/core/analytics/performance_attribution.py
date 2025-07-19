import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

class AttributionMethod(Enum):
    BRINSON_HOOD_BEEBOWER = "brinson_hood_beebower"
    BRINSON_FACHLER = "brinson_fachler"
    GEOMETRIC_ATTRIBUTION = "geometric_attribution"
    ARITHMETIC_ATTRIBUTION = "arithmetic_attribution"
    FAMA_FRENCH = "fama_french"
    MULTI_FACTOR = "multi_factor"
    RISK_ADJUSTED = "risk_adjusted"
    CURRENCY_ATTRIBUTION = "currency_attribution"
    SECTOR_ATTRIBUTION = "sector_attribution"
    SECURITY_SELECTION = "security_selection"

class AttributionLevel(Enum):
    PORTFOLIO = "portfolio"
    SECTOR = "sector"
    ASSET_CLASS = "asset_class"
    SECURITY = "security"
    FACTOR = "factor"
    STYLE = "style"
    COUNTRY = "country"
    CURRENCY = "currency"

class AttributionPeriod(Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    INCEPTION_TO_DATE = "inception_to_date"
    CUSTOM = "custom"

@dataclass
class AttributionComponent:
    """归因组件"""
    component_name: str
    component_type: str
    allocation_effect: float
    selection_effect: float
    interaction_effect: float
    total_effect: float
    weight_portfolio: float
    weight_benchmark: float
    return_portfolio: float
    return_benchmark: float
    contribution_portfolio: float
    contribution_benchmark: float
    active_weight: float
    active_return: float
    information_ratio: float
    tracking_error: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SecurityAttribution:
    """证券级归因"""
    security_id: str
    security_name: str
    sector: str
    weight_portfolio: float
    weight_benchmark: float
    return_portfolio: float
    return_benchmark: float
    contribution_portfolio: float
    contribution_benchmark: float
    active_contribution: float
    selection_effect: float
    allocation_effect: float
    total_effect: float
    risk_contribution: float
    alpha: float
    beta: float
    residual_risk: float
    specific_risk: float
    factor_exposures: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FactorAttribution:
    """因子归因"""
    factor_name: str
    factor_type: str
    factor_return: float
    factor_exposure_portfolio: float
    factor_exposure_benchmark: float
    active_exposure: float
    factor_contribution: float
    selection_return: float
    allocation_return: float
    total_return: float
    risk_contribution: float
    information_ratio: float
    factor_timing: float
    factor_selection: float
    volatility: float
    sharpe_ratio: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CurrencyAttribution:
    """货币归因"""
    currency_pair: str
    currency_return: float
    currency_exposure_portfolio: float
    currency_exposure_benchmark: float
    hedging_ratio: float
    hedged_return: float
    unhedged_return: float
    currency_contribution: float
    hedging_contribution: float
    total_currency_effect: float
    forward_points: float
    spot_return: float
    carry_return: float
    volatility: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AttributionSummary:
    """归因总结"""
    attribution_method: AttributionMethod
    attribution_level: AttributionLevel
    period_start: datetime
    period_end: datetime
    portfolio_return: float
    benchmark_return: float
    active_return: float
    total_allocation_effect: float
    total_selection_effect: float
    total_interaction_effect: float
    total_attribution: float
    attribution_residual: float
    tracking_error: float
    information_ratio: float
    attribution_components: List[AttributionComponent]
    security_attribution: List[SecurityAttribution]
    factor_attribution: List[FactorAttribution]
    currency_attribution: List[CurrencyAttribution]
    risk_attribution: Dict[str, float]
    style_attribution: Dict[str, float]
    sector_attribution: Dict[str, float]
    country_attribution: Dict[str, float]
    top_contributors: List[str]
    top_detractors: List[str]
    attribution_quality: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)

class PerformanceAttribution:
    """
    投资组合绩效归因分析
    
    提供多层次、多维度的绩效归因分析，包括资产配置效应、
    证券选择效应、因子归因、货币归因等。
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 归因配置
        self.default_method = AttributionMethod(config.get('default_method', 'brinson_hood_beebower'))
        self.default_level = AttributionLevel(config.get('default_level', 'sector'))
        self.default_period = AttributionPeriod(config.get('default_period', 'monthly'))
        
        # 数据配置
        self.base_currency = config.get('base_currency', 'USD')
        self.risk_free_rate = config.get('risk_free_rate', 0.02)
        self.confidence_level = config.get('confidence_level', 0.95)
        
        # 因子模型
        self.factor_models = {
            'fama_french_3': ['market', 'size', 'value'],
            'fama_french_5': ['market', 'size', 'value', 'profitability', 'investment'],
            'carhart_4': ['market', 'size', 'value', 'momentum'],
            'custom': config.get('custom_factors', [])
        }
        
        # 缓存
        self.attribution_cache = {}
        self.benchmark_cache = {}
        
        # 性能监控
        self.performance_metrics = {}
        
    async def calculate_attribution(self, 
                                  portfolio_returns: pd.DataFrame,
                                  portfolio_weights: pd.DataFrame,
                                  benchmark_returns: pd.DataFrame,
                                  benchmark_weights: pd.DataFrame,
                                  method: AttributionMethod = None,
                                  level: AttributionLevel = None,
                                  period: AttributionPeriod = None) -> AttributionSummary:
        """计算绩效归因"""
        try:
            if method is None:
                method = self.default_method
            if level is None:
                level = self.default_level
            if period is None:
                period = self.default_period
            
            self.logger.info(f"开始计算绩效归因: {method.value} - {level.value}")
            
            # 数据预处理
            portfolio_data, benchmark_data = await self._preprocess_data(
                portfolio_returns, portfolio_weights, benchmark_returns, benchmark_weights
            )
            
            # 计算基础收益指标
            portfolio_return = await self._calculate_portfolio_return(portfolio_data)
            benchmark_return = await self._calculate_benchmark_return(benchmark_data)
            active_return = portfolio_return - benchmark_return
            
            # 根据方法计算归因
            if method == AttributionMethod.BRINSON_HOOD_BEEBOWER:
                attribution_result = await self._brinson_hood_beebower_attribution(
                    portfolio_data, benchmark_data
                )
            elif method == AttributionMethod.BRINSON_FACHLER:
                attribution_result = await self._brinson_fachler_attribution(
                    portfolio_data, benchmark_data
                )
            elif method == AttributionMethod.GEOMETRIC_ATTRIBUTION:
                attribution_result = await self._geometric_attribution(
                    portfolio_data, benchmark_data
                )
            elif method == AttributionMethod.FAMA_FRENCH:
                attribution_result = await self._fama_french_attribution(
                    portfolio_data, benchmark_data
                )
            elif method == AttributionMethod.MULTI_FACTOR:
                attribution_result = await self._multi_factor_attribution(
                    portfolio_data, benchmark_data
                )
            elif method == AttributionMethod.CURRENCY_ATTRIBUTION:
                attribution_result = await self._currency_attribution(
                    portfolio_data, benchmark_data
                )
            else:
                attribution_result = await self._brinson_hood_beebower_attribution(
                    portfolio_data, benchmark_data
                )
            
            # 计算风险归因
            risk_attribution = await self._calculate_risk_attribution(
                portfolio_data, benchmark_data
            )
            
            # 计算归因质量指标
            attribution_quality = await self._calculate_attribution_quality(
                attribution_result, portfolio_return, benchmark_return
            )
            
            # 识别主要贡献者和拖累者
            top_contributors, top_detractors = await self._identify_top_contributors(
                attribution_result
            )
            
            # 构建归因总结
            summary = AttributionSummary(
                attribution_method=method,
                attribution_level=level,
                period_start=portfolio_data.index[0],
                period_end=portfolio_data.index[-1],
                portfolio_return=portfolio_return,
                benchmark_return=benchmark_return,
                active_return=active_return,
                total_allocation_effect=attribution_result['total_allocation'],
                total_selection_effect=attribution_result['total_selection'],
                total_interaction_effect=attribution_result['total_interaction'],
                total_attribution=attribution_result['total_attribution'],
                attribution_residual=attribution_result['residual'],
                tracking_error=attribution_result['tracking_error'],
                information_ratio=attribution_result['information_ratio'],
                attribution_components=attribution_result['components'],
                security_attribution=attribution_result.get('security_attribution', []),
                factor_attribution=attribution_result.get('factor_attribution', []),
                currency_attribution=attribution_result.get('currency_attribution', []),
                risk_attribution=risk_attribution,
                style_attribution=attribution_result.get('style_attribution', {}),
                sector_attribution=attribution_result.get('sector_attribution', {}),
                country_attribution=attribution_result.get('country_attribution', {}),
                top_contributors=top_contributors,
                top_detractors=top_detractors,
                attribution_quality=attribution_quality
            )
            
            self.logger.info(f"归因计算完成，主动收益: {active_return:.4f}")
            return summary
            
        except Exception as e:
            self.logger.error(f"绩效归因计算失败: {e}")
            raise
    
    async def _preprocess_data(self, portfolio_returns: pd.DataFrame,
                             portfolio_weights: pd.DataFrame,
                             benchmark_returns: pd.DataFrame,
                             benchmark_weights: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """数据预处理"""
        # 对齐时间序列
        common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
        portfolio_returns = portfolio_returns.loc[common_dates]
        benchmark_returns = benchmark_returns.loc[common_dates]
        
        # 对齐权重数据
        portfolio_weights = portfolio_weights.reindex(common_dates).fillna(method='ffill')
        benchmark_weights = benchmark_weights.reindex(common_dates).fillna(method='ffill')
        
        # 确保权重归一化
        portfolio_weights = portfolio_weights.div(portfolio_weights.sum(axis=1), axis=0)
        benchmark_weights = benchmark_weights.div(benchmark_weights.sum(axis=1), axis=0)
        
        # 组合数据
        portfolio_data = pd.concat([portfolio_returns, portfolio_weights], axis=1, keys=['returns', 'weights'])
        benchmark_data = pd.concat([benchmark_returns, benchmark_weights], axis=1, keys=['returns', 'weights'])
        
        return portfolio_data, benchmark_data
    
    async def _calculate_portfolio_return(self, portfolio_data: pd.DataFrame) -> float:
        """计算投资组合收益率"""
        returns = portfolio_data['returns']
        weights = portfolio_data['weights']
        
        # 计算加权收益率
        portfolio_returns = (returns * weights.shift(1)).sum(axis=1)
        
        # 计算累积收益率
        cumulative_return = (1 + portfolio_returns).prod() - 1
        
        return cumulative_return
    
    async def _calculate_benchmark_return(self, benchmark_data: pd.DataFrame) -> float:
        """计算基准收益率"""
        returns = benchmark_data['returns']
        weights = benchmark_data['weights']
        
        # 计算加权收益率
        benchmark_returns = (returns * weights.shift(1)).sum(axis=1)
        
        # 计算累积收益率
        cumulative_return = (1 + benchmark_returns).prod() - 1
        
        return cumulative_return
    
    async def _brinson_hood_beebower_attribution(self, 
                                               portfolio_data: pd.DataFrame,
                                               benchmark_data: pd.DataFrame) -> Dict[str, Any]:
        """Brinson-Hood-Beebower归因方法"""
        # 获取收益率和权重
        Rp = portfolio_data['returns']  # 投资组合收益率
        Rb = benchmark_data['returns']  # 基准收益率
        Wp = portfolio_data['weights']  # 投资组合权重
        Wb = benchmark_data['weights']  # 基准权重
        
        # 计算各个效应
        components = []
        
        for asset in Rp.columns:
            if asset in Rb.columns:
                # 平均权重和收益率
                wp_avg = Wp[asset].mean()
                wb_avg = Wb[asset].mean()
                rp_avg = Rp[asset].mean()
                rb_avg = Rb[asset].mean()
                
                # 资产配置效应 (Allocation Effect)
                allocation_effect = (wp_avg - wb_avg) * rb_avg
                
                # 证券选择效应 (Selection Effect)
                selection_effect = wb_avg * (rp_avg - rb_avg)
                
                # 交互效应 (Interaction Effect)
                interaction_effect = (wp_avg - wb_avg) * (rp_avg - rb_avg)
                
                # 总效应
                total_effect = allocation_effect + selection_effect + interaction_effect
                
                # 贡献度
                contribution_portfolio = wp_avg * rp_avg
                contribution_benchmark = wb_avg * rb_avg
                
                # 主动权重和收益率
                active_weight = wp_avg - wb_avg
                active_return = rp_avg - rb_avg
                
                # 信息比率和跟踪误差
                active_returns = (Rp[asset] - Rb[asset]).dropna()
                tracking_error = active_returns.std()
                information_ratio = active_returns.mean() / tracking_error if tracking_error > 0 else 0
                
                component = AttributionComponent(
                    component_name=asset,
                    component_type="security",
                    allocation_effect=allocation_effect,
                    selection_effect=selection_effect,
                    interaction_effect=interaction_effect,
                    total_effect=total_effect,
                    weight_portfolio=wp_avg,
                    weight_benchmark=wb_avg,
                    return_portfolio=rp_avg,
                    return_benchmark=rb_avg,
                    contribution_portfolio=contribution_portfolio,
                    contribution_benchmark=contribution_benchmark,
                    active_weight=active_weight,
                    active_return=active_return,
                    information_ratio=information_ratio,
                    tracking_error=tracking_error
                )
                
                components.append(component)
        
        # 汇总效应
        total_allocation = sum(c.allocation_effect for c in components)
        total_selection = sum(c.selection_effect for c in components)
        total_interaction = sum(c.interaction_effect for c in components)
        total_attribution = total_allocation + total_selection + total_interaction
        
        # 计算投资组合层面的指标
        portfolio_returns = (Rp * Wp.shift(1)).sum(axis=1)
        benchmark_returns = (Rb * Wb.shift(1)).sum(axis=1)
        active_returns = portfolio_returns - benchmark_returns
        
        tracking_error = active_returns.std()
        information_ratio = active_returns.mean() / tracking_error if tracking_error > 0 else 0
        
        # 计算残差
        portfolio_total_return = (1 + portfolio_returns).prod() - 1
        benchmark_total_return = (1 + benchmark_returns).prod() - 1
        active_total_return = portfolio_total_return - benchmark_total_return
        residual = active_total_return - total_attribution
        
        return {
            'components': components,
            'total_allocation': total_allocation,
            'total_selection': total_selection,
            'total_interaction': total_interaction,
            'total_attribution': total_attribution,
            'residual': residual,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio
        }
    
    async def _brinson_fachler_attribution(self, 
                                         portfolio_data: pd.DataFrame,
                                         benchmark_data: pd.DataFrame) -> Dict[str, Any]:
        """Brinson-Fachler归因方法"""
        # 获取收益率和权重
        Rp = portfolio_data['returns']
        Rb = benchmark_data['returns']
        Wp = portfolio_data['weights']
        Wb = benchmark_data['weights']
        
        components = []
        
        for asset in Rp.columns:
            if asset in Rb.columns:
                # 计算各期权重和收益率
                wp_series = Wp[asset]
                wb_series = Wb[asset]
                rp_series = Rp[asset]
                rb_series = Rb[asset]
                
                # 几何平均
                wp_geom = np.exp(np.log(wp_series + 1e-10).mean()) - 1e-10
                wb_geom = np.exp(np.log(wb_series + 1e-10).mean()) - 1e-10
                rp_geom = np.exp(np.log(rp_series + 1).mean()) - 1
                rb_geom = np.exp(np.log(rb_series + 1).mean()) - 1
                
                # 资产配置效应
                allocation_effect = (wp_geom - wb_geom) * rb_geom
                
                # 证券选择效应
                selection_effect = wb_geom * (rp_geom - rb_geom)
                
                # 交互效应
                interaction_effect = (wp_geom - wb_geom) * (rp_geom - rb_geom)
                
                # 总效应
                total_effect = allocation_effect + selection_effect + interaction_effect
                
                component = AttributionComponent(
                    component_name=asset,
                    component_type="security",
                    allocation_effect=allocation_effect,
                    selection_effect=selection_effect,
                    interaction_effect=interaction_effect,
                    total_effect=total_effect,
                    weight_portfolio=wp_geom,
                    weight_benchmark=wb_geom,
                    return_portfolio=rp_geom,
                    return_benchmark=rb_geom,
                    contribution_portfolio=wp_geom * rp_geom,
                    contribution_benchmark=wb_geom * rb_geom,
                    active_weight=wp_geom - wb_geom,
                    active_return=rp_geom - rb_geom,
                    information_ratio=0,  # 简化处理
                    tracking_error=0      # 简化处理
                )
                
                components.append(component)
        
        # 汇总效应
        total_allocation = sum(c.allocation_effect for c in components)
        total_selection = sum(c.selection_effect for c in components)
        total_interaction = sum(c.interaction_effect for c in components)
        total_attribution = total_allocation + total_selection + total_interaction
        
        return {
            'components': components,
            'total_allocation': total_allocation,
            'total_selection': total_selection,
            'total_interaction': total_interaction,
            'total_attribution': total_attribution,
            'residual': 0,  # 简化处理
            'tracking_error': 0,
            'information_ratio': 0
        }
    
    async def _geometric_attribution(self, 
                                   portfolio_data: pd.DataFrame,
                                   benchmark_data: pd.DataFrame) -> Dict[str, Any]:
        """几何归因方法"""
        # 计算累积收益率
        Rp = portfolio_data['returns']
        Rb = benchmark_data['returns']
        Wp = portfolio_data['weights']
        Wb = benchmark_data['weights']
        
        # 计算投资组合和基准的累积收益率
        portfolio_cumulative = (1 + (Rp * Wp.shift(1)).sum(axis=1)).cumprod()
        benchmark_cumulative = (1 + (Rb * Wb.shift(1)).sum(axis=1)).cumprod()
        
        # 几何归因基于对数收益率
        portfolio_log_returns = np.log(portfolio_cumulative / portfolio_cumulative.shift(1)).dropna()
        benchmark_log_returns = np.log(benchmark_cumulative / benchmark_cumulative.shift(1)).dropna()
        
        components = []
        
        for asset in Rp.columns:
            if asset in Rb.columns:
                # 计算对数收益率
                rp_log = np.log(1 + Rp[asset]).mean()
                rb_log = np.log(1 + Rb[asset]).mean()
                
                wp_avg = Wp[asset].mean()
                wb_avg = Wb[asset].mean()
                
                # 几何归因效应
                allocation_effect = (wp_avg - wb_avg) * rb_log
                selection_effect = wb_avg * (rp_log - rb_log)
                interaction_effect = (wp_avg - wb_avg) * (rp_log - rb_log)
                
                total_effect = allocation_effect + selection_effect + interaction_effect
                
                component = AttributionComponent(
                    component_name=asset,
                    component_type="security",
                    allocation_effect=allocation_effect,
                    selection_effect=selection_effect,
                    interaction_effect=interaction_effect,
                    total_effect=total_effect,
                    weight_portfolio=wp_avg,
                    weight_benchmark=wb_avg,
                    return_portfolio=np.exp(rp_log) - 1,
                    return_benchmark=np.exp(rb_log) - 1,
                    contribution_portfolio=wp_avg * (np.exp(rp_log) - 1),
                    contribution_benchmark=wb_avg * (np.exp(rb_log) - 1),
                    active_weight=wp_avg - wb_avg,
                    active_return=np.exp(rp_log) - np.exp(rb_log),
                    information_ratio=0,
                    tracking_error=0
                )
                
                components.append(component)
        
        # 汇总效应
        total_allocation = sum(c.allocation_effect for c in components)
        total_selection = sum(c.selection_effect for c in components)
        total_interaction = sum(c.interaction_effect for c in components)
        total_attribution = total_allocation + total_selection + total_interaction
        
        return {
            'components': components,
            'total_allocation': total_allocation,
            'total_selection': total_selection,
            'total_interaction': total_interaction,
            'total_attribution': total_attribution,
            'residual': 0,
            'tracking_error': 0,
            'information_ratio': 0
        }
    
    async def _fama_french_attribution(self, 
                                     portfolio_data: pd.DataFrame,
                                     benchmark_data: pd.DataFrame) -> Dict[str, Any]:
        """Fama-French因子归因"""
        # 获取因子收益率数据（简化模拟）
        dates = portfolio_data.index
        
        # 模拟因子收益率
        market_factor = np.random.normal(0.0008, 0.02, len(dates))
        size_factor = np.random.normal(0.0003, 0.015, len(dates))
        value_factor = np.random.normal(0.0002, 0.012, len(dates))
        
        factor_returns = pd.DataFrame({
            'market': market_factor,
            'size': size_factor,
            'value': value_factor
        }, index=dates)
        
        # 计算投资组合和基准的因子暴露
        portfolio_returns = (portfolio_data['returns'] * portfolio_data['weights'].shift(1)).sum(axis=1)
        benchmark_returns = (benchmark_data['returns'] * benchmark_data['weights'].shift(1)).sum(axis=1)
        
        # 简化的因子暴露计算
        portfolio_exposures = {
            'market': 1.0,
            'size': 0.2,
            'value': 0.1
        }
        
        benchmark_exposures = {
            'market': 1.0,
            'size': 0.0,
            'value': 0.0
        }
        
        factor_attribution = []
        
        for factor in ['market', 'size', 'value']:
            factor_return = factor_returns[factor].mean()
            portfolio_exposure = portfolio_exposures[factor]
            benchmark_exposure = benchmark_exposures[factor]
            active_exposure = portfolio_exposure - benchmark_exposure
            
            factor_contribution = active_exposure * factor_return
            
            factor_attr = FactorAttribution(
                factor_name=factor,
                factor_type="style",
                factor_return=factor_return,
                factor_exposure_portfolio=portfolio_exposure,
                factor_exposure_benchmark=benchmark_exposure,
                active_exposure=active_exposure,
                factor_contribution=factor_contribution,
                selection_return=0,  # 简化处理
                allocation_return=0,  # 简化处理
                total_return=factor_contribution,
                risk_contribution=0,  # 简化处理
                information_ratio=0,  # 简化处理
                factor_timing=0,     # 简化处理
                factor_selection=0,  # 简化处理
                volatility=factor_returns[factor].std(),
                sharpe_ratio=factor_return / factor_returns[factor].std() if factor_returns[factor].std() > 0 else 0
            )
            
            factor_attribution.append(factor_attr)
        
        # 计算总归因
        total_factor_attribution = sum(f.factor_contribution for f in factor_attribution)
        
        return {
            'components': [],
            'factor_attribution': factor_attribution,
            'total_allocation': 0,
            'total_selection': 0,
            'total_interaction': 0,
            'total_attribution': total_factor_attribution,
            'residual': 0,
            'tracking_error': 0,
            'information_ratio': 0
        }
    
    async def _multi_factor_attribution(self, 
                                      portfolio_data: pd.DataFrame,
                                      benchmark_data: pd.DataFrame) -> Dict[str, Any]:
        """多因子归因"""
        # 扩展因子模型
        factors = ['market', 'size', 'value', 'momentum', 'quality', 'low_volatility']
        
        # 模拟因子收益率
        dates = portfolio_data.index
        factor_returns = pd.DataFrame({
            factor: np.random.normal(0.0005, 0.01, len(dates))
            for factor in factors
        }, index=dates)
        
        # 模拟因子暴露
        portfolio_exposures = {
            'market': 1.0,
            'size': 0.15,
            'value': 0.08,
            'momentum': -0.05,
            'quality': 0.12,
            'low_volatility': 0.06
        }
        
        benchmark_exposures = {
            'market': 1.0,
            'size': 0.0,
            'value': 0.0,
            'momentum': 0.0,
            'quality': 0.0,
            'low_volatility': 0.0
        }
        
        factor_attribution = []
        
        for factor in factors:
            factor_return = factor_returns[factor].mean()
            portfolio_exposure = portfolio_exposures[factor]
            benchmark_exposure = benchmark_exposures[factor]
            active_exposure = portfolio_exposure - benchmark_exposure
            
            factor_contribution = active_exposure * factor_return
            
            # 计算因子风险贡献
            factor_volatility = factor_returns[factor].std()
            risk_contribution = (active_exposure ** 2) * (factor_volatility ** 2)
            
            factor_attr = FactorAttribution(
                factor_name=factor,
                factor_type="style",
                factor_return=factor_return,
                factor_exposure_portfolio=portfolio_exposure,
                factor_exposure_benchmark=benchmark_exposure,
                active_exposure=active_exposure,
                factor_contribution=factor_contribution,
                selection_return=0,
                allocation_return=0,
                total_return=factor_contribution,
                risk_contribution=risk_contribution,
                information_ratio=factor_return / factor_volatility if factor_volatility > 0 else 0,
                factor_timing=0,
                factor_selection=0,
                volatility=factor_volatility,
                sharpe_ratio=factor_return / factor_volatility if factor_volatility > 0 else 0
            )
            
            factor_attribution.append(factor_attr)
        
        # 计算总归因
        total_factor_attribution = sum(f.factor_contribution for f in factor_attribution)
        total_risk_contribution = sum(f.risk_contribution for f in factor_attribution)
        
        return {
            'components': [],
            'factor_attribution': factor_attribution,
            'total_allocation': 0,
            'total_selection': 0,
            'total_interaction': 0,
            'total_attribution': total_factor_attribution,
            'total_risk_contribution': total_risk_contribution,
            'residual': 0,
            'tracking_error': 0,
            'information_ratio': 0
        }
    
    async def _currency_attribution(self, 
                                  portfolio_data: pd.DataFrame,
                                  benchmark_data: pd.DataFrame) -> Dict[str, Any]:
        """货币归因分析"""
        # 模拟货币数据
        currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CHF']
        dates = portfolio_data.index
        
        currency_attribution = []
        
        for currency in currencies:
            if currency == self.base_currency:
                continue
            
            # 模拟汇率收益率
            currency_return = np.random.normal(0.0001, 0.08, len(dates)).mean()
            
            # 模拟货币暴露
            portfolio_exposure = np.random.uniform(0.05, 0.25)
            benchmark_exposure = np.random.uniform(0.0, 0.15)
            
            # 对冲比率
            hedging_ratio = np.random.uniform(0.0, 1.0)
            
            # 计算货币效应
            currency_pair = f"{currency}/{self.base_currency}"
            unhedged_return = currency_return
            hedged_return = currency_return * (1 - hedging_ratio)
            
            currency_contribution = (portfolio_exposure - benchmark_exposure) * unhedged_return
            hedging_contribution = -portfolio_exposure * hedging_ratio * currency_return
            total_currency_effect = currency_contribution + hedging_contribution
            
            currency_attr = CurrencyAttribution(
                currency_pair=currency_pair,
                currency_return=currency_return,
                currency_exposure_portfolio=portfolio_exposure,
                currency_exposure_benchmark=benchmark_exposure,
                hedging_ratio=hedging_ratio,
                hedged_return=hedged_return,
                unhedged_return=unhedged_return,
                currency_contribution=currency_contribution,
                hedging_contribution=hedging_contribution,
                total_currency_effect=total_currency_effect,
                forward_points=0.0001,  # 简化处理
                spot_return=currency_return,
                carry_return=0.0002,    # 简化处理
                volatility=0.08
            )
            
            currency_attribution.append(currency_attr)
        
        # 计算总货币归因
        total_currency_attribution = sum(c.total_currency_effect for c in currency_attribution)
        
        return {
            'components': [],
            'currency_attribution': currency_attribution,
            'total_allocation': 0,
            'total_selection': 0,
            'total_interaction': 0,
            'total_attribution': total_currency_attribution,
            'residual': 0,
            'tracking_error': 0,
            'information_ratio': 0
        }
    
    async def _calculate_risk_attribution(self, 
                                        portfolio_data: pd.DataFrame,
                                        benchmark_data: pd.DataFrame) -> Dict[str, float]:
        """计算风险归因"""
        # 计算投资组合和基准的收益率
        portfolio_returns = (portfolio_data['returns'] * portfolio_data['weights'].shift(1)).sum(axis=1)
        benchmark_returns = (benchmark_data['returns'] * benchmark_data['weights'].shift(1)).sum(axis=1)
        
        # 计算风险指标
        portfolio_volatility = portfolio_returns.std()
        benchmark_volatility = benchmark_returns.std()
        active_volatility = (portfolio_returns - benchmark_returns).std()
        
        # 计算风险分解
        systematic_risk = portfolio_volatility ** 2 - active_volatility ** 2
        idiosyncratic_risk = active_volatility ** 2
        
        return {
            'total_risk': portfolio_volatility ** 2,
            'systematic_risk': systematic_risk,
            'idiosyncratic_risk': idiosyncratic_risk,
            'active_risk': active_volatility ** 2,
            'tracking_error': active_volatility,
            'risk_ratio': systematic_risk / (portfolio_volatility ** 2) if portfolio_volatility > 0 else 0
        }
    
    async def _calculate_attribution_quality(self, 
                                           attribution_result: Dict[str, Any],
                                           portfolio_return: float,
                                           benchmark_return: float) -> Dict[str, float]:
        """计算归因质量指标"""
        active_return = portfolio_return - benchmark_return
        total_attribution = attribution_result['total_attribution']
        residual = attribution_result['residual']
        
        # 归因完整性
        attribution_completeness = abs(total_attribution) / abs(active_return) if active_return != 0 else 0
        
        # 归因精度
        attribution_accuracy = 1 - abs(residual) / abs(active_return) if active_return != 0 else 0
        
        # 归因稳定性
        attribution_stability = 1 - abs(residual) / abs(total_attribution) if total_attribution != 0 else 0
        
        return {
            'completeness': attribution_completeness,
            'accuracy': attribution_accuracy,
            'stability': attribution_stability,
            'residual_ratio': abs(residual) / abs(active_return) if active_return != 0 else 0,
            'r_squared': 1 - (residual ** 2) / (active_return ** 2) if active_return != 0 else 0
        }
    
    async def _identify_top_contributors(self, 
                                       attribution_result: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """识别主要贡献者和拖累者"""
        components = attribution_result.get('components', [])
        
        if not components:
            return [], []
        
        # 按总效应排序
        sorted_components = sorted(components, key=lambda x: x.total_effect, reverse=True)
        
        # 前5名贡献者
        top_contributors = [c.component_name for c in sorted_components[:5] if c.total_effect > 0]
        
        # 前5名拖累者
        top_detractors = [c.component_name for c in sorted_components[-5:] if c.total_effect < 0]
        top_detractors.reverse()  # 从最大拖累者开始
        
        return top_contributors, top_detractors
    
    async def generate_attribution_report(self, attribution_summary: AttributionSummary) -> Dict[str, Any]:
        """生成归因报告"""
        report = {
            'executive_summary': {
                'period': f"{attribution_summary.period_start.strftime('%Y-%m-%d')} to {attribution_summary.period_end.strftime('%Y-%m-%d')}",
                'portfolio_return': f"{attribution_summary.portfolio_return:.2%}",
                'benchmark_return': f"{attribution_summary.benchmark_return:.2%}",
                'active_return': f"{attribution_summary.active_return:.2%}",
                'total_attribution': f"{attribution_summary.total_attribution:.2%}",
                'attribution_residual': f"{attribution_summary.attribution_residual:.2%}",
                'information_ratio': f"{attribution_summary.information_ratio:.2f}",
                'tracking_error': f"{attribution_summary.tracking_error:.2%}"
            },
            'attribution_breakdown': {
                'allocation_effect': f"{attribution_summary.total_allocation_effect:.2%}",
                'selection_effect': f"{attribution_summary.total_selection_effect:.2%}",
                'interaction_effect': f"{attribution_summary.total_interaction_effect:.2%}",
                'allocation_percentage': f"{attribution_summary.total_allocation_effect / attribution_summary.active_return * 100:.1f}%" if attribution_summary.active_return != 0 else "N/A",
                'selection_percentage': f"{attribution_summary.total_selection_effect / attribution_summary.active_return * 100:.1f}%" if attribution_summary.active_return != 0 else "N/A"
            },
            'top_contributors': attribution_summary.top_contributors,
            'top_detractors': attribution_summary.top_detractors,
            'attribution_quality': {
                'completeness': f"{attribution_summary.attribution_quality.get('completeness', 0):.1%}",
                'accuracy': f"{attribution_summary.attribution_quality.get('accuracy', 0):.1%}",
                'stability': f"{attribution_summary.attribution_quality.get('stability', 0):.1%}",
                'r_squared': f"{attribution_summary.attribution_quality.get('r_squared', 0):.2f}"
            },
            'risk_attribution': attribution_summary.risk_attribution,
            'component_details': [
                {
                    'name': c.component_name,
                    'allocation_effect': f"{c.allocation_effect:.4f}",
                    'selection_effect': f"{c.selection_effect:.4f}",
                    'total_effect': f"{c.total_effect:.4f}",
                    'active_weight': f"{c.active_weight:.2%}",
                    'active_return': f"{c.active_return:.2%}",
                    'information_ratio': f"{c.information_ratio:.2f}"
                }
                for c in attribution_summary.attribution_components
            ]
        }
        
        return report
    
    async def visualize_attribution(self, attribution_summary: AttributionSummary, 
                                  save_path: str = None):
        """可视化归因分析"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 归因分解
        categories = ['Allocation', 'Selection', 'Interaction']
        values = [
            attribution_summary.total_allocation_effect,
            attribution_summary.total_selection_effect,
            attribution_summary.total_interaction_effect
        ]
        
        axes[0, 0].bar(categories, values, color=['blue', 'green', 'orange'])
        axes[0, 0].set_title('Attribution Breakdown')
        axes[0, 0].set_ylabel('Contribution')
        axes[0, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 组件贡献
        if attribution_summary.attribution_components:
            components = attribution_summary.attribution_components[:10]  # 前10个
            comp_names = [c.component_name for c in components]
            comp_effects = [c.total_effect for c in components]
            
            colors = ['green' if e > 0 else 'red' for e in comp_effects]
            axes[0, 1].barh(comp_names, comp_effects, color=colors)
            axes[0, 1].set_title('Component Contributions')
            axes[0, 1].set_xlabel('Total Effect')
        
        # 因子归因
        if attribution_summary.factor_attribution:
            factors = [f.factor_name for f in attribution_summary.factor_attribution]
            factor_contributions = [f.factor_contribution for f in attribution_summary.factor_attribution]
            
            axes[1, 0].bar(factors, factor_contributions, color='purple', alpha=0.7)
            axes[1, 0].set_title('Factor Attribution')
            axes[1, 0].set_ylabel('Factor Contribution')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 风险分解
        if attribution_summary.risk_attribution:
            risk_types = list(attribution_summary.risk_attribution.keys())
            risk_values = list(attribution_summary.risk_attribution.values())
            
            axes[1, 1].pie(risk_values, labels=risk_types, autopct='%1.1f%%')
            axes[1, 1].set_title('Risk Attribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    async def calculate_rolling_attribution(self, 
                                          portfolio_returns: pd.DataFrame,
                                          portfolio_weights: pd.DataFrame,
                                          benchmark_returns: pd.DataFrame,
                                          benchmark_weights: pd.DataFrame,
                                          window_size: int = 252) -> pd.DataFrame:
        """计算滚动归因"""
        results = []
        
        for i in range(window_size, len(portfolio_returns)):
            start_idx = i - window_size
            end_idx = i
            
            # 提取窗口数据
            p_returns = portfolio_returns.iloc[start_idx:end_idx]
            p_weights = portfolio_weights.iloc[start_idx:end_idx]
            b_returns = benchmark_returns.iloc[start_idx:end_idx]
            b_weights = benchmark_weights.iloc[start_idx:end_idx]
            
            # 计算归因
            attribution = await self.calculate_attribution(
                p_returns, p_weights, b_returns, b_weights
            )
            
            results.append({
                'date': portfolio_returns.index[end_idx-1],
                'active_return': attribution.active_return,
                'allocation_effect': attribution.total_allocation_effect,
                'selection_effect': attribution.total_selection_effect,
                'interaction_effect': attribution.total_interaction_effect,
                'tracking_error': attribution.tracking_error,
                'information_ratio': attribution.information_ratio
            })
        
        return pd.DataFrame(results).set_index('date')
    
    async def compare_attribution_methods(self, 
                                        portfolio_returns: pd.DataFrame,
                                        portfolio_weights: pd.DataFrame,
                                        benchmark_returns: pd.DataFrame,
                                        benchmark_weights: pd.DataFrame) -> Dict[str, AttributionSummary]:
        """比较不同归因方法"""
        methods = [
            AttributionMethod.BRINSON_HOOD_BEEBOWER,
            AttributionMethod.BRINSON_FACHLER,
            AttributionMethod.GEOMETRIC_ATTRIBUTION
        ]
        
        results = {}
        
        for method in methods:
            try:
                attribution = await self.calculate_attribution(
                    portfolio_returns, portfolio_weights,
                    benchmark_returns, benchmark_weights,
                    method=method
                )
                results[method.value] = attribution
            except Exception as e:
                self.logger.error(f"归因方法 {method.value} 计算失败: {e}")
        
        return results
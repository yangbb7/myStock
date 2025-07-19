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

class ESGCategory(Enum):
    ENVIRONMENTAL = "environmental"
    SOCIAL = "social"
    GOVERNANCE = "governance"
    OVERALL = "overall"

class ESGRatingProvider(Enum):
    MSCI = "msci"
    SUSTAINALYTICS = "sustainalytics"
    REFINITIV = "refinitiv"
    BLOOMBERG = "bloomberg"
    FTSE_RUSSELL = "ftse_russell"
    CDP = "cdp"
    SASB = "sasb"
    CUSTOM = "custom"

class ESGIntegrationMethod(Enum):
    SCREENING = "screening"
    INTEGRATION = "integration"
    THEMATIC = "thematic"
    ENGAGEMENT = "engagement"
    IMPACT = "impact"
    BEST_IN_CLASS = "best_in_class"
    NEGATIVE_SCREENING = "negative_screening"
    POSITIVE_SCREENING = "positive_screening"
    NORM_BASED_SCREENING = "norm_based_screening"
    ESG_MOMENTUM = "esg_momentum"

class ESGRiskType(Enum):
    TRANSITION_RISK = "transition_risk"
    PHYSICAL_RISK = "physical_risk"
    REGULATORY_RISK = "regulatory_risk"
    REPUTATION_RISK = "reputation_risk"
    LITIGATION_RISK = "litigation_risk"
    STRANDED_ASSETS = "stranded_assets"
    SUPPLY_CHAIN_RISK = "supply_chain_risk"
    TECHNOLOGY_RISK = "technology_risk"

@dataclass
class ESGScore:
    """ESG评分"""
    asset_id: str
    provider: ESGRatingProvider
    environmental_score: float
    social_score: float
    governance_score: float
    overall_score: float
    score_date: datetime
    percentile_rank: float
    industry_group: str
    controversy_score: float = 0.0
    carbon_intensity: float = 0.0
    water_intensity: float = 0.0
    waste_intensity: float = 0.0
    data_quality: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ESGRiskMetrics:
    """ESG风险指标"""
    asset_id: str
    transition_risk: float
    physical_risk: float
    regulatory_risk: float
    reputation_risk: float
    overall_esg_risk: float
    carbon_footprint: float
    water_footprint: float
    waste_footprint: float
    climate_var: float
    stranded_asset_risk: float
    supply_chain_risk: float
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class ESGConstraint:
    """ESG约束条件"""
    constraint_type: str
    category: ESGCategory
    threshold: float
    operator: str  # 'min', 'max', 'range', 'exclude'
    exclusion_list: List[str] = field(default_factory=list)
    inclusion_list: List[str] = field(default_factory=list)
    weight_limit: float = 1.0
    priority: int = 1

@dataclass
class ESGPortfolioMetrics:
    """ESG组合指标"""
    portfolio_id: str
    weighted_avg_esg_score: float
    environmental_score: float
    social_score: float
    governance_score: float
    esg_risk_score: float
    carbon_intensity: float
    water_intensity: float
    waste_intensity: float
    esg_momentum: float
    controversy_exposure: float
    stranded_asset_exposure: float
    alignment_score: float
    diversification_score: float
    last_updated: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ESGOptimizationResult:
    """ESG优化结果"""
    optimization_method: ESGIntegrationMethod
    optimal_weights: np.ndarray
    asset_ids: List[str]
    portfolio_metrics: ESGPortfolioMetrics
    esg_constraints_satisfied: bool
    traditional_metrics: Dict[str, float]
    esg_risk_metrics: Dict[str, float]
    excluded_assets: List[str]
    tilted_weights: Dict[str, float]
    optimization_success: bool
    objective_value: float
    tracking_error: float
    esg_improvement: float
    cost_of_esg: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class ESGIntegrationEngine:
    """
    ESG整合引擎
    
    提供全面的ESG投资整合功能，包括：
    - ESG评分数据管理
    - ESG约束优化
    - ESG风险建模
    - ESG投资组合构建
    - ESG绩效归因分析
    - ESG报告生成
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 基础配置
        self.default_provider = ESGRatingProvider(config.get('default_provider', 'msci'))
        self.min_esg_score = config.get('min_esg_score', 3.0)
        self.max_esg_score = config.get('max_esg_score', 10.0)
        self.esg_weight = config.get('esg_weight', 0.3)
        
        # 约束参数
        self.carbon_intensity_limit = config.get('carbon_intensity_limit', 100.0)
        self.controversy_threshold = config.get('controversy_threshold', 5.0)
        self.min_data_quality = config.get('min_data_quality', 0.7)
        
        # 优化参数
        self.optimization_method = config.get('optimization_method', 'SLSQP')
        self.max_iterations = config.get('max_iterations', 1000)
        self.tolerance = config.get('tolerance', 1e-6)
        
        # 风险参数
        self.climate_var_threshold = config.get('climate_var_threshold', 0.05)
        self.stranded_asset_threshold = config.get('stranded_asset_threshold', 0.1)
        
        # 数据存储
        self.esg_scores: Dict[str, ESGScore] = {}
        self.esg_risk_metrics: Dict[str, ESGRiskMetrics] = {}
        self.portfolio_history: List[ESGOptimizationResult] = []
        
        # 行业映射
        self.industry_mappings = config.get('industry_mappings', {})
        self.sector_carbon_intensities = config.get('sector_carbon_intensities', {})
        
        # 执行器
        self.executor = ThreadPoolExecutor(max_workers=config.get('max_workers', 4))
        
        # 缓存
        self.correlation_cache = {}
        self.regression_cache = {}
        
        self.logger.info(f"ESG integration engine initialized with {self.default_provider.value} provider")
    
    async def load_esg_data(self, 
                           asset_ids: List[str],
                           data_sources: Dict[str, Any],
                           **kwargs) -> Dict[str, ESGScore]:
        """
        加载ESG数据
        """
        try:
            self.logger.info(f"Loading ESG data for {len(asset_ids)} assets")
            
            esg_scores = {}
            
            for asset_id in asset_ids:
                # 从不同数据源加载ESG数据
                if 'esg_scores' in data_sources and asset_id in data_sources['esg_scores']:
                    raw_score = data_sources['esg_scores'][asset_id]
                    
                    esg_score = ESGScore(
                        asset_id=asset_id,
                        provider=self.default_provider,
                        environmental_score=raw_score.get('environmental', 5.0),
                        social_score=raw_score.get('social', 5.0),
                        governance_score=raw_score.get('governance', 5.0),
                        overall_score=raw_score.get('overall', 5.0),
                        score_date=datetime.now(),
                        percentile_rank=raw_score.get('percentile_rank', 50.0),
                        industry_group=raw_score.get('industry_group', 'Unknown'),
                        controversy_score=raw_score.get('controversy_score', 0.0),
                        carbon_intensity=raw_score.get('carbon_intensity', 0.0),
                        water_intensity=raw_score.get('water_intensity', 0.0),
                        waste_intensity=raw_score.get('waste_intensity', 0.0),
                        data_quality=raw_score.get('data_quality', 1.0)
                    )
                    
                    esg_scores[asset_id] = esg_score
                    self.esg_scores[asset_id] = esg_score
                else:
                    # 生成模拟ESG数据
                    esg_score = await self._generate_synthetic_esg_score(asset_id, **kwargs)
                    esg_scores[asset_id] = esg_score
                    self.esg_scores[asset_id] = esg_score
            
            # 加载ESG风险指标
            await self._load_esg_risk_metrics(asset_ids, data_sources, **kwargs)
            
            self.logger.info(f"Successfully loaded ESG data for {len(esg_scores)} assets")
            
            return esg_scores
            
        except Exception as e:
            self.logger.error(f"Error loading ESG data: {e}")
            raise
    
    async def optimize_esg_portfolio(self, 
                                   asset_ids: List[str],
                                   expected_returns: np.ndarray,
                                   covariance_matrix: np.ndarray,
                                   integration_method: ESGIntegrationMethod,
                                   esg_constraints: List[ESGConstraint],
                                   benchmark_weights: Optional[np.ndarray] = None,
                                   **kwargs) -> ESGOptimizationResult:
        """
        ESG投资组合优化
        """
        try:
            self.logger.info(f"Starting ESG portfolio optimization using {integration_method.value}")
            
            # 验证ESG数据
            if not self._validate_esg_data(asset_ids):
                raise ValueError("ESG data validation failed")
            
            # 选择整合方法
            if integration_method == ESGIntegrationMethod.SCREENING:
                result = await self._optimize_screening(
                    asset_ids, expected_returns, covariance_matrix, esg_constraints, **kwargs
                )
            elif integration_method == ESGIntegrationMethod.INTEGRATION:
                result = await self._optimize_integration(
                    asset_ids, expected_returns, covariance_matrix, esg_constraints, **kwargs
                )
            elif integration_method == ESGIntegrationMethod.THEMATIC:
                result = await self._optimize_thematic(
                    asset_ids, expected_returns, covariance_matrix, esg_constraints, **kwargs
                )
            elif integration_method == ESGIntegrationMethod.BEST_IN_CLASS:
                result = await self._optimize_best_in_class(
                    asset_ids, expected_returns, covariance_matrix, esg_constraints, **kwargs
                )
            elif integration_method == ESGIntegrationMethod.NEGATIVE_SCREENING:
                result = await self._optimize_negative_screening(
                    asset_ids, expected_returns, covariance_matrix, esg_constraints, **kwargs
                )
            elif integration_method == ESGIntegrationMethod.POSITIVE_SCREENING:
                result = await self._optimize_positive_screening(
                    asset_ids, expected_returns, covariance_matrix, esg_constraints, **kwargs
                )
            elif integration_method == ESGIntegrationMethod.ESG_MOMENTUM:
                result = await self._optimize_esg_momentum(
                    asset_ids, expected_returns, covariance_matrix, esg_constraints, **kwargs
                )
            else:
                raise ValueError(f"Unsupported integration method: {integration_method}")
            
            # 计算传统指标
            traditional_metrics = await self._calculate_traditional_metrics(
                result.optimal_weights, expected_returns, covariance_matrix, **kwargs
            )
            result.traditional_metrics = traditional_metrics
            
            # 计算ESG改善
            if benchmark_weights is not None:
                result.esg_improvement = await self._calculate_esg_improvement(
                    result.optimal_weights, benchmark_weights, asset_ids
                )
            
            # 计算ESG成本
            result.cost_of_esg = await self._calculate_esg_cost(
                result.optimal_weights, benchmark_weights, expected_returns, covariance_matrix
            )
            
            # 保存结果
            self.portfolio_history.append(result)
            
            self.logger.info(f"ESG portfolio optimization completed successfully")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in ESG portfolio optimization: {e}")
            raise
    
    async def _optimize_screening(self, 
                                asset_ids: List[str],
                                expected_returns: np.ndarray,
                                covariance_matrix: np.ndarray,
                                esg_constraints: List[ESGConstraint],
                                **kwargs) -> ESGOptimizationResult:
        """
        ESG筛选优化
        """
        try:
            # 应用ESG筛选
            eligible_assets = []
            eligible_indices = []
            
            for i, asset_id in enumerate(asset_ids):
                if asset_id in self.esg_scores:
                    esg_score = self.esg_scores[asset_id]
                    
                    # 检查是否满足ESG约束
                    if await self._check_esg_constraints(esg_score, esg_constraints):
                        eligible_assets.append(asset_id)
                        eligible_indices.append(i)
            
            if not eligible_assets:
                raise ValueError("No assets pass ESG screening criteria")
            
            # 构建筛选后的数据
            filtered_returns = expected_returns[eligible_indices]
            filtered_covariance = covariance_matrix[np.ix_(eligible_indices, eligible_indices)]
            
            # 优化筛选后的组合
            def objective_function(weights):
                portfolio_return = np.dot(weights, filtered_returns)
                portfolio_variance = np.dot(weights, np.dot(filtered_covariance, weights))
                
                # 最大化夏普比率
                return -(portfolio_return / np.sqrt(portfolio_variance))
            
            # 约束条件
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # 权重和为1
            ]
            
            # 权重界限
            bounds = [(0.0, 1.0) for _ in eligible_assets]
            
            # 初始权重
            initial_weights = np.ones(len(eligible_assets)) / len(eligible_assets)
            
            # 优化
            result = minimize(
                objective_function,
                initial_weights,
                method=self.optimization_method,
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
            )
            
            # 构建完整的权重向量
            optimal_weights = np.zeros(len(asset_ids))
            for i, idx in enumerate(eligible_indices):
                optimal_weights[idx] = result.x[i]
            
            # 计算组合ESG指标
            portfolio_metrics = await self._calculate_portfolio_esg_metrics(
                optimal_weights, asset_ids, "screening_portfolio"
            )
            
            # 被排除的资产
            excluded_assets = [asset_id for asset_id in asset_ids if asset_id not in eligible_assets]
            
            return ESGOptimizationResult(
                optimization_method=ESGIntegrationMethod.SCREENING,
                optimal_weights=optimal_weights,
                asset_ids=asset_ids,
                portfolio_metrics=portfolio_metrics,
                esg_constraints_satisfied=True,
                traditional_metrics={},
                esg_risk_metrics={},
                excluded_assets=excluded_assets,
                tilted_weights={},
                optimization_success=result.success,
                objective_value=result.fun,
                tracking_error=0.0,
                esg_improvement=0.0,
                cost_of_esg=0.0
            )
            
        except Exception as e:
            self.logger.error(f"Error in screening optimization: {e}")
            raise
    
    async def _optimize_integration(self, 
                                  asset_ids: List[str],
                                  expected_returns: np.ndarray,
                                  covariance_matrix: np.ndarray,
                                  esg_constraints: List[ESGConstraint],
                                  **kwargs) -> ESGOptimizationResult:
        """
        ESG整合优化
        """
        try:
            # 构建ESG调整后的预期收益
            esg_adjusted_returns = await self._calculate_esg_adjusted_returns(
                asset_ids, expected_returns, **kwargs
            )
            
            # 构建ESG调整后的风险模型
            esg_adjusted_covariance = await self._calculate_esg_adjusted_covariance(
                asset_ids, covariance_matrix, **kwargs
            )
            
            # 多目标优化函数
            def objective_function(weights):
                # 传统收益-风险目标
                portfolio_return = np.dot(weights, esg_adjusted_returns)
                portfolio_variance = np.dot(weights, np.dot(esg_adjusted_covariance, weights))
                
                # ESG目标
                esg_score = self._calculate_portfolio_esg_score(weights, asset_ids)
                
                # 组合目标函数
                traditional_utility = portfolio_return - 0.5 * portfolio_variance
                esg_utility = self.esg_weight * esg_score
                
                return -(traditional_utility + esg_utility)
            
            # 约束条件
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # 权重和为1
            ]
            
            # 添加ESG约束
            for constraint in esg_constraints:
                if constraint.constraint_type == 'min_esg_score':
                    def esg_constraint(w, threshold=constraint.threshold):
                        return self._calculate_portfolio_esg_score(w, asset_ids) - threshold
                    constraints.append({'type': 'ineq', 'fun': esg_constraint})
                
                elif constraint.constraint_type == 'max_carbon_intensity':
                    def carbon_constraint(w, threshold=constraint.threshold):
                        return threshold - self._calculate_portfolio_carbon_intensity(w, asset_ids)
                    constraints.append({'type': 'ineq', 'fun': carbon_constraint})
            
            # 权重界限
            bounds = [(0.0, 1.0) for _ in asset_ids]
            
            # 初始权重
            initial_weights = np.ones(len(asset_ids)) / len(asset_ids)
            
            # 优化
            result = minimize(
                objective_function,
                initial_weights,
                method=self.optimization_method,
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
            )
            
            # 计算组合ESG指标
            portfolio_metrics = await self._calculate_portfolio_esg_metrics(
                result.x, asset_ids, "integration_portfolio"
            )
            
            # 计算倾斜权重
            equal_weights = np.ones(len(asset_ids)) / len(asset_ids)
            tilted_weights = {}
            for i, asset_id in enumerate(asset_ids):
                tilt = result.x[i] - equal_weights[i]
                if abs(tilt) > 0.001:  # 0.1%的倾斜阈值
                    tilted_weights[asset_id] = tilt
            
            return ESGOptimizationResult(
                optimization_method=ESGIntegrationMethod.INTEGRATION,
                optimal_weights=result.x,
                asset_ids=asset_ids,
                portfolio_metrics=portfolio_metrics,
                esg_constraints_satisfied=result.success,
                traditional_metrics={},
                esg_risk_metrics={},
                excluded_assets=[],
                tilted_weights=tilted_weights,
                optimization_success=result.success,
                objective_value=result.fun,
                tracking_error=0.0,
                esg_improvement=0.0,
                cost_of_esg=0.0
            )
            
        except Exception as e:
            self.logger.error(f"Error in integration optimization: {e}")
            raise
    
    async def _optimize_best_in_class(self, 
                                    asset_ids: List[str],
                                    expected_returns: np.ndarray,
                                    covariance_matrix: np.ndarray,
                                    esg_constraints: List[ESGConstraint],
                                    **kwargs) -> ESGOptimizationResult:
        """
        同类最佳ESG优化
        """
        try:
            # 按行业分组
            industry_groups = {}
            for asset_id in asset_ids:
                if asset_id in self.esg_scores:
                    industry = self.esg_scores[asset_id].industry_group
                    if industry not in industry_groups:
                        industry_groups[industry] = []
                    industry_groups[industry].append(asset_id)
            
            # 选择每个行业的最佳ESG资产
            best_in_class_assets = []
            best_in_class_indices = []
            
            for industry, assets in industry_groups.items():
                if not assets:
                    continue
                
                # 按ESG得分排序
                industry_scores = [
                    (asset_id, self.esg_scores[asset_id].overall_score)
                    for asset_id in assets if asset_id in self.esg_scores
                ]
                
                if industry_scores:
                    # 选择前25%的资产
                    num_to_select = max(1, len(industry_scores) // 4)
                    industry_scores.sort(key=lambda x: x[1], reverse=True)
                    
                    for i in range(num_to_select):
                        asset_id = industry_scores[i][0]
                        asset_index = asset_ids.index(asset_id)
                        best_in_class_assets.append(asset_id)
                        best_in_class_indices.append(asset_index)
            
            if not best_in_class_assets:
                raise ValueError("No best-in-class assets found")
            
            # 构建筛选后的数据
            filtered_returns = expected_returns[best_in_class_indices]
            filtered_covariance = covariance_matrix[np.ix_(best_in_class_indices, best_in_class_indices)]
            
            # 优化最佳资产组合
            def objective_function(weights):
                portfolio_return = np.dot(weights, filtered_returns)
                portfolio_variance = np.dot(weights, np.dot(filtered_covariance, weights))
                
                # ESG增强的效用函数
                esg_score = self._calculate_portfolio_esg_score(weights, best_in_class_assets)
                
                utility = portfolio_return - 0.5 * portfolio_variance + self.esg_weight * esg_score
                return -utility
            
            # 约束条件
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
            ]
            
            # 权重界限
            bounds = [(0.0, 1.0) for _ in best_in_class_assets]
            
            # 初始权重
            initial_weights = np.ones(len(best_in_class_assets)) / len(best_in_class_assets)
            
            # 优化
            result = minimize(
                objective_function,
                initial_weights,
                method=self.optimization_method,
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
            )
            
            # 构建完整的权重向量
            optimal_weights = np.zeros(len(asset_ids))
            for i, idx in enumerate(best_in_class_indices):
                optimal_weights[idx] = result.x[i]
            
            # 计算组合ESG指标
            portfolio_metrics = await self._calculate_portfolio_esg_metrics(
                optimal_weights, asset_ids, "best_in_class_portfolio"
            )
            
            # 被排除的资产
            excluded_assets = [asset_id for asset_id in asset_ids if asset_id not in best_in_class_assets]
            
            return ESGOptimizationResult(
                optimization_method=ESGIntegrationMethod.BEST_IN_CLASS,
                optimal_weights=optimal_weights,
                asset_ids=asset_ids,
                portfolio_metrics=portfolio_metrics,
                esg_constraints_satisfied=True,
                traditional_metrics={},
                esg_risk_metrics={},
                excluded_assets=excluded_assets,
                tilted_weights={},
                optimization_success=result.success,
                objective_value=result.fun,
                tracking_error=0.0,
                esg_improvement=0.0,
                cost_of_esg=0.0,
                metadata={'industry_groups': len(industry_groups)}
            )
            
        except Exception as e:
            self.logger.error(f"Error in best-in-class optimization: {e}")
            raise
    
    async def _optimize_esg_momentum(self, 
                                   asset_ids: List[str],
                                   expected_returns: np.ndarray,
                                   covariance_matrix: np.ndarray,
                                   esg_constraints: List[ESGConstraint],
                                   **kwargs) -> ESGOptimizationResult:
        """
        ESG动量优化
        """
        try:
            # 计算ESG动量分数
            esg_momentum_scores = await self._calculate_esg_momentum_scores(asset_ids, **kwargs)
            
            # 构建动量调整的收益率
            momentum_adjustment = np.array([
                esg_momentum_scores.get(asset_id, 0.0) * 0.02  # 2%的动量调整
                for asset_id in asset_ids
            ])
            
            momentum_adjusted_returns = expected_returns + momentum_adjustment
            
            # 优化函数
            def objective_function(weights):
                portfolio_return = np.dot(weights, momentum_adjusted_returns)
                portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
                
                # ESG动量因子
                momentum_factor = np.dot(weights, [esg_momentum_scores.get(asset_id, 0.0) for asset_id in asset_ids])
                
                # 组合效用函数
                utility = portfolio_return - 0.5 * portfolio_variance + 0.1 * momentum_factor
                return -utility
            
            # 约束条件
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
            ]
            
            # 添加ESG约束
            for constraint in esg_constraints:
                if constraint.constraint_type == 'min_esg_momentum':
                    def momentum_constraint(w, threshold=constraint.threshold):
                        momentum = np.dot(w, [esg_momentum_scores.get(asset_id, 0.0) for asset_id in asset_ids])
                        return momentum - threshold
                    constraints.append({'type': 'ineq', 'fun': momentum_constraint})
            
            # 权重界限
            bounds = [(0.0, 1.0) for _ in asset_ids]
            
            # 初始权重
            initial_weights = np.ones(len(asset_ids)) / len(asset_ids)
            
            # 优化
            result = minimize(
                objective_function,
                initial_weights,
                method=self.optimization_method,
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
            )
            
            # 计算组合ESG指标
            portfolio_metrics = await self._calculate_portfolio_esg_metrics(
                result.x, asset_ids, "esg_momentum_portfolio"
            )
            
            # 添加动量指标
            portfolio_momentum = np.dot(result.x, [esg_momentum_scores.get(asset_id, 0.0) for asset_id in asset_ids])
            portfolio_metrics.esg_momentum = portfolio_momentum
            
            return ESGOptimizationResult(
                optimization_method=ESGIntegrationMethod.ESG_MOMENTUM,
                optimal_weights=result.x,
                asset_ids=asset_ids,
                portfolio_metrics=portfolio_metrics,
                esg_constraints_satisfied=result.success,
                traditional_metrics={},
                esg_risk_metrics={},
                excluded_assets=[],
                tilted_weights={},
                optimization_success=result.success,
                objective_value=result.fun,
                tracking_error=0.0,
                esg_improvement=0.0,
                cost_of_esg=0.0,
                metadata={'momentum_scores': esg_momentum_scores}
            )
            
        except Exception as e:
            self.logger.error(f"Error in ESG momentum optimization: {e}")
            raise
    
    async def analyze_esg_performance(self, 
                                    portfolio_result: ESGOptimizationResult,
                                    benchmark_weights: np.ndarray,
                                    market_data: Dict[str, Any],
                                    **kwargs) -> Dict[str, Any]:
        """
        分析ESG绩效
        """
        try:
            # 计算基准组合的ESG指标
            benchmark_metrics = await self._calculate_portfolio_esg_metrics(
                benchmark_weights, portfolio_result.asset_ids, "benchmark_portfolio"
            )
            
            # ESG绩效归因
            esg_attribution = await self._calculate_esg_attribution(
                portfolio_result, benchmark_weights, market_data
            )
            
            # 风险调整后的ESG表现
            risk_adjusted_esg = await self._calculate_risk_adjusted_esg_performance(
                portfolio_result, benchmark_weights, market_data
            )
            
            # ESG因子暴露分析
            factor_exposure = await self._analyze_esg_factor_exposure(
                portfolio_result, market_data
            )
            
            # 气候风险分析
            climate_risk_analysis = await self._analyze_climate_risk(
                portfolio_result, market_data
            )
            
            performance_analysis = {
                'portfolio_metrics': portfolio_result.portfolio_metrics,
                'benchmark_metrics': benchmark_metrics,
                'esg_attribution': esg_attribution,
                'risk_adjusted_esg': risk_adjusted_esg,
                'factor_exposure': factor_exposure,
                'climate_risk_analysis': climate_risk_analysis,
                'esg_improvement': {
                    'overall_score': portfolio_result.portfolio_metrics.weighted_avg_esg_score - benchmark_metrics.weighted_avg_esg_score,
                    'environmental': portfolio_result.portfolio_metrics.environmental_score - benchmark_metrics.environmental_score,
                    'social': portfolio_result.portfolio_metrics.social_score - benchmark_metrics.social_score,
                    'governance': portfolio_result.portfolio_metrics.governance_score - benchmark_metrics.governance_score,
                    'carbon_intensity': benchmark_metrics.carbon_intensity - portfolio_result.portfolio_metrics.carbon_intensity,
                    'controversy_reduction': benchmark_metrics.controversy_exposure - portfolio_result.portfolio_metrics.controversy_exposure
                },
                'cost_benefit_analysis': {
                    'esg_cost': portfolio_result.cost_of_esg,
                    'tracking_error': portfolio_result.tracking_error,
                    'information_ratio': portfolio_result.esg_improvement / portfolio_result.tracking_error if portfolio_result.tracking_error > 0 else 0,
                    'esg_efficiency': portfolio_result.esg_improvement / portfolio_result.cost_of_esg if portfolio_result.cost_of_esg > 0 else 0
                }
            }
            
            return performance_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing ESG performance: {e}")
            return {}
    
    # 辅助方法
    async def _generate_synthetic_esg_score(self, asset_id: str, **kwargs) -> ESGScore:
        """生成模拟ESG评分"""
        try:
            # 生成随机但相关的ESG评分
            np.random.seed(hash(asset_id) % 2**32)
            
            # 基础分数
            base_score = np.random.uniform(3.0, 8.0)
            
            # 各维度分数（有相关性）
            environmental = np.clip(base_score + np.random.normal(0, 1.0), 0, 10)
            social = np.clip(base_score + np.random.normal(0, 1.0), 0, 10)
            governance = np.clip(base_score + np.random.normal(0, 1.0), 0, 10)
            overall = (environmental + social + governance) / 3
            
            # 其他指标
            percentile_rank = norm.cdf(overall, 5.0, 1.5) * 100
            controversy_score = max(0, np.random.exponential(1.0))
            carbon_intensity = np.random.lognormal(3.0, 1.0)
            
            return ESGScore(
                asset_id=asset_id,
                provider=self.default_provider,
                environmental_score=environmental,
                social_score=social,
                governance_score=governance,
                overall_score=overall,
                score_date=datetime.now(),
                percentile_rank=percentile_rank,
                industry_group=np.random.choice(['Technology', 'Healthcare', 'Financial', 'Energy', 'Consumer']),
                controversy_score=controversy_score,
                carbon_intensity=carbon_intensity,
                water_intensity=np.random.lognormal(2.0, 0.5),
                waste_intensity=np.random.lognormal(1.5, 0.5),
                data_quality=np.random.uniform(0.7, 1.0)
            )
            
        except Exception as e:
            self.logger.error(f"Error generating synthetic ESG score: {e}")
            raise
    
    async def _load_esg_risk_metrics(self, 
                                   asset_ids: List[str],
                                   data_sources: Dict[str, Any],
                                   **kwargs):
        """加载ESG风险指标"""
        try:
            for asset_id in asset_ids:
                # 计算或加载ESG风险指标
                if asset_id in self.esg_scores:
                    esg_score = self.esg_scores[asset_id]
                    
                    # 基于ESG评分计算风险指标
                    transition_risk = max(0, (10 - esg_score.environmental_score) / 10)
                    physical_risk = max(0, (10 - esg_score.environmental_score) / 10) * 0.8
                    regulatory_risk = max(0, (10 - esg_score.governance_score) / 10)
                    reputation_risk = esg_score.controversy_score / 10
                    
                    overall_esg_risk = (transition_risk + physical_risk + regulatory_risk + reputation_risk) / 4
                    
                    # 气候风险指标
                    climate_var = esg_score.carbon_intensity / 1000 * 0.05  # 简化的气候VaR
                    stranded_asset_risk = transition_risk * 0.1 if esg_score.industry_group == 'Energy' else 0.01
                    
                    risk_metrics = ESGRiskMetrics(
                        asset_id=asset_id,
                        transition_risk=transition_risk,
                        physical_risk=physical_risk,
                        regulatory_risk=regulatory_risk,
                        reputation_risk=reputation_risk,
                        overall_esg_risk=overall_esg_risk,
                        carbon_footprint=esg_score.carbon_intensity,
                        water_footprint=esg_score.water_intensity,
                        waste_footprint=esg_score.waste_intensity,
                        climate_var=climate_var,
                        stranded_asset_risk=stranded_asset_risk,
                        supply_chain_risk=np.random.uniform(0.01, 0.05)
                    )
                    
                    self.esg_risk_metrics[asset_id] = risk_metrics
            
        except Exception as e:
            self.logger.error(f"Error loading ESG risk metrics: {e}")
    
    def _validate_esg_data(self, asset_ids: List[str]) -> bool:
        """验证ESG数据"""
        try:
            missing_assets = []
            for asset_id in asset_ids:
                if asset_id not in self.esg_scores:
                    missing_assets.append(asset_id)
            
            if missing_assets:
                self.logger.warning(f"Missing ESG data for assets: {missing_assets}")
                return False
            
            # 检查数据质量
            low_quality_assets = []
            for asset_id in asset_ids:
                esg_score = self.esg_scores[asset_id]
                if esg_score.data_quality < self.min_data_quality:
                    low_quality_assets.append(asset_id)
            
            if low_quality_assets:
                self.logger.warning(f"Low quality ESG data for assets: {low_quality_assets}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating ESG data: {e}")
            return False
    
    async def _check_esg_constraints(self, 
                                   esg_score: ESGScore,
                                   esg_constraints: List[ESGConstraint]) -> bool:
        """检查ESG约束"""
        try:
            for constraint in esg_constraints:
                if constraint.category == ESGCategory.OVERALL:
                    score = esg_score.overall_score
                elif constraint.category == ESGCategory.ENVIRONMENTAL:
                    score = esg_score.environmental_score
                elif constraint.category == ESGCategory.SOCIAL:
                    score = esg_score.social_score
                elif constraint.category == ESGCategory.GOVERNANCE:
                    score = esg_score.governance_score
                else:
                    continue
                
                # 检查约束条件
                if constraint.operator == 'min' and score < constraint.threshold:
                    return False
                elif constraint.operator == 'max' and score > constraint.threshold:
                    return False
                elif constraint.operator == 'exclude' and esg_score.asset_id in constraint.exclusion_list:
                    return False
                
                # 检查争议分数
                if constraint.constraint_type == 'max_controversy' and esg_score.controversy_score > constraint.threshold:
                    return False
                
                # 检查碳强度
                if constraint.constraint_type == 'max_carbon_intensity' and esg_score.carbon_intensity > constraint.threshold:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking ESG constraints: {e}")
            return False
    
    async def _calculate_esg_adjusted_returns(self, 
                                            asset_ids: List[str],
                                            expected_returns: np.ndarray,
                                            **kwargs) -> np.ndarray:
        """计算ESG调整后的预期收益"""
        try:
            esg_adjustments = []
            
            for asset_id in asset_ids:
                if asset_id in self.esg_scores:
                    esg_score = self.esg_scores[asset_id]
                    
                    # ESG溢价/折价
                    esg_premium = (esg_score.overall_score - 5.0) / 5.0 * 0.01  # 1%的ESG溢价
                    
                    # 争议折价
                    controversy_discount = -esg_score.controversy_score * 0.002  # 争议折价
                    
                    # 碳风险折价
                    carbon_discount = -min(esg_score.carbon_intensity / 1000, 0.05)  # 最高5%的碳风险折价
                    
                    total_adjustment = esg_premium + controversy_discount + carbon_discount
                    esg_adjustments.append(total_adjustment)
                else:
                    esg_adjustments.append(0.0)
            
            return expected_returns + np.array(esg_adjustments)
            
        except Exception as e:
            self.logger.error(f"Error calculating ESG adjusted returns: {e}")
            return expected_returns
    
    async def _calculate_esg_adjusted_covariance(self, 
                                               asset_ids: List[str],
                                               covariance_matrix: np.ndarray,
                                               **kwargs) -> np.ndarray:
        """计算ESG调整后的协方差矩阵"""
        try:
            # 简化实现：基于ESG相似性调整相关性
            esg_similarity_matrix = np.eye(len(asset_ids))
            
            for i, asset_i in enumerate(asset_ids):
                for j, asset_j in enumerate(asset_ids):
                    if i != j and asset_i in self.esg_scores and asset_j in self.esg_scores:
                        esg_i = self.esg_scores[asset_i]
                        esg_j = self.esg_scores[asset_j]
                        
                        # 计算ESG相似性
                        esg_similarity = 1 - abs(esg_i.overall_score - esg_j.overall_score) / 10
                        esg_similarity_matrix[i, j] = esg_similarity
            
            # 调整协方差矩阵
            adjusted_covariance = covariance_matrix * (1 + 0.1 * (esg_similarity_matrix - 1))
            
            return adjusted_covariance
            
        except Exception as e:
            self.logger.error(f"Error calculating ESG adjusted covariance: {e}")
            return covariance_matrix
    
    def _calculate_portfolio_esg_score(self, weights: np.ndarray, asset_ids: List[str]) -> float:
        """计算组合ESG得分"""
        try:
            weighted_score = 0.0
            total_weight = 0.0
            
            for i, asset_id in enumerate(asset_ids):
                if asset_id in self.esg_scores and weights[i] > 0:
                    weighted_score += weights[i] * self.esg_scores[asset_id].overall_score
                    total_weight += weights[i]
            
            return weighted_score / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio ESG score: {e}")
            return 0.0
    
    def _calculate_portfolio_carbon_intensity(self, weights: np.ndarray, asset_ids: List[str]) -> float:
        """计算组合碳强度"""
        try:
            weighted_carbon = 0.0
            total_weight = 0.0
            
            for i, asset_id in enumerate(asset_ids):
                if asset_id in self.esg_scores and weights[i] > 0:
                    weighted_carbon += weights[i] * self.esg_scores[asset_id].carbon_intensity
                    total_weight += weights[i]
            
            return weighted_carbon / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio carbon intensity: {e}")
            return 0.0
    
    async def _calculate_portfolio_esg_metrics(self, 
                                             weights: np.ndarray,
                                             asset_ids: List[str],
                                             portfolio_id: str) -> ESGPortfolioMetrics:
        """计算组合ESG指标"""
        try:
            # 权重标准化
            total_weight = np.sum(weights)
            if total_weight > 0:
                normalized_weights = weights / total_weight
            else:
                normalized_weights = weights
            
            # 计算加权平均ESG指标
            weighted_env = 0.0
            weighted_social = 0.0
            weighted_gov = 0.0
            weighted_overall = 0.0
            weighted_carbon = 0.0
            weighted_water = 0.0
            weighted_waste = 0.0
            weighted_controversy = 0.0
            
            active_weight = 0.0
            
            for i, asset_id in enumerate(asset_ids):
                if asset_id in self.esg_scores and normalized_weights[i] > 0:
                    esg_score = self.esg_scores[asset_id]
                    weight = normalized_weights[i]
                    
                    weighted_env += weight * esg_score.environmental_score
                    weighted_social += weight * esg_score.social_score
                    weighted_gov += weight * esg_score.governance_score
                    weighted_overall += weight * esg_score.overall_score
                    weighted_carbon += weight * esg_score.carbon_intensity
                    weighted_water += weight * esg_score.water_intensity
                    weighted_waste += weight * esg_score.waste_intensity
                    weighted_controversy += weight * esg_score.controversy_score
                    active_weight += weight
            
            # 计算ESG风险得分
            esg_risk_score = 0.0
            for i, asset_id in enumerate(asset_ids):
                if asset_id in self.esg_risk_metrics and normalized_weights[i] > 0:
                    risk_metrics = self.esg_risk_metrics[asset_id]
                    esg_risk_score += normalized_weights[i] * risk_metrics.overall_esg_risk
            
            # 计算多元化得分
            esg_scores = [self.esg_scores[asset_id].overall_score for asset_id in asset_ids if asset_id in self.esg_scores]
            esg_std = np.std(esg_scores) if esg_scores else 0.0
            diversification_score = min(1.0, esg_std / 2.0)  # 标准化到0-1
            
            # 计算对齐得分（与目标ESG水平的对齐程度）
            target_esg_score = 7.0  # 假设目标ESG得分为7
            alignment_score = max(0, 1 - abs(weighted_overall - target_esg_score) / target_esg_score)
            
            return ESGPortfolioMetrics(
                portfolio_id=portfolio_id,
                weighted_avg_esg_score=weighted_overall,
                environmental_score=weighted_env,
                social_score=weighted_social,
                governance_score=weighted_gov,
                esg_risk_score=esg_risk_score,
                carbon_intensity=weighted_carbon,
                water_intensity=weighted_water,
                waste_intensity=weighted_waste,
                esg_momentum=0.0,  # 将在后续计算
                controversy_exposure=weighted_controversy,
                stranded_asset_exposure=0.0,  # 将在后续计算
                alignment_score=alignment_score,
                diversification_score=diversification_score
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio ESG metrics: {e}")
            return ESGPortfolioMetrics(
                portfolio_id=portfolio_id,
                weighted_avg_esg_score=0.0,
                environmental_score=0.0,
                social_score=0.0,
                governance_score=0.0,
                esg_risk_score=0.0,
                carbon_intensity=0.0,
                water_intensity=0.0,
                waste_intensity=0.0,
                esg_momentum=0.0,
                controversy_exposure=0.0,
                stranded_asset_exposure=0.0,
                alignment_score=0.0,
                diversification_score=0.0
            )
    
    async def _calculate_esg_momentum_scores(self, 
                                           asset_ids: List[str],
                                           **kwargs) -> Dict[str, float]:
        """计算ESG动量分数"""
        try:
            momentum_scores = {}
            
            for asset_id in asset_ids:
                if asset_id in self.esg_scores:
                    esg_score = self.esg_scores[asset_id]
                    
                    # 简化的动量计算
                    # 实际应用中需要历史ESG数据
                    base_momentum = (esg_score.overall_score - 5.0) / 5.0
                    
                    # 行业调整
                    industry_adjustment = 0.0
                    if esg_score.industry_group == 'Technology':
                        industry_adjustment = 0.1
                    elif esg_score.industry_group == 'Energy':
                        industry_adjustment = -0.1
                    
                    # 数据质量调整
                    quality_adjustment = (esg_score.data_quality - 0.5) * 0.2
                    
                    momentum_score = base_momentum + industry_adjustment + quality_adjustment
                    momentum_scores[asset_id] = momentum_score
            
            return momentum_scores
            
        except Exception as e:
            self.logger.error(f"Error calculating ESG momentum scores: {e}")
            return {}
    
    async def _calculate_traditional_metrics(self, 
                                           weights: np.ndarray,
                                           expected_returns: np.ndarray,
                                           covariance_matrix: np.ndarray,
                                           **kwargs) -> Dict[str, float]:
        """计算传统金融指标"""
        try:
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            risk_free_rate = kwargs.get('risk_free_rate', 0.02)
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
            
            return {
                'expected_return': portfolio_return,
                'volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio,
                'var_95': portfolio_volatility * norm.ppf(0.95),
                'max_drawdown': portfolio_volatility * 2.0  # 估算
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating traditional metrics: {e}")
            return {}
    
    async def _calculate_esg_improvement(self, 
                                       optimal_weights: np.ndarray,
                                       benchmark_weights: np.ndarray,
                                       asset_ids: List[str]) -> float:
        """计算ESG改善"""
        try:
            portfolio_esg = self._calculate_portfolio_esg_score(optimal_weights, asset_ids)
            benchmark_esg = self._calculate_portfolio_esg_score(benchmark_weights, asset_ids)
            
            return portfolio_esg - benchmark_esg
            
        except Exception as e:
            self.logger.error(f"Error calculating ESG improvement: {e}")
            return 0.0
    
    async def _calculate_esg_cost(self, 
                                optimal_weights: np.ndarray,
                                benchmark_weights: Optional[np.ndarray],
                                expected_returns: np.ndarray,
                                covariance_matrix: np.ndarray) -> float:
        """计算ESG成本"""
        try:
            if benchmark_weights is None:
                return 0.0
            
            # 计算跟踪误差
            active_weights = optimal_weights - benchmark_weights
            tracking_variance = np.dot(active_weights, np.dot(covariance_matrix, active_weights))
            tracking_error = np.sqrt(tracking_variance)
            
            # 计算收益率差异
            return_difference = np.dot(active_weights, expected_returns)
            
            # ESG成本 = 负的超额收益 + 跟踪误差成本
            esg_cost = max(0, -return_difference) + 0.5 * tracking_error
            
            return esg_cost
            
        except Exception as e:
            self.logger.error(f"Error calculating ESG cost: {e}")
            return 0.0
    
    async def _calculate_esg_attribution(self, 
                                       portfolio_result: ESGOptimizationResult,
                                       benchmark_weights: np.ndarray,
                                       market_data: Dict[str, Any]) -> Dict[str, float]:
        """计算ESG绩效归因"""
        try:
            # 简化的ESG归因分析
            attribution = {
                'security_selection': 0.0,
                'sector_allocation': 0.0,
                'esg_tilting': 0.0,
                'interaction': 0.0
            }
            
            # ESG倾斜归因
            active_weights = portfolio_result.optimal_weights - benchmark_weights
            esg_tilting_effect = 0.0
            
            for i, asset_id in enumerate(portfolio_result.asset_ids):
                if asset_id in self.esg_scores:
                    esg_score = self.esg_scores[asset_id]
                    esg_premium = (esg_score.overall_score - 5.0) / 5.0 * 0.01
                    esg_tilting_effect += active_weights[i] * esg_premium
            
            attribution['esg_tilting'] = esg_tilting_effect
            
            return attribution
            
        except Exception as e:
            self.logger.error(f"Error calculating ESG attribution: {e}")
            return {}
    
    async def _calculate_risk_adjusted_esg_performance(self, 
                                                     portfolio_result: ESGOptimizationResult,
                                                     benchmark_weights: np.ndarray,
                                                     market_data: Dict[str, Any]) -> Dict[str, float]:
        """计算风险调整后的ESG表现"""
        try:
            # 计算信息比率
            esg_improvement = portfolio_result.esg_improvement
            tracking_error = portfolio_result.tracking_error
            
            information_ratio = esg_improvement / tracking_error if tracking_error > 0 else 0
            
            # 计算ESG夏普比率
            esg_sharpe = esg_improvement / 0.01  # 假设ESG风险为1%
            
            return {
                'information_ratio': information_ratio,
                'esg_sharpe_ratio': esg_sharpe,
                'risk_adjusted_esg_score': esg_improvement * 100 / max(tracking_error, 0.01)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating risk-adjusted ESG performance: {e}")
            return {}
    
    async def _analyze_esg_factor_exposure(self, 
                                         portfolio_result: ESGOptimizationResult,
                                         market_data: Dict[str, Any]) -> Dict[str, float]:
        """分析ESG因子暴露"""
        try:
            factor_exposure = {
                'carbon_beta': 0.0,
                'esg_momentum_beta': 0.0,
                'controversy_sensitivity': 0.0,
                'governance_quality': 0.0
            }
            
            # 计算碳贝塔
            carbon_intensities = [
                self.esg_scores[asset_id].carbon_intensity 
                for asset_id in portfolio_result.asset_ids 
                if asset_id in self.esg_scores
            ]
            
            if carbon_intensities:
                portfolio_carbon = np.dot(portfolio_result.optimal_weights, carbon_intensities)
                market_carbon = np.mean(carbon_intensities)
                factor_exposure['carbon_beta'] = portfolio_carbon / market_carbon if market_carbon > 0 else 1.0
            
            # 计算治理质量
            governance_scores = [
                self.esg_scores[asset_id].governance_score 
                for asset_id in portfolio_result.asset_ids 
                if asset_id in self.esg_scores
            ]
            
            if governance_scores:
                portfolio_governance = np.dot(portfolio_result.optimal_weights, governance_scores)
                factor_exposure['governance_quality'] = portfolio_governance / 10.0  # 标准化到0-1
            
            return factor_exposure
            
        except Exception as e:
            self.logger.error(f"Error analyzing ESG factor exposure: {e}")
            return {}
    
    async def _analyze_climate_risk(self, 
                                  portfolio_result: ESGOptimizationResult,
                                  market_data: Dict[str, Any]) -> Dict[str, float]:
        """分析气候风险"""
        try:
            climate_risk = {
                'transition_risk': 0.0,
                'physical_risk': 0.0,
                'stranded_asset_risk': 0.0,
                'climate_var': 0.0
            }
            
            # 计算组合气候风险
            for i, asset_id in enumerate(portfolio_result.asset_ids):
                if asset_id in self.esg_risk_metrics:
                    risk_metrics = self.esg_risk_metrics[asset_id]
                    weight = portfolio_result.optimal_weights[i]
                    
                    climate_risk['transition_risk'] += weight * risk_metrics.transition_risk
                    climate_risk['physical_risk'] += weight * risk_metrics.physical_risk
                    climate_risk['stranded_asset_risk'] += weight * risk_metrics.stranded_asset_risk
                    climate_risk['climate_var'] += weight * risk_metrics.climate_var
            
            return climate_risk
            
        except Exception as e:
            self.logger.error(f"Error analyzing climate risk: {e}")
            return {}
    
    def get_esg_summary(self) -> Dict[str, Any]:
        """获取ESG摘要"""
        try:
            if not self.portfolio_history:
                return {}
            
            recent_portfolios = self.portfolio_history[-10:]
            
            summary = {
                'total_optimizations': len(self.portfolio_history),
                'integration_methods_used': list(set(p.optimization_method.value for p in recent_portfolios)),
                'average_esg_score': np.mean([p.portfolio_metrics.weighted_avg_esg_score for p in recent_portfolios]),
                'average_carbon_intensity': np.mean([p.portfolio_metrics.carbon_intensity for p in recent_portfolios]),
                'average_esg_improvement': np.mean([p.esg_improvement for p in recent_portfolios]),
                'average_esg_cost': np.mean([p.cost_of_esg for p in recent_portfolios]),
                'assets_with_esg_data': len(self.esg_scores),
                'esg_data_providers': [self.default_provider.value],
                'last_optimization_date': self.portfolio_history[-1].metadata.get('timestamp', datetime.now())
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting ESG summary: {e}")
            return {}
    
    def plot_esg_analysis(self, 
                         portfolio_result: ESGOptimizationResult,
                         benchmark_weights: Optional[np.ndarray] = None,
                         save_path: Optional[str] = None):
        """绘制ESG分析图"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. ESG得分分布
            esg_scores = [
                self.esg_scores[asset_id].overall_score 
                for asset_id in portfolio_result.asset_ids 
                if asset_id in self.esg_scores
            ]
            
            if esg_scores:
                ax1.hist(esg_scores, bins=20, alpha=0.7, edgecolor='black')
                ax1.axvline(portfolio_result.portfolio_metrics.weighted_avg_esg_score, 
                           color='red', linestyle='--', label='组合加权平均')
                ax1.set_xlabel('ESG得分')
                ax1.set_ylabel('频数')
                ax1.set_title('ESG得分分布')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            
            # 2. ESG维度雷达图
            categories = ['环境', '社会', '治理', '综合']
            values = [
                portfolio_result.portfolio_metrics.environmental_score,
                portfolio_result.portfolio_metrics.social_score,
                portfolio_result.portfolio_metrics.governance_score,
                portfolio_result.portfolio_metrics.weighted_avg_esg_score
            ]
            
            # 简化的雷达图
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
            values_plot = values + [values[0]]  # 闭合图形
            angles_plot = np.concatenate([angles, [angles[0]]])
            
            ax2.plot(angles_plot, values_plot, 'o-', linewidth=2, label='投资组合')
            ax2.fill(angles_plot, values_plot, alpha=0.25)
            ax2.set_xticks(angles)
            ax2.set_xticklabels(categories)
            ax2.set_ylim(0, 10)
            ax2.set_title('ESG维度分析')
            ax2.grid(True)
            
            # 3. 权重vs ESG得分散点图
            weights = portfolio_result.optimal_weights
            esg_scores_full = [
                self.esg_scores[asset_id].overall_score if asset_id in self.esg_scores else 0
                for asset_id in portfolio_result.asset_ids
            ]
            
            # 只显示有权重的资产
            active_indices = [i for i, w in enumerate(weights) if w > 0.001]
            active_weights = [weights[i] for i in active_indices]
            active_esg_scores = [esg_scores_full[i] for i in active_indices]
            
            if active_weights:
                ax3.scatter(active_esg_scores, active_weights, alpha=0.6, s=100)
                ax3.set_xlabel('ESG得分')
                ax3.set_ylabel('权重')
                ax3.set_title('权重 vs ESG得分')
                ax3.grid(True, alpha=0.3)
            
            # 4. 碳强度分析
            carbon_intensities = [
                self.esg_scores[asset_id].carbon_intensity 
                for asset_id in portfolio_result.asset_ids 
                if asset_id in self.esg_scores
            ]
            
            if carbon_intensities:
                ax4.bar(range(len(carbon_intensities)), carbon_intensities, alpha=0.7)
                ax4.axhline(portfolio_result.portfolio_metrics.carbon_intensity, 
                           color='red', linestyle='--', label='组合加权平均')
                ax4.set_xlabel('资产')
                ax4.set_ylabel('碳强度')
                ax4.set_title('碳强度分析')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Error plotting ESG analysis: {e}")
    
    def export_esg_report(self, 
                         portfolio_result: ESGOptimizationResult,
                         file_path: str):
        """导出ESG报告"""
        try:
            report_data = {
                'report_metadata': {
                    'optimization_method': portfolio_result.optimization_method.value,
                    'generation_date': datetime.now().isoformat(),
                    'portfolio_id': portfolio_result.portfolio_metrics.portfolio_id,
                    'esg_provider': self.default_provider.value
                },
                'portfolio_metrics': {
                    'weighted_avg_esg_score': portfolio_result.portfolio_metrics.weighted_avg_esg_score,
                    'environmental_score': portfolio_result.portfolio_metrics.environmental_score,
                    'social_score': portfolio_result.portfolio_metrics.social_score,
                    'governance_score': portfolio_result.portfolio_metrics.governance_score,
                    'carbon_intensity': portfolio_result.portfolio_metrics.carbon_intensity,
                    'water_intensity': portfolio_result.portfolio_metrics.water_intensity,
                    'waste_intensity': portfolio_result.portfolio_metrics.waste_intensity,
                    'controversy_exposure': portfolio_result.portfolio_metrics.controversy_exposure,
                    'alignment_score': portfolio_result.portfolio_metrics.alignment_score,
                    'diversification_score': portfolio_result.portfolio_metrics.diversification_score
                },
                'traditional_metrics': portfolio_result.traditional_metrics,
                'esg_risk_metrics': portfolio_result.esg_risk_metrics,
                'optimization_results': {
                    'optimization_success': portfolio_result.optimization_success,
                    'objective_value': portfolio_result.objective_value,
                    'esg_improvement': portfolio_result.esg_improvement,
                    'cost_of_esg': portfolio_result.cost_of_esg,
                    'tracking_error': portfolio_result.tracking_error
                },
                'asset_details': [
                    {
                        'asset_id': asset_id,
                        'weight': float(portfolio_result.optimal_weights[i]),
                        'esg_score': self.esg_scores[asset_id].overall_score if asset_id in self.esg_scores else None,
                        'carbon_intensity': self.esg_scores[asset_id].carbon_intensity if asset_id in self.esg_scores else None,
                        'controversy_score': self.esg_scores[asset_id].controversy_score if asset_id in self.esg_scores else None
                    }
                    for i, asset_id in enumerate(portfolio_result.asset_ids)
                    if portfolio_result.optimal_weights[i] > 0.001
                ],
                'excluded_assets': portfolio_result.excluded_assets,
                'tilted_weights': portfolio_result.tilted_weights
            }
            
            import json
            with open(file_path, 'w') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"ESG report exported to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error exporting ESG report: {e}")
            raise
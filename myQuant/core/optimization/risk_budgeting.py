import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor

class RiskBudgetingMethod(Enum):
    EQUAL_RISK_CONTRIBUTION = "equal_risk_contribution"
    RISK_PARITY = "risk_parity"
    EQUAL_WEIGHTED = "equal_weighted"
    MAXIMUM_DIVERSIFICATION = "maximum_diversification"
    MINIMUM_VARIANCE = "minimum_variance"
    INVERSE_VOLATILITY = "inverse_volatility"
    HIERARCHICAL_RISK_PARITY = "hierarchical_risk_parity"
    CONDITIONAL_RISK_PARITY = "conditional_risk_parity"

class RiskMeasure(Enum):
    VOLATILITY = "volatility"
    VALUE_AT_RISK = "value_at_risk"
    CONDITIONAL_VALUE_AT_RISK = "conditional_value_at_risk"
    EXPECTED_SHORTFALL = "expected_shortfall"
    MAXIMUM_DRAWDOWN = "maximum_drawdown"
    DOWNSIDE_DEVIATION = "downside_deviation"

@dataclass
class RiskContribution:
    """风险贡献度"""
    asset: str
    weight: float
    marginal_risk: float
    component_risk: float
    risk_contribution: float
    risk_contribution_pct: float
    diversification_ratio: float

@dataclass
class RiskBudget:
    """风险预算"""
    asset: str
    target_risk_contribution: float
    actual_risk_contribution: float
    deviation: float
    constraint_type: str = "equality"  # equality, upper_bound, lower_bound
    priority: int = 1

@dataclass
class RiskBudgetingResult:
    """风险预算结果"""
    method: RiskBudgetingMethod
    optimal_weights: np.ndarray
    risk_contributions: List[RiskContribution]
    portfolio_risk: float
    portfolio_return: float
    diversification_ratio: float
    effective_number_of_assets: float
    risk_concentration: float
    optimization_success: bool
    iterations: int
    objective_value: float
    constraints_satisfied: bool
    risk_budgets: List[RiskBudget]
    rebalancing_frequency: int
    transaction_costs: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class RiskBudgetingModel:
    """
    风险预算模型
    
    风险预算模型旨在根据预定义的风险贡献度分配资产权重，
    而不是基于预期收益率。主要方法包括等风险贡献、风险平价等。
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 模型参数
        self.risk_measure = RiskMeasure(config.get('risk_measure', 'volatility'))
        self.max_iterations = config.get('max_iterations', 1000)
        self.tolerance = config.get('tolerance', 1e-8)
        self.regularization = config.get('regularization', 1e-8)
        
        # 约束参数
        self.min_weight = config.get('min_weight', 0.0)
        self.max_weight = config.get('max_weight', 1.0)
        self.weight_sum = config.get('weight_sum', 1.0)
        
        # 优化参数
        self.optimization_method = config.get('optimization_method', 'SLSQP')
        self.use_analytical_gradient = config.get('use_analytical_gradient', True)
        
        # 执行器
        self.executor = ThreadPoolExecutor(max_workers=config.get('max_workers', 4))
        
        # 历史结果
        self.optimization_history: List[RiskBudgetingResult] = []
        
        # 缓存
        self.cache = {}
        
        self.logger.info(f"Risk budgeting model initialized with {self.risk_measure.value} risk measure")
    
    async def optimize_portfolio(self, 
                               assets: List[str],
                               returns: np.ndarray,
                               covariance_matrix: np.ndarray,
                               method: RiskBudgetingMethod = RiskBudgetingMethod.EQUAL_RISK_CONTRIBUTION,
                               risk_budgets: Optional[List[RiskBudget]] = None,
                               constraints: Optional[Dict[str, Any]] = None,
                               **kwargs) -> RiskBudgetingResult:
        """
        风险预算优化
        """
        try:
            start_time = datetime.now()
            
            self.logger.info(f"Starting risk budgeting optimization using {method.value}")
            
            # 验证输入
            if not self._validate_inputs(assets, returns, covariance_matrix):
                raise ValueError("Invalid inputs for risk budgeting optimization")
            
            # 选择优化方法
            if method == RiskBudgetingMethod.EQUAL_RISK_CONTRIBUTION:
                result = await self._optimize_equal_risk_contribution(
                    assets, returns, covariance_matrix, constraints, **kwargs
                )
            elif method == RiskBudgetingMethod.RISK_PARITY:
                result = await self._optimize_risk_parity(
                    assets, returns, covariance_matrix, risk_budgets, constraints, **kwargs
                )
            elif method == RiskBudgetingMethod.EQUAL_WEIGHTED:
                result = await self._optimize_equal_weighted(
                    assets, returns, covariance_matrix, constraints, **kwargs
                )
            elif method == RiskBudgetingMethod.MAXIMUM_DIVERSIFICATION:
                result = await self._optimize_maximum_diversification(
                    assets, returns, covariance_matrix, constraints, **kwargs
                )
            elif method == RiskBudgetingMethod.MINIMUM_VARIANCE:
                result = await self._optimize_minimum_variance(
                    assets, returns, covariance_matrix, constraints, **kwargs
                )
            elif method == RiskBudgetingMethod.INVERSE_VOLATILITY:
                result = await self._optimize_inverse_volatility(
                    assets, returns, covariance_matrix, constraints, **kwargs
                )
            elif method == RiskBudgetingMethod.HIERARCHICAL_RISK_PARITY:
                result = await self._optimize_hierarchical_risk_parity(
                    assets, returns, covariance_matrix, constraints, **kwargs
                )
            elif method == RiskBudgetingMethod.CONDITIONAL_RISK_PARITY:
                result = await self._optimize_conditional_risk_parity(
                    assets, returns, covariance_matrix, constraints, **kwargs
                )
            else:
                raise ValueError(f"Unsupported risk budgeting method: {method}")
            
            # 记录优化时间
            optimization_time = (datetime.now() - start_time).total_seconds()
            result.metadata['optimization_time'] = optimization_time
            
            # 计算额外指标
            await self._calculate_additional_metrics(result, returns, covariance_matrix)
            
            # 保存结果
            self.optimization_history.append(result)
            
            # 保持历史记录数量
            if len(self.optimization_history) > 100:
                self.optimization_history = self.optimization_history[-100:]
            
            self.logger.info(f"Risk budgeting optimization completed in {optimization_time:.2f} seconds")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in risk budgeting optimization: {e}")
            raise
    
    async def _optimize_equal_risk_contribution(self, 
                                              assets: List[str],
                                              returns: np.ndarray,
                                              covariance_matrix: np.ndarray,
                                              constraints: Optional[Dict[str, Any]] = None,
                                              **kwargs) -> RiskBudgetingResult:
        """
        等风险贡献优化 (Equal Risk Contribution)
        """
        try:
            n_assets = len(assets)
            
            def objective_function(weights):
                # 计算风险贡献度
                risk_contributions = self._calculate_risk_contributions(weights, covariance_matrix)
                
                # 目标：最小化风险贡献度的方差
                target_contribution = 1.0 / n_assets
                deviations = risk_contributions - target_contribution
                
                return np.sum(deviations ** 2)
            
            def jacobian(weights):
                # 解析梯度
                return self._calculate_erc_gradient(weights, covariance_matrix)
            
            # 构建约束
            constraints_list = self._build_constraints(constraints, n_assets)
            
            # 权重界限
            bounds = self._build_bounds(constraints, n_assets)
            
            # 初始权重
            initial_weights = np.ones(n_assets) / n_assets
            
            # 优化
            if self.use_analytical_gradient:
                result = minimize(
                    objective_function,
                    initial_weights,
                    method=self.optimization_method,
                    jac=jacobian,
                    bounds=bounds,
                    constraints=constraints_list,
                    options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
                )
            else:
                result = minimize(
                    objective_function,
                    initial_weights,
                    method=self.optimization_method,
                    bounds=bounds,
                    constraints=constraints_list,
                    options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
                )
            
            # 计算风险贡献度
            risk_contributions = self._calculate_detailed_risk_contributions(
                assets, result.x, covariance_matrix
            )
            
            # 计算风险预算
            risk_budgets = [
                RiskBudget(
                    asset=asset,
                    target_risk_contribution=1.0 / n_assets,
                    actual_risk_contribution=rc.risk_contribution_pct,
                    deviation=rc.risk_contribution_pct - 1.0 / n_assets
                ) for asset, rc in zip(assets, risk_contributions)
            ]
            
            # 计算组合风险
            portfolio_risk = np.sqrt(np.dot(result.x, np.dot(covariance_matrix, result.x)))
            
            # 计算组合收益
            portfolio_return = np.dot(result.x, returns)
            
            return RiskBudgetingResult(
                method=RiskBudgetingMethod.EQUAL_RISK_CONTRIBUTION,
                optimal_weights=result.x,
                risk_contributions=risk_contributions,
                portfolio_risk=portfolio_risk,
                portfolio_return=portfolio_return,
                diversification_ratio=self._calculate_diversification_ratio(result.x, covariance_matrix),
                effective_number_of_assets=1.0 / np.sum(result.x ** 2),
                risk_concentration=self._calculate_risk_concentration(risk_contributions),
                optimization_success=result.success,
                iterations=result.nit if hasattr(result, 'nit') else 0,
                objective_value=result.fun,
                constraints_satisfied=result.success,
                risk_budgets=risk_budgets,
                rebalancing_frequency=kwargs.get('rebalancing_frequency', 252),
                transaction_costs=kwargs.get('transaction_costs', 0.0)
            )
            
        except Exception as e:
            self.logger.error(f"Error in equal risk contribution optimization: {e}")
            raise
    
    async def _optimize_risk_parity(self, 
                                  assets: List[str],
                                  returns: np.ndarray,
                                  covariance_matrix: np.ndarray,
                                  risk_budgets: Optional[List[RiskBudget]] = None,
                                  constraints: Optional[Dict[str, Any]] = None,
                                  **kwargs) -> RiskBudgetingResult:
        """
        风险平价优化
        """
        try:
            n_assets = len(assets)
            
            # 目标风险贡献度
            if risk_budgets is None:
                target_contributions = np.ones(n_assets) / n_assets
            else:
                target_contributions = np.array([rb.target_risk_contribution for rb in risk_budgets])
                target_contributions = target_contributions / np.sum(target_contributions)
            
            def objective_function(weights):
                # 计算风险贡献度
                risk_contributions = self._calculate_risk_contributions(weights, covariance_matrix)
                
                # 目标：最小化与目标风险贡献度的偏差
                deviations = risk_contributions - target_contributions
                
                return np.sum(deviations ** 2)
            
            # 构建约束
            constraints_list = self._build_constraints(constraints, n_assets)
            
            # 权重界限
            bounds = self._build_bounds(constraints, n_assets)
            
            # 初始权重
            initial_weights = target_contributions.copy()
            
            # 优化
            result = minimize(
                objective_function,
                initial_weights,
                method=self.optimization_method,
                bounds=bounds,
                constraints=constraints_list,
                options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
            )
            
            # 计算风险贡献度
            risk_contributions = self._calculate_detailed_risk_contributions(
                assets, result.x, covariance_matrix
            )
            
            # 计算风险预算
            risk_budgets_result = [
                RiskBudget(
                    asset=asset,
                    target_risk_contribution=target_contributions[i],
                    actual_risk_contribution=rc.risk_contribution_pct,
                    deviation=rc.risk_contribution_pct - target_contributions[i]
                ) for i, (asset, rc) in enumerate(zip(assets, risk_contributions))
            ]
            
            # 计算组合风险
            portfolio_risk = np.sqrt(np.dot(result.x, np.dot(covariance_matrix, result.x)))
            
            # 计算组合收益
            portfolio_return = np.dot(result.x, returns)
            
            return RiskBudgetingResult(
                method=RiskBudgetingMethod.RISK_PARITY,
                optimal_weights=result.x,
                risk_contributions=risk_contributions,
                portfolio_risk=portfolio_risk,
                portfolio_return=portfolio_return,
                diversification_ratio=self._calculate_diversification_ratio(result.x, covariance_matrix),
                effective_number_of_assets=1.0 / np.sum(result.x ** 2),
                risk_concentration=self._calculate_risk_concentration(risk_contributions),
                optimization_success=result.success,
                iterations=result.nit if hasattr(result, 'nit') else 0,
                objective_value=result.fun,
                constraints_satisfied=result.success,
                risk_budgets=risk_budgets_result,
                rebalancing_frequency=kwargs.get('rebalancing_frequency', 252),
                transaction_costs=kwargs.get('transaction_costs', 0.0)
            )
            
        except Exception as e:
            self.logger.error(f"Error in risk parity optimization: {e}")
            raise
    
    async def _optimize_equal_weighted(self, 
                                     assets: List[str],
                                     returns: np.ndarray,
                                     covariance_matrix: np.ndarray,
                                     constraints: Optional[Dict[str, Any]] = None,
                                     **kwargs) -> RiskBudgetingResult:
        """
        等权重优化
        """
        try:
            n_assets = len(assets)
            
            # 等权重
            weights = np.ones(n_assets) / n_assets
            
            # 计算风险贡献度
            risk_contributions = self._calculate_detailed_risk_contributions(
                assets, weights, covariance_matrix
            )
            
            # 计算风险预算
            risk_budgets = [
                RiskBudget(
                    asset=asset,
                    target_risk_contribution=1.0 / n_assets,
                    actual_risk_contribution=rc.risk_contribution_pct,
                    deviation=rc.risk_contribution_pct - 1.0 / n_assets
                ) for asset, rc in zip(assets, risk_contributions)
            ]
            
            # 计算组合风险
            portfolio_risk = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
            
            # 计算组合收益
            portfolio_return = np.dot(weights, returns)
            
            return RiskBudgetingResult(
                method=RiskBudgetingMethod.EQUAL_WEIGHTED,
                optimal_weights=weights,
                risk_contributions=risk_contributions,
                portfolio_risk=portfolio_risk,
                portfolio_return=portfolio_return,
                diversification_ratio=self._calculate_diversification_ratio(weights, covariance_matrix),
                effective_number_of_assets=1.0 / np.sum(weights ** 2),
                risk_concentration=self._calculate_risk_concentration(risk_contributions),
                optimization_success=True,
                iterations=0,
                objective_value=0.0,
                constraints_satisfied=True,
                risk_budgets=risk_budgets,
                rebalancing_frequency=kwargs.get('rebalancing_frequency', 252),
                transaction_costs=kwargs.get('transaction_costs', 0.0)
            )
            
        except Exception as e:
            self.logger.error(f"Error in equal weighted optimization: {e}")
            raise
    
    async def _optimize_maximum_diversification(self, 
                                              assets: List[str],
                                              returns: np.ndarray,
                                              covariance_matrix: np.ndarray,
                                              constraints: Optional[Dict[str, Any]] = None,
                                              **kwargs) -> RiskBudgetingResult:
        """
        最大多元化优化
        """
        try:
            n_assets = len(assets)
            
            # 计算个股波动率
            individual_volatilities = np.sqrt(np.diag(covariance_matrix))
            
            def objective_function(weights):
                # 多元化比率 = 加权平均波动率 / 组合波动率
                weighted_avg_volatility = np.dot(weights, individual_volatilities)
                portfolio_volatility = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
                
                # 最大化多元化比率 = 最小化负多元化比率
                if portfolio_volatility > 0:
                    diversification_ratio = weighted_avg_volatility / portfolio_volatility
                    return -diversification_ratio
                else:
                    return 1e6
            
            # 构建约束
            constraints_list = self._build_constraints(constraints, n_assets)
            
            # 权重界限
            bounds = self._build_bounds(constraints, n_assets)
            
            # 初始权重
            initial_weights = np.ones(n_assets) / n_assets
            
            # 优化
            result = minimize(
                objective_function,
                initial_weights,
                method=self.optimization_method,
                bounds=bounds,
                constraints=constraints_list,
                options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
            )
            
            # 计算风险贡献度
            risk_contributions = self._calculate_detailed_risk_contributions(
                assets, result.x, covariance_matrix
            )
            
            # 计算风险预算
            risk_budgets = [
                RiskBudget(
                    asset=asset,
                    target_risk_contribution=0.0,  # 最大多元化没有明确的目标风险贡献
                    actual_risk_contribution=rc.risk_contribution_pct,
                    deviation=0.0
                ) for asset, rc in zip(assets, risk_contributions)
            ]
            
            # 计算组合风险
            portfolio_risk = np.sqrt(np.dot(result.x, np.dot(covariance_matrix, result.x)))
            
            # 计算组合收益
            portfolio_return = np.dot(result.x, returns)
            
            return RiskBudgetingResult(
                method=RiskBudgetingMethod.MAXIMUM_DIVERSIFICATION,
                optimal_weights=result.x,
                risk_contributions=risk_contributions,
                portfolio_risk=portfolio_risk,
                portfolio_return=portfolio_return,
                diversification_ratio=self._calculate_diversification_ratio(result.x, covariance_matrix),
                effective_number_of_assets=1.0 / np.sum(result.x ** 2),
                risk_concentration=self._calculate_risk_concentration(risk_contributions),
                optimization_success=result.success,
                iterations=result.nit if hasattr(result, 'nit') else 0,
                objective_value=result.fun,
                constraints_satisfied=result.success,
                risk_budgets=risk_budgets,
                rebalancing_frequency=kwargs.get('rebalancing_frequency', 252),
                transaction_costs=kwargs.get('transaction_costs', 0.0)
            )
            
        except Exception as e:
            self.logger.error(f"Error in maximum diversification optimization: {e}")
            raise
    
    async def _optimize_minimum_variance(self, 
                                       assets: List[str],
                                       returns: np.ndarray,
                                       covariance_matrix: np.ndarray,
                                       constraints: Optional[Dict[str, Any]] = None,
                                       **kwargs) -> RiskBudgetingResult:
        """
        最小方差优化
        """
        try:
            n_assets = len(assets)
            
            def objective_function(weights):
                # 最小化组合方差
                return np.dot(weights, np.dot(covariance_matrix, weights))
            
            # 构建约束
            constraints_list = self._build_constraints(constraints, n_assets)
            
            # 权重界限
            bounds = self._build_bounds(constraints, n_assets)
            
            # 初始权重
            initial_weights = np.ones(n_assets) / n_assets
            
            # 优化
            result = minimize(
                objective_function,
                initial_weights,
                method=self.optimization_method,
                bounds=bounds,
                constraints=constraints_list,
                options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
            )
            
            # 计算风险贡献度
            risk_contributions = self._calculate_detailed_risk_contributions(
                assets, result.x, covariance_matrix
            )
            
            # 计算风险预算
            risk_budgets = [
                RiskBudget(
                    asset=asset,
                    target_risk_contribution=0.0,  # 最小方差没有明确的目标风险贡献
                    actual_risk_contribution=rc.risk_contribution_pct,
                    deviation=0.0
                ) for asset, rc in zip(assets, risk_contributions)
            ]
            
            # 计算组合风险
            portfolio_risk = np.sqrt(result.fun)
            
            # 计算组合收益
            portfolio_return = np.dot(result.x, returns)
            
            return RiskBudgetingResult(
                method=RiskBudgetingMethod.MINIMUM_VARIANCE,
                optimal_weights=result.x,
                risk_contributions=risk_contributions,
                portfolio_risk=portfolio_risk,
                portfolio_return=portfolio_return,
                diversification_ratio=self._calculate_diversification_ratio(result.x, covariance_matrix),
                effective_number_of_assets=1.0 / np.sum(result.x ** 2),
                risk_concentration=self._calculate_risk_concentration(risk_contributions),
                optimization_success=result.success,
                iterations=result.nit if hasattr(result, 'nit') else 0,
                objective_value=result.fun,
                constraints_satisfied=result.success,
                risk_budgets=risk_budgets,
                rebalancing_frequency=kwargs.get('rebalancing_frequency', 252),
                transaction_costs=kwargs.get('transaction_costs', 0.0)
            )
            
        except Exception as e:
            self.logger.error(f"Error in minimum variance optimization: {e}")
            raise
    
    async def _optimize_inverse_volatility(self, 
                                         assets: List[str],
                                         returns: np.ndarray,
                                         covariance_matrix: np.ndarray,
                                         constraints: Optional[Dict[str, Any]] = None,
                                         **kwargs) -> RiskBudgetingResult:
        """
        逆波动率优化
        """
        try:
            n_assets = len(assets)
            
            # 计算个股波动率
            individual_volatilities = np.sqrt(np.diag(covariance_matrix))
            
            # 逆波动率权重
            inverse_volatilities = 1.0 / individual_volatilities
            weights = inverse_volatilities / np.sum(inverse_volatilities)
            
            # 计算风险贡献度
            risk_contributions = self._calculate_detailed_risk_contributions(
                assets, weights, covariance_matrix
            )
            
            # 计算风险预算
            risk_budgets = [
                RiskBudget(
                    asset=asset,
                    target_risk_contribution=1.0 / individual_volatilities[i] / np.sum(1.0 / individual_volatilities),
                    actual_risk_contribution=rc.risk_contribution_pct,
                    deviation=rc.risk_contribution_pct - (1.0 / individual_volatilities[i] / np.sum(1.0 / individual_volatilities))
                ) for i, (asset, rc) in enumerate(zip(assets, risk_contributions))
            ]
            
            # 计算组合风险
            portfolio_risk = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
            
            # 计算组合收益
            portfolio_return = np.dot(weights, returns)
            
            return RiskBudgetingResult(
                method=RiskBudgetingMethod.INVERSE_VOLATILITY,
                optimal_weights=weights,
                risk_contributions=risk_contributions,
                portfolio_risk=portfolio_risk,
                portfolio_return=portfolio_return,
                diversification_ratio=self._calculate_diversification_ratio(weights, covariance_matrix),
                effective_number_of_assets=1.0 / np.sum(weights ** 2),
                risk_concentration=self._calculate_risk_concentration(risk_contributions),
                optimization_success=True,
                iterations=0,
                objective_value=0.0,
                constraints_satisfied=True,
                risk_budgets=risk_budgets,
                rebalancing_frequency=kwargs.get('rebalancing_frequency', 252),
                transaction_costs=kwargs.get('transaction_costs', 0.0)
            )
            
        except Exception as e:
            self.logger.error(f"Error in inverse volatility optimization: {e}")
            raise
    
    async def _optimize_hierarchical_risk_parity(self, 
                                               assets: List[str],
                                               returns: np.ndarray,
                                               covariance_matrix: np.ndarray,
                                               constraints: Optional[Dict[str, Any]] = None,
                                               **kwargs) -> RiskBudgetingResult:
        """
        层次风险平价优化 (Hierarchical Risk Parity)
        """
        try:
            from scipy.cluster.hierarchy import linkage, dendrogram
            from scipy.spatial.distance import pdist
            
            n_assets = len(assets)
            
            # 计算相关性矩阵
            correlation_matrix = self._covariance_to_correlation(covariance_matrix)
            
            # 计算距离矩阵
            distance_matrix = np.sqrt(0.5 * (1 - correlation_matrix))
            
            # 层次聚类
            condensed_distances = pdist(distance_matrix, metric='euclidean')
            linkage_matrix = linkage(condensed_distances, method='single')
            
            # 递归二分法分配权重
            weights = self._recursive_bisection(linkage_matrix, covariance_matrix, n_assets)
            
            # 计算风险贡献度
            risk_contributions = self._calculate_detailed_risk_contributions(
                assets, weights, covariance_matrix
            )
            
            # 计算风险预算
            risk_budgets = [
                RiskBudget(
                    asset=asset,
                    target_risk_contribution=1.0 / n_assets,
                    actual_risk_contribution=rc.risk_contribution_pct,
                    deviation=rc.risk_contribution_pct - 1.0 / n_assets
                ) for asset, rc in zip(assets, risk_contributions)
            ]
            
            # 计算组合风险
            portfolio_risk = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
            
            # 计算组合收益
            portfolio_return = np.dot(weights, returns)
            
            return RiskBudgetingResult(
                method=RiskBudgetingMethod.HIERARCHICAL_RISK_PARITY,
                optimal_weights=weights,
                risk_contributions=risk_contributions,
                portfolio_risk=portfolio_risk,
                portfolio_return=portfolio_return,
                diversification_ratio=self._calculate_diversification_ratio(weights, covariance_matrix),
                effective_number_of_assets=1.0 / np.sum(weights ** 2),
                risk_concentration=self._calculate_risk_concentration(risk_contributions),
                optimization_success=True,
                iterations=0,
                objective_value=0.0,
                constraints_satisfied=True,
                risk_budgets=risk_budgets,
                rebalancing_frequency=kwargs.get('rebalancing_frequency', 252),
                transaction_costs=kwargs.get('transaction_costs', 0.0)
            )
            
        except Exception as e:
            self.logger.error(f"Error in hierarchical risk parity optimization: {e}")
            # 降级到等风险贡献
            return await self._optimize_equal_risk_contribution(assets, returns, covariance_matrix, constraints, **kwargs)
    
    async def _optimize_conditional_risk_parity(self, 
                                              assets: List[str],
                                              returns: np.ndarray,
                                              covariance_matrix: np.ndarray,
                                              constraints: Optional[Dict[str, Any]] = None,
                                              **kwargs) -> RiskBudgetingResult:
        """
        条件风险平价优化
        """
        try:
            n_assets = len(assets)
            
            # 获取市场条件
            market_regime = kwargs.get('market_regime', 'normal')
            
            # 根据市场条件调整协方差矩阵
            if market_regime == 'stress':
                # 压力情况下增加相关性
                adjusted_covariance = self._adjust_covariance_for_stress(covariance_matrix)
            elif market_regime == 'low_vol':
                # 低波动情况下减少波动率
                adjusted_covariance = covariance_matrix * 0.5
            else:
                adjusted_covariance = covariance_matrix
            
            # 使用调整后的协方差矩阵进行等风险贡献优化
            result = await self._optimize_equal_risk_contribution(
                assets, returns, adjusted_covariance, constraints, **kwargs
            )
            
            # 更新方法名称
            result.method = RiskBudgetingMethod.CONDITIONAL_RISK_PARITY
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in conditional risk parity optimization: {e}")
            raise
    
    def _calculate_risk_contributions(self, weights: np.ndarray, covariance_matrix: np.ndarray) -> np.ndarray:
        """计算风险贡献度"""
        try:
            portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
            
            if portfolio_variance == 0:
                return np.zeros(len(weights))
            
            # 边际风险贡献
            marginal_contributions = np.dot(covariance_matrix, weights)
            
            # 风险贡献度
            risk_contributions = weights * marginal_contributions / portfolio_variance
            
            return risk_contributions
            
        except Exception as e:
            self.logger.error(f"Error calculating risk contributions: {e}")
            return np.zeros(len(weights))
    
    def _calculate_detailed_risk_contributions(self, 
                                             assets: List[str],
                                             weights: np.ndarray,
                                             covariance_matrix: np.ndarray) -> List[RiskContribution]:
        """计算详细的风险贡献度"""
        try:
            portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
            portfolio_risk = np.sqrt(portfolio_variance)
            
            risk_contributions = []
            
            for i, asset in enumerate(assets):
                # 边际风险贡献
                marginal_risk = np.dot(covariance_matrix[i, :], weights) / portfolio_risk
                
                # 组件风险贡献
                component_risk = weights[i] * marginal_risk
                
                # 风险贡献度
                risk_contribution = component_risk / portfolio_risk if portfolio_risk > 0 else 0
                
                # 风险贡献百分比
                risk_contribution_pct = risk_contribution * 100
                
                # 多元化比率
                individual_volatility = np.sqrt(covariance_matrix[i, i])
                diversification_ratio = individual_volatility / portfolio_risk if portfolio_risk > 0 else 0
                
                risk_contributions.append(RiskContribution(
                    asset=asset,
                    weight=weights[i],
                    marginal_risk=marginal_risk,
                    component_risk=component_risk,
                    risk_contribution=risk_contribution,
                    risk_contribution_pct=risk_contribution_pct,
                    diversification_ratio=diversification_ratio
                ))
            
            return risk_contributions
            
        except Exception as e:
            self.logger.error(f"Error calculating detailed risk contributions: {e}")
            return []
    
    def _calculate_erc_gradient(self, weights: np.ndarray, covariance_matrix: np.ndarray) -> np.ndarray:
        """计算等风险贡献的解析梯度"""
        try:
            n_assets = len(weights)
            portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
            
            if portfolio_variance == 0:
                return np.zeros(n_assets)
            
            # 风险贡献度
            risk_contributions = self._calculate_risk_contributions(weights, covariance_matrix)
            
            # 目标风险贡献度
            target_contribution = 1.0 / n_assets
            
            # 梯度计算
            gradient = np.zeros(n_assets)
            
            for i in range(n_assets):
                deviation = risk_contributions[i] - target_contribution
                
                # 计算偏导数
                partial_derivative = self._calculate_partial_derivative(i, weights, covariance_matrix, portfolio_variance)
                
                gradient[i] = 2 * deviation * partial_derivative
            
            return gradient
            
        except Exception as e:
            self.logger.error(f"Error calculating ERC gradient: {e}")
            return np.zeros(len(weights))
    
    def _calculate_partial_derivative(self, 
                                    asset_index: int,
                                    weights: np.ndarray,
                                    covariance_matrix: np.ndarray,
                                    portfolio_variance: float) -> float:
        """计算风险贡献度的偏导数"""
        try:
            n_assets = len(weights)
            
            # 边际风险贡献
            marginal_risk = np.dot(covariance_matrix[asset_index, :], weights)
            
            # 二阶偏导数
            second_derivative = covariance_matrix[asset_index, asset_index]
            
            # 计算偏导数
            numerator = marginal_risk * portfolio_variance + weights[asset_index] * (second_derivative * portfolio_variance - marginal_risk ** 2)
            denominator = portfolio_variance ** 2
            
            return numerator / denominator if denominator > 0 else 0
            
        except Exception as e:
            self.logger.error(f"Error calculating partial derivative: {e}")
            return 0.0
    
    def _calculate_diversification_ratio(self, weights: np.ndarray, covariance_matrix: np.ndarray) -> float:
        """计算多元化比率"""
        try:
            # 加权平均波动率
            individual_volatilities = np.sqrt(np.diag(covariance_matrix))
            weighted_avg_volatility = np.dot(weights, individual_volatilities)
            
            # 组合波动率
            portfolio_volatility = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
            
            return weighted_avg_volatility / portfolio_volatility if portfolio_volatility > 0 else 0
            
        except Exception as e:
            self.logger.error(f"Error calculating diversification ratio: {e}")
            return 0.0
    
    def _calculate_risk_concentration(self, risk_contributions: List[RiskContribution]) -> float:
        """计算风险集中度"""
        try:
            # 使用Herfindahl指数
            risk_contribution_pcts = [rc.risk_contribution_pct / 100 for rc in risk_contributions]
            herfindahl_index = sum(x ** 2 for x in risk_contribution_pcts)
            
            return herfindahl_index
            
        except Exception as e:
            self.logger.error(f"Error calculating risk concentration: {e}")
            return 0.0
    
    def _build_constraints(self, constraints: Optional[Dict[str, Any]], n_assets: int) -> List[Dict]:
        """构建优化约束"""
        constraints_list = []
        
        # 权重和约束
        constraints_list.append({
            'type': 'eq',
            'fun': lambda x: np.sum(x) - self.weight_sum
        })
        
        # 自定义约束
        if constraints:
            # 行业限制
            if 'sector_limits' in constraints:
                sector_limits = constraints['sector_limits']
                for sector, limit in sector_limits.items():
                    sector_assets = constraints.get(f'{sector}_assets', [])
                    if sector_assets:
                        sector_indices = [i for i, asset in enumerate(sector_assets) if i < n_assets]
                        if sector_indices:
                            constraints_list.append({
                                'type': 'ineq',
                                'fun': lambda x, indices=sector_indices: limit - np.sum(x[indices])
                            })
            
            # 集中度限制
            if 'max_weight' in constraints:
                max_weight = constraints['max_weight']
                constraints_list.append({
                    'type': 'ineq',
                    'fun': lambda x: max_weight - np.max(x)
                })
        
        return constraints_list
    
    def _build_bounds(self, constraints: Optional[Dict[str, Any]], n_assets: int) -> List[Tuple]:
        """构建权重界限"""
        if constraints and 'weight_bounds' in constraints:
            return constraints['weight_bounds']
        else:
            return [(self.min_weight, self.max_weight) for _ in range(n_assets)]
    
    def _recursive_bisection(self, linkage_matrix: np.ndarray, covariance_matrix: np.ndarray, n_assets: int) -> np.ndarray:
        """递归二分法权重分配"""
        try:
            # 简化的HRP实现
            weights = np.ones(n_assets) / n_assets
            
            # 这里应该实现完整的HRP算法
            # 由于复杂性，这里提供简化版本
            
            return weights
            
        except Exception as e:
            self.logger.error(f"Error in recursive bisection: {e}")
            return np.ones(n_assets) / n_assets
    
    def _covariance_to_correlation(self, covariance_matrix: np.ndarray) -> np.ndarray:
        """协方差矩阵转换为相关性矩阵"""
        try:
            std_devs = np.sqrt(np.diag(covariance_matrix))
            correlation_matrix = covariance_matrix / np.outer(std_devs, std_devs)
            
            return correlation_matrix
            
        except Exception as e:
            self.logger.error(f"Error converting covariance to correlation: {e}")
            return np.eye(covariance_matrix.shape[0])
    
    def _adjust_covariance_for_stress(self, covariance_matrix: np.ndarray) -> np.ndarray:
        """为压力情况调整协方差矩阵"""
        try:
            # 增加所有相关性
            correlation_matrix = self._covariance_to_correlation(covariance_matrix)
            
            # 增加相关性（但不超过0.95）
            adjusted_correlation = np.minimum(correlation_matrix * 1.5, 0.95)
            np.fill_diagonal(adjusted_correlation, 1.0)
            
            # 转换回协方差矩阵
            std_devs = np.sqrt(np.diag(covariance_matrix))
            adjusted_covariance = adjusted_correlation * np.outer(std_devs, std_devs)
            
            return adjusted_covariance
            
        except Exception as e:
            self.logger.error(f"Error adjusting covariance for stress: {e}")
            return covariance_matrix
    
    async def _calculate_additional_metrics(self, 
                                          result: RiskBudgetingResult,
                                          returns: np.ndarray,
                                          covariance_matrix: np.ndarray):
        """计算额外指标"""
        try:
            # 计算交易成本
            if hasattr(self, 'previous_weights'):
                turnover = np.sum(np.abs(result.optimal_weights - self.previous_weights))
                result.transaction_costs = turnover * result.transaction_costs
            
            # 保存当前权重
            self.previous_weights = result.optimal_weights.copy()
            
            # 计算其他指标
            result.metadata.update({
                'individual_volatilities': np.sqrt(np.diag(covariance_matrix)).tolist(),
                'portfolio_volatility': result.portfolio_risk,
                'weight_entropy': -np.sum(result.optimal_weights * np.log(result.optimal_weights + 1e-10)),
                'maximum_weight': np.max(result.optimal_weights),
                'minimum_weight': np.min(result.optimal_weights),
                'weight_range': np.max(result.optimal_weights) - np.min(result.optimal_weights)
            })
            
        except Exception as e:
            self.logger.error(f"Error calculating additional metrics: {e}")
    
    def _validate_inputs(self, 
                        assets: List[str],
                        returns: np.ndarray,
                        covariance_matrix: np.ndarray) -> bool:
        """验证输入参数"""
        try:
            if len(assets) != len(returns):
                return False
            
            if len(assets) != covariance_matrix.shape[0]:
                return False
            
            if covariance_matrix.shape[0] != covariance_matrix.shape[1]:
                return False
            
            # 检查协方差矩阵是否正定
            eigenvalues = np.linalg.eigvals(covariance_matrix)
            if np.any(eigenvalues <= 0):
                self.logger.warning("Covariance matrix is not positive definite")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating inputs: {e}")
            return False
    
    def plot_risk_contributions(self, 
                              result: RiskBudgetingResult,
                              save_path: Optional[str] = None):
        """绘制风险贡献度"""
        try:
            assets = [rc.asset for rc in result.risk_contributions]
            weights = [rc.weight for rc in result.risk_contributions]
            risk_contributions = [rc.risk_contribution_pct for rc in result.risk_contributions]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 权重分布
            ax1.bar(assets, weights, alpha=0.7, color='skyblue')
            ax1.set_title('Portfolio Weights')
            ax1.set_ylabel('Weight')
            ax1.set_xlabel('Assets')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
            
            # 风险贡献度
            ax2.bar(assets, risk_contributions, alpha=0.7, color='lightcoral')
            ax2.set_title('Risk Contributions')
            ax2.set_ylabel('Risk Contribution (%)')
            ax2.set_xlabel('Assets')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
            
            # 添加平均线
            if result.method == RiskBudgetingMethod.EQUAL_RISK_CONTRIBUTION:
                avg_risk_contribution = 100 / len(assets)
                ax2.axhline(y=avg_risk_contribution, color='red', linestyle='--', 
                           label=f'Target ({avg_risk_contribution:.1f}%)')
                ax2.legend()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Error plotting risk contributions: {e}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """获取模型摘要"""
        try:
            summary = {
                'model_configuration': {
                    'risk_measure': self.risk_measure.value,
                    'max_iterations': self.max_iterations,
                    'tolerance': self.tolerance,
                    'optimization_method': self.optimization_method
                },
                'optimization_history': {
                    'total_optimizations': len(self.optimization_history),
                    'methods_used': list(set(r.method.value for r in self.optimization_history)),
                    'success_rate': np.mean([r.optimization_success for r in self.optimization_history]) if self.optimization_history else 0,
                    'avg_diversification_ratio': np.mean([r.diversification_ratio for r in self.optimization_history]) if self.optimization_history else 0
                }
            }
            
            if self.optimization_history:
                latest = self.optimization_history[-1]
                summary['latest_result'] = {
                    'method': latest.method.value,
                    'portfolio_risk': latest.portfolio_risk,
                    'portfolio_return': latest.portfolio_return,
                    'diversification_ratio': latest.diversification_ratio,
                    'effective_number_of_assets': latest.effective_number_of_assets,
                    'risk_concentration': latest.risk_concentration
                }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting model summary: {e}")
            return {}
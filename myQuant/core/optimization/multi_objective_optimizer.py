import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod

class OptimizationMethod(Enum):
    PARETO_FRONTIER = "pareto_frontier"
    WEIGHTED_SUM = "weighted_sum"
    EPSILON_CONSTRAINT = "epsilon_constraint"
    GOAL_PROGRAMMING = "goal_programming"
    LEXICOGRAPHIC = "lexicographic"
    NSGA_II = "nsga_ii"

class OptimizationObjective(Enum):
    MAXIMIZE_RETURN = "maximize_return"
    MINIMIZE_RISK = "minimize_risk"
    MAXIMIZE_SHARPE = "maximize_sharpe"
    MINIMIZE_DRAWDOWN = "minimize_drawdown"
    MAXIMIZE_DIVERSIFICATION = "maximize_diversification"
    MINIMIZE_TRACKING_ERROR = "minimize_tracking_error"
    MAXIMIZE_ESG_SCORE = "maximize_esg_score"
    MINIMIZE_TURNOVER = "minimize_turnover"
    MAXIMIZE_ALPHA = "maximize_alpha"
    MINIMIZE_CORRELATION = "minimize_correlation"

class ConstraintType(Enum):
    WEIGHT_SUM = "weight_sum"
    WEIGHT_BOUNDS = "weight_bounds"
    SECTOR_LIMITS = "sector_limits"
    TURNOVER_LIMIT = "turnover_limit"
    CONCENTRATION_LIMIT = "concentration_limit"
    TRACKING_ERROR_LIMIT = "tracking_error_limit"
    ESG_MINIMUM = "esg_minimum"
    LIQUIDITY_MINIMUM = "liquidity_minimum"
    BENCHMARK_DEVIATION = "benchmark_deviation"

@dataclass
class OptimizationConstraint:
    constraint_type: ConstraintType
    name: str
    bounds: Optional[Tuple[float, float]] = None
    value: Optional[float] = None
    assets: Optional[List[str]] = None
    sectors: Optional[Dict[str, float]] = None
    reference_weights: Optional[np.ndarray] = None
    tolerance: float = 1e-6

@dataclass
class ObjectiveFunction:
    objective: OptimizationObjective
    weight: float = 1.0
    target_value: Optional[float] = None
    priority: int = 1
    scaler: float = 1.0

@dataclass
class OptimizationResult:
    method: OptimizationMethod
    optimal_weights: np.ndarray
    objective_values: Dict[str, float]
    constraints_satisfied: bool
    optimization_time: float
    convergence_achieved: bool
    frontier_points: Optional[List[Tuple[float, float]]] = None
    pareto_optimal: bool = False
    dominated_solutions: Optional[List[np.ndarray]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseObjectiveFunction(ABC):
    @abstractmethod
    def calculate(self, weights: np.ndarray, returns: np.ndarray, 
                  covariance: np.ndarray, **kwargs) -> float:
        pass

class ReturnObjective(BaseObjectiveFunction):
    def calculate(self, weights: np.ndarray, returns: np.ndarray, 
                  covariance: np.ndarray, **kwargs) -> float:
        return np.dot(weights, returns)

class RiskObjective(BaseObjectiveFunction):
    def calculate(self, weights: np.ndarray, returns: np.ndarray, 
                  covariance: np.ndarray, **kwargs) -> float:
        return np.sqrt(np.dot(weights, np.dot(covariance, weights)))

class SharpeObjective(BaseObjectiveFunction):
    def calculate(self, weights: np.ndarray, returns: np.ndarray, 
                  covariance: np.ndarray, **kwargs) -> float:
        portfolio_return = np.dot(weights, returns)
        portfolio_risk = np.sqrt(np.dot(weights, np.dot(covariance, weights)))
        risk_free_rate = kwargs.get('risk_free_rate', 0.0)
        
        if portfolio_risk == 0:
            return 0.0
        return (portfolio_return - risk_free_rate) / portfolio_risk

class DiversificationObjective(BaseObjectiveFunction):
    def calculate(self, weights: np.ndarray, returns: np.ndarray, 
                  covariance: np.ndarray, **kwargs) -> float:
        # 使用逆Herfindahl指数衡量多样化
        return 1.0 / np.sum(weights ** 2)

class TrackingErrorObjective(BaseObjectiveFunction):
    def calculate(self, weights: np.ndarray, returns: np.ndarray, 
                  covariance: np.ndarray, **kwargs) -> float:
        benchmark_weights = kwargs.get('benchmark_weights', np.ones(len(weights)) / len(weights))
        active_weights = weights - benchmark_weights
        return np.sqrt(np.dot(active_weights, np.dot(covariance, active_weights)))

class ESGObjective(BaseObjectiveFunction):
    def calculate(self, weights: np.ndarray, returns: np.ndarray, 
                  covariance: np.ndarray, **kwargs) -> float:
        esg_scores = kwargs.get('esg_scores', np.ones(len(weights)))
        return np.dot(weights, esg_scores)

class MultiObjectiveOptimizer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 优化器配置
        self.max_iterations = config.get('max_iterations', 1000)
        self.tolerance = config.get('tolerance', 1e-6)
        self.population_size = config.get('population_size', 50)
        self.crossover_rate = config.get('crossover_rate', 0.8)
        self.mutation_rate = config.get('mutation_rate', 0.1)
        
        # 目标函数映射
        self.objective_functions = {
            OptimizationObjective.MAXIMIZE_RETURN: ReturnObjective(),
            OptimizationObjective.MINIMIZE_RISK: RiskObjective(),
            OptimizationObjective.MAXIMIZE_SHARPE: SharpeObjective(),
            OptimizationObjective.MAXIMIZE_DIVERSIFICATION: DiversificationObjective(),
            OptimizationObjective.MINIMIZE_TRACKING_ERROR: TrackingErrorObjective(),
            OptimizationObjective.MAXIMIZE_ESG_SCORE: ESGObjective()
        }
        
        # 执行器
        self.executor = ThreadPoolExecutor(max_workers=config.get('max_workers', 4))
        
        # 缓存
        self.optimization_cache = {}
        
        # 历史结果
        self.optimization_history = []
        
    async def optimize(self, 
                      assets: List[str],
                      returns: np.ndarray,
                      covariance: np.ndarray,
                      objectives: List[ObjectiveFunction],
                      constraints: List[OptimizationConstraint],
                      method: OptimizationMethod = OptimizationMethod.PARETO_FRONTIER,
                      **kwargs) -> OptimizationResult:
        """
        多目标优化主函数
        """
        try:
            start_time = datetime.now()
            
            self.logger.info(f"Starting multi-objective optimization with {len(objectives)} objectives")
            
            # 验证输入
            if not self._validate_inputs(assets, returns, covariance, objectives, constraints):
                raise ValueError("Invalid optimization inputs")
            
            # 选择优化方法
            if method == OptimizationMethod.PARETO_FRONTIER:
                result = await self._optimize_pareto_frontier(
                    assets, returns, covariance, objectives, constraints, **kwargs
                )
            elif method == OptimizationMethod.WEIGHTED_SUM:
                result = await self._optimize_weighted_sum(
                    assets, returns, covariance, objectives, constraints, **kwargs
                )
            elif method == OptimizationMethod.EPSILON_CONSTRAINT:
                result = await self._optimize_epsilon_constraint(
                    assets, returns, covariance, objectives, constraints, **kwargs
                )
            elif method == OptimizationMethod.GOAL_PROGRAMMING:
                result = await self._optimize_goal_programming(
                    assets, returns, covariance, objectives, constraints, **kwargs
                )
            elif method == OptimizationMethod.LEXICOGRAPHIC:
                result = await self._optimize_lexicographic(
                    assets, returns, covariance, objectives, constraints, **kwargs
                )
            elif method == OptimizationMethod.NSGA_II:
                result = await self._optimize_nsga_ii(
                    assets, returns, covariance, objectives, constraints, **kwargs
                )
            else:
                raise ValueError(f"Unsupported optimization method: {method}")
            
            # 记录执行时间
            result.optimization_time = (datetime.now() - start_time).total_seconds()
            
            # 验证约束
            result.constraints_satisfied = self._verify_constraints(
                result.optimal_weights, constraints, assets, **kwargs
            )
            
            # 保存结果
            self.optimization_history.append(result)
            
            self.logger.info(f"Multi-objective optimization completed in {result.optimization_time:.2f} seconds")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in multi-objective optimization: {e}")
            raise
    
    async def _optimize_pareto_frontier(self, 
                                      assets: List[str],
                                      returns: np.ndarray,
                                      covariance: np.ndarray,
                                      objectives: List[ObjectiveFunction],
                                      constraints: List[OptimizationConstraint],
                                      **kwargs) -> OptimizationResult:
        """
        帕累托前沿优化
        """
        try:
            if len(objectives) != 2:
                raise ValueError("Pareto frontier optimization requires exactly 2 objectives")
            
            # 生成帕累托前沿点
            frontier_points = []
            pareto_weights = []
            
            # 在目标空间中生成权重组合
            num_points = kwargs.get('num_frontier_points', 50)
            
            for i in range(num_points + 1):
                # 线性组合权重
                w1 = i / num_points
                w2 = 1 - w1
                
                # 创建加权目标函数
                weighted_objectives = [
                    ObjectiveFunction(objectives[0].objective, w1, objectives[0].target_value),
                    ObjectiveFunction(objectives[1].objective, w2, objectives[1].target_value)
                ]
                
                # 优化加权目标
                result = await self._optimize_weighted_sum(
                    assets, returns, covariance, weighted_objectives, constraints, **kwargs
                )
                
                if result.constraints_satisfied:
                    # 计算目标值
                    obj1_value = self._calculate_objective_value(
                        result.optimal_weights, objectives[0], returns, covariance, **kwargs
                    )
                    obj2_value = self._calculate_objective_value(
                        result.optimal_weights, objectives[1], returns, covariance, **kwargs
                    )
                    
                    frontier_points.append((obj1_value, obj2_value))
                    pareto_weights.append(result.optimal_weights)
            
            # 找到帕累托最优解
            if frontier_points:
                # 选择中间点作为代表性解
                mid_index = len(frontier_points) // 2
                optimal_weights = pareto_weights[mid_index]
                
                # 计算所有目标值
                objective_values = {}
                for obj in objectives:
                    obj_value = self._calculate_objective_value(
                        optimal_weights, obj, returns, covariance, **kwargs
                    )
                    objective_values[obj.objective.value] = obj_value
                
                return OptimizationResult(
                    method=OptimizationMethod.PARETO_FRONTIER,
                    optimal_weights=optimal_weights,
                    objective_values=objective_values,
                    constraints_satisfied=True,
                    optimization_time=0.0,
                    convergence_achieved=True,
                    frontier_points=frontier_points,
                    pareto_optimal=True,
                    dominated_solutions=pareto_weights[:mid_index] + pareto_weights[mid_index+1:]
                )
            else:
                raise ValueError("No feasible solutions found for Pareto frontier")
                
        except Exception as e:
            self.logger.error(f"Error in Pareto frontier optimization: {e}")
            raise
    
    async def _optimize_weighted_sum(self, 
                                   assets: List[str],
                                   returns: np.ndarray,
                                   covariance: np.ndarray,
                                   objectives: List[ObjectiveFunction],
                                   constraints: List[OptimizationConstraint],
                                   **kwargs) -> OptimizationResult:
        """
        加权和优化
        """
        try:
            def objective_function(weights):
                total_objective = 0.0
                
                for obj in objectives:
                    obj_value = self._calculate_objective_value(
                        weights, obj, returns, covariance, **kwargs
                    )
                    
                    # 最小化问题，需要对最大化目标取负
                    if obj.objective in [OptimizationObjective.MAXIMIZE_RETURN, 
                                       OptimizationObjective.MAXIMIZE_SHARPE,
                                       OptimizationObjective.MAXIMIZE_DIVERSIFICATION,
                                       OptimizationObjective.MAXIMIZE_ESG_SCORE,
                                       OptimizationObjective.MAXIMIZE_ALPHA]:
                        obj_value = -obj_value
                    
                    total_objective += obj.weight * obj_value
                
                return total_objective
            
            # 构建约束
            scipy_constraints = self._build_scipy_constraints(constraints, assets, **kwargs)
            
            # 权重界限
            bounds = [(0.0, 1.0) for _ in range(len(assets))]
            
            # 初始权重
            initial_weights = np.ones(len(assets)) / len(assets)
            
            # 优化
            result = minimize(
                objective_function,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=scipy_constraints,
                options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
            )
            
            # 计算所有目标值
            objective_values = {}
            for obj in objectives:
                obj_value = self._calculate_objective_value(
                    result.x, obj, returns, covariance, **kwargs
                )
                objective_values[obj.objective.value] = obj_value
            
            return OptimizationResult(
                method=OptimizationMethod.WEIGHTED_SUM,
                optimal_weights=result.x,
                objective_values=objective_values,
                constraints_satisfied=result.success,
                optimization_time=0.0,
                convergence_achieved=result.success
            )
            
        except Exception as e:
            self.logger.error(f"Error in weighted sum optimization: {e}")
            raise
    
    async def _optimize_epsilon_constraint(self, 
                                         assets: List[str],
                                         returns: np.ndarray,
                                         covariance: np.ndarray,
                                         objectives: List[ObjectiveFunction],
                                         constraints: List[OptimizationConstraint],
                                         **kwargs) -> OptimizationResult:
        """
        ε约束优化
        """
        try:
            if len(objectives) < 2:
                raise ValueError("Epsilon constraint requires at least 2 objectives")
            
            # 选择主要目标（第一个）
            primary_objective = objectives[0]
            secondary_objectives = objectives[1:]
            
            def objective_function(weights):
                obj_value = self._calculate_objective_value(
                    weights, primary_objective, returns, covariance, **kwargs
                )
                
                # 最小化问题
                if primary_objective.objective in [OptimizationObjective.MAXIMIZE_RETURN, 
                                                 OptimizationObjective.MAXIMIZE_SHARPE,
                                                 OptimizationObjective.MAXIMIZE_DIVERSIFICATION,
                                                 OptimizationObjective.MAXIMIZE_ESG_SCORE,
                                                 OptimizationObjective.MAXIMIZE_ALPHA]:
                    obj_value = -obj_value
                
                return obj_value
            
            # 构建约束（包括ε约束）
            scipy_constraints = self._build_scipy_constraints(constraints, assets, **kwargs)
            
            # 添加ε约束
            for obj in secondary_objectives:
                if obj.target_value is not None:
                    def epsilon_constraint(weights, target_obj=obj):
                        obj_value = self._calculate_objective_value(
                            weights, target_obj, returns, covariance, **kwargs
                        )
                        
                        if target_obj.objective in [OptimizationObjective.MINIMIZE_RISK,
                                                   OptimizationObjective.MINIMIZE_DRAWDOWN,
                                                   OptimizationObjective.MINIMIZE_TRACKING_ERROR,
                                                   OptimizationObjective.MINIMIZE_TURNOVER]:
                            return target_obj.target_value - obj_value  # obj_value <= target
                        else:
                            return obj_value - target_obj.target_value  # obj_value >= target
                    
                    scipy_constraints.append({
                        'type': 'ineq',
                        'fun': epsilon_constraint
                    })
            
            # 权重界限
            bounds = [(0.0, 1.0) for _ in range(len(assets))]
            
            # 初始权重
            initial_weights = np.ones(len(assets)) / len(assets)
            
            # 优化
            result = minimize(
                objective_function,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=scipy_constraints,
                options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
            )
            
            # 计算所有目标值
            objective_values = {}
            for obj in objectives:
                obj_value = self._calculate_objective_value(
                    result.x, obj, returns, covariance, **kwargs
                )
                objective_values[obj.objective.value] = obj_value
            
            return OptimizationResult(
                method=OptimizationMethod.EPSILON_CONSTRAINT,
                optimal_weights=result.x,
                objective_values=objective_values,
                constraints_satisfied=result.success,
                optimization_time=0.0,
                convergence_achieved=result.success
            )
            
        except Exception as e:
            self.logger.error(f"Error in epsilon constraint optimization: {e}")
            raise
    
    async def _optimize_goal_programming(self, 
                                       assets: List[str],
                                       returns: np.ndarray,
                                       covariance: np.ndarray,
                                       objectives: List[ObjectiveFunction],
                                       constraints: List[OptimizationConstraint],
                                       **kwargs) -> OptimizationResult:
        """
        目标规划优化
        """
        try:
            def objective_function(weights):
                total_deviation = 0.0
                
                for obj in objectives:
                    obj_value = self._calculate_objective_value(
                        weights, obj, returns, covariance, **kwargs
                    )
                    
                    if obj.target_value is not None:
                        # 计算与目标值的偏差
                        deviation = abs(obj_value - obj.target_value)
                        total_deviation += obj.weight * deviation
                
                return total_deviation
            
            # 构建约束
            scipy_constraints = self._build_scipy_constraints(constraints, assets, **kwargs)
            
            # 权重界限
            bounds = [(0.0, 1.0) for _ in range(len(assets))]
            
            # 初始权重
            initial_weights = np.ones(len(assets)) / len(assets)
            
            # 优化
            result = minimize(
                objective_function,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=scipy_constraints,
                options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
            )
            
            # 计算所有目标值
            objective_values = {}
            for obj in objectives:
                obj_value = self._calculate_objective_value(
                    result.x, obj, returns, covariance, **kwargs
                )
                objective_values[obj.objective.value] = obj_value
            
            return OptimizationResult(
                method=OptimizationMethod.GOAL_PROGRAMMING,
                optimal_weights=result.x,
                objective_values=objective_values,
                constraints_satisfied=result.success,
                optimization_time=0.0,
                convergence_achieved=result.success
            )
            
        except Exception as e:
            self.logger.error(f"Error in goal programming optimization: {e}")
            raise
    
    async def _optimize_lexicographic(self, 
                                    assets: List[str],
                                    returns: np.ndarray,
                                    covariance: np.ndarray,
                                    objectives: List[ObjectiveFunction],
                                    constraints: List[OptimizationConstraint],
                                    **kwargs) -> OptimizationResult:
        """
        词典序优化
        """
        try:
            # 按优先级排序目标
            sorted_objectives = sorted(objectives, key=lambda x: x.priority)
            
            current_constraints = constraints.copy()
            final_weights = None
            
            for i, obj in enumerate(sorted_objectives):
                # 优化当前目标
                def objective_function(weights):
                    obj_value = self._calculate_objective_value(
                        weights, obj, returns, covariance, **kwargs
                    )
                    
                    # 最小化问题
                    if obj.objective in [OptimizationObjective.MAXIMIZE_RETURN, 
                                       OptimizationObjective.MAXIMIZE_SHARPE,
                                       OptimizationObjective.MAXIMIZE_DIVERSIFICATION,
                                       OptimizationObjective.MAXIMIZE_ESG_SCORE,
                                       OptimizationObjective.MAXIMIZE_ALPHA]:
                        obj_value = -obj_value
                    
                    return obj_value
                
                # 构建约束
                scipy_constraints = self._build_scipy_constraints(current_constraints, assets, **kwargs)
                
                # 权重界限
                bounds = [(0.0, 1.0) for _ in range(len(assets))]
                
                # 初始权重
                initial_weights = np.ones(len(assets)) / len(assets)
                
                # 优化
                result = minimize(
                    objective_function,
                    initial_weights,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=scipy_constraints,
                    options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
                )
                
                if not result.success:
                    break
                
                final_weights = result.x
                
                # 如果不是最后一个目标，添加约束固定当前目标值
                if i < len(sorted_objectives) - 1:
                    optimal_value = self._calculate_objective_value(
                        final_weights, obj, returns, covariance, **kwargs
                    )
                    
                    # 添加约束保持当前目标值
                    current_constraints.append(OptimizationConstraint(
                        constraint_type=ConstraintType.BENCHMARK_DEVIATION,
                        name=f"fixed_{obj.objective.value}",
                        value=optimal_value,
                        tolerance=self.tolerance
                    ))
            
            # 计算所有目标值
            objective_values = {}
            for obj in objectives:
                obj_value = self._calculate_objective_value(
                    final_weights, obj, returns, covariance, **kwargs
                )
                objective_values[obj.objective.value] = obj_value
            
            return OptimizationResult(
                method=OptimizationMethod.LEXICOGRAPHIC,
                optimal_weights=final_weights,
                objective_values=objective_values,
                constraints_satisfied=True,
                optimization_time=0.0,
                convergence_achieved=True
            )
            
        except Exception as e:
            self.logger.error(f"Error in lexicographic optimization: {e}")
            raise
    
    async def _optimize_nsga_ii(self, 
                              assets: List[str],
                              returns: np.ndarray,
                              covariance: np.ndarray,
                              objectives: List[ObjectiveFunction],
                              constraints: List[OptimizationConstraint],
                              **kwargs) -> OptimizationResult:
        """
        NSGA-II多目标遗传算法
        """
        try:
            def multi_objective_function(weights):
                objective_values = []
                
                for obj in objectives:
                    obj_value = self._calculate_objective_value(
                        weights, obj, returns, covariance, **kwargs
                    )
                    
                    # 最小化问题
                    if obj.objective in [OptimizationObjective.MAXIMIZE_RETURN, 
                                       OptimizationObjective.MAXIMIZE_SHARPE,
                                       OptimizationObjective.MAXIMIZE_DIVERSIFICATION,
                                       OptimizationObjective.MAXIMIZE_ESG_SCORE,
                                       OptimizationObjective.MAXIMIZE_ALPHA]:
                        obj_value = -obj_value
                    
                    objective_values.append(obj_value)
                
                return objective_values
            
            # 约束函数
            def constraint_function(weights):
                violations = []
                
                for constraint in constraints:
                    violation = self._evaluate_constraint(constraint, weights, assets, **kwargs)
                    if violation > 0:
                        violations.append(violation)
                
                return sum(violations) if violations else 0
            
            # 权重界限
            bounds = [(0.0, 1.0) for _ in range(len(assets))]
            
            # 使用差分进化作为NSGA-II的简化版本
            def combined_objective(weights):
                # 加权组合多目标
                obj_values = multi_objective_function(weights)
                constraint_penalty = constraint_function(weights)
                
                # 组合目标函数
                combined_obj = sum(obj.weight * obj_val for obj, obj_val in zip(objectives, obj_values))
                
                # 添加约束惩罚
                combined_obj += 1000 * constraint_penalty
                
                return combined_obj
            
            # 优化
            result = differential_evolution(
                combined_objective,
                bounds,
                maxiter=self.max_iterations,
                popsize=self.population_size,
                tol=self.tolerance,
                seed=42
            )
            
            # 计算所有目标值
            objective_values = {}
            for obj in objectives:
                obj_value = self._calculate_objective_value(
                    result.x, obj, returns, covariance, **kwargs
                )
                objective_values[obj.objective.value] = obj_value
            
            return OptimizationResult(
                method=OptimizationMethod.NSGA_II,
                optimal_weights=result.x,
                objective_values=objective_values,
                constraints_satisfied=result.success,
                optimization_time=0.0,
                convergence_achieved=result.success
            )
            
        except Exception as e:
            self.logger.error(f"Error in NSGA-II optimization: {e}")
            raise
    
    def _calculate_objective_value(self, 
                                  weights: np.ndarray,
                                  objective: ObjectiveFunction,
                                  returns: np.ndarray,
                                  covariance: np.ndarray,
                                  **kwargs) -> float:
        """
        计算目标函数值
        """
        try:
            obj_func = self.objective_functions.get(objective.objective)
            if obj_func is None:
                raise ValueError(f"Unknown objective: {objective.objective}")
            
            return obj_func.calculate(weights, returns, covariance, **kwargs) * objective.scaler
            
        except Exception as e:
            self.logger.error(f"Error calculating objective value: {e}")
            return 0.0
    
    def _build_scipy_constraints(self, 
                                constraints: List[OptimizationConstraint],
                                assets: List[str],
                                **kwargs) -> List[Dict]:
        """
        构建scipy约束
        """
        scipy_constraints = []
        
        for constraint in constraints:
            if constraint.constraint_type == ConstraintType.WEIGHT_SUM:
                # 权重和约束
                scipy_constraints.append({
                    'type': 'eq',
                    'fun': lambda x: np.sum(x) - 1.0
                })
            
            elif constraint.constraint_type == ConstraintType.SECTOR_LIMITS:
                # 行业限制约束
                if constraint.sectors:
                    for sector, limit in constraint.sectors.items():
                        sector_assets = kwargs.get(f'{sector}_assets', [])
                        if sector_assets:
                            sector_indices = [assets.index(asset) for asset in sector_assets if asset in assets]
                            if sector_indices:
                                scipy_constraints.append({
                                    'type': 'ineq',
                                    'fun': lambda x, indices=sector_indices: limit - np.sum(x[indices])
                                })
            
            elif constraint.constraint_type == ConstraintType.CONCENTRATION_LIMIT:
                # 集中度限制
                if constraint.value:
                    scipy_constraints.append({
                        'type': 'ineq',
                        'fun': lambda x: constraint.value - np.max(x)
                    })
            
            elif constraint.constraint_type == ConstraintType.TURNOVER_LIMIT:
                # 换手率限制
                if constraint.value and constraint.reference_weights is not None:
                    scipy_constraints.append({
                        'type': 'ineq',
                        'fun': lambda x: constraint.value - np.sum(np.abs(x - constraint.reference_weights))
                    })
        
        return scipy_constraints
    
    def _evaluate_constraint(self, 
                            constraint: OptimizationConstraint,
                            weights: np.ndarray,
                            assets: List[str],
                            **kwargs) -> float:
        """
        评估约束违规程度
        """
        try:
            if constraint.constraint_type == ConstraintType.WEIGHT_SUM:
                return abs(np.sum(weights) - 1.0)
            
            elif constraint.constraint_type == ConstraintType.CONCENTRATION_LIMIT:
                if constraint.value:
                    max_weight = np.max(weights)
                    return max(0, max_weight - constraint.value)
            
            elif constraint.constraint_type == ConstraintType.TURNOVER_LIMIT:
                if constraint.value and constraint.reference_weights is not None:
                    turnover = np.sum(np.abs(weights - constraint.reference_weights))
                    return max(0, turnover - constraint.value)
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error evaluating constraint: {e}")
            return 0.0
    
    def _verify_constraints(self, 
                           weights: np.ndarray,
                           constraints: List[OptimizationConstraint],
                           assets: List[str],
                           **kwargs) -> bool:
        """
        验证约束是否满足
        """
        try:
            for constraint in constraints:
                violation = self._evaluate_constraint(constraint, weights, assets, **kwargs)
                if violation > constraint.tolerance:
                    return False
            return True
            
        except Exception as e:
            self.logger.error(f"Error verifying constraints: {e}")
            return False
    
    def _validate_inputs(self, 
                        assets: List[str],
                        returns: np.ndarray,
                        covariance: np.ndarray,
                        objectives: List[ObjectiveFunction],
                        constraints: List[OptimizationConstraint]) -> bool:
        """
        验证输入参数
        """
        try:
            # 检查资产数量
            if len(assets) != len(returns) or len(assets) != covariance.shape[0]:
                return False
            
            # 检查协方差矩阵
            if covariance.shape[0] != covariance.shape[1]:
                return False
            
            # 检查目标函数
            if not objectives:
                return False
            
            # 检查权重和
            weight_sum_constraint = any(
                c.constraint_type == ConstraintType.WEIGHT_SUM for c in constraints
            )
            if not weight_sum_constraint:
                # 自动添加权重和约束
                constraints.append(OptimizationConstraint(
                    constraint_type=ConstraintType.WEIGHT_SUM,
                    name="weight_sum"
                ))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating inputs: {e}")
            return False
    
    def plot_pareto_frontier(self, result: OptimizationResult, 
                           objective_names: List[str],
                           save_path: Optional[str] = None):
        """
        绘制帕累托前沿
        """
        try:
            if result.frontier_points is None or len(objective_names) != 2:
                return
            
            plt.figure(figsize=(10, 8))
            
            # 提取帕累托前沿点
            x_values = [point[0] for point in result.frontier_points]
            y_values = [point[1] for point in result.frontier_points]
            
            # 绘制帕累托前沿
            plt.plot(x_values, y_values, 'b-', linewidth=2, label='Pareto Frontier')
            plt.scatter(x_values, y_values, c='red', s=50, alpha=0.6)
            
            # 标记最优解
            optimal_obj1 = result.objective_values.get(objective_names[0], 0)
            optimal_obj2 = result.objective_values.get(objective_names[1], 0)
            plt.scatter([optimal_obj1], [optimal_obj2], c='green', s=100, 
                       marker='*', label='Optimal Solution')
            
            plt.xlabel(objective_names[0])
            plt.ylabel(objective_names[1])
            plt.title('Pareto Frontier')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Error plotting Pareto frontier: {e}")
    
    def analyze_trade_offs(self, result: OptimizationResult) -> Dict[str, Any]:
        """
        分析目标间的权衡关系
        """
        try:
            analysis = {
                'optimization_method': result.method.value,
                'constraints_satisfied': result.constraints_satisfied,
                'convergence_achieved': result.convergence_achieved,
                'objective_values': result.objective_values,
                'portfolio_weights': result.optimal_weights.tolist(),
                'diversification_ratio': 1.0 / np.sum(result.optimal_weights ** 2),
                'max_weight': np.max(result.optimal_weights),
                'min_weight': np.min(result.optimal_weights),
                'non_zero_weights': np.sum(result.optimal_weights > 1e-6),
                'pareto_optimal': result.pareto_optimal
            }
            
            # 权衡分析
            if len(result.objective_values) >= 2:
                objectives = list(result.objective_values.keys())
                analysis['trade_offs'] = {}
                
                for i, obj1 in enumerate(objectives):
                    for j, obj2 in enumerate(objectives[i+1:], i+1):
                        ratio = result.objective_values[obj1] / result.objective_values[obj2]
                        analysis['trade_offs'][f'{obj1}_vs_{obj2}'] = ratio
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing trade-offs: {e}")
            return {}
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """
        获取优化摘要
        """
        try:
            if not self.optimization_history:
                return {}
            
            return {
                'total_optimizations': len(self.optimization_history),
                'methods_used': list(set(r.method.value for r in self.optimization_history)),
                'avg_optimization_time': np.mean([r.optimization_time for r in self.optimization_history]),
                'success_rate': np.mean([r.constraints_satisfied for r in self.optimization_history]),
                'convergence_rate': np.mean([r.convergence_achieved for r in self.optimization_history]),
                'latest_result': self.optimization_history[-1] if self.optimization_history else None
            }
            
        except Exception as e:
            self.logger.error(f"Error getting optimization summary: {e}")
            return {}
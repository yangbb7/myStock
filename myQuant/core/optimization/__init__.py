# -*- coding: utf-8 -*-
"""
投资组合优化模块 - 提供多目标优化、Black-Litterman模型、风险预算和动态对冲功能
"""

# 多目标优化
from .multi_objective_optimizer import (
    MultiObjectiveOptimizer,
    OptimizationMethod,
    OptimizationObjective,
    ConstraintType,
    OptimizationConstraint,
    ObjectiveFunction,
    OptimizationResult,
    BaseObjectiveFunction,
    ReturnObjective,
    RiskObjective,
    SharpeObjective,
    DiversificationObjective,
    TrackingErrorObjective,
    ESGObjective
)

# Black-Litterman模型
from .black_litterman import (
    BlackLittermanModel,
    BlackLittermanResult,
    ViewType,
    ConfidenceLevel,
    MarketParameters,
    ViewParameters
)

# 风险预算模型
from .risk_budgeting import (
    RiskBudgetingModel,
    RiskBudgetingResult,
    RiskBudgetingMethod,
    RiskMeasure,
    RiskContribution,
    RiskBudget
)

# 动态对冲策略
from .dynamic_hedging import (
    DynamicHedgingEngine,
    HedgingStrategy,
    HedgingFrequency,
    HedgingInstrument,
    HedgeRatio,
    HedgingPosition,
    HedgingResult
)

# 参数优化器
from .parameter_optimizer import ParameterOptimizer

# 优化算法
from .optimization_algorithms import (
    BaseOptimizer,
    GridSearchOptimizer,
    RandomSearchOptimizer,
    BayesianOptimizer,
    DifferentialEvolutionOptimizer,
    HyperOptimizer,
    OptimizationResult
)

# 目标函数
from .objective_functions import (
    BaseObjectiveFunction,
    PerformanceMetrics,
    SharpeRatioObjective,
    SortinoRatioObjective,
    CalmarRatioObjective,
    MaxDrawdownObjective,
    ProfitFactorObjective,
    TotalReturnObjective,
    WinRateObjective,
    CompositeObjective,
    RiskAdjustedObjective,
    MultiObjective,
    ConstrainedObjective,
    CommonConstraints
)

# 参数空间
from .parameter_space import (
    Parameter,
    ParameterType,
    ParameterSpace,
    ParameterSpaceBuilder,
    CommonParameterSpaces
)

__all__ = [
    # 多目标优化
    'MultiObjectiveOptimizer',
    'OptimizationMethod',
    'OptimizationObjective',
    'ConstraintType',
    'OptimizationConstraint',
    'ObjectiveFunction',
    'OptimizationResult',
    'BaseObjectiveFunction',
    'ReturnObjective',
    'RiskObjective',
    'SharpeObjective',
    'DiversificationObjective',
    'TrackingErrorObjective',
    'ESGObjective',
    
    # Black-Litterman模型
    'BlackLittermanModel',
    'BlackLittermanResult',
    'ViewType',
    'ConfidenceLevel',
    'MarketParameters',
    'ViewParameters',
    
    # 风险预算模型
    'RiskBudgetingModel',
    'RiskBudgetingResult',
    'RiskBudgetingMethod',
    'RiskMeasure',
    'RiskContribution',
    'RiskBudget',
    
    # 动态对冲策略
    'DynamicHedgingEngine',
    'HedgingStrategy',
    'HedgingFrequency',
    'HedgingInstrument',
    'HedgeRatio',
    'HedgingPosition',
    'HedgingResult',
    
    # 参数优化器
    'ParameterOptimizer',
    
    # 优化算法
    'BaseOptimizer',
    'GridSearchOptimizer',
    'RandomSearchOptimizer',
    'BayesianOptimizer',
    'DifferentialEvolutionOptimizer',
    'HyperOptimizer',
    'OptimizationResult',
    
    # 目标函数
    'BaseObjectiveFunction',
    'PerformanceMetrics',
    'SharpeRatioObjective',
    'SortinoRatioObjective',
    'CalmarRatioObjective',
    'MaxDrawdownObjective',
    'ProfitFactorObjective',
    'TotalReturnObjective',
    'WinRateObjective',
    'CompositeObjective',
    'RiskAdjustedObjective',
    'MultiObjective',
    'ConstrainedObjective',
    'CommonConstraints',
    
    # 参数空间
    'Parameter',
    'ParameterType',
    'ParameterSpace',
    'ParameterSpaceBuilder',
    'CommonParameterSpaces'
]
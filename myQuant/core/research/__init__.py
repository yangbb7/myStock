# -*- coding: utf-8 -*-
"""
回测与研究平台模块 - 提供前瞻性分析、多资产回测、交易成本模型等高级功能
"""

# 前瞻性分析
from .forward_analysis import (
    ForwardAnalysisEngine,
    ForwardAnalysisMethod,
    ScenarioType,
    RiskFactor,
    ForwardScenario,
    ForwardPrediction,
    PortfolioProjection,
    ForwardAnalysisResult
)

# 多资产回测
from .multi_asset_backtester import (
    MultiAssetBacktester,
    AssetClass,
    BacktestFrequency,
    RebalanceMethod,
    BenchmarkType,
    AssetData,
    BacktestConfig,
    Trade,
    Position,
    PerformanceMetrics,
    BacktestResult
)

# 交易成本模型
from .transaction_cost_models import (
    TransactionCostModel,
    CostComponent,
    MarketRegime,
    TradingVenue,
    OrderType,
    MarketData,
    TradeInstruction,
    CostBreakdown,
    TransactionCostEstimate,
    ExecutionResult
)

# 市场微观结构模拟
from .market_microstructure_simulator import (
    MarketMicrostructureSimulator,
    OrderType as MSOrderType,
    OrderSide,
    OrderStatus,
    TraderType,
    MarketEvent,
    Order,
    Trade as MSSTrade,
    OrderBookLevel,
    OrderBook,
    MarketState,
    TraderBehavior,
    SimulationResult
)

# 滑点优化
from .slippage_optimizer import (
    SlippageOptimizer,
    SlippageModel,
    MarketCondition,
    ExecutionStyle,
    SlippageParameters,
    MarketImpactFactors,
    SlippageEstimate,
    SlippageCalibration
)

# 容量分析
from .capacity_analyzer import (
    CapacityAnalyzer,
    CapacityConstraint,
    CapacityMetric,
    ScalingMethod,
    CapacityInput,
    CapacityConstraintAnalysis,
    CapacityEstimate,
    CapacityScenario
)

__all__ = [
    # 前瞻性分析
    'ForwardAnalysisEngine',
    'ForwardAnalysisMethod',
    'ScenarioType',
    'RiskFactor',
    'ForwardScenario',
    'ForwardPrediction',
    'PortfolioProjection',
    'ForwardAnalysisResult',
    
    # 多资产回测
    'MultiAssetBacktester',
    'AssetClass',
    'BacktestFrequency',
    'RebalanceMethod',
    'BenchmarkType',
    'AssetData',
    'BacktestConfig',
    'Trade',
    'Position',
    'PerformanceMetrics',
    'BacktestResult',
    
    # 交易成本模型
    'TransactionCostModel',
    'CostComponent',
    'MarketRegime',
    'TradingVenue',
    'OrderType',
    'MarketData',
    'TradeInstruction',
    'CostBreakdown',
    'TransactionCostEstimate',
    'ExecutionResult',
    
    # 市场微观结构模拟
    'MarketMicrostructureSimulator',
    'MSOrderType',
    'OrderSide',
    'OrderStatus',
    'TraderType',
    'MarketEvent',
    'Order',
    'MSSTrade',
    'OrderBookLevel',
    'OrderBook',
    'MarketState',
    'TraderBehavior',
    'SimulationResult',
    
    # 滑点优化
    'SlippageOptimizer',
    'SlippageModel',
    'MarketCondition',
    'ExecutionStyle',
    'SlippageParameters',
    'MarketImpactFactors',
    'SlippageEstimate',
    'SlippageCalibration',
    
    # 容量分析
    'CapacityAnalyzer',
    'CapacityConstraint',
    'CapacityMetric',
    'ScalingMethod',
    'CapacityInput',
    'CapacityConstraintAnalysis',
    'CapacityEstimate',
    'CapacityScenario'
]
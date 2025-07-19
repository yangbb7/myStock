# -*- coding: utf-8 -*-
"""
分析模块 - 提供投资组合分析仪表板和绩效归因分析功能
"""

# 投资组合仪表板
from .portfolio_dashboard import (
    PortfolioDashboard,
    DashboardType,
    TimeFrame,
    ChartType,
    PortfolioMetrics,
    HoldingData,
    DashboardConfig
)

# 绩效归因分析
from .performance_attribution import (
    PerformanceAttribution,
    AttributionMethod,
    AttributionLevel,
    AttributionPeriod,
    AttributionComponent,
    SecurityAttribution,
    FactorAttribution,
    CurrencyAttribution,
    AttributionSummary
)

__all__ = [
    # 投资组合仪表板
    'PortfolioDashboard',
    'DashboardType',
    'TimeFrame',
    'ChartType',
    'PortfolioMetrics',
    'HoldingData',
    'DashboardConfig',
    
    # 绩效归因分析
    'PerformanceAttribution',
    'AttributionMethod',
    'AttributionLevel',
    'AttributionPeriod',
    'AttributionComponent',
    'SecurityAttribution',
    'FactorAttribution',
    'CurrencyAttribution',
    'AttributionSummary'
]
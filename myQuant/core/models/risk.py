# -*- coding: utf-8 -*-
"""
风险模型 - 定义风险管理相关的数据结构
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
from decimal import Decimal


class RiskType(Enum):
    """风险类型"""
    MARKET_RISK = "market_risk"
    CREDIT_risk = "credit_risk"
    LIQUIDITY_RISK = "liquidity_risk"
    OPERATIONAL_RISK = "operational_risk"
    CONCENTRATION_RISK = "concentration_risk"
    MODEL_RISK = "model_risk"


class RiskLevel(Enum):
    """风险级别"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """预警状态"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class RiskLimits:
    """风险限制"""
    portfolio_id: str
    
    # 位置限制
    max_position_size: Optional[Decimal] = None
    max_position_percentage: Optional[float] = None
    max_sector_exposure: Optional[float] = None
    max_single_stock_exposure: Optional[float] = None
    
    # 风险限制
    max_var_daily: Optional[Decimal] = None
    max_var_weekly: Optional[Decimal] = None
    max_drawdown: Optional[float] = None
    max_leverage: Optional[float] = None
    
    # 流动性限制
    min_cash_ratio: Optional[float] = None
    max_illiquid_percentage: Optional[float] = None
    
    # 时间限制
    effective_from: datetime = field(default_factory=datetime.now)
    effective_until: Optional[datetime] = None
    
    # 元数据
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: str = ""
    notes: str = ""


@dataclass
class RiskMetrics:
    """风险指标"""
    portfolio_id: str
    timestamp: datetime
    
    # VaR指标
    var_1d_95: Optional[Decimal] = None
    var_1d_99: Optional[Decimal] = None
    var_5d_95: Optional[Decimal] = None
    var_10d_95: Optional[Decimal] = None
    
    # CVaR指标
    cvar_1d_95: Optional[Decimal] = None
    cvar_1d_99: Optional[Decimal] = None
    
    # 波动性指标
    portfolio_volatility: Optional[float] = None
    realized_volatility: Optional[float] = None
    implied_volatility: Optional[float] = None
    
    # 回撤指标
    current_drawdown: Optional[float] = None
    max_drawdown: Optional[float] = None
    max_drawdown_duration: Optional[int] = None
    
    # 集中度指标
    concentration_ratio: Optional[float] = None
    herfindahl_index: Optional[float] = None
    effective_number_positions: Optional[int] = None
    
    # 流动性指标
    liquidity_ratio: Optional[float] = None
    cash_ratio: Optional[float] = None
    turnover_ratio: Optional[float] = None
    
    # 相关性指标
    beta_to_market: Optional[float] = None
    correlation_to_benchmark: Optional[float] = None
    tracking_error: Optional[float] = None
    
    # 绩效指标
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    information_ratio: Optional[float] = None
    calmar_ratio: Optional[float] = None
    
    # 元数据
    calculation_method: str = "historical"
    data_window_days: int = 252
    confidence_level: float = 0.95
    notes: str = ""


@dataclass
class RiskAlert:
    """风险预警"""
    alert_id: str
    portfolio_id: Optional[str] = None
    symbol: Optional[str] = None
    
    # 预警信息
    risk_type: RiskType = RiskType.MARKET_RISK
    level: RiskLevel = RiskLevel.MEDIUM
    title: str = ""
    message: str = ""
    description: str = ""
    
    # 阈值信息
    current_value: Optional[Decimal] = None
    threshold_value: Optional[Decimal] = None
    limit_type: str = ""
    
    # 状态信息
    status: AlertStatus = AlertStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    triggered_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    
    # 处理信息
    acknowledged_by: Optional[str] = None
    resolved_by: Optional[str] = None
    resolution_notes: str = ""
    
    # 通知信息
    notification_sent: bool = False
    notification_channels: List[str] = field(default_factory=list)
    notification_attempts: int = 0
    
    # 元数据
    source_system: str = ""
    rule_id: Optional[str] = None
    priority: int = 1  # 1(最高) - 5(最低)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskExposure:
    """风险暴露"""
    portfolio_id: str
    symbol: str
    timestamp: datetime
    
    # 位置信息
    position_size: Decimal
    market_value: Decimal
    percentage_of_portfolio: float
    
    # 风险指标
    var_contribution: Optional[Decimal] = None
    beta: Optional[float] = None
    volatility: Optional[float] = None
    correlation_to_portfolio: Optional[float] = None
    
    # 行业/部门信息
    sector: Optional[str] = None
    industry: Optional[str] = None
    country: Optional[str] = None
    currency: Optional[str] = None
    
    # 流动性信息
    liquidity_score: Optional[float] = None
    avg_daily_volume: Optional[int] = None
    bid_ask_spread: Optional[float] = None
    
    # 元数据
    notes: str = ""


@dataclass
class StressTestScenario:
    """压力测试场景"""
    scenario_id: str
    name: str
    description: str
    
    # 场景参数
    market_shock: Dict[str, float] = field(default_factory=dict)  # symbol -> shock_percentage
    volatility_multiplier: float = 1.0
    correlation_adjustment: float = 0.0
    liquidity_impact: float = 0.0
    
    # 时间参数
    shock_duration_days: int = 1
    recovery_period_days: int = 0
    
    # 元数据
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = ""
    category: str = "custom"  # "historical", "hypothetical", "regulatory", "custom"
    tags: List[str] = field(default_factory=list)


@dataclass
class StressTestResult:
    """压力测试结果"""
    test_id: str
    portfolio_id: str
    scenario_id: str
    timestamp: datetime
    
    # 结果指标
    portfolio_pnl: Decimal
    portfolio_pnl_percentage: float
    worst_position_pnl: Decimal
    worst_position_symbol: str
    
    # 风险指标
    post_stress_var: Optional[Decimal] = None
    post_stress_leverage: Optional[float] = None
    liquidity_shortfall: Optional[Decimal] = None
    
    # 位置级别结果
    position_results: Dict[str, Decimal] = field(default_factory=dict)  # symbol -> pnl
    
    # 元数据
    calculation_time_ms: int = 0
    notes: str = ""


@dataclass
class RiskReport:
    """风险报告"""
    report_id: str
    portfolio_id: str
    report_type: str  # "daily", "weekly", "monthly", "ad_hoc"
    period_start: datetime
    period_end: datetime
    generated_at: datetime
    
    # 报告内容
    executive_summary: str = ""
    risk_metrics: Optional[RiskMetrics] = None
    limit_utilization: Dict[str, float] = field(default_factory=dict)
    active_alerts: List[RiskAlert] = field(default_factory=list)
    stress_test_results: List[StressTestResult] = field(default_factory=list)
    
    # 分析结果
    key_risks: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # 元数据
    generated_by: str = ""
    approved_by: Optional[str] = None
    distribution_list: List[str] = field(default_factory=list)
    attachments: List[str] = field(default_factory=list)

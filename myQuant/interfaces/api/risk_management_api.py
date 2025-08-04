# -*- coding: utf-8 -*-
"""
风险管理API

提供风险评估、风险指标计算、风险预警等功能
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, date
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from enum import Enum

from myQuant.core.managers.risk_manager import RiskManager
from myQuant.infrastructure.database.repositories import RiskMetricRepository


class RiskLevel(str, Enum):
    """风险等级"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


class RiskAlert(BaseModel):
    """风险预警"""
    id: str
    type: str
    level: RiskLevel
    message: str
    symbol: Optional[str] = None
    portfolio_id: Optional[int] = None
    threshold: float
    current_value: float
    created_at: datetime
    is_active: bool


class RiskMetrics(BaseModel):
    """风险指标"""
    portfolio_id: int
    date: date
    portfolio_value: float
    daily_pnl: float
    max_drawdown: float
    var_95: float  # 95% VaR
    var_99: Optional[float] = None  # 99% VaR
    beta: Optional[float] = None
    sharpe_ratio: float
    volatility: float
    tracking_error: Optional[float] = None
    information_ratio: Optional[float] = None
    calculated_at: datetime


class PositionRisk(BaseModel):
    """持仓风险"""
    symbol: str
    quantity: int
    market_value: float
    weight: float
    beta: Optional[float] = None
    volatility: float
    var_daily: float
    correlation_risk: float
    liquidity_risk: RiskLevel
    concentration_risk: RiskLevel


class RiskAssessment(BaseModel):
    """风险评估结果"""
    portfolio_id: int
    overall_risk_level: RiskLevel
    risk_score: float
    metrics: RiskMetrics
    position_risks: List[PositionRisk]
    alerts: List[RiskAlert]
    recommendations: List[str]
    assessment_date: datetime


class RiskLimits(BaseModel):
    """风险限额设置"""
    max_position_weight: float = Field(default=0.1, ge=0.01, le=0.5)
    max_sector_weight: float = Field(default=0.3, ge=0.1, le=0.8)
    max_drawdown_limit: float = Field(default=-0.2, le=-0.05, ge=-0.5)
    var_limit: float = Field(default=0.05, ge=0.01, le=0.2)
    beta_range: Dict[str, float] = Field(default={"min": 0.5, "max": 1.5})
    volatility_limit: float = Field(default=0.25, ge=0.1, le=1.0)


class RiskManagementAPI:
    """风险管理API"""
    
    def __init__(self, risk_repo: RiskMetricRepository, risk_manager: RiskManager):
        self.risk_manager = risk_manager
        self.risk_repo = risk_repo
        self.router = APIRouter(prefix="/risk", tags=["risk-management"])
        self._setup_routes()
    
    def _setup_routes(self):
        """设置路由"""
        
        @self.router.get("/assessment/{portfolio_id}", response_model=RiskAssessment)
        async def get_risk_assessment(
            portfolio_id: int,
            user_id: int = Depends(self._get_current_user)
        ):
            """获取投资组合风险评估"""
            try:
                # 验证用户权限
                await self._verify_portfolio_access(portfolio_id, user_id)
                
                # 计算风险指标
                metrics = await self.risk_manager.calculate_portfolio_risk(portfolio_id)
                
                # 评估持仓风险
                position_risks = await self.risk_manager.assess_position_risks(portfolio_id)
                
                # 获取风险预警
                alerts = await self.risk_manager.get_active_alerts(portfolio_id)
                
                # 生成风险建议
                recommendations = await self.risk_manager.generate_recommendations(portfolio_id)
                
                # 计算总体风险等级
                overall_risk = self._calculate_overall_risk_level(metrics)
                
                return RiskAssessment(
                    portfolio_id=portfolio_id,
                    overall_risk_level=overall_risk,
                    risk_score=metrics.get('risk_score', 0),
                    metrics=RiskMetrics(**metrics),
                    position_risks=[PositionRisk(**pos) for pos in position_risks],
                    alerts=[RiskAlert(**alert) for alert in alerts],
                    recommendations=recommendations,
                    assessment_date=datetime.utcnow()
                )
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.router.get("/metrics/{portfolio_id}", response_model=List[RiskMetrics])
        async def get_risk_metrics_history(
            portfolio_id: int,
            start_date: date = None,
            end_date: date = None,
            user_id: int = Depends(self._get_current_user)
        ):
            """获取风险指标历史"""
            try:
                await self._verify_portfolio_access(portfolio_id, user_id)
                
                if start_date and end_date:
                    metrics_data = await self.risk_repo.get_by_user_range(
                        user_id, str(start_date), str(end_date)
                    )
                else:
                    # 默认获取最近30天
                    end_date = date.today()
                    start_date = date.today().replace(day=1)  # 当前月第一天
                    metrics_data = await self.risk_repo.get_by_user_range(
                        user_id, str(start_date), str(end_date)
                    )
                
                return [RiskMetrics(**data) for data in metrics_data]
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.router.get("/alerts/{portfolio_id}", response_model=List[RiskAlert])
        async def get_risk_alerts(
            portfolio_id: int,
            active_only: bool = True,
            user_id: int = Depends(self._get_current_user)
        ):
            """获取风险预警"""
            try:
                await self._verify_portfolio_access(portfolio_id, user_id)
                
                alerts = await self.risk_manager.get_alerts(portfolio_id, active_only)
                return [RiskAlert(**alert) for alert in alerts]
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.router.post("/limits/{portfolio_id}", response_model=Dict[str, str])
        async def set_risk_limits(
            portfolio_id: int,
            limits: RiskLimits,
            user_id: int = Depends(self._get_current_user)
        ):
            """设置风险限额"""
            try:
                await self._verify_portfolio_access(portfolio_id, user_id)
                
                await self.risk_manager.set_risk_limits(portfolio_id, limits.dict())
                return {"message": "Risk limits updated successfully"}
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.router.get("/limits/{portfolio_id}", response_model=RiskLimits)
        async def get_risk_limits(
            portfolio_id: int,
            user_id: int = Depends(self._get_current_user)
        ):
            """获取风险限额设置"""
            try:
                await self._verify_portfolio_access(portfolio_id, user_id)
                
                limits = await self.risk_manager.get_risk_limits(portfolio_id)
                return RiskLimits(**limits)
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.router.post("/stress-test/{portfolio_id}", response_model=Dict[str, Any])
        async def run_stress_test(
            portfolio_id: int,
            scenarios: List[Dict[str, Any]],
            user_id: int = Depends(self._get_current_user)
        ):
            """运行压力测试"""
            try:
                await self._verify_portfolio_access(portfolio_id, user_id)
                
                results = await self.risk_manager.run_stress_test(portfolio_id, scenarios)
                return results
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.router.post("/correlation-analysis/{portfolio_id}", response_model=Dict[str, Any])
        async def analyze_correlation(
            portfolio_id: int,
            period_days: int = 252,
            user_id: int = Depends(self._get_current_user)
        ):
            """相关性分析"""
            try:
                await self._verify_portfolio_access(portfolio_id, user_id)
                
                correlation_matrix = await self.risk_manager.calculate_correlation_matrix(
                    portfolio_id, period_days
                )
                return correlation_matrix
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.router.post("/alerts/{alert_id}/acknowledge", response_model=Dict[str, str])
        async def acknowledge_alert(
            alert_id: str,
            user_id: int = Depends(self._get_current_user)
        ):
            """确认风险预警"""
            try:
                await self.risk_manager.acknowledge_alert(alert_id, user_id)
                return {"message": "Alert acknowledged successfully"}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
    
    async def get_risk_metrics(self, user_id: int):
        """获取风险指标（测试方法）"""
        risk_data = await self.risk_manager.calculate_portfolio_risk(user_id)
        return {
            "success": True,
            "data": risk_data
        }
    
    async def check_order_risk(self, user_id: int, order: Dict[str, Any]):
        """检查订单风险（测试方法）"""
        risk_data = await self.risk_manager.check_order_risk(user_id, order)
        return {
            "success": True,
            "data": risk_data
        }
    
    async def set_risk_limits(self, user_id: int, limits: Dict[str, Any]):
        """设置风险限制（测试方法）"""
        result = await self.risk_manager.set_risk_limits(user_id, limits)
        return {
            "success": True,
            "message": result["message"]
        }
    
    async def get_risk_alerts(self, user_id: int, status: str = "ACTIVE"):
        """获取风险提醒（测试方法）"""
        alerts = await self.risk_manager.get_alerts(user_id, status == "ACTIVE")
        return {
            "success": True,
            "data": alerts
        }
    
    async def calculate_portfolio_var(self, user_id: int, confidence_level: float, 
                                    holding_period: int, method: str):
        """计算投资组合VaR（测试方法）"""
        var_data = await self.risk_manager.calculate_var(
            user_id, confidence_level, holding_period, method
        )
        return {
            "success": True,
            "data": var_data
        }
    
    async def run_stress_tests(self, user_id: int, scenarios: Dict[str, Any]):
        """运行压力测试（测试方法）"""
        results = await self.risk_manager.run_stress_tests(user_id, scenarios)
        return {
            "success": True,
            "data": results
        }
    
    async def start_real_time_monitoring(self, user_id: int):
        """开始实时风险监控（测试方法）"""
        monitoring_result = await self.risk_manager.start_monitoring(user_id)
        return {
            "success": True,
            "data": monitoring_result
        }
    
    async def execute_emergency_action(self, user_id: int, action_type: str, reason: str):
        """执行紧急风险操作（测试方法）"""
        action_result = await self.risk_manager.execute_emergency_action(user_id, action_type, reason)
        return {
            "success": True,
            "data": action_result
        }
    
    def _get_current_user(self) -> int:
        """获取当前用户ID（占位符，实际实现需要从认证系统获取）"""
        # TODO: 实现真实的用户认证
        return 1
    
    async def _verify_portfolio_access(self, portfolio_id: int, user_id: int):
        """验证用户对投资组合的访问权限"""
        # TODO: 实现真实的权限验证
        pass
    
    def _calculate_overall_risk_level(self, metrics: Dict[str, Any]) -> RiskLevel:
        """计算总体风险等级"""
        risk_score = metrics.get('risk_score', 0)
        
        if risk_score >= 80:
            return RiskLevel.EXTREME
        elif risk_score >= 60:
            return RiskLevel.HIGH
        elif risk_score >= 30:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
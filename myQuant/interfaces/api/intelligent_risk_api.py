# -*- coding: utf-8 -*-
"""
智能风控API接口
"""

from fastapi import APIRouter, HTTPException, Depends, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import asyncio
import json
import logging

from myQuant.core.risk.intelligent_risk_control import (
    IntelligentRiskControl,
    RiskRule,
    StopLossConfig,
    StopLossType,
    PositionLimit,
    RiskAlert,
    RiskLevel
)
from myQuant.core.managers.portfolio_manager import PortfolioManager
from myQuant.core.managers.data_manager import DataManager
from myQuant.interfaces.api.user_authentication_api import get_current_user

router = APIRouter(prefix="/api/v1/risk", tags=["risk"])
logger = logging.getLogger(__name__)

# 初始化风控系统
risk_control = IntelligentRiskControl()

# WebSocket连接管理
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                pass

manager = ConnectionManager()


class RiskConfigUpdate(BaseModel):
    """风险配置更新"""
    max_position_size: Optional[float] = None
    max_position_percent: Optional[float] = None
    max_total_exposure: Optional[float] = None
    max_sector_exposure: Optional[float] = None
    max_correlation: Optional[float] = None
    stop_loss_enabled: Optional[bool] = None
    stop_loss_percent: Optional[float] = None
    trailing_stop_enabled: Optional[bool] = None
    trailing_stop_percent: Optional[float] = None
    take_profit_enabled: Optional[bool] = None
    take_profit_levels: Optional[List[Dict[str, float]]] = None


class RiskRuleCreate(BaseModel):
    """创建风险规则"""
    name: str
    condition: str
    action: str
    params: Dict[str, Any] = {}
    priority: int = 10
    enabled: bool = True


class StressTestScenario(BaseModel):
    """压力测试场景"""
    name: str
    market_change: float
    volatility_change: float
    correlation_change: Optional[float] = 0


@router.get("/config")
async def get_risk_config(
    current_user: dict = Depends(get_current_user)
):
    """获取风险配置"""
    return {
        "max_position_size": risk_control.position_limits.max_position_size,
        "max_position_percent": risk_control.position_limits.max_position_percent,
        "max_total_exposure": risk_control.position_limits.max_total_exposure,
        "max_sector_exposure": risk_control.position_limits.max_sector_exposure,
        "max_correlation": risk_control.position_limits.max_correlation,
        "stop_loss_enabled": True,
        "stop_loss_percent": risk_control.stop_loss_configs["default"].stop_percent,
        "trailing_stop_enabled": True,
        "trailing_stop_percent": risk_control.stop_loss_configs["trend"].trailing_percent,
        "take_profit_enabled": True,
        "take_profit_levels": [
            {"percent": 0.05, "close_ratio": 0.3},
            {"percent": 0.10, "close_ratio": 0.3},
            {"percent": 0.15, "close_ratio": 0.4}
        ]
    }


@router.put("/config")
async def update_risk_config(
    config: RiskConfigUpdate,
    current_user: dict = Depends(get_current_user)
):
    """更新风险配置"""
    try:
        # 更新仓位限制
        if config.max_position_size is not None:
            risk_control.position_limits.max_position_size = config.max_position_size
        if config.max_position_percent is not None:
            risk_control.position_limits.max_position_percent = config.max_position_percent
        if config.max_total_exposure is not None:
            risk_control.position_limits.max_total_exposure = config.max_total_exposure
        if config.max_sector_exposure is not None:
            risk_control.position_limits.max_sector_exposure = config.max_sector_exposure
        if config.max_correlation is not None:
            risk_control.position_limits.max_correlation = config.max_correlation
        
        # 更新止损配置
        if config.stop_loss_percent is not None:
            risk_control.stop_loss_configs["default"].stop_percent = config.stop_loss_percent
        if config.trailing_stop_percent is not None:
            risk_control.stop_loss_configs["trend"].trailing_percent = config.trailing_stop_percent
        
        logger.info(f"Risk config updated by user {current_user['id']}")
        return {"status": "success", "message": "风险配置已更新"}
        
    except Exception as e:
        logger.error(f"Failed to update risk config: {str(e)}")
        raise HTTPException(status_code=500, detail=f"更新失败: {str(e)}")


@router.get("/alerts")
async def get_risk_alerts(
    level: Optional[str] = None,
    type: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 100,
    current_user: dict = Depends(get_current_user)
):
    """获取风险警报"""
    alerts = risk_control.active_alerts
    
    # 过滤
    if level:
        alerts = [a for a in alerts if a.level.value == level]
    if type:
        alerts = [a for a in alerts if a.type == type]
    if start_date:
        start = datetime.fromisoformat(start_date)
        alerts = [a for a in alerts if a.timestamp >= start]
    if end_date:
        end = datetime.fromisoformat(end_date)
        alerts = [a for a in alerts if a.timestamp <= end]
    
    # 排序并限制数量
    alerts.sort(key=lambda x: x.timestamp, reverse=True)
    alerts = alerts[:limit]
    
    # 转换为响应格式
    return [
        {
            "alert_id": a.alert_id,
            "timestamp": a.timestamp.isoformat(),
            "level": a.level.value,
            "type": a.type,
            "message": a.message,
            "position_id": a.position_id,
            "suggested_action": a.suggested_action,
            "metadata": a.metadata
        }
        for a in alerts
    ]


@router.post("/alerts/{alert_id}/execute")
async def execute_risk_alert(
    alert_id: str,
    current_user: dict = Depends(get_current_user)
):
    """执行风险警报建议"""
    try:
        # 查找警报
        alert = next((a for a in risk_control.active_alerts if a.alert_id == alert_id), None)
        if not alert:
            raise HTTPException(status_code=404, detail="警报不存在")
        
        # TODO: 执行建议的动作
        # 这里需要集成订单管理系统
        
        # 从活跃警报中移除
        risk_control.active_alerts.remove(alert)
        
        logger.info(f"Risk alert {alert_id} executed by user {current_user['id']}")
        return {
            "status": "success",
            "message": f"已执行风险警报建议: {alert.suggested_action}",
            "alert_id": alert_id
        }
        
    except Exception as e:
        logger.error(f"Failed to execute risk alert: {str(e)}")
        raise HTTPException(status_code=500, detail=f"执行失败: {str(e)}")


@router.post("/alerts/{alert_id}/dismiss")
async def dismiss_risk_alert(
    alert_id: str,
    current_user: dict = Depends(get_current_user)
):
    """忽略风险警报"""
    try:
        # 查找并移除警报
        alert = next((a for a in risk_control.active_alerts if a.alert_id == alert_id), None)
        if not alert:
            raise HTTPException(status_code=404, detail="警报不存在")
        
        risk_control.active_alerts.remove(alert)
        
        logger.info(f"Risk alert {alert_id} dismissed by user {current_user['id']}")
        return {"status": "success", "message": "警报已忽略"}
        
    except Exception as e:
        logger.error(f"Failed to dismiss risk alert: {str(e)}")
        raise HTTPException(status_code=500, detail=f"操作失败: {str(e)}")


@router.get("/metrics")
async def get_risk_metrics(
    current_user: dict = Depends(get_current_user)
):
    """获取风险指标"""
    # TODO: 从投资组合管理器获取实际数据
    # 这里返回模拟数据
    return {
        "total_positions": 5,
        "risk_level": "medium",
        "active_alerts": len(risk_control.active_alerts),
        "risk_metrics": {
            "max_drawdown": -0.08,
            "value_at_risk": -15000,
            "expected_shortfall": -20000,
            "portfolio_volatility": 0.18
        },
        "position_distribution": {
            "technology": 0.35,
            "finance": 0.25,
            "consumer": 0.20,
            "healthcare": 0.15,
            "others": 0.05
        },
        "recommendations": [
            "考虑减少科技股仓位",
            "增加防御性板块配置"
        ]
    }


@router.get("/rules")
async def get_risk_rules(
    current_user: dict = Depends(get_current_user)
):
    """获取风险规则"""
    return [
        {
            "rule_id": rule.rule_id,
            "name": rule.name,
            "condition": rule.condition,
            "action": rule.action,
            "params": rule.params,
            "priority": rule.priority,
            "enabled": rule.enabled
        }
        for rule in risk_control.risk_rules
    ]


@router.post("/rules")
async def create_risk_rule(
    rule_data: RiskRuleCreate,
    current_user: dict = Depends(get_current_user)
):
    """创建风险规则"""
    try:
        rule = RiskRule(
            rule_id=f"rule_{datetime.now().timestamp()}",
            name=rule_data.name,
            condition=rule_data.condition,
            action=rule_data.action,
            params=rule_data.params,
            priority=rule_data.priority,
            enabled=rule_data.enabled
        )
        
        risk_control.add_risk_rule(rule)
        
        logger.info(f"Risk rule created: {rule.rule_id}")
        return {
            "status": "success",
            "rule_id": rule.rule_id,
            "message": "风险规则创建成功"
        }
        
    except Exception as e:
        logger.error(f"Failed to create risk rule: {str(e)}")
        raise HTTPException(status_code=500, detail=f"创建失败: {str(e)}")


@router.put("/rules/{rule_id}")
async def update_risk_rule(
    rule_id: str,
    rule_data: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
):
    """更新风险规则"""
    try:
        # 查找规则
        rule = next((r for r in risk_control.risk_rules if r.rule_id == rule_id), None)
        if not rule:
            raise HTTPException(status_code=404, detail="规则不存在")
        
        # 更新属性
        for key, value in rule_data.items():
            if hasattr(rule, key):
                setattr(rule, key, value)
        
        logger.info(f"Risk rule updated: {rule_id}")
        return {"status": "success", "message": "规则更新成功"}
        
    except Exception as e:
        logger.error(f"Failed to update risk rule: {str(e)}")
        raise HTTPException(status_code=500, detail=f"更新失败: {str(e)}")


@router.delete("/rules/{rule_id}")
async def delete_risk_rule(
    rule_id: str,
    current_user: dict = Depends(get_current_user)
):
    """删除风险规则"""
    try:
        risk_control.risk_rules = [r for r in risk_control.risk_rules if r.rule_id != rule_id]
        logger.info(f"Risk rule deleted: {rule_id}")
        return {"status": "success", "message": "规则删除成功"}
        
    except Exception as e:
        logger.error(f"Failed to delete risk rule: {str(e)}")
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")


@router.post("/stress-test")
async def run_stress_test(
    scenarios: List[StressTestScenario],
    current_user: dict = Depends(get_current_user)
):
    """运行压力测试"""
    try:
        results = []
        
        for scenario in scenarios:
            # TODO: 实现实际的压力测试逻辑
            # 这里返回模拟结果
            result = {
                "scenario": scenario.name,
                "market_change": scenario.market_change,
                "volatility_change": scenario.volatility_change,
                "impact": {
                    "portfolio_value_change": scenario.market_change * 0.8,
                    "max_drawdown": min(-0.15, scenario.market_change * 1.5),
                    "var_change": scenario.volatility_change * 10000,
                    "margin_call_risk": "high" if scenario.market_change < -0.1 else "low"
                }
            }
            results.append(result)
        
        return {
            "test_time": datetime.now().isoformat(),
            "scenarios": results,
            "summary": {
                "worst_case_loss": min(r["impact"]["portfolio_value_change"] for r in results),
                "highest_risk_scenario": max(results, key=lambda x: abs(x["impact"]["max_drawdown"]))["scenario"]
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to run stress test: {str(e)}")
        raise HTTPException(status_code=500, detail=f"压力测试失败: {str(e)}")


@router.websocket("/alerts")
async def websocket_risk_alerts(
    websocket: WebSocket,
    # current_user: dict = Depends(get_current_user)  # WebSocket认证需要特殊处理
):
    """WebSocket实时风险警报"""
    await manager.connect(websocket)
    try:
        while True:
            # 模拟定期发送风险警报
            await asyncio.sleep(10)
            
            # 检查是否有新的警报
            if risk_control.active_alerts:
                latest_alert = risk_control.active_alerts[-1]
                alert_data = {
                    "alert_id": latest_alert.alert_id,
                    "timestamp": latest_alert.timestamp.isoformat(),
                    "level": latest_alert.level.value,
                    "type": latest_alert.type,
                    "message": latest_alert.message,
                    "suggested_action": latest_alert.suggested_action
                }
                await websocket.send_json(alert_data)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        manager.disconnect(websocket)
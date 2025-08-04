# -*- coding: utf-8 -*-
"""
AI策略助手API接口
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging

from myQuant.core.ai.strategy_assistant import (
    StrategyAssistant,
    MarketCondition,
    StrategyRecommendation,
    OptimizationSuggestion
)
from myQuant.core.managers.data_manager import DataManager
from myQuant.interfaces.api.user_authentication_api import get_current_user

router = APIRouter(prefix="/api/v1/ai-assistant", tags=["ai-assistant"])
logger = logging.getLogger(__name__)

# 初始化AI助手
strategy_assistant = StrategyAssistant()


class MarketAnalysisRequest(BaseModel):
    """市场分析请求"""
    symbols: List[str]
    period: str = "1d"  # 1d, 1h, 5m
    lookback_days: int = 60


class StrategyRecommendationRequest(BaseModel):
    """策略推荐请求"""
    symbols: List[str]
    risk_preference: str = "medium"  # low, medium, high
    capital: float = 100000.0
    existing_strategies: List[str] = []


class OptimizationRequest(BaseModel):
    """策略优化请求"""
    strategy_id: str
    optimization_metric: str = "sharpe_ratio"  # sharpe_ratio, total_return, win_rate
    lookback_days: int = 90


class RiskControlRequest(BaseModel):
    """风险控制建议请求"""
    strategy_id: str
    portfolio_value: float
    risk_tolerance: str = "medium"


@router.post("/analyze-market")
async def analyze_market(
    request: MarketAnalysisRequest,
    current_user: dict = Depends(get_current_user)
):
    """分析市场状态"""
    try:
        # 获取市场数据
        data_manager = DataManager()
        market_data = {}
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=request.lookback_days)
        
        for symbol in request.symbols:
            try:
                data = data_manager.get_historical_data(
                    symbol,
                    start_date.strftime("%Y-%m-%d"),
                    end_date.strftime("%Y-%m-%d"),
                    request.period
                )
                if data is not None and len(data) > 0:
                    market_data[symbol] = data
            except Exception as e:
                logger.warning(f"获取{symbol}数据失败: {str(e)}")
        
        if not market_data:
            raise HTTPException(status_code=404, detail="无法获取市场数据")
        
        # 分析市场状态
        conditions = {}
        for symbol, data in market_data.items():
            condition = strategy_assistant.analyze_market_condition(data)
            conditions[symbol] = {
                "trend": condition.trend,
                "volatility": condition.volatility,
                "volume_trend": condition.volume_trend,
                "confidence": condition.confidence
            }
        
        # 综合市场状态
        trends = [c["trend"] for c in conditions.values()]
        volatilities = [c["volatility"] for c in conditions.values()]
        
        overall_trend = max(set(trends), key=trends.count)
        overall_volatility = max(set(volatilities), key=volatilities.count)
        avg_confidence = sum(c["confidence"] for c in conditions.values()) / len(conditions)
        
        return {
            "overall": {
                "trend": overall_trend,
                "volatility": overall_volatility,
                "confidence": avg_confidence
            },
            "individual": conditions,
            "analysis_time": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"市场分析失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"市场分析失败: {str(e)}")


@router.post("/recommend-strategies")
async def recommend_strategies(
    request: StrategyRecommendationRequest,
    current_user: dict = Depends(get_current_user)
):
    """获取策略推荐"""
    try:
        # 获取市场数据
        data_manager = DataManager()
        market_data = {}
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)
        
        for symbol in request.symbols:
            try:
                data = data_manager.get_historical_data(
                    symbol,
                    start_date.strftime("%Y-%m-%d"),
                    end_date.strftime("%Y-%m-%d"),
                    "1d"
                )
                if data is not None and len(data) > 0:
                    market_data[symbol] = data
            except Exception as e:
                logger.warning(f"获取{symbol}数据失败: {str(e)}")
        
        # 获取推荐
        recommendations = strategy_assistant.recommend_strategies(
            request.symbols,
            market_data,
            request.risk_preference
        )
        
        # 转换为响应格式
        response_data = []
        for rec in recommendations:
            response_data.append({
                "strategy_type": rec.strategy_type,
                "reason": rec.reason,
                "expected_performance": rec.expected_performance,
                "risk_level": rec.risk_level,
                "suitable_symbols": rec.suitable_symbols,
                "parameters": rec.parameters,
                "confidence": rec.confidence
            })
        
        return {
            "recommendations": response_data,
            "total_capital": request.capital,
            "risk_preference": request.risk_preference,
            "recommendation_time": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"策略推荐失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"策略推荐失败: {str(e)}")


@router.post("/optimize-strategy")
async def optimize_strategy(
    request: OptimizationRequest,
    current_user: dict = Depends(get_current_user)
):
    """优化策略参数"""
    try:
        # TODO: 从数据库获取策略实例
        # strategy = get_strategy_by_id(request.strategy_id)
        
        # 模拟优化建议
        suggestions = [
            {
                "parameter": "fast_period",
                "current_value": 20,
                "suggested_value": 15,
                "expected_improvement": 0.12,
                "reason": "回测显示15日均线在当前市场环境下表现更好"
            },
            {
                "parameter": "stop_loss",
                "current_value": 0.05,
                "suggested_value": 0.03,
                "expected_improvement": 0.08,
                "reason": "收紧止损可以减少回撤，提高风险调整收益"
            },
            {
                "parameter": "position_size",
                "current_value": 0.8,
                "suggested_value": 0.6,
                "expected_improvement": 0.05,
                "reason": "降低仓位可以改善夏普比率"
            }
        ]
        
        return {
            "strategy_id": request.strategy_id,
            "optimization_metric": request.optimization_metric,
            "suggestions": suggestions,
            "optimization_time": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"策略优化失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"策略优化失败: {str(e)}")


@router.post("/risk-control-suggestions")
async def get_risk_control_suggestions(
    request: RiskControlRequest,
    current_user: dict = Depends(get_current_user)
):
    """获取风险控制建议"""
    try:
        # TODO: 从数据库获取策略实例
        # strategy = get_strategy_by_id(request.strategy_id)
        
        # 生成风控建议
        suggestions = {
            "position_limits": {
                "max_position_size": request.portfolio_value * 0.1,
                "max_position_percent": 0.1,
                "max_sector_exposure": 0.3,
                "max_correlation": 0.7
            },
            "stop_loss_levels": {
                "initial_stop_loss": 0.02 if request.risk_tolerance == "low" else 0.03,
                "trailing_stop_loss": 0.03,
                "time_stop_days": 30,
                "volatility_adjusted": True
            },
            "risk_alerts": [
                {
                    "type": "concentration",
                    "message": "建议增加持仓分散度，当前持仓过于集中",
                    "severity": "medium"
                },
                {
                    "type": "correlation",
                    "message": "部分持仓相关性较高，建议选择不同行业标的",
                    "severity": "low"
                }
            ],
            "diversification": {
                "recommended_positions": 8,
                "sector_allocation": {
                    "technology": 0.25,
                    "finance": 0.20,
                    "consumer": 0.20,
                    "healthcare": 0.15,
                    "others": 0.20
                }
            },
            "risk_metrics": {
                "var_95": request.portfolio_value * 0.02,
                "expected_shortfall": request.portfolio_value * 0.03,
                "max_acceptable_drawdown": 0.15
            }
        }
        
        return {
            "strategy_id": request.strategy_id,
            "portfolio_value": request.portfolio_value,
            "risk_tolerance": request.risk_tolerance,
            "suggestions": suggestions,
            "generated_time": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"风控建议生成失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"风控建议生成失败: {str(e)}")


@router.get("/strategy-performance-analysis/{strategy_id}")
async def analyze_strategy_performance(
    strategy_id: str,
    current_user: dict = Depends(get_current_user)
):
    """分析策略表现"""
    try:
        # TODO: 从数据库获取策略和其历史表现
        
        # 模拟分析结果
        analysis = {
            "performance_score": 75,
            "strengths": [
                "良好的风险调整收益（夏普比率 > 1.2）",
                "较低的最大回撤（< 10%）",
                "稳定的月度收益"
            ],
            "weaknesses": [
                "在震荡市场表现不佳",
                "换手率较高导致交易成本增加"
            ],
            "improvement_suggestions": [
                "考虑加入市场状态过滤器",
                "优化进出场时机以降低交易频率",
                "增加止盈策略以锁定利润"
            ],
            "market_fit": "当前市场环境适合该策略",
            "peer_comparison": {
                "rank": 15,
                "total": 100,
                "percentile": 85
            },
            "analysis_time": datetime.now().isoformat()
        }
        
        return analysis
        
    except Exception as e:
        logger.error(f"策略表现分析失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"策略表现分析失败: {str(e)}")


@router.get("/market-insights")
async def get_market_insights(
    current_user: dict = Depends(get_current_user)
):
    """获取市场洞察"""
    try:
        insights = {
            "market_summary": {
                "trend": "震荡上行",
                "key_levels": {
                    "support": 3200,
                    "resistance": 3500
                },
                "sentiment": "中性偏乐观"
            },
            "sector_rotation": {
                "strong_sectors": ["新能源", "半导体", "医药"],
                "weak_sectors": ["地产", "银行"],
                "emerging_themes": ["人工智能", "消费升级"]
            },
            "risk_factors": [
                {
                    "factor": "美联储政策",
                    "impact": "medium",
                    "probability": 0.6
                },
                {
                    "factor": "地缘政治",
                    "impact": "high",
                    "probability": 0.3
                }
            ],
            "opportunities": [
                {
                    "type": "sector",
                    "description": "新能源车产业链机会",
                    "confidence": 0.8
                },
                {
                    "type": "event",
                    "description": "季报业绩超预期个股",
                    "confidence": 0.7
                }
            ],
            "update_time": datetime.now().isoformat()
        }
        
        return insights
        
    except Exception as e:
        logger.error(f"获取市场洞察失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取市场洞察失败: {str(e)}")
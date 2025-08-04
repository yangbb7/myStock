# -*- coding: utf-8 -*-
"""
可视化策略API接口
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import logging

from myQuant.core.strategy.visual_strategy_builder import (
    VisualStrategyBuilder,
    DataBlock,
    IndicatorBlock,
    ConditionBlock,
    ActionBlock,
    BlockType,
    OperatorType
)
from myQuant.interfaces.api.user_authentication_api import get_current_user

router = APIRouter(prefix="/api/v1/visual-strategy", tags=["visual-strategy"])
logger = logging.getLogger(__name__)


class BlockData(BaseModel):
    """策略块数据"""
    id: str
    type: str
    params: Dict[str, Any] = {}
    position: Dict[str, float] = {}


class EdgeData(BaseModel):
    """连接数据"""
    source: str
    target: str
    targetHandle: Optional[str] = None


class VisualStrategyData(BaseModel):
    """可视化策略数据"""
    nodes: List[BlockData]
    edges: List[EdgeData]
    name: Optional[str] = "未命名策略"
    symbols: List[str] = ["000001.SZ"]


class StrategyTemplate(BaseModel):
    """策略模板"""
    id: str
    name: str
    description: str
    category: str
    difficulty: str  # beginner, intermediate, advanced
    strategy_data: Dict[str, Any]
    performance_metrics: Optional[Dict[str, float]] = None


# 存储用户策略（实际应用中应该使用数据库）
user_strategies: Dict[str, List[Dict]] = {}

# 预定义的策略模板
STRATEGY_TEMPLATES = [
    {
        "id": "ma_cross",
        "name": "双均线交叉",
        "description": "经典的移动平均线交叉策略，当短期均线上穿长期均线时买入",
        "category": "趋势跟踪",
        "difficulty": "beginner",
        "strategy_data": {
            "nodes": [
                {"id": "data1", "type": "data", "params": {"field": "close"}, "position": {"x": 50, "y": 100}},
                {"id": "sma5", "type": "indicator", "params": {"type": "SMA", "period": 5}, "position": {"x": 250, "y": 50}},
                {"id": "sma20", "type": "indicator", "params": {"type": "SMA", "period": 20}, "position": {"x": 250, "y": 150}},
                {"id": "cross", "type": "condition", "params": {"operator": "cross_above"}, "position": {"x": 450, "y": 100}},
                {"id": "buy", "type": "action", "params": {"type": "buy", "position_size": 1.0}, "position": {"x": 650, "y": 100}}
            ],
            "edges": [
                {"source": "data1", "target": "sma5"},
                {"source": "data1", "target": "sma20"},
                {"source": "sma5", "target": "cross"},
                {"source": "sma20", "target": "cross", "targetHandle": "input2"},
                {"source": "cross", "target": "buy"}
            ]
        },
        "performance_metrics": {
            "annual_return": 0.15,
            "sharpe_ratio": 1.2,
            "max_drawdown": -0.08,
            "win_rate": 0.55
        }
    },
    {
        "id": "rsi_oversold",
        "name": "RSI超卖反弹",
        "description": "当RSI指标低于30时买入，利用超卖反弹获利",
        "category": "均值回归",
        "difficulty": "beginner",
        "strategy_data": {
            "nodes": [
                {"id": "data1", "type": "data", "params": {"field": "close"}, "position": {"x": 50, "y": 100}},
                {"id": "rsi", "type": "indicator", "params": {"type": "RSI", "period": 14}, "position": {"x": 250, "y": 100}},
                {"id": "threshold", "type": "data", "params": {"value": 30}, "position": {"x": 250, "y": 200}},
                {"id": "compare", "type": "condition", "params": {"operator": "<"}, "position": {"x": 450, "y": 150}},
                {"id": "buy", "type": "action", "params": {"type": "buy", "position_size": 0.5}, "position": {"x": 650, "y": 150}}
            ],
            "edges": [
                {"source": "data1", "target": "rsi"},
                {"source": "rsi", "target": "compare"},
                {"source": "threshold", "target": "compare", "targetHandle": "input2"},
                {"source": "compare", "target": "buy"}
            ]
        },
        "performance_metrics": {
            "annual_return": 0.12,
            "sharpe_ratio": 0.9,
            "max_drawdown": -0.12,
            "win_rate": 0.60
        }
    },
    {
        "id": "bollinger_breakout",
        "name": "布林带突破",
        "description": "价格突破布林带上轨时买入，跟随趋势",
        "category": "突破策略",
        "difficulty": "intermediate",
        "strategy_data": {
            "nodes": [
                {"id": "data1", "type": "data", "params": {"field": "close"}, "position": {"x": 50, "y": 150}},
                {"id": "boll", "type": "indicator", "params": {"type": "BOLL", "period": 20, "std": 2}, "position": {"x": 250, "y": 150}},
                {"id": "upper_band", "type": "data", "params": {"field": "upper"}, "position": {"x": 450, "y": 100}},
                {"id": "compare", "type": "condition", "params": {"operator": ">"}, "position": {"x": 650, "y": 150}},
                {"id": "buy", "type": "action", "params": {"type": "buy", "position_size": 0.8}, "position": {"x": 850, "y": 150}}
            ],
            "edges": [
                {"source": "data1", "target": "boll"},
                {"source": "boll", "target": "upper_band"},
                {"source": "data1", "target": "compare"},
                {"source": "upper_band", "target": "compare", "targetHandle": "input2"},
                {"source": "compare", "target": "buy"}
            ]
        },
        "performance_metrics": {
            "annual_return": 0.18,
            "sharpe_ratio": 1.1,
            "max_drawdown": -0.15,
            "win_rate": 0.45
        }
    }
]


@router.post("/save")
async def save_visual_strategy(
    strategy_data: VisualStrategyData,
    current_user: dict = Depends(get_current_user)
):
    """保存可视化策略"""
    try:
        # 创建策略构建器
        builder = VisualStrategyBuilder()
        
        # 重建策略结构
        for node in strategy_data.nodes:
            block_type = BlockType(node.type)
            
            if block_type == BlockType.DATA:
                block = DataBlock(node.id, params=node.params)
            elif block_type == BlockType.INDICATOR:
                block = IndicatorBlock(node.id, node.params.get("type", "SMA"), node.params)
            elif block_type == BlockType.CONDITION:
                block = ConditionBlock(
                    node.id,
                    OperatorType(node.params.get("operator", ">")),
                    node.params
                )
            elif block_type == BlockType.ACTION:
                block = ActionBlock(node.id, node.params.get("type", "buy"), node.params)
            else:
                continue
            
            builder.add_block(block)
        
        # 重建连接
        for edge in strategy_data.edges:
            input_slot = 1 if edge.targetHandle == "input2" else 0
            builder.connect_blocks(edge.source, edge.target, input_slot)
        
        # 验证策略
        if not builder.validate_strategy():
            raise HTTPException(status_code=400, detail="策略验证失败")
        
        # 保存策略
        strategy_json = builder.export_json()
        user_id = current_user["id"]
        
        if user_id not in user_strategies:
            user_strategies[user_id] = []
        
        strategy_record = {
            "id": f"strategy_{datetime.now().timestamp()}",
            "name": strategy_data.name,
            "symbols": strategy_data.symbols,
            "created_at": datetime.now().isoformat(),
            "strategy_json": strategy_json,
            "nodes": [n.dict() for n in strategy_data.nodes],
            "edges": [e.dict() for e in strategy_data.edges]
        }
        
        user_strategies[user_id].append(strategy_record)
        
        return {
            "status": "success",
            "strategy_id": strategy_record["id"],
            "message": "策略保存成功"
        }
        
    except Exception as e:
        logger.error(f"保存策略失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"保存策略失败: {str(e)}")


@router.get("/list")
async def list_user_strategies(
    current_user: dict = Depends(get_current_user)
):
    """获取用户的策略列表"""
    user_id = current_user["id"]
    strategies = user_strategies.get(user_id, [])
    
    return {
        "strategies": [
            {
                "id": s["id"],
                "name": s["name"],
                "symbols": s["symbols"],
                "created_at": s["created_at"],
                "node_count": len(s["nodes"]),
                "edge_count": len(s["edges"])
            }
            for s in strategies
        ]
    }


@router.get("/load/{strategy_id}")
async def load_strategy(
    strategy_id: str,
    current_user: dict = Depends(get_current_user)
):
    """加载策略详情"""
    user_id = current_user["id"]
    strategies = user_strategies.get(user_id, [])
    
    strategy = next((s for s in strategies if s["id"] == strategy_id), None)
    if not strategy:
        raise HTTPException(status_code=404, detail="策略不存在")
    
    return {
        "id": strategy["id"],
        "name": strategy["name"],
        "symbols": strategy["symbols"],
        "nodes": strategy["nodes"],
        "edges": strategy["edges"],
        "created_at": strategy["created_at"]
    }


@router.delete("/{strategy_id}")
async def delete_strategy(
    strategy_id: str,
    current_user: dict = Depends(get_current_user)
):
    """删除策略"""
    user_id = current_user["id"]
    
    if user_id in user_strategies:
        user_strategies[user_id] = [
            s for s in user_strategies[user_id] if s["id"] != strategy_id
        ]
    
    return {"status": "success", "message": "策略删除成功"}


@router.get("/templates")
async def get_strategy_templates():
    """获取策略模板列表"""
    return {
        "templates": STRATEGY_TEMPLATES
    }


@router.post("/create-from-template/{template_id}")
async def create_from_template(
    template_id: str,
    name: str,
    symbols: List[str],
    current_user: dict = Depends(get_current_user)
):
    """从模板创建策略"""
    template = next((t for t in STRATEGY_TEMPLATES if t["id"] == template_id), None)
    if not template:
        raise HTTPException(status_code=404, detail="模板不存在")
    
    # 创建策略数据
    strategy_data = VisualStrategyData(
        nodes=[BlockData(**node) for node in template["strategy_data"]["nodes"]],
        edges=[EdgeData(**edge) for edge in template["strategy_data"]["edges"]],
        name=name,
        symbols=symbols
    )
    
    # 保存策略
    return await save_visual_strategy(strategy_data, current_user)


@router.post("/test-run/{strategy_id}")
async def test_run_strategy(
    strategy_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """测试运行策略"""
    user_id = current_user["id"]
    strategies = user_strategies.get(user_id, [])
    
    strategy = next((s for s in strategies if s["id"] == strategy_id), None)
    if not strategy:
        raise HTTPException(status_code=404, detail="策略不存在")
    
    # TODO: 实现策略回测逻辑
    # 这里先返回模拟结果
    return {
        "status": "success",
        "results": {
            "total_return": 0.15,
            "sharpe_ratio": 1.2,
            "max_drawdown": -0.08,
            "win_rate": 0.55,
            "trade_count": 42,
            "message": "测试运行完成（模拟结果）"
        }
    }


@router.get("/indicator-params/{indicator_type}")
async def get_indicator_params(indicator_type: str):
    """获取指标的默认参数"""
    default_params = {
        "SMA": {"period": 20},
        "EMA": {"period": 20},
        "RSI": {"period": 14},
        "MACD": {"fast": 12, "slow": 26, "signal": 9},
        "BOLL": {"period": 20, "std": 2},
        "KDJ": {"n": 9, "m1": 3, "m2": 3},
        "ATR": {"period": 14},
        "VOL": {}
    }
    
    params = default_params.get(indicator_type, {})
    return {
        "indicator_type": indicator_type,
        "default_params": params
    }
# -*- coding: utf-8 -*-
"""
可视化策略构建器 - 支持拖拽式策略创建
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import pandas as pd
import numpy as np
from datetime import datetime
import logging

from myQuant.core.strategy.base_strategy import BaseStrategy
from myQuant.core.models.signals import TradingSignal, SignalType


class BlockType(Enum):
    """策略块类型"""
    DATA = "data"              # 数据块
    INDICATOR = "indicator"    # 指标块
    CONDITION = "condition"    # 条件块
    ACTION = "action"          # 动作块
    LOGIC = "logic"           # 逻辑块


class OperatorType(Enum):
    """操作符类型"""
    GREATER = ">"
    LESS = "<"
    EQUAL = "=="
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="
    NOT_EQUAL = "!="
    CROSS_ABOVE = "cross_above"
    CROSS_BELOW = "cross_below"
    AND = "and"
    OR = "or"
    NOT = "not"


@dataclass
class StrategyBlock:
    """策略块基类"""
    id: str
    type: BlockType
    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    inputs: List[str] = field(default_factory=list)  # 输入连接的块ID
    output_type: str = "value"  # value, signal, condition


@dataclass
class DataBlock(StrategyBlock):
    """数据块 - 获取市场数据"""
    def __init__(self, id: str, data_type: str = "price", params: Dict = None):
        super().__init__(
            id=id,
            type=BlockType.DATA,
            name=f"数据源_{data_type}",
            params=params or {"field": "close", "period": 1}
        )


@dataclass
class IndicatorBlock(StrategyBlock):
    """指标块 - 计算技术指标"""
    def __init__(self, id: str, indicator_type: str, params: Dict = None):
        super().__init__(
            id=id,
            type=BlockType.INDICATOR,
            name=f"指标_{indicator_type}",
            params=params or {}
        )
        self.indicator_type = indicator_type


@dataclass
class ConditionBlock(StrategyBlock):
    """条件块 - 比较和逻辑判断"""
    def __init__(self, id: str, operator: OperatorType, params: Dict = None):
        super().__init__(
            id=id,
            type=BlockType.CONDITION,
            name=f"条件_{operator.value}",
            params=params or {},
            output_type="condition"
        )
        self.operator = operator


@dataclass
class ActionBlock(StrategyBlock):
    """动作块 - 交易动作"""
    def __init__(self, id: str, action_type: str, params: Dict = None):
        super().__init__(
            id=id,
            type=BlockType.ACTION,
            name=f"动作_{action_type}",
            params=params or {"position_size": 1.0},
            output_type="signal"
        )
        self.action_type = action_type  # buy, sell, close


class VisualStrategyBuilder:
    """可视化策略构建器"""
    
    # 支持的指标库
    INDICATORS = {
        "SMA": {"name": "简单移动平均", "params": {"period": 20}},
        "EMA": {"name": "指数移动平均", "params": {"period": 20}},
        "RSI": {"name": "相对强弱指标", "params": {"period": 14}},
        "MACD": {"name": "MACD", "params": {"fast": 12, "slow": 26, "signal": 9}},
        "BOLL": {"name": "布林带", "params": {"period": 20, "std": 2}},
        "KDJ": {"name": "KDJ指标", "params": {"n": 9, "m1": 3, "m2": 3}},
        "ATR": {"name": "平均真实波幅", "params": {"period": 14}},
        "VOL": {"name": "成交量", "params": {}},
    }
    
    def __init__(self):
        self.blocks: Dict[str, StrategyBlock] = {}
        self.connections: List[Dict[str, str]] = []
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def add_block(self, block: StrategyBlock) -> str:
        """添加策略块"""
        self.blocks[block.id] = block
        self.logger.info(f"Added block: {block.id} ({block.type.value})")
        return block.id
    
    def connect_blocks(self, from_id: str, to_id: str, input_slot: int = 0):
        """连接策略块"""
        if from_id not in self.blocks or to_id not in self.blocks:
            raise ValueError("无效的块ID")
        
        # 更新目标块的输入
        to_block = self.blocks[to_id]
        if len(to_block.inputs) <= input_slot:
            to_block.inputs.extend([None] * (input_slot + 1 - len(to_block.inputs)))
        to_block.inputs[input_slot] = from_id
        
        # 记录连接
        self.connections.append({
            "from": from_id,
            "to": to_id,
            "slot": input_slot
        })
        self.logger.info(f"Connected {from_id} -> {to_id} (slot {input_slot})")
    
    def validate_strategy(self) -> bool:
        """验证策略结构的合法性"""
        # 检查是否有动作块
        action_blocks = [b for b in self.blocks.values() if b.type == BlockType.ACTION]
        if not action_blocks:
            self.logger.error("策略必须包含至少一个动作块")
            return False
        
        # 检查动作块是否有条件输入
        for action in action_blocks:
            if not action.inputs:
                self.logger.error(f"动作块 {action.id} 没有条件输入")
                return False
        
        # 检查是否有循环依赖
        if self._has_circular_dependency():
            self.logger.error("策略中存在循环依赖")
            return False
        
        return True
    
    def _has_circular_dependency(self) -> bool:
        """检查是否有循环依赖"""
        visited = set()
        rec_stack = set()
        
        def has_cycle(block_id: str) -> bool:
            visited.add(block_id)
            rec_stack.add(block_id)
            
            block = self.blocks.get(block_id)
            if block:
                for input_id in block.inputs:
                    if input_id and input_id in self.blocks:
                        if input_id not in visited:
                            if has_cycle(input_id):
                                return True
                        elif input_id in rec_stack:
                            return True
            
            rec_stack.remove(block_id)
            return False
        
        for block_id in self.blocks:
            if block_id not in visited:
                if has_cycle(block_id):
                    return True
        
        return False
    
    def compile_strategy(self, name: str, symbols: List[str]) -> 'CompiledVisualStrategy':
        """编译策略为可执行的策略对象"""
        if not self.validate_strategy():
            raise ValueError("策略验证失败")
        
        return CompiledVisualStrategy(
            name=name,
            symbols=symbols,
            blocks=self.blocks,
            connections=self.connections,
            builder=self
        )
    
    def export_json(self) -> str:
        """导出策略为JSON格式"""
        data = {
            "blocks": [
                {
                    "id": b.id,
                    "type": b.type.value,
                    "name": b.name,
                    "params": b.params,
                    "inputs": b.inputs,
                    "output_type": b.output_type
                }
                for b in self.blocks.values()
            ],
            "connections": self.connections
        }
        return json.dumps(data, ensure_ascii=False, indent=2)
    
    def import_json(self, json_str: str):
        """从JSON导入策略"""
        data = json.loads(json_str)
        
        self.blocks.clear()
        self.connections.clear()
        
        # 恢复块
        for block_data in data["blocks"]:
            block_type = BlockType(block_data["type"])
            if block_type == BlockType.DATA:
                block = DataBlock(block_data["id"], params=block_data["params"])
            elif block_type == BlockType.INDICATOR:
                block = IndicatorBlock(
                    block_data["id"],
                    block_data["params"].get("type", "SMA"),
                    block_data["params"]
                )
            elif block_type == BlockType.CONDITION:
                block = ConditionBlock(
                    block_data["id"],
                    OperatorType(block_data["params"].get("operator", ">")),
                    block_data["params"]
                )
            elif block_type == BlockType.ACTION:
                block = ActionBlock(
                    block_data["id"],
                    block_data["params"].get("type", "buy"),
                    block_data["params"]
                )
            else:
                continue
            
            block.inputs = block_data.get("inputs", [])
            self.blocks[block.id] = block
        
        # 恢复连接
        self.connections = data["connections"]


class CompiledVisualStrategy(BaseStrategy):
    """编译后的可视化策略"""
    
    def __init__(self, name: str, symbols: List[str], blocks: Dict, connections: List, builder: VisualStrategyBuilder):
        super().__init__(name, symbols)
        self.blocks = blocks
        self.connections = connections
        self.builder = builder
        self.block_outputs: Dict[str, Any] = {}
        self.indicator_cache: Dict[str, pd.Series] = {}
    
    def initialize(self, context: Any = None) -> None:
        """初始化策略"""
        super().initialize(context)
        self.logger.info(f"Visual strategy '{self.name}' initialized with {len(self.blocks)} blocks")
    
    def on_bar(self, symbol: str, data: pd.DataFrame) -> List[TradingSignal]:
        """处理K线数据"""
        if len(data) < 2:
            return []
        
        # 清空输出缓存
        self.block_outputs.clear()
        
        # 按拓扑顺序执行块
        execution_order = self._get_execution_order()
        signals = []
        
        for block_id in execution_order:
            block = self.blocks[block_id]
            output = self._execute_block(block, symbol, data)
            self.block_outputs[block_id] = output
            
            # 如果是动作块且条件满足，生成信号
            if block.type == BlockType.ACTION and output:
                signal = self._create_signal_from_action(block, symbol, data)
                if signal:
                    signals.append(signal)
        
        return signals
    
    def _get_execution_order(self) -> List[str]:
        """获取块的执行顺序（拓扑排序）"""
        in_degree = {block_id: 0 for block_id in self.blocks}
        
        # 计算入度
        for block in self.blocks.values():
            for input_id in block.inputs:
                if input_id and input_id in in_degree:
                    in_degree[block.id] += 1
        
        # 拓扑排序
        queue = [block_id for block_id, degree in in_degree.items() if degree == 0]
        order = []
        
        while queue:
            block_id = queue.pop(0)
            order.append(block_id)
            
            # 更新依赖此块的其他块的入度
            for other_id, other_block in self.blocks.items():
                if block_id in other_block.inputs:
                    in_degree[other_id] -= 1
                    if in_degree[other_id] == 0:
                        queue.append(other_id)
        
        return order
    
    def _execute_block(self, block: StrategyBlock, symbol: str, data: pd.DataFrame) -> Any:
        """执行单个块"""
        if block.type == BlockType.DATA:
            return self._execute_data_block(block, data)
        elif block.type == BlockType.INDICATOR:
            return self._execute_indicator_block(block, symbol, data)
        elif block.type == BlockType.CONDITION:
            return self._execute_condition_block(block, data)
        elif block.type == BlockType.ACTION:
            return self._execute_action_block(block, data)
        
        return None
    
    def _execute_data_block(self, block: DataBlock, data: pd.DataFrame) -> pd.Series:
        """执行数据块"""
        field = block.params.get("field", "close")
        period = block.params.get("period", 1)
        
        if field in data.columns:
            return data[field].iloc[-period:]
        else:
            self.logger.warning(f"Data field {field} not found")
            return pd.Series()
    
    def _execute_indicator_block(self, block: IndicatorBlock, symbol: str, data: pd.DataFrame) -> pd.Series:
        """执行指标块"""
        indicator_type = block.params.get("type", "SMA")
        
        # 获取输入数据
        if block.inputs and block.inputs[0]:
            input_data = self.block_outputs.get(block.inputs[0])
            if input_data is None or len(input_data) == 0:
                input_data = data["close"]
        else:
            input_data = data["close"]
        
        # 计算指标
        cache_key = f"{symbol}_{block.id}_{indicator_type}"
        
        if indicator_type == "SMA":
            period = block.params.get("period", 20)
            result = input_data.rolling(window=period).mean()
        elif indicator_type == "EMA":
            period = block.params.get("period", 20)
            result = input_data.ewm(span=period, adjust=False).mean()
        elif indicator_type == "RSI":
            period = block.params.get("period", 14)
            result = self._calculate_rsi(input_data, period)
        elif indicator_type == "VOL":
            result = data["volume"] if "volume" in data.columns else pd.Series()
        else:
            result = input_data
        
        self.indicator_cache[cache_key] = result
        return result
    
    def _calculate_rsi(self, data: pd.Series, period: int) -> pd.Series:
        """计算RSI指标"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _execute_condition_block(self, block: ConditionBlock, data: pd.DataFrame) -> bool:
        """执行条件块"""
        operator = block.params.get("operator", ">")
        
        # 获取输入值
        if len(block.inputs) < 2:
            return False
        
        input1 = self._get_block_output_value(block.inputs[0], data)
        input2 = self._get_block_output_value(block.inputs[1], data)
        
        if input1 is None or input2 is None:
            return False
        
        # 执行比较
        if operator == ">":
            return input1 > input2
        elif operator == "<":
            return input1 < input2
        elif operator == ">=":
            return input1 >= input2
        elif operator == "<=":
            return input1 <= input2
        elif operator == "==":
            return input1 == input2
        elif operator == "!=":
            return input1 != input2
        elif operator == "cross_above":
            return self._check_cross_above(block.inputs[0], block.inputs[1])
        elif operator == "cross_below":
            return self._check_cross_below(block.inputs[0], block.inputs[1])
        
        return False
    
    def _get_block_output_value(self, block_id: str, data: pd.DataFrame) -> Optional[float]:
        """获取块的输出值"""
        output = self.block_outputs.get(block_id)
        
        if output is None:
            return None
        
        if isinstance(output, pd.Series):
            return output.iloc[-1] if len(output) > 0 else None
        elif isinstance(output, (int, float)):
            return float(output)
        elif isinstance(output, bool):
            return 1.0 if output else 0.0
        
        return None
    
    def _check_cross_above(self, series1_id: str, series2_id: str) -> bool:
        """检查上穿"""
        s1 = self.block_outputs.get(series1_id)
        s2 = self.block_outputs.get(series2_id)
        
        if not isinstance(s1, pd.Series) or not isinstance(s2, pd.Series):
            return False
        
        if len(s1) < 2 or len(s2) < 2:
            return False
        
        return s1.iloc[-2] <= s2.iloc[-2] and s1.iloc[-1] > s2.iloc[-1]
    
    def _check_cross_below(self, series1_id: str, series2_id: str) -> bool:
        """检查下穿"""
        s1 = self.block_outputs.get(series1_id)
        s2 = self.block_outputs.get(series2_id)
        
        if not isinstance(s1, pd.Series) or not isinstance(s2, pd.Series):
            return False
        
        if len(s1) < 2 or len(s2) < 2:
            return False
        
        return s1.iloc[-2] >= s2.iloc[-2] and s1.iloc[-1] < s2.iloc[-1]
    
    def _execute_action_block(self, block: ActionBlock, data: pd.DataFrame) -> bool:
        """执行动作块"""
        # 检查条件输入
        if not block.inputs or not block.inputs[0]:
            return False
        
        condition = self.block_outputs.get(block.inputs[0])
        return bool(condition)
    
    def _create_signal_from_action(self, block: ActionBlock, symbol: str, data: pd.DataFrame) -> Optional[TradingSignal]:
        """从动作块创建交易信号"""
        action_type = block.params.get("type", "buy")
        position_size = block.params.get("position_size", 1.0)
        
        current_price = data["close"].iloc[-1]
        current_time = data.index[-1]
        
        if action_type == "buy":
            signal_type = SignalType.BUY
        elif action_type == "sell":
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.HOLD
        
        return TradingSignal(
            symbol=symbol,
            timestamp=current_time,
            signal_type=signal_type,
            strength=position_size,
            price=current_price,
            reason=f"Visual strategy action: {action_type}",
            metadata={"block_id": block.id, "block_name": block.name}
        )
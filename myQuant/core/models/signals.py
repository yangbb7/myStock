# -*- coding: utf-8 -*-
"""
Signals - 交易信号相关数据模型
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class SignalType(Enum):
    """信号类型"""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    STRONG_BUY = "STRONG_BUY"
    STRONG_SELL = "STRONG_SELL"


class SignalStrength(Enum):
    """信号强度"""

    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4


class SignalSource(Enum):
    """信号来源"""

    TECHNICAL = "TECHNICAL"  # 技术分析
    FUNDAMENTAL = "FUNDAMENTAL"  # 基本面分析
    QUANTITATIVE = "QUANTITATIVE"  # 量化分析
    NEWS = "NEWS"  # 新闻事件
    SENTIMENT = "SENTIMENT"  # 情绪分析
    COMBINED = "COMBINED"  # 综合信号


class TradingSignal:
    """交易信号基础类"""

    def __init__(
        self,
        symbol: str,
        signal_type: SignalType,
        strength: SignalStrength = SignalStrength.MODERATE,
        source: SignalSource = SignalSource.TECHNICAL,
        strategy_name: str = "",
    ):

        self.signal_id = str(uuid.uuid4())
        self.symbol = symbol
        self.signal_type = signal_type
        self.strength = strength
        self.source = source
        self.strategy_name = strategy_name

        # 价格信息
        self.price = None
        self.target_price = None
        self.stop_loss_price = None
        self.take_profit_price = None

        # 数量信息
        self.quantity = None
        self.quantity_ratio = None  # 相对于持仓的比例

        # 时间信息
        self.timestamp = datetime.now()
        self.valid_until = None
        self.delay_until = None  # 延迟执行时间

        # 置信度和概率
        self.confidence = 0.0  # 0-1之间
        self.success_probability = 0.0

        # 预期收益和风险
        self.expected_return = 0.0
        self.expected_risk = 0.0
        self.risk_reward_ratio = 0.0

        # 技术指标信息
        self.technical_indicators: Dict[str, float] = {}

        # 基本面信息
        self.fundamental_data: Dict[str, Any] = {}

        # 附加信息
        self.metadata: Dict[str, Any] = {}
        self.description = ""
        self.reasoning = ""

        # 执行状态
        self.is_executed = False
        self.execution_time = None
        self.execution_price = None

    def set_price_targets(
        self,
        current_price: float,
        target_price: float = None,
        stop_loss_price: float = None,
        take_profit_price: float = None,
    ):
        """设置价格目标"""
        self.price = current_price
        self.target_price = target_price
        self.stop_loss_price = stop_loss_price
        self.take_profit_price = take_profit_price

        # 计算风险收益比
        if stop_loss_price and take_profit_price:
            risk = abs(current_price - stop_loss_price)
            reward = abs(take_profit_price - current_price)
            if risk > 0:
                self.risk_reward_ratio = reward / risk

    def set_quantity(self, quantity: int = None, quantity_ratio: float = None):
        """设置交易数量"""
        self.quantity = quantity
        self.quantity_ratio = quantity_ratio

    def add_technical_indicator(self, name: str, value: float):
        """添加技术指标"""
        self.technical_indicators[name] = value

    def add_fundamental_data(self, key: str, value: Any):
        """添加基本面数据"""
        self.fundamental_data[key] = value

    def is_valid(self) -> bool:
        """检查信号是否有效"""
        if self.valid_until:
            return datetime.now() <= self.valid_until
        return True

    def is_ready_for_execution(self) -> bool:
        """检查是否可以执行"""
        if self.delay_until:
            return datetime.now() >= self.delay_until
        return True

    def mark_executed(self, execution_price: float = None):
        """标记为已执行"""
        self.is_executed = True
        self.execution_time = datetime.now()
        self.execution_price = execution_price

    def calculate_score(self) -> float:
        """计算信号综合评分"""
        # 基础评分基于强度
        base_score = self.strength.value * 25  # 最高100分

        # 置信度调整
        confidence_adjustment = self.confidence * 20

        # 风险收益比调整
        risk_reward_adjustment = (
            min(self.risk_reward_ratio * 10, 20) if self.risk_reward_ratio > 0 else 0
        )

        # 成功概率调整
        probability_adjustment = self.success_probability * 15

        total_score = (
            base_score
            + confidence_adjustment
            + risk_reward_adjustment
            + probability_adjustment
        )
        return min(total_score, 100.0)  # 最高100分

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "signal_type": self.signal_type.value,
            "strength": self.strength.value,
            "source": self.source.value,
            "strategy_name": self.strategy_name,
            "price": self.price,
            "target_price": self.target_price,
            "stop_loss_price": self.stop_loss_price,
            "take_profit_price": self.take_profit_price,
            "quantity": self.quantity,
            "quantity_ratio": self.quantity_ratio,
            "timestamp": self.timestamp.isoformat(),
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
            "delay_until": self.delay_until.isoformat() if self.delay_until else None,
            "confidence": self.confidence,
            "success_probability": self.success_probability,
            "expected_return": self.expected_return,
            "expected_risk": self.expected_risk,
            "risk_reward_ratio": self.risk_reward_ratio,
            "technical_indicators": self.technical_indicators,
            "fundamental_data": self.fundamental_data,
            "metadata": self.metadata,
            "description": self.description,
            "reasoning": self.reasoning,
            "is_executed": self.is_executed,
            "execution_time": (
                self.execution_time.isoformat() if self.execution_time else None
            ),
            "execution_price": self.execution_price,
            "score": self.calculate_score(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TradingSignal":
        """从字典创建信号对象"""
        signal = cls(
            symbol=data["symbol"],
            signal_type=SignalType(data["signal_type"]),
            strength=SignalStrength(data.get("strength", 2)),
            source=SignalSource(data.get("source", "TECHNICAL")),
            strategy_name=data.get("strategy_name", ""),
        )

        # 恢复基本信息
        signal.signal_id = data.get("signal_id", signal.signal_id)
        signal.price = data.get("price")
        signal.target_price = data.get("target_price")
        signal.stop_loss_price = data.get("stop_loss_price")
        signal.take_profit_price = data.get("take_profit_price")
        signal.quantity = data.get("quantity")
        signal.quantity_ratio = data.get("quantity_ratio")

        # 恢复时间信息
        if data.get("timestamp"):
            signal.timestamp = datetime.fromisoformat(data["timestamp"])
        if data.get("valid_until"):
            signal.valid_until = datetime.fromisoformat(data["valid_until"])
        if data.get("delay_until"):
            signal.delay_until = datetime.fromisoformat(data["delay_until"])

        # 恢复数值信息
        signal.confidence = data.get("confidence", 0.0)
        signal.success_probability = data.get("success_probability", 0.0)
        signal.expected_return = data.get("expected_return", 0.0)
        signal.expected_risk = data.get("expected_risk", 0.0)
        signal.risk_reward_ratio = data.get("risk_reward_ratio", 0.0)

        # 恢复复合信息
        signal.technical_indicators = data.get("technical_indicators", {})
        signal.fundamental_data = data.get("fundamental_data", {})
        signal.metadata = data.get("metadata", {})
        signal.description = data.get("description", "")
        signal.reasoning = data.get("reasoning", "")

        # 恢复执行状态
        signal.is_executed = data.get("is_executed", False)
        if data.get("execution_time"):
            signal.execution_time = datetime.fromisoformat(data["execution_time"])
        signal.execution_price = data.get("execution_price")

        return signal


class SignalCollection:
    """信号集合管理类"""

    def __init__(self):
        self.signals: Dict[str, TradingSignal] = {}
        self.signals_by_symbol: Dict[str, List[str]] = {}
        self.signals_by_strategy: Dict[str, List[str]] = {}
        self.signals_by_type: Dict[SignalType, List[str]] = {}

        self.created_time = datetime.now()
        self.last_update_time = datetime.now()

    def add_signal(self, signal: TradingSignal):
        """添加信号"""
        self.signals[signal.signal_id] = signal

        # 更新索引
        if signal.symbol not in self.signals_by_symbol:
            self.signals_by_symbol[signal.symbol] = []
        self.signals_by_symbol[signal.symbol].append(signal.signal_id)

        if signal.strategy_name:
            if signal.strategy_name not in self.signals_by_strategy:
                self.signals_by_strategy[signal.strategy_name] = []
            self.signals_by_strategy[signal.strategy_name].append(signal.signal_id)

        if signal.signal_type not in self.signals_by_type:
            self.signals_by_type[signal.signal_type] = []
        self.signals_by_type[signal.signal_type].append(signal.signal_id)

        self.last_update_time = datetime.now()

    def get_signal(self, signal_id: str) -> Optional[TradingSignal]:
        """获取信号"""
        return self.signals.get(signal_id)

    def get_signals_by_symbol(self, symbol: str) -> List[TradingSignal]:
        """获取指定股票的信号"""
        signal_ids = self.signals_by_symbol.get(symbol, [])
        return [self.signals[sid] for sid in signal_ids if sid in self.signals]

    def get_signals_by_strategy(self, strategy_name: str) -> List[TradingSignal]:
        """获取指定策略的信号"""
        signal_ids = self.signals_by_strategy.get(strategy_name, [])
        return [self.signals[sid] for sid in signal_ids if sid in self.signals]

    def get_signals_by_type(self, signal_type: SignalType) -> List[TradingSignal]:
        """获取指定类型的信号"""
        signal_ids = self.signals_by_type.get(signal_type, [])
        return [self.signals[sid] for sid in signal_ids if sid in self.signals]

    def get_active_signals(self) -> List[TradingSignal]:
        """获取有效且未执行的信号"""
        return [
            signal
            for signal in self.signals.values()
            if signal.is_valid() and not signal.is_executed
        ]

    def get_executable_signals(self) -> List[TradingSignal]:
        """获取可执行的信号"""
        return [
            signal
            for signal in self.get_active_signals()
            if signal.is_ready_for_execution()
        ]

    def remove_signal(self, signal_id: str):
        """移除信号"""
        if signal_id in self.signals:
            signal = self.signals[signal_id]

            # 更新索引
            if signal.symbol in self.signals_by_symbol:
                if signal_id in self.signals_by_symbol[signal.symbol]:
                    self.signals_by_symbol[signal.symbol].remove(signal_id)

            if (
                signal.strategy_name
                and signal.strategy_name in self.signals_by_strategy
            ):
                if signal_id in self.signals_by_strategy[signal.strategy_name]:
                    self.signals_by_strategy[signal.strategy_name].remove(signal_id)

            if signal.signal_type in self.signals_by_type:
                if signal_id in self.signals_by_type[signal.signal_type]:
                    self.signals_by_type[signal.signal_type].remove(signal_id)

            # 删除信号
            del self.signals[signal_id]
            self.last_update_time = datetime.now()

    def cleanup_expired_signals(self):
        """清理过期信号"""
        expired_signals = [
            sid for sid, signal in self.signals.items() if not signal.is_valid()
        ]

        for signal_id in expired_signals:
            self.remove_signal(signal_id)

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_signals = len(self.signals)
        active_signals = len(self.get_active_signals())
        executed_signals = len([s for s in self.signals.values() if s.is_executed])

        # 按类型统计
        type_stats = {}
        for signal_type in SignalType:
            count = len(self.get_signals_by_type(signal_type))
            type_stats[signal_type.value] = count

        # 按策略统计
        strategy_stats = {}
        for strategy_name, signal_ids in self.signals_by_strategy.items():
            strategy_stats[strategy_name] = len(signal_ids)

        return {
            "total_signals": total_signals,
            "active_signals": active_signals,
            "executed_signals": executed_signals,
            "expired_signals": total_signals - active_signals - executed_signals,
            "execution_rate": executed_signals / max(total_signals, 1),
            "by_type": type_stats,
            "by_strategy": strategy_stats,
            "symbols_count": len(self.signals_by_symbol),
            "strategies_count": len(self.signals_by_strategy),
        }

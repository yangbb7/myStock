"""
投资组合模型

包含Portfolio、PortfolioSummary、PerformanceMetrics等投资组合相关数据结构
"""

import re
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional
from dataclasses import dataclass

from .positions import Position, PositionStatus


@dataclass
class PortfolioSummary:
    """投资组合摘要"""
    
    total_value: Decimal
    cash_balance: Decimal
    position_value: Decimal
    total_pnl: Decimal
    total_return: Decimal
    positions_count: int
    last_updated: datetime


@dataclass
class PerformanceMetrics:
    """绩效指标"""
    
    total_return: Decimal
    annualized_return: Optional[Decimal] = None
    volatility: Optional[Decimal] = None
    sharpe_ratio: Optional[Decimal] = None
    max_drawdown: Optional[Decimal] = None
    var_95: Optional[Decimal] = None
    beta: Optional[Decimal] = None


class Portfolio:
    """投资组合模型"""
    
    def __init__(
        self,
        user_id: int = 1,  # 默认用户ID
        initial_capital: Decimal = Decimal('1000000'),
        commission_rate: Optional[float] = 0.0003,  # 手续费率
        min_commission: Optional[float] = 5.0  # 最低手续费
    ):
        self.user_id = user_id
        self.initial_capital = initial_capital
        self.cash_balance = initial_capital
        self.cash = initial_capital  # 别名，兼容旧代码
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.positions: Dict[str, Position] = {}
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
        self.total_value = initial_capital  # 总价值
    
    def add_position(self, position: Position) -> bool:
        """添加持仓
        
        Args:
            position: 持仓对象
            
        Returns:
            bool: 添加是否成功
        """
        if position.user_id != self.user_id:
            return False
        
        self.positions[position.symbol] = position
        self.updated_at = datetime.now(timezone.utc)
        return True
    
    def remove_position(self, symbol: str) -> bool:
        """移除持仓
        
        Args:
            symbol: 股票代码
            
        Returns:
            bool: 移除是否成功
        """
        if symbol in self.positions:
            del self.positions[symbol]
            self.updated_at = datetime.now(timezone.utc)
            return True
        return False
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """获取持仓
        
        Args:
            symbol: 股票代码
            
        Returns:
            Optional[Position]: 持仓对象
        """
        return self.positions.get(symbol)
    
    def calculate_total_value(self, current_prices: Dict[str, Decimal]) -> Decimal:
        """计算总价值
        
        Args:
            current_prices: 当前价格字典
            
        Returns:
            Decimal: 总价值
        """
        position_value = Decimal('0')
        
        for symbol, position in self.positions.items():
            if position.status == PositionStatus.ACTIVE and symbol in current_prices:
                position_value += position.calculate_market_value(current_prices[symbol])
        
        return self.cash_balance + position_value
    
    def get_total_value(self) -> Decimal:
        """获取总价值（不需要价格参数的简化版本）
        
        Returns:
            Decimal: 总价值
        """
        # 对于没有当前价格的情况，使用平均价格估算
        position_value = Decimal('0')
        
        for symbol, position in self.positions.items():
            if position.status == PositionStatus.ACTIVE:
                position_value += position.calculate_market_value(position.average_price)
        
        return self.cash_balance + position_value
    
    def get_cash_balance(self) -> Decimal:
        """获取现金余额
        
        Returns:
            Decimal: 现金余额
        """
        return self.cash_balance
    
    def get_total_exposure(self) -> Decimal:
        """获取总敞口
        
        Returns:
            Decimal: 总敞口（持仓价值占总价值的比例）
        """
        total_value = self.get_total_value()
        if total_value <= 0:
            return Decimal('0')
        
        position_value = Decimal('0')
        for symbol, position in self.positions.items():
            if position.status == PositionStatus.ACTIVE:
                position_value += position.calculate_market_value(position.average_price)
        
        return position_value / total_value
    
    def calculate_position_value(self, current_prices: Dict[str, Decimal]) -> Decimal:
        """计算持仓价值
        
        Args:
            current_prices: 当前价格字典
            
        Returns:
            Decimal: 持仓价值
        """
        position_value = Decimal('0')
        
        for symbol, position in self.positions.items():
            if position.status == PositionStatus.ACTIVE and symbol in current_prices:
                position_value += position.calculate_market_value(current_prices[symbol])
        
        return position_value
    
    def calculate_total_pnl(self, current_prices: Dict[str, Decimal]) -> Decimal:
        """计算总盈亏
        
        Args:
            current_prices: 当前价格字典
            
        Returns:
            Decimal: 总盈亏
        """
        total_pnl = Decimal('0')
        
        for symbol, position in self.positions.items():
            if position.status == PositionStatus.ACTIVE and symbol in current_prices:
                total_pnl += position.calculate_unrealized_pnl(current_prices[symbol])
            # 加上已实现盈亏
            total_pnl += position.realized_pnl
        
        return total_pnl
    
    def update_cash_balance(self, amount: Decimal) -> bool:
        """更新现金余额
        
        Args:
            amount: 变动金额（正数为增加，负数为减少）
            
        Returns:
            bool: 更新是否成功
        """
        new_balance = self.cash_balance + amount
        
        # 检查余额是否足够
        if new_balance < 0:
            return False
        
        self.cash_balance = new_balance
        self.updated_at = datetime.now(timezone.utc)
        return True
    
    def get_summary(self, current_prices: Dict[str, Decimal]) -> PortfolioSummary:
        """获取投资组合摘要
        
        Args:
            current_prices: 当前价格字典
            
        Returns:
            PortfolioSummary: 投资组合摘要
        """
        total_value = self.calculate_total_value(current_prices)
        position_value = self.calculate_position_value(current_prices)
        total_pnl = self.calculate_total_pnl(current_prices)
        
        # 计算总收益率
        if self.initial_capital > 0:
            total_return = (total_pnl / self.initial_capital * 100).quantize(Decimal('0.01'))
        else:
            total_return = Decimal('0')
        
        # 计算活跃持仓数量
        active_positions = sum(
            1 for pos in self.positions.values() 
            if pos.status == PositionStatus.ACTIVE and pos.quantity > 0
        )
        
        return PortfolioSummary(
            total_value=total_value,
            cash_balance=self.cash_balance,
            position_value=position_value,
            total_pnl=total_pnl,
            total_return=total_return,
            positions_count=active_positions,
            last_updated=self.updated_at
        )
    
    def calculate_performance_metrics(
        self, 
        current_prices: Dict[str, Decimal],
        historical_values: List[Decimal]
    ) -> PerformanceMetrics:
        """计算绩效指标
        
        Args:
            current_prices: 当前价格字典
            historical_values: 历史价值数据
            
        Returns:
            PerformanceMetrics: 绩效指标
        """
        total_pnl = self.calculate_total_pnl(current_prices)
        total_return = total_pnl / self.initial_capital * 100 if self.initial_capital > 0 else Decimal('0')
        
        # 计算波动率
        volatility = None
        if len(historical_values) > 1:
            returns = []
            for i in range(1, len(historical_values)):
                if historical_values[i-1] > 0:
                    daily_return = (historical_values[i] - historical_values[i-1]) / historical_values[i-1]
                    returns.append(daily_return)
            
            if returns:
                mean_return = sum(returns) / len(returns)
                variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
                volatility = (variance ** Decimal('0.5') * Decimal('252') ** Decimal('0.5') * 100).quantize(Decimal('0.01'))
        
        # 计算最大回撤
        max_drawdown = None
        if len(historical_values) > 1:
            peak = historical_values[0]
            max_dd = Decimal('0')
            
            for value in historical_values[1:]:
                if value > peak:
                    peak = value
                else:
                    drawdown = (peak - value) / peak * 100 if peak > 0 else Decimal('0')
                    if drawdown > max_dd:
                        max_dd = drawdown
            
            max_drawdown = -max_dd.quantize(Decimal('0.01'))
        
        # 简化的夏普比率计算（假设无风险利率为0）
        sharpe_ratio = None
        if volatility and volatility > 0:
            annualized_return = total_return if len(historical_values) <= 252 else total_return * Decimal('252') / len(historical_values)
            sharpe_ratio = (annualized_return / volatility).quantize(Decimal('0.01'))
        
        return PerformanceMetrics(
            total_return=total_return.quantize(Decimal('0.01')),
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown
        )
    
    def get_active_positions(self) -> Dict[str, Position]:
        """获取活跃持仓
        
        Returns:
            Dict[str, Position]: 活跃持仓字典
        """
        return {
            symbol: position 
            for symbol, position in self.positions.items() 
            if position.status == PositionStatus.ACTIVE and position.quantity > 0
        }
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'user_id': self.user_id,
            'initial_capital': float(self.initial_capital),
            'cash_balance': float(self.cash_balance),
            'positions': {symbol: pos.to_dict() for symbol, pos in self.positions.items()},
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
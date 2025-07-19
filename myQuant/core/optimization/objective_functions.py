# -*- coding: utf-8 -*-
"""
目标函数 - 定义常用的策略优化目标函数
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
    """性能指标"""
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_return: float = 0.0
    total_trades: int = 0
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典"""
        return {
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'calmar_ratio': self.calmar_ratio,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'avg_trade_return': self.avg_trade_return,
            'total_trades': self.total_trades
        }


class BaseObjectiveFunction(ABC):
    """目标函数基类"""
    
    def __init__(self, 
                 strategy_class: type,
                 data: pd.DataFrame,
                 benchmark_return: Optional[float] = None,
                 risk_free_rate: float = 0.02):
        """
        初始化目标函数
        
        Args:
            strategy_class: 策略类
            data: 历史数据
            benchmark_return: 基准收益率
            risk_free_rate: 无风险利率
        """
        self.strategy_class = strategy_class
        self.data = data
        self.benchmark_return = benchmark_return or 0.0
        self.risk_free_rate = risk_free_rate
        
        # 缓存
        self.performance_cache: Dict[str, PerformanceMetrics] = {}
    
    @abstractmethod
    def __call__(self, params: Dict[str, Any]) -> float:
        """
        计算目标函数值
        
        Args:
            params: 参数字典
            
        Returns:
            float: 目标函数值
        """
        pass
    
    def calculate_performance_metrics(self, params: Dict[str, Any]) -> PerformanceMetrics:
        """
        计算性能指标
        
        Args:
            params: 策略参数
            
        Returns:
            PerformanceMetrics: 性能指标
        """
        # 生成缓存键
        cache_key = self._generate_cache_key(params)
        
        if cache_key in self.performance_cache:
            return self.performance_cache[cache_key]
        
        try:
            # 创建策略实例
            strategy = self.strategy_class(
                name=f"optimization_strategy",
                symbols=self._get_symbols_from_data(),
                params=params
            )
            
            # 运行回测
            returns = self._run_backtest(strategy)
            
            # 计算性能指标
            metrics = self._calculate_metrics(returns)
            
            # 缓存结果
            self.performance_cache[cache_key] = metrics
            
            return metrics
            
        except Exception as e:
            # 返回最差性能指标
            return PerformanceMetrics(
                total_return=-1.0,
                sharpe_ratio=-10.0,
                max_drawdown=1.0,
                profit_factor=0.0
            )
    
    def _generate_cache_key(self, params: Dict[str, Any]) -> str:
        """生成缓存键"""
        sorted_items = sorted(params.items())
        return str(hash(tuple(sorted_items)))
    
    def _get_symbols_from_data(self) -> List[str]:
        """从数据中获取交易标的"""
        if 'symbol' in self.data.columns:
            return self.data['symbol'].unique().tolist()
        else:
            return ['default_symbol']
    
    def _run_backtest(self, strategy) -> pd.Series:
        """
        运行回测
        
        Args:
            strategy: 策略实例
            
        Returns:
            pd.Series: 收益率序列
        """
        # 这里应该调用实际的回测引擎
        # 为了演示，我们生成一个简单的模拟收益率序列
        
        # 模拟策略执行
        positions = []
        current_position = 0
        
        for i, row in self.data.iterrows():
            # 模拟信号生成（这里应该调用真实的策略逻辑）
            price = row.get('close', row.get('Close', 100))
            
            # 简单的移动平均策略示例
            if i >= 20:  # 确保有足够的历史数据
                ma_short = self.data['close'].iloc[i-5:i+1].mean() if 'close' in self.data.columns else 100
                ma_long = self.data['close'].iloc[i-20:i+1].mean() if 'close' in self.data.columns else 100
                
                if ma_short > ma_long and current_position <= 0:
                    current_position = 1  # 买入
                elif ma_short < ma_long and current_position >= 0:
                    current_position = -1  # 卖出
            
            positions.append(current_position)
        
        # 计算收益率
        if 'close' in self.data.columns:
            price_series = self.data['close']
        elif 'Close' in self.data.columns:
            price_series = self.data['Close']
        else:
            # 生成模拟价格序列
            price_series = pd.Series(100 * np.cumprod(1 + np.random.normal(0.001, 0.02, len(self.data))))
        
        returns = price_series.pct_change().fillna(0)
        
        # 计算策略收益
        strategy_returns = pd.Series(positions[:-1]) * returns.iloc[1:]
        
        return strategy_returns
    
    def _calculate_metrics(self, returns: pd.Series) -> PerformanceMetrics:
        """计算性能指标"""
        if len(returns) == 0 or returns.isna().all():
            return PerformanceMetrics()
        
        # 基础指标
        total_return = (1 + returns).cumprod().iloc[-1] - 1 if len(returns) > 0 else 0
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0
        volatility = returns.std() * np.sqrt(252)
        
        # 风险调整收益率
        excess_returns = returns - self.risk_free_rate / 252
        sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Sortino比率
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else volatility
        sortino_ratio = (annualized_return - self.risk_free_rate) / downside_std if downside_std > 0 else 0
        
        # 最大回撤
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calmar比率
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # 交易统计
        winning_trades = (returns > 0).sum()
        total_trades = (returns != 0).sum()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # 盈利因子
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
        
        # 平均交易收益
        avg_trade_return = returns[returns != 0].mean() if total_trades > 0 else 0
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_trade_return=avg_trade_return,
            total_trades=int(total_trades)
        )


class SharpeRatioObjective(BaseObjectiveFunction):
    """夏普比率目标函数"""
    
    def __call__(self, params: Dict[str, Any]) -> float:
        """计算夏普比率"""
        metrics = self.calculate_performance_metrics(params)
        return metrics.sharpe_ratio


class SortinoRatioObjective(BaseObjectiveFunction):
    """Sortino比率目标函数"""
    
    def __call__(self, params: Dict[str, Any]) -> float:
        """计算Sortino比率"""
        metrics = self.calculate_performance_metrics(params)
        return metrics.sortino_ratio


class CalmarRatioObjective(BaseObjectiveFunction):
    """Calmar比率目标函数"""
    
    def __call__(self, params: Dict[str, Any]) -> float:
        """计算Calmar比率"""
        metrics = self.calculate_performance_metrics(params)
        return metrics.calmar_ratio


class MaxDrawdownObjective(BaseObjectiveFunction):
    """最大回撤目标函数（最小化）"""
    
    def __call__(self, params: Dict[str, Any]) -> float:
        """计算最大回撤的负值（用于最小化）"""
        metrics = self.calculate_performance_metrics(params)
        return -metrics.max_drawdown  # 负值，因为我们要最小化回撤


class ProfitFactorObjective(BaseObjectiveFunction):
    """盈利因子目标函数"""
    
    def __call__(self, params: Dict[str, Any]) -> float:
        """计算盈利因子"""
        metrics = self.calculate_performance_metrics(params)
        return metrics.profit_factor


class TotalReturnObjective(BaseObjectiveFunction):
    """总收益率目标函数"""
    
    def __call__(self, params: Dict[str, Any]) -> float:
        """计算总收益率"""
        metrics = self.calculate_performance_metrics(params)
        return metrics.total_return


class WinRateObjective(BaseObjectiveFunction):
    """胜率目标函数"""
    
    def __call__(self, params: Dict[str, Any]) -> float:
        """计算胜率"""
        metrics = self.calculate_performance_metrics(params)
        return metrics.win_rate


class CompositeObjective(BaseObjectiveFunction):
    """复合目标函数"""
    
    def __init__(self, 
                 strategy_class: type,
                 data: pd.DataFrame,
                 weights: Dict[str, float],
                 **kwargs):
        """
        初始化复合目标函数
        
        Args:
            strategy_class: 策略类
            data: 历史数据
            weights: 指标权重字典
        """
        super().__init__(strategy_class, data, **kwargs)
        self.weights = weights
        
        # 验证权重
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
    
    def __call__(self, params: Dict[str, Any]) -> float:
        """计算加权复合得分"""
        metrics = self.calculate_performance_metrics(params)
        metrics_dict = metrics.to_dict()
        
        composite_score = 0.0
        for metric_name, weight in self.weights.items():
            if metric_name in metrics_dict:
                value = metrics_dict[metric_name]
                
                # 对于需要最小化的指标（如最大回撤），取负值
                if metric_name == 'max_drawdown':
                    value = -value
                elif metric_name == 'volatility':
                    value = -value
                
                composite_score += weight * value
        
        return composite_score


class RiskAdjustedObjective(BaseObjectiveFunction):
    """风险调整目标函数"""
    
    def __init__(self, 
                 strategy_class: type,
                 data: pd.DataFrame,
                 risk_penalty: float = 1.0,
                 **kwargs):
        """
        初始化风险调整目标函数
        
        Args:
            strategy_class: 策略类
            data: 历史数据
            risk_penalty: 风险惩罚系数
        """
        super().__init__(strategy_class, data, **kwargs)
        self.risk_penalty = risk_penalty
    
    def __call__(self, params: Dict[str, Any]) -> float:
        """计算风险调整收益"""
        metrics = self.calculate_performance_metrics(params)
        
        # 风险调整收益 = 收益率 - 风险惩罚 * 最大回撤
        risk_adjusted_return = (
            metrics.annualized_return - 
            self.risk_penalty * abs(metrics.max_drawdown)
        )
        
        return risk_adjusted_return


class MultiObjective(BaseObjectiveFunction):
    """多目标优化函数"""
    
    def __init__(self, 
                 strategy_class: type,
                 data: pd.DataFrame,
                 objectives: List[BaseObjectiveFunction],
                 **kwargs):
        """
        初始化多目标函数
        
        Args:
            strategy_class: 策略类
            data: 历史数据
            objectives: 目标函数列表
        """
        super().__init__(strategy_class, data, **kwargs)
        self.objectives = objectives
    
    def __call__(self, params: Dict[str, Any]) -> float:
        """计算帕累托前沿得分"""
        scores = []
        for objective in self.objectives:
            score = objective(params)
            scores.append(score)
        
        # 简单的加权平均（可以使用更复杂的多目标优化方法）
        return np.mean(scores)
    
    def get_pareto_scores(self, params: Dict[str, Any]) -> List[float]:
        """获取帕累托得分"""
        return [objective(params) for objective in self.objectives]


class ConstrainedObjective(BaseObjectiveFunction):
    """约束目标函数"""
    
    def __init__(self, 
                 strategy_class: type,
                 data: pd.DataFrame,
                 base_objective: BaseObjectiveFunction,
                 constraints: List[Callable[[PerformanceMetrics], bool]],
                 penalty: float = -1000.0,
                 **kwargs):
        """
        初始化约束目标函数
        
        Args:
            strategy_class: 策略类
            data: 历史数据
            base_objective: 基础目标函数
            constraints: 约束条件列表
            penalty: 违反约束的惩罚值
        """
        super().__init__(strategy_class, data, **kwargs)
        self.base_objective = base_objective
        self.constraints = constraints
        self.penalty = penalty
    
    def __call__(self, params: Dict[str, Any]) -> float:
        """计算约束目标函数值"""
        metrics = self.calculate_performance_metrics(params)
        
        # 检查约束条件
        for constraint in self.constraints:
            if not constraint(metrics):
                return self.penalty
        
        # 计算基础目标函数值
        return self.base_objective(params)


# 预定义约束条件
class CommonConstraints:
    """常用约束条件"""
    
    @staticmethod
    def max_drawdown_constraint(max_dd: float) -> Callable[[PerformanceMetrics], bool]:
        """最大回撤约束"""
        def constraint(metrics: PerformanceMetrics) -> bool:
            return abs(metrics.max_drawdown) <= max_dd
        return constraint
    
    @staticmethod
    def min_sharpe_constraint(min_sharpe: float) -> Callable[[PerformanceMetrics], bool]:
        """最小夏普比率约束"""
        def constraint(metrics: PerformanceMetrics) -> bool:
            return metrics.sharpe_ratio >= min_sharpe
        return constraint
    
    @staticmethod
    def min_trades_constraint(min_trades: int) -> Callable[[PerformanceMetrics], bool]:
        """最小交易次数约束"""
        def constraint(metrics: PerformanceMetrics) -> bool:
            return metrics.total_trades >= min_trades
        return constraint
    
    @staticmethod
    def max_volatility_constraint(max_vol: float) -> Callable[[PerformanceMetrics], bool]:
        """最大波动率约束"""
        def constraint(metrics: PerformanceMetrics) -> bool:
            return metrics.volatility <= max_vol
        return constraint
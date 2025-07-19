# -*- coding: utf-8 -*-
"""
策略性能分析 - 提供策略性能评估和分析功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    # 收益指标
    total_return: float = 0.0
    annualized_return: float = 0.0
    cumulative_return: float = 0.0
    
    # 风险指标
    volatility: float = 0.0
    max_drawdown: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    
    # 风险调整收益
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    omega_ratio: float = 0.0
    
    # 交易统计
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    
    # 其他指标
    beta: float = 0.0
    alpha: float = 0.0
    information_ratio: float = 0.0
    tracking_error: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典"""
        return {
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'cumulative_return': self.cumulative_return,
            'volatility': self.volatility,
            'max_drawdown': self.max_drawdown,
            'var_95': self.var_95,
            'cvar_95': self.cvar_95,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'omega_ratio': self.omega_ratio,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'profit_factor': self.profit_factor,
            'beta': self.beta,
            'alpha': self.alpha,
            'information_ratio': self.information_ratio,
            'tracking_error': self.tracking_error
        }


class StrategyPerformance:
    """策略性能分析器"""
    
    def __init__(self,
                 returns: pd.Series,
                 benchmark_returns: Optional[pd.Series] = None,
                 risk_free_rate: float = 0.02,
                 trading_days: int = 252):
        """
        初始化性能分析器
        
        Args:
            returns: 策略收益率序列
            benchmark_returns: 基准收益率序列
            risk_free_rate: 无风险利率（年化）
            trading_days: 每年交易日数
        """
        self.returns = returns.dropna()
        self.benchmark_returns = benchmark_returns.dropna() if benchmark_returns is not None else None
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days
        
        # 对齐数据
        if self.benchmark_returns is not None:
            self.returns, self.benchmark_returns = self._align_series(self.returns, self.benchmark_returns)
        
        # 计算累积收益
        self.cumulative_returns = (1 + self.returns).cumprod()
        
        # 日志
        self.logger = logging.getLogger(__name__)
    
    def _align_series(self, series1: pd.Series, series2: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """对齐两个时间序列"""
        common_index = series1.index.intersection(series2.index)
        return series1.loc[common_index], series2.loc[common_index]
    
    def calculate_metrics(self) -> PerformanceMetrics:
        """计算所有性能指标"""
        try:
            metrics = PerformanceMetrics()
            
            if len(self.returns) == 0:
                return metrics
            
            # 收益指标
            metrics.total_return = self.calculate_total_return()
            metrics.annualized_return = self.calculate_annualized_return()
            metrics.cumulative_return = self.cumulative_returns.iloc[-1] - 1
            
            # 风险指标
            metrics.volatility = self.calculate_volatility()
            metrics.max_drawdown = self.calculate_max_drawdown()
            metrics.var_95 = self.calculate_var(confidence_level=0.05)
            metrics.cvar_95 = self.calculate_cvar(confidence_level=0.05)
            
            # 风险调整收益
            metrics.sharpe_ratio = self.calculate_sharpe_ratio()
            metrics.sortino_ratio = self.calculate_sortino_ratio()
            metrics.calmar_ratio = self.calculate_calmar_ratio()
            metrics.omega_ratio = self.calculate_omega_ratio()
            
            # 交易统计
            trade_stats = self.calculate_trade_statistics()
            metrics.total_trades = trade_stats['total_trades']
            metrics.winning_trades = trade_stats['winning_trades']
            metrics.losing_trades = trade_stats['losing_trades']
            metrics.win_rate = trade_stats['win_rate']
            metrics.avg_win = trade_stats['avg_win']
            metrics.avg_loss = trade_stats['avg_loss']
            metrics.profit_factor = trade_stats['profit_factor']
            
            # 基准相关指标
            if self.benchmark_returns is not None:
                metrics.beta = self.calculate_beta()
                metrics.alpha = self.calculate_alpha()
                metrics.information_ratio = self.calculate_information_ratio()
                metrics.tracking_error = self.calculate_tracking_error()
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            return PerformanceMetrics()
    
    def calculate_total_return(self) -> float:
        """计算总收益率"""
        if len(self.returns) == 0:
            return 0.0
        return (1 + self.returns).prod() - 1
    
    def calculate_annualized_return(self) -> float:
        """计算年化收益率"""
        total_return = self.calculate_total_return()
        if len(self.returns) == 0:
            return 0.0
        years = len(self.returns) / self.trading_days
        return (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0
    
    def calculate_volatility(self) -> float:
        """计算年化波动率"""
        if len(self.returns) == 0:
            return 0.0
        return self.returns.std() * np.sqrt(self.trading_days)
    
    def calculate_max_drawdown(self) -> float:
        """计算最大回撤"""
        if len(self.cumulative_returns) == 0:
            return 0.0
        
        running_max = self.cumulative_returns.expanding().max()
        drawdown = (self.cumulative_returns - running_max) / running_max
        return drawdown.min()
    
    def calculate_drawdown_series(self) -> pd.Series:
        """计算回撤序列"""
        if len(self.cumulative_returns) == 0:
            return pd.Series()
        
        running_max = self.cumulative_returns.expanding().max()
        drawdown = (self.cumulative_returns - running_max) / running_max
        return drawdown
    
    def calculate_var(self, confidence_level: float = 0.05) -> float:
        """计算风险价值(VaR)"""
        if len(self.returns) == 0:
            return 0.0
        return np.percentile(self.returns, confidence_level * 100)
    
    def calculate_cvar(self, confidence_level: float = 0.05) -> float:
        """计算条件风险价值(CVaR)"""
        if len(self.returns) == 0:
            return 0.0
        var = self.calculate_var(confidence_level)
        return self.returns[self.returns <= var].mean()
    
    def calculate_sharpe_ratio(self) -> float:
        """计算夏普比率"""
        if len(self.returns) == 0:
            return 0.0
        
        excess_returns = self.returns - self.risk_free_rate / self.trading_days
        if excess_returns.std() == 0:
            return 0.0
        
        return excess_returns.mean() / excess_returns.std() * np.sqrt(self.trading_days)
    
    def calculate_sortino_ratio(self) -> float:
        """计算索提诺比率"""
        if len(self.returns) == 0:
            return 0.0
        
        excess_returns = self.returns - self.risk_free_rate / self.trading_days
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        
        return excess_returns.mean() / downside_returns.std() * np.sqrt(self.trading_days)
    
    def calculate_calmar_ratio(self) -> float:
        """计算卡尔马比率"""
        annualized_return = self.calculate_annualized_return()
        max_drawdown = abs(self.calculate_max_drawdown())
        
        if max_drawdown == 0:
            return 0.0
        
        return annualized_return / max_drawdown
    
    def calculate_omega_ratio(self, threshold: float = 0.0) -> float:
        """计算欧米茄比率"""
        if len(self.returns) == 0:
            return 0.0
        
        excess_returns = self.returns - threshold
        positive_returns = excess_returns[excess_returns > 0]
        negative_returns = excess_returns[excess_returns < 0]
        
        if len(negative_returns) == 0 or negative_returns.sum() == 0:
            return float('inf') if len(positive_returns) > 0 else 0.0
        
        return positive_returns.sum() / abs(negative_returns.sum())
    
    def calculate_trade_statistics(self) -> Dict[str, Any]:
        """计算交易统计"""
        # 识别交易
        non_zero_returns = self.returns[self.returns != 0]
        
        if len(non_zero_returns) == 0:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0
            }
        
        winning_trades = non_zero_returns[non_zero_returns > 0]
        losing_trades = non_zero_returns[non_zero_returns < 0]
        
        total_trades = len(non_zero_returns)
        num_winning = len(winning_trades)
        num_losing = len(losing_trades)
        
        win_rate = num_winning / total_trades if total_trades > 0 else 0.0
        avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0.0
        avg_loss = losing_trades.mean() if len(losing_trades) > 0 else 0.0
        
        # 盈利因子
        gross_profit = winning_trades.sum() if len(winning_trades) > 0 else 0.0
        gross_loss = abs(losing_trades.sum()) if len(losing_trades) > 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float('inf') if gross_profit > 0 else 0.0)
        
        return {
            'total_trades': total_trades,
            'winning_trades': num_winning,
            'losing_trades': num_losing,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor
        }
    
    def calculate_beta(self) -> float:
        """计算贝塔系数"""
        if self.benchmark_returns is None or len(self.returns) == 0:
            return 0.0
        
        covariance = np.cov(self.returns, self.benchmark_returns)[0, 1]
        benchmark_variance = np.var(self.benchmark_returns)
        
        if benchmark_variance == 0:
            return 0.0
        
        return covariance / benchmark_variance
    
    def calculate_alpha(self) -> float:
        """计算阿尔法系数"""
        if self.benchmark_returns is None:
            return 0.0
        
        strategy_return = self.calculate_annualized_return()
        benchmark_return = (1 + self.benchmark_returns).prod() ** (self.trading_days / len(self.benchmark_returns)) - 1
        beta = self.calculate_beta()
        
        return strategy_return - (self.risk_free_rate + beta * (benchmark_return - self.risk_free_rate))
    
    def calculate_information_ratio(self) -> float:
        """计算信息比率"""
        if self.benchmark_returns is None or len(self.returns) == 0:
            return 0.0
        
        active_returns = self.returns - self.benchmark_returns
        tracking_error = active_returns.std() * np.sqrt(self.trading_days)
        
        if tracking_error == 0:
            return 0.0
        
        return active_returns.mean() * self.trading_days / tracking_error
    
    def calculate_tracking_error(self) -> float:
        """计算跟踪误差"""
        if self.benchmark_returns is None or len(self.returns) == 0:
            return 0.0
        
        active_returns = self.returns - self.benchmark_returns
        return active_returns.std() * np.sqrt(self.trading_days)
    
    def calculate_rolling_metrics(self, window: int = 252) -> pd.DataFrame:
        """计算滚动指标"""
        if len(self.returns) < window:
            return pd.DataFrame()
        
        rolling_data = []
        
        for i in range(window, len(self.returns) + 1):
            window_returns = self.returns.iloc[i-window:i]
            window_cum_returns = (1 + window_returns).cumprod()
            
            # 计算窗口指标
            annualized_return = (window_cum_returns.iloc[-1]) ** (self.trading_days / window) - 1
            volatility = window_returns.std() * np.sqrt(self.trading_days)
            
            # 最大回撤
            window_running_max = window_cum_returns.expanding().max()
            window_drawdown = (window_cum_returns - window_running_max) / window_running_max
            max_drawdown = window_drawdown.min()
            
            # 夏普比率
            excess_returns = window_returns - self.risk_free_rate / self.trading_days
            sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(self.trading_days) if excess_returns.std() > 0 else 0
            
            rolling_data.append({
                'date': self.returns.index[i-1],
                'annualized_return': annualized_return,
                'volatility': volatility,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio
            })
        
        return pd.DataFrame(rolling_data).set_index('date')
    
    def generate_performance_report(self, save_path: Optional[str] = None) -> str:
        """生成性能报告"""
        metrics = self.calculate_metrics()
        
        report = f"""
策略性能分析报告
{'='*50}

收益指标:
  总收益率: {metrics.total_return:.2%}
  年化收益率: {metrics.annualized_return:.2%}
  累积收益率: {metrics.cumulative_return:.2%}

风险指标:
  年化波动率: {metrics.volatility:.2%}
  最大回撤: {metrics.max_drawdown:.2%}
  95% VaR: {metrics.var_95:.2%}
  95% CVaR: {metrics.cvar_95:.2%}

风险调整收益:
  夏普比率: {metrics.sharpe_ratio:.4f}
  索提诺比率: {metrics.sortino_ratio:.4f}
  卡尔马比率: {metrics.calmar_ratio:.4f}
  欧米茄比率: {metrics.omega_ratio:.4f}

交易统计:
  总交易次数: {metrics.total_trades}
  胜率: {metrics.win_rate:.2%}
  平均盈利: {metrics.avg_win:.2%}
  平均亏损: {metrics.avg_loss:.2%}
  盈利因子: {metrics.profit_factor:.4f}
"""
        
        if self.benchmark_returns is not None:
            report += f"""
基准比较:
  贝塔系数: {metrics.beta:.4f}
  阿尔法系数: {metrics.alpha:.4f}
  信息比率: {metrics.information_ratio:.4f}
  跟踪误差: {metrics.tracking_error:.2%}
"""
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
        
        return report
    
    def create_performance_charts(self, save_dir: Optional[str] = None, figsize: Tuple[int, int] = (15, 12)) -> None:
        """创建性能图表"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=figsize)
            
            # 1. 累积收益曲线
            axes[0, 0].plot(self.cumulative_returns.index, self.cumulative_returns.values, 'b-', linewidth=2, label='Strategy')
            if self.benchmark_returns is not None:
                benchmark_cum = (1 + self.benchmark_returns).cumprod()
                axes[0, 0].plot(benchmark_cum.index, benchmark_cum.values, 'r--', linewidth=2, label='Benchmark')
            axes[0, 0].set_title('Cumulative Returns')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. 回撤曲线
            drawdown = self.calculate_drawdown_series()
            axes[0, 1].fill_between(drawdown.index, drawdown.values, 0, alpha=0.7, color='red')
            axes[0, 1].set_title('Drawdown')
            axes[0, 1].set_ylabel('Drawdown')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. 收益率分布
            axes[0, 2].hist(self.returns, bins=50, alpha=0.7, density=True)
            axes[0, 2].axvline(self.returns.mean(), color='r', linestyle='--', label=f'Mean: {self.returns.mean():.4f}')
            axes[0, 2].axvline(self.returns.median(), color='g', linestyle='--', label=f'Median: {self.returns.median():.4f}')
            axes[0, 2].set_title('Returns Distribution')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
            
            # 4. 滚动夏普比率
            if len(self.returns) > 252:
                rolling_metrics = self.calculate_rolling_metrics(window=252)
                if not rolling_metrics.empty:
                    axes[1, 0].plot(rolling_metrics.index, rolling_metrics['sharpe_ratio'], 'b-', linewidth=2)
                    axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.7)
                    axes[1, 0].set_title('Rolling Sharpe Ratio (1Y)')
                    axes[1, 0].grid(True, alpha=0.3)
            
            # 5. 月度收益热图
            monthly_returns = self.returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            if len(monthly_returns) > 0:
                monthly_table = monthly_returns.groupby([monthly_returns.index.year, monthly_returns.index.month]).first().unstack()
                if not monthly_table.empty:
                    sns.heatmap(monthly_table, annot=True, fmt='.2%', cmap='RdYlGn', center=0, ax=axes[1, 1])
                    axes[1, 1].set_title('Monthly Returns Heatmap')
            
            # 6. 收益率散点图（vs基准）
            if self.benchmark_returns is not None:
                axes[1, 2].scatter(self.benchmark_returns, self.returns, alpha=0.6)
                
                # 添加回归线
                slope, intercept, r_value, p_value, std_err = stats.linregress(self.benchmark_returns, self.returns)
                line = slope * self.benchmark_returns + intercept
                axes[1, 2].plot(self.benchmark_returns, line, 'r-', linewidth=2)
                
                axes[1, 2].set_xlabel('Benchmark Returns')
                axes[1, 2].set_ylabel('Strategy Returns')
                axes[1, 2].set_title(f'Returns Scatter (R² = {r_value**2:.3f})')
                axes[1, 2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_dir:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                plt.savefig(f"{save_dir}/performance_charts_{timestamp}.png", dpi=300, bbox_inches='tight')
            
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Error creating performance charts: {e}")
    
    def compare_with_benchmark(self) -> Dict[str, Any]:
        """与基准进行详细比较"""
        if self.benchmark_returns is None:
            return {"error": "No benchmark data available"}
        
        strategy_metrics = self.calculate_metrics()
        
        # 计算基准指标
        benchmark_analyzer = StrategyPerformance(
            self.benchmark_returns,
            risk_free_rate=self.risk_free_rate,
            trading_days=self.trading_days
        )
        benchmark_metrics = benchmark_analyzer.calculate_metrics()
        
        # 比较结果
        comparison = {
            'strategy': strategy_metrics.to_dict(),
            'benchmark': benchmark_metrics.to_dict(),
            'outperformance': {
                'total_return': strategy_metrics.total_return - benchmark_metrics.total_return,
                'annualized_return': strategy_metrics.annualized_return - benchmark_metrics.annualized_return,
                'sharpe_ratio': strategy_metrics.sharpe_ratio - benchmark_metrics.sharpe_ratio,
                'max_drawdown': strategy_metrics.max_drawdown - benchmark_metrics.max_drawdown,
                'volatility': strategy_metrics.volatility - benchmark_metrics.volatility
            }
        }
        
        return comparison
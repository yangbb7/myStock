# -*- coding: utf-8 -*-
"""
基准比较器 - 提供投资组合与基准的比较分析
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class BenchmarkComparison:
    """基准比较结果"""
    portfolio_return: float
    benchmark_return: float
    excess_return: float
    alpha: float
    beta: float
    information_ratio: float
    tracking_error: float
    correlation: float
    up_capture: float
    down_capture: float


class BenchmarkComparator:
    """基准比较器"""
    
    def __init__(self):
        # 模拟基准数据存储
        self._benchmark_data = {
            "沪深300": {
                "returns": [0.008, -0.003, 0.015, 0.002, 0.012, 0.005, -0.001, 0.018, 0.004, 0.009],
                "weights": {"金融": 0.3, "科技": 0.4, "消费": 0.3}
            },
            "上证50": {
                "returns": [0.006, -0.002, 0.012, 0.001, 0.010, 0.004, -0.002, 0.015, 0.003, 0.007],
                "weights": {"金融": 0.4, "科技": 0.3, "消费": 0.3}
            }
        }
    
    def calculate_alpha_beta(self, portfolio_returns: pd.Series, 
                           benchmark_returns: pd.Series, risk_free_rate: float = 0.02) -> Tuple[float, float]:
        """计算Alpha和Beta"""
        # 超额收益
        portfolio_excess = portfolio_returns - risk_free_rate / 252
        benchmark_excess = benchmark_returns - risk_free_rate / 252
        
        # 计算Beta
        covariance = np.cov(portfolio_excess, benchmark_excess)[0][1]
        benchmark_variance = np.var(benchmark_excess)
        beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
        
        # 计算Alpha
        portfolio_mean = portfolio_excess.mean() * 252
        benchmark_mean = benchmark_excess.mean() * 252
        alpha = portfolio_mean - beta * benchmark_mean
        
        return alpha, beta
    
    def calculate_information_ratio(self, portfolio_returns: pd.Series, 
                                  benchmark_returns: pd.Series) -> float:
        """计算信息比率"""
        excess_returns = portfolio_returns - benchmark_returns
        tracking_error = excess_returns.std() * np.sqrt(252)
        excess_return = excess_returns.mean() * 252
        
        return excess_return / tracking_error if tracking_error != 0 else 0
    
    def calculate_tracking_error(self, portfolio_returns: pd.Series, 
                               benchmark_returns: pd.Series) -> float:
        """计算跟踪误差"""
        excess_returns = portfolio_returns - benchmark_returns
        return excess_returns.std() * np.sqrt(252)
    
    def calculate_capture_ratios(self, portfolio_returns: pd.Series, 
                               benchmark_returns: pd.Series) -> Tuple[float, float]:
        """计算上行捕获率和下行捕获率"""
        # 上行市场（基准收益 > 0）
        up_mask = benchmark_returns > 0
        if up_mask.sum() > 0:
            up_portfolio = portfolio_returns[up_mask].mean()
            up_benchmark = benchmark_returns[up_mask].mean()
            up_capture = up_portfolio / up_benchmark if up_benchmark != 0 else 0
        else:
            up_capture = 0
        
        # 下行市场（基准收益 < 0）
        down_mask = benchmark_returns < 0
        if down_mask.sum() > 0:
            down_portfolio = portfolio_returns[down_mask].mean()
            down_benchmark = benchmark_returns[down_mask].mean()
            down_capture = down_portfolio / down_benchmark if down_benchmark != 0 else 0
        else:
            down_capture = 0
        
        return up_capture, down_capture
    
    def compare_with_benchmark(self, portfolio_returns: pd.Series, 
                             benchmark_returns: pd.Series, 
                             risk_free_rate: float = 0.02) -> BenchmarkComparison:
        """全面比较投资组合与基准"""
        # 基本收益统计
        portfolio_return = (portfolio_returns + 1).prod() - 1
        benchmark_return = (benchmark_returns + 1).prod() - 1
        excess_return = portfolio_return - benchmark_return
        
        # Alpha和Beta
        alpha, beta = self.calculate_alpha_beta(portfolio_returns, benchmark_returns, risk_free_rate)
        
        # 信息比率和跟踪误差
        information_ratio = self.calculate_information_ratio(portfolio_returns, benchmark_returns)
        tracking_error = self.calculate_tracking_error(portfolio_returns, benchmark_returns)
        
        # 相关性
        correlation = portfolio_returns.corr(benchmark_returns)
        
        # 上下行捕获率
        up_capture, down_capture = self.calculate_capture_ratios(portfolio_returns, benchmark_returns)
        
        return BenchmarkComparison(
            portfolio_return=portfolio_return,
            benchmark_return=benchmark_return,
            excess_return=excess_return,
            alpha=alpha,
            beta=beta,
            information_ratio=information_ratio,
            tracking_error=tracking_error,
            correlation=correlation,
            up_capture=up_capture,
            down_capture=down_capture
        )
    
    def calculate_rolling_metrics(self, portfolio_returns: pd.Series, 
                                benchmark_returns: pd.Series, 
                                window: int = 252) -> pd.DataFrame:
        """计算滚动指标"""
        results = []
        
        for i in range(window, len(portfolio_returns) + 1):
            portfolio_window = portfolio_returns.iloc[i-window:i]
            benchmark_window = benchmark_returns.iloc[i-window:i]
            
            comparison = self.compare_with_benchmark(portfolio_window, benchmark_window)
            
            results.append({
                'date': portfolio_returns.index[i-1],
                'alpha': comparison.alpha,
                'beta': comparison.beta,
                'information_ratio': comparison.information_ratio,
                'tracking_error': comparison.tracking_error,
                'correlation': comparison.correlation
            })
        
        return pd.DataFrame(results).set_index('date')
    
    def load_benchmark_data(self, benchmark_name: str, start_date: str, end_date: str) -> pd.DataFrame:
        """加载基准数据"""
        if benchmark_name not in self._benchmark_data:
            raise ValueError(f"未找到基准数据: {benchmark_name}")
        
        data = self._benchmark_data[benchmark_name]
        
        # 生成日期范围
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        dates = pd.date_range(start, end, freq='D')
        
        # 创建模拟数据
        returns = data["returns"]
        extended_returns = (returns * (len(dates) // len(returns) + 1))[:len(dates)]
        
        return pd.DataFrame({
            'date': dates,
            'return': extended_returns,
            'price': np.cumprod(1 + np.array(extended_returns))
        }).set_index('date')
    
    def calculate_relative_performance(self, portfolio_returns: List[float], 
                                     benchmark_returns: List[float]) -> Dict[str, float]:
        """计算相对绩效"""
        portfolio_returns = np.array(portfolio_returns)
        benchmark_returns = np.array(benchmark_returns)
        
        # 计算累计收益
        portfolio_cumulative = np.prod(1 + portfolio_returns) - 1
        benchmark_cumulative = np.prod(1 + benchmark_returns) - 1
        
        # 计算超额收益
        excess_returns = portfolio_returns - benchmark_returns
        excess_cumulative = portfolio_cumulative - benchmark_cumulative
        
        # 计算统计指标
        excess_mean = np.mean(excess_returns)
        excess_std = np.std(excess_returns)
        
        # 计算上行和下行捕获率
        benchmark_series = pd.Series(benchmark_returns)
        portfolio_series = pd.Series(portfolio_returns)
        
        up_mask = benchmark_series > 0
        down_mask = benchmark_series < 0
        
        if up_mask.sum() > 0:
            up_capture = portfolio_series[up_mask].mean() / benchmark_series[up_mask].mean()
        else:
            up_capture = 0
            
        if down_mask.sum() > 0:
            down_capture = portfolio_series[down_mask].mean() / benchmark_series[down_mask].mean()
        else:
            down_capture = 0
        
        return {
            'portfolio_return': portfolio_cumulative,
            'benchmark_return': benchmark_cumulative,
            'excess_return': excess_cumulative,
            'excess_mean': excess_mean,
            'excess_std': excess_std,
            'information_ratio': excess_mean / excess_std if excess_std != 0 else 0,
            'win_rate': np.sum(excess_returns > 0) / len(excess_returns),
            'outperformance_ratio': portfolio_cumulative / benchmark_cumulative if benchmark_cumulative != 0 else 0,
            'up_capture': up_capture,
            'down_capture': down_capture
        }
    
    def perform_attribution_analysis(self, portfolio_weights: Dict[str, float], 
                                   benchmark_weights: Dict[str, float], 
                                   sector_returns: Dict[str, float]) -> Dict[str, float]:
        """执行归因分析"""
        allocation_effect = 0.0
        selection_effect = 0.0
        interaction_effect = 0.0
        
        # 计算基准总收益
        benchmark_return = sum(benchmark_weights[sector] * sector_returns[sector] 
                             for sector in benchmark_weights)
        
        for sector in set(portfolio_weights.keys()) | set(benchmark_weights.keys()):
            pw = portfolio_weights.get(sector, 0)
            bw = benchmark_weights.get(sector, 0)
            sr = sector_returns.get(sector, 0)
            
            # 配置效应：(wp - wb) * rb
            allocation_effect += (pw - bw) * benchmark_return
            
            # 选择效应：wb * (rp - rb)
            selection_effect += bw * (sr - benchmark_return)
            
            # 交互效应：(wp - wb) * (rp - rb)
            interaction_effect += (pw - bw) * (sr - benchmark_return)
        
        total_effect = allocation_effect + selection_effect + interaction_effect
        
        return {
            'allocation_effect': allocation_effect,
            'selection_effect': selection_effect,
            'interaction_effect': interaction_effect,
            'total_effect': total_effect,
            'benchmark_return': benchmark_return
        }

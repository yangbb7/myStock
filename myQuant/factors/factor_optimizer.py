# -*- coding: utf-8 -*-
"""
因子优化器 - 提供因子组合、权重优化、因子选择等功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from enum import Enum
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import warnings
from scipy import stats
from scipy.optimize import minimize, differential_evolution
import cvxpy as cp
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


class OptimizationMethod(Enum):
    """优化方法枚举"""
    MEAN_VARIANCE = "mean_variance"         # 均值方差优化
    RISK_PARITY = "risk_parity"            # 风险平价
    HIERARCHICAL = "hierarchical"          # 层次风险平价
    MAXIMUM_SHARPE = "maximum_sharpe"      # 最大夏普比率
    MINIMUM_VARIANCE = "minimum_variance"   # 最小方差
    MAXIMUM_DIVERSIFICATION = "max_div"    # 最大分散化
    BLACK_LITTERMAN = "black_litterman"    # Black-Litterman
    ROBUST_OPTIMIZATION = "robust"         # 鲁棒优化


class FactorSelectionMethod(Enum):
    """因子选择方法枚举"""
    CORRELATION_FILTER = "correlation"     # 相关性过滤
    IC_RANKING = "ic_ranking"              # IC排序
    FORWARD_SELECTION = "forward"          # 前向选择
    BACKWARD_ELIMINATION = "backward"      # 后向消除
    LASSO_SELECTION = "lasso"              # LASSO选择
    MUTUAL_INFO = "mutual_info"            # 互信息
    PCA_SELECTION = "pca"                  # 主成分分析
    RANDOM_FOREST = "random_forest"        # 随机森林重要性


@dataclass
class OptimizationConstraints:
    """优化约束条件"""
    max_weight: float = 0.5                # 单个因子最大权重
    min_weight: float = 0.0                # 单个因子最小权重
    max_leverage: float = 1.0              # 最大杠杆
    target_volatility: Optional[float] = None  # 目标波动率
    max_drawdown: Optional[float] = None   # 最大回撤限制
    sector_constraints: Optional[Dict[str, float]] = None  # 行业约束
    turnover_constraint: Optional[float] = None  # 换手率约束
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'max_weight': self.max_weight,
            'min_weight': self.min_weight,
            'max_leverage': self.max_leverage,
            'target_volatility': self.target_volatility,
            'max_drawdown': self.max_drawdown,
            'sector_constraints': self.sector_constraints,
            'turnover_constraint': self.turnover_constraint
        }


@dataclass
class OptimizationResult:
    """优化结果"""
    weights: Dict[str, float] = None        # 因子权重
    expected_return: float = 0.0            # 预期收益
    expected_risk: float = 0.0              # 预期风险
    sharpe_ratio: float = 0.0               # 夏普比率
    diversification_ratio: float = 0.0      # 分散化比率
    optimization_status: str = "success"    # 优化状态
    iterations: int = 0                     # 迭代次数
    objective_value: float = 0.0            # 目标函数值
    constraints_satisfied: bool = True      # 约束是否满足
    risk_breakdown: Dict[str, float] = None # 风险分解
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'weights': self.weights or {},
            'expected_return': self.expected_return,
            'expected_risk': self.expected_risk,
            'sharpe_ratio': self.sharpe_ratio,
            'diversification_ratio': self.diversification_ratio,
            'optimization_status': self.optimization_status,
            'iterations': self.iterations,
            'objective_value': self.objective_value,
            'constraints_satisfied': self.constraints_satisfied,
            'risk_breakdown': self.risk_breakdown or {}
        }


class FactorOptimizer:
    """因子优化器"""
    
    def __init__(self,
                 method: OptimizationMethod = OptimizationMethod.MEAN_VARIANCE,
                 risk_free_rate: float = 0.02,
                 lookback_window: int = 252,
                 rebalance_frequency: int = 20,
                 transaction_cost: float = 0.001):
        """
        初始化因子优化器
        
        Args:
            method: 优化方法
            risk_free_rate: 无风险利率
            lookback_window: 回望窗口
            rebalance_frequency: 再平衡频率
            transaction_cost: 交易成本
        """
        self.method = method
        self.risk_free_rate = risk_free_rate
        self.lookback_window = lookback_window
        self.rebalance_frequency = rebalance_frequency
        self.transaction_cost = transaction_cost
        
        # 历史数据缓存
        self.factor_returns_cache = {}
        self.covariance_cache = {}
        self.optimization_history = []
        
        self.logger = logging.getLogger(__name__)
        
        # 标准化器
        self.scaler = StandardScaler()
    
    def optimize_portfolio(self,
                          factor_returns: pd.DataFrame,
                          constraints: OptimizationConstraints = None,
                          benchmark_returns: Optional[pd.Series] = None,
                          factor_exposures: Optional[pd.DataFrame] = None) -> OptimizationResult:
        """
        优化因子组合
        
        Args:
            factor_returns: 因子收益矩阵
            constraints: 优化约束
            benchmark_returns: 基准收益
            factor_exposures: 因子暴露
            
        Returns:
            OptimizationResult: 优化结果
        """
        try:
            self.logger.info(f"Optimizing portfolio using {self.method.value}")
            
            if factor_returns.empty:
                self.logger.warning("Empty factor returns data")
                return OptimizationResult(optimization_status="failed")
            
            # 设置默认约束
            if constraints is None:
                constraints = OptimizationConstraints()
            
            # 数据预处理
            clean_returns = self._preprocess_returns(factor_returns)
            
            if clean_returns.empty:
                return OptimizationResult(optimization_status="failed")
            
            # 计算预期收益和协方差矩阵
            expected_returns = self._calculate_expected_returns(clean_returns, benchmark_returns)
            covariance_matrix = self._calculate_covariance_matrix(clean_returns)
            
            # 执行优化
            if self.method == OptimizationMethod.MEAN_VARIANCE:
                result = self._mean_variance_optimization(
                    expected_returns, covariance_matrix, constraints
                )
            elif self.method == OptimizationMethod.RISK_PARITY:
                result = self._risk_parity_optimization(
                    covariance_matrix, constraints
                )
            elif self.method == OptimizationMethod.HIERARCHICAL:
                result = self._hierarchical_risk_parity_optimization(
                    clean_returns, constraints
                )
            elif self.method == OptimizationMethod.MAXIMUM_SHARPE:
                result = self._maximum_sharpe_optimization(
                    expected_returns, covariance_matrix, constraints
                )
            elif self.method == OptimizationMethod.MINIMUM_VARIANCE:
                result = self._minimum_variance_optimization(
                    covariance_matrix, constraints
                )
            elif self.method == OptimizationMethod.MAXIMUM_DIVERSIFICATION:
                result = self._maximum_diversification_optimization(
                    expected_returns, covariance_matrix, constraints
                )
            elif self.method == OptimizationMethod.BLACK_LITTERMAN:
                result = self._black_litterman_optimization(
                    expected_returns, covariance_matrix, constraints, benchmark_returns
                )
            elif self.method == OptimizationMethod.ROBUST_OPTIMIZATION:
                result = self._robust_optimization(
                    clean_returns, constraints
                )
            else:
                raise ValueError(f"Unknown optimization method: {self.method}")
            
            # 计算绩效指标
            result = self._calculate_performance_metrics(
                result, expected_returns, covariance_matrix
            )
            
            # 保存优化历史
            self.optimization_history.append({
                'timestamp': datetime.now(),
                'method': self.method.value,
                'result': result
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in portfolio optimization: {e}")
            return OptimizationResult(optimization_status="failed")
    
    def _preprocess_returns(self, factor_returns: pd.DataFrame) -> pd.DataFrame:
        """预处理收益数据"""
        # 删除缺失值
        clean_returns = factor_returns.dropna()
        
        # 异常值处理
        for col in clean_returns.columns:
            # Winsorize处理
            lower = clean_returns[col].quantile(0.01)
            upper = clean_returns[col].quantile(0.99)
            clean_returns[col] = clean_returns[col].clip(lower, upper)
        
        # 标准化（可选）
        # clean_returns = pd.DataFrame(
        #     self.scaler.fit_transform(clean_returns),
        #     index=clean_returns.index,
        #     columns=clean_returns.columns
        # )
        
        return clean_returns
    
    def _calculate_expected_returns(self,
                                   factor_returns: pd.DataFrame,
                                   benchmark_returns: Optional[pd.Series] = None) -> pd.Series:
        """计算预期收益"""
        # 简单均值方法
        expected_returns = factor_returns.mean()
        
        # 如果有基准，可以使用CAPM或其他方法调整
        if benchmark_returns is not None:
            # 简化的超额收益计算
            benchmark_mean = benchmark_returns.mean()
            expected_returns = expected_returns - benchmark_mean
        
        return expected_returns
    
    def _calculate_covariance_matrix(self, factor_returns: pd.DataFrame) -> pd.DataFrame:
        """计算协方差矩阵"""
        # 样本协方差矩阵
        cov_matrix = factor_returns.cov()
        
        # 可以添加协方差矩阵调整方法
        # 如：收缩估计、Ledoit-Wolf等
        
        return cov_matrix
    
    def _mean_variance_optimization(self,
                                   expected_returns: pd.Series,
                                   covariance_matrix: pd.DataFrame,
                                   constraints: OptimizationConstraints) -> OptimizationResult:
        """均值方差优化"""
        n = len(expected_returns)
        
        # 决策变量：权重
        w = cp.Variable(n)
        
        # 目标函数：最大化 μ'w - γ/2 * w'Σw
        gamma = 1.0  # 风险厌恶系数
        objective = cp.Maximize(
            expected_returns.values @ w - gamma / 2 * cp.quad_form(w, covariance_matrix.values)
        )
        
        # 约束条件
        constraints_list = [
            cp.sum(w) == 1,  # 权重和为1
            w >= constraints.min_weight,  # 最小权重
            w <= constraints.max_weight   # 最大权重
        ]
        
        # 添加额外约束
        if constraints.max_leverage is not None:
            constraints_list.append(cp.norm(w, 1) <= constraints.max_leverage)
        
        # 求解
        prob = cp.Problem(objective, constraints_list)
        prob.solve(solver=cp.ECOS)
        
        if prob.status not in ["infeasible", "unbounded"]:
            weights = dict(zip(expected_returns.index, w.value))
            return OptimizationResult(
                weights=weights,
                optimization_status="success",
                objective_value=prob.value
            )
        else:
            return OptimizationResult(optimization_status="failed")
    
    def _risk_parity_optimization(self,
                                 covariance_matrix: pd.DataFrame,
                                 constraints: OptimizationConstraints) -> OptimizationResult:
        """风险平价优化"""
        n = len(covariance_matrix)
        
        def risk_parity_objective(weights):
            """风险平价目标函数"""
            weights = np.array(weights)
            portfolio_var = weights.T @ covariance_matrix.values @ weights
            
            # 计算边际风险贡献
            marginal_contrib = covariance_matrix.values @ weights
            risk_contrib = weights * marginal_contrib / portfolio_var
            
            # 目标：最小化风险贡献的方差
            target_contrib = 1.0 / n
            return np.sum((risk_contrib - target_contrib) ** 2)
        
        # 初始权重
        x0 = np.ones(n) / n
        
        # 约束条件
        constraints_list = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # 权重和为1
        ]
        
        # 边界约束
        bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n)]
        
        # 优化
        result = minimize(
            risk_parity_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        if result.success:
            weights = dict(zip(covariance_matrix.index, result.x))
            return OptimizationResult(
                weights=weights,
                optimization_status="success",
                iterations=result.nit,
                objective_value=result.fun
            )
        else:
            return OptimizationResult(optimization_status="failed")
    
    def _hierarchical_risk_parity_optimization(self,
                                              factor_returns: pd.DataFrame,
                                              constraints: OptimizationConstraints) -> OptimizationResult:
        """层次风险平价优化"""
        try:
            from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
            from scipy.spatial.distance import squareform
            
            # 计算距离矩阵
            corr_matrix = factor_returns.corr()
            distance_matrix = np.sqrt(0.5 * (1 - corr_matrix))
            
            # 层次聚类
            linkage_matrix = linkage(squareform(distance_matrix), method='ward')
            
            # 递归二分法计算权重
            def _get_cluster_var(cov, cluster_items):
                """计算聚类方差"""
                cov_slice = cov.loc[cluster_items, cluster_items]
                inv_diag = 1 / np.diag(cov_slice)
                inv_diag /= inv_diag.sum()
                w = inv_diag.reshape(-1, 1)
                cluster_var = np.dot(w.T, np.dot(cov_slice, w))[0, 0]
                return cluster_var
            
            def _get_rec_bipart(cov, sort_ix):
                """递归二分"""
                w = pd.Series(1, index=sort_ix)
                c_items = [sort_ix]
                
                while len(c_items) > 0:
                    c_items = [i[j:k] for i in c_items for j, k in 
                              ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
                    
                    for i in range(0, len(c_items), 2):
                        c_items0 = c_items[i]
                        c_items1 = c_items[i + 1]
                        
                        c_var0 = _get_cluster_var(cov, c_items0)
                        c_var1 = _get_cluster_var(cov, c_items1)
                        
                        alpha = 1 - c_var0 / (c_var0 + c_var1)
                        
                        w[c_items0] *= alpha
                        w[c_items1] *= 1 - alpha
                
                return w
            
            # 获取排序后的资产
            sort_ix = factor_returns.columns.tolist()
            
            # 计算HRP权重
            hrp_weights = _get_rec_bipart(factor_returns.cov(), sort_ix)
            
            # 标准化权重
            hrp_weights = hrp_weights / hrp_weights.sum()
            
            weights = hrp_weights.to_dict()
            
            return OptimizationResult(
                weights=weights,
                optimization_status="success"
            )
            
        except Exception as e:
            self.logger.error(f"Error in hierarchical risk parity: {e}")
            return OptimizationResult(optimization_status="failed")
    
    def _maximum_sharpe_optimization(self,
                                    expected_returns: pd.Series,
                                    covariance_matrix: pd.DataFrame,
                                    constraints: OptimizationConstraints) -> OptimizationResult:
        """最大夏普比率优化"""
        n = len(expected_returns)
        
        # 决策变量
        w = cp.Variable(n)
        
        # 转换为二次规划问题
        # 最大化 (μ - rf)'w / sqrt(w'Σw)
        # 等价于最小化 w'Σw s.t. (μ - rf)'w = 1
        
        excess_returns = expected_returns - self.risk_free_rate
        
        # 目标函数：最小化方差
        objective = cp.Minimize(cp.quad_form(w, covariance_matrix.values))
        
        # 约束条件
        constraints_list = [
            excess_returns.values @ w == 1,  # 超额收益约束
            w >= constraints.min_weight,
            w <= constraints.max_weight
        ]
        
        # 求解
        prob = cp.Problem(objective, constraints_list)
        prob.solve(solver=cp.ECOS)
        
        if prob.status not in ["infeasible", "unbounded"]:
            # 标准化权重
            raw_weights = w.value
            weights = raw_weights / np.sum(raw_weights)
            
            weights_dict = dict(zip(expected_returns.index, weights))
            
            return OptimizationResult(
                weights=weights_dict,
                optimization_status="success",
                objective_value=prob.value
            )
        else:
            return OptimizationResult(optimization_status="failed")
    
    def _minimum_variance_optimization(self,
                                      covariance_matrix: pd.DataFrame,
                                      constraints: OptimizationConstraints) -> OptimizationResult:
        """最小方差优化"""
        n = len(covariance_matrix)
        
        # 决策变量
        w = cp.Variable(n)
        
        # 目标函数：最小化方差
        objective = cp.Minimize(cp.quad_form(w, covariance_matrix.values))
        
        # 约束条件
        constraints_list = [
            cp.sum(w) == 1,
            w >= constraints.min_weight,
            w <= constraints.max_weight
        ]
        
        # 求解
        prob = cp.Problem(objective, constraints_list)
        prob.solve(solver=cp.ECOS)
        
        if prob.status not in ["infeasible", "unbounded"]:
            weights = dict(zip(covariance_matrix.index, w.value))
            return OptimizationResult(
                weights=weights,
                optimization_status="success",
                objective_value=prob.value
            )
        else:
            return OptimizationResult(optimization_status="failed")
    
    def _maximum_diversification_optimization(self,
                                             expected_returns: pd.Series,
                                             covariance_matrix: pd.DataFrame,
                                             constraints: OptimizationConstraints) -> OptimizationResult:
        """最大分散化优化"""
        n = len(expected_returns)
        
        def diversification_objective(weights):
            """分散化比率：加权平均波动率 / 组合波动率"""
            weights = np.array(weights)
            
            # 个股波动率
            individual_vols = np.sqrt(np.diag(covariance_matrix.values))
            
            # 加权平均波动率
            weighted_avg_vol = np.sum(weights * individual_vols)
            
            # 组合波动率
            portfolio_vol = np.sqrt(weights.T @ covariance_matrix.values @ weights)
            
            # 最大化分散化比率 = 最小化其负值
            return -weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 0
        
        # 初始权重
        x0 = np.ones(n) / n
        
        # 约束条件
        constraints_list = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        ]
        
        # 边界约束
        bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n)]
        
        # 优化
        result = minimize(
            diversification_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        if result.success:
            weights = dict(zip(expected_returns.index, result.x))
            return OptimizationResult(
                weights=weights,
                optimization_status="success",
                iterations=result.nit,
                objective_value=-result.fun  # 转换回正值
            )
        else:
            return OptimizationResult(optimization_status="failed")
    
    def _black_litterman_optimization(self,
                                     expected_returns: pd.Series,
                                     covariance_matrix: pd.DataFrame,
                                     constraints: OptimizationConstraints,
                                     benchmark_returns: Optional[pd.Series] = None) -> OptimizationResult:
        """Black-Litterman优化"""
        # 简化的BL实现
        # 市场隐含收益（使用历史均值作为先验）
        pi = expected_returns.copy()
        
        # 观点矩阵（这里使用简化版本）
        # 在实际应用中，需要根据具体观点设置P和Q矩阵
        tau = 0.05  # 缩放参数
        
        # BL预期收益（简化版本，实际需要观点矩阵）
        bl_returns = pi  # 这里简化为使用历史收益
        
        # 使用BL预期收益进行均值方差优化
        return self._mean_variance_optimization(bl_returns, covariance_matrix, constraints)
    
    def _robust_optimization(self,
                            factor_returns: pd.DataFrame,
                            constraints: OptimizationConstraints) -> OptimizationResult:
        """鲁棒优化"""
        # 使用样本协方差矩阵的收缩估计
        from sklearn.covariance import LedoitWolf
        
        # LedoitWolf收缩估计
        lw = LedoitWolf()
        robust_cov = lw.fit(factor_returns.values).covariance_
        robust_cov_df = pd.DataFrame(
            robust_cov, 
            index=factor_returns.columns,
            columns=factor_returns.columns
        )
        
        # 使用鲁棒协方差矩阵进行最小方差优化
        return self._minimum_variance_optimization(robust_cov_df, constraints)
    
    def _calculate_performance_metrics(self,
                                      result: OptimizationResult,
                                      expected_returns: pd.Series,
                                      covariance_matrix: pd.DataFrame) -> OptimizationResult:
        """计算绩效指标"""
        if result.weights is None:
            return result
        
        weights = np.array(list(result.weights.values()))
        
        # 预期收益
        result.expected_return = np.sum(weights * expected_returns.values)
        
        # 预期风险
        result.expected_risk = np.sqrt(weights.T @ covariance_matrix.values @ weights)
        
        # 夏普比率
        if result.expected_risk > 0:
            result.sharpe_ratio = (result.expected_return - self.risk_free_rate) / result.expected_risk
        
        # 分散化比率
        individual_vols = np.sqrt(np.diag(covariance_matrix.values))
        weighted_avg_vol = np.sum(weights * individual_vols)
        if result.expected_risk > 0:
            result.diversification_ratio = weighted_avg_vol / result.expected_risk
        
        # 风险分解
        marginal_contrib = covariance_matrix.values @ weights
        risk_contrib = weights * marginal_contrib / (result.expected_risk ** 2)
        
        result.risk_breakdown = dict(zip(
            expected_returns.index,
            risk_contrib
        ))
        
        return result
    
    def backtest_strategy(self,
                         factor_returns: pd.DataFrame,
                         constraints: OptimizationConstraints = None,
                         rebalance_dates: Optional[List[datetime]] = None) -> pd.DataFrame:
        """回测策略"""
        if constraints is None:
            constraints = OptimizationConstraints()
        
        # 生成再平衡日期
        if rebalance_dates is None:
            rebalance_dates = factor_returns.index[::self.rebalance_frequency]
        
        backtest_results = []
        current_weights = None
        
        for i, rebalance_date in enumerate(rebalance_dates):
            # 获取历史数据
            start_date = max(0, i * self.rebalance_frequency - self.lookback_window)
            end_date = i * self.rebalance_frequency
            
            if end_date <= start_date:
                continue
            
            hist_data = factor_returns.iloc[start_date:end_date]
            
            # 优化权重
            opt_result = self.optimize_portfolio(hist_data, constraints)
            
            if opt_result.optimization_status == "success":
                new_weights = opt_result.weights
                
                # 计算换手率
                if current_weights is not None:
                    turnover = self._calculate_turnover(current_weights, new_weights)
                else:
                    turnover = 0.0
                
                current_weights = new_weights
                
                # 计算未来收益
                next_period_end = min(len(factor_returns), end_date + self.rebalance_frequency)
                future_returns = factor_returns.iloc[end_date:next_period_end]
                
                if not future_returns.empty:
                    portfolio_returns = self._calculate_portfolio_returns(
                        future_returns, current_weights
                    )
                    
                    backtest_results.extend([
                        {
                            'date': date,
                            'portfolio_return': ret,
                            'turnover': turnover if idx == 0 else 0,
                            **{f'weight_{factor}': weight for factor, weight in current_weights.items()}
                        }
                        for idx, (date, ret) in enumerate(portfolio_returns.items())
                    ])
        
        return pd.DataFrame(backtest_results).set_index('date')
    
    def _calculate_turnover(self, old_weights: Dict[str, float], new_weights: Dict[str, float]) -> float:
        """计算换手率"""
        all_factors = set(old_weights.keys()) | set(new_weights.keys())
        
        turnover = 0.0
        for factor in all_factors:
            old_w = old_weights.get(factor, 0.0)
            new_w = new_weights.get(factor, 0.0)
            turnover += abs(new_w - old_w)
        
        return turnover / 2  # 单边换手率
    
    def _calculate_portfolio_returns(self,
                                    factor_returns: pd.DataFrame,
                                    weights: Dict[str, float]) -> pd.Series:
        """计算组合收益"""
        portfolio_returns = pd.Series(0.0, index=factor_returns.index)
        
        for factor, weight in weights.items():
            if factor in factor_returns.columns:
                portfolio_returns += weight * factor_returns[factor]
        
        return portfolio_returns


class FactorCombiner:
    """因子组合器 - 专门用于因子选择和组合"""
    
    def __init__(self, 
                 selection_method: FactorSelectionMethod = FactorSelectionMethod.IC_RANKING,
                 max_factors: int = 10,
                 correlation_threshold: float = 0.7):
        """
        初始化因子组合器
        
        Args:
            selection_method: 因子选择方法
            max_factors: 最大因子数量
            correlation_threshold: 相关性阈值
        """
        self.selection_method = selection_method
        self.max_factors = max_factors
        self.correlation_threshold = correlation_threshold
        
        self.logger = logging.getLogger(__name__)
    
    def select_factors(self,
                      factor_data: Dict[str, pd.DataFrame],
                      return_data: pd.DataFrame,
                      factor_ic_scores: Optional[Dict[str, float]] = None) -> List[str]:
        """
        选择因子
        
        Args:
            factor_data: 因子数据字典
            return_data: 收益数据
            factor_ic_scores: 因子IC得分
            
        Returns:
            List[str]: 选中的因子名称列表
        """
        try:
            if self.selection_method == FactorSelectionMethod.IC_RANKING:
                return self._ic_ranking_selection(factor_ic_scores)
            elif self.selection_method == FactorSelectionMethod.CORRELATION_FILTER:
                return self._correlation_filter_selection(factor_data)
            elif self.selection_method == FactorSelectionMethod.FORWARD_SELECTION:
                return self._forward_selection(factor_data, return_data)
            elif self.selection_method == FactorSelectionMethod.BACKWARD_ELIMINATION:
                return self._backward_elimination(factor_data, return_data)
            elif self.selection_method == FactorSelectionMethod.LASSO_SELECTION:
                return self._lasso_selection(factor_data, return_data)
            elif self.selection_method == FactorSelectionMethod.MUTUAL_INFO:
                return self._mutual_info_selection(factor_data, return_data)
            elif self.selection_method == FactorSelectionMethod.PCA_SELECTION:
                return self._pca_selection(factor_data)
            elif self.selection_method == FactorSelectionMethod.RANDOM_FOREST:
                return self._random_forest_selection(factor_data, return_data)
            else:
                raise ValueError(f"Unknown selection method: {self.selection_method}")
                
        except Exception as e:
            self.logger.error(f"Error in factor selection: {e}")
            return list(factor_data.keys())[:self.max_factors]
    
    def _ic_ranking_selection(self, factor_ic_scores: Dict[str, float]) -> List[str]:
        """基于IC排序的因子选择"""
        if not factor_ic_scores:
            return []
        
        # 按IC绝对值排序
        sorted_factors = sorted(
            factor_ic_scores.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        return [factor for factor, _ in sorted_factors[:self.max_factors]]
    
    def _correlation_filter_selection(self, factor_data: Dict[str, pd.DataFrame]) -> List[str]:
        """基于相关性过滤的因子选择"""
        if not factor_data:
            return []
        
        # 构建因子矩阵
        factor_matrix = self._build_factor_matrix(factor_data)
        
        if factor_matrix.empty:
            return []
        
        # 计算相关性矩阵
        corr_matrix = factor_matrix.corr().abs()
        
        # 贪心算法选择低相关性因子
        selected_factors = []
        remaining_factors = list(factor_matrix.columns)
        
        while len(selected_factors) < self.max_factors and remaining_factors:
            if not selected_factors:
                # 选择第一个因子（可以基于其他标准）
                selected_factors.append(remaining_factors[0])
                remaining_factors.remove(remaining_factors[0])
            else:
                # 选择与已选因子相关性最低的因子
                best_factor = None
                best_max_corr = float('inf')
                
                for factor in remaining_factors:
                    max_corr = max([corr_matrix.loc[factor, selected] for selected in selected_factors])
                    if max_corr < best_max_corr:
                        best_max_corr = max_corr
                        best_factor = factor
                
                if best_factor and best_max_corr < self.correlation_threshold:
                    selected_factors.append(best_factor)
                    remaining_factors.remove(best_factor)
                else:
                    break
        
        return selected_factors
    
    def _forward_selection(self,
                          factor_data: Dict[str, pd.DataFrame],
                          return_data: pd.DataFrame) -> List[str]:
        """前向选择"""
        factor_matrix = self._build_factor_matrix(factor_data)
        target = self._align_target_data(factor_matrix, return_data)
        
        if factor_matrix.empty or target.empty:
            return []
        
        from sklearn.feature_selection import SequentialFeatureSelector
        from sklearn.linear_model import LinearRegression
        
        # 前向选择
        sfs = SequentialFeatureSelector(
            LinearRegression(),
            n_features_to_select=min(self.max_factors, len(factor_matrix.columns)),
            direction='forward',
            scoring='r2'
        )
        
        sfs.fit(factor_matrix, target)
        selected_features = factor_matrix.columns[sfs.get_support()].tolist()
        
        return selected_features
    
    def _backward_elimination(self,
                             factor_data: Dict[str, pd.DataFrame],
                             return_data: pd.DataFrame) -> List[str]:
        """后向消除"""
        factor_matrix = self._build_factor_matrix(factor_data)
        target = self._align_target_data(factor_matrix, return_data)
        
        if factor_matrix.empty or target.empty:
            return []
        
        from sklearn.feature_selection import SequentialFeatureSelector
        from sklearn.linear_model import LinearRegression
        
        # 后向消除
        sfs = SequentialFeatureSelector(
            LinearRegression(),
            n_features_to_select=min(self.max_factors, len(factor_matrix.columns)),
            direction='backward',
            scoring='r2'
        )
        
        sfs.fit(factor_matrix, target)
        selected_features = factor_matrix.columns[sfs.get_support()].tolist()
        
        return selected_features
    
    def _lasso_selection(self,
                        factor_data: Dict[str, pd.DataFrame],
                        return_data: pd.DataFrame) -> List[str]:
        """LASSO选择"""
        factor_matrix = self._build_factor_matrix(factor_data)
        target = self._align_target_data(factor_matrix, return_data)
        
        if factor_matrix.empty or target.empty:
            return []
        
        # LASSO回归
        lasso = Lasso(alpha=0.01, max_iter=1000)
        lasso.fit(factor_matrix, target)
        
        # 选择非零系数的因子
        selected_features = factor_matrix.columns[lasso.coef_ != 0].tolist()
        
        return selected_features[:self.max_factors]
    
    def _mutual_info_selection(self,
                              factor_data: Dict[str, pd.DataFrame],
                              return_data: pd.DataFrame) -> List[str]:
        """互信息选择"""
        factor_matrix = self._build_factor_matrix(factor_data)
        target = self._align_target_data(factor_matrix, return_data)
        
        if factor_matrix.empty or target.empty:
            return []
        
        # 互信息特征选择
        selector = SelectKBest(
            score_func=mutual_info_regression,
            k=min(self.max_factors, len(factor_matrix.columns))
        )
        
        selector.fit(factor_matrix, target)
        selected_features = factor_matrix.columns[selector.get_support()].tolist()
        
        return selected_features
    
    def _pca_selection(self, factor_data: Dict[str, pd.DataFrame]) -> List[str]:
        """主成分分析选择"""
        factor_matrix = self._build_factor_matrix(factor_data)
        
        if factor_matrix.empty:
            return []
        
        # PCA分析
        pca = PCA(n_components=min(self.max_factors, len(factor_matrix.columns)))
        pca.fit(factor_matrix)
        
        # 选择对主成分贡献最大的因子
        # 这里使用第一主成分的载荷
        loadings = abs(pca.components_[0])
        factor_importance = dict(zip(factor_matrix.columns, loadings))
        
        # 按重要性排序
        sorted_factors = sorted(
            factor_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [factor for factor, _ in sorted_factors[:self.max_factors]]
    
    def _random_forest_selection(self,
                                factor_data: Dict[str, pd.DataFrame],
                                return_data: pd.DataFrame) -> List[str]:
        """随机森林重要性选择"""
        factor_matrix = self._build_factor_matrix(factor_data)
        target = self._align_target_data(factor_matrix, return_data)
        
        if factor_matrix.empty or target.empty:
            return []
        
        # 随机森林
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(factor_matrix, target)
        
        # 按特征重要性排序
        feature_importance = dict(zip(factor_matrix.columns, rf.feature_importances_))
        sorted_factors = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [factor for factor, _ in sorted_factors[:self.max_factors]]
    
    def _build_factor_matrix(self, factor_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """构建因子矩阵"""
        factor_dfs = []
        
        for factor_name, factor_df in factor_data.items():
            # 假设因子数据格式为 (date, symbol, factor_value)
            if 'date' in factor_df.columns and 'symbol' in factor_df.columns:
                factor_series = factor_df.set_index(['date', 'symbol']).iloc[:, -1]
                factor_series.name = factor_name
                factor_dfs.append(factor_series)
            elif isinstance(factor_df.index, pd.MultiIndex):
                factor_series = factor_df.iloc[:, -1]
                factor_series.name = factor_name
                factor_dfs.append(factor_series)
        
        if not factor_dfs:
            return pd.DataFrame()
        
        # 合并所有因子
        factor_matrix = pd.concat(factor_dfs, axis=1)
        return factor_matrix.dropna()
    
    def _align_target_data(self,
                          factor_matrix: pd.DataFrame,
                          return_data: pd.DataFrame) -> pd.Series:
        """对齐目标数据"""
        if factor_matrix.empty:
            return pd.Series()
        
        # 确保return_data有正确的索引
        if 'date' in return_data.columns and 'symbol' in return_data.columns:
            return_series = return_data.set_index(['date', 'symbol']).iloc[:, -1]
        elif isinstance(return_data.index, pd.MultiIndex):
            return_series = return_data.iloc[:, -1]
        else:
            return pd.Series()
        
        # 对齐索引
        common_index = factor_matrix.index.intersection(return_series.index)
        
        if len(common_index) == 0:
            return pd.Series()
        
        return return_series.loc[common_index]
    
    def combine_factors(self,
                       selected_factors: List[str],
                       factor_data: Dict[str, pd.DataFrame],
                       combination_method: str = 'equal_weight') -> pd.DataFrame:
        """
        组合因子
        
        Args:
            selected_factors: 选中的因子列表
            factor_data: 因子数据
            combination_method: 组合方法
            
        Returns:
            pd.DataFrame: 组合后的因子
        """
        if not selected_factors:
            return pd.DataFrame()
        
        # 构建选中因子的矩阵
        selected_factor_data = {f: factor_data[f] for f in selected_factors if f in factor_data}
        factor_matrix = self._build_factor_matrix(selected_factor_data)
        
        if factor_matrix.empty:
            return pd.DataFrame()
        
        if combination_method == 'equal_weight':
            # 等权重组合
            combined_factor = factor_matrix.mean(axis=1)
        elif combination_method == 'ic_weight':
            # 基于IC加权（需要额外的IC数据）
            combined_factor = factor_matrix.mean(axis=1)  # 简化为等权重
        elif combination_method == 'pca':
            # 主成分组合
            pca = PCA(n_components=1)
            combined_values = pca.fit_transform(factor_matrix)
            combined_factor = pd.Series(
                combined_values.flatten(),
                index=factor_matrix.index,
                name='combined_factor'
            )
        else:
            combined_factor = factor_matrix.mean(axis=1)
        
        # 转换回DataFrame格式
        combined_df = combined_factor.reset_index()
        combined_df.columns = ['date', 'symbol', 'factor_value']
        
        return combined_df
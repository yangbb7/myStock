# -*- coding: utf-8 -*-
"""
优化算法 - 实现多种参数优化算法
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import logging
import time
from scipy.optimize import minimize, differential_evolution
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.model_selection import ParameterGrid
import warnings

from .parameter_space import ParameterSpace

warnings.filterwarnings('ignore')


@dataclass
class OptimizationResult:
    """优化结果"""
    best_params: Dict[str, Any]
    best_score: float
    optimization_history: List[Dict[str, Any]]
    total_evaluations: int
    optimization_time: float
    convergence_info: Dict[str, Any]
    algorithm_name: str
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'total_evaluations': self.total_evaluations,
            'optimization_time': self.optimization_time,
            'algorithm_name': self.algorithm_name,
            'convergence_info': self.convergence_info,
            'history_length': len(self.optimization_history)
        }


class BaseOptimizer(ABC):
    """优化器基类"""
    
    def __init__(self, 
                 parameter_space: ParameterSpace,
                 objective_function: Callable[[Dict[str, Any]], float],
                 maximize: bool = True,
                 random_state: Optional[int] = None,
                 n_jobs: int = 1,
                 verbose: bool = False):
        """
        初始化优化器
        
        Args:
            parameter_space: 参数空间
            objective_function: 目标函数
            maximize: 是否最大化目标函数
            random_state: 随机种子
            n_jobs: 并行任务数
            verbose: 是否显示详细信息
        """
        self.parameter_space = parameter_space
        self.objective_function = objective_function
        self.maximize = maximize
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        # 优化状态
        self.history: List[Dict[str, Any]] = []
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: float = float('-inf') if maximize else float('inf')
        self.evaluations: int = 0
        
        # 日志
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if random_state is not None:
            np.random.seed(random_state)
    
    @abstractmethod
    def optimize(self, n_trials: int = 100, **kwargs) -> OptimizationResult:
        """
        执行优化
        
        Args:
            n_trials: 优化试验次数
            **kwargs: 其他参数
            
        Returns:
            OptimizationResult: 优化结果
        """
        pass
    
    def evaluate(self, params: Dict[str, Any]) -> float:
        """
        评估参数组合
        
        Args:
            params: 参数组合
            
        Returns:
            float: 目标函数值
        """
        try:
            # 验证参数
            if not self.parameter_space.validate_sample(params):
                params = self.parameter_space.clip_sample(params)
            
            # 计算目标函数
            score = self.objective_function(params)
            
            # 处理无效结果
            if np.isnan(score) or np.isinf(score):
                score = float('-inf') if self.maximize else float('inf')
            
            # 更新最佳结果
            is_better = (score > self.best_score) if self.maximize else (score < self.best_score)
            if is_better:
                self.best_score = score
                self.best_params = params.copy()
            
            # 记录历史
            evaluation_record = {
                'evaluation': self.evaluations,
                'params': params.copy(),
                'score': score,
                'is_best': is_better,
                'timestamp': time.time()
            }
            self.history.append(evaluation_record)
            
            self.evaluations += 1
            
            if self.verbose:
                self.logger.info(f"Evaluation {self.evaluations}: score={score:.4f}, best={self.best_score:.4f}")
            
            return score
            
        except Exception as e:
            self.logger.error(f"Error evaluating params {params}: {e}")
            return float('-inf') if self.maximize else float('inf')
    
    def parallel_evaluate(self, params_list: List[Dict[str, Any]]) -> List[float]:
        """
        并行评估参数组合列表
        
        Args:
            params_list: 参数组合列表
            
        Returns:
            List[float]: 目标函数值列表
        """
        if self.n_jobs == 1:
            return [self.evaluate(params) for params in params_list]
        
        # 并行处理
        scores = [None] * len(params_list)
        
        executor_class = ThreadPoolExecutor if self.n_jobs > 0 else ProcessPoolExecutor
        max_workers = min(abs(self.n_jobs), len(params_list))
        
        with executor_class(max_workers=max_workers) as executor:
            # 提交任务
            future_to_index = {
                executor.submit(self._safe_evaluate, params): i 
                for i, params in enumerate(params_list)
            }
            
            # 收集结果
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    scores[index] = future.result()
                except Exception as e:
                    self.logger.error(f"Error in parallel evaluation: {e}")
                    scores[index] = float('-inf') if self.maximize else float('inf')
        
        return scores
    
    def _safe_evaluate(self, params: Dict[str, Any]) -> float:
        """安全的评估函数（用于并行处理）"""
        try:
            return self.objective_function(params)
        except Exception as e:
            self.logger.error(f"Error in objective function: {e}")
            return float('-inf') if self.maximize else float('inf')


class GridSearchOptimizer(BaseOptimizer):
    """网格搜索优化器"""
    
    def optimize(self, n_trials: int = 100, **kwargs) -> OptimizationResult:
        """
        执行网格搜索优化
        
        Args:
            n_trials: 网格点数（用于计算每个参数的网格密度）
            **kwargs: 其他参数
        """
        start_time = time.time()
        
        # 估算每个参数的网格点数
        n_params = len(self.parameter_space.parameters)
        points_per_param = max(2, int(n_trials ** (1.0 / n_params))) if n_params > 0 else 10
        
        # 生成网格
        grid_combinations = self.parameter_space.get_grid(points_per_param)
        
        if self.verbose:
            self.logger.info(f"Grid search with {len(grid_combinations)} combinations")
        
        # 评估所有组合
        for params in grid_combinations:
            self.evaluate(params)
        
        optimization_time = time.time() - start_time
        
        return OptimizationResult(
            best_params=self.best_params or {},
            best_score=self.best_score,
            optimization_history=self.history,
            total_evaluations=self.evaluations,
            optimization_time=optimization_time,
            convergence_info={'grid_size': len(grid_combinations)},
            algorithm_name='GridSearch'
        )


class RandomSearchOptimizer(BaseOptimizer):
    """随机搜索优化器"""
    
    def optimize(self, n_trials: int = 100, **kwargs) -> OptimizationResult:
        """
        执行随机搜索优化
        
        Args:
            n_trials: 随机试验次数
            **kwargs: 其他参数
        """
        start_time = time.time()
        
        if self.verbose:
            self.logger.info(f"Random search with {n_trials} trials")
        
        # 随机采样和评估
        for i in range(n_trials):
            params_list = self.parameter_space.sample(1, random_state=self.random_state + i if self.random_state else None)
            if params_list:
                self.evaluate(params_list[0])
        
        optimization_time = time.time() - start_time
        
        return OptimizationResult(
            best_params=self.best_params or {},
            best_score=self.best_score,
            optimization_history=self.history,
            total_evaluations=self.evaluations,
            optimization_time=optimization_time,
            convergence_info={'n_trials': n_trials},
            algorithm_name='RandomSearch'
        )


class BayesianOptimizer(BaseOptimizer):
    """贝叶斯优化器"""
    
    def __init__(self, *args, acquisition_function: str = 'EI', xi: float = 0.01, **kwargs):
        """
        初始化贝叶斯优化器
        
        Args:
            acquisition_function: 获取函数类型 ('EI', 'PI', 'UCB')
            xi: 探索参数
        """
        super().__init__(*args, **kwargs)
        self.acquisition_function = acquisition_function
        self.xi = xi
        self.gp = None
        self.X_observed = []
        self.y_observed = []
    
    def optimize(self, n_trials: int = 100, n_initial_points: int = 10, **kwargs) -> OptimizationResult:
        """
        执行贝叶斯优化
        
        Args:
            n_trials: 总试验次数
            n_initial_points: 初始随机点数
            **kwargs: 其他参数
        """
        start_time = time.time()
        
        # 初始化高斯过程
        kernel = Matern(length_scale=1.0, nu=2.5)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=self.random_state
        )
        
        if self.verbose:
            self.logger.info(f"Bayesian optimization with {n_trials} trials ({n_initial_points} initial)")
        
        # 1. 初始随机采样
        initial_trials = min(n_initial_points, n_trials)
        for i in range(initial_trials):
            params_list = self.parameter_space.sample(1, random_state=self.random_state + i if self.random_state else None)
            if params_list:
                params = params_list[0]
                score = self.evaluate(params)
                
                # 编码参数用于GP
                encoded_params = self.parameter_space.encode_sample(params)
                self.X_observed.append(encoded_params)
                self.y_observed.append(score)
        
        # 2. 贝叶斯优化循环
        for i in range(initial_trials, n_trials):
            if len(self.X_observed) < 2:
                break
            
            try:
                # 拟合高斯过程
                X_array = np.array(self.X_observed)
                y_array = np.array(self.y_observed)
                
                self.gp.fit(X_array, y_array)
                
                # 寻找下一个采样点
                next_params = self._find_next_sample()
                score = self.evaluate(next_params)
                
                # 更新观测数据
                encoded_params = self.parameter_space.encode_sample(next_params)
                self.X_observed.append(encoded_params)
                self.y_observed.append(score)
                
            except Exception as e:
                self.logger.error(f"Error in Bayesian optimization iteration {i}: {e}")
                # 回退到随机采样
                params_list = self.parameter_space.sample(1, random_state=self.random_state + i if self.random_state else None)
                if params_list:
                    score = self.evaluate(params_list[0])
                    encoded_params = self.parameter_space.encode_sample(params_list[0])
                    self.X_observed.append(encoded_params)
                    self.y_observed.append(score)
        
        optimization_time = time.time() - start_time
        
        # 计算收敛信息
        convergence_info = {
            'n_initial_points': initial_trials,
            'acquisition_function': self.acquisition_function,
            'final_gp_score': getattr(self.gp, 'log_marginal_likelihood_value_', None)
        }
        
        return OptimizationResult(
            best_params=self.best_params or {},
            best_score=self.best_score,
            optimization_history=self.history,
            total_evaluations=self.evaluations,
            optimization_time=optimization_time,
            convergence_info=convergence_info,
            algorithm_name='BayesianOptimization'
        )
    
    def _find_next_sample(self) -> Dict[str, Any]:
        """寻找下一个采样点"""
        bounds = self.parameter_space.get_bounds()
        
        def acquisition(x):
            return -self._acquisition_function(x.reshape(1, -1))
        
        # 多次随机初始化以找到全局最优
        best_x = None
        best_acquisition = float('inf')
        
        for _ in range(10):
            # 随机初始点
            x0 = np.array([np.random.uniform(low, high) for low, high in bounds])
            
            try:
                result = minimize(
                    acquisition,
                    x0,
                    bounds=bounds,
                    method='L-BFGS-B'
                )
                
                if result.fun < best_acquisition:
                    best_acquisition = result.fun
                    best_x = result.x
                    
            except Exception as e:
                self.logger.warning(f"Optimization failed for initial point: {e}")
                continue
        
        if best_x is None:
            # 回退到随机采样
            return self.parameter_space.sample(1)[0]
        
        # 解码参数
        return self.parameter_space.decode_sample(best_x.tolist())
    
    def _acquisition_function(self, X: np.ndarray) -> np.ndarray:
        """计算获取函数值"""
        if self.gp is None:
            return np.zeros(X.shape[0])
        
        # 预测均值和标准差
        mu, std = self.gp.predict(X, return_std=True)
        
        if self.acquisition_function == 'EI':
            # Expected Improvement
            if self.maximize:
                imp = mu - self.best_score - self.xi
            else:
                imp = self.best_score - mu - self.xi
            
            with np.errstate(divide='ignore', invalid='ignore'):
                Z = imp / std
                ei = imp * self._normal_cdf(Z) + std * self._normal_pdf(Z)
                ei[std == 0.0] = 0.0
            
            return ei
        
        elif self.acquisition_function == 'PI':
            # Probability of Improvement
            if self.maximize:
                imp = mu - self.best_score - self.xi
            else:
                imp = self.best_score - mu - self.xi
            
            with np.errstate(divide='ignore', invalid='ignore'):
                Z = imp / std
                pi = self._normal_cdf(Z)
                pi[std == 0.0] = 0.0
            
            return pi
        
        elif self.acquisition_function == 'UCB':
            # Upper Confidence Bound
            kappa = 2.576  # 99% confidence
            if self.maximize:
                return mu + kappa * std
            else:
                return -(mu - kappa * std)
        
        else:
            raise ValueError(f"Unknown acquisition function: {self.acquisition_function}")
    
    @staticmethod
    def _normal_cdf(x):
        """标准正态分布累积分布函数"""
        return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x**2 / np.pi)))
    
    @staticmethod
    def _normal_pdf(x):
        """标准正态分布概率密度函数"""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)


class DifferentialEvolutionOptimizer(BaseOptimizer):
    """差分进化优化器"""
    
    def __init__(self, *args, strategy: str = 'best1bin', mutation: float = 0.5, recombination: float = 0.7, **kwargs):
        """
        初始化差分进化优化器
        
        Args:
            strategy: 进化策略
            mutation: 变异系数
            recombination: 重组概率
        """
        super().__init__(*args, **kwargs)
        self.strategy = strategy
        self.mutation = mutation
        self.recombination = recombination
    
    def optimize(self, n_trials: int = 100, population_size: int = 15, **kwargs) -> OptimizationResult:
        """
        执行差分进化优化
        
        Args:
            n_trials: 最大迭代次数
            population_size: 种群大小
            **kwargs: 其他参数
        """
        start_time = time.time()
        
        # 获取参数边界
        bounds = self.parameter_space.get_bounds()
        
        def objective_wrapper(x):
            params = self.parameter_space.decode_sample(x.tolist())
            score = self.objective_function(params)
            
            # 记录评估历史
            is_better = (score > self.best_score) if self.maximize else (score < self.best_score)
            if is_better:
                self.best_score = score
                self.best_params = params.copy()
            
            evaluation_record = {
                'evaluation': self.evaluations,
                'params': params.copy(),
                'score': score,
                'is_best': is_better,
                'timestamp': time.time()
            }
            self.history.append(evaluation_record)
            self.evaluations += 1
            
            if self.verbose and self.evaluations % 10 == 0:
                self.logger.info(f"Evaluation {self.evaluations}: score={score:.4f}, best={self.best_score:.4f}")
            
            # 差分进化是最小化算法
            return -score if self.maximize else score
        
        if self.verbose:
            self.logger.info(f"Differential evolution with population_size={population_size}, maxiter={n_trials}")
        
        try:
            result = differential_evolution(
                objective_wrapper,
                bounds,
                strategy=self.strategy,
                maxiter=n_trials,
                popsize=population_size,
                mutation=self.mutation,
                recombination=self.recombination,
                seed=self.random_state,
                workers=self.n_jobs if self.n_jobs > 1 else 1,
                polish=True
            )
            
            # 更新最佳结果
            final_params = self.parameter_space.decode_sample(result.x.tolist())
            final_score = -result.fun if self.maximize else result.fun
            
            if self.best_params is None or ((final_score > self.best_score) if self.maximize else (final_score < self.best_score)):
                self.best_params = final_params
                self.best_score = final_score
            
            convergence_info = {
                'success': result.success,
                'message': result.message,
                'nfev': result.nfev,
                'nit': result.nit
            }
            
        except Exception as e:
            self.logger.error(f"Differential evolution failed: {e}")
            convergence_info = {'error': str(e)}
        
        optimization_time = time.time() - start_time
        
        return OptimizationResult(
            best_params=self.best_params or {},
            best_score=self.best_score,
            optimization_history=self.history,
            total_evaluations=self.evaluations,
            optimization_time=optimization_time,
            convergence_info=convergence_info,
            algorithm_name='DifferentialEvolution'
        )


class HyperOptimizer(BaseOptimizer):
    """混合优化器 - 结合多种优化算法"""
    
    def __init__(self, *args, algorithms: Optional[List[str]] = None, **kwargs):
        """
        初始化混合优化器
        
        Args:
            algorithms: 使用的算法列表
        """
        super().__init__(*args, **kwargs)
        self.algorithms = algorithms or ['RandomSearch', 'BayesianOptimization', 'DifferentialEvolution']
        
    def optimize(self, n_trials: int = 100, **kwargs) -> OptimizationResult:
        """
        执行混合优化
        
        Args:
            n_trials: 总试验次数
            **kwargs: 其他参数
        """
        start_time = time.time()
        
        # 为每个算法分配试验次数
        trials_per_algorithm = n_trials // len(self.algorithms)
        remaining_trials = n_trials % len(self.algorithms)
        
        all_results = []
        
        for i, algorithm in enumerate(self.algorithms):
            current_trials = trials_per_algorithm + (1 if i < remaining_trials else 0)
            
            if self.verbose:
                self.logger.info(f"Running {algorithm} with {current_trials} trials")
            
            # 创建算法实例
            if algorithm == 'GridSearch':
                optimizer = GridSearchOptimizer(
                    self.parameter_space, self.objective_function, self.maximize,
                    self.random_state, self.n_jobs, False
                )
            elif algorithm == 'RandomSearch':
                optimizer = RandomSearchOptimizer(
                    self.parameter_space, self.objective_function, self.maximize,
                    self.random_state, self.n_jobs, False
                )
            elif algorithm == 'BayesianOptimization':
                optimizer = BayesianOptimizer(
                    self.parameter_space, self.objective_function, self.maximize,
                    self.random_state, self.n_jobs, False
                )
            elif algorithm == 'DifferentialEvolution':
                optimizer = DifferentialEvolutionOptimizer(
                    self.parameter_space, self.objective_function, self.maximize,
                    self.random_state, self.n_jobs, False
                )
            else:
                self.logger.warning(f"Unknown algorithm {algorithm}, skipping")
                continue
            
            # 运行优化
            result = optimizer.optimize(current_trials, **kwargs)
            all_results.append(result)
            
            # 更新全局最佳结果
            if self.best_params is None or ((result.best_score > self.best_score) if self.maximize else (result.best_score < self.best_score)):
                self.best_params = result.best_params
                self.best_score = result.best_score
            
            # 合并历史记录
            self.history.extend(result.optimization_history)
            self.evaluations += result.total_evaluations
        
        optimization_time = time.time() - start_time
        
        # 汇总收敛信息
        convergence_info = {
            'algorithms_used': self.algorithms,
            'algorithm_results': [result.to_dict() for result in all_results]
        }
        
        return OptimizationResult(
            best_params=self.best_params or {},
            best_score=self.best_score,
            optimization_history=self.history,
            total_evaluations=self.evaluations,
            optimization_time=optimization_time,
            convergence_info=convergence_info,
            algorithm_name='HyperOptimizer'
        )
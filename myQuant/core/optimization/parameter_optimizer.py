# -*- coding: utf-8 -*-
"""
参数优化器 - 策略参数优化的主要接口
"""

import logging
import json
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from .parameter_space import ParameterSpace
from .optimization_algorithms import (
    BaseOptimizer, GridSearchOptimizer, RandomSearchOptimizer, 
    BayesianOptimizer, DifferentialEvolutionOptimizer, HyperOptimizer,
    OptimizationResult
)
from .objective_functions import (
    BaseObjectiveFunction, SharpeRatioObjective, CompositeObjective
)


class ParameterOptimizer:
    """
    参数优化器 - 策略参数优化的统一接口
    """
    
    def __init__(self,
                 strategy_class: type,
                 parameter_space: ParameterSpace,
                 data: pd.DataFrame,
                 objective_function: Optional[BaseObjectiveFunction] = None,
                 algorithm: str = 'BayesianOptimization',
                 cv_folds: int = 1,
                 test_size: float = 0.2,
                 random_state: Optional[int] = None,
                 n_jobs: int = 1,
                 verbose: bool = True,
                 results_dir: Optional[str] = None):
        """
        初始化参数优化器
        
        Args:
            strategy_class: 策略类
            parameter_space: 参数空间
            data: 历史数据
            objective_function: 目标函数
            algorithm: 优化算法
            cv_folds: 交叉验证折数
            test_size: 测试集比例
            random_state: 随机种子
            n_jobs: 并行任务数
            verbose: 是否显示详细信息
            results_dir: 结果保存目录
        """
        self.strategy_class = strategy_class
        self.parameter_space = parameter_space
        self.data = data
        self.algorithm = algorithm
        self.cv_folds = cv_folds
        self.test_size = test_size
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        # 设置目标函数
        if objective_function is None:
            self.objective_function = SharpeRatioObjective(strategy_class, data)
        else:
            self.objective_function = objective_function
        
        # 结果存储
        self.results_dir = results_dir
        if self.results_dir:
            Path(self.results_dir).mkdir(parents=True, exist_ok=True)
        
        # 优化历史
        self.optimization_results: List[OptimizationResult] = []
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: Optional[float] = None
        
        # 数据分割
        self.train_data, self.validation_data, self.test_data = self._split_data()
        
        # 日志
        self.logger = logging.getLogger(__name__)
        
        if verbose:
            self.logger.info(f"ParameterOptimizer initialized with algorithm: {algorithm}")
            self.logger.info(f"Data shape: {data.shape}")
            self.logger.info(f"Parameter space: {len(parameter_space.parameters)} parameters")
    
    def _split_data(self) -> tuple:
        """分割数据为训练、验证、测试集"""
        n = len(self.data)
        
        if self.cv_folds > 1:
            # 交叉验证模式
            test_start = int(n * (1 - self.test_size))
            train_data = self.data.iloc[:test_start]
            test_data = self.data.iloc[test_start:]
            validation_data = None  # 在CV中动态生成
        else:
            # 简单分割
            val_start = int(n * 0.6)
            test_start = int(n * (1 - self.test_size))
            
            train_data = self.data.iloc[:val_start]
            validation_data = self.data.iloc[val_start:test_start]
            test_data = self.data.iloc[test_start:]
        
        return train_data, validation_data, test_data
    
    def optimize(self, 
                 n_trials: int = 100,
                 algorithm_params: Optional[Dict[str, Any]] = None,
                 **kwargs) -> OptimizationResult:
        """
        执行参数优化
        
        Args:
            n_trials: 优化试验次数
            algorithm_params: 算法特定参数
            **kwargs: 其他参数
            
        Returns:
            OptimizationResult: 优化结果
        """
        start_time = time.time()
        
        if self.verbose:
            self.logger.info(f"Starting optimization with {n_trials} trials")
        
        # 设置目标函数（使用训练数据）
        if self.cv_folds > 1:
            objective_func = self._create_cv_objective_function()
        else:
            train_objective = self._create_objective_function(self.train_data)
            objective_func = train_objective
        
        # 创建优化器
        optimizer = self._create_optimizer(objective_func, algorithm_params or {})
        
        # 执行优化
        result = optimizer.optimize(n_trials, **kwargs)
        
        # 验证最佳参数
        if self.validation_data is not None:
            validation_score = self._validate_params(result.best_params)
            result.convergence_info['validation_score'] = validation_score
        
        # 测试最佳参数
        if self.test_data is not None and len(self.test_data) > 0:
            test_score = self._test_params(result.best_params)
            result.convergence_info['test_score'] = test_score
        
        # 更新最佳结果
        if self.best_score is None or result.best_score > self.best_score:
            self.best_params = result.best_params
            self.best_score = result.best_score
        
        # 保存结果
        self.optimization_results.append(result)
        
        optimization_time = time.time() - start_time
        
        if self.verbose:
            self.logger.info(f"Optimization completed in {optimization_time:.2f} seconds")
            self.logger.info(f"Best score: {result.best_score:.4f}")
            self.logger.info(f"Best params: {result.best_params}")
        
        # 保存结果到文件
        if self.results_dir:
            self._save_results(result)
        
        return result
    
    def _create_optimizer(self, objective_func: Callable, algorithm_params: Dict[str, Any]) -> BaseOptimizer:
        """创建优化器实例"""
        common_params = {
            'parameter_space': self.parameter_space,
            'objective_function': objective_func,
            'maximize': True,  # 假设我们要最大化目标函数
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            'verbose': self.verbose
        }
        
        if self.algorithm == 'GridSearch':
            return GridSearchOptimizer(**common_params, **algorithm_params)
        elif self.algorithm == 'RandomSearch':
            return RandomSearchOptimizer(**common_params, **algorithm_params)
        elif self.algorithm == 'BayesianOptimization':
            return BayesianOptimizer(**common_params, **algorithm_params)
        elif self.algorithm == 'DifferentialEvolution':
            return DifferentialEvolutionOptimizer(**common_params, **algorithm_params)
        elif self.algorithm == 'HyperOptimizer':
            return HyperOptimizer(**common_params, **algorithm_params)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def _create_objective_function(self, data: pd.DataFrame) -> Callable:
        """创建目标函数"""
        # 创建基于特定数据的目标函数
        if hasattr(self.objective_function, '__class__'):
            # 创建新实例
            objective_class = self.objective_function.__class__
            return objective_class(self.strategy_class, data)
        else:
            # 使用现有实例但更新数据
            self.objective_function.data = data
            return self.objective_function
    
    def _create_cv_objective_function(self) -> Callable:
        """创建交叉验证目标函数"""
        def cv_objective(params: Dict[str, Any]) -> float:
            scores = []
            
            # K折交叉验证
            fold_size = len(self.train_data) // self.cv_folds
            
            for fold in range(self.cv_folds):
                # 分割训练和验证数据
                val_start = fold * fold_size
                val_end = (fold + 1) * fold_size if fold < self.cv_folds - 1 else len(self.train_data)
                
                train_fold = pd.concat([
                    self.train_data.iloc[:val_start],
                    self.train_data.iloc[val_end:]
                ])
                val_fold = self.train_data.iloc[val_start:val_end]
                
                # 在训练折上训练，在验证折上评估
                train_objective = self._create_objective_function(train_fold)
                val_objective = self._create_objective_function(val_fold)
                
                # 使用验证数据评估
                score = val_objective(params)
                scores.append(score)
            
            # 返回平均得分
            return np.mean(scores)
        
        return cv_objective
    
    def _validate_params(self, params: Dict[str, Any]) -> float:
        """在验证集上验证参数"""
        val_objective = self._create_objective_function(self.validation_data)
        return val_objective(params)
    
    def _test_params(self, params: Dict[str, Any]) -> float:
        """在测试集上测试参数"""
        test_objective = self._create_objective_function(self.test_data)
        return test_objective(params)
    
    def _save_results(self, result: OptimizationResult) -> None:
        """保存优化结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存JSON格式的摘要
        summary_file = Path(self.results_dir) / f"optimization_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'algorithm': self.algorithm,
                'best_params': result.best_params,
                'best_score': result.best_score,
                'total_evaluations': result.total_evaluations,
                'optimization_time': result.optimization_time,
                'convergence_info': result.convergence_info
            }, f, indent=2)
        
        # 保存完整结果的pickle文件
        result_file = Path(self.results_dir) / f"optimization_result_{timestamp}.pkl"
        with open(result_file, 'wb') as f:
            pickle.dump(result, f)
        
        # 保存优化历史CSV
        if result.optimization_history:
            history_df = pd.DataFrame([
                {
                    'evaluation': record['evaluation'],
                    'score': record['score'],
                    'is_best': record['is_best'],
                    **record['params']
                }
                for record in result.optimization_history
            ])
            
            history_file = Path(self.results_dir) / f"optimization_history_{timestamp}.csv"
            history_df.to_csv(history_file, index=False)
    
    def compare_algorithms(self, 
                          algorithms: List[str],
                          n_trials: int = 100,
                          algorithm_params: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, OptimizationResult]:
        """
        比较多个优化算法
        
        Args:
            algorithms: 算法列表
            n_trials: 每个算法的试验次数
            algorithm_params: 每个算法的特定参数
            
        Returns:
            Dict[str, OptimizationResult]: 算法结果字典
        """
        results = {}
        algorithm_params = algorithm_params or {}
        
        for algorithm in algorithms:
            if self.verbose:
                self.logger.info(f"Running {algorithm}...")
            
            # 临时更改算法
            original_algorithm = self.algorithm
            self.algorithm = algorithm
            
            try:
                # 运行优化
                result = self.optimize(
                    n_trials=n_trials,
                    algorithm_params=algorithm_params.get(algorithm, {})
                )
                results[algorithm] = result
                
            except Exception as e:
                self.logger.error(f"Error running {algorithm}: {e}")
                continue
            
            finally:
                # 恢复原始算法
                self.algorithm = original_algorithm
        
        # 生成比较报告
        if self.results_dir:
            self._save_comparison_report(results)
        
        return results
    
    def _save_comparison_report(self, results: Dict[str, OptimizationResult]) -> None:
        """保存算法比较报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建比较数据
        comparison_data = []
        for algorithm, result in results.items():
            comparison_data.append({
                'algorithm': algorithm,
                'best_score': result.best_score,
                'total_evaluations': result.total_evaluations,
                'optimization_time': result.optimization_time,
                'best_params': result.best_params
            })
        
        # 保存比较结果
        comparison_file = Path(self.results_dir) / f"algorithm_comparison_{timestamp}.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        # 创建比较图表
        self._create_comparison_plots(results, timestamp)
    
    def _create_comparison_plots(self, results: Dict[str, OptimizationResult], timestamp: str) -> None:
        """创建比较图表"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # 设置样式
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. 最佳得分比较
            algorithms = list(results.keys())
            best_scores = [results[alg].best_score for alg in algorithms]
            
            axes[0, 0].bar(algorithms, best_scores)
            axes[0, 0].set_title('Best Score Comparison')
            axes[0, 0].set_ylabel('Best Score')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # 2. 优化时间比较
            opt_times = [results[alg].optimization_time for alg in algorithms]
            
            axes[0, 1].bar(algorithms, opt_times)
            axes[0, 1].set_title('Optimization Time Comparison')
            axes[0, 1].set_ylabel('Time (seconds)')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # 3. 评估次数比较
            evaluations = [results[alg].total_evaluations for alg in algorithms]
            
            axes[1, 0].bar(algorithms, evaluations)
            axes[1, 0].set_title('Total Evaluations Comparison')
            axes[1, 0].set_ylabel('Number of Evaluations')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # 4. 收敛曲线
            for alg in algorithms:
                history = results[alg].optimization_history
                if history:
                    scores = [record['score'] for record in history]
                    best_so_far = []
                    current_best = float('-inf')
                    for score in scores:
                        if score > current_best:
                            current_best = score
                        best_so_far.append(current_best)
                    
                    axes[1, 1].plot(best_so_far, label=alg)
            
            axes[1, 1].set_title('Convergence Curves')
            axes[1, 1].set_xlabel('Evaluation')
            axes[1, 1].set_ylabel('Best Score So Far')
            axes[1, 1].legend()
            
            plt.tight_layout()
            
            # 保存图表
            plot_file = Path(self.results_dir) / f"algorithm_comparison_{timestamp}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Failed to create comparison plots: {e}")
    
    def analyze_parameter_sensitivity(self, 
                                    param_name: str,
                                    n_points: int = 20,
                                    fix_other_params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        分析参数敏感性
        
        Args:
            param_name: 要分析的参数名
            n_points: 分析点数
            fix_other_params: 固定其他参数值
            
        Returns:
            pd.DataFrame: 敏感性分析结果
        """
        if param_name not in self.parameter_space.parameters:
            raise ValueError(f"Parameter {param_name} not found in parameter space")
        
        param = self.parameter_space.parameters[param_name]
        
        # 生成参数值
        param_values = param.get_grid_values(n_points)
        
        # 固定其他参数
        if fix_other_params is None:
            # 使用最佳参数或默认参数
            if self.best_params:
                fix_other_params = self.best_params.copy()
            else:
                fix_other_params = {
                    name: p.default for name, p in self.parameter_space.parameters.items()
                    if p.default is not None
                }
        
        # 计算每个参数值的得分
        results = []
        objective_func = self._create_objective_function(self.train_data)
        
        for value in param_values:
            params = fix_other_params.copy()
            params[param_name] = value
            
            try:
                score = objective_func(params)
                results.append({
                    'param_value': value,
                    'score': score
                })
            except Exception as e:
                self.logger.warning(f"Error evaluating {param_name}={value}: {e}")
                continue
        
        sensitivity_df = pd.DataFrame(results)
        
        # 保存结果
        if self.results_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sensitivity_file = Path(self.results_dir) / f"sensitivity_{param_name}_{timestamp}.csv"
            sensitivity_df.to_csv(sensitivity_file, index=False)
            
            # 创建敏感性图表
            self._create_sensitivity_plot(sensitivity_df, param_name, timestamp)
        
        return sensitivity_df
    
    def _create_sensitivity_plot(self, sensitivity_df: pd.DataFrame, param_name: str, timestamp: str) -> None:
        """创建参数敏感性图表"""
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(sensitivity_df['param_value'], sensitivity_df['score'], 'b-o', linewidth=2, markersize=6)
            plt.xlabel(param_name)
            plt.ylabel('Objective Score')
            plt.title(f'Parameter Sensitivity Analysis: {param_name}')
            plt.grid(True, alpha=0.3)
            
            # 标记最佳值
            best_idx = sensitivity_df['score'].idxmax()
            best_value = sensitivity_df.loc[best_idx, 'param_value']
            best_score = sensitivity_df.loc[best_idx, 'score']
            
            plt.axvline(x=best_value, color='r', linestyle='--', alpha=0.7, label=f'Best: {best_value:.3f}')
            plt.legend()
            
            # 保存图表
            plot_file = Path(self.results_dir) / f"sensitivity_{param_name}_{timestamp}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Failed to create sensitivity plot: {e}")
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """获取优化摘要"""
        if not self.optimization_results:
            return {"message": "No optimization results available"}
        
        latest_result = self.optimization_results[-1]
        
        return {
            'total_optimizations': len(self.optimization_results),
            'best_overall_score': self.best_score,
            'best_overall_params': self.best_params,
            'latest_result': {
                'algorithm': latest_result.algorithm_name,
                'best_score': latest_result.best_score,
                'total_evaluations': latest_result.total_evaluations,
                'optimization_time': latest_result.optimization_time
            },
            'parameter_space_summary': self.parameter_space.get_summary()
        }
    
    def load_results(self, results_file: str) -> OptimizationResult:
        """
        加载保存的优化结果
        
        Args:
            results_file: 结果文件路径
            
        Returns:
            OptimizationResult: 优化结果
        """
        with open(results_file, 'rb') as f:
            result = pickle.load(f)
        
        self.optimization_results.append(result)
        
        # 更新最佳结果
        if self.best_score is None or result.best_score > self.best_score:
            self.best_params = result.best_params
            self.best_score = result.best_score
        
        return result
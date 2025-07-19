import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from scipy import stats
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

class SimulationType(Enum):
    GEOMETRIC_BROWNIAN_MOTION = "gbm"
    JUMP_DIFFUSION = "jump_diffusion"
    HESTON_STOCHASTIC_VOLATILITY = "heston"
    LEVY_PROCESS = "levy"
    COPULA_SIMULATION = "copula"
    REGIME_SWITCHING = "regime_switching"

class DistributionType(Enum):
    NORMAL = "normal"
    STUDENT_T = "student_t"
    SKEWED_T = "skewed_t"
    GENERALIZED_ERROR = "ged"
    LAPLACE = "laplace"

@dataclass
class SimulationConfig:
    simulation_type: SimulationType
    num_simulations: int = 10000
    time_horizon_days: int = 252
    time_steps: int = 252
    confidence_levels: List[float] = field(default_factory=lambda: [0.95, 0.99, 0.999])
    distribution: DistributionType = DistributionType.NORMAL
    random_seed: Optional[int] = None
    antithetic_variates: bool = True
    control_variates: bool = True
    
@dataclass
class AssetParameters:
    symbol: str
    mu: float  # 预期收益率
    sigma: float  # 波动率
    initial_price: float
    weight: float  # 组合权重
    
    # 跳跃扩散参数
    lambda_jump: float = 0.1  # 跳跃强度
    mu_jump: float = 0.0  # 跳跃幅度均值
    sigma_jump: float = 0.1  # 跳跃幅度标准差
    
    # Heston参数
    kappa: float = 2.0  # 均值回归速度
    theta: float = 0.04  # 长期方差
    xi: float = 0.3  # 波动率的波动率
    rho: float = -0.7  # 相关系数
    
@dataclass
class SimulationResult:
    simulation_id: str
    config: SimulationConfig
    timestamp: datetime
    portfolio_paths: np.ndarray
    final_values: np.ndarray
    var_estimates: Dict[float, float]
    cvar_estimates: Dict[float, float]
    expected_return: float
    volatility: float
    max_drawdown: float
    sharpe_ratio: float
    skewness: float
    kurtosis: float
    tail_expectation: float
    probability_of_loss: float
    execution_time_seconds: float
    convergence_achieved: bool

class MonteCarloSimulationEngine:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 执行器
        self.executor = ThreadPoolExecutor(max_workers=config.get('max_workers', 4))
        
        # 随机数生成器
        self.rng = np.random.RandomState(config.get('random_seed', 42))
        
        # 模拟结果缓存
        self.simulation_results: Dict[str, SimulationResult] = {}
        
        # 并行计算配置
        self.chunk_size = config.get('chunk_size', 1000)
        
    async def run_portfolio_simulation(self, 
                                     assets: List[AssetParameters],
                                     simulation_config: SimulationConfig) -> SimulationResult:
        """运行组合蒙特卡洛模拟"""
        start_time = datetime.now()
        
        try:
            simulation_id = f"MC_{int(start_time.timestamp() * 1000)}"
            
            self.logger.info(f"Starting Monte Carlo simulation: {simulation_config.simulation_type.value}")
            self.logger.info(f"Simulations: {simulation_config.num_simulations}, Assets: {len(assets)}")
            
            # 设置随机种子
            if simulation_config.random_seed:
                self.rng.seed(simulation_config.random_seed)
            
            # 运行模拟
            if simulation_config.simulation_type == SimulationType.GEOMETRIC_BROWNIAN_MOTION:
                portfolio_paths, final_values = await self._run_gbm_simulation(assets, simulation_config)
            elif simulation_config.simulation_type == SimulationType.JUMP_DIFFUSION:
                portfolio_paths, final_values = await self._run_jump_diffusion_simulation(assets, simulation_config)
            elif simulation_config.simulation_type == SimulationType.HESTON_STOCHASTIC_VOLATILITY:
                portfolio_paths, final_values = await self._run_heston_simulation(assets, simulation_config)
            elif simulation_config.simulation_type == SimulationType.LEVY_PROCESS:
                portfolio_paths, final_values = await self._run_levy_simulation(assets, simulation_config)
            elif simulation_config.simulation_type == SimulationType.COPULA_SIMULATION:
                portfolio_paths, final_values = await self._run_copula_simulation(assets, simulation_config)
            elif simulation_config.simulation_type == SimulationType.REGIME_SWITCHING:
                portfolio_paths, final_values = await self._run_regime_switching_simulation(assets, simulation_config)
            else:
                raise ValueError(f"Unsupported simulation type: {simulation_config.simulation_type}")
            
            # 计算统计量
            var_estimates = self._calculate_var_estimates(final_values, simulation_config.confidence_levels)
            cvar_estimates = self._calculate_cvar_estimates(final_values, simulation_config.confidence_levels)
            
            # 计算其他风险指标
            expected_return = np.mean(final_values)
            volatility = np.std(final_values)
            max_drawdown = self._calculate_max_drawdown(portfolio_paths)
            sharpe_ratio = self._calculate_sharpe_ratio(final_values)
            skewness = stats.skew(final_values)
            kurtosis = stats.kurtosis(final_values)
            tail_expectation = self._calculate_tail_expectation(final_values)
            probability_of_loss = np.mean(final_values < 0)
            
            # 检查收敛性
            convergence_achieved = self._check_convergence(final_values, simulation_config.num_simulations)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # 创建结果对象
            result = SimulationResult(
                simulation_id=simulation_id,
                config=simulation_config,
                timestamp=start_time,
                portfolio_paths=portfolio_paths,
                final_values=final_values,
                var_estimates=var_estimates,
                cvar_estimates=cvar_estimates,
                expected_return=expected_return,
                volatility=volatility,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                skewness=skewness,
                kurtosis=kurtosis,
                tail_expectation=tail_expectation,
                probability_of_loss=probability_of_loss,
                execution_time_seconds=execution_time,
                convergence_achieved=convergence_achieved
            )
            
            # 缓存结果
            self.simulation_results[simulation_id] = result
            
            self.logger.info(f"Monte Carlo simulation completed: {simulation_id}")
            self.logger.info(f"Expected Return: {expected_return:.4f}, Volatility: {volatility:.4f}")
            self.logger.info(f"VaR (95%): {var_estimates.get(0.95, 0):.4f}, CVaR (95%): {cvar_estimates.get(0.95, 0):.4f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in Monte Carlo simulation: {e}")
            raise
    
    async def _run_gbm_simulation(self, assets: List[AssetParameters], config: SimulationConfig) -> Tuple[np.ndarray, np.ndarray]:
        """几何布朗运动模拟"""
        dt = 1.0 / config.time_steps
        num_assets = len(assets)
        
        # 初始化数组
        portfolio_paths = np.zeros((config.num_simulations, config.time_steps + 1))
        
        # 计算初始组合价值
        initial_portfolio_value = sum(asset.initial_price * asset.weight for asset in assets)
        portfolio_paths[:, 0] = initial_portfolio_value
        
        # 构建相关矩阵
        correlation_matrix = self._build_correlation_matrix(assets)
        cholesky_matrix = np.linalg.cholesky(correlation_matrix)
        
        # 并行计算
        tasks = []
        chunk_size = max(1, config.num_simulations // self.config.get('max_workers', 4))
        
        for i in range(0, config.num_simulations, chunk_size):
            end_i = min(i + chunk_size, config.num_simulations)
            task = self.executor.submit(
                self._simulate_gbm_chunk,
                assets, i, end_i, config.time_steps, dt, cholesky_matrix
            )
            tasks.append(task)
        
        # 收集结果
        all_paths = []
        for task in as_completed(tasks):
            paths = task.result()
            all_paths.extend(paths)
        
        # 重新排序
        all_paths = np.array(all_paths)
        
        # 计算组合路径
        for sim in range(config.num_simulations):
            for t in range(1, config.time_steps + 1):
                portfolio_value = 0
                for j, asset in enumerate(assets):
                    asset_value = asset.initial_price * np.exp(all_paths[sim, j, t])
                    portfolio_value += asset_value * asset.weight
                portfolio_paths[sim, t] = portfolio_value
        
        final_values = portfolio_paths[:, -1] - initial_portfolio_value
        
        return portfolio_paths, final_values
    
    def _simulate_gbm_chunk(self, assets: List[AssetParameters], start_idx: int, end_idx: int, 
                           time_steps: int, dt: float, cholesky_matrix: np.ndarray) -> List[np.ndarray]:
        """模拟GBM路径块"""
        num_sims = end_idx - start_idx
        num_assets = len(assets)
        
        paths = []
        
        for sim in range(num_sims):
            # 初始化路径
            path = np.zeros((num_assets, time_steps + 1))
            
            for t in range(1, time_steps + 1):
                # 生成相关随机数
                random_vars = self.rng.randn(num_assets)
                correlated_vars = cholesky_matrix @ random_vars
                
                for j, asset in enumerate(assets):
                    # GBM公式
                    drift = (asset.mu - 0.5 * asset.sigma**2) * dt
                    diffusion = asset.sigma * np.sqrt(dt) * correlated_vars[j]
                    path[j, t] = path[j, t-1] + drift + diffusion
            
            paths.append(path)
        
        return paths
    
    async def _run_jump_diffusion_simulation(self, assets: List[AssetParameters], config: SimulationConfig) -> Tuple[np.ndarray, np.ndarray]:
        """跳跃扩散模拟（Merton模型）"""
        dt = 1.0 / config.time_steps
        num_assets = len(assets)
        
        portfolio_paths = np.zeros((config.num_simulations, config.time_steps + 1))
        initial_portfolio_value = sum(asset.initial_price * asset.weight for asset in assets)
        portfolio_paths[:, 0] = initial_portfolio_value
        
        # 构建相关矩阵
        correlation_matrix = self._build_correlation_matrix(assets)
        cholesky_matrix = np.linalg.cholesky(correlation_matrix)
        
        for sim in range(config.num_simulations):
            asset_paths = np.zeros((num_assets, config.time_steps + 1))
            
            for t in range(1, config.time_steps + 1):
                # 生成相关随机数
                random_vars = self.rng.randn(num_assets)
                correlated_vars = cholesky_matrix @ random_vars
                
                for j, asset in enumerate(assets):
                    # 连续部分（GBM）
                    drift = (asset.mu - 0.5 * asset.sigma**2) * dt
                    diffusion = asset.sigma * np.sqrt(dt) * correlated_vars[j]
                    
                    # 跳跃部分
                    jump_occurred = self.rng.poisson(asset.lambda_jump * dt)
                    jump_size = 0
                    if jump_occurred:
                        jump_size = self.rng.normal(asset.mu_jump, asset.sigma_jump)
                    
                    asset_paths[j, t] = asset_paths[j, t-1] + drift + diffusion + jump_size
            
            # 计算组合路径
            for t in range(1, config.time_steps + 1):
                portfolio_value = 0
                for j, asset in enumerate(assets):
                    asset_value = asset.initial_price * np.exp(asset_paths[j, t])
                    portfolio_value += asset_value * asset.weight
                portfolio_paths[sim, t] = portfolio_value
        
        final_values = portfolio_paths[:, -1] - initial_portfolio_value
        
        return portfolio_paths, final_values
    
    async def _run_heston_simulation(self, assets: List[AssetParameters], config: SimulationConfig) -> Tuple[np.ndarray, np.ndarray]:
        """Heston随机波动率模拟"""
        dt = 1.0 / config.time_steps
        num_assets = len(assets)
        
        portfolio_paths = np.zeros((config.num_simulations, config.time_steps + 1))
        initial_portfolio_value = sum(asset.initial_price * asset.weight for asset in assets)
        portfolio_paths[:, 0] = initial_portfolio_value
        
        for sim in range(config.num_simulations):
            asset_paths = np.zeros((num_assets, config.time_steps + 1))
            variance_paths = np.zeros((num_assets, config.time_steps + 1))
            
            # 初始化方差
            for j, asset in enumerate(assets):
                variance_paths[j, 0] = asset.sigma**2
            
            for t in range(1, config.time_steps + 1):
                for j, asset in enumerate(assets):
                    # 生成相关随机数
                    z1 = self.rng.randn()
                    z2 = self.rng.randn()
                    z2_corr = asset.rho * z1 + np.sqrt(1 - asset.rho**2) * z2
                    
                    # 方差过程
                    variance = max(variance_paths[j, t-1], 0)
                    dv = asset.kappa * (asset.theta - variance) * dt + asset.xi * np.sqrt(variance) * np.sqrt(dt) * z2_corr
                    variance_paths[j, t] = variance + dv
                    
                    # 价格过程
                    ds = asset.mu * dt + np.sqrt(variance) * np.sqrt(dt) * z1
                    asset_paths[j, t] = asset_paths[j, t-1] + ds
            
            # 计算组合路径
            for t in range(1, config.time_steps + 1):
                portfolio_value = 0
                for j, asset in enumerate(assets):
                    asset_value = asset.initial_price * np.exp(asset_paths[j, t])
                    portfolio_value += asset_value * asset.weight
                portfolio_paths[sim, t] = portfolio_value
        
        final_values = portfolio_paths[:, -1] - initial_portfolio_value
        
        return portfolio_paths, final_values
    
    async def _run_levy_simulation(self, assets: List[AssetParameters], config: SimulationConfig) -> Tuple[np.ndarray, np.ndarray]:
        """Levy过程模拟"""
        # 简化的Levy过程实现（使用NIG分布）
        dt = 1.0 / config.time_steps
        num_assets = len(assets)
        
        portfolio_paths = np.zeros((config.num_simulations, config.time_steps + 1))
        initial_portfolio_value = sum(asset.initial_price * asset.weight for asset in assets)
        portfolio_paths[:, 0] = initial_portfolio_value
        
        for sim in range(config.num_simulations):
            asset_paths = np.zeros((num_assets, config.time_steps + 1))
            
            for t in range(1, config.time_steps + 1):
                for j, asset in enumerate(assets):
                    # 使用正态逆高斯分布
                    alpha = 2.0  # 形状参数
                    beta = 0.5   # 偏斜参数
                    delta = dt   # 尺度参数
                    
                    # 生成NIG随机数（简化实现）
                    ig_sample = self._sample_inverse_gaussian(delta, alpha**2 - beta**2)
                    normal_sample = self.rng.normal(0, 1)
                    
                    levy_increment = beta * ig_sample + np.sqrt(ig_sample) * normal_sample
                    
                    # 添加漂移
                    drift = asset.mu * dt
                    
                    asset_paths[j, t] = asset_paths[j, t-1] + drift + levy_increment
            
            # 计算组合路径
            for t in range(1, config.time_steps + 1):
                portfolio_value = 0
                for j, asset in enumerate(assets):
                    asset_value = asset.initial_price * np.exp(asset_paths[j, t])
                    portfolio_value += asset_value * asset.weight
                portfolio_paths[sim, t] = portfolio_value
        
        final_values = portfolio_paths[:, -1] - initial_portfolio_value
        
        return portfolio_paths, final_values
    
    async def _run_copula_simulation(self, assets: List[AssetParameters], config: SimulationConfig) -> Tuple[np.ndarray, np.ndarray]:
        """Copula模拟"""
        dt = 1.0 / config.time_steps
        num_assets = len(assets)
        
        portfolio_paths = np.zeros((config.num_simulations, config.time_steps + 1))
        initial_portfolio_value = sum(asset.initial_price * asset.weight for asset in assets)
        portfolio_paths[:, 0] = initial_portfolio_value
        
        # 使用高斯Copula
        correlation_matrix = self._build_correlation_matrix(assets)
        
        for sim in range(config.num_simulations):
            asset_paths = np.zeros((num_assets, config.time_steps + 1))
            
            for t in range(1, config.time_steps + 1):
                # 生成相关正态随机数
                multivariate_normal = self.rng.multivariate_normal(
                    np.zeros(num_assets), correlation_matrix
                )
                
                # 转换为均匀分布
                uniform_vars = stats.norm.cdf(multivariate_normal)
                
                for j, asset in enumerate(assets):
                    # 使用边际分布（这里使用Student-t分布）
                    df = 5  # 自由度
                    t_var = stats.t.ppf(uniform_vars[j], df)
                    
                    # 标准化
                    standardized_var = t_var / np.sqrt(df / (df - 2))
                    
                    # 应用到价格路径
                    drift = (asset.mu - 0.5 * asset.sigma**2) * dt
                    diffusion = asset.sigma * np.sqrt(dt) * standardized_var
                    
                    asset_paths[j, t] = asset_paths[j, t-1] + drift + diffusion
            
            # 计算组合路径
            for t in range(1, config.time_steps + 1):
                portfolio_value = 0
                for j, asset in enumerate(assets):
                    asset_value = asset.initial_price * np.exp(asset_paths[j, t])
                    portfolio_value += asset_value * asset.weight
                portfolio_paths[sim, t] = portfolio_value
        
        final_values = portfolio_paths[:, -1] - initial_portfolio_value
        
        return portfolio_paths, final_values
    
    async def _run_regime_switching_simulation(self, assets: List[AssetParameters], config: SimulationConfig) -> Tuple[np.ndarray, np.ndarray]:
        """制度转换模拟"""
        dt = 1.0 / config.time_steps
        num_assets = len(assets)
        
        # 定义两个制度
        regime_params = {
            0: {'mu_multiplier': 1.0, 'sigma_multiplier': 1.0},    # 正常制度
            1: {'mu_multiplier': -0.5, 'sigma_multiplier': 2.0}    # 危机制度
        }
        
        # 转换概率矩阵
        transition_matrix = np.array([
            [0.95, 0.05],  # 从正常制度转换的概率
            [0.20, 0.80]   # 从危机制度转换的概率
        ])
        
        portfolio_paths = np.zeros((config.num_simulations, config.time_steps + 1))
        initial_portfolio_value = sum(asset.initial_price * asset.weight for asset in assets)
        portfolio_paths[:, 0] = initial_portfolio_value
        
        for sim in range(config.num_simulations):
            asset_paths = np.zeros((num_assets, config.time_steps + 1))
            current_regime = 0  # 开始于正常制度
            
            for t in range(1, config.time_steps + 1):
                # 制度转换
                if self.rng.rand() < transition_matrix[current_regime, 1-current_regime]:
                    current_regime = 1 - current_regime
                
                regime_param = regime_params[current_regime]
                
                for j, asset in enumerate(assets):
                    # 根据当前制度调整参数
                    adjusted_mu = asset.mu * regime_param['mu_multiplier']
                    adjusted_sigma = asset.sigma * regime_param['sigma_multiplier']
                    
                    # GBM with regime switching
                    drift = (adjusted_mu - 0.5 * adjusted_sigma**2) * dt
                    diffusion = adjusted_sigma * np.sqrt(dt) * self.rng.randn()
                    
                    asset_paths[j, t] = asset_paths[j, t-1] + drift + diffusion
            
            # 计算组合路径
            for t in range(1, config.time_steps + 1):
                portfolio_value = 0
                for j, asset in enumerate(assets):
                    asset_value = asset.initial_price * np.exp(asset_paths[j, t])
                    portfolio_value += asset_value * asset.weight
                portfolio_paths[sim, t] = portfolio_value
        
        final_values = portfolio_paths[:, -1] - initial_portfolio_value
        
        return portfolio_paths, final_values
    
    def _build_correlation_matrix(self, assets: List[AssetParameters]) -> np.ndarray:
        """构建相关矩阵"""
        n = len(assets)
        correlation_matrix = np.eye(n)
        
        # 简化的相关矩阵生成
        for i in range(n):
            for j in range(i+1, n):
                # 基于资产类型设置相关性
                if 'tech' in assets[i].symbol.lower() and 'tech' in assets[j].symbol.lower():
                    correlation = 0.7
                elif 'utility' in assets[i].symbol.lower() and 'utility' in assets[j].symbol.lower():
                    correlation = 0.5
                else:
                    correlation = 0.3
                
                correlation_matrix[i, j] = correlation
                correlation_matrix[j, i] = correlation
        
        return correlation_matrix
    
    def _sample_inverse_gaussian(self, mu: float, lambda_param: float) -> float:
        """采样逆高斯分布"""
        # 简化的逆高斯采样
        nu = self.rng.randn()
        y = nu**2
        
        x = mu + (mu**2 * y) / (2 * lambda_param) - (mu / (2 * lambda_param)) * np.sqrt(4 * mu * lambda_param * y + mu**2 * y**2)
        
        test = self.rng.rand()
        if test <= mu / (mu + x):
            return x
        else:
            return mu**2 / x
    
    def _calculate_var_estimates(self, final_values: np.ndarray, confidence_levels: List[float]) -> Dict[float, float]:
        """计算VaR估计"""
        var_estimates = {}
        
        for confidence_level in confidence_levels:
            percentile = (1 - confidence_level) * 100
            var_estimate = np.percentile(final_values, percentile)
            var_estimates[confidence_level] = var_estimate
        
        return var_estimates
    
    def _calculate_cvar_estimates(self, final_values: np.ndarray, confidence_levels: List[float]) -> Dict[float, float]:
        """计算CVaR估计"""
        cvar_estimates = {}
        
        for confidence_level in confidence_levels:
            percentile = (1 - confidence_level) * 100
            var_threshold = np.percentile(final_values, percentile)
            
            # CVaR是低于VaR的损失的条件期望
            tail_losses = final_values[final_values <= var_threshold]
            cvar_estimate = np.mean(tail_losses) if len(tail_losses) > 0 else var_threshold
            
            cvar_estimates[confidence_level] = cvar_estimate
        
        return cvar_estimates
    
    def _calculate_max_drawdown(self, portfolio_paths: np.ndarray) -> float:
        """计算最大回撤"""
        max_drawdowns = []
        
        for sim in range(portfolio_paths.shape[0]):
            path = portfolio_paths[sim, :]
            cumulative_max = np.maximum.accumulate(path)
            drawdown = (path - cumulative_max) / cumulative_max
            max_drawdown = np.min(drawdown)
            max_drawdowns.append(max_drawdown)
        
        return np.mean(max_drawdowns)
    
    def _calculate_sharpe_ratio(self, final_values: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """计算夏普比率"""
        excess_return = np.mean(final_values) - risk_free_rate
        volatility = np.std(final_values)
        
        if volatility == 0:
            return 0
        
        return excess_return / volatility
    
    def _calculate_tail_expectation(self, final_values: np.ndarray, threshold: float = 0.05) -> float:
        """计算尾部期望"""
        tail_threshold = np.percentile(final_values, threshold * 100)
        tail_values = final_values[final_values <= tail_threshold]
        
        return np.mean(tail_values) if len(tail_values) > 0 else 0
    
    def _check_convergence(self, final_values: np.ndarray, num_simulations: int) -> bool:
        """检查收敛性"""
        if num_simulations < 1000:
            return False
        
        # 检查均值收敛
        batch_size = num_simulations // 10
        batch_means = []
        
        for i in range(0, num_simulations, batch_size):
            batch = final_values[i:i+batch_size]
            batch_means.append(np.mean(batch))
        
        # 计算批次间的变异系数
        cv = np.std(batch_means) / np.abs(np.mean(batch_means))
        
        return cv < 0.05  # 变异系数小于5%认为收敛
    
    def get_simulation_result(self, simulation_id: str) -> Optional[SimulationResult]:
        """获取模拟结果"""
        return self.simulation_results.get(simulation_id)
    
    def get_all_results(self) -> Dict[str, SimulationResult]:
        """获取所有模拟结果"""
        return self.simulation_results.copy()
    
    def calculate_scenario_probabilities(self, result: SimulationResult, scenarios: Dict[str, float]) -> Dict[str, float]:
        """计算情景概率"""
        probabilities = {}
        
        for scenario_name, threshold in scenarios.items():
            if 'loss' in scenario_name.lower():
                prob = np.mean(result.final_values < threshold)
            else:
                prob = np.mean(result.final_values > threshold)
            
            probabilities[scenario_name] = prob
        
        return probabilities
    
    def generate_stress_scenarios(self, result: SimulationResult, num_scenarios: int = 100) -> List[Dict[str, Any]]:
        """生成压力情景"""
        # 选择最坏的情景
        worst_indices = np.argsort(result.final_values)[:num_scenarios]
        
        scenarios = []
        for i, idx in enumerate(worst_indices):
            scenario = {
                'scenario_id': f"stress_{i+1}",
                'final_value': result.final_values[idx],
                'portfolio_path': result.portfolio_paths[idx, :],
                'loss_percentage': result.final_values[idx] / result.portfolio_paths[idx, 0] * 100,
                'max_drawdown': self._calculate_single_path_drawdown(result.portfolio_paths[idx, :])
            }
            scenarios.append(scenario)
        
        return scenarios
    
    def _calculate_single_path_drawdown(self, path: np.ndarray) -> float:
        """计算单条路径的最大回撤"""
        cumulative_max = np.maximum.accumulate(path)
        drawdown = (path - cumulative_max) / cumulative_max
        return np.min(drawdown)
    
    def export_results_to_dataframe(self, simulation_id: str) -> Optional[pd.DataFrame]:
        """导出结果到DataFrame"""
        result = self.get_simulation_result(simulation_id)
        
        if not result:
            return None
        
        # 创建汇总数据
        summary_data = {
            'simulation_id': result.simulation_id,
            'simulation_type': result.config.simulation_type.value,
            'num_simulations': result.config.num_simulations,
            'expected_return': result.expected_return,
            'volatility': result.volatility,
            'max_drawdown': result.max_drawdown,
            'sharpe_ratio': result.sharpe_ratio,
            'skewness': result.skewness,
            'kurtosis': result.kurtosis,
            'probability_of_loss': result.probability_of_loss,
            'execution_time': result.execution_time_seconds,
            'convergence_achieved': result.convergence_achieved
        }
        
        # 添加VaR和CVaR
        for confidence_level in result.config.confidence_levels:
            summary_data[f'var_{confidence_level}'] = result.var_estimates.get(confidence_level, 0)
            summary_data[f'cvar_{confidence_level}'] = result.cvar_estimates.get(confidence_level, 0)
        
        return pd.DataFrame([summary_data])
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """获取性能统计"""
        if not self.simulation_results:
            return {}
        
        results = list(self.simulation_results.values())
        
        return {
            'total_simulations': len(results),
            'avg_execution_time': np.mean([r.execution_time_seconds for r in results]),
            'total_execution_time': sum(r.execution_time_seconds for r in results),
            'convergence_rate': np.mean([r.convergence_achieved for r in results]),
            'simulation_types': list(set(r.config.simulation_type.value for r in results))
        }
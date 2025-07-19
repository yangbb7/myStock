import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, multivariate_normal
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

class ForwardAnalysisMethod(Enum):
    MONTE_CARLO = "monte_carlo"
    SCENARIO_ANALYSIS = "scenario_analysis"
    STRESS_TESTING = "stress_testing"
    SENSITIVITY_ANALYSIS = "sensitivity_analysis"
    WHAT_IF_ANALYSIS = "what_if_analysis"
    PREDICTIVE_MODELING = "predictive_modeling"
    MACHINE_LEARNING = "machine_learning"
    REGIME_SWITCHING = "regime_switching"
    FACTOR_MODELING = "factor_modeling"
    ECONOMETRIC_MODELING = "econometric_modeling"

class ScenarioType(Enum):
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    RECESSION = "recession"
    RECOVERY = "recovery"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    INTEREST_RATE_RISE = "interest_rate_rise"
    INTEREST_RATE_FALL = "interest_rate_fall"
    INFLATION_SHOCK = "inflation_shock"
    CURRENCY_CRISIS = "currency_crisis"
    MARKET_CRASH = "market_crash"
    CUSTOM = "custom"

class RiskFactor(Enum):
    MARKET_RISK = "market_risk"
    CREDIT_RISK = "credit_risk"
    LIQUIDITY_RISK = "liquidity_risk"
    OPERATIONAL_RISK = "operational_risk"
    INTEREST_RATE_RISK = "interest_rate_risk"
    CURRENCY_RISK = "currency_risk"
    VOLATILITY_RISK = "volatility_risk"
    CONCENTRATION_RISK = "concentration_risk"
    COUNTERPARTY_RISK = "counterparty_risk"
    REGULATORY_RISK = "regulatory_risk"

@dataclass
class ForwardScenario:
    """前瞻性分析场景"""
    scenario_id: str
    scenario_type: ScenarioType
    name: str
    description: str
    probability: float
    horizon_days: int
    market_conditions: Dict[str, float]
    risk_factors: Dict[RiskFactor, float]
    macroeconomic_variables: Dict[str, float]
    asset_correlations: np.ndarray = None
    volatility_multiplier: float = 1.0
    return_multiplier: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ForwardPrediction:
    """前瞻性预测"""
    asset_id: str
    prediction_date: datetime
    horizon_days: int
    predicted_price: float
    predicted_return: float
    confidence_interval: Tuple[float, float]
    probability_distribution: Dict[str, float]
    risk_metrics: Dict[str, float]
    model_used: str
    model_confidence: float
    feature_importance: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PortfolioProjection:
    """投资组合预测"""
    portfolio_id: str
    projection_date: datetime
    horizon_days: int
    projected_value: float
    projected_return: float
    projected_volatility: float
    var_95: float
    var_99: float
    cvar_95: float
    maximum_drawdown: float
    sharpe_ratio: float
    scenario_analysis: Dict[str, float]
    stress_test_results: Dict[str, float]
    optimization_suggestions: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ForwardAnalysisResult:
    """前瞻性分析结果"""
    analysis_id: str
    analysis_type: ForwardAnalysisMethod
    creation_date: datetime
    asset_predictions: List[ForwardPrediction]
    portfolio_projections: List[PortfolioProjection]
    scenario_outcomes: Dict[str, Dict[str, float]]
    risk_decomposition: Dict[str, float]
    sensitivity_analysis: Dict[str, float]
    optimization_recommendations: List[str]
    model_performance: Dict[str, float]
    analysis_confidence: float
    limitations: List[str]
    assumptions: List[str]
    methodology: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class ForwardAnalysisEngine:
    """
    前瞻性分析引擎
    
    提供多种前瞻性分析方法，包括蒙特卡洛模拟、情景分析、
    压力测试、敏感性分析等，帮助投资者预测未来的市场表现。
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 模型参数
        self.default_horizon = config.get('default_horizon', 252)  # 一年
        self.simulation_runs = config.get('simulation_runs', 10000)
        self.confidence_levels = config.get('confidence_levels', [0.95, 0.99])
        
        # 模型配置
        self.random_seed = config.get('random_seed', 42)
        self.use_parallel_processing = config.get('use_parallel_processing', True)
        self.max_workers = config.get('max_workers', 4)
        
        # 风险参数
        self.risk_free_rate = config.get('risk_free_rate', 0.03)
        self.market_risk_premium = config.get('market_risk_premium', 0.06)
        
        # 场景库
        self.scenarios = {}
        self._initialize_default_scenarios()
        
        # 缓存
        self.prediction_cache = {}
        self.model_cache = {}
        
        np.random.seed(self.random_seed)
    
    def _initialize_default_scenarios(self):
        """初始化默认场景"""
        # 牛市场景
        self.scenarios['bull_market'] = ForwardScenario(
            scenario_id='bull_market',
            scenario_type=ScenarioType.BULL_MARKET,
            name='牛市场景',
            description='股票市场持续上涨，经济增长强劲',
            probability=0.25,
            horizon_days=252,
            market_conditions={
                'market_return': 0.15,
                'market_volatility': 0.12,
                'bond_yield': 0.04,
                'vix': 15.0
            },
            risk_factors={
                RiskFactor.MARKET_RISK: 0.8,
                RiskFactor.CREDIT_RISK: 0.3,
                RiskFactor.LIQUIDITY_RISK: 0.2
            },
            macroeconomic_variables={
                'gdp_growth': 0.04,
                'inflation': 0.025,
                'unemployment': 0.04
            },
            return_multiplier=1.5,
            volatility_multiplier=0.8
        )
        
        # 熊市场景
        self.scenarios['bear_market'] = ForwardScenario(
            scenario_id='bear_market',
            scenario_type=ScenarioType.BEAR_MARKET,
            name='熊市场景',
            description='股票市场持续下跌，经济衰退',
            probability=0.15,
            horizon_days=252,
            market_conditions={
                'market_return': -0.20,
                'market_volatility': 0.25,
                'bond_yield': 0.02,
                'vix': 35.0
            },
            risk_factors={
                RiskFactor.MARKET_RISK: 1.5,
                RiskFactor.CREDIT_RISK: 1.2,
                RiskFactor.LIQUIDITY_RISK: 1.8
            },
            macroeconomic_variables={
                'gdp_growth': -0.02,
                'inflation': 0.015,
                'unemployment': 0.08
            },
            return_multiplier=0.3,
            volatility_multiplier=2.0
        )
        
        # 高波动场景
        self.scenarios['high_volatility'] = ForwardScenario(
            scenario_id='high_volatility',
            scenario_type=ScenarioType.HIGH_VOLATILITY,
            name='高波动场景',
            description='市场波动性显著增加',
            probability=0.20,
            horizon_days=126,
            market_conditions={
                'market_return': 0.05,
                'market_volatility': 0.30,
                'bond_yield': 0.03,
                'vix': 40.0
            },
            risk_factors={
                RiskFactor.MARKET_RISK: 1.3,
                RiskFactor.VOLATILITY_RISK: 2.0,
                RiskFactor.LIQUIDITY_RISK: 1.4
            },
            macroeconomic_variables={
                'gdp_growth': 0.02,
                'inflation': 0.03,
                'unemployment': 0.05
            },
            return_multiplier=1.0,
            volatility_multiplier=2.5
        )
    
    async def run_forward_analysis(self, 
                                 portfolio_weights: np.ndarray,
                                 asset_returns: pd.DataFrame,
                                 method: ForwardAnalysisMethod = ForwardAnalysisMethod.MONTE_CARLO,
                                 horizon_days: int = None,
                                 scenarios: List[str] = None,
                                 **kwargs) -> ForwardAnalysisResult:
        """运行前瞻性分析"""
        try:
            if horizon_days is None:
                horizon_days = self.default_horizon
            
            analysis_id = f"forward_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # 根据方法执行不同的分析
            if method == ForwardAnalysisMethod.MONTE_CARLO:
                result = await self._run_monte_carlo_analysis(
                    portfolio_weights, asset_returns, horizon_days, **kwargs
                )
            elif method == ForwardAnalysisMethod.SCENARIO_ANALYSIS:
                result = await self._run_scenario_analysis(
                    portfolio_weights, asset_returns, scenarios, **kwargs
                )
            elif method == ForwardAnalysisMethod.STRESS_TESTING:
                result = await self._run_stress_testing(
                    portfolio_weights, asset_returns, **kwargs
                )
            elif method == ForwardAnalysisMethod.SENSITIVITY_ANALYSIS:
                result = await self._run_sensitivity_analysis(
                    portfolio_weights, asset_returns, **kwargs
                )
            else:
                raise ValueError(f"Unsupported analysis method: {method}")
            
            # 构建分析结果
            analysis_result = ForwardAnalysisResult(
                analysis_id=analysis_id,
                analysis_type=method,
                creation_date=datetime.now(),
                asset_predictions=result.get('asset_predictions', []),
                portfolio_projections=result.get('portfolio_projections', []),
                scenario_outcomes=result.get('scenario_outcomes', {}),
                risk_decomposition=result.get('risk_decomposition', {}),
                sensitivity_analysis=result.get('sensitivity_analysis', {}),
                optimization_recommendations=result.get('optimization_recommendations', []),
                model_performance=result.get('model_performance', {}),
                analysis_confidence=result.get('analysis_confidence', 0.8),
                limitations=result.get('limitations', []),
                assumptions=result.get('assumptions', []),
                methodology=method.value,
                metadata=result.get('metadata', {})
            )
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"前瞻性分析失败: {e}")
            raise
    
    async def _run_monte_carlo_analysis(self,
                                      portfolio_weights: np.ndarray,
                                      asset_returns: pd.DataFrame,
                                      horizon_days: int,
                                      **kwargs) -> Dict[str, Any]:
        """运行蒙特卡洛分析"""
        try:
            # 计算资产的统计特征
            mean_returns = asset_returns.mean().values
            cov_matrix = asset_returns.cov().values
            
            # 蒙特卡洛模拟
            simulated_returns = []
            simulated_values = []
            
            for _ in range(self.simulation_runs):
                # 生成随机收益率路径
                random_returns = np.random.multivariate_normal(
                    mean_returns, cov_matrix, horizon_days
                )
                
                # 计算投资组合收益率
                portfolio_returns = np.dot(random_returns, portfolio_weights)
                
                # 计算累积收益率
                cumulative_return = np.prod(1 + portfolio_returns) - 1
                final_value = (1 + cumulative_return)
                
                simulated_returns.append(cumulative_return)
                simulated_values.append(final_value)
            
            # 计算统计指标
            simulated_returns = np.array(simulated_returns)
            simulated_values = np.array(simulated_values)
            
            # 投资组合预测
            portfolio_projection = PortfolioProjection(
                portfolio_id="current_portfolio",
                projection_date=datetime.now(),
                horizon_days=horizon_days,
                projected_value=np.mean(simulated_values),
                projected_return=np.mean(simulated_returns),
                projected_volatility=np.std(simulated_returns),
                var_95=np.percentile(simulated_returns, 5),
                var_99=np.percentile(simulated_returns, 1),
                cvar_95=np.mean(simulated_returns[simulated_returns <= np.percentile(simulated_returns, 5)]),
                maximum_drawdown=self._calculate_max_drawdown_mc(simulated_returns),
                sharpe_ratio=np.mean(simulated_returns) / np.std(simulated_returns) if np.std(simulated_returns) > 0 else 0,
                scenario_analysis={},
                stress_test_results={},
                optimization_suggestions=[]
            )
            
            # 资产预测
            asset_predictions = []
            for i, asset in enumerate(asset_returns.columns):
                asset_simulated_returns = []
                for _ in range(self.simulation_runs):
                    random_returns = np.random.multivariate_normal(
                        mean_returns, cov_matrix, horizon_days
                    )
                    asset_return = np.prod(1 + random_returns[:, i]) - 1
                    asset_simulated_returns.append(asset_return)
                
                asset_simulated_returns = np.array(asset_simulated_returns)
                
                prediction = ForwardPrediction(
                    asset_id=asset,
                    prediction_date=datetime.now(),
                    horizon_days=horizon_days,
                    predicted_price=1.0 + np.mean(asset_simulated_returns),
                    predicted_return=np.mean(asset_simulated_returns),
                    confidence_interval=(
                        np.percentile(asset_simulated_returns, 5),
                        np.percentile(asset_simulated_returns, 95)
                    ),
                    probability_distribution={},
                    risk_metrics={
                        'volatility': np.std(asset_simulated_returns),
                        'var_95': np.percentile(asset_simulated_returns, 5),
                        'var_99': np.percentile(asset_simulated_returns, 1)
                    },
                    model_used='Monte Carlo',
                    model_confidence=0.85
                )
                asset_predictions.append(prediction)
            
            return {
                'asset_predictions': asset_predictions,
                'portfolio_projections': [portfolio_projection],
                'scenario_outcomes': {},
                'risk_decomposition': self._calculate_risk_decomposition(portfolio_weights, cov_matrix),
                'model_performance': {
                    'simulation_runs': self.simulation_runs,
                    'convergence_metric': np.std(simulated_returns[-1000:]) / np.std(simulated_returns)
                },
                'analysis_confidence': 0.85,
                'limitations': [
                    '假设收益率服从正态分布',
                    '未考虑市场制度变化',
                    '基于历史数据的协方差矩阵'
                ],
                'assumptions': [
                    '收益率独立同分布',
                    '交易成本忽略不计',
                    '流动性充足'
                ]
            }
            
        except Exception as e:
            self.logger.error(f"蒙特卡洛分析失败: {e}")
            raise
    
    async def _run_scenario_analysis(self,
                                   portfolio_weights: np.ndarray,
                                   asset_returns: pd.DataFrame,
                                   scenarios: List[str] = None,
                                   **kwargs) -> Dict[str, Any]:
        """运行情景分析"""
        try:
            if scenarios is None:
                scenarios = list(self.scenarios.keys())
            
            scenario_outcomes = {}
            portfolio_projections = []
            
            for scenario_name in scenarios:
                if scenario_name not in self.scenarios:
                    continue
                
                scenario = self.scenarios[scenario_name]
                
                # 调整收益率和波动率
                adjusted_returns = asset_returns.mean() * scenario.return_multiplier
                adjusted_cov = asset_returns.cov() * (scenario.volatility_multiplier ** 2)
                
                # 计算情景下的投资组合表现
                portfolio_return = np.dot(adjusted_returns.values, portfolio_weights)
                portfolio_volatility = np.sqrt(
                    np.dot(portfolio_weights.T, np.dot(adjusted_cov.values, portfolio_weights))
                )
                
                # VaR计算
                var_95 = portfolio_return - 1.645 * portfolio_volatility
                var_99 = portfolio_return - 2.326 * portfolio_volatility
                
                scenario_outcomes[scenario_name] = {
                    'expected_return': portfolio_return,
                    'volatility': portfolio_volatility,
                    'var_95': var_95,
                    'var_99': var_99,
                    'probability': scenario.probability,
                    'sharpe_ratio': (portfolio_return - self.risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
                }
                
                # 创建投资组合预测
                projection = PortfolioProjection(
                    portfolio_id=f"portfolio_{scenario_name}",
                    projection_date=datetime.now(),
                    horizon_days=scenario.horizon_days,
                    projected_value=1.0 + portfolio_return,
                    projected_return=portfolio_return,
                    projected_volatility=portfolio_volatility,
                    var_95=var_95,
                    var_99=var_99,
                    cvar_95=var_95 - portfolio_volatility * 0.5,
                    maximum_drawdown=min(-0.05, var_99 * 1.5),
                    sharpe_ratio=(portfolio_return - self.risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0,
                    scenario_analysis=scenario_outcomes,
                    stress_test_results={},
                    optimization_suggestions=[]
                )
                portfolio_projections.append(projection)
            
            return {
                'portfolio_projections': portfolio_projections,
                'scenario_outcomes': scenario_outcomes,
                'risk_decomposition': self._calculate_risk_decomposition(portfolio_weights, asset_returns.cov().values),
                'model_performance': {
                    'scenarios_analyzed': len(scenarios),
                    'coverage_probability': sum(self.scenarios[s].probability for s in scenarios if s in self.scenarios)
                },
                'analysis_confidence': 0.80,
                'limitations': [
                    '场景设定基于历史经验',
                    '未考虑场景之间的相关性',
                    '场景概率主观设定'
                ],
                'assumptions': [
                    '场景独立发生',
                    '资产相关性保持稳定',
                    '无套利机会'
                ]
            }
            
        except Exception as e:
            self.logger.error(f"情景分析失败: {e}")
            raise
    
    async def _run_stress_testing(self,
                                portfolio_weights: np.ndarray,
                                asset_returns: pd.DataFrame,
                                **kwargs) -> Dict[str, Any]:
        """运行压力测试"""
        try:
            stress_scenarios = {
                'market_crash': {
                    'market_shock': -0.30,
                    'volatility_shock': 2.0,
                    'correlation_shock': 0.2
                },
                'credit_crisis': {
                    'credit_spread_shock': 0.05,
                    'liquidity_shock': 0.15,
                    'flight_to_quality': 0.20
                },
                'interest_rate_shock': {
                    'rate_shock': 0.03,
                    'yield_curve_shock': 0.02,
                    'duration_risk': 0.10
                },
                'currency_crisis': {
                    'fx_shock': 0.25,
                    'emerging_market_shock': 0.40,
                    'commodity_shock': 0.30
                }
            }
            
            stress_results = {}
            
            for stress_name, stress_params in stress_scenarios.items():
                # 应用压力冲击
                stressed_returns = asset_returns.copy()
                
                if 'market_shock' in stress_params:
                    # 市场冲击
                    shock_magnitude = stress_params['market_shock']
                    stressed_returns += shock_magnitude
                
                if 'volatility_shock' in stress_params:
                    # 波动率冲击
                    vol_multiplier = stress_params['volatility_shock']
                    stressed_cov = asset_returns.cov() * (vol_multiplier ** 2)
                else:
                    stressed_cov = asset_returns.cov()
                
                # 计算压力测试结果
                stressed_portfolio_return = np.dot(stressed_returns.mean().values, portfolio_weights)
                stressed_portfolio_vol = np.sqrt(
                    np.dot(portfolio_weights.T, np.dot(stressed_cov.values, portfolio_weights))
                )
                
                stress_results[stress_name] = {
                    'stressed_return': stressed_portfolio_return,
                    'stressed_volatility': stressed_portfolio_vol,
                    'stressed_var_95': stressed_portfolio_return - 1.645 * stressed_portfolio_vol,
                    'stressed_var_99': stressed_portfolio_return - 2.326 * stressed_portfolio_vol,
                    'return_impact': stressed_portfolio_return - np.dot(asset_returns.mean().values, portfolio_weights),
                    'vol_impact': stressed_portfolio_vol - np.sqrt(np.dot(portfolio_weights.T, np.dot(asset_returns.cov().values, portfolio_weights)))
                }
            
            return {
                'stress_test_results': stress_results,
                'risk_decomposition': self._calculate_risk_decomposition(portfolio_weights, asset_returns.cov().values),
                'model_performance': {
                    'stress_scenarios': len(stress_scenarios),
                    'max_loss_scenario': max(stress_results.keys(), key=lambda x: abs(stress_results[x]['stressed_var_99']))
                },
                'analysis_confidence': 0.75,
                'limitations': [
                    '压力情景基于历史极端事件',
                    '未考虑流动性风险',
                    '冲击大小主观设定'
                ],
                'assumptions': [
                    '冲击立即发生',
                    '市场结构保持不变',
                    '无政策干预'
                ]
            }
            
        except Exception as e:
            self.logger.error(f"压力测试失败: {e}")
            raise
    
    async def _run_sensitivity_analysis(self,
                                      portfolio_weights: np.ndarray,
                                      asset_returns: pd.DataFrame,
                                      **kwargs) -> Dict[str, Any]:
        """运行敏感性分析"""
        try:
            base_return = np.dot(asset_returns.mean().values, portfolio_weights)
            base_volatility = np.sqrt(
                np.dot(portfolio_weights.T, np.dot(asset_returns.cov().values, portfolio_weights))
            )
            
            sensitivity_results = {}
            
            # 收益率敏感性
            return_shocks = [-0.05, -0.02, -0.01, 0.01, 0.02, 0.05]
            for shock in return_shocks:
                shocked_returns = asset_returns.mean() + shock
                shocked_portfolio_return = np.dot(shocked_returns.values, portfolio_weights)
                sensitivity_results[f'return_shock_{shock}'] = {
                    'shocked_return': shocked_portfolio_return,
                    'return_sensitivity': shocked_portfolio_return - base_return
                }
            
            # 波动率敏感性
            vol_multipliers = [0.5, 0.75, 1.25, 1.5, 2.0]
            for multiplier in vol_multipliers:
                shocked_cov = asset_returns.cov() * (multiplier ** 2)
                shocked_volatility = np.sqrt(
                    np.dot(portfolio_weights.T, np.dot(shocked_cov.values, portfolio_weights))
                )
                sensitivity_results[f'vol_shock_{multiplier}'] = {
                    'shocked_volatility': shocked_volatility,
                    'volatility_sensitivity': shocked_volatility - base_volatility
                }
            
            # 相关性敏感性
            correlation_shocks = [0.1, 0.2, 0.3, -0.1, -0.2]
            for shock in correlation_shocks:
                shocked_corr = asset_returns.corr() + shock
                shocked_corr = np.clip(shocked_corr, -0.99, 0.99)
                
                # 重构协方差矩阵
                std_devs = asset_returns.std()
                shocked_cov = np.outer(std_devs, std_devs) * shocked_corr
                
                shocked_volatility = np.sqrt(
                    np.dot(portfolio_weights.T, np.dot(shocked_cov.values, portfolio_weights))
                )
                sensitivity_results[f'corr_shock_{shock}'] = {
                    'shocked_volatility': shocked_volatility,
                    'correlation_sensitivity': shocked_volatility - base_volatility
                }
            
            return {
                'sensitivity_analysis': sensitivity_results,
                'risk_decomposition': self._calculate_risk_decomposition(portfolio_weights, asset_returns.cov().values),
                'model_performance': {
                    'sensitivity_factors': len(sensitivity_results),
                    'max_sensitivity': max(abs(v.get('return_sensitivity', 0)) for v in sensitivity_results.values())
                },
                'analysis_confidence': 0.90,
                'limitations': [
                    '单因子敏感性分析',
                    '线性假设',
                    '其他因子保持不变'
                ],
                'assumptions': [
                    '因子独立变化',
                    '无交互效应',
                    '市场有效性'
                ]
            }
            
        except Exception as e:
            self.logger.error(f"敏感性分析失败: {e}")
            raise
    
    def _calculate_risk_decomposition(self, weights: np.ndarray, cov_matrix: np.ndarray) -> Dict[str, float]:
        """计算风险分解"""
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        
        # 边际风险贡献
        marginal_contrib = np.dot(cov_matrix, weights)
        
        # 风险贡献
        risk_contrib = weights * marginal_contrib
        
        # 风险贡献占比
        risk_contrib_pct = risk_contrib / portfolio_variance
        
        return {
            'portfolio_variance': portfolio_variance,
            'portfolio_volatility': np.sqrt(portfolio_variance),
            'total_risk_contribution': np.sum(risk_contrib),
            'max_risk_contribution': np.max(risk_contrib_pct),
            'risk_concentration': np.sum(risk_contrib_pct ** 2)
        }
    
    def _calculate_max_drawdown_mc(self, returns: np.ndarray) -> float:
        """计算蒙特卡洛最大回撤"""
        # 简化版本，实际应该计算路径的最大回撤
        sorted_returns = np.sort(returns)
        worst_5_percent = sorted_returns[:int(len(sorted_returns) * 0.05)]
        return np.mean(worst_5_percent)
    
    async def create_custom_scenario(self, scenario_config: Dict[str, Any]) -> str:
        """创建自定义场景"""
        scenario_id = scenario_config.get('scenario_id', f"custom_{len(self.scenarios)}")
        
        scenario = ForwardScenario(
            scenario_id=scenario_id,
            scenario_type=ScenarioType.CUSTOM,
            name=scenario_config.get('name', 'Custom Scenario'),
            description=scenario_config.get('description', ''),
            probability=scenario_config.get('probability', 0.1),
            horizon_days=scenario_config.get('horizon_days', 252),
            market_conditions=scenario_config.get('market_conditions', {}),
            risk_factors=scenario_config.get('risk_factors', {}),
            macroeconomic_variables=scenario_config.get('macroeconomic_variables', {}),
            return_multiplier=scenario_config.get('return_multiplier', 1.0),
            volatility_multiplier=scenario_config.get('volatility_multiplier', 1.0)
        )
        
        self.scenarios[scenario_id] = scenario
        return scenario_id
    
    async def generate_forecast_report(self, analysis_result: ForwardAnalysisResult) -> Dict[str, Any]:
        """生成预测报告"""
        report = {
            'executive_summary': {
                'analysis_type': analysis_result.analysis_type.value,
                'analysis_confidence': analysis_result.analysis_confidence,
                'key_findings': [],
                'recommendations': analysis_result.optimization_recommendations
            },
            'portfolio_outlook': {},
            'risk_assessment': analysis_result.risk_decomposition,
            'scenario_analysis': analysis_result.scenario_outcomes,
            'model_validation': analysis_result.model_performance,
            'limitations_and_assumptions': {
                'limitations': analysis_result.limitations,
                'assumptions': analysis_result.assumptions
            }
        }
        
        # 投资组合展望
        if analysis_result.portfolio_projections:
            main_projection = analysis_result.portfolio_projections[0]
            report['portfolio_outlook'] = {
                'expected_return': main_projection.projected_return,
                'expected_volatility': main_projection.projected_volatility,
                'value_at_risk_95': main_projection.var_95,
                'value_at_risk_99': main_projection.var_99,
                'sharpe_ratio': main_projection.sharpe_ratio,
                'maximum_drawdown': main_projection.maximum_drawdown
            }
        
        # 关键发现
        if analysis_result.portfolio_projections:
            proj = analysis_result.portfolio_projections[0]
            if proj.projected_return > 0.1:
                report['executive_summary']['key_findings'].append('预期收益率较高')
            if proj.var_95 < -0.1:
                report['executive_summary']['key_findings'].append('下行风险显著')
            if proj.sharpe_ratio > 1.0:
                report['executive_summary']['key_findings'].append('风险调整后收益良好')
        
        return report
    
    async def visualize_forward_analysis(self, analysis_result: ForwardAnalysisResult, 
                                       save_path: str = None) -> Dict[str, Any]:
        """可视化前瞻性分析结果"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 投资组合预测分布
        if analysis_result.portfolio_projections:
            projection = analysis_result.portfolio_projections[0]
            
            # 模拟收益率分布
            returns = np.random.normal(projection.projected_return, projection.projected_volatility, 1000)
            
            axes[0, 0].hist(returns, bins=50, alpha=0.7, density=True)
            axes[0, 0].axvline(projection.projected_return, color='red', linestyle='--', label='预期收益')
            axes[0, 0].axvline(projection.var_95, color='orange', linestyle='--', label='VaR 95%')
            axes[0, 0].set_title('投资组合收益率分布')
            axes[0, 0].legend()
        
        # 情景分析结果
        if analysis_result.scenario_outcomes:
            scenarios = list(analysis_result.scenario_outcomes.keys())
            returns = [analysis_result.scenario_outcomes[s]['expected_return'] for s in scenarios]
            
            axes[0, 1].bar(scenarios, returns, alpha=0.7)
            axes[0, 1].set_title('情景分析结果')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 风险分解
        if analysis_result.risk_decomposition:
            risk_data = analysis_result.risk_decomposition
            axes[1, 0].pie([risk_data.get('portfolio_variance', 1)], labels=['投资组合风险'], autopct='%1.1f%%')
            axes[1, 0].set_title('风险分解')
        
        # 敏感性分析
        if analysis_result.sensitivity_analysis:
            sensitivity_data = analysis_result.sensitivity_analysis
            factors = list(sensitivity_data.keys())[:5]  # 前5个因子
            sensitivities = [sensitivity_data[f].get('return_sensitivity', 0) for f in factors]
            
            axes[1, 1].bar(factors, sensitivities, alpha=0.7)
            axes[1, 1].set_title('敏感性分析')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return {
            'visualization_created': True,
            'save_path': save_path,
            'charts': ['收益率分布', '情景分析', '风险分解', '敏感性分析']
        }
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm, t, skew, kurtosis
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

class SlippageModel(Enum):
    LINEAR = "linear"
    SQUARE_ROOT = "square_root"
    LOGARITHMIC = "logarithmic"
    POWER_LAW = "power_law"
    ALMGREN_CHRISS = "almgren_chriss"
    BERTSIMAS_LO = "bertsimas_lo"
    KYLE = "kyle"
    OBIZHAEVA_WANG = "obizhaeva_wang"
    MACHINE_LEARNING = "machine_learning"
    ADAPTIVE = "adaptive"

class MarketCondition(Enum):
    NORMAL = "normal"
    VOLATILE = "volatile"
    ILLIQUID = "illiquid"
    TRENDING = "trending"
    REVERTING = "reverting"
    CRISIS = "crisis"

class ExecutionStyle(Enum):
    AGGRESSIVE = "aggressive"
    MODERATE = "moderate"
    PASSIVE = "passive"
    STEALTH = "stealth"
    OPPORTUNISTIC = "opportunistic"

@dataclass
class SlippageParameters:
    """滑点模型参数"""
    model_type: SlippageModel
    alpha: float = 0.6  # 参与率指数
    beta: float = 0.5   # 波动率指数
    gamma: float = 0.3  # 流动性指数
    delta: float = 0.2  # 时间指数
    epsilon: float = 0.1  # 基础滑点
    lambda_param: float = 0.01  # 冲击系数
    eta: float = 0.001  # 临时冲击
    theta: float = 0.0001  # 永久冲击
    kappa: float = 0.5  # 市场深度系数
    sigma: float = 0.02  # 波动率
    phi: float = 0.1    # 订单大小敏感性
    tau: float = 1.0    # 时间尺度
    regime_adjustments: Dict[MarketCondition, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MarketImpactFactors:
    """市场冲击因子"""
    participation_rate: float
    volatility: float
    liquidity: float
    order_size: float
    time_horizon: int
    market_cap: float
    adv: float  # Average Daily Volume
    spread: float
    momentum: float
    mean_reversion: float
    volume_profile: List[float]
    intraday_pattern: List[float]
    news_impact: float = 0.0
    earnings_proximity: float = 0.0
    sector_rotation: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SlippageEstimate:
    """滑点估算结果"""
    symbol: str
    order_size: float
    expected_slippage: float
    slippage_std: float
    confidence_interval: Tuple[float, float]
    temporary_impact: float
    permanent_impact: float
    total_impact: float
    model_used: SlippageModel
    market_condition: MarketCondition
    parameters: SlippageParameters
    feature_importance: Dict[str, float]
    prediction_confidence: float
    alternative_estimates: Dict[SlippageModel, float]
    sensitivity_analysis: Dict[str, float]
    execution_recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SlippageCalibration:
    """滑点模型校准"""
    model_type: SlippageModel
    calibration_period: Tuple[datetime, datetime]
    sample_size: int
    in_sample_r2: float
    out_sample_r2: float
    mae: float
    rmse: float
    mape: float
    parameters: SlippageParameters
    feature_importance: Dict[str, float]
    residual_analysis: Dict[str, float]
    validation_metrics: Dict[str, float]
    stability_test: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)

class SlippageOptimizer:
    """
    滑点模型优化器
    
    提供多种滑点模型的校准、优化和预测功能，
    支持机器学习方法和自适应参数调整。
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 模型配置
        self.default_model = SlippageModel(config.get('default_model', 'square_root'))
        self.calibration_window = config.get('calibration_window', 252)
        self.min_observations = config.get('min_observations', 100)
        
        # 模型参数
        self.models = {}
        self.calibrations = {}
        self.feature_scalers = {}
        
        # 历史数据
        self.execution_history = []
        self.market_data_history = []
        
        # 机器学习模型
        self.ml_models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'neural_network': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000)
        }
        
        # 初始化默认参数
        self._initialize_default_parameters()
        
        # 性能监控
        self.performance_metrics = {}
        self.model_rankings = {}
        
    def _initialize_default_parameters(self):
        """初始化默认参数"""
        # 线性模型
        self.models[SlippageModel.LINEAR] = SlippageParameters(
            model_type=SlippageModel.LINEAR,
            alpha=1.0, beta=0.5, epsilon=0.001,
            regime_adjustments={
                MarketCondition.NORMAL: 1.0,
                MarketCondition.VOLATILE: 1.5,
                MarketCondition.ILLIQUID: 2.0,
                MarketCondition.CRISIS: 3.0
            }
        )
        
        # 平方根模型
        self.models[SlippageModel.SQUARE_ROOT] = SlippageParameters(
            model_type=SlippageModel.SQUARE_ROOT,
            alpha=0.5, beta=0.6, epsilon=0.001,
            regime_adjustments={
                MarketCondition.NORMAL: 1.0,
                MarketCondition.VOLATILE: 1.3,
                MarketCondition.ILLIQUID: 1.8,
                MarketCondition.CRISIS: 2.5
            }
        )
        
        # Almgren-Chriss模型
        self.models[SlippageModel.ALMGREN_CHRISS] = SlippageParameters(
            model_type=SlippageModel.ALMGREN_CHRISS,
            eta=0.001, theta=0.0001, sigma=0.02,
            regime_adjustments={
                MarketCondition.NORMAL: 1.0,
                MarketCondition.VOLATILE: 1.4,
                MarketCondition.ILLIQUID: 1.9,
                MarketCondition.CRISIS: 2.8
            }
        )
        
        # 机器学习模型
        self.models[SlippageModel.MACHINE_LEARNING] = SlippageParameters(
            model_type=SlippageModel.MACHINE_LEARNING,
            alpha=0.6, beta=0.5, gamma=0.3,
            regime_adjustments={
                MarketCondition.NORMAL: 1.0,
                MarketCondition.VOLATILE: 1.2,
                MarketCondition.ILLIQUID: 1.6,
                MarketCondition.CRISIS: 2.2
            }
        )
    
    async def estimate_slippage(self, 
                              symbol: str,
                              order_size: float,
                              market_factors: MarketImpactFactors,
                              model_type: SlippageModel = None,
                              execution_style: ExecutionStyle = ExecutionStyle.MODERATE) -> SlippageEstimate:
        """估算滑点"""
        try:
            if model_type is None:
                model_type = self.default_model
            
            # 检测市场条件
            market_condition = await self._detect_market_condition(market_factors)
            
            # 获取模型参数
            parameters = self.models.get(model_type, self.models[self.default_model])
            
            # 计算滑点
            if model_type == SlippageModel.LINEAR:
                slippage = await self._linear_slippage(order_size, market_factors, parameters)
            elif model_type == SlippageModel.SQUARE_ROOT:
                slippage = await self._square_root_slippage(order_size, market_factors, parameters)
            elif model_type == SlippageModel.LOGARITHMIC:
                slippage = await self._logarithmic_slippage(order_size, market_factors, parameters)
            elif model_type == SlippageModel.POWER_LAW:
                slippage = await self._power_law_slippage(order_size, market_factors, parameters)
            elif model_type == SlippageModel.ALMGREN_CHRISS:
                slippage = await self._almgren_chriss_slippage(order_size, market_factors, parameters)
            elif model_type == SlippageModel.BERTSIMAS_LO:
                slippage = await self._bertsimas_lo_slippage(order_size, market_factors, parameters)
            elif model_type == SlippageModel.KYLE:
                slippage = await self._kyle_slippage(order_size, market_factors, parameters)
            elif model_type == SlippageModel.OBIZHAEVA_WANG:
                slippage = await self._obizhaeva_wang_slippage(order_size, market_factors, parameters)
            elif model_type == SlippageModel.MACHINE_LEARNING:
                slippage = await self._ml_slippage(order_size, market_factors, parameters)
            elif model_type == SlippageModel.ADAPTIVE:
                slippage = await self._adaptive_slippage(order_size, market_factors, parameters)
            else:
                slippage = await self._square_root_slippage(order_size, market_factors, parameters)
            
            # 应用市场制度调整
            regime_adjustment = parameters.regime_adjustments.get(market_condition, 1.0)
            adjusted_slippage = slippage * regime_adjustment
            
            # 应用执行风格调整
            style_adjustment = await self._get_execution_style_adjustment(execution_style)
            final_slippage = adjusted_slippage * style_adjustment
            
            # 计算不确定性
            slippage_std = await self._estimate_slippage_uncertainty(
                final_slippage, market_factors, market_condition
            )
            
            # 置信区间
            confidence_interval = (
                final_slippage - 1.96 * slippage_std,
                final_slippage + 1.96 * slippage_std
            )
            
            # 分解临时和永久冲击
            temporary_impact, permanent_impact = await self._decompose_impact(
                final_slippage, market_factors, parameters
            )
            
            # 特征重要性
            feature_importance = await self._calculate_feature_importance(
                market_factors, model_type
            )
            
            # 敏感性分析
            sensitivity_analysis = await self._perform_sensitivity_analysis(
                order_size, market_factors, parameters
            )
            
            # 替代估算
            alternative_estimates = await self._get_alternative_estimates(
                order_size, market_factors, model_type
            )
            
            # 执行建议
            execution_recommendations = await self._generate_execution_recommendations(
                final_slippage, market_factors, market_condition
            )
            
            return SlippageEstimate(
                symbol=symbol,
                order_size=order_size,
                expected_slippage=final_slippage,
                slippage_std=slippage_std,
                confidence_interval=confidence_interval,
                temporary_impact=temporary_impact,
                permanent_impact=permanent_impact,
                total_impact=final_slippage,
                model_used=model_type,
                market_condition=market_condition,
                parameters=parameters,
                feature_importance=feature_importance,
                prediction_confidence=0.85,  # 根据模型校准结果调整
                alternative_estimates=alternative_estimates,
                sensitivity_analysis=sensitivity_analysis,
                execution_recommendations=execution_recommendations
            )
            
        except Exception as e:
            self.logger.error(f"滑点估算失败: {e}")
            raise
    
    async def _detect_market_condition(self, market_factors: MarketImpactFactors) -> MarketCondition:
        """检测市场条件"""
        # 基于多个因子的市场制度分类
        volatility_threshold = 0.25
        liquidity_threshold = 0.3
        momentum_threshold = 0.1
        
        if market_factors.volatility > volatility_threshold:
            if market_factors.liquidity < liquidity_threshold:
                return MarketCondition.CRISIS
            else:
                return MarketCondition.VOLATILE
        elif market_factors.liquidity < liquidity_threshold:
            return MarketCondition.ILLIQUID
        elif abs(market_factors.momentum) > momentum_threshold:
            return MarketCondition.TRENDING
        elif market_factors.mean_reversion > 0.5:
            return MarketCondition.REVERTING
        else:
            return MarketCondition.NORMAL
    
    async def _linear_slippage(self, order_size: float, 
                             market_factors: MarketImpactFactors,
                             parameters: SlippageParameters) -> float:
        """线性滑点模型"""
        participation_rate = market_factors.participation_rate
        volatility = market_factors.volatility
        
        slippage = (
            parameters.epsilon + 
            parameters.alpha * participation_rate +
            parameters.beta * volatility
        )
        
        return slippage
    
    async def _square_root_slippage(self, order_size: float,
                                  market_factors: MarketImpactFactors,
                                  parameters: SlippageParameters) -> float:
        """平方根滑点模型"""
        participation_rate = market_factors.participation_rate
        volatility = market_factors.volatility
        
        slippage = (
            parameters.epsilon + 
            parameters.alpha * np.sqrt(participation_rate) +
            parameters.beta * volatility
        )
        
        return slippage
    
    async def _logarithmic_slippage(self, order_size: float,
                                  market_factors: MarketImpactFactors,
                                  parameters: SlippageParameters) -> float:
        """对数滑点模型"""
        participation_rate = max(market_factors.participation_rate, 1e-6)
        volatility = market_factors.volatility
        
        slippage = (
            parameters.epsilon + 
            parameters.alpha * np.log(1 + participation_rate) +
            parameters.beta * volatility
        )
        
        return slippage
    
    async def _power_law_slippage(self, order_size: float,
                                market_factors: MarketImpactFactors,
                                parameters: SlippageParameters) -> float:
        """幂律滑点模型"""
        participation_rate = market_factors.participation_rate
        volatility = market_factors.volatility
        
        slippage = (
            parameters.epsilon + 
            parameters.alpha * (participation_rate ** parameters.gamma) +
            parameters.beta * volatility
        )
        
        return slippage
    
    async def _almgren_chriss_slippage(self, order_size: float,
                                     market_factors: MarketImpactFactors,
                                     parameters: SlippageParameters) -> float:
        """Almgren-Chriss滑点模型"""
        participation_rate = market_factors.participation_rate
        time_horizon = market_factors.time_horizon
        volatility = market_factors.volatility
        
        # 临时冲击
        temporary_impact = parameters.eta * participation_rate
        
        # 永久冲击
        permanent_impact = parameters.theta * participation_rate
        
        # 总冲击
        total_impact = temporary_impact + permanent_impact
        
        return total_impact
    
    async def _bertsimas_lo_slippage(self, order_size: float,
                                   market_factors: MarketImpactFactors,
                                   parameters: SlippageParameters) -> float:
        """Bertsimas-Lo滑点模型"""
        participation_rate = market_factors.participation_rate
        volatility = market_factors.volatility
        liquidity = market_factors.liquidity
        
        # 考虑流动性的修正
        liquidity_adjustment = 1.0 / max(liquidity, 0.1)
        
        slippage = (
            parameters.epsilon + 
            parameters.alpha * np.sqrt(participation_rate) * liquidity_adjustment +
            parameters.beta * volatility
        )
        
        return slippage
    
    async def _kyle_slippage(self, order_size: float,
                           market_factors: MarketImpactFactors,
                           parameters: SlippageParameters) -> float:
        """Kyle滑点模型"""
        participation_rate = market_factors.participation_rate
        volatility = market_factors.volatility
        
        # Kyle's lambda
        kyle_lambda = parameters.lambda_param * volatility / np.sqrt(market_factors.liquidity)
        
        slippage = kyle_lambda * participation_rate
        
        return slippage
    
    async def _obizhaeva_wang_slippage(self, order_size: float,
                                     market_factors: MarketImpactFactors,
                                     parameters: SlippageParameters) -> float:
        """Obizhaeva-Wang滑点模型"""
        participation_rate = market_factors.participation_rate
        volatility = market_factors.volatility
        time_horizon = market_factors.time_horizon
        
        # 考虑时间维度的冲击
        time_adjustment = np.sqrt(time_horizon / 252)  # 年化调整
        
        slippage = (
            parameters.alpha * (participation_rate ** parameters.gamma) * 
            volatility * time_adjustment
        )
        
        return slippage
    
    async def _ml_slippage(self, order_size: float,
                         market_factors: MarketImpactFactors,
                         parameters: SlippageParameters) -> float:
        """机器学习滑点模型"""
        try:
            # 构建特征向量
            features = self._build_feature_vector(market_factors)
            
            # 使用最佳ML模型预测
            best_model = self._get_best_ml_model()
            
            if best_model and hasattr(best_model, 'predict'):
                # 标准化特征
                scaler = self.feature_scalers.get('ml_scaler')
                if scaler:
                    features_scaled = scaler.transform([features])
                    slippage = best_model.predict(features_scaled)[0]
                else:
                    slippage = best_model.predict([features])[0]
            else:
                # 回退到平方根模型
                slippage = await self._square_root_slippage(order_size, market_factors, parameters)
            
            return max(0, slippage)  # 确保非负
            
        except Exception as e:
            self.logger.warning(f"ML滑点预测失败，回退到默认模型: {e}")
            return await self._square_root_slippage(order_size, market_factors, parameters)
    
    def _build_feature_vector(self, market_factors: MarketImpactFactors) -> List[float]:
        """构建特征向量"""
        features = [
            market_factors.participation_rate,
            market_factors.volatility,
            market_factors.liquidity,
            market_factors.order_size,
            market_factors.time_horizon,
            np.log(market_factors.market_cap),
            np.log(market_factors.adv),
            market_factors.spread,
            market_factors.momentum,
            market_factors.mean_reversion,
            np.mean(market_factors.volume_profile) if market_factors.volume_profile else 0,
            np.std(market_factors.volume_profile) if market_factors.volume_profile else 0,
            market_factors.news_impact,
            market_factors.earnings_proximity,
            market_factors.sector_rotation
        ]
        
        return features
    
    def _get_best_ml_model(self):
        """获取最佳ML模型"""
        if not self.model_rankings:
            return None
        
        best_model_name = min(self.model_rankings, key=self.model_rankings.get)
        return self.ml_models.get(best_model_name)
    
    async def _adaptive_slippage(self, order_size: float,
                               market_factors: MarketImpactFactors,
                               parameters: SlippageParameters) -> float:
        """自适应滑点模型"""
        # 基于实时市场条件选择最佳模型
        market_condition = await self._detect_market_condition(market_factors)
        
        # 根据市场条件选择模型
        if market_condition == MarketCondition.NORMAL:
            return await self._square_root_slippage(order_size, market_factors, parameters)
        elif market_condition == MarketCondition.VOLATILE:
            return await self._almgren_chriss_slippage(order_size, market_factors, parameters)
        elif market_condition == MarketCondition.ILLIQUID:
            return await self._bertsimas_lo_slippage(order_size, market_factors, parameters)
        elif market_condition == MarketCondition.TRENDING:
            return await self._kyle_slippage(order_size, market_factors, parameters)
        elif market_condition == MarketCondition.CRISIS:
            return await self._power_law_slippage(order_size, market_factors, parameters)
        else:
            return await self._ml_slippage(order_size, market_factors, parameters)
    
    async def _get_execution_style_adjustment(self, execution_style: ExecutionStyle) -> float:
        """获取执行风格调整因子"""
        adjustments = {
            ExecutionStyle.AGGRESSIVE: 1.5,
            ExecutionStyle.MODERATE: 1.0,
            ExecutionStyle.PASSIVE: 0.7,
            ExecutionStyle.STEALTH: 0.5,
            ExecutionStyle.OPPORTUNISTIC: 0.8
        }
        
        return adjustments.get(execution_style, 1.0)
    
    async def _estimate_slippage_uncertainty(self, slippage: float,
                                           market_factors: MarketImpactFactors,
                                           market_condition: MarketCondition) -> float:
        """估算滑点不确定性"""
        # 基于市场条件的不确定性因子
        uncertainty_factors = {
            MarketCondition.NORMAL: 0.2,
            MarketCondition.VOLATILE: 0.5,
            MarketCondition.ILLIQUID: 0.8,
            MarketCondition.TRENDING: 0.3,
            MarketCondition.REVERTING: 0.4,
            MarketCondition.CRISIS: 1.2
        }
        
        base_uncertainty = uncertainty_factors.get(market_condition, 0.3)
        
        # 考虑订单大小的影响
        size_factor = 1 + market_factors.participation_rate
        
        # 考虑波动率的影响
        volatility_factor = 1 + market_factors.volatility
        
        uncertainty = slippage * base_uncertainty * size_factor * volatility_factor
        
        return uncertainty
    
    async def _decompose_impact(self, total_slippage: float,
                              market_factors: MarketImpactFactors,
                              parameters: SlippageParameters) -> Tuple[float, float]:
        """分解临时和永久冲击"""
        # 临时冲击通常占总冲击的70-80%
        temporary_ratio = 0.75
        
        # 基于参与率调整比例
        if market_factors.participation_rate > 0.2:
            temporary_ratio = 0.6  # 大单的永久冲击更大
        elif market_factors.participation_rate < 0.05:
            temporary_ratio = 0.85  # 小单的临时冲击更大
        
        temporary_impact = total_slippage * temporary_ratio
        permanent_impact = total_slippage * (1 - temporary_ratio)
        
        return temporary_impact, permanent_impact
    
    async def _calculate_feature_importance(self, market_factors: MarketImpactFactors,
                                          model_type: SlippageModel) -> Dict[str, float]:
        """计算特征重要性"""
        if model_type == SlippageModel.MACHINE_LEARNING:
            # 从ML模型获取特征重要性
            best_model = self._get_best_ml_model()
            if hasattr(best_model, 'feature_importances_'):
                feature_names = [
                    'participation_rate', 'volatility', 'liquidity', 'order_size',
                    'time_horizon', 'market_cap', 'adv', 'spread', 'momentum',
                    'mean_reversion', 'volume_profile_mean', 'volume_profile_std',
                    'news_impact', 'earnings_proximity', 'sector_rotation'
                ]
                
                importances = best_model.feature_importances_
                return dict(zip(feature_names, importances))
        
        # 默认特征重要性
        return {
            'participation_rate': 0.35,
            'volatility': 0.25,
            'liquidity': 0.20,
            'spread': 0.10,
            'momentum': 0.05,
            'other': 0.05
        }
    
    async def _perform_sensitivity_analysis(self, order_size: float,
                                          market_factors: MarketImpactFactors,
                                          parameters: SlippageParameters) -> Dict[str, float]:
        """执行敏感性分析"""
        base_slippage = await self._square_root_slippage(order_size, market_factors, parameters)
        
        sensitivity = {}
        
        # 参与率敏感性
        high_participation = MarketImpactFactors(
            **{k: v for k, v in market_factors.__dict__.items() 
               if k != 'participation_rate'},
            participation_rate=market_factors.participation_rate * 1.5
        )
        high_part_slippage = await self._square_root_slippage(order_size, high_participation, parameters)
        sensitivity['participation_rate'] = (high_part_slippage - base_slippage) / base_slippage
        
        # 波动率敏感性
        high_volatility = MarketImpactFactors(
            **{k: v for k, v in market_factors.__dict__.items() 
               if k != 'volatility'},
            volatility=market_factors.volatility * 1.5
        )
        high_vol_slippage = await self._square_root_slippage(order_size, high_volatility, parameters)
        sensitivity['volatility'] = (high_vol_slippage - base_slippage) / base_slippage
        
        # 流动性敏感性
        low_liquidity = MarketImpactFactors(
            **{k: v for k, v in market_factors.__dict__.items() 
               if k != 'liquidity'},
            liquidity=market_factors.liquidity * 0.5
        )
        low_liq_slippage = await self._square_root_slippage(order_size, low_liquidity, parameters)
        sensitivity['liquidity'] = (low_liq_slippage - base_slippage) / base_slippage
        
        return sensitivity
    
    async def _get_alternative_estimates(self, order_size: float,
                                       market_factors: MarketImpactFactors,
                                       primary_model: SlippageModel) -> Dict[SlippageModel, float]:
        """获取替代估算"""
        alternatives = {}
        
        for model_type in [SlippageModel.LINEAR, SlippageModel.SQUARE_ROOT, 
                          SlippageModel.ALMGREN_CHRISS, SlippageModel.KYLE]:
            if model_type != primary_model:
                try:
                    estimate = await self.estimate_slippage(
                        "ALTERNATIVE", order_size, market_factors, model_type
                    )
                    alternatives[model_type] = estimate.expected_slippage
                except Exception as e:
                    self.logger.warning(f"替代估算失败 {model_type}: {e}")
        
        return alternatives
    
    async def _generate_execution_recommendations(self, slippage: float,
                                                market_factors: MarketImpactFactors,
                                                market_condition: MarketCondition) -> List[str]:
        """生成执行建议"""
        recommendations = []
        
        # 基于滑点水平的建议
        if slippage > 0.01:  # 1%
            recommendations.append("考虑分批执行以降低市场冲击")
            recommendations.append("使用TWAP或VWAP算法")
        elif slippage > 0.005:  # 0.5%
            recommendations.append("适度分批执行")
            recommendations.append("考虑在流动性较好的时段执行")
        else:
            recommendations.append("可以考虑积极执行")
        
        # 基于市场条件的建议
        if market_condition == MarketCondition.VOLATILE:
            recommendations.append("避免在高波动期执行大单")
            recommendations.append("考虑使用限价单减少滑点")
        elif market_condition == MarketCondition.ILLIQUID:
            recommendations.append("延长执行时间窗口")
            recommendations.append("考虑使用暗池或其他流动性来源")
        elif market_condition == MarketCondition.CRISIS:
            recommendations.append("谨慎执行，考虑延迟交易")
            recommendations.append("增加缓冲时间和价格容忍度")
        
        # 基于参与率的建议
        if market_factors.participation_rate > 0.2:
            recommendations.append("参与率过高，建议延长执行时间")
        elif market_factors.participation_rate < 0.05:
            recommendations.append("参与率较低，可以适当加快执行")
        
        return recommendations
    
    async def calibrate_model(self, execution_data: List[Dict[str, Any]],
                            model_type: SlippageModel = None) -> SlippageCalibration:
        """校准滑点模型"""
        try:
            if model_type is None:
                model_type = self.default_model
            
            # 准备数据
            X, y = self._prepare_calibration_data(execution_data)
            
            if len(X) < self.min_observations:
                raise ValueError(f"需要至少 {self.min_observations} 个观测值进行校准")
            
            # 分割数据
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # 校准模型
            if model_type == SlippageModel.MACHINE_LEARNING:
                calibration = await self._calibrate_ml_model(X_train, X_test, y_train, y_test)
            else:
                calibration = await self._calibrate_parametric_model(
                    X_train, X_test, y_train, y_test, model_type
                )
            
            # 保存校准结果
            self.calibrations[model_type] = calibration
            
            return calibration
            
        except Exception as e:
            self.logger.error(f"模型校准失败: {e}")
            raise
    
    def _prepare_calibration_data(self, execution_data: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """准备校准数据"""
        X = []
        y = []
        
        for record in execution_data:
            # 构建特征向量
            features = [
                record.get('participation_rate', 0),
                record.get('volatility', 0),
                record.get('liquidity', 1),
                record.get('order_size', 0),
                record.get('time_horizon', 60),
                np.log(record.get('market_cap', 1e9)),
                np.log(record.get('adv', 1e6)),
                record.get('spread', 0),
                record.get('momentum', 0),
                record.get('mean_reversion', 0),
                record.get('volume_profile_mean', 0),
                record.get('volume_profile_std', 0),
                record.get('news_impact', 0),
                record.get('earnings_proximity', 0),
                record.get('sector_rotation', 0)
            ]
            
            X.append(features)
            y.append(record.get('actual_slippage', 0))
        
        return np.array(X), np.array(y)
    
    async def _calibrate_ml_model(self, X_train: np.ndarray, X_test: np.ndarray,
                                y_train: np.ndarray, y_test: np.ndarray) -> SlippageCalibration:
        """校准机器学习模型"""
        # 特征标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.feature_scalers['ml_scaler'] = scaler
        
        # 训练多个模型
        model_performances = {}
        
        for name, model in self.ml_models.items():
            try:
                # 训练模型
                model.fit(X_train_scaled, y_train)
                
                # 评估模型
                train_score = model.score(X_train_scaled, y_train)
                test_score = model.score(X_test_scaled, y_test)
                
                # 交叉验证
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
                
                model_performances[name] = {
                    'train_score': train_score,
                    'test_score': test_score,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
                
            except Exception as e:
                self.logger.warning(f"模型 {name} 训练失败: {e}")
                model_performances[name] = {
                    'train_score': 0,
                    'test_score': 0,
                    'cv_mean': 0,
                    'cv_std': 1
                }
        
        # 选择最佳模型
        best_model_name = max(model_performances, 
                             key=lambda x: model_performances[x]['test_score'])
        
        self.model_rankings = {name: -perf['test_score'] 
                              for name, perf in model_performances.items()}
        
        best_performance = model_performances[best_model_name]
        
        # 计算误差指标
        best_model = self.ml_models[best_model_name]
        y_pred = best_model.predict(X_test_scaled)
        
        mae = np.mean(np.abs(y_test - y_pred))
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        # 特征重要性
        feature_importance = {}
        if hasattr(best_model, 'feature_importances_'):
            feature_names = [
                'participation_rate', 'volatility', 'liquidity', 'order_size',
                'time_horizon', 'market_cap', 'adv', 'spread', 'momentum',
                'mean_reversion', 'volume_profile_mean', 'volume_profile_std',
                'news_impact', 'earnings_proximity', 'sector_rotation'
            ]
            feature_importance = dict(zip(feature_names, best_model.feature_importances_))
        
        return SlippageCalibration(
            model_type=SlippageModel.MACHINE_LEARNING,
            calibration_period=(datetime.now() - timedelta(days=self.calibration_window), 
                              datetime.now()),
            sample_size=len(X_train),
            in_sample_r2=best_performance['train_score'],
            out_sample_r2=best_performance['test_score'],
            mae=mae,
            rmse=rmse,
            mape=mape,
            parameters=self.models[SlippageModel.MACHINE_LEARNING],
            feature_importance=feature_importance,
            residual_analysis={
                'mean_residual': np.mean(y_test - y_pred),
                'residual_std': np.std(y_test - y_pred),
                'residual_skew': skew(y_test - y_pred),
                'residual_kurtosis': kurtosis(y_test - y_pred)
            },
            validation_metrics=model_performances,
            stability_test={
                'cv_mean': best_performance['cv_mean'],
                'cv_std': best_performance['cv_std']
            }
        )
    
    async def _calibrate_parametric_model(self, X_train: np.ndarray, X_test: np.ndarray,
                                        y_train: np.ndarray, y_test: np.ndarray,
                                        model_type: SlippageModel) -> SlippageCalibration:
        """校准参数化模型"""
        # 定义优化目标函数
        def objective(params):
            # 更新模型参数
            temp_params = SlippageParameters(
                model_type=model_type,
                alpha=params[0],
                beta=params[1],
                epsilon=params[2] if len(params) > 2 else 0.001,
                gamma=params[3] if len(params) > 3 else 0.3
            )
            
            # 计算预测误差
            errors = []
            for i, features in enumerate(X_train):
                # 构建MarketImpactFactors
                market_factors = MarketImpactFactors(
                    participation_rate=features[0],
                    volatility=features[1],
                    liquidity=features[2],
                    order_size=features[3],
                    time_horizon=int(features[4]),
                    market_cap=np.exp(features[5]),
                    adv=np.exp(features[6]),
                    spread=features[7],
                    momentum=features[8],
                    mean_reversion=features[9],
                    volume_profile=[],
                    intraday_pattern=[]
                )
                
                # 计算预测滑点
                if model_type == SlippageModel.LINEAR:
                    predicted = asyncio.run(self._linear_slippage(
                        market_factors.order_size, market_factors, temp_params
                    ))
                elif model_type == SlippageModel.SQUARE_ROOT:
                    predicted = asyncio.run(self._square_root_slippage(
                        market_factors.order_size, market_factors, temp_params
                    ))
                else:
                    predicted = asyncio.run(self._square_root_slippage(
                        market_factors.order_size, market_factors, temp_params
                    ))
                
                errors.append((predicted - y_train[i]) ** 2)
            
            return np.mean(errors)
        
        # 参数边界
        if model_type == SlippageModel.LINEAR:
            bounds = [(0.1, 2.0), (0.1, 2.0), (0.0001, 0.01)]
            x0 = [1.0, 0.5, 0.001]
        else:  # SQUARE_ROOT
            bounds = [(0.1, 2.0), (0.1, 2.0), (0.0001, 0.01)]
            x0 = [0.5, 0.5, 0.001]
        
        # 优化
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
        
        # 更新模型参数
        optimized_params = SlippageParameters(
            model_type=model_type,
            alpha=result.x[0],
            beta=result.x[1],
            epsilon=result.x[2] if len(result.x) > 2 else 0.001,
            gamma=result.x[3] if len(result.x) > 3 else 0.3
        )
        
        self.models[model_type] = optimized_params
        
        # 计算验证指标
        train_predictions = []
        test_predictions = []
        
        for features in X_train:
            market_factors = MarketImpactFactors(
                participation_rate=features[0],
                volatility=features[1],
                liquidity=features[2],
                order_size=features[3],
                time_horizon=int(features[4]),
                market_cap=np.exp(features[5]),
                adv=np.exp(features[6]),
                spread=features[7],
                momentum=features[8],
                mean_reversion=features[9],
                volume_profile=[],
                intraday_pattern=[]
            )
            
            if model_type == SlippageModel.LINEAR:
                pred = await self._linear_slippage(
                    market_factors.order_size, market_factors, optimized_params
                )
            else:
                pred = await self._square_root_slippage(
                    market_factors.order_size, market_factors, optimized_params
                )
            
            train_predictions.append(pred)
        
        for features in X_test:
            market_factors = MarketImpactFactors(
                participation_rate=features[0],
                volatility=features[1],
                liquidity=features[2],
                order_size=features[3],
                time_horizon=int(features[4]),
                market_cap=np.exp(features[5]),
                adv=np.exp(features[6]),
                spread=features[7],
                momentum=features[8],
                mean_reversion=features[9],
                volume_profile=[],
                intraday_pattern=[]
            )
            
            if model_type == SlippageModel.LINEAR:
                pred = await self._linear_slippage(
                    market_factors.order_size, market_factors, optimized_params
                )
            else:
                pred = await self._square_root_slippage(
                    market_factors.order_size, market_factors, optimized_params
                )
            
            test_predictions.append(pred)
        
        # 计算R²
        train_r2 = 1 - np.sum((y_train - train_predictions) ** 2) / np.sum((y_train - np.mean(y_train)) ** 2)
        test_r2 = 1 - np.sum((y_test - test_predictions) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
        
        # 计算误差指标
        mae = np.mean(np.abs(y_test - test_predictions))
        rmse = np.sqrt(np.mean((y_test - test_predictions) ** 2))
        mape = np.mean(np.abs((y_test - test_predictions) / y_test)) * 100
        
        return SlippageCalibration(
            model_type=model_type,
            calibration_period=(datetime.now() - timedelta(days=self.calibration_window), 
                              datetime.now()),
            sample_size=len(X_train),
            in_sample_r2=train_r2,
            out_sample_r2=test_r2,
            mae=mae,
            rmse=rmse,
            mape=mape,
            parameters=optimized_params,
            feature_importance={'participation_rate': 0.6, 'volatility': 0.4},
            residual_analysis={
                'mean_residual': np.mean(y_test - test_predictions),
                'residual_std': np.std(y_test - test_predictions)
            },
            validation_metrics={'optimization_success': result.success},
            stability_test={'parameter_stability': 1.0}
        )
    
    async def optimize_execution_parameters(self, symbol: str, order_size: float,
                                          market_factors: MarketImpactFactors,
                                          constraints: Dict[str, Any] = None) -> Dict[str, Any]:
        """优化执行参数"""
        try:
            # 定义优化目标
            def objective(params):
                # params: [participation_rate, time_horizon]
                modified_factors = MarketImpactFactors(
                    **{k: v for k, v in market_factors.__dict__.items() 
                       if k not in ['participation_rate', 'time_horizon']},
                    participation_rate=params[0],
                    time_horizon=int(params[1])
                )
                
                # 计算预期滑点
                estimate = asyncio.run(self.estimate_slippage(
                    symbol, order_size, modified_factors
                ))
                
                return estimate.expected_slippage
            
            # 约束条件
            constraints_list = []
            
            # 参与率约束
            if constraints and 'max_participation_rate' in constraints:
                constraints_list.append({
                    'type': 'ineq',
                    'fun': lambda x: constraints['max_participation_rate'] - x[0]
                })
            
            # 时间约束
            if constraints and 'max_time_horizon' in constraints:
                constraints_list.append({
                    'type': 'ineq',
                    'fun': lambda x: constraints['max_time_horizon'] - x[1]
                })
            
            # 初始值和边界
            x0 = [0.1, 60]  # 10%参与率，60分钟
            bounds = [(0.01, 0.30), (5, 480)]  # 参与率1-30%，时间5-480分钟
            
            # 优化
            result = minimize(
                objective, x0, method='SLSQP', bounds=bounds, constraints=constraints_list
            )
            
            if result.success:
                optimal_participation = result.x[0]
                optimal_time = int(result.x[1])
                optimal_cost = result.fun
                
                # 计算节省
                base_estimate = await self.estimate_slippage(symbol, order_size, market_factors)
                savings = base_estimate.expected_slippage - optimal_cost
                
                return {
                    'optimization_success': True,
                    'optimal_participation_rate': optimal_participation,
                    'optimal_time_horizon': optimal_time,
                    'optimal_slippage': optimal_cost,
                    'expected_savings': savings,
                    'savings_bps': (savings / (order_size * 100)) * 10000,  # 假设股价100
                    'execution_strategy': 'optimized',
                    'confidence': 0.85
                }
            else:
                return {
                    'optimization_success': False,
                    'error': result.message,
                    'fallback_strategy': 'moderate'
                }
                
        except Exception as e:
            self.logger.error(f"执行参数优化失败: {e}")
            return {
                'optimization_success': False,
                'error': str(e),
                'fallback_strategy': 'moderate'
            }
    
    async def generate_slippage_report(self, estimate: SlippageEstimate) -> Dict[str, Any]:
        """生成滑点报告"""
        report = {
            'executive_summary': {
                'symbol': estimate.symbol,
                'order_size': f"{estimate.order_size:,.0f}",
                'expected_slippage': f"{estimate.expected_slippage:.4f}",
                'slippage_bps': f"{estimate.expected_slippage * 10000:.1f} bp",
                'confidence_interval': f"{estimate.confidence_interval[0]:.4f} - {estimate.confidence_interval[1]:.4f}",
                'model_used': estimate.model_used.value,
                'market_condition': estimate.market_condition.value,
                'prediction_confidence': f"{estimate.prediction_confidence:.1%}"
            },
            'impact_breakdown': {
                'temporary_impact': f"{estimate.temporary_impact:.4f}",
                'permanent_impact': f"{estimate.permanent_impact:.4f}",
                'total_impact': f"{estimate.total_impact:.4f}",
                'temporary_percentage': f"{estimate.temporary_impact / estimate.total_impact * 100:.1f}%",
                'permanent_percentage': f"{estimate.permanent_impact / estimate.total_impact * 100:.1f}%"
            },
            'feature_importance': estimate.feature_importance,
            'alternative_estimates': {
                model.value: f"{slippage:.4f}" 
                for model, slippage in estimate.alternative_estimates.items()
            },
            'sensitivity_analysis': {
                factor: f"{sensitivity:.2%}" 
                for factor, sensitivity in estimate.sensitivity_analysis.items()
            },
            'execution_recommendations': estimate.execution_recommendations,
            'risk_assessment': {
                'uncertainty_level': 'High' if estimate.slippage_std > estimate.expected_slippage * 0.5 else 'Medium' if estimate.slippage_std > estimate.expected_slippage * 0.2 else 'Low',
                'market_risk': estimate.market_condition.value,
                'execution_risk': 'High' if estimate.expected_slippage > 0.01 else 'Medium' if estimate.expected_slippage > 0.005 else 'Low'
            }
        }
        
        return report
    
    async def visualize_slippage_analysis(self, estimate: SlippageEstimate, 
                                        save_path: str = None):
        """可视化滑点分析"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 冲击分解
        impact_labels = ['Temporary', 'Permanent']
        impact_values = [estimate.temporary_impact, estimate.permanent_impact]
        
        axes[0, 0].pie(impact_values, labels=impact_labels, autopct='%1.1f%%')
        axes[0, 0].set_title('Market Impact Decomposition')
        
        # 替代估算对比
        if estimate.alternative_estimates:
            models = list(estimate.alternative_estimates.keys())
            values = list(estimate.alternative_estimates.values())
            values.append(estimate.expected_slippage)
            models.append(estimate.model_used)
            
            axes[0, 1].bar([m.value for m in models], values)
            axes[0, 1].set_title('Alternative Model Estimates')
            axes[0, 1].set_ylabel('Slippage')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 特征重要性
        if estimate.feature_importance:
            features = list(estimate.feature_importance.keys())
            importances = list(estimate.feature_importance.values())
            
            axes[1, 0].barh(features, importances)
            axes[1, 0].set_title('Feature Importance')
            axes[1, 0].set_xlabel('Importance')
        
        # 敏感性分析
        if estimate.sensitivity_analysis:
            factors = list(estimate.sensitivity_analysis.keys())
            sensitivities = list(estimate.sensitivity_analysis.values())
            
            axes[1, 1].bar(factors, sensitivities)
            axes[1, 1].set_title('Sensitivity Analysis')
            axes[1, 1].set_ylabel('Sensitivity')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
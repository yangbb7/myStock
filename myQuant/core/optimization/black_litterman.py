import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from scipy.linalg import inv, pinv
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns

class ViewType(Enum):
    ABSOLUTE = "absolute"
    RELATIVE = "relative"
    CONDITIONAL = "conditional"

class ConfidenceLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class MarketParameters:
    """市场参数"""
    market_cap_weights: np.ndarray
    risk_aversion: float
    expected_returns: np.ndarray
    covariance_matrix: np.ndarray
    risk_free_rate: float = 0.0
    market_return: float = 0.0
    market_volatility: float = 0.0
    tau: float = 0.025  # 不确定性参数

@dataclass
class ViewParameters:
    """观点参数"""
    view_id: str
    view_type: ViewType
    assets: List[str]
    picking_matrix: np.ndarray  # P矩阵
    view_returns: np.ndarray    # Q向量
    uncertainty_matrix: np.ndarray  # Omega矩阵
    confidence: ConfidenceLevel
    horizon_days: int = 252
    description: str = ""
    active: bool = True

@dataclass
class BlackLittermanResult:
    """Black-Litterman结果"""
    new_expected_returns: np.ndarray
    new_covariance_matrix: np.ndarray
    optimal_weights: np.ndarray
    implied_returns: np.ndarray
    posterior_returns: np.ndarray
    posterior_risk: float
    posterior_sharpe: float
    views_incorporated: int
    confidence_weighted_returns: np.ndarray
    tracking_error: float
    active_weights: np.ndarray
    view_contributions: Dict[str, float]
    uncertainty_adjustments: Dict[str, float]

class BlackLittermanModel:
    """
    Black-Litterman模型实现
    
    Black-Litterman模型是一种贝叶斯方法，用于结合市场均衡观点和投资者的主观观点
    来生成更稳健的预期收益率估计。
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 模型参数
        self.tau = config.get('tau', 0.025)  # 不确定性参数
        self.risk_aversion = config.get('risk_aversion', 3.0)
        self.confidence_scaling = config.get('confidence_scaling', {
            ConfidenceLevel.LOW: 0.25,
            ConfidenceLevel.MEDIUM: 0.5,
            ConfidenceLevel.HIGH: 0.75,
            ConfidenceLevel.VERY_HIGH: 1.0
        })
        
        # 视图管理
        self.views: Dict[str, ViewParameters] = {}
        self.active_views: List[str] = []
        
        # 历史结果
        self.optimization_history: List[BlackLittermanResult] = []
        
        # 市场参数
        self.market_params: Optional[MarketParameters] = None
        
        # 计算缓存
        self.cache = {}
        
        self.logger.info("Black-Litterman model initialized")
    
    def set_market_parameters(self, market_params: MarketParameters):
        """设置市场参数"""
        try:
            self.market_params = market_params
            
            # 验证参数一致性
            n_assets = len(market_params.market_cap_weights)
            
            if len(market_params.expected_returns) != n_assets:
                raise ValueError("Expected returns dimension mismatch")
            
            if market_params.covariance_matrix.shape != (n_assets, n_assets):
                raise ValueError("Covariance matrix dimension mismatch")
            
            # 计算隐含收益率
            self._calculate_implied_returns()
            
            self.logger.info(f"Market parameters set for {n_assets} assets")
            
        except Exception as e:
            self.logger.error(f"Error setting market parameters: {e}")
            raise
    
    def add_view(self, view: ViewParameters):
        """添加投资者观点"""
        try:
            # 验证观点参数
            if not self._validate_view(view):
                raise ValueError(f"Invalid view parameters: {view.view_id}")
            
            self.views[view.view_id] = view
            
            if view.active:
                self.active_views.append(view.view_id)
            
            self.logger.info(f"Added view: {view.view_id} ({view.view_type.value})")
            
        except Exception as e:
            self.logger.error(f"Error adding view: {e}")
            raise
    
    def remove_view(self, view_id: str):
        """移除投资者观点"""
        try:
            if view_id in self.views:
                del self.views[view_id]
                
                if view_id in self.active_views:
                    self.active_views.remove(view_id)
                
                self.logger.info(f"Removed view: {view_id}")
            
        except Exception as e:
            self.logger.error(f"Error removing view: {e}")
    
    def update_view_confidence(self, view_id: str, confidence: ConfidenceLevel):
        """更新观点置信度"""
        try:
            if view_id in self.views:
                old_confidence = self.views[view_id].confidence
                self.views[view_id].confidence = confidence
                
                # 重新计算不确定性矩阵
                self._update_view_uncertainty(view_id)
                
                self.logger.info(f"Updated view {view_id} confidence: {old_confidence.value} -> {confidence.value}")
            
        except Exception as e:
            self.logger.error(f"Error updating view confidence: {e}")
    
    def activate_view(self, view_id: str):
        """激活观点"""
        try:
            if view_id in self.views and view_id not in self.active_views:
                self.active_views.append(view_id)
                self.views[view_id].active = True
                self.logger.info(f"Activated view: {view_id}")
            
        except Exception as e:
            self.logger.error(f"Error activating view: {e}")
    
    def deactivate_view(self, view_id: str):
        """停用观点"""
        try:
            if view_id in self.active_views:
                self.active_views.remove(view_id)
                self.views[view_id].active = False
                self.logger.info(f"Deactivated view: {view_id}")
            
        except Exception as e:
            self.logger.error(f"Error deactivating view: {e}")
    
    async def optimize_portfolio(self, 
                               assets: List[str],
                               constraints: Optional[Dict[str, Any]] = None) -> BlackLittermanResult:
        """
        使用Black-Litterman模型优化投资组合
        """
        try:
            if self.market_params is None:
                raise ValueError("Market parameters not set")
            
            self.logger.info("Starting Black-Litterman optimization")
            
            # 构建观点矩阵
            P, Q, Omega = self._build_view_matrices()
            
            # 执行Black-Litterman计算
            result = await self._calculate_black_litterman(P, Q, Omega, assets, constraints)
            
            # 保存结果
            self.optimization_history.append(result)
            
            # 保持历史记录数量
            if len(self.optimization_history) > 100:
                self.optimization_history = self.optimization_history[-100:]
            
            self.logger.info("Black-Litterman optimization completed")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in Black-Litterman optimization: {e}")
            raise
    
    def _calculate_implied_returns(self):
        """计算隐含收益率"""
        try:
            if self.market_params is None:
                return
            
            # 逆向优化：mu = lambda * Sigma * w
            # 其中lambda是市场风险厌恶系数
            implied_returns = (
                self.market_params.risk_aversion * 
                np.dot(self.market_params.covariance_matrix, self.market_params.market_cap_weights)
            )
            
            self.market_params.expected_returns = implied_returns
            
            self.logger.info("Calculated implied returns from market equilibrium")
            
        except Exception as e:
            self.logger.error(f"Error calculating implied returns: {e}")
    
    def _build_view_matrices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """构建观点矩阵P、Q、Omega"""
        try:
            if not self.active_views:
                # 没有活跃观点，返回空矩阵
                n_assets = len(self.market_params.market_cap_weights)
                return np.zeros((0, n_assets)), np.zeros(0), np.zeros((0, 0))
            
            # 收集活跃观点
            active_view_objects = [self.views[view_id] for view_id in self.active_views]
            
            # 构建P矩阵（观点选择矩阵）
            P_list = []
            Q_list = []
            Omega_diag = []
            
            for view in active_view_objects:
                P_list.append(view.picking_matrix)
                Q_list.append(view.view_returns)
                
                # 计算观点不确定性
                uncertainty = self._calculate_view_uncertainty(view)
                Omega_diag.append(uncertainty)
            
            # 组装矩阵
            P = np.vstack(P_list) if P_list else np.zeros((0, len(self.market_params.market_cap_weights)))
            Q = np.concatenate(Q_list) if Q_list else np.zeros(0)
            Omega = np.diag(Omega_diag) if Omega_diag else np.zeros((0, 0))
            
            return P, Q, Omega
            
        except Exception as e:
            self.logger.error(f"Error building view matrices: {e}")
            raise
    
    def _calculate_view_uncertainty(self, view: ViewParameters) -> float:
        """计算观点不确定性"""
        try:
            # 基础不确定性
            base_uncertainty = np.dot(view.picking_matrix, 
                                    np.dot(self.market_params.covariance_matrix, 
                                          view.picking_matrix.T))
            
            # 根据置信度调整
            confidence_factor = self.confidence_scaling[view.confidence]
            
            # 时间调整
            time_factor = view.horizon_days / 252.0
            
            # 最终不确定性
            uncertainty = base_uncertainty * (1 / confidence_factor) * time_factor
            
            return max(uncertainty, 1e-8)  # 避免数值问题
            
        except Exception as e:
            self.logger.error(f"Error calculating view uncertainty: {e}")
            return 1e-4
    
    def _update_view_uncertainty(self, view_id: str):
        """更新观点不确定性"""
        try:
            if view_id not in self.views:
                return
            
            view = self.views[view_id]
            uncertainty = self._calculate_view_uncertainty(view)
            
            # 更新不确定性矩阵
            view.uncertainty_matrix = np.array([[uncertainty]])
            
        except Exception as e:
            self.logger.error(f"Error updating view uncertainty: {e}")
    
    async def _calculate_black_litterman(self, 
                                       P: np.ndarray, 
                                       Q: np.ndarray, 
                                       Omega: np.ndarray,
                                       assets: List[str],
                                       constraints: Optional[Dict[str, Any]] = None) -> BlackLittermanResult:
        """
        执行Black-Litterman计算
        """
        try:
            # 获取市场参数
            mu_market = self.market_params.expected_returns
            Sigma = self.market_params.covariance_matrix
            w_market = self.market_params.market_cap_weights
            tau = self.tau
            
            # Black-Litterman公式
            # 新的预期收益率 = mu_BL = [(tau*Sigma)^-1 + P'*Omega^-1*P]^-1 * [(tau*Sigma)^-1*mu + P'*Omega^-1*Q]
            # 新的协方差矩阵 = Sigma_BL = [(tau*Sigma)^-1 + P'*Omega^-1*P]^-1
            
            # 计算先验精度矩阵
            prior_precision = inv(tau * Sigma)
            
            # 计算观点精度矩阵
            if P.shape[0] > 0:  # 有观点
                view_precision = np.dot(P.T, np.dot(inv(Omega), P))
            else:  # 无观点
                view_precision = np.zeros_like(prior_precision)
            
            # 计算后验精度矩阵
            posterior_precision = prior_precision + view_precision
            
            # 计算后验协方差矩阵
            posterior_covariance = inv(posterior_precision)
            
            # 计算后验期望收益率
            prior_term = np.dot(prior_precision, mu_market)
            
            if P.shape[0] > 0:  # 有观点
                view_term = np.dot(P.T, np.dot(inv(Omega), Q))
            else:  # 无观点
                view_term = np.zeros_like(mu_market)
            
            posterior_returns = np.dot(posterior_covariance, prior_term + view_term)
            
            # 计算最优权重
            optimal_weights = await self._optimize_weights(
                posterior_returns, posterior_covariance, constraints
            )
            
            # 计算绩效指标
            posterior_risk = np.sqrt(np.dot(optimal_weights, np.dot(posterior_covariance, optimal_weights)))
            posterior_sharpe = (np.dot(optimal_weights, posterior_returns) - self.market_params.risk_free_rate) / posterior_risk
            
            # 计算跟踪误差
            active_weights = optimal_weights - w_market
            tracking_error = np.sqrt(np.dot(active_weights, np.dot(Sigma, active_weights)))
            
            # 计算观点贡献
            view_contributions = self._calculate_view_contributions(P, Q, Omega, posterior_returns, mu_market)
            
            # 计算不确定性调整
            uncertainty_adjustments = self._calculate_uncertainty_adjustments(posterior_covariance, Sigma)
            
            # 置信度加权收益率
            confidence_weighted_returns = self._calculate_confidence_weighted_returns(posterior_returns, mu_market)
            
            result = BlackLittermanResult(
                new_expected_returns=posterior_returns,
                new_covariance_matrix=posterior_covariance,
                optimal_weights=optimal_weights,
                implied_returns=mu_market,
                posterior_returns=posterior_returns,
                posterior_risk=posterior_risk,
                posterior_sharpe=posterior_sharpe,
                views_incorporated=len(self.active_views),
                confidence_weighted_returns=confidence_weighted_returns,
                tracking_error=tracking_error,
                active_weights=active_weights,
                view_contributions=view_contributions,
                uncertainty_adjustments=uncertainty_adjustments
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in Black-Litterman calculation: {e}")
            raise
    
    async def _optimize_weights(self, 
                              expected_returns: np.ndarray,
                              covariance_matrix: np.ndarray,
                              constraints: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        基于新的预期收益率和协方差矩阵优化权重
        """
        try:
            n_assets = len(expected_returns)
            
            # 目标函数：最大化效用 = mu'w - (lambda/2)*w'*Sigma*w
            def objective(weights):
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_risk = np.dot(weights, np.dot(covariance_matrix, weights))
                utility = portfolio_return - 0.5 * self.market_params.risk_aversion * portfolio_risk
                return -utility  # 最小化负效用
            
            # 约束条件
            constraints_list = []
            
            # 权重和约束
            constraints_list.append({
                'type': 'eq',
                'fun': lambda w: np.sum(w) - 1.0
            })
            
            # 自定义约束
            if constraints:
                # 权重界限
                if 'weight_bounds' in constraints:
                    bounds = constraints['weight_bounds']
                else:
                    bounds = [(0.0, 1.0) for _ in range(n_assets)]
                
                # 行业限制
                if 'sector_limits' in constraints:
                    sector_limits = constraints['sector_limits']
                    for sector, limit in sector_limits.items():
                        sector_assets = constraints.get(f'{sector}_assets', [])
                        if sector_assets:
                            sector_indices = [i for i, asset in enumerate(sector_assets) if i < n_assets]
                            if sector_indices:
                                constraints_list.append({
                                    'type': 'ineq',
                                    'fun': lambda w, indices=sector_indices: limit - np.sum(w[indices])
                                })
                
                # 集中度限制
                if 'max_weight' in constraints:
                    max_weight = constraints['max_weight']
                    constraints_list.append({
                        'type': 'ineq',
                        'fun': lambda w: max_weight - np.max(w)
                    })
            else:
                bounds = [(0.0, 1.0) for _ in range(n_assets)]
            
            # 初始权重
            initial_weights = self.market_params.market_cap_weights.copy()
            
            # 优化
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_list,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            if result.success:
                return result.x
            else:
                self.logger.warning(f"Optimization failed: {result.message}")
                return initial_weights
                
        except Exception as e:
            self.logger.error(f"Error optimizing weights: {e}")
            return self.market_params.market_cap_weights.copy()
    
    def _calculate_view_contributions(self, 
                                    P: np.ndarray, 
                                    Q: np.ndarray, 
                                    Omega: np.ndarray,
                                    posterior_returns: np.ndarray,
                                    prior_returns: np.ndarray) -> Dict[str, float]:
        """计算各观点的贡献"""
        try:
            contributions = {}
            
            if P.shape[0] == 0:
                return contributions
            
            # 计算每个观点的贡献
            for i, view_id in enumerate(self.active_views):
                P_i = P[i:i+1, :]
                Q_i = Q[i:i+1]
                Omega_i = Omega[i:i+1, i:i+1]
                
                # 计算该观点的影响
                view_precision = np.dot(P_i.T, np.dot(inv(Omega_i), P_i))
                view_term = np.dot(P_i.T, np.dot(inv(Omega_i), Q_i))
                
                # 贡献度量
                contribution = np.linalg.norm(view_term)
                contributions[view_id] = contribution
            
            return contributions
            
        except Exception as e:
            self.logger.error(f"Error calculating view contributions: {e}")
            return {}
    
    def _calculate_uncertainty_adjustments(self, 
                                         posterior_cov: np.ndarray,
                                         prior_cov: np.ndarray) -> Dict[str, float]:
        """计算不确定性调整"""
        try:
            adjustments = {}
            
            # 整体不确定性变化
            prior_uncertainty = np.trace(prior_cov)
            posterior_uncertainty = np.trace(posterior_cov)
            
            adjustments['overall_uncertainty_change'] = (posterior_uncertainty - prior_uncertainty) / prior_uncertainty
            
            # 各资产的不确定性变化
            for i in range(len(prior_cov)):
                prior_var = prior_cov[i, i]
                posterior_var = posterior_cov[i, i]
                
                if prior_var > 0:
                    change = (posterior_var - prior_var) / prior_var
                    adjustments[f'asset_{i}_uncertainty_change'] = change
            
            return adjustments
            
        except Exception as e:
            self.logger.error(f"Error calculating uncertainty adjustments: {e}")
            return {}
    
    def _calculate_confidence_weighted_returns(self, 
                                             posterior_returns: np.ndarray,
                                             prior_returns: np.ndarray) -> np.ndarray:
        """计算置信度加权收益率"""
        try:
            if not self.active_views:
                return posterior_returns
            
            # 计算整体置信度
            total_confidence = sum(
                self.confidence_scaling[self.views[view_id].confidence] 
                for view_id in self.active_views
            )
            
            avg_confidence = total_confidence / len(self.active_views) if self.active_views else 0
            
            # 置信度加权
            confidence_weighted = avg_confidence * posterior_returns + (1 - avg_confidence) * prior_returns
            
            return confidence_weighted
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence weighted returns: {e}")
            return posterior_returns
    
    def _validate_view(self, view: ViewParameters) -> bool:
        """验证观点参数"""
        try:
            if self.market_params is None:
                return False
            
            n_assets = len(self.market_params.market_cap_weights)
            
            # 检查选择矩阵维度
            if view.picking_matrix.shape[1] != n_assets:
                return False
            
            # 检查观点收益率维度
            if len(view.view_returns) != view.picking_matrix.shape[0]:
                return False
            
            # 检查不确定性矩阵维度
            expected_shape = (view.picking_matrix.shape[0], view.picking_matrix.shape[0])
            if view.uncertainty_matrix.shape != expected_shape:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating view: {e}")
            return False
    
    def create_absolute_view(self, 
                           view_id: str,
                           asset_index: int,
                           expected_return: float,
                           confidence: ConfidenceLevel,
                           description: str = "") -> ViewParameters:
        """
        创建绝对观点
        """
        try:
            if self.market_params is None:
                raise ValueError("Market parameters not set")
            
            n_assets = len(self.market_params.market_cap_weights)
            
            # 构建选择矩阵
            P = np.zeros((1, n_assets))
            P[0, asset_index] = 1.0
            
            # 观点收益率
            Q = np.array([expected_return])
            
            # 不确定性矩阵（临时，会被重新计算）
            Omega = np.array([[1.0]])
            
            view = ViewParameters(
                view_id=view_id,
                view_type=ViewType.ABSOLUTE,
                assets=[f"asset_{asset_index}"],
                picking_matrix=P,
                view_returns=Q,
                uncertainty_matrix=Omega,
                confidence=confidence,
                description=description
            )
            
            # 重新计算不确定性
            uncertainty = self._calculate_view_uncertainty(view)
            view.uncertainty_matrix = np.array([[uncertainty]])
            
            return view
            
        except Exception as e:
            self.logger.error(f"Error creating absolute view: {e}")
            raise
    
    def create_relative_view(self, 
                           view_id: str,
                           asset_index_1: int,
                           asset_index_2: int,
                           expected_outperformance: float,
                           confidence: ConfidenceLevel,
                           description: str = "") -> ViewParameters:
        """
        创建相对观点
        """
        try:
            if self.market_params is None:
                raise ValueError("Market parameters not set")
            
            n_assets = len(self.market_params.market_cap_weights)
            
            # 构建选择矩阵
            P = np.zeros((1, n_assets))
            P[0, asset_index_1] = 1.0
            P[0, asset_index_2] = -1.0
            
            # 观点收益率
            Q = np.array([expected_outperformance])
            
            # 不确定性矩阵（临时，会被重新计算）
            Omega = np.array([[1.0]])
            
            view = ViewParameters(
                view_id=view_id,
                view_type=ViewType.RELATIVE,
                assets=[f"asset_{asset_index_1}", f"asset_{asset_index_2}"],
                picking_matrix=P,
                view_returns=Q,
                uncertainty_matrix=Omega,
                confidence=confidence,
                description=description
            )
            
            # 重新计算不确定性
            uncertainty = self._calculate_view_uncertainty(view)
            view.uncertainty_matrix = np.array([[uncertainty]])
            
            return view
            
        except Exception as e:
            self.logger.error(f"Error creating relative view: {e}")
            raise
    
    def analyze_view_impact(self, view_id: str) -> Dict[str, Any]:
        """分析观点影响"""
        try:
            if view_id not in self.views:
                return {}
            
            view = self.views[view_id]
            
            # 计算没有该观点时的优化结果
            original_active = self.active_views.copy()
            
            # 暂时停用该观点
            if view_id in self.active_views:
                self.active_views.remove(view_id)
            
            # 优化without view
            P_without, Q_without, Omega_without = self._build_view_matrices()
            
            # 恢复原始状态
            self.active_views = original_active
            
            # 优化with view
            P_with, Q_with, Omega_with = self._build_view_matrices()
            
            # 比较结果
            analysis = {
                'view_id': view_id,
                'view_type': view.view_type.value,
                'confidence': view.confidence.value,
                'expected_return': view.view_returns.tolist(),
                'assets_involved': view.assets,
                'uncertainty': self._calculate_view_uncertainty(view),
                'impact_analysis': {
                    'views_without': P_without.shape[0],
                    'views_with': P_with.shape[0],
                    'view_strength': np.linalg.norm(view.picking_matrix),
                    'view_magnitude': np.linalg.norm(view.view_returns)
                }
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing view impact: {e}")
            return {}
    
    def get_model_summary(self) -> Dict[str, Any]:
        """获取模型摘要"""
        try:
            summary = {
                'model_parameters': {
                    'tau': self.tau,
                    'risk_aversion': self.risk_aversion,
                    'has_market_params': self.market_params is not None
                },
                'views_summary': {
                    'total_views': len(self.views),
                    'active_views': len(self.active_views),
                    'view_types': {},
                    'confidence_distribution': {}
                },
                'optimization_history': {
                    'total_optimizations': len(self.optimization_history),
                    'latest_result': None
                }
            }
            
            # 统计观点类型
            for view in self.views.values():
                view_type = view.view_type.value
                confidence = view.confidence.value
                
                summary['views_summary']['view_types'][view_type] = \
                    summary['views_summary']['view_types'].get(view_type, 0) + 1
                
                summary['views_summary']['confidence_distribution'][confidence] = \
                    summary['views_summary']['confidence_distribution'].get(confidence, 0) + 1
            
            # 最新结果
            if self.optimization_history:
                latest = self.optimization_history[-1]
                summary['optimization_history']['latest_result'] = {
                    'views_incorporated': latest.views_incorporated,
                    'posterior_sharpe': latest.posterior_sharpe,
                    'tracking_error': latest.tracking_error,
                    'posterior_risk': latest.posterior_risk
                }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting model summary: {e}")
            return {}
    
    def plot_return_comparison(self, 
                             result: BlackLittermanResult,
                             asset_names: List[str],
                             save_path: Optional[str] = None):
        """绘制收益率比较图"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 隐含收益率 vs 后验收益率
            x = np.arange(len(asset_names))
            width = 0.35
            
            ax1.bar(x - width/2, result.implied_returns, width, 
                   label='Implied Returns', alpha=0.7)
            ax1.bar(x + width/2, result.posterior_returns, width, 
                   label='Posterior Returns', alpha=0.7)
            
            ax1.set_xlabel('Assets')
            ax1.set_ylabel('Expected Return')
            ax1.set_title('Implied vs Posterior Returns')
            ax1.set_xticks(x)
            ax1.set_xticklabels(asset_names, rotation=45)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 权重比较
            ax2.bar(x - width/2, self.market_params.market_cap_weights, width, 
                   label='Market Cap Weights', alpha=0.7)
            ax2.bar(x + width/2, result.optimal_weights, width, 
                   label='Optimal Weights', alpha=0.7)
            
            ax2.set_xlabel('Assets')
            ax2.set_ylabel('Weight')
            ax2.set_title('Market Cap vs Optimal Weights')
            ax2.set_xticks(x)
            ax2.set_xticklabels(asset_names, rotation=45)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Error plotting return comparison: {e}")
    
    def plot_efficient_frontier(self, 
                               result: BlackLittermanResult,
                               save_path: Optional[str] = None):
        """绘制有效前沿"""
        try:
            # 生成有效前沿点
            target_returns = np.linspace(
                result.posterior_returns.min(), 
                result.posterior_returns.max(), 
                50
            )
            
            risks = []
            weights_list = []
            
            for target_return in target_returns:
                # 最小化风险，给定目标收益率
                def objective(weights):
                    return np.dot(weights, np.dot(result.new_covariance_matrix, weights))
                
                constraints = [
                    {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
                    {'type': 'eq', 'fun': lambda w: np.dot(w, result.posterior_returns) - target_return}
                ]
                
                bounds = [(0.0, 1.0) for _ in range(len(result.posterior_returns))]
                
                result_opt = minimize(
                    objective,
                    result.optimal_weights,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints
                )
                
                if result_opt.success:
                    risk = np.sqrt(result_opt.fun)
                    risks.append(risk)
                    weights_list.append(result_opt.x)
            
            # 绘制有效前沿
            plt.figure(figsize=(10, 8))
            
            # 绘制前沿
            plt.plot(risks, target_returns, 'b-', linewidth=2, label='Efficient Frontier')
            
            # 标记Black-Litterman最优点
            plt.scatter([result.posterior_risk], 
                       [np.dot(result.optimal_weights, result.posterior_returns)], 
                       c='red', s=100, marker='*', 
                       label='Black-Litterman Optimal', zorder=5)
            
            # 标记市场组合
            market_return = np.dot(self.market_params.market_cap_weights, result.posterior_returns)
            market_risk = np.sqrt(np.dot(self.market_params.market_cap_weights, 
                                        np.dot(result.new_covariance_matrix, 
                                              self.market_params.market_cap_weights)))
            
            plt.scatter([market_risk], [market_return], 
                       c='green', s=100, marker='o', 
                       label='Market Portfolio', zorder=5)
            
            plt.xlabel('Risk (Standard Deviation)')
            plt.ylabel('Expected Return')
            plt.title('Efficient Frontier with Black-Litterman')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Error plotting efficient frontier: {e}")
    
    def export_results(self, result: BlackLittermanResult, file_path: str):
        """导出结果到文件"""
        try:
            export_data = {
                'model_parameters': {
                    'tau': self.tau,
                    'risk_aversion': self.risk_aversion,
                    'views_incorporated': result.views_incorporated
                },
                'results': {
                    'implied_returns': result.implied_returns.tolist(),
                    'posterior_returns': result.posterior_returns.tolist(),
                    'optimal_weights': result.optimal_weights.tolist(),
                    'market_cap_weights': self.market_params.market_cap_weights.tolist(),
                    'posterior_risk': result.posterior_risk,
                    'posterior_sharpe': result.posterior_sharpe,
                    'tracking_error': result.tracking_error
                },
                'views': {
                    view_id: {
                        'view_type': view.view_type.value,
                        'confidence': view.confidence.value,
                        'expected_returns': view.view_returns.tolist(),
                        'assets': view.assets,
                        'description': view.description
                    } for view_id, view in self.views.items()
                },
                'analysis': {
                    'view_contributions': result.view_contributions,
                    'uncertainty_adjustments': result.uncertainty_adjustments
                }
            }
            
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.logger.info(f"Results exported to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error exporting results: {e}")
            raise
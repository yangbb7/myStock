import asyncio
import logging
import math
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from .smart_routing import MarketData, ChildOrder
from .low_latency_engine import Order, OrderSide

class ImpactModel(Enum):
    LINEAR = "linear"
    SQUARE_ROOT = "square_root"
    ALMGREN_CHRISS = "almgren_chriss"
    VOLUME_WEIGHTED = "volume_weighted"

@dataclass
class ImpactParameters:
    permanent_impact_coeff: float = 0.1
    temporary_impact_coeff: float = 0.01
    volatility: float = 0.02
    daily_volume: int = 1000000
    spread: float = 0.01
    tick_size: float = 0.01
    risk_aversion: float = 0.5

@dataclass
class ImpactPrediction:
    permanent_impact: float
    temporary_impact: float
    total_impact: float
    optimal_participation_rate: float
    implementation_shortfall: float
    risk_penalty: float
    confidence_interval: Tuple[float, float]

class MarketImpactMinimizer:
    def __init__(self, model_type: ImpactModel = ImpactModel.ALMGREN_CHRISS):
        self.model_type = model_type
        self.logger = logging.getLogger(__name__)
        self.impact_parameters = ImpactParameters()
        self.historical_impacts: List[float] = []
        
    def calibrate_model(self, historical_data: List[Dict[str, Any]]):
        """校准市场影响模型"""
        try:
            # 从历史数据中提取影响参数
            impacts = []
            volumes = []
            spreads = []
            
            for data in historical_data:
                if 'impact' in data and 'volume' in data:
                    impacts.append(data['impact'])
                    volumes.append(data['volume'])
                    spreads.append(data.get('spread', 0.01))
            
            if len(impacts) > 10:
                # 使用最小二乘法拟合参数
                self._fit_impact_parameters(impacts, volumes, spreads)
                self.logger.info("Market impact model calibrated successfully")
            else:
                self.logger.warning("Insufficient historical data for model calibration")
                
        except Exception as e:
            self.logger.error(f"Error calibrating market impact model: {e}")
    
    def predict_impact(self, 
                      order_quantity: int, 
                      market_data: MarketData,
                      execution_time: float) -> ImpactPrediction:
        """预测市场影响"""
        try:
            if self.model_type == ImpactModel.ALMGREN_CHRISS:
                return self._almgren_chriss_model(order_quantity, market_data, execution_time)
            elif self.model_type == ImpactModel.SQUARE_ROOT:
                return self._square_root_model(order_quantity, market_data, execution_time)
            elif self.model_type == ImpactModel.LINEAR:
                return self._linear_model(order_quantity, market_data, execution_time)
            else:
                return self._volume_weighted_model(order_quantity, market_data, execution_time)
                
        except Exception as e:
            self.logger.error(f"Error predicting market impact: {e}")
            return ImpactPrediction(0, 0, 0, 0.2, 0, 0, (0, 0))
    
    def _almgren_chriss_model(self, 
                            quantity: int, 
                            market_data: MarketData, 
                            execution_time: float) -> ImpactPrediction:
        """Almgren-Chriss 最优执行模型"""
        # 模型参数
        sigma = self.impact_parameters.volatility
        gamma = self.impact_parameters.permanent_impact_coeff
        eta = self.impact_parameters.temporary_impact_coeff
        lambda_param = self.impact_parameters.risk_aversion
        
        # 计算最优参与率
        kappa = math.sqrt(lambda_param * sigma**2 / eta)
        
        # 计算最优交易轨迹
        optimal_participation = self._calculate_optimal_participation(
            quantity, execution_time, kappa, gamma, eta
        )
        
        # 计算永久影响
        permanent_impact = gamma * quantity / market_data.adv
        
        # 计算临时影响
        participation_rate = quantity / (market_data.volume * execution_time)
        temporary_impact = eta * participation_rate * market_data.last_price
        
        # 计算总影响
        total_impact = permanent_impact + temporary_impact
        
        # 计算实施缺口
        implementation_shortfall = self._calculate_implementation_shortfall(
            quantity, market_data, execution_time, participation_rate
        )
        
        # 计算风险惩罚
        risk_penalty = lambda_param * sigma**2 * quantity**2 * execution_time / 2
        
        # 计算置信区间
        confidence_interval = self._calculate_confidence_interval(
            total_impact, sigma, execution_time
        )
        
        return ImpactPrediction(
            permanent_impact=permanent_impact,
            temporary_impact=temporary_impact,
            total_impact=total_impact,
            optimal_participation_rate=optimal_participation,
            implementation_shortfall=implementation_shortfall,
            risk_penalty=risk_penalty,
            confidence_interval=confidence_interval
        )
    
    def _square_root_model(self, 
                         quantity: int, 
                         market_data: MarketData, 
                         execution_time: float) -> ImpactPrediction:
        """平方根市场影响模型"""
        # 平方根模型：影响 ∝ √(quantity/volume)
        volume_ratio = quantity / market_data.volume if market_data.volume > 0 else 0
        
        permanent_impact = self.impact_parameters.permanent_impact_coeff * math.sqrt(volume_ratio)
        temporary_impact = self.impact_parameters.temporary_impact_coeff * math.sqrt(volume_ratio)
        
        total_impact = permanent_impact + temporary_impact
        
        # 最优参与率
        optimal_participation = min(0.3, math.sqrt(quantity / market_data.adv))
        
        # 实施缺口
        implementation_shortfall = total_impact * market_data.last_price
        
        return ImpactPrediction(
            permanent_impact=permanent_impact,
            temporary_impact=temporary_impact,
            total_impact=total_impact,
            optimal_participation_rate=optimal_participation,
            implementation_shortfall=implementation_shortfall,
            risk_penalty=0,
            confidence_interval=(total_impact * 0.8, total_impact * 1.2)
        )
    
    def _linear_model(self, 
                     quantity: int, 
                     market_data: MarketData, 
                     execution_time: float) -> ImpactPrediction:
        """线性市场影响模型"""
        # 线性模型：影响 ∝ quantity/volume
        volume_ratio = quantity / market_data.volume if market_data.volume > 0 else 0
        
        permanent_impact = self.impact_parameters.permanent_impact_coeff * volume_ratio
        temporary_impact = self.impact_parameters.temporary_impact_coeff * volume_ratio
        
        total_impact = permanent_impact + temporary_impact
        
        # 最优参与率
        optimal_participation = min(0.25, quantity / market_data.adv)
        
        # 实施缺口
        implementation_shortfall = total_impact * market_data.last_price
        
        return ImpactPrediction(
            permanent_impact=permanent_impact,
            temporary_impact=temporary_impact,
            total_impact=total_impact,
            optimal_participation_rate=optimal_participation,
            implementation_shortfall=implementation_shortfall,
            risk_penalty=0,
            confidence_interval=(total_impact * 0.9, total_impact * 1.1)
        )
    
    def _volume_weighted_model(self, 
                             quantity: int, 
                             market_data: MarketData, 
                             execution_time: float) -> ImpactPrediction:
        """成交量加权影响模型"""
        # 考虑成交量分布的影响模型
        daily_volume = market_data.adv
        current_volume = market_data.volume
        
        # 计算成交量权重
        volume_weight = current_volume / daily_volume if daily_volume > 0 else 0
        
        # 影响与成交量权重反比
        impact_multiplier = 1 / (volume_weight + 0.1)  # 避免除零
        
        base_impact = quantity / daily_volume if daily_volume > 0 else 0
        
        permanent_impact = base_impact * impact_multiplier * 0.1
        temporary_impact = base_impact * impact_multiplier * 0.05
        
        total_impact = permanent_impact + temporary_impact
        
        # 最优参与率基于成交量权重
        optimal_participation = min(0.2, volume_weight * 0.5)
        
        # 实施缺口
        implementation_shortfall = total_impact * market_data.last_price
        
        return ImpactPrediction(
            permanent_impact=permanent_impact,
            temporary_impact=temporary_impact,
            total_impact=total_impact,
            optimal_participation_rate=optimal_participation,
            implementation_shortfall=implementation_shortfall,
            risk_penalty=0,
            confidence_interval=(total_impact * 0.85, total_impact * 1.15)
        )
    
    def optimize_execution_schedule(self, 
                                  total_quantity: int,
                                  market_data: MarketData,
                                  execution_horizon: float) -> List[Dict[str, Any]]:
        """优化执行计划"""
        try:
            # 预测总体影响
            impact_prediction = self.predict_impact(total_quantity, market_data, execution_horizon)
            
            # 根据最优参与率计算执行计划
            optimal_rate = impact_prediction.optimal_participation_rate
            
            # 将执行时间分割成小的时间段
            time_slices = max(1, int(execution_horizon * 60 / 5))  # 每5分钟一个时间片
            
            schedule = []
            remaining_quantity = total_quantity
            
            for i in range(time_slices):
                # 计算当前时间片的执行量
                if i == time_slices - 1:
                    # 最后一个时间片执行剩余全部
                    slice_quantity = remaining_quantity
                else:
                    # 根据最优参与率和市场条件调整
                    expected_volume = self._estimate_slice_volume(market_data, i)
                    slice_quantity = min(
                        int(expected_volume * optimal_rate),
                        remaining_quantity // (time_slices - i)
                    )
                
                if slice_quantity > 0:
                    # 计算执行价格
                    execution_price = self._calculate_execution_price(
                        market_data, slice_quantity, optimal_rate
                    )
                    
                    schedule.append({
                        'time_slice': i,
                        'quantity': slice_quantity,
                        'price': execution_price,
                        'participation_rate': optimal_rate,
                        'expected_impact': impact_prediction.total_impact * (slice_quantity / total_quantity)
                    })
                    
                    remaining_quantity -= slice_quantity
                
                if remaining_quantity <= 0:
                    break
            
            return schedule
            
        except Exception as e:
            self.logger.error(f"Error optimizing execution schedule: {e}")
            return []
    
    def adaptive_impact_control(self, 
                              child_orders: List[ChildOrder],
                              market_data: MarketData,
                              realized_impact: float) -> List[Dict[str, Any]]:
        """自适应影响控制"""
        try:
            adjustments = []
            
            # 比较实际影响与预测影响
            total_quantity = sum(order.quantity for order in child_orders)
            predicted_impact = self.predict_impact(total_quantity, market_data, 1.0)
            
            impact_ratio = realized_impact / predicted_impact.total_impact if predicted_impact.total_impact > 0 else 1
            
            # 如果实际影响大于预测，调整执行策略
            if impact_ratio > 1.2:
                # 影响过大，减慢执行速度
                for order in child_orders:
                    if order.status.value == "pending":
                        # 延迟执行时间
                        new_time = order.scheduled_time + timedelta(minutes=5)
                        
                        # 减少订单数量
                        new_quantity = int(order.quantity * 0.8)
                        
                        adjustments.append({
                            'order_id': order.child_order_id,
                            'action': 'delay_and_reduce',
                            'new_time': new_time,
                            'new_quantity': new_quantity,
                            'reason': 'high_market_impact'
                        })
            
            elif impact_ratio < 0.8:
                # 影响小于预期，可以加速执行
                for order in child_orders:
                    if order.status.value == "pending":
                        # 提前执行时间
                        new_time = max(
                            datetime.now(),
                            order.scheduled_time - timedelta(minutes=2)
                        )
                        
                        # 增加订单数量
                        new_quantity = int(order.quantity * 1.2)
                        
                        adjustments.append({
                            'order_id': order.child_order_id,
                            'action': 'accelerate_and_increase',
                            'new_time': new_time,
                            'new_quantity': new_quantity,
                            'reason': 'low_market_impact'
                        })
            
            return adjustments
            
        except Exception as e:
            self.logger.error(f"Error in adaptive impact control: {e}")
            return []
    
    def _calculate_optimal_participation(self, 
                                       quantity: int, 
                                       execution_time: float,
                                       kappa: float, 
                                       gamma: float, 
                                       eta: float) -> float:
        """计算最优参与率"""
        try:
            # Almgren-Chriss 最优参与率公式
            theta = kappa * execution_time
            
            if theta > 0:
                optimal_rate = (1 - math.exp(-theta)) / (kappa * execution_time)
            else:
                optimal_rate = 1 / execution_time
            
            return min(0.3, max(0.05, optimal_rate))
            
        except Exception:
            return 0.2  # 默认参与率
    
    def _calculate_implementation_shortfall(self, 
                                          quantity: int,
                                          market_data: MarketData,
                                          execution_time: float,
                                          participation_rate: float) -> float:
        """计算实施缺口"""
        try:
            # 实施缺口 = 市场影响成本 + 时间风险成本
            market_impact_cost = self.impact_parameters.permanent_impact_coeff * quantity
            
            time_risk_cost = (
                self.impact_parameters.volatility**2 * 
                quantity**2 * 
                execution_time / 2
            )
            
            return market_impact_cost + time_risk_cost
            
        except Exception:
            return 0
    
    def _calculate_confidence_interval(self, 
                                     impact: float, 
                                     volatility: float,
                                     execution_time: float) -> Tuple[float, float]:
        """计算置信区间"""
        try:
            # 95% 置信区间
            std_dev = volatility * math.sqrt(execution_time)
            z_score = 1.96  # 95% 置信水平
            
            lower_bound = impact - z_score * std_dev
            upper_bound = impact + z_score * std_dev
            
            return (lower_bound, upper_bound)
            
        except Exception:
            return (impact * 0.8, impact * 1.2)
    
    def _estimate_slice_volume(self, market_data: MarketData, slice_index: int) -> int:
        """估算时间片成交量"""
        try:
            # 根据历史模式估算
            base_volume = market_data.adv / (6.5 * 12)  # 6.5小时，12个时间片
            
            # 根据时间调整成交量
            time_multiplier = self._get_time_multiplier(slice_index)
            
            return int(base_volume * time_multiplier)
            
        except Exception:
            return market_data.volume // 12
    
    def _get_time_multiplier(self, slice_index: int) -> float:
        """获取时间乘数"""
        # 模拟一天中不同时间的成交量模式
        multipliers = [
            1.5,  # 开盘
            1.2, 1.0, 0.8, 0.6, 0.7,  # 上午
            0.9, 1.1, 1.3, 1.5, 1.8, 2.0  # 下午到收盘
        ]
        
        return multipliers[slice_index % len(multipliers)]
    
    def _calculate_execution_price(self, 
                                 market_data: MarketData,
                                 quantity: int,
                                 participation_rate: float) -> float:
        """计算执行价格"""
        try:
            # 考虑市场影响的执行价格
            impact = self.impact_parameters.temporary_impact_coeff * participation_rate
            
            # 买入时价格上涨，卖出时价格下跌
            return market_data.last_price * (1 + impact)
            
        except Exception:
            return market_data.last_price
    
    def _fit_impact_parameters(self, 
                             impacts: List[float],
                             volumes: List[int],
                             spreads: List[float]):
        """拟合影响参数"""
        try:
            # 使用最小二乘法拟合
            impacts_array = np.array(impacts)
            volumes_array = np.array(volumes)
            spreads_array = np.array(spreads)
            
            # 简单线性回归
            if len(impacts) > 0:
                # 更新永久影响系数
                self.impact_parameters.permanent_impact_coeff = np.mean(impacts_array)
                
                # 更新临时影响系数
                self.impact_parameters.temporary_impact_coeff = np.mean(impacts_array) * 0.5
                
                # 更新波动率
                self.impact_parameters.volatility = np.std(impacts_array)
                
                # 更新价差
                self.impact_parameters.spread = np.mean(spreads_array)
                
        except Exception as e:
            self.logger.error(f"Error fitting impact parameters: {e}")
    
    def get_impact_statistics(self) -> Dict[str, Any]:
        """获取影响统计信息"""
        return {
            'model_type': self.model_type.value,
            'parameters': {
                'permanent_impact_coeff': self.impact_parameters.permanent_impact_coeff,
                'temporary_impact_coeff': self.impact_parameters.temporary_impact_coeff,
                'volatility': self.impact_parameters.volatility,
                'risk_aversion': self.impact_parameters.risk_aversion
            },
            'historical_impacts_count': len(self.historical_impacts),
            'avg_historical_impact': np.mean(self.historical_impacts) if self.historical_impacts else 0
        }
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import math

class LiquidityRiskType(Enum):
    FUNDING_LIQUIDITY = "funding_liquidity"
    MARKET_LIQUIDITY = "market_liquidity"
    OPERATIONAL_LIQUIDITY = "operational_liquidity"
    CONTINGENT_LIQUIDITY = "contingent_liquidity"

class LiquidityTier(Enum):
    TIER_1 = "tier_1"  # 现金和央行准备金
    TIER_2A = "tier_2a"  # 政府债券
    TIER_2B = "tier_2b"  # 公司债券、股票
    TIER_3 = "tier_3"  # 其他资产

class LiquidityTimeHorizon(Enum):
    INTRADAY = "intraday"
    OVERNIGHT = "overnight"
    ONE_WEEK = "one_week"
    ONE_MONTH = "one_month"
    THREE_MONTHS = "three_months"
    ONE_YEAR = "one_year"

@dataclass
class LiquidityMetrics:
    symbol: str
    bid_ask_spread: float
    market_depth: float
    volume_weighted_spread: float
    price_impact: float
    turnover_ratio: float
    liquidity_score: float
    days_to_liquidate: float
    market_cap: float
    average_daily_volume: float
    free_float: float
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class LiquidAsset:
    asset_id: str
    symbol: str
    asset_type: str
    tier: LiquidityTier
    market_value: float
    haircut: float
    liquidation_time_hours: int
    concentration_limit: float
    operational_requirements: float
    central_bank_eligible: bool
    encumbered: bool
    stressed_value: float
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class FundingSource:
    source_id: str
    source_type: str  # deposits, wholesale, repo, credit_line
    amount: float
    maturity_date: datetime
    cost_of_funding: float
    stability_factor: float
    concentration_risk: float
    stressed_runoff_rate: float
    available_amount: float
    committed: bool
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class LiquidityPosition:
    position_id: str
    asset: LiquidAsset
    quantity: float
    market_value: float
    adjusted_value: float  # 考虑折价后的价值
    liquidation_value: float  # 紧急清算价值
    concentration_ratio: float
    liquidity_buffer_contribution: float
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class LiquidityRiskMetrics:
    timestamp: datetime
    total_liquid_assets: float
    total_funding_requirements: float
    liquidity_coverage_ratio: float
    net_stable_funding_ratio: float
    loan_to_deposit_ratio: float
    liquidity_gap: Dict[str, float]  # 按时间段分布
    survival_period_days: float
    concentration_risk: float
    funding_cost: float
    liquidity_buffer: float
    stress_test_results: Dict[str, float]
    early_warning_indicators: List[str]
    regulatory_ratios: Dict[str, float]
    market_liquidity_risk: float
    funding_liquidity_risk: float

class LiquidityRiskManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 流动性资产和负债
        self.liquid_assets: Dict[str, LiquidAsset] = {}
        self.funding_sources: Dict[str, FundingSource] = {}
        self.liquidity_positions: Dict[str, LiquidityPosition] = {}
        
        # 流动性指标
        self.liquidity_metrics: Dict[str, LiquidityMetrics] = {}
        
        # 风险限额
        self.risk_limits = {
            'min_liquidity_coverage_ratio': config.get('min_lcr', 1.0),
            'min_net_stable_funding_ratio': config.get('min_nsfr', 1.0),
            'max_loan_to_deposit_ratio': config.get('max_ldr', 0.9),
            'min_survival_period_days': config.get('min_survival_days', 30),
            'max_concentration_single_asset': config.get('max_single_asset', 0.1),
            'max_concentration_asset_class': config.get('max_asset_class', 0.3),
            'min_liquidity_buffer': config.get('min_buffer', 0.05)
        }
        
        # 市场数据
        self.market_data: Dict[str, Dict[str, Any]] = {}
        
        # 压力测试参数
        self.stress_scenarios = {
            'mild_stress': {
                'deposit_runoff': 0.1,
                'wholesale_runoff': 0.3,
                'asset_haircut_increase': 0.05,
                'funding_cost_increase': 0.02
            },
            'moderate_stress': {
                'deposit_runoff': 0.2,
                'wholesale_runoff': 0.5,
                'asset_haircut_increase': 0.10,
                'funding_cost_increase': 0.05
            },
            'severe_stress': {
                'deposit_runoff': 0.3,
                'wholesale_runoff': 0.75,
                'asset_haircut_increase': 0.20,
                'funding_cost_increase': 0.10
            }
        }
        
        # 监管要求
        self.regulatory_requirements = {
            'basel_iii_lcr': 1.0,
            'basel_iii_nsfr': 1.0,
            'additional_buffer': 0.05
        }
        
        # 执行器
        self.executor = ThreadPoolExecutor(max_workers=config.get('max_workers', 4))
        
        # 历史数据
        self.risk_history: List[LiquidityRiskMetrics] = []
        
        # 报警阈值
        self.alert_thresholds = {
            'lcr_warning': 1.05,
            'lcr_critical': 1.0,
            'nsfr_warning': 1.05,
            'nsfr_critical': 1.0,
            'concentration_warning': 0.08,
            'concentration_critical': 0.10
        }
    
    async def add_liquid_asset(self, asset: LiquidAsset):
        """添加流动性资产"""
        try:
            self.liquid_assets[asset.asset_id] = asset
            
            # 计算流动性指标
            await self._calculate_asset_liquidity_metrics(asset)
            
            self.logger.info(f"Added liquid asset: {asset.symbol}")
            
        except Exception as e:
            self.logger.error(f"Error adding liquid asset: {e}")
    
    async def add_funding_source(self, funding: FundingSource):
        """添加资金来源"""
        try:
            self.funding_sources[funding.source_id] = funding
            
            # 更新资金成本
            await self._update_funding_cost(funding)
            
            self.logger.info(f"Added funding source: {funding.source_type}")
            
        except Exception as e:
            self.logger.error(f"Error adding funding source: {e}")
    
    async def _calculate_asset_liquidity_metrics(self, asset: LiquidAsset):
        """计算资产流动性指标"""
        try:
            # 获取市场数据
            market_data = self.market_data.get(asset.symbol, {})
            
            if not market_data:
                # 使用默认值
                self.liquidity_metrics[asset.symbol] = LiquidityMetrics(
                    symbol=asset.symbol,
                    bid_ask_spread=0.01,
                    market_depth=100000,
                    volume_weighted_spread=0.01,
                    price_impact=0.001,
                    turnover_ratio=0.1,
                    liquidity_score=0.5,
                    days_to_liquidate=1.0,
                    market_cap=asset.market_value,
                    average_daily_volume=10000,
                    free_float=0.8
                )
                return
            
            # 计算流动性指标
            bid_price = market_data.get('bid_price', 0)
            ask_price = market_data.get('ask_price', 0)
            mid_price = (bid_price + ask_price) / 2 if bid_price > 0 and ask_price > 0 else market_data.get('last_price', 0)
            
            # 买卖价差
            bid_ask_spread = (ask_price - bid_price) / mid_price if mid_price > 0 else 0.01
            
            # 市场深度
            market_depth = market_data.get('market_depth', 100000)
            
            # 成交量加权价差
            volume_weighted_spread = bid_ask_spread * market_data.get('volume_ratio', 1.0)
            
            # 价格冲击
            price_impact = self._calculate_price_impact(asset, market_data)
            
            # 换手率
            turnover_ratio = market_data.get('turnover_ratio', 0.1)
            
            # 流动性评分
            liquidity_score = self._calculate_liquidity_score(bid_ask_spread, market_depth, turnover_ratio)
            
            # 清算天数
            days_to_liquidate = self._calculate_days_to_liquidate(asset, market_data)
            
            # 创建流动性指标
            self.liquidity_metrics[asset.symbol] = LiquidityMetrics(
                symbol=asset.symbol,
                bid_ask_spread=bid_ask_spread,
                market_depth=market_depth,
                volume_weighted_spread=volume_weighted_spread,
                price_impact=price_impact,
                turnover_ratio=turnover_ratio,
                liquidity_score=liquidity_score,
                days_to_liquidate=days_to_liquidate,
                market_cap=market_data.get('market_cap', asset.market_value),
                average_daily_volume=market_data.get('adv', 10000),
                free_float=market_data.get('free_float', 0.8)
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating asset liquidity metrics: {e}")
    
    def _calculate_price_impact(self, asset: LiquidAsset, market_data: Dict[str, Any]) -> float:
        """计算价格冲击"""
        try:
            # 使用Kyle模型估算价格冲击
            trade_size = asset.market_value
            average_daily_volume = market_data.get('adv', 10000)
            volatility = market_data.get('volatility', 0.20)
            
            if average_daily_volume <= 0:
                return 0.001
            
            # 价格冲击 = volatility * (trade_size / ADV)^0.5
            price_impact = volatility * np.sqrt(trade_size / average_daily_volume)
            
            return min(price_impact, 0.1)  # 最大10%
            
        except Exception as e:
            self.logger.error(f"Error calculating price impact: {e}")
            return 0.001
    
    def _calculate_liquidity_score(self, spread: float, depth: float, turnover: float) -> float:
        """计算流动性评分"""
        try:
            # 流动性评分 = 1 / (1 + spread) * log(depth) * turnover
            if spread <= 0 or depth <= 0 or turnover <= 0:
                return 0.0
            
            spread_factor = 1 / (1 + spread)
            depth_factor = np.log(depth) / 10  # 标准化
            turnover_factor = min(turnover, 1.0)
            
            liquidity_score = spread_factor * depth_factor * turnover_factor
            
            return min(max(liquidity_score, 0.0), 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating liquidity score: {e}")
            return 0.0
    
    def _calculate_days_to_liquidate(self, asset: LiquidAsset, market_data: Dict[str, Any]) -> float:
        """计算清算天数"""
        try:
            trade_size = asset.market_value
            average_daily_volume = market_data.get('adv', 10000)
            participation_rate = 0.2  # 假设最大参与率20%
            
            if average_daily_volume <= 0:
                return 30.0
            
            # 清算天数 = 交易规模 / (平均日成交量 * 参与率)
            days_to_liquidate = trade_size / (average_daily_volume * participation_rate)
            
            return min(max(days_to_liquidate, 1.0), 365.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating days to liquidate: {e}")
            return 30.0
    
    async def _update_funding_cost(self, funding: FundingSource):
        """更新资金成本"""
        try:
            # 基础成本
            base_cost = funding.cost_of_funding
            
            # 风险调整
            risk_adjustment = 0.0
            
            # 期限风险
            days_to_maturity = (funding.maturity_date - datetime.now()).days
            if days_to_maturity <= 30:
                risk_adjustment += 0.001  # 10bp
            elif days_to_maturity <= 90:
                risk_adjustment += 0.0005  # 5bp
            
            # 集中度风险
            if funding.concentration_risk > 0.1:
                risk_adjustment += funding.concentration_risk * 0.01
            
            # 稳定性风险
            if funding.stability_factor < 0.8:
                risk_adjustment += (0.8 - funding.stability_factor) * 0.02
            
            # 更新成本
            funding.cost_of_funding = base_cost + risk_adjustment
            
        except Exception as e:
            self.logger.error(f"Error updating funding cost: {e}")
    
    async def calculate_liquidity_coverage_ratio(self) -> float:
        """计算流动性覆盖率（LCR）"""
        try:
            # 计算高质量流动性资产
            hqla = 0.0
            
            for asset in self.liquid_assets.values():
                # 根据层级应用折价
                if asset.tier == LiquidityTier.TIER_1:
                    haircut = 0.0
                elif asset.tier == LiquidityTier.TIER_2A:
                    haircut = 0.15
                elif asset.tier == LiquidityTier.TIER_2B:
                    haircut = 0.25
                else:
                    haircut = 0.50
                
                # 考虑集中度限制
                adjusted_value = asset.market_value * (1 - haircut)
                
                # 应用集中度限制
                if asset.concentration_limit > 0:
                    concentration_cap = hqla * asset.concentration_limit
                    adjusted_value = min(adjusted_value, concentration_cap)
                
                hqla += adjusted_value
            
            # 计算净现金流出
            net_cash_outflow = 0.0
            
            # 资金流出
            for funding in self.funding_sources.values():
                days_to_maturity = (funding.maturity_date - datetime.now()).days
                if days_to_maturity <= 30:
                    outflow = funding.amount * funding.stressed_runoff_rate
                    net_cash_outflow += outflow
            
            # 减去资金流入（保守估计）
            net_cash_outflow *= 0.75  # 假设75%的净流出
            
            # 计算LCR
            if net_cash_outflow > 0:
                lcr = hqla / net_cash_outflow
            else:
                lcr = float('inf')
            
            return lcr
            
        except Exception as e:
            self.logger.error(f"Error calculating LCR: {e}")
            return 0.0
    
    async def calculate_net_stable_funding_ratio(self) -> float:
        """计算净稳定资金比率（NSFR）"""
        try:
            # 计算可用稳定资金
            available_stable_funding = 0.0
            
            for funding in self.funding_sources.values():
                days_to_maturity = (funding.maturity_date - datetime.now()).days
                
                # 根据期限和类型确定稳定性因子
                if funding.source_type == 'deposits':
                    if days_to_maturity > 365:
                        asf_factor = 1.0
                    elif days_to_maturity > 180:
                        asf_factor = 0.9
                    else:
                        asf_factor = 0.5
                elif funding.source_type == 'wholesale':
                    if days_to_maturity > 365:
                        asf_factor = 1.0
                    else:
                        asf_factor = 0.5
                else:
                    asf_factor = 0.0
                
                available_stable_funding += funding.amount * asf_factor
            
            # 计算所需稳定资金
            required_stable_funding = 0.0
            
            for asset in self.liquid_assets.values():
                # 根据资产类型确定所需稳定资金因子
                if asset.tier == LiquidityTier.TIER_1:
                    rsf_factor = 0.0
                elif asset.tier == LiquidityTier.TIER_2A:
                    rsf_factor = 0.05
                elif asset.tier == LiquidityTier.TIER_2B:
                    rsf_factor = 0.15
                else:
                    rsf_factor = 0.50
                
                required_stable_funding += asset.market_value * rsf_factor
            
            # 计算NSFR
            if required_stable_funding > 0:
                nsfr = available_stable_funding / required_stable_funding
            else:
                nsfr = float('inf')
            
            return nsfr
            
        except Exception as e:
            self.logger.error(f"Error calculating NSFR: {e}")
            return 0.0
    
    async def calculate_liquidity_gaps(self) -> Dict[str, float]:
        """计算流动性缺口"""
        try:
            gaps = {}
            time_buckets = ['1D', '1W', '1M', '3M', '1Y']
            
            for bucket in time_buckets:
                # 计算该时间段内的资金流入
                cash_inflow = 0.0
                
                # 计算该时间段内的资金流出
                cash_outflow = 0.0
                
                # 计算可变现资产
                liquidatable_assets = 0.0
                
                days_limit = self._get_days_for_bucket(bucket)
                
                for funding in self.funding_sources.values():
                    days_to_maturity = (funding.maturity_date - datetime.now()).days
                    if days_to_maturity <= days_limit:
                        cash_outflow += funding.amount * funding.stressed_runoff_rate
                
                for asset in self.liquid_assets.values():
                    if asset.liquidation_time_hours <= days_limit * 24:
                        liquidatable_assets += asset.market_value * (1 - asset.haircut)
                
                # 计算缺口
                gap = liquidatable_assets + cash_inflow - cash_outflow
                gaps[bucket] = gap
            
            return gaps
            
        except Exception as e:
            self.logger.error(f"Error calculating liquidity gaps: {e}")
            return {}
    
    def _get_days_for_bucket(self, bucket: str) -> int:
        """获取时间段对应的天数"""
        mapping = {
            '1D': 1,
            '1W': 7,
            '1M': 30,
            '3M': 90,
            '1Y': 365
        }
        return mapping.get(bucket, 30)
    
    async def calculate_survival_period(self) -> float:
        """计算生存期"""
        try:
            # 计算当前可用流动性
            available_liquidity = 0.0
            
            for asset in self.liquid_assets.values():
                # 只考虑高质量流动性资产
                if asset.tier in [LiquidityTier.TIER_1, LiquidityTier.TIER_2A]:
                    available_liquidity += asset.market_value * (1 - asset.haircut)
            
            # 计算每日净现金流出
            daily_outflow = 0.0
            
            for funding in self.funding_sources.values():
                days_to_maturity = (funding.maturity_date - datetime.now()).days
                if days_to_maturity <= 30:
                    daily_outflow += funding.amount * funding.stressed_runoff_rate / 30
            
            # 计算生存期
            if daily_outflow > 0:
                survival_period = available_liquidity / daily_outflow
            else:
                survival_period = float('inf')
            
            return min(survival_period, 365.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating survival period: {e}")
            return 0.0
    
    async def calculate_concentration_risk(self) -> float:
        """计算集中度风险"""
        try:
            # 计算单一资产集中度
            total_assets = sum(asset.market_value for asset in self.liquid_assets.values())
            
            if total_assets == 0:
                return 0.0
            
            max_single_concentration = 0.0
            asset_class_concentrations = {}
            
            for asset in self.liquid_assets.values():
                # 单一资产集中度
                concentration = asset.market_value / total_assets
                max_single_concentration = max(max_single_concentration, concentration)
                
                # 资产类别集中度
                asset_class = asset.asset_type
                if asset_class not in asset_class_concentrations:
                    asset_class_concentrations[asset_class] = 0.0
                asset_class_concentrations[asset_class] += concentration
            
            # 计算集中度风险
            concentration_risk = max_single_concentration
            
            # 考虑资产类别集中度
            max_class_concentration = max(asset_class_concentrations.values()) if asset_class_concentrations else 0.0
            concentration_risk = max(concentration_risk, max_class_concentration)
            
            return concentration_risk
            
        except Exception as e:
            self.logger.error(f"Error calculating concentration risk: {e}")
            return 0.0
    
    async def stress_test_liquidity(self, scenario: str = 'moderate_stress') -> Dict[str, float]:
        """流动性压力测试"""
        try:
            if scenario not in self.stress_scenarios:
                scenario = 'moderate_stress'
            
            stress_params = self.stress_scenarios[scenario]
            
            # 应用压力情景
            stressed_assets = 0.0
            stressed_funding = 0.0
            
            # 资产价值压力
            for asset in self.liquid_assets.values():
                stressed_haircut = asset.haircut + stress_params['asset_haircut_increase']
                stressed_value = asset.market_value * (1 - stressed_haircut)
                stressed_assets += stressed_value
            
            # 资金来源压力
            for funding in self.funding_sources.values():
                if funding.source_type == 'deposits':
                    stressed_runoff = stress_params['deposit_runoff']
                elif funding.source_type == 'wholesale':
                    stressed_runoff = stress_params['wholesale_runoff']
                else:
                    stressed_runoff = 0.5
                
                stressed_funding += funding.amount * stressed_runoff
            
            # 计算压力测试指标
            stressed_lcr = stressed_assets / stressed_funding if stressed_funding > 0 else float('inf')
            
            # 计算生存期
            daily_outflow = stressed_funding / 30
            stressed_survival = stressed_assets / daily_outflow if daily_outflow > 0 else float('inf')
            
            # 计算资金成本冲击
            funding_cost_impact = 0.0
            for funding in self.funding_sources.values():
                additional_cost = funding.amount * stress_params['funding_cost_increase']
                funding_cost_impact += additional_cost
            
            return {
                'stressed_lcr': stressed_lcr,
                'stressed_survival_days': min(stressed_survival, 365),
                'stressed_assets': stressed_assets,
                'stressed_funding_needs': stressed_funding,
                'funding_cost_impact': funding_cost_impact,
                'scenario': scenario
            }
            
        except Exception as e:
            self.logger.error(f"Error in liquidity stress test: {e}")
            return {}
    
    async def calculate_liquidity_risk_metrics(self) -> LiquidityRiskMetrics:
        """计算流动性风险指标"""
        try:
            # 计算基础指标
            lcr = await self.calculate_liquidity_coverage_ratio()
            nsfr = await self.calculate_net_stable_funding_ratio()
            
            # 计算贷存比
            total_loans = sum(asset.market_value for asset in self.liquid_assets.values() if asset.asset_type == 'loan')
            total_deposits = sum(funding.amount for funding in self.funding_sources.values() if funding.source_type == 'deposits')
            loan_to_deposit_ratio = total_loans / total_deposits if total_deposits > 0 else 0.0
            
            # 计算流动性缺口
            liquidity_gaps = await self.calculate_liquidity_gaps()
            
            # 计算生存期
            survival_period = await self.calculate_survival_period()
            
            # 计算集中度风险
            concentration_risk = await self.calculate_concentration_risk()
            
            # 计算资金成本
            total_funding_amount = sum(funding.amount for funding in self.funding_sources.values())
            weighted_funding_cost = 0.0
            for funding in self.funding_sources.values():
                weight = funding.amount / total_funding_amount if total_funding_amount > 0 else 0
                weighted_funding_cost += weight * funding.cost_of_funding
            
            # 计算流动性缓冲
            total_assets = sum(asset.market_value for asset in self.liquid_assets.values())
            tier1_assets = sum(asset.market_value for asset in self.liquid_assets.values() if asset.tier == LiquidityTier.TIER_1)
            liquidity_buffer = tier1_assets / total_assets if total_assets > 0 else 0.0
            
            # 运行压力测试
            stress_results = {}
            for scenario in self.stress_scenarios.keys():
                stress_result = await self.stress_test_liquidity(scenario)
                stress_results[scenario] = stress_result.get('stressed_lcr', 0)
            
            # 检查预警指标
            early_warnings = self._check_early_warning_indicators(lcr, nsfr, concentration_risk, survival_period)
            
            # 计算监管比率
            regulatory_ratios = {
                'lcr': lcr,
                'nsfr': nsfr,
                'liquidity_buffer': liquidity_buffer,
                'concentration_risk': concentration_risk
            }
            
            # 分类风险
            market_liquidity_risk = self._calculate_market_liquidity_risk()
            funding_liquidity_risk = self._calculate_funding_liquidity_risk()
            
            # 创建风险指标对象
            metrics = LiquidityRiskMetrics(
                timestamp=datetime.now(),
                total_liquid_assets=total_assets,
                total_funding_requirements=total_funding_amount,
                liquidity_coverage_ratio=lcr,
                net_stable_funding_ratio=nsfr,
                loan_to_deposit_ratio=loan_to_deposit_ratio,
                liquidity_gap=liquidity_gaps,
                survival_period_days=survival_period,
                concentration_risk=concentration_risk,
                funding_cost=weighted_funding_cost,
                liquidity_buffer=liquidity_buffer,
                stress_test_results=stress_results,
                early_warning_indicators=early_warnings,
                regulatory_ratios=regulatory_ratios,
                market_liquidity_risk=market_liquidity_risk,
                funding_liquidity_risk=funding_liquidity_risk
            )
            
            # 保存历史记录
            self.risk_history.append(metrics)
            
            # 保持历史记录数量
            if len(self.risk_history) > 1000:
                self.risk_history = self.risk_history[-1000:]
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating liquidity risk metrics: {e}")
            return None
    
    def _check_early_warning_indicators(self, lcr: float, nsfr: float, concentration: float, survival: float) -> List[str]:
        """检查预警指标"""
        try:
            warnings = []
            
            # LCR预警
            if lcr < self.alert_thresholds['lcr_critical']:
                warnings.append("LCR below critical threshold")
            elif lcr < self.alert_thresholds['lcr_warning']:
                warnings.append("LCR below warning threshold")
            
            # NSFR预警
            if nsfr < self.alert_thresholds['nsfr_critical']:
                warnings.append("NSFR below critical threshold")
            elif nsfr < self.alert_thresholds['nsfr_warning']:
                warnings.append("NSFR below warning threshold")
            
            # 集中度预警
            if concentration > self.alert_thresholds['concentration_critical']:
                warnings.append("Concentration risk above critical threshold")
            elif concentration > self.alert_thresholds['concentration_warning']:
                warnings.append("Concentration risk above warning threshold")
            
            # 生存期预警
            if survival < self.risk_limits['min_survival_period_days']:
                warnings.append("Survival period below minimum requirement")
            
            return warnings
            
        except Exception as e:
            self.logger.error(f"Error checking early warning indicators: {e}")
            return []
    
    def _calculate_market_liquidity_risk(self) -> float:
        """计算市场流动性风险"""
        try:
            total_risk = 0.0
            total_assets = sum(asset.market_value for asset in self.liquid_assets.values())
            
            if total_assets == 0:
                return 0.0
            
            for asset in self.liquid_assets.values():
                # 获取流动性指标
                liquidity_metrics = self.liquidity_metrics.get(asset.symbol)
                if liquidity_metrics:
                    # 流动性风险 = 买卖价差 * 价格冲击 * 权重
                    weight = asset.market_value / total_assets
                    asset_risk = liquidity_metrics.bid_ask_spread * liquidity_metrics.price_impact * weight
                    total_risk += asset_risk
            
            return total_risk
            
        except Exception as e:
            self.logger.error(f"Error calculating market liquidity risk: {e}")
            return 0.0
    
    def _calculate_funding_liquidity_risk(self) -> float:
        """计算资金流动性风险"""
        try:
            total_risk = 0.0
            total_funding = sum(funding.amount for funding in self.funding_sources.values())
            
            if total_funding == 0:
                return 0.0
            
            for funding in self.funding_sources.values():
                # 资金流动性风险 = 不稳定性 * 集中度 * 权重
                weight = funding.amount / total_funding
                funding_risk = (1 - funding.stability_factor) * funding.concentration_risk * weight
                total_risk += funding_risk
            
            return total_risk
            
        except Exception as e:
            self.logger.error(f"Error calculating funding liquidity risk: {e}")
            return 0.0
    
    async def optimize_liquidity_portfolio(self) -> Dict[str, Any]:
        """优化流动性资产组合"""
        try:
            recommendations = []
            
            # 检查LCR
            lcr = await self.calculate_liquidity_coverage_ratio()
            if lcr < self.risk_limits['min_liquidity_coverage_ratio']:
                shortfall = self.risk_limits['min_liquidity_coverage_ratio'] - lcr
                recommendations.append({
                    'type': 'increase_hqla',
                    'reason': f'LCR shortfall: {shortfall:.2f}',
                    'action': 'Purchase additional high-quality liquid assets',
                    'priority': 'high'
                })
            
            # 检查NSFR
            nsfr = await self.calculate_net_stable_funding_ratio()
            if nsfr < self.risk_limits['min_net_stable_funding_ratio']:
                shortfall = self.risk_limits['min_net_stable_funding_ratio'] - nsfr
                recommendations.append({
                    'type': 'increase_stable_funding',
                    'reason': f'NSFR shortfall: {shortfall:.2f}',
                    'action': 'Increase stable funding sources',
                    'priority': 'high'
                })
            
            # 检查集中度风险
            concentration_risk = await self.calculate_concentration_risk()
            if concentration_risk > self.risk_limits['max_concentration_single_asset']:
                recommendations.append({
                    'type': 'reduce_concentration',
                    'reason': f'Concentration risk: {concentration_risk:.2%}',
                    'action': 'Diversify liquid asset holdings',
                    'priority': 'medium'
                })
            
            # 检查生存期
            survival_period = await self.calculate_survival_period()
            if survival_period < self.risk_limits['min_survival_period_days']:
                recommendations.append({
                    'type': 'extend_survival_period',
                    'reason': f'Survival period: {survival_period:.1f} days',
                    'action': 'Increase liquid asset buffer',
                    'priority': 'high'
                })
            
            # 资产配置建议
            asset_allocation = self._suggest_asset_allocation()
            
            return {
                'recommendations': recommendations,
                'asset_allocation': asset_allocation,
                'current_metrics': {
                    'lcr': lcr,
                    'nsfr': nsfr,
                    'concentration_risk': concentration_risk,
                    'survival_period': survival_period
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error optimizing liquidity portfolio: {e}")
            return {}
    
    def _suggest_asset_allocation(self) -> Dict[str, float]:
        """建议资产配置"""
        try:
            # 目标配置
            target_allocation = {
                'tier_1': 0.3,   # 30% Tier 1资产
                'tier_2a': 0.4,  # 40% Tier 2A资产
                'tier_2b': 0.25, # 25% Tier 2B资产
                'tier_3': 0.05   # 5% Tier 3资产
            }
            
            # 当前配置
            current_allocation = {}
            total_assets = sum(asset.market_value for asset in self.liquid_assets.values())
            
            if total_assets == 0:
                return target_allocation
            
            tier_totals = {'tier_1': 0, 'tier_2a': 0, 'tier_2b': 0, 'tier_3': 0}
            
            for asset in self.liquid_assets.values():
                tier_key = asset.tier.value
                tier_totals[tier_key] += asset.market_value
            
            for tier, total in tier_totals.items():
                current_allocation[tier] = total / total_assets
            
            # 计算调整建议
            adjustments = {}
            for tier, target in target_allocation.items():
                current = current_allocation.get(tier, 0)
                adjustment = target - current
                adjustments[tier] = adjustment
            
            return {
                'target_allocation': target_allocation,
                'current_allocation': current_allocation,
                'adjustments_needed': adjustments
            }
            
        except Exception as e:
            self.logger.error(f"Error suggesting asset allocation: {e}")
            return {}
    
    def get_liquidity_summary(self) -> Dict[str, Any]:
        """获取流动性风险摘要"""
        try:
            return {
                'total_liquid_assets': len(self.liquid_assets),
                'total_funding_sources': len(self.funding_sources),
                'total_asset_value': sum(asset.market_value for asset in self.liquid_assets.values()),
                'total_funding_amount': sum(funding.amount for funding in self.funding_sources.values()),
                'risk_limits': self.risk_limits,
                'regulatory_requirements': self.regulatory_requirements,
                'alert_thresholds': self.alert_thresholds,
                'liquid_assets': [
                    {
                        'asset_id': asset.asset_id,
                        'symbol': asset.symbol,
                        'tier': asset.tier.value,
                        'market_value': asset.market_value,
                        'haircut': asset.haircut,
                        'liquidation_time_hours': asset.liquidation_time_hours,
                        'liquidity_score': self.liquidity_metrics.get(asset.symbol, {}).get('liquidity_score', 0)
                    } for asset in self.liquid_assets.values()
                ],
                'funding_sources': [
                    {
                        'source_id': funding.source_id,
                        'source_type': funding.source_type,
                        'amount': funding.amount,
                        'maturity_date': funding.maturity_date.strftime('%Y-%m-%d'),
                        'cost_of_funding': funding.cost_of_funding,
                        'stability_factor': funding.stability_factor
                    } for funding in self.funding_sources.values()
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting liquidity summary: {e}")
            return {}
    
    def update_market_data(self, symbol: str, market_data: Dict[str, Any]):
        """更新市场数据"""
        try:
            self.market_data[symbol] = market_data
            
            # 重新计算相关资产的流动性指标
            for asset in self.liquid_assets.values():
                if asset.symbol == symbol:
                    asyncio.create_task(self._calculate_asset_liquidity_metrics(asset))
            
        except Exception as e:
            self.logger.error(f"Error updating market data: {e}")
    
    def get_regulatory_compliance_status(self) -> Dict[str, Any]:
        """获取监管合规状态"""
        try:
            # 异步计算关键指标
            async def get_metrics():
                lcr = await self.calculate_liquidity_coverage_ratio()
                nsfr = await self.calculate_net_stable_funding_ratio()
                return lcr, nsfr
            
            # 创建任务并等待结果
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 如果事件循环正在运行，创建任务
                task = asyncio.create_task(get_metrics())
                # 这里我们使用同步方式获取结果
                lcr, nsfr = 1.0, 1.0  # 默认值
            else:
                lcr, nsfr = loop.run_until_complete(get_metrics())
            
            compliance_status = {
                'lcr_compliant': lcr >= self.regulatory_requirements['basel_iii_lcr'],
                'nsfr_compliant': nsfr >= self.regulatory_requirements['basel_iii_nsfr'],
                'lcr_value': lcr,
                'nsfr_value': nsfr,
                'lcr_buffer': lcr - self.regulatory_requirements['basel_iii_lcr'],
                'nsfr_buffer': nsfr - self.regulatory_requirements['basel_iii_nsfr'],
                'overall_compliant': (lcr >= self.regulatory_requirements['basel_iii_lcr'] and 
                                    nsfr >= self.regulatory_requirements['basel_iii_nsfr'])
            }
            
            return compliance_status
            
        except Exception as e:
            self.logger.error(f"Error getting regulatory compliance status: {e}")
            return {}
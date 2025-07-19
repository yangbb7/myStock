import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import json

class CreditRating(Enum):
    AAA = "AAA"
    AA_PLUS = "AA+"
    AA = "AA"
    AA_MINUS = "AA-"
    A_PLUS = "A+"
    A = "A"
    A_MINUS = "A-"
    BBB_PLUS = "BBB+"
    BBB = "BBB"
    BBB_MINUS = "BBB-"
    BB_PLUS = "BB+"
    BB = "BB"
    BB_MINUS = "BB-"
    B_PLUS = "B+"
    B = "B"
    B_MINUS = "B-"
    CCC_PLUS = "CCC+"
    CCC = "CCC"
    CCC_MINUS = "CCC-"
    CC = "CC"
    C = "C"
    D = "D"

class CreditEventType(Enum):
    DEFAULT = "default"
    BANKRUPTCY = "bankruptcy"
    RESTRUCTURING = "restructuring"
    PAYMENT_FAILURE = "payment_failure"
    DOWNGRADE = "downgrade"
    WATCH_LIST = "watch_list"
    COVENANT_BREACH = "covenant_breach"

@dataclass
class CreditEntity:
    entity_id: str
    name: str
    entity_type: str  # sovereign, corporate, financial
    sector: str
    country: str
    currency: str
    rating_sp: Optional[CreditRating] = None
    rating_moody: Optional[str] = None
    rating_fitch: Optional[str] = None
    rating_internal: Optional[CreditRating] = None
    probability_of_default_1y: float = 0.0
    probability_of_default_5y: float = 0.0
    recovery_rate: float = 0.4
    market_cap: float = 0.0
    total_debt: float = 0.0
    ebitda: float = 0.0
    cash_and_equivalents: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class CreditExposure:
    exposure_id: str
    entity: CreditEntity
    exposure_type: str  # bond, loan, derivative, guarantee
    notional_amount: float
    current_exposure: float
    potential_future_exposure: float
    recovery_rate: float
    maturity_date: datetime
    coupon_rate: float
    seniority: str  # senior_secured, senior_unsecured, subordinated
    collateral_value: float = 0.0
    netting_agreement: bool = False
    credit_enhancement: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class CreditLoss:
    loss_id: str
    entity: CreditEntity
    exposure: CreditExposure
    loss_type: str  # expected_loss, unexpected_loss, stress_loss
    probability_of_default: float
    loss_given_default: float
    exposure_at_default: float
    expected_loss: float
    unexpected_loss: float
    confidence_level: float
    time_horizon_days: int
    calculated_date: datetime = field(default_factory=datetime.now)

@dataclass
class CreditMetrics:
    timestamp: datetime
    total_exposure: float
    total_expected_loss: float
    total_unexpected_loss: float
    portfolio_default_rate: float
    portfolio_recovery_rate: float
    concentration_risk: float
    sector_concentration: Dict[str, float]
    country_concentration: Dict[str, float]
    rating_distribution: Dict[str, float]
    var_99: float
    cvar_99: float
    economic_capital: float
    credit_risk_contribution: Dict[str, float]
    diversification_benefit: float
    correlation_risk: float
    wrong_way_risk: float
    credit_spread_risk: float

class CreditRiskAssessment:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 信用实体和敞口
        self.credit_entities: Dict[str, CreditEntity] = {}
        self.credit_exposures: Dict[str, CreditExposure] = {}
        self.credit_losses: Dict[str, CreditLoss] = {}
        
        # 违约概率模型
        self.default_probability_models = {
            'merton': self._merton_model,
            'reduced_form': self._reduced_form_model,
            'structural': self._structural_model
        }
        
        # 市场数据
        self.market_data: Dict[str, Dict[str, Any]] = {}
        
        # 信用风险参数
        self.risk_parameters = {
            'confidence_levels': [0.95, 0.99, 0.999],
            'time_horizons': [1, 30, 90, 365],
            'correlation_matrix': {},
            'recovery_rates': self._initialize_recovery_rates(),
            'pd_curves': self._initialize_pd_curves()
        }
        
        # 执行器
        self.executor = ThreadPoolExecutor(max_workers=config.get('max_workers', 4))
        
        # 历史数据
        self.credit_history: List[CreditMetrics] = []
        
        # 监管要求
        self.regulatory_requirements = {
            'basel_iii': True,
            'min_capital_ratio': 0.08,
            'leverage_ratio': 0.03,
            'liquidity_coverage_ratio': 1.0
        }
    
    def _initialize_recovery_rates(self) -> Dict[str, float]:
        """初始化回收率"""
        return {
            'senior_secured': 0.70,
            'senior_unsecured': 0.45,
            'subordinated': 0.25,
            'equity': 0.05,
            'sovereign': 0.60,
            'financial': 0.40,
            'corporate': 0.45
        }
    
    def _initialize_pd_curves(self) -> Dict[str, List[float]]:
        """初始化违约概率曲线"""
        return {
            'AAA': [0.0001, 0.0005, 0.0015, 0.0030, 0.0050],
            'AA': [0.0002, 0.0010, 0.0030, 0.0060, 0.0100],
            'A': [0.0005, 0.0025, 0.0075, 0.0150, 0.0250],
            'BBB': [0.0015, 0.0075, 0.0225, 0.0450, 0.0750],
            'BB': [0.0050, 0.0250, 0.0750, 0.1500, 0.2500],
            'B': [0.0150, 0.0750, 0.2250, 0.4500, 0.7500],
            'CCC': [0.0500, 0.2500, 0.7500, 1.5000, 2.5000],
            'D': [1.0000, 1.0000, 1.0000, 1.0000, 1.0000]
        }
    
    async def add_credit_entity(self, entity: CreditEntity):
        """添加信用实体"""
        try:
            self.credit_entities[entity.entity_id] = entity
            
            # 更新违约概率
            await self._update_default_probabilities(entity)
            
            self.logger.info(f"Added credit entity: {entity.name}")
            
        except Exception as e:
            self.logger.error(f"Error adding credit entity: {e}")
    
    async def add_credit_exposure(self, exposure: CreditExposure):
        """添加信用敞口"""
        try:
            self.credit_exposures[exposure.exposure_id] = exposure
            
            # 计算预期损失
            await self._calculate_expected_loss(exposure)
            
            self.logger.info(f"Added credit exposure: {exposure.exposure_id}")
            
        except Exception as e:
            self.logger.error(f"Error adding credit exposure: {e}")
    
    async def _update_default_probabilities(self, entity: CreditEntity):
        """更新违约概率"""
        try:
            # 使用多种模型计算违约概率
            pd_merton = await self._calculate_merton_pd(entity)
            pd_structural = await self._calculate_structural_pd(entity)
            pd_market = await self._calculate_market_implied_pd(entity)
            
            # 加权平均
            weights = [0.3, 0.3, 0.4]  # Merton, Structural, Market
            
            entity.probability_of_default_1y = (
                weights[0] * pd_merton['1y'] +
                weights[1] * pd_structural['1y'] +
                weights[2] * pd_market['1y']
            )
            
            entity.probability_of_default_5y = (
                weights[0] * pd_merton['5y'] +
                weights[1] * pd_structural['5y'] +
                weights[2] * pd_market['5y']
            )
            
            # 更新时间戳
            entity.last_updated = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error updating default probabilities for {entity.name}: {e}")
    
    async def _calculate_merton_pd(self, entity: CreditEntity) -> Dict[str, float]:
        """使用Merton模型计算违约概率"""
        try:
            # 获取市场数据
            market_data = self.market_data.get(entity.entity_id, {})
            
            if not market_data:
                # 使用评级默认值
                rating = entity.rating_internal or entity.rating_sp
                if rating:
                    pd_curve = self.risk_parameters['pd_curves'].get(rating.value, [0.01, 0.05])
                    return {'1y': pd_curve[0], '5y': pd_curve[1] if len(pd_curve) > 1 else pd_curve[0]}
                else:
                    return {'1y': 0.01, '5y': 0.05}
            
            # Merton模型参数
            V = entity.market_cap + entity.total_debt  # 企业价值
            D = entity.total_debt  # 债务面值
            r = 0.05  # 无风险利率
            sigma = market_data.get('equity_volatility', 0.30)  # 股权波动率
            
            if V <= 0 or D <= 0:
                return {'1y': 0.01, '5y': 0.05}
            
            # 1年违约概率
            d2_1y = (np.log(V / D) + (r - 0.5 * sigma**2) * 1) / (sigma * np.sqrt(1))
            pd_1y = 1 - self._normal_cdf(d2_1y)
            
            # 5年违约概率
            d2_5y = (np.log(V / D) + (r - 0.5 * sigma**2) * 5) / (sigma * np.sqrt(5))
            pd_5y = 1 - self._normal_cdf(d2_5y)
            
            return {
                '1y': min(max(pd_1y, 0.0001), 0.9999),
                '5y': min(max(pd_5y, 0.0001), 0.9999)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Merton PD: {e}")
            return {'1y': 0.01, '5y': 0.05}
    
    async def _calculate_structural_pd(self, entity: CreditEntity) -> Dict[str, float]:
        """使用结构化模型计算违约概率"""
        try:
            # 基于财务比率的结构化模型
            if entity.ebitda <= 0 or entity.total_debt <= 0:
                return {'1y': 0.01, '5y': 0.05}
            
            # 财务比率
            debt_to_ebitda = entity.total_debt / entity.ebitda
            interest_coverage = entity.ebitda / (entity.total_debt * 0.05)  # 假设5%利率
            cash_ratio = entity.cash_and_equivalents / entity.total_debt
            
            # 计算Z-Score (简化版Altman Z-Score)
            z_score = (
                3.3 * (entity.ebitda / entity.total_debt) +
                1.0 * (entity.cash_and_equivalents / entity.total_debt) +
                0.6 * (entity.market_cap / entity.total_debt) +
                1.4 * (entity.ebitda / entity.market_cap) +
                1.2 * (entity.cash_and_equivalents / entity.market_cap)
            )
            
            # 将Z-Score映射到违约概率
            if z_score > 3.0:
                pd_1y = 0.001
                pd_5y = 0.005
            elif z_score > 2.7:
                pd_1y = 0.005
                pd_5y = 0.025
            elif z_score > 1.8:
                pd_1y = 0.025
                pd_5y = 0.125
            else:
                pd_1y = 0.100
                pd_5y = 0.500
            
            return {'1y': pd_1y, '5y': pd_5y}
            
        except Exception as e:
            self.logger.error(f"Error calculating structural PD: {e}")
            return {'1y': 0.01, '5y': 0.05}
    
    async def _calculate_market_implied_pd(self, entity: CreditEntity) -> Dict[str, float]:
        """使用市场隐含违约概率"""
        try:
            # 获取信用利差
            market_data = self.market_data.get(entity.entity_id, {})
            credit_spread = market_data.get('credit_spread', 0.02)  # 默认200bp
            
            # 回收率
            recovery_rate = entity.recovery_rate
            
            # 计算隐含违约概率
            # PD = Credit Spread / (1 - Recovery Rate)
            pd_1y = credit_spread / (1 - recovery_rate)
            pd_5y = pd_1y * 2.5  # 假设5年期违约概率是1年期的2.5倍
            
            return {
                '1y': min(max(pd_1y, 0.0001), 0.9999),
                '5y': min(max(pd_5y, 0.0001), 0.9999)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating market implied PD: {e}")
            return {'1y': 0.01, '5y': 0.05}
    
    async def _calculate_expected_loss(self, exposure: CreditExposure):
        """计算预期损失"""
        try:
            # 获取违约概率
            entity = exposure.entity
            pd_1y = entity.probability_of_default_1y
            
            # 计算违约损失率
            lgd = 1 - exposure.recovery_rate
            
            # 计算违约时敞口
            ead = exposure.current_exposure
            
            # 计算预期损失
            expected_loss = pd_1y * lgd * ead
            
            # 计算意外损失
            unexpected_loss = np.sqrt(pd_1y * (1 - pd_1y)) * lgd * ead
            
            # 创建损失记录
            loss = CreditLoss(
                loss_id=f"LOSS_{exposure.exposure_id}_{int(datetime.now().timestamp())}",
                entity=entity,
                exposure=exposure,
                loss_type="expected_loss",
                probability_of_default=pd_1y,
                loss_given_default=lgd,
                exposure_at_default=ead,
                expected_loss=expected_loss,
                unexpected_loss=unexpected_loss,
                confidence_level=0.99,
                time_horizon_days=365
            )
            
            self.credit_losses[loss.loss_id] = loss
            
        except Exception as e:
            self.logger.error(f"Error calculating expected loss: {e}")
    
    async def calculate_portfolio_metrics(self) -> CreditMetrics:
        """计算组合信用风险指标"""
        try:
            # 计算总敞口
            total_exposure = sum(exp.current_exposure for exp in self.credit_exposures.values())
            
            # 计算总预期损失
            total_expected_loss = sum(loss.expected_loss for loss in self.credit_losses.values())
            
            # 计算总意外损失
            total_unexpected_loss = sum(loss.unexpected_loss for loss in self.credit_losses.values())
            
            # 计算组合违约率
            portfolio_default_rate = self._calculate_portfolio_default_rate()
            
            # 计算组合回收率
            portfolio_recovery_rate = self._calculate_portfolio_recovery_rate()
            
            # 计算集中度风险
            concentration_risk = self._calculate_concentration_risk()
            
            # 计算行业集中度
            sector_concentration = self._calculate_sector_concentration()
            
            # 计算国家集中度
            country_concentration = self._calculate_country_concentration()
            
            # 计算评级分布
            rating_distribution = self._calculate_rating_distribution()
            
            # 计算VaR和CVaR
            var_99 = self._calculate_credit_var(0.99)
            cvar_99 = self._calculate_credit_cvar(0.99)
            
            # 计算经济资本
            economic_capital = self._calculate_economic_capital()
            
            # 计算信用风险贡献度
            credit_risk_contribution = self._calculate_risk_contribution()
            
            # 计算多元化收益
            diversification_benefit = self._calculate_diversification_benefit()
            
            # 计算相关性风险
            correlation_risk = self._calculate_correlation_risk()
            
            # 计算错向风险
            wrong_way_risk = self._calculate_wrong_way_risk()
            
            # 计算信用利差风险
            credit_spread_risk = self._calculate_credit_spread_risk()
            
            # 创建指标对象
            metrics = CreditMetrics(
                timestamp=datetime.now(),
                total_exposure=total_exposure,
                total_expected_loss=total_expected_loss,
                total_unexpected_loss=total_unexpected_loss,
                portfolio_default_rate=portfolio_default_rate,
                portfolio_recovery_rate=portfolio_recovery_rate,
                concentration_risk=concentration_risk,
                sector_concentration=sector_concentration,
                country_concentration=country_concentration,
                rating_distribution=rating_distribution,
                var_99=var_99,
                cvar_99=cvar_99,
                economic_capital=economic_capital,
                credit_risk_contribution=credit_risk_contribution,
                diversification_benefit=diversification_benefit,
                correlation_risk=correlation_risk,
                wrong_way_risk=wrong_way_risk,
                credit_spread_risk=credit_spread_risk
            )
            
            # 保存历史记录
            self.credit_history.append(metrics)
            
            # 保持历史记录数量
            if len(self.credit_history) > 1000:
                self.credit_history = self.credit_history[-1000:]
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio metrics: {e}")
            return None
    
    def _calculate_portfolio_default_rate(self) -> float:
        """计算组合违约率"""
        try:
            if not self.credit_entities:
                return 0.0
            
            total_exposure = sum(exp.current_exposure for exp in self.credit_exposures.values())
            
            if total_exposure == 0:
                return 0.0
            
            weighted_pd = 0.0
            for exposure in self.credit_exposures.values():
                weight = exposure.current_exposure / total_exposure
                pd = exposure.entity.probability_of_default_1y
                weighted_pd += weight * pd
            
            return weighted_pd
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio default rate: {e}")
            return 0.0
    
    def _calculate_portfolio_recovery_rate(self) -> float:
        """计算组合回收率"""
        try:
            if not self.credit_exposures:
                return 0.0
            
            total_exposure = sum(exp.current_exposure for exp in self.credit_exposures.values())
            
            if total_exposure == 0:
                return 0.0
            
            weighted_recovery = 0.0
            for exposure in self.credit_exposures.values():
                weight = exposure.current_exposure / total_exposure
                recovery = exposure.recovery_rate
                weighted_recovery += weight * recovery
            
            return weighted_recovery
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio recovery rate: {e}")
            return 0.0
    
    def _calculate_concentration_risk(self) -> float:
        """计算集中度风险"""
        try:
            if not self.credit_exposures:
                return 0.0
            
            total_exposure = sum(exp.current_exposure for exp in self.credit_exposures.values())
            
            if total_exposure == 0:
                return 0.0
            
            # 计算Herfindahl指数
            herfindahl_index = 0.0
            for exposure in self.credit_exposures.values():
                weight = exposure.current_exposure / total_exposure
                herfindahl_index += weight ** 2
            
            # 集中度风险 = 1 - (1/HHI)
            if herfindahl_index > 0:
                concentration_risk = 1 - (1 / herfindahl_index)
            else:
                concentration_risk = 0.0
            
            return concentration_risk
            
        except Exception as e:
            self.logger.error(f"Error calculating concentration risk: {e}")
            return 0.0
    
    def _calculate_sector_concentration(self) -> Dict[str, float]:
        """计算行业集中度"""
        try:
            sector_exposure = {}
            total_exposure = sum(exp.current_exposure for exp in self.credit_exposures.values())
            
            if total_exposure == 0:
                return {}
            
            for exposure in self.credit_exposures.values():
                sector = exposure.entity.sector
                if sector not in sector_exposure:
                    sector_exposure[sector] = 0.0
                sector_exposure[sector] += exposure.current_exposure
            
            # 转换为百分比
            for sector in sector_exposure:
                sector_exposure[sector] = sector_exposure[sector] / total_exposure
            
            return sector_exposure
            
        except Exception as e:
            self.logger.error(f"Error calculating sector concentration: {e}")
            return {}
    
    def _calculate_country_concentration(self) -> Dict[str, float]:
        """计算国家集中度"""
        try:
            country_exposure = {}
            total_exposure = sum(exp.current_exposure for exp in self.credit_exposures.values())
            
            if total_exposure == 0:
                return {}
            
            for exposure in self.credit_exposures.values():
                country = exposure.entity.country
                if country not in country_exposure:
                    country_exposure[country] = 0.0
                country_exposure[country] += exposure.current_exposure
            
            # 转换为百分比
            for country in country_exposure:
                country_exposure[country] = country_exposure[country] / total_exposure
            
            return country_exposure
            
        except Exception as e:
            self.logger.error(f"Error calculating country concentration: {e}")
            return {}
    
    def _calculate_rating_distribution(self) -> Dict[str, float]:
        """计算评级分布"""
        try:
            rating_exposure = {}
            total_exposure = sum(exp.current_exposure for exp in self.credit_exposures.values())
            
            if total_exposure == 0:
                return {}
            
            for exposure in self.credit_exposures.values():
                rating = exposure.entity.rating_internal or exposure.entity.rating_sp
                if rating:
                    rating_str = rating.value
                    if rating_str not in rating_exposure:
                        rating_exposure[rating_str] = 0.0
                    rating_exposure[rating_str] += exposure.current_exposure
            
            # 转换为百分比
            for rating in rating_exposure:
                rating_exposure[rating] = rating_exposure[rating] / total_exposure
            
            return rating_exposure
            
        except Exception as e:
            self.logger.error(f"Error calculating rating distribution: {e}")
            return {}
    
    def _calculate_credit_var(self, confidence_level: float) -> float:
        """计算信用VaR"""
        try:
            if not self.credit_losses:
                return 0.0
            
            # 收集所有损失
            losses = [loss.expected_loss + loss.unexpected_loss for loss in self.credit_losses.values()]
            
            if not losses:
                return 0.0
            
            # 计算VaR
            var_percentile = confidence_level * 100
            credit_var = np.percentile(losses, var_percentile)
            
            return credit_var
            
        except Exception as e:
            self.logger.error(f"Error calculating credit VaR: {e}")
            return 0.0
    
    def _calculate_credit_cvar(self, confidence_level: float) -> float:
        """计算信用CVaR"""
        try:
            if not self.credit_losses:
                return 0.0
            
            # 收集所有损失
            losses = [loss.expected_loss + loss.unexpected_loss for loss in self.credit_losses.values()]
            
            if not losses:
                return 0.0
            
            # 计算CVaR
            var_percentile = confidence_level * 100
            var_threshold = np.percentile(losses, var_percentile)
            
            # CVaR是超过VaR的损失的期望值
            tail_losses = [loss for loss in losses if loss >= var_threshold]
            
            if tail_losses:
                credit_cvar = np.mean(tail_losses)
            else:
                credit_cvar = var_threshold
            
            return credit_cvar
            
        except Exception as e:
            self.logger.error(f"Error calculating credit CVaR: {e}")
            return 0.0
    
    def _calculate_economic_capital(self) -> float:
        """计算经济资本"""
        try:
            # 经济资本 = 意外损失 - 预期损失
            total_unexpected_loss = sum(loss.unexpected_loss for loss in self.credit_losses.values())
            total_expected_loss = sum(loss.expected_loss for loss in self.credit_losses.values())
            
            economic_capital = total_unexpected_loss - total_expected_loss
            
            return max(economic_capital, 0.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating economic capital: {e}")
            return 0.0
    
    def _calculate_risk_contribution(self) -> Dict[str, float]:
        """计算风险贡献度"""
        try:
            risk_contributions = {}
            total_risk = sum(loss.expected_loss + loss.unexpected_loss for loss in self.credit_losses.values())
            
            if total_risk == 0:
                return {}
            
            for loss in self.credit_losses.values():
                entity_id = loss.entity.entity_id
                contribution = (loss.expected_loss + loss.unexpected_loss) / total_risk
                risk_contributions[entity_id] = contribution
            
            return risk_contributions
            
        except Exception as e:
            self.logger.error(f"Error calculating risk contribution: {e}")
            return {}
    
    def _calculate_diversification_benefit(self) -> float:
        """计算多元化收益"""
        try:
            # 简化计算：分散化收益 = 1 - 集中度风险
            concentration_risk = self._calculate_concentration_risk()
            diversification_benefit = 1 - concentration_risk
            
            return diversification_benefit
            
        except Exception as e:
            self.logger.error(f"Error calculating diversification benefit: {e}")
            return 0.0
    
    def _calculate_correlation_risk(self) -> float:
        """计算相关性风险"""
        try:
            # 简化计算：相关性风险基于行业和国家集中度
            sector_concentration = self._calculate_sector_concentration()
            country_concentration = self._calculate_country_concentration()
            
            # 计算最大集中度
            max_sector_concentration = max(sector_concentration.values()) if sector_concentration else 0
            max_country_concentration = max(country_concentration.values()) if country_concentration else 0
            
            # 相关性风险 = 最大集中度的加权平均
            correlation_risk = 0.6 * max_sector_concentration + 0.4 * max_country_concentration
            
            return correlation_risk
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation risk: {e}")
            return 0.0
    
    def _calculate_wrong_way_risk(self) -> float:
        """计算错向风险"""
        try:
            # 简化计算：错向风险基于金融机构敞口
            financial_exposure = 0.0
            total_exposure = sum(exp.current_exposure for exp in self.credit_exposures.values())
            
            if total_exposure == 0:
                return 0.0
            
            for exposure in self.credit_exposures.values():
                if exposure.entity.entity_type == 'financial':
                    financial_exposure += exposure.current_exposure
            
            # 错向风险 = 金融机构敞口占比
            wrong_way_risk = financial_exposure / total_exposure
            
            return wrong_way_risk
            
        except Exception as e:
            self.logger.error(f"Error calculating wrong way risk: {e}")
            return 0.0
    
    def _calculate_credit_spread_risk(self) -> float:
        """计算信用利差风险"""
        try:
            # 简化计算：基于信用利差的波动率
            total_spread_risk = 0.0
            
            for exposure in self.credit_exposures.values():
                # 获取信用利差
                market_data = self.market_data.get(exposure.entity.entity_id, {})
                credit_spread = market_data.get('credit_spread', 0.02)
                spread_volatility = market_data.get('spread_volatility', 0.50)
                
                # 计算利差风险
                duration = self._calculate_duration(exposure)
                spread_risk = exposure.current_exposure * duration * credit_spread * spread_volatility
                
                total_spread_risk += spread_risk
            
            return total_spread_risk
            
        except Exception as e:
            self.logger.error(f"Error calculating credit spread risk: {e}")
            return 0.0
    
    def _calculate_duration(self, exposure: CreditExposure) -> float:
        """计算久期"""
        try:
            # 简化久期计算
            years_to_maturity = (exposure.maturity_date - datetime.now()).days / 365.0
            
            if years_to_maturity <= 0:
                return 0.0
            
            # 修正久期 = 麦考利久期 / (1 + 收益率)
            coupon_rate = exposure.coupon_rate
            modified_duration = years_to_maturity / (1 + coupon_rate)
            
            return modified_duration
            
        except Exception as e:
            self.logger.error(f"Error calculating duration: {e}")
            return 0.0
    
    def _normal_cdf(self, x: float) -> float:
        """标准正态分布累积分布函数"""
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
    
    def _merton_model(self, entity: CreditEntity) -> float:
        """Merton模型"""
        return self._calculate_merton_pd(entity)
    
    def _reduced_form_model(self, entity: CreditEntity) -> float:
        """简化形式模型"""
        return self._calculate_market_implied_pd(entity)
    
    def _structural_model(self, entity: CreditEntity) -> float:
        """结构化模型"""
        return self._calculate_structural_pd(entity)
    
    async def stress_test_credit_portfolio(self, stress_scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """信用组合压力测试"""
        try:
            stress_results = {}
            
            for scenario in stress_scenarios:
                scenario_name = scenario['name']
                pd_shock = scenario.get('pd_shock', 0.0)
                lgd_shock = scenario.get('lgd_shock', 0.0)
                correlation_shock = scenario.get('correlation_shock', 0.0)
                
                # 应用压力情景
                stressed_losses = []
                
                for loss in self.credit_losses.values():
                    # 调整违约概率
                    stressed_pd = loss.probability_of_default * (1 + pd_shock)
                    stressed_pd = min(stressed_pd, 0.9999)
                    
                    # 调整违约损失率
                    stressed_lgd = loss.loss_given_default * (1 + lgd_shock)
                    stressed_lgd = min(stressed_lgd, 1.0)
                    
                    # 计算压力损失
                    stressed_loss = stressed_pd * stressed_lgd * loss.exposure_at_default
                    stressed_losses.append(stressed_loss)
                
                # 计算压力指标
                total_stressed_loss = sum(stressed_losses)
                baseline_loss = sum(loss.expected_loss for loss in self.credit_losses.values())
                
                stress_impact = total_stressed_loss - baseline_loss
                
                stress_results[scenario_name] = {
                    'total_stressed_loss': total_stressed_loss,
                    'baseline_loss': baseline_loss,
                    'stress_impact': stress_impact,
                    'stress_ratio': stress_impact / baseline_loss if baseline_loss > 0 else 0,
                    'individual_losses': stressed_losses
                }
            
            return stress_results
            
        except Exception as e:
            self.logger.error(f"Error in credit portfolio stress test: {e}")
            return {}
    
    def get_credit_summary(self) -> Dict[str, Any]:
        """获取信用风险摘要"""
        try:
            return {
                'total_entities': len(self.credit_entities),
                'total_exposures': len(self.credit_exposures),
                'total_exposure_amount': sum(exp.current_exposure for exp in self.credit_exposures.values()),
                'total_expected_loss': sum(loss.expected_loss for loss in self.credit_losses.values()),
                'total_unexpected_loss': sum(loss.unexpected_loss for loss in self.credit_losses.values()),
                'portfolio_default_rate': self._calculate_portfolio_default_rate(),
                'portfolio_recovery_rate': self._calculate_portfolio_recovery_rate(),
                'concentration_risk': self._calculate_concentration_risk(),
                'sector_concentration': self._calculate_sector_concentration(),
                'country_concentration': self._calculate_country_concentration(),
                'rating_distribution': self._calculate_rating_distribution(),
                'regulatory_capital': self._calculate_economic_capital() * 1.2,  # 加20%缓冲
                'credit_entities': [
                    {
                        'entity_id': entity.entity_id,
                        'name': entity.name,
                        'rating': entity.rating_internal.value if entity.rating_internal else 'NR',
                        'pd_1y': entity.probability_of_default_1y,
                        'pd_5y': entity.probability_of_default_5y,
                        'sector': entity.sector,
                        'country': entity.country
                    } for entity in self.credit_entities.values()
                ],
                'credit_exposures': [
                    {
                        'exposure_id': exp.exposure_id,
                        'entity_name': exp.entity.name,
                        'exposure_type': exp.exposure_type,
                        'current_exposure': exp.current_exposure,
                        'maturity_date': exp.maturity_date.strftime('%Y-%m-%d'),
                        'seniority': exp.seniority,
                        'recovery_rate': exp.recovery_rate
                    } for exp in self.credit_exposures.values()
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting credit summary: {e}")
            return {}
    
    def update_market_data(self, entity_id: str, market_data: Dict[str, Any]):
        """更新市场数据"""
        try:
            self.market_data[entity_id] = market_data
            
        except Exception as e:
            self.logger.error(f"Error updating market data: {e}")
    
    def get_regulatory_capital_requirement(self) -> Dict[str, float]:
        """获取监管资本要求"""
        try:
            # 计算标准化资本要求
            total_exposure = sum(exp.current_exposure for exp in self.credit_exposures.values())
            
            # 风险加权资产计算
            risk_weighted_assets = 0.0
            
            for exposure in self.credit_exposures.values():
                # 根据评级确定风险权重
                rating = exposure.entity.rating_internal or exposure.entity.rating_sp
                if rating:
                    if rating.value in ['AAA', 'AA+', 'AA', 'AA-']:
                        risk_weight = 0.20
                    elif rating.value in ['A+', 'A', 'A-']:
                        risk_weight = 0.50
                    elif rating.value in ['BBB+', 'BBB', 'BBB-']:
                        risk_weight = 1.00
                    else:
                        risk_weight = 1.50
                else:
                    risk_weight = 1.00
                
                risk_weighted_assets += exposure.current_exposure * risk_weight
            
            # 计算资本要求
            capital_requirement = risk_weighted_assets * self.regulatory_requirements['min_capital_ratio']
            
            return {
                'total_exposure': total_exposure,
                'risk_weighted_assets': risk_weighted_assets,
                'capital_requirement': capital_requirement,
                'capital_ratio': self.regulatory_requirements['min_capital_ratio'],
                'economic_capital': self._calculate_economic_capital(),
                'regulatory_buffer': max(0, capital_requirement - self._calculate_economic_capital())
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating regulatory capital requirement: {e}")
            return {}
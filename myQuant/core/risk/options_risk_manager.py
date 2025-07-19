import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from scipy.stats import norm
from concurrent.futures import ThreadPoolExecutor
import math

class OptionType(Enum):
    CALL = "call"
    PUT = "put"

class OptionStrategy(Enum):
    COVERED_CALL = "covered_call"
    PROTECTIVE_PUT = "protective_put"
    COLLAR = "collar"
    STRADDLE = "straddle"
    STRANGLE = "strangle"
    BUTTERFLY = "butterfly"
    CONDOR = "condor"
    IRON_CONDOR = "iron_condor"
    CALENDAR_SPREAD = "calendar_spread"
    DIAGONAL_SPREAD = "diagonal_spread"

@dataclass
class OptionContract:
    symbol: str
    underlying_symbol: str
    option_type: OptionType
    strike_price: float
    expiration_date: datetime
    quantity: int
    market_price: float
    bid_price: float
    ask_price: float
    implied_volatility: float
    volume: int
    open_interest: int
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class OptionPosition:
    contract: OptionContract
    quantity: int
    entry_price: float
    current_price: float
    unrealized_pnl: float
    delta_exposure: float
    gamma_exposure: float
    theta_exposure: float
    vega_exposure: float
    rho_exposure: float
    days_to_expiration: int
    position_value: float

@dataclass
class OptionPortfolio:
    positions: List[OptionPosition]
    total_value: float
    total_delta: float
    total_gamma: float
    total_theta: float
    total_vega: float
    total_rho: float
    net_premium: float
    unrealized_pnl: float
    margin_requirement: float
    buying_power: float
    cash_requirement: float

@dataclass
class OptionRiskMetrics:
    timestamp: datetime
    portfolio_value: float
    total_delta: float
    total_gamma: float
    total_theta: float
    total_vega: float
    total_rho: float
    delta_adjusted_exposure: float
    gamma_risk: float
    vega_risk: float
    time_decay_risk: float
    pin_risk: float
    early_exercise_risk: float
    assignment_risk: float
    max_loss: float
    max_profit: float
    break_even_points: List[float]
    profit_probability: float
    kelly_criterion: float
    sharpe_ratio: float
    var_1d: float
    cvar_1d: float
    expected_move: float
    realized_volatility: float
    iv_rank: float
    skew_risk: float

class OptionRiskManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 期权组合
        self.option_portfolio = OptionPortfolio(
            positions=[],
            total_value=0.0,
            total_delta=0.0,
            total_gamma=0.0,
            total_theta=0.0,
            total_vega=0.0,
            total_rho=0.0,
            net_premium=0.0,
            unrealized_pnl=0.0,
            margin_requirement=0.0,
            buying_power=0.0,
            cash_requirement=0.0
        )
        
        # 风险限制
        self.risk_limits = {
            'max_delta_exposure': config.get('max_delta_exposure', 10000),
            'max_gamma_exposure': config.get('max_gamma_exposure', 1000),
            'max_vega_exposure': config.get('max_vega_exposure', 5000),
            'max_theta_exposure': config.get('max_theta_exposure', 1000),
            'max_single_position_size': config.get('max_single_position_size', 100),
            'max_portfolio_value': config.get('max_portfolio_value', 1000000),
            'min_days_to_expiration': config.get('min_days_to_expiration', 7),
            'max_implied_volatility': config.get('max_implied_volatility', 1.0),
            'min_liquidity_threshold': config.get('min_liquidity_threshold', 10)
        }
        
        # 市场数据
        self.market_data: Dict[str, Dict[str, Any]] = {}
        self.risk_free_rate = config.get('risk_free_rate', 0.05)
        
        # 计算引擎
        self.executor = ThreadPoolExecutor(max_workers=config.get('max_workers', 4))
        
        # 历史数据
        self.risk_history: List[OptionRiskMetrics] = []
        
    async def calculate_option_greeks(self, contract: OptionContract, spot_price: float) -> Dict[str, float]:
        """计算期权希腊字母"""
        try:
            # 计算参数
            S = spot_price
            K = contract.strike_price
            T = (contract.expiration_date - datetime.now()).days / 365.0
            r = self.risk_free_rate
            sigma = contract.implied_volatility
            
            if T <= 0:
                return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
            
            # 计算d1和d2
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            # 标准正态分布函数
            N_d1 = norm.cdf(d1)
            N_d2 = norm.cdf(d2)
            n_d1 = norm.pdf(d1)
            
            if contract.option_type == OptionType.CALL:
                # 看涨期权
                delta = N_d1
                gamma = n_d1 / (S * sigma * np.sqrt(T))
                theta = (-(S * n_d1 * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * N_d2) / 365
                vega = S * n_d1 * np.sqrt(T) / 100
                rho = K * T * np.exp(-r * T) * N_d2 / 100
            else:
                # 看跌期权
                delta = N_d1 - 1
                gamma = n_d1 / (S * sigma * np.sqrt(T))
                theta = (-(S * n_d1 * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * (1 - N_d2)) / 365
                vega = S * n_d1 * np.sqrt(T) / 100
                rho = -K * T * np.exp(-r * T) * (1 - N_d2) / 100
            
            return {
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega,
                'rho': rho
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Greeks for {contract.symbol}: {e}")
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
    
    async def calculate_option_price(self, contract: OptionContract, spot_price: float) -> float:
        """使用Black-Scholes模型计算期权价格"""
        try:
            S = spot_price
            K = contract.strike_price
            T = (contract.expiration_date - datetime.now()).days / 365.0
            r = self.risk_free_rate
            sigma = contract.implied_volatility
            
            if T <= 0:
                if contract.option_type == OptionType.CALL:
                    return max(S - K, 0)
                else:
                    return max(K - S, 0)
            
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            if contract.option_type == OptionType.CALL:
                price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            else:
                price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            
            return max(price, 0)
            
        except Exception as e:
            self.logger.error(f"Error calculating option price for {contract.symbol}: {e}")
            return 0.0
    
    async def update_position_greeks(self, position: OptionPosition, spot_price: float):
        """更新持仓的希腊字母"""
        try:
            greeks = await self.calculate_option_greeks(position.contract, spot_price)
            
            # 更新希腊字母
            position.contract.delta = greeks['delta']
            position.contract.gamma = greeks['gamma']
            position.contract.theta = greeks['theta']
            position.contract.vega = greeks['vega']
            position.contract.rho = greeks['rho']
            
            # 更新持仓敞口
            position.delta_exposure = position.quantity * greeks['delta'] * spot_price
            position.gamma_exposure = position.quantity * greeks['gamma'] * spot_price
            position.theta_exposure = position.quantity * greeks['theta']
            position.vega_exposure = position.quantity * greeks['vega']
            position.rho_exposure = position.quantity * greeks['rho']
            
            # 更新到期天数
            position.days_to_expiration = (position.contract.expiration_date - datetime.now()).days
            
            # 更新持仓价值
            position.position_value = position.quantity * position.current_price * 100  # 期权合约乘数
            
            # 更新未实现盈亏
            position.unrealized_pnl = position.quantity * (position.current_price - position.entry_price) * 100
            
        except Exception as e:
            self.logger.error(f"Error updating position Greeks: {e}")
    
    async def calculate_portfolio_risk_metrics(self) -> OptionRiskMetrics:
        """计算组合风险指标"""
        try:
            # 更新组合总值
            self.option_portfolio.total_value = sum(pos.position_value for pos in self.option_portfolio.positions)
            
            # 计算总希腊字母
            self.option_portfolio.total_delta = sum(pos.delta_exposure for pos in self.option_portfolio.positions)
            self.option_portfolio.total_gamma = sum(pos.gamma_exposure for pos in self.option_portfolio.positions)
            self.option_portfolio.total_theta = sum(pos.theta_exposure for pos in self.option_portfolio.positions)
            self.option_portfolio.total_vega = sum(pos.vega_exposure for pos in self.option_portfolio.positions)
            self.option_portfolio.total_rho = sum(pos.rho_exposure for pos in self.option_portfolio.positions)
            
            # 计算风险指标
            delta_adjusted_exposure = abs(self.option_portfolio.total_delta)
            gamma_risk = self.option_portfolio.total_gamma * 0.01  # 1%股价变动的Gamma风险
            vega_risk = self.option_portfolio.total_vega * 0.01  # 1%波动率变动的Vega风险
            time_decay_risk = abs(self.option_portfolio.total_theta)  # 时间衰减风险
            
            # 计算其他风险指标
            pin_risk = self._calculate_pin_risk()
            early_exercise_risk = self._calculate_early_exercise_risk()
            assignment_risk = self._calculate_assignment_risk()
            
            # 计算最大盈亏
            max_loss, max_profit = self._calculate_max_profit_loss()
            
            # 计算盈亏平衡点
            break_even_points = self._calculate_break_even_points()
            
            # 计算获利概率
            profit_probability = self._calculate_profit_probability()
            
            # 计算Kelly准则
            kelly_criterion = self._calculate_kelly_criterion()
            
            # 计算VaR和CVaR
            var_1d = self._calculate_option_var()
            cvar_1d = self._calculate_option_cvar()
            
            # 计算期望移动
            expected_move = self._calculate_expected_move()
            
            # 计算IV等级
            iv_rank = self._calculate_iv_rank()
            
            # 计算偏度风险
            skew_risk = self._calculate_skew_risk()
            
            # 创建风险指标对象
            risk_metrics = OptionRiskMetrics(
                timestamp=datetime.now(),
                portfolio_value=self.option_portfolio.total_value,
                total_delta=self.option_portfolio.total_delta,
                total_gamma=self.option_portfolio.total_gamma,
                total_theta=self.option_portfolio.total_theta,
                total_vega=self.option_portfolio.total_vega,
                total_rho=self.option_portfolio.total_rho,
                delta_adjusted_exposure=delta_adjusted_exposure,
                gamma_risk=gamma_risk,
                vega_risk=vega_risk,
                time_decay_risk=time_decay_risk,
                pin_risk=pin_risk,
                early_exercise_risk=early_exercise_risk,
                assignment_risk=assignment_risk,
                max_loss=max_loss,
                max_profit=max_profit,
                break_even_points=break_even_points,
                profit_probability=profit_probability,
                kelly_criterion=kelly_criterion,
                sharpe_ratio=self._calculate_sharpe_ratio(),
                var_1d=var_1d,
                cvar_1d=cvar_1d,
                expected_move=expected_move,
                realized_volatility=self._calculate_realized_volatility(),
                iv_rank=iv_rank,
                skew_risk=skew_risk
            )
            
            # 记录历史
            self.risk_history.append(risk_metrics)
            
            # 保持历史记录数量
            if len(self.risk_history) > 1000:
                self.risk_history = self.risk_history[-1000:]
            
            return risk_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio risk metrics: {e}")
            return None
    
    def _calculate_pin_risk(self) -> float:
        """计算Pin风险"""
        try:
            pin_risk = 0.0
            
            for position in self.option_portfolio.positions:
                if position.days_to_expiration <= 7:  # 临近到期
                    # 获取标的价格
                    underlying_price = self.market_data.get(position.contract.underlying_symbol, {}).get('price', 0)
                    
                    # 计算距离执行价的距离
                    distance_to_strike = abs(underlying_price - position.contract.strike_price)
                    strike_ratio = distance_to_strike / position.contract.strike_price
                    
                    # Pin风险随着距离执行价越近而增加
                    if strike_ratio < 0.02:  # 2%以内
                        pin_risk += abs(position.quantity) * position.contract.strike_price * 0.1
                    elif strike_ratio < 0.05:  # 5%以内
                        pin_risk += abs(position.quantity) * position.contract.strike_price * 0.05
            
            return pin_risk
            
        except Exception as e:
            self.logger.error(f"Error calculating pin risk: {e}")
            return 0.0
    
    def _calculate_early_exercise_risk(self) -> float:
        """计算早期行权风险"""
        try:
            early_exercise_risk = 0.0
            
            for position in self.option_portfolio.positions:
                if position.quantity < 0:  # 只有空头有早期行权风险
                    # 获取标的价格
                    underlying_price = self.market_data.get(position.contract.underlying_symbol, {}).get('price', 0)
                    
                    # 计算内在价值
                    if position.contract.option_type == OptionType.CALL:
                        intrinsic_value = max(underlying_price - position.contract.strike_price, 0)
                    else:
                        intrinsic_value = max(position.contract.strike_price - underlying_price, 0)
                    
                    # 计算时间价值
                    time_value = position.contract.market_price - intrinsic_value
                    
                    # 时间价值低时早期行权风险增加
                    if time_value < 0.05 and intrinsic_value > 0:
                        early_exercise_risk += abs(position.quantity) * intrinsic_value * 100
            
            return early_exercise_risk
            
        except Exception as e:
            self.logger.error(f"Error calculating early exercise risk: {e}")
            return 0.0
    
    def _calculate_assignment_risk(self) -> float:
        """计算指派风险"""
        try:
            assignment_risk = 0.0
            
            for position in self.option_portfolio.positions:
                if position.quantity < 0:  # 只有空头有指派风险
                    # 获取标的价格
                    underlying_price = self.market_data.get(position.contract.underlying_symbol, {}).get('price', 0)
                    
                    # 计算价内程度
                    if position.contract.option_type == OptionType.CALL:
                        moneyness = underlying_price / position.contract.strike_price
                        in_the_money = moneyness > 1.0
                    else:
                        moneyness = position.contract.strike_price / underlying_price
                        in_the_money = moneyness > 1.0
                    
                    # 价内期权的指派风险
                    if in_the_money:
                        assignment_probability = min(moneyness - 1.0, 0.5)  # 最大50%概率
                        assignment_risk += abs(position.quantity) * assignment_probability * position.contract.strike_price * 100
            
            return assignment_risk
            
        except Exception as e:
            self.logger.error(f"Error calculating assignment risk: {e}")
            return 0.0
    
    def _calculate_max_profit_loss(self) -> Tuple[float, float]:
        """计算最大盈亏"""
        try:
            # 简化计算，实际应该考虑所有可能的标的价格
            max_loss = 0.0
            max_profit = 0.0
            
            for position in self.option_portfolio.positions:
                if position.quantity > 0:  # 多头
                    # 最大损失是权利金
                    max_loss += position.quantity * position.entry_price * 100
                    
                    # 最大收益取决于期权类型
                    if position.contract.option_type == OptionType.CALL:
                        max_profit += float('inf')  # 理论上无限
                    else:
                        max_profit += position.quantity * (position.contract.strike_price - position.entry_price) * 100
                else:  # 空头
                    # 最大收益是权利金
                    max_profit += abs(position.quantity) * position.entry_price * 100
                    
                    # 最大损失取决于期权类型
                    if position.contract.option_type == OptionType.CALL:
                        max_loss += float('inf')  # 理论上无限
                    else:
                        max_loss += abs(position.quantity) * (position.contract.strike_price + position.entry_price) * 100
            
            return max_loss, max_profit
            
        except Exception as e:
            self.logger.error(f"Error calculating max profit/loss: {e}")
            return 0.0, 0.0
    
    def _calculate_break_even_points(self) -> List[float]:
        """计算盈亏平衡点"""
        try:
            # 简化计算，实际应该根据具体策略计算
            break_even_points = []
            
            for position in self.option_portfolio.positions:
                if position.contract.option_type == OptionType.CALL:
                    if position.quantity > 0:  # 多头看涨
                        break_even = position.contract.strike_price + position.entry_price
                    else:  # 空头看涨
                        break_even = position.contract.strike_price + position.entry_price
                else:  # PUT
                    if position.quantity > 0:  # 多头看跌
                        break_even = position.contract.strike_price - position.entry_price
                    else:  # 空头看跌
                        break_even = position.contract.strike_price - position.entry_price
                
                break_even_points.append(break_even)
            
            return list(set(break_even_points))  # 去重
            
        except Exception as e:
            self.logger.error(f"Error calculating break even points: {e}")
            return []
    
    def _calculate_profit_probability(self) -> float:
        """计算获利概率"""
        try:
            # 使用Black-Scholes模型和蒙特卡洛模拟
            total_probability = 0.0
            total_positions = len(self.option_portfolio.positions)
            
            if total_positions == 0:
                return 0.0
            
            for position in self.option_portfolio.positions:
                underlying_price = self.market_data.get(position.contract.underlying_symbol, {}).get('price', 100)
                
                # 计算到期时获利的概率
                if position.quantity > 0:  # 多头
                    if position.contract.option_type == OptionType.CALL:
                        # 看涨期权多头获利概率
                        break_even = position.contract.strike_price + position.entry_price
                        prob = 1 - norm.cdf(break_even, underlying_price, underlying_price * 0.2)
                    else:
                        # 看跌期权多头获利概率
                        break_even = position.contract.strike_price - position.entry_price
                        prob = norm.cdf(break_even, underlying_price, underlying_price * 0.2)
                else:  # 空头
                    if position.contract.option_type == OptionType.CALL:
                        # 看涨期权空头获利概率
                        break_even = position.contract.strike_price + position.entry_price
                        prob = norm.cdf(break_even, underlying_price, underlying_price * 0.2)
                    else:
                        # 看跌期权空头获利概率
                        break_even = position.contract.strike_price - position.entry_price
                        prob = 1 - norm.cdf(break_even, underlying_price, underlying_price * 0.2)
                
                total_probability += prob
            
            return total_probability / total_positions
            
        except Exception as e:
            self.logger.error(f"Error calculating profit probability: {e}")
            return 0.0
    
    def _calculate_kelly_criterion(self) -> float:
        """计算Kelly准则"""
        try:
            # 简化Kelly准则计算
            win_rate = self._calculate_profit_probability()
            
            if win_rate <= 0 or win_rate >= 1:
                return 0.0
            
            # 假设平均收益率
            avg_win = 0.20  # 20%
            avg_loss = 0.15  # 15%
            
            # Kelly公式: f = (bp - q) / b
            # 其中 b = 赔率, p = 胜率, q = 败率
            b = avg_win / avg_loss
            p = win_rate
            q = 1 - p
            
            kelly = (b * p - q) / b
            
            return max(0, min(kelly, 0.25))  # 限制在0-25%之间
            
        except Exception as e:
            self.logger.error(f"Error calculating Kelly criterion: {e}")
            return 0.0
    
    def _calculate_sharpe_ratio(self) -> float:
        """计算夏普比率"""
        try:
            if len(self.risk_history) < 30:
                return 0.0
            
            # 计算近30天的收益率
            recent_metrics = self.risk_history[-30:]
            returns = []
            
            for i in range(1, len(recent_metrics)):
                prev_value = recent_metrics[i-1].portfolio_value
                curr_value = recent_metrics[i].portfolio_value
                
                if prev_value > 0:
                    daily_return = (curr_value - prev_value) / prev_value
                    returns.append(daily_return)
            
            if len(returns) < 2:
                return 0.0
            
            # 计算夏普比率
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            if std_return == 0:
                return 0.0
            
            # 年化夏普比率
            sharpe = (mean_return - self.risk_free_rate / 365) / std_return * np.sqrt(365)
            
            return sharpe
            
        except Exception as e:
            self.logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def _calculate_option_var(self) -> float:
        """计算期权VaR"""
        try:
            # 使用Delta-Normal方法
            confidence_level = 0.95
            
            # 计算组合Delta
            total_delta = abs(self.option_portfolio.total_delta)
            
            # 假设标的资产波动率
            volatility = 0.20  # 20%年化波动率
            
            # 计算1日VaR
            daily_volatility = volatility / np.sqrt(252)
            z_score = norm.ppf(confidence_level)
            
            var_1d = total_delta * daily_volatility * z_score
            
            return var_1d
            
        except Exception as e:
            self.logger.error(f"Error calculating option VaR: {e}")
            return 0.0
    
    def _calculate_option_cvar(self) -> float:
        """计算期权CVaR"""
        try:
            # 使用条件期望
            confidence_level = 0.95
            
            # 计算VaR
            var_1d = self._calculate_option_var()
            
            # CVaR通常比VaR高20-30%
            cvar_1d = var_1d * 1.3
            
            return cvar_1d
            
        except Exception as e:
            self.logger.error(f"Error calculating option CVaR: {e}")
            return 0.0
    
    def _calculate_expected_move(self) -> float:
        """计算期望移动"""
        try:
            # 使用ATM跨式期权计算期望移动
            expected_move = 0.0
            
            for position in self.option_portfolio.positions:
                if position.days_to_expiration <= 30:  # 近月期权
                    # 期望移动 = ATM跨式价格
                    straddle_price = position.contract.market_price * 2  # 简化计算
                    expected_move = max(expected_move, straddle_price)
            
            return expected_move
            
        except Exception as e:
            self.logger.error(f"Error calculating expected move: {e}")
            return 0.0
    
    def _calculate_realized_volatility(self) -> float:
        """计算实现波动率"""
        try:
            # 使用历史价格计算实现波动率
            if len(self.risk_history) < 20:
                return 0.20  # 默认20%
            
            recent_values = [m.portfolio_value for m in self.risk_history[-20:]]
            returns = []
            
            for i in range(1, len(recent_values)):
                if recent_values[i-1] > 0:
                    ret = np.log(recent_values[i] / recent_values[i-1])
                    returns.append(ret)
            
            if len(returns) < 2:
                return 0.20
            
            # 年化波动率
            realized_vol = np.std(returns) * np.sqrt(252)
            
            return realized_vol
            
        except Exception as e:
            self.logger.error(f"Error calculating realized volatility: {e}")
            return 0.20
    
    def _calculate_iv_rank(self) -> float:
        """计算IV等级"""
        try:
            # 计算当前IV在历史IV中的百分位
            current_iv = np.mean([pos.contract.implied_volatility for pos in self.option_portfolio.positions])
            
            # 模拟历史IV数据
            historical_iv = [0.15, 0.18, 0.22, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60]
            
            # 计算百分位
            iv_rank = len([iv for iv in historical_iv if iv <= current_iv]) / len(historical_iv)
            
            return iv_rank
            
        except Exception as e:
            self.logger.error(f"Error calculating IV rank: {e}")
            return 0.5
    
    def _calculate_skew_risk(self) -> float:
        """计算偏度风险"""
        try:
            # 计算不同执行价的IV差异
            skew_risk = 0.0
            
            # 按标的资产分组
            underlying_groups = {}
            for position in self.option_portfolio.positions:
                underlying = position.contract.underlying_symbol
                if underlying not in underlying_groups:
                    underlying_groups[underlying] = []
                underlying_groups[underlying].append(position)
            
            # 计算每个标的的偏度风险
            for underlying, positions in underlying_groups.items():
                if len(positions) > 1:
                    ivs = [pos.contract.implied_volatility for pos in positions]
                    strikes = [pos.contract.strike_price for pos in positions]
                    
                    # 计算IV的标准差作为偏度风险
                    iv_std = np.std(ivs)
                    skew_risk += iv_std * sum(abs(pos.quantity) for pos in positions)
            
            return skew_risk
            
        except Exception as e:
            self.logger.error(f"Error calculating skew risk: {e}")
            return 0.0
    
    async def check_risk_limits(self) -> List[Dict[str, Any]]:
        """检查风险限制"""
        try:
            violations = []
            
            # 检查Delta敞口
            if abs(self.option_portfolio.total_delta) > self.risk_limits['max_delta_exposure']:
                violations.append({
                    'type': 'delta_exposure',
                    'current': abs(self.option_portfolio.total_delta),
                    'limit': self.risk_limits['max_delta_exposure'],
                    'severity': 'high'
                })
            
            # 检查Gamma敞口
            if abs(self.option_portfolio.total_gamma) > self.risk_limits['max_gamma_exposure']:
                violations.append({
                    'type': 'gamma_exposure',
                    'current': abs(self.option_portfolio.total_gamma),
                    'limit': self.risk_limits['max_gamma_exposure'],
                    'severity': 'high'
                })
            
            # 检查Vega敞口
            if abs(self.option_portfolio.total_vega) > self.risk_limits['max_vega_exposure']:
                violations.append({
                    'type': 'vega_exposure',
                    'current': abs(self.option_portfolio.total_vega),
                    'limit': self.risk_limits['max_vega_exposure'],
                    'severity': 'medium'
                })
            
            # 检查Theta敞口
            if abs(self.option_portfolio.total_theta) > self.risk_limits['max_theta_exposure']:
                violations.append({
                    'type': 'theta_exposure',
                    'current': abs(self.option_portfolio.total_theta),
                    'limit': self.risk_limits['max_theta_exposure'],
                    'severity': 'medium'
                })
            
            # 检查组合价值
            if self.option_portfolio.total_value > self.risk_limits['max_portfolio_value']:
                violations.append({
                    'type': 'portfolio_value',
                    'current': self.option_portfolio.total_value,
                    'limit': self.risk_limits['max_portfolio_value'],
                    'severity': 'high'
                })
            
            # 检查单一持仓大小
            for position in self.option_portfolio.positions:
                if abs(position.quantity) > self.risk_limits['max_single_position_size']:
                    violations.append({
                        'type': 'single_position_size',
                        'symbol': position.contract.symbol,
                        'current': abs(position.quantity),
                        'limit': self.risk_limits['max_single_position_size'],
                        'severity': 'medium'
                    })
            
            # 检查到期日风险
            for position in self.option_portfolio.positions:
                if position.days_to_expiration < self.risk_limits['min_days_to_expiration']:
                    violations.append({
                        'type': 'expiration_risk',
                        'symbol': position.contract.symbol,
                        'current': position.days_to_expiration,
                        'limit': self.risk_limits['min_days_to_expiration'],
                        'severity': 'high'
                    })
            
            return violations
            
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {e}")
            return []
    
    async def generate_hedge_recommendations(self) -> List[Dict[str, Any]]:
        """生成对冲建议"""
        try:
            recommendations = []
            
            # Delta对冲
            if abs(self.option_portfolio.total_delta) > self.risk_limits['max_delta_exposure'] * 0.8:
                hedge_quantity = -self.option_portfolio.total_delta
                recommendations.append({
                    'type': 'delta_hedge',
                    'action': 'buy' if hedge_quantity > 0 else 'sell',
                    'instrument': 'underlying_stock',
                    'quantity': abs(hedge_quantity),
                    'reason': f'Delta exposure: {self.option_portfolio.total_delta:.2f}',
                    'priority': 'high'
                })
            
            # Gamma对冲
            if abs(self.option_portfolio.total_gamma) > self.risk_limits['max_gamma_exposure'] * 0.8:
                recommendations.append({
                    'type': 'gamma_hedge',
                    'action': 'buy' if self.option_portfolio.total_gamma < 0 else 'sell',
                    'instrument': 'atm_options',
                    'quantity': abs(self.option_portfolio.total_gamma) / 100,
                    'reason': f'Gamma exposure: {self.option_portfolio.total_gamma:.2f}',
                    'priority': 'medium'
                })
            
            # Vega对冲
            if abs(self.option_portfolio.total_vega) > self.risk_limits['max_vega_exposure'] * 0.8:
                recommendations.append({
                    'type': 'vega_hedge',
                    'action': 'buy' if self.option_portfolio.total_vega < 0 else 'sell',
                    'instrument': 'long_dated_options',
                    'quantity': abs(self.option_portfolio.total_vega) / 100,
                    'reason': f'Vega exposure: {self.option_portfolio.total_vega:.2f}',
                    'priority': 'medium'
                })
            
            # Theta对冲
            if self.option_portfolio.total_theta < -self.risk_limits['max_theta_exposure'] * 0.8:
                recommendations.append({
                    'type': 'theta_hedge',
                    'action': 'sell',
                    'instrument': 'short_dated_options',
                    'quantity': abs(self.option_portfolio.total_theta) / 10,
                    'reason': f'Theta exposure: {self.option_portfolio.total_theta:.2f}',
                    'priority': 'low'
                })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating hedge recommendations: {e}")
            return []
    
    async def simulate_scenarios(self, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """模拟情景分析"""
        try:
            scenario_results = {}
            
            for scenario in scenarios:
                scenario_name = scenario['name']
                price_change = scenario.get('price_change', 0)
                vol_change = scenario.get('vol_change', 0)
                time_decay = scenario.get('time_decay', 0)
                
                total_pnl = 0
                position_pnls = []
                
                for position in self.option_portfolio.positions:
                    # 获取当前标的价格
                    current_price = self.market_data.get(position.contract.underlying_symbol, {}).get('price', 100)
                    
                    # 计算情景下的新价格
                    new_price = current_price * (1 + price_change)
                    
                    # 计算新的期权价格
                    new_contract = position.contract
                    new_contract.implied_volatility *= (1 + vol_change)
                    
                    # 调整到期时间
                    new_contract.expiration_date -= timedelta(days=time_decay)
                    
                    # 计算新的期权价格
                    new_option_price = await self.calculate_option_price(new_contract, new_price)
                    
                    # 计算盈亏
                    position_pnl = position.quantity * (new_option_price - position.current_price) * 100
                    total_pnl += position_pnl
                    
                    position_pnls.append({
                        'symbol': position.contract.symbol,
                        'pnl': position_pnl,
                        'new_price': new_option_price
                    })
                
                scenario_results[scenario_name] = {
                    'total_pnl': total_pnl,
                    'position_pnls': position_pnls,
                    'parameters': scenario
                }
            
            return scenario_results
            
        except Exception as e:
            self.logger.error(f"Error simulating scenarios: {e}")
            return {}
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """获取组合摘要"""
        try:
            return {
                'total_positions': len(self.option_portfolio.positions),
                'total_value': self.option_portfolio.total_value,
                'total_delta': self.option_portfolio.total_delta,
                'total_gamma': self.option_portfolio.total_gamma,
                'total_theta': self.option_portfolio.total_theta,
                'total_vega': self.option_portfolio.total_vega,
                'total_rho': self.option_portfolio.total_rho,
                'unrealized_pnl': self.option_portfolio.unrealized_pnl,
                'margin_requirement': self.option_portfolio.margin_requirement,
                'buying_power': self.option_portfolio.buying_power,
                'risk_limits': self.risk_limits,
                'portfolio_positions': [
                    {
                        'symbol': pos.contract.symbol,
                        'quantity': pos.quantity,
                        'current_price': pos.current_price,
                        'unrealized_pnl': pos.unrealized_pnl,
                        'delta': pos.delta_exposure,
                        'gamma': pos.gamma_exposure,
                        'theta': pos.theta_exposure,
                        'vega': pos.vega_exposure,
                        'days_to_expiration': pos.days_to_expiration
                    } for pos in self.option_portfolio.positions
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting portfolio summary: {e}")
            return {}
    
    def add_position(self, position: OptionPosition):
        """添加持仓"""
        try:
            self.option_portfolio.positions.append(position)
            self.logger.info(f"Added option position: {position.contract.symbol}")
            
        except Exception as e:
            self.logger.error(f"Error adding position: {e}")
    
    def remove_position(self, symbol: str):
        """移除持仓"""
        try:
            self.option_portfolio.positions = [
                pos for pos in self.option_portfolio.positions 
                if pos.contract.symbol != symbol
            ]
            self.logger.info(f"Removed option position: {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error removing position: {e}")
    
    def update_market_data(self, symbol: str, market_data: Dict[str, Any]):
        """更新市场数据"""
        try:
            self.market_data[symbol] = market_data
            
        except Exception as e:
            self.logger.error(f"Error updating market data: {e}")
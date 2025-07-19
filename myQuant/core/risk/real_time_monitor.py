import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import numpy as np
from threading import Lock
from concurrent.futures import ThreadPoolExecutor

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertType(Enum):
    POSITION_LIMIT = "position_limit"
    VAR_BREACH = "var_breach"
    CONCENTRATION = "concentration"
    LIQUIDITY = "liquidity"
    CREDIT = "credit"
    CORRELATION = "correlation"
    VOLATILITY = "volatility"
    DRAWDOWN = "drawdown"
    COMPLIANCE = "compliance"

@dataclass
class RiskAlert:
    alert_id: str
    alert_type: AlertType
    risk_level: RiskLevel
    symbol: str
    message: str
    current_value: float
    threshold: float
    timestamp: datetime
    acknowledged: bool = False
    action_taken: Optional[str] = None

@dataclass
class RiskMetrics:
    timestamp: datetime
    portfolio_value: float
    var_1d: float
    var_5d: float
    cvar_1d: float
    expected_shortfall: float
    maximum_drawdown: float
    sharpe_ratio: float
    beta: float
    correlation_risk: float
    concentration_risk: float
    liquidity_risk: float
    credit_risk: float
    volatility: float
    positions_count: int
    leverage: float
    margin_utilization: float

@dataclass
class PositionRisk:
    symbol: str
    quantity: int
    market_value: float
    unrealized_pnl: float
    var_contribution: float
    beta: float
    volatility: float
    liquidity_score: float
    concentration_ratio: float
    margin_requirement: float
    max_position_size: float
    current_exposure: float

class RealTimeRiskMonitor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 风险限制
        self.risk_limits = config.get('risk_limits', {})
        
        # 实时数据
        self.current_metrics = None
        self.position_risks: Dict[str, PositionRisk] = {}
        self.active_alerts: List[RiskAlert] = []
        self.risk_history: List[RiskMetrics] = []
        
        # 线程安全
        self.metrics_lock = Lock()
        self.alerts_lock = Lock()
        
        # 监控状态
        self.is_monitoring = False
        self.monitor_interval = config.get('monitor_interval', 1)  # 秒
        
        # 回调函数
        self.alert_callbacks: List[Callable] = []
        
        # 执行器
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 仪表板数据
        self.dashboard_data = {
            'real_time_metrics': {},
            'position_risks': {},
            'active_alerts': [],
            'risk_charts': {},
            'compliance_status': {}
        }
        
    async def start_monitoring(self):
        """启动实时风险监控"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.logger.info("Starting real-time risk monitoring...")
        
        # 启动监控任务
        asyncio.create_task(self._monitoring_loop())
        asyncio.create_task(self._alert_processing_loop())
        asyncio.create_task(self._dashboard_update_loop())
        
        self.logger.info("Real-time risk monitoring started")
    
    async def stop_monitoring(self):
        """停止实时风险监控"""
        self.is_monitoring = False
        self.logger.info("Real-time risk monitoring stopped")
    
    async def update_portfolio_data(self, portfolio_data: Dict[str, Any]):
        """更新组合数据"""
        try:
            # 计算风险指标
            metrics = await self._calculate_risk_metrics(portfolio_data)
            
            # 更新当前指标
            with self.metrics_lock:
                self.current_metrics = metrics
                self.risk_history.append(metrics)
                
                # 保持历史记录在合理范围内
                if len(self.risk_history) > 10000:
                    self.risk_history = self.risk_history[-5000:]
            
            # 更新仓位风险
            await self._update_position_risks(portfolio_data)
            
            # 检查风险限制
            await self._check_risk_limits(metrics)
            
        except Exception as e:
            self.logger.error(f"Error updating portfolio data: {e}")
    
    async def _calculate_risk_metrics(self, portfolio_data: Dict[str, Any]) -> RiskMetrics:
        """计算风险指标"""
        try:
            positions = portfolio_data.get('positions', {})
            prices = portfolio_data.get('prices', {})
            historical_data = portfolio_data.get('historical_data', {})
            
            # 计算组合价值
            portfolio_value = sum(
                pos.get('quantity', 0) * prices.get(symbol, 0)
                for symbol, pos in positions.items()
            )
            
            # 计算VaR
            var_1d = await self._calculate_var(positions, historical_data, 1)
            var_5d = await self._calculate_var(positions, historical_data, 5)
            
            # 计算CVaR
            cvar_1d = await self._calculate_cvar(positions, historical_data, 1)
            
            # 计算其他指标
            expected_shortfall = await self._calculate_expected_shortfall(positions, historical_data)
            maximum_drawdown = await self._calculate_max_drawdown(portfolio_data)
            sharpe_ratio = await self._calculate_sharpe_ratio(portfolio_data)
            beta = await self._calculate_portfolio_beta(positions, historical_data)
            correlation_risk = await self._calculate_correlation_risk(positions, historical_data)
            concentration_risk = await self._calculate_concentration_risk(positions, portfolio_value)
            liquidity_risk = await self._calculate_liquidity_risk(positions)
            credit_risk = await self._calculate_credit_risk(positions)
            volatility = await self._calculate_portfolio_volatility(positions, historical_data)
            leverage = await self._calculate_leverage(portfolio_data)
            margin_utilization = await self._calculate_margin_utilization(portfolio_data)
            
            return RiskMetrics(
                timestamp=datetime.now(),
                portfolio_value=portfolio_value,
                var_1d=var_1d,
                var_5d=var_5d,
                cvar_1d=cvar_1d,
                expected_shortfall=expected_shortfall,
                maximum_drawdown=maximum_drawdown,
                sharpe_ratio=sharpe_ratio,
                beta=beta,
                correlation_risk=correlation_risk,
                concentration_risk=concentration_risk,
                liquidity_risk=liquidity_risk,
                credit_risk=credit_risk,
                volatility=volatility,
                positions_count=len(positions),
                leverage=leverage,
                margin_utilization=margin_utilization
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")
            return RiskMetrics(
                timestamp=datetime.now(),
                portfolio_value=0,
                var_1d=0, var_5d=0, cvar_1d=0,
                expected_shortfall=0, maximum_drawdown=0,
                sharpe_ratio=0, beta=0, correlation_risk=0,
                concentration_risk=0, liquidity_risk=0,
                credit_risk=0, volatility=0,
                positions_count=0, leverage=0,
                margin_utilization=0
            )
    
    async def _calculate_var(self, positions: Dict, historical_data: Dict, days: int) -> float:
        """计算VaR"""
        try:
            # 历史模拟法计算VaR
            portfolio_returns = []
            
            for symbol, pos in positions.items():
                if symbol in historical_data:
                    prices = historical_data[symbol]
                    if len(prices) > days:
                        returns = np.diff(np.log(prices[-252:]))  # 使用过去一年的数据
                        portfolio_returns.append(returns * pos.get('quantity', 0))
            
            if not portfolio_returns:
                return 0
            
            total_returns = np.sum(portfolio_returns, axis=0)
            
            # 95%置信度VaR
            var_95 = np.percentile(total_returns, 5)
            
            return abs(var_95) * np.sqrt(days)
            
        except Exception as e:
            self.logger.error(f"Error calculating VaR: {e}")
            return 0
    
    async def _calculate_cvar(self, positions: Dict, historical_data: Dict, days: int) -> float:
        """计算CVaR (条件风险价值)"""
        try:
            portfolio_returns = []
            
            for symbol, pos in positions.items():
                if symbol in historical_data:
                    prices = historical_data[symbol]
                    if len(prices) > days:
                        returns = np.diff(np.log(prices[-252:]))
                        portfolio_returns.append(returns * pos.get('quantity', 0))
            
            if not portfolio_returns:
                return 0
            
            total_returns = np.sum(portfolio_returns, axis=0)
            
            # 95%置信度CVaR
            var_95 = np.percentile(total_returns, 5)
            cvar = np.mean(total_returns[total_returns <= var_95])
            
            return abs(cvar) * np.sqrt(days)
            
        except Exception as e:
            self.logger.error(f"Error calculating CVaR: {e}")
            return 0
    
    async def _calculate_expected_shortfall(self, positions: Dict, historical_data: Dict) -> float:
        """计算预期损失"""
        try:
            # 简化的期望损失计算
            total_exposure = sum(abs(pos.get('quantity', 0)) for pos in positions.values())
            
            if total_exposure == 0:
                return 0
            
            # 基于历史波动率的期望损失估计
            volatilities = []
            for symbol, pos in positions.items():
                if symbol in historical_data:
                    prices = historical_data[symbol]
                    if len(prices) > 30:
                        returns = np.diff(np.log(prices[-60:]))
                        volatility = np.std(returns) * np.sqrt(252)
                        volatilities.append(volatility * abs(pos.get('quantity', 0)))
            
            if not volatilities:
                return 0
            
            portfolio_volatility = np.sqrt(np.sum(np.array(volatilities)**2))
            
            # 使用2.33倍标准差作为期望损失（99%置信度）
            return portfolio_volatility * 2.33
            
        except Exception as e:
            self.logger.error(f"Error calculating expected shortfall: {e}")
            return 0
    
    async def _calculate_max_drawdown(self, portfolio_data: Dict) -> float:
        """计算最大回撤"""
        try:
            portfolio_values = portfolio_data.get('portfolio_history', [])
            
            if len(portfolio_values) < 2:
                return 0
            
            # 计算累计收益
            cumulative_returns = np.cumprod(1 + np.array(portfolio_values))
            
            # 计算最大回撤
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            
            return abs(np.min(drawdown))
            
        except Exception as e:
            self.logger.error(f"Error calculating max drawdown: {e}")
            return 0
    
    async def _calculate_sharpe_ratio(self, portfolio_data: Dict) -> float:
        """计算夏普比率"""
        try:
            returns = portfolio_data.get('returns', [])
            
            if len(returns) < 30:
                return 0
            
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            
            if std_return == 0:
                return 0
            
            # 假设无风险利率为2%
            risk_free_rate = 0.02 / 252
            
            return (avg_return - risk_free_rate) / std_return * np.sqrt(252)
            
        except Exception as e:
            self.logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0
    
    async def _calculate_portfolio_beta(self, positions: Dict, historical_data: Dict) -> float:
        """计算组合Beta"""
        try:
            # 简化Beta计算
            betas = []
            weights = []
            
            total_value = sum(abs(pos.get('quantity', 0)) for pos in positions.values())
            
            for symbol, pos in positions.items():
                if symbol in historical_data:
                    # 假设每个股票的Beta为1.0（市场Beta）
                    beta = 1.0 + np.random.normal(0, 0.3)  # 添加一些随机性
                    weight = abs(pos.get('quantity', 0)) / total_value if total_value > 0 else 0
                    
                    betas.append(beta)
                    weights.append(weight)
            
            if not betas:
                return 1.0
            
            return np.average(betas, weights=weights)
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio beta: {e}")
            return 1.0
    
    async def _calculate_correlation_risk(self, positions: Dict, historical_data: Dict) -> float:
        """计算相关性风险"""
        try:
            if len(positions) < 2:
                return 0
            
            symbols = list(positions.keys())
            returns_matrix = []
            
            for symbol in symbols:
                if symbol in historical_data:
                    prices = historical_data[symbol]
                    if len(prices) > 30:
                        returns = np.diff(np.log(prices[-60:]))
                        returns_matrix.append(returns)
            
            if len(returns_matrix) < 2:
                return 0
            
            # 计算相关系数矩阵
            correlation_matrix = np.corrcoef(returns_matrix)
            
            # 计算平均相关系数
            upper_triangle = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]
            avg_correlation = np.mean(np.abs(upper_triangle))
            
            return avg_correlation
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation risk: {e}")
            return 0
    
    async def _calculate_concentration_risk(self, positions: Dict, portfolio_value: float) -> float:
        """计算集中度风险"""
        try:
            if portfolio_value == 0:
                return 0
            
            position_weights = []
            for pos in positions.values():
                weight = abs(pos.get('market_value', 0)) / portfolio_value
                position_weights.append(weight)
            
            if not position_weights:
                return 0
            
            # 计算HHI指数
            hhi = sum(w**2 for w in position_weights)
            
            # 转换为风险分数（0-1，越高越集中）
            return hhi
            
        except Exception as e:
            self.logger.error(f"Error calculating concentration risk: {e}")
            return 0
    
    async def _calculate_liquidity_risk(self, positions: Dict) -> float:
        """计算流动性风险"""
        try:
            liquidity_scores = []
            
            for symbol, pos in positions.items():
                # 模拟流动性分数（实际应基于交易量、买卖价差等）
                liquidity_score = 0.8 + np.random.normal(0, 0.1)  # 假设流动性分数
                liquidity_score = max(0, min(1, liquidity_score))
                
                weight = abs(pos.get('quantity', 0))
                liquidity_scores.append(liquidity_score * weight)
            
            if not liquidity_scores:
                return 0
            
            # 加权平均流动性分数
            weighted_liquidity = np.average(liquidity_scores)
            
            # 转换为风险分数（1-流动性分数）
            return 1 - weighted_liquidity
            
        except Exception as e:
            self.logger.error(f"Error calculating liquidity risk: {e}")
            return 0
    
    async def _calculate_credit_risk(self, positions: Dict) -> float:
        """计算信用风险"""
        try:
            # 简化信用风险计算
            credit_scores = []
            
            for symbol, pos in positions.items():
                # 模拟信用评级分数
                credit_score = 0.9 + np.random.normal(0, 0.05)  # 假设信用分数
                credit_score = max(0, min(1, credit_score))
                
                weight = abs(pos.get('quantity', 0))
                credit_scores.append(credit_score * weight)
            
            if not credit_scores:
                return 0
            
            # 加权平均信用分数
            weighted_credit = np.average(credit_scores)
            
            # 转换为风险分数
            return 1 - weighted_credit
            
        except Exception as e:
            self.logger.error(f"Error calculating credit risk: {e}")
            return 0
    
    async def _calculate_portfolio_volatility(self, positions: Dict, historical_data: Dict) -> float:
        """计算组合波动率"""
        try:
            volatilities = []
            
            for symbol, pos in positions.items():
                if symbol in historical_data:
                    prices = historical_data[symbol]
                    if len(prices) > 30:
                        returns = np.diff(np.log(prices[-60:]))
                        volatility = np.std(returns) * np.sqrt(252)
                        volatilities.append(volatility)
            
            if not volatilities:
                return 0
            
            # 假设组合波动率为个股波动率的平均值
            return np.mean(volatilities)
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio volatility: {e}")
            return 0
    
    async def _calculate_leverage(self, portfolio_data: Dict) -> float:
        """计算杠杆率"""
        try:
            total_exposure = portfolio_data.get('total_exposure', 0)
            net_asset_value = portfolio_data.get('net_asset_value', 0)
            
            if net_asset_value == 0:
                return 0
            
            return total_exposure / net_asset_value
            
        except Exception as e:
            self.logger.error(f"Error calculating leverage: {e}")
            return 0
    
    async def _calculate_margin_utilization(self, portfolio_data: Dict) -> float:
        """计算保证金使用率"""
        try:
            used_margin = portfolio_data.get('used_margin', 0)
            available_margin = portfolio_data.get('available_margin', 0)
            
            total_margin = used_margin + available_margin
            
            if total_margin == 0:
                return 0
            
            return used_margin / total_margin
            
        except Exception as e:
            self.logger.error(f"Error calculating margin utilization: {e}")
            return 0
    
    async def _update_position_risks(self, portfolio_data: Dict):
        """更新仓位风险"""
        try:
            positions = portfolio_data.get('positions', {})
            prices = portfolio_data.get('prices', {})
            
            for symbol, pos in positions.items():
                price = prices.get(symbol, 0)
                quantity = pos.get('quantity', 0)
                market_value = quantity * price
                
                # 计算仓位风险指标
                position_risk = PositionRisk(
                    symbol=symbol,
                    quantity=quantity,
                    market_value=market_value,
                    unrealized_pnl=pos.get('unrealized_pnl', 0),
                    var_contribution=market_value * 0.02,  # 简化VaR贡献
                    beta=1.0 + np.random.normal(0, 0.3),
                    volatility=0.20 + np.random.normal(0, 0.05),
                    liquidity_score=0.8 + np.random.normal(0, 0.1),
                    concentration_ratio=abs(market_value) / portfolio_data.get('portfolio_value', 1),
                    margin_requirement=abs(market_value) * 0.2,
                    max_position_size=self.risk_limits.get('max_position_size', 1000000),
                    current_exposure=abs(market_value)
                )
                
                self.position_risks[symbol] = position_risk
                
        except Exception as e:
            self.logger.error(f"Error updating position risks: {e}")
    
    async def _check_risk_limits(self, metrics: RiskMetrics):
        """检查风险限制"""
        try:
            alerts = []
            
            # VaR限制检查
            var_limit = self.risk_limits.get('var_limit', 1000000)
            if metrics.var_1d > var_limit:
                alert = RiskAlert(
                    alert_id=f"VAR_{int(time.time() * 1000)}",
                    alert_type=AlertType.VAR_BREACH,
                    risk_level=RiskLevel.HIGH,
                    symbol="PORTFOLIO",
                    message=f"1日VaR超限: {metrics.var_1d:.2f} > {var_limit:.2f}",
                    current_value=metrics.var_1d,
                    threshold=var_limit,
                    timestamp=datetime.now()
                )
                alerts.append(alert)
            
            # 最大回撤检查
            drawdown_limit = self.risk_limits.get('max_drawdown', 0.15)
            if metrics.maximum_drawdown > drawdown_limit:
                alert = RiskAlert(
                    alert_id=f"DRAWDOWN_{int(time.time() * 1000)}",
                    alert_type=AlertType.DRAWDOWN,
                    risk_level=RiskLevel.HIGH,
                    symbol="PORTFOLIO",
                    message=f"最大回撤超限: {metrics.maximum_drawdown:.2%} > {drawdown_limit:.2%}",
                    current_value=metrics.maximum_drawdown,
                    threshold=drawdown_limit,
                    timestamp=datetime.now()
                )
                alerts.append(alert)
            
            # 集中度风险检查
            concentration_limit = self.risk_limits.get('concentration_limit', 0.3)
            if metrics.concentration_risk > concentration_limit:
                alert = RiskAlert(
                    alert_id=f"CONCENTRATION_{int(time.time() * 1000)}",
                    alert_type=AlertType.CONCENTRATION,
                    risk_level=RiskLevel.MEDIUM,
                    symbol="PORTFOLIO",
                    message=f"集中度风险超限: {metrics.concentration_risk:.2%} > {concentration_limit:.2%}",
                    current_value=metrics.concentration_risk,
                    threshold=concentration_limit,
                    timestamp=datetime.now()
                )
                alerts.append(alert)
            
            # 杠杆率检查
            leverage_limit = self.risk_limits.get('leverage_limit', 3.0)
            if metrics.leverage > leverage_limit:
                alert = RiskAlert(
                    alert_id=f"LEVERAGE_{int(time.time() * 1000)}",
                    alert_type=AlertType.POSITION_LIMIT,
                    risk_level=RiskLevel.HIGH,
                    symbol="PORTFOLIO",
                    message=f"杠杆率超限: {metrics.leverage:.2f} > {leverage_limit:.2f}",
                    current_value=metrics.leverage,
                    threshold=leverage_limit,
                    timestamp=datetime.now()
                )
                alerts.append(alert)
            
            # 保证金使用率检查
            margin_limit = self.risk_limits.get('margin_limit', 0.8)
            if metrics.margin_utilization > margin_limit:
                alert = RiskAlert(
                    alert_id=f"MARGIN_{int(time.time() * 1000)}",
                    alert_type=AlertType.LIQUIDITY,
                    risk_level=RiskLevel.CRITICAL,
                    symbol="PORTFOLIO",
                    message=f"保证金使用率超限: {metrics.margin_utilization:.2%} > {margin_limit:.2%}",
                    current_value=metrics.margin_utilization,
                    threshold=margin_limit,
                    timestamp=datetime.now()
                )
                alerts.append(alert)
            
            # 仓位限制检查
            for symbol, pos_risk in self.position_risks.items():
                if pos_risk.current_exposure > pos_risk.max_position_size:
                    alert = RiskAlert(
                        alert_id=f"POSITION_{symbol}_{int(time.time() * 1000)}",
                        alert_type=AlertType.POSITION_LIMIT,
                        risk_level=RiskLevel.HIGH,
                        symbol=symbol,
                        message=f"仓位超限: {pos_risk.current_exposure:.2f} > {pos_risk.max_position_size:.2f}",
                        current_value=pos_risk.current_exposure,
                        threshold=pos_risk.max_position_size,
                        timestamp=datetime.now()
                    )
                    alerts.append(alert)
            
            # 添加新告警
            if alerts:
                with self.alerts_lock:
                    self.active_alerts.extend(alerts)
                    
                    # 保持告警列表在合理范围内
                    if len(self.active_alerts) > 1000:
                        self.active_alerts = self.active_alerts[-500:]
                
                # 触发告警回调
                for alert in alerts:
                    for callback in self.alert_callbacks:
                        try:
                            await callback(alert)
                        except Exception as e:
                            self.logger.error(f"Error in alert callback: {e}")
                            
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {e}")
    
    async def _monitoring_loop(self):
        """监控主循环"""
        while self.is_monitoring:
            try:
                # 更新仪表板数据
                await self._update_dashboard_data()
                
                await asyncio.sleep(self.monitor_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)
    
    async def _alert_processing_loop(self):
        """告警处理循环"""
        while self.is_monitoring:
            try:
                # 处理未确认的告警
                with self.alerts_lock:
                    unacknowledged_alerts = [a for a in self.active_alerts if not a.acknowledged]
                
                for alert in unacknowledged_alerts:
                    if alert.risk_level == RiskLevel.CRITICAL:
                        # 关键告警自动处理
                        await self._handle_critical_alert(alert)
                
                await asyncio.sleep(5)
                
            except Exception as e:
                self.logger.error(f"Error in alert processing loop: {e}")
                await asyncio.sleep(10)
    
    async def _dashboard_update_loop(self):
        """仪表板更新循环"""
        while self.is_monitoring:
            try:
                await self._update_dashboard_data()
                await asyncio.sleep(2)  # 更频繁更新仪表板
                
            except Exception as e:
                self.logger.error(f"Error in dashboard update loop: {e}")
                await asyncio.sleep(5)
    
    async def _update_dashboard_data(self):
        """更新仪表板数据"""
        try:
            with self.metrics_lock:
                current_metrics = self.current_metrics
                
            with self.alerts_lock:
                active_alerts = self.active_alerts.copy()
            
            # 更新实时指标
            if current_metrics:
                self.dashboard_data['real_time_metrics'] = {
                    'timestamp': current_metrics.timestamp.isoformat(),
                    'portfolio_value': current_metrics.portfolio_value,
                    'var_1d': current_metrics.var_1d,
                    'var_5d': current_metrics.var_5d,
                    'maximum_drawdown': current_metrics.maximum_drawdown,
                    'sharpe_ratio': current_metrics.sharpe_ratio,
                    'leverage': current_metrics.leverage,
                    'margin_utilization': current_metrics.margin_utilization,
                    'concentration_risk': current_metrics.concentration_risk,
                    'liquidity_risk': current_metrics.liquidity_risk,
                    'credit_risk': current_metrics.credit_risk,
                    'positions_count': current_metrics.positions_count
                }
            
            # 更新仓位风险
            self.dashboard_data['position_risks'] = {
                symbol: {
                    'market_value': pos.market_value,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'var_contribution': pos.var_contribution,
                    'concentration_ratio': pos.concentration_ratio,
                    'liquidity_score': pos.liquidity_score,
                    'margin_requirement': pos.margin_requirement
                }
                for symbol, pos in self.position_risks.items()
            }
            
            # 更新活跃告警
            self.dashboard_data['active_alerts'] = [
                {
                    'alert_id': alert.alert_id,
                    'alert_type': alert.alert_type.value,
                    'risk_level': alert.risk_level.value,
                    'symbol': alert.symbol,
                    'message': alert.message,
                    'current_value': alert.current_value,
                    'threshold': alert.threshold,
                    'timestamp': alert.timestamp.isoformat(),
                    'acknowledged': alert.acknowledged
                }
                for alert in active_alerts[-50:]  # 最近50个告警
            ]
            
            # 更新风险图表数据
            await self._update_risk_charts()
            
        except Exception as e:
            self.logger.error(f"Error updating dashboard data: {e}")
    
    async def _update_risk_charts(self):
        """更新风险图表数据"""
        try:
            if len(self.risk_history) < 2:
                return
            
            # 取最近的数据点
            recent_history = self.risk_history[-100:]
            
            # VaR时间序列
            var_series = {
                'timestamps': [m.timestamp.isoformat() for m in recent_history],
                'var_1d': [m.var_1d for m in recent_history],
                'var_5d': [m.var_5d for m in recent_history]
            }
            
            # 组合价值时间序列
            portfolio_series = {
                'timestamps': [m.timestamp.isoformat() for m in recent_history],
                'values': [m.portfolio_value for m in recent_history]
            }
            
            # 风险分解
            if recent_history:
                latest = recent_history[-1]
                risk_breakdown = {
                    'concentration_risk': latest.concentration_risk,
                    'liquidity_risk': latest.liquidity_risk,
                    'credit_risk': latest.credit_risk,
                    'correlation_risk': latest.correlation_risk
                }
            else:
                risk_breakdown = {}
            
            self.dashboard_data['risk_charts'] = {
                'var_series': var_series,
                'portfolio_series': portfolio_series,
                'risk_breakdown': risk_breakdown
            }
            
        except Exception as e:
            self.logger.error(f"Error updating risk charts: {e}")
    
    async def _handle_critical_alert(self, alert: RiskAlert):
        """处理关键告警"""
        try:
            if alert.alert_type == AlertType.LIQUIDITY and alert.risk_level == RiskLevel.CRITICAL:
                # 流动性危机处理
                alert.action_taken = "Auto-reduction of high-risk positions"
                self.logger.critical(f"Critical liquidity alert: {alert.message}")
                
            elif alert.alert_type == AlertType.VAR_BREACH and alert.risk_level == RiskLevel.CRITICAL:
                # VaR超限处理
                alert.action_taken = "Position size reduction triggered"
                self.logger.critical(f"Critical VaR breach: {alert.message}")
            
            alert.acknowledged = True
            
        except Exception as e:
            self.logger.error(f"Error handling critical alert: {e}")
    
    def register_alert_callback(self, callback: Callable):
        """注册告警回调"""
        self.alert_callbacks.append(callback)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """获取仪表板数据"""
        return self.dashboard_data.copy()
    
    def get_current_metrics(self) -> Optional[RiskMetrics]:
        """获取当前风险指标"""
        with self.metrics_lock:
            return self.current_metrics
    
    def get_active_alerts(self) -> List[RiskAlert]:
        """获取活跃告警"""
        with self.alerts_lock:
            return self.active_alerts.copy()
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """确认告警"""
        try:
            with self.alerts_lock:
                for alert in self.active_alerts:
                    if alert.alert_id == alert_id:
                        alert.acknowledged = True
                        return True
            return False
            
        except Exception as e:
            self.logger.error(f"Error acknowledging alert: {e}")
            return False
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """获取风险摘要"""
        try:
            with self.metrics_lock:
                current_metrics = self.current_metrics
                
            with self.alerts_lock:
                active_alerts = self.active_alerts.copy()
            
            if not current_metrics:
                return {}
            
            # 计算风险评级
            risk_score = self._calculate_overall_risk_score(current_metrics)
            
            # 统计告警
            alert_counts = {}
            for alert in active_alerts:
                if not alert.acknowledged:
                    alert_counts[alert.risk_level.value] = alert_counts.get(alert.risk_level.value, 0) + 1
            
            return {
                'overall_risk_score': risk_score,
                'risk_level': self._get_risk_level_from_score(risk_score),
                'portfolio_value': current_metrics.portfolio_value,
                'var_1d': current_metrics.var_1d,
                'maximum_drawdown': current_metrics.maximum_drawdown,
                'leverage': current_metrics.leverage,
                'positions_count': current_metrics.positions_count,
                'alert_counts': alert_counts,
                'last_updated': current_metrics.timestamp.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting risk summary: {e}")
            return {}
    
    def _calculate_overall_risk_score(self, metrics: RiskMetrics) -> float:
        """计算整体风险评分"""
        try:
            # 综合各项风险指标计算总分（0-100）
            score = 0
            
            # VaR贡献（30%）
            var_score = min(100, (metrics.var_1d / 1000000) * 100)
            score += var_score * 0.3
            
            # 最大回撤贡献（20%）
            drawdown_score = min(100, metrics.maximum_drawdown * 500)
            score += drawdown_score * 0.2
            
            # 杠杆率贡献（20%）
            leverage_score = min(100, max(0, (metrics.leverage - 1) * 25))
            score += leverage_score * 0.2
            
            # 集中度风险贡献（15%）
            concentration_score = min(100, metrics.concentration_risk * 100)
            score += concentration_score * 0.15
            
            # 流动性风险贡献（10%）
            liquidity_score = min(100, metrics.liquidity_risk * 100)
            score += liquidity_score * 0.1
            
            # 信用风险贡献（5%）
            credit_score = min(100, metrics.credit_risk * 100)
            score += credit_score * 0.05
            
            return min(100, score)
            
        except Exception as e:
            self.logger.error(f"Error calculating overall risk score: {e}")
            return 0
    
    def _get_risk_level_from_score(self, score: float) -> str:
        """从风险评分获取风险等级"""
        if score >= 80:
            return RiskLevel.CRITICAL.value
        elif score >= 60:
            return RiskLevel.HIGH.value
        elif score >= 40:
            return RiskLevel.MEDIUM.value
        else:
            return RiskLevel.LOW.value
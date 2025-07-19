import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json

from .real_time_monitor import RealTimeRiskMonitor
from .stress_testing import StressTestingFramework
from .monte_carlo import MonteCarloSimulationEngine
from .options_risk_manager import OptionRiskManager
from .credit_risk_assessment import CreditRiskAssessment
from .liquidity_risk_manager import LiquidityRiskManager
from .regulatory_compliance import RegulatoryComplianceManager, RegulatoryFramework

class RiskSystemStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    OFFLINE = "offline"

@dataclass
class IntegratedRiskReport:
    report_id: str
    timestamp: datetime
    overall_risk_score: float
    system_status: RiskSystemStatus
    market_risk_score: float
    credit_risk_score: float
    liquidity_risk_score: float
    operational_risk_score: float
    compliance_score: float
    key_alerts: List[Dict[str, Any]]
    recommendations: List[str]
    next_review_date: datetime

class IntegratedRiskManager:
    """
    集成风险管理系统
    
    整合所有风险管理组件，提供统一的风险管理接口
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 初始化所有风险管理组件
        self.real_time_monitor = RealTimeRiskMonitor(config.get('real_time_monitor', {}))
        self.stress_testing = StressTestingFramework(config.get('stress_testing', {}))
        self.monte_carlo = MonteCarloSimulationEngine(config.get('monte_carlo', {}))
        self.options_risk = OptionRiskManager(config.get('options_risk', {}))
        self.credit_risk = CreditRiskAssessment(config.get('credit_risk', {}))
        self.liquidity_risk = LiquidityRiskManager(config.get('liquidity_risk', {}))
        self.regulatory_compliance = RegulatoryComplianceManager(config.get('regulatory_compliance', {}))
        
        # 设置数据源连接
        self._setup_data_source_connections()
        
        # 系统状态
        self.system_status = RiskSystemStatus.OFFLINE
        self.is_running = False
        
        # 报告历史
        self.risk_reports: List[IntegratedRiskReport] = []
        
        # 监控任务
        self.monitoring_tasks: List[asyncio.Task] = []
        
        # 风险阈值
        self.risk_thresholds = {
            'overall_risk_critical': 0.8,
            'overall_risk_warning': 0.6,
            'market_risk_critical': 0.7,
            'credit_risk_critical': 0.7,
            'liquidity_risk_critical': 0.7,
            'compliance_critical': 0.8
        }
        
        self.logger.info("Integrated Risk Manager initialized")
    
    def _setup_data_source_connections(self):
        """设置数据源连接"""
        try:
            # 连接监管合规系统到其他风险系统
            self.regulatory_compliance.set_risk_data_source('market_risk', self.real_time_monitor)
            self.regulatory_compliance.set_risk_data_source('credit_risk', self.credit_risk)
            self.regulatory_compliance.set_risk_data_source('liquidity_risk', self.liquidity_risk)
            
            self.logger.info("Data source connections established")
            
        except Exception as e:
            self.logger.error(f"Error setting up data source connections: {e}")
    
    async def start(self):
        """启动集成风险管理系统"""
        try:
            self.logger.info("Starting Integrated Risk Management System...")
            
            # 启动实时监控
            await self.real_time_monitor.start_monitoring()
            
            # 启动监控任务
            self.monitoring_tasks = [
                asyncio.create_task(self._risk_monitoring_loop()),
                asyncio.create_task(self._compliance_monitoring_loop()),
                asyncio.create_task(self._system_health_monitoring_loop())
            ]
            
            self.is_running = True
            self.system_status = RiskSystemStatus.HEALTHY
            
            self.logger.info("Integrated Risk Management System started successfully")
            
        except Exception as e:
            self.logger.error(f"Error starting Integrated Risk Management System: {e}")
            self.system_status = RiskSystemStatus.CRITICAL
            raise
    
    async def stop(self):
        """停止集成风险管理系统"""
        try:
            self.logger.info("Stopping Integrated Risk Management System...")
            
            self.is_running = False
            
            # 停止监控任务
            for task in self.monitoring_tasks:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            # 停止实时监控
            await self.real_time_monitor.stop_monitoring()
            
            self.system_status = RiskSystemStatus.OFFLINE
            
            self.logger.info("Integrated Risk Management System stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping Integrated Risk Management System: {e}")
    
    async def _risk_monitoring_loop(self):
        """风险监控循环"""
        while self.is_running:
            try:
                # 生成综合风险报告
                report = await self.generate_integrated_risk_report()
                
                if report:
                    # 检查风险阈值
                    await self._check_risk_thresholds(report)
                    
                    # 保存报告
                    self.risk_reports.append(report)
                    
                    # 保持报告历史数量
                    if len(self.risk_reports) > 1000:
                        self.risk_reports = self.risk_reports[-1000:]
                
                # 等待下次监控
                await asyncio.sleep(300)  # 5分钟
                
            except Exception as e:
                self.logger.error(f"Error in risk monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _compliance_monitoring_loop(self):
        """合规监控循环"""
        while self.is_running:
            try:
                # 运行合规检查
                await self.regulatory_compliance.run_compliance_check_batch()
                
                # 等待下次检查
                await asyncio.sleep(3600)  # 1小时
                
            except Exception as e:
                self.logger.error(f"Error in compliance monitoring loop: {e}")
                await asyncio.sleep(300)
    
    async def _system_health_monitoring_loop(self):
        """系统健康监控循环"""
        while self.is_running:
            try:
                # 检查系统健康状态
                health_status = await self._check_system_health()
                
                # 更新系统状态
                self.system_status = health_status
                
                # 等待下次检查
                await asyncio.sleep(60)  # 1分钟
                
            except Exception as e:
                self.logger.error(f"Error in system health monitoring loop: {e}")
                await asyncio.sleep(30)
    
    async def _check_system_health(self) -> RiskSystemStatus:
        """检查系统健康状态"""
        try:
            health_checks = {
                'real_time_monitor': self.real_time_monitor.is_monitoring,
                'stress_testing': True,  # 压力测试系统无状态
                'monte_carlo': True,     # 蒙特卡洛系统无状态
                'options_risk': True,    # 期权风险系统无状态
                'credit_risk': True,     # 信用风险系统无状态
                'liquidity_risk': True,  # 流动性风险系统无状态
                'regulatory_compliance': True  # 监管合规系统无状态
            }
            
            # 检查所有组件是否正常
            all_healthy = all(health_checks.values())
            
            if all_healthy:
                return RiskSystemStatus.HEALTHY
            else:
                failed_components = [k for k, v in health_checks.items() if not v]
                self.logger.warning(f"Unhealthy components: {failed_components}")
                return RiskSystemStatus.WARNING
                
        except Exception as e:
            self.logger.error(f"Error checking system health: {e}")
            return RiskSystemStatus.CRITICAL
    
    async def generate_integrated_risk_report(self) -> Optional[IntegratedRiskReport]:
        """生成综合风险报告"""
        try:
            timestamp = datetime.now()
            
            # 收集各风险模块的指标
            market_risk_metrics = await self._get_market_risk_metrics()
            credit_risk_metrics = await self._get_credit_risk_metrics()
            liquidity_risk_metrics = await self._get_liquidity_risk_metrics()
            compliance_metrics = await self._get_compliance_metrics()
            
            # 计算风险评分
            market_risk_score = self._calculate_market_risk_score(market_risk_metrics)
            credit_risk_score = self._calculate_credit_risk_score(credit_risk_metrics)
            liquidity_risk_score = self._calculate_liquidity_risk_score(liquidity_risk_metrics)
            compliance_score = self._calculate_compliance_score(compliance_metrics)
            
            # 计算综合风险评分
            overall_risk_score = self._calculate_overall_risk_score(
                market_risk_score, credit_risk_score, liquidity_risk_score, compliance_score
            )
            
            # 确定系统状态
            system_status = self._determine_system_status(overall_risk_score)
            
            # 收集关键警报
            key_alerts = await self._collect_key_alerts()
            
            # 生成建议
            recommendations = self._generate_integrated_recommendations(
                market_risk_score, credit_risk_score, liquidity_risk_score, compliance_score, key_alerts
            )
            
            # 创建报告
            report = IntegratedRiskReport(
                report_id=f"RISK_REPORT_{int(timestamp.timestamp())}",
                timestamp=timestamp,
                overall_risk_score=overall_risk_score,
                system_status=system_status,
                market_risk_score=market_risk_score,
                credit_risk_score=credit_risk_score,
                liquidity_risk_score=liquidity_risk_score,
                operational_risk_score=0.0,  # 暂未实现
                compliance_score=compliance_score,
                key_alerts=key_alerts,
                recommendations=recommendations,
                next_review_date=timestamp + timedelta(hours=4)
            )
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating integrated risk report: {e}")
            return None
    
    async def _get_market_risk_metrics(self) -> Dict[str, Any]:
        """获取市场风险指标"""
        try:
            if self.real_time_monitor.current_metrics:
                return {
                    'var_1d': self.real_time_monitor.current_metrics.var_1d,
                    'var_5d': self.real_time_monitor.current_metrics.var_5d,
                    'max_drawdown': self.real_time_monitor.current_metrics.maximum_drawdown,
                    'volatility': self.real_time_monitor.current_metrics.volatility,
                    'sharpe_ratio': self.real_time_monitor.current_metrics.sharpe_ratio,
                    'leverage': self.real_time_monitor.current_metrics.leverage
                }
            return {}
            
        except Exception as e:
            self.logger.error(f"Error getting market risk metrics: {e}")
            return {}
    
    async def _get_credit_risk_metrics(self) -> Dict[str, Any]:
        """获取信用风险指标"""
        try:
            credit_metrics = await self.credit_risk.calculate_portfolio_metrics()
            if credit_metrics:
                return {
                    'total_exposure': credit_metrics.total_exposure,
                    'expected_loss': credit_metrics.total_expected_loss,
                    'var_99': credit_metrics.var_99,
                    'concentration_risk': credit_metrics.concentration_risk,
                    'portfolio_default_rate': credit_metrics.portfolio_default_rate
                }
            return {}
            
        except Exception as e:
            self.logger.error(f"Error getting credit risk metrics: {e}")
            return {}
    
    async def _get_liquidity_risk_metrics(self) -> Dict[str, Any]:
        """获取流动性风险指标"""
        try:
            liquidity_metrics = await self.liquidity_risk.calculate_liquidity_risk_metrics()
            if liquidity_metrics:
                return {
                    'lcr': liquidity_metrics.liquidity_coverage_ratio,
                    'nsfr': liquidity_metrics.net_stable_funding_ratio,
                    'survival_period': liquidity_metrics.survival_period_days,
                    'concentration_risk': liquidity_metrics.concentration_risk,
                    'funding_cost': liquidity_metrics.funding_cost
                }
            return {}
            
        except Exception as e:
            self.logger.error(f"Error getting liquidity risk metrics: {e}")
            return {}
    
    async def _get_compliance_metrics(self) -> Dict[str, Any]:
        """获取合规指标"""
        try:
            dashboard_data = self.regulatory_compliance.get_compliance_dashboard_data()
            if dashboard_data:
                return {
                    'total_rules': dashboard_data.get('summary', {}).get('total_rules', 0),
                    'active_alerts': dashboard_data.get('summary', {}).get('active_alerts', 0),
                    'framework_stats': dashboard_data.get('framework_stats', {}),
                    'recent_checks': dashboard_data.get('recent_checks', [])
                }
            return {}
            
        except Exception as e:
            self.logger.error(f"Error getting compliance metrics: {e}")
            return {}
    
    def _calculate_market_risk_score(self, metrics: Dict[str, Any]) -> float:
        """计算市场风险评分"""
        try:
            if not metrics:
                return 0.0
            
            score = 0.0
            
            # VaR评分
            var_1d = metrics.get('var_1d', 0)
            if var_1d > 0.05:
                score += 0.3
            elif var_1d > 0.03:
                score += 0.2
            elif var_1d > 0.01:
                score += 0.1
            
            # 最大回撤评分
            max_drawdown = abs(metrics.get('max_drawdown', 0))
            if max_drawdown > 0.20:
                score += 0.3
            elif max_drawdown > 0.15:
                score += 0.2
            elif max_drawdown > 0.10:
                score += 0.1
            
            # 杠杆评分
            leverage = metrics.get('leverage', 0)
            if leverage > 3.0:
                score += 0.2
            elif leverage > 2.0:
                score += 0.1
            
            # 夏普比率评分（负向）
            sharpe_ratio = metrics.get('sharpe_ratio', 0)
            if sharpe_ratio < 0:
                score += 0.2
            elif sharpe_ratio < 0.5:
                score += 0.1
            
            return min(score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating market risk score: {e}")
            return 0.0
    
    def _calculate_credit_risk_score(self, metrics: Dict[str, Any]) -> float:
        """计算信用风险评分"""
        try:
            if not metrics:
                return 0.0
            
            score = 0.0
            
            # 违约率评分
            default_rate = metrics.get('portfolio_default_rate', 0)
            if default_rate > 0.05:
                score += 0.4
            elif default_rate > 0.03:
                score += 0.3
            elif default_rate > 0.01:
                score += 0.2
            
            # 集中度风险评分
            concentration = metrics.get('concentration_risk', 0)
            if concentration > 0.3:
                score += 0.3
            elif concentration > 0.2:
                score += 0.2
            elif concentration > 0.1:
                score += 0.1
            
            # VaR评分
            var_99 = metrics.get('var_99', 0)
            total_exposure = metrics.get('total_exposure', 1)
            var_ratio = var_99 / total_exposure if total_exposure > 0 else 0
            if var_ratio > 0.10:
                score += 0.3
            elif var_ratio > 0.05:
                score += 0.2
            elif var_ratio > 0.02:
                score += 0.1
            
            return min(score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating credit risk score: {e}")
            return 0.0
    
    def _calculate_liquidity_risk_score(self, metrics: Dict[str, Any]) -> float:
        """计算流动性风险评分"""
        try:
            if not metrics:
                return 0.0
            
            score = 0.0
            
            # LCR评分
            lcr = metrics.get('lcr', 1.0)
            if lcr < 1.0:
                score += 0.4
            elif lcr < 1.05:
                score += 0.3
            elif lcr < 1.1:
                score += 0.1
            
            # NSFR评分
            nsfr = metrics.get('nsfr', 1.0)
            if nsfr < 1.0:
                score += 0.3
            elif nsfr < 1.05:
                score += 0.2
            elif nsfr < 1.1:
                score += 0.1
            
            # 生存期评分
            survival_period = metrics.get('survival_period', 30)
            if survival_period < 7:
                score += 0.3
            elif survival_period < 15:
                score += 0.2
            elif survival_period < 30:
                score += 0.1
            
            return min(score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating liquidity risk score: {e}")
            return 0.0
    
    def _calculate_compliance_score(self, metrics: Dict[str, Any]) -> float:
        """计算合规评分"""
        try:
            if not metrics:
                return 0.0
            
            active_alerts = metrics.get('active_alerts', 0)
            total_rules = metrics.get('total_rules', 1)
            
            # 合规违规率
            violation_rate = active_alerts / total_rules if total_rules > 0 else 0
            
            if violation_rate > 0.2:
                return 0.8
            elif violation_rate > 0.1:
                return 0.6
            elif violation_rate > 0.05:
                return 0.4
            elif violation_rate > 0.02:
                return 0.2
            else:
                return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating compliance score: {e}")
            return 0.0
    
    def _calculate_overall_risk_score(self, market: float, credit: float, liquidity: float, compliance: float) -> float:
        """计算综合风险评分"""
        try:
            # 权重配置
            weights = {
                'market': 0.3,
                'credit': 0.25,
                'liquidity': 0.25,
                'compliance': 0.2
            }
            
            overall_score = (
                market * weights['market'] +
                credit * weights['credit'] +
                liquidity * weights['liquidity'] +
                compliance * weights['compliance']
            )
            
            return min(overall_score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating overall risk score: {e}")
            return 0.0
    
    def _determine_system_status(self, overall_risk_score: float) -> RiskSystemStatus:
        """确定系统状态"""
        if overall_risk_score >= self.risk_thresholds['overall_risk_critical']:
            return RiskSystemStatus.CRITICAL
        elif overall_risk_score >= self.risk_thresholds['overall_risk_warning']:
            return RiskSystemStatus.WARNING
        else:
            return RiskSystemStatus.HEALTHY
    
    async def _collect_key_alerts(self) -> List[Dict[str, Any]]:
        """收集关键警报"""
        try:
            alerts = []
            
            # 市场风险警报
            if self.real_time_monitor.active_alerts:
                for alert in self.real_time_monitor.active_alerts[-5:]:  # 最近5个
                    alerts.append({
                        'type': 'market_risk',
                        'source': 'real_time_monitor',
                        'alert_type': alert.alert_type.value,
                        'risk_level': alert.risk_level.value,
                        'message': alert.message,
                        'timestamp': alert.timestamp.isoformat()
                    })
            
            # 合规警报
            compliance_data = self.regulatory_compliance.get_compliance_dashboard_data()
            recent_checks = compliance_data.get('recent_checks', [])
            for check in recent_checks[:3]:  # 最近3个
                if check.get('status') == 'breach':
                    alerts.append({
                        'type': 'compliance',
                        'source': 'regulatory_compliance',
                        'rule_name': check.get('rule_name'),
                        'framework': check.get('framework'),
                        'severity': check.get('severity'),
                        'timestamp': check.get('timestamp')
                    })
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Error collecting key alerts: {e}")
            return []
    
    def _generate_integrated_recommendations(self, market_score: float, credit_score: float, 
                                           liquidity_score: float, compliance_score: float, 
                                           alerts: List[Dict[str, Any]]) -> List[str]:
        """生成综合建议"""
        try:
            recommendations = []
            
            # 市场风险建议
            if market_score > 0.6:
                recommendations.append("Reduce market risk exposure through hedging or position sizing")
            if market_score > 0.4:
                recommendations.append("Enhance market risk monitoring and VaR calculations")
            
            # 信用风险建议
            if credit_score > 0.6:
                recommendations.append("Review credit exposure limits and counterparty risks")
            if credit_score > 0.4:
                recommendations.append("Diversify credit portfolio to reduce concentration risk")
            
            # 流动性风险建议
            if liquidity_score > 0.6:
                recommendations.append("Increase liquid asset buffer and improve funding stability")
            if liquidity_score > 0.4:
                recommendations.append("Review liquidity contingency plans and stress testing")
            
            # 合规建议
            if compliance_score > 0.6:
                recommendations.append("Address regulatory compliance breaches immediately")
            if compliance_score > 0.4:
                recommendations.append("Enhance compliance monitoring and reporting procedures")
            
            # 基于警报的建议
            alert_types = [alert['type'] for alert in alerts]
            if 'market_risk' in alert_types:
                recommendations.append("Investigate market risk alerts and take corrective action")
            if 'compliance' in alert_types:
                recommendations.append("Review compliance violations and implement remediation")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating integrated recommendations: {e}")
            return []
    
    async def _check_risk_thresholds(self, report: IntegratedRiskReport):
        """检查风险阈值"""
        try:
            # 检查综合风险阈值
            if report.overall_risk_score >= self.risk_thresholds['overall_risk_critical']:
                await self._send_critical_alert("Overall risk score exceeds critical threshold", report)
            
            # 检查各类风险阈值
            if report.market_risk_score >= self.risk_thresholds['market_risk_critical']:
                await self._send_critical_alert("Market risk score exceeds critical threshold", report)
            
            if report.credit_risk_score >= self.risk_thresholds['credit_risk_critical']:
                await self._send_critical_alert("Credit risk score exceeds critical threshold", report)
            
            if report.liquidity_risk_score >= self.risk_thresholds['liquidity_risk_critical']:
                await self._send_critical_alert("Liquidity risk score exceeds critical threshold", report)
            
            if report.compliance_score >= self.risk_thresholds['compliance_critical']:
                await self._send_critical_alert("Compliance score exceeds critical threshold", report)
            
        except Exception as e:
            self.logger.error(f"Error checking risk thresholds: {e}")
    
    async def _send_critical_alert(self, message: str, report: IntegratedRiskReport):
        """发送关键警报"""
        try:
            alert_data = {
                'type': 'critical_risk_alert',
                'message': message,
                'report_id': report.report_id,
                'timestamp': report.timestamp.isoformat(),
                'overall_risk_score': report.overall_risk_score,
                'system_status': report.system_status.value,
                'recommendations': report.recommendations
            }
            
            self.logger.critical(f"CRITICAL RISK ALERT: {json.dumps(alert_data, indent=2)}")
            
        except Exception as e:
            self.logger.error(f"Error sending critical alert: {e}")
    
    async def run_comprehensive_stress_test(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """运行综合压力测试"""
        try:
            self.logger.info("Starting comprehensive stress test...")
            
            # 运行各类压力测试
            stress_results = {}
            
            # 市场压力测试
            market_stress = await self.stress_testing.run_comprehensive_stress_test(portfolio_data)
            stress_results['market_stress'] = market_stress
            
            # 信用压力测试
            credit_stress = await self.credit_risk.stress_test_credit_portfolio([
                {'name': 'severe_recession', 'pd_shock': 0.5, 'lgd_shock': 0.3, 'correlation_shock': 0.2}
            ])
            stress_results['credit_stress'] = credit_stress
            
            # 流动性压力测试
            liquidity_stress = await self.liquidity_risk.stress_test_liquidity('severe_stress')
            stress_results['liquidity_stress'] = liquidity_stress
            
            # 综合分析
            stress_results['summary'] = self._analyze_stress_test_results(stress_results)
            
            self.logger.info("Comprehensive stress test completed")
            
            return stress_results
            
        except Exception as e:
            self.logger.error(f"Error running comprehensive stress test: {e}")
            return {}
    
    def _analyze_stress_test_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """分析压力测试结果"""
        try:
            summary = {
                'overall_resilience': 'good',
                'key_vulnerabilities': [],
                'recommendations': [],
                'risk_concentration': {}
            }
            
            # 分析市场压力测试
            market_results = results.get('market_stress', {})
            if market_results:
                worst_scenario = None
                worst_loss = 0
                
                for scenario, result in market_results.items():
                    if hasattr(result, 'relative_loss'):
                        loss = abs(result.relative_loss)
                        if loss > worst_loss:
                            worst_loss = loss
                            worst_scenario = scenario
                
                if worst_loss > 0.3:
                    summary['overall_resilience'] = 'poor'
                    summary['key_vulnerabilities'].append(f"High market risk exposure: {worst_scenario}")
                elif worst_loss > 0.2:
                    summary['overall_resilience'] = 'moderate'
                    summary['key_vulnerabilities'].append(f"Moderate market risk exposure: {worst_scenario}")
            
            # 分析信用压力测试
            credit_results = results.get('credit_stress', {})
            if credit_results:
                for scenario, result in credit_results.items():
                    stress_ratio = result.get('stress_ratio', 0)
                    if stress_ratio > 0.5:
                        summary['key_vulnerabilities'].append(f"High credit risk sensitivity: {scenario}")
            
            # 分析流动性压力测试
            liquidity_results = results.get('liquidity_stress', {})
            if liquidity_results:
                stressed_lcr = liquidity_results.get('stressed_lcr', 1.0)
                if stressed_lcr < 1.0:
                    summary['key_vulnerabilities'].append("Liquidity shortfall under stress")
            
            # 生成建议
            if summary['overall_resilience'] == 'poor':
                summary['recommendations'].extend([
                    "Significantly reduce risk exposure",
                    "Implement comprehensive hedging strategy",
                    "Review and strengthen risk limits"
                ])
            elif summary['overall_resilience'] == 'moderate':
                summary['recommendations'].extend([
                    "Consider additional risk mitigation measures",
                    "Enhance monitoring of key risk factors",
                    "Review risk tolerance and appetite"
                ])
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error analyzing stress test results: {e}")
            return {}
    
    def get_system_dashboard(self) -> Dict[str, Any]:
        """获取系统仪表板数据"""
        try:
            latest_report = self.risk_reports[-1] if self.risk_reports else None
            
            dashboard = {
                'system_status': self.system_status.value,
                'is_running': self.is_running,
                'last_update': datetime.now().isoformat(),
                'components_status': {
                    'real_time_monitor': self.real_time_monitor.is_monitoring,
                    'stress_testing': True,
                    'monte_carlo': True,
                    'options_risk': True,
                    'credit_risk': True,
                    'liquidity_risk': True,
                    'regulatory_compliance': True
                }
            }
            
            if latest_report:
                dashboard.update({
                    'latest_report': {
                        'report_id': latest_report.report_id,
                        'timestamp': latest_report.timestamp.isoformat(),
                        'overall_risk_score': latest_report.overall_risk_score,
                        'market_risk_score': latest_report.market_risk_score,
                        'credit_risk_score': latest_report.credit_risk_score,
                        'liquidity_risk_score': latest_report.liquidity_risk_score,
                        'compliance_score': latest_report.compliance_score,
                        'key_alerts_count': len(latest_report.key_alerts),
                        'recommendations_count': len(latest_report.recommendations)
                    }
                })
            
            return dashboard
            
        except Exception as e:
            self.logger.error(f"Error getting system dashboard: {e}")
            return {}
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """获取风险摘要"""
        try:
            return {
                'integrated_risk_manager': {
                    'system_status': self.system_status.value,
                    'is_running': self.is_running,
                    'reports_count': len(self.risk_reports),
                    'risk_thresholds': self.risk_thresholds
                },
                'components': {
                    'real_time_monitor': self.real_time_monitor.get_dashboard_data(),
                    'stress_testing': self.stress_testing.get_test_summary(),
                    'monte_carlo': self.monte_carlo.get_performance_statistics(),
                    'options_risk': self.options_risk.get_portfolio_summary(),
                    'credit_risk': self.credit_risk.get_credit_summary(),
                    'liquidity_risk': self.liquidity_risk.get_liquidity_summary(),
                    'regulatory_compliance': self.regulatory_compliance.get_regulatory_summary()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting risk summary: {e}")
            return {}
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

class RegulatoryFramework(Enum):
    BASEL_III = "basel_iii"
    DODD_FRANK = "dodd_frank"
    MIFID_II = "mifid_ii"
    SOLVENCY_II = "solvency_ii"
    CFTC = "cftc"
    SEC = "sec"
    FINRA = "finra"
    CSRC = "csrc"  # 中国证监会
    PBOC = "pboc"  # 中国人民银行
    CBIRC = "cbirc"  # 中国银保监会

class ComplianceStatus(Enum):
    COMPLIANT = "compliant"
    WARNING = "warning"
    BREACH = "breach"
    UNKNOWN = "unknown"

class RiskType(Enum):
    MARKET_RISK = "market_risk"
    CREDIT_RISK = "credit_risk"
    OPERATIONAL_RISK = "operational_risk"
    LIQUIDITY_RISK = "liquidity_risk"
    CONCENTRATION_RISK = "concentration_risk"
    COUNTERPARTY_RISK = "counterparty_risk"

@dataclass
class RegulatoryRule:
    rule_id: str
    framework: RegulatoryFramework
    rule_name: str
    description: str
    risk_type: RiskType
    metric_name: str
    threshold_value: float
    threshold_operator: str  # >, <, >=, <=, ==
    warning_threshold: Optional[float] = None
    frequency: str = "daily"  # daily, weekly, monthly, quarterly
    mandatory: bool = True
    effective_date: datetime = field(default_factory=datetime.now)
    expiry_date: Optional[datetime] = None
    jurisdiction: str = "global"

@dataclass
class ComplianceCheck:
    check_id: str
    rule: RegulatoryRule
    timestamp: datetime
    current_value: float
    threshold_value: float
    status: ComplianceStatus
    deviation: float
    severity: str  # low, medium, high, critical
    message: str
    recommendation: str
    next_check_date: datetime
    historical_values: List[float] = field(default_factory=list)

@dataclass
class ComplianceReport:
    report_id: str
    timestamp: datetime
    framework: RegulatoryFramework
    total_rules: int
    compliant_rules: int
    warning_rules: int
    breach_rules: int
    overall_status: ComplianceStatus
    risk_score: float
    compliance_ratio: float
    critical_breaches: List[ComplianceCheck]
    warnings: List[ComplianceCheck]
    recommendations: List[str]
    next_review_date: datetime

class RegulatoryComplianceManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 监管规则
        self.regulatory_rules: Dict[str, RegulatoryRule] = {}
        
        # 合规检查历史
        self.compliance_checks: Dict[str, List[ComplianceCheck]] = {}
        
        # 合规报告
        self.compliance_reports: Dict[str, ComplianceReport] = {}
        
        # 监管框架配置
        self.frameworks_config = self._initialize_frameworks()
        
        # 风险数据源
        self.risk_data_sources = {
            'market_risk': None,
            'credit_risk': None,
            'operational_risk': None,
            'liquidity_risk': None
        }
        
        # 执行器
        self.executor = ThreadPoolExecutor(max_workers=config.get('max_workers', 4))
        
        # 报警系统
        self.alert_system = config.get('alert_system', {})
        
        # 初始化默认规则
        self._initialize_default_rules()
    
    def _initialize_frameworks(self) -> Dict[str, Dict[str, Any]]:
        """初始化监管框架"""
        return {
            'basel_iii': {
                'name': 'Basel III',
                'description': 'International banking regulations',
                'jurisdiction': 'global',
                'key_metrics': ['capital_adequacy_ratio', 'leverage_ratio', 'lcr', 'nsfr'],
                'reporting_frequency': 'quarterly'
            },
            'dodd_frank': {
                'name': 'Dodd-Frank Act',
                'description': 'US financial reform regulations',
                'jurisdiction': 'us',
                'key_metrics': ['volcker_rule', 'stress_testing', 'risk_retention'],
                'reporting_frequency': 'quarterly'
            },
            'mifid_ii': {
                'name': 'MiFID II',
                'description': 'European financial markets regulation',
                'jurisdiction': 'eu',
                'key_metrics': ['best_execution', 'position_limits', 'transparency'],
                'reporting_frequency': 'daily'
            },
            'csrc': {
                'name': 'China Securities Regulatory Commission',
                'description': 'Chinese securities market regulation',
                'jurisdiction': 'china',
                'key_metrics': ['position_limits', 'margin_requirements', 'risk_control'],
                'reporting_frequency': 'daily'
            }
        }
    
    def _initialize_default_rules(self):
        """初始化默认监管规则"""
        try:
            # Basel III 规则
            self.add_regulatory_rule(RegulatoryRule(
                rule_id="basel_iii_capital_adequacy",
                framework=RegulatoryFramework.BASEL_III,
                rule_name="Capital Adequacy Ratio",
                description="Minimum capital adequacy ratio requirement",
                risk_type=RiskType.CREDIT_RISK,
                metric_name="capital_adequacy_ratio",
                threshold_value=0.08,
                threshold_operator=">=",
                warning_threshold=0.085,
                frequency="daily",
                mandatory=True,
                jurisdiction="global"
            ))
            
            self.add_regulatory_rule(RegulatoryRule(
                rule_id="basel_iii_leverage_ratio",
                framework=RegulatoryFramework.BASEL_III,
                rule_name="Leverage Ratio",
                description="Minimum leverage ratio requirement",
                risk_type=RiskType.CREDIT_RISK,
                metric_name="leverage_ratio",
                threshold_value=0.03,
                threshold_operator=">=",
                warning_threshold=0.035,
                frequency="daily",
                mandatory=True,
                jurisdiction="global"
            ))
            
            self.add_regulatory_rule(RegulatoryRule(
                rule_id="basel_iii_lcr",
                framework=RegulatoryFramework.BASEL_III,
                rule_name="Liquidity Coverage Ratio",
                description="Minimum liquidity coverage ratio",
                risk_type=RiskType.LIQUIDITY_RISK,
                metric_name="lcr",
                threshold_value=1.0,
                threshold_operator=">=",
                warning_threshold=1.05,
                frequency="daily",
                mandatory=True,
                jurisdiction="global"
            ))
            
            self.add_regulatory_rule(RegulatoryRule(
                rule_id="basel_iii_nsfr",
                framework=RegulatoryFramework.BASEL_III,
                rule_name="Net Stable Funding Ratio",
                description="Minimum net stable funding ratio",
                risk_type=RiskType.LIQUIDITY_RISK,
                metric_name="nsfr",
                threshold_value=1.0,
                threshold_operator=">=",
                warning_threshold=1.05,
                frequency="monthly",
                mandatory=True,
                jurisdiction="global"
            ))
            
            # 市场风险规则
            self.add_regulatory_rule(RegulatoryRule(
                rule_id="market_risk_var_limit",
                framework=RegulatoryFramework.BASEL_III,
                rule_name="Market Risk VaR Limit",
                description="Maximum Value at Risk limit",
                risk_type=RiskType.MARKET_RISK,
                metric_name="var_99",
                threshold_value=0.05,
                threshold_operator="<=",
                warning_threshold=0.04,
                frequency="daily",
                mandatory=True,
                jurisdiction="global"
            ))
            
            # 集中度风险规则
            self.add_regulatory_rule(RegulatoryRule(
                rule_id="concentration_single_counterparty",
                framework=RegulatoryFramework.BASEL_III,
                rule_name="Single Counterparty Concentration",
                description="Maximum exposure to single counterparty",
                risk_type=RiskType.CONCENTRATION_RISK,
                metric_name="single_counterparty_exposure",
                threshold_value=0.25,
                threshold_operator="<=",
                warning_threshold=0.20,
                frequency="daily",
                mandatory=True,
                jurisdiction="global"
            ))
            
            # 中国监管规则
            self.add_regulatory_rule(RegulatoryRule(
                rule_id="csrc_position_limit",
                framework=RegulatoryFramework.CSRC,
                rule_name="Position Concentration Limit",
                description="Maximum position concentration in single security",
                risk_type=RiskType.CONCENTRATION_RISK,
                metric_name="position_concentration",
                threshold_value=0.10,
                threshold_operator="<=",
                warning_threshold=0.08,
                frequency="daily",
                mandatory=True,
                jurisdiction="china"
            ))
            
            self.add_regulatory_rule(RegulatoryRule(
                rule_id="csrc_margin_requirement",
                framework=RegulatoryFramework.CSRC,
                rule_name="Margin Requirement",
                description="Minimum margin requirement for leveraged positions",
                risk_type=RiskType.MARKET_RISK,
                metric_name="margin_ratio",
                threshold_value=0.20,
                threshold_operator=">=",
                warning_threshold=0.25,
                frequency="daily",
                mandatory=True,
                jurisdiction="china"
            ))
            
            self.logger.info(f"Initialized {len(self.regulatory_rules)} default regulatory rules")
            
        except Exception as e:
            self.logger.error(f"Error initializing default rules: {e}")
    
    def add_regulatory_rule(self, rule: RegulatoryRule):
        """添加监管规则"""
        try:
            self.regulatory_rules[rule.rule_id] = rule
            self.logger.info(f"Added regulatory rule: {rule.rule_name}")
            
        except Exception as e:
            self.logger.error(f"Error adding regulatory rule: {e}")
    
    def remove_regulatory_rule(self, rule_id: str):
        """移除监管规则"""
        try:
            if rule_id in self.regulatory_rules:
                del self.regulatory_rules[rule_id]
                self.logger.info(f"Removed regulatory rule: {rule_id}")
            
        except Exception as e:
            self.logger.error(f"Error removing regulatory rule: {e}")
    
    def set_risk_data_source(self, risk_type: str, data_source: Any):
        """设置风险数据源"""
        try:
            self.risk_data_sources[risk_type] = data_source
            self.logger.info(f"Set risk data source for {risk_type}")
            
        except Exception as e:
            self.logger.error(f"Error setting risk data source: {e}")
    
    async def get_metric_value(self, metric_name: str) -> float:
        """获取指标值"""
        try:
            # 根据指标名称从不同数据源获取数据
            if metric_name in ['capital_adequacy_ratio', 'leverage_ratio']:
                # 从信用风险数据源获取
                if self.risk_data_sources['credit_risk']:
                    return await self._get_credit_risk_metric(metric_name)
                
            elif metric_name in ['lcr', 'nsfr']:
                # 从流动性风险数据源获取
                if self.risk_data_sources['liquidity_risk']:
                    return await self._get_liquidity_risk_metric(metric_name)
                
            elif metric_name in ['var_99', 'margin_ratio']:
                # 从市场风险数据源获取
                if self.risk_data_sources['market_risk']:
                    return await self._get_market_risk_metric(metric_name)
                
            elif metric_name in ['single_counterparty_exposure', 'position_concentration']:
                # 从集中度风险数据源获取
                return await self._get_concentration_metric(metric_name)
            
            # 默认模拟值
            return self._get_simulated_metric_value(metric_name)
            
        except Exception as e:
            self.logger.error(f"Error getting metric value for {metric_name}: {e}")
            return 0.0
    
    async def _get_credit_risk_metric(self, metric_name: str) -> float:
        """获取信用风险指标"""
        try:
            credit_source = self.risk_data_sources['credit_risk']
            
            if metric_name == 'capital_adequacy_ratio':
                # 资本充足率 = 资本 / 风险加权资产
                regulatory_capital = await credit_source.get_regulatory_capital_requirement()
                return regulatory_capital.get('capital_ratio', 0.08)
                
            elif metric_name == 'leverage_ratio':
                # 杠杆率 = 一级资本 / 总敞口
                regulatory_capital = await credit_source.get_regulatory_capital_requirement()
                return regulatory_capital.get('capital_ratio', 0.03) * 0.8  # 简化计算
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error getting credit risk metric: {e}")
            return 0.0
    
    async def _get_liquidity_risk_metric(self, metric_name: str) -> float:
        """获取流动性风险指标"""
        try:
            liquidity_source = self.risk_data_sources['liquidity_risk']
            
            if metric_name == 'lcr':
                return await liquidity_source.calculate_liquidity_coverage_ratio()
                
            elif metric_name == 'nsfr':
                return await liquidity_source.calculate_net_stable_funding_ratio()
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error getting liquidity risk metric: {e}")
            return 0.0
    
    async def _get_market_risk_metric(self, metric_name: str) -> float:
        """获取市场风险指标"""
        try:
            market_source = self.risk_data_sources['market_risk']
            
            if metric_name == 'var_99':
                # 从实时风险监控获取VaR
                if hasattr(market_source, 'current_metrics') and market_source.current_metrics:
                    return market_source.current_metrics.var_5d * 0.01  # 转换为百分比
                return 0.03
                
            elif metric_name == 'margin_ratio':
                # 保证金比率
                if hasattr(market_source, 'current_metrics') and market_source.current_metrics:
                    return market_source.current_metrics.margin_utilization
                return 0.25
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error getting market risk metric: {e}")
            return 0.0
    
    async def _get_concentration_metric(self, metric_name: str) -> float:
        """获取集中度风险指标"""
        try:
            if metric_name == 'single_counterparty_exposure':
                # 单一对手方敞口
                return 0.15  # 模拟值
                
            elif metric_name == 'position_concentration':
                # 持仓集中度
                return 0.08  # 模拟值
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error getting concentration metric: {e}")
            return 0.0
    
    def _get_simulated_metric_value(self, metric_name: str) -> float:
        """获取模拟指标值"""
        # 提供默认模拟值
        simulation_values = {
            'capital_adequacy_ratio': 0.12,
            'leverage_ratio': 0.05,
            'lcr': 1.15,
            'nsfr': 1.10,
            'var_99': 0.035,
            'margin_ratio': 0.30,
            'single_counterparty_exposure': 0.15,
            'position_concentration': 0.08
        }
        
        return simulation_values.get(metric_name, 0.0)
    
    async def check_compliance(self, rule_id: str) -> ComplianceCheck:
        """检查单个规则合规性"""
        try:
            rule = self.regulatory_rules.get(rule_id)
            if not rule:
                raise ValueError(f"Rule {rule_id} not found")
            
            # 获取当前指标值
            current_value = await self.get_metric_value(rule.metric_name)
            
            # 检查合规性
            status = self._evaluate_compliance(current_value, rule)
            
            # 计算偏差
            deviation = self._calculate_deviation(current_value, rule)
            
            # 确定严重程度
            severity = self._determine_severity(status, deviation, rule)
            
            # 生成消息和建议
            message = self._generate_compliance_message(status, current_value, rule)
            recommendation = self._generate_recommendation(status, rule)
            
            # 计算下次检查时间
            next_check_date = self._calculate_next_check_date(rule)
            
            # 创建合规检查对象
            check = ComplianceCheck(
                check_id=f"{rule_id}_{int(datetime.now().timestamp())}",
                rule=rule,
                timestamp=datetime.now(),
                current_value=current_value,
                threshold_value=rule.threshold_value,
                status=status,
                deviation=deviation,
                severity=severity,
                message=message,
                recommendation=recommendation,
                next_check_date=next_check_date
            )
            
            # 保存检查历史
            if rule_id not in self.compliance_checks:
                self.compliance_checks[rule_id] = []
            self.compliance_checks[rule_id].append(check)
            
            # 保持历史记录数量
            if len(self.compliance_checks[rule_id]) > 100:
                self.compliance_checks[rule_id] = self.compliance_checks[rule_id][-100:]
            
            # 如果有违规，发送警报
            if status == ComplianceStatus.BREACH:
                await self._send_compliance_alert(check)
            
            return check
            
        except Exception as e:
            self.logger.error(f"Error checking compliance for rule {rule_id}: {e}")
            return None
    
    def _evaluate_compliance(self, current_value: float, rule: RegulatoryRule) -> ComplianceStatus:
        """评估合规状态"""
        try:
            threshold = rule.threshold_value
            operator = rule.threshold_operator
            
            # 评估主要阈值
            if operator == ">=":
                compliant = current_value >= threshold
            elif operator == "<=":
                compliant = current_value <= threshold
            elif operator == ">":
                compliant = current_value > threshold
            elif operator == "<":
                compliant = current_value < threshold
            elif operator == "==":
                compliant = abs(current_value - threshold) < 0.001
            else:
                compliant = False
            
            if not compliant:
                return ComplianceStatus.BREACH
            
            # 检查警告阈值
            if rule.warning_threshold:
                if operator == ">=":
                    warning = current_value < rule.warning_threshold
                elif operator == "<=":
                    warning = current_value > rule.warning_threshold
                elif operator == ">":
                    warning = current_value <= rule.warning_threshold
                elif operator == "<":
                    warning = current_value >= rule.warning_threshold
                else:
                    warning = False
                
                if warning:
                    return ComplianceStatus.WARNING
            
            return ComplianceStatus.COMPLIANT
            
        except Exception as e:
            self.logger.error(f"Error evaluating compliance: {e}")
            return ComplianceStatus.UNKNOWN
    
    def _calculate_deviation(self, current_value: float, rule: RegulatoryRule) -> float:
        """计算偏差"""
        try:
            threshold = rule.threshold_value
            
            if rule.threshold_operator in [">=", ">"]:
                # 数值应该大于阈值
                deviation = (current_value - threshold) / threshold
            else:
                # 数值应该小于阈值
                deviation = (threshold - current_value) / threshold
            
            return deviation
            
        except Exception as e:
            self.logger.error(f"Error calculating deviation: {e}")
            return 0.0
    
    def _determine_severity(self, status: ComplianceStatus, deviation: float, rule: RegulatoryRule) -> str:
        """确定严重程度"""
        try:
            if status == ComplianceStatus.COMPLIANT:
                return "low"
            elif status == ComplianceStatus.WARNING:
                return "medium"
            elif status == ComplianceStatus.BREACH:
                if rule.mandatory:
                    if abs(deviation) > 0.2:
                        return "critical"
                    else:
                        return "high"
                else:
                    return "medium"
            else:
                return "low"
                
        except Exception as e:
            self.logger.error(f"Error determining severity: {e}")
            return "low"
    
    def _generate_compliance_message(self, status: ComplianceStatus, current_value: float, rule: RegulatoryRule) -> str:
        """生成合规消息"""
        try:
            if status == ComplianceStatus.COMPLIANT:
                return f"{rule.rule_name} is compliant. Current value: {current_value:.4f}, Threshold: {rule.threshold_value:.4f}"
            elif status == ComplianceStatus.WARNING:
                return f"{rule.rule_name} is approaching breach. Current value: {current_value:.4f}, Warning threshold: {rule.warning_threshold:.4f}"
            elif status == ComplianceStatus.BREACH:
                return f"{rule.rule_name} is in breach. Current value: {current_value:.4f}, Threshold: {rule.threshold_value:.4f}"
            else:
                return f"{rule.rule_name} status unknown"
                
        except Exception as e:
            self.logger.error(f"Error generating compliance message: {e}")
            return "Error generating message"
    
    def _generate_recommendation(self, status: ComplianceStatus, rule: RegulatoryRule) -> str:
        """生成建议"""
        try:
            if status == ComplianceStatus.COMPLIANT:
                return "Continue monitoring"
            elif status == ComplianceStatus.WARNING:
                return f"Take preventive action to avoid breach of {rule.rule_name}"
            elif status == ComplianceStatus.BREACH:
                return f"Immediate action required to address {rule.rule_name} breach"
            else:
                return "Review rule configuration"
                
        except Exception as e:
            self.logger.error(f"Error generating recommendation: {e}")
            return "No recommendation available"
    
    def _calculate_next_check_date(self, rule: RegulatoryRule) -> datetime:
        """计算下次检查时间"""
        try:
            now = datetime.now()
            
            if rule.frequency == "daily":
                return now + timedelta(days=1)
            elif rule.frequency == "weekly":
                return now + timedelta(weeks=1)
            elif rule.frequency == "monthly":
                return now + timedelta(days=30)
            elif rule.frequency == "quarterly":
                return now + timedelta(days=90)
            else:
                return now + timedelta(days=1)
                
        except Exception as e:
            self.logger.error(f"Error calculating next check date: {e}")
            return datetime.now() + timedelta(days=1)
    
    async def _send_compliance_alert(self, check: ComplianceCheck):
        """发送合规警报"""
        try:
            alert_config = self.alert_system.get('compliance_breach', {})
            
            if alert_config.get('enabled', True):
                # 构造警报消息
                alert_message = {
                    'type': 'compliance_breach',
                    'rule_id': check.rule.rule_id,
                    'rule_name': check.rule.rule_name,
                    'framework': check.rule.framework.value,
                    'current_value': check.current_value,
                    'threshold_value': check.threshold_value,
                    'deviation': check.deviation,
                    'severity': check.severity,
                    'message': check.message,
                    'recommendation': check.recommendation,
                    'timestamp': check.timestamp.isoformat()
                }
                
                # 发送警报（这里可以集成邮件、SMS、webhook等）
                self.logger.warning(f"COMPLIANCE BREACH ALERT: {json.dumps(alert_message, indent=2)}")
                
                # 记录警报
                if 'alerts' not in self.config:
                    self.config['alerts'] = []
                self.config['alerts'].append(alert_message)
            
        except Exception as e:
            self.logger.error(f"Error sending compliance alert: {e}")
    
    async def run_compliance_check_batch(self, framework: Optional[RegulatoryFramework] = None) -> List[ComplianceCheck]:
        """批量运行合规检查"""
        try:
            checks = []
            
            # 筛选要检查的规则
            rules_to_check = []
            for rule in self.regulatory_rules.values():
                if framework is None or rule.framework == framework:
                    rules_to_check.append(rule)
            
            # 并行执行检查
            tasks = []
            for rule in rules_to_check:
                task = asyncio.create_task(self.check_compliance(rule.rule_id))
                tasks.append(task)
            
            # 等待所有检查完成
            check_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理结果
            for result in check_results:
                if isinstance(result, ComplianceCheck):
                    checks.append(result)
                elif isinstance(result, Exception):
                    self.logger.error(f"Error in compliance check: {result}")
            
            self.logger.info(f"Completed {len(checks)} compliance checks")
            
            return checks
            
        except Exception as e:
            self.logger.error(f"Error running compliance check batch: {e}")
            return []
    
    async def generate_compliance_report(self, framework: RegulatoryFramework) -> ComplianceReport:
        """生成合规报告"""
        try:
            # 运行合规检查
            checks = await self.run_compliance_check_batch(framework)
            
            # 统计结果
            total_rules = len(checks)
            compliant_rules = len([c for c in checks if c.status == ComplianceStatus.COMPLIANT])
            warning_rules = len([c for c in checks if c.status == ComplianceStatus.WARNING])
            breach_rules = len([c for c in checks if c.status == ComplianceStatus.BREACH])
            
            # 确定总体状态
            if breach_rules > 0:
                overall_status = ComplianceStatus.BREACH
            elif warning_rules > 0:
                overall_status = ComplianceStatus.WARNING
            else:
                overall_status = ComplianceStatus.COMPLIANT
            
            # 计算风险评分
            risk_score = self._calculate_risk_score(checks)
            
            # 计算合规比率
            compliance_ratio = compliant_rules / total_rules if total_rules > 0 else 0.0
            
            # 收集重要违规和警告
            critical_breaches = [c for c in checks if c.status == ComplianceStatus.BREACH and c.severity == "critical"]
            warnings = [c for c in checks if c.status == ComplianceStatus.WARNING]
            
            # 生成建议
            recommendations = self._generate_report_recommendations(checks)
            
            # 计算下次审查时间
            next_review_date = datetime.now() + timedelta(days=30)
            
            # 创建报告
            report = ComplianceReport(
                report_id=f"COMPLIANCE_{framework.value}_{int(datetime.now().timestamp())}",
                timestamp=datetime.now(),
                framework=framework,
                total_rules=total_rules,
                compliant_rules=compliant_rules,
                warning_rules=warning_rules,
                breach_rules=breach_rules,
                overall_status=overall_status,
                risk_score=risk_score,
                compliance_ratio=compliance_ratio,
                critical_breaches=critical_breaches,
                warnings=warnings,
                recommendations=recommendations,
                next_review_date=next_review_date
            )
            
            # 保存报告
            self.compliance_reports[report.report_id] = report
            
            self.logger.info(f"Generated compliance report for {framework.value}: {compliance_ratio:.2%} compliant")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating compliance report: {e}")
            return None
    
    def _calculate_risk_score(self, checks: List[ComplianceCheck]) -> float:
        """计算风险评分"""
        try:
            if not checks:
                return 0.0
            
            total_score = 0.0
            
            for check in checks:
                # 基础分数
                if check.status == ComplianceStatus.COMPLIANT:
                    score = 0.0
                elif check.status == ComplianceStatus.WARNING:
                    score = 0.3
                elif check.status == ComplianceStatus.BREACH:
                    score = 1.0
                else:
                    score = 0.0
                
                # 严重程度调整
                if check.severity == "critical":
                    score *= 2.0
                elif check.severity == "high":
                    score *= 1.5
                elif check.severity == "medium":
                    score *= 1.2
                
                # 强制性调整
                if check.rule.mandatory:
                    score *= 1.5
                
                total_score += score
            
            # 标准化到0-1范围
            max_possible_score = len(checks) * 3.0  # 最大可能分数
            normalized_score = min(total_score / max_possible_score, 1.0) if max_possible_score > 0 else 0.0
            
            return normalized_score
            
        except Exception as e:
            self.logger.error(f"Error calculating risk score: {e}")
            return 0.0
    
    def _generate_report_recommendations(self, checks: List[ComplianceCheck]) -> List[str]:
        """生成报告建议"""
        try:
            recommendations = []
            
            # 分析违规模式
            breach_types = {}
            for check in checks:
                if check.status == ComplianceStatus.BREACH:
                    risk_type = check.rule.risk_type.value
                    if risk_type not in breach_types:
                        breach_types[risk_type] = []
                    breach_types[risk_type].append(check)
            
            # 生成针对性建议
            for risk_type, breaches in breach_types.items():
                if risk_type == "market_risk":
                    recommendations.append("Implement additional market risk controls and hedging strategies")
                elif risk_type == "credit_risk":
                    recommendations.append("Review credit exposure limits and enhance due diligence processes")
                elif risk_type == "liquidity_risk":
                    recommendations.append("Increase liquid asset buffer and diversify funding sources")
                elif risk_type == "concentration_risk":
                    recommendations.append("Reduce concentration in single counterparties and asset classes")
                elif risk_type == "operational_risk":
                    recommendations.append("Strengthen operational controls and business continuity planning")
            
            # 通用建议
            if len(checks) > 0:
                breach_ratio = len([c for c in checks if c.status == ComplianceStatus.BREACH]) / len(checks)
                if breach_ratio > 0.2:
                    recommendations.append("Conduct comprehensive risk management review")
                if breach_ratio > 0.1:
                    recommendations.append("Enhance compliance monitoring and reporting procedures")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating report recommendations: {e}")
            return []
    
    def get_compliance_dashboard_data(self) -> Dict[str, Any]:
        """获取合规仪表板数据"""
        try:
            # 最近的检查统计
            recent_checks = []
            for rule_id, checks in self.compliance_checks.items():
                if checks:
                    recent_checks.extend(checks[-5:])  # 最近5次检查
            
            recent_checks.sort(key=lambda x: x.timestamp, reverse=True)
            
            # 按框架分组统计
            framework_stats = {}
            for rule in self.regulatory_rules.values():
                framework = rule.framework.value
                if framework not in framework_stats:
                    framework_stats[framework] = {
                        'total_rules': 0,
                        'compliant': 0,
                        'warnings': 0,
                        'breaches': 0
                    }
                framework_stats[framework]['total_rules'] += 1
                
                # 获取最近的检查结果
                recent_check = self.compliance_checks.get(rule.rule_id, [])
                if recent_check:
                    latest_check = recent_check[-1]
                    if latest_check.status == ComplianceStatus.COMPLIANT:
                        framework_stats[framework]['compliant'] += 1
                    elif latest_check.status == ComplianceStatus.WARNING:
                        framework_stats[framework]['warnings'] += 1
                    elif latest_check.status == ComplianceStatus.BREACH:
                        framework_stats[framework]['breaches'] += 1
            
            # 风险类型分布
            risk_type_stats = {}
            for rule in self.regulatory_rules.values():
                risk_type = rule.risk_type.value
                if risk_type not in risk_type_stats:
                    risk_type_stats[risk_type] = 0
                risk_type_stats[risk_type] += 1
            
            # 趋势数据
            trend_data = self._calculate_compliance_trends()
            
            return {
                'summary': {
                    'total_rules': len(self.regulatory_rules),
                    'total_frameworks': len(set(rule.framework.value for rule in self.regulatory_rules.values())),
                    'recent_checks': len(recent_checks),
                    'active_alerts': len([c for c in recent_checks if c.status == ComplianceStatus.BREACH])
                },
                'framework_stats': framework_stats,
                'risk_type_distribution': risk_type_stats,
                'recent_checks': [
                    {
                        'rule_name': check.rule.rule_name,
                        'framework': check.rule.framework.value,
                        'status': check.status.value,
                        'current_value': check.current_value,
                        'threshold_value': check.threshold_value,
                        'timestamp': check.timestamp.isoformat(),
                        'severity': check.severity
                    } for check in recent_checks[:10]
                ],
                'trends': trend_data
            }
            
        except Exception as e:
            self.logger.error(f"Error getting compliance dashboard data: {e}")
            return {}
    
    def _calculate_compliance_trends(self) -> Dict[str, Any]:
        """计算合规趋势"""
        try:
            trends = {
                'daily_compliance_rate': [],
                'breach_trend': [],
                'warning_trend': []
            }
            
            # 计算过去30天的合规趋势
            for i in range(30):
                date = datetime.now() - timedelta(days=i)
                
                daily_checks = []
                for rule_id, checks in self.compliance_checks.items():
                    for check in checks:
                        if check.timestamp.date() == date.date():
                            daily_checks.append(check)
                
                if daily_checks:
                    compliant_count = len([c for c in daily_checks if c.status == ComplianceStatus.COMPLIANT])
                    warning_count = len([c for c in daily_checks if c.status == ComplianceStatus.WARNING])
                    breach_count = len([c for c in daily_checks if c.status == ComplianceStatus.BREACH])
                    
                    compliance_rate = compliant_count / len(daily_checks) if daily_checks else 0.0
                    
                    trends['daily_compliance_rate'].append({
                        'date': date.strftime('%Y-%m-%d'),
                        'rate': compliance_rate
                    })
                    trends['breach_trend'].append({
                        'date': date.strftime('%Y-%m-%d'),
                        'count': breach_count
                    })
                    trends['warning_trend'].append({
                        'date': date.strftime('%Y-%m-%d'),
                        'count': warning_count
                    })
            
            return trends
            
        except Exception as e:
            self.logger.error(f"Error calculating compliance trends: {e}")
            return {}
    
    def get_regulatory_summary(self) -> Dict[str, Any]:
        """获取监管摘要"""
        try:
            return {
                'total_rules': len(self.regulatory_rules),
                'frameworks': list(self.frameworks_config.keys()),
                'risk_types': list(set(rule.risk_type.value for rule in self.regulatory_rules.values())),
                'mandatory_rules': len([rule for rule in self.regulatory_rules.values() if rule.mandatory]),
                'daily_checks': len([rule for rule in self.regulatory_rules.values() if rule.frequency == 'daily']),
                'framework_summary': self.frameworks_config,
                'rules_by_framework': {
                    framework: [
                        {
                            'rule_id': rule.rule_id,
                            'rule_name': rule.rule_name,
                            'risk_type': rule.risk_type.value,
                            'metric_name': rule.metric_name,
                            'threshold_value': rule.threshold_value,
                            'mandatory': rule.mandatory,
                            'frequency': rule.frequency
                        } for rule in self.regulatory_rules.values()
                        if rule.framework.value == framework
                    ] for framework in self.frameworks_config.keys()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting regulatory summary: {e}")
            return {}
    
    def export_compliance_report(self, report_id: str, format: str = 'json') -> str:
        """导出合规报告"""
        try:
            report = self.compliance_reports.get(report_id)
            if not report:
                return ""
            
            if format == 'json':
                return json.dumps({
                    'report_id': report.report_id,
                    'timestamp': report.timestamp.isoformat(),
                    'framework': report.framework.value,
                    'summary': {
                        'total_rules': report.total_rules,
                        'compliant_rules': report.compliant_rules,
                        'warning_rules': report.warning_rules,
                        'breach_rules': report.breach_rules,
                        'overall_status': report.overall_status.value,
                        'risk_score': report.risk_score,
                        'compliance_ratio': report.compliance_ratio
                    },
                    'critical_breaches': [
                        {
                            'rule_name': check.rule.rule_name,
                            'current_value': check.current_value,
                            'threshold_value': check.threshold_value,
                            'deviation': check.deviation,
                            'message': check.message,
                            'recommendation': check.recommendation
                        } for check in report.critical_breaches
                    ],
                    'warnings': [
                        {
                            'rule_name': check.rule.rule_name,
                            'current_value': check.current_value,
                            'threshold_value': check.threshold_value,
                            'message': check.message
                        } for check in report.warnings
                    ],
                    'recommendations': report.recommendations,
                    'next_review_date': report.next_review_date.isoformat()
                }, indent=2)
            
            return ""
            
        except Exception as e:
            self.logger.error(f"Error exporting compliance report: {e}")
            return ""
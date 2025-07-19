import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from pathlib import Path
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

class ControlType(Enum):
    PREVENTIVE = "preventive"
    DETECTIVE = "detective"
    CORRECTIVE = "corrective"
    DIRECTIVE = "directive"
    COMPENSATING = "compensating"

class ControlCategory(Enum):
    TRADING = "trading"
    RISK_MANAGEMENT = "risk_management"
    COMPLIANCE = "compliance"
    OPERATIONAL = "operational"
    FINANCIAL = "financial"
    TECHNOLOGY = "technology"
    HUMAN_RESOURCES = "human_resources"
    GOVERNANCE = "governance"

class ControlFrequency(Enum):
    REAL_TIME = "real_time"
    CONTINUOUS = "continuous"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUALLY = "annually"
    EVENT_DRIVEN = "event_driven"

class ControlStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    TESTING = "testing"
    REMEDIATION = "remediation"
    UNDER_REVIEW = "under_review"
    DEPRECATED = "deprecated"

class ControlEffectiveness(Enum):
    EFFECTIVE = "effective"
    PARTIALLY_EFFECTIVE = "partially_effective"
    INEFFECTIVE = "ineffective"
    NOT_TESTED = "not_tested"
    NEEDS_IMPROVEMENT = "needs_improvement"

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ControlObjective:
    """控制目标"""
    objective_id: str
    objective_name: str
    description: str
    category: ControlCategory
    risk_level: RiskLevel
    regulatory_requirements: List[str]
    business_impact: str
    success_criteria: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class InternalControl:
    """内部控制"""
    control_id: str
    control_name: str
    description: str
    control_type: ControlType
    category: ControlCategory
    frequency: ControlFrequency
    control_objective: ControlObjective
    control_owner: str
    control_operator: str
    implementation_date: datetime
    last_review_date: Optional[datetime]
    next_review_date: datetime
    status: ControlStatus
    effectiveness: ControlEffectiveness
    control_procedures: List[str]
    key_controls: List[str]
    supporting_evidence: List[str]
    risk_mitigation: List[str]
    failure_scenarios: List[str]
    escalation_procedures: List[str]
    automation_level: float
    cost_benefit_ratio: float
    control_dependencies: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ControlTest:
    """控制测试"""
    test_id: str
    control_id: str
    test_name: str
    test_type: str
    test_date: datetime
    tester: str
    test_procedures: List[str]
    sample_size: int
    test_results: Dict[str, Any]
    exceptions_found: List[Dict[str, Any]]
    test_conclusion: str
    effectiveness_rating: ControlEffectiveness
    recommendations: List[str]
    follow_up_actions: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ControlDeficiency:
    """控制缺陷"""
    deficiency_id: str
    control_id: str
    deficiency_type: str
    severity: str
    description: str
    identified_date: datetime
    identified_by: str
    root_cause: str
    business_impact: str
    risk_rating: RiskLevel
    remediation_plan: str
    remediation_owner: str
    target_completion_date: datetime
    actual_completion_date: Optional[datetime]
    status: str
    verification_date: Optional[datetime]
    verification_by: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ControlMetrics:
    """控制指标"""
    metric_id: str
    control_id: str
    metric_name: str
    metric_type: str
    calculation_method: str
    target_value: float
    actual_value: float
    tolerance_range: Tuple[float, float]
    measurement_date: datetime
    frequency: ControlFrequency
    trend_analysis: Dict[str, Any]
    status: str
    alerts_triggered: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ControlAssessment:
    """控制评估"""
    assessment_id: str
    assessment_type: str
    assessment_date: datetime
    assessor: str
    scope: List[str]
    controls_assessed: List[str]
    assessment_results: Dict[str, Any]
    overall_rating: str
    key_findings: List[str]
    recommendations: List[str]
    action_items: List[Dict[str, Any]]
    next_assessment_date: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ControlAlert:
    """控制告警"""
    alert_id: str
    control_id: str
    alert_type: str
    severity: str
    alert_time: datetime
    description: str
    triggered_by: str
    threshold_value: float
    actual_value: float
    impact_assessment: str
    immediate_actions: List[str]
    escalation_level: int
    resolved_time: Optional[datetime]
    resolution_actions: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

class InternalControlSystems:
    """
    内部控制系统
    
    提供完整的内部控制管理功能，包括控制设计、实施、
    监控、测试和评估的全生命周期管理。
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 配置参数
        self.control_library = {}
        self.control_objectives = {}
        self.control_tests = {}
        self.control_deficiencies = {}
        self.control_metrics = {}
        self.control_assessments = {}
        self.control_alerts = {}
        
        # 监控配置
        self.monitoring_config = {
            'real_time_monitoring': config.get('real_time_monitoring', True),
            'alert_thresholds': config.get('alert_thresholds', {
                'control_failure_rate': 0.05,
                'deficiency_aging_days': 30,
                'effectiveness_threshold': 0.8
            }),
            'escalation_matrix': config.get('escalation_matrix', {
                'low': ['control_operator'],
                'medium': ['control_owner', 'supervisor'],
                'high': ['control_owner', 'supervisor', 'management'],
                'critical': ['control_owner', 'supervisor', 'management', 'board']
            })
        }
        
        # 自动化控制
        self.automated_controls = {}
        self.control_workflows = {}
        
        # 统计信息
        self.control_statistics = {
            'total_controls': 0,
            'active_controls': 0,
            'effective_controls': 0,
            'controls_by_category': {},
            'controls_by_type': {},
            'deficiencies_by_severity': {},
            'tests_conducted': 0,
            'automated_control_percentage': 0.0
        }
        
        # 初始化
        self._initialize_control_framework()
        self._initialize_control_library()
        
        self.logger.info("内部控制系统初始化完成")
    
    def _initialize_control_framework(self):
        """初始化控制框架"""
        # 创建控制目标
        self._create_standard_control_objectives()
        
        # 设置控制工作流
        self._setup_control_workflows()
        
        # 配置自动化控制
        self._configure_automated_controls()
    
    def _create_standard_control_objectives(self):
        """创建标准控制目标"""
        # 交易控制目标
        trading_objectives = [
            ControlObjective(
                objective_id="TRADE_001",
                objective_name="交易授权控制",
                description="确保所有交易都经过适当授权",
                category=ControlCategory.TRADING,
                risk_level=RiskLevel.HIGH,
                regulatory_requirements=["MiFID II", "SOX"],
                business_impact="防止未授权交易造成损失",
                success_criteria=["100%交易授权覆盖率", "零未授权交易事件"]
            ),
            ControlObjective(
                objective_id="TRADE_002",
                objective_name="交易限额控制",
                description="确保交易不超过预设限额",
                category=ControlCategory.TRADING,
                risk_level=RiskLevel.HIGH,
                regulatory_requirements=["Basel III", "Internal Policy"],
                business_impact="防止过度风险敞口",
                success_criteria=["零限额超限事件", "实时监控覆盖率100%"]
            ),
            ControlObjective(
                objective_id="TRADE_003",
                objective_name="交易确认控制",
                description="确保交易准确记录和确认",
                category=ControlCategory.TRADING,
                risk_level=RiskLevel.MEDIUM,
                regulatory_requirements=["MiFID II"],
                business_impact="确保交易记录准确性",
                success_criteria=["交易确认准确率>99.9%", "确认时效性<T+1"]
            )
        ]
        
        # 风险管理控制目标
        risk_objectives = [
            ControlObjective(
                objective_id="RISK_001",
                objective_name="风险限额监控",
                description="实时监控各类风险限额",
                category=ControlCategory.RISK_MANAGEMENT,
                risk_level=RiskLevel.CRITICAL,
                regulatory_requirements=["Basel III", "Internal Policy"],
                business_impact="防止风险超限",
                success_criteria=["实时监控", "零超限事件"]
            ),
            ControlObjective(
                objective_id="RISK_002",
                objective_name="压力测试控制",
                description="定期进行压力测试",
                category=ControlCategory.RISK_MANAGEMENT,
                risk_level=RiskLevel.HIGH,
                regulatory_requirements=["Basel III", "CCAR"],
                business_impact="评估极端情况下的风险",
                success_criteria=["月度压力测试", "测试覆盖率100%"]
            )
        ]
        
        # 合规控制目标
        compliance_objectives = [
            ControlObjective(
                objective_id="COMP_001",
                objective_name="监管报告控制",
                description="确保监管报告及时准确",
                category=ControlCategory.COMPLIANCE,
                risk_level=RiskLevel.HIGH,
                regulatory_requirements=["All Applicable"],
                business_impact="避免监管处罚",
                success_criteria=["报告及时率100%", "报告准确率>99.9%"]
            ),
            ControlObjective(
                objective_id="COMP_002",
                objective_name="反洗钱控制",
                description="防止洗钱和恐怖主义融资",
                category=ControlCategory.COMPLIANCE,
                risk_level=RiskLevel.CRITICAL,
                regulatory_requirements=["AML/CFT"],
                business_impact="防止洗钱风险",
                success_criteria=["可疑交易识别率>95%", "零洗钱事件"]
            )
        ]
        
        # 添加到控制目标库
        for objectives in [trading_objectives, risk_objectives, compliance_objectives]:
            for objective in objectives:
                self.control_objectives[objective.objective_id] = objective
    
    def _setup_control_workflows(self):
        """设置控制工作流"""
        self.control_workflows = {
            'control_design': {
                'steps': [
                    'identify_risks',
                    'define_objectives',
                    'design_controls',
                    'review_design',
                    'approve_design'
                ],
                'approvers': ['risk_manager', 'compliance_officer', 'cro']
            },
            'control_implementation': {
                'steps': [
                    'develop_procedures',
                    'train_personnel',
                    'configure_systems',
                    'test_controls',
                    'deploy_controls'
                ],
                'approvers': ['control_owner', 'it_manager']
            },
            'control_testing': {
                'steps': [
                    'plan_testing',
                    'execute_tests',
                    'analyze_results',
                    'report_findings',
                    'follow_up_actions'
                ],
                'approvers': ['internal_auditor', 'control_owner']
            },
            'deficiency_remediation': {
                'steps': [
                    'identify_deficiency',
                    'assess_impact',
                    'develop_remediation',
                    'implement_remediation',
                    'validate_remediation'
                ],
                'approvers': ['control_owner', 'risk_manager']
            }
        }
    
    def _configure_automated_controls(self):
        """配置自动化控制"""
        self.automated_controls = {
            'trading_limits': {
                'description': '交易限额自动检查',
                'frequency': ControlFrequency.REAL_TIME,
                'check_function': self._check_trading_limits,
                'alert_threshold': 0.9,
                'escalation_threshold': 1.0
            },
            'risk_limits': {
                'description': '风险限额自动检查',
                'frequency': ControlFrequency.REAL_TIME,
                'check_function': self._check_risk_limits,
                'alert_threshold': 0.85,
                'escalation_threshold': 0.95
            },
            'compliance_screening': {
                'description': '合规筛查',
                'frequency': ControlFrequency.REAL_TIME,
                'check_function': self._compliance_screening,
                'alert_threshold': 0.7,
                'escalation_threshold': 0.9
            },
            'data_validation': {
                'description': '数据验证',
                'frequency': ControlFrequency.CONTINUOUS,
                'check_function': self._validate_data,
                'alert_threshold': 0.05,
                'escalation_threshold': 0.1
            }
        }
    
    def _initialize_control_library(self):
        """初始化控制库"""
        # 创建标准控制
        self._create_standard_controls()
        
        # 更新统计信息
        self._update_control_statistics()
    
    def _create_standard_controls(self):
        """创建标准控制"""
        # 交易授权控制
        trading_auth_control = InternalControl(
            control_id="CTRL_TRADE_001",
            control_name="交易授权控制",
            description="确保所有交易都经过适当的授权批准",
            control_type=ControlType.PREVENTIVE,
            category=ControlCategory.TRADING,
            frequency=ControlFrequency.REAL_TIME,
            control_objective=self.control_objectives["TRADE_001"],
            control_owner="trading_manager",
            control_operator="trading_system",
            implementation_date=datetime.now() - timedelta(days=365),
            last_review_date=datetime.now() - timedelta(days=30),
            next_review_date=datetime.now() + timedelta(days=90),
            status=ControlStatus.ACTIVE,
            effectiveness=ControlEffectiveness.EFFECTIVE,
            control_procedures=[
                "验证交易员授权级别",
                "检查交易限额",
                "记录授权决策",
                "生成异常报告"
            ],
            key_controls=[
                "实时授权检查",
                "双重授权要求",
                "授权日志记录"
            ],
            supporting_evidence=[
                "授权矩阵",
                "交易日志",
                "系统配置文档"
            ],
            risk_mitigation=[
                "防止未授权交易",
                "降低操作风险",
                "满足监管要求"
            ],
            failure_scenarios=[
                "系统故障导致授权失效",
                "授权矩阵配置错误",
                "网络延迟影响实时检查"
            ],
            escalation_procedures=[
                "立即通知交易主管",
                "暂停相关交易活动",
                "启动应急授权程序"
            ],
            automation_level=0.95,
            cost_benefit_ratio=4.2,
            control_dependencies=["user_authentication", "trading_system"]
        )
        
        # 风险限额控制
        risk_limit_control = InternalControl(
            control_id="CTRL_RISK_001",
            control_name="风险限额监控控制",
            description="实时监控各类风险指标，确保不超过预设限额",
            control_type=ControlType.DETECTIVE,
            category=ControlCategory.RISK_MANAGEMENT,
            frequency=ControlFrequency.REAL_TIME,
            control_objective=self.control_objectives["RISK_001"],
            control_owner="risk_manager",
            control_operator="risk_system",
            implementation_date=datetime.now() - timedelta(days=300),
            last_review_date=datetime.now() - timedelta(days=15),
            next_review_date=datetime.now() + timedelta(days=60),
            status=ControlStatus.ACTIVE,
            effectiveness=ControlEffectiveness.EFFECTIVE,
            control_procedures=[
                "实时计算风险指标",
                "与限额进行比较",
                "触发告警机制",
                "生成风险报告"
            ],
            key_controls=[
                "实时VaR监控",
                "限额预警系统",
                "风险仪表板"
            ],
            supporting_evidence=[
                "风险报告",
                "限额配置",
                "监控日志"
            ],
            risk_mitigation=[
                "防止风险超限",
                "及时风险预警",
                "支持风险决策"
            ],
            failure_scenarios=[
                "数据延迟导致监控失效",
                "计算错误",
                "告警系统故障"
            ],
            escalation_procedures=[
                "立即通知风险官",
                "暂停高风险交易",
                "启动风险管理程序"
            ],
            automation_level=0.98,
            cost_benefit_ratio=5.1,
            control_dependencies=["market_data", "position_data", "risk_engine"]
        )
        
        # 合规筛查控制
        compliance_screening_control = InternalControl(
            control_id="CTRL_COMP_001",
            control_name="合规筛查控制",
            description="对交易和客户进行合规性筛查",
            control_type=ControlType.PREVENTIVE,
            category=ControlCategory.COMPLIANCE,
            frequency=ControlFrequency.REAL_TIME,
            control_objective=self.control_objectives["COMP_002"],
            control_owner="compliance_officer",
            control_operator="compliance_system",
            implementation_date=datetime.now() - timedelta(days=180),
            last_review_date=datetime.now() - timedelta(days=45),
            next_review_date=datetime.now() + timedelta(days=90),
            status=ControlStatus.ACTIVE,
            effectiveness=ControlEffectiveness.EFFECTIVE,
            control_procedures=[
                "检查制裁名单",
                "监控可疑交易模式",
                "验证客户身份",
                "生成合规报告"
            ],
            key_controls=[
                "反洗钱系统",
                "制裁名单筛查",
                "可疑交易监控"
            ],
            supporting_evidence=[
                "筛查日志",
                "合规报告",
                "客户尽职调查"
            ],
            risk_mitigation=[
                "防止洗钱风险",
                "避免监管处罚",
                "保护机构声誉"
            ],
            failure_scenarios=[
                "筛查规则不全",
                "数据质量问题",
                "系统性能问题"
            ],
            escalation_procedures=[
                "立即通知合规官",
                "冻结可疑交易",
                "启动调查程序"
            ],
            automation_level=0.85,
            cost_benefit_ratio=3.8,
            control_dependencies=["customer_data", "transaction_data", "sanctions_data"]
        )
        
        # 添加到控制库
        controls = [trading_auth_control, risk_limit_control, compliance_screening_control]
        for control in controls:
            self.control_library[control.control_id] = control
    
    async def add_control(self, control: InternalControl):
        """添加控制"""
        self.control_library[control.control_id] = control
        self._update_control_statistics()
        self.logger.info(f"已添加控制: {control.control_id} - {control.control_name}")
    
    async def update_control(self, control_id: str, updates: Dict[str, Any]):
        """更新控制"""
        if control_id in self.control_library:
            control = self.control_library[control_id]
            for key, value in updates.items():
                if hasattr(control, key):
                    setattr(control, key, value)
            self._update_control_statistics()
            self.logger.info(f"已更新控制: {control_id}")
        else:
            self.logger.error(f"控制不存在: {control_id}")
    
    async def deactivate_control(self, control_id: str, reason: str):
        """停用控制"""
        if control_id in self.control_library:
            control = self.control_library[control_id]
            control.status = ControlStatus.INACTIVE
            control.metadata['deactivation_reason'] = reason
            control.metadata['deactivation_date'] = datetime.now()
            self._update_control_statistics()
            self.logger.info(f"已停用控制: {control_id}")
        else:
            self.logger.error(f"控制不存在: {control_id}")
    
    async def execute_control_test(self, control_id: str, test_config: Dict[str, Any]) -> ControlTest:
        """执行控制测试"""
        if control_id not in self.control_library:
            raise ValueError(f"控制不存在: {control_id}")
        
        control = self.control_library[control_id]
        test_id = f"TEST_{control_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 执行测试
        test_results = await self._perform_control_test(control, test_config)
        
        # 分析结果
        exceptions_found = test_results.get('exceptions', [])
        effectiveness_rating = self._assess_control_effectiveness(test_results)
        
        # 生成建议
        recommendations = self._generate_test_recommendations(test_results, exceptions_found)
        
        # 创建测试记录
        control_test = ControlTest(
            test_id=test_id,
            control_id=control_id,
            test_name=test_config.get('test_name', f"Test of {control.control_name}"),
            test_type=test_config.get('test_type', 'effectiveness'),
            test_date=datetime.now(),
            tester=test_config.get('tester', 'system'),
            test_procedures=test_config.get('test_procedures', []),
            sample_size=test_config.get('sample_size', 100),
            test_results=test_results,
            exceptions_found=exceptions_found,
            test_conclusion=test_results.get('conclusion', ''),
            effectiveness_rating=effectiveness_rating,
            recommendations=recommendations,
            follow_up_actions=test_config.get('follow_up_actions', [])
        )
        
        self.control_tests[test_id] = control_test
        
        # 更新控制有效性
        control.effectiveness = effectiveness_rating
        control.last_review_date = datetime.now()
        
        # 创建缺陷（如果有）
        if exceptions_found:
            await self._create_deficiencies_from_exceptions(control_id, exceptions_found)
        
        self.logger.info(f"控制测试完成: {test_id}")
        return control_test
    
    async def _perform_control_test(self, control: InternalControl, test_config: Dict[str, Any]) -> Dict[str, Any]:
        """执行控制测试"""
        test_results = {
            'test_passed': True,
            'exceptions': [],
            'metrics': {},
            'conclusion': '',
            'detailed_results': []
        }
        
        # 基于控制类型执行不同的测试
        if control.category == ControlCategory.TRADING:
            test_results = await self._test_trading_control(control, test_config)
        elif control.category == ControlCategory.RISK_MANAGEMENT:
            test_results = await self._test_risk_control(control, test_config)
        elif control.category == ControlCategory.COMPLIANCE:
            test_results = await self._test_compliance_control(control, test_config)
        else:
            test_results = await self._test_generic_control(control, test_config)
        
        return test_results
    
    async def _test_trading_control(self, control: InternalControl, test_config: Dict[str, Any]) -> Dict[str, Any]:
        """测试交易控制"""
        # 模拟交易控制测试
        sample_size = test_config.get('sample_size', 100)
        exceptions = []
        
        # 模拟测试数据
        for i in range(sample_size):
            # 模拟授权检查
            if np.random.random() < 0.02:  # 2%的异常率
                exceptions.append({
                    'exception_type': 'authorization_failure',
                    'description': f'交易 {i} 授权检查失败',
                    'severity': 'medium',
                    'impact': 'potential_unauthorized_trade'
                })
        
        test_passed = len(exceptions) == 0
        
        return {
            'test_passed': test_passed,
            'exceptions': exceptions,
            'metrics': {
                'sample_size': sample_size,
                'exception_rate': len(exceptions) / sample_size,
                'pass_rate': 1 - (len(exceptions) / sample_size)
            },
            'conclusion': 'Control is effective' if test_passed else 'Control has deficiencies',
            'detailed_results': [f'Tested {sample_size} transactions', f'Found {len(exceptions)} exceptions']
        }
    
    async def _test_risk_control(self, control: InternalControl, test_config: Dict[str, Any]) -> Dict[str, Any]:
        """测试风险控制"""
        # 模拟风险控制测试
        sample_size = test_config.get('sample_size', 100)
        exceptions = []
        
        # 模拟风险限额测试
        for i in range(sample_size):
            if np.random.random() < 0.01:  # 1%的异常率
                exceptions.append({
                    'exception_type': 'limit_breach',
                    'description': f'风险限额 {i} 超限未及时检测',
                    'severity': 'high',
                    'impact': 'potential_risk_exposure'
                })
        
        test_passed = len(exceptions) == 0
        
        return {
            'test_passed': test_passed,
            'exceptions': exceptions,
            'metrics': {
                'sample_size': sample_size,
                'exception_rate': len(exceptions) / sample_size,
                'detection_rate': 1 - (len(exceptions) / sample_size)
            },
            'conclusion': 'Control is effective' if test_passed else 'Control has deficiencies',
            'detailed_results': [f'Tested {sample_size} risk scenarios', f'Found {len(exceptions)} exceptions']
        }
    
    async def _test_compliance_control(self, control: InternalControl, test_config: Dict[str, Any]) -> Dict[str, Any]:
        """测试合规控制"""
        # 模拟合规控制测试
        sample_size = test_config.get('sample_size', 100)
        exceptions = []
        
        # 模拟合规筛查测试
        for i in range(sample_size):
            if np.random.random() < 0.005:  # 0.5%的异常率
                exceptions.append({
                    'exception_type': 'screening_failure',
                    'description': f'合规筛查 {i} 未检测到风险',
                    'severity': 'critical',
                    'impact': 'regulatory_violation'
                })
        
        test_passed = len(exceptions) == 0
        
        return {
            'test_passed': test_passed,
            'exceptions': exceptions,
            'metrics': {
                'sample_size': sample_size,
                'exception_rate': len(exceptions) / sample_size,
                'screening_accuracy': 1 - (len(exceptions) / sample_size)
            },
            'conclusion': 'Control is effective' if test_passed else 'Control has deficiencies',
            'detailed_results': [f'Tested {sample_size} compliance scenarios', f'Found {len(exceptions)} exceptions']
        }
    
    async def _test_generic_control(self, control: InternalControl, test_config: Dict[str, Any]) -> Dict[str, Any]:
        """测试通用控制"""
        # 模拟通用控制测试
        sample_size = test_config.get('sample_size', 100)
        exceptions = []
        
        # 基于控制自动化程度调整异常率
        base_exception_rate = 0.05 * (1 - control.automation_level)
        
        for i in range(sample_size):
            if np.random.random() < base_exception_rate:
                exceptions.append({
                    'exception_type': 'control_failure',
                    'description': f'控制执行 {i} 失败',
                    'severity': 'medium',
                    'impact': 'control_objective_not_met'
                })
        
        test_passed = len(exceptions) == 0
        
        return {
            'test_passed': test_passed,
            'exceptions': exceptions,
            'metrics': {
                'sample_size': sample_size,
                'exception_rate': len(exceptions) / sample_size,
                'control_reliability': 1 - (len(exceptions) / sample_size)
            },
            'conclusion': 'Control is effective' if test_passed else 'Control has deficiencies',
            'detailed_results': [f'Tested {sample_size} control executions', f'Found {len(exceptions)} exceptions']
        }
    
    def _assess_control_effectiveness(self, test_results: Dict[str, Any]) -> ControlEffectiveness:
        """评估控制有效性"""
        if test_results['test_passed']:
            return ControlEffectiveness.EFFECTIVE
        
        exception_rate = test_results['metrics'].get('exception_rate', 0)
        
        if exception_rate < 0.01:
            return ControlEffectiveness.EFFECTIVE
        elif exception_rate < 0.05:
            return ControlEffectiveness.PARTIALLY_EFFECTIVE
        else:
            return ControlEffectiveness.INEFFECTIVE
    
    def _generate_test_recommendations(self, test_results: Dict[str, Any], exceptions: List[Dict[str, Any]]) -> List[str]:
        """生成测试建议"""
        recommendations = []
        
        if test_results['test_passed']:
            recommendations.append("控制运行良好，建议继续监控")
        else:
            exception_rate = test_results['metrics'].get('exception_rate', 0)
            
            if exception_rate > 0.1:
                recommendations.append("控制失效率过高，建议立即重新设计控制")
            elif exception_rate > 0.05:
                recommendations.append("控制需要改进，建议加强监控和培训")
            else:
                recommendations.append("控制基本有效，建议针对具体异常进行优化")
            
            # 基于异常类型的建议
            exception_types = [e['exception_type'] for e in exceptions]
            if 'authorization_failure' in exception_types:
                recommendations.append("加强授权控制的系统配置和流程")
            if 'limit_breach' in exception_types:
                recommendations.append("优化风险限额监控的实时性")
            if 'screening_failure' in exception_types:
                recommendations.append("更新合规筛查规则和数据源")
        
        return recommendations
    
    async def _create_deficiencies_from_exceptions(self, control_id: str, exceptions: List[Dict[str, Any]]):
        """从异常创建缺陷"""
        for exception in exceptions:
            deficiency_id = f"DEF_{control_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            # 确定严重程度
            severity_map = {
                'low': RiskLevel.LOW,
                'medium': RiskLevel.MEDIUM,
                'high': RiskLevel.HIGH,
                'critical': RiskLevel.CRITICAL
            }
            
            deficiency = ControlDeficiency(
                deficiency_id=deficiency_id,
                control_id=control_id,
                deficiency_type=exception['exception_type'],
                severity=exception['severity'],
                description=exception['description'],
                identified_date=datetime.now(),
                identified_by='control_test',
                root_cause='To be determined',
                business_impact=exception['impact'],
                risk_rating=severity_map.get(exception['severity'], RiskLevel.MEDIUM),
                remediation_plan='To be developed',
                remediation_owner='control_owner',
                target_completion_date=datetime.now() + timedelta(days=30),
                status='open'
            )
            
            self.control_deficiencies[deficiency_id] = deficiency
            self.logger.warning(f"创建控制缺陷: {deficiency_id}")
    
    async def monitor_automated_controls(self):
        """监控自动化控制"""
        for control_name, control_config in self.automated_controls.items():
            try:
                # 执行控制检查
                check_result = await control_config['check_function']()
                
                # 评估结果
                if check_result['status'] == 'alert':
                    await self._create_control_alert(control_name, check_result)
                elif check_result['status'] == 'escalation':
                    await self._escalate_control_issue(control_name, check_result)
                
            except Exception as e:
                self.logger.error(f"自动化控制监控失败: {control_name} - {e}")
                await self._create_control_alert(control_name, {
                    'status': 'error',
                    'message': f'Control monitoring failed: {e}',
                    'severity': 'high'
                })
    
    async def _check_trading_limits(self) -> Dict[str, Any]:
        """检查交易限额"""
        # 模拟交易限额检查
        current_usage = np.random.uniform(0.5, 1.2)
        limit_threshold = 1.0
        
        if current_usage >= limit_threshold:
            return {
                'status': 'escalation',
                'message': f'交易限额超限: {current_usage:.2f} >= {limit_threshold}',
                'severity': 'critical',
                'metric_value': current_usage,
                'threshold_value': limit_threshold
            }
        elif current_usage >= 0.9:
            return {
                'status': 'alert',
                'message': f'交易限额接近上限: {current_usage:.2f}',
                'severity': 'high',
                'metric_value': current_usage,
                'threshold_value': 0.9
            }
        else:
            return {
                'status': 'normal',
                'message': '交易限额正常',
                'severity': 'low',
                'metric_value': current_usage,
                'threshold_value': limit_threshold
            }
    
    async def _check_risk_limits(self) -> Dict[str, Any]:
        """检查风险限额"""
        # 模拟风险限额检查
        current_var = np.random.uniform(0.01, 0.05)
        var_limit = 0.03
        
        if current_var >= var_limit:
            return {
                'status': 'escalation',
                'message': f'VaR超限: {current_var:.3f} >= {var_limit:.3f}',
                'severity': 'critical',
                'metric_value': current_var,
                'threshold_value': var_limit
            }
        elif current_var >= var_limit * 0.85:
            return {
                'status': 'alert',
                'message': f'VaR接近上限: {current_var:.3f}',
                'severity': 'high',
                'metric_value': current_var,
                'threshold_value': var_limit * 0.85
            }
        else:
            return {
                'status': 'normal',
                'message': 'VaR正常',
                'severity': 'low',
                'metric_value': current_var,
                'threshold_value': var_limit
            }
    
    async def _compliance_screening(self) -> Dict[str, Any]:
        """合规筛查"""
        # 模拟合规筛查
        suspicious_score = np.random.uniform(0.0, 1.0)
        
        if suspicious_score >= 0.9:
            return {
                'status': 'escalation',
                'message': f'高度可疑交易: 评分 {suspicious_score:.2f}',
                'severity': 'critical',
                'metric_value': suspicious_score,
                'threshold_value': 0.9
            }
        elif suspicious_score >= 0.7:
            return {
                'status': 'alert',
                'message': f'可疑交易: 评分 {suspicious_score:.2f}',
                'severity': 'medium',
                'metric_value': suspicious_score,
                'threshold_value': 0.7
            }
        else:
            return {
                'status': 'normal',
                'message': '合规筛查正常',
                'severity': 'low',
                'metric_value': suspicious_score,
                'threshold_value': 0.7
            }
    
    async def _validate_data(self) -> Dict[str, Any]:
        """验证数据"""
        # 模拟数据验证
        error_rate = np.random.uniform(0.0, 0.2)
        
        if error_rate >= 0.1:
            return {
                'status': 'escalation',
                'message': f'数据错误率过高: {error_rate:.3f}',
                'severity': 'high',
                'metric_value': error_rate,
                'threshold_value': 0.1
            }
        elif error_rate >= 0.05:
            return {
                'status': 'alert',
                'message': f'数据错误率告警: {error_rate:.3f}',
                'severity': 'medium',
                'metric_value': error_rate,
                'threshold_value': 0.05
            }
        else:
            return {
                'status': 'normal',
                'message': '数据验证正常',
                'severity': 'low',
                'metric_value': error_rate,
                'threshold_value': 0.05
            }
    
    async def _create_control_alert(self, control_name: str, check_result: Dict[str, Any]):
        """创建控制告警"""
        alert_id = f"ALERT_{control_name}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # 查找相关控制ID
        control_id = None
        for ctrl_id, ctrl in self.control_library.items():
            if control_name in ctrl.control_name.lower():
                control_id = ctrl_id
                break
        
        alert = ControlAlert(
            alert_id=alert_id,
            control_id=control_id or 'unknown',
            alert_type='automated_control',
            severity=check_result['severity'],
            alert_time=datetime.now(),
            description=check_result['message'],
            triggered_by=control_name,
            threshold_value=check_result.get('threshold_value', 0.0),
            actual_value=check_result.get('metric_value', 0.0),
            impact_assessment=self._assess_control_impact(check_result),
            immediate_actions=self._get_immediate_actions(check_result),
            escalation_level=self._get_escalation_level(check_result['severity'])
        )
        
        self.control_alerts[alert_id] = alert
        self.logger.warning(f"控制告警: {alert_id} - {check_result['message']}")
    
    async def _escalate_control_issue(self, control_name: str, check_result: Dict[str, Any]):
        """升级控制问题"""
        # 创建告警
        await self._create_control_alert(control_name, check_result)
        
        # 执行升级程序
        escalation_level = self._get_escalation_level(check_result['severity'])
        recipients = self.monitoring_config['escalation_matrix'].get(check_result['severity'], [])
        
        self.logger.critical(f"控制升级: {control_name} - {check_result['message']}")
        self.logger.critical(f"升级级别: {escalation_level}, 接收人: {recipients}")
        
        # 这里可以添加实际的通知逻辑（邮件、短信等）
    
    def _assess_control_impact(self, check_result: Dict[str, Any]) -> str:
        """评估控制影响"""
        severity = check_result['severity']
        
        impact_map = {
            'low': '影响较小，建议持续监控',
            'medium': '中等影响，需要关注并采取措施',
            'high': '高影响，需要立即关注',
            'critical': '严重影响，需要紧急处理'
        }
        
        return impact_map.get(severity, '影响未知')
    
    def _get_immediate_actions(self, check_result: Dict[str, Any]) -> List[str]:
        """获取立即行动"""
        severity = check_result['severity']
        
        actions_map = {
            'low': ['继续监控', '记录事件'],
            'medium': ['通知相关人员', '分析根本原因', '制定改进计划'],
            'high': ['立即通知管理层', '暂停相关活动', '启动应急程序'],
            'critical': ['立即升级', '紧急停止相关操作', '启动危机管理']
        }
        
        return actions_map.get(severity, ['联系管理员'])
    
    def _get_escalation_level(self, severity: str) -> int:
        """获取升级级别"""
        level_map = {
            'low': 1,
            'medium': 2,
            'high': 3,
            'critical': 4
        }
        
        return level_map.get(severity, 2)
    
    def _update_control_statistics(self):
        """更新控制统计"""
        total_controls = len(self.control_library)
        active_controls = len([c for c in self.control_library.values() if c.status == ControlStatus.ACTIVE])
        effective_controls = len([c for c in self.control_library.values() if c.effectiveness == ControlEffectiveness.EFFECTIVE])
        
        # 按类别统计
        controls_by_category = {}
        for control in self.control_library.values():
            category = control.category.value
            controls_by_category[category] = controls_by_category.get(category, 0) + 1
        
        # 按类型统计
        controls_by_type = {}
        for control in self.control_library.values():
            control_type = control.control_type.value
            controls_by_type[control_type] = controls_by_type.get(control_type, 0) + 1
        
        # 按缺陷严重程度统计
        deficiencies_by_severity = {}
        for deficiency in self.control_deficiencies.values():
            severity = deficiency.severity
            deficiencies_by_severity[severity] = deficiencies_by_severity.get(severity, 0) + 1
        
        # 自动化控制百分比
        automated_controls = len([c for c in self.control_library.values() if c.automation_level >= 0.8])
        automation_percentage = (automated_controls / total_controls * 100) if total_controls > 0 else 0
        
        self.control_statistics = {
            'total_controls': total_controls,
            'active_controls': active_controls,
            'effective_controls': effective_controls,
            'controls_by_category': controls_by_category,
            'controls_by_type': controls_by_type,
            'deficiencies_by_severity': deficiencies_by_severity,
            'tests_conducted': len(self.control_tests),
            'automated_control_percentage': automation_percentage
        }
    
    async def generate_control_assessment(self, assessment_type: str, scope: List[str]) -> ControlAssessment:
        """生成控制评估"""
        assessment_id = f"ASSESS_{assessment_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 获取评估范围内的控制
        controls_to_assess = []
        for control_id, control in self.control_library.items():
            if not scope or control.category.value in scope:
                controls_to_assess.append(control_id)
        
        # 执行评估
        assessment_results = await self._perform_control_assessment(controls_to_assess)
        
        # 生成发现和建议
        key_findings = self._generate_assessment_findings(assessment_results)
        recommendations = self._generate_assessment_recommendations(assessment_results)
        
        # 创建行动项
        action_items = self._create_action_items(key_findings, recommendations)
        
        # 确定整体评级
        overall_rating = self._determine_overall_rating(assessment_results)
        
        assessment = ControlAssessment(
            assessment_id=assessment_id,
            assessment_type=assessment_type,
            assessment_date=datetime.now(),
            assessor='internal_control_system',
            scope=scope,
            controls_assessed=controls_to_assess,
            assessment_results=assessment_results,
            overall_rating=overall_rating,
            key_findings=key_findings,
            recommendations=recommendations,
            action_items=action_items,
            next_assessment_date=datetime.now() + timedelta(days=90)
        )
        
        self.control_assessments[assessment_id] = assessment
        self.logger.info(f"控制评估完成: {assessment_id}")
        return assessment
    
    async def _perform_control_assessment(self, control_ids: List[str]) -> Dict[str, Any]:
        """执行控制评估"""
        results = {
            'controls_assessed': len(control_ids),
            'effective_controls': 0,
            'partially_effective_controls': 0,
            'ineffective_controls': 0,
            'not_tested_controls': 0,
            'control_details': {},
            'risk_assessment': {},
            'compliance_status': {}
        }
        
        for control_id in control_ids:
            if control_id in self.control_library:
                control = self.control_library[control_id]
                
                # 评估控制有效性
                if control.effectiveness == ControlEffectiveness.EFFECTIVE:
                    results['effective_controls'] += 1
                elif control.effectiveness == ControlEffectiveness.PARTIALLY_EFFECTIVE:
                    results['partially_effective_controls'] += 1
                elif control.effectiveness == ControlEffectiveness.INEFFECTIVE:
                    results['ineffective_controls'] += 1
                else:
                    results['not_tested_controls'] += 1
                
                # 记录控制详情
                results['control_details'][control_id] = {
                    'name': control.control_name,
                    'type': control.control_type.value,
                    'category': control.category.value,
                    'effectiveness': control.effectiveness.value,
                    'automation_level': control.automation_level,
                    'last_review': control.last_review_date.isoformat() if control.last_review_date else None
                }
        
        return results
    
    def _generate_assessment_findings(self, assessment_results: Dict[str, Any]) -> List[str]:
        """生成评估发现"""
        findings = []
        
        total_controls = assessment_results['controls_assessed']
        effective_rate = assessment_results['effective_controls'] / total_controls if total_controls > 0 else 0
        
        if effective_rate >= 0.9:
            findings.append("控制整体运行良好，有效性达到90%以上")
        elif effective_rate >= 0.7:
            findings.append("控制基本有效，但仍有改进空间")
        else:
            findings.append("控制有效性不足，需要重点关注")
        
        if assessment_results['ineffective_controls'] > 0:
            findings.append(f"发现 {assessment_results['ineffective_controls']} 个无效控制，需要立即处理")
        
        if assessment_results['not_tested_controls'] > 0:
            findings.append(f"有 {assessment_results['not_tested_controls']} 个控制未经测试，建议尽快测试")
        
        return findings
    
    def _generate_assessment_recommendations(self, assessment_results: Dict[str, Any]) -> List[str]:
        """生成评估建议"""
        recommendations = []
        
        if assessment_results['ineffective_controls'] > 0:
            recommendations.append("重新设计无效控制并加强监控")
        
        if assessment_results['not_tested_controls'] > 0:
            recommendations.append("制定控制测试计划并及时执行")
        
        if assessment_results['partially_effective_controls'] > 0:
            recommendations.append("分析部分有效控制的改进机会")
        
        recommendations.extend([
            "定期审查控制有效性",
            "加强控制培训和意识",
            "持续优化控制流程",
            "建立控制绩效指标"
        ])
        
        return recommendations
    
    def _create_action_items(self, findings: List[str], recommendations: List[str]) -> List[Dict[str, Any]]:
        """创建行动项"""
        action_items = []
        
        for i, recommendation in enumerate(recommendations):
            action_items.append({
                'action_id': f"ACTION_{i+1}",
                'description': recommendation,
                'priority': 'high' if i < 2 else 'medium',
                'assigned_to': 'control_owner',
                'due_date': (datetime.now() + timedelta(days=30)).isoformat(),
                'status': 'open'
            })
        
        return action_items
    
    def _determine_overall_rating(self, assessment_results: Dict[str, Any]) -> str:
        """确定整体评级"""
        total_controls = assessment_results['controls_assessed']
        effective_rate = assessment_results['effective_controls'] / total_controls if total_controls > 0 else 0
        
        if effective_rate >= 0.9:
            return 'excellent'
        elif effective_rate >= 0.8:
            return 'good'
        elif effective_rate >= 0.7:
            return 'satisfactory'
        elif effective_rate >= 0.6:
            return 'needs_improvement'
        else:
            return 'unsatisfactory'
    
    async def get_control_dashboard(self) -> Dict[str, Any]:
        """获取控制仪表板"""
        # 更新统计信息
        self._update_control_statistics()
        
        # 获取最新告警
        recent_alerts = list(self.control_alerts.values())[-10:]
        
        # 获取待处理缺陷
        open_deficiencies = [d for d in self.control_deficiencies.values() if d.status == 'open']
        
        # 获取最近测试
        recent_tests = list(self.control_tests.values())[-5:]
        
        dashboard = {
            'summary': {
                'total_controls': self.control_statistics['total_controls'],
                'active_controls': self.control_statistics['active_controls'],
                'effective_controls': self.control_statistics['effective_controls'],
                'automation_percentage': self.control_statistics['automated_control_percentage'],
                'open_deficiencies': len(open_deficiencies),
                'recent_alerts': len(recent_alerts)
            },
            'controls_by_category': self.control_statistics['controls_by_category'],
            'controls_by_type': self.control_statistics['controls_by_type'],
            'deficiencies_by_severity': self.control_statistics['deficiencies_by_severity'],
            'recent_alerts': [
                {
                    'alert_id': alert.alert_id,
                    'control_id': alert.control_id,
                    'severity': alert.severity,
                    'description': alert.description,
                    'alert_time': alert.alert_time.isoformat()
                }
                for alert in recent_alerts
            ],
            'open_deficiencies': [
                {
                    'deficiency_id': def_.deficiency_id,
                    'control_id': def_.control_id,
                    'severity': def_.severity,
                    'description': def_.description,
                    'target_completion_date': def_.target_completion_date.isoformat()
                }
                for def_ in open_deficiencies
            ],
            'recent_tests': [
                {
                    'test_id': test.test_id,
                    'control_id': test.control_id,
                    'test_date': test.test_date.isoformat(),
                    'effectiveness_rating': test.effectiveness_rating.value,
                    'exceptions_found': len(test.exceptions_found)
                }
                for test in recent_tests
            ]
        }
        
        return dashboard
    
    async def export_control_report(self, format: str = "json") -> str:
        """导出控制报告"""
        report_data = {
            'export_time': datetime.now().isoformat(),
            'control_library': {
                control_id: {
                    'control_name': control.control_name,
                    'control_type': control.control_type.value,
                    'category': control.category.value,
                    'status': control.status.value,
                    'effectiveness': control.effectiveness.value,
                    'automation_level': control.automation_level,
                    'last_review_date': control.last_review_date.isoformat() if control.last_review_date else None
                }
                for control_id, control in self.control_library.items()
            },
            'control_statistics': self.control_statistics,
            'control_deficiencies': {
                def_id: {
                    'control_id': def_.control_id,
                    'severity': def_.severity,
                    'description': def_.description,
                    'status': def_.status,
                    'target_completion_date': def_.target_completion_date.isoformat()
                }
                for def_id, def_ in self.control_deficiencies.items()
            },
            'control_tests': {
                test_id: {
                    'control_id': test.control_id,
                    'test_date': test.test_date.isoformat(),
                    'effectiveness_rating': test.effectiveness_rating.value,
                    'exceptions_count': len(test.exceptions_found)
                }
                for test_id, test in self.control_tests.items()
            }
        }
        
        filename = f"control_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}"
        
        if format.lower() == "json":
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
        elif format.lower() == "csv":
            # 导出控制清单
            df = pd.DataFrame([
                {
                    'Control ID': control_id,
                    'Control Name': data['control_name'],
                    'Type': data['control_type'],
                    'Category': data['category'],
                    'Status': data['status'],
                    'Effectiveness': data['effectiveness'],
                    'Automation Level': data['automation_level']
                }
                for control_id, data in report_data['control_library'].items()
            ])
            df.to_csv(filename, index=False)
        
        self.logger.info(f"控制报告已导出: {filename}")
        return filename
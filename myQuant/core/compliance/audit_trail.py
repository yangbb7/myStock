import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from pathlib import Path
import json
import hashlib
import uuid
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

class AuditEventType(Enum):
    TRADE_EXECUTION = "trade_execution"
    ORDER_PLACEMENT = "order_placement"
    ORDER_MODIFICATION = "order_modification"
    ORDER_CANCELLATION = "order_cancellation"
    POSITION_CHANGE = "position_change"
    CASH_MOVEMENT = "cash_movement"
    RISK_LIMIT_BREACH = "risk_limit_breach"
    COMPLIANCE_VIOLATION = "compliance_violation"
    SYSTEM_ACCESS = "system_access"
    DATA_MODIFICATION = "data_modification"
    REPORT_GENERATION = "report_generation"
    APPROVAL_WORKFLOW = "approval_workflow"
    EXCEPTION_HANDLING = "exception_handling"
    CONFIGURATION_CHANGE = "configuration_change"
    BACKUP_OPERATION = "backup_operation"

class AuditSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AuditDataType(Enum):
    TRANSACTION = "transaction"
    SYSTEM = "system"
    SECURITY = "security"
    COMPLIANCE = "compliance"
    OPERATIONAL = "operational"

class AuditStatus(Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    SEALED = "sealed"
    DELETED = "deleted"

@dataclass
class AuditEvent:
    """审计事件"""
    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    user_id: str
    session_id: str
    source_system: str
    severity: AuditSeverity
    data_type: AuditDataType
    event_description: str
    before_state: Dict[str, Any]
    after_state: Dict[str, Any]
    affected_entities: List[str]
    ip_address: str
    user_agent: str
    transaction_id: Optional[str] = None
    correlation_id: Optional[str] = None
    risk_score: float = 0.0
    compliance_flags: List[str] = field(default_factory=list)
    digital_signature: Optional[str] = None
    hash_value: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AuditQuery:
    """审计查询"""
    query_id: str
    user_id: str
    query_timestamp: datetime
    event_types: List[AuditEventType]
    date_range: Tuple[datetime, datetime]
    user_filter: Optional[str] = None
    severity_filter: Optional[AuditSeverity] = None
    entity_filter: Optional[str] = None
    transaction_filter: Optional[str] = None
    max_results: int = 1000
    include_metadata: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AuditReport:
    """审计报告"""
    report_id: str
    report_type: str
    generation_time: datetime
    reporting_period: Tuple[datetime, datetime]
    total_events: int
    event_summary: Dict[str, int]
    severity_breakdown: Dict[str, int]
    user_activity: Dict[str, int]
    compliance_issues: List[Dict[str, Any]]
    risk_analysis: Dict[str, Any]
    recommendations: List[str]
    detailed_events: List[AuditEvent]
    digital_signature: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AuditConfiguration:
    """审计配置"""
    config_id: str
    retention_days: int
    encryption_enabled: bool
    signature_required: bool
    real_time_monitoring: bool
    alert_thresholds: Dict[str, Any]
    backup_frequency: str
    archive_frequency: str
    compliance_standards: List[str]
    sensitive_fields: List[str]
    access_controls: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AuditIntegrityCheck:
    """审计完整性检查"""
    check_id: str
    check_timestamp: datetime
    check_type: str
    events_checked: int
    integrity_status: str
    hash_mismatches: List[str]
    missing_events: List[str]
    corrupted_events: List[str]
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

class AuditTrail:
    """
    综合审计跟踪系统
    
    提供完整的交易和系统活动审计功能，包括事件记录、
    完整性验证、合规报告和风险分析。
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 配置参数
        self.database_path = config.get('database_path', './audit_trail.db')
        self.encryption_key = self._generate_encryption_key(config.get('encryption_password', 'default_password'))
        self.max_memory_events = config.get('max_memory_events', 10000)
        self.batch_size = config.get('batch_size', 100)
        
        # 审计配置
        self.audit_config = AuditConfiguration(
            config_id="default_config",
            retention_days=config.get('retention_days', 2555),  # 7年
            encryption_enabled=config.get('encryption_enabled', True),
            signature_required=config.get('signature_required', True),
            real_time_monitoring=config.get('real_time_monitoring', True),
            alert_thresholds=config.get('alert_thresholds', {
                'high_risk_threshold': 0.8,
                'critical_events_per_hour': 10,
                'failed_logins_per_hour': 5
            }),
            backup_frequency=config.get('backup_frequency', 'daily'),
            archive_frequency=config.get('archive_frequency', 'monthly'),
            compliance_standards=config.get('compliance_standards', ['SOX', 'MiFID II', 'GDPR']),
            sensitive_fields=config.get('sensitive_fields', ['password', 'ssn', 'account_number']),
            access_controls=config.get('access_controls', {
                'read_roles': ['auditor', 'compliance_officer', 'admin'],
                'write_roles': ['admin'],
                'delete_roles': ['admin']
            })
        )
        
        # 内存事件缓存
        self.memory_events = []
        self.event_index = {}
        
        # 监控统计
        self.monitoring_stats = {
            'total_events': 0,
            'events_by_type': {},
            'events_by_severity': {},
            'events_by_user': {},
            'integrity_checks': 0,
            'compliance_violations': 0,
            'risk_alerts': 0
        }
        
        # 实时监控
        self.real_time_alerts = []
        self.risk_patterns = {}
        
        # 数据库连接池
        self.db_pool = []
        
        # 初始化
        self._initialize_database()
        self._initialize_risk_patterns()
        
        self.logger.info("审计跟踪系统初始化完成")
    
    def _generate_encryption_key(self, password: str) -> bytes:
        """生成加密密钥"""
        password_bytes = password.encode()
        salt = b'salt_for_audit_trail'  # 实际应用中应使用随机salt
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password_bytes))
        return key
    
    def _initialize_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        # 创建审计事件表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                user_id TEXT NOT NULL,
                session_id TEXT,
                source_system TEXT,
                severity TEXT,
                data_type TEXT,
                event_description TEXT,
                before_state TEXT,
                after_state TEXT,
                affected_entities TEXT,
                ip_address TEXT,
                user_agent TEXT,
                transaction_id TEXT,
                correlation_id TEXT,
                risk_score REAL,
                compliance_flags TEXT,
                digital_signature TEXT,
                hash_value TEXT,
                encrypted_data TEXT,
                metadata TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 创建审计查询表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_queries (
                query_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                query_timestamp DATETIME NOT NULL,
                query_criteria TEXT,
                results_count INTEGER,
                execution_time REAL,
                metadata TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 创建审计报告表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_reports (
                report_id TEXT PRIMARY KEY,
                report_type TEXT NOT NULL,
                generation_time DATETIME NOT NULL,
                reporting_period_start DATETIME,
                reporting_period_end DATETIME,
                total_events INTEGER,
                report_data TEXT,
                digital_signature TEXT,
                metadata TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 创建完整性检查表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS integrity_checks (
                check_id TEXT PRIMARY KEY,
                check_timestamp DATETIME NOT NULL,
                check_type TEXT,
                events_checked INTEGER,
                integrity_status TEXT,
                check_results TEXT,
                recommendations TEXT,
                metadata TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 创建索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_events(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_id ON audit_events(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_event_type ON audit_events(event_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_severity ON audit_events(severity)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_transaction_id ON audit_events(transaction_id)')
        
        conn.commit()
        conn.close()
    
    def _initialize_risk_patterns(self):
        """初始化风险模式"""
        self.risk_patterns = {
            'rapid_trading': {
                'description': '快速交易模式',
                'threshold': 10,  # 10秒内多笔交易
                'time_window': 10,
                'severity': AuditSeverity.HIGH
            },
            'after_hours_activity': {
                'description': '非工作时间活动',
                'threshold': 1,
                'time_window': 3600,
                'severity': AuditSeverity.MEDIUM
            },
            'privilege_escalation': {
                'description': '权限提升',
                'threshold': 1,
                'time_window': 300,
                'severity': AuditSeverity.CRITICAL
            },
            'data_exfiltration': {
                'description': '数据泄露',
                'threshold': 5,
                'time_window': 300,
                'severity': AuditSeverity.CRITICAL
            },
            'failed_authentication': {
                'description': '认证失败',
                'threshold': 5,
                'time_window': 300,
                'severity': AuditSeverity.HIGH
            }
        }
    
    async def log_event(self, event_data: Dict[str, Any]) -> str:
        """记录审计事件"""
        # 生成事件ID
        event_id = str(uuid.uuid4())
        
        # 创建审计事件
        event = AuditEvent(
            event_id=event_id,
            event_type=AuditEventType(event_data.get('event_type', 'system_access')),
            timestamp=event_data.get('timestamp', datetime.now()),
            user_id=event_data.get('user_id', 'system'),
            session_id=event_data.get('session_id', ''),
            source_system=event_data.get('source_system', 'myquant'),
            severity=AuditSeverity(event_data.get('severity', 'medium')),
            data_type=AuditDataType(event_data.get('data_type', 'operational')),
            event_description=event_data.get('event_description', ''),
            before_state=event_data.get('before_state', {}),
            after_state=event_data.get('after_state', {}),
            affected_entities=event_data.get('affected_entities', []),
            ip_address=event_data.get('ip_address', ''),
            user_agent=event_data.get('user_agent', ''),
            transaction_id=event_data.get('transaction_id'),
            correlation_id=event_data.get('correlation_id'),
            risk_score=event_data.get('risk_score', 0.0),
            compliance_flags=event_data.get('compliance_flags', []),
            metadata=event_data.get('metadata', {})
        )
        
        # 计算风险评分
        event.risk_score = await self._calculate_risk_score(event)
        
        # 生成数字签名
        if self.audit_config.signature_required:
            event.digital_signature = self._generate_digital_signature(event)
        
        # 生成哈希值
        event.hash_value = self._generate_hash(event)
        
        # 添加到内存缓存
        self.memory_events.append(event)
        self.event_index[event_id] = event
        
        # 限制内存事件数量
        if len(self.memory_events) > self.max_memory_events:
            await self._flush_to_database()
        
        # 实时监控
        if self.audit_config.real_time_monitoring:
            await self._monitor_event(event)
        
        # 更新统计
        self._update_statistics(event)
        
        self.logger.debug(f"审计事件已记录: {event_id}")
        return event_id
    
    async def _calculate_risk_score(self, event: AuditEvent) -> float:
        """计算风险评分"""
        base_score = 0.0
        
        # 基于事件类型的基础评分
        type_scores = {
            AuditEventType.TRADE_EXECUTION: 0.3,
            AuditEventType.ORDER_PLACEMENT: 0.2,
            AuditEventType.RISK_LIMIT_BREACH: 0.8,
            AuditEventType.COMPLIANCE_VIOLATION: 0.9,
            AuditEventType.SYSTEM_ACCESS: 0.1,
            AuditEventType.DATA_MODIFICATION: 0.6,
            AuditEventType.CONFIGURATION_CHANGE: 0.7
        }
        
        base_score = type_scores.get(event.event_type, 0.3)
        
        # 基于严重程度的调整
        severity_multipliers = {
            AuditSeverity.LOW: 1.0,
            AuditSeverity.MEDIUM: 1.5,
            AuditSeverity.HIGH: 2.0,
            AuditSeverity.CRITICAL: 3.0
        }
        
        base_score *= severity_multipliers.get(event.severity, 1.0)
        
        # 基于时间的调整（非工作时间增加风险）
        if event.timestamp.hour < 8 or event.timestamp.hour > 18:
            base_score *= 1.2
        
        # 基于用户历史行为的调整
        user_risk_factor = await self._get_user_risk_factor(event.user_id)
        base_score *= user_risk_factor
        
        # 基于合规标志的调整
        if event.compliance_flags:
            base_score *= 1.5
        
        return min(1.0, base_score)
    
    async def _get_user_risk_factor(self, user_id: str) -> float:
        """获取用户风险因子"""
        # 简化版本 - 实际应用中应基于用户历史行为
        user_stats = self.monitoring_stats['events_by_user'].get(user_id, {})
        total_events = user_stats.get('total', 0)
        
        if total_events > 100:
            return 0.8  # 活跃用户风险较低
        elif total_events > 50:
            return 1.0  # 普通用户
        else:
            return 1.2  # 新用户或不活跃用户风险较高
    
    def _generate_digital_signature(self, event: AuditEvent) -> str:
        """生成数字签名"""
        # 简化版本 - 实际应用中应使用RSA或其他数字签名算法
        event_data = f"{event.event_id}{event.timestamp}{event.user_id}{event.event_description}"
        signature = hashlib.sha256(event_data.encode()).hexdigest()
        return signature
    
    def _generate_hash(self, event: AuditEvent) -> str:
        """生成哈希值"""
        event_json = json.dumps({
            'event_id': event.event_id,
            'event_type': event.event_type.value,
            'timestamp': event.timestamp.isoformat(),
            'user_id': event.user_id,
            'event_description': event.event_description,
            'before_state': event.before_state,
            'after_state': event.after_state
        }, sort_keys=True)
        
        return hashlib.sha256(event_json.encode()).hexdigest()
    
    async def _monitor_event(self, event: AuditEvent):
        """监控事件"""
        # 检查风险模式
        await self._check_risk_patterns(event)
        
        # 检查阈值
        await self._check_thresholds(event)
        
        # 检查合规性
        await self._check_compliance(event)
    
    async def _check_risk_patterns(self, event: AuditEvent):
        """检查风险模式"""
        for pattern_name, pattern_config in self.risk_patterns.items():
            if await self._matches_pattern(event, pattern_config):
                await self._create_risk_alert(event, pattern_name, pattern_config)
    
    async def _matches_pattern(self, event: AuditEvent, pattern_config: Dict[str, Any]) -> bool:
        """检查事件是否匹配风险模式"""
        # 简化版本 - 实际应用中应实现更复杂的模式匹配
        if pattern_config['description'] == '快速交易模式':
            return event.event_type == AuditEventType.TRADE_EXECUTION
        elif pattern_config['description'] == '非工作时间活动':
            return event.timestamp.hour < 8 or event.timestamp.hour > 18
        elif pattern_config['description'] == '权限提升':
            return 'privilege' in event.event_description.lower()
        
        return False
    
    async def _create_risk_alert(self, event: AuditEvent, pattern_name: str, pattern_config: Dict[str, Any]):
        """创建风险告警"""
        alert = {
            'alert_id': str(uuid.uuid4()),
            'timestamp': datetime.now(),
            'event_id': event.event_id,
            'pattern_name': pattern_name,
            'severity': pattern_config['severity'],
            'description': f"检测到风险模式: {pattern_config['description']}",
            'user_id': event.user_id,
            'ip_address': event.ip_address,
            'recommendations': self._get_risk_recommendations(pattern_name)
        }
        
        self.real_time_alerts.append(alert)
        self.monitoring_stats['risk_alerts'] += 1
        
        self.logger.warning(f"风险告警: {alert['alert_id']} - {alert['description']}")
    
    def _get_risk_recommendations(self, pattern_name: str) -> List[str]:
        """获取风险建议"""
        recommendations_map = {
            'rapid_trading': [
                "审查交易策略合理性",
                "检查是否存在算法交易异常",
                "确认交易者身份和权限"
            ],
            'after_hours_activity': [
                "验证用户身份和访问权限",
                "检查活动的业务合理性",
                "考虑限制非工作时间访问"
            ],
            'privilege_escalation': [
                "立即审查用户权限变更",
                "检查权限变更的审批流程",
                "监控后续活动"
            ],
            'data_exfiltration': [
                "立即锁定相关账户",
                "审查数据访问日志",
                "通知安全团队"
            ],
            'failed_authentication': [
                "检查是否存在暴力破解攻击",
                "考虑临时锁定账户",
                "通知用户和安全团队"
            ]
        }
        
        return recommendations_map.get(pattern_name, ["联系安全团队"])
    
    async def _check_thresholds(self, event: AuditEvent):
        """检查阈值"""
        # 检查高风险事件
        if event.risk_score >= self.audit_config.alert_thresholds['high_risk_threshold']:
            await self._create_threshold_alert(event, 'high_risk_event')
        
        # 检查关键事件频率
        if event.severity == AuditSeverity.CRITICAL:
            recent_critical_events = await self._count_recent_events(
                event_type=event.event_type,
                severity=AuditSeverity.CRITICAL,
                time_window=3600  # 1小时
            )
            
            if recent_critical_events >= self.audit_config.alert_thresholds['critical_events_per_hour']:
                await self._create_threshold_alert(event, 'critical_event_threshold')
    
    async def _count_recent_events(self, event_type: AuditEventType = None, 
                                 severity: AuditSeverity = None, 
                                 time_window: int = 3600) -> int:
        """统计最近事件数量"""
        cutoff_time = datetime.now() - timedelta(seconds=time_window)
        count = 0
        
        for event in self.memory_events:
            if event.timestamp >= cutoff_time:
                if event_type and event.event_type != event_type:
                    continue
                if severity and event.severity != severity:
                    continue
                count += 1
        
        return count
    
    async def _create_threshold_alert(self, event: AuditEvent, alert_type: str):
        """创建阈值告警"""
        alert = {
            'alert_id': str(uuid.uuid4()),
            'timestamp': datetime.now(),
            'event_id': event.event_id,
            'alert_type': alert_type,
            'severity': event.severity,
            'description': f"阈值告警: {alert_type}",
            'user_id': event.user_id,
            'event_type': event.event_type.value,
            'risk_score': event.risk_score
        }
        
        self.real_time_alerts.append(alert)
        self.logger.warning(f"阈值告警: {alert['alert_id']} - {alert['description']}")
    
    async def _check_compliance(self, event: AuditEvent):
        """检查合规性"""
        compliance_violations = []
        
        # 检查敏感数据访问
        if event.data_type == AuditDataType.SECURITY:
            if not self._has_appropriate_access(event.user_id, event.event_type):
                compliance_violations.append("未授权访问敏感数据")
        
        # 检查交易合规性
        if event.event_type == AuditEventType.TRADE_EXECUTION:
            if not self._is_compliant_trade(event):
                compliance_violations.append("交易不符合合规要求")
        
        # 检查数据完整性
        if event.event_type == AuditEventType.DATA_MODIFICATION:
            if not self._has_data_modification_approval(event):
                compliance_violations.append("数据修改缺少审批")
        
        if compliance_violations:
            await self._create_compliance_alert(event, compliance_violations)
    
    def _has_appropriate_access(self, user_id: str, event_type: AuditEventType) -> bool:
        """检查用户是否有适当的访问权限"""
        # 简化版本 - 实际应用中应查询权限管理系统
        return True  # 假设都有权限
    
    def _is_compliant_trade(self, event: AuditEvent) -> bool:
        """检查交易是否合规"""
        # 简化版本 - 实际应用中应实现复杂的合规检查
        return event.risk_score < 0.8
    
    def _has_data_modification_approval(self, event: AuditEvent) -> bool:
        """检查数据修改是否有审批"""
        # 简化版本 - 实际应用中应查询审批系统
        return 'approval_id' in event.metadata
    
    async def _create_compliance_alert(self, event: AuditEvent, violations: List[str]):
        """创建合规告警"""
        alert = {
            'alert_id': str(uuid.uuid4()),
            'timestamp': datetime.now(),
            'event_id': event.event_id,
            'alert_type': 'compliance_violation',
            'severity': AuditSeverity.HIGH,
            'violations': violations,
            'user_id': event.user_id,
            'event_type': event.event_type.value,
            'recommendations': ["立即审查合规性", "联系合规官员", "暂停相关活动"]
        }
        
        self.real_time_alerts.append(alert)
        self.monitoring_stats['compliance_violations'] += 1
        
        self.logger.error(f"合规告警: {alert['alert_id']} - {violations}")
    
    def _update_statistics(self, event: AuditEvent):
        """更新统计信息"""
        self.monitoring_stats['total_events'] += 1
        
        # 按类型统计
        event_type = event.event_type.value
        if event_type not in self.monitoring_stats['events_by_type']:
            self.monitoring_stats['events_by_type'][event_type] = 0
        self.monitoring_stats['events_by_type'][event_type] += 1
        
        # 按严重程度统计
        severity = event.severity.value
        if severity not in self.monitoring_stats['events_by_severity']:
            self.monitoring_stats['events_by_severity'][severity] = 0
        self.monitoring_stats['events_by_severity'][severity] += 1
        
        # 按用户统计
        user_id = event.user_id
        if user_id not in self.monitoring_stats['events_by_user']:
            self.monitoring_stats['events_by_user'][user_id] = {'total': 0, 'risk_events': 0}
        self.monitoring_stats['events_by_user'][user_id]['total'] += 1
        
        if event.risk_score >= 0.5:
            self.monitoring_stats['events_by_user'][user_id]['risk_events'] += 1
    
    async def _flush_to_database(self):
        """刷新到数据库"""
        if not self.memory_events:
            return
        
        # 批量插入事件
        events_to_insert = self.memory_events[:self.batch_size]
        
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        for event in events_to_insert:
            # 加密敏感数据
            encrypted_data = None
            if self.audit_config.encryption_enabled:
                encrypted_data = self._encrypt_sensitive_data(event)
            
            cursor.execute('''
                INSERT INTO audit_events (
                    event_id, event_type, timestamp, user_id, session_id, source_system,
                    severity, data_type, event_description, before_state, after_state,
                    affected_entities, ip_address, user_agent, transaction_id, correlation_id,
                    risk_score, compliance_flags, digital_signature, hash_value,
                    encrypted_data, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                event.event_id,
                event.event_type.value,
                event.timestamp.isoformat(),
                event.user_id,
                event.session_id,
                event.source_system,
                event.severity.value,
                event.data_type.value,
                event.event_description,
                json.dumps(event.before_state),
                json.dumps(event.after_state),
                json.dumps(event.affected_entities),
                event.ip_address,
                event.user_agent,
                event.transaction_id,
                event.correlation_id,
                event.risk_score,
                json.dumps(event.compliance_flags),
                event.digital_signature,
                event.hash_value,
                encrypted_data,
                json.dumps(event.metadata)
            ))
        
        conn.commit()
        conn.close()
        
        # 从内存中移除已保存的事件
        self.memory_events = self.memory_events[self.batch_size:]
        
        # 更新索引
        for event in events_to_insert:
            if event.event_id in self.event_index:
                del self.event_index[event.event_id]
        
        self.logger.debug(f"已刷新 {len(events_to_insert)} 个事件到数据库")
    
    def _encrypt_sensitive_data(self, event: AuditEvent) -> str:
        """加密敏感数据"""
        fernet = Fernet(self.encryption_key)
        
        sensitive_data = {
            'before_state': event.before_state,
            'after_state': event.after_state,
            'metadata': event.metadata
        }
        
        # 移除敏感字段
        for field in self.audit_config.sensitive_fields:
            if field in sensitive_data['before_state']:
                sensitive_data['before_state'][field] = '[ENCRYPTED]'
            if field in sensitive_data['after_state']:
                sensitive_data['after_state'][field] = '[ENCRYPTED]'
            if field in sensitive_data['metadata']:
                sensitive_data['metadata'][field] = '[ENCRYPTED]'
        
        data_json = json.dumps(sensitive_data)
        encrypted_data = fernet.encrypt(data_json.encode())
        
        return base64.b64encode(encrypted_data).decode()
    
    async def query_events(self, query: AuditQuery) -> List[AuditEvent]:
        """查询审计事件"""
        start_time = datetime.now()
        
        # 记录查询请求
        await self._log_query(query)
        
        # 构建查询条件
        where_conditions = []
        params = []
        
        # 时间范围
        where_conditions.append("timestamp BETWEEN ? AND ?")
        params.extend([query.date_range[0].isoformat(), query.date_range[1].isoformat()])
        
        # 事件类型
        if query.event_types:
            type_placeholders = ','.join(['?' for _ in query.event_types])
            where_conditions.append(f"event_type IN ({type_placeholders})")
            params.extend([et.value for et in query.event_types])
        
        # 用户过滤
        if query.user_filter:
            where_conditions.append("user_id = ?")
            params.append(query.user_filter)
        
        # 严重程度过滤
        if query.severity_filter:
            where_conditions.append("severity = ?")
            params.append(query.severity_filter.value)
        
        # 实体过滤
        if query.entity_filter:
            where_conditions.append("affected_entities LIKE ?")
            params.append(f"%{query.entity_filter}%")
        
        # 交易过滤
        if query.transaction_filter:
            where_conditions.append("transaction_id = ?")
            params.append(query.transaction_filter)
        
        # 构建SQL查询
        sql = f"""
            SELECT * FROM audit_events
            WHERE {' AND '.join(where_conditions)}
            ORDER BY timestamp DESC
            LIMIT ?
        """
        params.append(query.max_results)
        
        # 执行查询
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        cursor.execute(sql, params)
        
        results = []
        for row in cursor.fetchall():
            event = self._row_to_event(row)
            results.append(event)
        
        conn.close()
        
        # 记录查询性能
        execution_time = (datetime.now() - start_time).total_seconds()
        self.logger.info(f"查询完成: {query.query_id}, 结果数: {len(results)}, 耗时: {execution_time:.3f}s")
        
        return results
    
    def _row_to_event(self, row: tuple) -> AuditEvent:
        """将数据库行转换为审计事件"""
        return AuditEvent(
            event_id=row[0],
            event_type=AuditEventType(row[1]),
            timestamp=datetime.fromisoformat(row[2]),
            user_id=row[3],
            session_id=row[4] or '',
            source_system=row[5] or '',
            severity=AuditSeverity(row[6]),
            data_type=AuditDataType(row[7]),
            event_description=row[8] or '',
            before_state=json.loads(row[9]) if row[9] else {},
            after_state=json.loads(row[10]) if row[10] else {},
            affected_entities=json.loads(row[11]) if row[11] else [],
            ip_address=row[12] or '',
            user_agent=row[13] or '',
            transaction_id=row[14],
            correlation_id=row[15],
            risk_score=row[16] or 0.0,
            compliance_flags=json.loads(row[17]) if row[17] else [],
            digital_signature=row[18],
            hash_value=row[19],
            metadata=json.loads(row[21]) if row[21] else {}
        )
    
    async def _log_query(self, query: AuditQuery):
        """记录查询请求"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO audit_queries (
                query_id, user_id, query_timestamp, query_criteria, metadata
            ) VALUES (?, ?, ?, ?, ?)
        ''', (
            query.query_id,
            query.user_id,
            query.query_timestamp.isoformat(),
            json.dumps({
                'event_types': [et.value for et in query.event_types],
                'date_range': [query.date_range[0].isoformat(), query.date_range[1].isoformat()],
                'user_filter': query.user_filter,
                'severity_filter': query.severity_filter.value if query.severity_filter else None,
                'entity_filter': query.entity_filter,
                'transaction_filter': query.transaction_filter,
                'max_results': query.max_results
            }),
            json.dumps(query.metadata)
        ))
        
        conn.commit()
        conn.close()
    
    async def generate_audit_report(self, report_type: str, start_date: datetime, 
                                  end_date: datetime, include_details: bool = False) -> AuditReport:
        """生成审计报告"""
        report_id = f"audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 查询事件
        query = AuditQuery(
            query_id=f"report_query_{report_id}",
            user_id="system",
            query_timestamp=datetime.now(),
            event_types=list(AuditEventType),
            date_range=(start_date, end_date),
            max_results=10000
        )
        
        events = await self.query_events(query)
        
        # 生成统计信息
        event_summary = {}
        severity_breakdown = {}
        user_activity = {}
        
        for event in events:
            # 事件类型统计
            event_type = event.event_type.value
            event_summary[event_type] = event_summary.get(event_type, 0) + 1
            
            # 严重程度统计
            severity = event.severity.value
            severity_breakdown[severity] = severity_breakdown.get(severity, 0) + 1
            
            # 用户活动统计
            user_id = event.user_id
            if user_id not in user_activity:
                user_activity[user_id] = {'total': 0, 'risk_events': 0}
            user_activity[user_id]['total'] += 1
            
            if event.risk_score >= 0.5:
                user_activity[user_id]['risk_events'] += 1
        
        # 分析合规问题
        compliance_issues = await self._analyze_compliance_issues(events)
        
        # 风险分析
        risk_analysis = await self._analyze_risks(events)
        
        # 生成建议
        recommendations = await self._generate_audit_recommendations(events, compliance_issues, risk_analysis)
        
        # 创建报告
        report = AuditReport(
            report_id=report_id,
            report_type=report_type,
            generation_time=datetime.now(),
            reporting_period=(start_date, end_date),
            total_events=len(events),
            event_summary=event_summary,
            severity_breakdown=severity_breakdown,
            user_activity=user_activity,
            compliance_issues=compliance_issues,
            risk_analysis=risk_analysis,
            recommendations=recommendations,
            detailed_events=events if include_details else []
        )
        
        # 生成数字签名
        if self.audit_config.signature_required:
            report.digital_signature = self._generate_report_signature(report)
        
        # 保存报告
        await self._save_audit_report(report)
        
        self.logger.info(f"审计报告已生成: {report_id}")
        return report
    
    async def _analyze_compliance_issues(self, events: List[AuditEvent]) -> List[Dict[str, Any]]:
        """分析合规问题"""
        compliance_issues = []
        
        # 分析未授权访问
        unauthorized_events = [e for e in events if 'unauthorized' in e.compliance_flags]
        if unauthorized_events:
            compliance_issues.append({
                'issue_type': 'unauthorized_access',
                'count': len(unauthorized_events),
                'severity': 'high',
                'description': f'发现 {len(unauthorized_events)} 次未授权访问',
                'affected_users': list(set(e.user_id for e in unauthorized_events))
            })
        
        # 分析数据完整性问题
        integrity_events = [e for e in events if e.event_type == AuditEventType.DATA_MODIFICATION 
                          and 'integrity_violation' in e.compliance_flags]
        if integrity_events:
            compliance_issues.append({
                'issue_type': 'data_integrity',
                'count': len(integrity_events),
                'severity': 'medium',
                'description': f'发现 {len(integrity_events)} 次数据完整性问题',
                'affected_entities': list(set(entity for e in integrity_events for entity in e.affected_entities))
            })
        
        # 分析交易合规问题
        trade_compliance_events = [e for e in events if e.event_type == AuditEventType.TRADE_EXECUTION 
                                 and e.risk_score >= 0.8]
        if trade_compliance_events:
            compliance_issues.append({
                'issue_type': 'trade_compliance',
                'count': len(trade_compliance_events),
                'severity': 'high',
                'description': f'发现 {len(trade_compliance_events)} 次高风险交易',
                'affected_transactions': list(set(e.transaction_id for e in trade_compliance_events if e.transaction_id))
            })
        
        return compliance_issues
    
    async def _analyze_risks(self, events: List[AuditEvent]) -> Dict[str, Any]:
        """分析风险"""
        risk_analysis = {
            'overall_risk_level': 'low',
            'risk_categories': {},
            'high_risk_users': [],
            'risk_trends': {},
            'mitigation_priorities': []
        }
        
        # 计算整体风险水平
        high_risk_events = [e for e in events if e.risk_score >= 0.7]
        if len(high_risk_events) > len(events) * 0.1:
            risk_analysis['overall_risk_level'] = 'high'
        elif len(high_risk_events) > len(events) * 0.05:
            risk_analysis['overall_risk_level'] = 'medium'
        
        # 分析风险类别
        for event in events:
            category = event.event_type.value
            if category not in risk_analysis['risk_categories']:
                risk_analysis['risk_categories'][category] = {
                    'total_events': 0,
                    'high_risk_events': 0,
                    'average_risk_score': 0.0
                }
            
            risk_analysis['risk_categories'][category]['total_events'] += 1
            if event.risk_score >= 0.7:
                risk_analysis['risk_categories'][category]['high_risk_events'] += 1
        
        # 计算平均风险评分
        for category, data in risk_analysis['risk_categories'].items():
            category_events = [e for e in events if e.event_type.value == category]
            if category_events:
                data['average_risk_score'] = sum(e.risk_score for e in category_events) / len(category_events)
        
        # 识别高风险用户
        user_risk_scores = {}
        for event in events:
            if event.user_id not in user_risk_scores:
                user_risk_scores[event.user_id] = []
            user_risk_scores[event.user_id].append(event.risk_score)
        
        for user_id, scores in user_risk_scores.items():
            avg_score = sum(scores) / len(scores)
            if avg_score >= 0.6:
                risk_analysis['high_risk_users'].append({
                    'user_id': user_id,
                    'average_risk_score': avg_score,
                    'total_events': len(scores),
                    'high_risk_events': len([s for s in scores if s >= 0.7])
                })
        
        # 缓解优先级
        if risk_analysis['overall_risk_level'] == 'high':
            risk_analysis['mitigation_priorities'] = [
                '立即审查高风险用户活动',
                '加强访问控制',
                '实施额外监控措施',
                '进行安全培训'
            ]
        
        return risk_analysis
    
    async def _generate_audit_recommendations(self, events: List[AuditEvent], 
                                            compliance_issues: List[Dict[str, Any]], 
                                            risk_analysis: Dict[str, Any]) -> List[str]:
        """生成审计建议"""
        recommendations = []
        
        # 基于合规问题的建议
        if compliance_issues:
            recommendations.append("建议立即处理发现的合规问题")
            
            for issue in compliance_issues:
                if issue['issue_type'] == 'unauthorized_access':
                    recommendations.append("加强访问控制和身份验证")
                elif issue['issue_type'] == 'data_integrity':
                    recommendations.append("实施数据完整性检查机制")
                elif issue['issue_type'] == 'trade_compliance':
                    recommendations.append("审查交易合规政策和限制")
        
        # 基于风险分析的建议
        if risk_analysis['overall_risk_level'] == 'high':
            recommendations.append("建议实施更严格的风险控制措施")
        
        if risk_analysis['high_risk_users']:
            recommendations.append("建议对高风险用户进行额外审查")
        
        # 基于事件类型的建议
        event_types = set(e.event_type for e in events)
        if AuditEventType.COMPLIANCE_VIOLATION in event_types:
            recommendations.append("建议加强合规培训和监控")
        
        if AuditEventType.SYSTEM_ACCESS in event_types:
            recommendations.append("建议实施多因素认证")
        
        # 通用建议
        recommendations.extend([
            "定期审查和更新审计策略",
            "加强员工安全意识培训",
            "实施持续监控和告警机制",
            "建立事件响应和处理流程"
        ])
        
        return recommendations
    
    def _generate_report_signature(self, report: AuditReport) -> str:
        """生成报告签名"""
        report_data = f"{report.report_id}{report.generation_time}{report.total_events}"
        signature = hashlib.sha256(report_data.encode()).hexdigest()
        return signature
    
    async def _save_audit_report(self, report: AuditReport):
        """保存审计报告"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO audit_reports (
                report_id, report_type, generation_time, reporting_period_start,
                reporting_period_end, total_events, report_data, digital_signature, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            report.report_id,
            report.report_type,
            report.generation_time.isoformat(),
            report.reporting_period[0].isoformat(),
            report.reporting_period[1].isoformat(),
            report.total_events,
            json.dumps({
                'event_summary': report.event_summary,
                'severity_breakdown': report.severity_breakdown,
                'user_activity': report.user_activity,
                'compliance_issues': report.compliance_issues,
                'risk_analysis': report.risk_analysis,
                'recommendations': report.recommendations
            }),
            report.digital_signature,
            json.dumps(report.metadata)
        ))
        
        conn.commit()
        conn.close()
    
    async def verify_integrity(self, start_date: datetime, end_date: datetime) -> AuditIntegrityCheck:
        """验证审计完整性"""
        check_id = f"integrity_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 查询事件
        query = AuditQuery(
            query_id=f"integrity_query_{check_id}",
            user_id="system",
            query_timestamp=datetime.now(),
            event_types=list(AuditEventType),
            date_range=(start_date, end_date),
            max_results=50000
        )
        
        events = await self.query_events(query)
        
        # 验证哈希值
        hash_mismatches = []
        for event in events:
            if event.hash_value:
                calculated_hash = self._generate_hash(event)
                if calculated_hash != event.hash_value:
                    hash_mismatches.append(event.event_id)
        
        # 检查缺失事件（简化版本）
        missing_events = []
        
        # 检查损坏事件
        corrupted_events = []
        for event in events:
            if not event.event_id or not event.timestamp or not event.user_id:
                corrupted_events.append(event.event_id)
        
        # 确定完整性状态
        if hash_mismatches or missing_events or corrupted_events:
            integrity_status = "compromised"
        else:
            integrity_status = "intact"
        
        # 生成建议
        recommendations = []
        if hash_mismatches:
            recommendations.append("发现哈希值不匹配，建议调查数据篡改")
        if missing_events:
            recommendations.append("发现缺失事件，建议检查数据完整性")
        if corrupted_events:
            recommendations.append("发现损坏事件，建议数据恢复")
        
        if integrity_status == "intact":
            recommendations.append("审计数据完整性良好")
        
        # 创建完整性检查结果
        check_result = AuditIntegrityCheck(
            check_id=check_id,
            check_timestamp=datetime.now(),
            check_type="full_integrity_check",
            events_checked=len(events),
            integrity_status=integrity_status,
            hash_mismatches=hash_mismatches,
            missing_events=missing_events,
            corrupted_events=corrupted_events,
            recommendations=recommendations
        )
        
        # 保存检查结果
        await self._save_integrity_check(check_result)
        
        # 更新统计
        self.monitoring_stats['integrity_checks'] += 1
        
        self.logger.info(f"完整性检查完成: {check_id}, 状态: {integrity_status}")
        return check_result
    
    async def _save_integrity_check(self, check_result: AuditIntegrityCheck):
        """保存完整性检查结果"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO integrity_checks (
                check_id, check_timestamp, check_type, events_checked, integrity_status,
                check_results, recommendations, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            check_result.check_id,
            check_result.check_timestamp.isoformat(),
            check_result.check_type,
            check_result.events_checked,
            check_result.integrity_status,
            json.dumps({
                'hash_mismatches': check_result.hash_mismatches,
                'missing_events': check_result.missing_events,
                'corrupted_events': check_result.corrupted_events
            }),
            json.dumps(check_result.recommendations),
            json.dumps(check_result.metadata)
        ))
        
        conn.commit()
        conn.close()
    
    async def get_real_time_alerts(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取实时告警"""
        return self.real_time_alerts[-limit:] if self.real_time_alerts else []
    
    async def get_monitoring_statistics(self) -> Dict[str, Any]:
        """获取监控统计"""
        return {
            'total_events': self.monitoring_stats['total_events'],
            'events_by_type': self.monitoring_stats['events_by_type'],
            'events_by_severity': self.monitoring_stats['events_by_severity'],
            'events_by_user': self.monitoring_stats['events_by_user'],
            'integrity_checks': self.monitoring_stats['integrity_checks'],
            'compliance_violations': self.monitoring_stats['compliance_violations'],
            'risk_alerts': self.monitoring_stats['risk_alerts'],
            'memory_events': len(self.memory_events),
            'real_time_alerts': len(self.real_time_alerts),
            'database_size': self._get_database_size()
        }
    
    def _get_database_size(self) -> int:
        """获取数据库大小"""
        try:
            db_path = Path(self.database_path)
            return db_path.stat().st_size if db_path.exists() else 0
        except:
            return 0
    
    async def archive_old_events(self, days_to_keep: int = 2555):
        """归档旧事件"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        # 统计要归档的事件
        cursor.execute('SELECT COUNT(*) FROM audit_events WHERE timestamp < ?', (cutoff_date.isoformat(),))
        events_to_archive = cursor.fetchone()[0]
        
        if events_to_archive > 0:
            # 创建归档表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS audit_events_archive AS 
                SELECT * FROM audit_events WHERE 1=0
            ''')
            
            # 移动到归档表
            cursor.execute('''
                INSERT INTO audit_events_archive 
                SELECT * FROM audit_events WHERE timestamp < ?
            ''', (cutoff_date.isoformat(),))
            
            # 删除原表中的旧数据
            cursor.execute('DELETE FROM audit_events WHERE timestamp < ?', (cutoff_date.isoformat(),))
            
            conn.commit()
            self.logger.info(f"已归档 {events_to_archive} 个审计事件")
        
        conn.close()
        return events_to_archive
    
    async def export_audit_data(self, start_date: datetime, end_date: datetime, 
                              format: str = "csv") -> str:
        """导出审计数据"""
        # 查询事件
        query = AuditQuery(
            query_id=f"export_query_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            user_id="system",
            query_timestamp=datetime.now(),
            event_types=list(AuditEventType),
            date_range=(start_date, end_date),
            max_results=100000
        )
        
        events = await self.query_events(query)
        
        # 导出数据
        if format.lower() == "csv":
            data = []
            for event in events:
                data.append({
                    'Event ID': event.event_id,
                    'Event Type': event.event_type.value,
                    'Timestamp': event.timestamp.isoformat(),
                    'User ID': event.user_id,
                    'Severity': event.severity.value,
                    'Description': event.event_description,
                    'Risk Score': event.risk_score,
                    'IP Address': event.ip_address,
                    'Transaction ID': event.transaction_id or '',
                    'Compliance Flags': ', '.join(event.compliance_flags)
                })
            
            df = pd.DataFrame(data)
            filename = f"audit_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(filename, index=False)
            
            self.logger.info(f"审计数据已导出: {filename}")
            return filename
        
        return ""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import threading
import time
import json
from pathlib import Path
from collections import deque, defaultdict
import subprocess
import sys
import psutil
import warnings
warnings.filterwarnings('ignore')

class FaultType(Enum):
    SYSTEM_CRASH = "system_crash"
    SERVICE_FAILURE = "service_failure"
    NETWORK_FAILURE = "network_failure"
    DATABASE_FAILURE = "database_failure"
    MEMORY_EXHAUSTION = "memory_exhaustion"
    DISK_FULL = "disk_full"
    CPU_OVERLOAD = "cpu_overload"
    TRADING_ENGINE_FAILURE = "trading_engine_failure"
    DATA_FEED_FAILURE = "data_feed_failure"
    CONNECTIVITY_LOSS = "connectivity_loss"
    AUTHENTICATION_FAILURE = "authentication_failure"
    CONFIGURATION_ERROR = "configuration_error"
    DEPENDENCY_FAILURE = "dependency_failure"
    CUSTOM_FAULT = "custom_fault"

class FaultSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RecoveryStrategy(Enum):
    RESTART_SERVICE = "restart_service"
    RESTART_SYSTEM = "restart_system"
    FAILOVER = "failover"
    ROLLBACK = "rollback"
    SCALE_UP = "scale_up"
    CIRCUIT_BREAKER = "circuit_breaker"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"
    CUSTOM_RECOVERY = "custom_recovery"

class RecoveryStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ComponentStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    RECOVERING = "recovering"
    UNKNOWN = "unknown"

@dataclass
class FaultEvent:
    """故障事件"""
    fault_id: str
    timestamp: datetime
    fault_type: FaultType
    severity: FaultSeverity
    component: str
    description: str
    error_details: Dict[str, Any]
    impact_assessment: str
    affected_services: List[str]
    root_cause: Optional[str] = None
    detected_by: str = "system"
    acknowledged: bool = False
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RecoveryPlan:
    """恢复计划"""
    plan_id: str
    fault_id: str
    strategy: RecoveryStrategy
    priority: int
    estimated_duration: int  # seconds
    prerequisites: List[str]
    recovery_steps: List[Dict[str, Any]]
    rollback_steps: List[Dict[str, Any]]
    success_criteria: List[str]
    risk_assessment: str
    automated: bool = True
    requires_approval: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RecoveryExecution:
    """恢复执行"""
    execution_id: str
    plan_id: str
    fault_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: RecoveryStatus = RecoveryStatus.PENDING
    current_step: int = 0
    total_steps: int = 0
    progress_percentage: float = 0.0
    executed_steps: List[Dict[str, Any]] = field(default_factory=list)
    error_message: Optional[str] = None
    success: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ComponentHealth:
    """组件健康状态"""
    component_id: str
    component_name: str
    status: ComponentStatus
    last_check: datetime
    uptime: float
    error_count: int
    last_error: Optional[str] = None
    health_score: float = 1.0
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class FaultRecoverySystem:
    """故障自动恢复系统"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 系统状态
        self.is_running = False
        self.monitoring_thread = None
        self.recovery_thread = None
        self.stop_event = threading.Event()
        
        # 故障管理
        self.fault_events = deque(maxlen=config.get('max_fault_events', 1000))
        self.active_faults: Dict[str, FaultEvent] = {}
        self.recovery_plans: Dict[str, RecoveryPlan] = {}
        self.recovery_executions = deque(maxlen=config.get('max_executions', 500))
        
        # 组件监控
        self.components: Dict[str, ComponentHealth] = {}
        self.health_checkers: Dict[str, Callable] = {}
        
        # 恢复策略
        self.recovery_strategies: Dict[FaultType, List[RecoveryStrategy]] = {}
        self.custom_recovery_handlers: Dict[str, Callable] = {}
        
        # 回调函数
        self.fault_callbacks: List[Callable] = []
        self.recovery_callbacks: List[Callable] = []
        
        # 初始化
        self._initialize_components()
        self._initialize_recovery_strategies()
        self._load_recovery_plans()
    
    def _initialize_components(self):
        """初始化组件"""
        default_components = [
            {
                'component_id': 'trading_engine',
                'component_name': 'Trading Engine',
                'dependencies': ['data_feed', 'database', 'risk_manager']
            },
            {
                'component_id': 'data_feed',
                'component_name': 'Market Data Feed',
                'dependencies': ['network', 'external_api']
            },
            {
                'component_id': 'risk_manager',
                'component_name': 'Risk Manager',
                'dependencies': ['database', 'portfolio_manager']
            },
            {
                'component_id': 'portfolio_manager',
                'component_name': 'Portfolio Manager',
                'dependencies': ['database', 'trading_engine']
            },
            {
                'component_id': 'database',
                'component_name': 'Database',
                'dependencies': ['disk_storage']
            },
            {
                'component_id': 'network',
                'component_name': 'Network',
                'dependencies': []
            },
            {
                'component_id': 'external_api',
                'component_name': 'External API',
                'dependencies': ['network']
            },
            {
                'component_id': 'disk_storage',
                'component_name': 'Disk Storage',
                'dependencies': []
            }
        ]
        
        # 创建组件
        for comp_config in default_components:
            component = ComponentHealth(
                component_id=comp_config['component_id'],
                component_name=comp_config['component_name'],
                status=ComponentStatus.UNKNOWN,
                last_check=datetime.now(),
                uptime=0.0,
                error_count=0,
                dependencies=comp_config.get('dependencies', [])
            )
            self.components[comp_config['component_id']] = component
        
        # 加载用户自定义组件
        custom_components = self.config.get('custom_components', [])
        for comp_config in custom_components:
            self._add_component(comp_config)
    
    def _add_component(self, config: Dict[str, Any]):
        """添加组件"""
        component = ComponentHealth(
            component_id=config['component_id'],
            component_name=config['component_name'],
            status=ComponentStatus.UNKNOWN,
            last_check=datetime.now(),
            uptime=0.0,
            error_count=0,
            dependencies=config.get('dependencies', [])
        )
        self.components[config['component_id']] = component
    
    def _initialize_recovery_strategies(self):
        """初始化恢复策略"""
        default_strategies = {
            FaultType.SYSTEM_CRASH: [
                RecoveryStrategy.RESTART_SYSTEM,
                RecoveryStrategy.FAILOVER,
                RecoveryStrategy.EMERGENCY_SHUTDOWN
            ],
            FaultType.SERVICE_FAILURE: [
                RecoveryStrategy.RESTART_SERVICE,
                RecoveryStrategy.FAILOVER,
                RecoveryStrategy.ROLLBACK
            ],
            FaultType.NETWORK_FAILURE: [
                RecoveryStrategy.FAILOVER,
                RecoveryStrategy.CIRCUIT_BREAKER,
                RecoveryStrategy.GRACEFUL_DEGRADATION
            ],
            FaultType.DATABASE_FAILURE: [
                RecoveryStrategy.RESTART_SERVICE,
                RecoveryStrategy.FAILOVER,
                RecoveryStrategy.ROLLBACK
            ],
            FaultType.MEMORY_EXHAUSTION: [
                RecoveryStrategy.RESTART_SERVICE,
                RecoveryStrategy.SCALE_UP,
                RecoveryStrategy.GRACEFUL_DEGRADATION
            ],
            FaultType.DISK_FULL: [
                RecoveryStrategy.SCALE_UP,
                RecoveryStrategy.GRACEFUL_DEGRADATION,
                RecoveryStrategy.EMERGENCY_SHUTDOWN
            ],
            FaultType.CPU_OVERLOAD: [
                RecoveryStrategy.SCALE_UP,
                RecoveryStrategy.GRACEFUL_DEGRADATION,
                RecoveryStrategy.RESTART_SERVICE
            ],
            FaultType.TRADING_ENGINE_FAILURE: [
                RecoveryStrategy.RESTART_SERVICE,
                RecoveryStrategy.FAILOVER,
                RecoveryStrategy.EMERGENCY_SHUTDOWN
            ],
            FaultType.DATA_FEED_FAILURE: [
                RecoveryStrategy.RESTART_SERVICE,
                RecoveryStrategy.FAILOVER,
                RecoveryStrategy.CIRCUIT_BREAKER
            ],
            FaultType.CONNECTIVITY_LOSS: [
                RecoveryStrategy.FAILOVER,
                RecoveryStrategy.CIRCUIT_BREAKER,
                RecoveryStrategy.GRACEFUL_DEGRADATION
            ]
        }
        
        # 加载用户自定义策略
        custom_strategies = self.config.get('recovery_strategies', {})
        
        for fault_type, strategies in default_strategies.items():
            if fault_type.value in custom_strategies:
                # 合并用户自定义策略
                user_strategies = [RecoveryStrategy(s) for s in custom_strategies[fault_type.value]]
                self.recovery_strategies[fault_type] = user_strategies + strategies
            else:
                self.recovery_strategies[fault_type] = strategies
    
    def _load_recovery_plans(self):
        """加载恢复计划"""
        plans_file = Path(self.config.get('recovery_plans_file', 'recovery_plans.json'))
        
        if plans_file.exists():
            try:
                with open(plans_file, 'r') as f:
                    plans_data = json.load(f)
                
                for plan_id, plan_data in plans_data.items():
                    plan = RecoveryPlan(**plan_data)
                    self.recovery_plans[plan_id] = plan
                
                self.logger.info(f"Loaded {len(self.recovery_plans)} recovery plans")
                
            except Exception as e:
                self.logger.error(f"Error loading recovery plans: {e}")
    
    def start(self):
        """启动故障恢复系统"""
        if self.is_running:
            return
        
        self.is_running = True
        self.stop_event.clear()
        
        # 启动监控线程
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        # 启动恢复线程
        self.recovery_thread = threading.Thread(target=self._recovery_loop, daemon=True)
        self.recovery_thread.start()
        
        self.logger.info("Fault recovery system started")
    
    def stop(self):
        """停止故障恢复系统"""
        if not self.is_running:
            return
        
        self.logger.info("Stopping fault recovery system...")
        self.stop_event.set()
        
        # 停止线程
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        if self.recovery_thread:
            self.recovery_thread.join(timeout=5)
        
        self.is_running = False
        self.logger.info("Fault recovery system stopped")
    
    def _monitoring_loop(self):
        """监控主循环"""
        while not self.stop_event.is_set():
            try:
                # 检查所有组件健康状态
                for component_id, component in self.components.items():
                    self._check_component_health(component_id)
                
                # 检查系统级别故障
                self._check_system_level_faults()
                
                time.sleep(self.config.get('monitoring_interval', 5))
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1)
    
    def _recovery_loop(self):
        """恢复主循环"""
        while not self.stop_event.is_set():
            try:
                # 处理活跃故障
                for fault_id, fault in list(self.active_faults.items()):
                    if not fault.resolved:
                        self._handle_fault_recovery(fault)
                
                time.sleep(self.config.get('recovery_interval', 2))
                
            except Exception as e:
                self.logger.error(f"Error in recovery loop: {e}")
                time.sleep(1)
    
    def _check_component_health(self, component_id: str):
        """检查组件健康状态"""
        try:
            component = self.components[component_id]
            
            # 使用自定义健康检查器
            if component_id in self.health_checkers:
                is_healthy = self.health_checkers[component_id]()
            else:
                is_healthy = self._default_health_check(component_id)
            
            # 更新组件状态
            previous_status = component.status
            
            if is_healthy:
                component.status = ComponentStatus.HEALTHY
                component.health_score = min(1.0, component.health_score + 0.1)
                component.error_count = 0
            else:
                component.status = ComponentStatus.FAILED
                component.health_score = max(0.0, component.health_score - 0.2)
                component.error_count += 1
                
                # 如果状态从健康变为失败，触发故障事件
                if previous_status == ComponentStatus.HEALTHY:
                    self._trigger_fault_event(component_id, FaultType.SERVICE_FAILURE)
            
            component.last_check = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error checking component {component_id}: {e}")
            self.components[component_id].status = ComponentStatus.UNKNOWN
    
    def _default_health_check(self, component_id: str) -> bool:
        """默认健康检查"""
        # 简化的健康检查逻辑
        if component_id in ['trading_engine', 'data_feed', 'risk_manager']:
            # 模拟服务健康检查
            return np.random.random() > 0.05  # 95%的概率健康
        elif component_id == 'database':
            # 模拟数据库健康检查
            return np.random.random() > 0.02  # 98%的概率健康
        elif component_id == 'network':
            # 模拟网络健康检查
            return np.random.random() > 0.03  # 97%的概率健康
        else:
            return True
    
    def _check_system_level_faults(self):
        """检查系统级别故障"""
        try:
            # 检查CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 95:
                self._trigger_fault_event('system', FaultType.CPU_OVERLOAD)
            
            # 检查内存使用率
            memory = psutil.virtual_memory()
            if memory.percent > 95:
                self._trigger_fault_event('system', FaultType.MEMORY_EXHAUSTION)
            
            # 检查磁盘使用率
            disk = psutil.disk_usage('/')
            if (disk.used / disk.total) > 0.95:
                self._trigger_fault_event('system', FaultType.DISK_FULL)
            
        except Exception as e:
            self.logger.error(f"Error checking system faults: {e}")
    
    def _trigger_fault_event(self, component: str, fault_type: FaultType, 
                            description: str = None, error_details: Dict[str, Any] = None):
        """触发故障事件"""
        fault_id = f"{component}_{fault_type.value}_{int(time.time())}"
        
        # 检查是否已经存在相同的活跃故障
        for existing_fault in self.active_faults.values():
            if (existing_fault.component == component and 
                existing_fault.fault_type == fault_type and 
                not existing_fault.resolved):
                return  # 不重复创建相同故障
        
        # 确定严重程度
        severity = self._determine_fault_severity(fault_type, component)
        
        # 评估影响
        affected_services = self._assess_impact(component, fault_type)
        impact_assessment = self._generate_impact_assessment(affected_services, severity)
        
        fault = FaultEvent(
            fault_id=fault_id,
            timestamp=datetime.now(),
            fault_type=fault_type,
            severity=severity,
            component=component,
            description=description or f"{fault_type.value} in {component}",
            error_details=error_details or {},
            impact_assessment=impact_assessment,
            affected_services=affected_services
        )
        
        # 记录故障
        self.fault_events.append(fault)
        self.active_faults[fault_id] = fault
        
        # 通知回调
        for callback in self.fault_callbacks:
            try:
                callback(fault)
            except Exception as e:
                self.logger.error(f"Error in fault callback: {e}")
        
        self.logger.error(f"Fault detected: {fault.description}")
    
    def _determine_fault_severity(self, fault_type: FaultType, component: str) -> FaultSeverity:
        """确定故障严重程度"""
        critical_faults = [
            FaultType.SYSTEM_CRASH,
            FaultType.TRADING_ENGINE_FAILURE,
            FaultType.DATABASE_FAILURE
        ]
        
        high_faults = [
            FaultType.MEMORY_EXHAUSTION,
            FaultType.DISK_FULL,
            FaultType.DATA_FEED_FAILURE
        ]
        
        if fault_type in critical_faults:
            return FaultSeverity.CRITICAL
        elif fault_type in high_faults:
            return FaultSeverity.HIGH
        elif component in ['trading_engine', 'risk_manager']:
            return FaultSeverity.HIGH
        else:
            return FaultSeverity.MEDIUM
    
    def _assess_impact(self, component: str, fault_type: FaultType) -> List[str]:
        """评估影响范围"""
        affected_services = [component]
        
        # 递归查找依赖于该组件的服务
        def find_dependents(comp_id: str):
            dependents = []
            for comp_id_iter, comp in self.components.items():
                if comp_id in comp.dependencies:
                    dependents.append(comp_id_iter)
                    dependents.extend(find_dependents(comp_id_iter))
            return dependents
        
        affected_services.extend(find_dependents(component))
        
        return list(set(affected_services))
    
    def _generate_impact_assessment(self, affected_services: List[str], severity: FaultSeverity) -> str:
        """生成影响评估"""
        if severity == FaultSeverity.CRITICAL:
            return f"Critical system failure affecting {len(affected_services)} services"
        elif severity == FaultSeverity.HIGH:
            return f"High impact failure affecting {len(affected_services)} services"
        else:
            return f"Medium impact failure affecting {len(affected_services)} services"
    
    def _handle_fault_recovery(self, fault: FaultEvent):
        """处理故障恢复"""
        try:
            # 检查是否有正在执行的恢复计划
            active_executions = [e for e in self.recovery_executions 
                               if e.fault_id == fault.fault_id and e.status == RecoveryStatus.IN_PROGRESS]
            
            if active_executions:
                return  # 已经在恢复中
            
            # 选择恢复策略
            recovery_strategy = self._select_recovery_strategy(fault)
            
            if recovery_strategy:
                # 创建恢复计划
                recovery_plan = self._create_recovery_plan(fault, recovery_strategy)
                
                # 执行恢复
                self._execute_recovery_plan(recovery_plan)
                
        except Exception as e:
            self.logger.error(f"Error handling fault recovery: {e}")
    
    def _select_recovery_strategy(self, fault: FaultEvent) -> Optional[RecoveryStrategy]:
        """选择恢复策略"""
        strategies = self.recovery_strategies.get(fault.fault_type, [])
        
        if not strategies:
            return None
        
        # 根据严重程度和组件类型选择策略
        if fault.severity == FaultSeverity.CRITICAL:
            # 关键故障使用最激进的策略
            return strategies[0]
        elif fault.severity == FaultSeverity.HIGH:
            # 高影响故障优先使用重启策略
            restart_strategies = [s for s in strategies if 'restart' in s.value.lower()]
            return restart_strategies[0] if restart_strategies else strategies[0]
        else:
            # 其他故障使用温和策略
            gentle_strategies = [s for s in strategies if s in [
                RecoveryStrategy.GRACEFUL_DEGRADATION,
                RecoveryStrategy.CIRCUIT_BREAKER,
                RecoveryStrategy.RESTART_SERVICE
            ]]
            return gentle_strategies[0] if gentle_strategies else strategies[0]
    
    def _create_recovery_plan(self, fault: FaultEvent, strategy: RecoveryStrategy) -> RecoveryPlan:
        """创建恢复计划"""
        plan_id = f"recovery_{fault.fault_id}_{strategy.value}"
        
        # 根据策略生成恢复步骤
        recovery_steps = self._generate_recovery_steps(fault, strategy)
        rollback_steps = self._generate_rollback_steps(fault, strategy)
        
        plan = RecoveryPlan(
            plan_id=plan_id,
            fault_id=fault.fault_id,
            strategy=strategy,
            priority=self._get_strategy_priority(strategy),
            estimated_duration=self._estimate_recovery_duration(strategy),
            prerequisites=self._get_recovery_prerequisites(fault, strategy),
            recovery_steps=recovery_steps,
            rollback_steps=rollback_steps,
            success_criteria=self._get_success_criteria(fault, strategy),
            risk_assessment=self._assess_recovery_risk(strategy),
            automated=self._is_strategy_automated(strategy),
            requires_approval=self._requires_approval(fault, strategy)
        )
        
        self.recovery_plans[plan_id] = plan
        return plan
    
    def _generate_recovery_steps(self, fault: FaultEvent, strategy: RecoveryStrategy) -> List[Dict[str, Any]]:
        """生成恢复步骤"""
        steps = []
        
        if strategy == RecoveryStrategy.RESTART_SERVICE:
            steps = [
                {
                    'step': 1,
                    'action': 'stop_service',
                    'component': fault.component,
                    'description': f'Stop {fault.component} service',
                    'timeout': 30
                },
                {
                    'step': 2,
                    'action': 'wait',
                    'duration': 5,
                    'description': 'Wait for service to fully stop'
                },
                {
                    'step': 3,
                    'action': 'start_service',
                    'component': fault.component,
                    'description': f'Start {fault.component} service',
                    'timeout': 60
                },
                {
                    'step': 4,
                    'action': 'verify_health',
                    'component': fault.component,
                    'description': f'Verify {fault.component} is healthy',
                    'timeout': 30
                }
            ]
        
        elif strategy == RecoveryStrategy.FAILOVER:
            steps = [
                {
                    'step': 1,
                    'action': 'identify_backup',
                    'component': fault.component,
                    'description': f'Identify backup for {fault.component}',
                    'timeout': 10
                },
                {
                    'step': 2,
                    'action': 'redirect_traffic',
                    'component': fault.component,
                    'description': f'Redirect traffic from {fault.component} to backup',
                    'timeout': 30
                },
                {
                    'step': 3,
                    'action': 'verify_failover',
                    'component': fault.component,
                    'description': 'Verify failover is successful',
                    'timeout': 60
                }
            ]
        
        elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
            steps = [
                {
                    'step': 1,
                    'action': 'enable_degraded_mode',
                    'component': fault.component,
                    'description': f'Enable degraded mode for {fault.component}',
                    'timeout': 30
                },
                {
                    'step': 2,
                    'action': 'reduce_load',
                    'component': fault.component,
                    'description': 'Reduce system load',
                    'timeout': 60
                },
                {
                    'step': 3,
                    'action': 'monitor_performance',
                    'component': fault.component,
                    'description': 'Monitor system performance',
                    'timeout': 120
                }
            ]
        
        elif strategy == RecoveryStrategy.CIRCUIT_BREAKER:
            steps = [
                {
                    'step': 1,
                    'action': 'activate_circuit_breaker',
                    'component': fault.component,
                    'description': f'Activate circuit breaker for {fault.component}',
                    'timeout': 10
                },
                {
                    'step': 2,
                    'action': 'monitor_health',
                    'component': fault.component,
                    'description': 'Monitor component health',
                    'timeout': 300
                },
                {
                    'step': 3,
                    'action': 'attempt_recovery',
                    'component': fault.component,
                    'description': 'Attempt to recover component',
                    'timeout': 60
                }
            ]
        
        else:
            # 默认恢复步骤
            steps = [
                {
                    'step': 1,
                    'action': 'diagnose_issue',
                    'component': fault.component,
                    'description': f'Diagnose issue with {fault.component}',
                    'timeout': 30
                },
                {
                    'step': 2,
                    'action': 'apply_fix',
                    'component': fault.component,
                    'description': f'Apply fix to {fault.component}',
                    'timeout': 120
                },
                {
                    'step': 3,
                    'action': 'verify_fix',
                    'component': fault.component,
                    'description': f'Verify fix for {fault.component}',
                    'timeout': 60
                }
            ]
        
        return steps
    
    def _generate_rollback_steps(self, fault: FaultEvent, strategy: RecoveryStrategy) -> List[Dict[str, Any]]:
        """生成回滚步骤"""
        # 简化的回滚步骤
        return [
            {
                'step': 1,
                'action': 'restore_previous_state',
                'component': fault.component,
                'description': f'Restore {fault.component} to previous state',
                'timeout': 120
            }
        ]
    
    def _get_strategy_priority(self, strategy: RecoveryStrategy) -> int:
        """获取策略优先级"""
        priority_map = {
            RecoveryStrategy.EMERGENCY_SHUTDOWN: 1,
            RecoveryStrategy.FAILOVER: 2,
            RecoveryStrategy.RESTART_SERVICE: 3,
            RecoveryStrategy.CIRCUIT_BREAKER: 4,
            RecoveryStrategy.GRACEFUL_DEGRADATION: 5,
            RecoveryStrategy.ROLLBACK: 6
        }
        return priority_map.get(strategy, 5)
    
    def _estimate_recovery_duration(self, strategy: RecoveryStrategy) -> int:
        """估算恢复时间"""
        duration_map = {
            RecoveryStrategy.CIRCUIT_BREAKER: 30,
            RecoveryStrategy.GRACEFUL_DEGRADATION: 60,
            RecoveryStrategy.RESTART_SERVICE: 120,
            RecoveryStrategy.FAILOVER: 180,
            RecoveryStrategy.ROLLBACK: 300,
            RecoveryStrategy.EMERGENCY_SHUTDOWN: 600
        }
        return duration_map.get(strategy, 300)
    
    def _get_recovery_prerequisites(self, fault: FaultEvent, strategy: RecoveryStrategy) -> List[str]:
        """获取恢复前提条件"""
        prerequisites = []
        
        if strategy == RecoveryStrategy.FAILOVER:
            prerequisites.append('backup_available')
        elif strategy == RecoveryStrategy.ROLLBACK:
            prerequisites.append('backup_state_available')
        
        return prerequisites
    
    def _get_success_criteria(self, fault: FaultEvent, strategy: RecoveryStrategy) -> List[str]:
        """获取成功标准"""
        return [
            f'{fault.component}_health_check_passed',
            'no_error_logs_in_last_5_minutes',
            'dependent_services_healthy'
        ]
    
    def _assess_recovery_risk(self, strategy: RecoveryStrategy) -> str:
        """评估恢复风险"""
        risk_map = {
            RecoveryStrategy.GRACEFUL_DEGRADATION: 'low',
            RecoveryStrategy.CIRCUIT_BREAKER: 'low',
            RecoveryStrategy.RESTART_SERVICE: 'medium',
            RecoveryStrategy.FAILOVER: 'medium',
            RecoveryStrategy.ROLLBACK: 'high',
            RecoveryStrategy.EMERGENCY_SHUTDOWN: 'high'
        }
        return risk_map.get(strategy, 'medium')
    
    def _is_strategy_automated(self, strategy: RecoveryStrategy) -> bool:
        """判断策略是否自动化"""
        automated_strategies = [
            RecoveryStrategy.CIRCUIT_BREAKER,
            RecoveryStrategy.GRACEFUL_DEGRADATION,
            RecoveryStrategy.RESTART_SERVICE,
            RecoveryStrategy.FAILOVER
        ]
        return strategy in automated_strategies
    
    def _requires_approval(self, fault: FaultEvent, strategy: RecoveryStrategy) -> bool:
        """判断是否需要审批"""
        if fault.severity == FaultSeverity.CRITICAL:
            return strategy in [RecoveryStrategy.EMERGENCY_SHUTDOWN, RecoveryStrategy.RESTART_SYSTEM]
        return False
    
    def _execute_recovery_plan(self, plan: RecoveryPlan):
        """执行恢复计划"""
        execution = RecoveryExecution(
            execution_id=f"exec_{plan.plan_id}_{int(time.time())}",
            plan_id=plan.plan_id,
            fault_id=plan.fault_id,
            start_time=datetime.now(),
            status=RecoveryStatus.IN_PROGRESS,
            total_steps=len(plan.recovery_steps)
        )
        
        self.recovery_executions.append(execution)
        
        # 在后台执行恢复
        def execute_recovery():
            try:
                self._perform_recovery_steps(plan, execution)
            except Exception as e:
                self.logger.error(f"Error executing recovery plan: {e}")
                execution.status = RecoveryStatus.FAILED
                execution.error_message = str(e)
                execution.end_time = datetime.now()
        
        threading.Thread(target=execute_recovery, daemon=True).start()
    
    def _perform_recovery_steps(self, plan: RecoveryPlan, execution: RecoveryExecution):
        """执行恢复步骤"""
        try:
            for i, step in enumerate(plan.recovery_steps):
                if self.stop_event.is_set():
                    break
                
                execution.current_step = i + 1
                execution.progress_percentage = (i + 1) / len(plan.recovery_steps) * 100
                
                # 执行步骤
                step_result = self._execute_recovery_step(step)
                
                step_record = {
                    'step_number': step['step'],
                    'action': step['action'],
                    'description': step['description'],
                    'success': step_result['success'],
                    'execution_time': step_result['execution_time'],
                    'error_message': step_result.get('error_message')
                }
                
                execution.executed_steps.append(step_record)
                
                if not step_result['success']:
                    # 步骤失败，停止执行
                    execution.status = RecoveryStatus.FAILED
                    execution.error_message = step_result.get('error_message', 'Step failed')
                    execution.end_time = datetime.now()
                    return
            
            # 所有步骤成功完成
            execution.status = RecoveryStatus.COMPLETED
            execution.success = True
            execution.end_time = datetime.now()
            
            # 标记故障为已解决
            if plan.fault_id in self.active_faults:
                fault = self.active_faults[plan.fault_id]
                fault.resolved = True
                fault.resolution_time = datetime.now()
                del self.active_faults[plan.fault_id]
            
            # 通知回调
            for callback in self.recovery_callbacks:
                try:
                    callback(execution)
                except Exception as e:
                    self.logger.error(f"Error in recovery callback: {e}")
            
            self.logger.info(f"Recovery completed successfully for {plan.fault_id}")
            
        except Exception as e:
            execution.status = RecoveryStatus.FAILED
            execution.error_message = str(e)
            execution.end_time = datetime.now()
            self.logger.error(f"Recovery execution failed: {e}")
    
    def _execute_recovery_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """执行单个恢复步骤"""
        start_time = time.time()
        
        try:
            action = step['action']
            
            if action == 'stop_service':
                success = self._stop_service(step['component'])
            elif action == 'start_service':
                success = self._start_service(step['component'])
            elif action == 'verify_health':
                success = self._verify_component_health(step['component'])
            elif action == 'wait':
                time.sleep(step['duration'])
                success = True
            elif action == 'enable_degraded_mode':
                success = self._enable_degraded_mode(step['component'])
            elif action == 'activate_circuit_breaker':
                success = self._activate_circuit_breaker(step['component'])
            else:
                # 模拟其他动作
                time.sleep(np.random.uniform(1, 5))
                success = np.random.random() > 0.1  # 90%成功率
            
            execution_time = time.time() - start_time
            
            return {
                'success': success,
                'execution_time': execution_time
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            return {
                'success': False,
                'execution_time': execution_time,
                'error_message': str(e)
            }
    
    def _stop_service(self, component: str) -> bool:
        """停止服务"""
        # 模拟停止服务
        time.sleep(2)
        return True
    
    def _start_service(self, component: str) -> bool:
        """启动服务"""
        # 模拟启动服务
        time.sleep(5)
        return True
    
    def _verify_component_health(self, component: str) -> bool:
        """验证组件健康"""
        if component in self.components:
            return self.components[component].status == ComponentStatus.HEALTHY
        return False
    
    def _enable_degraded_mode(self, component: str) -> bool:
        """启用降级模式"""
        # 模拟启用降级模式
        time.sleep(1)
        return True
    
    def _activate_circuit_breaker(self, component: str) -> bool:
        """激活断路器"""
        # 模拟激活断路器
        time.sleep(1)
        return True
    
    def register_health_checker(self, component_id: str, checker: Callable) -> None:
        """注册健康检查器"""
        self.health_checkers[component_id] = checker
    
    def register_recovery_handler(self, handler_id: str, handler: Callable) -> None:
        """注册自定义恢复处理器"""
        self.custom_recovery_handlers[handler_id] = handler
    
    def add_fault_callback(self, callback: Callable) -> None:
        """添加故障回调"""
        self.fault_callbacks.append(callback)
    
    def add_recovery_callback(self, callback: Callable) -> None:
        """添加恢复回调"""
        self.recovery_callbacks.append(callback)
    
    def manual_fault_trigger(self, component: str, fault_type: FaultType, 
                           description: str = None) -> str:
        """手动触发故障"""
        self._trigger_fault_event(component, fault_type, description)
        return f"Fault triggered for {component}: {fault_type.value}"
    
    def get_active_faults(self) -> List[FaultEvent]:
        """获取活跃故障"""
        return list(self.active_faults.values())
    
    def get_fault_history(self, component: Optional[str] = None) -> List[FaultEvent]:
        """获取故障历史"""
        faults = list(self.fault_events)
        
        if component:
            faults = [f for f in faults if f.component == component]
        
        return faults
    
    def get_recovery_history(self, fault_id: Optional[str] = None) -> List[RecoveryExecution]:
        """获取恢复历史"""
        executions = list(self.recovery_executions)
        
        if fault_id:
            executions = [e for e in executions if e.fault_id == fault_id]
        
        return executions
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'recovery_system_running': self.is_running,
            'active_faults': len(self.active_faults),
            'total_components': len(self.components),
            'healthy_components': len([c for c in self.components.values() if c.status == ComponentStatus.HEALTHY]),
            'failed_components': len([c for c in self.components.values() if c.status == ComponentStatus.FAILED]),
            'recovery_plans': len(self.recovery_plans),
            'total_fault_events': len(self.fault_events),
            'total_recovery_executions': len(self.recovery_executions),
            'successful_recoveries': len([e for e in self.recovery_executions if e.success])
        }
    
    def get_component_status(self, component_id: Optional[str] = None) -> Union[ComponentHealth, Dict[str, ComponentHealth]]:
        """获取组件状态"""
        if component_id:
            return self.components.get(component_id)
        return dict(self.components)
    
    def acknowledge_fault(self, fault_id: str) -> bool:
        """确认故障"""
        if fault_id in self.active_faults:
            self.active_faults[fault_id].acknowledged = True
            return True
        return False
    
    def resolve_fault(self, fault_id: str) -> bool:
        """手动解决故障"""
        if fault_id in self.active_faults:
            fault = self.active_faults[fault_id]
            fault.resolved = True
            fault.resolution_time = datetime.now()
            del self.active_faults[fault_id]
            return True
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'system_running': self.is_running,
            'monitored_components': len(self.components),
            'active_faults': len(self.active_faults),
            'fault_history': len(self.fault_events),
            'recovery_executions': len(self.recovery_executions),
            'recovery_plans': len(self.recovery_plans),
            'success_rate': len([e for e in self.recovery_executions if e.success]) / len(self.recovery_executions) * 100 if self.recovery_executions else 0.0
        }
    
    def __del__(self):
        """析构函数"""
        if hasattr(self, 'is_running') and self.is_running:
            self.stop()
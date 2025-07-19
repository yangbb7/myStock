# -*- coding: utf-8 -*-
"""
策略生命周期管理 - 提供完整的策略生命周期管理功能
"""

import logging
import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Set
from dataclasses import dataclass, field
import json
import os
from pathlib import Path

from .base_strategy import BaseStrategy, StrategyState
from ..events.enhanced_event_types import EventFactory, StrategyEvent


class LifecyclePhase(Enum):
    """生命周期阶段"""
    CREATED = "created"
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    STARTING = "starting"
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    RESUMING = "resuming"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FINALIZING = "finalizing"
    FINALIZED = "finalized"
    ERROR = "error"


class LifecycleTransition(Enum):
    """生命周期转换"""
    INITIALIZE = "initialize"
    START = "start"
    PAUSE = "pause"
    RESUME = "resume"
    STOP = "stop"
    FINALIZE = "finalize"
    RESTART = "restart"
    RESET = "reset"


@dataclass
class LifecycleEvent:
    """生命周期事件"""
    strategy_name: str
    transition: LifecycleTransition
    from_phase: LifecyclePhase
    to_phase: LifecyclePhase
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None


class LifecycleHook(ABC):
    """生命周期钩子基类"""
    
    @abstractmethod
    def execute(self, strategy: BaseStrategy, phase: LifecyclePhase, **kwargs) -> bool:
        """
        执行钩子
        
        Args:
            strategy: 策略实例
            phase: 生命周期阶段
            **kwargs: 额外参数
            
        Returns:
            bool: 是否执行成功
        """
        pass


class PreInitializeHook(LifecycleHook):
    """初始化前钩子"""
    
    def execute(self, strategy: BaseStrategy, phase: LifecyclePhase, **kwargs) -> bool:
        """执行初始化前的准备工作"""
        # 检查策略参数
        if not strategy.validate_params():
            return False
        
        # 清理旧状态
        strategy.clear_cache()
        
        return True


class PostInitializeHook(LifecycleHook):
    """初始化后钩子"""
    
    def execute(self, strategy: BaseStrategy, phase: LifecyclePhase, **kwargs) -> bool:
        """执行初始化后的验证工作"""
        # 验证策略状态
        if not strategy.is_active():
            return False
        
        return True


class PreStartHook(LifecycleHook):
    """启动前钩子"""
    
    def execute(self, strategy: BaseStrategy, phase: LifecyclePhase, **kwargs) -> bool:
        """执行启动前的检查"""
        # 检查必要的数据源
        if not strategy.symbols:
            return False
        
        return True


class PostStopHook(LifecycleHook):
    """停止后钩子"""
    
    def execute(self, strategy: BaseStrategy, phase: LifecyclePhase, **kwargs) -> bool:
        """执行停止后的清理工作"""
        # 清理资源
        strategy.clear_cache()
        
        # 保存状态
        if hasattr(strategy, 'save_state'):
            strategy.save_state()
        
        return True


class StrategyLifecycleManager:
    """策略生命周期管理器"""
    
    def __init__(self, 
                 persistence_path: Optional[str] = None,
                 enable_health_check: bool = True,
                 health_check_interval: int = 30):
        """
        初始化生命周期管理器
        
        Args:
            persistence_path: 持久化路径
            enable_health_check: 是否启用健康检查
            health_check_interval: 健康检查间隔（秒）
        """
        self.strategies: Dict[str, BaseStrategy] = {}
        self.lifecycle_states: Dict[str, LifecyclePhase] = {}
        self.lifecycle_history: Dict[str, List[LifecycleEvent]] = {}
        
        # 生命周期钩子
        self.hooks: Dict[LifecyclePhase, List[LifecycleHook]] = {
            LifecyclePhase.CREATED: [],
            LifecyclePhase.INITIALIZING: [PreInitializeHook()],
            LifecyclePhase.INITIALIZED: [PostInitializeHook()],
            LifecyclePhase.STARTING: [PreStartHook()],
            LifecyclePhase.RUNNING: [],
            LifecyclePhase.PAUSING: [],
            LifecyclePhase.PAUSED: [],
            LifecyclePhase.RESUMING: [],
            LifecyclePhase.STOPPING: [],
            LifecyclePhase.STOPPED: [PostStopHook()],
            LifecyclePhase.FINALIZING: [],
            LifecyclePhase.FINALIZED: [],
            LifecyclePhase.ERROR: []
        }
        
        # 允许的状态转换
        self.allowed_transitions: Dict[LifecyclePhase, Set[LifecyclePhase]] = {
            LifecyclePhase.CREATED: {LifecyclePhase.INITIALIZING},
            LifecyclePhase.INITIALIZING: {LifecyclePhase.INITIALIZED, LifecyclePhase.ERROR},
            LifecyclePhase.INITIALIZED: {LifecyclePhase.STARTING, LifecyclePhase.FINALIZING},
            LifecyclePhase.STARTING: {LifecyclePhase.RUNNING, LifecyclePhase.ERROR},
            LifecyclePhase.RUNNING: {LifecyclePhase.PAUSING, LifecyclePhase.STOPPING},
            LifecyclePhase.PAUSING: {LifecyclePhase.PAUSED, LifecyclePhase.ERROR},
            LifecyclePhase.PAUSED: {LifecyclePhase.RESUMING, LifecyclePhase.STOPPING},
            LifecyclePhase.RESUMING: {LifecyclePhase.RUNNING, LifecyclePhase.ERROR},
            LifecyclePhase.STOPPING: {LifecyclePhase.STOPPED, LifecyclePhase.ERROR},
            LifecyclePhase.STOPPED: {LifecyclePhase.FINALIZING, LifecyclePhase.INITIALIZING},
            LifecyclePhase.FINALIZING: {LifecyclePhase.FINALIZED, LifecyclePhase.ERROR},
            LifecyclePhase.FINALIZED: {LifecyclePhase.CREATED},
            LifecyclePhase.ERROR: {LifecyclePhase.INITIALIZING, LifecyclePhase.FINALIZING}
        }
        
        # 持久化配置
        self.persistence_path = persistence_path
        if self.persistence_path:
            Path(self.persistence_path).mkdir(parents=True, exist_ok=True)
        
        # 健康检查
        self.enable_health_check = enable_health_check
        self.health_check_interval = health_check_interval
        self.health_check_thread = None
        self.health_check_running = False
        
        # 依赖管理
        self.dependencies: Dict[str, Set[str]] = {}  # strategy_name -> dependent_strategies
        self.dependents: Dict[str, Set[str]] = {}    # strategy_name -> strategies_that_depend_on_it
        
        # 线程安全
        self._lock = threading.RLock()
        
        # 事件总线（可选）
        self.event_bus = None
        
        # 日志
        self.logger = logging.getLogger(__name__)
        
        # 启动健康检查
        if self.enable_health_check:
            self.start_health_check()
    
    def register_strategy(self, strategy: BaseStrategy) -> None:
        """
        注册策略
        
        Args:
            strategy: 策略实例
        """
        with self._lock:
            self.strategies[strategy.name] = strategy
            self.lifecycle_states[strategy.name] = LifecyclePhase.CREATED
            self.lifecycle_history[strategy.name] = []
            
            # 记录创建事件
            event = LifecycleEvent(
                strategy_name=strategy.name,
                transition=LifecycleTransition.INITIALIZE,
                from_phase=LifecyclePhase.CREATED,
                to_phase=LifecyclePhase.CREATED
            )
            self.lifecycle_history[strategy.name].append(event)
            
            self.logger.info(f"Strategy {strategy.name} registered")
    
    def unregister_strategy(self, strategy_name: str) -> bool:
        """
        注销策略
        
        Args:
            strategy_name: 策略名称
            
        Returns:
            bool: 是否成功注销
        """
        with self._lock:
            if strategy_name not in self.strategies:
                return False
            
            # 检查是否有依赖
            if strategy_name in self.dependents and self.dependents[strategy_name]:
                self.logger.error(f"Cannot unregister strategy {strategy_name}: has dependents")
                return False
            
            # 先停止策略
            if self.lifecycle_states[strategy_name] != LifecyclePhase.STOPPED:
                self.stop_strategy(strategy_name)
            
            # 移除策略
            del self.strategies[strategy_name]
            del self.lifecycle_states[strategy_name]
            del self.lifecycle_history[strategy_name]
            
            # 清理依赖关系
            if strategy_name in self.dependencies:
                del self.dependencies[strategy_name]
            
            for deps in self.dependents.values():
                deps.discard(strategy_name)
            
            self.logger.info(f"Strategy {strategy_name} unregistered")
            return True
    
    def add_dependency(self, strategy_name: str, dependency_name: str) -> bool:
        """
        添加策略依赖
        
        Args:
            strategy_name: 策略名称
            dependency_name: 依赖的策略名称
            
        Returns:
            bool: 是否成功添加
        """
        with self._lock:
            if strategy_name not in self.strategies or dependency_name not in self.strategies:
                return False
            
            # 检查循环依赖
            if self._has_circular_dependency(strategy_name, dependency_name):
                self.logger.error(f"Circular dependency detected: {strategy_name} -> {dependency_name}")
                return False
            
            # 添加依赖关系
            if strategy_name not in self.dependencies:
                self.dependencies[strategy_name] = set()
            self.dependencies[strategy_name].add(dependency_name)
            
            if dependency_name not in self.dependents:
                self.dependents[dependency_name] = set()
            self.dependents[dependency_name].add(strategy_name)
            
            self.logger.info(f"Added dependency: {strategy_name} depends on {dependency_name}")
            return True
    
    def _has_circular_dependency(self, strategy_name: str, dependency_name: str) -> bool:
        """检查是否存在循环依赖"""
        visited = set()
        
        def dfs(current: str) -> bool:
            if current in visited:
                return True
            if current == strategy_name:
                return True
            
            visited.add(current)
            for dep in self.dependencies.get(current, set()):
                if dfs(dep):
                    return True
            visited.remove(current)
            return False
        
        return dfs(dependency_name)
    
    def transition_strategy(self, strategy_name: str, transition: LifecycleTransition) -> bool:
        """
        执行策略生命周期转换
        
        Args:
            strategy_name: 策略名称
            transition: 转换类型
            
        Returns:
            bool: 是否成功转换
        """
        with self._lock:
            if strategy_name not in self.strategies:
                return False
            
            strategy = self.strategies[strategy_name]
            current_phase = self.lifecycle_states[strategy_name]
            
            # 确定目标阶段
            target_phase = self._get_target_phase(transition, current_phase)
            if not target_phase:
                self.logger.error(f"Invalid transition {transition} from {current_phase}")
                return False
            
            # 检查是否允许转换
            if target_phase not in self.allowed_transitions.get(current_phase, set()):
                self.logger.error(f"Transition not allowed: {current_phase} -> {target_phase}")
                return False
            
            # 执行转换
            return self._execute_transition(strategy, transition, current_phase, target_phase)
    
    def _get_target_phase(self, transition: LifecycleTransition, current_phase: LifecyclePhase) -> Optional[LifecyclePhase]:
        """获取目标阶段"""
        transition_map = {
            LifecycleTransition.INITIALIZE: LifecyclePhase.INITIALIZING,
            LifecycleTransition.START: LifecyclePhase.STARTING,
            LifecycleTransition.PAUSE: LifecyclePhase.PAUSING,
            LifecycleTransition.RESUME: LifecyclePhase.RESUMING,
            LifecycleTransition.STOP: LifecyclePhase.STOPPING,
            LifecycleTransition.FINALIZE: LifecyclePhase.FINALIZING,
            LifecycleTransition.RESTART: LifecyclePhase.INITIALIZING,
            LifecycleTransition.RESET: LifecyclePhase.CREATED
        }
        
        return transition_map.get(transition)
    
    def _execute_transition(self, strategy: BaseStrategy, transition: LifecycleTransition,
                          from_phase: LifecyclePhase, to_phase: LifecyclePhase) -> bool:
        """执行状态转换"""
        start_time = datetime.now()
        
        try:
            # 更新状态
            self.lifecycle_states[strategy.name] = to_phase
            
            # 执行钩子
            hooks = self.hooks.get(to_phase, [])
            for hook in hooks:
                if not hook.execute(strategy, to_phase):
                    raise Exception(f"Hook {hook.__class__.__name__} failed")
            
            # 执行具体的转换逻辑
            success = self._execute_transition_logic(strategy, transition, to_phase)
            
            if success:
                # 转换成功，更新到最终状态
                final_phase = self._get_final_phase(to_phase)
                if final_phase != to_phase:
                    self.lifecycle_states[strategy.name] = final_phase
                
                # 记录事件
                event = LifecycleEvent(
                    strategy_name=strategy.name,
                    transition=transition,
                    from_phase=from_phase,
                    to_phase=final_phase,
                    timestamp=start_time,
                    success=True
                )
                self.lifecycle_history[strategy.name].append(event)
                
                # 发布事件
                self._publish_lifecycle_event(strategy, event)
                
                self.logger.info(f"Strategy {strategy.name} transitioned: {from_phase} -> {final_phase}")
                return True
            else:
                # 转换失败，恢复状态
                self.lifecycle_states[strategy.name] = LifecyclePhase.ERROR
                raise Exception("Transition logic failed")
                
        except Exception as e:
            # 记录失败事件
            event = LifecycleEvent(
                strategy_name=strategy.name,
                transition=transition,
                from_phase=from_phase,
                to_phase=LifecyclePhase.ERROR,
                timestamp=start_time,
                success=False,
                error_message=str(e)
            )
            self.lifecycle_history[strategy.name].append(event)
            
            # 更新状态为错误
            self.lifecycle_states[strategy.name] = LifecyclePhase.ERROR
            
            self.logger.error(f"Strategy {strategy.name} transition failed: {e}")
            return False
    
    def _get_final_phase(self, intermediate_phase: LifecyclePhase) -> LifecyclePhase:
        """获取最终阶段"""
        phase_map = {
            LifecyclePhase.INITIALIZING: LifecyclePhase.INITIALIZED,
            LifecyclePhase.STARTING: LifecyclePhase.RUNNING,
            LifecyclePhase.PAUSING: LifecyclePhase.PAUSED,
            LifecyclePhase.RESUMING: LifecyclePhase.RUNNING,
            LifecyclePhase.STOPPING: LifecyclePhase.STOPPED,
            LifecyclePhase.FINALIZING: LifecyclePhase.FINALIZED
        }
        
        return phase_map.get(intermediate_phase, intermediate_phase)
    
    def _execute_transition_logic(self, strategy: BaseStrategy, transition: LifecycleTransition,
                                phase: LifecyclePhase) -> bool:
        """执行转换逻辑"""
        try:
            if transition == LifecycleTransition.INITIALIZE:
                strategy.initialize()
            elif transition == LifecycleTransition.START:
                strategy.start()
            elif transition == LifecycleTransition.PAUSE:
                strategy.pause()
            elif transition == LifecycleTransition.RESUME:
                strategy.resume()
            elif transition == LifecycleTransition.STOP:
                strategy.stop()
            elif transition == LifecycleTransition.FINALIZE:
                strategy.finalize()
            elif transition == LifecycleTransition.RESTART:
                strategy.stop()
                strategy.initialize()
            elif transition == LifecycleTransition.RESET:
                strategy.finalize()
                strategy.clear_cache()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Transition logic failed: {e}")
            return False
    
    def _publish_lifecycle_event(self, strategy: BaseStrategy, event: LifecycleEvent) -> None:
        """发布生命周期事件"""
        if self.event_bus:
            strategy_event = EventFactory.create_signal_event(
                event_type='LIFECYCLE',
                strategy_name=strategy.name,
                old_state=event.from_phase.value,
                new_state=event.to_phase.value,
                metadata=event.metadata
            )
            self.event_bus.publish(strategy_event)
    
    def initialize_strategy(self, strategy_name: str) -> bool:
        """初始化策略"""
        return self.transition_strategy(strategy_name, LifecycleTransition.INITIALIZE)
    
    def start_strategy(self, strategy_name: str) -> bool:
        """启动策略"""
        # 检查依赖
        if not self._check_dependencies(strategy_name):
            return False
        
        return self.transition_strategy(strategy_name, LifecycleTransition.START)
    
    def pause_strategy(self, strategy_name: str) -> bool:
        """暂停策略"""
        return self.transition_strategy(strategy_name, LifecycleTransition.PAUSE)
    
    def resume_strategy(self, strategy_name: str) -> bool:
        """恢复策略"""
        return self.transition_strategy(strategy_name, LifecycleTransition.RESUME)
    
    def stop_strategy(self, strategy_name: str) -> bool:
        """停止策略"""
        # 先停止依赖此策略的其他策略
        dependents = self.dependents.get(strategy_name, set())
        for dependent in dependents:
            if self.lifecycle_states[dependent] == LifecyclePhase.RUNNING:
                self.stop_strategy(dependent)
        
        return self.transition_strategy(strategy_name, LifecycleTransition.STOP)
    
    def restart_strategy(self, strategy_name: str) -> bool:
        """重启策略"""
        return self.transition_strategy(strategy_name, LifecycleTransition.RESTART)
    
    def _check_dependencies(self, strategy_name: str) -> bool:
        """检查依赖是否满足"""
        dependencies = self.dependencies.get(strategy_name, set())
        
        for dep_name in dependencies:
            if self.lifecycle_states[dep_name] != LifecyclePhase.RUNNING:
                self.logger.error(f"Dependency {dep_name} not running for strategy {strategy_name}")
                return False
        
        return True
    
    def start_health_check(self) -> None:
        """启动健康检查"""
        if self.health_check_running:
            return
        
        self.health_check_running = True
        self.health_check_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self.health_check_thread.start()
        
        self.logger.info("Health check started")
    
    def stop_health_check(self) -> None:
        """停止健康检查"""
        self.health_check_running = False
        if self.health_check_thread:
            self.health_check_thread.join()
        
        self.logger.info("Health check stopped")
    
    def _health_check_loop(self) -> None:
        """健康检查循环"""
        while self.health_check_running:
            try:
                self._perform_health_check()
                time.sleep(self.health_check_interval)
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
                time.sleep(self.health_check_interval)
    
    def _perform_health_check(self) -> None:
        """执行健康检查"""
        with self._lock:
            for strategy_name, strategy in self.strategies.items():
                phase = self.lifecycle_states[strategy_name]
                
                # 检查运行中的策略
                if phase == LifecyclePhase.RUNNING:
                    if not strategy.is_active():
                        self.logger.warning(f"Strategy {strategy_name} is not active, stopping")
                        self.stop_strategy(strategy_name)
                    
                    # 检查策略健康状态
                    if hasattr(strategy, 'check_health'):
                        if not strategy.check_health():
                            self.logger.warning(f"Strategy {strategy_name} health check failed")
                
                # 检查错误状态的策略
                elif phase == LifecyclePhase.ERROR:
                    # 可以实现自动恢复逻辑
                    pass
    
    def get_strategy_state(self, strategy_name: str) -> Optional[LifecyclePhase]:
        """获取策略状态"""
        return self.lifecycle_states.get(strategy_name)
    
    def get_strategy_history(self, strategy_name: str) -> List[LifecycleEvent]:
        """获取策略历史"""
        return self.lifecycle_history.get(strategy_name, [])
    
    def get_all_strategies_status(self) -> Dict[str, Dict[str, Any]]:
        """获取所有策略状态"""
        result = {}
        
        with self._lock:
            for strategy_name, strategy in self.strategies.items():
                phase = self.lifecycle_states[strategy_name]
                history = self.lifecycle_history[strategy_name]
                
                result[strategy_name] = {
                    'phase': phase.value,
                    'active': strategy.is_active(),
                    'created_time': strategy.created_time.isoformat(),
                    'started_time': strategy.started_time.isoformat() if strategy.started_time else None,
                    'stopped_time': strategy.stopped_time.isoformat() if strategy.stopped_time else None,
                    'event_count': len(history),
                    'last_event': history[-1].timestamp.isoformat() if history else None,
                    'dependencies': list(self.dependencies.get(strategy_name, set())),
                    'dependents': list(self.dependents.get(strategy_name, set()))
                }
        
        return result
    
    def add_hook(self, phase: LifecyclePhase, hook: LifecycleHook) -> None:
        """添加生命周期钩子"""
        self.hooks[phase].append(hook)
    
    def remove_hook(self, phase: LifecyclePhase, hook: LifecycleHook) -> None:
        """移除生命周期钩子"""
        if hook in self.hooks[phase]:
            self.hooks[phase].remove(hook)
    
    def save_state(self, strategy_name: Optional[str] = None) -> bool:
        """保存状态到文件"""
        if not self.persistence_path:
            return False
        
        try:
            if strategy_name:
                # 保存单个策略状态
                state_file = os.path.join(self.persistence_path, f"{strategy_name}_state.json")
                state_data = {
                    'strategy_name': strategy_name,
                    'phase': self.lifecycle_states[strategy_name].value,
                    'history': [
                        {
                            'strategy_name': event.strategy_name,
                            'transition': event.transition.value,
                            'from_phase': event.from_phase.value,
                            'to_phase': event.to_phase.value,
                            'timestamp': event.timestamp.isoformat(),
                            'success': event.success,
                            'error_message': event.error_message
                        }
                        for event in self.lifecycle_history[strategy_name]
                    ]
                }
                
                with open(state_file, 'w') as f:
                    json.dump(state_data, f, indent=2)
            else:
                # 保存所有策略状态
                state_file = os.path.join(self.persistence_path, "all_strategies_state.json")
                state_data = {
                    'strategies': {
                        name: {
                            'phase': phase.value,
                            'dependencies': list(self.dependencies.get(name, set())),
                            'dependents': list(self.dependents.get(name, set()))
                        }
                        for name, phase in self.lifecycle_states.items()
                    },
                    'timestamp': datetime.now().isoformat()
                }
                
                with open(state_file, 'w') as f:
                    json.dump(state_data, f, indent=2)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
            return False
    
    def load_state(self, strategy_name: Optional[str] = None) -> bool:
        """从文件加载状态"""
        if not self.persistence_path:
            return False
        
        try:
            if strategy_name:
                # 加载单个策略状态
                state_file = os.path.join(self.persistence_path, f"{strategy_name}_state.json")
                if not os.path.exists(state_file):
                    return False
                
                with open(state_file, 'r') as f:
                    state_data = json.load(f)
                
                # 恢复状态
                phase = LifecyclePhase(state_data['phase'])
                self.lifecycle_states[strategy_name] = phase
                
                # 恢复历史
                history = []
                for event_data in state_data['history']:
                    event = LifecycleEvent(
                        strategy_name=event_data['strategy_name'],
                        transition=LifecycleTransition(event_data['transition']),
                        from_phase=LifecyclePhase(event_data['from_phase']),
                        to_phase=LifecyclePhase(event_data['to_phase']),
                        timestamp=datetime.fromisoformat(event_data['timestamp']),
                        success=event_data['success'],
                        error_message=event_data.get('error_message')
                    )
                    history.append(event)
                
                self.lifecycle_history[strategy_name] = history
                
            else:
                # 加载所有策略状态
                state_file = os.path.join(self.persistence_path, "all_strategies_state.json")
                if not os.path.exists(state_file):
                    return False
                
                with open(state_file, 'r') as f:
                    state_data = json.load(f)
                
                # 恢复状态
                for name, strategy_data in state_data['strategies'].items():
                    if name in self.strategies:
                        self.lifecycle_states[name] = LifecyclePhase(strategy_data['phase'])
                        
                        # 恢复依赖关系
                        self.dependencies[name] = set(strategy_data.get('dependencies', []))
                        self.dependents[name] = set(strategy_data.get('dependents', []))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load state: {e}")
            return False
    
    def shutdown(self) -> None:
        """关闭生命周期管理器"""
        self.logger.info("Shutting down lifecycle manager")
        
        # 停止健康检查
        self.stop_health_check()
        
        # 停止所有策略
        with self._lock:
            for strategy_name in list(self.strategies.keys()):
                if self.lifecycle_states[strategy_name] == LifecyclePhase.RUNNING:
                    self.stop_strategy(strategy_name)
        
        # 保存状态
        self.save_state()
        
        self.logger.info("Lifecycle manager shutdown complete")
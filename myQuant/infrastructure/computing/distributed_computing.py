import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
import threading
import time
import pickle
import json
from pathlib import Path
import queue
import uuid
import socket
import zmq
from functools import wraps
import dask
from dask.distributed import Client, as_completed as dask_as_completed
from dask import delayed
import ray
import warnings
warnings.filterwarnings('ignore')

class ComputingBackend(Enum):
    LOCAL = "local"
    DASK = "dask"
    RAY = "ray"
    CELERY = "celery"
    SPARK = "spark"
    CUSTOM = "custom"

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"

class TaskPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4

class ResourceType(Enum):
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    DISK = "disk"
    NETWORK = "network"

@dataclass
class TaskConfig:
    """任务配置"""
    task_id: str
    function_name: str
    args: Tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.MEDIUM
    timeout: int = 300
    retry_count: int = 3
    retry_delay: int = 5
    resource_requirements: Dict[ResourceType, float] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TaskResult:
    """任务结果"""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    execution_time: Optional[float] = None
    worker_id: Optional[str] = None
    resource_usage: Dict[ResourceType, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WorkerNode:
    """工作节点"""
    worker_id: str
    host: str
    port: int
    status: str
    resources: Dict[ResourceType, float]
    available_resources: Dict[ResourceType, float]
    active_tasks: List[str]
    last_heartbeat: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ComputingCluster:
    """计算集群"""
    cluster_id: str
    nodes: List[WorkerNode]
    total_resources: Dict[ResourceType, float]
    available_resources: Dict[ResourceType, float]
    active_tasks: int
    completed_tasks: int
    failed_tasks: int
    last_updated: datetime

@dataclass
class ComputingConfig:
    """分布式计算配置"""
    backend: ComputingBackend
    max_workers: int = 10
    worker_threads: int = 4
    worker_memory_limit: str = "2GB"
    scheduler_address: Optional[str] = None
    enable_gpu: bool = False
    enable_clustering: bool = False
    cluster_nodes: List[str] = field(default_factory=list)
    task_queue_size: int = 1000
    result_cache_size: int = 500
    heartbeat_interval: int = 30
    task_timeout: int = 300
    enable_monitoring: bool = True
    enable_auto_scaling: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

class DistributedComputingEngine:
    """分布式计算引擎"""
    
    def __init__(self, config: ComputingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 计算后端
        self.backend = None
        self.client = None
        
        # 任务管理
        self.task_queue = queue.PriorityQueue(maxsize=config.task_queue_size)
        self.active_tasks: Dict[str, TaskConfig] = {}
        self.task_results: Dict[str, TaskResult] = {}
        self.task_dependencies: Dict[str, List[str]] = {}
        
        # 工作节点
        self.worker_nodes: Dict[str, WorkerNode] = {}
        self.local_worker_pool = None
        
        # 运行状态
        self.is_running = False
        self.scheduler_thread = None
        self.monitor_thread = None
        self.stop_event = threading.Event()
        
        # 统计信息
        self.stats = {
            'tasks_submitted': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'tasks_cancelled': 0,
            'total_execution_time': 0,
            'average_execution_time': 0,
            'tasks_per_second': 0,
            'last_updated': datetime.now()
        }
        
        # 注册的函数
        self.registered_functions: Dict[str, Callable] = {}
        
        # 初始化
        self._initialize_backend()
        self._initialize_monitoring()
    
    def _initialize_backend(self):
        """初始化计算后端"""
        try:
            if self.config.backend == ComputingBackend.LOCAL:
                self._initialize_local_backend()
            elif self.config.backend == ComputingBackend.DASK:
                self._initialize_dask_backend()
            elif self.config.backend == ComputingBackend.RAY:
                self._initialize_ray_backend()
            elif self.config.backend == ComputingBackend.CELERY:
                self._initialize_celery_backend()
            elif self.config.backend == ComputingBackend.SPARK:
                self._initialize_spark_backend()
            else:
                raise ValueError(f"Unsupported backend: {self.config.backend}")
            
            self.logger.info(f"Initialized {self.config.backend.value} computing backend")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize backend: {e}")
            raise
    
    def _initialize_local_backend(self):
        """初始化本地计算后端"""
        try:
            # 创建线程池和进程池
            self.thread_pool = ThreadPoolExecutor(
                max_workers=self.config.max_workers,
                thread_name_prefix="computing_thread"
            )
            
            self.process_pool = ProcessPoolExecutor(
                max_workers=min(self.config.max_workers, mp.cpu_count()),
                mp_context=mp.get_context('spawn')
            )
            
            # 创建本地工作节点
            self._create_local_worker_node()
            
        except Exception as e:
            self.logger.error(f"Error initializing local backend: {e}")
            raise
    
    def _initialize_dask_backend(self):
        """初始化Dask计算后端"""
        try:
            if self.config.scheduler_address:
                # 连接到现有的Dask集群
                self.client = Client(self.config.scheduler_address)
            else:
                # 创建本地Dask集群
                self.client = Client(
                    n_workers=self.config.max_workers,
                    threads_per_worker=self.config.worker_threads,
                    memory_limit=self.config.worker_memory_limit
                )
            
            self.logger.info(f"Dask cluster: {self.client.cluster}")
            
        except Exception as e:
            self.logger.error(f"Error initializing Dask backend: {e}")
            raise
    
    def _initialize_ray_backend(self):
        """初始化Ray计算后端"""
        try:
            if self.config.scheduler_address:
                # 连接到现有的Ray集群
                ray.init(address=self.config.scheduler_address)
            else:
                # 创建本地Ray集群
                ray.init(
                    num_cpus=self.config.max_workers,
                    ignore_reinit_error=True
                )
            
            self.logger.info(f"Ray cluster initialized: {ray.cluster_resources()}")
            
        except Exception as e:
            self.logger.error(f"Error initializing Ray backend: {e}")
            raise
    
    def _initialize_celery_backend(self):
        """初始化Celery计算后端"""
        try:
            from celery import Celery
            
            # 创建Celery应用
            self.celery_app = Celery(
                'myquant_computing',
                broker=self.config.metadata.get('broker_url', 'redis://localhost:6379'),
                backend=self.config.metadata.get('result_backend', 'redis://localhost:6379')
            )
            
            # 配置Celery
            self.celery_app.conf.update(
                task_serializer='pickle',
                accept_content=['pickle'],
                result_serializer='pickle',
                timezone='UTC',
                enable_utc=True,
                task_track_started=True,
                task_time_limit=self.config.task_timeout,
                worker_prefetch_multiplier=1,
                worker_max_tasks_per_child=1000
            )
            
        except Exception as e:
            self.logger.error(f"Error initializing Celery backend: {e}")
            raise
    
    def _initialize_spark_backend(self):
        """初始化Spark计算后端"""
        try:
            from pyspark.sql import SparkSession
            
            # 创建Spark会话
            self.spark = SparkSession.builder \
                .appName("MyQuant Computing") \
                .config("spark.executor.memory", self.config.worker_memory_limit) \
                .config("spark.executor.cores", str(self.config.worker_threads)) \
                .config("spark.executor.instances", str(self.config.max_workers)) \
                .getOrCreate()
            
            self.spark_context = self.spark.sparkContext
            
        except Exception as e:
            self.logger.error(f"Error initializing Spark backend: {e}")
            raise
    
    def _initialize_monitoring(self):
        """初始化监控"""
        if self.config.enable_monitoring:
            # 创建监控线程
            self.monitor_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True
            )
    
    def _create_local_worker_node(self):
        """创建本地工作节点"""
        worker_id = f"local_worker_{uuid.uuid4().hex[:8]}"
        
        # 获取系统资源信息
        import psutil
        
        resources = {
            ResourceType.CPU: float(psutil.cpu_count()),
            ResourceType.MEMORY: float(psutil.virtual_memory().total / 1024 / 1024 / 1024),  # GB
            ResourceType.DISK: float(psutil.disk_usage('/').free / 1024 / 1024 / 1024),  # GB
            ResourceType.NETWORK: 1000.0  # Mbps (假设值)
        }
        
        if self.config.enable_gpu:
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    resources[ResourceType.GPU] = float(len(gpus))
            except ImportError:
                pass
        
        worker_node = WorkerNode(
            worker_id=worker_id,
            host=socket.gethostname(),
            port=0,
            status="active",
            resources=resources,
            available_resources=resources.copy(),
            active_tasks=[],
            last_heartbeat=datetime.now()
        )
        
        self.worker_nodes[worker_id] = worker_node
    
    def start(self):
        """启动分布式计算引擎"""
        if self.is_running:
            return
        
        self.is_running = True
        self.stop_event.clear()
        
        # 启动调度器线程
        self.scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            daemon=True
        )
        self.scheduler_thread.start()
        
        # 启动监控线程
        if self.config.enable_monitoring and self.monitor_thread:
            self.monitor_thread.start()
        
        self.logger.info("Distributed computing engine started")
    
    def stop(self):
        """停止分布式计算引擎"""
        if not self.is_running:
            return
        
        self.logger.info("Stopping distributed computing engine...")
        self.stop_event.set()
        
        # 停止线程
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        # 关闭计算后端
        self._shutdown_backend()
        
        self.is_running = False
        self.logger.info("Distributed computing engine stopped")
    
    def _scheduler_loop(self):
        """调度器主循环"""
        while not self.stop_event.is_set():
            try:
                # 处理任务队列
                self._process_task_queue()
                
                # 检查任务状态
                self._check_task_status()
                
                # 清理完成的任务
                self._cleanup_completed_tasks()
                
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error in scheduler loop: {e}")
                time.sleep(1)
    
    def _monitoring_loop(self):
        """监控主循环"""
        while not self.stop_event.is_set():
            try:
                # 更新统计信息
                self._update_statistics()
                
                # 检查工作节点状态
                self._check_worker_status()
                
                # 自动扩缩容
                if self.config.enable_auto_scaling:
                    self._auto_scale_workers()
                
                time.sleep(self.config.heartbeat_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1)
    
    def register_function(self, func: Callable, name: Optional[str] = None):
        """注册函数"""
        function_name = name or func.__name__
        self.registered_functions[function_name] = func
        self.logger.info(f"Registered function: {function_name}")
    
    def submit_task(self, task_config: TaskConfig) -> str:
        """提交任务"""
        try:
            # 验证任务配置
            if task_config.function_name not in self.registered_functions:
                raise ValueError(f"Function {task_config.function_name} not registered")
            
            # 检查依赖
            if task_config.dependencies:
                for dep_task_id in task_config.dependencies:
                    if dep_task_id not in self.task_results:
                        raise ValueError(f"Dependency task {dep_task_id} not found")
                    
                    result = self.task_results[dep_task_id]
                    if result.status != TaskStatus.COMPLETED:
                        raise ValueError(f"Dependency task {dep_task_id} not completed")
            
            # 添加到任务队列
            priority = -task_config.priority.value  # 负值使高优先级排在前面
            self.task_queue.put((priority, time.time(), task_config))
            
            # 添加到活动任务
            self.active_tasks[task_config.task_id] = task_config
            
            # 创建任务结果
            self.task_results[task_config.task_id] = TaskResult(
                task_id=task_config.task_id,
                status=TaskStatus.PENDING
            )
            
            self.stats['tasks_submitted'] += 1
            
            return task_config.task_id
            
        except Exception as e:
            self.logger.error(f"Error submitting task: {e}")
            raise
    
    def _process_task_queue(self):
        """处理任务队列"""
        try:
            while not self.task_queue.empty():
                try:
                    priority, timestamp, task_config = self.task_queue.get_nowait()
                    
                    # 检查资源可用性
                    if self._check_resource_availability(task_config):
                        # 执行任务
                        self._execute_task(task_config)
                    else:
                        # 重新放入队列
                        self.task_queue.put((priority, timestamp, task_config))
                        break
                    
                except queue.Empty:
                    break
                    
        except Exception as e:
            self.logger.error(f"Error processing task queue: {e}")
    
    def _check_resource_availability(self, task_config: TaskConfig) -> bool:
        """检查资源可用性"""
        for resource_type, required_amount in task_config.resource_requirements.items():
            total_available = sum(
                worker.available_resources.get(resource_type, 0)
                for worker in self.worker_nodes.values()
                if worker.status == "active"
            )
            
            if total_available < required_amount:
                return False
        
        return True
    
    def _execute_task(self, task_config: TaskConfig):
        """执行任务"""
        try:
            # 获取函数
            func = self.registered_functions[task_config.function_name]
            
            # 更新任务状态
            result = self.task_results[task_config.task_id]
            result.status = TaskStatus.RUNNING
            result.start_time = datetime.now()
            
            # 根据后端类型执行任务
            if self.config.backend == ComputingBackend.LOCAL:
                self._execute_local_task(task_config, func)
            elif self.config.backend == ComputingBackend.DASK:
                self._execute_dask_task(task_config, func)
            elif self.config.backend == ComputingBackend.RAY:
                self._execute_ray_task(task_config, func)
            elif self.config.backend == ComputingBackend.CELERY:
                self._execute_celery_task(task_config, func)
            elif self.config.backend == ComputingBackend.SPARK:
                self._execute_spark_task(task_config, func)
            
        except Exception as e:
            self.logger.error(f"Error executing task {task_config.task_id}: {e}")
            result = self.task_results[task_config.task_id]
            result.status = TaskStatus.FAILED
            result.error = str(e)
            result.end_time = datetime.now()
    
    def _execute_local_task(self, task_config: TaskConfig, func: Callable):
        """执行本地任务"""
        def task_wrapper():
            try:
                result = func(*task_config.args, **task_config.kwargs)
                
                # 更新任务结果
                task_result = self.task_results[task_config.task_id]
                task_result.status = TaskStatus.COMPLETED
                task_result.result = result
                task_result.end_time = datetime.now()
                task_result.execution_time = (task_result.end_time - task_result.start_time).total_seconds()
                
                # 从活动任务中移除
                self.active_tasks.pop(task_config.task_id, None)
                
                return result
                
            except Exception as e:
                # 更新任务结果
                task_result = self.task_results[task_config.task_id]
                task_result.status = TaskStatus.FAILED
                task_result.error = str(e)
                task_result.end_time = datetime.now()
                
                # 从活动任务中移除
                self.active_tasks.pop(task_config.task_id, None)
                
                raise
        
        # 根据任务类型选择执行器
        if task_config.metadata.get('use_process', False):
            self.process_pool.submit(task_wrapper)
        else:
            self.thread_pool.submit(task_wrapper)
    
    def _execute_dask_task(self, task_config: TaskConfig, func: Callable):
        """执行Dask任务"""
        try:
            # 使用Dask delayed装饰器
            delayed_func = delayed(func)
            future = delayed_func(*task_config.args, **task_config.kwargs)
            
            # 提交计算
            dask_future = self.client.compute(future)
            
            # 添加回调
            def on_complete(future):
                try:
                    result = future.result()
                    
                    # 更新任务结果
                    task_result = self.task_results[task_config.task_id]
                    task_result.status = TaskStatus.COMPLETED
                    task_result.result = result
                    task_result.end_time = datetime.now()
                    task_result.execution_time = (task_result.end_time - task_result.start_time).total_seconds()
                    
                except Exception as e:
                    # 更新任务结果
                    task_result = self.task_results[task_config.task_id]
                    task_result.status = TaskStatus.FAILED
                    task_result.error = str(e)
                    task_result.end_time = datetime.now()
                
                # 从活动任务中移除
                self.active_tasks.pop(task_config.task_id, None)
            
            dask_future.add_done_callback(on_complete)
            
        except Exception as e:
            self.logger.error(f"Error executing Dask task: {e}")
            raise
    
    def _execute_ray_task(self, task_config: TaskConfig, func: Callable):
        """执行Ray任务"""
        try:
            # 使用Ray remote装饰器
            remote_func = ray.remote(func)
            future = remote_func.remote(*task_config.args, **task_config.kwargs)
            
            # 异步获取结果
            def get_result():
                try:
                    result = ray.get(future)
                    
                    # 更新任务结果
                    task_result = self.task_results[task_config.task_id]
                    task_result.status = TaskStatus.COMPLETED
                    task_result.result = result
                    task_result.end_time = datetime.now()
                    task_result.execution_time = (task_result.end_time - task_result.start_time).total_seconds()
                    
                except Exception as e:
                    # 更新任务结果
                    task_result = self.task_results[task_config.task_id]
                    task_result.status = TaskStatus.FAILED
                    task_result.error = str(e)
                    task_result.end_time = datetime.now()
                
                # 从活动任务中移除
                self.active_tasks.pop(task_config.task_id, None)
            
            # 在线程池中异步获取结果
            self.thread_pool.submit(get_result)
            
        except Exception as e:
            self.logger.error(f"Error executing Ray task: {e}")
            raise
    
    def _execute_celery_task(self, task_config: TaskConfig, func: Callable):
        """执行Celery任务"""
        try:
            # 创建Celery任务
            @self.celery_app.task(bind=True)
            def celery_task_wrapper(self, *args, **kwargs):
                return func(*args, **kwargs)
            
            # 提交任务
            async_result = celery_task_wrapper.apply_async(
                args=task_config.args,
                kwargs=task_config.kwargs,
                task_id=task_config.task_id
            )
            
            # 监控任务状态
            def monitor_task():
                try:
                    result = async_result.get(timeout=task_config.timeout)
                    
                    # 更新任务结果
                    task_result = self.task_results[task_config.task_id]
                    task_result.status = TaskStatus.COMPLETED
                    task_result.result = result
                    task_result.end_time = datetime.now()
                    task_result.execution_time = (task_result.end_time - task_result.start_time).total_seconds()
                    
                except Exception as e:
                    # 更新任务结果
                    task_result = self.task_results[task_config.task_id]
                    task_result.status = TaskStatus.FAILED
                    task_result.error = str(e)
                    task_result.end_time = datetime.now()
                
                # 从活动任务中移除
                self.active_tasks.pop(task_config.task_id, None)
            
            # 在线程池中监控任务
            self.thread_pool.submit(monitor_task)
            
        except Exception as e:
            self.logger.error(f"Error executing Celery task: {e}")
            raise
    
    def _execute_spark_task(self, task_config: TaskConfig, func: Callable):
        """执行Spark任务"""
        try:
            # 使用Spark的函数式编程接口
            def spark_task_wrapper():
                try:
                    result = func(*task_config.args, **task_config.kwargs)
                    
                    # 更新任务结果
                    task_result = self.task_results[task_config.task_id]
                    task_result.status = TaskStatus.COMPLETED
                    task_result.result = result
                    task_result.end_time = datetime.now()
                    task_result.execution_time = (task_result.end_time - task_result.start_time).total_seconds()
                    
                except Exception as e:
                    # 更新任务结果
                    task_result = self.task_results[task_config.task_id]
                    task_result.status = TaskStatus.FAILED
                    task_result.error = str(e)
                    task_result.end_time = datetime.now()
                
                # 从活动任务中移除
                self.active_tasks.pop(task_config.task_id, None)
            
            # 在线程池中执行Spark任务
            self.thread_pool.submit(spark_task_wrapper)
            
        except Exception as e:
            self.logger.error(f"Error executing Spark task: {e}")
            raise
    
    def _check_task_status(self):
        """检查任务状态"""
        current_time = datetime.now()
        
        for task_id, task_config in list(self.active_tasks.items()):
            task_result = self.task_results[task_id]
            
            # 检查超时
            if (task_result.start_time and 
                (current_time - task_result.start_time).total_seconds() > task_config.timeout):
                
                task_result.status = TaskStatus.FAILED
                task_result.error = "Task timeout"
                task_result.end_time = current_time
                
                self.active_tasks.pop(task_id, None)
                self.stats['tasks_failed'] += 1
    
    def _cleanup_completed_tasks(self):
        """清理已完成的任务"""
        # 保留最近的结果
        if len(self.task_results) > self.config.result_cache_size:
            completed_tasks = [
                (task_id, result) for task_id, result in self.task_results.items()
                if result.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]
            ]
            
            # 按完成时间排序
            completed_tasks.sort(key=lambda x: x[1].end_time or datetime.min)
            
            # 移除旧的任务结果
            tasks_to_remove = completed_tasks[:-self.config.result_cache_size]
            for task_id, _ in tasks_to_remove:
                self.task_results.pop(task_id, None)
    
    def _update_statistics(self):
        """更新统计信息"""
        current_time = datetime.now()
        
        # 计算完成和失败的任务数
        completed_tasks = sum(1 for result in self.task_results.values() 
                            if result.status == TaskStatus.COMPLETED)
        failed_tasks = sum(1 for result in self.task_results.values() 
                          if result.status == TaskStatus.FAILED)
        
        # 计算平均执行时间
        execution_times = [result.execution_time for result in self.task_results.values() 
                          if result.execution_time is not None]
        
        self.stats.update({
            'tasks_completed': completed_tasks,
            'tasks_failed': failed_tasks,
            'total_execution_time': sum(execution_times),
            'average_execution_time': np.mean(execution_times) if execution_times else 0,
            'last_updated': current_time
        })
    
    def _check_worker_status(self):
        """检查工作节点状态"""
        current_time = datetime.now()
        
        for worker_id, worker in list(self.worker_nodes.items()):
            # 检查心跳
            if (current_time - worker.last_heartbeat).total_seconds() > self.config.heartbeat_interval * 2:
                worker.status = "inactive"
                self.logger.warning(f"Worker {worker_id} is inactive")
    
    def _auto_scale_workers(self):
        """自动扩缩容"""
        # 简单的自动扩缩容逻辑
        active_workers = sum(1 for worker in self.worker_nodes.values() 
                           if worker.status == "active")
        
        queue_size = self.task_queue.qsize()
        
        # 如果队列太长，尝试添加更多工作节点
        if queue_size > active_workers * 2:
            self.logger.info("High task queue, considering scaling up")
        
        # 如果队列很空，尝试减少工作节点
        elif queue_size < active_workers * 0.5:
            self.logger.info("Low task queue, considering scaling down")
    
    def _shutdown_backend(self):
        """关闭计算后端"""
        try:
            if hasattr(self, 'thread_pool'):
                self.thread_pool.shutdown(wait=True)
            
            if hasattr(self, 'process_pool'):
                self.process_pool.shutdown(wait=True)
            
            if hasattr(self, 'client') and self.client:
                self.client.close()
            
            if hasattr(self, 'celery_app'):
                self.celery_app.control.purge()
            
            if hasattr(self, 'spark'):
                self.spark.stop()
            
        except Exception as e:
            self.logger.error(f"Error shutting down backend: {e}")
    
    def get_task_result(self, task_id: str) -> Optional[TaskResult]:
        """获取任务结果"""
        return self.task_results.get(task_id)
    
    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        if task_id in self.active_tasks:
            result = self.task_results.get(task_id)
            if result:
                result.status = TaskStatus.CANCELLED
                result.end_time = datetime.now()
            
            self.active_tasks.pop(task_id, None)
            self.stats['tasks_cancelled'] += 1
            return True
        
        return False
    
    def get_cluster_status(self) -> ComputingCluster:
        """获取集群状态"""
        total_resources = {}
        available_resources = {}
        
        for resource_type in ResourceType:
            total_resources[resource_type] = sum(
                worker.resources.get(resource_type, 0)
                for worker in self.worker_nodes.values()
            )
            available_resources[resource_type] = sum(
                worker.available_resources.get(resource_type, 0)
                for worker in self.worker_nodes.values()
                if worker.status == "active"
            )
        
        return ComputingCluster(
            cluster_id=f"cluster_{uuid.uuid4().hex[:8]}",
            nodes=list(self.worker_nodes.values()),
            total_resources=total_resources,
            available_resources=available_resources,
            active_tasks=len(self.active_tasks),
            completed_tasks=self.stats['tasks_completed'],
            failed_tasks=self.stats['tasks_failed'],
            last_updated=datetime.now()
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'backend': self.config.backend.value,
            'is_running': self.is_running,
            'active_workers': len([w for w in self.worker_nodes.values() if w.status == "active"]),
            'total_workers': len(self.worker_nodes),
            'queue_size': self.task_queue.qsize(),
            'active_tasks': len(self.active_tasks),
            'registered_functions': len(self.registered_functions),
            **self.stats
        }
    
    def __del__(self):
        """析构函数"""
        if hasattr(self, 'is_running') and self.is_running:
            self.stop()


# 装饰器函数
def distributed_task(engine: DistributedComputingEngine, 
                    priority: TaskPriority = TaskPriority.MEDIUM,
                    timeout: int = 300,
                    retry_count: int = 3):
    """分布式任务装饰器"""
    def decorator(func):
        # 注册函数
        engine.register_function(func)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 创建任务配置
            task_config = TaskConfig(
                task_id=f"{func.__name__}_{uuid.uuid4().hex[:8]}",
                function_name=func.__name__,
                args=args,
                kwargs=kwargs,
                priority=priority,
                timeout=timeout,
                retry_count=retry_count
            )
            
            # 提交任务
            task_id = engine.submit_task(task_config)
            
            # 返回任务ID
            return task_id
        
        return wrapper
    return decorator
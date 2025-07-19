import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Union, Callable, Generator
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
import threading
import time
import gc
import mmap
import os
import psutil
import sys
from pathlib import Path
from functools import wraps, lru_cache
from contextlib import contextmanager
import weakref
from collections import OrderedDict
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

# 尝试导入可选的性能库
try:
    import pympler
    from pympler import tracker, muppy, summary
    PYMPLER_AVAILABLE = True
except ImportError:
    PYMPLER_AVAILABLE = False

try:
    import memory_profiler
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False

try:
    import numpy as np
    from numba import jit, njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

class MemoryStrategy(Enum):
    LAZY_LOADING = "lazy_loading"
    CHUNK_PROCESSING = "chunk_processing"
    MEMORY_MAPPING = "memory_mapping"
    COMPRESSION = "compression"
    STREAMING = "streaming"
    CACHE_EVICTION = "cache_eviction"
    COPY_ON_WRITE = "copy_on_write"
    OBJECT_POOLING = "object_pooling"

class ComputeOptimization(Enum):
    NONE = "none"
    VECTORIZATION = "vectorization"
    PARALLELIZATION = "parallelization"
    GPU_ACCELERATION = "gpu_acceleration"
    JIT_COMPILATION = "jit_compilation"
    MEMORY_LAYOUT = "memory_layout"
    CACHE_OPTIMIZATION = "cache_optimization"

class DataLayout(Enum):
    ROW_MAJOR = "row_major"
    COLUMN_MAJOR = "column_major"
    BLOCK_COMPRESSED = "block_compressed"
    SPARSE = "sparse"
    CHUNKED = "chunked"

@dataclass
class MemoryConfig:
    """内存优化配置"""
    max_memory_usage: str = "4GB"
    chunk_size: int = 10000
    cache_size: int = 1000
    enable_lazy_loading: bool = True
    enable_memory_mapping: bool = True
    enable_compression: bool = True
    compression_level: int = 6
    enable_object_pooling: bool = True
    pool_size: int = 100
    enable_gc_optimization: bool = True
    gc_threshold: float = 0.8
    memory_strategies: List[MemoryStrategy] = field(default_factory=lambda: [
        MemoryStrategy.LAZY_LOADING,
        MemoryStrategy.CHUNK_PROCESSING,
        MemoryStrategy.CACHE_EVICTION
    ])
    compute_optimizations: List[ComputeOptimization] = field(default_factory=lambda: [
        ComputeOptimization.VECTORIZATION,
        ComputeOptimization.PARALLELIZATION,
        ComputeOptimization.JIT_COMPILATION
    ])
    data_layout: DataLayout = DataLayout.COLUMN_MAJOR
    enable_profiling: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MemoryStats:
    """内存统计信息"""
    timestamp: datetime
    total_memory: float
    used_memory: float
    available_memory: float
    memory_percent: float
    peak_memory: float
    cache_size: int
    active_objects: int
    gc_collections: int
    memory_leaks: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ComputeTask:
    """计算任务"""
    task_id: str
    function: Callable
    args: Tuple
    kwargs: Dict[str, Any]
    memory_limit: Optional[float] = None
    optimization_level: ComputeOptimization = ComputeOptimization.VECTORIZATION
    use_cache: bool = True
    chunk_size: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class MemoryPool:
    """内存池管理器"""
    
    def __init__(self, pool_size: int = 100, object_factory: Optional[Callable] = None):
        self.pool_size = pool_size
        self.object_factory = object_factory
        self.available_objects = []
        self.in_use_objects = weakref.WeakSet()
        self.lock = threading.Lock()
        self.stats = {
            'created': 0,
            'reused': 0,
            'destroyed': 0
        }
    
    def acquire(self) -> Any:
        """获取对象"""
        with self.lock:
            if self.available_objects:
                obj = self.available_objects.pop()
                self.in_use_objects.add(obj)
                self.stats['reused'] += 1
                return obj
            else:
                if self.object_factory:
                    obj = self.object_factory()
                    self.in_use_objects.add(obj)
                    self.stats['created'] += 1
                    return obj
                else:
                    return None
    
    def release(self, obj: Any):
        """释放对象"""
        with self.lock:
            if obj in self.in_use_objects:
                self.in_use_objects.remove(obj)
                
                if len(self.available_objects) < self.pool_size:
                    # 重置对象状态
                    if hasattr(obj, 'reset'):
                        obj.reset()
                    self.available_objects.append(obj)
                else:
                    # 池满，销毁对象
                    if hasattr(obj, 'cleanup'):
                        obj.cleanup()
                    self.stats['destroyed'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self.lock:
            return {
                'pool_size': self.pool_size,
                'available': len(self.available_objects),
                'in_use': len(self.in_use_objects),
                'created': self.stats['created'],
                'reused': self.stats['reused'],
                'destroyed': self.stats['destroyed']
            }

class LazyDataLoader:
    """惰性数据加载器"""
    
    def __init__(self, data_source: Union[str, Path, Callable], 
                 loader_func: Optional[Callable] = None,
                 cache_size: int = 100):
        self.data_source = data_source
        self.loader_func = loader_func
        self.cache = OrderedDict()
        self.cache_size = cache_size
        self._data = None
        self._loaded = False
        self.lock = threading.Lock()
    
    def load(self) -> Any:
        """加载数据"""
        with self.lock:
            if not self._loaded:
                if callable(self.data_source):
                    self._data = self.data_source()
                elif self.loader_func:
                    self._data = self.loader_func(self.data_source)
                else:
                    # 默认加载逻辑
                    self._data = self._default_load()
                
                self._loaded = True
            
            return self._data
    
    def _default_load(self) -> Any:
        """默认加载逻辑"""
        if isinstance(self.data_source, (str, Path)):
            path = Path(self.data_source)
            if path.suffix == '.csv':
                return pd.read_csv(path)
            elif path.suffix == '.parquet':
                return pd.read_parquet(path)
            elif path.suffix == '.pickle':
                return pd.read_pickle(path)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
        else:
            return self.data_source
    
    def get_chunk(self, start: int, end: int) -> Any:
        """获取数据块"""
        cache_key = (start, end)
        
        if cache_key in self.cache:
            # 移到末尾（LRU策略）
            self.cache.move_to_end(cache_key)
            return self.cache[cache_key]
        
        # 加载完整数据
        data = self.load()
        
        # 提取数据块
        chunk = data.iloc[start:end] if hasattr(data, 'iloc') else data[start:end]
        
        # 缓存数据块
        self.cache[cache_key] = chunk
        
        # 清理旧缓存
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        
        return chunk
    
    def unload(self):
        """卸载数据"""
        with self.lock:
            self._data = None
            self._loaded = False
            self.cache.clear()

class MemoryMappedArray:
    """内存映射数组"""
    
    def __init__(self, file_path: Union[str, Path], 
                 shape: Tuple[int, ...], 
                 dtype: np.dtype = np.float64,
                 mode: str = 'r+'):
        self.file_path = Path(file_path)
        self.shape = shape
        self.dtype = dtype
        self.mode = mode
        self.size = np.prod(shape) * np.dtype(dtype).itemsize
        
        # 创建内存映射文件
        self._create_memmap_file()
        
        # 创建内存映射数组
        self.array = np.memmap(
            self.file_path, 
            dtype=dtype, 
            mode=mode, 
            shape=shape
        )
    
    def _create_memmap_file(self):
        """创建内存映射文件"""
        if not self.file_path.exists():
            # 创建文件
            with open(self.file_path, 'wb') as f:
                f.write(b'\x00' * self.size)
    
    def __getitem__(self, key):
        return self.array[key]
    
    def __setitem__(self, key, value):
        self.array[key] = value
    
    def flush(self):
        """刷新到磁盘"""
        self.array.flush()
    
    def close(self):
        """关闭映射"""
        del self.array
        self.array = None
    
    def __del__(self):
        if hasattr(self, 'array') and self.array is not None:
            self.close()

class ChunkedDataProcessor:
    """分块数据处理器"""
    
    def __init__(self, data: Any, chunk_size: int = 10000):
        self.data = data
        self.chunk_size = chunk_size
        self.total_size = len(data) if hasattr(data, '__len__') else 0
        self.current_chunk = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        start_idx = self.current_chunk * self.chunk_size
        end_idx = min(start_idx + self.chunk_size, self.total_size)
        
        if start_idx >= self.total_size:
            raise StopIteration
        
        chunk = self.data[start_idx:end_idx]
        self.current_chunk += 1
        
        return chunk
    
    def process_chunks(self, func: Callable, *args, **kwargs) -> Generator[Any, None, None]:
        """处理数据块"""
        for chunk in self:
            yield func(chunk, *args, **kwargs)
    
    def reduce_chunks(self, func: Callable, reducer: Callable, 
                     initial_value: Any = None) -> Any:
        """归约数据块"""
        result = initial_value
        
        for chunk in self:
            chunk_result = func(chunk)
            if result is None:
                result = chunk_result
            else:
                result = reducer(result, chunk_result)
        
        return result

class MemoryOptimizedCompute:
    """内存优化计算引擎"""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 内存管理
        self.memory_pools = {}
        self.lazy_loaders = {}
        self.memory_maps = {}
        self.cache = OrderedDict()
        
        # 性能监控
        self.memory_stats = []
        self.profiler = None
        
        # 垃圾回收
        self.gc_enabled = config.enable_gc_optimization
        self.gc_threshold = config.gc_threshold
        
        # 运行状态
        self.is_running = False
        self.monitor_thread = None
        self.stop_event = threading.Event()
        
        # 初始化
        self._initialize_memory_pools()
        self._initialize_profiler()
        self._configure_gc()
    
    def _initialize_memory_pools(self):
        """初始化内存池"""
        if MemoryStrategy.OBJECT_POOLING in self.config.memory_strategies:
            # 创建常用对象的内存池
            self.memory_pools['dataframe'] = MemoryPool(
                pool_size=self.config.pool_size,
                object_factory=lambda: pd.DataFrame()
            )
            
            self.memory_pools['array'] = MemoryPool(
                pool_size=self.config.pool_size,
                object_factory=lambda: np.empty(0)
            )
            
            self.memory_pools['dict'] = MemoryPool(
                pool_size=self.config.pool_size,
                object_factory=lambda: {}
            )
    
    def _initialize_profiler(self):
        """初始化性能分析器"""
        if self.config.enable_profiling and PYMPLER_AVAILABLE:
            self.profiler = tracker.SummaryTracker()
    
    def _configure_gc(self):
        """配置垃圾回收"""
        if self.config.enable_gc_optimization:
            # 调整垃圾回收阈值
            import gc
            gc.set_threshold(700, 10, 10)
            
            # 启用增量垃圾回收
            if hasattr(gc, 'set_debug'):
                gc.set_debug(gc.DEBUG_STATS)
    
    def start(self):
        """启动内存优化引擎"""
        if self.is_running:
            return
        
        self.is_running = True
        self.stop_event.clear()
        
        # 启动内存监控线程
        self.monitor_thread = threading.Thread(
            target=self._memory_monitor_loop,
            daemon=True
        )
        self.monitor_thread.start()
        
        self.logger.info("Memory optimized compute engine started")
    
    def stop(self):
        """停止内存优化引擎"""
        if not self.is_running:
            return
        
        self.logger.info("Stopping memory optimized compute engine...")
        self.stop_event.set()
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        # 清理资源
        self._cleanup_resources()
        
        self.is_running = False
        self.logger.info("Memory optimized compute engine stopped")
    
    def _memory_monitor_loop(self):
        """内存监控循环"""
        while not self.stop_event.is_set():
            try:
                # 收集内存统计
                stats = self._collect_memory_stats()
                self.memory_stats.append(stats)
                
                # 保持历史记录大小
                if len(self.memory_stats) > 1000:
                    self.memory_stats = self.memory_stats[-1000:]
                
                # 检查内存使用
                if stats.memory_percent > self.config.gc_threshold * 100:
                    self._trigger_memory_cleanup()
                
                time.sleep(30)  # 30秒监控间隔
                
            except Exception as e:
                self.logger.error(f"Error in memory monitor: {e}")
                time.sleep(5)
    
    def _collect_memory_stats(self) -> MemoryStats:
        """收集内存统计信息"""
        process = psutil.Process()
        memory_info = process.memory_info()
        virtual_memory = psutil.virtual_memory()
        
        stats = MemoryStats(
            timestamp=datetime.now(),
            total_memory=virtual_memory.total / 1024 / 1024,  # MB
            used_memory=memory_info.rss / 1024 / 1024,  # MB
            available_memory=virtual_memory.available / 1024 / 1024,  # MB
            memory_percent=process.memory_percent(),
            peak_memory=getattr(process, 'memory_info_ex', lambda: memory_info)().peak_wset / 1024 / 1024 if hasattr(process, 'memory_info_ex') else 0,
            cache_size=len(self.cache),
            active_objects=len(gc.get_objects()),
            gc_collections=sum(gc.get_stats()) if hasattr(gc, 'get_stats') else 0
        )
        
        return stats
    
    def _trigger_memory_cleanup(self):
        """触发内存清理"""
        self.logger.info("Triggering memory cleanup...")
        
        # 清理缓存
        self._cleanup_cache()
        
        # 清理对象池
        self._cleanup_pools()
        
        # 强制垃圾回收
        gc.collect()
        
        self.logger.info("Memory cleanup completed")
    
    def _cleanup_cache(self):
        """清理缓存"""
        cache_size_before = len(self.cache)
        
        # 清理一半的缓存项
        items_to_remove = len(self.cache) // 2
        
        for _ in range(items_to_remove):
            if self.cache:
                self.cache.popitem(last=False)
        
        self.logger.info(f"Cleaned cache: {cache_size_before} -> {len(self.cache)}")
    
    def _cleanup_pools(self):
        """清理对象池"""
        for pool_name, pool in self.memory_pools.items():
            stats_before = pool.get_stats()
            
            # 清理一半的可用对象
            with pool.lock:
                items_to_remove = len(pool.available_objects) // 2
                for _ in range(items_to_remove):
                    if pool.available_objects:
                        obj = pool.available_objects.pop()
                        if hasattr(obj, 'cleanup'):
                            obj.cleanup()
                        pool.stats['destroyed'] += 1
            
            stats_after = pool.get_stats()
            self.logger.info(f"Cleaned pool {pool_name}: {stats_before['available']} -> {stats_after['available']}")
    
    @contextmanager
    def memory_limit(self, max_memory: str):
        """内存限制上下文管理器"""
        original_limit = self.config.max_memory_usage
        self.config.max_memory_usage = max_memory
        
        try:
            yield
        finally:
            self.config.max_memory_usage = original_limit
    
    @contextmanager
    def object_pool(self, pool_name: str):
        """对象池上下文管理器"""
        obj = None
        try:
            if pool_name in self.memory_pools:
                obj = self.memory_pools[pool_name].acquire()
            yield obj
        finally:
            if obj is not None and pool_name in self.memory_pools:
                self.memory_pools[pool_name].release(obj)
    
    def create_lazy_loader(self, data_source: Union[str, Path, Callable], 
                          loader_id: str,
                          loader_func: Optional[Callable] = None) -> LazyDataLoader:
        """创建惰性加载器"""
        loader = LazyDataLoader(data_source, loader_func, self.config.cache_size)
        self.lazy_loaders[loader_id] = loader
        return loader
    
    def create_memory_map(self, file_path: Union[str, Path], 
                         shape: Tuple[int, ...],
                         dtype: np.dtype = np.float64,
                         map_id: str = None) -> MemoryMappedArray:
        """创建内存映射数组"""
        if map_id is None:
            map_id = str(uuid.uuid4())
        
        mem_map = MemoryMappedArray(file_path, shape, dtype)
        self.memory_maps[map_id] = mem_map
        return mem_map
    
    def process_chunked(self, data: Any, 
                       func: Callable, 
                       chunk_size: Optional[int] = None,
                       **kwargs) -> Generator[Any, None, None]:
        """分块处理数据"""
        chunk_size = chunk_size or self.config.chunk_size
        processor = ChunkedDataProcessor(data, chunk_size)
        
        for chunk_result in processor.process_chunks(func, **kwargs):
            yield chunk_result
    
    def reduce_chunked(self, data: Any, 
                      func: Callable, 
                      reducer: Callable,
                      initial_value: Any = None,
                      chunk_size: Optional[int] = None) -> Any:
        """分块归约数据"""
        chunk_size = chunk_size or self.config.chunk_size
        processor = ChunkedDataProcessor(data, chunk_size)
        
        return processor.reduce_chunks(func, reducer, initial_value)
    
    @lru_cache(maxsize=1000)
    def cached_compute(self, func: Callable, *args, **kwargs) -> Any:
        """缓存计算结果"""
        return func(*args, **kwargs)
    
    def optimized_compute(self, task: ComputeTask) -> Any:
        """优化计算"""
        start_time = time.time()
        
        try:
            # 应用内存优化策略
            if MemoryStrategy.LAZY_LOADING in self.config.memory_strategies:
                # 惰性加载参数
                args = self._apply_lazy_loading(task.args)
                kwargs = self._apply_lazy_loading(task.kwargs)
            else:
                args = task.args
                kwargs = task.kwargs
            
            # 应用计算优化
            if task.optimization_level == ComputeOptimization.JIT_COMPILATION and NUMBA_AVAILABLE:
                # JIT编译优化
                optimized_func = self._apply_jit_optimization(task.function)
            else:
                optimized_func = task.function
            
            # 分块处理
            if task.chunk_size and hasattr(args[0], '__len__'):
                result = self.reduce_chunked(
                    args[0], 
                    lambda chunk: optimized_func(chunk, *args[1:], **kwargs),
                    self._get_reducer_func(task.function),
                    chunk_size=task.chunk_size
                )
            else:
                result = optimized_func(*args, **kwargs)
            
            # 缓存结果
            if task.use_cache:
                cache_key = self._generate_cache_key(task.task_id, args, kwargs)
                self.cache[cache_key] = result
                
                # 清理旧缓存
                while len(self.cache) > self.config.cache_size:
                    self.cache.popitem(last=False)
            
            execution_time = time.time() - start_time
            self.logger.debug(f"Task {task.task_id} completed in {execution_time:.3f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in optimized compute: {e}")
            raise
    
    def _apply_lazy_loading(self, obj: Any) -> Any:
        """应用惰性加载"""
        if isinstance(obj, (list, tuple)):
            return type(obj)(self._apply_lazy_loading(item) for item in obj)
        elif isinstance(obj, dict):
            return {k: self._apply_lazy_loading(v) for k, v in obj.items()}
        elif isinstance(obj, (str, Path)) and Path(obj).exists():
            # 创建惰性加载器
            loader_id = f"lazy_{hash(str(obj))}"
            if loader_id not in self.lazy_loaders:
                self.create_lazy_loader(obj, loader_id)
            return self.lazy_loaders[loader_id]
        else:
            return obj
    
    def _apply_jit_optimization(self, func: Callable) -> Callable:
        """应用JIT编译优化"""
        if NUMBA_AVAILABLE:
            try:
                return jit(nopython=True)(func)
            except Exception as e:
                self.logger.warning(f"JIT compilation failed: {e}")
                return func
        else:
            return func
    
    def _get_reducer_func(self, func: Callable) -> Callable:
        """获取归约函数"""
        func_name = func.__name__ if hasattr(func, '__name__') else str(func)
        
        if 'sum' in func_name:
            return lambda x, y: x + y
        elif 'mean' in func_name:
            return lambda x, y: (x + y) / 2
        elif 'max' in func_name:
            return lambda x, y: max(x, y)
        elif 'min' in func_name:
            return lambda x, y: min(x, y)
        else:
            return lambda x, y: y  # 默认返回最后一个结果
    
    def _generate_cache_key(self, task_id: str, args: Tuple, kwargs: Dict) -> str:
        """生成缓存键"""
        import hashlib
        
        # 简化的缓存键生成
        key_data = f"{task_id}_{hash(args)}_{hash(frozenset(kwargs.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _cleanup_resources(self):
        """清理资源"""
        # 清理惰性加载器
        for loader in self.lazy_loaders.values():
            loader.unload()
        self.lazy_loaders.clear()
        
        # 清理内存映射
        for mem_map in self.memory_maps.values():
            mem_map.close()
        self.memory_maps.clear()
        
        # 清理缓存
        self.cache.clear()
        
        # 清理对象池
        self.memory_pools.clear()
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """获取内存使用情况"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss': memory_info.rss / 1024 / 1024,  # MB
            'vms': memory_info.vms / 1024 / 1024,  # MB
            'percent': process.memory_percent(),
            'cache_size': len(self.cache),
            'lazy_loaders': len(self.lazy_loaders),
            'memory_maps': len(self.memory_maps),
            'object_pools': len(self.memory_pools)
        }
    
    def get_memory_stats(self) -> List[MemoryStats]:
        """获取内存统计历史"""
        return self.memory_stats
    
    def profile_memory(self, func: Callable, *args, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """内存性能分析"""
        if not self.config.enable_profiling:
            return func(*args, **kwargs), {}
        
        start_memory = psutil.Process().memory_info().rss
        
        if PYMPLER_AVAILABLE and self.profiler:
            self.profiler.print_diff()
        
        result = func(*args, **kwargs)
        
        end_memory = psutil.Process().memory_info().rss
        memory_delta = (end_memory - start_memory) / 1024 / 1024  # MB
        
        profile_info = {
            'memory_delta': memory_delta,
            'start_memory': start_memory / 1024 / 1024,
            'end_memory': end_memory / 1024 / 1024
        }
        
        if PYMPLER_AVAILABLE and self.profiler:
            profile_info['memory_diff'] = self.profiler.format_diff()
        
        return result, profile_info
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        pool_stats = {}
        for pool_name, pool in self.memory_pools.items():
            pool_stats[pool_name] = pool.get_stats()
        
        return {
            'is_running': self.is_running,
            'memory_usage': self.get_memory_usage(),
            'memory_strategies': [s.value for s in self.config.memory_strategies],
            'compute_optimizations': [o.value for o in self.config.compute_optimizations],
            'pool_stats': pool_stats,
            'cache_hit_rate': self._calculate_cache_hit_rate(),
            'gc_enabled': self.gc_enabled,
            'profiling_enabled': self.config.enable_profiling
        }
    
    def _calculate_cache_hit_rate(self) -> float:
        """计算缓存命中率"""
        # 简化的缓存命中率计算
        if hasattr(self.cached_compute, 'cache_info'):
            cache_info = self.cached_compute.cache_info()
            total_calls = cache_info.hits + cache_info.misses
            return cache_info.hits / total_calls if total_calls > 0 else 0.0
        return 0.0
    
    def __del__(self):
        """析构函数"""
        if hasattr(self, 'is_running') and self.is_running:
            self.stop()


# 内存优化装饰器
def memory_optimized(config: Optional[MemoryConfig] = None,
                    chunk_size: Optional[int] = None,
                    use_cache: bool = True,
                    memory_limit: Optional[str] = None):
    """内存优化装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 创建计算任务
            task = ComputeTask(
                task_id=f"{func.__name__}_{int(time.time())}",
                function=func,
                args=args,
                kwargs=kwargs,
                chunk_size=chunk_size,
                use_cache=use_cache
            )
            
            # 创建内存优化引擎
            engine_config = config or MemoryConfig()
            if memory_limit:
                engine_config.max_memory_usage = memory_limit
            
            engine = MemoryOptimizedCompute(engine_config)
            
            try:
                engine.start()
                return engine.optimized_compute(task)
            finally:
                engine.stop()
        
        return wrapper
    return decorator


# 内存监控装饰器
def monitor_memory(func):
    """内存监控装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_memory = psutil.Process().memory_info().rss
        
        result = func(*args, **kwargs)
        
        end_memory = psutil.Process().memory_info().rss
        memory_delta = (end_memory - start_memory) / 1024 / 1024  # MB
        
        logging.getLogger(__name__).info(
            f"Function {func.__name__} used {memory_delta:.2f} MB memory"
        )
        
        return result
    
    return wrapper


# 高性能NumPy函数
if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True)
    def fast_array_sum(arr: np.ndarray) -> float:
        """快速数组求和"""
        return np.sum(arr)
    
    @jit(nopython=True, parallel=True)
    def fast_array_mean(arr: np.ndarray) -> float:
        """快速数组平均值"""
        return np.mean(arr)
    
    @jit(nopython=True, parallel=True)
    def fast_array_std(arr: np.ndarray) -> float:
        """快速数组标准差"""
        return np.std(arr)
    
    @jit(nopython=True, parallel=True)
    def fast_rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
        """快速滚动平均"""
        n = len(arr)
        result = np.empty(n)
        
        for i in prange(n):
            start = max(0, i - window + 1)
            end = i + 1
            result[i] = np.mean(arr[start:end])
        
        return result
    
    @jit(nopython=True, parallel=True)
    def fast_correlation(x: np.ndarray, y: np.ndarray) -> float:
        """快速相关系数计算"""
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x * x)
        sum_y2 = np.sum(y * y)
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator = np.sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y))
        
        return numerator / denominator if denominator != 0 else 0.0

else:
    # 回退到标准NumPy实现
    def fast_array_sum(arr: np.ndarray) -> float:
        return np.sum(arr)
    
    def fast_array_mean(arr: np.ndarray) -> float:
        return np.mean(arr)
    
    def fast_array_std(arr: np.ndarray) -> float:
        return np.std(arr)
    
    def fast_rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
        return pd.Series(arr).rolling(window=window).mean().values
    
    def fast_correlation(x: np.ndarray, y: np.ndarray) -> float:
        return np.corrcoef(x, y)[0, 1]
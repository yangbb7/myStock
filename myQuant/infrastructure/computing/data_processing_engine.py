import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Union, Callable, Generator
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
import time
import pickle
import json
import uuid
from pathlib import Path
import queue
import gc
from functools import wraps, lru_cache
import numba
from numba import jit, prange, cuda
import cupy as cp
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import dask.dataframe as dd
from dask.array import Array as DaskArray
import vaex
import modin.pandas as mpd
import ray
from memory_profiler import profile
import psutil
import warnings
warnings.filterwarnings('ignore')

class ProcessingEngine(Enum):
    PANDAS = "pandas"
    POLARS = "polars"
    DASK = "dask"
    VAEX = "vaex"
    MODIN = "modin"
    CUDF = "cudf"
    ARROW = "arrow"
    NUMPY = "numpy"
    NUMBA = "numba"

class DataFormat(Enum):
    CSV = "csv"
    PARQUET = "parquet"
    FEATHER = "feather"
    HDF5 = "hdf5"
    PICKLE = "pickle"
    JSON = "json"
    AVRO = "avro"
    ORC = "orc"

class ComputeBackend(Enum):
    CPU = "cpu"
    GPU = "gpu"
    MULTI_GPU = "multi_gpu"
    DISTRIBUTED = "distributed"

class OptimizationLevel(Enum):
    NONE = 0
    BASIC = 1
    ADVANCED = 2
    AGGRESSIVE = 3

@dataclass
class ProcessingConfig:
    """数据处理配置"""
    engine: ProcessingEngine
    compute_backend: ComputeBackend
    optimization_level: OptimizationLevel
    chunk_size: int = 10000
    batch_size: int = 1000
    max_memory_usage: str = "2GB"
    enable_caching: bool = True
    cache_size: int = 1000
    enable_parallel: bool = True
    n_workers: int = 4
    enable_gpu: bool = False
    gpu_memory_limit: str = "1GB"
    enable_profiling: bool = False
    enable_lazy_evaluation: bool = True
    compression_level: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProcessingMetrics:
    """处理指标"""
    operation_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None
    gpu_usage: Optional[float] = None
    rows_processed: Optional[int] = None
    bytes_processed: Optional[int] = None
    cache_hits: int = 0
    cache_misses: int = 0
    optimization_applied: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DataPipeline:
    """数据处理管道"""
    pipeline_id: str
    operations: List[Dict[str, Any]]
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    enable_validation: bool = True
    enable_monitoring: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

class HighPerformanceDataEngine:
    """高性能数据处理引擎"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 处理引擎
        self.current_engine = None
        self.engine_instances = {}
        
        # 缓存系统
        self.cache = {} if config.enable_caching else None
        self.cache_lock = threading.Lock()
        
        # 任务队列
        self.task_queue = queue.Queue()
        self.result_cache = {}
        
        # 性能监控
        self.metrics_history = []
        self.profiling_enabled = config.enable_profiling
        
        # 运行状态
        self.is_running = False
        self.worker_pool = None
        self.process_pool = None
        
        # 内存监控
        self.memory_monitor = MemoryMonitor()
        
        # 优化器
        self.query_optimizer = QueryOptimizer()
        
        # 初始化
        self._initialize_engine()
        self._initialize_workers()
        self._initialize_gpu_support()
    
    def _initialize_engine(self):
        """初始化数据处理引擎"""
        try:
            if self.config.engine == ProcessingEngine.PANDAS:
                self._initialize_pandas()
            elif self.config.engine == ProcessingEngine.POLARS:
                self._initialize_polars()
            elif self.config.engine == ProcessingEngine.DASK:
                self._initialize_dask()
            elif self.config.engine == ProcessingEngine.VAEX:
                self._initialize_vaex()
            elif self.config.engine == ProcessingEngine.MODIN:
                self._initialize_modin()
            elif self.config.engine == ProcessingEngine.CUDF:
                self._initialize_cudf()
            elif self.config.engine == ProcessingEngine.ARROW:
                self._initialize_arrow()
            elif self.config.engine == ProcessingEngine.NUMPY:
                self._initialize_numpy()
            elif self.config.engine == ProcessingEngine.NUMBA:
                self._initialize_numba()
            
            self.logger.info(f"Initialized {self.config.engine.value} engine")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize engine: {e}")
            raise
    
    def _initialize_pandas(self):
        """初始化Pandas引擎"""
        # 配置Pandas性能选项
        pd.set_option('compute.use_bottleneck', True)
        pd.set_option('compute.use_numexpr', True)
        pd.set_option('mode.chained_assignment', None)
        
        # 设置内存使用限制
        memory_limit = self._parse_memory_limit(self.config.max_memory_usage)
        pd.set_option('display.memory_usage', 'deep')
        
        self.current_engine = pd
    
    def _initialize_polars(self):
        """初始化Polars引擎"""
        # Polars配置
        pl.Config.set_tbl_rows(20)
        pl.Config.set_tbl_cols(10)
        pl.Config.set_streaming_chunk_size(self.config.chunk_size)
        
        self.current_engine = pl
    
    def _initialize_dask(self):
        """初始化Dask引擎"""
        import dask
        from dask.distributed import Client
        
        # Dask配置
        dask.config.set({
            'array.chunk-size': f"{self.config.chunk_size}MiB",
            'distributed.worker.memory.target': 0.8,
            'distributed.worker.memory.spill': 0.9,
            'distributed.worker.memory.pause': 0.95,
            'distributed.worker.memory.terminate': 0.98
        })
        
        # 创建客户端
        if self.config.compute_backend == ComputeBackend.DISTRIBUTED:
            self.dask_client = Client(n_workers=self.config.n_workers)
        else:
            self.dask_client = Client(processes=False, threads_per_worker=2)
        
        self.current_engine = dd
    
    def _initialize_vaex(self):
        """初始化Vaex引擎"""
        # Vaex配置
        vaex.settings.main.cache_size_bytes = self._parse_memory_limit(self.config.max_memory_usage)
        
        self.current_engine = vaex
    
    def _initialize_modin(self):
        """初始化Modin引擎"""
        import modin.config as cfg
        
        # Modin配置
        cfg.Engine.put('ray')
        cfg.NPartitions.put(self.config.n_workers)
        
        self.current_engine = mpd
    
    def _initialize_cudf(self):
        """初始化cuDF引擎"""
        try:
            import cudf
            import cupy as cp
            
            # GPU内存设置
            gpu_memory_limit = self._parse_memory_limit(self.config.gpu_memory_limit)
            cp.cuda.MemoryPool().set_limit(size=gpu_memory_limit)
            
            self.current_engine = cudf
            
        except ImportError:
            self.logger.warning("cuDF not available, falling back to pandas")
            self._initialize_pandas()
    
    def _initialize_arrow(self):
        """初始化Arrow引擎"""
        # Arrow配置
        pa.set_cpu_count(self.config.n_workers)
        pa.set_io_thread_count(self.config.n_workers)
        
        self.current_engine = pa
    
    def _initialize_numpy(self):
        """初始化NumPy引擎"""
        # NumPy配置
        if self.config.enable_parallel:
            import os
            os.environ['OMP_NUM_THREADS'] = str(self.config.n_workers)
            os.environ['MKL_NUM_THREADS'] = str(self.config.n_workers)
            os.environ['NUMEXPR_NUM_THREADS'] = str(self.config.n_workers)
        
        self.current_engine = np
    
    def _initialize_numba(self):
        """初始化Numba引擎"""
        # Numba配置
        numba.set_num_threads(self.config.n_workers)
        
        if self.config.enable_gpu:
            # 检查CUDA可用性
            if cuda.is_available():
                self.logger.info(f"CUDA devices: {cuda.list_devices()}")
            else:
                self.logger.warning("CUDA not available")
        
        self.current_engine = numba
    
    def _initialize_workers(self):
        """初始化工作线程"""
        if self.config.enable_parallel:
            self.worker_pool = ThreadPoolExecutor(
                max_workers=self.config.n_workers,
                thread_name_prefix="data_engine"
            )
            
            self.process_pool = ProcessPoolExecutor(
                max_workers=min(self.config.n_workers, 4)
            )
    
    def _initialize_gpu_support(self):
        """初始化GPU支持"""
        if self.config.enable_gpu:
            try:
                import cupy as cp
                
                # 检查GPU可用性
                if cp.cuda.is_available():
                    self.gpu_available = True
                    self.gpu_count = cp.cuda.runtime.getDeviceCount()
                    self.logger.info(f"GPU support enabled: {self.gpu_count} devices")
                else:
                    self.gpu_available = False
                    self.logger.warning("GPU not available")
            except ImportError:
                self.gpu_available = False
                self.logger.warning("CuPy not installed, GPU support disabled")
    
    def _parse_memory_limit(self, memory_str: str) -> int:
        """解析内存限制字符串"""
        if memory_str.endswith('GB'):
            return int(float(memory_str[:-2]) * 1024 * 1024 * 1024)
        elif memory_str.endswith('MB'):
            return int(float(memory_str[:-2]) * 1024 * 1024)
        elif memory_str.endswith('KB'):
            return int(float(memory_str[:-2]) * 1024)
        else:
            return int(memory_str)
    
    def start(self):
        """启动数据处理引擎"""
        if self.is_running:
            return
        
        self.is_running = True
        self.logger.info("High-performance data engine started")
    
    def stop(self):
        """停止数据处理引擎"""
        if not self.is_running:
            return
        
        self.logger.info("Stopping high-performance data engine...")
        
        # 关闭工作线程
        if self.worker_pool:
            self.worker_pool.shutdown(wait=True)
        
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
        
        # 关闭Dask客户端
        if hasattr(self, 'dask_client'):
            self.dask_client.close()
        
        self.is_running = False
        self.logger.info("High-performance data engine stopped")
    
    @profile
    def load_data(self, data_source: Union[str, Path, Dict[str, Any]], 
                  format: DataFormat = DataFormat.CSV,
                  **kwargs) -> Any:
        """加载数据"""
        start_time = datetime.now()
        operation_id = f"load_{uuid.uuid4().hex[:8]}"
        
        try:
            # 缓存检查
            cache_key = f"load_{hash(str(data_source))}"
            if self.cache and cache_key in self.cache:
                return self._get_cached_result(cache_key)
            
            # 根据引擎类型加载数据
            if self.config.engine == ProcessingEngine.PANDAS:
                data = self._load_pandas(data_source, format, **kwargs)
            elif self.config.engine == ProcessingEngine.POLARS:
                data = self._load_polars(data_source, format, **kwargs)
            elif self.config.engine == ProcessingEngine.DASK:
                data = self._load_dask(data_source, format, **kwargs)
            elif self.config.engine == ProcessingEngine.VAEX:
                data = self._load_vaex(data_source, format, **kwargs)
            elif self.config.engine == ProcessingEngine.CUDF:
                data = self._load_cudf(data_source, format, **kwargs)
            elif self.config.engine == ProcessingEngine.ARROW:
                data = self._load_arrow(data_source, format, **kwargs)
            else:
                raise ValueError(f"Unsupported engine: {self.config.engine}")
            
            # 缓存结果
            if self.cache:
                self._cache_result(cache_key, data)
            
            # 记录指标
            self._record_metrics(operation_id, start_time, data)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    def _load_pandas(self, data_source: Union[str, Path], format: DataFormat, **kwargs) -> pd.DataFrame:
        """使用Pandas加载数据"""
        if format == DataFormat.CSV:
            return pd.read_csv(data_source, **kwargs)
        elif format == DataFormat.PARQUET:
            return pd.read_parquet(data_source, **kwargs)
        elif format == DataFormat.FEATHER:
            return pd.read_feather(data_source, **kwargs)
        elif format == DataFormat.HDF5:
            return pd.read_hdf(data_source, **kwargs)
        elif format == DataFormat.PICKLE:
            return pd.read_pickle(data_source, **kwargs)
        elif format == DataFormat.JSON:
            return pd.read_json(data_source, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _load_polars(self, data_source: Union[str, Path], format: DataFormat, **kwargs) -> pl.DataFrame:
        """使用Polars加载数据"""
        if format == DataFormat.CSV:
            return pl.read_csv(data_source, **kwargs)
        elif format == DataFormat.PARQUET:
            return pl.read_parquet(data_source, **kwargs)
        elif format == DataFormat.JSON:
            return pl.read_json(data_source, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _load_dask(self, data_source: Union[str, Path], format: DataFormat, **kwargs) -> dd.DataFrame:
        """使用Dask加载数据"""
        if format == DataFormat.CSV:
            return dd.read_csv(data_source, **kwargs)
        elif format == DataFormat.PARQUET:
            return dd.read_parquet(data_source, **kwargs)
        elif format == DataFormat.JSON:
            return dd.read_json(data_source, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _load_vaex(self, data_source: Union[str, Path], format: DataFormat, **kwargs) -> vaex.DataFrame:
        """使用Vaex加载数据"""
        if format == DataFormat.CSV:
            return vaex.from_csv(data_source, **kwargs)
        elif format == DataFormat.PARQUET:
            return vaex.open(data_source, **kwargs)
        elif format == DataFormat.HDF5:
            return vaex.open(data_source, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _load_cudf(self, data_source: Union[str, Path], format: DataFormat, **kwargs):
        """使用cuDF加载数据"""
        import cudf
        
        if format == DataFormat.CSV:
            return cudf.read_csv(data_source, **kwargs)
        elif format == DataFormat.PARQUET:
            return cudf.read_parquet(data_source, **kwargs)
        elif format == DataFormat.JSON:
            return cudf.read_json(data_source, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _load_arrow(self, data_source: Union[str, Path], format: DataFormat, **kwargs) -> pa.Table:
        """使用Arrow加载数据"""
        if format == DataFormat.CSV:
            return pa.csv.read_csv(data_source, **kwargs)
        elif format == DataFormat.PARQUET:
            return pq.read_table(data_source, **kwargs)
        elif format == DataFormat.FEATHER:
            return pa.feather.read_table(data_source, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def process_data(self, data: Any, operations: List[Dict[str, Any]], 
                    pipeline_id: Optional[str] = None) -> Any:
        """处理数据"""
        start_time = datetime.now()
        operation_id = f"process_{uuid.uuid4().hex[:8]}"
        
        try:
            # 创建数据管道
            if pipeline_id:
                pipeline = DataPipeline(
                    pipeline_id=pipeline_id,
                    operations=operations
                )
            
            # 优化查询
            if self.config.optimization_level != OptimizationLevel.NONE:
                operations = self.query_optimizer.optimize(operations, self.config.optimization_level)
            
            # 应用操作
            result = data
            for operation in operations:
                result = self._apply_operation(result, operation)
            
            # 记录指标
            self._record_metrics(operation_id, start_time, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing data: {e}")
            raise
    
    def _apply_operation(self, data: Any, operation: Dict[str, Any]) -> Any:
        """应用单个操作"""
        op_type = operation.get('type')
        params = operation.get('params', {})
        
        if op_type == 'filter':
            return self._apply_filter(data, params)
        elif op_type == 'select':
            return self._apply_select(data, params)
        elif op_type == 'groupby':
            return self._apply_groupby(data, params)
        elif op_type == 'join':
            return self._apply_join(data, params)
        elif op_type == 'sort':
            return self._apply_sort(data, params)
        elif op_type == 'aggregate':
            return self._apply_aggregate(data, params)
        elif op_type == 'transform':
            return self._apply_transform(data, params)
        elif op_type == 'window':
            return self._apply_window(data, params)
        else:
            raise ValueError(f"Unsupported operation: {op_type}")
    
    def _apply_filter(self, data: Any, params: Dict[str, Any]) -> Any:
        """应用过滤操作"""
        condition = params.get('condition')
        
        if self.config.engine == ProcessingEngine.PANDAS:
            return data.query(condition) if isinstance(condition, str) else data[condition]
        elif self.config.engine == ProcessingEngine.POLARS:
            return data.filter(pl.expr(condition))
        elif self.config.engine == ProcessingEngine.DASK:
            return data.query(condition) if isinstance(condition, str) else data[condition]
        elif self.config.engine == ProcessingEngine.VAEX:
            return data[condition]
        else:
            raise ValueError(f"Filter not supported for {self.config.engine}")
    
    def _apply_select(self, data: Any, params: Dict[str, Any]) -> Any:
        """应用选择操作"""
        columns = params.get('columns')
        
        if self.config.engine == ProcessingEngine.PANDAS:
            return data[columns]
        elif self.config.engine == ProcessingEngine.POLARS:
            return data.select(columns)
        elif self.config.engine == ProcessingEngine.DASK:
            return data[columns]
        elif self.config.engine == ProcessingEngine.VAEX:
            return data[columns]
        else:
            raise ValueError(f"Select not supported for {self.config.engine}")
    
    def _apply_groupby(self, data: Any, params: Dict[str, Any]) -> Any:
        """应用分组操作"""
        by = params.get('by')
        agg_func = params.get('agg', 'mean')
        
        if self.config.engine == ProcessingEngine.PANDAS:
            return data.groupby(by).agg(agg_func)
        elif self.config.engine == ProcessingEngine.POLARS:
            return data.group_by(by).agg(getattr(pl.col("*"), agg_func)())
        elif self.config.engine == ProcessingEngine.DASK:
            return data.groupby(by).agg(agg_func)
        elif self.config.engine == ProcessingEngine.VAEX:
            return data.groupby(by).agg(agg_func)
        else:
            raise ValueError(f"GroupBy not supported for {self.config.engine}")
    
    def _apply_join(self, data: Any, params: Dict[str, Any]) -> Any:
        """应用连接操作"""
        right = params.get('right')
        on = params.get('on')
        how = params.get('how', 'inner')
        
        if self.config.engine == ProcessingEngine.PANDAS:
            return data.merge(right, on=on, how=how)
        elif self.config.engine == ProcessingEngine.POLARS:
            return data.join(right, on=on, how=how)
        elif self.config.engine == ProcessingEngine.DASK:
            return data.merge(right, on=on, how=how)
        elif self.config.engine == ProcessingEngine.VAEX:
            return data.join(right, on=on, how=how)
        else:
            raise ValueError(f"Join not supported for {self.config.engine}")
    
    def _apply_sort(self, data: Any, params: Dict[str, Any]) -> Any:
        """应用排序操作"""
        by = params.get('by')
        ascending = params.get('ascending', True)
        
        if self.config.engine == ProcessingEngine.PANDAS:
            return data.sort_values(by, ascending=ascending)
        elif self.config.engine == ProcessingEngine.POLARS:
            return data.sort(by, descending=not ascending)
        elif self.config.engine == ProcessingEngine.DASK:
            return data.sort_values(by, ascending=ascending)
        elif self.config.engine == ProcessingEngine.VAEX:
            return data.sort(by, ascending=ascending)
        else:
            raise ValueError(f"Sort not supported for {self.config.engine}")
    
    def _apply_aggregate(self, data: Any, params: Dict[str, Any]) -> Any:
        """应用聚合操作"""
        agg_func = params.get('func')
        
        if self.config.engine == ProcessingEngine.PANDAS:
            return data.agg(agg_func)
        elif self.config.engine == ProcessingEngine.POLARS:
            return data.select(getattr(pl.col("*"), agg_func)())
        elif self.config.engine == ProcessingEngine.DASK:
            return data.agg(agg_func)
        elif self.config.engine == ProcessingEngine.VAEX:
            return getattr(data, agg_func)()
        else:
            raise ValueError(f"Aggregate not supported for {self.config.engine}")
    
    def _apply_transform(self, data: Any, params: Dict[str, Any]) -> Any:
        """应用转换操作"""
        func = params.get('func')
        columns = params.get('columns')
        
        if self.config.engine == ProcessingEngine.PANDAS:
            if columns:
                return data.assign(**{col: data[col].transform(func) for col in columns})
            else:
                return data.transform(func)
        elif self.config.engine == ProcessingEngine.POLARS:
            if columns:
                return data.with_columns([pl.col(col).map(func).alias(col) for col in columns])
            else:
                return data.map(func)
        else:
            raise ValueError(f"Transform not supported for {self.config.engine}")
    
    def _apply_window(self, data: Any, params: Dict[str, Any]) -> Any:
        """应用窗口操作"""
        window_size = params.get('window_size')
        func = params.get('func')
        
        if self.config.engine == ProcessingEngine.PANDAS:
            return data.rolling(window=window_size).agg(func)
        elif self.config.engine == ProcessingEngine.POLARS:
            return data.rolling(window_size).agg(getattr(pl.col("*"), func)())
        else:
            raise ValueError(f"Window not supported for {self.config.engine}")
    
    @lru_cache(maxsize=1000)
    def _get_cached_result(self, cache_key: str) -> Any:
        """获取缓存结果"""
        with self.cache_lock:
            return self.cache.get(cache_key)
    
    def _cache_result(self, cache_key: str, result: Any):
        """缓存结果"""
        if len(self.cache) >= self.config.cache_size:
            # 删除最旧的缓存项
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        with self.cache_lock:
            self.cache[cache_key] = result
    
    def _record_metrics(self, operation_id: str, start_time: datetime, result: Any):
        """记录性能指标"""
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # 计算数据大小
        rows_processed = None
        bytes_processed = None
        
        if hasattr(result, 'shape'):
            rows_processed = result.shape[0]
        if hasattr(result, 'memory_usage'):
            bytes_processed = result.memory_usage(deep=True).sum()
        
        # 获取系统指标
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        cpu_usage = process.cpu_percent()
        
        metrics = ProcessingMetrics(
            operation_id=operation_id,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            rows_processed=rows_processed,
            bytes_processed=bytes_processed
        )
        
        self.metrics_history.append(metrics)
        
        # 保持历史记录大小
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
    
    def save_data(self, data: Any, file_path: Union[str, Path], 
                  format: DataFormat = DataFormat.PARQUET, **kwargs):
        """保存数据"""
        try:
            if format == DataFormat.CSV:
                if hasattr(data, 'to_csv'):
                    data.to_csv(file_path, **kwargs)
                else:
                    raise ValueError("Data does not support CSV export")
            elif format == DataFormat.PARQUET:
                if hasattr(data, 'to_parquet'):
                    data.to_parquet(file_path, **kwargs)
                else:
                    raise ValueError("Data does not support Parquet export")
            elif format == DataFormat.FEATHER:
                if hasattr(data, 'to_feather'):
                    data.to_feather(file_path, **kwargs)
                else:
                    raise ValueError("Data does not support Feather export")
            elif format == DataFormat.PICKLE:
                if hasattr(data, 'to_pickle'):
                    data.to_pickle(file_path, **kwargs)
                else:
                    with open(file_path, 'wb') as f:
                        pickle.dump(data, f)
            elif format == DataFormat.JSON:
                if hasattr(data, 'to_json'):
                    data.to_json(file_path, **kwargs)
                else:
                    with open(file_path, 'w') as f:
                        json.dump(data, f, **kwargs)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Data saved to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving data: {e}")
            raise
    
    def get_performance_metrics(self) -> List[ProcessingMetrics]:
        """获取性能指标"""
        return self.metrics_history
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """获取内存使用情况"""
        return self.memory_monitor.get_memory_usage()
    
    def optimize_memory(self):
        """优化内存使用"""
        # 清理缓存
        if self.cache:
            with self.cache_lock:
                self.cache.clear()
        
        # 强制垃圾回收
        gc.collect()
        
        # GPU内存清理
        if self.config.enable_gpu:
            try:
                import cupy as cp
                cp.get_default_memory_pool().free_all_blocks()
            except ImportError:
                pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_operations = len(self.metrics_history)
        total_duration = sum(m.duration for m in self.metrics_history if m.duration)
        avg_duration = total_duration / total_operations if total_operations > 0 else 0
        
        return {
            'engine': self.config.engine.value,
            'compute_backend': self.config.compute_backend.value,
            'is_running': self.is_running,
            'total_operations': total_operations,
            'total_duration': total_duration,
            'average_duration': avg_duration,
            'cache_size': len(self.cache) if self.cache else 0,
            'cache_hit_rate': self._calculate_cache_hit_rate(),
            'memory_usage': self.get_memory_usage(),
            'gpu_available': getattr(self, 'gpu_available', False),
            'optimization_level': self.config.optimization_level.value
        }
    
    def _calculate_cache_hit_rate(self) -> float:
        """计算缓存命中率"""
        if not self.metrics_history:
            return 0.0
        
        total_hits = sum(m.cache_hits for m in self.metrics_history)
        total_misses = sum(m.cache_misses for m in self.metrics_history)
        total_requests = total_hits + total_misses
        
        return total_hits / total_requests if total_requests > 0 else 0.0
    
    def __del__(self):
        """析构函数"""
        if hasattr(self, 'is_running') and self.is_running:
            self.stop()


class MemoryMonitor:
    """内存监控器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """获取内存使用情况"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss': memory_info.rss / 1024 / 1024,  # MB
            'vms': memory_info.vms / 1024 / 1024,  # MB
            'percent': process.memory_percent(),
            'available': psutil.virtual_memory().available / 1024 / 1024,  # MB
            'total': psutil.virtual_memory().total / 1024 / 1024  # MB
        }


class QueryOptimizer:
    """查询优化器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def optimize(self, operations: List[Dict[str, Any]], 
                optimization_level: OptimizationLevel) -> List[Dict[str, Any]]:
        """优化查询操作"""
        optimized_ops = operations.copy()
        
        if optimization_level == OptimizationLevel.NONE:
            return optimized_ops
        
        # 基础优化
        if optimization_level >= OptimizationLevel.BASIC:
            optimized_ops = self._push_down_filters(optimized_ops)
            optimized_ops = self._combine_filters(optimized_ops)
        
        # 高级优化
        if optimization_level >= OptimizationLevel.ADVANCED:
            optimized_ops = self._reorder_operations(optimized_ops)
            optimized_ops = self._optimize_joins(optimized_ops)
        
        # 激进优化
        if optimization_level >= OptimizationLevel.AGGRESSIVE:
            optimized_ops = self._eliminate_redundant_operations(optimized_ops)
            optimized_ops = self._vectorize_operations(optimized_ops)
        
        return optimized_ops
    
    def _push_down_filters(self, operations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """下推过滤器"""
        # 将过滤操作尽可能移到前面
        filters = [op for op in operations if op.get('type') == 'filter']
        non_filters = [op for op in operations if op.get('type') != 'filter']
        
        return filters + non_filters
    
    def _combine_filters(self, operations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """合并过滤器"""
        combined_ops = []
        current_filter = None
        
        for op in operations:
            if op.get('type') == 'filter':
                if current_filter is None:
                    current_filter = op
                else:
                    # 合并条件
                    current_condition = current_filter['params']['condition']
                    new_condition = op['params']['condition']
                    current_filter['params']['condition'] = f"({current_condition}) & ({new_condition})"
            else:
                if current_filter is not None:
                    combined_ops.append(current_filter)
                    current_filter = None
                combined_ops.append(op)
        
        if current_filter is not None:
            combined_ops.append(current_filter)
        
        return combined_ops
    
    def _reorder_operations(self, operations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """重新排序操作"""
        # 按照优化顺序排序：filter -> select -> sort -> groupby -> join
        order_priority = {
            'filter': 1,
            'select': 2,
            'sort': 3,
            'groupby': 4,
            'join': 5,
            'aggregate': 6,
            'transform': 7,
            'window': 8
        }
        
        return sorted(operations, key=lambda x: order_priority.get(x.get('type'), 999))
    
    def _optimize_joins(self, operations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """优化连接操作"""
        # 简单的连接优化逻辑
        for op in operations:
            if op.get('type') == 'join':
                # 可以添加连接顺序优化、索引建议等
                pass
        
        return operations
    
    def _eliminate_redundant_operations(self, operations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """消除冗余操作"""
        # 移除重复的操作
        seen = set()
        unique_ops = []
        
        for op in operations:
            op_signature = f"{op.get('type')}_{hash(str(op.get('params', {})))}"
            if op_signature not in seen:
                seen.add(op_signature)
                unique_ops.append(op)
        
        return unique_ops
    
    def _vectorize_operations(self, operations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """向量化操作"""
        # 标记可以向量化的操作
        for op in operations:
            if op.get('type') in ['transform', 'aggregate', 'window']:
                op['vectorized'] = True
        
        return operations


# 高性能计算装饰器
def high_performance_compute(engine: ProcessingEngine = ProcessingEngine.PANDAS,
                           use_gpu: bool = False,
                           cache_result: bool = True):
    """高性能计算装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 性能监控
            start_time = time.time()
            
            # 执行函数
            result = func(*args, **kwargs)
            
            # 记录性能
            duration = time.time() - start_time
            
            return result
        
        return wrapper
    return decorator


# Numba优化函数
@jit(nopython=True, parallel=True)
def fast_moving_average(data: np.ndarray, window: int) -> np.ndarray:
    """快速移动平均计算"""
    result = np.empty_like(data)
    
    for i in prange(len(data)):
        if i < window - 1:
            result[i] = np.mean(data[:i+1])
        else:
            result[i] = np.mean(data[i-window+1:i+1])
    
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


# GPU加速函数
def gpu_matrix_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """GPU矩阵乘法"""
    try:
        import cupy as cp
        
        # 转换到GPU
        a_gpu = cp.asarray(a)
        b_gpu = cp.asarray(b)
        
        # 执行矩阵乘法
        result_gpu = cp.dot(a_gpu, b_gpu)
        
        # 转换回CPU
        return cp.asnumpy(result_gpu)
        
    except ImportError:
        # 回退到CPU
        return np.dot(a, b)
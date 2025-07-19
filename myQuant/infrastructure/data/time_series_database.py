import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, Column, DateTime, Float, String, Integer, Boolean, Index, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import JSONB
import psycopg2
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
import redis
import pickle
import threading
import time
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

class DatabaseType(Enum):
    TIMESCALE = "timescale"
    INFLUXDB = "influxdb"
    REDIS = "redis"
    POSTGRESQL = "postgresql"
    HYBRID = "hybrid"

class CompressionType(Enum):
    NONE = "none"
    GZIP = "gzip"
    SNAPPY = "snappy"
    LZ4 = "lz4"
    ZSTD = "zstd"

class DataRetentionPolicy(Enum):
    MINUTES = "minutes"
    HOURS = "hours"
    DAYS = "days"
    WEEKS = "weeks"
    MONTHS = "months"
    YEARS = "years"

class AggregationMethod(Enum):
    MEAN = "mean"
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    FIRST = "first"
    LAST = "last"
    STDDEV = "stddev"
    MEDIAN = "median"
    PERCENTILE = "percentile"

@dataclass
class TimeSeriesPoint:
    """时间序列数据点"""
    timestamp: datetime
    metric_name: str
    value: float
    tags: Dict[str, str] = field(default_factory=dict)
    fields: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RetentionRule:
    """数据保留规则"""
    duration: int
    period: DataRetentionPolicy
    aggregation: AggregationMethod
    compression: CompressionType
    description: str = ""

@dataclass
class QueryOptions:
    """查询选项"""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    aggregation: Optional[AggregationMethod] = None
    group_by: Optional[List[str]] = None
    window: Optional[str] = None
    limit: Optional[int] = None
    offset: Optional[int] = None
    order_by: Optional[str] = None
    descending: bool = False

@dataclass
class DatabaseConfig:
    """数据库配置"""
    db_type: DatabaseType
    connection_string: str
    max_connections: int = 100
    connection_timeout: int = 30
    query_timeout: int = 300
    enable_compression: bool = True
    compression_type: CompressionType = CompressionType.GZIP
    batch_size: int = 1000
    flush_interval: int = 5
    retention_rules: List[RetentionRule] = field(default_factory=list)
    enable_clustering: bool = False
    cluster_nodes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class TimeSeriesDatabase:
    """时间序列数据库"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 连接池
        self.connections: Dict[str, Any] = {}
        self.connection_pool = None
        
        # 数据缓冲
        self.write_buffer = []
        self.buffer_lock = threading.Lock()
        
        # 运行状态
        self.is_running = False
        self.flush_thread = None
        self.stop_event = threading.Event()
        
        # 统计信息
        self.stats = {
            'points_written': 0,
            'points_read': 0,
            'queries_executed': 0,
            'errors': 0,
            'last_flush': None,
            'buffer_size': 0
        }
        
        # 初始化
        self._initialize_database()
        self._setup_retention_policies()
    
    def _initialize_database(self):
        """初始化数据库"""
        try:
            if self.config.db_type == DatabaseType.TIMESCALE:
                self._initialize_timescale()
            elif self.config.db_type == DatabaseType.INFLUXDB:
                self._initialize_influxdb()
            elif self.config.db_type == DatabaseType.REDIS:
                self._initialize_redis()
            elif self.config.db_type == DatabaseType.HYBRID:
                self._initialize_hybrid()
            else:
                raise ValueError(f"Unsupported database type: {self.config.db_type}")
            
            self.logger.info(f"Initialized {self.config.db_type.value} database")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _initialize_timescale(self):
        """初始化TimescaleDB"""
        try:
            self.engine = create_engine(
                self.config.connection_string,
                pool_size=self.config.max_connections,
                max_overflow=20,
                pool_timeout=self.config.connection_timeout,
                pool_recycle=3600
            )
            
            self.SessionLocal = sessionmaker(bind=self.engine)
            self.metadata = MetaData()
            
            # 创建时间序列表
            self.time_series_table = Table(
                'time_series_data',
                self.metadata,
                Column('timestamp', DateTime, primary_key=True),
                Column('metric_name', String(255), primary_key=True),
                Column('value', Float),
                Column('tags', JSONB),
                Column('fields', JSONB),
                Column('metadata', JSONB),
                Index('idx_metric_time', 'metric_name', 'timestamp')
            )
            
            # 创建表
            self.metadata.create_all(self.engine)
            
            # 启用TimescaleDB扩展
            with self.engine.connect() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"))
                
                # 创建超表
                try:
                    conn.execute(text(
                        "SELECT create_hypertable('time_series_data', 'timestamp', if_not_exists => TRUE);"
                    ))
                except Exception as e:
                    self.logger.warning(f"Hypertable might already exist: {e}")
                
                conn.commit()
            
        except Exception as e:
            self.logger.error(f"Error initializing TimescaleDB: {e}")
            raise
    
    def _initialize_influxdb(self):
        """初始化InfluxDB"""
        try:
            # 解析连接字符串
            # Format: influxdb://username:password@host:port/database?org=org&token=token
            import urllib.parse
            parsed = urllib.parse.urlparse(self.config.connection_string)
            
            self.influx_client = InfluxDBClient(
                url=f"{parsed.scheme}://{parsed.hostname}:{parsed.port}",
                token=parsed.fragment.split('token=')[1] if 'token=' in parsed.fragment else None,
                org=parsed.fragment.split('org=')[1].split('&')[0] if 'org=' in parsed.fragment else "myorg"
            )
            
            self.bucket = parsed.path.lstrip('/')
            self.org = parsed.fragment.split('org=')[1].split('&')[0] if 'org=' in parsed.fragment else "myorg"
            
            self.write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)
            self.query_api = self.influx_client.query_api()
            
        except Exception as e:
            self.logger.error(f"Error initializing InfluxDB: {e}")
            raise
    
    def _initialize_redis(self):
        """初始化Redis"""
        try:
            # 解析连接字符串
            import urllib.parse
            parsed = urllib.parse.urlparse(self.config.connection_string)
            
            self.redis_client = redis.Redis(
                host=parsed.hostname,
                port=parsed.port,
                password=parsed.password,
                decode_responses=False
            )
            
            # 测试连接
            self.redis_client.ping()
            
        except Exception as e:
            self.logger.error(f"Error initializing Redis: {e}")
            raise
    
    def _initialize_hybrid(self):
        """初始化混合数据库"""
        try:
            # 实时数据使用Redis
            self._initialize_redis()
            
            # 历史数据使用TimescaleDB
            self._initialize_timescale()
            
            self.logger.info("Initialized hybrid database (Redis + TimescaleDB)")
            
        except Exception as e:
            self.logger.error(f"Error initializing hybrid database: {e}")
            raise
    
    def _setup_retention_policies(self):
        """设置数据保留策略"""
        try:
            if self.config.db_type == DatabaseType.TIMESCALE:
                self._setup_timescale_retention()
            elif self.config.db_type == DatabaseType.INFLUXDB:
                self._setup_influxdb_retention()
            
        except Exception as e:
            self.logger.error(f"Error setting up retention policies: {e}")
    
    def _setup_timescale_retention(self):
        """设置TimescaleDB保留策略"""
        try:
            with self.engine.connect() as conn:
                for rule in self.config.retention_rules:
                    interval = f"{rule.duration} {rule.period.value}"
                    
                    # 设置数据保留策略
                    conn.execute(text(f"""
                        SELECT add_retention_policy('time_series_data', INTERVAL '{interval}', if_not_exists => TRUE);
                    """))
                    
                    # 设置压缩策略
                    if self.config.enable_compression:
                        conn.execute(text(f"""
                            ALTER TABLE time_series_data SET (
                                timescaledb.compress,
                                timescaledb.compress_segmentby = 'metric_name',
                                timescaledb.compress_orderby = 'timestamp'
                            );
                        """))
                        
                        conn.execute(text(f"""
                            SELECT add_compression_policy('time_series_data', INTERVAL '{interval}', if_not_exists => TRUE);
                        """))
                
                conn.commit()
            
        except Exception as e:
            self.logger.error(f"Error setting up TimescaleDB retention: {e}")
    
    def _setup_influxdb_retention(self):
        """设置InfluxDB保留策略"""
        try:
            # InfluxDB v2.x 使用bucket的retention period
            for rule in self.config.retention_rules:
                duration_seconds = self._convert_to_seconds(rule.duration, rule.period)
                
                # 创建bucket with retention
                # 注意：这里需要根据实际的InfluxDB API进行调整
                pass
            
        except Exception as e:
            self.logger.error(f"Error setting up InfluxDB retention: {e}")
    
    def _convert_to_seconds(self, duration: int, period: DataRetentionPolicy) -> int:
        """转换保留时间为秒"""
        multipliers = {
            DataRetentionPolicy.MINUTES: 60,
            DataRetentionPolicy.HOURS: 3600,
            DataRetentionPolicy.DAYS: 86400,
            DataRetentionPolicy.WEEKS: 604800,
            DataRetentionPolicy.MONTHS: 2592000,
            DataRetentionPolicy.YEARS: 31536000
        }
        
        return duration * multipliers.get(period, 86400)
    
    def start(self):
        """启动数据库服务"""
        if self.is_running:
            return
        
        self.is_running = True
        self.stop_event.clear()
        
        # 启动数据刷新线程
        self.flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self.flush_thread.start()
        
        self.logger.info("Time series database started")
    
    def stop(self):
        """停止数据库服务"""
        if not self.is_running:
            return
        
        self.logger.info("Stopping time series database...")
        self.stop_event.set()
        
        # 刷新剩余数据
        self._flush_buffer()
        
        if self.flush_thread:
            self.flush_thread.join(timeout=5)
        
        # 关闭连接
        self._close_connections()
        
        self.is_running = False
        self.logger.info("Time series database stopped")
    
    def _flush_loop(self):
        """数据刷新主循环"""
        while not self.stop_event.is_set():
            try:
                self._flush_buffer()
                time.sleep(self.config.flush_interval)
                
            except Exception as e:
                self.logger.error(f"Error in flush loop: {e}")
                time.sleep(1)
    
    def _flush_buffer(self):
        """刷新缓冲区"""
        if not self.write_buffer:
            return
        
        with self.buffer_lock:
            buffer_to_flush = self.write_buffer.copy()
            self.write_buffer.clear()
        
        try:
            if self.config.db_type == DatabaseType.TIMESCALE:
                self._flush_to_timescale(buffer_to_flush)
            elif self.config.db_type == DatabaseType.INFLUXDB:
                self._flush_to_influxdb(buffer_to_flush)
            elif self.config.db_type == DatabaseType.REDIS:
                self._flush_to_redis(buffer_to_flush)
            elif self.config.db_type == DatabaseType.HYBRID:
                self._flush_to_hybrid(buffer_to_flush)
            
            self.stats['points_written'] += len(buffer_to_flush)
            self.stats['last_flush'] = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error flushing buffer: {e}")
            self.stats['errors'] += 1
            
            # 重新添加到缓冲区
            with self.buffer_lock:
                self.write_buffer.extend(buffer_to_flush)
    
    def _flush_to_timescale(self, points: List[TimeSeriesPoint]):
        """刷新数据到TimescaleDB"""
        try:
            with self.SessionLocal() as session:
                for point in points:
                    session.execute(
                        self.time_series_table.insert().values(
                            timestamp=point.timestamp,
                            metric_name=point.metric_name,
                            value=point.value,
                            tags=point.tags,
                            fields=point.fields,
                            metadata=point.metadata
                        )
                    )
                
                session.commit()
            
        except Exception as e:
            self.logger.error(f"Error flushing to TimescaleDB: {e}")
            raise
    
    def _flush_to_influxdb(self, points: List[TimeSeriesPoint]):
        """刷新数据到InfluxDB"""
        try:
            influx_points = []
            
            for point in points:
                influx_point = Point(point.metric_name) \
                    .time(point.timestamp, WritePrecision.MS) \
                    .field("value", point.value)
                
                # 添加标签
                for key, value in point.tags.items():
                    influx_point = influx_point.tag(key, value)
                
                # 添加字段
                for key, value in point.fields.items():
                    influx_point = influx_point.field(key, value)
                
                influx_points.append(influx_point)
            
            self.write_api.write(bucket=self.bucket, org=self.org, record=influx_points)
            
        except Exception as e:
            self.logger.error(f"Error flushing to InfluxDB: {e}")
            raise
    
    def _flush_to_redis(self, points: List[TimeSeriesPoint]):
        """刷新数据到Redis"""
        try:
            pipe = self.redis_client.pipeline()
            
            for point in points:
                # 使用时间戳作为分数的有序集合
                score = int(point.timestamp.timestamp() * 1000)
                value = pickle.dumps({
                    'value': point.value,
                    'tags': point.tags,
                    'fields': point.fields,
                    'metadata': point.metadata
                })
                
                pipe.zadd(f"ts:{point.metric_name}", {value: score})
            
            pipe.execute()
            
        except Exception as e:
            self.logger.error(f"Error flushing to Redis: {e}")
            raise
    
    def _flush_to_hybrid(self, points: List[TimeSeriesPoint]):
        """刷新数据到混合数据库"""
        try:
            # 实时数据写入Redis
            self._flush_to_redis(points)
            
            # 历史数据写入TimescaleDB（可以异步进行）
            self._flush_to_timescale(points)
            
        except Exception as e:
            self.logger.error(f"Error flushing to hybrid database: {e}")
            raise
    
    def write_point(self, point: TimeSeriesPoint):
        """写入单个数据点"""
        with self.buffer_lock:
            self.write_buffer.append(point)
            self.stats['buffer_size'] = len(self.write_buffer)
        
        # 如果缓冲区满了，立即刷新
        if len(self.write_buffer) >= self.config.batch_size:
            self._flush_buffer()
    
    def write_points(self, points: List[TimeSeriesPoint]):
        """批量写入数据点"""
        with self.buffer_lock:
            self.write_buffer.extend(points)
            self.stats['buffer_size'] = len(self.write_buffer)
        
        # 如果缓冲区满了，立即刷新
        if len(self.write_buffer) >= self.config.batch_size:
            self._flush_buffer()
    
    def query(self, metric_name: str, options: QueryOptions = None) -> pd.DataFrame:
        """查询数据"""
        try:
            if options is None:
                options = QueryOptions()
            
            if self.config.db_type == DatabaseType.TIMESCALE:
                return self._query_timescale(metric_name, options)
            elif self.config.db_type == DatabaseType.INFLUXDB:
                return self._query_influxdb(metric_name, options)
            elif self.config.db_type == DatabaseType.REDIS:
                return self._query_redis(metric_name, options)
            elif self.config.db_type == DatabaseType.HYBRID:
                return self._query_hybrid(metric_name, options)
            
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error querying data: {e}")
            self.stats['errors'] += 1
            return pd.DataFrame()
        finally:
            self.stats['queries_executed'] += 1
    
    def _query_timescale(self, metric_name: str, options: QueryOptions) -> pd.DataFrame:
        """查询TimescaleDB"""
        try:
            query = f"SELECT * FROM time_series_data WHERE metric_name = '{metric_name}'"
            
            # 添加时间条件
            if options.start_time:
                query += f" AND timestamp >= '{options.start_time}'"
            if options.end_time:
                query += f" AND timestamp <= '{options.end_time}'"
            
            # 添加排序
            if options.order_by:
                order_direction = "DESC" if options.descending else "ASC"
                query += f" ORDER BY {options.order_by} {order_direction}"
            else:
                query += " ORDER BY timestamp ASC"
            
            # 添加限制
            if options.limit:
                query += f" LIMIT {options.limit}"
            if options.offset:
                query += f" OFFSET {options.offset}"
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                data = result.fetchall()
                
                if data:
                    columns = result.keys()
                    df = pd.DataFrame(data, columns=columns)
                    self.stats['points_read'] += len(df)
                    return df
                else:
                    return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error querying TimescaleDB: {e}")
            raise
    
    def _query_influxdb(self, metric_name: str, options: QueryOptions) -> pd.DataFrame:
        """查询InfluxDB"""
        try:
            query = f'from(bucket: "{self.bucket}") |> range(start: -1h) |> filter(fn: (r) => r._measurement == "{metric_name}")'
            
            # 添加时间范围
            if options.start_time and options.end_time:
                start_str = options.start_time.strftime('%Y-%m-%dT%H:%M:%SZ')
                end_str = options.end_time.strftime('%Y-%m-%dT%H:%M:%SZ')
                query = f'from(bucket: "{self.bucket}") |> range(start: {start_str}, stop: {end_str}) |> filter(fn: (r) => r._measurement == "{metric_name}")'
            
            # 添加聚合
            if options.aggregation:
                if options.window:
                    query += f' |> aggregateWindow(every: {options.window}, fn: {options.aggregation.value})'
            
            # 添加限制
            if options.limit:
                query += f' |> limit(n: {options.limit})'
            
            result = self.query_api.query_data_frame(query)
            
            if not result.empty:
                self.stats['points_read'] += len(result)
                return result
            else:
                return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error querying InfluxDB: {e}")
            raise
    
    def _query_redis(self, metric_name: str, options: QueryOptions) -> pd.DataFrame:
        """查询Redis"""
        try:
            key = f"ts:{metric_name}"
            
            # 时间范围
            min_score = 0
            max_score = int(time.time() * 1000)
            
            if options.start_time:
                min_score = int(options.start_time.timestamp() * 1000)
            if options.end_time:
                max_score = int(options.end_time.timestamp() * 1000)
            
            # 查询数据
            data = self.redis_client.zrangebyscore(key, min_score, max_score, withscores=True)
            
            if data:
                rows = []
                for value, score in data:
                    timestamp = datetime.fromtimestamp(score / 1000)
                    point_data = pickle.loads(value)
                    
                    row = {
                        'timestamp': timestamp,
                        'metric_name': metric_name,
                        'value': point_data['value'],
                        'tags': point_data['tags'],
                        'fields': point_data['fields'],
                        'metadata': point_data['metadata']
                    }
                    rows.append(row)
                
                df = pd.DataFrame(rows)
                
                # 应用限制
                if options.limit:
                    df = df.head(options.limit)
                
                self.stats['points_read'] += len(df)
                return df
            else:
                return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error querying Redis: {e}")
            raise
    
    def _query_hybrid(self, metric_name: str, options: QueryOptions) -> pd.DataFrame:
        """查询混合数据库"""
        try:
            # 优先查询Redis中的实时数据
            recent_data = self._query_redis(metric_name, options)
            
            # 如果需要更多历史数据，查询TimescaleDB
            if options.start_time and options.start_time < datetime.now() - timedelta(hours=1):
                historical_data = self._query_timescale(metric_name, options)
                
                # 合并数据
                if not recent_data.empty and not historical_data.empty:
                    combined_data = pd.concat([historical_data, recent_data], ignore_index=True)
                    combined_data = combined_data.drop_duplicates(subset=['timestamp'], keep='last')
                    combined_data = combined_data.sort_values('timestamp')
                    return combined_data
                elif not recent_data.empty:
                    return recent_data
                elif not historical_data.empty:
                    return historical_data
            
            return recent_data
            
        except Exception as e:
            self.logger.error(f"Error querying hybrid database: {e}")
            raise
    
    def get_metrics(self) -> List[str]:
        """获取所有指标名称"""
        try:
            if self.config.db_type == DatabaseType.TIMESCALE:
                with self.engine.connect() as conn:
                    result = conn.execute(text("SELECT DISTINCT metric_name FROM time_series_data"))
                    return [row[0] for row in result.fetchall()]
            elif self.config.db_type == DatabaseType.REDIS:
                pattern = "ts:*"
                keys = self.redis_client.keys(pattern)
                return [key.decode('utf-8')[3:] for key in keys]
            else:
                return []
        except Exception as e:
            self.logger.error(f"Error getting metrics: {e}")
            return []
    
    def delete_metric(self, metric_name: str):
        """删除指标数据"""
        try:
            if self.config.db_type == DatabaseType.TIMESCALE:
                with self.engine.connect() as conn:
                    conn.execute(text(f"DELETE FROM time_series_data WHERE metric_name = '{metric_name}'"))
                    conn.commit()
            elif self.config.db_type == DatabaseType.REDIS:
                key = f"ts:{metric_name}"
                self.redis_client.delete(key)
            
            self.logger.info(f"Deleted metric: {metric_name}")
            
        except Exception as e:
            self.logger.error(f"Error deleting metric: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'database_type': self.config.db_type.value,
            'is_running': self.is_running,
            'points_written': self.stats['points_written'],
            'points_read': self.stats['points_read'],
            'queries_executed': self.stats['queries_executed'],
            'errors': self.stats['errors'],
            'buffer_size': self.stats['buffer_size'],
            'last_flush': self.stats['last_flush'].isoformat() if self.stats['last_flush'] else None,
            'total_metrics': len(self.get_metrics())
        }
    
    def _close_connections(self):
        """关闭所有连接"""
        try:
            if hasattr(self, 'influx_client'):
                self.influx_client.close()
            
            if hasattr(self, 'redis_client'):
                self.redis_client.close()
            
            if hasattr(self, 'engine'):
                self.engine.dispose()
            
        except Exception as e:
            self.logger.error(f"Error closing connections: {e}")
    
    def __del__(self):
        """析构函数"""
        if hasattr(self, 'is_running') and self.is_running:
            self.stop()
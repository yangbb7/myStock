import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
import os
import shutil
import threading
import time
import pickle
import json
import gzip
import bz2
import lzma
import zipfile
import tarfile
from pathlib import Path
import hashlib
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# 压缩库
try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False

try:
    import lz4.frame as lz4
    LZ4_AVAILABLE = True
except ImportError:
    LZ4_AVAILABLE = False

try:
    import snappy
    SNAPPY_AVAILABLE = True
except ImportError:
    SNAPPY_AVAILABLE = False

try:
    import blosc
    BLOSC_AVAILABLE = True
except ImportError:
    BLOSC_AVAILABLE = False

# 存储库
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    ARROW_AVAILABLE = True
except ImportError:
    ARROW_AVAILABLE = False

try:
    import tables
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False

class CompressionType(Enum):
    GZIP = "gzip"
    BZIP2 = "bzip2"
    LZMA = "lzma"
    ZSTD = "zstd"
    LZ4 = "lz4"
    SNAPPY = "snappy"
    BLOSC = "blosc"
    ZLIB = "zlib"
    NONE = "none"

class StorageFormat(Enum):
    PARQUET = "parquet"
    HDF5 = "hdf5"
    FEATHER = "feather"
    PICKLE = "pickle"
    CSV = "csv"
    JSON = "json"
    BINARY = "binary"
    DELTA = "delta"

class ArchiveFrequency(Enum):
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"
    CUSTOM = "custom"

class CompressionLevel(Enum):
    FASTEST = 1
    FAST = 3
    BALANCED = 6
    BEST = 9

class DataTier(Enum):
    HOT = "hot"          # 频繁访问，高性能存储
    WARM = "warm"        # 偶尔访问，平衡存储
    COLD = "cold"        # 很少访问，高压缩存储
    FROZEN = "frozen"    # 归档数据，最高压缩

@dataclass
class CompressionConfig:
    """压缩配置"""
    compression_type: CompressionType
    compression_level: CompressionLevel
    storage_format: StorageFormat
    enable_checksums: bool = True
    enable_encryption: bool = False
    encryption_key: Optional[str] = None
    block_size: int = 65536
    chunk_size: int = 1024 * 1024
    buffer_size: int = 8192
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ArchivePolicy:
    """归档策略"""
    policy_id: str
    data_pattern: str
    frequency: ArchiveFrequency
    retention_days: int
    compression_config: CompressionConfig
    target_tier: DataTier
    auto_cleanup: bool = True
    cleanup_delay_days: int = 30
    enable_deduplication: bool = True
    enable_delta_compression: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CompressionStats:
    """压缩统计"""
    original_size: int
    compressed_size: int
    compression_ratio: float
    compression_time: float
    decompression_time: float
    checksum: str
    algorithm: CompressionType
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ArchiveRecord:
    """归档记录"""
    record_id: str
    original_path: str
    archive_path: str
    archive_size: int
    compression_stats: CompressionStats
    archive_date: datetime
    retention_until: datetime
    data_tier: DataTier
    checksum: str
    encrypted: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

class CompressionEngine:
    """压缩引擎"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.compressors = self._initialize_compressors()
    
    def _initialize_compressors(self) -> Dict[CompressionType, Callable]:
        """初始化压缩器"""
        compressors = {
            CompressionType.GZIP: self._compress_gzip,
            CompressionType.BZIP2: self._compress_bzip2,
            CompressionType.LZMA: self._compress_lzma,
            CompressionType.ZLIB: self._compress_zlib,
            CompressionType.NONE: self._compress_none
        }
        
        if ZSTD_AVAILABLE:
            compressors[CompressionType.ZSTD] = self._compress_zstd
        
        if LZ4_AVAILABLE:
            compressors[CompressionType.LZ4] = self._compress_lz4
        
        if SNAPPY_AVAILABLE:
            compressors[CompressionType.SNAPPY] = self._compress_snappy
        
        if BLOSC_AVAILABLE:
            compressors[CompressionType.BLOSC] = self._compress_blosc
        
        return compressors
    
    def compress(self, data: bytes, config: CompressionConfig) -> Tuple[bytes, CompressionStats]:
        """压缩数据"""
        start_time = time.time()
        original_size = len(data)
        
        if config.compression_type not in self.compressors:
            raise ValueError(f"Unsupported compression type: {config.compression_type}")
        
        compressed_data = self.compressors[config.compression_type](data, config)
        compressed_size = len(compressed_data)
        
        compression_time = time.time() - start_time
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
        
        # 计算校验和
        checksum = hashlib.sha256(compressed_data).hexdigest() if config.enable_checksums else ""
        
        # 测试解压缩时间
        decompression_start = time.time()
        self.decompress(compressed_data, config)
        decompression_time = time.time() - decompression_start
        
        stats = CompressionStats(
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=compression_ratio,
            compression_time=compression_time,
            decompression_time=decompression_time,
            checksum=checksum,
            algorithm=config.compression_type,
            timestamp=datetime.now()
        )
        
        return compressed_data, stats
    
    def decompress(self, compressed_data: bytes, config: CompressionConfig) -> bytes:
        """解压缩数据"""
        decompressors = {
            CompressionType.GZIP: self._decompress_gzip,
            CompressionType.BZIP2: self._decompress_bzip2,
            CompressionType.LZMA: self._decompress_lzma,
            CompressionType.ZLIB: self._decompress_zlib,
            CompressionType.NONE: self._decompress_none
        }
        
        if ZSTD_AVAILABLE:
            decompressors[CompressionType.ZSTD] = self._decompress_zstd
        
        if LZ4_AVAILABLE:
            decompressors[CompressionType.LZ4] = self._decompress_lz4
        
        if SNAPPY_AVAILABLE:
            decompressors[CompressionType.SNAPPY] = self._decompress_snappy
        
        if BLOSC_AVAILABLE:
            decompressors[CompressionType.BLOSC] = self._decompress_blosc
        
        if config.compression_type not in decompressors:
            raise ValueError(f"Unsupported compression type: {config.compression_type}")
        
        return decompressors[config.compression_type](compressed_data, config)
    
    def _compress_gzip(self, data: bytes, config: CompressionConfig) -> bytes:
        """GZIP压缩"""
        return gzip.compress(data, compresslevel=config.compression_level.value)
    
    def _decompress_gzip(self, data: bytes, config: CompressionConfig) -> bytes:
        """GZIP解压"""
        return gzip.decompress(data)
    
    def _compress_bzip2(self, data: bytes, config: CompressionConfig) -> bytes:
        """BZIP2压缩"""
        return bz2.compress(data, compresslevel=config.compression_level.value)
    
    def _decompress_bzip2(self, data: bytes, config: CompressionConfig) -> bytes:
        """BZIP2解压"""
        return bz2.decompress(data)
    
    def _compress_lzma(self, data: bytes, config: CompressionConfig) -> bytes:
        """LZMA压缩"""
        return lzma.compress(data, preset=config.compression_level.value)
    
    def _decompress_lzma(self, data: bytes, config: CompressionConfig) -> bytes:
        """LZMA解压"""
        return lzma.decompress(data)
    
    def _compress_zlib(self, data: bytes, config: CompressionConfig) -> bytes:
        """ZLIB压缩"""
        return zlib.compress(data, level=config.compression_level.value)
    
    def _decompress_zlib(self, data: bytes, config: CompressionConfig) -> bytes:
        """ZLIB解压"""
        return zlib.decompress(data)
    
    def _compress_none(self, data: bytes, config: CompressionConfig) -> bytes:
        """不压缩"""
        return data
    
    def _decompress_none(self, data: bytes, config: CompressionConfig) -> bytes:
        """不解压"""
        return data
    
    def _compress_zstd(self, data: bytes, config: CompressionConfig) -> bytes:
        """ZSTD压缩"""
        cctx = zstd.ZstdCompressor(level=config.compression_level.value)
        return cctx.compress(data)
    
    def _decompress_zstd(self, data: bytes, config: CompressionConfig) -> bytes:
        """ZSTD解压"""
        dctx = zstd.ZstdDecompressor()
        return dctx.decompress(data)
    
    def _compress_lz4(self, data: bytes, config: CompressionConfig) -> bytes:
        """LZ4压缩"""
        return lz4.compress(data, compression_level=config.compression_level.value)
    
    def _decompress_lz4(self, data: bytes, config: CompressionConfig) -> bytes:
        """LZ4解压"""
        return lz4.decompress(data)
    
    def _compress_snappy(self, data: bytes, config: CompressionConfig) -> bytes:
        """Snappy压缩"""
        return snappy.compress(data)
    
    def _decompress_snappy(self, data: bytes, config: CompressionConfig) -> bytes:
        """Snappy解压"""
        return snappy.decompress(data)
    
    def _compress_blosc(self, data: bytes, config: CompressionConfig) -> bytes:
        """Blosc压缩"""
        return blosc.compress(data, clevel=config.compression_level.value)
    
    def _decompress_blosc(self, data: bytes, config: CompressionConfig) -> bytes:
        """Blosc解压"""
        return blosc.decompress(data)

class DataArchiver:
    """数据归档器"""
    
    def __init__(self, archive_root: Path, config: Dict[str, Any]):
        self.archive_root = Path(archive_root)
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 确保归档目录存在
        self.archive_root.mkdir(parents=True, exist_ok=True)
        
        # 压缩引擎
        self.compression_engine = CompressionEngine()
        
        # 归档策略
        self.archive_policies: Dict[str, ArchivePolicy] = {}
        
        # 归档记录
        self.archive_records: Dict[str, ArchiveRecord] = {}
        
        # 运行状态
        self.is_running = False
        self.scheduler_thread = None
        self.stop_event = threading.Event()
        
        # 线程池
        self.thread_pool = ThreadPoolExecutor(
            max_workers=config.get('max_workers', 4),
            thread_name_prefix="archiver"
        )
        
        # 统计信息
        self.stats = {
            'files_archived': 0,
            'total_original_size': 0,
            'total_compressed_size': 0,
            'total_compression_time': 0,
            'files_restored': 0,
            'total_restoration_time': 0
        }
        
        # 加载现有记录
        self._load_archive_records()
    
    def add_archive_policy(self, policy: ArchivePolicy):
        """添加归档策略"""
        self.archive_policies[policy.policy_id] = policy
        self.logger.info(f"Added archive policy: {policy.policy_id}")
    
    def remove_archive_policy(self, policy_id: str):
        """移除归档策略"""
        if policy_id in self.archive_policies:
            del self.archive_policies[policy_id]
            self.logger.info(f"Removed archive policy: {policy_id}")
    
    def start(self):
        """启动归档器"""
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
        
        self.logger.info("Data archiver started")
    
    def stop(self):
        """停止归档器"""
        if not self.is_running:
            return
        
        self.logger.info("Stopping data archiver...")
        self.stop_event.set()
        
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=10)
        
        self.thread_pool.shutdown(wait=True)
        
        # 保存记录
        self._save_archive_records()
        
        self.is_running = False
        self.logger.info("Data archiver stopped")
    
    def _scheduler_loop(self):
        """调度器循环"""
        while not self.stop_event.is_set():
            try:
                # 检查归档策略
                for policy in self.archive_policies.values():
                    self._check_policy_conditions(policy)
                
                # 清理过期文件
                self._cleanup_expired_archives()
                
                # 等待下一次检查
                time.sleep(self.config.get('check_interval', 3600))  # 默认1小时
                
            except Exception as e:
                self.logger.error(f"Error in scheduler loop: {e}")
                time.sleep(60)
    
    def _check_policy_conditions(self, policy: ArchivePolicy):
        """检查策略条件"""
        try:
            # 查找匹配的文件
            files_to_archive = self._find_files_to_archive(policy)
            
            if files_to_archive:
                # 批量归档
                self._batch_archive_files(files_to_archive, policy)
                
        except Exception as e:
            self.logger.error(f"Error checking policy {policy.policy_id}: {e}")
    
    def _find_files_to_archive(self, policy: ArchivePolicy) -> List[Path]:
        """查找需要归档的文件"""
        files_to_archive = []
        
        # 解析数据模式
        pattern_path = Path(policy.data_pattern)
        
        if pattern_path.is_absolute():
            search_path = pattern_path.parent
            pattern = pattern_path.name
        else:
            search_path = Path.cwd()
            pattern = policy.data_pattern
        
        # 查找匹配文件
        for file_path in search_path.glob(pattern):
            if file_path.is_file():
                # 检查是否需要归档
                if self._should_archive_file(file_path, policy):
                    files_to_archive.append(file_path)
        
        return files_to_archive
    
    def _should_archive_file(self, file_path: Path, policy: ArchivePolicy) -> bool:
        """判断是否需要归档文件"""
        # 检查文件是否已被归档
        if str(file_path) in [record.original_path for record in self.archive_records.values()]:
            return False
        
        # 检查文件修改时间
        file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
        current_time = datetime.now()
        
        # 根据频率判断
        if policy.frequency == ArchiveFrequency.HOURLY:
            return (current_time - file_mtime).total_seconds() > 3600
        elif policy.frequency == ArchiveFrequency.DAILY:
            return (current_time - file_mtime).days >= 1
        elif policy.frequency == ArchiveFrequency.WEEKLY:
            return (current_time - file_mtime).days >= 7
        elif policy.frequency == ArchiveFrequency.MONTHLY:
            return (current_time - file_mtime).days >= 30
        elif policy.frequency == ArchiveFrequency.YEARLY:
            return (current_time - file_mtime).days >= 365
        
        return False
    
    def _batch_archive_files(self, files: List[Path], policy: ArchivePolicy):
        """批量归档文件"""
        futures = []
        
        for file_path in files:
            future = self.thread_pool.submit(
                self._archive_single_file,
                file_path,
                policy
            )
            futures.append(future)
        
        # 等待完成
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    self.stats['files_archived'] += 1
                    self.logger.info(f"Archived file: {result.original_path}")
            except Exception as e:
                self.logger.error(f"Error archiving file: {e}")
    
    def _archive_single_file(self, file_path: Path, policy: ArchivePolicy) -> Optional[ArchiveRecord]:
        """归档单个文件"""
        try:
            # 生成归档路径
            archive_path = self._generate_archive_path(file_path, policy)
            
            # 确保目录存在
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 读取文件数据
            with open(file_path, 'rb') as f:
                data = f.read()
            
            # 压缩数据
            compressed_data, compression_stats = self.compression_engine.compress(
                data,
                policy.compression_config
            )
            
            # 加密数据（如果需要）
            if policy.compression_config.enable_encryption:
                compressed_data = self._encrypt_data(compressed_data, policy.compression_config)
            
            # 写入归档文件
            with open(archive_path, 'wb') as f:
                f.write(compressed_data)
            
            # 创建归档记录
            record = ArchiveRecord(
                record_id=self._generate_record_id(),
                original_path=str(file_path),
                archive_path=str(archive_path),
                archive_size=len(compressed_data),
                compression_stats=compression_stats,
                archive_date=datetime.now(),
                retention_until=datetime.now() + timedelta(days=policy.retention_days),
                data_tier=policy.target_tier,
                checksum=compression_stats.checksum,
                encrypted=policy.compression_config.enable_encryption
            )
            
            # 保存记录
            self.archive_records[record.record_id] = record
            
            # 更新统计
            self.stats['total_original_size'] += compression_stats.original_size
            self.stats['total_compressed_size'] += compression_stats.compressed_size
            self.stats['total_compression_time'] += compression_stats.compression_time
            
            # 删除原文件（如果配置了自动清理）
            if policy.auto_cleanup:
                cleanup_time = datetime.now() + timedelta(days=policy.cleanup_delay_days)
                # 这里可以添加延迟删除逻辑
                pass
            
            return record
            
        except Exception as e:
            self.logger.error(f"Error archiving file {file_path}: {e}")
            return None
    
    def _generate_archive_path(self, file_path: Path, policy: ArchivePolicy) -> Path:
        """生成归档路径"""
        # 基于时间和策略生成路径
        now = datetime.now()
        
        tier_dir = policy.target_tier.value
        date_dir = now.strftime("%Y/%m/%d")
        
        # 生成文件名
        original_name = file_path.stem
        timestamp = now.strftime("%H%M%S")
        extension = self._get_archive_extension(policy.compression_config)
        
        archive_name = f"{original_name}_{timestamp}{extension}"
        
        return self.archive_root / tier_dir / date_dir / archive_name
    
    def _get_archive_extension(self, config: CompressionConfig) -> str:
        """获取归档文件扩展名"""
        format_extensions = {
            StorageFormat.PARQUET: ".parquet",
            StorageFormat.HDF5: ".h5",
            StorageFormat.FEATHER: ".feather",
            StorageFormat.PICKLE: ".pkl",
            StorageFormat.CSV: ".csv",
            StorageFormat.JSON: ".json",
            StorageFormat.BINARY: ".bin",
            StorageFormat.DELTA: ".delta"
        }
        
        compression_extensions = {
            CompressionType.GZIP: ".gz",
            CompressionType.BZIP2: ".bz2",
            CompressionType.LZMA: ".xz",
            CompressionType.ZSTD: ".zst",
            CompressionType.LZ4: ".lz4",
            CompressionType.SNAPPY: ".snappy",
            CompressionType.BLOSC: ".blosc",
            CompressionType.ZLIB: ".zlib",
            CompressionType.NONE: ""
        }
        
        base_ext = format_extensions.get(config.storage_format, ".bin")
        comp_ext = compression_extensions.get(config.compression_type, "")
        
        return base_ext + comp_ext
    
    def _generate_record_id(self) -> str:
        """生成记录ID"""
        return hashlib.sha256(f"{datetime.now().isoformat()}_{os.getpid()}".encode()).hexdigest()[:16]
    
    def _encrypt_data(self, data: bytes, config: CompressionConfig) -> bytes:
        """加密数据"""
        if not config.enable_encryption or not config.encryption_key:
            return data
        
        try:
            from cryptography.fernet import Fernet
            
            # 使用提供的密钥或生成新密钥
            if config.encryption_key:
                key = config.encryption_key.encode()
                # 确保密钥长度正确
                key = hashlib.sha256(key).digest()
                key = Fernet.generate_key()  # 简化实现
            else:
                key = Fernet.generate_key()
            
            fernet = Fernet(key)
            return fernet.encrypt(data)
            
        except ImportError:
            self.logger.warning("Cryptography library not available, skipping encryption")
            return data
    
    def _decrypt_data(self, data: bytes, config: CompressionConfig) -> bytes:
        """解密数据"""
        if not config.enable_encryption or not config.encryption_key:
            return data
        
        try:
            from cryptography.fernet import Fernet
            
            key = config.encryption_key.encode()
            key = hashlib.sha256(key).digest()
            key = Fernet.generate_key()  # 简化实现
            
            fernet = Fernet(key)
            return fernet.decrypt(data)
            
        except ImportError:
            self.logger.warning("Cryptography library not available, skipping decryption")
            return data
    
    def restore_file(self, record_id: str, restore_path: Optional[Path] = None) -> bool:
        """恢复文件"""
        start_time = time.time()
        
        try:
            if record_id not in self.archive_records:
                self.logger.error(f"Archive record not found: {record_id}")
                return False
            
            record = self.archive_records[record_id]
            archive_path = Path(record.archive_path)
            
            if not archive_path.exists():
                self.logger.error(f"Archive file not found: {archive_path}")
                return False
            
            # 确定恢复路径
            if restore_path is None:
                restore_path = Path(record.original_path)
            
            # 读取归档文件
            with open(archive_path, 'rb') as f:
                compressed_data = f.read()
            
            # 解密数据（如果需要）
            if record.encrypted:
                # 这里需要配置信息来解密
                pass
            
            # 解压缩数据
            # 这里需要从记录中恢复压缩配置
            compression_config = CompressionConfig(
                compression_type=record.compression_stats.algorithm,
                compression_level=CompressionLevel.BALANCED,
                storage_format=StorageFormat.BINARY
            )
            
            data = self.compression_engine.decompress(compressed_data, compression_config)
            
            # 确保目录存在
            restore_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 写入恢复的文件
            with open(restore_path, 'wb') as f:
                f.write(data)
            
            # 更新统计
            self.stats['files_restored'] += 1
            self.stats['total_restoration_time'] += time.time() - start_time
            
            self.logger.info(f"Restored file: {restore_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error restoring file {record_id}: {e}")
            return False
    
    def _cleanup_expired_archives(self):
        """清理过期归档"""
        current_time = datetime.now()
        expired_records = []
        
        for record_id, record in self.archive_records.items():
            if record.retention_until < current_time:
                expired_records.append(record_id)
        
        for record_id in expired_records:
            try:
                record = self.archive_records[record_id]
                archive_path = Path(record.archive_path)
                
                if archive_path.exists():
                    archive_path.unlink()
                    self.logger.info(f"Deleted expired archive: {archive_path}")
                
                del self.archive_records[record_id]
                
            except Exception as e:
                self.logger.error(f"Error cleaning up expired archive {record_id}: {e}")
    
    def _load_archive_records(self):
        """加载归档记录"""
        records_file = self.archive_root / "archive_records.json"
        
        if records_file.exists():
            try:
                with open(records_file, 'r') as f:
                    records_data = json.load(f)
                
                for record_id, record_data in records_data.items():
                    # 重建记录对象
                    record = ArchiveRecord(
                        record_id=record_data['record_id'],
                        original_path=record_data['original_path'],
                        archive_path=record_data['archive_path'],
                        archive_size=record_data['archive_size'],
                        compression_stats=CompressionStats(**record_data['compression_stats']),
                        archive_date=datetime.fromisoformat(record_data['archive_date']),
                        retention_until=datetime.fromisoformat(record_data['retention_until']),
                        data_tier=DataTier(record_data['data_tier']),
                        checksum=record_data['checksum'],
                        encrypted=record_data.get('encrypted', False)
                    )
                    
                    self.archive_records[record_id] = record
                
                self.logger.info(f"Loaded {len(self.archive_records)} archive records")
                
            except Exception as e:
                self.logger.error(f"Error loading archive records: {e}")
    
    def _save_archive_records(self):
        """保存归档记录"""
        records_file = self.archive_root / "archive_records.json"
        
        try:
            records_data = {}
            
            for record_id, record in self.archive_records.items():
                records_data[record_id] = {
                    'record_id': record.record_id,
                    'original_path': record.original_path,
                    'archive_path': record.archive_path,
                    'archive_size': record.archive_size,
                    'compression_stats': {
                        'original_size': record.compression_stats.original_size,
                        'compressed_size': record.compression_stats.compressed_size,
                        'compression_ratio': record.compression_stats.compression_ratio,
                        'compression_time': record.compression_stats.compression_time,
                        'decompression_time': record.compression_stats.decompression_time,
                        'checksum': record.compression_stats.checksum,
                        'algorithm': record.compression_stats.algorithm.value,
                        'timestamp': record.compression_stats.timestamp.isoformat()
                    },
                    'archive_date': record.archive_date.isoformat(),
                    'retention_until': record.retention_until.isoformat(),
                    'data_tier': record.data_tier.value,
                    'checksum': record.checksum,
                    'encrypted': record.encrypted
                }
            
            with open(records_file, 'w') as f:
                json.dump(records_data, f, indent=2)
            
            self.logger.info(f"Saved {len(self.archive_records)} archive records")
            
        except Exception as e:
            self.logger.error(f"Error saving archive records: {e}")
    
    def get_archive_statistics(self) -> Dict[str, Any]:
        """获取归档统计信息"""
        total_records = len(self.archive_records)
        total_savings = self.stats['total_original_size'] - self.stats['total_compressed_size']
        avg_compression_ratio = (self.stats['total_original_size'] / self.stats['total_compressed_size']) if self.stats['total_compressed_size'] > 0 else 0
        
        # 按数据层级统计
        tier_stats = {}
        for tier in DataTier:
            tier_records = [r for r in self.archive_records.values() if r.data_tier == tier]
            tier_stats[tier.value] = {
                'count': len(tier_records),
                'total_size': sum(r.archive_size for r in tier_records)
            }
        
        return {
            'total_records': total_records,
            'active_policies': len(self.archive_policies),
            'total_original_size': self.stats['total_original_size'],
            'total_compressed_size': self.stats['total_compressed_size'],
            'total_savings': total_savings,
            'average_compression_ratio': avg_compression_ratio,
            'files_archived': self.stats['files_archived'],
            'files_restored': self.stats['files_restored'],
            'average_compression_time': self.stats['total_compression_time'] / max(self.stats['files_archived'], 1),
            'average_restoration_time': self.stats['total_restoration_time'] / max(self.stats['files_restored'], 1),
            'tier_statistics': tier_stats,
            'is_running': self.is_running
        }
    
    def __del__(self):
        """析构函数"""
        if hasattr(self, 'is_running') and self.is_running:
            self.stop()


class CompressionBenchmark:
    """压缩性能基准测试"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.compression_engine = CompressionEngine()
    
    def benchmark_algorithms(self, data: bytes, 
                           algorithms: List[CompressionType] = None,
                           levels: List[CompressionLevel] = None) -> Dict[str, Any]:
        """基准测试压缩算法"""
        if algorithms is None:
            algorithms = [CompressionType.GZIP, CompressionType.BZIP2, CompressionType.LZMA]
            if ZSTD_AVAILABLE:
                algorithms.append(CompressionType.ZSTD)
            if LZ4_AVAILABLE:
                algorithms.append(CompressionType.LZ4)
        
        if levels is None:
            levels = [CompressionLevel.FASTEST, CompressionLevel.BALANCED, CompressionLevel.BEST]
        
        results = {}
        
        for algorithm in algorithms:
            for level in levels:
                try:
                    config = CompressionConfig(
                        compression_type=algorithm,
                        compression_level=level,
                        storage_format=StorageFormat.BINARY
                    )
                    
                    compressed_data, stats = self.compression_engine.compress(data, config)
                    
                    key = f"{algorithm.value}_{level.value}"
                    results[key] = {
                        'algorithm': algorithm.value,
                        'level': level.value,
                        'original_size': stats.original_size,
                        'compressed_size': stats.compressed_size,
                        'compression_ratio': stats.compression_ratio,
                        'compression_time': stats.compression_time,
                        'decompression_time': stats.decompression_time,
                        'throughput_mb_s': (stats.original_size / 1024 / 1024) / stats.compression_time if stats.compression_time > 0 else 0
                    }
                    
                except Exception as e:
                    self.logger.error(f"Error benchmarking {algorithm.value} level {level.value}: {e}")
        
        return results
    
    def recommend_algorithm(self, data: bytes, 
                          priority: str = "balanced") -> Tuple[CompressionType, CompressionLevel]:
        """推荐压缩算法"""
        benchmark_results = self.benchmark_algorithms(data)
        
        if not benchmark_results:
            return CompressionType.GZIP, CompressionLevel.BALANCED
        
        if priority == "speed":
            # 优先考虑速度
            best_result = min(benchmark_results.values(), 
                            key=lambda x: x['compression_time'])
        elif priority == "ratio":
            # 优先考虑压缩率
            best_result = max(benchmark_results.values(), 
                            key=lambda x: x['compression_ratio'])
        else:
            # 平衡考虑
            best_result = max(benchmark_results.values(), 
                            key=lambda x: x['compression_ratio'] / (x['compression_time'] + 0.001))
        
        return (CompressionType(best_result['algorithm']), 
                CompressionLevel(best_result['level']))


# 使用示例
def create_default_archive_policies() -> List[ArchivePolicy]:
    """创建默认归档策略"""
    policies = []
    
    # 日志文件策略
    log_policy = ArchivePolicy(
        policy_id="log_files",
        data_pattern="logs/*.log",
        frequency=ArchiveFrequency.DAILY,
        retention_days=90,
        compression_config=CompressionConfig(
            compression_type=CompressionType.GZIP,
            compression_level=CompressionLevel.BALANCED,
            storage_format=StorageFormat.BINARY
        ),
        target_tier=DataTier.COLD
    )
    policies.append(log_policy)
    
    # 数据文件策略
    data_policy = ArchivePolicy(
        policy_id="data_files",
        data_pattern="data/*.csv",
        frequency=ArchiveFrequency.WEEKLY,
        retention_days=365,
        compression_config=CompressionConfig(
            compression_type=CompressionType.ZSTD if ZSTD_AVAILABLE else CompressionType.GZIP,
            compression_level=CompressionLevel.BEST,
            storage_format=StorageFormat.PARQUET if ARROW_AVAILABLE else StorageFormat.CSV
        ),
        target_tier=DataTier.WARM
    )
    policies.append(data_policy)
    
    return policies
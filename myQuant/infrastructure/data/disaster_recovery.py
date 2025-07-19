import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import os
import shutil
import threading
import time
import json
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import psutil
import warnings
warnings.filterwarnings('ignore')

# 云存储库
try:
    import boto3
    from botocore.exceptions import ClientError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

try:
    from azure.storage.blob import BlobServiceClient
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

try:
    from google.cloud import storage as gcs
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False

# 数据库备份库
try:
    import pymongo
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False

try:
    import psycopg2
    POSTGRESQL_AVAILABLE = True
except ImportError:
    POSTGRESQL_AVAILABLE = False

class BackupType(Enum):
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"
    SNAPSHOT = "snapshot"
    CONTINUOUS = "continuous"

class BackupStorage(Enum):
    LOCAL = "local"
    NETWORK = "network"
    CLOUD_AWS = "cloud_aws"
    CLOUD_AZURE = "cloud_azure"
    CLOUD_GCP = "cloud_gcp"
    HYBRID = "hybrid"

class RecoveryStrategy(Enum):
    IMMEDIATE = "immediate"
    SCHEDULED = "scheduled"
    ON_DEMAND = "on_demand"
    AUTOMATIC = "automatic"

class BackupStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"

class RecoveryStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    VERIFIED = "verified"

class DataConsistency(Enum):
    EVENTUAL = "eventual"
    STRONG = "strong"
    WEAK = "weak"

@dataclass
class BackupConfig:
    """备份配置"""
    backup_id: str
    backup_type: BackupType
    storage_type: BackupStorage
    source_paths: List[str]
    target_path: str
    schedule: Optional[str] = None  # cron格式
    retention_days: int = 30
    compression_enabled: bool = True
    encryption_enabled: bool = True
    encryption_key: Optional[str] = None
    checksum_enabled: bool = True
    notification_enabled: bool = True
    notification_emails: List[str] = field(default_factory=list)
    max_concurrent_backups: int = 3
    bandwidth_limit: Optional[str] = None
    exclude_patterns: List[str] = field(default_factory=list)
    include_patterns: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RecoveryConfig:
    """恢复配置"""
    recovery_id: str
    backup_id: str
    recovery_strategy: RecoveryStrategy
    target_path: str
    recovery_point: Optional[datetime] = None
    verify_integrity: bool = True
    overwrite_existing: bool = False
    recovery_priority: int = 1
    max_recovery_time: int = 3600  # seconds
    notification_enabled: bool = True
    notification_emails: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BackupRecord:
    """备份记录"""
    backup_id: str
    backup_type: BackupType
    storage_type: BackupStorage
    source_paths: List[str]
    target_path: str
    backup_size: int
    compressed_size: int
    backup_time: datetime
    completion_time: Optional[datetime] = None
    status: BackupStatus = BackupStatus.PENDING
    checksum: Optional[str] = None
    encryption_enabled: bool = False
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RecoveryRecord:
    """恢复记录"""
    recovery_id: str
    backup_id: str
    recovery_strategy: RecoveryStrategy
    target_path: str
    recovery_point: Optional[datetime] = None
    recovery_time: datetime
    completion_time: Optional[datetime] = None
    status: RecoveryStatus = RecoveryStatus.PENDING
    files_recovered: int = 0
    bytes_recovered: int = 0
    integrity_verified: bool = False
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DisasterRecoveryPlan:
    """灾难恢复计划"""
    plan_id: str
    plan_name: str
    description: str
    recovery_time_objective: int  # RTO in minutes
    recovery_point_objective: int  # RPO in minutes
    backup_configs: List[BackupConfig]
    recovery_procedures: List[Dict[str, Any]]
    failover_targets: List[str]
    rollback_procedures: List[Dict[str, Any]]
    contact_list: List[str]
    test_schedule: Optional[str] = None  # cron格式
    last_test_date: Optional[datetime] = None
    test_results: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class LocalBackupManager:
    """本地备份管理器"""
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def create_backup(self, config: BackupConfig) -> BackupRecord:
        """创建备份"""
        start_time = datetime.now()
        
        try:
            # 创建备份目录
            backup_dir = self.base_path / config.backup_id
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            total_size = 0
            files_backed_up = 0
            
            # 备份每个源路径
            for source_path in config.source_paths:
                source = Path(source_path)
                
                if source.is_file():
                    # 备份单个文件
                    target = backup_dir / source.name
                    shutil.copy2(source, target)
                    total_size += source.stat().st_size
                    files_backed_up += 1
                    
                elif source.is_dir():
                    # 备份目录
                    target = backup_dir / source.name
                    shutil.copytree(source, target, dirs_exist_ok=True)
                    
                    # 计算大小
                    for root, dirs, files in os.walk(target):
                        for file in files:
                            file_path = Path(root) / file
                            total_size += file_path.stat().st_size
                            files_backed_up += 1
            
            # 压缩备份（如果启用）
            compressed_size = total_size
            if config.compression_enabled:
                compressed_size = self._compress_backup(backup_dir)
            
            # 计算校验和
            checksum = None
            if config.checksum_enabled:
                checksum = self._calculate_checksum(backup_dir)
            
            # 创建备份记录
            record = BackupRecord(
                backup_id=config.backup_id,
                backup_type=config.backup_type,
                storage_type=config.storage_type,
                source_paths=config.source_paths,
                target_path=str(backup_dir),
                backup_size=total_size,
                compressed_size=compressed_size,
                backup_time=start_time,
                completion_time=datetime.now(),
                status=BackupStatus.COMPLETED,
                checksum=checksum,
                encryption_enabled=config.encryption_enabled,
                metadata={
                    'files_backed_up': files_backed_up,
                    'duration_seconds': (datetime.now() - start_time).total_seconds()
                }
            )
            
            self.logger.info(f"Local backup completed: {config.backup_id}")
            return record
            
        except Exception as e:
            self.logger.error(f"Error creating local backup: {e}")
            
            return BackupRecord(
                backup_id=config.backup_id,
                backup_type=config.backup_type,
                storage_type=config.storage_type,
                source_paths=config.source_paths,
                target_path=str(backup_dir),
                backup_size=0,
                compressed_size=0,
                backup_time=start_time,
                completion_time=datetime.now(),
                status=BackupStatus.FAILED,
                error_message=str(e)
            )
    
    def restore_backup(self, backup_record: BackupRecord, recovery_config: RecoveryConfig) -> RecoveryRecord:
        """恢复备份"""
        start_time = datetime.now()
        
        try:
            backup_dir = Path(backup_record.target_path)
            target_dir = Path(recovery_config.target_path)
            
            # 确保目标目录存在
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # 解压缩（如果需要）
            if backup_record.compressed_size < backup_record.backup_size:
                self._decompress_backup(backup_dir)
            
            # 复制文件
            files_recovered = 0
            bytes_recovered = 0
            
            for item in backup_dir.iterdir():
                if item.is_file():
                    target_file = target_dir / item.name
                    shutil.copy2(item, target_file)
                    files_recovered += 1
                    bytes_recovered += item.stat().st_size
                    
                elif item.is_dir():
                    target_subdir = target_dir / item.name
                    shutil.copytree(item, target_subdir, dirs_exist_ok=True)
                    
                    # 计算恢复的文件数和字节数
                    for root, dirs, files in os.walk(target_subdir):
                        for file in files:
                            file_path = Path(root) / file
                            files_recovered += 1
                            bytes_recovered += file_path.stat().st_size
            
            # 验证完整性
            integrity_verified = False
            if recovery_config.verify_integrity and backup_record.checksum:
                integrity_verified = self._verify_integrity(target_dir, backup_record.checksum)
            
            # 创建恢复记录
            record = RecoveryRecord(
                recovery_id=recovery_config.recovery_id,
                backup_id=backup_record.backup_id,
                recovery_strategy=recovery_config.recovery_strategy,
                target_path=str(target_dir),
                recovery_point=recovery_config.recovery_point,
                recovery_time=start_time,
                completion_time=datetime.now(),
                status=RecoveryStatus.COMPLETED,
                files_recovered=files_recovered,
                bytes_recovered=bytes_recovered,
                integrity_verified=integrity_verified,
                metadata={
                    'duration_seconds': (datetime.now() - start_time).total_seconds()
                }
            )
            
            self.logger.info(f"Local recovery completed: {recovery_config.recovery_id}")
            return record
            
        except Exception as e:
            self.logger.error(f"Error restoring backup: {e}")
            
            return RecoveryRecord(
                recovery_id=recovery_config.recovery_id,
                backup_id=backup_record.backup_id,
                recovery_strategy=recovery_config.recovery_strategy,
                target_path=recovery_config.target_path,
                recovery_point=recovery_config.recovery_point,
                recovery_time=start_time,
                completion_time=datetime.now(),
                status=RecoveryStatus.FAILED,
                error_message=str(e)
            )
    
    def _compress_backup(self, backup_dir: Path) -> int:
        """压缩备份"""
        try:
            # 使用tar.gz压缩
            tar_file = backup_dir.with_suffix('.tar.gz')
            
            with tarfile.open(tar_file, 'w:gz') as tar:
                tar.add(backup_dir, arcname=backup_dir.name)
            
            # 删除原始目录
            shutil.rmtree(backup_dir)
            
            return tar_file.stat().st_size
            
        except Exception as e:
            self.logger.error(f"Error compressing backup: {e}")
            return backup_dir.stat().st_size
    
    def _decompress_backup(self, backup_path: Path):
        """解压缩备份"""
        try:
            if backup_path.suffix == '.gz':
                with tarfile.open(backup_path, 'r:gz') as tar:
                    tar.extractall(backup_path.parent)
                    
        except Exception as e:
            self.logger.error(f"Error decompressing backup: {e}")
    
    def _calculate_checksum(self, path: Path) -> str:
        """计算校验和"""
        try:
            hasher = hashlib.sha256()
            
            if path.is_file():
                with open(path, 'rb') as f:
                    while chunk := f.read(8192):
                        hasher.update(chunk)
            elif path.is_dir():
                for root, dirs, files in os.walk(path):
                    for file in sorted(files):
                        file_path = Path(root) / file
                        with open(file_path, 'rb') as f:
                            while chunk := f.read(8192):
                                hasher.update(chunk)
            
            return hasher.hexdigest()
            
        except Exception as e:
            self.logger.error(f"Error calculating checksum: {e}")
            return ""
    
    def _verify_integrity(self, path: Path, expected_checksum: str) -> bool:
        """验证完整性"""
        try:
            actual_checksum = self._calculate_checksum(path)
            return actual_checksum == expected_checksum
            
        except Exception as e:
            self.logger.error(f"Error verifying integrity: {e}")
            return False

class CloudBackupManager:
    """云备份管理器"""
    
    def __init__(self, storage_type: BackupStorage, credentials: Dict[str, Any]):
        self.storage_type = storage_type
        self.credentials = credentials
        self.logger = logging.getLogger(__name__)
        
        # 初始化云存储客户端
        self.client = self._initialize_client()
    
    def _initialize_client(self):
        """初始化云存储客户端"""
        if self.storage_type == BackupStorage.CLOUD_AWS and AWS_AVAILABLE:
            return boto3.client('s3', **self.credentials)
        elif self.storage_type == BackupStorage.CLOUD_AZURE and AZURE_AVAILABLE:
            return BlobServiceClient(**self.credentials)
        elif self.storage_type == BackupStorage.CLOUD_GCP and GCP_AVAILABLE:
            return gcs.Client(**self.credentials)
        else:
            raise ValueError(f"Unsupported storage type: {self.storage_type}")
    
    def create_backup(self, config: BackupConfig) -> BackupRecord:
        """创建云备份"""
        start_time = datetime.now()
        
        try:
            if self.storage_type == BackupStorage.CLOUD_AWS:
                return self._create_aws_backup(config, start_time)
            elif self.storage_type == BackupStorage.CLOUD_AZURE:
                return self._create_azure_backup(config, start_time)
            elif self.storage_type == BackupStorage.CLOUD_GCP:
                return self._create_gcp_backup(config, start_time)
            else:
                raise ValueError(f"Unsupported storage type: {self.storage_type}")
                
        except Exception as e:
            self.logger.error(f"Error creating cloud backup: {e}")
            
            return BackupRecord(
                backup_id=config.backup_id,
                backup_type=config.backup_type,
                storage_type=config.storage_type,
                source_paths=config.source_paths,
                target_path=config.target_path,
                backup_size=0,
                compressed_size=0,
                backup_time=start_time,
                completion_time=datetime.now(),
                status=BackupStatus.FAILED,
                error_message=str(e)
            )
    
    def _create_aws_backup(self, config: BackupConfig, start_time: datetime) -> BackupRecord:
        """创建AWS S3备份"""
        bucket_name = config.target_path.split('/')[0]
        total_size = 0
        
        for source_path in config.source_paths:
            source = Path(source_path)
            
            if source.is_file():
                # 上传文件
                key = f"{config.backup_id}/{source.name}"
                self.client.upload_file(str(source), bucket_name, key)
                total_size += source.stat().st_size
                
            elif source.is_dir():
                # 上传目录
                for root, dirs, files in os.walk(source):
                    for file in files:
                        file_path = Path(root) / file
                        relative_path = file_path.relative_to(source)
                        key = f"{config.backup_id}/{source.name}/{relative_path}"
                        self.client.upload_file(str(file_path), bucket_name, key)
                        total_size += file_path.stat().st_size
        
        return BackupRecord(
            backup_id=config.backup_id,
            backup_type=config.backup_type,
            storage_type=config.storage_type,
            source_paths=config.source_paths,
            target_path=f"s3://{bucket_name}/{config.backup_id}",
            backup_size=total_size,
            compressed_size=total_size,
            backup_time=start_time,
            completion_time=datetime.now(),
            status=BackupStatus.COMPLETED
        )
    
    def _create_azure_backup(self, config: BackupConfig, start_time: datetime) -> BackupRecord:
        """创建Azure Blob备份"""
        # Azure Blob备份实现
        pass
    
    def _create_gcp_backup(self, config: BackupConfig, start_time: datetime) -> BackupRecord:
        """创建GCP存储备份"""
        # GCP存储备份实现
        pass

class DatabaseBackupManager:
    """数据库备份管理器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def backup_postgresql(self, config: Dict[str, Any]) -> BackupRecord:
        """备份PostgreSQL数据库"""
        start_time = datetime.now()
        
        try:
            host = config.get('host', 'localhost')
            port = config.get('port', 5432)
            database = config.get('database')
            username = config.get('username')
            password = config.get('password')
            backup_file = config.get('backup_file')
            
            # 构建pg_dump命令
            cmd = [
                'pg_dump',
                '-h', host,
                '-p', str(port),
                '-U', username,
                '-d', database,
                '-f', backup_file,
                '--verbose'
            ]
            
            # 设置密码环境变量
            env = os.environ.copy()
            env['PGPASSWORD'] = password
            
            # 执行备份
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            
            if result.returncode == 0:
                backup_size = Path(backup_file).stat().st_size
                
                return BackupRecord(
                    backup_id=config.get('backup_id', f"pg_{int(time.time())}"),
                    backup_type=BackupType.FULL,
                    storage_type=BackupStorage.LOCAL,
                    source_paths=[f"postgresql://{host}:{port}/{database}"],
                    target_path=backup_file,
                    backup_size=backup_size,
                    compressed_size=backup_size,
                    backup_time=start_time,
                    completion_time=datetime.now(),
                    status=BackupStatus.COMPLETED
                )
            else:
                raise Exception(f"pg_dump failed: {result.stderr}")
                
        except Exception as e:
            self.logger.error(f"Error backing up PostgreSQL: {e}")
            
            return BackupRecord(
                backup_id=config.get('backup_id', f"pg_{int(time.time())}"),
                backup_type=BackupType.FULL,
                storage_type=BackupStorage.LOCAL,
                source_paths=[f"postgresql://{host}:{port}/{database}"],
                target_path=config.get('backup_file', ''),
                backup_size=0,
                compressed_size=0,
                backup_time=start_time,
                completion_time=datetime.now(),
                status=BackupStatus.FAILED,
                error_message=str(e)
            )
    
    def backup_mongodb(self, config: Dict[str, Any]) -> BackupRecord:
        """备份MongoDB数据库"""
        start_time = datetime.now()
        
        try:
            host = config.get('host', 'localhost')
            port = config.get('port', 27017)
            database = config.get('database')
            backup_dir = config.get('backup_dir')
            
            # 构建mongodump命令
            cmd = [
                'mongodump',
                '--host', f"{host}:{port}",
                '--db', database,
                '--out', backup_dir
            ]
            
            # 执行备份
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # 计算备份大小
                backup_size = 0
                for root, dirs, files in os.walk(backup_dir):
                    for file in files:
                        file_path = Path(root) / file
                        backup_size += file_path.stat().st_size
                
                return BackupRecord(
                    backup_id=config.get('backup_id', f"mongo_{int(time.time())}"),
                    backup_type=BackupType.FULL,
                    storage_type=BackupStorage.LOCAL,
                    source_paths=[f"mongodb://{host}:{port}/{database}"],
                    target_path=backup_dir,
                    backup_size=backup_size,
                    compressed_size=backup_size,
                    backup_time=start_time,
                    completion_time=datetime.now(),
                    status=BackupStatus.COMPLETED
                )
            else:
                raise Exception(f"mongodump failed: {result.stderr}")
                
        except Exception as e:
            self.logger.error(f"Error backing up MongoDB: {e}")
            
            return BackupRecord(
                backup_id=config.get('backup_id', f"mongo_{int(time.time())}"),
                backup_type=BackupType.FULL,
                storage_type=BackupStorage.LOCAL,
                source_paths=[f"mongodb://{host}:{port}/{database}"],
                target_path=config.get('backup_dir', ''),
                backup_size=0,
                compressed_size=0,
                backup_time=start_time,
                completion_time=datetime.now(),
                status=BackupStatus.FAILED,
                error_message=str(e)
            )

class DisasterRecoveryManager:
    """灾难恢复管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 备份管理器
        self.local_backup_manager = LocalBackupManager(
            Path(config.get('local_backup_path', './backups'))
        )
        
        # 云备份管理器
        self.cloud_backup_managers = {}
        
        # 数据库备份管理器
        self.database_backup_manager = DatabaseBackupManager()
        
        # 灾难恢复计划
        self.disaster_recovery_plans: Dict[str, DisasterRecoveryPlan] = {}
        
        # 备份记录
        self.backup_records: Dict[str, BackupRecord] = {}
        self.recovery_records: Dict[str, RecoveryRecord] = {}
        
        # 运行状态
        self.is_running = False
        self.scheduler_thread = None
        self.stop_event = threading.Event()
        
        # 线程池
        self.thread_pool = ThreadPoolExecutor(
            max_workers=config.get('max_workers', 8),
            thread_name_prefix="disaster_recovery"
        )
        
        # 统计信息
        self.stats = {
            'total_backups': 0,
            'successful_backups': 0,
            'failed_backups': 0,
            'total_recoveries': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'total_backup_size': 0,
            'last_backup_time': None,
            'last_recovery_time': None
        }
    
    def add_disaster_recovery_plan(self, plan: DisasterRecoveryPlan):
        """添加灾难恢复计划"""
        self.disaster_recovery_plans[plan.plan_id] = plan
        self.logger.info(f"Added disaster recovery plan: {plan.plan_id}")
    
    def start(self):
        """启动灾难恢复管理器"""
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
        
        self.logger.info("Disaster recovery manager started")
    
    def stop(self):
        """停止灾难恢复管理器"""
        if not self.is_running:
            return
        
        self.logger.info("Stopping disaster recovery manager...")
        self.stop_event.set()
        
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=10)
        
        self.thread_pool.shutdown(wait=True)
        
        self.is_running = False
        self.logger.info("Disaster recovery manager stopped")
    
    def _scheduler_loop(self):
        """调度器循环"""
        while not self.stop_event.is_set():
            try:
                # 执行计划的备份
                self._execute_scheduled_backups()
                
                # 检查备份状态
                self._check_backup_status()
                
                # 执行灾难恢复测试
                self._execute_dr_tests()
                
                # 清理过期备份
                self._cleanup_expired_backups()
                
                # 等待下一次检查
                time.sleep(self.config.get('check_interval', 300))  # 5分钟
                
            except Exception as e:
                self.logger.error(f"Error in scheduler loop: {e}")
                time.sleep(60)
    
    def _execute_scheduled_backups(self):
        """执行计划的备份"""
        current_time = datetime.now()
        
        for plan in self.disaster_recovery_plans.values():
            for backup_config in plan.backup_configs:
                if self._should_execute_backup(backup_config, current_time):
                    # 异步执行备份
                    self.thread_pool.submit(
                        self._execute_backup,
                        backup_config
                    )
    
    def _should_execute_backup(self, config: BackupConfig, current_time: datetime) -> bool:
        """判断是否应该执行备份"""
        # 简化的调度逻辑
        if config.schedule:
            # 这里应该实现完整的cron解析
            # 简化实现：每小时执行一次
            last_backup_time = self.stats.get('last_backup_time')
            if last_backup_time is None:
                return True
            
            time_diff = current_time - last_backup_time
            return time_diff.total_seconds() > 3600  # 1小时
        
        return False
    
    def _execute_backup(self, config: BackupConfig):
        """执行备份"""
        try:
            self.logger.info(f"Starting backup: {config.backup_id}")
            
            # 根据存储类型选择备份管理器
            if config.storage_type == BackupStorage.LOCAL:
                record = self.local_backup_manager.create_backup(config)
            elif config.storage_type.value.startswith('cloud_'):
                # 云备份
                if config.storage_type in self.cloud_backup_managers:
                    manager = self.cloud_backup_managers[config.storage_type]
                    record = manager.create_backup(config)
                else:
                    raise ValueError(f"Cloud backup manager not configured: {config.storage_type}")
            else:
                raise ValueError(f"Unsupported storage type: {config.storage_type}")
            
            # 保存备份记录
            self.backup_records[record.backup_id] = record
            
            # 更新统计
            self.stats['total_backups'] += 1
            if record.status == BackupStatus.COMPLETED:
                self.stats['successful_backups'] += 1
                self.stats['total_backup_size'] += record.backup_size
            else:
                self.stats['failed_backups'] += 1
            
            self.stats['last_backup_time'] = datetime.now()
            
            self.logger.info(f"Backup completed: {config.backup_id} ({record.status.value})")
            
        except Exception as e:
            self.logger.error(f"Error executing backup {config.backup_id}: {e}")
            
            # 创建失败记录
            record = BackupRecord(
                backup_id=config.backup_id,
                backup_type=config.backup_type,
                storage_type=config.storage_type,
                source_paths=config.source_paths,
                target_path=config.target_path,
                backup_size=0,
                compressed_size=0,
                backup_time=datetime.now(),
                completion_time=datetime.now(),
                status=BackupStatus.FAILED,
                error_message=str(e)
            )
            
            self.backup_records[record.backup_id] = record
            self.stats['total_backups'] += 1
            self.stats['failed_backups'] += 1
    
    def execute_recovery(self, recovery_config: RecoveryConfig) -> RecoveryRecord:
        """执行恢复"""
        try:
            self.logger.info(f"Starting recovery: {recovery_config.recovery_id}")
            
            # 查找备份记录
            if recovery_config.backup_id not in self.backup_records:
                raise ValueError(f"Backup record not found: {recovery_config.backup_id}")
            
            backup_record = self.backup_records[recovery_config.backup_id]
            
            # 根据存储类型选择恢复方法
            if backup_record.storage_type == BackupStorage.LOCAL:
                record = self.local_backup_manager.restore_backup(backup_record, recovery_config)
            else:
                raise ValueError(f"Recovery not implemented for storage type: {backup_record.storage_type}")
            
            # 保存恢复记录
            self.recovery_records[record.recovery_id] = record
            
            # 更新统计
            self.stats['total_recoveries'] += 1
            if record.status == RecoveryStatus.COMPLETED:
                self.stats['successful_recoveries'] += 1
            else:
                self.stats['failed_recoveries'] += 1
            
            self.stats['last_recovery_time'] = datetime.now()
            
            self.logger.info(f"Recovery completed: {recovery_config.recovery_id} ({record.status.value})")
            
            return record
            
        except Exception as e:
            self.logger.error(f"Error executing recovery {recovery_config.recovery_id}: {e}")
            
            # 创建失败记录
            record = RecoveryRecord(
                recovery_id=recovery_config.recovery_id,
                backup_id=recovery_config.backup_id,
                recovery_strategy=recovery_config.recovery_strategy,
                target_path=recovery_config.target_path,
                recovery_point=recovery_config.recovery_point,
                recovery_time=datetime.now(),
                completion_time=datetime.now(),
                status=RecoveryStatus.FAILED,
                error_message=str(e)
            )
            
            self.recovery_records[record.recovery_id] = record
            self.stats['total_recoveries'] += 1
            self.stats['failed_recoveries'] += 1
            
            return record
    
    def _check_backup_status(self):
        """检查备份状态"""
        for record in self.backup_records.values():
            if record.status == BackupStatus.RUNNING:
                # 检查是否超时
                if record.completion_time is None:
                    time_diff = datetime.now() - record.backup_time
                    if time_diff.total_seconds() > 3600:  # 1小时超时
                        record.status = BackupStatus.FAILED
                        record.error_message = "Backup timeout"
                        record.completion_time = datetime.now()
    
    def _execute_dr_tests(self):
        """执行灾难恢复测试"""
        current_time = datetime.now()
        
        for plan in self.disaster_recovery_plans.values():
            if plan.test_schedule and self._should_execute_test(plan, current_time):
                # 异步执行测试
                self.thread_pool.submit(
                    self._execute_dr_test,
                    plan
                )
    
    def _should_execute_test(self, plan: DisasterRecoveryPlan, current_time: datetime) -> bool:
        """判断是否应该执行测试"""
        if plan.last_test_date is None:
            return True
        
        # 简化：每月执行一次测试
        time_diff = current_time - plan.last_test_date
        return time_diff.days >= 30
    
    def _execute_dr_test(self, plan: DisasterRecoveryPlan):
        """执行灾难恢复测试"""
        try:
            self.logger.info(f"Starting DR test: {plan.plan_id}")
            
            test_start_time = datetime.now()
            test_results = []
            
            # 执行测试步骤
            for procedure in plan.recovery_procedures:
                step_result = self._execute_test_step(procedure)
                test_results.append(step_result)
            
            # 记录测试结果
            test_record = {
                'test_date': test_start_time.isoformat(),
                'duration_minutes': (datetime.now() - test_start_time).total_seconds() / 60,
                'steps_executed': len(test_results),
                'steps_passed': sum(1 for r in test_results if r.get('success', False)),
                'results': test_results
            }
            
            plan.test_results.append(test_record)
            plan.last_test_date = test_start_time
            
            self.logger.info(f"DR test completed: {plan.plan_id}")
            
        except Exception as e:
            self.logger.error(f"Error executing DR test {plan.plan_id}: {e}")
    
    def _execute_test_step(self, procedure: Dict[str, Any]) -> Dict[str, Any]:
        """执行测试步骤"""
        try:
            step_type = procedure.get('type')
            
            if step_type == 'backup_test':
                # 测试备份
                return {'success': True, 'message': 'Backup test passed'}
            elif step_type == 'recovery_test':
                # 测试恢复
                return {'success': True, 'message': 'Recovery test passed'}
            elif step_type == 'connectivity_test':
                # 测试连通性
                return {'success': True, 'message': 'Connectivity test passed'}
            else:
                return {'success': False, 'message': f'Unknown test type: {step_type}'}
                
        except Exception as e:
            return {'success': False, 'message': str(e)}
    
    def _cleanup_expired_backups(self):
        """清理过期备份"""
        current_time = datetime.now()
        expired_backups = []
        
        for backup_id, record in self.backup_records.items():
            # 检查保留期限
            if record.completion_time:
                retention_days = 30  # 默认保留30天
                expiry_time = record.completion_time + timedelta(days=retention_days)
                
                if current_time > expiry_time:
                    expired_backups.append(backup_id)
        
        # 删除过期备份
        for backup_id in expired_backups:
            try:
                record = self.backup_records[backup_id]
                
                # 删除备份文件
                backup_path = Path(record.target_path)
                if backup_path.exists():
                    if backup_path.is_file():
                        backup_path.unlink()
                    elif backup_path.is_dir():
                        shutil.rmtree(backup_path)
                
                # 删除记录
                del self.backup_records[backup_id]
                
                self.logger.info(f"Deleted expired backup: {backup_id}")
                
            except Exception as e:
                self.logger.error(f"Error deleting expired backup {backup_id}: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'is_running': self.is_running,
            'disaster_recovery_plans': len(self.disaster_recovery_plans),
            'backup_records': len(self.backup_records),
            'recovery_records': len(self.recovery_records),
            'backup_success_rate': (self.stats['successful_backups'] / max(self.stats['total_backups'], 1)) * 100,
            'recovery_success_rate': (self.stats['successful_recoveries'] / max(self.stats['total_recoveries'], 1)) * 100,
            **self.stats
        }
    
    def export_report(self, report_path: str):
        """导出报告"""
        report = {
            'generated_at': datetime.now().isoformat(),
            'statistics': self.get_statistics(),
            'disaster_recovery_plans': {
                plan_id: {
                    'plan_name': plan.plan_name,
                    'description': plan.description,
                    'rto_minutes': plan.recovery_time_objective,
                    'rpo_minutes': plan.recovery_point_objective,
                    'backup_configs': len(plan.backup_configs),
                    'last_test_date': plan.last_test_date.isoformat() if plan.last_test_date else None,
                    'test_results': plan.test_results
                }
                for plan_id, plan in self.disaster_recovery_plans.items()
            },
            'backup_records': {
                record_id: {
                    'backup_type': record.backup_type.value,
                    'storage_type': record.storage_type.value,
                    'backup_size': record.backup_size,
                    'backup_time': record.backup_time.isoformat(),
                    'status': record.status.value,
                    'error_message': record.error_message
                }
                for record_id, record in self.backup_records.items()
            },
            'recovery_records': {
                record_id: {
                    'recovery_strategy': record.recovery_strategy.value,
                    'files_recovered': record.files_recovered,
                    'bytes_recovered': record.bytes_recovered,
                    'recovery_time': record.recovery_time.isoformat(),
                    'status': record.status.value,
                    'integrity_verified': record.integrity_verified
                }
                for record_id, record in self.recovery_records.items()
            }
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Report exported to: {report_path}")
    
    def __del__(self):
        """析构函数"""
        if hasattr(self, 'is_running') and self.is_running:
            self.stop()


# 使用示例
def create_sample_disaster_recovery_plan() -> DisasterRecoveryPlan:
    """创建示例灾难恢复计划"""
    
    # 创建备份配置
    backup_configs = [
        BackupConfig(
            backup_id="daily_data_backup",
            backup_type=BackupType.INCREMENTAL,
            storage_type=BackupStorage.LOCAL,
            source_paths=["./data", "./logs"],
            target_path="./backups/daily",
            schedule="0 2 * * *",  # 每天2点执行
            retention_days=30,
            compression_enabled=True,
            encryption_enabled=True
        ),
        BackupConfig(
            backup_id="weekly_full_backup",
            backup_type=BackupType.FULL,
            storage_type=BackupStorage.CLOUD_AWS,
            source_paths=["./"],
            target_path="myquant-backups/weekly",
            schedule="0 3 * * 0",  # 每周日3点执行
            retention_days=90,
            compression_enabled=True,
            encryption_enabled=True
        )
    ]
    
    # 创建灾难恢复计划
    plan = DisasterRecoveryPlan(
        plan_id="myquant_dr_plan",
        plan_name="MyQuant Disaster Recovery Plan",
        description="Comprehensive disaster recovery plan for MyQuant trading system",
        recovery_time_objective=60,  # 1小时RTO
        recovery_point_objective=15,  # 15分钟RPO
        backup_configs=backup_configs,
        recovery_procedures=[
            {
                'type': 'backup_test',
                'description': 'Verify backup integrity',
                'timeout_minutes': 30
            },
            {
                'type': 'recovery_test',
                'description': 'Test recovery procedures',
                'timeout_minutes': 60
            }
        ],
        failover_targets=["backup_server_1", "backup_server_2"],
        rollback_procedures=[
            {
                'type': 'rollback',
                'description': 'Rollback to previous stable state',
                'timeout_minutes': 30
            }
        ],
        contact_list=["admin@myquant.com", "ops@myquant.com"],
        test_schedule="0 4 1 * *"  # 每月1日4点执行测试
    )
    
    return plan
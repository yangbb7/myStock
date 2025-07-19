# -*- coding: utf-8 -*-
"""
统一数据库配置管理
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass, field


class DatabaseType(Enum):
    """数据库类型"""
    SQLITE = "sqlite"
    MYSQL = "mysql"
    POSTGRESQL = "postgresql"


class Environment(Enum):
    """运行环境"""
    DEVELOPMENT = "development"
    TEST = "test"
    PRODUCTION = "production"


@dataclass
class DatabaseConfig:
    """统一数据库配置"""
    
    # 基础配置
    environment: Environment = Environment.DEVELOPMENT
    database_type: DatabaseType = DatabaseType.SQLITE
    
    # 路径配置 - 全局最优解：统一使用项目根目录
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent.parent)
    data_directory: str = "data"
    
    # 数据库文件配置
    main_database_name: str = "myquant.db"
    test_database_name: str = "myquant_test.db"
    
    # 连接配置
    host: Optional[str] = None
    port: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = None
    
    # 性能配置
    connection_pool_size: int = 5
    connection_timeout: int = 30
    query_timeout: int = 30
    
    # 缓存配置
    enable_cache: bool = True
    cache_size: int = 1000
    cache_ttl: int = 3600
    
    def __post_init__(self):
        """初始化后处理"""
        # 确保数据目录存在
        self.data_path.mkdir(parents=True, exist_ok=True)
    
    @property
    def data_path(self) -> Path:
        """数据目录路径"""
        return self.project_root / self.data_directory
    
    @property
    def database_path(self) -> Path:
        """数据库文件路径"""
        db_name = self.test_database_name if self.environment == Environment.TEST else self.main_database_name
        return self.data_path / db_name
    
    @property
    def database_url(self) -> str:
        """数据库连接URL"""
        if self.database_type == DatabaseType.SQLITE:
            return f"sqlite:///{self.database_path}"
        elif self.database_type == DatabaseType.MYSQL:
            return f"mysql://{self.username}:{self.password}@{self.host}:{self.port}/{self.main_database_name}"
        elif self.database_type == DatabaseType.POSTGRESQL:
            return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.main_database_name}"
        else:
            raise ValueError(f"Unsupported database type: {self.database_type}")
    
    @property
    def shards_path(self) -> Path:
        """分片数据库目录"""
        return self.data_path / "shards"
    
    def get_shard_path(self, shard_type: str) -> Path:
        """获取特定分片路径"""
        return self.shards_path / shard_type
    
    def get_environment_config(self) -> Dict[str, Any]:
        """获取环境特定配置"""
        return {
            "database_url": self.database_url,
            "database_path": str(self.database_path),
            "data_path": str(self.data_path),
            "shards_path": str(self.shards_path),
            "environment": self.environment.value,
            "database_type": self.database_type.value,
            "cache_enabled": self.enable_cache,
            "cache_size": self.cache_size,
            "cache_ttl": self.cache_ttl
        }
    
    @classmethod
    def from_environment(cls) -> 'DatabaseConfig':
        """从环境变量创建配置"""
        env_name = os.getenv('MYQUANT_ENV', 'development').lower()
        
        if env_name == 'test':
            environment = Environment.TEST
        elif env_name == 'production':
            environment = Environment.PRODUCTION
        else:
            environment = Environment.DEVELOPMENT
        
        return cls(
            environment=environment,
            database_type=DatabaseType(os.getenv('MYQUANT_DB_TYPE', 'sqlite')),
            host=os.getenv('MYQUANT_DB_HOST'),
            port=int(os.getenv('MYQUANT_DB_PORT', '5432')) if os.getenv('MYQUANT_DB_PORT') else None,
            username=os.getenv('MYQUANT_DB_USER'),
            password=os.getenv('MYQUANT_DB_PASSWORD'),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'environment': self.environment.value,
            'database_type': self.database_type.value,
            'database_url': self.database_url,
            'database_path': str(self.database_path),
            'data_path': str(self.data_path),
            'shards_path': str(self.shards_path),
            'connection_pool_size': self.connection_pool_size,
            'connection_timeout': self.connection_timeout,
            'query_timeout': self.query_timeout,
            'enable_cache': self.enable_cache,
            'cache_size': self.cache_size,
            'cache_ttl': self.cache_ttl
        }


# 全局配置实例
_global_config: Optional[DatabaseConfig] = None


def get_database_config() -> DatabaseConfig:
    """获取全局数据库配置"""
    global _global_config
    if _global_config is None:
        _global_config = DatabaseConfig.from_environment()
    return _global_config


def set_database_config(config: DatabaseConfig) -> None:
    """设置全局数据库配置"""
    global _global_config
    _global_config = config


def reset_database_config() -> None:
    """重置全局数据库配置"""
    global _global_config
    _global_config = None
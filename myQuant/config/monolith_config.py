# -*- coding: utf-8 -*-
"""
模块化单体配置管理系统
提供灵活的配置管理，支持环境变量、配置文件等多种配置方式
"""

import os
import yaml
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from enum import Enum

from ..core.enhanced_trading_system import SystemModule


class ConfigSource(Enum):
    """配置来源"""
    DEFAULT = "default"
    FILE = "file"
    ENV = "environment"
    OVERRIDE = "override"


@dataclass
class DatabaseConfig:
    """数据库配置"""
    url: str = "sqlite:///data/myquant.db"
    echo: bool = False
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30
    
    @classmethod
    def from_env(cls):
        """从环境变量创建配置"""
        return cls(
            url=os.getenv("DATABASE_URL", "sqlite:///data/myquant.db"),
            echo=os.getenv("DATABASE_ECHO", "false").lower() == "true",
            pool_size=int(os.getenv("DATABASE_POOL_SIZE", "5")),
            max_overflow=int(os.getenv("DATABASE_MAX_OVERFLOW", "10")),
            pool_timeout=int(os.getenv("DATABASE_POOL_TIMEOUT", "30"))
        )


@dataclass
class TradingConfig:
    """交易配置"""
    initial_capital: float = 1000000.0
    commission_rate: float = 0.0003
    min_commission: float = 5.0
    max_position_size: float = 0.1
    max_drawdown_limit: float = 0.2
    max_daily_loss: float = 0.05
    
    @classmethod
    def from_env(cls):
        """从环境变量创建配置"""
        return cls(
            initial_capital=float(os.getenv("TRADING_INITIAL_CAPITAL", "1000000.0")),
            commission_rate=float(os.getenv("TRADING_COMMISSION_RATE", "0.0003")),
            min_commission=float(os.getenv("TRADING_MIN_COMMISSION", "5.0")),
            max_position_size=float(os.getenv("TRADING_MAX_POSITION_SIZE", "0.1")),
            max_drawdown_limit=float(os.getenv("TRADING_MAX_DRAWDOWN_LIMIT", "0.2")),
            max_daily_loss=float(os.getenv("TRADING_MAX_DAILY_LOSS", "0.05"))
        )


@dataclass
class PerformanceConfig:
    """性能配置"""
    max_concurrent_orders: int = 50
    order_timeout: float = 10.0
    data_buffer_size: int = 1000
    enable_cache: bool = True
    cache_size: int = 10000
    batch_size: int = 100
    
    @classmethod
    def from_env(cls):
        """从环境变量创建配置"""
        return cls(
            max_concurrent_orders=int(os.getenv("PERF_MAX_CONCURRENT_ORDERS", "50")),
            order_timeout=float(os.getenv("PERF_ORDER_TIMEOUT", "10.0")),
            data_buffer_size=int(os.getenv("PERF_DATA_BUFFER_SIZE", "1000")),
            enable_cache=os.getenv("PERF_ENABLE_CACHE", "true").lower() == "true",
            cache_size=int(os.getenv("PERF_CACHE_SIZE", "10000")),
            batch_size=int(os.getenv("PERF_BATCH_SIZE", "100"))
        )


@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = "INFO"
    file: str = "logs/monolith.log"
    max_size: int = 10485760  # 10MB
    backup_count: int = 5
    enable_console: bool = True
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @classmethod
    def from_env(cls):
        """从环境变量创建配置"""
        return cls(
            level=os.getenv("LOG_LEVEL", "INFO"),
            file=os.getenv("LOG_FILE", "logs/monolith.log"),
            max_size=int(os.getenv("LOG_MAX_SIZE", "10485760")),
            backup_count=int(os.getenv("LOG_BACKUP_COUNT", "5")),
            enable_console=os.getenv("LOG_ENABLE_CONSOLE", "true").lower() == "true",
            format=os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )


@dataclass
class APIConfig:
    """API配置"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    enable_docs: bool = True
    enable_cors: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    
    @classmethod
    def from_env(cls):
        """从环境变量创建配置"""
        cors_origins = os.getenv("API_CORS_ORIGINS", "*").split(",")
        return cls(
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", "8000")),
            debug=os.getenv("API_DEBUG", "false").lower() == "true",
            enable_docs=os.getenv("API_ENABLE_DOCS", "true").lower() == "true",
            enable_cors=os.getenv("API_ENABLE_CORS", "true").lower() == "true",
            cors_origins=cors_origins
        )


@dataclass
class MonitoringConfig:
    """监控配置"""
    enable_metrics: bool = True
    metrics_port: int = 8080
    enable_health_check: bool = True
    health_check_interval: int = 30
    enable_prometheus: bool = False
    prometheus_port: int = 9090
    
    @classmethod
    def from_env(cls):
        """从环境变量创建配置"""
        return cls(
            enable_metrics=os.getenv("MONITOR_ENABLE_METRICS", "true").lower() == "true",
            metrics_port=int(os.getenv("MONITOR_METRICS_PORT", "8080")),
            enable_health_check=os.getenv("MONITOR_ENABLE_HEALTH_CHECK", "true").lower() == "true",
            health_check_interval=int(os.getenv("MONITOR_HEALTH_CHECK_INTERVAL", "30")),
            enable_prometheus=os.getenv("MONITOR_ENABLE_PROMETHEUS", "false").lower() == "true",
            prometheus_port=int(os.getenv("MONITOR_PROMETHEUS_PORT", "9090"))
        )


@dataclass
class MonolithConfig:
    """模块化单体完整配置"""
    # 基础配置
    environment: str = "development"
    debug: bool = False
    
    # 模块配置
    enabled_modules: List[str] = field(default_factory=lambda: [
        "data", "strategy", "execution", "risk", "portfolio", "analytics"
    ])
    
    # 各子系统配置
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    api: APIConfig = field(default_factory=APIConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # 配置元数据
    config_source: ConfigSource = ConfigSource.DEFAULT
    loaded_at: Optional[str] = None
    
    @classmethod
    def from_env(cls):
        """从环境变量创建配置"""
        enabled_modules = os.getenv("ENABLED_MODULES", "data,strategy,execution,risk,portfolio,analytics").split(",")
        
        return cls(
            environment=os.getenv("ENVIRONMENT", "development"),
            debug=os.getenv("DEBUG", "false").lower() == "true",
            enabled_modules=enabled_modules,
            database=DatabaseConfig.from_env(),
            trading=TradingConfig.from_env(),
            performance=PerformanceConfig.from_env(),
            logging=LoggingConfig.from_env(),
            api=APIConfig.from_env(),
            monitoring=MonitoringConfig.from_env(),
            config_source=ConfigSource.ENV
        )
    
    @classmethod
    def from_file(cls, file_path: str):
        """从文件创建配置"""
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        config = cls(**data)
        config.config_source = ConfigSource.FILE
        return config
    
    def save_to_file(self, file_path: str):
        """保存配置到文件"""
        data = asdict(self)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            else:
                json.dump(data, f, indent=2, ensure_ascii=False)
    
    def to_system_config(self):
        """转换为系统配置"""
        from ..core.enhanced_trading_system import SystemConfig, SystemModule
        
        # 转换模块列表
        modules = []
        for module_name in self.enabled_modules:
            try:
                modules.append(SystemModule(module_name))
            except ValueError:
                pass  # 忽略不支持的模块
        
        return SystemConfig(
            # 基础配置
            initial_capital=self.trading.initial_capital,
            commission_rate=self.trading.commission_rate,
            min_commission=self.trading.min_commission,
            
            # 性能配置
            max_concurrent_orders=self.performance.max_concurrent_orders,
            order_timeout=self.performance.order_timeout,
            data_buffer_size=self.performance.data_buffer_size,
            
            # 风险管理
            max_position_size=self.trading.max_position_size,
            max_drawdown_limit=self.trading.max_drawdown_limit,
            max_daily_loss=self.trading.max_daily_loss,
            
            # 模块配置
            enabled_modules=modules,
            
            # 数据库配置
            database_url=self.database.url,
            enable_persistence=True,
            
            # 监控配置
            enable_metrics=self.monitoring.enable_metrics,
            metrics_port=self.monitoring.metrics_port,
            
            # 日志配置
            log_level=self.logging.level,
            log_file=self.logging.file
        )
    
    def validate(self) -> List[str]:
        """验证配置"""
        errors = []
        
        # 验证交易配置
        if self.trading.initial_capital <= 0:
            errors.append("初始资金必须大于0")
        
        if self.trading.commission_rate < 0 or self.trading.commission_rate > 1:
            errors.append("佣金率必须在0-1之间")
        
        if self.trading.max_position_size <= 0 or self.trading.max_position_size > 1:
            errors.append("最大持仓比例必须在0-1之间")
        
        # 验证性能配置
        if self.performance.max_concurrent_orders <= 0:
            errors.append("最大并发订单数必须大于0")
        
        if self.performance.order_timeout <= 0:
            errors.append("订单超时时间必须大于0")
        
        # 验证API配置
        if self.api.port <= 0 or self.api.port > 65535:
            errors.append("API端口必须在1-65535之间")
        
        return errors


class ConfigManager:
    """配置管理器"""
    
    def __init__(self):
        self.config: Optional[MonolithConfig] = None
        self.config_file: Optional[str] = None
        
    def load_config(self, 
                   file_path: Optional[str] = None,
                   use_env: bool = True,
                   overrides: Optional[Dict[str, Any]] = None) -> MonolithConfig:
        """加载配置"""
        
        # 1. 从默认配置开始
        config = MonolithConfig()
        
        # 2. 从文件加载
        if file_path and os.path.exists(file_path):
            config = MonolithConfig.from_file(file_path)
            self.config_file = file_path
        
        # 3. 从环境变量覆盖
        if use_env:
            env_config = MonolithConfig.from_env()
            # 合并环境变量配置
            config = self._merge_configs(config, env_config)
        
        # 4. 应用覆盖配置
        if overrides:
            config = self._apply_overrides(config, overrides)
        
        # 5. 验证配置
        errors = config.validate()
        if errors:
            raise ValueError(f"配置验证失败: {', '.join(errors)}")
        
        # 6. 设置加载时间
        from datetime import datetime
        config.loaded_at = datetime.now().isoformat()
        
        self.config = config
        return config
    
    def _merge_configs(self, base: MonolithConfig, overlay: MonolithConfig) -> MonolithConfig:
        """合并两个配置"""
        # 简单的深度合并
        base_dict = asdict(base)
        overlay_dict = asdict(overlay)
        
        merged = self._deep_merge(base_dict, overlay_dict)
        return MonolithConfig(**merged)
    
    def _deep_merge(self, base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
        """深度合并字典"""
        result = base.copy()
        
        for key, value in overlay.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _apply_overrides(self, config: MonolithConfig, overrides: Dict[str, Any]) -> MonolithConfig:
        """应用覆盖配置"""
        config_dict = asdict(config)
        merged = self._deep_merge(config_dict, overrides)
        return MonolithConfig(**merged)
    
    def get_config(self) -> MonolithConfig:
        """获取当前配置"""
        if self.config is None:
            self.config = self.load_config()
        return self.config
    
    def reload_config(self) -> MonolithConfig:
        """重新加载配置"""
        if self.config_file:
            return self.load_config(self.config_file)
        else:
            return self.load_config()
    
    def save_config(self, file_path: str):
        """保存当前配置"""
        if self.config:
            self.config.save_to_file(file_path)
            self.config_file = file_path


# 全局配置管理器实例
config_manager = ConfigManager()


def get_config() -> MonolithConfig:
    """获取全局配置"""
    return config_manager.get_config()


def load_config(file_path: Optional[str] = None, **kwargs) -> MonolithConfig:
    """加载配置"""
    return config_manager.load_config(file_path, **kwargs)
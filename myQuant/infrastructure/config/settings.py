# -*- coding: utf-8 -*-
"""
Settings - 统一配置管理模块
"""

import json
import os
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


class Environment(Enum):
    """环境类型"""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class DatabaseConfig:
    """数据库配置"""
    
    # 支持两种配置方式：URL方式（推荐）和传统方式
    url: str = "sqlite:///data/myquant.db"  # 数据库连接URL
    echo: bool = False  # 是否打印SQL语句
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    
    # 传统配置方式（向后兼容）
    host: str = "localhost"
    port: int = 5432
    database: str = "myquant"
    username: str = "postgres"
    password: str = ""


@dataclass
class DataProviderConfig:
    """数据提供商配置"""

    primary_provider: str = "tushare"
    fallback_providers: list = None
    api_keys: Dict[str, str] = None
    timeout: int = 30
    retry_attempts: int = 3

    def __post_init__(self):
        if self.fallback_providers is None:
            self.fallback_providers = ["yahoo", "eastmoney"]
        if self.api_keys is None:
            self.api_keys = {}


@dataclass
class TradingConfig:
    """交易配置"""

    initial_capital: float = 1000000.0
    commission_rate: float = 0.0003
    min_commission: float = 5.0
    max_position_size: float = 0.1  # 10%
    max_total_positions: int = 50
    order_timeout: int = 3600  # 1小时

    # 风险管理
    max_drawdown_limit: float = 0.2  # 20%
    max_daily_loss: float = 0.05  # 5%
    position_size_limit: float = 0.1  # 10%
    sector_exposure_limit: float = 0.3  # 30%
    var_confidence: float = 0.95


@dataclass
class BacktestConfig:
    """回测配置"""

    start_date: str = "2020-01-01"
    end_date: str = "2023-12-31"
    benchmark: str = "000300.SH"  # 沪深300
    frequency: str = "daily"  # daily, hourly, minute

    # 回测参数
    slippage: float = 0.001  # 0.1%
    latency: int = 0  # 延迟（毫秒）

    # 性能分析
    calculate_metrics: bool = True
    save_trades: bool = True
    save_portfolio_history: bool = True


@dataclass
class LoggingConfig:
    """日志配置"""

    level: str = "INFO"
    console_output: bool = True
    file_output: bool = True
    json_output: bool = False
    log_dir: str = "logs"
    file: str = "logs/myquant.log"  # 日志文件路径
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    max_size: int = 10 * 1024 * 1024  # 别名，向后兼容
    backup_count: int = 5
    enable_console: bool = True  # 向后兼容
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"  # 日志格式


@dataclass
class MonitoringConfig:
    """监控配置"""

    enabled: bool = True
    metrics_collection_interval: int = 10  # 秒
    system_metrics: bool = True
    performance_metrics: bool = True
    export_prometheus: bool = False
    prometheus_port: int = 8000
    
    # 向后兼容的字段
    enable_metrics: bool = True
    metrics_port: int = 8080
    enable_health_check: bool = True
    health_check_interval: int = 30
    enable_prometheus: bool = False


@dataclass
class ApplicationConfig:
    """应用配置"""

    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False

    # 组件配置
    database: DatabaseConfig = None
    data_provider: DataProviderConfig = None
    trading: TradingConfig = None
    backtest: BacktestConfig = None
    logging: LoggingConfig = None
    monitoring: MonitoringConfig = None

    # 自定义配置
    custom: Dict[str, Any] = None
    
    # 启用的模块
    enabled_modules: List[str] = None

    def __post_init__(self):
        if self.database is None:
            self.database = DatabaseConfig()
        if self.data_provider is None:
            self.data_provider = DataProviderConfig()
        if self.trading is None:
            self.trading = TradingConfig()
        if self.backtest is None:
            self.backtest = BacktestConfig()
        if self.logging is None:
            self.logging = LoggingConfig()
        if self.monitoring is None:
            self.monitoring = MonitoringConfig()
        if self.custom is None:
            self.custom = {}
        if self.enabled_modules is None:
            self.enabled_modules = [
                "data", "strategy", "execution", "risk", "portfolio", "analytics"
            ]


class ConfigManager:
    """配置管理器"""

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config: Optional[ApplicationConfig] = None
        self._environment = Environment.DEVELOPMENT

    def load_config(
        self, config_file: str = None, environment: Environment = None
    ) -> ApplicationConfig:
        """加载配置"""
        if environment:
            self._environment = environment
        else:
            # 从环境变量获取
            env_name = os.getenv("MYQUANT_ENV", "development")
            self._environment = Environment(env_name)

        # 确定配置文件
        if config_file is None:
            config_file = f"{self._environment.value}.yaml"

        config_path = self.config_dir / config_file

        if config_path.exists():
            self.config = self._load_from_file(config_path)
        else:
            # 使用默认配置
            self.config = ApplicationConfig(environment=self._environment)

        # 应用环境变量覆盖
        self._apply_environment_overrides()

        return self.config

    def _load_from_file(self, config_path: Path) -> ApplicationConfig:
        """从文件加载配置"""
        with open(config_path, "r", encoding="utf-8") as f:
            if config_path.suffix.lower() in [".yaml", ".yml"]:
                data = yaml.safe_load(f)
            elif config_path.suffix.lower() == ".json":
                data = json.load(f)
            else:
                raise ValueError(
                    f"Unsupported config file format: {config_path.suffix}"
                )

        return self._dict_to_config(data)

    def _dict_to_config(self, data: Dict[str, Any]) -> ApplicationConfig:
        """将字典转换为配置对象"""
        # 处理嵌套配置
        config_data = {}

        if "environment" in data:
            config_data["environment"] = Environment(data["environment"])

        if "database" in data:
            config_data["database"] = DatabaseConfig(**data["database"])

        if "data_provider" in data:
            config_data["data_provider"] = DataProviderConfig(**data["data_provider"])

        if "trading" in data:
            config_data["trading"] = TradingConfig(**data["trading"])

        if "backtest" in data:
            config_data["backtest"] = BacktestConfig(**data["backtest"])

        if "logging" in data:
            config_data["logging"] = LoggingConfig(**data["logging"])

        if "monitoring" in data:
            config_data["monitoring"] = MonitoringConfig(**data["monitoring"])

        # 其他字段 - 只包含 ApplicationConfig 支持的字段
        allowed_keys = {
            "debug", "custom", "enabled_modules"
        }
        for key, value in data.items():
            if key in allowed_keys:
                config_data[key] = value

        return ApplicationConfig(**config_data)

    def _apply_environment_overrides(self):
        """应用环境变量覆盖"""
        if not self.config:
            return

        # 数据库配置覆盖
        if db_host := os.getenv("DB_HOST"):
            self.config.database.host = db_host
        if db_port := os.getenv("DB_PORT"):
            self.config.database.port = int(db_port)
        if db_name := os.getenv("DB_NAME"):
            self.config.database.database = db_name
        if db_user := os.getenv("DB_USER"):
            self.config.database.username = db_user
        if db_pass := os.getenv("DB_PASSWORD"):
            self.config.database.password = db_pass

        # API密钥覆盖
        if tushare_key := os.getenv("TUSHARE_API_KEY"):
            self.config.data_provider.api_keys["tushare"] = tushare_key
        if yahoo_key := os.getenv("YAHOO_API_KEY"):
            self.config.data_provider.api_keys["yahoo"] = yahoo_key

        # 交易配置覆盖
        if initial_capital := os.getenv("INITIAL_CAPITAL"):
            self.config.trading.initial_capital = float(initial_capital)

        # 日志级别覆盖
        if log_level := os.getenv("LOG_LEVEL"):
            self.config.logging.level = log_level

        # 调试模式覆盖
        if debug := os.getenv("DEBUG"):
            self.config.debug = debug.lower() in ["true", "1", "yes", "on"]

    def save_config(self, config_file: str = None, config: ApplicationConfig = None):
        """保存配置"""
        if config is None:
            config = self.config

        if config_file is None:
            config_file = f"{self._environment.value}.yaml"

        config_path = self.config_dir / config_file

        # 确保目录存在
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # 转换为字典
        config_dict = self._config_to_dict(config)

        # 保存文件
        with open(config_path, "w", encoding="utf-8") as f:
            if config_path.suffix.lower() in [".yaml", ".yml"]:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
            elif config_path.suffix.lower() == ".json":
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

    def _config_to_dict(self, config: ApplicationConfig) -> Dict[str, Any]:
        """将配置对象转换为字典"""
        result = {}

        result["environment"] = config.environment.value
        result["debug"] = config.debug

        if config.database:
            result["database"] = asdict(config.database)

        if config.data_provider:
            result["data_provider"] = asdict(config.data_provider)

        if config.trading:
            result["trading"] = asdict(config.trading)

        if config.backtest:
            result["backtest"] = asdict(config.backtest)

        if config.logging:
            result["logging"] = asdict(config.logging)

        if config.monitoring:
            result["monitoring"] = asdict(config.monitoring)

        if config.custom:
            result["custom"] = config.custom

        return result

    def get_config(self) -> ApplicationConfig:
        """获取当前配置"""
        if self.config is None:
            self.load_config()
        return self.config

    def update_config(self, updates: Dict[str, Any]):
        """更新配置"""
        if self.config is None:
            self.load_config()

        # 安全地处理配置更新
        try:
            config_dict = asdict(self.config) if self.config else {}
        except (TypeError, ValueError):
            # 如果asdict失败，使用空字典
            config_dict = {}

        # 深度更新配置
        self._deep_update(config_dict, updates)
        self.config = self._dict_to_config(config_dict)

    def _deep_update(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]):
        """深度更新字典"""
        for key, value in update_dict.items():
            # 处理点号分隔的键名
            if '.' in key:
                keys = key.split('.')
                current_dict = base_dict
                
                # 遍历到最后一个键之前的所有键
                for k in keys[:-1]:
                    if k not in current_dict:
                        current_dict[k] = {}
                    elif not isinstance(current_dict[k], dict):
                        current_dict[k] = {}
                    current_dict = current_dict[k]
                
                # 设置最后一个键的值
                current_dict[keys[-1]] = value
            elif (
                key in base_dict
                and isinstance(base_dict[key], dict)
                and isinstance(value, dict)
            ):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value


# 全局配置管理器
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """获取全局配置管理器"""
    global _config_manager

    if _config_manager is None:
        _config_manager = ConfigManager()

    return _config_manager


def get_config() -> ApplicationConfig:
    """获取应用配置"""
    return get_config_manager().get_config()


def load_config(
    config_file: str = None, environment: Environment = None
) -> ApplicationConfig:
    """加载配置"""
    return get_config_manager().load_config(config_file, environment)


def save_config(config_file: str = None, config: ApplicationConfig = None):
    """保存配置"""
    get_config_manager().save_config(config_file, config)


def update_config(updates: Dict[str, Any]):
    """更新配置"""
    get_config_manager().update_config(updates)

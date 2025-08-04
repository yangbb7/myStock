"""
统一配置管理系统

使用 Pydantic 实现类型安全的配置管理，支持环境变量、配置文件和默认值。
"""

import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings


class LogLevel(str, Enum):
    """日志级别"""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DatabaseType(str, Enum):
    """数据库类型"""

    SQLITE = "sqlite"
    MYSQL = "mysql"
    POSTGRESQL = "postgresql"


class DataSource(str, Enum):
    """数据源类型"""

    EMQUANT = "emquant"
    TUSHARE = "tushare"
    AKSHARE = "akshare"
    YAHOO = "yahoo"
    MOCK = "mock"


class TradingMode(str, Enum):
    """交易模式"""

    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


class DatabaseSettings(BaseSettings):
    """数据库配置"""

    type: DatabaseType = DatabaseType.SQLITE
    url: str = "sqlite:///data/myquant.db"
    host: Optional[str] = None
    port: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = None
    database: Optional[str] = None

    # 连接池配置
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30

    @field_validator("url", mode="before")
    @classmethod
    def build_database_url(cls, v, info):
        """构建数据库URL"""
        if v and v != "sqlite:///data/myquant.db":
            return v

        data = info.data if info else {}
        db_type = data.get("type", DatabaseType.SQLITE)
        if db_type == DatabaseType.SQLITE:
            return "sqlite:///data/myquant.db"
        elif db_type in [DatabaseType.MYSQL, DatabaseType.POSTGRESQL]:
            host = data.get("host", "localhost")
            port = data.get("port")
            username = data.get("username")
            password = data.get("password")
            database = data.get("database")

            if not all([host, username, database]):
                raise ValueError(f"Missing required fields for {db_type}")

            auth = f"{username}:{password}@" if password else f"{username}@"
            port_str = f":{port}" if port else ""
            return f"{db_type}://{auth}{host}{port_str}/{database}"

        return v


class RiskSettings(BaseSettings):
    """风险管理配置"""

    # 仓位控制
    max_position_size: float = Field(0.1, ge=0.0, le=1.0)
    max_sector_exposure: float = Field(0.3, ge=0.0, le=1.0)
    max_total_exposure: float = Field(0.95, ge=0.0, le=1.0)

    # 回撤控制
    max_drawdown_limit: float = Field(0.2, ge=0.0, le=1.0)
    stop_loss_threshold: float = Field(0.05, ge=0.0, le=1.0)

    # 风险价值
    var_confidence: float = Field(0.95, ge=0.9, le=0.99)
    var_window: int = Field(252, ge=30, le=500)

    # 流动性控制
    min_volume_ratio: float = Field(0.01, ge=0.0, le=1.0)
    max_volume_participation: float = Field(0.1, ge=0.0, le=1.0)


class PerformanceSettings(BaseSettings):
    """性能配置"""

    # 缓存配置
    cache_size: int = Field(1000, ge=100, le=10000)
    cache_ttl: int = Field(3600, ge=60, le=86400)  # 秒

    # 线程池配置
    thread_pool_size: int = Field(4, ge=1, le=32)
    max_workers: int = Field(8, ge=1, le=64)

    # 批处理配置
    batch_size: int = Field(1000, ge=100, le=10000)
    chunk_size: int = Field(10000, ge=1000, le=100000)

    # 内存管理
    max_memory_usage: float = Field(0.8, ge=0.1, le=0.95)  # 最大内存使用率


class DataSettings(BaseSettings):
    """数据配置"""

    # 数据源配置 - 优先使用真实数据源，移除Mock数据源
    primary_source: DataSource = DataSource.TUSHARE
    fallback_sources: List[DataSource] = [DataSource.YAHOO, DataSource.EMQUANT]

    # 数据获取配置
    max_retries: int = Field(3, ge=0, le=10)
    retry_delay: float = Field(1.0, ge=0.1, le=10.0)
    timeout: int = Field(30, ge=5, le=300)

    # 数据质量控制
    min_data_points: int = Field(20, ge=1, le=1000)
    max_missing_ratio: float = Field(0.1, ge=0.0, le=0.5)

    # 实时数据配置
    realtime_update_interval: int = Field(60, ge=1, le=3600)  # 秒
    price_tolerance: float = Field(0.05, ge=0.0, le=1.0)
    
    # 券商API集成配置
    enable_broker_realtime: bool = Field(True, description="启用券商实时数据")
    broker_apis: Dict[str, Any] = Field(default_factory=dict, description="券商API配置")
    
    # 数据验证配置
    enable_data_validation: bool = Field(True, description="启用数据质量验证")
    data_freshness_threshold: int = Field(300, ge=60, le=3600, description="数据新鲜度阈值(秒)")


class TradingSettings(BaseSettings):
    """交易配置"""

    # 交易模式
    mode: TradingMode = TradingMode.BACKTEST

    # 订单配置
    default_order_type: str = "MARKET"
    max_order_size: float = Field(1000000.0, ge=0.0)
    min_order_size: float = Field(100.0, ge=0.0)

    # 手续费配置
    commission_rate: float = Field(0.0003, ge=0.0, le=0.01)
    min_commission: float = Field(5.0, ge=0.0)
    stamp_tax_rate: float = Field(0.001, ge=0.0, le=0.01)

    # 市场时间配置
    market_open_time: str = "09:30"
    market_close_time: str = "15:00"
    lunch_break_start: str = "11:30"
    lunch_break_end: str = "13:00"


class LoggingSettings(BaseSettings):
    """日志配置"""

    # 基本配置
    level: LogLevel = LogLevel.INFO
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # 文件配置
    log_dir: str = "logs"
    max_file_size: int = Field(10 * 1024 * 1024, ge=1024 * 1024)  # 10MB
    backup_count: int = Field(5, ge=1, le=50)

    # 控制台输出
    console_output: bool = True
    structured_logging: bool = True

    # 特定日志器配置
    disable_loggers: List[str] = ["urllib3", "requests"]


class BacktestSettings(BaseSettings):
    """回测配置"""

    # 基本配置
    initial_capital: float = Field(1000000.0, ge=10000.0)
    start_date: str = "2020-01-01"
    end_date: str = "2023-12-31"

    # 执行配置
    benchmark: str = "000300.SH"  # 沪深300
    frequency: str = "D"  # 日频

    # 滑点和延迟
    slippage_rate: float = Field(0.001, ge=0.0, le=0.01)
    execution_delay: int = Field(0, ge=0, le=5)  # 执行延迟（分钟）

    # 结果分析
    save_results: bool = True
    generate_report: bool = True
    plot_results: bool = True


class APISettings(BaseSettings):
    """API配置"""

    # 东方财富配置
    emquant_username: Optional[str] = None
    emquant_password: Optional[str] = None

    # Tushare配置
    tushare_token: Optional[str] = None

    # 其他API配置
    rate_limit: int = Field(100, ge=1, le=1000)  # 每分钟请求数
    api_timeout: int = Field(30, ge=5, le=300)


class Settings(BaseSettings):
    """主配置类"""

    # 应用配置
    app_name: str = "MyQuant"
    version: str = "1.0.0"
    debug: bool = False

    # 子配置
    database: DatabaseSettings = DatabaseSettings()
    risk: RiskSettings = RiskSettings()
    performance: PerformanceSettings = PerformanceSettings()
    data: DataSettings = DataSettings()
    trading: TradingSettings = TradingSettings()
    logging: LoggingSettings = LoggingSettings()
    backtest: BacktestSettings = BacktestSettings()
    api: APISettings = APISettings()

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    @field_validator("debug", mode="before")
    @classmethod
    def parse_debug(cls, v):
        """解析debug配置"""
        if isinstance(v, str):
            return v.lower() in ["true", "1", "yes", "on"]
        return bool(v)

    def get_database_url(self) -> str:
        """获取数据库连接URL"""
        return self.database.url

    def get_log_config(self) -> Dict[str, Any]:
        """获取日志配置字典"""
        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {"format": self.logging.format},
                "structured": {
                    "class": "infrastructure.monitoring.exception_logger.StructuredFormatter"
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": self.logging.level.value,
                    "formatter": (
                        "structured" if self.logging.structured_logging else "standard"
                    ),
                    "stream": "ext://sys.stdout",
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": self.logging.level.value,
                    "formatter": (
                        "structured" if self.logging.structured_logging else "standard"
                    ),
                    "filename": f"{self.logging.log_dir}/myquant.log",
                    "maxBytes": self.logging.max_file_size,
                    "backupCount": self.logging.backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "": {
                    "handlers": (
                        ["console", "file"] if self.logging.console_output else ["file"]
                    ),
                    "level": self.logging.level.value,
                    "propagate": False,
                }
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self.dict()

    def save_to_file(self, file_path: str):
        """保存配置到文件"""
        import json

        config_dict = self.to_dict()

        # 创建目录
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False, default=str)

    @classmethod
    def load_from_file(cls, file_path: str) -> "Settings":
        """从文件加载配置"""
        import json

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"配置文件不存在: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)

        return cls(**config_dict)


# 全局配置实例
settings = Settings()


def get_settings() -> Settings:
    """获取配置实例"""
    return settings


def reload_settings():
    """重新加载配置"""
    global settings
    settings = Settings()
    return settings


def update_settings(**kwargs):
    """更新配置"""
    global settings
    settings = get_settings()  # Ensure settings is initialized
    for key, value in kwargs.items():
        if hasattr(settings, key):
            setattr(settings, key, value)
        else:
            raise ValueError(f"未知配置项: {key}")


# 配置验证函数
def validate_settings(settings_obj: Settings) -> List[str]:
    """验证配置设置"""
    errors = []

    # 验证风险参数逻辑关系
    if settings_obj.risk.max_position_size > settings_obj.risk.max_total_exposure:
        errors.append("最大单仓位不能超过最大总仓位")

    if settings_obj.risk.stop_loss_threshold > settings_obj.risk.max_drawdown_limit:
        errors.append("止损阈值不能超过最大回撤限制")

    # 验证交易参数
    if settings_obj.trading.min_order_size > settings_obj.trading.max_order_size:
        errors.append("最小订单大小不能超过最大订单大小")

    # 验证回测日期
    try:
        from datetime import datetime

        start = datetime.strptime(settings_obj.backtest.start_date, "%Y-%m-%d")
        end = datetime.strptime(settings_obj.backtest.end_date, "%Y-%m-%d")
        if start >= end:
            errors.append("回测开始日期必须早于结束日期")
    except ValueError:
        errors.append("回测日期格式错误，应为YYYY-MM-DD")

    return errors

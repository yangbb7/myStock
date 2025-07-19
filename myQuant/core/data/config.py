from typing import Dict, Any, List
from .multi_source_manager import ProviderConfig

# 数据源配置
DATA_PROVIDER_CONFIGS = [
    ProviderConfig(
        name='wind',
        provider_class='WindDataProvider',
        config={
            'api_key': 'your_wind_api_key',
            'username': 'your_wind_username',
            'password': 'your_wind_password',
            'server_url': 'https://api.wind.com.cn'
        },
        priority=1,
        enabled=True,
        fallback_providers=['bloomberg', 'reuters']
    ),
    ProviderConfig(
        name='bloomberg',
        provider_class='BloombergDataProvider',
        config={
            'api_key': 'your_bloomberg_api_key',
            'server_host': 'localhost',
            'server_port': 8194
        },
        priority=2,
        enabled=True,
        fallback_providers=['reuters', 'alpha']
    ),
    ProviderConfig(
        name='reuters',
        provider_class='ReutersDataProvider',
        config={
            'api_key': 'your_reuters_api_key',
            'base_url': 'https://api.refinitiv.com'
        },
        priority=3,
        enabled=True,
        fallback_providers=['alpha', 'wind']
    ),
    ProviderConfig(
        name='alpha',
        provider_class='AlphaDataProvider',
        config={
            'api_key': 'your_alpha_api_key',
            'base_url': 'https://www.alphavantage.co'
        },
        priority=4,
        enabled=True,
        fallback_providers=['wind', 'bloomberg']
    ),
    ProviderConfig(
        name='level2',
        provider_class='Level2DataProvider',
        config={
            'websocket_url': 'wss://api.level2.com/ws',
            'api_key': 'your_level2_api_key',
            'depth_levels': 10,
            'update_frequency': 100
        },
        priority=5,
        enabled=True,
        fallback_providers=[]
    )
]

# 数据质量监控配置
DATA_QUALITY_CONFIG = {
    'completeness_threshold': 0.95,
    'accuracy_threshold': 0.90,
    'timeliness_threshold': 0.85,
    'consistency_threshold': 0.80,
    'validity_threshold': 0.90,
    'latency_threshold': 1000,
    'anomaly_threshold': 3.0,
    'freshness_threshold': 300,
    'alert_enabled': True,
    'alert_channels': ['log', 'email', 'webhook']
}

# 存储配置
STORAGE_CONFIG = {
    'storage_path': 'data/shards',
    'compression_enabled': True,
    'backup_enabled': True,
    'backup_interval': 3600,
    'cleanup_enabled': True,
    'cleanup_interval': 86400
}

# 数据源优先级映射
DATA_TYPE_PRIORITY_MAP = {
    'tick': ['level2', 'bloomberg', 'wind', 'reuters', 'alpha'],
    'kline': ['bloomberg', 'wind', 'reuters', 'alpha'],
    'orderbook': ['level2', 'bloomberg'],
    'trade': ['level2', 'bloomberg', 'reuters'],
    'fundamental': ['bloomberg', 'reuters', 'wind', 'alpha'],
    'news': ['bloomberg', 'reuters', 'alpha'],
    'alternative': ['alpha', 'bloomberg']
}

# 实时数据配置
REALTIME_CONFIG = {
    'max_subscriptions': 100,
    'buffer_size': 1000,
    'heartbeat_interval': 30,
    'reconnect_attempts': 3,
    'reconnect_delay': 5
}

# 历史数据回填配置
BACKFILL_CONFIG = {
    'batch_size': 1000,
    'max_concurrent': 10,
    'retry_attempts': 3,
    'retry_delay': 1,
    'progress_report_interval': 100
}
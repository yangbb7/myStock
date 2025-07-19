from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import asyncio
from enum import Enum
from dataclasses import dataclass
import logging

class DataType(Enum):
    TICK = "tick"
    KLINE = "kline"
    ORDERBOOK = "orderbook"
    TRADE = "trade"
    FUNDAMENTAL = "fundamental"
    NEWS = "news"
    ALTERNATIVE = "alternative"

class DataQuality(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INVALID = "invalid"

@dataclass
class DataQualityMetrics:
    completeness: float
    accuracy: float
    timeliness: float
    consistency: float
    validity: float
    overall_score: float
    anomaly_count: int
    last_updated: datetime

@dataclass
class DataRequest:
    symbol: str
    data_type: DataType
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    frequency: Optional[str] = None
    fields: Optional[List[str]] = None
    params: Optional[Dict[str, Any]] = None

@dataclass
class DataResponse:
    data: Any
    metadata: Dict[str, Any]
    quality_metrics: DataQualityMetrics
    source: str
    timestamp: datetime
    latency_ms: float

class BaseDataProvider(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.is_connected = False
        self.connection_stats = {
            'success_count': 0,
            'error_count': 0,
            'last_error': None,
            'uptime': 0
        }
    
    @abstractmethod
    async def connect(self) -> bool:
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        pass
    
    @abstractmethod
    async def get_data(self, request: DataRequest) -> DataResponse:
        pass
    
    @abstractmethod
    async def subscribe_realtime(self, symbols: List[str], data_type: DataType, callback) -> bool:
        pass
    
    @abstractmethod
    async def unsubscribe_realtime(self, symbols: List[str], data_type: DataType) -> bool:
        pass
    
    @abstractmethod
    def get_supported_data_types(self) -> List[DataType]:
        pass
    
    @abstractmethod
    def get_supported_symbols(self) -> List[str]:
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        return {
            'is_connected': self.is_connected,
            'connection_stats': self.connection_stats,
            'provider_name': self.__class__.__name__
        }
    
    def _calculate_quality_metrics(self, data: Any) -> DataQualityMetrics:
        if not data:
            return DataQualityMetrics(0, 0, 0, 0, 0, 0, 1, datetime.now())
        
        completeness = 0.9
        accuracy = 0.95
        timeliness = 0.9
        consistency = 0.92
        validity = 0.88
        overall_score = (completeness + accuracy + timeliness + consistency + validity) / 5
        
        return DataQualityMetrics(
            completeness=completeness,
            accuracy=accuracy,
            timeliness=timeliness,
            consistency=consistency,
            validity=validity,
            overall_score=overall_score,
            anomaly_count=0,
            last_updated=datetime.now()
        )
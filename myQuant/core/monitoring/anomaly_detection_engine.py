import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import pickle
from pathlib import Path
import json
import threading
import time
from collections import deque, defaultdict
import warnings
warnings.filterwarnings('ignore')

class AnomalyType(Enum):
    STATISTICAL = "statistical"
    CLUSTERING = "clustering"
    ISOLATION = "isolation"
    PATTERN = "pattern"
    THRESHOLD = "threshold"
    SEASONAL = "seasonal"
    TREND = "trend"
    POINT = "point"
    CONTEXTUAL = "contextual"
    COLLECTIVE = "collective"

class AnomalySeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class DetectionMethod(Enum):
    Z_SCORE = "z_score"
    MODIFIED_Z_SCORE = "modified_z_score"
    IQR = "iqr"
    ISOLATION_FOREST = "isolation_forest"
    DBSCAN = "dbscan"
    LOCAL_OUTLIER_FACTOR = "local_outlier_factor"
    ONE_CLASS_SVM = "one_class_svm"
    AUTOENCODER = "autoencoder"
    LSTM = "lstm"
    ARIMA = "arima"

class AnomalyStatus(Enum):
    DETECTED = "detected"
    CONFIRMED = "confirmed"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    FALSE_POSITIVE = "false_positive"
    IGNORED = "ignored"

@dataclass
class AnomalyDetectionConfig:
    """异常检测配置"""
    method: DetectionMethod
    sensitivity: float = 0.95
    min_samples: int = 100
    lookback_window: int = 1000
    update_frequency: int = 60
    contamination: float = 0.1
    n_estimators: int = 100
    max_features: int = 10
    random_state: int = 42
    enable_auto_tuning: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AnomalyPoint:
    """异常数据点"""
    timestamp: datetime
    metric_name: str
    value: float
    expected_value: float
    deviation: float
    anomaly_score: float
    method: DetectionMethod
    severity: AnomalySeverity
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AnomalyEvent:
    """异常事件"""
    event_id: str
    timestamp: datetime
    anomaly_type: AnomalyType
    metric_name: str
    description: str
    severity: AnomalySeverity
    detection_method: DetectionMethod
    confidence: float
    affected_metrics: List[str]
    anomaly_points: List[AnomalyPoint]
    context: Dict[str, Any]
    status: AnomalyStatus = AnomalyStatus.DETECTED
    investigation_notes: str = ""
    resolution_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelPerformance:
    """模型性能指标"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    false_positive_rate: float
    false_negative_rate: float
    last_updated: datetime
    training_samples: int
    validation_samples: int
    metadata: Dict[str, Any] = field(default_factory=dict)

class AnomalyDetectionEngine:
    """异常检测引擎"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 检测器配置
        self.detection_configs: Dict[str, AnomalyDetectionConfig] = {}
        self.active_detectors: Dict[str, Any] = {}
        
        # 数据存储
        self.data_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.anomaly_buffer = deque(maxlen=5000)
        self.training_data: Dict[str, pd.DataFrame] = {}
        
        # 模型存储
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.model_performance: Dict[str, ModelPerformance] = {}
        
        # 运行状态
        self.is_running = False
        self.detection_thread = None
        self.stop_event = threading.Event()
        
        # 订阅者
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        
        # 初始化
        self._initialize_detectors()
    
    def _initialize_detectors(self):
        """初始化检测器"""
        try:
            # 加载配置
            detector_configs = self.config.get('detectors', {})
            
            for metric_name, config in detector_configs.items():
                self.detection_configs[metric_name] = AnomalyDetectionConfig(**config)
            
            # 创建默认检测器
            self._create_default_detectors()
            
            # 加载预训练模型
            self._load_pretrained_models()
            
            self.logger.info(f"Initialized {len(self.detection_configs)} anomaly detectors")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize detectors: {e}")
            raise
    
    def _create_default_detectors(self):
        """创建默认检测器"""
        default_metrics = [
            'cpu_usage', 'memory_usage', 'disk_usage', 'network_io',
            'order_latency_ms', 'fill_rate', 'slippage_bps', 'error_rate',
            'orders_per_second', 'trades_per_second'
        ]
        
        for metric in default_metrics:
            if metric not in self.detection_configs:
                # 根据指标类型选择合适的检测方法
                if 'latency' in metric:
                    method = DetectionMethod.ISOLATION_FOREST
                    sensitivity = 0.98
                elif 'rate' in metric:
                    method = DetectionMethod.Z_SCORE
                    sensitivity = 0.95
                elif 'usage' in metric:
                    method = DetectionMethod.IQR
                    sensitivity = 0.90
                else:
                    method = DetectionMethod.MODIFIED_Z_SCORE
                    sensitivity = 0.95
                
                self.detection_configs[metric] = AnomalyDetectionConfig(
                    method=method,
                    sensitivity=sensitivity,
                    min_samples=50,
                    lookback_window=500
                )
    
    def _load_pretrained_models(self):
        """加载预训练模型"""
        model_dir = Path(self.config.get('model_dir', './models'))
        
        if model_dir.exists():
            for model_file in model_dir.glob('*.pkl'):
                try:
                    with open(model_file, 'rb') as f:
                        model_data = pickle.load(f)
                    
                    metric_name = model_file.stem
                    self.models[metric_name] = model_data['model']
                    self.scalers[metric_name] = model_data.get('scaler')
                    self.model_performance[metric_name] = ModelPerformance(**model_data['performance'])
                    
                    self.logger.info(f"Loaded model for {metric_name}")
                    
                except Exception as e:
                    self.logger.error(f"Error loading model {model_file}: {e}")
    
    def start(self):
        """启动异常检测"""
        if self.is_running:
            return
        
        self.is_running = True
        self.stop_event.clear()
        
        # 启动检测线程
        self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.detection_thread.start()
        
        self.logger.info("Anomaly detection engine started")
    
    def stop(self):
        """停止异常检测"""
        if not self.is_running:
            return
        
        self.logger.info("Stopping anomaly detection engine...")
        self.stop_event.set()
        
        if self.detection_thread:
            self.detection_thread.join(timeout=5)
        
        self.is_running = False
        self.logger.info("Anomaly detection engine stopped")
    
    def _detection_loop(self):
        """检测主循环"""
        while not self.stop_event.is_set():
            try:
                # 对每个指标进行异常检测
                for metric_name, config in self.detection_configs.items():
                    if metric_name in self.data_buffer:
                        anomalies = self._detect_anomalies(metric_name, config)
                        
                        for anomaly in anomalies:
                            self._handle_anomaly(anomaly)
                
                # 更新模型
                self._update_models()
                
                time.sleep(self.config.get('detection_interval', 10))
                
            except Exception as e:
                self.logger.error(f"Error in detection loop: {e}")
                time.sleep(1)
    
    def add_data_point(self, metric_name: str, timestamp: datetime, value: float, metadata: Dict[str, Any] = None):
        """添加数据点"""
        data_point = {
            'timestamp': timestamp,
            'value': value,
            'metadata': metadata or {}
        }
        
        self.data_buffer[metric_name].append(data_point)
        
        # 如果需要实时检测
        if self.config.get('real_time_detection', True):
            config = self.detection_configs.get(metric_name)
            if config:
                anomalies = self._detect_anomalies(metric_name, config)
                for anomaly in anomalies:
                    self._handle_anomaly(anomaly)
    
    def _detect_anomalies(self, metric_name: str, config: AnomalyDetectionConfig) -> List[AnomalyEvent]:
        """检测异常"""
        try:
            data = self.data_buffer[metric_name]
            
            if len(data) < config.min_samples:
                return []
            
            # 准备数据
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            # 取最近的数据
            recent_data = df.tail(config.lookback_window)
            
            # 根据检测方法进行异常检测
            if config.method == DetectionMethod.Z_SCORE:
                anomalies = self._detect_zscore_anomalies(metric_name, recent_data, config)
            elif config.method == DetectionMethod.MODIFIED_Z_SCORE:
                anomalies = self._detect_modified_zscore_anomalies(metric_name, recent_data, config)
            elif config.method == DetectionMethod.IQR:
                anomalies = self._detect_iqr_anomalies(metric_name, recent_data, config)
            elif config.method == DetectionMethod.ISOLATION_FOREST:
                anomalies = self._detect_isolation_forest_anomalies(metric_name, recent_data, config)
            elif config.method == DetectionMethod.DBSCAN:
                anomalies = self._detect_dbscan_anomalies(metric_name, recent_data, config)
            else:
                self.logger.warning(f"Unsupported detection method: {config.method}")
                return []
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error detecting anomalies for {metric_name}: {e}")
            return []
    
    def _detect_zscore_anomalies(self, metric_name: str, data: pd.DataFrame, config: AnomalyDetectionConfig) -> List[AnomalyEvent]:
        """Z-Score异常检测"""
        anomalies = []
        
        values = data['value'].values
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            return anomalies
        
        z_scores = np.abs((values - mean) / std)
        threshold = stats.norm.ppf(config.sensitivity)
        
        anomaly_indices = np.where(z_scores > threshold)[0]
        
        for idx in anomaly_indices:
            point = AnomalyPoint(
                timestamp=data.iloc[idx]['timestamp'],
                metric_name=metric_name,
                value=values[idx],
                expected_value=mean,
                deviation=abs(values[idx] - mean),
                anomaly_score=z_scores[idx],
                method=config.method,
                severity=self._calculate_severity(z_scores[idx], threshold)
            )
            
            event = AnomalyEvent(
                event_id=f"{metric_name}_{int(time.time())}_{idx}",
                timestamp=data.iloc[idx]['timestamp'],
                anomaly_type=AnomalyType.STATISTICAL,
                metric_name=metric_name,
                description=f"Z-Score anomaly detected: {values[idx]:.2f} (z-score: {z_scores[idx]:.2f})",
                severity=point.severity,
                detection_method=config.method,
                confidence=min(z_scores[idx] / threshold, 1.0),
                affected_metrics=[metric_name],
                anomaly_points=[point],
                context={'mean': mean, 'std': std, 'threshold': threshold}
            )
            
            anomalies.append(event)
        
        return anomalies
    
    def _detect_modified_zscore_anomalies(self, metric_name: str, data: pd.DataFrame, config: AnomalyDetectionConfig) -> List[AnomalyEvent]:
        """修正Z-Score异常检测"""
        anomalies = []
        
        values = data['value'].values
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        
        if mad == 0:
            return anomalies
        
        modified_z_scores = 0.6745 * (values - median) / mad
        threshold = stats.norm.ppf(config.sensitivity)
        
        anomaly_indices = np.where(np.abs(modified_z_scores) > threshold)[0]
        
        for idx in anomaly_indices:
            point = AnomalyPoint(
                timestamp=data.iloc[idx]['timestamp'],
                metric_name=metric_name,
                value=values[idx],
                expected_value=median,
                deviation=abs(values[idx] - median),
                anomaly_score=abs(modified_z_scores[idx]),
                method=config.method,
                severity=self._calculate_severity(abs(modified_z_scores[idx]), threshold)
            )
            
            event = AnomalyEvent(
                event_id=f"{metric_name}_{int(time.time())}_{idx}",
                timestamp=data.iloc[idx]['timestamp'],
                anomaly_type=AnomalyType.STATISTICAL,
                metric_name=metric_name,
                description=f"Modified Z-Score anomaly detected: {values[idx]:.2f} (modified z-score: {modified_z_scores[idx]:.2f})",
                severity=point.severity,
                detection_method=config.method,
                confidence=min(abs(modified_z_scores[idx]) / threshold, 1.0),
                affected_metrics=[metric_name],
                anomaly_points=[point],
                context={'median': median, 'mad': mad, 'threshold': threshold}
            )
            
            anomalies.append(event)
        
        return anomalies
    
    def _detect_iqr_anomalies(self, metric_name: str, data: pd.DataFrame, config: AnomalyDetectionConfig) -> List[AnomalyEvent]:
        """IQR异常检测"""
        anomalies = []
        
        values = data['value'].values
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        
        if iqr == 0:
            return anomalies
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        anomaly_indices = np.where((values < lower_bound) | (values > upper_bound))[0]
        
        for idx in anomaly_indices:
            expected_value = q1 if values[idx] < lower_bound else q3
            anomaly_score = abs(values[idx] - expected_value) / iqr
            
            point = AnomalyPoint(
                timestamp=data.iloc[idx]['timestamp'],
                metric_name=metric_name,
                value=values[idx],
                expected_value=expected_value,
                deviation=abs(values[idx] - expected_value),
                anomaly_score=anomaly_score,
                method=config.method,
                severity=self._calculate_severity(anomaly_score, 1.5)
            )
            
            event = AnomalyEvent(
                event_id=f"{metric_name}_{int(time.time())}_{idx}",
                timestamp=data.iloc[idx]['timestamp'],
                anomaly_type=AnomalyType.STATISTICAL,
                metric_name=metric_name,
                description=f"IQR anomaly detected: {values[idx]:.2f} (bounds: {lower_bound:.2f}-{upper_bound:.2f})",
                severity=point.severity,
                detection_method=config.method,
                confidence=min(anomaly_score / 1.5, 1.0),
                affected_metrics=[metric_name],
                anomaly_points=[point],
                context={'q1': q1, 'q3': q3, 'iqr': iqr, 'lower_bound': lower_bound, 'upper_bound': upper_bound}
            )
            
            anomalies.append(event)
        
        return anomalies
    
    def _detect_isolation_forest_anomalies(self, metric_name: str, data: pd.DataFrame, config: AnomalyDetectionConfig) -> List[AnomalyEvent]:
        """孤立森林异常检测"""
        anomalies = []
        
        try:
            # 准备特征
            features = self._prepare_features(data)
            
            if len(features) < config.min_samples:
                return anomalies
            
            # 获取或创建模型
            model_key = f"{metric_name}_isolation_forest"
            if model_key not in self.models:
                self.models[model_key] = IsolationForest(
                    contamination=config.contamination,
                    n_estimators=config.n_estimators,
                    max_features=min(config.max_features, features.shape[1]),
                    random_state=config.random_state
                )
                
                # 训练模型
                self.models[model_key].fit(features)
            
            # 预测异常
            predictions = self.models[model_key].predict(features)
            anomaly_scores = -self.models[model_key].score_samples(features)
            
            anomaly_indices = np.where(predictions == -1)[0]
            
            for idx in anomaly_indices:
                point = AnomalyPoint(
                    timestamp=data.iloc[idx]['timestamp'],
                    metric_name=metric_name,
                    value=data.iloc[idx]['value'],
                    expected_value=np.mean(data['value']),
                    deviation=abs(data.iloc[idx]['value'] - np.mean(data['value'])),
                    anomaly_score=anomaly_scores[idx],
                    method=config.method,
                    severity=self._calculate_severity(anomaly_scores[idx], np.mean(anomaly_scores))
                )
                
                event = AnomalyEvent(
                    event_id=f"{metric_name}_{int(time.time())}_{idx}",
                    timestamp=data.iloc[idx]['timestamp'],
                    anomaly_type=AnomalyType.ISOLATION,
                    metric_name=metric_name,
                    description=f"Isolation Forest anomaly detected: {data.iloc[idx]['value']:.2f} (score: {anomaly_scores[idx]:.3f})",
                    severity=point.severity,
                    detection_method=config.method,
                    confidence=min(anomaly_scores[idx] / np.max(anomaly_scores), 1.0),
                    affected_metrics=[metric_name],
                    anomaly_points=[point],
                    context={'contamination': config.contamination, 'n_estimators': config.n_estimators}
                )
                
                anomalies.append(event)
            
        except Exception as e:
            self.logger.error(f"Error in isolation forest detection: {e}")
        
        return anomalies
    
    def _detect_dbscan_anomalies(self, metric_name: str, data: pd.DataFrame, config: AnomalyDetectionConfig) -> List[AnomalyEvent]:
        """DBSCAN异常检测"""
        anomalies = []
        
        try:
            # 准备特征
            features = self._prepare_features(data)
            
            if len(features) < config.min_samples:
                return anomalies
            
            # 标准化特征
            scaler_key = f"{metric_name}_scaler"
            if scaler_key not in self.scalers:
                self.scalers[scaler_key] = StandardScaler()
                features_scaled = self.scalers[scaler_key].fit_transform(features)
            else:
                features_scaled = self.scalers[scaler_key].transform(features)
            
            # DBSCAN聚类
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            cluster_labels = dbscan.fit_predict(features_scaled)
            
            # 标记为-1的点是异常点
            anomaly_indices = np.where(cluster_labels == -1)[0]
            
            for idx in anomaly_indices:
                point = AnomalyPoint(
                    timestamp=data.iloc[idx]['timestamp'],
                    metric_name=metric_name,
                    value=data.iloc[idx]['value'],
                    expected_value=np.mean(data['value']),
                    deviation=abs(data.iloc[idx]['value'] - np.mean(data['value'])),
                    anomaly_score=1.0,  # DBSCAN不提供异常分数
                    method=config.method,
                    severity=AnomalySeverity.MEDIUM
                )
                
                event = AnomalyEvent(
                    event_id=f"{metric_name}_{int(time.time())}_{idx}",
                    timestamp=data.iloc[idx]['timestamp'],
                    anomaly_type=AnomalyType.CLUSTERING,
                    metric_name=metric_name,
                    description=f"DBSCAN anomaly detected: {data.iloc[idx]['value']:.2f} (outlier)",
                    severity=point.severity,
                    detection_method=config.method,
                    confidence=1.0,
                    affected_metrics=[metric_name],
                    anomaly_points=[point],
                    context={'eps': 0.5, 'min_samples': 5}
                )
                
                anomalies.append(event)
            
        except Exception as e:
            self.logger.error(f"Error in DBSCAN detection: {e}")
        
        return anomalies
    
    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """准备特征"""
        features = []
        
        # 基础特征
        features.append(data['value'].values)
        
        # 时间特征
        timestamps = pd.to_datetime(data['timestamp'])
        features.append(timestamps.dt.hour.values)
        features.append(timestamps.dt.minute.values)
        features.append(timestamps.dt.dayofweek.values)
        
        # 统计特征
        rolling_mean = data['value'].rolling(window=10, min_periods=1).mean().values
        rolling_std = data['value'].rolling(window=10, min_periods=1).std().fillna(0).values
        
        features.append(rolling_mean)
        features.append(rolling_std)
        
        # 差分特征
        diff = data['value'].diff().fillna(0).values
        features.append(diff)
        
        return np.column_stack(features)
    
    def _calculate_severity(self, score: float, threshold: float) -> AnomalySeverity:
        """计算异常严重程度"""
        ratio = score / threshold
        
        if ratio >= 3.0:
            return AnomalySeverity.CRITICAL
        elif ratio >= 2.0:
            return AnomalySeverity.HIGH
        elif ratio >= 1.5:
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW
    
    def _handle_anomaly(self, anomaly: AnomalyEvent):
        """处理异常事件"""
        try:
            # 添加到缓冲区
            self.anomaly_buffer.append(anomaly)
            
            # 通知订阅者
            self._notify_subscribers('anomaly_detected', anomaly)
            
            # 记录日志
            self.logger.warning(f"Anomaly detected: {anomaly.description}")
            
        except Exception as e:
            self.logger.error(f"Error handling anomaly: {e}")
    
    def _update_models(self):
        """更新模型"""
        try:
            # 定期重新训练模型
            update_interval = self.config.get('model_update_interval', 3600)  # 1小时
            
            for metric_name, config in self.detection_configs.items():
                if len(self.data_buffer[metric_name]) >= config.min_samples * 2:
                    # 重新训练模型
                    self._retrain_model(metric_name, config)
            
        except Exception as e:
            self.logger.error(f"Error updating models: {e}")
    
    def _retrain_model(self, metric_name: str, config: AnomalyDetectionConfig):
        """重新训练模型"""
        try:
            data = pd.DataFrame(self.data_buffer[metric_name])
            features = self._prepare_features(data)
            
            if config.method == DetectionMethod.ISOLATION_FOREST:
                model_key = f"{metric_name}_isolation_forest"
                self.models[model_key] = IsolationForest(
                    contamination=config.contamination,
                    n_estimators=config.n_estimators,
                    max_features=min(config.max_features, features.shape[1]),
                    random_state=config.random_state
                )
                self.models[model_key].fit(features)
                
                self.logger.info(f"Retrained isolation forest model for {metric_name}")
            
        except Exception as e:
            self.logger.error(f"Error retraining model for {metric_name}: {e}")
    
    def subscribe(self, event_type: str, callback: Callable):
        """订阅事件"""
        self.subscribers[event_type].append(callback)
    
    def _notify_subscribers(self, event_type: str, data: Any):
        """通知订阅者"""
        for callback in self.subscribers[event_type]:
            try:
                callback(data)
            except Exception as e:
                self.logger.error(f"Error notifying subscriber: {e}")
    
    def get_anomalies(self, metric_name: Optional[str] = None,
                     time_range: Optional[Tuple[datetime, datetime]] = None,
                     severity: Optional[AnomalySeverity] = None) -> List[AnomalyEvent]:
        """获取异常事件"""
        anomalies = list(self.anomaly_buffer)
        
        if metric_name:
            anomalies = [a for a in anomalies if a.metric_name == metric_name]
        
        if time_range:
            start_time, end_time = time_range
            anomalies = [a for a in anomalies if start_time <= a.timestamp <= end_time]
        
        if severity:
            anomalies = [a for a in anomalies if a.severity == severity]
        
        return anomalies
    
    def update_anomaly_status(self, event_id: str, status: AnomalyStatus, notes: str = ""):
        """更新异常状态"""
        for anomaly in self.anomaly_buffer:
            if anomaly.event_id == event_id:
                anomaly.status = status
                anomaly.investigation_notes = notes
                if status == AnomalyStatus.RESOLVED:
                    anomaly.resolution_time = datetime.now()
                break
    
    def get_model_performance(self, metric_name: Optional[str] = None) -> Dict[str, ModelPerformance]:
        """获取模型性能"""
        if metric_name:
            return {metric_name: self.model_performance.get(metric_name)}
        return self.model_performance
    
    def save_models(self, model_dir: str):
        """保存模型"""
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)
        
        for metric_name, model in self.models.items():
            try:
                model_data = {
                    'model': model,
                    'scaler': self.scalers.get(f"{metric_name}_scaler"),
                    'performance': self.model_performance.get(metric_name).__dict__ if metric_name in self.model_performance else {}
                }
                
                with open(model_path / f"{metric_name}.pkl", 'wb') as f:
                    pickle.dump(model_data, f)
                
                self.logger.info(f"Saved model for {metric_name}")
                
            except Exception as e:
                self.logger.error(f"Error saving model for {metric_name}: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_anomalies = len(self.anomaly_buffer)
        severity_counts = defaultdict(int)
        status_counts = defaultdict(int)
        
        for anomaly in self.anomaly_buffer:
            severity_counts[anomaly.severity.value] += 1
            status_counts[anomaly.status.value] += 1
        
        return {
            'total_anomalies': total_anomalies,
            'severity_distribution': dict(severity_counts),
            'status_distribution': dict(status_counts),
            'active_detectors': len(self.detection_configs),
            'trained_models': len(self.models),
            'is_running': self.is_running
        }
    
    def __del__(self):
        """析构函数"""
        if hasattr(self, 'is_running') and self.is_running:
            self.stop()
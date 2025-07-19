import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import statistics
import numpy as np

from .base_provider import DataResponse, DataQualityMetrics, DataType

@dataclass
class QualityRule:
    name: str
    rule_type: str
    threshold: float
    severity: str
    enabled: bool = True

@dataclass
class QualityAlert:
    rule_name: str
    severity: str
    message: str
    timestamp: datetime
    data_source: str
    symbol: str
    metrics: Dict[str, Any]

class DataQualityMonitor:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.quality_history = {}
        self.alerts = []
        self.rules = self._initialize_rules()
        
    def _initialize_rules(self) -> List[QualityRule]:
        return [
            QualityRule("completeness_check", "threshold", 0.95, "high"),
            QualityRule("accuracy_check", "threshold", 0.90, "high"),
            QualityRule("timeliness_check", "threshold", 0.85, "medium"),
            QualityRule("consistency_check", "threshold", 0.80, "medium"),
            QualityRule("validity_check", "threshold", 0.90, "high"),
            QualityRule("latency_check", "threshold", 1000, "medium"),
            QualityRule("anomaly_detection", "statistical", 3.0, "high"),
            QualityRule("data_freshness", "time_based", 300, "medium"),
        ]
    
    async def evaluate_data(self, response: DataResponse) -> DataQualityMetrics:
        metrics = self._calculate_comprehensive_metrics(response)
        
        await self._apply_quality_rules(response, metrics)
        
        self._store_quality_history(response, metrics)
        
        return metrics
    
    def _calculate_comprehensive_metrics(self, response: DataResponse) -> DataQualityMetrics:
        data = response.data
        
        completeness = self._calculate_completeness(data)
        accuracy = self._calculate_accuracy(data, response.source)
        timeliness = self._calculate_timeliness(response.timestamp)
        consistency = self._calculate_consistency(data, response.source)
        validity = self._calculate_validity(data)
        
        overall_score = (completeness + accuracy + timeliness + consistency + validity) / 5
        anomaly_count = self._detect_anomalies(data)
        
        return DataQualityMetrics(
            completeness=completeness,
            accuracy=accuracy,
            timeliness=timeliness,
            consistency=consistency,
            validity=validity,
            overall_score=overall_score,
            anomaly_count=anomaly_count,
            last_updated=datetime.now()
        )
    
    def _calculate_completeness(self, data: Any) -> float:
        if not data:
            return 0.0
        
        if isinstance(data, dict):
            expected_fields = ['price', 'volume', 'timestamp']
            present_fields = sum(1 for field in expected_fields if field in data and data[field] is not None)
            return present_fields / len(expected_fields)
        
        elif isinstance(data, list):
            if not data:
                return 0.0
            
            total_records = len(data)
            complete_records = sum(1 for record in data if self._is_record_complete(record))
            return complete_records / total_records
        
        return 0.9
    
    def _calculate_accuracy(self, data: Any, source: str) -> float:
        if not data:
            return 0.0
        
        if isinstance(data, dict):
            accuracy_score = 0.0
            checks = 0
            
            if 'price' in data and data['price'] is not None:
                if data['price'] > 0:
                    accuracy_score += 1
                checks += 1
            
            if 'volume' in data and data['volume'] is not None:
                if data['volume'] >= 0:
                    accuracy_score += 1
                checks += 1
            
            return accuracy_score / checks if checks > 0 else 0.9
        
        return 0.9
    
    def _calculate_timeliness(self, timestamp: datetime) -> float:
        now = datetime.now()
        delay = (now - timestamp).total_seconds()
        
        if delay <= 1:
            return 1.0
        elif delay <= 5:
            return 0.95
        elif delay <= 10:
            return 0.9
        elif delay <= 30:
            return 0.8
        elif delay <= 60:
            return 0.7
        else:
            return 0.5
    
    def _calculate_consistency(self, data: Any, source: str) -> float:
        symbol = data.get('symbol', 'unknown') if isinstance(data, dict) else 'unknown'
        
        if source not in self.quality_history:
            return 0.9
        
        if symbol not in self.quality_history[source]:
            return 0.9
        
        history = self.quality_history[source][symbol]
        if len(history) < 2:
            return 0.9
        
        recent_data = history[-10:]
        prices = [item.get('price', 0) for item in recent_data if 'price' in item]
        
        if len(prices) < 2:
            return 0.9
        
        price_changes = [abs(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices)) if prices[i-1] != 0]
        
        if not price_changes:
            return 0.9
        
        avg_change = statistics.mean(price_changes)
        
        if avg_change > 0.1:
            return 0.6
        elif avg_change > 0.05:
            return 0.8
        else:
            return 0.95
    
    def _calculate_validity(self, data: Any) -> float:
        if not data:
            return 0.0
        
        if isinstance(data, dict):
            validity_score = 0.0
            checks = 0
            
            if 'price' in data:
                if isinstance(data['price'], (int, float)) and data['price'] > 0:
                    validity_score += 1
                checks += 1
            
            if 'volume' in data:
                if isinstance(data['volume'], (int, float)) and data['volume'] >= 0:
                    validity_score += 1
                checks += 1
            
            if 'symbol' in data:
                if isinstance(data['symbol'], str) and len(data['symbol']) > 0:
                    validity_score += 1
                checks += 1
            
            return validity_score / checks if checks > 0 else 0.9
        
        return 0.9
    
    def _detect_anomalies(self, data: Any) -> int:
        if not isinstance(data, dict) or 'price' not in data:
            return 0
        
        price = data['price']
        
        if price <= 0 or price > 1000000:
            return 1
        
        return 0
    
    def _is_record_complete(self, record: Any) -> bool:
        if not isinstance(record, dict):
            return False
        
        required_fields = ['price', 'volume', 'timestamp']
        return all(field in record and record[field] is not None for field in required_fields)
    
    async def _apply_quality_rules(self, response: DataResponse, metrics: DataQualityMetrics):
        for rule in self.rules:
            if not rule.enabled:
                continue
            
            alert = None
            
            if rule.rule_type == "threshold":
                if rule.name == "completeness_check" and metrics.completeness < rule.threshold:
                    alert = self._create_alert(rule, f"Completeness below threshold: {metrics.completeness:.2f}", response)
                elif rule.name == "accuracy_check" and metrics.accuracy < rule.threshold:
                    alert = self._create_alert(rule, f"Accuracy below threshold: {metrics.accuracy:.2f}", response)
                elif rule.name == "timeliness_check" and metrics.timeliness < rule.threshold:
                    alert = self._create_alert(rule, f"Timeliness below threshold: {metrics.timeliness:.2f}", response)
                elif rule.name == "consistency_check" and metrics.consistency < rule.threshold:
                    alert = self._create_alert(rule, f"Consistency below threshold: {metrics.consistency:.2f}", response)
                elif rule.name == "validity_check" and metrics.validity < rule.threshold:
                    alert = self._create_alert(rule, f"Validity below threshold: {metrics.validity:.2f}", response)
                elif rule.name == "latency_check" and response.latency_ms > rule.threshold:
                    alert = self._create_alert(rule, f"Latency above threshold: {response.latency_ms:.2f}ms", response)
            
            elif rule.rule_type == "statistical":
                if rule.name == "anomaly_detection" and metrics.anomaly_count > 0:
                    alert = self._create_alert(rule, f"Anomalies detected: {metrics.anomaly_count}", response)
            
            elif rule.rule_type == "time_based":
                if rule.name == "data_freshness":
                    age = (datetime.now() - response.timestamp).total_seconds()
                    if age > rule.threshold:
                        alert = self._create_alert(rule, f"Data too old: {age:.2f}s", response)
            
            if alert:
                self.alerts.append(alert)
                await self._handle_alert(alert)
    
    def _create_alert(self, rule: QualityRule, message: str, response: DataResponse) -> QualityAlert:
        symbol = response.data.get('symbol', 'unknown') if isinstance(response.data, dict) else 'unknown'
        
        return QualityAlert(
            rule_name=rule.name,
            severity=rule.severity,
            message=message,
            timestamp=datetime.now(),
            data_source=response.source,
            symbol=symbol,
            metrics={'overall_score': response.quality_metrics.overall_score}
        )
    
    async def _handle_alert(self, alert: QualityAlert):
        self.logger.warning(f"Quality Alert [{alert.severity}]: {alert.message} - {alert.data_source}:{alert.symbol}")
        
        if alert.severity == "high":
            pass
    
    def _store_quality_history(self, response: DataResponse, metrics: DataQualityMetrics):
        source = response.source
        symbol = response.data.get('symbol', 'unknown') if isinstance(response.data, dict) else 'unknown'
        
        if source not in self.quality_history:
            self.quality_history[source] = {}
        
        if symbol not in self.quality_history[source]:
            self.quality_history[source][symbol] = []
        
        history_entry = {
            'timestamp': datetime.now(),
            'metrics': metrics,
            'data': response.data
        }
        
        self.quality_history[source][symbol].append(history_entry)
        
        if len(self.quality_history[source][symbol]) > 100:
            self.quality_history[source][symbol] = self.quality_history[source][symbol][-100:]
    
    async def get_overall_metrics(self) -> Dict[str, Any]:
        if not self.quality_history:
            return {'overall_score': 0.0, 'total_alerts': 0}
        
        all_scores = []
        for source_data in self.quality_history.values():
            for symbol_data in source_data.values():
                for entry in symbol_data[-10:]:
                    all_scores.append(entry['metrics'].overall_score)
        
        overall_score = statistics.mean(all_scores) if all_scores else 0.0
        
        recent_alerts = [alert for alert in self.alerts if (datetime.now() - alert.timestamp).total_seconds() < 3600]
        
        return {
            'overall_score': overall_score,
            'total_alerts': len(self.alerts),
            'recent_alerts': len(recent_alerts),
            'high_severity_alerts': len([a for a in recent_alerts if a.severity == "high"]),
            'data_sources_monitored': len(self.quality_history),
            'symbols_monitored': sum(len(source_data) for source_data in self.quality_history.values())
        }
    
    def get_recent_alerts(self, hours: int = 24) -> List[QualityAlert]:
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alerts if alert.timestamp >= cutoff_time]
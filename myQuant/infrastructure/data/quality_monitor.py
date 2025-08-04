# -*- coding: utf-8 -*-
"""
数据质量监控器 - 监控和验证真实数据质量
"""

import logging
import statistics
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class DataQualityMetrics:
    """数据质量指标"""
    
    def __init__(self):
        self.completeness = 0.0  # 完整性
        self.accuracy = 0.0      # 准确性  
        self.consistency = 0.0   # 一致性
        self.timeliness = 0.0    # 及时性
        self.validity = 0.0      # 有效性
        self.overall_score = 0.0 # 综合评分
        self.issues = []         # 质量问题列表
        self.timestamp = datetime.now()


class DataQualityMonitor:
    """数据质量监控器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 质量标准配置
        self.quality_thresholds = {
            'min_completeness': self.config.get('min_completeness', 0.95),
            'max_price_change_percent': self.config.get('max_price_change_percent', 0.2),
            'max_volume_change_ratio': self.config.get('max_volume_change_ratio', 10.0),
            'data_freshness_seconds': self.config.get('data_freshness_seconds', 300),
            'min_data_points': self.config.get('min_data_points', 1),
            'price_range_tolerance': self.config.get('price_range_tolerance', 0.1)
        }
        
        # 历史数据缓存用于一致性检查
        self.historical_cache = {}
        self.quality_history = []
        
    def validate_realtime_data(self, data: Dict[str, Dict[str, Any]]) -> DataQualityMetrics:
        """验证实时数据质量
        
        Args:
            data: 实时数据字典 {symbol: {price, volume, timestamp, ...}}
            
        Returns:
            DataQualityMetrics: 数据质量评估结果
        """
        metrics = DataQualityMetrics()
        
        if not data:
            metrics.issues.append("数据为空")
            return metrics
            
        total_symbols = len(data)
        valid_symbols = 0
        
        completeness_issues = []
        accuracy_issues = []
        consistency_issues = []
        timeliness_issues = []
        validity_issues = []
        
        for symbol, symbol_data in data.items():
            symbol_valid = True
            
            # 1. 完整性检查
            required_fields = ['current_price', 'timestamp', 'symbol']
            missing_fields = [field for field in required_fields if field not in symbol_data]
            if missing_fields:
                completeness_issues.append(f"{symbol}缺少字段: {missing_fields}")
                symbol_valid = False
            
            # 2. 有效性检查
            if 'current_price' in symbol_data:
                price = symbol_data['current_price']
                if not isinstance(price, (int, float)) or price <= 0:
                    validity_issues.append(f"{symbol}价格无效: {price}")
                    symbol_valid = False
                elif price > 10000:  # 异常高价检查
                    validity_issues.append(f"{symbol}价格异常高: {price}")
                    
            if 'volume' in symbol_data:
                volume = symbol_data.get('volume', 0)
                if volume < 0:
                    validity_issues.append(f"{symbol}成交量无效: {volume}")
                    symbol_valid = False
            
            # 3. 及时性检查
            if 'timestamp' in symbol_data:
                try:
                    if isinstance(symbol_data['timestamp'], str):
                        timestamp = datetime.fromisoformat(symbol_data['timestamp'].replace('Z', '+00:00'))
                    else:
                        timestamp = symbol_data['timestamp']
                    
                    age_seconds = (datetime.now() - timestamp.replace(tzinfo=None)).total_seconds()
                    if age_seconds > self.quality_thresholds['data_freshness_seconds']:
                        timeliness_issues.append(f"{symbol}数据过期: {age_seconds:.1f}秒")
                        
                except Exception as e:
                    timeliness_issues.append(f"{symbol}时间戳格式错误: {e}")
                    symbol_valid = False
            
            # 4. 一致性检查（与历史数据对比）
            if symbol in self.historical_cache and 'current_price' in symbol_data:
                historical_price = self.historical_cache[symbol].get('price', 0)
                if historical_price > 0:
                    current_price = symbol_data['current_price']
                    change_percent = abs(current_price - historical_price) / historical_price
                    if change_percent > self.quality_thresholds['max_price_change_percent']:
                        consistency_issues.append(
                            f"{symbol}价格变动异常: {change_percent:.2%} "
                            f"(从{historical_price}到{current_price})"
                        )
                        
                # 成交量一致性检查
                if 'volume' in symbol_data:
                    historical_volume = self.historical_cache[symbol].get('volume', 0)
                    current_volume = symbol_data['volume']
                    if historical_volume > 0 and current_volume > 0:
                        volume_ratio = current_volume / historical_volume
                        if volume_ratio > self.quality_thresholds['max_volume_change_ratio']:
                            consistency_issues.append(
                                f"{symbol}成交量变动异常: {volume_ratio:.1f}倍"
                            )
            
            if symbol_valid:
                valid_symbols += 1
                # 更新历史缓存
                self.historical_cache[symbol] = {
                    'price': symbol_data.get('current_price', 0),
                    'volume': symbol_data.get('volume', 0),
                    'timestamp': datetime.now()
                }
        
        # 计算质量指标
        metrics.completeness = 1.0 - len(completeness_issues) / total_symbols
        metrics.validity = 1.0 - len(validity_issues) / total_symbols
        metrics.timeliness = 1.0 - len(timeliness_issues) / total_symbols
        metrics.consistency = 1.0 - len(consistency_issues) / total_symbols
        metrics.accuracy = valid_symbols / total_symbols if total_symbols > 0 else 0.0
        
        # 综合评分（加权平均）
        weights = {'completeness': 0.25, 'validity': 0.25, 'timeliness': 0.2, 
                  'consistency': 0.15, 'accuracy': 0.15}
        metrics.overall_score = (
            metrics.completeness * weights['completeness'] +
            metrics.validity * weights['validity'] +
            metrics.timeliness * weights['timeliness'] +
            metrics.consistency * weights['consistency'] +
            metrics.accuracy * weights['accuracy']
        )
        
        # 收集所有问题
        metrics.issues = (completeness_issues + accuracy_issues + 
                         consistency_issues + timeliness_issues + validity_issues)
        
        # 记录质量历史
        self.quality_history.append({
            'timestamp': datetime.now(),
            'overall_score': metrics.overall_score,
            'completeness': metrics.completeness,
            'validity': metrics.validity,
            'timeliness': metrics.timeliness,
            'consistency': metrics.consistency,
            'accuracy': metrics.accuracy,
            'issues_count': len(metrics.issues)
        })
        
        # 保持历史记录在合理范围内
        if len(self.quality_history) > 1000:
            self.quality_history = self.quality_history[-500:]
        
        return metrics
    
    def validate_historical_data(self, data: pd.DataFrame, symbol: str) -> DataQualityMetrics:
        """验证历史数据质量
        
        Args:
            data: 历史数据DataFrame
            symbol: 股票代码
            
        Returns:
            DataQualityMetrics: 数据质量评估结果
        """
        metrics = DataQualityMetrics()
        
        if data.empty:
            metrics.issues.append("历史数据为空")
            return metrics
        
        total_records = len(data)
        issues = []
        
        # 1. 完整性检查
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            issues.append(f"缺少必要列: {missing_columns}")
        
        # 检查缺失值
        missing_count = data[required_columns].isnull().sum().sum()
        metrics.completeness = 1.0 - missing_count / (total_records * len(required_columns))
        
        # 2. 有效性检查
        validity_issues = []
        
        # 价格有效性
        if 'open' in data.columns:
            invalid_open = (data['open'] <= 0).sum()
            if invalid_open > 0:
                validity_issues.append(f"无效开盘价数量: {invalid_open}")
        
        if 'close' in data.columns:
            invalid_close = (data['close'] <= 0).sum()
            if invalid_close > 0:
                validity_issues.append(f"无效收盘价数量: {invalid_close}")
        
        # 成交量有效性
        if 'volume' in data.columns:
            invalid_volume = (data['volume'] < 0).sum()
            if invalid_volume > 0:
                validity_issues.append(f"无效成交量数量: {invalid_volume}")
        
        # OHLC逻辑一致性
        if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            inconsistent_ohlc = (
                (data['high'] < data['low']) |
                (data['high'] < data['open']) |
                (data['high'] < data['close']) |
                (data['low'] > data['open']) |
                (data['low'] > data['close'])
            ).sum()
            if inconsistent_ohlc > 0:
                validity_issues.append(f"OHLC逻辑不一致数量: {inconsistent_ohlc}")
        
        metrics.validity = 1.0 - len(validity_issues) / max(1, total_records)
        
        # 3. 一致性检查（价格连续性）
        consistency_issues = []
        if 'close' in data.columns and len(data) > 1:
            data_sorted = data.sort_values('date')
            price_changes = data_sorted['close'].pct_change().fillna(0)
            extreme_changes = (abs(price_changes) > 0.5).sum()  # 50%以上变动
            if extreme_changes > 0:
                consistency_issues.append(f"极端价格变动数量: {extreme_changes}")
        
        metrics.consistency = 1.0 - len(consistency_issues) / max(1, total_records)
        
        # 4. 及时性检查（数据连续性）
        timeliness_issues = []
        if 'date' in data.columns and len(data) > 1:
            data_sorted = data.sort_values('date')
            date_gaps = data_sorted['date'].diff().dt.days
            large_gaps = (date_gaps > 7).sum()  # 超过7天的间隔
            if large_gaps > 0:
                timeliness_issues.append(f"数据间隔过大数量: {large_gaps}")
        
        metrics.timeliness = 1.0 - len(timeliness_issues) / max(1, total_records)
        
        # 5. 准确性评估（基于数据分布合理性）
        accuracy_score = 1.0
        if 'close' in data.columns:
            close_std = data['close'].std()
            close_mean = data['close'].mean()
            if close_mean > 0:
                cv = close_std / close_mean  # 变异系数
                if cv > 2.0:  # 变异系数过大可能表示数据异常
                    accuracy_score -= 0.2
        
        metrics.accuracy = accuracy_score
        
        # 综合评分
        weights = {'completeness': 0.3, 'validity': 0.25, 'consistency': 0.2, 
                  'timeliness': 0.15, 'accuracy': 0.1}
        metrics.overall_score = (
            metrics.completeness * weights['completeness'] +
            metrics.validity * weights['validity'] +
            metrics.consistency * weights['consistency'] +
            metrics.timeliness * weights['timeliness'] +
            metrics.accuracy * weights['accuracy']
        )
        
        # 收集所有问题
        metrics.issues = issues + validity_issues + consistency_issues + timeliness_issues
        
        return metrics
    
    def is_data_acceptable(self, metrics: DataQualityMetrics) -> bool:
        """判断数据质量是否可接受
        
        Args:
            metrics: 数据质量指标
            
        Returns:
            bool: 数据质量是否可接受
        """
        min_acceptable_score = self.config.get('min_acceptable_score', 0.8)
        return metrics.overall_score >= min_acceptable_score
    
    def get_quality_report(self) -> Dict[str, Any]:
        """获取数据质量报告
        
        Returns:
            Dict[str, Any]: 质量报告
        """
        if not self.quality_history:
            return {'status': 'no_data', 'message': '暂无质量数据'}
        
        recent_scores = [item['overall_score'] for item in self.quality_history[-50:]]
        recent_issues = [item['issues_count'] for item in self.quality_history[-50:]]
        
        return {
            'status': 'active',
            'current_score': recent_scores[-1] if recent_scores else 0,
            'average_score_50': statistics.mean(recent_scores) if recent_scores else 0,
            'min_score_50': min(recent_scores) if recent_scores else 0,
            'max_score_50': max(recent_scores) if recent_scores else 0,
            'total_evaluations': len(self.quality_history),
            'recent_issues_avg': statistics.mean(recent_issues) if recent_issues else 0,
            'last_updated': self.quality_history[-1]['timestamp'].isoformat(),
            'quality_trend': self._calculate_quality_trend(),
            'cached_symbols': len(self.historical_cache)
        }
    
    def _calculate_quality_trend(self) -> str:
        """计算质量趋势
        
        Returns:
            str: 趋势描述 ('improving', 'stable', 'declining')
        """
        if len(self.quality_history) < 10:
            return 'insufficient_data'
        
        recent_scores = [item['overall_score'] for item in self.quality_history[-10:]]
        earlier_scores = [item['overall_score'] for item in self.quality_history[-20:-10]]
        
        if not earlier_scores:
            return 'insufficient_data'
        
        recent_avg = statistics.mean(recent_scores)
        earlier_avg = statistics.mean(earlier_scores)
        
        diff = recent_avg - earlier_avg
        
        if diff > 0.05:
            return 'improving'
        elif diff < -0.05:
            return 'declining'
        else:
            return 'stable'
    
    def cleanup_cache(self, max_age_hours: int = 24):
        """清理过期的历史缓存
        
        Args:
            max_age_hours: 最大保留时间（小时）
        """
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        # 清理历史缓存
        expired_symbols = []
        for symbol, data in self.historical_cache.items():
            if data.get('timestamp', datetime.min) < cutoff_time:
                expired_symbols.append(symbol)
        
        for symbol in expired_symbols:
            del self.historical_cache[symbol]
        
        if expired_symbols:
            self.logger.info(f"清理了{len(expired_symbols)}个过期缓存项")
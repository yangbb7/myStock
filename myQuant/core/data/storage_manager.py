import asyncio
import logging
import sqlite3
import json
import os
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import hashlib
import pandas as pd

from .base_provider import DataResponse, DataType

@dataclass
class ShardConfig:
    shard_type: str
    partition_key: str
    retention_days: int
    compression: bool = True
    index_fields: List[str] = None

class ShardedStorageManager:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.base_path = self.config.get('storage_path', 'data/shards')
        self.shard_configs = self._initialize_shard_configs()
        self.connection_pool = {}
        
        os.makedirs(self.base_path, exist_ok=True)
    
    def _initialize_shard_configs(self) -> Dict[str, ShardConfig]:
        return {
            'tick_data': ShardConfig(
                shard_type='time_based',
                partition_key='date',
                retention_days=30,
                compression=True,
                index_fields=['symbol', 'timestamp']
            ),
            'kline_data': ShardConfig(
                shard_type='symbol_based',
                partition_key='symbol',
                retention_days=365,
                compression=True,
                index_fields=['symbol', 'timestamp', 'frequency']
            ),
            'orderbook_data': ShardConfig(
                shard_type='time_based',
                partition_key='date',
                retention_days=7,
                compression=True,
                index_fields=['symbol', 'timestamp']
            ),
            'fundamental_data': ShardConfig(
                shard_type='symbol_based',
                partition_key='symbol',
                retention_days=1095,
                compression=False,
                index_fields=['symbol', 'report_date', 'data_type']
            ),
            'news_data': ShardConfig(
                shard_type='time_based',
                partition_key='date',
                retention_days=180,
                compression=True,
                index_fields=['timestamp', 'category', 'symbols']
            )
        }
    
    async def store_data(self, response: DataResponse) -> bool:
        try:
            shard_key = self._determine_shard_key(response)
            shard_path = self._get_shard_path(shard_key, response.metadata.get('data_type', 'unknown'))
            
            await self._write_to_shard(shard_path, response)
            
            await self._update_metadata(shard_key, response)
            
            self.logger.debug(f"Stored data to shard: {shard_key}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store data: {e}")
            return False
    
    async def retrieve_data(self, 
                          symbol: str, 
                          data_type: DataType, 
                          start_time: datetime, 
                          end_time: datetime,
                          filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        
        shard_keys = self._get_relevant_shards(symbol, data_type, start_time, end_time)
        
        all_data = []
        
        for shard_key in shard_keys:
            shard_path = self._get_shard_path(shard_key, data_type.value)
            
            try:
                shard_data = await self._read_from_shard(shard_path, symbol, start_time, end_time, filters)
                all_data.extend(shard_data)
            except Exception as e:
                self.logger.error(f"Failed to read from shard {shard_key}: {e}")
        
        return sorted(all_data, key=lambda x: x.get('timestamp', datetime.min))
    
    async def cleanup_old_data(self):
        cleanup_count = 0
        
        for data_type, config in self.shard_configs.items():
            cutoff_date = datetime.now() - timedelta(days=config.retention_days)
            
            shard_pattern = f"{data_type}_*"
            shard_dir = os.path.join(self.base_path, data_type)
            
            if not os.path.exists(shard_dir):
                continue
            
            for shard_file in os.listdir(shard_dir):
                shard_path = os.path.join(shard_dir, shard_file)
                
                if os.path.isfile(shard_path):
                    file_time = datetime.fromtimestamp(os.path.getctime(shard_path))
                    
                    if file_time < cutoff_date:
                        try:
                            os.remove(shard_path)
                            cleanup_count += 1
                            self.logger.info(f"Cleaned up old shard: {shard_path}")
                        except Exception as e:
                            self.logger.error(f"Failed to cleanup shard {shard_path}: {e}")
        
        self.logger.info(f"Cleanup completed. Removed {cleanup_count} old shards.")
        return cleanup_count
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        stats = {
            'total_shards': 0,
            'total_size_mb': 0,
            'data_type_stats': {},
            'oldest_data': None,
            'newest_data': None
        }
        
        for data_type in self.shard_configs.keys():
            shard_dir = os.path.join(self.base_path, data_type)
            
            if not os.path.exists(shard_dir):
                continue
            
            type_stats = {
                'shard_count': 0,
                'size_mb': 0,
                'oldest': None,
                'newest': None
            }
            
            for shard_file in os.listdir(shard_dir):
                shard_path = os.path.join(shard_dir, shard_file)
                
                if os.path.isfile(shard_path):
                    type_stats['shard_count'] += 1
                    stats['total_shards'] += 1
                    
                    size_mb = os.path.getsize(shard_path) / (1024 * 1024)
                    type_stats['size_mb'] += size_mb
                    stats['total_size_mb'] += size_mb
                    
                    file_time = datetime.fromtimestamp(os.path.getctime(shard_path))
                    
                    if type_stats['oldest'] is None or file_time < type_stats['oldest']:
                        type_stats['oldest'] = file_time
                    
                    if type_stats['newest'] is None or file_time > type_stats['newest']:
                        type_stats['newest'] = file_time
                    
                    if stats['oldest_data'] is None or file_time < stats['oldest_data']:
                        stats['oldest_data'] = file_time
                    
                    if stats['newest_data'] is None or file_time > stats['newest_data']:
                        stats['newest_data'] = file_time
            
            stats['data_type_stats'][data_type] = type_stats
        
        return stats
    
    def _determine_shard_key(self, response: DataResponse) -> str:
        data_type = response.metadata.get('data_type', 'unknown')
        config = self.shard_configs.get(data_type)
        
        if not config:
            return f"unknown_{datetime.now().strftime('%Y%m%d')}"
        
        if config.shard_type == 'time_based':
            return f"{data_type}_{response.timestamp.strftime('%Y%m%d')}"
        
        elif config.shard_type == 'symbol_based':
            symbol = response.data.get('symbol', 'unknown') if isinstance(response.data, dict) else 'unknown'
            symbol_hash = hashlib.md5(symbol.encode()).hexdigest()[:8]
            return f"{data_type}_{symbol_hash}"
        
        else:
            return f"{data_type}_{datetime.now().strftime('%Y%m%d')}"
    
    def _get_shard_path(self, shard_key: str, data_type: str) -> str:
        shard_dir = os.path.join(self.base_path, data_type)
        os.makedirs(shard_dir, exist_ok=True)
        return os.path.join(shard_dir, f"{shard_key}.db")
    
    async def _write_to_shard(self, shard_path: str, response: DataResponse):
        conn = sqlite3.connect(shard_path)
        
        try:
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS data_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    timestamp REAL,
                    data_type TEXT,
                    source TEXT,
                    data TEXT,
                    metadata TEXT,
                    quality_score REAL,
                    created_at REAL
                )
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_symbol_timestamp 
                ON data_records(symbol, timestamp)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_data_type_timestamp 
                ON data_records(data_type, timestamp)
            ''')
            
            symbol = response.data.get('symbol', 'unknown') if isinstance(response.data, dict) else 'unknown'
            
            cursor.execute('''
                INSERT INTO data_records 
                (symbol, timestamp, data_type, source, data, metadata, quality_score, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol,
                response.timestamp.timestamp(),
                response.metadata.get('data_type', 'unknown'),
                response.source,
                json.dumps(response.data, default=str),
                json.dumps(response.metadata, default=str),
                response.quality_metrics.overall_score,
                datetime.now().timestamp()
            ))
            
            conn.commit()
            
        finally:
            conn.close()
    
    async def _read_from_shard(self, 
                             shard_path: str, 
                             symbol: str, 
                             start_time: datetime, 
                             end_time: datetime,
                             filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        
        if not os.path.exists(shard_path):
            return []
        
        conn = sqlite3.connect(shard_path)
        
        try:
            cursor = conn.cursor()
            
            query = '''
                SELECT symbol, timestamp, data_type, source, data, metadata, quality_score, created_at
                FROM data_records
                WHERE symbol = ? AND timestamp BETWEEN ? AND ?
            '''
            
            params = [symbol, start_time.timestamp(), end_time.timestamp()]
            
            if filters:
                for key, value in filters.items():
                    if key == 'min_quality':
                        query += ' AND quality_score >= ?'
                        params.append(value)
                    elif key == 'source':
                        query += ' AND source = ?'
                        params.append(value)
            
            query += ' ORDER BY timestamp'
            
            cursor.execute(query, params)
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'symbol': row[0],
                    'timestamp': datetime.fromtimestamp(row[1]),
                    'data_type': row[2],
                    'source': row[3],
                    'data': json.loads(row[4]),
                    'metadata': json.loads(row[5]),
                    'quality_score': row[6],
                    'created_at': datetime.fromtimestamp(row[7])
                })
            
            return results
            
        finally:
            conn.close()
    
    def _get_relevant_shards(self, 
                           symbol: str, 
                           data_type: DataType, 
                           start_time: datetime, 
                           end_time: datetime) -> List[str]:
        
        config = self.shard_configs.get(data_type.value)
        if not config:
            return []
        
        shards = []
        
        if config.shard_type == 'time_based':
            current_date = start_time.date()
            end_date = end_time.date()
            
            while current_date <= end_date:
                shard_key = f"{data_type.value}_{current_date.strftime('%Y%m%d')}"
                shards.append(shard_key)
                current_date += timedelta(days=1)
        
        elif config.shard_type == 'symbol_based':
            symbol_hash = hashlib.md5(symbol.encode()).hexdigest()[:8]
            shard_key = f"{data_type.value}_{symbol_hash}"
            shards.append(shard_key)
        
        return shards
    
    async def _update_metadata(self, shard_key: str, response: DataResponse):
        metadata_path = os.path.join(self.base_path, 'metadata', f"{shard_key}.json")
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        
        try:
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {
                    'shard_key': shard_key,
                    'created_at': datetime.now().isoformat(),
                    'record_count': 0,
                    'size_bytes': 0,
                    'symbols': set(),
                    'data_sources': set(),
                    'quality_avg': 0.0
                }
            
            metadata['record_count'] += 1
            metadata['last_updated'] = datetime.now().isoformat()
            
            if isinstance(response.data, dict) and 'symbol' in response.data:
                if isinstance(metadata['symbols'], list):
                    metadata['symbols'] = set(metadata['symbols'])
                metadata['symbols'].add(response.data['symbol'])
            
            if isinstance(metadata['data_sources'], list):
                metadata['data_sources'] = set(metadata['data_sources'])
            metadata['data_sources'].add(response.source)
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, default=list, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to update metadata for {shard_key}: {e}")
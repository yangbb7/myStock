# -*- coding: utf-8 -*-
"""
统一数据库管理器
"""

import sqlite3
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from contextlib import contextmanager
from datetime import datetime
import json
import shutil

from .database_config import DatabaseConfig, get_database_config, Environment


class DatabaseManager:
    """统一数据库管理器 - 全局最优解实现"""
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or get_database_config()
        self.logger = logging.getLogger(__name__)
        self._connections: Dict[str, sqlite3.Connection] = {}
        
    def initialize(self) -> None:
        """初始化数据库系统"""
        self.logger.info("Initializing unified database system...")
        
        # 创建必要的目录
        self.config.data_path.mkdir(parents=True, exist_ok=True)
        self.config.shards_path.mkdir(parents=True, exist_ok=True)
        
        # 创建主数据库
        self._create_main_database()
        
        # 创建分片数据库目录
        self._create_shard_directories()
        
        self.logger.info("Database system initialized successfully")
    
    def _create_main_database(self) -> None:
        """创建主数据库"""
        if not self.config.database_path.exists():
            self.logger.info(f"Creating main database at {self.config.database_path}")
            
            with self.get_connection() as conn:
                # 创建价格数据表
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS price_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        open REAL NOT NULL,
                        high REAL NOT NULL,
                        low REAL NOT NULL,
                        close REAL NOT NULL,
                        volume INTEGER NOT NULL,
                        adj_close REAL NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(date, symbol)
                    )
                ''')
                
                # 创建财务数据表
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS financial_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        report_date TEXT NOT NULL,
                        eps REAL,
                        revenue REAL,
                        net_profit REAL,
                        roe REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(symbol, report_date)
                    )
                ''')
                
                # 创建元数据表
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS metadata (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        key TEXT UNIQUE NOT NULL,
                        value TEXT NOT NULL,
                        description TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # 插入初始元数据
                conn.execute('''
                    INSERT OR REPLACE INTO metadata (key, value, description) 
                    VALUES (?, ?, ?)
                ''', (
                    'database_version', '1.0.0', 'Database schema version'
                ))
                
                conn.execute('''
                    INSERT OR REPLACE INTO metadata (key, value, description) 
                    VALUES (?, ?, ?)
                ''', (
                    'created_at', datetime.now().isoformat(), 'Database creation timestamp'
                ))
                
                conn.commit()
    
    def _create_shard_directories(self) -> None:
        """创建分片数据库目录"""
        shard_types = ['tick', 'kline', 'fundamental', 'alternative']
        for shard_type in shard_types:
            shard_path = self.config.get_shard_path(shard_type)
            shard_path.mkdir(parents=True, exist_ok=True)
    
    @contextmanager
    def get_connection(self, database_path: Optional[Path] = None):
        """获取数据库连接"""
        db_path = database_path or self.config.database_path
        
        try:
            conn = sqlite3.connect(str(db_path), timeout=self.config.connection_timeout)
            conn.row_factory = sqlite3.Row  # 启用字典式访问
            conn.execute("PRAGMA journal_mode=WAL")  # 启用WAL模式
            conn.execute("PRAGMA synchronous=NORMAL")  # 优化同步模式
            conn.execute("PRAGMA cache_size=10000")  # 设置缓存大小
            conn.execute("PRAGMA temp_store=MEMORY")  # 临时存储在内存中
            yield conn
        except Exception as e:
            self.logger.error(f"Database connection error: {e}")
            raise
        finally:
            if 'conn' in locals():
                conn.close()
    
    def migrate_existing_databases(self) -> None:
        """迁移现有数据库到统一架构"""
        self.logger.info("Starting database migration to unified architecture...")
        
        # 查找所有现有的数据库文件
        existing_databases = []
        
        # 检查myQuant/data/myquant.db
        old_db_path = self.config.project_root / "myQuant" / "data" / "myquant.db"
        if old_db_path.exists():
            existing_databases.append(old_db_path)
        
        # 检查tests/data/myquant.db
        test_db_path = self.config.project_root / "tests" / "data" / "myquant.db"
        if test_db_path.exists():
            existing_databases.append(test_db_path)
        
        # 合并数据到主数据库
        if existing_databases:
            self.logger.info(f"Found {len(existing_databases)} existing databases to migrate")
            self._merge_databases(existing_databases)
        
        # 清理旧数据库文件
        self._cleanup_old_databases(existing_databases)
        
        self.logger.info("Database migration completed successfully")
    
    def _merge_databases(self, source_databases: List[Path]) -> None:
        """合并多个数据库到主数据库"""
        with self.get_connection() as main_conn:
            for db_path in source_databases:
                self.logger.info(f"Merging database: {db_path}")
                
                # 附加源数据库
                main_conn.execute(f"ATTACH DATABASE '{db_path}' AS source_db")
                
                try:
                    # 迁移价格数据
                    main_conn.execute('''
                        INSERT OR REPLACE INTO price_data 
                        (date, symbol, open, high, low, close, volume, adj_close)
                        SELECT date, symbol, open, high, low, close, volume, adj_close 
                        FROM source_db.price_data
                    ''')
                    
                    # 迁移财务数据
                    main_conn.execute('''
                        INSERT OR REPLACE INTO financial_data 
                        (symbol, report_date, eps, revenue, net_profit, roe)
                        SELECT symbol, report_date, eps, revenue, net_profit, roe 
                        FROM source_db.financial_data
                    ''')
                    
                    main_conn.commit()
                    self.logger.info(f"Successfully merged data from {db_path}")
                    
                except sqlite3.Error as e:
                    self.logger.warning(f"Error merging database {db_path}: {e}")
                    main_conn.rollback()
                
                finally:
                    # 分离源数据库
                    main_conn.execute("DETACH DATABASE source_db")
    
    def _cleanup_old_databases(self, old_databases: List[Path]) -> None:
        """清理旧数据库文件"""
        backup_dir = self.config.data_path / "backups" / datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        for db_path in old_databases:
            if db_path.exists():
                # 创建备份
                backup_path = backup_dir / f"{db_path.parent.name}_{db_path.name}"
                shutil.copy2(db_path, backup_path)
                self.logger.info(f"Backed up {db_path} to {backup_path}")
                
                # 删除原文件
                db_path.unlink()
                self.logger.info(f"Removed old database: {db_path}")
    
    def get_database_info(self) -> Dict[str, Any]:
        """获取数据库信息"""
        info = {
            'config': self.config.to_dict(),
            'main_database': {
                'path': str(self.config.database_path),
                'exists': self.config.database_path.exists(),
                'size': 0,
                'tables': []
            },
            'shards': {},
            'environment': self.config.environment.value
        }
        
        if self.config.database_path.exists():
            info['main_database']['size'] = self.config.database_path.stat().st_size
            
            with self.get_connection() as conn:
                # 获取表信息
                tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                for table in tables:
                    table_name = table['name']
                    if table_name != 'sqlite_sequence':
                        count = conn.execute(f"SELECT COUNT(*) as count FROM {table_name}").fetchone()
                        info['main_database']['tables'].append({
                            'name': table_name,
                            'record_count': count['count']
                        })
        
        # 获取分片信息
        if self.config.shards_path.exists():
            for shard_dir in self.config.shards_path.iterdir():
                if shard_dir.is_dir():
                    shard_files = list(shard_dir.glob("*.db"))
                    info['shards'][shard_dir.name] = {
                        'path': str(shard_dir),
                        'file_count': len(shard_files),
                        'total_size': sum(f.stat().st_size for f in shard_files)
                    }
        
        return info
    
    def health_check(self) -> Dict[str, Any]:
        """数据库健康检查"""
        health = {
            'status': 'healthy',
            'issues': [],
            'recommendations': []
        }
        
        # 检查主数据库
        if not self.config.database_path.exists():
            health['status'] = 'error'
            health['issues'].append('Main database does not exist')
            health['recommendations'].append('Run database initialization')
        else:
            try:
                with self.get_connection() as conn:
                    conn.execute("SELECT 1").fetchone()
            except Exception as e:
                health['status'] = 'error'
                health['issues'].append(f'Cannot connect to main database: {e}')
                health['recommendations'].append('Check database file permissions and integrity')
        
        # 检查数据目录
        if not self.config.data_path.exists():
            health['status'] = 'warning'
            health['issues'].append('Data directory does not exist')
            health['recommendations'].append('Create data directory structure')
        
        # 检查分片目录
        if not self.config.shards_path.exists():
            health['status'] = 'warning'
            health['issues'].append('Shards directory does not exist')
            health['recommendations'].append('Create shards directory structure')
        
        return health
    
    def optimize_database(self) -> None:
        """优化数据库性能"""
        self.logger.info("Starting database optimization...")
        
        with self.get_connection() as conn:
            # 分析查询计划
            conn.execute("ANALYZE")
            
            # 重建索引
            conn.execute("REINDEX")
            
            # 清理数据库
            conn.execute("VACUUM")
            
            # 更新统计信息
            conn.execute("PRAGMA optimize")
            
            conn.commit()
        
        self.logger.info("Database optimization completed")
    
    def backup_database(self, backup_name: Optional[str] = None) -> Path:
        """备份数据库"""
        if backup_name is None:
            backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backup_dir = self.config.data_path / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        backup_path = backup_dir / f"{backup_name}.db"
        
        with self.get_connection() as conn:
            backup_conn = sqlite3.connect(str(backup_path))
            conn.backup(backup_conn)
            backup_conn.close()
        
        self.logger.info(f"Database backed up to {backup_path}")
        return backup_path
    
    def restore_database(self, backup_path: Path) -> None:
        """从备份恢复数据库"""
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_path}")
        
        # 备份当前数据库
        current_backup = self.backup_database("pre_restore")
        
        try:
            # 恢复数据库
            shutil.copy2(backup_path, self.config.database_path)
            self.logger.info(f"Database restored from {backup_path}")
        except Exception as e:
            # 恢复失败，回滚
            shutil.copy2(current_backup, self.config.database_path)
            self.logger.error(f"Database restore failed, rolled back: {e}")
            raise
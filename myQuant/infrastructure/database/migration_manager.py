# -*- coding: utf-8 -*-
"""
数据库迁移管理器
"""

import sqlite3
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from .database_config import DatabaseConfig, get_database_config


class MigrationManager:
    """数据库迁移管理器"""
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or get_database_config()
        self.logger = logging.getLogger(__name__)
        self.migrations_dir = Path(__file__).parent / "migrations"
        self.migrations_dir.mkdir(exist_ok=True)
    
    def create_migration_script(self, name: str, sql_up: str, sql_down: str) -> Path:
        """创建迁移脚本"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{name}.json"
        migration_path = self.migrations_dir / filename
        
        migration_data = {
            "name": name,
            "timestamp": timestamp,
            "sql_up": sql_up,
            "sql_down": sql_down,
            "created_at": datetime.now().isoformat()
        }
        
        with open(migration_path, 'w', encoding='utf-8') as f:
            json.dump(migration_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Created migration script: {migration_path}")
        return migration_path
    
    def run_migration(self, migration_file: Path) -> None:
        """运行迁移"""
        with open(migration_file, 'r', encoding='utf-8') as f:
            migration_data = json.load(f)
        
        migration_name = migration_data["name"]
        sql_up = migration_data["sql_up"]
        
        self.logger.info(f"Running migration: {migration_name}")
        
        with sqlite3.connect(str(self.config.database_path)) as conn:
            try:
                # 创建迁移历史表
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS migration_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        migration_name TEXT UNIQUE NOT NULL,
                        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        sql_executed TEXT
                    )
                ''')
                
                # 检查是否已应用
                existing = conn.execute(
                    "SELECT 1 FROM migration_history WHERE migration_name = ?",
                    (migration_name,)
                ).fetchone()
                
                if existing:
                    self.logger.info(f"Migration {migration_name} already applied")
                    return
                
                # 执行迁移
                conn.executescript(sql_up)
                
                # 记录迁移历史
                conn.execute(
                    "INSERT INTO migration_history (migration_name, sql_executed) VALUES (?, ?)",
                    (migration_name, sql_up)
                )
                
                conn.commit()
                self.logger.info(f"Migration {migration_name} completed successfully")
                
            except Exception as e:
                conn.rollback()
                self.logger.error(f"Migration {migration_name} failed: {e}")
                raise
    
    def rollback_migration(self, migration_file: Path) -> None:
        """回滚迁移"""
        with open(migration_file, 'r', encoding='utf-8') as f:
            migration_data = json.load(f)
        
        migration_name = migration_data["name"]
        sql_down = migration_data["sql_down"]
        
        self.logger.info(f"Rolling back migration: {migration_name}")
        
        with sqlite3.connect(str(self.config.database_path)) as conn:
            try:
                # 执行回滚
                conn.executescript(sql_down)
                
                # 删除迁移历史记录
                conn.execute(
                    "DELETE FROM migration_history WHERE migration_name = ?",
                    (migration_name,)
                )
                
                conn.commit()
                self.logger.info(f"Migration {migration_name} rolled back successfully")
                
            except Exception as e:
                conn.rollback()
                self.logger.error(f"Migration rollback {migration_name} failed: {e}")
                raise
    
    def get_migration_status(self) -> Dict[str, Any]:
        """获取迁移状态"""
        status = {
            "applied_migrations": [],
            "pending_migrations": [],
            "total_migrations": 0
        }
        
        # 获取所有迁移文件
        migration_files = sorted(self.migrations_dir.glob("*.json"))
        status["total_migrations"] = len(migration_files)
        
        if not self.config.database_path.exists():
            status["pending_migrations"] = [f.stem for f in migration_files]
            return status
        
        with sqlite3.connect(str(self.config.database_path)) as conn:
            # 获取已应用的迁移
            try:
                applied = conn.execute(
                    "SELECT migration_name FROM migration_history ORDER BY applied_at"
                ).fetchall()
                status["applied_migrations"] = [row[0] for row in applied]
            except sqlite3.OperationalError:
                # 迁移历史表不存在
                status["applied_migrations"] = []
        
        # 计算待应用的迁移
        applied_names = set(status["applied_migrations"])
        for migration_file in migration_files:
            with open(migration_file, 'r', encoding='utf-8') as f:
                migration_data = json.load(f)
                migration_name = migration_data["name"]
                
                if migration_name not in applied_names:
                    status["pending_migrations"].append(migration_name)
        
        return status
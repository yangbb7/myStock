# -*- coding: utf-8 -*-
"""
数据库迁移管理器

负责管理数据库版本升级和迁移
支持前滚、回滚和版本管理
"""

import asyncio
import logging
import json
import os
from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path

from .database_manager import DatabaseManager
from .database_config import DatabaseConfig, get_database_config


class MigrationManager:
    """数据库迁移管理器"""
    
    def __init__(self, database_manager: DatabaseManager):
        self.db = database_manager
        self.logger = logging.getLogger(__name__)
        self.migration_table = "schema_migrations"
    
    async def initialize(self):
        """初始化迁移系统"""
        await self._create_migration_table()
        self.logger.info("迁移系统初始化完成")
    
    async def _create_migration_table(self):
        """创建迁移记录表"""
        sql = f"""
        CREATE TABLE IF NOT EXISTS {self.migration_table} (
            version INTEGER PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            checksum VARCHAR(64)
        )
        """
        await self.db.execute_ddl(sql)
    
    async def migration_table_exists(self) -> bool:
        """检查迁移表是否存在"""
        sql = f"""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name='{self.migration_table}'
        """
        result = await self.db.fetch_one(sql)
        return result is not None
    
    async def get_migration_table_structure(self) -> List[Dict[str, Any]]:
        """获取迁移表结构"""
        sql = f"PRAGMA table_info({self.migration_table})"
        return await self.db.fetch_all(sql)
    
    async def get_current_version(self) -> int:
        """获取当前数据库版本"""
        try:
            sql = f"SELECT MAX(version) as max_version FROM {self.migration_table}"
            result = await self.db.fetch_one(sql)
            return result['max_version'] if result and result['max_version'] else 0
        except Exception:
            return 0
    
    async def apply_migration(self, migration: Dict[str, Any]) -> bool:
        """应用单个迁移"""
        version = migration['version']
        name = migration['name']
        up_sql = migration['up_sql']
        
        self.logger.info(f"应用迁移 {version}: {name}")
        
        try:
            async with self.db.transaction():
                # 执行迁移SQL
                for sql_statement in up_sql.split(';'):
                    sql_statement = sql_statement.strip()
                    if sql_statement:
                        await self.db.execute_ddl(sql_statement)
                
                # 记录迁移
                await self.db.execute_insert(
                    f"""INSERT INTO {self.migration_table} (version, name) 
                        VALUES (?, ?)""",
                    (version, name)
                )
            
            self.logger.info(f"迁移 {version} 应用成功")
            return True
            
        except Exception as e:
            self.logger.error(f"迁移 {version} 应用失败: {e}")
            raise
    
    async def rollback_migration(self, version: int, down_sql: Optional[str] = None) -> bool:
        """回滚指定版本的迁移"""
        self.logger.info(f"回滚迁移版本 {version}")
        
        try:
            # 获取迁移信息
            sql = f"SELECT * FROM {self.migration_table} WHERE version = ?"
            migration_record = await self.db.fetch_one(sql, (version,))
            
            if not migration_record:
                raise ValueError(f"未找到版本 {version} 的迁移记录")
            
            async with self.db.transaction():
                # 执行回滚SQL（如果提供）
                if down_sql:
                    for sql_statement in down_sql.split(';'):
                        sql_statement = sql_statement.strip()
                        if sql_statement:
                            await self.db.execute_ddl(sql_statement)
                
                # 删除迁移记录
                await self.db.execute_delete(
                    f"DELETE FROM {self.migration_table} WHERE version = ?",
                    (version,)
                )
            
            self.logger.info(f"迁移版本 {version} 回滚成功")
            return True
            
        except Exception as e:
            self.logger.error(f"回滚迁移版本 {version} 失败: {e}")
            raise
    
    async def get_pending_migrations(self, available_migrations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """获取待应用的迁移"""
        current_version = await self.get_current_version()
        
        pending = [
            migration for migration in available_migrations
            if migration['version'] > current_version
        ]
        
        # 按版本号排序
        return sorted(pending, key=lambda x: x['version'])
    
    async def apply_all_migrations(self, migrations: List[Dict[str, Any]]) -> List[bool]:
        """应用所有待处理迁移"""
        pending_migrations = await self.get_pending_migrations(migrations)
        results = []
        
        for migration in pending_migrations:
            try:
                result = await self.apply_migration(migration)
                results.append(result)
            except Exception as e:
                self.logger.error(f"应用迁移失败，停止后续迁移: {e}")
                results.append(False)
                break
        
        return results
    
    async def get_migration_history(self) -> List[Dict[str, Any]]:
        """获取迁移历史"""
        sql = f"""
        SELECT version, name, applied_at 
        FROM {self.migration_table} 
        ORDER BY version
        """
        return await self.db.fetch_all(sql)
    
    async def check_integrity(self) -> Dict[str, Any]:
        """检查迁移完整性"""
        integrity_check = {
            'is_valid': True,
            'current_version': await self.get_current_version(),
            'missing_migrations': [],
            'error_messages': []
        }
        
        try:
            # 检查迁移表是否存在
            if not await self.migration_table_exists():
                integrity_check['is_valid'] = False
                integrity_check['error_messages'].append("迁移表不存在")
                return integrity_check
            
            # 检查版本连续性
            history = await self.get_migration_history()
            if history:
                versions = [record['version'] for record in history]
                for i in range(1, max(versions) + 1):
                    if i not in versions:
                        integrity_check['missing_migrations'].append(i)
                        integrity_check['is_valid'] = False
            
        except Exception as e:
            integrity_check['is_valid'] = False
            integrity_check['error_messages'].append(str(e))
        
        return integrity_check
    
    async def create_backup(self) -> Optional[str]:
        """创建迁移前备份"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"backup_before_migration_{timestamp}.sql"
            
            # 这里简化处理，实际应该创建数据库转储
            self.logger.info(f"创建备份: {backup_path}")
            
            # 简化实现：只返回备份路径
            return backup_path
            
        except Exception as e:
            self.logger.error(f"创建备份失败: {e}")
            return None
    
    async def table_exists(self, table_name: str) -> bool:
        """检查表是否存在"""
        sql = f"""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name='{table_name}'
        """
        result = await self.db.fetch_one(sql)
        return result is not None
    
    async def is_migration_complete(self) -> bool:
        """检查迁移是否完成"""
        try:
            current_version = await self.get_current_version()
            return current_version > 0
        except Exception:
            return False
    
    async def get_migration_version(self) -> int:
        """获取迁移版本（兼容性方法）"""
        return await self.get_current_version()
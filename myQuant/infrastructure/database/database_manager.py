"""
数据库管理器

负责数据库连接、表创建、事务管理等核心功能
"""

import os
import sqlite3
import asyncio
import aiosqlite
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from contextlib import asynccontextmanager
from datetime import datetime
import logging


class DatabaseManager:
    """统一数据库管理器"""
    
    def __init__(self, database_url: str):
        """初始化数据库管理器
        
        Args:
            database_url: 数据库连接URL
        """
        self.database_url = database_url
        self.connection: Optional[aiosqlite.Connection] = None
        self.logger = logging.getLogger(__name__)
        self._is_connected = False
        self._migration_version = 1
        
    async def initialize(self) -> None:
        """初始化数据库"""
        self.logger.info("Initializing database...")
        
        # 解析数据库URL
        if self.database_url.startswith("sqlite://"):
            db_path = self.database_url[10:]  # 移除 "sqlite://" 前缀
            
            # 如果不是内存数据库，创建目录
            if db_path != ":memory:":
                db_file = Path(db_path)
                db_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 连接数据库
            self.connection = await aiosqlite.connect(db_path)
            self.connection.row_factory = aiosqlite.Row
            
            # 启用外键约束
            await self.connection.execute("PRAGMA foreign_keys = ON")
            
            self._is_connected = True
            
            # 创建表结构
            await self._create_tables()
            
            self.logger.info("Database initialized successfully")
        else:
            raise ValueError(f"Unsupported database URL: {self.database_url}")
    
    async def close(self) -> None:
        """关闭数据库连接"""
        if self.connection:
            await self.connection.close()
            self.connection = None
            self._is_connected = False
            self.logger.info("Database connection closed")
    
    def is_connected(self) -> bool:
        """检查数据库是否连接"""
        return self._is_connected
    
    async def get_migration_version(self) -> int:
        """获取迁移版本"""
        return self._migration_version
    
    async def is_migration_complete(self) -> bool:
        """检查迁移是否完成"""
        return self._migration_version > 0
    
    async def get_table_names(self) -> List[str]:
        """获取所有表名"""
        if not self.connection:
            raise RuntimeError("Database not connected")
        
        cursor = await self.connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        rows = await cursor.fetchall()
        await cursor.close()
        
        return [row['name'] for row in rows]
    
    @asynccontextmanager
    async def transaction(self):
        """事务上下文管理器"""
        if not self.connection:
            raise RuntimeError("Database not connected")
        
        await self.connection.execute("BEGIN")
        try:
            yield
            await self.connection.commit()
        except Exception:
            await self.connection.rollback()
            raise
    
    async def execute_insert(self, query: str, params: Tuple = (), auto_commit: bool = True) -> Optional[int]:
        """执行插入操作并返回ID"""
        if not self.connection:
            raise RuntimeError("Database not connected")
        
        cursor = await self.connection.execute(query, params)
        row_id = cursor.lastrowid
        await cursor.close()
        if auto_commit:
            await self.connection.commit()
        return row_id
    
    async def execute_update(self, query: str, params: Tuple = (), auto_commit: bool = True) -> int:
        """执行更新操作并返回影响行数"""
        if not self.connection:
            raise RuntimeError("Database not connected")
        
        cursor = await self.connection.execute(query, params)
        rows_affected = cursor.rowcount
        await cursor.close()
        if auto_commit:
            await self.connection.commit()
        return rows_affected
    
    async def execute_delete(self, query: str, params: Tuple = (), auto_commit: bool = True) -> int:
        """执行删除操作并返回影响行数"""
        if not self.connection:
            raise RuntimeError("Database not connected")
        
        cursor = await self.connection.execute(query, params)
        rows_affected = cursor.rowcount
        await cursor.close()
        if auto_commit:
            await self.connection.commit()
        return rows_affected
    
    async def fetch_one(self, query: str, params: Tuple = ()) -> Optional[Dict[str, Any]]:
        """查询单条记录"""
        if not self.connection:
            raise RuntimeError("Database not connected")
        
        cursor = await self.connection.execute(query, params)
        row = await cursor.fetchone()
        await cursor.close()
        
        return dict(row) if row else None
    
    async def fetch_all(self, query: str, params: Tuple = ()) -> List[Dict[str, Any]]:
        """查询多条记录"""
        if not self.connection:
            raise RuntimeError("Database not connected")
        
        cursor = await self.connection.execute(query, params)
        rows = await cursor.fetchall()
        await cursor.close()
        
        return [dict(row) for row in rows]
    
    async def fetch_scalar(self, query: str, params: Tuple = ()) -> Any:
        """查询单个值"""
        if not self.connection:
            raise RuntimeError("Database not connected")
        
        cursor = await self.connection.execute(query, params)
        row = await cursor.fetchone()
        await cursor.close()
        
        return row[0] if row else None
    
    async def execute_batch_insert(self, query: str, params_list: List[Tuple]) -> None:
        """批量插入"""
        if not self.connection:
            raise RuntimeError("Database not connected")
        
        await self.connection.executemany(query, params_list)
        await self.connection.commit()
    
    async def execute_ddl(self, query: str, params: Tuple = ()) -> None:
        """执行DDL语句（CREATE, ALTER, DROP等）"""
        if not self.connection:
            raise RuntimeError("Database not connected")
        
        cursor = await self.connection.execute(query, params)
        await cursor.close()
        await self.connection.commit()
    
    async def _create_tables(self) -> None:
        """创建所有表"""
        # 用户表
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username VARCHAR(50) UNIQUE NOT NULL,
                email VARCHAR(100) UNIQUE,
                password_hash VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE
            )
        """)
        
        # 用户配置表
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS user_configs (
                user_id INTEGER PRIMARY KEY,
                risk_tolerance DECIMAL(3,2) DEFAULT 0.02,
                max_position_size DECIMAL(3,2) DEFAULT 0.10,
                notification_settings TEXT,
                trading_preferences TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        # 股票基础信息表
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS stocks (
                symbol VARCHAR(20) PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                sector VARCHAR(50),
                industry VARCHAR(100),
                market VARCHAR(10) NOT NULL,
                listing_date DATE,
                total_shares BIGINT,
                float_shares BIGINT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # K线数据表
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS kline_daily (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol VARCHAR(20) NOT NULL,
                trade_date DATE NOT NULL,
                open_price DECIMAL(10,3) NOT NULL,
                high_price DECIMAL(10,3) NOT NULL,
                low_price DECIMAL(10,3) NOT NULL,
                close_price DECIMAL(10,3) NOT NULL,
                volume BIGINT DEFAULT 0,
                turnover DECIMAL(15,2) DEFAULT 0,
                UNIQUE(symbol, trade_date)
            )
        """)
        
        # 实时行情表
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS real_time_quotes (
                symbol VARCHAR(20) PRIMARY KEY,
                current_price DECIMAL(10,3),
                change_amount DECIMAL(10,3),
                change_percent DECIMAL(5,2),
                volume BIGINT,
                turnover DECIMAL(15,2),
                bid_price_1 DECIMAL(10,3),
                bid_volume_1 BIGINT,
                ask_price_1 DECIMAL(10,3),
                ask_volume_1 BIGINT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 订单表
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                id VARCHAR(36) PRIMARY KEY,
                user_id INTEGER NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                order_type VARCHAR(20) NOT NULL,
                side VARCHAR(10) NOT NULL,
                quantity INTEGER NOT NULL,
                price DECIMAL(10,3),
                stop_price DECIMAL(10,3),
                time_in_force VARCHAR(10) DEFAULT 'DAY',
                filled_quantity INTEGER DEFAULT 0,
                average_fill_price DECIMAL(10,3),
                status VARCHAR(20) DEFAULT 'PENDING',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        # 持仓表
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                quantity INTEGER NOT NULL,
                average_price DECIMAL(10,3) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, symbol),
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        # 交易记录表
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id VARCHAR(36) NOT NULL,
                user_id INTEGER NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                side VARCHAR(10) NOT NULL,
                quantity INTEGER NOT NULL,
                price DECIMAL(10,3) NOT NULL,
                commission DECIMAL(8,2) DEFAULT 0,
                executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (order_id) REFERENCES orders(id),
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        # 策略配置表
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS strategies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                name VARCHAR(100) NOT NULL,
                type VARCHAR(50) NOT NULL,
                parameters TEXT NOT NULL,
                is_active BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        # 提醒设置表
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                alert_type VARCHAR(50) NOT NULL,
                condition_type VARCHAR(20) NOT NULL,
                threshold_value DECIMAL(15,6),
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                triggered_at TIMESTAMP NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        # 风险管理表
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS risk_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                date DATE NOT NULL,
                portfolio_value DECIMAL(15,2),
                daily_pnl DECIMAL(15,2),
                max_drawdown DECIMAL(5,2),
                var_95 DECIMAL(15,2),
                beta DECIMAL(5,3),
                sharpe_ratio DECIMAL(5,3),
                calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, date),
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        # 创建索引
        await self._create_indexes()
        
        await self.connection.commit()
        self.logger.info("All tables created successfully")
    
    async def _create_indexes(self) -> None:
        """创建索引"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_kline_symbol_date ON kline_daily(symbol, trade_date DESC)",
            "CREATE INDEX IF NOT EXISTS idx_orders_user_status_created ON orders(user_id, status, created_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_transactions_user_date ON transactions(user_id, executed_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_positions_user_symbol ON positions(user_id, symbol)",
            "CREATE INDEX IF NOT EXISTS idx_alerts_user_active ON alerts(user_id, is_active)",
            "CREATE INDEX IF NOT EXISTS idx_risk_metrics_user_date ON risk_metrics(user_id, date DESC)"
        ]
        
        for index_sql in indexes:
            await self.connection.execute(index_sql)
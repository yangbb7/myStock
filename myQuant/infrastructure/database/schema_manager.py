"""
数据库Schema管理器

负责创建、管理和验证数据库表结构
包括表创建、索引优化、约束管理等功能
"""

import asyncio
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime

from myQuant.infrastructure.database.database_manager import DatabaseManager


class SchemaManager:
    """数据库Schema管理器"""
    
    def __init__(self, database_manager: DatabaseManager):
        self.db = database_manager
        self.logger = logging.getLogger(__name__)
    
    async def create_all_tables(self) -> bool:
        """创建所有表结构"""
        try:
            self.logger.info("开始创建所有数据库表")
            
            # 按依赖顺序创建表
            await self.create_users_table()
            await self.create_user_configs_table()
            await self.create_stocks_table()
            await self.create_kline_daily_table()
            await self.create_real_time_quotes_table()
            await self.create_orders_table()
            await self.create_positions_table()
            await self.create_transactions_table()
            await self.create_strategies_table()
            await self.create_alerts_table()
            await self.create_risk_metrics_table()
            
            # 创建索引
            await self.create_performance_indexes()
            await self.create_partial_indexes()
            
            self.logger.info("所有数据库表创建完成")
            return True
            
        except Exception as e:
            self.logger.error(f"创建数据库表失败: {e}")
            raise

    async def create_users_table(self):
        """创建用户表"""
        sql = """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username VARCHAR(50) UNIQUE NOT NULL,
            email VARCHAR(100) UNIQUE,
            password_hash VARCHAR(255) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT TRUE
        )
        """
        await self.db.execute_ddl(sql)

    async def create_user_configs_table(self):
        """创建用户配置表"""
        sql = """
        CREATE TABLE IF NOT EXISTS user_configs (
            user_id INTEGER PRIMARY KEY,
            risk_tolerance DECIMAL(3,2) DEFAULT 0.02,
            max_position_size DECIMAL(3,2) DEFAULT 0.10,
            notification_settings JSON,
            trading_preferences JSON,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """
        await self.db.execute_ddl(sql)

    async def create_stocks_table(self):
        """创建股票基础信息表"""
        sql = """
        CREATE TABLE IF NOT EXISTS stocks (
            symbol VARCHAR(20) PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            sector VARCHAR(50),
            industry VARCHAR(100),
            market VARCHAR(10) NOT NULL, -- SH/SZ
            listing_date DATE,
            total_shares BIGINT,
            float_shares BIGINT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        await self.db.execute_ddl(sql)

    async def create_kline_daily_table(self):
        """创建K线数据表"""
        sql = """
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
        """
        await self.db.execute_ddl(sql)

    async def create_real_time_quotes_table(self):
        """创建实时行情表"""
        sql = """
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
        """
        await self.db.execute_ddl(sql)

    async def create_orders_table(self):
        """创建订单表"""
        sql = """
        CREATE TABLE IF NOT EXISTS orders (
            id VARCHAR(36) PRIMARY KEY, -- UUID
            user_id INTEGER NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            order_type VARCHAR(20) NOT NULL, -- MARKET/LIMIT/STOP
            side VARCHAR(10) NOT NULL, -- BUY/SELL
            quantity INTEGER NOT NULL,
            price DECIMAL(10,3),
            stop_price DECIMAL(10,3),
            filled_quantity INTEGER DEFAULT 0,
            average_fill_price DECIMAL(10,3),
            status VARCHAR(20) DEFAULT 'PENDING', -- PENDING/FILLED/CANCELLED/REJECTED
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """
        await self.db.execute_ddl(sql)

    async def create_positions_table(self):
        """创建持仓表"""
        sql = """
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
        """
        await self.db.execute_ddl(sql)

    async def create_transactions_table(self):
        """创建交易记录表"""
        sql = """
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
        """
        await self.db.execute_ddl(sql)

    async def create_strategies_table(self):
        """创建策略配置表"""
        sql = """
        CREATE TABLE IF NOT EXISTS strategies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            name VARCHAR(100) NOT NULL,
            type VARCHAR(50) NOT NULL,
            parameters JSON NOT NULL,
            is_active BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """
        await self.db.execute_ddl(sql)

    async def create_alerts_table(self):
        """创建提醒设置表"""
        sql = """
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            alert_type VARCHAR(50) NOT NULL, -- PRICE/VOLUME/INDICATOR
            condition_type VARCHAR(20) NOT NULL, -- ABOVE/BELOW/CROSS
            threshold_value DECIMAL(15,6),
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            triggered_at TIMESTAMP NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """
        await self.db.execute_ddl(sql)

    async def create_risk_metrics_table(self):
        """创建风险管理表"""
        sql = """
        CREATE TABLE IF NOT EXISTS risk_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            date DATE NOT NULL,
            portfolio_value DECIMAL(15,2),
            daily_pnl DECIMAL(15,2),
            max_drawdown DECIMAL(5,2),
            var_95 DECIMAL(15,2), -- Value at Risk
            beta DECIMAL(5,3),
            sharpe_ratio DECIMAL(5,3),
            calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, date),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """
        await self.db.execute_ddl(sql)

    async def create_performance_indexes(self):
        """创建性能优化索引"""
        indexes = [
            # K线数据复合索引
            "CREATE INDEX IF NOT EXISTS idx_kline_symbol_date ON kline_daily(symbol, trade_date DESC)",
            
            # 订单表索引
            "CREATE INDEX IF NOT EXISTS idx_orders_user_status_created ON orders(user_id, status, created_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_orders_symbol_created ON orders(symbol, created_at DESC)",
            
            # 交易记录索引
            "CREATE INDEX IF NOT EXISTS idx_transactions_user_date ON transactions(user_id, executed_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_transactions_symbol_date ON transactions(symbol, executed_at DESC)",
            
            # 持仓表索引
            "CREATE INDEX IF NOT EXISTS idx_positions_user ON positions(user_id)",
            
            # 提醒表索引
            "CREATE INDEX IF NOT EXISTS idx_alerts_user_active ON alerts(user_id, is_active)",
            
            # 风险指标索引
            "CREATE INDEX IF NOT EXISTS idx_risk_metrics_user_date ON risk_metrics(user_id, date DESC)"
        ]
        
        for index_sql in indexes:
            await self.db.execute_ddl(index_sql)

    async def create_partial_indexes(self):
        """创建部分索引优化查询"""
        partial_indexes = [
            # 活跃订单部分索引
            "CREATE INDEX IF NOT EXISTS idx_pending_orders ON orders(user_id, symbol) WHERE status = 'PENDING'",
            
            # 活跃提醒部分索引
            "CREATE INDEX IF NOT EXISTS idx_active_alerts ON alerts(user_id, symbol) WHERE is_active = TRUE"
        ]
        
        for index_sql in partial_indexes:
            try:
                await self.db.execute_ddl(index_sql)
            except Exception as e:
                # SQLite某些版本可能不支持部分索引
                self.logger.warning(f"创建部分索引失败（可能不支持）: {e}")

    async def get_existing_tables(self) -> List[str]:
        """获取已存在的表列表"""
        sql = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        result = await self.db.fetch_all(sql)
        return [row['name'] for row in result]

    async def get_table_structure(self, table_name: str) -> List[Dict[str, Any]]:
        """获取表结构信息"""
        sql = f"PRAGMA table_info({table_name})"
        result = await self.db.fetch_all(sql)
        return result

    async def get_table_constraints(self, table_name: str) -> List[str]:
        """获取表约束信息"""
        # SQLite中获取约束信息
        sql = f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'"
        result = await self.db.fetch_one(sql)
        if result and result['sql']:
            return [result['sql']]
        return []

    async def get_foreign_keys(self, table_name: str) -> List[Dict[str, Any]]:
        """获取外键信息"""
        sql = f"PRAGMA foreign_key_list({table_name})"
        result = await self.db.fetch_all(sql)
        return result

    async def get_table_indexes(self, table_name: str) -> List[Dict[str, Any]]:
        """获取表索引信息"""
        sql = f"PRAGMA index_list({table_name})"
        result = await self.db.fetch_all(sql)
        return result

    async def get_partial_indexes(self, table_name: str) -> List[str]:
        """获取部分索引信息"""
        sql = f"""
        SELECT sql FROM sqlite_master 
        WHERE type='index' AND tbl_name='{table_name}' AND sql LIKE '%WHERE%'
        """
        result = await self.db.fetch_all(sql)
        return [row['sql'] for row in result if row['sql']]

    async def validate_schema(self) -> Dict[str, Any]:
        """验证Schema完整性"""
        validation_result = {
            'is_valid': True,
            'missing_tables': [],
            'missing_indexes': [],
            'constraint_violations': []
        }
        
        # 检查必需的表
        required_tables = [
            'users', 'user_configs', 'stocks', 'kline_daily', 'real_time_quotes',
            'orders', 'positions', 'transactions', 'strategies', 'alerts', 'risk_metrics'
        ]
        
        existing_tables = await self.get_existing_tables()
        
        for table in required_tables:
            if table not in existing_tables:
                validation_result['missing_tables'].append(table)
                validation_result['is_valid'] = False
        
        # 检查关键索引
        if 'kline_daily' in existing_tables:
            indexes = await self.get_table_indexes('kline_daily')
            index_names = [idx['name'] for idx in indexes]
            if not any('symbol' in name and 'date' in name for name in index_names):
                validation_result['missing_indexes'].append('kline_daily symbol-date index')
                validation_result['is_valid'] = False
        
        return validation_result

    async def drop_all_tables(self):
        """删除所有用户表（谨慎使用）"""
        self.logger.warning("正在删除所有数据库表")
        
        tables = await self.get_existing_tables()
        
        # 按相反顺序删除表（考虑外键约束）
        drop_order = [
            'risk_metrics', 'alerts', 'strategies', 'transactions', 
            'positions', 'orders', 'real_time_quotes', 'kline_daily',
            'stocks', 'user_configs', 'users'
        ]
        
        for table in drop_order:
            if table in tables:
                await self.db.execute_ddl(f"DROP TABLE IF EXISTS {table}")
        
        self.logger.info("所有表已删除")

    async def table_exists(self, table_name: str) -> bool:
        """检查表是否存在"""
        existing_tables = await self.get_existing_tables()
        return table_name in existing_tables
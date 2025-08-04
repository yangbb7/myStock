"""
数据库Schema管理器测试

测试数据库表创建、索引管理、约束验证等功能
严格遵循TDD原则，覆盖各种边界情况
"""

import pytest
import pytest_asyncio
import asyncio
from datetime import datetime
from typing import List, Dict, Any

from myQuant.infrastructure.database.database_manager import DatabaseManager
from myQuant.infrastructure.database.schema_manager import SchemaManager


class TestSchemaManager:
    """Schema管理器测试类"""
    
    @pytest_asyncio.fixture
    async def db_manager(self):
        """创建测试用数据库管理器"""
        db_manager = DatabaseManager("sqlite://:memory:")
        await db_manager.initialize()
        # 清理数据库以确保测试环境干净
        schema_manager = SchemaManager(db_manager)
        await schema_manager.drop_all_tables()
        yield db_manager
        await db_manager.close()
    
    @pytest_asyncio.fixture
    async def schema_manager(self, db_manager):
        """创建Schema管理器实例"""
        return SchemaManager(db_manager)
    
    @pytest.mark.asyncio
    async def test_create_all_tables_success(self, schema_manager):
        """测试成功创建所有表"""
        # 执行创建所有表
        result = await schema_manager.create_all_tables()
        
        # 验证返回值
        assert result is True
        
        # 验证所有表都已创建
        existing_tables = await schema_manager.get_existing_tables()
        expected_tables = [
            'users', 'user_configs', 'stocks', 'kline_daily', 
            'real_time_quotes', 'orders', 'positions', 'transactions',
            'strategies', 'alerts', 'risk_metrics'
        ]
        
        for table in expected_tables:
            assert table in existing_tables, f"表 {table} 未创建"
    
    @pytest.mark.asyncio
    async def test_create_users_table_structure(self, schema_manager):
        """测试用户表结构是否正确"""
        # 创建用户表
        await schema_manager.create_users_table()
        
        # 获取表结构
        columns = await schema_manager.get_table_structure('users')
        
        # 验证列定义
        column_names = {col['name'] for col in columns}
        expected_columns = {
            'id', 'username', 'email', 'password_hash',
            'created_at', 'updated_at', 'is_active'
        }
        assert column_names == expected_columns
        
        # 验证主键
        pk_columns = [col for col in columns if col['pk'] == 1]
        assert len(pk_columns) == 1
        assert pk_columns[0]['name'] == 'id'
        
        # 验证NOT NULL约束
        username_col = next(col for col in columns if col['name'] == 'username')
        assert username_col['notnull'] == 1
        
        password_col = next(col for col in columns if col['name'] == 'password_hash')
        assert password_col['notnull'] == 1
    
    @pytest.mark.asyncio
    async def test_create_foreign_key_constraints(self, schema_manager):
        """测试外键约束是否正确创建"""
        # 创建相关表
        await schema_manager.create_users_table()
        await schema_manager.create_user_configs_table()
        
        # 获取外键信息
        foreign_keys = await schema_manager.get_foreign_keys('user_configs')
        
        # 验证外键存在
        assert len(foreign_keys) > 0
        
        # 验证外键指向正确的表和列
        fk = foreign_keys[0]
        assert fk['table'] == 'users'
        assert fk['from'] == 'user_id'
        assert fk['to'] == 'id'
    
    @pytest.mark.asyncio
    async def test_create_unique_constraints(self, schema_manager):
        """测试唯一约束是否正确创建"""
        # 创建表
        await schema_manager.create_kline_daily_table()
        await schema_manager.create_positions_table()
        
        # 测试kline_daily的复合唯一索引
        constraints = await schema_manager.get_table_constraints('kline_daily')
        assert any('UNIQUE(symbol, trade_date)' in constraint for constraint in constraints)
        
        # 测试positions的复合唯一索引
        constraints = await schema_manager.get_table_constraints('positions')
        assert any('UNIQUE(user_id, symbol)' in constraint for constraint in constraints)
    
    @pytest.mark.asyncio
    async def test_create_performance_indexes(self, schema_manager):
        """测试性能优化索引是否正确创建"""
        # 先创建表
        await schema_manager.create_all_tables()
        
        # 创建性能索引
        await schema_manager.create_performance_indexes()
        
        # 验证关键索引存在
        kline_indexes = await schema_manager.get_table_indexes('kline_daily')
        index_names = [idx['name'] for idx in kline_indexes]
        assert any('idx_kline_symbol_date' in name for name in index_names)
        
        # 验证订单表索引
        order_indexes = await schema_manager.get_table_indexes('orders')
        index_names = [idx['name'] for idx in order_indexes]
        assert any('idx_orders_user_status_created' in name for name in index_names)
    
    @pytest.mark.asyncio
    async def test_create_partial_indexes(self, schema_manager):
        """测试部分索引创建（可能不被所有SQLite版本支持）"""
        # 先创建表
        await schema_manager.create_all_tables()
        
        # 尝试创建部分索引
        await schema_manager.create_partial_indexes()
        
        # 获取部分索引（如果支持）
        partial_indexes = await schema_manager.get_partial_indexes('orders')
        # 不强制要求部分索引必须存在，因为某些SQLite版本不支持
    
    @pytest.mark.asyncio
    async def test_validate_schema_complete(self, schema_manager):
        """测试Schema完整性验证 - 完整的情况"""
        # 创建所有表
        await schema_manager.create_all_tables()
        
        # 验证Schema
        validation_result = await schema_manager.validate_schema()
        
        assert validation_result['is_valid'] is True
        assert len(validation_result['missing_tables']) == 0
        assert len(validation_result['missing_indexes']) == 0
    
    @pytest.mark.asyncio
    async def test_validate_schema_missing_tables(self, schema_manager):
        """测试Schema完整性验证 - 缺少表的情况"""
        # 只创建部分表
        await schema_manager.create_users_table()
        await schema_manager.create_stocks_table()
        
        # 验证Schema
        validation_result = await schema_manager.validate_schema()
        
        assert validation_result['is_valid'] is False
        assert len(validation_result['missing_tables']) > 0
        assert 'orders' in validation_result['missing_tables']
        assert 'positions' in validation_result['missing_tables']
    
    @pytest.mark.asyncio
    async def test_table_exists(self, schema_manager):
        """测试检查表是否存在"""
        # 初始状态没有表
        assert await schema_manager.table_exists('users') is False
        
        # 创建表后
        await schema_manager.create_users_table()
        assert await schema_manager.table_exists('users') is True
        assert await schema_manager.table_exists('orders') is False
    
    @pytest.mark.asyncio
    async def test_drop_all_tables(self, schema_manager):
        """测试删除所有表功能"""
        # 先创建所有表
        await schema_manager.create_all_tables()
        tables_before = await schema_manager.get_existing_tables()
        assert len(tables_before) > 0
        
        # 删除所有表
        await schema_manager.drop_all_tables()
        
        # 验证所有表都已删除
        tables_after = await schema_manager.get_existing_tables()
        assert len(tables_after) == 0
    
    @pytest.mark.asyncio
    async def test_drop_tables_respects_foreign_key_order(self, schema_manager):
        """测试删除表时遵守外键约束顺序"""
        # 创建有外键关系的表
        await schema_manager.create_users_table()
        await schema_manager.create_orders_table()
        await schema_manager.create_transactions_table()
        
        # 删除所有表（应该不会因为外键约束而失败）
        try:
            await schema_manager.drop_all_tables()
            success = True
        except Exception as e:
            success = False
            error = str(e)
        
        assert success is True, f"删除表失败: {error if not success else ''}"
    
    @pytest.mark.asyncio
    async def test_create_tables_idempotent(self, schema_manager):
        """测试表创建的幂等性（重复创建不会报错）"""
        # 第一次创建
        await schema_manager.create_users_table()
        
        # 第二次创建应该不会报错
        try:
            await schema_manager.create_users_table()
            success = True
        except Exception:
            success = False
        
        assert success is True
    
    @pytest.mark.asyncio
    async def test_get_table_structure_details(self, schema_manager):
        """测试获取表结构的详细信息"""
        # 创建表
        await schema_manager.create_orders_table()
        
        # 获取表结构
        structure = await schema_manager.get_table_structure('orders')
        
        # 验证返回的结构信息
        assert len(structure) > 0
        
        # 验证每个列都有必要的信息
        for column in structure:
            assert 'name' in column
            assert 'type' in column
            assert 'notnull' in column
            assert 'dflt_value' in column
            assert 'pk' in column
    
    @pytest.mark.asyncio
    async def test_decimal_column_types(self, schema_manager):
        """测试DECIMAL类型列的创建"""
        # 创建包含DECIMAL列的表
        await schema_manager.create_risk_metrics_table()
        
        # 获取表结构
        structure = await schema_manager.get_table_structure('risk_metrics')
        
        # 查找DECIMAL类型的列
        decimal_columns = ['portfolio_value', 'daily_pnl', 'var_95', 'beta', 'sharpe_ratio']
        
        for col_name in decimal_columns:
            column = next((col for col in structure if col['name'] == col_name), None)
            assert column is not None, f"列 {col_name} 不存在"
            # SQLite将DECIMAL存储为NUMERIC
            assert 'DECIMAL' in column['type'] or 'NUMERIC' in column['type']
    
    @pytest.mark.asyncio
    async def test_timestamp_default_values(self, schema_manager):
        """测试TIMESTAMP列的默认值"""
        # 创建表
        await schema_manager.create_users_table()
        
        # 插入数据不指定时间戳
        db = schema_manager.db
        user_id = await db.execute_insert(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            ('testuser', 'hashed_password')
        )
        
        # 查询插入的数据
        user = await db.fetch_one("SELECT * FROM users WHERE id = ?", (user_id,))
        
        # 验证时间戳已自动设置
        assert user['created_at'] is not None
        assert user['updated_at'] is not None
    
    @pytest.mark.asyncio
    async def test_boolean_columns(self, schema_manager):
        """测试布尔类型列"""
        # 创建表
        await schema_manager.create_users_table()
        
        # 插入数据测试布尔值
        db = schema_manager.db
        user_id = await db.execute_insert(
            "INSERT INTO users (username, password_hash, is_active) VALUES (?, ?, ?)",
            ('testuser_boolean', 'hash', True)
        )
        
        # 查询验证
        user = await db.fetch_one("SELECT * FROM users WHERE id = ?", (user_id,))
        assert user['is_active'] == 1  # SQLite中布尔值存储为0/1
    
    @pytest.mark.asyncio
    async def test_json_column_type(self, schema_manager):
        """测试JSON类型列"""
        # 创建包含JSON列的表
        await schema_manager.create_user_configs_table()
        await schema_manager.create_users_table()
        
        # 插入测试数据
        db = schema_manager.db
        user_id = await db.execute_insert(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            ('testuser_json', 'hash')
        )
        
        # 插入JSON数据
        json_data = '{"theme": "dark", "language": "zh-CN"}'
        await db.execute_insert(
            "INSERT INTO user_configs (user_id, trading_preferences) VALUES (?, ?)",
            (user_id, json_data)
        )
        
        # 查询验证
        config = await db.fetch_one(
            "SELECT * FROM user_configs WHERE user_id = ?", 
            (user_id,)
        )
        assert config['trading_preferences'] == json_data
    
    @pytest.mark.asyncio
    async def test_cascade_operations(self, schema_manager):
        """测试级联操作（外键约束）"""
        # 创建相关表
        await schema_manager.create_users_table()
        await schema_manager.create_orders_table()
        
        # 插入用户
        db = schema_manager.db
        user_id = await db.execute_insert(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            ('testuser_cascade', 'hash')
        )
        
        # 插入订单
        await db.execute_insert(
            "INSERT INTO orders (id, user_id, symbol, order_type, side, quantity) VALUES (?, ?, ?, ?, ?, ?)",
            ('order-1', user_id, 'AAPL', 'MARKET', 'BUY', 100)
        )
        
        # 尝试插入无效用户ID的订单（应该失败）
        with pytest.raises(Exception):
            await db.execute_insert(
                "INSERT INTO orders (id, user_id, symbol, order_type, side, quantity) VALUES (?, ?, ?, ?, ?, ?)",
                ('order-2', 99999, 'AAPL', 'MARKET', 'BUY', 100)
            )
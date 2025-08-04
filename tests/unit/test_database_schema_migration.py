"""
数据库Schema创建和迁移系统TDD测试

按照TDD原则，先编写完整的测试确保测试全部失败，然后实现功能代码
测试数据库表结构创建、索引优化、分片策略和迁移管理
"""

import pytest
import pytest_asyncio
import sqlite3
import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import List, Dict, Optional
from unittest.mock import AsyncMock, patch, MagicMock

# 待实现的模块
from myQuant.infrastructure.database.schema_manager import SchemaManager
from myQuant.infrastructure.database.migration_manager import MigrationManager
from myQuant.infrastructure.database.database_manager import DatabaseManager


class TestSchemaManager:
    """数据库Schema管理器测试"""

    @pytest.fixture
    def test_db_url(self):
        """测试数据库URL"""
        return "sqlite:///:memory:"

    @pytest_asyncio.fixture
    async def database_manager(self, test_db_url):
        """测试用数据库管理器"""
        manager = DatabaseManager(test_db_url)
        await manager.initialize()
        yield manager
        await manager.close()

    @pytest.fixture
    def schema_manager(self, database_manager):
        """Schema管理器实例"""
        return SchemaManager(database_manager)

    @pytest.mark.asyncio
    async def test_create_all_tables(self, schema_manager):
        """测试创建所有表结构"""
        # Act
        result = await schema_manager.create_all_tables()
        
        # Assert
        assert result is True
        
        # 验证所有必需的表都已创建
        expected_tables = [
            'users', 'user_configs', 'stocks', 'kline_daily', 'real_time_quotes',
            'orders', 'positions', 'transactions', 'strategies', 'alerts', 'risk_metrics'
        ]
        
        existing_tables = await schema_manager.get_existing_tables()
        for table in expected_tables:
            assert table in existing_tables

    @pytest.mark.asyncio
    async def test_create_users_table_with_constraints(self, schema_manager):
        """测试创建用户表及其约束"""
        # Act
        await schema_manager.create_users_table()
        
        # Assert - 验证表结构
        table_info = await schema_manager.get_table_structure('users')
        
        # 验证字段存在
        field_names = [field['name'] for field in table_info]
        assert 'id' in field_names
        assert 'username' in field_names
        assert 'email' in field_names
        assert 'password_hash' in field_names
        assert 'created_at' in field_names
        assert 'updated_at' in field_names
        assert 'is_active' in field_names
        
        # 验证约束
        constraints = await schema_manager.get_table_constraints('users')
        assert any('username' in str(c) and 'UNIQUE' in str(c) for c in constraints)
        assert any('email' in str(c) and 'UNIQUE' in str(c) for c in constraints)

    @pytest.mark.asyncio
    async def test_create_stocks_table_with_proper_types(self, schema_manager):
        """测试创建股票表及其数据类型"""
        # Act
        await schema_manager.create_stocks_table()
        
        # Assert
        table_info = await schema_manager.get_table_structure('stocks')
        
        # 验证主键
        pk_field = next((f for f in table_info if f['pk'] == 1), None)
        assert pk_field is not None
        assert pk_field['name'] == 'symbol'
        
        # 验证字段类型
        field_types = {field['name']: field['type'] for field in table_info}
        assert 'VARCHAR' in field_types['symbol']
        assert 'VARCHAR' in field_types['name']
        assert 'DATE' in field_types.get('listing_date', '')

    @pytest.mark.asyncio
    async def test_create_kline_table_with_partitioning_support(self, schema_manager):
        """测试创建K线表支持分区"""
        # Act
        await schema_manager.create_kline_daily_table()
        
        # Assert
        table_info = await schema_manager.get_table_structure('kline_daily')
        
        # 验证必要字段
        field_names = [field['name'] for field in table_info]
        required_fields = ['symbol', 'trade_date', 'open_price', 'high_price', 
                          'low_price', 'close_price', 'volume', 'turnover']
        for field in required_fields:
            assert field in field_names
        
        # 验证数值类型精度
        decimal_fields = ['open_price', 'high_price', 'low_price', 'close_price', 'turnover']
        field_types = {field['name']: field['type'] for field in table_info}
        for field in decimal_fields:
            assert 'DECIMAL' in field_types[field]

    @pytest.mark.asyncio
    async def test_create_orders_table_with_foreign_keys(self, schema_manager):
        """测试创建订单表及其外键约束"""
        # Arrange - 先创建依赖表
        await schema_manager.create_users_table()
        await schema_manager.create_stocks_table()
        
        # Act
        await schema_manager.create_orders_table()
        
        # Assert
        foreign_keys = await schema_manager.get_foreign_keys('orders')
        
        # 验证外键约束
        assert any('user_id' in str(fk) and 'users' in str(fk) for fk in foreign_keys)
        
        # 验证订单状态枚举
        table_info = await schema_manager.get_table_structure('orders')
        status_field = next((f for f in table_info if f['name'] == 'status'), None)
        assert status_field is not None

    @pytest.mark.asyncio
    async def test_create_positions_table_with_unique_constraints(self, schema_manager):
        """测试创建持仓表及其唯一约束"""
        # Arrange
        await schema_manager.create_users_table()
        await schema_manager.create_stocks_table()
        
        # Act
        await schema_manager.create_positions_table()
        
        # Assert
        constraints = await schema_manager.get_table_constraints('positions')
        
        # 验证唯一约束 (user_id, symbol)
        unique_constraint_exists = any(
            'user_id' in str(c) and 'symbol' in str(c) and 'UNIQUE' in str(c) 
            for c in constraints
        )
        assert unique_constraint_exists

    @pytest.mark.asyncio
    async def test_create_indexes_for_performance(self, schema_manager):
        """测试创建性能优化索引"""
        # Arrange
        await schema_manager.create_all_tables()
        
        # Act
        await schema_manager.create_performance_indexes()
        
        # Assert
        indexes = await schema_manager.get_table_indexes('kline_daily')
        
        # 验证重要索引存在
        index_names = [idx['name'] for idx in indexes]
        assert any('symbol' in name and 'date' in name for name in index_names)
        
        # 验证订单表索引
        order_indexes = await schema_manager.get_table_indexes('orders')
        order_index_names = [idx['name'] for idx in order_indexes]
        assert any('user' in name and 'status' in name for name in order_index_names)

    @pytest.mark.asyncio
    async def test_create_partial_indexes_for_active_data(self, schema_manager):
        """测试为活跃数据创建部分索引"""
        # Arrange
        await schema_manager.create_all_tables()
        
        # Act
        await schema_manager.create_partial_indexes()
        
        # Assert
        # 验证活跃订单的部分索引
        partial_indexes = await schema_manager.get_partial_indexes('orders')
        assert any('PENDING' in str(idx) for idx in partial_indexes)
        
        # 验证活跃提醒的部分索引
        alert_partial_indexes = await schema_manager.get_partial_indexes('alerts')
        assert any('is_active' in str(idx) for idx in alert_partial_indexes)

    @pytest.mark.asyncio
    async def test_schema_validation(self, schema_manager):
        """测试Schema验证"""
        # Arrange
        await schema_manager.create_all_tables()
        
        # Act
        validation_result = await schema_manager.validate_schema()
        
        # Assert
        assert validation_result['is_valid'] is True
        assert validation_result['missing_tables'] == []
        assert validation_result['missing_indexes'] == []
        assert validation_result['constraint_violations'] == []

    @pytest.mark.asyncio
    async def test_schema_validation_detects_missing_tables(self, schema_manager):
        """测试Schema验证能检测缺失的表"""
        # Arrange - 删除所有表以模拟缺失状态
        await schema_manager.drop_all_tables()
        
        # Act - 不创建表就验证
        validation_result = await schema_manager.validate_schema()
        
        # Assert
        assert validation_result['is_valid'] is False
        assert len(validation_result['missing_tables']) > 0
        assert 'users' in validation_result['missing_tables']

    @pytest.mark.asyncio
    async def test_drop_all_tables(self, schema_manager):
        """测试删除所有表"""
        # Arrange
        await schema_manager.create_all_tables()
        tables_before = await schema_manager.get_existing_tables()
        assert len(tables_before) > 0
        
        # Act
        await schema_manager.drop_all_tables()
        
        # Assert
        tables_after = await schema_manager.get_existing_tables()
        # 除了系统表，用户表应该都被删除
        user_tables = [t for t in tables_after if not t.startswith('sqlite_')]
        assert len(user_tables) == 0


class TestMigrationManager:
    """数据库迁移管理器测试"""

    @pytest.fixture
    def test_db_url(self):
        return "sqlite:///:memory:"

    @pytest_asyncio.fixture
    async def database_manager(self, test_db_url):
        manager = DatabaseManager(test_db_url)
        await manager.initialize()
        yield manager
        await manager.close()

    @pytest.fixture
    def migration_manager(self, database_manager):
        return MigrationManager(database_manager)

    @pytest.mark.asyncio
    async def test_initialize_migration_table(self, migration_manager):
        """测试初始化迁移表"""
        # Act
        await migration_manager.initialize()
        
        # Assert
        exists = await migration_manager.migration_table_exists()
        assert exists is True
        
        # 验证迁移表结构
        table_info = await migration_manager.get_migration_table_structure()
        field_names = [field['name'] for field in table_info]
        assert 'version' in field_names
        assert 'name' in field_names
        assert 'applied_at' in field_names

    @pytest.mark.asyncio
    async def test_get_current_version(self, migration_manager):
        """测试获取当前迁移版本"""
        # Arrange
        await migration_manager.initialize()
        
        # Act
        version = await migration_manager.get_current_version()
        
        # Assert
        assert version == 0  # 初始版本应该是0

    @pytest.mark.asyncio
    async def test_apply_migration(self, migration_manager):
        """测试应用迁移"""
        # Arrange
        await migration_manager.initialize()
        
        migration = {
            'version': 1,
            'name': 'create_initial_tables',
            'up_sql': '''
                CREATE TABLE test_table (
                    id INTEGER PRIMARY KEY,
                    name VARCHAR(100) NOT NULL
                );
            ''',
            'down_sql': 'DROP TABLE IF EXISTS test_table;'
        }
        
        # Act
        result = await migration_manager.apply_migration(migration)
        
        # Assert
        assert result is True
        
        # 验证版本已更新
        current_version = await migration_manager.get_current_version()
        assert current_version == 1
        
        # 验证表已创建
        exists = await migration_manager.table_exists('test_table')
        assert exists is True

    @pytest.mark.asyncio
    async def test_rollback_migration(self, migration_manager):
        """测试回滚迁移"""
        # Arrange
        await migration_manager.initialize()
        
        migration = {
            'version': 1,
            'name': 'create_test_table',
            'up_sql': 'CREATE TABLE test_table (id INTEGER PRIMARY KEY);',
            'down_sql': 'DROP TABLE IF EXISTS test_table;'
        }
        
        await migration_manager.apply_migration(migration)
        
        # Act
        result = await migration_manager.rollback_migration(1, migration['down_sql'])
        
        # Assert
        assert result is True
        
        # 验证版本已回滚
        current_version = await migration_manager.get_current_version()
        assert current_version == 0
        
        # 验证表已删除
        exists = await migration_manager.table_exists('test_table')
        assert exists is False

    @pytest.mark.asyncio
    async def test_get_pending_migrations(self, migration_manager):
        """测试获取待应用的迁移"""
        # Arrange
        await migration_manager.initialize()
        
        available_migrations = [
            {'version': 1, 'name': 'create_users'},
            {'version': 2, 'name': 'create_orders'},
            {'version': 3, 'name': 'add_indexes'}
        ]
        
        # Act
        pending = await migration_manager.get_pending_migrations(available_migrations)
        
        # Assert
        assert len(pending) == 3
        assert pending[0]['version'] == 1

    @pytest.mark.asyncio
    async def test_apply_all_pending_migrations(self, migration_manager):
        """测试应用所有待处理迁移"""
        # Arrange
        await migration_manager.initialize()
        
        migrations = [
            {
                'version': 1,
                'name': 'create_test_users',
                'up_sql': 'CREATE TABLE test_users (id INTEGER PRIMARY KEY);',
                'down_sql': 'DROP TABLE test_users;'
            },
            {
                'version': 2,
                'name': 'create_test_orders',
                'up_sql': 'CREATE TABLE test_orders (id INTEGER PRIMARY KEY);',
                'down_sql': 'DROP TABLE test_orders;'
            }
        ]
        
        # Act
        results = await migration_manager.apply_all_migrations(migrations)
        
        # Assert
        assert all(results)
        
        # 验证最终版本
        current_version = await migration_manager.get_current_version()
        assert current_version == 2
        
        # 验证表都已创建
        assert await migration_manager.table_exists('users')
        assert await migration_manager.table_exists('orders')

    @pytest.mark.asyncio
    async def test_migration_transaction_rollback_on_error(self, migration_manager):
        """测试迁移出错时的事务回滚"""
        # Arrange
        await migration_manager.initialize()
        
        invalid_migration = {
            'version': 1,
            'name': 'invalid_migration',
            'up_sql': 'CREATE TABLE invalid_syntax INVALID SQL;',  # 故意的语法错误
            'down_sql': 'DROP TABLE invalid_table;'
        }
        
        # Act & Assert
        with pytest.raises(Exception):
            await migration_manager.apply_migration(invalid_migration)
        
        # 验证版本没有改变
        current_version = await migration_manager.get_current_version()
        assert current_version == 0

    @pytest.mark.asyncio
    async def test_get_migration_history(self, migration_manager):
        """测试获取迁移历史"""
        # Arrange
        await migration_manager.initialize()
        
        migrations = [
            {
                'version': 1,
                'name': 'create_test_users_hist',
                'up_sql': 'CREATE TABLE test_users_hist (id INTEGER PRIMARY KEY);',
                'down_sql': 'DROP TABLE test_users_hist;'
            },
            {
                'version': 2,
                'name': 'create_test_orders_hist',
                'up_sql': 'CREATE TABLE test_orders_hist (id INTEGER PRIMARY KEY);',
                'down_sql': 'DROP TABLE test_orders_hist;'
            }
        ]
        
        for migration in migrations:
            await migration_manager.apply_migration(migration)
        
        # Act
        history = await migration_manager.get_migration_history()
        
        # Assert
        assert len(history) == 2
        assert history[0]['version'] == 1
        assert history[1]['version'] == 2
        assert all('applied_at' in record for record in history)

    @pytest.mark.asyncio
    async def test_check_migration_integrity(self, migration_manager):
        """测试检查迁移完整性"""
        # Arrange
        await migration_manager.initialize()
        
        # 应用一些迁移
        migration = {
            'version': 1,
            'name': 'create_test_users_integ',
            'up_sql': 'CREATE TABLE test_users_integ (id INTEGER PRIMARY KEY);',
            'down_sql': 'DROP TABLE test_users_integ;'
        }
        await migration_manager.apply_migration(migration)
        
        # Act
        integrity_check = await migration_manager.check_integrity()
        
        # Assert
        assert integrity_check['is_valid'] is True
        assert integrity_check['current_version'] == 1
        assert len(integrity_check['missing_migrations']) == 0

    @pytest.mark.asyncio
    async def test_backup_before_migration(self, migration_manager):
        """测试迁移前备份"""
        # Arrange
        await migration_manager.initialize()
        
        # Act
        backup_path = await migration_manager.create_backup()
        
        # Assert
        assert backup_path is not None
        assert backup_path.endswith('.sql') or backup_path.endswith('.db')


if __name__ == "__main__":
    # 运行测试确保全部失败（TDD第一步）
    pytest.main([__file__, "-v", "--tb=short"])
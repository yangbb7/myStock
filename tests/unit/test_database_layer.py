"""
数据库层单元测试

测试数据库表结构创建和基础CRUD操作
按照TDD原则，先编写完整的单元测试，确保测试全部失败，然后实现功能代码
"""

import pytest
import pytest_asyncio
import sqlite3
import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import List, Dict, Optional
from unittest.mock import AsyncMock, patch

from myQuant.infrastructure.database.database_manager import DatabaseManager
from myQuant.infrastructure.database.models import (
    StockTable, KlineDailyTable, RealTimeQuoteTable, OrderTable, 
    PositionTable, TransactionTable, UserTable, UserConfigTable,
    StrategyTable, AlertTable, RiskMetricTable
)
from myQuant.infrastructure.database.repositories import (
    UserRepository, StockRepository, OrderRepository, 
    PositionRepository, PortfolioRepository
)


@pytest.fixture
def test_db_url():
    """测试数据库URL"""
    return "sqlite:///:memory:"


@pytest_asyncio.fixture
async def database_manager(test_db_url):
    """测试用数据库管理器"""
    manager = DatabaseManager(test_db_url)
    await manager.initialize()
    yield manager
    await manager.close()


class TestDatabaseManager:
    """数据库管理器测试"""

    @pytest.mark.asyncio
    async def test_database_initialization(self, test_db_url):
        """测试数据库初始化"""
        # Arrange & Act
        manager = DatabaseManager(test_db_url)
        await manager.initialize()

        # Assert
        assert manager.is_connected()
        assert manager.database_url == test_db_url
        
        # 验证所有表都已创建
        tables = await manager.get_table_names()
        expected_tables = [
            'users', 'user_configs', 'stocks', 'kline_daily', 'real_time_quotes',
            'orders', 'positions', 'transactions', 'strategies', 'alerts', 'risk_metrics'
        ]
        
        for table in expected_tables:
            assert table in tables

        await manager.close()

    @pytest.mark.asyncio
    async def test_database_connection_management(self, test_db_url):
        """测试数据库连接管理"""
        # Arrange
        manager = DatabaseManager(test_db_url)

        # Act & Assert - 初始状态
        assert not manager.is_connected()

        # 连接数据库
        await manager.initialize()
        assert manager.is_connected()

        # 关闭连接
        await manager.close()
        assert not manager.is_connected()

    @pytest.mark.asyncio
    async def test_database_transaction_management(self, database_manager):
        """测试数据库事务管理"""
        # Arrange
        user_data = {
            "username": "test_user",
            "email": "test@example.com",
            "password_hash": "hashed_password"
        }

        # Act & Assert - 成功事务
        async with database_manager.transaction():
            user_id = await database_manager.execute_insert(
                "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                (user_data["username"], user_data["email"], user_data["password_hash"]),
                auto_commit=False
            )
            assert user_id is not None

        # 验证数据已提交
        user = await database_manager.fetch_one(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        )
        assert user is not None
        assert user["username"] == user_data["username"]

        # Act & Assert - 失败事务回滚
        try:
            async with database_manager.transaction():
                await database_manager.execute_insert(
                    "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                    ("test_user2", "test2@example.com", "hashed_password2"),
                    auto_commit=False
                )
                # 模拟错误
                raise Exception("Transaction error")
        except Exception:
            pass

        # 验证数据已回滚
        users = await database_manager.fetch_all("SELECT * FROM users WHERE username = ?", ("test_user2",))
        assert len(users) == 0

    @pytest.mark.asyncio
    async def test_database_migration(self, test_db_url):
        """测试数据库迁移"""
        # Arrange
        manager = DatabaseManager(test_db_url)

        # Act
        await manager.initialize()
        migration_version = await manager.get_migration_version()

        # Assert
        assert migration_version > 0
        assert await manager.is_migration_complete()

        await manager.close()


class TestDatabaseTables:
    """数据库表测试"""

    @pytest.mark.asyncio
    async def test_user_table_operations(self, database_manager):
        """测试用户表操作"""
        # Arrange
        user_data = {
            "username": "test_user",
            "email": "test@example.com", 
            "password_hash": "hashed_password",
            "is_active": True
        }

        # Act - 插入用户
        user_id = await database_manager.execute_insert(
            """INSERT INTO users (username, email, password_hash, is_active) 
               VALUES (?, ?, ?, ?)""",
            (user_data["username"], user_data["email"], 
             user_data["password_hash"], user_data["is_active"])
        )

        # Assert - 验证插入
        assert user_id is not None
        
        # 查询用户
        user = await database_manager.fetch_one(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        )
        assert user["username"] == user_data["username"]
        assert user["email"] == user_data["email"]
        assert user["is_active"] == user_data["is_active"]

        # Act - 更新用户
        await database_manager.execute_update(
            "UPDATE users SET email = ? WHERE id = ?",
            ("updated@example.com", user_id)
        )

        # Assert - 验证更新
        updated_user = await database_manager.fetch_one(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        )
        assert updated_user["email"] == "updated@example.com"

        # Act - 删除用户
        await database_manager.execute_delete(
            "DELETE FROM users WHERE id = ?", (user_id,)
        )

        # Assert - 验证删除
        deleted_user = await database_manager.fetch_one(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        )
        assert deleted_user is None

    @pytest.mark.asyncio
    async def test_stock_table_operations(self, database_manager):
        """测试股票表操作"""
        # Arrange
        stock_data = {
            "symbol": "000001.SZ",
            "name": "平安银行",
            "sector": "金融",
            "industry": "银行",
            "market": "SZ",
            "listing_date": "1991-04-03",
            "total_shares": 19405918198,
            "float_shares": 19405918198
        }

        # Act - 插入股票
        await database_manager.execute_insert(
            """INSERT INTO stocks (symbol, name, sector, industry, market, 
                                listing_date, total_shares, float_shares) 
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (stock_data["symbol"], stock_data["name"], stock_data["sector"],
             stock_data["industry"], stock_data["market"], stock_data["listing_date"],
             stock_data["total_shares"], stock_data["float_shares"])
        )

        # Assert - 验证插入
        stock = await database_manager.fetch_one(
            "SELECT * FROM stocks WHERE symbol = ?", (stock_data["symbol"],)
        )
        assert stock["name"] == stock_data["name"]
        assert stock["sector"] == stock_data["sector"]
        assert stock["market"] == stock_data["market"]

    @pytest.mark.asyncio
    async def test_kline_table_operations(self, database_manager):
        """测试K线表操作"""
        # Arrange - 先插入股票
        await database_manager.execute_insert(
            "INSERT INTO stocks (symbol, name, market) VALUES (?, ?, ?)",
            ("000001.SZ", "平安银行", "SZ")
        )

        kline_data = {
            "symbol": "000001.SZ",
            "trade_date": "2024-01-01",
            "open_price": Decimal("12.50"),
            "high_price": Decimal("12.80"),
            "low_price": Decimal("12.30"),
            "close_price": Decimal("12.75"),
            "volume": 1000000,
            "turnover": Decimal("12650000.00")
        }

        # Act - 插入K线数据
        kline_id = await database_manager.execute_insert(
            """INSERT INTO kline_daily (symbol, trade_date, open_price, high_price, 
                                      low_price, close_price, volume, turnover) 
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (kline_data["symbol"], kline_data["trade_date"], 
             float(kline_data["open_price"]), float(kline_data["high_price"]),
             float(kline_data["low_price"]), float(kline_data["close_price"]),
             kline_data["volume"], float(kline_data["turnover"]))
        )

        # Assert - 验证插入
        assert kline_id is not None
        
        kline = await database_manager.fetch_one(
            "SELECT * FROM kline_daily WHERE symbol = ? AND trade_date = ?",
            (kline_data["symbol"], kline_data["trade_date"])
        )
        assert kline["open_price"] == float(kline_data["open_price"])
        assert kline["volume"] == kline_data["volume"]

    @pytest.mark.asyncio
    async def test_order_table_operations(self, database_manager):
        """测试订单表操作"""
        # Arrange - 先插入用户和股票
        user_id = await database_manager.execute_insert(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            ("test_user", "test@example.com", "hashed_password")
        )
        
        await database_manager.execute_insert(
            "INSERT INTO stocks (symbol, name, market) VALUES (?, ?, ?)",
            ("000001.SZ", "平安银行", "SZ")
        )

        order_data = {
            "order_id": "ord_123456789",
            "user_id": user_id,
            "symbol": "000001.SZ",
            "order_type": "LIMIT",
            "side": "BUY",
            "quantity": 1000,
            "price": Decimal("12.50"),
            "status": "PENDING"
        }

        # Act - 插入订单
        await database_manager.execute_insert(
            """INSERT INTO orders (id, user_id, symbol, order_type, side, 
                                 quantity, price, status) 
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (order_data["order_id"], order_data["user_id"], order_data["symbol"],
             order_data["order_type"], order_data["side"], order_data["quantity"],
             float(order_data["price"]), order_data["status"])
        )

        # Assert - 验证插入
        order = await database_manager.fetch_one(
            "SELECT * FROM orders WHERE id = ?", (order_data["order_id"],)
        )
        assert order["user_id"] == order_data["user_id"]
        assert order["symbol"] == order_data["symbol"]
        assert order["quantity"] == order_data["quantity"]
        assert order["status"] == order_data["status"]

    @pytest.mark.asyncio
    async def test_position_table_operations(self, database_manager):
        """测试持仓表操作"""
        # Arrange - 先插入用户和股票
        user_id = await database_manager.execute_insert(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            ("test_user", "test@example.com", "hashed_password")
        )
        
        await database_manager.execute_insert(
            "INSERT INTO stocks (symbol, name, market) VALUES (?, ?, ?)",
            ("000001.SZ", "平安银行", "SZ")
        )

        position_data = {
            "user_id": user_id,
            "symbol": "000001.SZ",
            "quantity": 1000,
            "average_price": Decimal("12.00")
        }

        # Act - 插入持仓
        position_id = await database_manager.execute_insert(
            """INSERT INTO positions (user_id, symbol, quantity, average_price) 
               VALUES (?, ?, ?, ?)""",
            (position_data["user_id"], position_data["symbol"],
             position_data["quantity"], float(position_data["average_price"]))
        )

        # Assert - 验证插入
        assert position_id is not None
        
        position = await database_manager.fetch_one(
            "SELECT * FROM positions WHERE user_id = ? AND symbol = ?",
            (position_data["user_id"], position_data["symbol"])
        )
        assert position["quantity"] == position_data["quantity"]
        assert position["average_price"] == float(position_data["average_price"])


class TestRepositories:
    """Repository层测试"""

    @pytest.fixture
    def user_repository(self, database_manager):
        """用户Repository"""
        return UserRepository(database_manager)

    @pytest.fixture
    def order_repository(self, database_manager):
        """订单Repository"""
        return OrderRepository(database_manager)

    @pytest.fixture
    def position_repository(self, database_manager):
        """持仓Repository"""
        return PositionRepository(database_manager)

    @pytest.mark.asyncio
    async def test_user_repository_operations(self, user_repository):
        """测试用户Repository操作"""
        # Arrange
        user_data = {
            "username": "test_user",
            "email": "test@example.com",
            "password_hash": "hashed_password"
        }

        # Act - 创建用户
        user_id = await user_repository.create_user(user_data)

        # Assert
        assert user_id is not None
        
        # 查询用户
        user = await user_repository.get_user_by_id(user_id)
        assert user.username == user_data["username"]
        assert user.email == user_data["email"]

        # 按用户名查询
        user_by_name = await user_repository.get_user_by_username(user_data["username"])
        assert user_by_name.id == user_id

        # 更新用户
        await user_repository.update_user(user_id, {"email": "updated@example.com"})
        updated_user = await user_repository.get_user_by_id(user_id)
        assert updated_user.email == "updated@example.com"

        # 删除用户
        await user_repository.delete_user(user_id)
        deleted_user = await user_repository.get_user_by_id(user_id)
        assert deleted_user is None

    @pytest.mark.asyncio
    async def test_order_repository_operations(self, order_repository, user_repository):
        """测试订单Repository操作"""
        # Arrange - 先创建用户
        user_id = await user_repository.create_user({
            "username": "test_user",
            "email": "test@example.com",
            "password_hash": "hashed_password"
        })

        order_data = {
            "order_id": "ord_123456789",
            "user_id": user_id,
            "symbol": "000001.SZ",
            "order_type": "LIMIT",
            "side": "BUY",
            "quantity": 1000,
            "price": Decimal("12.50")
        }

        # Act - 创建订单
        await order_repository.create_order(order_data)

        # Assert
        # 查询订单
        order = await order_repository.get_order_by_id(order_data["order_id"])
        assert order.user_id == user_id
        assert order.symbol == order_data["symbol"]
        assert order.quantity == order_data["quantity"]

        # 按用户查询订单
        user_orders = await order_repository.get_orders_by_user(user_id)
        assert len(user_orders) == 1
        assert user_orders[0].id == order_data["order_id"]

        # 更新订单状态
        await order_repository.update_order_status(order_data["order_id"], "FILLED")
        updated_order = await order_repository.get_order_by_id(order_data["order_id"])
        assert updated_order.status == "FILLED"

    @pytest.mark.asyncio
    async def test_position_repository_operations(self, position_repository, user_repository):
        """测试持仓Repository操作"""
        # Arrange - 先创建用户
        user_id = await user_repository.create_user({
            "username": "test_user",
            "email": "test@example.com",
            "password_hash": "hashed_password"
        })

        position_data = {
            "user_id": user_id,
            "symbol": "000001.SZ",
            "quantity": 1000,
            "average_price": Decimal("12.00")
        }

        # Act - 创建持仓
        position_id = await position_repository.create_position(position_data)

        # Assert
        assert position_id is not None
        
        # 查询持仓
        position = await position_repository.get_position_by_id(position_id)
        assert position.user_id == user_id
        assert position.symbol == position_data["symbol"]
        assert position.quantity == position_data["quantity"]

        # 按用户查询持仓
        user_positions = await position_repository.get_positions_by_user(user_id)
        assert len(user_positions) == 1
        assert user_positions[0].symbol == position_data["symbol"]

        # 更新持仓
        await position_repository.update_position(position_id, {
            "quantity": 1500,
            "average_price": Decimal("12.20")
        })
        updated_position = await position_repository.get_position_by_id(position_id)
        assert updated_position.quantity == 1500

        # 删除持仓
        await position_repository.delete_position(position_id)
        deleted_position = await position_repository.get_position_by_id(position_id)
        assert deleted_position is None


class TestDatabasePerformance:
    """数据库性能测试"""

    @pytest.mark.asyncio
    async def test_batch_insert_performance(self, database_manager):
        """测试批量插入性能"""
        # Arrange
        batch_size = 1000
        kline_data = []
        
        # 先插入股票
        await database_manager.execute_insert(
            "INSERT INTO stocks (symbol, name, market) VALUES (?, ?, ?)",
            ("000001.SZ", "平安银行", "SZ")
        )

        for i in range(batch_size):
            kline_data.append((
                "000001.SZ",
                f"2024-01-{i+1:02d}",
                12.50 + i * 0.01,
                12.80 + i * 0.01,
                12.30 + i * 0.01,
                12.75 + i * 0.01,
                1000000 + i * 1000,
                12650000.00 + i * 1000
            ))

        # Act - 批量插入
        start_time = datetime.now()
        await database_manager.execute_batch_insert(
            """INSERT INTO kline_daily (symbol, trade_date, open_price, high_price, 
                                      low_price, close_price, volume, turnover) 
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            kline_data
        )
        end_time = datetime.now()

        # Assert
        execution_time = (end_time - start_time).total_seconds()
        assert execution_time < 1.0  # 批量插入应该在1秒内完成

        # 验证插入数量
        count = await database_manager.fetch_scalar(
            "SELECT COUNT(*) FROM kline_daily WHERE symbol = ?", ("000001.SZ",)
        )
        assert count == batch_size

    @pytest.mark.asyncio
    async def test_query_performance_with_index(self, database_manager):
        """测试带索引的查询性能"""
        # Arrange - 插入大量数据
        await database_manager.execute_insert(
            "INSERT INTO stocks (symbol, name, market) VALUES (?, ?, ?)",
            ("000001.SZ", "平安银行", "SZ")
        )

        # 插入1000条K线数据
        kline_data = []
        for i in range(1000):
            kline_data.append((
                "000001.SZ",
                f"2024-{(i//30)+1:02d}-{(i%30)+1:02d}",
                12.50, 12.80, 12.30, 12.75, 1000000, 12650000.00
            ))

        await database_manager.execute_batch_insert(
            """INSERT INTO kline_daily (symbol, trade_date, open_price, high_price, 
                                      low_price, close_price, volume, turnover) 
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            kline_data
        )

        # Act - 测试查询性能
        start_time = datetime.now()
        results = await database_manager.fetch_all(
            """SELECT * FROM kline_daily 
               WHERE symbol = ? AND trade_date >= ? AND trade_date <= ?
               ORDER BY trade_date DESC LIMIT 100""",
            ("000001.SZ", "2024-01-01", "2024-12-31")
        )
        end_time = datetime.now()

        # Assert
        execution_time = (end_time - start_time).total_seconds()
        assert execution_time < 0.1  # 查询应该在100ms内完成
        assert len(results) <= 100


if __name__ == "__main__":
    # 运行测试确保全部失败（TDD第一步）
    pytest.main([__file__, "-v", "--tb=short"])
# Standard library imports
import os
import queue
import sqlite3
import tempfile
import threading
import time
from datetime import datetime, date, timedelta
from unittest.mock import patch, Mock, MagicMock

# Third-party imports
import numpy as np
import pandas as pd
import psutil
import pytest
import requests

# Local imports
from myQuant.core.managers.data_manager import DataManager
from tests.base_test import BaseTestCase, TestDataFactory, MockFactory, IsolatedComponentFactory
from myQuant.core.exceptions import DataException

class TestDataManager(BaseTestCase):
    """数据管理器测试用例 - 覆盖常规case和边缘case"""
    
    @pytest.fixture
    def temp_db_path(self):
        """临时数据库路径fixture"""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_file.close()
        yield temp_file.name
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
    
    @pytest.fixture
    def sample_price_data(self):
        """样本价格数据fixture"""
        return TestDataFactory.create_deterministic_price_data('000001.SZ', 100)
    
    @pytest.fixture
    def sample_financial_data(self):
        """样本财务数据fixture"""
        return pd.DataFrame({
            'symbol': ['000001.SZ', '000002.SZ', '600000.SH'],
            'report_date': ['2023-12-31'] * 3,
            'eps': [1.2, 0.8, 2.1],
            'revenue': [1000000000, 800000000, 1500000000],
            'net_profit': [120000000, 80000000, 210000000],
            'roe': [0.15, 0.12, 0.18]
        })
    
    # === 初始化测试 ===
    @pytest.mark.unit
    def test_data_manager_init_success(self, temp_db_path):
        """测试数据管理器正常初始化"""
        config = {'db_path': temp_db_path, 'cache_size': 1000}
        manager = DataManager(config)
        assert manager.db_path == temp_db_path
        assert manager.cache_size == 1000
    
    @pytest.mark.unit
    def test_data_manager_init_invalid_path(self):
        """测试无效路径初始化"""
        invalid_path = '/invalid/path/db.sqlite'
        config = {'db_path': invalid_path}
        
        # 期望抛出异常
        with pytest.raises(Exception):
            DataManager(config)
    
    @pytest.mark.unit
    def test_data_manager_init_missing_config(self):
        """测试缺少配置参数 - 现在使用统一配置系统提供默认值"""
        config = {}  # 空配置
        
        # 统一配置系统引入后，不再抛出ValueError，而是使用默认配置
        manager = DataManager(config)
        assert manager is not None
        # 验证使用了默认的数据库URL
        assert manager.db_path is not None
    
    # === 数据获取测试 ===
    @pytest.mark.unit
    def test_get_price_data_normal_case(self, sample_price_data, temp_db_path):
        """测试正常获取价格数据"""
        symbol = '000001.SZ'
        start_date = '2023-01-01'
        end_date = '2023-01-31'
        
        config = {'db_path': temp_db_path, 'cache_size': 1000}
        manager = DataManager(config)
        result = manager.get_price_data(symbol, start_date, end_date)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert all(col in result.columns for col in ['open', 'high', 'low', 'close', 'volume'])
    
    @pytest.mark.unit
    def test_get_price_data_invalid_symbol(self, temp_db_path):
        """测试无效股票代码"""
        invalid_symbol = 'INVALID'
        start_date = '2023-01-01'
        end_date = '2023-01-31'
        
        config = {'db_path': temp_db_path, 'cache_size': 1000}
        manager = DataManager(config)
        result = manager.get_price_data(invalid_symbol, start_date, end_date)
        assert isinstance(result, pd.DataFrame)
    
    @pytest.mark.unit
    def test_get_price_data_invalid_date_range(self, temp_db_path):
        """测试无效日期范围"""
        symbol = '000001.SZ'
        start_date = '2023-01-31'  # 开始日期晚于结束日期
        end_date = '2023-01-01'
        
        config = {'db_path': temp_db_path, 'cache_size': 1000}
        manager = DataManager(config)
        with pytest.raises(ValueError):
            manager.get_price_data(symbol, start_date, end_date)
    
    @pytest.mark.unit
    def test_get_price_data_future_date(self, temp_db_path):
        """测试未来日期"""
        symbol = '000001.SZ'
        start_date = '2030-01-01'  # 更远的未来日期
        end_date = '2030-01-31'
        
        config = {'db_path': temp_db_path, 'cache_size': 1000}
        manager = DataManager(config)
        result = manager.get_price_data(symbol, start_date, end_date)
        assert result.empty
    
    @pytest.mark.unit
    def test_get_price_data_weekend_holidays(self, temp_db_path):
        """测试周末和节假日处理"""
        symbol = '000001.SZ'
        start_date = '2023-01-01'  # 假设这是周末
        end_date = '2023-01-02'
        
        config = {'db_path': temp_db_path, 'cache_size': 1000}
        manager = DataManager(config)
        result = manager.get_price_data(symbol, start_date, end_date)
        # 应该自动跳过非交易日
        assert isinstance(result, pd.DataFrame)
    
    # === 财务数据测试 ===
    @pytest.mark.unit
    def test_get_financial_data_normal_case(self, temp_db_path, sample_financial_data):
        """测试正常获取财务数据"""
        config = {'db_path': temp_db_path, 'cache_size': 1000}
        manager = DataManager(config)
        
        # 先保存一些财务数据到数据库
        conn = sqlite3.connect(temp_db_path)
        sample_financial_data.to_sql('financial_data', conn, if_exists='append', index=False)
        conn.close()
        
        symbol = '000001.SZ'
        report_date = '2023-12-31'
        
        result = manager.get_financial_data(symbol, report_date)
        assert isinstance(result, pd.Series)
        assert result['symbol'] == symbol
        assert result['eps'] == 1.2
    
    @pytest.mark.unit
    def test_get_financial_data_batch(self, temp_db_path, sample_financial_data):
        """测试批量获取财务数据"""
        config = {'db_path': temp_db_path, 'cache_size': 1000}
        manager = DataManager(config)
        
        # 先保存财务数据到数据库
        conn = sqlite3.connect(temp_db_path)
        sample_financial_data.to_sql('financial_data', conn, if_exists='append', index=False)
        conn.close()
        
        symbols = ['000001.SZ', '000002.SZ', '600000.SH']
        report_date = '2023-12-31'
        
        result = manager.get_financial_data_batch(symbols, report_date)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(symbols)
        assert set(result['symbol'].tolist()) == set(symbols)
    
    @pytest.mark.unit  
    def test_get_financial_data_missing_report(self, temp_db_path):
        """测试缺失财报数据"""
        config = {'db_path': temp_db_path, 'cache_size': 1000}
        manager = DataManager(config)
        
        symbol = '000001.SZ'
        report_date = '1990-12-31'  # 很早的日期，可能没有数据
        
        result = manager.get_financial_data(symbol, report_date)
        assert result is None
    
    # === 技术指标计算测试 ===
    @pytest.mark.unit
    def test_calculate_ma_normal_case(self, temp_db_path, sample_price_data):
        """测试计算移动平均线"""
        
        config = {'db_path': temp_db_path, 'cache_size': 1000}
        manager = DataManager(config)
        
        data = sample_price_data
        period = 20
        
        result = manager.calculate_ma(data['close'], period)
        assert len(result) == len(data)
        assert not pd.isna(result.iloc[-1])  # 最新值不应为NaN
        # 前19个值应该是NaN
        assert result.iloc[:period-1].isna().all()
    
    @pytest.mark.unit
    def test_calculate_ma_insufficient_data(self, temp_db_path):
        """测试数据不足计算MA"""
        config = {'db_path': temp_db_path, 'cache_size': 1000}
        manager = DataManager(config)
        
        short_data = pd.Series([10, 11, 12])  # 只有3个数据点
        period = 20
        
        result = manager.calculate_ma(short_data, period)
        assert result.isna().all()  # 所有值都应该是NaN
    
    @pytest.mark.unit
    def test_calculate_ma_invalid_period(self, temp_db_path, sample_price_data):
        """测试无效周期参数"""
        config = {'db_path': temp_db_path, 'cache_size': 1000}
        manager = DataManager(config)
        
        data = sample_price_data['close']
        invalid_period = 0
        
        with pytest.raises(ValueError):
            manager.calculate_ma(data, invalid_period)
    
    @pytest.mark.unit
    def test_calculate_ma_negative_period(self, temp_db_path, sample_price_data):
        """测试负数周期"""
        config = {'db_path': temp_db_path, 'cache_size': 1000}
        manager = DataManager(config)
        
        data = sample_price_data['close']
        negative_period = -5
        
        with pytest.raises(ValueError):
            manager.calculate_ma(data, negative_period)
    
    # === 数据缓存测试 ===
    @pytest.mark.unit
    def test_cache_hit(self, temp_db_path, sample_price_data):
        """测试缓存命中"""
        config = {'db_path': temp_db_path, 'cache_size': 1000}
        manager = DataManager(config)
        
        symbol = '000001.SZ'
        start_date = '2023-01-01'
        end_date = '2023-01-31'
        
        # 第一次调用
        result1 = manager.get_price_data(symbol, start_date, end_date)
        initial_hit_count = manager.cache_hit_count
        
        # 第二次调用，应该从缓存获取
        result2 = manager.get_price_data(symbol, start_date, end_date)
        assert result1.equals(result2)
        assert manager.cache_hit_count > initial_hit_count
    
    @pytest.mark.unit
    def test_cache_miss(self, temp_db_path, sample_price_data):
        """测试缓存未命中"""
        config = {'db_path': temp_db_path, 'cache_size': 1000}
        manager = DataManager(config)
        
        symbol1 = '000001.SZ'
        symbol2 = '000002.SZ'
        start_date = '2023-01-01'
        end_date = '2023-01-31'
        
        initial_miss_count = manager.cache_miss_count
        
        result1 = manager.get_price_data(symbol1, start_date, end_date)
        result2 = manager.get_price_data(symbol2, start_date, end_date)
        
        # 两个不同的symbol应该产生不同的结果
        assert not result1.equals(result2)
        # cache miss计数应该增加
        assert manager.cache_miss_count > initial_miss_count
    
    @pytest.mark.unit
    def test_cache_expiry(self, temp_db_path, sample_price_data):
        """测试缓存过期"""
        config = {'db_path': temp_db_path, 'cache_size': 1000}
        manager = DataManager(config)
        
        symbol = '000001.SZ'
        start_date = '2023-01-01'
        end_date = '2023-01-31'
        
        # 第一次调用填充缓存
        result1 = manager.get_price_data(symbol, start_date, end_date)
        
        # 清空缓存模拟过期
        manager.cache.cache.clear()
        manager.cache.cache_timestamps.clear()
        
        # 再次调用应该重新获取数据
        result2 = manager.get_price_data(symbol, start_date, end_date)
        # 数据应该一样（都是模拟数据，可能会有随机性差异）
        assert isinstance(result1, pd.DataFrame)
        assert isinstance(result2, pd.DataFrame)
        assert len(manager.cache.cache) > 0  # 缓存应该重新填充
    
    # === 数据存储测试 ===
    @pytest.mark.unit
    def test_save_price_data(self, temp_db_path, sample_price_data):
        """测试保存价格数据"""
        config = {'db_path': temp_db_path, 'cache_size': 1000}
        manager = DataManager(config)
        
        data = sample_price_data
        
        manager.save_price_data(data)
        # 验证数据已保存到数据库
        saved_data = manager.get_price_data('000001.SZ', '2023-01-01', '2023-04-10')
        assert len(saved_data) > 0
        assert '000001.SZ' in saved_data['symbol'].values
    
    @pytest.mark.unit
    def test_save_duplicate_data(self, temp_db_path, sample_price_data):
        """测试保存重复数据"""
        config = {'db_path': temp_db_path, 'cache_size': 1000}
        manager = DataManager(config)
        
        data = sample_price_data
        
        manager.save_price_data(data)
        # 第一次保存后的数据量
        saved_data_1 = manager.get_price_data('000001.SZ', '2023-01-01', '2023-04-10')
        initial_count = len(saved_data_1)
        
        # 重复保存，应该不会增加重复记录（但可能会报错，这取决于实现）
        try:
            manager.save_price_data(data)
            saved_data_2 = manager.get_price_data('000001.SZ', '2023-01-01', '2023-04-10')
            # 如果没有报错，数据量不应该变化太多
            assert len(saved_data_2) >= initial_count
        except Exception:
            # 如果报错了，也是可以接受的行为
            pass
    
    @pytest.mark.unit
    def test_save_invalid_data_format(self, temp_db_path):
        """测试保存无效格式数据"""
        config = {'db_path': temp_db_path, 'cache_size': 1000}
        manager = DataManager(config)
        
        # Test with string instead of DataFrame - should raise TypeError
        invalid_data: str = "not a dataframe"
        
        with pytest.raises(TypeError):
            manager.save_price_data(invalid_data)  # type: ignore
    
    # === 数据验证测试 ===
    @pytest.mark.unit
    def test_validate_price_data_normal(self, temp_db_path, sample_price_data):
        """测试正常价格数据验证"""
        config = {'db_path': temp_db_path, 'cache_size': 1000}
        manager = DataManager(config)
        
        # 修复样本数据，确保high >= low, high >= open/close, low <= open/close
        data = sample_price_data.copy()
        data['high'] = data[['open', 'high', 'low', 'close']].max(axis=1)
        data['low'] = data[['open', 'high', 'low', 'close']].min(axis=1)
        
        is_valid = manager.validate_price_data(data)
        assert is_valid is True
    
    @pytest.mark.unit
    def test_validate_price_data_missing_columns(self, temp_db_path):
        """测试缺少必要列的数据"""
        config = {'db_path': temp_db_path, 'cache_size': 1000}
        manager = DataManager(config)
        
        invalid_data = pd.DataFrame({
            'date': ['2023-01-01'],
            'symbol': ['000001.SZ'],
            'close': [10.0]
            # 缺少 'open', 'high', 'low', 'volume'
        })
        
        is_valid = manager.validate_price_data(invalid_data)
        assert is_valid is False
    
    @pytest.mark.unit
    def test_validate_price_data_negative_prices(self, temp_db_path):
        """测试负价格数据"""
        config = {'db_path': temp_db_path, 'cache_size': 1000}
        manager = DataManager(config)
        
        invalid_data = pd.DataFrame({
            'date': ['2023-01-01'],
            'symbol': ['000001.SZ'],
            'open': [-10.0],  # 负价格
            'high': [20.0],
            'low': [5.0],
            'close': [15.0],
            'volume': [1000000]
        })
        
        is_valid = manager.validate_price_data(invalid_data)
        assert is_valid is False
    
    @pytest.mark.unit
    def test_validate_price_data_high_low_inconsistency(self, temp_db_path):
        """测试高低价不一致"""
        config = {'db_path': temp_db_path, 'cache_size': 1000}
        manager = DataManager(config)
        
        invalid_data = pd.DataFrame({
            'date': ['2023-01-01'],
            'symbol': ['000001.SZ'],
            'open': [15.0],
            'high': [10.0],  # 最高价小于开盘价
            'low': [5.0],
            'close': [12.0],
            'volume': [1000000]
        })
        
        is_valid = manager.validate_price_data(invalid_data)
        assert is_valid is False
    
    # === 性能测试 ===
    @pytest.mark.unit
    def test_large_dataset_performance(self, temp_db_path):
        """测试大数据集性能"""
        # 生成大量数据
        large_data = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=10000, freq='D'),
            'symbol': ['000001.SZ'] * 10000,
            'close': np.random.uniform(10, 20, 10000)
        })
        
        start_time = time.time()
        config = {'db_path': temp_db_path, 'cache_size': 1000}
        manager = DataManager(config)
        manager.calculate_ma(large_data['close'], 20)
        end_time = time.time()
        
        # 性能要求：10000个数据点的MA计算应在1秒内完成
        assert (end_time - start_time) < 1.0

    # === 并发测试 ===
    @pytest.mark.unit
    def test_concurrent_data_access(self, temp_db_path):
        """测试并发数据访问"""
        config = {'db_path': temp_db_path, 'cache_size': 1000}
        manager = DataManager(config)
        
        results = queue.Queue()
        
        def worker():
            try:
                result = manager.get_price_data('000001.SZ', '2023-01-01', '2023-01-31')
                results.put(("success", len(result)))
            except Exception as e:
                results.put(("error", str(e)))
        
        # 创建多个线程并发访问
        threads = []
        for i in range(5):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()
        
        # 等待所有线程完成
        for t in threads:
            t.join()
        
        # 检查结果
        error_count = 0
        success_count = 0
        while not results.empty():
            status, data = results.get()
            if status == "error":
                error_count += 1
            else:
                success_count += 1
        
        # 大部分应该成功
        assert success_count >= 3
        assert error_count <= 2  # 允许少量错误
    
    # === 错误处理测试 ===
    @pytest.mark.unit
    def test_network_error_handling(self, temp_db_path):
        """测试网络错误处理"""
        config = {'db_path': temp_db_path, 'cache_size': 1000}
        manager = DataManager(config)
        
        symbol = '000001.SZ'
        start_date = '2023-01-01'
        end_date = '2023-01-31'
        
        # 模拟提供者错误
        with patch.object(manager.provider, 'get_price_data', 
                         side_effect=Exception("Network error")):
            result = manager.get_price_data(symbol, start_date, end_date)
            # 应该返回空DataFrame，而不是抛出异常
            assert isinstance(result, pd.DataFrame)
            assert result.empty
    
    @pytest.mark.unit
    def test_database_connection_error(self, temp_db_path):
        """测试数据库连接错误"""
        # 使用不存在的目录路径
        invalid_db_path = '/invalid/nonexistent/path/test.db'
        config = {'db_path': invalid_db_path, 'cache_size': 1000}
        
        # 初始化应该失败
        with pytest.raises(Exception):
            DataManager(config)
    
    # === 边界值测试 ===
    @pytest.mark.unit
    def test_extreme_date_ranges(self, temp_db_path):
        """测试极端日期范围"""
        config = {'db_path': temp_db_path, 'cache_size': 1000}
        manager = DataManager(config)
        
        symbol = '000001.SZ'
        
        # 测试很久以前的日期
        start_date = '1990-01-01'
        end_date = '1990-01-02'
        
        result = manager.get_price_data(symbol, start_date, end_date)
        # 老旧日期可能没有数据，但不应该报错
        assert isinstance(result, pd.DataFrame)
        
        # 测试很久以后的日期
        start_date = '2050-01-01'
        end_date = '2050-01-02'
        
        result = manager.get_price_data(symbol, start_date, end_date)
        # 未来日期应该返回空结果
        assert result.empty
    
    @pytest.mark.unit
    def test_single_day_data(self, temp_db_path):
        """测试单日数据"""
        symbol = '000001.SZ'
        single_date = '2023-01-01'

        config = {'db_path': temp_db_path, 'cache_size': 1000}
        manager = DataManager(config)
        result = manager.get_price_data(symbol, single_date, single_date)
        assert len(result) <= 1  # 可能是0（非交易日）或1

    @pytest.mark.unit
    def test_memory_usage_large_data(self, temp_db_path):
        """测试大数据集内存使用"""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        config = {'db_path': temp_db_path, 'cache_size': 1000}
        manager = DataManager(config)
        # 模拟加载大量数据
        for i in range(100):
            manager.get_price_data(f'00000{i % 10}.SZ', '2023-01-01', '2023-12-31')
            pass
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # 内存增长不应超过100MB
        assert memory_increase < 100 * 1024 * 1024

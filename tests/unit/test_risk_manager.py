# Standard library imports
from datetime import datetime

# Third-party imports
import numpy as np
import pandas as pd
import pytest

# Local imports
from core.managers.risk_manager import RiskManager

class TestRiskManager:
    """风险管理器测试用例 - 覆盖常规case和边缘case"""
    
    @pytest.fixture
    def risk_config(self):
        """风险管理配置fixture"""
        return {
            'max_position_size': 0.1,      # 单票最大仓位10%
            'max_sector_exposure': 0.3,    # 行业最大敞口30%
            'max_total_exposure': 0.95,    # 最大总仓位95%
            'stop_loss_pct': 0.05,         # 止损5%
            'take_profit_pct': 0.15,       # 止盈15%
            'max_drawdown_limit': 0.2,     # 最大回撤20%
            'var_confidence': 0.95,        # VaR置信度95%
            'var_window': 252,             # VaR计算窗口252天
            'correlation_threshold': 0.8    # 相关性阈值80%
        }
    
    @pytest.fixture
    def sample_portfolio(self):
        """样本投资组合fixture"""
        return {
            'total_value': 1000000,
            'cash': 200000,
            'positions': {
                '000001.SZ': {'quantity': 5000, 'price': 15.0, 'value': 75000, 'sector': 'Finance'},
                '000002.SZ': {'quantity': 3000, 'price': 20.0, 'value': 60000, 'sector': 'Finance'},
                '600000.SH': {'quantity': 2000, 'price': 25.0, 'value': 50000, 'sector': 'Finance'},
                '000858.SZ': {'quantity': 4000, 'price': 30.0, 'value': 120000, 'sector': 'Technology'},
                '600519.SH': {'quantity': 1000, 'price': 180.0, 'value': 180000, 'sector': 'Consumer'}
            }
        }
    
    @pytest.fixture
    def sample_price_history(self):
        """样本价格历史数据fixture"""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        symbols = ['000001.SZ', '000002.SZ', '600000.SH']
        
        data = {}
        for symbol in symbols:
            np.random.seed(42)  # 确保可重现性
            returns = np.random.normal(0.001, 0.02, 100)  # 日收益率
            prices = [15.0]  # 初始价格
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            data[symbol] = pd.Series(prices, index=dates)
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_order(self):
        """样本订单fixture"""
        return {
            'symbol': '000001.SZ',
            'side': 'BUY',
            'quantity': 1000,
            'price': 15.0,
            'order_type': 'MARKET',
            'timestamp': datetime.now()
        }
    
    # === 初始化测试 ===
    @pytest.mark.unit
    def test_risk_manager_init_success(self, risk_config):
        """测试风险管理器正常初始化"""
        risk_manager = RiskManager(risk_config)
        assert risk_manager.max_position_size == 0.1
        assert risk_manager.max_drawdown_limit == 0.2
        assert risk_manager.var_confidence == 0.95
    
    @pytest.mark.unit
    def test_risk_manager_init_default_config(self):
        """测试默认配置初始化"""
        risk_manager = RiskManager()
        assert risk_manager.max_position_size > 0
        assert risk_manager.max_drawdown_limit > 0
    
    @pytest.mark.unit
    def test_risk_manager_init_invalid_config(self):
        """测试无效配置"""
        invalid_configs = [
            {'max_position_size': -0.1},     # 负数
            {'max_position_size': 1.5},      # 大于1
            {'var_confidence': 1.5},         # 大于1
            {'var_confidence': -0.1}         # 负数
        ]
        
        for config in invalid_configs:
            with pytest.raises(ValueError):
                RiskManager(config)
    
    # === 仓位风险检查测试 ===
    @pytest.mark.unit
    def test_check_position_size_within_limit(self, risk_config, sample_portfolio, sample_order):
        """测试仓位在限制内"""
        risk_manager = RiskManager(risk_config)
        
        # 小订单应该通过
        small_order = sample_order.copy()
        small_order['quantity'] = 100
        
        result = risk_manager.check_position_size(small_order, sample_portfolio)
        assert result['allowed'] is True
        assert 'risk_level' in result
    
    @pytest.mark.unit
    def test_check_position_size_exceed_limit(self, risk_config, sample_portfolio, sample_order):
        """测试仓位超过限制"""
        risk_manager = RiskManager(risk_config)
        
        # 大订单应该被拒绝
        large_order = sample_order.copy()
        large_order['quantity'] = 10000  # 价值150万，超过10%限制
        
        result = risk_manager.check_position_size(large_order, sample_portfolio)
        assert result['allowed'] is False
        assert 'exceeds maximum position size' in result['reason']
    
    @pytest.mark.unit
    def test_check_sector_exposure_within_limit(self, risk_config, sample_portfolio):
        """测试行业敞口在限制内"""
        risk_manager = RiskManager(risk_config)
        
        # 金融行业当前敞口：(75000+60000+50000)/1000000 = 18.5% < 30%
        new_finance_order = {
            'symbol': '000001.SZ',
            'side': 'BUY',
            'quantity': 500,
            'price': 15.0,
            'sector': 'Finance'
        }
        
        result = risk_manager.check_sector_exposure(new_finance_order, sample_portfolio)
        assert result['allowed'] is True
    
    @pytest.mark.unit
    def test_check_total_exposure_within_limit(self, risk_config, sample_portfolio, sample_order):
        """测试总仓位在限制内"""
        risk_manager = RiskManager(risk_config)
        
        # 当前总仓位：(1000000-200000)/1000000 = 80% < 95%
        result = risk_manager.check_total_exposure(sample_order, sample_portfolio)
        assert result['allowed'] is True
    
    # === 止损测试 ===
    @pytest.mark.unit
    def test_check_stop_loss_no_trigger(self, risk_config, sample_portfolio):
        """测试未触发止损"""
        risk_manager = RiskManager(risk_config)
        
        # 价格变化不大，不触发止损
        current_prices = {
            '000001.SZ': 14.7,  # 从15.0跌到14.7，跌幅2% < 5%
            '000002.SZ': 19.5,  # 从20.0跌到19.5，跌幅2.5% < 5%
        }
        
        stop_loss_orders = risk_manager.check_stop_loss(sample_portfolio, current_prices)
        
        # 不应该触发止损
        assert len(stop_loss_orders) == 0
    
    # === 信号风险检查测试 ===
    @pytest.mark.unit
    def test_check_signal_risk_normal(self, risk_config, sample_order, sample_portfolio):
        """测试正常信号风险检查"""
        risk_manager = RiskManager(risk_config)
        
        # 获取当前持仓
        current_positions = sample_portfolio.get('positions', {})
        
        # 检查信号风险
        risk_check = risk_manager.check_signal_risk(sample_order, current_positions)
        
        # 应该有风险检查结果
        assert hasattr(risk_check, 'allowed') or 'allowed' in risk_check
    
    # === 配置更新测试 ===
    @pytest.mark.unit
    def test_update_config_success(self, risk_config):
        """测试配置更新成功"""
        risk_manager = RiskManager(risk_config)
        
        # 更新配置
        new_config = {
            'max_position_size': 0.05,  # 从10%降到5%
            'stop_loss_pct': 0.03       # 从5%降到3%
        }
        
        risk_manager.update_config(new_config)
        
        assert risk_manager.max_position_size == 0.05
        assert risk_manager.stop_loss_pct == 0.03
    
    @pytest.mark.unit
    def test_update_config_invalid(self, risk_config):
        """测试无效配置更新"""
        risk_manager = RiskManager(risk_config)
        
        # 无效配置应该被拒绝
        invalid_updates = [
            {'max_position_size': 1.5},    # 超过100%
            {'max_position_size': -0.1},   # 负数
            {'stop_loss_pct': -0.1}        # 负数
        ]
        
        for invalid_config in invalid_updates:
            with pytest.raises(ValueError):
                risk_manager.update_config(invalid_config)
    
    # === 综合风险检查测试 ===
    @pytest.mark.unit
    def test_comprehensive_risk_check(self, risk_config, sample_portfolio):
        """测试综合风险检查"""
        risk_manager = RiskManager(risk_config)
        
        # 运行综合风险检查
        result = risk_manager.comprehensive_risk_check(sample_portfolio)
        
        # 应该有完整的风险检查结果
        assert isinstance(result, dict)
        assert 'risk_level' in result
        assert 'total_positions' in result
        assert 'total_value' in result
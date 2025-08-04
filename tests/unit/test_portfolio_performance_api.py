"""
投资组合绩效分析API的TDD测试

按照TDD原则，先编写完整的测试确保测试全部失败，然后实现功能代码
测试投资组合分析、绩效计算、风险指标、基准比较等功能
"""

import pytest
import pytest_asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Optional
from unittest.mock import AsyncMock, patch, MagicMock

# 待实现的模块
from myQuant.interfaces.api.portfolio_api import PortfolioAPI
from myQuant.core.analysis.performance_analyzer import PerformanceAnalyzer
from myQuant.core.analysis.risk_analyzer import RiskAnalyzer
from myQuant.core.analysis.benchmark_comparator import BenchmarkComparator
from myQuant.core.models.portfolio import Portfolio, Position, PerformanceMetrics


class TestPortfolioAPI:
    """投资组合API测试"""

    @pytest.fixture
    def mock_database_manager(self):
        """模拟数据库管理器"""
        return AsyncMock()

    @pytest.fixture
    def mock_portfolio_manager(self):
        """模拟投资组合管理器"""
        return AsyncMock()

    @pytest.fixture
    def mock_portfolio_repo(self):
        """模拟投资组合仓储"""
        return AsyncMock()

    @pytest.fixture
    def api(self, mock_portfolio_manager, mock_portfolio_repo, mock_database_manager):
        """投资组合API实例"""
        return PortfolioAPI(mock_portfolio_manager, mock_portfolio_repo, mock_database_manager)

    @pytest.fixture
    def sample_portfolio_data(self):
        """样本投资组合数据"""
        return {
            "user_id": 1,
            "total_value": 1250000.00,
            "cash_balance": 250000.00,
            "position_value": 1000000.00,
            "total_pnl": 250000.00,
            "total_return": 25.00,
            "positions": [
                {
                    "symbol": "000001.SZ",
                    "name": "平安银行",
                    "quantity": 10000,
                    "average_price": 12.00,
                    "current_price": 12.75,
                    "market_value": 127500.00,
                    "unrealized_pnl": 7500.00,
                    "percentage": 10.20,
                    "sector": "金融"
                },
                {
                    "symbol": "000002.SZ", 
                    "name": "万科A",
                    "quantity": 5000,
                    "average_price": 20.00,
                    "current_price": 22.50,
                    "market_value": 112500.00,
                    "unrealized_pnl": 12500.00,
                    "percentage": 9.00,
                    "sector": "房地产"
                }
            ]
        }

    @pytest.mark.asyncio
    async def test_get_portfolio_summary_endpoint(self, api, mock_portfolio_manager, sample_portfolio_data):
        """测试获取投资组合概览端点"""
        # Arrange
        mock_portfolio_manager.get_portfolio_summary.return_value = sample_portfolio_data
        
        # Act
        result = await api.get_portfolio_summary(user_id=1)
        
        # Assert
        assert result['success'] is True
        assert result['data']['total_value'] == 1250000.00
        assert result['data']['total_return'] == 25.00
        assert len(result['data']['positions']) == 2
        
        mock_portfolio_manager.get_portfolio_summary.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_get_positions_endpoint(self, api, mock_database_manager, sample_portfolio_data):
        """测试获取持仓详情端点"""
        # Arrange
        mock_database_manager.fetch_all.return_value = sample_portfolio_data['positions']
        
        # Act
        result = await api.get_positions(user_id=1)
        
        # Assert
        assert result['success'] is True
        assert len(result['data']['positions']) == 2
        assert result['data']['total_value'] == 240000.00  # 127500 + 112500
        
        mock_database_manager.fetch_all.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_portfolio_performance_endpoint(self, api, mock_database_manager):
        """测试获取投资组合绩效端点"""
        # Arrange
        mock_performance_data = {
            "returns": {
                "total_return": 25.00,
                "annualized_return": 18.50,
                "daily_returns": [0.01, -0.005, 0.02, 0.001],
                "cumulative_returns": [1.01, 1.005, 1.025, 1.026]
            },
            "risk_metrics": {
                "volatility": 15.20,
                "sharpe_ratio": 1.22,
                "max_drawdown": -8.50,
                "var_95": -25000.00
            },
            "benchmark_comparison": {
                "benchmark": "沪深300",
                "alpha": 3.20,
                "beta": 1.15,
                "tracking_error": 5.80
            }
        }
        
        with patch.object(api, '_calculate_performance_metrics') as mock_calc:
            mock_calc.return_value = mock_performance_data
            
            # Act
            result = await api.get_portfolio_performance(user_id=1, period="1y")
            
            # Assert
            assert result['success'] is True
            assert result['data']['returns']['total_return'] == 25.00
            assert result['data']['risk_metrics']['sharpe_ratio'] == 1.22
            assert result['data']['benchmark_comparison']['alpha'] == 3.20

    @pytest.mark.asyncio
    async def test_get_transactions_endpoint(self, api, mock_database_manager):
        """测试获取交易记录端点"""
        # Arrange
        mock_transactions = [
            {
                "id": 1,
                "symbol": "000001.SZ",
                "side": "BUY",
                "quantity": 1000,
                "price": 12.00,
                "commission": 12.00,
                "executed_at": "2024-01-01T09:30:00"
            },
            {
                "id": 2,
                "symbol": "000001.SZ",
                "side": "SELL",
                "quantity": 500,
                "price": 12.75,
                "commission": 6.38,
                "executed_at": "2024-01-15T14:30:00"
            }
        ]
        mock_database_manager.fetch_all.return_value = mock_transactions
        
        # Act
        result = await api.get_transactions(
            user_id=1,
            symbol="000001.SZ",
            start_date="2024-01-01",
            end_date="2024-01-31",
            page=1,
            size=10
        )
        
        # Assert
        assert result['success'] is True
        assert len(result['data']['transactions']) == 2
        assert result['data']['transactions'][0]['symbol'] == "000001.SZ"

    @pytest.mark.asyncio
    async def test_get_sector_allocation(self, api, sample_portfolio_data):
        """测试获取行业配置"""
        # Arrange
        with patch.object(api, '_get_portfolio_positions') as mock_positions:
            mock_positions.return_value = sample_portfolio_data['positions']
            
            # Act
            result = await api.get_sector_allocation(user_id=1)
            
            # Assert
            assert result['success'] is True
            assert len(result['data']['sectors']) == 2
            
            sectors = {sector['sector']: sector for sector in result['data']['sectors']}
            assert '金融' in sectors
            assert '房地产' in sectors
            assert sectors['金融']['percentage'] > 0
            assert sectors['房地产']['percentage'] > 0

    @pytest.mark.asyncio
    async def test_calculate_risk_metrics(self, api):
        """测试计算风险指标"""
        # Arrange
        portfolio_values = [100000 + i * 1000 + np.random.normal(0, 5000) for i in range(252)]  # 一年的数据
        
        with patch.object(api, '_get_portfolio_value_history') as mock_history:
            mock_history.return_value = portfolio_values
            
            # Act
            result = await api.calculate_risk_metrics(user_id=1, period="1y")
            
            # Assert
            assert result['success'] is True
            assert 'volatility' in result['data']
            assert 'var_95' in result['data']
            assert 'expected_shortfall' in result['data']
            assert 'max_drawdown' in result['data']
            assert 'sharpe_ratio' in result['data']

    @pytest.mark.asyncio
    async def test_compare_with_benchmark(self, api):
        """测试与基准比较"""
        # Arrange
        portfolio_returns = [0.01, -0.005, 0.02, 0.001, 0.015]
        benchmark_returns = [0.008, -0.003, 0.015, 0.002, 0.012]
        
        with patch.object(api, '_get_portfolio_returns') as mock_portfolio_returns:
            with patch.object(api, '_get_benchmark_returns') as mock_benchmark_returns:
                mock_portfolio_returns.return_value = portfolio_returns
                mock_benchmark_returns.return_value = benchmark_returns
                
                # Act
                result = await api.compare_with_benchmark(
                    user_id=1, 
                    benchmark="沪深300", 
                    period="1m"
                )
                
                # Assert
                assert result['success'] is True
                assert 'alpha' in result['data']
                assert 'beta' in result['data']
                assert 'tracking_error' in result['data']
                assert 'information_ratio' in result['data']

    @pytest.mark.asyncio
    async def test_get_performance_attribution(self, api):
        """测试绩效归因分析"""
        # Act
        result = await api.get_performance_attribution(user_id=1, period="1m")
        
        # Assert
        assert result['success'] is True
        assert 'sector_attribution' in result['data']
        assert 'stock_selection' in result['data']
        assert 'asset_allocation' in result['data']

    @pytest.mark.asyncio
    async def test_generate_performance_report(self, api):
        """测试生成绩效报告"""
        # Act
        result = await api.generate_performance_report(user_id=1, period="1y", format="json")
        
        # Assert
        assert result['success'] is True
        assert 'report' in result['data']
        assert 'summary' in result['data']['report']
        assert 'detailed_analysis' in result['data']['report']
        assert 'charts' in result['data']['report']

    @pytest.mark.asyncio
    async def test_portfolio_optimization_suggestions(self, api):
        """测试投资组合优化建议"""
        # Act
        result = await api.get_optimization_suggestions(user_id=1)
        
        # Assert
        assert result['success'] is True
        assert 'suggestions' in result['data']
        assert len(result['data']['suggestions']) > 0
        
        for suggestion in result['data']['suggestions']:
            assert 'type' in suggestion
            assert 'description' in suggestion
            assert 'priority' in suggestion


class TestPerformanceAnalyzer:
    """绩效分析器测试"""

    @pytest.fixture
    def analyzer(self):
        """绩效分析器实例"""
        return PerformanceAnalyzer()

    @pytest.fixture
    def sample_returns(self):
        """样本收益率数据"""
        np.random.seed(42)  # 确保测试结果可重现
        return np.random.normal(0.001, 0.02, 252)  # 252个交易日的日收益率

    def test_calculate_total_return(self, analyzer, sample_returns):
        """测试总收益率计算"""
        # Act
        total_return = analyzer.calculate_total_return(sample_returns)
        
        # Assert
        assert isinstance(total_return, float)
        assert -1 < total_return < 10  # 合理的收益率范围

    def test_calculate_annualized_return(self, analyzer, sample_returns):
        """测试年化收益率计算"""
        # Act
        annualized_return = analyzer.calculate_annualized_return(sample_returns, periods_per_year=252)
        
        # Assert
        assert isinstance(annualized_return, float)
        assert -1 < annualized_return < 10

    def test_calculate_volatility(self, analyzer, sample_returns):
        """测试波动率计算"""
        # Act
        volatility = analyzer.calculate_volatility(sample_returns, periods_per_year=252)
        
        # Assert
        assert isinstance(volatility, float)
        assert volatility > 0
        assert volatility < 5  # 合理的年化波动率

    def test_calculate_sharpe_ratio(self, analyzer, sample_returns):
        """测试夏普比率计算"""
        # Arrange
        risk_free_rate = 0.03  # 3%无风险利率
        
        # Act
        sharpe_ratio = analyzer.calculate_sharpe_ratio(sample_returns, risk_free_rate)
        
        # Assert
        assert isinstance(sharpe_ratio, float)
        assert -5 < sharpe_ratio < 5  # 合理的夏普比率范围

    def test_calculate_max_drawdown(self, analyzer):
        """测试最大回撤计算"""
        # Arrange
        portfolio_values = [100, 110, 105, 120, 115, 100, 95, 105, 110]
        
        # Act
        result = analyzer.calculate_max_drawdown(portfolio_values)
        
        # Assert
        assert isinstance(result, dict)
        assert "max_drawdown" in result
        assert "max_drawdown_duration" in result
        assert isinstance(result["max_drawdown"], float)
        assert result["max_drawdown"] <= 0  # 回撤应该是负值或零
        assert isinstance(result["max_drawdown_duration"], int)
        assert result["max_drawdown_duration"] >= 0

    def test_calculate_var(self, analyzer, sample_returns):
        """测试VaR计算"""
        # Act
        var_95 = analyzer.calculate_var(sample_returns, confidence_level=0.95)
        var_99 = analyzer.calculate_var(sample_returns, confidence_level=0.99)
        
        # Assert
        assert isinstance(var_95, float)
        assert isinstance(var_99, float)
        assert var_95 > var_99  # 95% VaR应该大于99% VaR (都是负值，但绝对值小)

    def test_calculate_expected_shortfall(self, analyzer, sample_returns):
        """测试期望损失(ES)计算"""
        # Act
        es_95 = analyzer.calculate_expected_shortfall(sample_returns, confidence_level=0.95)
        
        # Assert
        assert isinstance(es_95, float)
        assert es_95 <= 0  # 期望损失应该是负值或零

    def test_calculate_beta(self, analyzer):
        """测试Beta系数计算"""
        # Arrange
        portfolio_returns = [0.01, -0.005, 0.02, 0.001, 0.015]
        benchmark_returns = [0.008, -0.003, 0.015, 0.002, 0.012]
        
        # Act
        beta = analyzer.calculate_beta(portfolio_returns, benchmark_returns)
        
        # Assert
        assert isinstance(beta, float)
        assert 0 < beta < 3  # 合理的Beta范围

    def test_calculate_alpha(self, analyzer):
        """测试Alpha计算"""
        # Arrange
        portfolio_returns = [0.01, -0.005, 0.02, 0.001, 0.015]
        benchmark_returns = [0.008, -0.003, 0.015, 0.002, 0.012]
        risk_free_rate = 0.03 / 252  # 日无风险利率
        
        # Act
        alpha = analyzer.calculate_alpha(portfolio_returns, benchmark_returns, risk_free_rate)
        
        # Assert
        assert isinstance(alpha, float)
        assert -1 < alpha < 1  # 合理的Alpha范围

    def test_calculate_tracking_error(self, analyzer):
        """测试跟踪误差计算"""
        # Arrange
        portfolio_returns = [0.01, -0.005, 0.02, 0.001, 0.015]
        benchmark_returns = [0.008, -0.003, 0.015, 0.002, 0.012]
        
        # Act
        tracking_error = analyzer.calculate_tracking_error(portfolio_returns, benchmark_returns)
        
        # Assert
        assert isinstance(tracking_error, float)
        assert tracking_error >= 0

    def test_calculate_information_ratio(self, analyzer):
        """测试信息比率计算"""
        # Arrange
        portfolio_returns = [0.01, -0.005, 0.02, 0.001, 0.015]
        benchmark_returns = [0.008, -0.003, 0.015, 0.002, 0.012]
        
        # Act
        information_ratio = analyzer.calculate_information_ratio(portfolio_returns, benchmark_returns)
        
        # Assert
        assert isinstance(information_ratio, float)


class TestRiskAnalyzer:
    """风险分析器测试"""

    @pytest.fixture
    def risk_analyzer(self):
        return RiskAnalyzer()

    @pytest.fixture
    def sample_positions(self):
        """样本持仓数据"""
        return [
            {"symbol": "000001.SZ", "weight": 0.3, "sector": "金融"},
            {"symbol": "000002.SZ", "weight": 0.2, "sector": "房地产"},
            {"symbol": "600000.SH", "weight": 0.25, "sector": "金融"},
            {"symbol": "600036.SH", "weight": 0.15, "sector": "银行"},
            {"symbol": "000858.SZ", "weight": 0.1, "sector": "科技"}
        ]

    def test_calculate_concentration_risk(self, risk_analyzer, sample_positions):
        """测试集中度风险计算"""
        # Act
        concentration_metrics = risk_analyzer.calculate_concentration_risk(sample_positions)
        
        # Assert
        assert 'largest_position_weight' in concentration_metrics
        assert 'top5_concentration' in concentration_metrics
        assert 'herfindahl_index' in concentration_metrics
        assert concentration_metrics['largest_position_weight'] == 0.3

    def test_calculate_sector_concentration(self, risk_analyzer, sample_positions):
        """测试行业集中度计算"""
        # Act
        sector_concentration = risk_analyzer.calculate_sector_concentration(sample_positions)
        
        # Assert
        assert '金融' in sector_concentration
        assert sector_concentration['金融'] == 0.55  # 0.3 + 0.25

    def test_calculate_portfolio_var(self, risk_analyzer):
        """测试投资组合VaR计算"""
        # Arrange
        correlation_matrix = np.array([
            [1.0, 0.3, 0.4],
            [0.3, 1.0, 0.2],
            [0.4, 0.2, 1.0]
        ])
        weights = np.array([0.4, 0.3, 0.3])
        volatilities = np.array([0.2, 0.25, 0.18])
        
        # Act
        portfolio_var = risk_analyzer.calculate_portfolio_var(
            weights, volatilities, correlation_matrix, confidence_level=0.95
        )
        
        # Assert
        assert isinstance(portfolio_var, float)
        assert portfolio_var < 0  # VaR应该是负值

    def test_stress_testing(self, risk_analyzer, sample_positions):
        """测试压力测试"""
        # Arrange
        stress_scenarios = {
            "market_crash": {"market_factor": -0.2, "volatility_factor": 2.0},
            "sector_rotation": {"sector_factors": {"金融": -0.1, "科技": 0.05}}
        }
        
        # Act
        stress_results = risk_analyzer.run_stress_tests(sample_positions, stress_scenarios)
        
        # Assert
        assert 'market_crash' in stress_results
        assert 'sector_rotation' in stress_results
        assert stress_results['market_crash']['portfolio_impact'] < 0

    def test_calculate_risk_budgets(self, risk_analyzer, sample_positions):
        """测试风险预算计算"""
        # Act
        risk_budgets = risk_analyzer.calculate_risk_budgets(sample_positions)
        
        # Assert
        assert len(risk_budgets) == len(sample_positions)
        assert sum(risk_budgets.values()) == pytest.approx(1.0, rel=1e-2)  # 风险预算总和应该接近1


class TestBenchmarkComparator:
    """基准比较器测试"""

    @pytest.fixture
    def comparator(self):
        return BenchmarkComparator()

    def test_load_benchmark_data(self, comparator):
        """测试加载基准数据"""
        # Act
        benchmark_data = comparator.load_benchmark_data("沪深300", "2024-01-01", "2024-12-31")
        
        # Assert
        assert benchmark_data is not None
        assert len(benchmark_data) > 0

    def test_calculate_relative_performance(self, comparator):
        """测试相对绩效计算"""
        # Arrange
        portfolio_returns = [0.01, -0.005, 0.02, 0.001, 0.015]
        benchmark_returns = [0.008, -0.003, 0.015, 0.002, 0.012]
        
        # Act
        relative_perf = comparator.calculate_relative_performance(portfolio_returns, benchmark_returns)
        
        # Assert
        assert 'excess_return' in relative_perf
        assert 'outperformance_ratio' in relative_perf
        assert 'up_capture' in relative_perf
        assert 'down_capture' in relative_perf

    def test_attribution_analysis(self, comparator):
        """测试归因分析"""
        # Arrange
        portfolio_weights = {"金融": 0.4, "科技": 0.3, "消费": 0.3}
        benchmark_weights = {"金融": 0.3, "科技": 0.4, "消费": 0.3}
        sector_returns = {"金融": 0.05, "科技": 0.08, "消费": 0.03}
        
        # Act
        attribution = comparator.perform_attribution_analysis(
            portfolio_weights, benchmark_weights, sector_returns
        )
        
        # Assert
        assert 'allocation_effect' in attribution
        assert 'selection_effect' in attribution
        assert 'interaction_effect' in attribution


if __name__ == "__main__":
    # 运行测试确保全部失败（TDD第一步）
    pytest.main([__file__, "-v", "--tb=short"])
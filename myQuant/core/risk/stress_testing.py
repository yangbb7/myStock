import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import json

class StressTestType(Enum):
    HISTORICAL = "historical"
    HYPOTHETICAL = "hypothetical"
    MONTE_CARLO = "monte_carlo"
    FACTOR_SHOCK = "factor_shock"
    TAIL_RISK = "tail_risk"
    LIQUIDITY_STRESS = "liquidity_stress"
    CREDIT_STRESS = "credit_stress"

class StressScenario(Enum):
    MARKET_CRASH = "market_crash"
    INTEREST_RATE_SHOCK = "interest_rate_shock"
    CURRENCY_CRISIS = "currency_crisis"
    SECTOR_ROTATION = "sector_rotation"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    CREDIT_CRISIS = "credit_crisis"
    VOLATILITY_SPIKE = "volatility_spike"
    CORRELATION_BREAKDOWN = "correlation_breakdown"

@dataclass
class StressTestConfig:
    test_type: StressTestType
    scenario: StressScenario
    name: str
    description: str
    shocks: Dict[str, float]
    duration_days: int
    confidence_level: float = 0.95
    num_simulations: int = 10000
    risk_factors: List[str] = field(default_factory=list)
    
@dataclass
class StressTestResult:
    test_id: str
    config: StressTestConfig
    timestamp: datetime
    portfolio_value_base: float
    portfolio_value_stressed: float
    absolute_loss: float
    relative_loss: float
    var_stressed: float
    max_drawdown: float
    positions_impact: Dict[str, float]
    risk_metrics: Dict[str, float]
    passed: bool
    recommendations: List[str]
    execution_time_seconds: float

class StressTestingFramework:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 执行器
        self.executor = ThreadPoolExecutor(max_workers=config.get('max_workers', 4))
        
        # 压力测试配置
        self.test_configs = self._initialize_test_configs()
        
        # 历史数据
        self.historical_data = {}
        
        # 测试结果
        self.test_results: Dict[str, StressTestResult] = {}
        
        # 风险因子
        self.risk_factors = [
            'equity_market', 'bond_market', 'fx_market', 'commodity_market',
            'interest_rate', 'credit_spread', 'volatility', 'liquidity'
        ]
        
    def _initialize_test_configs(self) -> Dict[str, StressTestConfig]:
        """初始化压力测试配置"""
        configs = {}
        
        # 市场崩盘情景
        configs['market_crash'] = StressTestConfig(
            test_type=StressTestType.HYPOTHETICAL,
            scenario=StressScenario.MARKET_CRASH,
            name="Market Crash Scenario",
            description="Global equity market crash similar to 2008",
            shocks={
                'equity_market': -0.40,  # 股市下跌40%
                'bond_market': 0.20,     # 债市上涨20%
                'fx_market': -0.15,      # 汇率贬值15%
                'volatility': 3.0,       # 波动率增加3倍
                'liquidity': -0.50       # 流动性下降50%
            },
            duration_days=30,
            confidence_level=0.99,
            risk_factors=['equity_market', 'volatility', 'liquidity']
        )
        
        # 利率冲击情景
        configs['interest_rate_shock'] = StressTestConfig(
            test_type=StressTestType.HYPOTHETICAL,
            scenario=StressScenario.INTEREST_RATE_SHOCK,
            name="Interest Rate Shock",
            description="Sudden 300bps interest rate increase",
            shocks={
                'interest_rate': 0.03,   # 利率上升300基点
                'bond_market': -0.25,    # 债市下跌25%
                'credit_spread': 0.02,   # 信用利差扩大200基点
                'fx_market': 0.10,       # 汇率升值10%
                'equity_market': -0.15   # 股市下跌15%
            },
            duration_days=7,
            confidence_level=0.95,
            risk_factors=['interest_rate', 'bond_market', 'credit_spread']
        )
        
        # 汇率危机情景
        configs['currency_crisis'] = StressTestConfig(
            test_type=StressTestType.HYPOTHETICAL,
            scenario=StressScenario.CURRENCY_CRISIS,
            name="Currency Crisis",
            description="Major currency devaluation",
            shocks={
                'fx_market': -0.30,      # 汇率贬值30%
                'equity_market': -0.20,  # 股市下跌20%
                'bond_market': -0.10,    # 债市下跌10%
                'interest_rate': 0.05,   # 利率上升500基点
                'volatility': 2.0        # 波动率增加2倍
            },
            duration_days=14,
            confidence_level=0.95,
            risk_factors=['fx_market', 'interest_rate', 'volatility']
        )
        
        # 流动性危机情景
        configs['liquidity_crisis'] = StressTestConfig(
            test_type=StressTestType.LIQUIDITY_STRESS,
            scenario=StressScenario.LIQUIDITY_CRISIS,
            name="Liquidity Crisis",
            description="Severe liquidity drought",
            shocks={
                'liquidity': -0.70,      # 流动性下降70%
                'credit_spread': 0.04,   # 信用利差扩大400基点
                'volatility': 2.5,       # 波动率增加2.5倍
                'equity_market': -0.25,  # 股市下跌25%
                'bond_market': -0.15     # 债市下跌15%
            },
            duration_days=21,
            confidence_level=0.99,
            risk_factors=['liquidity', 'credit_spread', 'volatility']
        )
        
        # 信用危机情景
        configs['credit_crisis'] = StressTestConfig(
            test_type=StressTestType.CREDIT_STRESS,
            scenario=StressScenario.CREDIT_CRISIS,
            name="Credit Crisis",
            description="Widespread credit defaults",
            shocks={
                'credit_spread': 0.06,   # 信用利差扩大600基点
                'bond_market': -0.30,    # 债市下跌30%
                'equity_market': -0.35,  # 股市下跌35%
                'liquidity': -0.60,      # 流动性下降60%
                'volatility': 2.8        # 波动率增加2.8倍
            },
            duration_days=45,
            confidence_level=0.99,
            risk_factors=['credit_spread', 'bond_market', 'liquidity']
        )
        
        # 波动率冲击情景
        configs['volatility_spike'] = StressTestConfig(
            test_type=StressTestType.FACTOR_SHOCK,
            scenario=StressScenario.VOLATILITY_SPIKE,
            name="Volatility Spike",
            description="Sudden volatility explosion",
            shocks={
                'volatility': 4.0,       # 波动率增加4倍
                'equity_market': -0.18,  # 股市下跌18%
                'bond_market': -0.08,    # 债市下跌8%
                'liquidity': -0.40,      # 流动性下降40%
                'correlation': 0.80      # 相关性增加到0.8
            },
            duration_days=5,
            confidence_level=0.95,
            risk_factors=['volatility', 'correlation']
        )
        
        return configs
    
    async def run_stress_test(self, test_name: str, portfolio_data: Dict[str, Any]) -> StressTestResult:
        """运行压力测试"""
        start_time = datetime.now()
        
        try:
            if test_name not in self.test_configs:
                raise ValueError(f"Unknown stress test: {test_name}")
            
            config = self.test_configs[test_name]
            test_id = f"{test_name}_{int(start_time.timestamp() * 1000)}"
            
            self.logger.info(f"Running stress test: {config.name}")
            
            # 获取基准组合价值
            portfolio_value_base = self._calculate_portfolio_value(portfolio_data)
            
            # 执行压力测试
            if config.test_type == StressTestType.HYPOTHETICAL:
                result = await self._run_hypothetical_test(config, portfolio_data)
            elif config.test_type == StressTestType.HISTORICAL:
                result = await self._run_historical_test(config, portfolio_data)
            elif config.test_type == StressTestType.MONTE_CARLO:
                result = await self._run_monte_carlo_test(config, portfolio_data)
            elif config.test_type == StressTestType.FACTOR_SHOCK:
                result = await self._run_factor_shock_test(config, portfolio_data)
            elif config.test_type == StressTestType.LIQUIDITY_STRESS:
                result = await self._run_liquidity_stress_test(config, portfolio_data)
            elif config.test_type == StressTestType.CREDIT_STRESS:
                result = await self._run_credit_stress_test(config, portfolio_data)
            else:
                raise ValueError(f"Unsupported test type: {config.test_type}")
            
            # 计算损失
            absolute_loss = portfolio_value_base - result['portfolio_value_stressed']
            relative_loss = absolute_loss / portfolio_value_base if portfolio_value_base > 0 else 0
            
            # 评估测试结果
            passed = self._evaluate_test_result(config, relative_loss)
            
            # 生成建议
            recommendations = self._generate_recommendations(config, result, relative_loss)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # 创建测试结果
            test_result = StressTestResult(
                test_id=test_id,
                config=config,
                timestamp=start_time,
                portfolio_value_base=portfolio_value_base,
                portfolio_value_stressed=result['portfolio_value_stressed'],
                absolute_loss=absolute_loss,
                relative_loss=relative_loss,
                var_stressed=result.get('var_stressed', 0),
                max_drawdown=result.get('max_drawdown', 0),
                positions_impact=result.get('positions_impact', {}),
                risk_metrics=result.get('risk_metrics', {}),
                passed=passed,
                recommendations=recommendations,
                execution_time_seconds=execution_time
            )
            
            # 存储结果
            self.test_results[test_id] = test_result
            
            self.logger.info(f"Stress test completed: {config.name}, Loss: {relative_loss:.2%}")
            
            return test_result
            
        except Exception as e:
            self.logger.error(f"Error running stress test {test_name}: {e}")
            raise
    
    async def _run_hypothetical_test(self, config: StressTestConfig, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """运行假设性压力测试"""
        positions = portfolio_data.get('positions', {})
        prices = portfolio_data.get('prices', {})
        
        stressed_positions = {}
        portfolio_value_stressed = 0
        positions_impact = {}
        
        for symbol, position in positions.items():
            original_price = prices.get(symbol, 0)
            quantity = position.get('quantity', 0)
            
            # 应用压力冲击
            shock_factor = self._calculate_position_shock(symbol, config.shocks)
            stressed_price = original_price * (1 + shock_factor)
            
            # 计算冲击后的价值
            stressed_value = quantity * stressed_price
            original_value = quantity * original_price
            
            stressed_positions[symbol] = {
                'quantity': quantity,
                'price': stressed_price,
                'value': stressed_value
            }
            
            portfolio_value_stressed += stressed_value
            positions_impact[symbol] = (stressed_value - original_value) / original_value if original_value > 0 else 0
        
        # 计算压力测试下的VaR
        var_stressed = await self._calculate_stressed_var(stressed_positions, config)
        
        # 计算最大回撤
        max_drawdown = abs(min(positions_impact.values())) if positions_impact else 0
        
        return {
            'portfolio_value_stressed': portfolio_value_stressed,
            'var_stressed': var_stressed,
            'max_drawdown': max_drawdown,
            'positions_impact': positions_impact,
            'risk_metrics': {
                'volatility_multiplier': config.shocks.get('volatility', 1.0),
                'liquidity_factor': config.shocks.get('liquidity', 0.0),
                'correlation_impact': config.shocks.get('correlation', 0.0)
            }
        }
    
    async def _run_historical_test(self, config: StressTestConfig, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """运行历史情景压力测试"""
        # 简化的历史测试实现
        positions = portfolio_data.get('positions', {})
        
        # 模拟历史情景数据
        historical_shocks = self._get_historical_shocks(config.scenario)
        
        stressed_positions = {}
        portfolio_value_stressed = 0
        positions_impact = {}
        
        for symbol, position in positions.items():
            original_value = position.get('market_value', 0)
            
            # 应用历史冲击
            shock = historical_shocks.get(symbol, -0.20)  # 默认-20%
            stressed_value = original_value * (1 + shock)
            
            stressed_positions[symbol] = stressed_value
            portfolio_value_stressed += stressed_value
            positions_impact[symbol] = shock
        
        var_stressed = portfolio_value_stressed * 0.05  # 简化VaR计算
        max_drawdown = abs(min(positions_impact.values())) if positions_impact else 0
        
        return {
            'portfolio_value_stressed': portfolio_value_stressed,
            'var_stressed': var_stressed,
            'max_drawdown': max_drawdown,
            'positions_impact': positions_impact,
            'risk_metrics': {
                'historical_period': '2008-2009',
                'scenario_duration': config.duration_days
            }
        }
    
    async def _run_monte_carlo_test(self, config: StressTestConfig, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """运行蒙特卡洛压力测试"""
        positions = portfolio_data.get('positions', {})
        num_simulations = config.num_simulations
        
        # 存储所有模拟结果
        simulation_results = []
        
        for _ in range(num_simulations):
            simulation_value = 0
            
            for symbol, position in positions.items():
                original_value = position.get('market_value', 0)
                
                # 生成随机冲击
                shock = np.random.normal(
                    config.shocks.get('equity_market', -0.1),
                    config.shocks.get('volatility', 0.2)
                )
                
                stressed_value = original_value * (1 + shock)
                simulation_value += stressed_value
            
            simulation_results.append(simulation_value)
        
        # 计算统计量
        simulation_results = np.array(simulation_results)
        portfolio_value_stressed = np.percentile(simulation_results, (1 - config.confidence_level) * 100)
        
        var_stressed = np.percentile(simulation_results, 5)
        max_drawdown = (np.min(simulation_results) - np.mean(simulation_results)) / np.mean(simulation_results)
        
        return {
            'portfolio_value_stressed': portfolio_value_stressed,
            'var_stressed': var_stressed,
            'max_drawdown': abs(max_drawdown),
            'positions_impact': {},
            'risk_metrics': {
                'num_simulations': num_simulations,
                'confidence_level': config.confidence_level,
                'mean_result': np.mean(simulation_results),
                'std_result': np.std(simulation_results)
            }
        }
    
    async def _run_factor_shock_test(self, config: StressTestConfig, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """运行因子冲击压力测试"""
        positions = portfolio_data.get('positions', {})
        factor_exposures = self._calculate_factor_exposures(positions)
        
        stressed_positions = {}
        portfolio_value_stressed = 0
        positions_impact = {}
        
        for symbol, position in positions.items():
            original_value = position.get('market_value', 0)
            
            # 计算因子冲击影响
            total_shock = 0
            for factor, exposure in factor_exposures.get(symbol, {}).items():
                if factor in config.shocks:
                    total_shock += exposure * config.shocks[factor]
            
            stressed_value = original_value * (1 + total_shock)
            
            stressed_positions[symbol] = stressed_value
            portfolio_value_stressed += stressed_value
            positions_impact[symbol] = total_shock
        
        var_stressed = portfolio_value_stressed * 0.08  # 简化VaR计算
        max_drawdown = abs(min(positions_impact.values())) if positions_impact else 0
        
        return {
            'portfolio_value_stressed': portfolio_value_stressed,
            'var_stressed': var_stressed,
            'max_drawdown': max_drawdown,
            'positions_impact': positions_impact,
            'risk_metrics': {
                'factor_count': len(config.shocks),
                'max_factor_shock': max(abs(s) for s in config.shocks.values())
            }
        }
    
    async def _run_liquidity_stress_test(self, config: StressTestConfig, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """运行流动性压力测试"""
        positions = portfolio_data.get('positions', {})
        
        stressed_positions = {}
        portfolio_value_stressed = 0
        positions_impact = {}
        liquidity_costs = {}
        
        for symbol, position in positions.items():
            original_value = position.get('market_value', 0)
            
            # 计算流动性成本
            liquidity_score = self._get_liquidity_score(symbol)
            liquidity_shock = config.shocks.get('liquidity', -0.5)
            
            # 流动性成本与仓位大小和流动性评分相关
            liquidity_cost = abs(liquidity_shock) * (1 - liquidity_score) * 0.1
            
            # 应用市场冲击和流动性成本
            market_shock = config.shocks.get('equity_market', -0.2)
            total_shock = market_shock - liquidity_cost
            
            stressed_value = original_value * (1 + total_shock)
            
            stressed_positions[symbol] = stressed_value
            portfolio_value_stressed += stressed_value
            positions_impact[symbol] = total_shock
            liquidity_costs[symbol] = liquidity_cost
        
        var_stressed = portfolio_value_stressed * 0.12  # 流动性危机下VaR更高
        max_drawdown = abs(min(positions_impact.values())) if positions_impact else 0
        
        return {
            'portfolio_value_stressed': portfolio_value_stressed,
            'var_stressed': var_stressed,
            'max_drawdown': max_drawdown,
            'positions_impact': positions_impact,
            'risk_metrics': {
                'liquidity_costs': liquidity_costs,
                'avg_liquidity_cost': np.mean(list(liquidity_costs.values())) if liquidity_costs else 0,
                'liquidity_stress_factor': config.shocks.get('liquidity', 0)
            }
        }
    
    async def _run_credit_stress_test(self, config: StressTestConfig, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """运行信用压力测试"""
        positions = portfolio_data.get('positions', {})
        
        stressed_positions = {}
        portfolio_value_stressed = 0
        positions_impact = {}
        credit_losses = {}
        
        for symbol, position in positions.items():
            original_value = position.get('market_value', 0)
            
            # 计算信用损失
            credit_rating = self._get_credit_rating(symbol)
            credit_shock = config.shocks.get('credit_spread', 0.04)
            
            # 信用损失与信用评级相关
            default_probability = self._get_default_probability(credit_rating, credit_shock)
            credit_loss = default_probability * 0.4  # 假设违约损失率40%
            
            # 应用市场冲击和信用损失
            market_shock = config.shocks.get('equity_market', -0.3)
            total_shock = market_shock - credit_loss
            
            stressed_value = original_value * (1 + total_shock)
            
            stressed_positions[symbol] = stressed_value
            portfolio_value_stressed += stressed_value
            positions_impact[symbol] = total_shock
            credit_losses[symbol] = credit_loss
        
        var_stressed = portfolio_value_stressed * 0.15  # 信用危机下VaR很高
        max_drawdown = abs(min(positions_impact.values())) if positions_impact else 0
        
        return {
            'portfolio_value_stressed': portfolio_value_stressed,
            'var_stressed': var_stressed,
            'max_drawdown': max_drawdown,
            'positions_impact': positions_impact,
            'risk_metrics': {
                'credit_losses': credit_losses,
                'avg_credit_loss': np.mean(list(credit_losses.values())) if credit_losses else 0,
                'credit_spread_shock': config.shocks.get('credit_spread', 0)
            }
        }
    
    def _calculate_portfolio_value(self, portfolio_data: Dict[str, Any]) -> float:
        """计算组合价值"""
        positions = portfolio_data.get('positions', {})
        prices = portfolio_data.get('prices', {})
        
        total_value = 0
        for symbol, position in positions.items():
            quantity = position.get('quantity', 0)
            price = prices.get(symbol, 0)
            total_value += quantity * price
        
        return total_value
    
    def _calculate_position_shock(self, symbol: str, shocks: Dict[str, float]) -> float:
        """计算仓位冲击"""
        # 简化的冲击计算
        base_shock = shocks.get('equity_market', 0)
        
        # 根据股票特征调整冲击
        if 'tech' in symbol.lower():
            base_shock *= 1.5  # 科技股冲击更大
        elif 'utility' in symbol.lower():
            base_shock *= 0.7  # 公用事业股冲击较小
        
        return base_shock
    
    async def _calculate_stressed_var(self, stressed_positions: Dict, config: StressTestConfig) -> float:
        """计算压力测试下的VaR"""
        total_value = sum(pos.get('value', 0) for pos in stressed_positions.values())
        
        # 压力测试下的VaR通常更高
        volatility_multiplier = config.shocks.get('volatility', 1.0)
        base_var_rate = 0.05  # 基础VaR率
        
        stressed_var = total_value * base_var_rate * volatility_multiplier
        
        return stressed_var
    
    def _get_historical_shocks(self, scenario: StressScenario) -> Dict[str, float]:
        """获取历史冲击数据"""
        # 简化的历史冲击数据
        if scenario == StressScenario.MARKET_CRASH:
            return {
                'AAPL': -0.45,
                'MSFT': -0.38,
                'GOOGL': -0.42,
                'AMZN': -0.35,
                'TSLA': -0.55
            }
        elif scenario == StressScenario.INTEREST_RATE_SHOCK:
            return {
                'AAPL': -0.15,
                'MSFT': -0.12,
                'GOOGL': -0.18,
                'AMZN': -0.20,
                'TSLA': -0.25
            }
        else:
            return {}
    
    def _calculate_factor_exposures(self, positions: Dict) -> Dict[str, Dict[str, float]]:
        """计算因子敞口"""
        exposures = {}
        
        for symbol in positions.keys():
            # 简化的因子敞口
            exposures[symbol] = {
                'equity_market': 1.0,
                'interest_rate': -0.5,
                'volatility': 0.8,
                'liquidity': 0.3
            }
        
        return exposures
    
    def _get_liquidity_score(self, symbol: str) -> float:
        """获取流动性评分"""
        # 简化的流动性评分
        if symbol in ['AAPL', 'MSFT', 'GOOGL']:
            return 0.9  # 高流动性
        elif symbol in ['AMZN', 'TSLA']:
            return 0.8  # 中等流动性
        else:
            return 0.6  # 低流动性
    
    def _get_credit_rating(self, symbol: str) -> str:
        """获取信用评级"""
        # 简化的信用评级
        if symbol in ['AAPL', 'MSFT']:
            return 'AAA'
        elif symbol in ['GOOGL', 'AMZN']:
            return 'AA'
        else:
            return 'A'
    
    def _get_default_probability(self, credit_rating: str, credit_shock: float) -> float:
        """获取违约概率"""
        # 简化的违约概率计算
        base_probabilities = {
            'AAA': 0.001,
            'AA': 0.002,
            'A': 0.005,
            'BBB': 0.010,
            'BB': 0.050,
            'B': 0.100
        }
        
        base_prob = base_probabilities.get(credit_rating, 0.010)
        
        # 信用冲击增加违约概率
        stressed_prob = base_prob * (1 + credit_shock * 10)
        
        return min(stressed_prob, 0.5)  # 最大违约概率50%
    
    def _evaluate_test_result(self, config: StressTestConfig, relative_loss: float) -> bool:
        """评估测试结果"""
        # 定义通过标准
        pass_thresholds = {
            StressScenario.MARKET_CRASH: 0.30,        # 市场崩盘最大损失30%
            StressScenario.INTEREST_RATE_SHOCK: 0.20,  # 利率冲击最大损失20%
            StressScenario.CURRENCY_CRISIS: 0.25,     # 汇率危机最大损失25%
            StressScenario.LIQUIDITY_CRISIS: 0.35,    # 流动性危机最大损失35%
            StressScenario.CREDIT_CRISIS: 0.40,       # 信用危机最大损失40%
            StressScenario.VOLATILITY_SPIKE: 0.15     # 波动率冲击最大损失15%
        }
        
        threshold = pass_thresholds.get(config.scenario, 0.25)
        
        return abs(relative_loss) <= threshold
    
    def _generate_recommendations(self, config: StressTestConfig, result: Dict[str, Any], relative_loss: float) -> List[str]:
        """生成建议"""
        recommendations = []
        
        if abs(relative_loss) > 0.25:
            recommendations.append("Consider reducing overall portfolio risk exposure")
        
        if abs(relative_loss) > 0.15:
            recommendations.append("Implement additional hedging strategies")
        
        # 根据不同情景类型给出具体建议
        if config.scenario == StressScenario.MARKET_CRASH:
            recommendations.append("Increase allocation to defensive assets")
            recommendations.append("Consider put options for downside protection")
        
        elif config.scenario == StressScenario.INTEREST_RATE_SHOCK:
            recommendations.append("Reduce duration risk in bond portfolio")
            recommendations.append("Consider interest rate hedging instruments")
        
        elif config.scenario == StressScenario.LIQUIDITY_CRISIS:
            recommendations.append("Maintain higher cash reserves")
            recommendations.append("Reduce positions in illiquid assets")
        
        elif config.scenario == StressScenario.CREDIT_CRISIS:
            recommendations.append("Upgrade credit quality of bond holdings")
            recommendations.append("Reduce exposure to high-yield securities")
        
        # 根据具体损失情况给出建议
        positions_impact = result.get('positions_impact', {})
        if positions_impact:
            worst_positions = sorted(positions_impact.items(), key=lambda x: x[1])[:3]
            for symbol, impact in worst_positions:
                if impact < -0.3:
                    recommendations.append(f"Consider reducing exposure to {symbol}")
        
        return recommendations
    
    async def run_comprehensive_stress_test(self, portfolio_data: Dict[str, Any]) -> Dict[str, StressTestResult]:
        """运行全面压力测试"""
        results = {}
        
        # 并行运行多个压力测试
        test_tasks = []
        for test_name in self.test_configs.keys():
            task = asyncio.create_task(self.run_stress_test(test_name, portfolio_data))
            test_tasks.append((test_name, task))
        
        # 等待所有测试完成
        for test_name, task in test_tasks:
            try:
                result = await task
                results[test_name] = result
            except Exception as e:
                self.logger.error(f"Error in stress test {test_name}: {e}")
        
        return results
    
    def get_test_results(self) -> Dict[str, StressTestResult]:
        """获取所有测试结果"""
        return self.test_results.copy()
    
    def get_test_result(self, test_id: str) -> Optional[StressTestResult]:
        """获取特定测试结果"""
        return self.test_results.get(test_id)
    
    def get_test_summary(self) -> Dict[str, Any]:
        """获取测试摘要"""
        if not self.test_results:
            return {}
        
        results = list(self.test_results.values())
        
        # 统计通过率
        passed_count = sum(1 for r in results if r.passed)
        total_count = len(results)
        pass_rate = passed_count / total_count if total_count > 0 else 0
        
        # 最大损失
        max_loss = max(abs(r.relative_loss) for r in results) if results else 0
        
        # 平均损失
        avg_loss = sum(abs(r.relative_loss) for r in results) / len(results) if results else 0
        
        # 最差情景
        worst_scenario = max(results, key=lambda r: abs(r.relative_loss)) if results else None
        
        return {
            'total_tests': total_count,
            'passed_tests': passed_count,
            'pass_rate': pass_rate,
            'max_loss': max_loss,
            'avg_loss': avg_loss,
            'worst_scenario': worst_scenario.config.name if worst_scenario else None,
            'worst_scenario_loss': worst_scenario.relative_loss if worst_scenario else 0,
            'last_test_time': max(r.timestamp for r in results) if results else None
        }
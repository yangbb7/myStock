#!/usr/bin/env python3
"""
风险管理系统演示

这个脚本演示了综合风险管理系统的主要功能，包括：
1. 实时风险监控
2. 压力测试
3. 蒙特卡洛模拟
4. 期权风险管理
5. 信用风险评估
6. 流动性风险管理
7. 监管合规检查
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from myQuant.core.risk.integrated_risk_manager import IntegratedRiskManager
    from myQuant.core.risk.options_risk_manager import OptionContract, OptionPosition, OptionType
    from myQuant.core.risk.credit_risk_assessment import CreditEntity, CreditExposure, CreditRating
    from myQuant.core.risk.liquidity_risk_manager import LiquidAsset, FundingSource, LiquidityTier
    from myQuant.core.risk.monte_carlo import AssetParameters, SimulationConfig, SimulationType
    from myQuant.core.risk.regulatory_compliance import RegulatoryFramework
except ImportError as e:
    logger.error(f"Failed to import risk management modules: {e}")
    logger.info("Please ensure the myQuant package is properly installed")
    exit(1)

class RiskManagementDemo:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 系统配置
        self.config = {
            'real_time_monitor': {
                'monitor_interval': 1,
                'risk_limits': {
                    'max_var_1d': 0.05,
                    'max_drawdown': 0.20,
                    'max_leverage': 3.0,
                    'max_concentration': 0.15,
                    'max_volatility': 0.25
                }
            },
            'stress_testing': {
                'max_workers': 2
            },
            'monte_carlo': {
                'max_workers': 2,
                'chunk_size': 1000
            },
            'options_risk': {
                'max_delta_exposure': 10000,
                'max_gamma_exposure': 1000,
                'max_vega_exposure': 5000
            },
            'credit_risk': {
                'max_workers': 2
            },
            'liquidity_risk': {
                'min_lcr': 1.0,
                'min_nsfr': 1.0,
                'min_survival_days': 30
            },
            'regulatory_compliance': {
                'max_workers': 2
            }
        }
        
        # 初始化风险管理系统
        self.risk_manager = IntegratedRiskManager(self.config)
        
    async def run_demo(self):
        """运行完整的风险管理系统演示"""
        try:
            self.logger.info("=== 风险管理系统演示开始 ===")
            
            # 1. 启动系统
            await self.start_system()
            
            # 2. 添加测试数据
            await self.setup_test_data()
            
            # 3. 演示各个模块功能
            await self.demo_real_time_monitoring()
            await self.demo_stress_testing()
            await self.demo_monte_carlo_simulation()
            await self.demo_options_risk_management()
            await self.demo_credit_risk_assessment()
            await self.demo_liquidity_risk_management()
            await self.demo_regulatory_compliance()
            
            # 4. 生成综合报告
            await self.demo_integrated_reporting()
            
            # 5. 关闭系统
            await self.stop_system()
            
            self.logger.info("=== 风险管理系统演示完成 ===")
            
        except Exception as e:
            self.logger.error(f"演示过程中发生错误: {e}")
            raise
    
    async def start_system(self):
        """启动风险管理系统"""
        self.logger.info("启动风险管理系统...")
        await self.risk_manager.start()
        
        # 等待系统稳定
        await asyncio.sleep(2)
        
        self.logger.info("风险管理系统启动成功")
    
    async def stop_system(self):
        """停止风险管理系统"""
        self.logger.info("停止风险管理系统...")
        await self.risk_manager.stop()
        self.logger.info("风险管理系统已停止")
    
    async def setup_test_data(self):
        """设置测试数据"""
        self.logger.info("设置测试数据...")
        
        # 添加期权合约
        await self.setup_options_data()
        
        # 添加信用实体
        await self.setup_credit_data()
        
        # 添加流动性资产
        await self.setup_liquidity_data()
        
        self.logger.info("测试数据设置完成")
    
    async def setup_options_data(self):
        """设置期权测试数据"""
        try:
            # 创建期权合约
            option_contract = OptionContract(
                symbol="AAPL240315C00150000",
                underlying_symbol="AAPL",
                option_type=OptionType.CALL,
                strike_price=150.0,
                expiration_date=datetime.now() + timedelta(days=30),
                quantity=100,
                market_price=5.50,
                bid_price=5.45,
                ask_price=5.55,
                implied_volatility=0.25,
                volume=1000,
                open_interest=5000
            )
            
            # 创建期权持仓
            option_position = OptionPosition(
                contract=option_contract,
                quantity=10,
                entry_price=5.00,
                current_price=5.50,
                unrealized_pnl=500.0,
                delta_exposure=0.0,
                gamma_exposure=0.0,
                theta_exposure=0.0,
                vega_exposure=0.0,
                rho_exposure=0.0,
                days_to_expiration=30,
                position_value=5500.0
            )
            
            # 添加到期权风险管理器
            self.risk_manager.options_risk.add_position(option_position)
            
            # 更新市场数据
            self.risk_manager.options_risk.update_market_data("AAPL", {
                'price': 152.0,
                'bid_price': 151.95,
                'ask_price': 152.05,
                'volume': 50000000,
                'volatility': 0.22
            })
            
        except Exception as e:
            self.logger.error(f"设置期权数据时发生错误: {e}")
    
    async def setup_credit_data(self):
        """设置信用测试数据"""
        try:
            # 创建信用实体
            credit_entity = CreditEntity(
                entity_id="AAPL_CORP",
                name="Apple Inc.",
                entity_type="corporate",
                sector="technology",
                country="US",
                currency="USD",
                rating_sp=CreditRating.AA,
                rating_internal=CreditRating.AA,
                market_cap=2500000000000,
                total_debt=100000000000,
                ebitda=120000000000,
                cash_and_equivalents=50000000000
            )
            
            # 添加信用实体
            await self.risk_manager.credit_risk.add_credit_entity(credit_entity)
            
            # 创建信用敞口
            credit_exposure = CreditExposure(
                exposure_id="AAPL_BOND_2024",
                entity=credit_entity,
                exposure_type="bond",
                notional_amount=10000000,
                current_exposure=10000000,
                potential_future_exposure=12000000,
                recovery_rate=0.6,
                maturity_date=datetime.now() + timedelta(days=365),
                coupon_rate=0.035,
                seniority="senior_unsecured"
            )
            
            # 添加信用敞口
            await self.risk_manager.credit_risk.add_credit_exposure(credit_exposure)
            
            # 更新市场数据
            self.risk_manager.credit_risk.update_market_data("AAPL_CORP", {
                'credit_spread': 0.025,
                'equity_volatility': 0.25,
                'market_cap': 2500000000000
            })
            
        except Exception as e:
            self.logger.error(f"设置信用数据时发生错误: {e}")
    
    async def setup_liquidity_data(self):
        """设置流动性测试数据"""
        try:
            # 创建流动性资产
            liquid_asset = LiquidAsset(
                asset_id="US_TREASURY_10Y",
                symbol="UST10Y",
                asset_type="government_bond",
                tier=LiquidityTier.TIER_2A,
                market_value=50000000,
                haircut=0.05,
                liquidation_time_hours=1,
                concentration_limit=0.20,
                operational_requirements=0.02,
                central_bank_eligible=True,
                encumbered=False,
                stressed_value=47500000
            )
            
            # 添加流动性资产
            await self.risk_manager.liquidity_risk.add_liquid_asset(liquid_asset)
            
            # 创建资金来源
            funding_source = FundingSource(
                source_id="RETAIL_DEPOSITS",
                source_type="deposits",
                amount=100000000,
                maturity_date=datetime.now() + timedelta(days=90),
                cost_of_funding=0.015,
                stability_factor=0.95,
                concentration_risk=0.05,
                stressed_runoff_rate=0.10,
                available_amount=100000000,
                committed=True
            )
            
            # 添加资金来源
            await self.risk_manager.liquidity_risk.add_funding_source(funding_source)
            
            # 更新市场数据
            self.risk_manager.liquidity_risk.update_market_data("UST10Y", {
                'bid_price': 99.95,
                'ask_price': 100.05,
                'volume': 1000000,
                'market_depth': 500000,
                'turnover_ratio': 0.15
            })
            
        except Exception as e:
            self.logger.error(f"设置流动性数据时发生错误: {e}")
    
    async def demo_real_time_monitoring(self):
        """演示实时风险监控"""
        self.logger.info("=== 实时风险监控演示 ===")
        
        try:
            # 模拟更新市场数据
            portfolio_data = {
                'positions': {
                    'AAPL': {'quantity': 1000, 'price': 152.0, 'market_value': 152000},
                    'GOOGL': {'quantity': 500, 'price': 2800.0, 'market_value': 1400000},
                    'MSFT': {'quantity': 800, 'price': 380.0, 'market_value': 304000}
                },
                'total_value': 1856000,
                'cash': 144000,
                'leverage': 1.5
            }
            
            # 更新组合数据
            await self.risk_manager.real_time_monitor.update_portfolio_data(portfolio_data)
            
            # 等待计算完成
            await asyncio.sleep(3)
            
            # 获取仪表板数据
            dashboard_data = self.risk_manager.real_time_monitor.get_dashboard_data()
            
            self.logger.info("实时风险监控结果:")
            self.logger.info(f"  - 组合价值: ${dashboard_data.get('portfolio_value', 0):,.2f}")
            self.logger.info(f"  - 1日VaR: ${dashboard_data.get('var_1d', 0):,.2f}")
            self.logger.info(f"  - 最大回撤: {dashboard_data.get('max_drawdown', 0):.2%}")
            self.logger.info(f"  - 夏普比率: {dashboard_data.get('sharpe_ratio', 0):.2f}")
            self.logger.info(f"  - 活跃警报: {dashboard_data.get('active_alerts', 0)}")
            
        except Exception as e:
            self.logger.error(f"实时风险监控演示失败: {e}")
    
    async def demo_stress_testing(self):
        """演示压力测试"""
        self.logger.info("=== 压力测试演示 ===")
        
        try:
            # 准备组合数据
            portfolio_data = {
                'positions': {
                    'AAPL': {'quantity': 1000, 'market_value': 152000},
                    'GOOGL': {'quantity': 500, 'market_value': 1400000},
                    'MSFT': {'quantity': 800, 'market_value': 304000}
                },
                'prices': {
                    'AAPL': 152.0,
                    'GOOGL': 2800.0,
                    'MSFT': 380.0
                }
            }
            
            # 运行市场崩盘压力测试
            stress_result = await self.risk_manager.stress_testing.run_stress_test(
                'market_crash', portfolio_data
            )
            
            if stress_result:
                self.logger.info("市场崩盘压力测试结果:")
                self.logger.info(f"  - 基准组合价值: ${stress_result.portfolio_value_base:,.2f}")
                self.logger.info(f"  - 压力测试价值: ${stress_result.portfolio_value_stressed:,.2f}")
                self.logger.info(f"  - 绝对损失: ${stress_result.absolute_loss:,.2f}")
                self.logger.info(f"  - 相对损失: {stress_result.relative_loss:.2%}")
                self.logger.info(f"  - 测试通过: {'是' if stress_result.passed else '否'}")
                
                if stress_result.recommendations:
                    self.logger.info("  - 建议措施:")
                    for rec in stress_result.recommendations:
                        self.logger.info(f"    • {rec}")
            
        except Exception as e:
            self.logger.error(f"压力测试演示失败: {e}")
    
    async def demo_monte_carlo_simulation(self):
        """演示蒙特卡洛模拟"""
        self.logger.info("=== 蒙特卡洛模拟演示 ===")
        
        try:
            # 创建资产参数
            assets = [
                AssetParameters(
                    symbol="AAPL",
                    mu=0.12,
                    sigma=0.25,
                    initial_price=152.0,
                    weight=0.3
                ),
                AssetParameters(
                    symbol="GOOGL",
                    mu=0.15,
                    sigma=0.30,
                    initial_price=2800.0,
                    weight=0.4
                ),
                AssetParameters(
                    symbol="MSFT",
                    mu=0.10,
                    sigma=0.22,
                    initial_price=380.0,
                    weight=0.3
                )
            ]
            
            # 创建模拟配置
            simulation_config = SimulationConfig(
                simulation_type=SimulationType.GEOMETRIC_BROWNIAN_MOTION,
                num_simulations=10000,
                time_horizon_days=252,
                time_steps=252,
                confidence_levels=[0.95, 0.99],
                random_seed=42
            )
            
            # 运行模拟
            result = await self.risk_manager.monte_carlo.run_portfolio_simulation(
                assets, simulation_config
            )
            
            if result:
                self.logger.info("蒙特卡洛模拟结果:")
                self.logger.info(f"  - 预期收益: ${result.expected_return:,.2f}")
                self.logger.info(f"  - 波动率: {result.volatility:.2%}")
                self.logger.info(f"  - 最大回撤: {result.max_drawdown:.2%}")
                self.logger.info(f"  - 夏普比率: {result.sharpe_ratio:.2f}")
                self.logger.info(f"  - VaR (95%): ${result.var_estimates.get(0.95, 0):,.2f}")
                self.logger.info(f"  - CVaR (95%): ${result.cvar_estimates.get(0.95, 0):,.2f}")
                self.logger.info(f"  - 执行时间: {result.execution_time_seconds:.2f}秒")
                self.logger.info(f"  - 收敛达成: {'是' if result.convergence_achieved else '否'}")
            
        except Exception as e:
            self.logger.error(f"蒙特卡洛模拟演示失败: {e}")
    
    async def demo_options_risk_management(self):
        """演示期权风险管理"""
        self.logger.info("=== 期权风险管理演示 ===")
        
        try:
            # 计算期权组合风险指标
            risk_metrics = await self.risk_manager.options_risk.calculate_portfolio_risk_metrics()
            
            if risk_metrics:
                self.logger.info("期权组合风险指标:")
                self.logger.info(f"  - 组合价值: ${risk_metrics.portfolio_value:,.2f}")
                self.logger.info(f"  - 总Delta: {risk_metrics.total_delta:.2f}")
                self.logger.info(f"  - 总Gamma: {risk_metrics.total_gamma:.2f}")
                self.logger.info(f"  - 总Theta: ${risk_metrics.total_theta:.2f}")
                self.logger.info(f"  - 总Vega: {risk_metrics.total_vega:.2f}")
                self.logger.info(f"  - Pin风险: ${risk_metrics.pin_risk:,.2f}")
                self.logger.info(f"  - 获利概率: {risk_metrics.profit_probability:.2%}")
                self.logger.info(f"  - Kelly准则: {risk_metrics.kelly_criterion:.2%}")
                
                # 检查风险限制
                violations = await self.risk_manager.options_risk.check_risk_limits()
                if violations:
                    self.logger.info("  - 风险限制违规:")
                    for violation in violations:
                        self.logger.info(f"    • {violation['type']}: {violation['current']:.2f} > {violation['limit']:.2f}")
                
                # 生成对冲建议
                recommendations = await self.risk_manager.options_risk.generate_hedge_recommendations()
                if recommendations:
                    self.logger.info("  - 对冲建议:")
                    for rec in recommendations:
                        self.logger.info(f"    • {rec['type']}: {rec['action']} {rec['quantity']:.0f} {rec['instrument']}")
            
        except Exception as e:
            self.logger.error(f"期权风险管理演示失败: {e}")
    
    async def demo_credit_risk_assessment(self):
        """演示信用风险评估"""
        self.logger.info("=== 信用风险评估演示 ===")
        
        try:
            # 计算组合信用风险指标
            credit_metrics = await self.risk_manager.credit_risk.calculate_portfolio_metrics()
            
            if credit_metrics:
                self.logger.info("信用风险评估结果:")
                self.logger.info(f"  - 总敞口: ${credit_metrics.total_exposure:,.2f}")
                self.logger.info(f"  - 预期损失: ${credit_metrics.total_expected_loss:,.2f}")
                self.logger.info(f"  - 意外损失: ${credit_metrics.total_unexpected_loss:,.2f}")
                self.logger.info(f"  - 组合违约率: {credit_metrics.portfolio_default_rate:.2%}")
                self.logger.info(f"  - 集中度风险: {credit_metrics.concentration_risk:.2%}")
                self.logger.info(f"  - 信用VaR (99%): ${credit_metrics.var_99:,.2f}")
                self.logger.info(f"  - 经济资本: ${credit_metrics.economic_capital:,.2f}")
                
                # 行业和国家分布
                self.logger.info("  - 行业分布:")
                for sector, ratio in credit_metrics.sector_concentration.items():
                    self.logger.info(f"    • {sector}: {ratio:.1%}")
                
                self.logger.info("  - 国家分布:")
                for country, ratio in credit_metrics.country_concentration.items():
                    self.logger.info(f"    • {country}: {ratio:.1%}")
            
        except Exception as e:
            self.logger.error(f"信用风险评估演示失败: {e}")
    
    async def demo_liquidity_risk_management(self):
        """演示流动性风险管理"""
        self.logger.info("=== 流动性风险管理演示 ===")
        
        try:
            # 计算流动性风险指标
            liquidity_metrics = await self.risk_manager.liquidity_risk.calculate_liquidity_risk_metrics()
            
            if liquidity_metrics:
                self.logger.info("流动性风险管理结果:")
                self.logger.info(f"  - 流动性覆盖率 (LCR): {liquidity_metrics.liquidity_coverage_ratio:.2f}")
                self.logger.info(f"  - 净稳定资金比率 (NSFR): {liquidity_metrics.net_stable_funding_ratio:.2f}")
                self.logger.info(f"  - 生存期: {liquidity_metrics.survival_period_days:.0f}天")
                self.logger.info(f"  - 集中度风险: {liquidity_metrics.concentration_risk:.2%}")
                self.logger.info(f"  - 资金成本: {liquidity_metrics.funding_cost:.2%}")
                self.logger.info(f"  - 流动性缓冲: {liquidity_metrics.liquidity_buffer:.2%}")
                
                # 流动性缺口
                self.logger.info("  - 流动性缺口:")
                for period, gap in liquidity_metrics.liquidity_gap.items():
                    self.logger.info(f"    • {period}: ${gap:,.2f}")
                
                # 压力测试结果
                self.logger.info("  - 压力测试结果:")
                for scenario, result in liquidity_metrics.stress_test_results.items():
                    self.logger.info(f"    • {scenario}: LCR {result:.2f}")
                
                # 预警指标
                if liquidity_metrics.early_warning_indicators:
                    self.logger.info("  - 预警指标:")
                    for warning in liquidity_metrics.early_warning_indicators:
                        self.logger.info(f"    • {warning}")
            
        except Exception as e:
            self.logger.error(f"流动性风险管理演示失败: {e}")
    
    async def demo_regulatory_compliance(self):
        """演示监管合规检查"""
        self.logger.info("=== 监管合规检查演示 ===")
        
        try:
            # 运行Basel III合规检查
            compliance_checks = await self.risk_manager.regulatory_compliance.run_compliance_check_batch(
                RegulatoryFramework.BASEL_III
            )
            
            if compliance_checks:
                self.logger.info("Basel III合规检查结果:")
                
                compliant_count = len([c for c in compliance_checks if c.status.value == 'compliant'])
                warning_count = len([c for c in compliance_checks if c.status.value == 'warning'])
                breach_count = len([c for c in compliance_checks if c.status.value == 'breach'])
                
                self.logger.info(f"  - 总规则数: {len(compliance_checks)}")
                self.logger.info(f"  - 合规: {compliant_count}")
                self.logger.info(f"  - 警告: {warning_count}")
                self.logger.info(f"  - 违规: {breach_count}")
                
                # 显示具体检查结果
                for check in compliance_checks:
                    status_emoji = "✓" if check.status.value == 'compliant' else "⚠️" if check.status.value == 'warning' else "❌"
                    self.logger.info(f"  {status_emoji} {check.rule.rule_name}: {check.current_value:.4f} ({check.status.value})")
            
            # 生成合规报告
            compliance_report = await self.risk_manager.regulatory_compliance.generate_compliance_report(
                RegulatoryFramework.BASEL_III
            )
            
            if compliance_report:
                self.logger.info("合规报告摘要:")
                self.logger.info(f"  - 总体状态: {compliance_report.overall_status.value}")
                self.logger.info(f"  - 合规比率: {compliance_report.compliance_ratio:.1%}")
                self.logger.info(f"  - 风险评分: {compliance_report.risk_score:.2f}")
                
                if compliance_report.recommendations:
                    self.logger.info("  - 建议措施:")
                    for rec in compliance_report.recommendations:
                        self.logger.info(f"    • {rec}")
            
        except Exception as e:
            self.logger.error(f"监管合规检查演示失败: {e}")
    
    async def demo_integrated_reporting(self):
        """演示综合报告生成"""
        self.logger.info("=== 综合风险报告演示 ===")
        
        try:
            # 生成综合风险报告
            integrated_report = await self.risk_manager.generate_integrated_risk_report()
            
            if integrated_report:
                self.logger.info("综合风险报告:")
                self.logger.info(f"  - 报告ID: {integrated_report.report_id}")
                self.logger.info(f"  - 时间戳: {integrated_report.timestamp}")
                self.logger.info(f"  - 综合风险评分: {integrated_report.overall_risk_score:.2f}")
                self.logger.info(f"  - 系统状态: {integrated_report.system_status.value}")
                self.logger.info(f"  - 市场风险评分: {integrated_report.market_risk_score:.2f}")
                self.logger.info(f"  - 信用风险评分: {integrated_report.credit_risk_score:.2f}")
                self.logger.info(f"  - 流动性风险评分: {integrated_report.liquidity_risk_score:.2f}")
                self.logger.info(f"  - 合规评分: {integrated_report.compliance_score:.2f}")
                
                # 关键警报
                if integrated_report.key_alerts:
                    self.logger.info("  - 关键警报:")
                    for alert in integrated_report.key_alerts:
                        self.logger.info(f"    • {alert['type']}: {alert.get('message', alert.get('rule_name', 'N/A'))}")
                
                # 建议措施
                if integrated_report.recommendations:
                    self.logger.info("  - 建议措施:")
                    for rec in integrated_report.recommendations:
                        self.logger.info(f"    • {rec}")
            
            # 获取系统仪表板
            dashboard = self.risk_manager.get_system_dashboard()
            
            self.logger.info("系统仪表板:")
            self.logger.info(f"  - 系统状态: {dashboard.get('system_status', 'unknown')}")
            self.logger.info(f"  - 运行状态: {'运行中' if dashboard.get('is_running', False) else '已停止'}")
            self.logger.info(f"  - 最后更新: {dashboard.get('last_update', 'N/A')}")
            
            # 组件状态
            components = dashboard.get('components_status', {})
            self.logger.info("  - 组件状态:")
            for component, status in components.items():
                status_emoji = "✓" if status else "❌"
                self.logger.info(f"    {status_emoji} {component}: {'正常' if status else '异常'}")
            
        except Exception as e:
            self.logger.error(f"综合报告演示失败: {e}")

async def main():
    """主函数"""
    logger.info("启动风险管理系统演示...")
    
    try:
        # 创建演示实例
        demo = RiskManagementDemo()
        
        # 运行演示
        await demo.run_demo()
        
        logger.info("风险管理系统演示成功完成!")
        
    except Exception as e:
        logger.error(f"演示失败: {e}")
        raise

if __name__ == "__main__":
    # 运行演示
    asyncio.run(main())
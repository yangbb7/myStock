# -*- coding: utf-8 -*-
"""
基本面因子计算引擎 - 提供财务报表和基本面数据的因子计算
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


class FundamentalCategory(Enum):
    """基本面因子分类"""
    PROFITABILITY = "profitability"     # 盈利能力
    GROWTH = "growth"                   # 成长性
    LEVERAGE = "leverage"               # 杠杆/债务
    LIQUIDITY = "liquidity"             # 流动性
    EFFICIENCY = "efficiency"           # 运营效率
    VALUATION = "valuation"             # 估值
    QUALITY = "quality"                 # 质量
    DIVIDEND = "dividend"               # 分红
    CASH_FLOW = "cash_flow"             # 现金流
    BALANCE_SHEET = "balance_sheet"     # 资产负债表


@dataclass
class FundamentalData:
    """基本面数据结构"""
    # 损益表数据
    revenue: float = 0.0                # 营业收入
    operating_income: float = 0.0       # 营业利润
    net_income: float = 0.0             # 净利润
    gross_profit: float = 0.0           # 毛利润
    ebitda: float = 0.0                 # 息税折旧摊销前利润
    ebit: float = 0.0                   # 息税前利润
    
    # 资产负债表数据
    total_assets: float = 0.0           # 总资产
    total_liabilities: float = 0.0      # 总负债
    total_equity: float = 0.0           # 总股本
    current_assets: float = 0.0         # 流动资产
    current_liabilities: float = 0.0    # 流动负债
    cash: float = 0.0                   # 现金及现金等价物
    inventory: float = 0.0              # 存货
    accounts_receivable: float = 0.0    # 应收账款
    
    # 现金流量表数据
    operating_cash_flow: float = 0.0    # 经营现金流
    investing_cash_flow: float = 0.0    # 投资现金流
    financing_cash_flow: float = 0.0    # 筹资现金流
    capex: float = 0.0                  # 资本支出
    
    # 市场数据
    market_cap: float = 0.0             # 市值
    shares_outstanding: float = 0.0     # 流通股本
    price: float = 0.0                  # 股价
    
    # 分红数据
    dividends_paid: float = 0.0         # 分红支出
    
    # 时间戳
    report_date: datetime = None        # 报告日期
    
    def __post_init__(self):
        """数据验证和计算派生字段"""
        if self.report_date is None:
            self.report_date = datetime.now()
        
        # 自动计算一些派生字段
        if self.gross_profit == 0.0 and self.revenue > 0:
            # 如果没有毛利润，尝试从其他数据推算
            self.gross_profit = self.revenue * 0.3  # 假设毛利率30%
        
        if self.total_equity == 0.0 and self.total_assets > 0 and self.total_liabilities > 0:
            self.total_equity = self.total_assets - self.total_liabilities


class FundamentalFactorEngine:
    """基本面因子计算引擎"""
    
    def __init__(self):
        """初始化基本面因子引擎"""
        self.factor_registry = {}
        self.data_cache = {}
        self._register_all_factors()
        self.logger = logging.getLogger(__name__)
    
    def _register_all_factors(self):
        """注册所有基本面因子"""
        # 盈利能力因子
        self._register_profitability_factors()
        # 成长性因子
        self._register_growth_factors()
        # 杠杆因子
        self._register_leverage_factors()
        # 流动性因子
        self._register_liquidity_factors()
        # 运营效率因子
        self._register_efficiency_factors()
        # 估值因子
        self._register_valuation_factors()
        # 质量因子
        self._register_quality_factors()
        # 分红因子
        self._register_dividend_factors()
        # 现金流因子
        self._register_cash_flow_factors()
        # 资产负债表因子
        self._register_balance_sheet_factors()
    
    def _register_profitability_factors(self):
        """注册盈利能力因子"""
        profitability_factors = {
            # 基础盈利能力
            'ROE': self.roe,                    # 净资产收益率
            'ROA': self.roa,                    # 总资产收益率
            'ROIC': self.roic,                  # 投入资本回报率
            'GROSS_MARGIN': self.gross_margin,   # 毛利率
            'OPERATING_MARGIN': self.operating_margin,  # 营业利润率
            'NET_MARGIN': self.net_margin,      # 净利率
            'EBITDA_MARGIN': self.ebitda_margin, # EBITDA利润率
            
            # 高级盈利能力
            'ROE_TTM': self.roe_ttm,           # 滚动净资产收益率
            'ROA_TTM': self.roa_ttm,           # 滚动总资产收益率
            'DUPONT_ROE': self.dupont_roe,     # 杜邦ROE分解
            'EARNINGS_POWER': self.earnings_power,  # 盈利能力
            'PROFIT_SUSTAINABILITY': self.profit_sustainability,  # 盈利持续性
        }
        
        for name, func in profitability_factors.items():
            self.factor_registry[name] = {
                'function': func,
                'category': FundamentalCategory.PROFITABILITY,
                'description': f'Profitability factor: {name}'
            }
    
    def _register_growth_factors(self):
        """注册成长性因子"""
        growth_factors = {
            # 收入增长
            'REVENUE_GROWTH': self.revenue_growth,      # 营收增长率
            'REVENUE_GROWTH_3Y': self.revenue_growth_3y, # 3年营收增长率
            'REVENUE_GROWTH_5Y': self.revenue_growth_5y, # 5年营收增长率
            'REVENUE_CAGR': self.revenue_cagr,          # 营收复合增长率
            
            # 利润增长
            'EARNINGS_GROWTH': self.earnings_growth,    # 利润增长率
            'EARNINGS_GROWTH_3Y': self.earnings_growth_3y, # 3年利润增长率
            'EARNINGS_GROWTH_5Y': self.earnings_growth_5y, # 5年利润增长率
            'EARNINGS_CAGR': self.earnings_cagr,        # 利润复合增长率
            
            # 其他增长
            'ASSET_GROWTH': self.asset_growth,          # 资产增长率
            'EQUITY_GROWTH': self.equity_growth,        # 股本增长率
            'BOOK_VALUE_GROWTH': self.book_value_growth, # 账面价值增长率
            'SUSTAINABLE_GROWTH': self.sustainable_growth, # 可持续增长率
            
            # 成长质量
            'GROWTH_QUALITY': self.growth_quality,      # 成长质量
            'GROWTH_STABILITY': self.growth_stability,  # 成长稳定性
        }
        
        for name, func in growth_factors.items():
            self.factor_registry[name] = {
                'function': func,
                'category': FundamentalCategory.GROWTH,
                'description': f'Growth factor: {name}'
            }
    
    def _register_leverage_factors(self):
        """注册杠杆因子"""
        leverage_factors = {
            # 债务比率
            'DEBT_TO_EQUITY': self.debt_to_equity,      # 债务股本比
            'DEBT_TO_ASSETS': self.debt_to_assets,      # 债务资产比
            'EQUITY_RATIO': self.equity_ratio,          # 股本比率
            'DEBT_RATIO': self.debt_ratio,              # 债务比率
            
            # 偿债能力
            'INTEREST_COVERAGE': self.interest_coverage, # 利息覆盖率
            'DEBT_SERVICE_COVERAGE': self.debt_service_coverage, # 债务偿还覆盖率
            'TIMES_INTEREST_EARNED': self.times_interest_earned, # 利息保障倍数
            
            # 杠杆变化
            'LEVERAGE_CHANGE': self.leverage_change,    # 杠杆变化
            'DEBT_GROWTH': self.debt_growth,            # 债务增长率
            'FINANCIAL_LEVERAGE': self.financial_leverage, # 财务杠杆
        }
        
        for name, func in leverage_factors.items():
            self.factor_registry[name] = {
                'function': func,
                'category': FundamentalCategory.LEVERAGE,
                'description': f'Leverage factor: {name}'
            }
    
    def _register_liquidity_factors(self):
        """注册流动性因子"""
        liquidity_factors = {
            # 流动性比率
            'CURRENT_RATIO': self.current_ratio,        # 流动比率
            'QUICK_RATIO': self.quick_ratio,            # 速动比率
            'CASH_RATIO': self.cash_ratio,              # 现金比率
            'ACID_TEST': self.acid_test,                # 酸性测试比率
            
            # 现金相关
            'CASH_TO_ASSETS': self.cash_to_assets,      # 现金资产比
            'CASH_TO_DEBT': self.cash_to_debt,          # 现金债务比
            'CASH_CONVERSION_CYCLE': self.cash_conversion_cycle, # 现金转换周期
            
            # 营运资本
            'WORKING_CAPITAL': self.working_capital,    # 营运资本
            'WORKING_CAPITAL_RATIO': self.working_capital_ratio, # 营运资本比率
            'NET_WORKING_CAPITAL': self.net_working_capital, # 净营运资本
        }
        
        for name, func in liquidity_factors.items():
            self.factor_registry[name] = {
                'function': func,
                'category': FundamentalCategory.LIQUIDITY,
                'description': f'Liquidity factor: {name}'
            }
    
    def _register_efficiency_factors(self):
        """注册运营效率因子"""
        efficiency_factors = {
            # 周转率
            'ASSET_TURNOVER': self.asset_turnover,      # 资产周转率
            'INVENTORY_TURNOVER': self.inventory_turnover, # 存货周转率
            'RECEIVABLES_TURNOVER': self.receivables_turnover, # 应收账款周转率
            'PAYABLES_TURNOVER': self.payables_turnover, # 应付账款周转率
            
            # 周转天数
            'DAYS_SALES_OUTSTANDING': self.days_sales_outstanding, # 应收账款周转天数
            'DAYS_INVENTORY_OUTSTANDING': self.days_inventory_outstanding, # 存货周转天数
            'DAYS_PAYABLE_OUTSTANDING': self.days_payable_outstanding, # 应付账款周转天数
            
            # 效率指标
            'CAPITAL_EFFICIENCY': self.capital_efficiency, # 资本效率
            'OPERATIONAL_EFFICIENCY': self.operational_efficiency, # 运营效率
            'MANAGEMENT_EFFICIENCY': self.management_efficiency, # 管理效率
        }
        
        for name, func in efficiency_factors.items():
            self.factor_registry[name] = {
                'function': func,
                'category': FundamentalCategory.EFFICIENCY,
                'description': f'Efficiency factor: {name}'
            }
    
    def _register_valuation_factors(self):
        """注册估值因子"""
        valuation_factors = {
            # 市盈率类
            'PE_RATIO': self.pe_ratio,                  # 市盈率
            'PE_TTM': self.pe_ttm,                      # 滚动市盈率
            'PEG_RATIO': self.peg_ratio,                # PEG比率
            'FORWARD_PE': self.forward_pe,              # 前瞻市盈率
            
            # 市净率类
            'PB_RATIO': self.pb_ratio,                  # 市净率
            'PRICE_TO_TANGIBLE_BOOK': self.price_to_tangible_book, # 有形资产市净率
            
            # 市销率类
            'PS_RATIO': self.ps_ratio,                  # 市销率
            'PS_TTM': self.ps_ttm,                      # 滚动市销率
            
            # 现金流估值
            'PCF_RATIO': self.pcf_ratio,                # 市现率
            'EV_EBITDA': self.ev_ebitda,                # 企业价值倍数
            'EV_SALES': self.ev_sales,                  # 企业价值销售比
            'EV_EBIT': self.ev_ebit,                    # 企业价值息税前利润比
            
            # 高级估值
            'PRICE_TO_BOOK_GROWTH': self.price_to_book_growth, # 市净率增长比
            'EARNINGS_YIELD': self.earnings_yield,      # 盈利收益率
            'FCFE_YIELD': self.fcfe_yield,              # 股权自由现金流收益率
            'DIVIDEND_YIELD': self.dividend_yield,      # 股息收益率
        }
        
        for name, func in valuation_factors.items():
            self.factor_registry[name] = {
                'function': func,
                'category': FundamentalCategory.VALUATION,
                'description': f'Valuation factor: {name}'
            }
    
    def _register_quality_factors(self):
        """注册质量因子"""
        quality_factors = {
            # 盈利质量
            'EARNINGS_QUALITY': self.earnings_quality,  # 盈利质量
            'ACCRUALS_RATIO': self.accruals_ratio,      # 应计项目比率
            'CASH_EARNINGS_RATIO': self.cash_earnings_ratio, # 现金盈利比率
            
            # 财务健康
            'ALTMAN_Z_SCORE': self.altman_z_score,      # 阿尔曼Z得分
            'PIOTROSKI_F_SCORE': self.piotroski_f_score, # 皮奥特罗斯基F得分
            'BENEISH_M_SCORE': self.beneish_m_score,    # 贝尼什M得分
            
            # 资产质量
            'ASSET_QUALITY': self.asset_quality,        # 资产质量
            'BALANCE_SHEET_QUALITY': self.balance_sheet_quality, # 资产负债表质量
            'GOODWILL_TO_ASSETS': self.goodwill_to_assets, # 商誉资产比
            
            # 管理质量
            'MANAGEMENT_QUALITY': self.management_quality, # 管理质量
            'CORPORATE_GOVERNANCE': self.corporate_governance, # 公司治理
        }
        
        for name, func in quality_factors.items():
            self.factor_registry[name] = {
                'function': func,
                'category': FundamentalCategory.QUALITY,
                'description': f'Quality factor: {name}'
            }
    
    def _register_dividend_factors(self):
        """注册分红因子"""
        dividend_factors = {
            # 分红率
            'DIVIDEND_YIELD': self.dividend_yield,      # 股息收益率
            'DIVIDEND_PAYOUT_RATIO': self.dividend_payout_ratio, # 分红率
            'DIVIDEND_COVERAGE': self.dividend_coverage, # 分红覆盖率
            
            # 分红增长
            'DIVIDEND_GROWTH': self.dividend_growth,    # 分红增长率
            'DIVIDEND_GROWTH_3Y': self.dividend_growth_3y, # 3年分红增长率
            'DIVIDEND_GROWTH_5Y': self.dividend_growth_5y, # 5年分红增长率
            'DIVIDEND_CAGR': self.dividend_cagr,        # 分红复合增长率
            
            # 分红稳定性
            'DIVIDEND_STABILITY': self.dividend_stability, # 分红稳定性
            'DIVIDEND_ARISTOCRAT': self.dividend_aristocrat, # 分红贵族
        }
        
        for name, func in dividend_factors.items():
            self.factor_registry[name] = {
                'function': func,
                'category': FundamentalCategory.DIVIDEND,
                'description': f'Dividend factor: {name}'
            }
    
    def _register_cash_flow_factors(self):
        """注册现金流因子"""
        cash_flow_factors = {
            # 现金流比率
            'OPERATING_CASH_FLOW_RATIO': self.operating_cash_flow_ratio, # 经营现金流比率
            'FCF_RATIO': self.fcf_ratio,                # 自由现金流比率
            'CASH_FLOW_TO_DEBT': self.cash_flow_to_debt, # 现金流债务比
            
            # 现金流收益率
            'CASH_FLOW_YIELD': self.cash_flow_yield,    # 现金流收益率
            'FCF_YIELD': self.fcf_yield,                # 自由现金流收益率
            'CASH_RETURN_ON_ASSETS': self.cash_return_on_assets, # 现金资产回报率
            
            # 现金流增长
            'CASH_FLOW_GROWTH': self.cash_flow_growth,  # 现金流增长率
            'FCF_GROWTH': self.fcf_growth,              # 自由现金流增长率
            'CASH_FLOW_STABILITY': self.cash_flow_stability, # 现金流稳定性
            
            # 现金流质量
            'CASH_FLOW_QUALITY': self.cash_flow_quality, # 现金流质量
            'CAPEX_TO_SALES': self.capex_to_sales,      # 资本支出销售比
            'CAPEX_TO_DEPRECIATION': self.capex_to_depreciation, # 资本支出折旧比
        }
        
        for name, func in cash_flow_factors.items():
            self.factor_registry[name] = {
                'function': func,
                'category': FundamentalCategory.CASH_FLOW,
                'description': f'Cash flow factor: {name}'
            }
    
    def _register_balance_sheet_factors(self):
        """注册资产负债表因子"""
        balance_sheet_factors = {
            # 资产结构
            'ASSET_STRUCTURE': self.asset_structure,    # 资产结构
            'CURRENT_ASSETS_RATIO': self.current_assets_ratio, # 流动资产比率
            'FIXED_ASSETS_RATIO': self.fixed_assets_ratio, # 固定资产比率
            'INTANGIBLE_ASSETS_RATIO': self.intangible_assets_ratio, # 无形资产比率
            
            # 负债结构
            'LIABILITY_STRUCTURE': self.liability_structure, # 负债结构
            'CURRENT_LIABILITIES_RATIO': self.current_liabilities_ratio, # 流动负债比率
            'LONG_TERM_DEBT_RATIO': self.long_term_debt_ratio, # 长期负债比率
            
            # 资本结构
            'CAPITAL_STRUCTURE': self.capital_structure, # 资本结构
            'EQUITY_MULTIPLIER': self.equity_multiplier, # 股本乘数
            'CAPITALIZATION_RATIO': self.capitalization_ratio, # 资本化比率
        }
        
        for name, func in balance_sheet_factors.items():
            self.factor_registry[name] = {
                'function': func,
                'category': FundamentalCategory.BALANCE_SHEET,
                'description': f'Balance sheet factor: {name}'
            }
    
    def calculate_factor(self, factor_name: str, data: Union[FundamentalData, List[FundamentalData]]) -> float:
        """
        计算单个因子
        
        Args:
            factor_name: 因子名称
            data: 基本面数据
            
        Returns:
            float: 因子值
        """
        if factor_name not in self.factor_registry:
            raise ValueError(f"Factor {factor_name} not found")
        
        factor_info = self.factor_registry[factor_name]
        factor_func = factor_info['function']
        
        try:
            return factor_func(data)
        except Exception as e:
            self.logger.error(f"Error calculating {factor_name}: {e}")
            return np.nan
    
    def calculate_factors(self, factor_names: List[str], data: Union[FundamentalData, List[FundamentalData]]) -> Dict[str, float]:
        """
        批量计算因子
        
        Args:
            factor_names: 因子名称列表
            data: 基本面数据
            
        Returns:
            Dict[str, float]: 因子值字典
        """
        results = {}
        for factor_name in factor_names:
            results[factor_name] = self.calculate_factor(factor_name, data)
        return results
    
    def get_available_factors(self, category: Optional[FundamentalCategory] = None) -> List[str]:
        """
        获取可用因子列表
        
        Args:
            category: 因子分类
            
        Returns:
            List[str]: 因子名称列表
        """
        if category is None:
            return list(self.factor_registry.keys())
        else:
            return [name for name, info in self.factor_registry.items() 
                   if info['category'] == category]
    
    # =============================================================================
    # 盈利能力因子实现
    # =============================================================================
    
    def roe(self, data: FundamentalData) -> float:
        """净资产收益率 (ROE)"""
        if data.total_equity <= 0:
            return np.nan
        return data.net_income / data.total_equity
    
    def roa(self, data: FundamentalData) -> float:
        """总资产收益率 (ROA)"""
        if data.total_assets <= 0:
            return np.nan
        return data.net_income / data.total_assets
    
    def roic(self, data: FundamentalData) -> float:
        """投入资本回报率 (ROIC)"""
        invested_capital = data.total_assets - data.current_liabilities
        if invested_capital <= 0:
            return np.nan
        return data.operating_income / invested_capital
    
    def gross_margin(self, data: FundamentalData) -> float:
        """毛利率"""
        if data.revenue <= 0:
            return np.nan
        return data.gross_profit / data.revenue
    
    def operating_margin(self, data: FundamentalData) -> float:
        """营业利润率"""
        if data.revenue <= 0:
            return np.nan
        return data.operating_income / data.revenue
    
    def net_margin(self, data: FundamentalData) -> float:
        """净利率"""
        if data.revenue <= 0:
            return np.nan
        return data.net_income / data.revenue
    
    def ebitda_margin(self, data: FundamentalData) -> float:
        """EBITDA利润率"""
        if data.revenue <= 0:
            return np.nan
        return data.ebitda / data.revenue
    
    def roe_ttm(self, data: List[FundamentalData]) -> float:
        """滚动净资产收益率 (TTM)"""
        if len(data) < 4:
            return np.nan
        
        # 计算过去4个季度的净利润总和
        ttm_net_income = sum(d.net_income for d in data[-4:])
        # 使用最新的总股本
        latest_equity = data[-1].total_equity
        
        if latest_equity <= 0:
            return np.nan
        return ttm_net_income / latest_equity
    
    def roa_ttm(self, data: List[FundamentalData]) -> float:
        """滚动总资产收益率 (TTM)"""
        if len(data) < 4:
            return np.nan
        
        ttm_net_income = sum(d.net_income for d in data[-4:])
        latest_assets = data[-1].total_assets
        
        if latest_assets <= 0:
            return np.nan
        return ttm_net_income / latest_assets
    
    def dupont_roe(self, data: FundamentalData) -> Dict[str, float]:
        """杜邦ROE分解"""
        net_margin = self.net_margin(data)
        asset_turnover = self.asset_turnover(data)
        equity_multiplier = self.equity_multiplier(data)
        
        return {
            'net_margin': net_margin,
            'asset_turnover': asset_turnover,
            'equity_multiplier': equity_multiplier,
            'roe': net_margin * asset_turnover * equity_multiplier
        }
    
    def earnings_power(self, data: FundamentalData) -> float:
        """盈利能力综合指标"""
        roe = self.roe(data)
        roa = self.roa(data)
        roic = self.roic(data)
        
        # 综合盈利能力评分
        if any(np.isnan(x) for x in [roe, roa, roic]):
            return np.nan
        
        return (roe + roa + roic) / 3
    
    def profit_sustainability(self, data: List[FundamentalData]) -> float:
        """盈利持续性"""
        if len(data) < 4:
            return np.nan
        
        # 计算过去4个季度的ROE
        roe_values = [self.roe(d) for d in data[-4:]]
        roe_values = [x for x in roe_values if not np.isnan(x)]
        
        if len(roe_values) < 4:
            return np.nan
        
        # 计算ROE的变异系数（标准差/均值）
        roe_mean = np.mean(roe_values)
        roe_std = np.std(roe_values)
        
        if roe_mean == 0:
            return np.nan
        
        # 持续性 = 1 - 变异系数
        return 1 - (roe_std / abs(roe_mean))
    
    # =============================================================================
    # 成长性因子实现
    # =============================================================================
    
    def revenue_growth(self, data: List[FundamentalData]) -> float:
        """营收增长率"""
        if len(data) < 2:
            return np.nan
        
        current_revenue = data[-1].revenue
        previous_revenue = data[-2].revenue
        
        if previous_revenue <= 0:
            return np.nan
        
        return (current_revenue - previous_revenue) / previous_revenue
    
    def revenue_growth_3y(self, data: List[FundamentalData]) -> float:
        """3年营收增长率"""
        if len(data) < 12:  # 12个季度
            return np.nan
        
        current_revenue = data[-1].revenue
        three_years_ago_revenue = data[-12].revenue
        
        if three_years_ago_revenue <= 0:
            return np.nan
        
        return (current_revenue - three_years_ago_revenue) / three_years_ago_revenue
    
    def revenue_growth_5y(self, data: List[FundamentalData]) -> float:
        """5年营收增长率"""
        if len(data) < 20:  # 20个季度
            return np.nan
        
        current_revenue = data[-1].revenue
        five_years_ago_revenue = data[-20].revenue
        
        if five_years_ago_revenue <= 0:
            return np.nan
        
        return (current_revenue - five_years_ago_revenue) / five_years_ago_revenue
    
    def revenue_cagr(self, data: List[FundamentalData], years: int = 5) -> float:
        """营收复合增长率"""
        periods = years * 4  # 假设是季度数据
        
        if len(data) < periods:
            return np.nan
        
        current_revenue = data[-1].revenue
        past_revenue = data[-periods].revenue
        
        if past_revenue <= 0:
            return np.nan
        
        return (current_revenue / past_revenue) ** (1 / years) - 1
    
    def earnings_growth(self, data: List[FundamentalData]) -> float:
        """利润增长率"""
        if len(data) < 2:
            return np.nan
        
        current_earnings = data[-1].net_income
        previous_earnings = data[-2].net_income
        
        if previous_earnings <= 0:
            return np.nan
        
        return (current_earnings - previous_earnings) / previous_earnings
    
    def earnings_growth_3y(self, data: List[FundamentalData]) -> float:
        """3年利润增长率"""
        if len(data) < 12:
            return np.nan
        
        current_earnings = data[-1].net_income
        three_years_ago_earnings = data[-12].net_income
        
        if three_years_ago_earnings <= 0:
            return np.nan
        
        return (current_earnings - three_years_ago_earnings) / three_years_ago_earnings
    
    def earnings_growth_5y(self, data: List[FundamentalData]) -> float:
        """5年利润增长率"""
        if len(data) < 20:
            return np.nan
        
        current_earnings = data[-1].net_income
        five_years_ago_earnings = data[-20].net_income
        
        if five_years_ago_earnings <= 0:
            return np.nan
        
        return (current_earnings - five_years_ago_earnings) / five_years_ago_earnings
    
    def earnings_cagr(self, data: List[FundamentalData], years: int = 5) -> float:
        """利润复合增长率"""
        periods = years * 4
        
        if len(data) < periods:
            return np.nan
        
        current_earnings = data[-1].net_income
        past_earnings = data[-periods].net_income
        
        if past_earnings <= 0:
            return np.nan
        
        return (current_earnings / past_earnings) ** (1 / years) - 1
    
    def asset_growth(self, data: List[FundamentalData]) -> float:
        """资产增长率"""
        if len(data) < 2:
            return np.nan
        
        current_assets = data[-1].total_assets
        previous_assets = data[-2].total_assets
        
        if previous_assets <= 0:
            return np.nan
        
        return (current_assets - previous_assets) / previous_assets
    
    def equity_growth(self, data: List[FundamentalData]) -> float:
        """股本增长率"""
        if len(data) < 2:
            return np.nan
        
        current_equity = data[-1].total_equity
        previous_equity = data[-2].total_equity
        
        if previous_equity <= 0:
            return np.nan
        
        return (current_equity - previous_equity) / previous_equity
    
    def book_value_growth(self, data: List[FundamentalData]) -> float:
        """账面价值增长率"""
        if len(data) < 2:
            return np.nan
        
        current_book_value = data[-1].total_equity / data[-1].shares_outstanding if data[-1].shares_outstanding > 0 else 0
        previous_book_value = data[-2].total_equity / data[-2].shares_outstanding if data[-2].shares_outstanding > 0 else 0
        
        if previous_book_value <= 0:
            return np.nan
        
        return (current_book_value - previous_book_value) / previous_book_value
    
    def sustainable_growth(self, data: FundamentalData) -> float:
        """可持续增长率"""
        roe = self.roe(data)
        payout_ratio = self.dividend_payout_ratio(data)
        
        if np.isnan(roe) or np.isnan(payout_ratio):
            return np.nan
        
        retention_ratio = 1 - payout_ratio
        return roe * retention_ratio
    
    def growth_quality(self, data: List[FundamentalData]) -> float:
        """成长质量"""
        if len(data) < 4:
            return np.nan
        
        # 计算收入增长和利润增长的一致性
        revenue_growth_rates = []
        earnings_growth_rates = []
        
        for i in range(1, min(len(data), 4)):
            revenue_gr = self.revenue_growth(data[i-1:i+1])
            earnings_gr = self.earnings_growth(data[i-1:i+1])
            
            if not np.isnan(revenue_gr) and not np.isnan(earnings_gr):
                revenue_growth_rates.append(revenue_gr)
                earnings_growth_rates.append(earnings_gr)
        
        if len(revenue_growth_rates) < 2:
            return np.nan
        
        # 计算相关性
        correlation = np.corrcoef(revenue_growth_rates, earnings_growth_rates)[0, 1]
        
        if np.isnan(correlation):
            return np.nan
        
        return correlation
    
    def growth_stability(self, data: List[FundamentalData]) -> float:
        """成长稳定性"""
        if len(data) < 4:
            return np.nan
        
        # 计算过去几个季度的增长率
        growth_rates = []
        for i in range(1, min(len(data), 4)):
            growth_rate = self.revenue_growth(data[i-1:i+1])
            if not np.isnan(growth_rate):
                growth_rates.append(growth_rate)
        
        if len(growth_rates) < 2:
            return np.nan
        
        # 计算增长率的变异系数
        mean_growth = np.mean(growth_rates)
        std_growth = np.std(growth_rates)
        
        if mean_growth == 0:
            return np.nan
        
        # 稳定性 = 1 - 变异系数
        return 1 - (std_growth / abs(mean_growth))
    
    # =============================================================================
    # 杠杆因子实现
    # =============================================================================
    
    def debt_to_equity(self, data: FundamentalData) -> float:
        """债务股本比"""
        if data.total_equity <= 0:
            return np.nan
        return data.total_liabilities / data.total_equity
    
    def debt_to_assets(self, data: FundamentalData) -> float:
        """债务资产比"""
        if data.total_assets <= 0:
            return np.nan
        return data.total_liabilities / data.total_assets
    
    def equity_ratio(self, data: FundamentalData) -> float:
        """股本比率"""
        if data.total_assets <= 0:
            return np.nan
        return data.total_equity / data.total_assets
    
    def debt_ratio(self, data: FundamentalData) -> float:
        """债务比率"""
        return self.debt_to_assets(data)
    
    def interest_coverage(self, data: FundamentalData) -> float:
        """利息覆盖率"""
        # 假设利息费用 = EBIT - 营业利润
        interest_expense = data.ebit - data.operating_income
        if interest_expense <= 0:
            return np.inf
        return data.ebit / interest_expense
    
    def debt_service_coverage(self, data: FundamentalData) -> float:
        """债务偿还覆盖率"""
        # 简化计算：现金流/总负债
        if data.total_liabilities <= 0:
            return np.inf
        return data.operating_cash_flow / data.total_liabilities
    
    def times_interest_earned(self, data: FundamentalData) -> float:
        """利息保障倍数"""
        return self.interest_coverage(data)
    
    def leverage_change(self, data: List[FundamentalData]) -> float:
        """杠杆变化"""
        if len(data) < 2:
            return np.nan
        
        current_leverage = self.debt_to_equity(data[-1])
        previous_leverage = self.debt_to_equity(data[-2])
        
        if np.isnan(current_leverage) or np.isnan(previous_leverage):
            return np.nan
        
        return current_leverage - previous_leverage
    
    def debt_growth(self, data: List[FundamentalData]) -> float:
        """债务增长率"""
        if len(data) < 2:
            return np.nan
        
        current_debt = data[-1].total_liabilities
        previous_debt = data[-2].total_liabilities
        
        if previous_debt <= 0:
            return np.nan
        
        return (current_debt - previous_debt) / previous_debt
    
    def financial_leverage(self, data: FundamentalData) -> float:
        """财务杠杆"""
        if data.total_equity <= 0:
            return np.nan
        return data.total_assets / data.total_equity
    
    # =============================================================================
    # 流动性因子实现
    # =============================================================================
    
    def current_ratio(self, data: FundamentalData) -> float:
        """流动比率"""
        if data.current_liabilities <= 0:
            return np.inf
        return data.current_assets / data.current_liabilities
    
    def quick_ratio(self, data: FundamentalData) -> float:
        """速动比率"""
        if data.current_liabilities <= 0:
            return np.inf
        quick_assets = data.current_assets - data.inventory
        return quick_assets / data.current_liabilities
    
    def cash_ratio(self, data: FundamentalData) -> float:
        """现金比率"""
        if data.current_liabilities <= 0:
            return np.inf
        return data.cash / data.current_liabilities
    
    def acid_test(self, data: FundamentalData) -> float:
        """酸性测试比率"""
        return self.quick_ratio(data)
    
    def cash_to_assets(self, data: FundamentalData) -> float:
        """现金资产比"""
        if data.total_assets <= 0:
            return np.nan
        return data.cash / data.total_assets
    
    def cash_to_debt(self, data: FundamentalData) -> float:
        """现金债务比"""
        if data.total_liabilities <= 0:
            return np.inf
        return data.cash / data.total_liabilities
    
    def cash_conversion_cycle(self, data: FundamentalData) -> float:
        """现金转换周期"""
        dso = self.days_sales_outstanding(data)
        dio = self.days_inventory_outstanding(data)
        dpo = self.days_payable_outstanding(data)
        
        if any(np.isnan(x) for x in [dso, dio, dpo]):
            return np.nan
        
        return dso + dio - dpo
    
    def working_capital(self, data: FundamentalData) -> float:
        """营运资本"""
        return data.current_assets - data.current_liabilities
    
    def working_capital_ratio(self, data: FundamentalData) -> float:
        """营运资本比率"""
        if data.total_assets <= 0:
            return np.nan
        working_cap = self.working_capital(data)
        return working_cap / data.total_assets
    
    def net_working_capital(self, data: FundamentalData) -> float:
        """净营运资本"""
        return self.working_capital(data)
    
    # =============================================================================
    # 运营效率因子实现
    # =============================================================================
    
    def asset_turnover(self, data: FundamentalData) -> float:
        """资产周转率"""
        if data.total_assets <= 0:
            return np.nan
        return data.revenue / data.total_assets
    
    def inventory_turnover(self, data: FundamentalData) -> float:
        """存货周转率"""
        if data.inventory <= 0:
            return np.inf
        # 假设销售成本 = 营业收入 - 毛利润
        cost_of_goods_sold = data.revenue - data.gross_profit
        return cost_of_goods_sold / data.inventory
    
    def receivables_turnover(self, data: FundamentalData) -> float:
        """应收账款周转率"""
        if data.accounts_receivable <= 0:
            return np.inf
        return data.revenue / data.accounts_receivable
    
    def payables_turnover(self, data: FundamentalData) -> float:
        """应付账款周转率"""
        # 简化计算：假设应付账款 = 流动负债的一部分
        estimated_payables = data.current_liabilities * 0.5
        if estimated_payables <= 0:
            return np.inf
        cost_of_goods_sold = data.revenue - data.gross_profit
        return cost_of_goods_sold / estimated_payables
    
    def days_sales_outstanding(self, data: FundamentalData) -> float:
        """应收账款周转天数"""
        receivables_turnover = self.receivables_turnover(data)
        if receivables_turnover <= 0 or np.isinf(receivables_turnover):
            return np.nan
        return 365 / receivables_turnover
    
    def days_inventory_outstanding(self, data: FundamentalData) -> float:
        """存货周转天数"""
        inventory_turnover = self.inventory_turnover(data)
        if inventory_turnover <= 0 or np.isinf(inventory_turnover):
            return np.nan
        return 365 / inventory_turnover
    
    def days_payable_outstanding(self, data: FundamentalData) -> float:
        """应付账款周转天数"""
        payables_turnover = self.payables_turnover(data)
        if payables_turnover <= 0 or np.isinf(payables_turnover):
            return np.nan
        return 365 / payables_turnover
    
    def capital_efficiency(self, data: FundamentalData) -> float:
        """资本效率"""
        # 综合资产周转率和ROA
        asset_turnover = self.asset_turnover(data)
        roa = self.roa(data)
        
        if np.isnan(asset_turnover) or np.isnan(roa):
            return np.nan
        
        return (asset_turnover + roa) / 2
    
    def operational_efficiency(self, data: FundamentalData) -> float:
        """运营效率"""
        # 综合多个周转率
        asset_turnover = self.asset_turnover(data)
        inventory_turnover = self.inventory_turnover(data)
        receivables_turnover = self.receivables_turnover(data)
        
        # 标准化处理
        turnovers = [asset_turnover]
        if not np.isinf(inventory_turnover):
            turnovers.append(min(inventory_turnover, 100))  # 上限100
        if not np.isinf(receivables_turnover):
            turnovers.append(min(receivables_turnover, 100))  # 上限100
        
        valid_turnovers = [t for t in turnovers if not np.isnan(t)]
        if not valid_turnovers:
            return np.nan
        
        return np.mean(valid_turnovers)
    
    def management_efficiency(self, data: FundamentalData) -> float:
        """管理效率"""
        # 综合盈利能力和运营效率
        roe = self.roe(data)
        operational_eff = self.operational_efficiency(data)
        
        if np.isnan(roe) or np.isnan(operational_eff):
            return np.nan
        
        return (roe + operational_eff) / 2
    
    # =============================================================================
    # 估值因子实现
    # =============================================================================
    
    def pe_ratio(self, data: FundamentalData) -> float:
        """市盈率"""
        if data.net_income <= 0:
            return np.nan
        eps = data.net_income / data.shares_outstanding if data.shares_outstanding > 0 else 0
        if eps <= 0:
            return np.nan
        return data.price / eps
    
    def pe_ttm(self, data: List[FundamentalData]) -> float:
        """滚动市盈率"""
        if len(data) < 4:
            return np.nan
        
        ttm_net_income = sum(d.net_income for d in data[-4:])
        latest_data = data[-1]
        
        if ttm_net_income <= 0 or latest_data.shares_outstanding <= 0:
            return np.nan
        
        ttm_eps = ttm_net_income / latest_data.shares_outstanding
        return latest_data.price / ttm_eps
    
    def peg_ratio(self, data: List[FundamentalData]) -> float:
        """PEG比率"""
        pe = self.pe_ratio(data[-1])
        earnings_growth = self.earnings_growth(data)
        
        if np.isnan(pe) or np.isnan(earnings_growth) or earnings_growth <= 0:
            return np.nan
        
        return pe / (earnings_growth * 100)
    
    def forward_pe(self, data: FundamentalData, forward_eps: float = None) -> float:
        """前瞻市盈率"""
        if forward_eps is None:
            # 简化：假设下一年EPS增长10%
            current_eps = data.net_income / data.shares_outstanding if data.shares_outstanding > 0 else 0
            forward_eps = current_eps * 1.1
        
        if forward_eps <= 0:
            return np.nan
        
        return data.price / forward_eps
    
    def pb_ratio(self, data: FundamentalData) -> float:
        """市净率"""
        if data.shares_outstanding <= 0 or data.total_equity <= 0:
            return np.nan
        
        book_value_per_share = data.total_equity / data.shares_outstanding
        return data.price / book_value_per_share
    
    def price_to_tangible_book(self, data: FundamentalData) -> float:
        """有形资产市净率"""
        # 简化：假设无形资产占总资产的10%
        tangible_assets = data.total_assets * 0.9
        tangible_equity = tangible_assets - data.total_liabilities
        
        if data.shares_outstanding <= 0 or tangible_equity <= 0:
            return np.nan
        
        tangible_book_value_per_share = tangible_equity / data.shares_outstanding
        return data.price / tangible_book_value_per_share
    
    def ps_ratio(self, data: FundamentalData) -> float:
        """市销率"""
        if data.shares_outstanding <= 0 or data.revenue <= 0:
            return np.nan
        
        sales_per_share = data.revenue / data.shares_outstanding
        return data.price / sales_per_share
    
    def ps_ttm(self, data: List[FundamentalData]) -> float:
        """滚动市销率"""
        if len(data) < 4:
            return np.nan
        
        ttm_revenue = sum(d.revenue for d in data[-4:])
        latest_data = data[-1]
        
        if ttm_revenue <= 0 or latest_data.shares_outstanding <= 0:
            return np.nan
        
        ttm_sales_per_share = ttm_revenue / latest_data.shares_outstanding
        return latest_data.price / ttm_sales_per_share
    
    def pcf_ratio(self, data: FundamentalData) -> float:
        """市现率"""
        if data.shares_outstanding <= 0 or data.operating_cash_flow <= 0:
            return np.nan
        
        cash_flow_per_share = data.operating_cash_flow / data.shares_outstanding
        return data.price / cash_flow_per_share
    
    def ev_ebitda(self, data: FundamentalData) -> float:
        """企业价值倍数"""
        if data.ebitda <= 0:
            return np.nan
        
        enterprise_value = data.market_cap + data.total_liabilities - data.cash
        return enterprise_value / data.ebitda
    
    def ev_sales(self, data: FundamentalData) -> float:
        """企业价值销售比"""
        if data.revenue <= 0:
            return np.nan
        
        enterprise_value = data.market_cap + data.total_liabilities - data.cash
        return enterprise_value / data.revenue
    
    def ev_ebit(self, data: FundamentalData) -> float:
        """企业价值息税前利润比"""
        if data.ebit <= 0:
            return np.nan
        
        enterprise_value = data.market_cap + data.total_liabilities - data.cash
        return enterprise_value / data.ebit
    
    def price_to_book_growth(self, data: List[FundamentalData]) -> float:
        """市净率增长比"""
        if len(data) < 2:
            return np.nan
        
        pb_ratio = self.pb_ratio(data[-1])
        book_value_growth = self.book_value_growth(data)
        
        if np.isnan(pb_ratio) or np.isnan(book_value_growth) or book_value_growth <= 0:
            return np.nan
        
        return pb_ratio / (book_value_growth * 100)
    
    def earnings_yield(self, data: FundamentalData) -> float:
        """盈利收益率"""
        pe = self.pe_ratio(data)
        if np.isnan(pe) or pe <= 0:
            return np.nan
        return 1 / pe
    
    def fcfe_yield(self, data: FundamentalData) -> float:
        """股权自由现金流收益率"""
        # 股权自由现金流 = 经营现金流 - 资本支出
        fcfe = data.operating_cash_flow - data.capex
        
        if data.market_cap <= 0 or fcfe <= 0:
            return np.nan
        
        return fcfe / data.market_cap
    
    def dividend_yield(self, data: FundamentalData) -> float:
        """股息收益率"""
        if data.shares_outstanding <= 0 or data.price <= 0:
            return np.nan
        
        dividend_per_share = data.dividends_paid / data.shares_outstanding
        return dividend_per_share / data.price
    
    # =============================================================================
    # 质量因子实现
    # =============================================================================
    
    def earnings_quality(self, data: FundamentalData) -> float:
        """盈利质量"""
        # 现金流与净利润的比率
        if data.net_income <= 0:
            return np.nan
        
        return data.operating_cash_flow / data.net_income
    
    def accruals_ratio(self, data: FundamentalData) -> float:
        """应计项目比率"""
        # 应计项目 = 净利润 - 经营现金流
        accruals = data.net_income - data.operating_cash_flow
        
        if data.total_assets <= 0:
            return np.nan
        
        return accruals / data.total_assets
    
    def cash_earnings_ratio(self, data: FundamentalData) -> float:
        """现金盈利比率"""
        return self.earnings_quality(data)
    
    def altman_z_score(self, data: FundamentalData) -> float:
        """阿尔曼Z得分"""
        if data.total_assets <= 0:
            return np.nan
        
        # 工作资本/总资产
        working_capital = self.working_capital(data)
        a = working_capital / data.total_assets
        
        # 留存收益/总资产 (简化为总股本/总资产)
        b = data.total_equity / data.total_assets
        
        # 息税前利润/总资产
        c = data.ebit / data.total_assets
        
        # 市值/总负债
        d = data.market_cap / data.total_liabilities if data.total_liabilities > 0 else 0
        
        # 营业收入/总资产
        e = data.revenue / data.total_assets
        
        # Z = 1.2A + 1.4B + 3.3C + 0.6D + 1.0E
        z_score = 1.2 * a + 1.4 * b + 3.3 * c + 0.6 * d + 1.0 * e
        
        return z_score
    
    def piotroski_f_score(self, data: List[FundamentalData]) -> float:
        """皮奥特罗斯基F得分"""
        if len(data) < 2:
            return np.nan
        
        current_data = data[-1]
        previous_data = data[-2]
        
        score = 0
        
        # 1. 正的净利润
        if current_data.net_income > 0:
            score += 1
        
        # 2. 正的经营现金流
        if current_data.operating_cash_flow > 0:
            score += 1
        
        # 3. ROA增长
        current_roa = self.roa(current_data)
        previous_roa = self.roa(previous_data)
        if not np.isnan(current_roa) and not np.isnan(previous_roa) and current_roa > previous_roa:
            score += 1
        
        # 4. 经营现金流 > 净利润
        if current_data.operating_cash_flow > current_data.net_income:
            score += 1
        
        # 5. 杠杆比率下降
        current_leverage = self.debt_to_assets(current_data)
        previous_leverage = self.debt_to_assets(previous_data)
        if not np.isnan(current_leverage) and not np.isnan(previous_leverage) and current_leverage < previous_leverage:
            score += 1
        
        # 6. 流动比率改善
        current_ratio = self.current_ratio(current_data)
        previous_ratio = self.current_ratio(previous_data)
        if not np.isnan(current_ratio) and not np.isnan(previous_ratio) and current_ratio > previous_ratio:
            score += 1
        
        # 7. 没有新增股份发行 (简化：股本没有增长)
        if current_data.shares_outstanding <= previous_data.shares_outstanding:
            score += 1
        
        # 8. 毛利率改善
        current_margin = self.gross_margin(current_data)
        previous_margin = self.gross_margin(previous_data)
        if not np.isnan(current_margin) and not np.isnan(previous_margin) and current_margin > previous_margin:
            score += 1
        
        # 9. 资产周转率改善
        current_turnover = self.asset_turnover(current_data)
        previous_turnover = self.asset_turnover(previous_data)
        if not np.isnan(current_turnover) and not np.isnan(previous_turnover) and current_turnover > previous_turnover:
            score += 1
        
        return score
    
    def beneish_m_score(self, data: List[FundamentalData]) -> float:
        """贝尼什M得分"""
        if len(data) < 2:
            return np.nan
        
        current_data = data[-1]
        previous_data = data[-2]
        
        # 简化的M得分计算
        try:
            # 应收账款增长vs营收增长
            receivables_growth = (current_data.accounts_receivable - previous_data.accounts_receivable) / previous_data.accounts_receivable if previous_data.accounts_receivable > 0 else 0
            revenue_growth = (current_data.revenue - previous_data.revenue) / previous_data.revenue if previous_data.revenue > 0 else 0
            
            dsri = (1 + receivables_growth) / (1 + revenue_growth) if revenue_growth > -1 else 1
            
            # 毛利率指数
            current_margin = self.gross_margin(current_data)
            previous_margin = self.gross_margin(previous_data)
            gmi = previous_margin / current_margin if current_margin > 0 else 1
            
            # 资产质量指数
            current_aqi = 1 - (current_data.current_assets + current_data.cash) / current_data.total_assets
            previous_aqi = 1 - (previous_data.current_assets + previous_data.cash) / previous_data.total_assets
            aqi = current_aqi / previous_aqi if previous_aqi > 0 else 1
            
            # 简化的M得分
            m_score = -4.84 + 0.92 * dsri + 0.58 * gmi + 0.404 * aqi
            
            return m_score
            
        except:
            return np.nan
    
    def asset_quality(self, data: FundamentalData) -> float:
        """资产质量"""
        # 现金和流动资产占总资产的比例
        if data.total_assets <= 0:
            return np.nan
        
        quality_assets = data.cash + data.current_assets
        return quality_assets / data.total_assets
    
    def balance_sheet_quality(self, data: FundamentalData) -> float:
        """资产负债表质量"""
        # 综合多个质量指标
        asset_quality = self.asset_quality(data)
        debt_ratio = self.debt_to_assets(data)
        
        if np.isnan(asset_quality) or np.isnan(debt_ratio):
            return np.nan
        
        # 资产质量高、债务比例低 = 高质量
        return asset_quality - debt_ratio
    
    def goodwill_to_assets(self, data: FundamentalData) -> float:
        """商誉资产比"""
        # 简化：假设无形资产的50%是商誉
        intangible_assets = data.total_assets * 0.1  # 假设无形资产占10%
        goodwill = intangible_assets * 0.5
        
        if data.total_assets <= 0:
            return np.nan
        
        return goodwill / data.total_assets
    
    def management_quality(self, data: List[FundamentalData]) -> float:
        """管理质量"""
        # 综合多个管理效率指标
        if len(data) < 2:
            return np.nan
        
        current_data = data[-1]
        
        # ROE、资产周转率、利润增长率
        roe = self.roe(current_data)
        asset_turnover = self.asset_turnover(current_data)
        earnings_growth = self.earnings_growth(data)
        
        scores = []
        if not np.isnan(roe):
            scores.append(min(max(roe, 0), 1))  # 限制在0-1之间
        if not np.isnan(asset_turnover):
            scores.append(min(asset_turnover, 5) / 5)  # 标准化到0-1
        if not np.isnan(earnings_growth):
            scores.append(min(max(earnings_growth, -1), 1))  # 限制在-1到1之间
        
        if not scores:
            return np.nan
        
        return np.mean(scores)
    
    def corporate_governance(self, data: FundamentalData) -> float:
        """公司治理"""
        # 简化的公司治理评分
        # 基于财务透明度和稳定性
        
        # 现金流质量
        cash_flow_quality = self.cash_flow_quality(data)
        
        # 债务管理
        debt_ratio = self.debt_to_assets(data)
        debt_score = 1 - min(debt_ratio, 1) if not np.isnan(debt_ratio) else 0.5
        
        # 盈利稳定性（简化）
        profit_margin = self.net_margin(data)
        profit_score = min(max(profit_margin, 0), 0.5) * 2 if not np.isnan(profit_margin) else 0.5
        
        scores = [cash_flow_quality, debt_score, profit_score]
        valid_scores = [s for s in scores if not np.isnan(s)]
        
        if not valid_scores:
            return np.nan
        
        return np.mean(valid_scores)
    
    # =============================================================================
    # 分红因子实现
    # =============================================================================
    
    def dividend_payout_ratio(self, data: FundamentalData) -> float:
        """分红率"""
        if data.net_income <= 0:
            return np.nan
        
        return data.dividends_paid / data.net_income
    
    def dividend_coverage(self, data: FundamentalData) -> float:
        """分红覆盖率"""
        if data.dividends_paid <= 0:
            return np.inf
        
        return data.net_income / data.dividends_paid
    
    def dividend_growth(self, data: List[FundamentalData]) -> float:
        """分红增长率"""
        if len(data) < 2:
            return np.nan
        
        current_div = data[-1].dividends_paid
        previous_div = data[-2].dividends_paid
        
        if previous_div <= 0:
            return np.nan
        
        return (current_div - previous_div) / previous_div
    
    def dividend_growth_3y(self, data: List[FundamentalData]) -> float:
        """3年分红增长率"""
        if len(data) < 12:
            return np.nan
        
        current_div = data[-1].dividends_paid
        three_years_ago_div = data[-12].dividends_paid
        
        if three_years_ago_div <= 0:
            return np.nan
        
        return (current_div - three_years_ago_div) / three_years_ago_div
    
    def dividend_growth_5y(self, data: List[FundamentalData]) -> float:
        """5年分红增长率"""
        if len(data) < 20:
            return np.nan
        
        current_div = data[-1].dividends_paid
        five_years_ago_div = data[-20].dividends_paid
        
        if five_years_ago_div <= 0:
            return np.nan
        
        return (current_div - five_years_ago_div) / five_years_ago_div
    
    def dividend_cagr(self, data: List[FundamentalData], years: int = 5) -> float:
        """分红复合增长率"""
        periods = years * 4
        
        if len(data) < periods:
            return np.nan
        
        current_div = data[-1].dividends_paid
        past_div = data[-periods].dividends_paid
        
        if past_div <= 0:
            return np.nan
        
        return (current_div / past_div) ** (1 / years) - 1
    
    def dividend_stability(self, data: List[FundamentalData]) -> float:
        """分红稳定性"""
        if len(data) < 4:
            return np.nan
        
        # 计算过去几个季度的分红
        dividends = [d.dividends_paid for d in data[-4:]]
        dividends = [d for d in dividends if d > 0]
        
        if len(dividends) < 2:
            return np.nan
        
        # 计算变异系数
        mean_div = np.mean(dividends)
        std_div = np.std(dividends)
        
        if mean_div == 0:
            return np.nan
        
        # 稳定性 = 1 - 变异系数
        return 1 - (std_div / mean_div)
    
    def dividend_aristocrat(self, data: List[FundamentalData]) -> float:
        """分红贵族"""
        # 连续分红增长的年数
        if len(data) < 4:
            return 0
        
        consecutive_growth = 0
        
        for i in range(1, len(data)):
            if data[i].dividends_paid > data[i-1].dividends_paid:
                consecutive_growth += 1
            else:
                break
        
        return consecutive_growth / 4  # 转换为年数
    
    # =============================================================================
    # 现金流因子实现
    # =============================================================================
    
    def operating_cash_flow_ratio(self, data: FundamentalData) -> float:
        """经营现金流比率"""
        if data.current_liabilities <= 0:
            return np.inf
        
        return data.operating_cash_flow / data.current_liabilities
    
    def fcf_ratio(self, data: FundamentalData) -> float:
        """自由现金流比率"""
        fcf = data.operating_cash_flow - data.capex
        
        if data.current_liabilities <= 0:
            return np.inf
        
        return fcf / data.current_liabilities
    
    def cash_flow_to_debt(self, data: FundamentalData) -> float:
        """现金流债务比"""
        if data.total_liabilities <= 0:
            return np.inf
        
        return data.operating_cash_flow / data.total_liabilities
    
    def cash_flow_yield(self, data: FundamentalData) -> float:
        """现金流收益率"""
        if data.market_cap <= 0:
            return np.nan
        
        return data.operating_cash_flow / data.market_cap
    
    def fcf_yield(self, data: FundamentalData) -> float:
        """自由现金流收益率"""
        fcf = data.operating_cash_flow - data.capex
        
        if data.market_cap <= 0:
            return np.nan
        
        return fcf / data.market_cap
    
    def cash_return_on_assets(self, data: FundamentalData) -> float:
        """现金资产回报率"""
        if data.total_assets <= 0:
            return np.nan
        
        return data.operating_cash_flow / data.total_assets
    
    def cash_flow_growth(self, data: List[FundamentalData]) -> float:
        """现金流增长率"""
        if len(data) < 2:
            return np.nan
        
        current_cf = data[-1].operating_cash_flow
        previous_cf = data[-2].operating_cash_flow
        
        if previous_cf <= 0:
            return np.nan
        
        return (current_cf - previous_cf) / previous_cf
    
    def fcf_growth(self, data: List[FundamentalData]) -> float:
        """自由现金流增长率"""
        if len(data) < 2:
            return np.nan
        
        current_fcf = data[-1].operating_cash_flow - data[-1].capex
        previous_fcf = data[-2].operating_cash_flow - data[-2].capex
        
        if previous_fcf <= 0:
            return np.nan
        
        return (current_fcf - previous_fcf) / previous_fcf
    
    def cash_flow_stability(self, data: List[FundamentalData]) -> float:
        """现金流稳定性"""
        if len(data) < 4:
            return np.nan
        
        # 计算过去几个季度的现金流
        cash_flows = [d.operating_cash_flow for d in data[-4:]]
        
        if not cash_flows:
            return np.nan
        
        # 计算变异系数
        mean_cf = np.mean(cash_flows)
        std_cf = np.std(cash_flows)
        
        if mean_cf == 0:
            return np.nan
        
        # 稳定性 = 1 - 变异系数
        return 1 - (std_cf / abs(mean_cf))
    
    def cash_flow_quality(self, data: FundamentalData) -> float:
        """现金流质量"""
        # 现金流质量 = 经营现金流 / (经营现金流 + 投资现金流 + 筹资现金流)
        total_cash_flow = abs(data.operating_cash_flow) + abs(data.investing_cash_flow) + abs(data.financing_cash_flow)
        
        if total_cash_flow <= 0:
            return np.nan
        
        return abs(data.operating_cash_flow) / total_cash_flow
    
    def capex_to_sales(self, data: FundamentalData) -> float:
        """资本支出销售比"""
        if data.revenue <= 0:
            return np.nan
        
        return data.capex / data.revenue
    
    def capex_to_depreciation(self, data: FundamentalData) -> float:
        """资本支出折旧比"""
        # 简化：假设折旧 = EBITDA - EBIT
        depreciation = data.ebitda - data.ebit
        
        if depreciation <= 0:
            return np.inf
        
        return data.capex / depreciation
    
    # =============================================================================
    # 资产负债表因子实现
    # =============================================================================
    
    def asset_structure(self, data: FundamentalData) -> Dict[str, float]:
        """资产结构"""
        if data.total_assets <= 0:
            return {'current_assets_ratio': np.nan, 'fixed_assets_ratio': np.nan}
        
        current_assets_ratio = data.current_assets / data.total_assets
        fixed_assets_ratio = (data.total_assets - data.current_assets) / data.total_assets
        
        return {
            'current_assets_ratio': current_assets_ratio,
            'fixed_assets_ratio': fixed_assets_ratio
        }
    
    def current_assets_ratio(self, data: FundamentalData) -> float:
        """流动资产比率"""
        if data.total_assets <= 0:
            return np.nan
        
        return data.current_assets / data.total_assets
    
    def fixed_assets_ratio(self, data: FundamentalData) -> float:
        """固定资产比率"""
        if data.total_assets <= 0:
            return np.nan
        
        fixed_assets = data.total_assets - data.current_assets
        return fixed_assets / data.total_assets
    
    def intangible_assets_ratio(self, data: FundamentalData) -> float:
        """无形资产比率"""
        # 简化：假设无形资产占总资产的10%
        intangible_assets = data.total_assets * 0.1
        
        if data.total_assets <= 0:
            return np.nan
        
        return intangible_assets / data.total_assets
    
    def liability_structure(self, data: FundamentalData) -> Dict[str, float]:
        """负债结构"""
        if data.total_liabilities <= 0:
            return {'current_liabilities_ratio': np.nan, 'long_term_debt_ratio': np.nan}
        
        current_liabilities_ratio = data.current_liabilities / data.total_liabilities
        long_term_debt_ratio = (data.total_liabilities - data.current_liabilities) / data.total_liabilities
        
        return {
            'current_liabilities_ratio': current_liabilities_ratio,
            'long_term_debt_ratio': long_term_debt_ratio
        }
    
    def current_liabilities_ratio(self, data: FundamentalData) -> float:
        """流动负债比率"""
        if data.total_liabilities <= 0:
            return np.nan
        
        return data.current_liabilities / data.total_liabilities
    
    def long_term_debt_ratio(self, data: FundamentalData) -> float:
        """长期负债比率"""
        if data.total_liabilities <= 0:
            return np.nan
        
        long_term_debt = data.total_liabilities - data.current_liabilities
        return long_term_debt / data.total_liabilities
    
    def capital_structure(self, data: FundamentalData) -> Dict[str, float]:
        """资本结构"""
        total_capital = data.total_assets
        
        if total_capital <= 0:
            return {'debt_ratio': np.nan, 'equity_ratio': np.nan}
        
        debt_ratio = data.total_liabilities / total_capital
        equity_ratio = data.total_equity / total_capital
        
        return {
            'debt_ratio': debt_ratio,
            'equity_ratio': equity_ratio
        }
    
    def equity_multiplier(self, data: FundamentalData) -> float:
        """股本乘数"""
        if data.total_equity <= 0:
            return np.nan
        
        return data.total_assets / data.total_equity
    
    def capitalization_ratio(self, data: FundamentalData) -> float:
        """资本化比率"""
        # 长期负债 / (长期负债 + 总股本)
        long_term_debt = data.total_liabilities - data.current_liabilities
        total_capitalization = long_term_debt + data.total_equity
        
        if total_capitalization <= 0:
            return np.nan
        
        return long_term_debt / total_capitalization
    
    def get_factor_description(self, factor_name: str) -> str:
        """获取因子描述"""
        if factor_name not in self.factor_registry:
            return f"Factor {factor_name} not found"
        
        return self.factor_registry[factor_name]['description']
    
    def get_factor_category(self, factor_name: str) -> FundamentalCategory:
        """获取因子分类"""
        if factor_name not in self.factor_registry:
            return None
        
        return self.factor_registry[factor_name]['category']
    
    def get_factors_summary(self) -> Dict[str, Any]:
        """获取因子库摘要"""
        summary = {
            'total_factors': len(self.factor_registry),
            'categories': {}
        }
        
        for category in FundamentalCategory:
            category_factors = self.get_available_factors(category)
            summary['categories'][category.value] = {
                'count': len(category_factors),
                'factors': category_factors
            }
        
        return summary
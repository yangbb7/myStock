import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from pathlib import Path
import json
import hashlib
import uuid
import re
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

class AMLRiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class TransactionType(Enum):
    CASH_DEPOSIT = "cash_deposit"
    CASH_WITHDRAWAL = "cash_withdrawal"
    WIRE_TRANSFER = "wire_transfer"
    STOCK_TRADE = "stock_trade"
    BOND_TRADE = "bond_trade"
    DERIVATIVE_TRADE = "derivative_trade"
    CRYPTOCURRENCY = "cryptocurrency"
    FOREIGN_EXCHANGE = "foreign_exchange"
    MONEY_MARKET = "money_market"
    INVESTMENT_FUND = "investment_fund"

class AlertType(Enum):
    SANCTIONS_MATCH = "sanctions_match"
    PEP_MATCH = "pep_match"
    ADVERSE_MEDIA = "adverse_media"
    UNUSUAL_ACTIVITY = "unusual_activity"
    STRUCTURING = "structuring"
    LARGE_AMOUNT = "large_amount"
    FREQUENT_TRANSACTIONS = "frequent_transactions"
    VELOCITY_ALERT = "velocity_alert"
    COUNTRY_RISK = "country_risk"
    PATTERN_MATCH = "pattern_match"

class AlertStatus(Enum):
    OPEN = "open"
    INVESTIGATING = "investigating"
    CLOSED_CLEARED = "closed_cleared"
    CLOSED_SUSPICIOUS = "closed_suspicious"
    ESCALATED = "escalated"
    FILED_SAR = "filed_sar"

class CustomerRiskRating(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    PROHIBITED = "prohibited"

@dataclass
class SanctionsList:
    """制裁名单"""
    list_id: str
    list_name: str
    list_type: str
    jurisdiction: str
    last_updated: datetime
    entries: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PEPList:
    """政治敏感人员名单"""
    list_id: str
    list_name: str
    jurisdiction: str
    last_updated: datetime
    entries: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CustomerProfile:
    """客户档案"""
    customer_id: str
    customer_name: str
    customer_type: str  # individual, corporate, trust
    date_of_birth: Optional[datetime]
    nationality: str
    country_of_residence: str
    occupation: str
    business_type: Optional[str]
    risk_rating: CustomerRiskRating
    kyc_status: str
    onboarding_date: datetime
    last_review_date: Optional[datetime]
    next_review_date: datetime
    expected_transaction_volume: float
    expected_transaction_frequency: int
    source_of_funds: str
    source_of_wealth: str
    beneficial_owners: List[Dict[str, Any]]
    risk_factors: List[str]
    sanctions_checked: bool
    pep_checked: bool
    adverse_media_checked: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TransactionRecord:
    """交易记录"""
    transaction_id: str
    customer_id: str
    transaction_type: TransactionType
    amount: float
    currency: str
    transaction_date: datetime
    value_date: datetime
    originator: Dict[str, Any]
    beneficiary: Dict[str, Any]
    originator_country: str
    beneficiary_country: str
    payment_method: str
    description: str
    reference_number: str
    channel: str
    branch_code: str
    employee_id: str
    risk_score: float
    flags: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AMLAlert:
    """AML告警"""
    alert_id: str
    alert_type: AlertType
    customer_id: str
    transaction_ids: List[str]
    risk_level: AMLRiskLevel
    alert_date: datetime
    alert_description: str
    triggered_rules: List[str]
    risk_score: float
    investigation_priority: int
    assigned_to: Optional[str]
    status: AlertStatus
    investigation_notes: List[str]
    supporting_documents: List[str]
    resolution_date: Optional[datetime]
    resolution_reason: Optional[str]
    sar_filed: bool
    sar_filing_date: Optional[datetime]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AMLRule:
    """AML规则"""
    rule_id: str
    rule_name: str
    rule_type: str
    description: str
    rule_logic: str
    threshold_values: Dict[str, Any]
    lookback_period: int
    is_active: bool
    severity_level: AMLRiskLevel
    false_positive_rate: float
    effectiveness_score: float
    last_updated: datetime
    created_by: str
    approved_by: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AMLReport:
    """AML报告"""
    report_id: str
    report_type: str
    reporting_period: Tuple[datetime, datetime]
    generation_date: datetime
    total_transactions: int
    total_alerts: int
    alerts_by_type: Dict[str, int]
    alerts_by_risk_level: Dict[str, int]
    sar_filings: int
    false_positive_rate: float
    investigation_metrics: Dict[str, Any]
    customer_risk_distribution: Dict[str, int]
    compliance_metrics: Dict[str, Any]
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

class AMLScreeningEngine:
    """
    反洗钱筛查引擎
    
    提供全面的AML合规功能，包括制裁名单筛查、
    PEP检查、可疑交易监控和报告生成。
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 数据存储
        self.sanctions_lists = {}
        self.pep_lists = {}
        self.customer_profiles = {}
        self.transaction_records = {}
        self.aml_alerts = {}
        self.aml_rules = {}
        self.aml_reports = {}
        
        # 筛查配置
        self.screening_config = {
            'match_threshold': config.get('match_threshold', 0.8),
            'fuzzy_matching': config.get('fuzzy_matching', True),
            'real_time_screening': config.get('real_time_screening', True),
            'batch_screening_interval': config.get('batch_screening_interval', 3600),
            'alert_auto_assignment': config.get('alert_auto_assignment', True),
            'suspicious_amount_threshold': config.get('suspicious_amount_threshold', 10000),
            'high_risk_countries': config.get('high_risk_countries', [
                'Afghanistan', 'Iran', 'North Korea', 'Syria', 'Yemen'
            ])
        }
        
        # 监控模式
        self.monitoring_patterns = {}
        
        # 统计信息
        self.screening_stats = {
            'total_screenings': 0,
            'sanctions_matches': 0,
            'pep_matches': 0,
            'alerts_generated': 0,
            'false_positives': 0,
            'sar_filings': 0,
            'screening_time_avg': 0.0
        }
        
        # 缓存
        self.screening_cache = {}
        
        # 初始化
        self._initialize_sanctions_lists()
        self._initialize_pep_lists()
        self._initialize_aml_rules()
        self._initialize_monitoring_patterns()
        
        self.logger.info("AML筛查引擎初始化完成")
    
    def _initialize_sanctions_lists(self):
        """初始化制裁名单"""
        # OFAC SDN List
        ofac_sdn = SanctionsList(
            list_id="OFAC_SDN",
            list_name="OFAC Specially Designated Nationals List",
            list_type="sanctions",
            jurisdiction="US",
            last_updated=datetime.now(),
            entries=[
                {
                    "name": "ABC Trading Company",
                    "type": "entity",
                    "country": "Country A",
                    "date_added": "2020-01-01",
                    "reason": "Money laundering activities",
                    "aliases": ["ABC Trade", "ABC Corp"],
                    "identification_numbers": ["123456789"]
                },
                {
                    "name": "John Doe",
                    "type": "individual",
                    "country": "Country B",
                    "date_of_birth": "1970-01-01",
                    "date_added": "2019-06-15",
                    "reason": "Terrorist financing",
                    "aliases": ["J. Doe", "Johnny Doe"],
                    "identification_numbers": ["987654321"]
                }
            ]
        )
        
        # EU Consolidated List
        eu_consolidated = SanctionsList(
            list_id="EU_CONSOLIDATED",
            list_name="EU Consolidated List",
            list_type="sanctions",
            jurisdiction="EU",
            last_updated=datetime.now(),
            entries=[
                {
                    "name": "XYZ Corporation",
                    "type": "entity",
                    "country": "Country C",
                    "date_added": "2021-03-10",
                    "reason": "Sanctions evasion",
                    "aliases": ["XYZ Corp", "XYZ Ltd"],
                    "identification_numbers": ["456789123"]
                }
            ]
        )
        
        # UN Sanctions List
        un_sanctions = SanctionsList(
            list_id="UN_SANCTIONS",
            list_name="UN Security Council Sanctions List",
            list_type="sanctions",
            jurisdiction="UN",
            last_updated=datetime.now(),
            entries=[
                {
                    "name": "DEF Organization",
                    "type": "entity",
                    "country": "Country D",
                    "date_added": "2018-11-20",
                    "reason": "Terrorism support",
                    "aliases": ["DEF Org", "DEF Group"],
                    "identification_numbers": ["789123456"]
                }
            ]
        )
        
        self.sanctions_lists = {
            ofac_sdn.list_id: ofac_sdn,
            eu_consolidated.list_id: eu_consolidated,
            un_sanctions.list_id: un_sanctions
        }
    
    def _initialize_pep_lists(self):
        """初始化PEP名单"""
        # World-Check PEP List
        world_check_pep = PEPList(
            list_id="WORLD_CHECK_PEP",
            list_name="World-Check PEP Database",
            jurisdiction="Global",
            last_updated=datetime.now(),
            entries=[
                {
                    "name": "Jane Smith",
                    "type": "individual",
                    "country": "Country E",
                    "position": "Minister of Finance",
                    "date_of_birth": "1965-05-15",
                    "pep_type": "foreign_pep",
                    "risk_level": "high",
                    "family_members": ["John Smith", "Mary Smith"],
                    "close_associates": ["Robert Johnson"]
                },
                {
                    "name": "Michael Johnson",
                    "type": "individual",
                    "country": "Country F",
                    "position": "Central Bank Governor",
                    "date_of_birth": "1960-12-03",
                    "pep_type": "domestic_pep",
                    "risk_level": "medium",
                    "family_members": ["Sarah Johnson"],
                    "close_associates": ["David Brown"]
                }
            ]
        )
        
        self.pep_lists = {
            world_check_pep.list_id: world_check_pep
        }
    
    def _initialize_aml_rules(self):
        """初始化AML规则"""
        # 大额交易规则
        large_transaction_rule = AMLRule(
            rule_id="RULE_LARGE_TRANSACTION",
            rule_name="Large Transaction Alert",
            rule_type="transaction_based",
            description="Alert for transactions exceeding specified threshold",
            rule_logic="amount >= threshold_amount",
            threshold_values={
                "threshold_amount": 10000,
                "currency": "USD"
            },
            lookback_period=0,
            is_active=True,
            severity_level=AMLRiskLevel.MEDIUM,
            false_positive_rate=0.15,
            effectiveness_score=0.85,
            last_updated=datetime.now(),
            created_by="system",
            approved_by="compliance_officer"
        )
        
        # 频繁交易规则
        frequent_transaction_rule = AMLRule(
            rule_id="RULE_FREQUENT_TRANSACTIONS",
            rule_name="Frequent Transaction Alert",
            rule_type="pattern_based",
            description="Alert for unusually high transaction frequency",
            rule_logic="transaction_count > threshold_count in lookback_period",
            threshold_values={
                "threshold_count": 20,
                "lookback_period_days": 1
            },
            lookback_period=1,
            is_active=True,
            severity_level=AMLRiskLevel.HIGH,
            false_positive_rate=0.25,
            effectiveness_score=0.75,
            last_updated=datetime.now(),
            created_by="system",
            approved_by="compliance_officer"
        )
        
        # 结构化交易规则
        structuring_rule = AMLRule(
            rule_id="RULE_STRUCTURING",
            rule_name="Structuring Alert",
            rule_type="pattern_based",
            description="Alert for potential structuring activities",
            rule_logic="multiple transactions just below reporting threshold",
            threshold_values={
                "individual_threshold": 9500,
                "aggregate_threshold": 10000,
                "transaction_count": 3,
                "lookback_period_days": 7
            },
            lookback_period=7,
            is_active=True,
            severity_level=AMLRiskLevel.HIGH,
            false_positive_rate=0.20,
            effectiveness_score=0.80,
            last_updated=datetime.now(),
            created_by="system",
            approved_by="compliance_officer"
        )
        
        # 高风险国家规则
        high_risk_country_rule = AMLRule(
            rule_id="RULE_HIGH_RISK_COUNTRY",
            rule_name="High Risk Country Alert",
            rule_type="location_based",
            description="Alert for transactions involving high-risk countries",
            rule_logic="originator_country in high_risk_countries or beneficiary_country in high_risk_countries",
            threshold_values={
                "high_risk_countries": self.screening_config['high_risk_countries'],
                "minimum_amount": 1000
            },
            lookback_period=0,
            is_active=True,
            severity_level=AMLRiskLevel.HIGH,
            false_positive_rate=0.30,
            effectiveness_score=0.70,
            last_updated=datetime.now(),
            created_by="system",
            approved_by="compliance_officer"
        )
        
        self.aml_rules = {
            large_transaction_rule.rule_id: large_transaction_rule,
            frequent_transaction_rule.rule_id: frequent_transaction_rule,
            structuring_rule.rule_id: structuring_rule,
            high_risk_country_rule.rule_id: high_risk_country_rule
        }
    
    def _initialize_monitoring_patterns(self):
        """初始化监控模式"""
        self.monitoring_patterns = {
            'velocity_monitoring': {
                'description': '交易速度监控',
                'parameters': {
                    'time_window': 3600,  # 1小时
                    'threshold_count': 10,
                    'threshold_amount': 50000
                }
            },
            'round_amount_monitoring': {
                'description': '整数金额监控',
                'parameters': {
                    'round_threshold': 1000,
                    'frequency_threshold': 5
                }
            },
            'cross_border_monitoring': {
                'description': '跨境交易监控',
                'parameters': {
                    'threshold_amount': 5000,
                    'high_risk_countries': self.screening_config['high_risk_countries']
                }
            },
            'dormant_account_monitoring': {
                'description': '休眠账户监控',
                'parameters': {
                    'dormancy_period': 180,  # 6个月
                    'reactivation_threshold': 1000
                }
            }
        }
    
    async def screen_customer(self, customer_profile: CustomerProfile) -> Dict[str, Any]:
        """筛查客户"""
        screening_results = {
            'customer_id': customer_profile.customer_id,
            'screening_date': datetime.now(),
            'sanctions_results': [],
            'pep_results': [],
            'adverse_media_results': [],
            'overall_risk_score': 0.0,
            'risk_factors': [],
            'recommendations': []
        }
        
        # 制裁名单筛查
        sanctions_results = await self._screen_sanctions(customer_profile)
        screening_results['sanctions_results'] = sanctions_results
        
        # PEP筛查
        pep_results = await self._screen_pep(customer_profile)
        screening_results['pep_results'] = pep_results
        
        # 负面媒体筛查
        adverse_media_results = await self._screen_adverse_media(customer_profile)
        screening_results['adverse_media_results'] = adverse_media_results
        
        # 计算整体风险评分
        overall_risk_score = self._calculate_customer_risk_score(
            sanctions_results, pep_results, adverse_media_results, customer_profile
        )
        screening_results['overall_risk_score'] = overall_risk_score
        
        # 识别风险因素
        risk_factors = self._identify_customer_risk_factors(customer_profile, screening_results)
        screening_results['risk_factors'] = risk_factors
        
        # 生成建议
        recommendations = self._generate_customer_recommendations(screening_results)
        screening_results['recommendations'] = recommendations
        
        # 更新统计
        self.screening_stats['total_screenings'] += 1
        if sanctions_results:
            self.screening_stats['sanctions_matches'] += 1
        if pep_results:
            self.screening_stats['pep_matches'] += 1
        
        # 创建告警（如果需要）
        if overall_risk_score >= 0.7:
            await self._create_customer_alert(customer_profile, screening_results)
        
        self.logger.info(f"客户筛查完成: {customer_profile.customer_id}")
        return screening_results
    
    async def _screen_sanctions(self, customer_profile: CustomerProfile) -> List[Dict[str, Any]]:
        """制裁名单筛查"""
        matches = []
        
        for list_id, sanctions_list in self.sanctions_lists.items():
            for entry in sanctions_list.entries:
                # 姓名匹配
                name_match_score = self._calculate_name_match_score(
                    customer_profile.customer_name, entry['name']
                )
                
                # 别名匹配
                alias_match_scores = []
                for alias in entry.get('aliases', []):
                    alias_match_scores.append(
                        self._calculate_name_match_score(customer_profile.customer_name, alias)
                    )
                
                max_match_score = max([name_match_score] + alias_match_scores)
                
                if max_match_score >= self.screening_config['match_threshold']:
                    matches.append({
                        'list_id': list_id,
                        'list_name': sanctions_list.list_name,
                        'entry_name': entry['name'],
                        'entry_type': entry['type'],
                        'match_score': max_match_score,
                        'match_type': 'name' if max_match_score == name_match_score else 'alias',
                        'entry_details': entry
                    })
        
        return matches
    
    async def _screen_pep(self, customer_profile: CustomerProfile) -> List[Dict[str, Any]]:
        """PEP筛查"""
        matches = []
        
        for list_id, pep_list in self.pep_lists.items():
            for entry in pep_list.entries:
                # 姓名匹配
                name_match_score = self._calculate_name_match_score(
                    customer_profile.customer_name, entry['name']
                )
                
                # 出生日期匹配（如果有）
                birth_date_match = False
                if customer_profile.date_of_birth and entry.get('date_of_birth'):
                    birth_date_match = customer_profile.date_of_birth.strftime('%Y-%m-%d') == entry['date_of_birth']
                
                # 国籍匹配
                country_match = customer_profile.nationality == entry['country']
                
                # 综合匹配分数
                total_match_score = name_match_score
                if birth_date_match:
                    total_match_score += 0.3
                if country_match:
                    total_match_score += 0.2
                
                if total_match_score >= self.screening_config['match_threshold']:
                    matches.append({
                        'list_id': list_id,
                        'list_name': pep_list.list_name,
                        'entry_name': entry['name'],
                        'pep_type': entry['pep_type'],
                        'position': entry['position'],
                        'match_score': total_match_score,
                        'entry_details': entry
                    })
        
        return matches
    
    async def _screen_adverse_media(self, customer_profile: CustomerProfile) -> List[Dict[str, Any]]:
        """负面媒体筛查"""
        # 模拟负面媒体筛查
        adverse_media_results = []
        
        # 简化版本 - 实际应用中需要连接到负面媒体数据库
        risk_keywords = ['money laundering', 'terrorist financing', 'sanctions', 'corruption', 'fraud']
        
        # 模拟基于客户姓名的风险评分
        if any(keyword in customer_profile.customer_name.lower() for keyword in ['suspicious', 'risk', 'alert']):
            adverse_media_results.append({
                'source': 'news_media',
                'headline': f'Investigation into {customer_profile.customer_name}',
                'date': datetime.now() - timedelta(days=30),
                'risk_score': 0.7,
                'keywords': ['investigation', 'financial irregularities'],
                'url': 'https://example.com/news/investigation'
            })
        
        return adverse_media_results
    
    def _calculate_name_match_score(self, name1: str, name2: str) -> float:
        """计算姓名匹配分数"""
        if not self.screening_config['fuzzy_matching']:
            return 1.0 if name1.lower() == name2.lower() else 0.0
        
        # 简化的模糊匹配算法
        name1_clean = re.sub(r'[^\w\s]', '', name1.lower())
        name2_clean = re.sub(r'[^\w\s]', '', name2.lower())
        
        # 计算Jaccard相似度
        words1 = set(name1_clean.split())
        words2 = set(name2_clean.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _calculate_customer_risk_score(self, sanctions_results: List[Dict[str, Any]], 
                                     pep_results: List[Dict[str, Any]], 
                                     adverse_media_results: List[Dict[str, Any]], 
                                     customer_profile: CustomerProfile) -> float:
        """计算客户风险评分"""
        base_score = 0.0
        
        # 制裁名单匹配
        if sanctions_results:
            max_sanctions_score = max([result['match_score'] for result in sanctions_results])
            base_score += max_sanctions_score * 0.8
        
        # PEP匹配
        if pep_results:
            max_pep_score = max([result['match_score'] for result in pep_results])
            base_score += max_pep_score * 0.6
        
        # 负面媒体
        if adverse_media_results:
            max_adverse_score = max([result['risk_score'] for result in adverse_media_results])
            base_score += max_adverse_score * 0.4
        
        # 客户特征风险
        if customer_profile.country_of_residence in self.screening_config['high_risk_countries']:
            base_score += 0.3
        
        if customer_profile.nationality in self.screening_config['high_risk_countries']:
            base_score += 0.2
        
        # 业务类型风险
        high_risk_business_types = ['money_service_business', 'cash_intensive_business', 'cryptocurrency']
        if customer_profile.business_type in high_risk_business_types:
            base_score += 0.3
        
        return min(1.0, base_score)
    
    def _identify_customer_risk_factors(self, customer_profile: CustomerProfile, 
                                      screening_results: Dict[str, Any]) -> List[str]:
        """识别客户风险因素"""
        risk_factors = []
        
        if screening_results['sanctions_results']:
            risk_factors.append('制裁名单匹配')
        
        if screening_results['pep_results']:
            risk_factors.append('政治敏感人员')
        
        if screening_results['adverse_media_results']:
            risk_factors.append('负面媒体报道')
        
        if customer_profile.country_of_residence in self.screening_config['high_risk_countries']:
            risk_factors.append('高风险国家居住')
        
        if customer_profile.nationality in self.screening_config['high_risk_countries']:
            risk_factors.append('高风险国家国籍')
        
        if customer_profile.customer_type == 'corporate' and not customer_profile.beneficial_owners:
            risk_factors.append('受益所有人信息缺失')
        
        if customer_profile.source_of_funds == 'unknown':
            risk_factors.append('资金来源不明')
        
        return risk_factors
    
    def _generate_customer_recommendations(self, screening_results: Dict[str, Any]) -> List[str]:
        """生成客户建议"""
        recommendations = []
        
        if screening_results['sanctions_results']:
            recommendations.append('立即冻结账户并上报监管机构')
        
        if screening_results['pep_results']:
            recommendations.append('实施增强尽职调查')
            recommendations.append('获得高级管理层批准')
        
        if screening_results['adverse_media_results']:
            recommendations.append('深入调查负面媒体报道')
            recommendations.append('评估声誉风险')
        
        if screening_results['overall_risk_score'] >= 0.7:
            recommendations.append('设置为高风险客户')
            recommendations.append('增加交易监控频率')
        
        if not recommendations:
            recommendations.append('客户风险可接受，继续正常监控')
        
        return recommendations
    
    async def _create_customer_alert(self, customer_profile: CustomerProfile, 
                                   screening_results: Dict[str, Any]):
        """创建客户告警"""
        alert_id = f"CUST_ALERT_{customer_profile.customer_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 确定告警类型
        alert_type = AlertType.UNUSUAL_ACTIVITY
        if screening_results['sanctions_results']:
            alert_type = AlertType.SANCTIONS_MATCH
        elif screening_results['pep_results']:
            alert_type = AlertType.PEP_MATCH
        elif screening_results['adverse_media_results']:
            alert_type = AlertType.ADVERSE_MEDIA
        
        # 确定风险等级
        risk_score = screening_results['overall_risk_score']
        if risk_score >= 0.9:
            risk_level = AMLRiskLevel.CRITICAL
        elif risk_score >= 0.7:
            risk_level = AMLRiskLevel.HIGH
        elif risk_score >= 0.5:
            risk_level = AMLRiskLevel.MEDIUM
        else:
            risk_level = AMLRiskLevel.LOW
        
        alert = AMLAlert(
            alert_id=alert_id,
            alert_type=alert_type,
            customer_id=customer_profile.customer_id,
            transaction_ids=[],
            risk_level=risk_level,
            alert_date=datetime.now(),
            alert_description=f"客户筛查发现风险: {', '.join(screening_results['risk_factors'])}",
            triggered_rules=['customer_screening'],
            risk_score=risk_score,
            investigation_priority=self._calculate_investigation_priority(risk_level),
            status=AlertStatus.OPEN,
            investigation_notes=[],
            supporting_documents=[],
            sar_filed=False
        )
        
        self.aml_alerts[alert_id] = alert
        self.screening_stats['alerts_generated'] += 1
        
        self.logger.warning(f"客户告警已创建: {alert_id}")
    
    async def screen_transaction(self, transaction: TransactionRecord) -> Dict[str, Any]:
        """筛查交易"""
        screening_results = {
            'transaction_id': transaction.transaction_id,
            'screening_date': datetime.now(),
            'rule_violations': [],
            'risk_score': 0.0,
            'alerts_generated': [],
            'recommendations': []
        }
        
        # 应用AML规则
        for rule_id, rule in self.aml_rules.items():
            if rule.is_active:
                violation = await self._evaluate_rule(transaction, rule)
                if violation:
                    screening_results['rule_violations'].append({
                        'rule_id': rule_id,
                        'rule_name': rule.rule_name,
                        'violation_details': violation
                    })
        
        # 计算交易风险评分
        risk_score = await self._calculate_transaction_risk_score(transaction, screening_results)
        screening_results['risk_score'] = risk_score
        
        # 生成告警
        if screening_results['rule_violations']:
            alert = await self._create_transaction_alert(transaction, screening_results)
            screening_results['alerts_generated'].append(alert.alert_id)
        
        # 生成建议
        recommendations = self._generate_transaction_recommendations(screening_results)
        screening_results['recommendations'] = recommendations
        
        # 更新交易记录
        transaction.risk_score = risk_score
        transaction.flags = [violation['rule_name'] for violation in screening_results['rule_violations']]
        
        self.logger.info(f"交易筛查完成: {transaction.transaction_id}")
        return screening_results
    
    async def _evaluate_rule(self, transaction: TransactionRecord, rule: AMLRule) -> Optional[Dict[str, Any]]:
        """评估规则"""
        if rule.rule_type == "transaction_based":
            return await self._evaluate_transaction_based_rule(transaction, rule)
        elif rule.rule_type == "pattern_based":
            return await self._evaluate_pattern_based_rule(transaction, rule)
        elif rule.rule_type == "location_based":
            return await self._evaluate_location_based_rule(transaction, rule)
        else:
            return None
    
    async def _evaluate_transaction_based_rule(self, transaction: TransactionRecord, 
                                             rule: AMLRule) -> Optional[Dict[str, Any]]:
        """评估基于交易的规则"""
        if rule.rule_id == "RULE_LARGE_TRANSACTION":
            threshold = rule.threshold_values['threshold_amount']
            if transaction.amount >= threshold:
                return {
                    'violation_type': 'large_transaction',
                    'threshold_value': threshold,
                    'actual_value': transaction.amount,
                    'description': f'交易金额 {transaction.amount} 超过阈值 {threshold}'
                }
        
        return None
    
    async def _evaluate_pattern_based_rule(self, transaction: TransactionRecord, 
                                         rule: AMLRule) -> Optional[Dict[str, Any]]:
        """评估基于模式的规则"""
        if rule.rule_id == "RULE_FREQUENT_TRANSACTIONS":
            # 查找同一客户在指定时间内的交易
            lookback_date = transaction.transaction_date - timedelta(days=rule.threshold_values['lookback_period_days'])
            
            customer_transactions = [
                t for t in self.transaction_records.values()
                if t.customer_id == transaction.customer_id and t.transaction_date >= lookback_date
            ]
            
            if len(customer_transactions) >= rule.threshold_values['threshold_count']:
                return {
                    'violation_type': 'frequent_transactions',
                    'threshold_value': rule.threshold_values['threshold_count'],
                    'actual_value': len(customer_transactions),
                    'description': f'客户在{rule.threshold_values["lookback_period_days"]}天内进行了{len(customer_transactions)}笔交易'
                }
        
        elif rule.rule_id == "RULE_STRUCTURING":
            # 检查潜在的结构化交易
            lookback_date = transaction.transaction_date - timedelta(days=rule.threshold_values['lookback_period_days'])
            
            customer_transactions = [
                t for t in self.transaction_records.values()
                if t.customer_id == transaction.customer_id and 
                t.transaction_date >= lookback_date and
                t.amount < rule.threshold_values['individual_threshold']
            ]
            
            total_amount = sum(t.amount for t in customer_transactions)
            
            if (len(customer_transactions) >= rule.threshold_values['transaction_count'] and 
                total_amount >= rule.threshold_values['aggregate_threshold']):
                return {
                    'violation_type': 'structuring',
                    'transaction_count': len(customer_transactions),
                    'total_amount': total_amount,
                    'description': f'客户可能进行结构化交易: {len(customer_transactions)}笔交易，总金额{total_amount}'
                }
        
        return None
    
    async def _evaluate_location_based_rule(self, transaction: TransactionRecord, 
                                          rule: AMLRule) -> Optional[Dict[str, Any]]:
        """评估基于位置的规则"""
        if rule.rule_id == "RULE_HIGH_RISK_COUNTRY":
            high_risk_countries = rule.threshold_values['high_risk_countries']
            minimum_amount = rule.threshold_values['minimum_amount']
            
            if (transaction.amount >= minimum_amount and 
                (transaction.originator_country in high_risk_countries or 
                 transaction.beneficiary_country in high_risk_countries)):
                return {
                    'violation_type': 'high_risk_country',
                    'originator_country': transaction.originator_country,
                    'beneficiary_country': transaction.beneficiary_country,
                    'amount': transaction.amount,
                    'description': f'交易涉及高风险国家: {transaction.originator_country} -> {transaction.beneficiary_country}'
                }
        
        return None
    
    async def _calculate_transaction_risk_score(self, transaction: TransactionRecord, 
                                              screening_results: Dict[str, Any]) -> float:
        """计算交易风险评分"""
        base_score = 0.0
        
        # 基于规则违规的评分
        for violation in screening_results['rule_violations']:
            rule_id = violation['rule_id']
            if rule_id in self.aml_rules:
                rule = self.aml_rules[rule_id]
                if rule.severity_level == AMLRiskLevel.CRITICAL:
                    base_score += 0.8
                elif rule.severity_level == AMLRiskLevel.HIGH:
                    base_score += 0.6
                elif rule.severity_level == AMLRiskLevel.MEDIUM:
                    base_score += 0.4
                else:
                    base_score += 0.2
        
        # 基于交易特征的评分
        if transaction.amount >= 50000:
            base_score += 0.3
        
        if transaction.originator_country != transaction.beneficiary_country:
            base_score += 0.2
        
        if transaction.originator_country in self.screening_config['high_risk_countries']:
            base_score += 0.4
        
        if transaction.beneficiary_country in self.screening_config['high_risk_countries']:
            base_score += 0.4
        
        # 基于交易类型的评分
        high_risk_types = [TransactionType.CASH_DEPOSIT, TransactionType.CASH_WITHDRAWAL, 
                          TransactionType.CRYPTOCURRENCY]
        if transaction.transaction_type in high_risk_types:
            base_score += 0.3
        
        return min(1.0, base_score)
    
    async def _create_transaction_alert(self, transaction: TransactionRecord, 
                                      screening_results: Dict[str, Any]) -> AMLAlert:
        """创建交易告警"""
        alert_id = f"TXN_ALERT_{transaction.transaction_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 确定告警类型
        alert_type = AlertType.UNUSUAL_ACTIVITY
        for violation in screening_results['rule_violations']:
            if violation['violation_details']['violation_type'] == 'large_transaction':
                alert_type = AlertType.LARGE_AMOUNT
            elif violation['violation_details']['violation_type'] == 'frequent_transactions':
                alert_type = AlertType.FREQUENT_TRANSACTIONS
            elif violation['violation_details']['violation_type'] == 'structuring':
                alert_type = AlertType.STRUCTURING
            elif violation['violation_details']['violation_type'] == 'high_risk_country':
                alert_type = AlertType.COUNTRY_RISK
        
        # 确定风险等级
        risk_score = screening_results['risk_score']
        if risk_score >= 0.8:
            risk_level = AMLRiskLevel.CRITICAL
        elif risk_score >= 0.6:
            risk_level = AMLRiskLevel.HIGH
        elif risk_score >= 0.4:
            risk_level = AMLRiskLevel.MEDIUM
        else:
            risk_level = AMLRiskLevel.LOW
        
        # 创建告警描述
        violation_descriptions = [v['violation_details']['description'] for v in screening_results['rule_violations']]
        alert_description = f"交易风险告警: {'; '.join(violation_descriptions)}"
        
        alert = AMLAlert(
            alert_id=alert_id,
            alert_type=alert_type,
            customer_id=transaction.customer_id,
            transaction_ids=[transaction.transaction_id],
            risk_level=risk_level,
            alert_date=datetime.now(),
            alert_description=alert_description,
            triggered_rules=[v['rule_id'] for v in screening_results['rule_violations']],
            risk_score=risk_score,
            investigation_priority=self._calculate_investigation_priority(risk_level),
            status=AlertStatus.OPEN,
            investigation_notes=[],
            supporting_documents=[],
            sar_filed=False
        )
        
        self.aml_alerts[alert_id] = alert
        self.screening_stats['alerts_generated'] += 1
        
        self.logger.warning(f"交易告警已创建: {alert_id}")
        return alert
    
    def _calculate_investigation_priority(self, risk_level: AMLRiskLevel) -> int:
        """计算调查优先级"""
        priority_map = {
            AMLRiskLevel.CRITICAL: 1,
            AMLRiskLevel.HIGH: 2,
            AMLRiskLevel.MEDIUM: 3,
            AMLRiskLevel.LOW: 4
        }
        return priority_map.get(risk_level, 3)
    
    def _generate_transaction_recommendations(self, screening_results: Dict[str, Any]) -> List[str]:
        """生成交易建议"""
        recommendations = []
        
        if screening_results['risk_score'] >= 0.8:
            recommendations.append('立即暂停交易并展开调查')
        elif screening_results['risk_score'] >= 0.6:
            recommendations.append('标记为高风险交易，加强监控')
        elif screening_results['risk_score'] >= 0.4:
            recommendations.append('进行额外的尽职调查')
        
        for violation in screening_results['rule_violations']:
            violation_type = violation['violation_details']['violation_type']
            if violation_type == 'large_transaction':
                recommendations.append('获得大额交易审批')
            elif violation_type == 'frequent_transactions':
                recommendations.append('分析客户交易模式')
            elif violation_type == 'structuring':
                recommendations.append('深入调查结构化交易嫌疑')
            elif violation_type == 'high_risk_country':
                recommendations.append('验证交易业务目的')
        
        if not recommendations:
            recommendations.append('交易风险可接受，继续处理')
        
        return recommendations
    
    async def investigate_alert(self, alert_id: str, investigation_notes: List[str], 
                              supporting_documents: List[str]) -> Dict[str, Any]:
        """调查告警"""
        if alert_id not in self.aml_alerts:
            raise ValueError(f"告警不存在: {alert_id}")
        
        alert = self.aml_alerts[alert_id]
        alert.status = AlertStatus.INVESTIGATING
        alert.investigation_notes.extend(investigation_notes)
        alert.supporting_documents.extend(supporting_documents)
        
        # 模拟调查结果
        investigation_results = {
            'alert_id': alert_id,
            'investigation_date': datetime.now(),
            'investigation_summary': '已完成初步调查',
            'findings': [],
            'conclusion': 'pending',
            'recommended_action': 'continue_investigation'
        }
        
        # 基于告警类型生成调查发现
        if alert.alert_type == AlertType.SANCTIONS_MATCH:
            investigation_results['findings'].append('制裁名单匹配需要进一步验证')
            investigation_results['conclusion'] = 'suspicious'
            investigation_results['recommended_action'] = 'file_sar'
        elif alert.alert_type == AlertType.STRUCTURING:
            investigation_results['findings'].append('发现可疑的结构化交易模式')
            investigation_results['conclusion'] = 'suspicious'
            investigation_results['recommended_action'] = 'file_sar'
        elif alert.alert_type == AlertType.LARGE_AMOUNT:
            investigation_results['findings'].append('大额交易已得到适当解释')
            investigation_results['conclusion'] = 'cleared'
            investigation_results['recommended_action'] = 'close_alert'
        
        self.logger.info(f"告警调查完成: {alert_id}")
        return investigation_results
    
    async def close_alert(self, alert_id: str, resolution_reason: str, file_sar: bool = False):
        """关闭告警"""
        if alert_id not in self.aml_alerts:
            raise ValueError(f"告警不存在: {alert_id}")
        
        alert = self.aml_alerts[alert_id]
        alert.resolution_date = datetime.now()
        alert.resolution_reason = resolution_reason
        
        if file_sar:
            alert.status = AlertStatus.FILED_SAR
            alert.sar_filed = True
            alert.sar_filing_date = datetime.now()
            self.screening_stats['sar_filings'] += 1
        else:
            alert.status = AlertStatus.CLOSED_CLEARED
        
        self.logger.info(f"告警已关闭: {alert_id}")
    
    async def generate_aml_report(self, report_type: str, start_date: datetime, 
                                end_date: datetime) -> AMLReport:
        """生成AML报告"""
        report_id = f"AML_REPORT_{report_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 筛选报告期间的数据
        period_transactions = [
            t for t in self.transaction_records.values()
            if start_date <= t.transaction_date <= end_date
        ]
        
        period_alerts = [
            a for a in self.aml_alerts.values()
            if start_date <= a.alert_date <= end_date
        ]
        
        # 统计告警类型
        alerts_by_type = {}
        for alert in period_alerts:
            alert_type = alert.alert_type.value
            alerts_by_type[alert_type] = alerts_by_type.get(alert_type, 0) + 1
        
        # 统计风险等级
        alerts_by_risk_level = {}
        for alert in period_alerts:
            risk_level = alert.risk_level.value
            alerts_by_risk_level[risk_level] = alerts_by_risk_level.get(risk_level, 0) + 1
        
        # 统计SAR申报
        sar_filings = len([a for a in period_alerts if a.sar_filed])
        
        # 计算误报率
        closed_alerts = [a for a in period_alerts if a.status in [AlertStatus.CLOSED_CLEARED, AlertStatus.FILED_SAR]]
        false_positives = len([a for a in closed_alerts if a.status == AlertStatus.CLOSED_CLEARED])
        false_positive_rate = false_positives / len(closed_alerts) if closed_alerts else 0.0
        
        # 调查指标
        investigation_metrics = {
            'total_alerts': len(period_alerts),
            'alerts_investigated': len([a for a in period_alerts if a.status != AlertStatus.OPEN]),
            'average_investigation_time': 3.5,  # 模拟数据
            'alerts_pending': len([a for a in period_alerts if a.status == AlertStatus.OPEN]),
            'alerts_escalated': len([a for a in period_alerts if a.status == AlertStatus.ESCALATED])
        }
        
        # 客户风险分布
        customer_risk_distribution = {
            'low': 0,
            'medium': 0,
            'high': 0,
            'prohibited': 0
        }
        
        for customer in self.customer_profiles.values():
            risk_rating = customer.risk_rating.value
            customer_risk_distribution[risk_rating] += 1
        
        # 合规指标
        compliance_metrics = {
            'screening_coverage': 100.0,  # 筛查覆盖率
            'sanctions_list_updates': 12,  # 制裁名单更新次数
            'pep_list_updates': 8,  # PEP名单更新次数
            'rule_effectiveness': 0.75,  # 规则有效性
            'system_uptime': 99.9  # 系统可用性
        }
        
        # 生成建议
        recommendations = self._generate_aml_recommendations(
            period_alerts, false_positive_rate, investigation_metrics
        )
        
        report = AMLReport(
            report_id=report_id,
            report_type=report_type,
            reporting_period=(start_date, end_date),
            generation_date=datetime.now(),
            total_transactions=len(period_transactions),
            total_alerts=len(period_alerts),
            alerts_by_type=alerts_by_type,
            alerts_by_risk_level=alerts_by_risk_level,
            sar_filings=sar_filings,
            false_positive_rate=false_positive_rate,
            investigation_metrics=investigation_metrics,
            customer_risk_distribution=customer_risk_distribution,
            compliance_metrics=compliance_metrics,
            recommendations=recommendations
        )
        
        self.aml_reports[report_id] = report
        self.logger.info(f"AML报告已生成: {report_id}")
        return report
    
    def _generate_aml_recommendations(self, alerts: List[AMLAlert], 
                                    false_positive_rate: float, 
                                    investigation_metrics: Dict[str, Any]) -> List[str]:
        """生成AML建议"""
        recommendations = []
        
        # 基于误报率的建议
        if false_positive_rate > 0.3:
            recommendations.append('误报率过高，建议优化AML规则和阈值')
        
        # 基于调查效率的建议
        if investigation_metrics['alerts_pending'] > 10:
            recommendations.append('待调查告警过多，建议增加调查人员或优化流程')
        
        # 基于告警类型的建议
        high_volume_alert_types = [
            alert_type for alert_type, count in 
            Counter([a.alert_type.value for a in alerts]).items()
            if count > 5
        ]
        
        if high_volume_alert_types:
            recommendations.append(f'以下告警类型量大，建议重点关注: {", ".join(high_volume_alert_types)}')
        
        # 基于风险等级的建议
        critical_alerts = [a for a in alerts if a.risk_level == AMLRiskLevel.CRITICAL]
        if critical_alerts:
            recommendations.append(f'发现{len(critical_alerts)}个关键风险告警，建议优先处理')
        
        # 通用建议
        recommendations.extend([
            '定期更新制裁名单和PEP名单',
            '持续监控和优化AML规则',
            '加强员工AML培训',
            '建立客户风险评级机制'
        ])
        
        return recommendations
    
    async def update_sanctions_list(self, list_id: str, new_entries: List[Dict[str, Any]]):
        """更新制裁名单"""
        if list_id in self.sanctions_lists:
            sanctions_list = self.sanctions_lists[list_id]
            sanctions_list.entries.extend(new_entries)
            sanctions_list.last_updated = datetime.now()
            self.logger.info(f"制裁名单已更新: {list_id}, 新增{len(new_entries)}个条目")
        else:
            self.logger.error(f"制裁名单不存在: {list_id}")
    
    async def update_pep_list(self, list_id: str, new_entries: List[Dict[str, Any]]):
        """更新PEP名单"""
        if list_id in self.pep_lists:
            pep_list = self.pep_lists[list_id]
            pep_list.entries.extend(new_entries)
            pep_list.last_updated = datetime.now()
            self.logger.info(f"PEP名单已更新: {list_id}, 新增{len(new_entries)}个条目")
        else:
            self.logger.error(f"PEP名单不存在: {list_id}")
    
    async def get_aml_dashboard(self) -> Dict[str, Any]:
        """获取AML仪表板"""
        # 最近30天的统计
        thirty_days_ago = datetime.now() - timedelta(days=30)
        recent_alerts = [a for a in self.aml_alerts.values() if a.alert_date >= thirty_days_ago]
        recent_transactions = [t for t in self.transaction_records.values() if t.transaction_date >= thirty_days_ago]
        
        dashboard = {
            'summary': {
                'total_customers': len(self.customer_profiles),
                'total_transactions_30d': len(recent_transactions),
                'total_alerts_30d': len(recent_alerts),
                'open_alerts': len([a for a in recent_alerts if a.status == AlertStatus.OPEN]),
                'sar_filings_30d': len([a for a in recent_alerts if a.sar_filed]),
                'false_positive_rate': self.screening_stats.get('false_positives', 0) / max(1, self.screening_stats.get('alerts_generated', 1))
            },
            'alerts_by_type': {
                alert_type.value: len([a for a in recent_alerts if a.alert_type == alert_type])
                for alert_type in AlertType
            },
            'alerts_by_risk_level': {
                risk_level.value: len([a for a in recent_alerts if a.risk_level == risk_level])
                for risk_level in AMLRiskLevel
            },
            'customer_risk_distribution': {
                risk_rating.value: len([c for c in self.customer_profiles.values() if c.risk_rating == risk_rating])
                for risk_rating in CustomerRiskRating
            },
            'screening_statistics': self.screening_stats.copy(),
            'recent_alerts': [
                {
                    'alert_id': alert.alert_id,
                    'alert_type': alert.alert_type.value,
                    'customer_id': alert.customer_id,
                    'risk_level': alert.risk_level.value,
                    'alert_date': alert.alert_date.isoformat(),
                    'status': alert.status.value
                }
                for alert in sorted(recent_alerts, key=lambda x: x.alert_date, reverse=True)[:10]
            ]
        }
        
        return dashboard
    
    async def export_aml_data(self, export_type: str, format: str = "csv") -> str:
        """导出AML数据"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"aml_{export_type}_{timestamp}.{format}"
        
        if export_type == "alerts":
            if format == "csv":
                data = []
                for alert in self.aml_alerts.values():
                    data.append({
                        'Alert ID': alert.alert_id,
                        'Alert Type': alert.alert_type.value,
                        'Customer ID': alert.customer_id,
                        'Risk Level': alert.risk_level.value,
                        'Alert Date': alert.alert_date.isoformat(),
                        'Status': alert.status.value,
                        'Risk Score': alert.risk_score,
                        'SAR Filed': alert.sar_filed,
                        'Description': alert.alert_description
                    })
                
                df = pd.DataFrame(data)
                df.to_csv(filename, index=False)
        
        elif export_type == "customers":
            if format == "csv":
                data = []
                for customer in self.customer_profiles.values():
                    data.append({
                        'Customer ID': customer.customer_id,
                        'Customer Name': customer.customer_name,
                        'Customer Type': customer.customer_type,
                        'Risk Rating': customer.risk_rating.value,
                        'Nationality': customer.nationality,
                        'Country of Residence': customer.country_of_residence,
                        'Onboarding Date': customer.onboarding_date.isoformat(),
                        'KYC Status': customer.kyc_status
                    })
                
                df = pd.DataFrame(data)
                df.to_csv(filename, index=False)
        
        self.logger.info(f"AML数据已导出: {filename}")
        return filename

# 导入Counter用于统计
from collections import Counter
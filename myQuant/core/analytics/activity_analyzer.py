"""
股票活跃度分析器
用于评估和筛选活跃度高的股票
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta


class ActivityAnalyzer:
    """股票活跃度分析器"""
    
    def __init__(self):
        self.activity_weights = {
            'volume_score': 0.3,        # 成交量权重
            'volatility_score': 0.25,   # 波动率权重
            'liquidity_score': 0.2,     # 流动性权重
            'turnover_score': 0.15,     # 换手率权重
            'consistency_score': 0.1,   # 持续性权重
        }
    
    def calculate_activity_score(self, stock_data: pd.DataFrame) -> float:
        """
        计算股票活跃度综合评分
        
        Args:
            stock_data: 股票历史数据，包含OHLCV
            
        Returns:
            活跃度评分 (0-100)
        """
        if len(stock_data) < 20:
            return 0.0
            
        scores = {}
        
        # 1. 成交量活跃度
        scores['volume_score'] = self._calculate_volume_score(stock_data)
        
        # 2. 价格波动活跃度
        scores['volatility_score'] = self._calculate_volatility_score(stock_data)
        
        # 3. 流动性活跃度
        scores['liquidity_score'] = self._calculate_liquidity_score(stock_data)
        
        # 4. 换手率活跃度
        scores['turnover_score'] = self._calculate_turnover_score(stock_data)
        
        # 5. 交易持续性
        scores['consistency_score'] = self._calculate_consistency_score(stock_data)
        
        # 加权计算总分
        total_score = sum(
            scores[key] * self.activity_weights[key] 
            for key in scores
        )
        
        return min(100.0, max(0.0, total_score))
    
    def _calculate_volume_score(self, data: pd.DataFrame) -> float:
        """计算成交量活跃度评分"""
        # 平均日成交额
        avg_amount = data['volume'].mean() * data['close'].mean()
        
        # 成交额标准
        amount_thresholds = [
            (500_000_000, 100),   # 5亿以上: 满分
            (200_000_000, 80),    # 2-5亿: 80分
            (100_000_000, 60),    # 1-2亿: 60分
            (50_000_000, 40),     # 5000万-1亿: 40分
            (20_000_000, 20),     # 2000万-5000万: 20分
        ]
        
        for threshold, score in amount_thresholds:
            if avg_amount >= threshold:
                return score
        
        return 0
    
    def _calculate_volatility_score(self, data: pd.DataFrame) -> float:
        """计算波动率活跃度评分"""
        # 计算日收益率
        data['returns'] = data['close'].pct_change()
        
        # 计算波动率
        volatility = data['returns'].std() * np.sqrt(252)  # 年化波动率
        
        # 计算平均真实波幅(ATR)
        data['high_low'] = data['high'] - data['low']
        data['high_close'] = np.abs(data['high'] - data['close'].shift())
        data['low_close'] = np.abs(data['low'] - data['close'].shift())
        data['tr'] = data[['high_low', 'high_close', 'low_close']].max(axis=1)
        atr = data['tr'].rolling(20).mean().iloc[-1]
        atr_ratio = atr / data['close'].iloc[-1]
        
        # 波动率评分
        volatility_score = min(100, volatility * 200)  # 50%波动率 = 100分
        atr_score = min(100, atr_ratio * 2000)         # 5%ATR = 100分
        
        return (volatility_score + atr_score) / 2
    
    def _calculate_liquidity_score(self, data: pd.DataFrame) -> float:
        """计算流动性活跃度评分"""
        # 成交量稳定性
        volume_cv = data['volume'].std() / data['volume'].mean()
        volume_stability = max(0, 100 - volume_cv * 100)
        
        # 价格连续性（无跳空）
        data['gap'] = np.abs(data['open'] - data['close'].shift()) / data['close'].shift()
        gap_score = max(0, 100 - data['gap'].mean() * 1000)
        
        return (volume_stability + gap_score) / 2
    
    def _calculate_turnover_score(self, data: pd.DataFrame) -> float:
        """计算换手率活跃度评分"""
        # 假设流通股本为市值的80%
        market_cap = data['close'].iloc[-1] * data['volume'].mean() * 100
        circulating_shares = market_cap * 0.8 / data['close'].iloc[-1]
        
        # 计算换手率
        turnover_rate = data['volume'].mean() / circulating_shares
        
        # 换手率评分
        turnover_thresholds = [
            (0.10, 100),  # 10%以上: 满分
            (0.05, 80),   # 5-10%: 80分
            (0.03, 60),   # 3-5%: 60分
            (0.02, 40),   # 2-3%: 40分
            (0.01, 20),   # 1-2%: 20分
        ]
        
        for threshold, score in turnover_thresholds:
            if turnover_rate >= threshold:
                return score
        
        return 0
    
    def _calculate_consistency_score(self, data: pd.DataFrame) -> float:
        """计算交易持续性评分"""
        # 交易天数占比
        trading_days = len(data[data['volume'] > 0])
        total_days = len(data)
        trading_ratio = trading_days / total_days
        
        # 成交量一致性
        volume_consistency = 1 - (data['volume'].std() / data['volume'].mean())
        
        return (trading_ratio + volume_consistency) * 50
    
    def get_activity_criteria(self) -> Dict:
        """获取活跃度筛选标准"""
        return {
            "high_activity": {
                "min_score": 70,
                "description": "高活跃度股票",
                "criteria": [
                    "日均成交额 > 2亿元",
                    "年化波动率 > 20%",
                    "换手率 > 3%",
                    "交易连续性 > 95%"
                ]
            },
            "medium_activity": {
                "min_score": 50,
                "description": "中等活跃度股票",
                "criteria": [
                    "日均成交额 > 1亿元",
                    "年化波动率 > 15%",
                    "换手率 > 2%",
                    "交易连续性 > 90%"
                ]
            },
            "low_activity": {
                "min_score": 30,
                "description": "低活跃度股票",
                "criteria": [
                    "日均成交额 > 5000万元",
                    "年化波动率 > 10%",
                    "换手率 > 1%",
                    "交易连续性 > 80%"
                ]
            }
        }
    
    def filter_active_stocks(self, stocks_data: Dict[str, pd.DataFrame], 
                           min_activity_score: float = 60) -> List[Tuple[str, float]]:
        """
        筛选活跃度高的股票
        
        Args:
            stocks_data: 股票数据字典 {symbol: DataFrame}
            min_activity_score: 最低活跃度评分
            
        Returns:
            [(symbol, activity_score), ...] 按活跃度排序
        """
        active_stocks = []
        
        for symbol, data in stocks_data.items():
            try:
                score = self.calculate_activity_score(data)
                if score >= min_activity_score:
                    active_stocks.append((symbol, score))
            except Exception as e:
                print(f"计算{symbol}活跃度时出错: {e}")
                continue
        
        # 按活跃度评分排序
        active_stocks.sort(key=lambda x: x[1], reverse=True)
        
        return active_stocks
    
    def get_activity_report(self, symbol: str, data: pd.DataFrame) -> Dict:
        """
        生成股票活跃度报告
        
        Args:
            symbol: 股票代码
            data: 股票数据
            
        Returns:
            活跃度详细报告
        """
        total_score = self.calculate_activity_score(data)
        
        # 计算各项子评分
        volume_score = self._calculate_volume_score(data)
        volatility_score = self._calculate_volatility_score(data)
        liquidity_score = self._calculate_liquidity_score(data)
        turnover_score = self._calculate_turnover_score(data)
        consistency_score = self._calculate_consistency_score(data)
        
        # 计算基础指标
        avg_volume = data['volume'].mean()
        avg_amount = avg_volume * data['close'].mean()
        data['returns'] = data['close'].pct_change()
        volatility = data['returns'].std() * np.sqrt(252)
        
        return {
            "symbol": symbol,
            "total_score": round(total_score, 2),
            "grade": self._get_activity_grade(total_score),
            "sub_scores": {
                "volume": round(volume_score, 2),
                "volatility": round(volatility_score, 2),
                "liquidity": round(liquidity_score, 2),
                "turnover": round(turnover_score, 2),
                "consistency": round(consistency_score, 2),
            },
            "metrics": {
                "avg_daily_volume": int(avg_volume),
                "avg_daily_amount": int(avg_amount),
                "annual_volatility": round(volatility, 4),
                "trading_days": len(data),
                "latest_price": round(data['close'].iloc[-1], 2),
            }
        }
    
    def _get_activity_grade(self, score: float) -> str:
        """根据评分获取活跃度等级"""
        if score >= 80:
            return "A+"
        elif score >= 70:
            return "A"
        elif score >= 60:
            return "B+"
        elif score >= 50:
            return "B"
        elif score >= 40:
            return "C+"
        elif score >= 30:
            return "C"
        else:
            return "D"


# 使用示例
if __name__ == "__main__":
    analyzer = ActivityAnalyzer()
    
    # 打印活跃度标准
    criteria = analyzer.get_activity_criteria()
    for level, info in criteria.items():
        print(f"\n{info['description']} (评分 >= {info['min_score']}):")
        for criterion in info['criteria']:
            print(f"  - {criterion}")
# -*- coding: utf-8 -*-
"""
市场时间管理器 - 管理交易时间和市场状态
"""

import logging
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum


class MarketStatus(Enum):
    """市场状态"""
    CLOSED = "CLOSED"           # 休市
    PRE_MARKET = "PRE_MARKET"   # 盘前
    OPEN = "OPEN"              # 开盘
    LUNCH_BREAK = "LUNCH_BREAK" # 午休
    CLOSE = "CLOSE"            # 收盘
    POST_MARKET = "POST_MARKET" # 盘后


class MarketTimeManager:
    """市场时间管理器"""
    
    def __init__(self, market_type: str = "A_SHARE"):
        """
        初始化市场时间管理器
        
        Args:
            market_type: 市场类型 (A_SHARE, US_STOCK, HK_STOCK)
        """
        self.market_type = market_type
        self.logger = logging.getLogger(__name__)
        
        # 设置不同市场的交易时间
        self._setup_market_hours()
        
        # 节假日列表（简化版）
        self.holidays = set([
            "2025-01-01",  # 元旦
            "2025-02-10",  # 春节
            "2025-02-11",
            "2025-02-12",
            "2025-02-13",
            "2025-02-14",
            "2025-02-15",
            "2025-02-16",
            "2025-02-17",
            "2025-04-05",  # 清明节
            "2025-05-01",  # 劳动节
            "2025-05-02",
            "2025-05-03",
            "2025-06-09",  # 端午节
            "2025-09-15",  # 中秋节
            "2025-09-16",
            "2025-09-17",
            "2025-10-01",  # 国庆节
            "2025-10-02",
            "2025-10-03",
            "2025-10-04",
            "2025-10-05",
            "2025-10-06",
            "2025-10-07",
        ])
    
    def _setup_market_hours(self):
        """设置市场交易时间"""
        if self.market_type == "A_SHARE":
            # 中国A股市场
            self.market_hours = {
                "morning_open": time(9, 30),
                "morning_close": time(11, 30),
                "afternoon_open": time(13, 0),
                "afternoon_close": time(15, 0),
                "pre_market_start": time(9, 15),  # 集合竞价
                "pre_market_end": time(9, 25),
                "post_market_start": time(15, 0),
                "post_market_end": time(15, 30),
            }
            self.trading_days = [0, 1, 2, 3, 4]  # 周一到周五
            
        elif self.market_type == "US_STOCK":
            # 美股市场
            self.market_hours = {
                "morning_open": time(9, 30),
                "morning_close": time(16, 0),
                "pre_market_start": time(4, 0),
                "pre_market_end": time(9, 30),
                "post_market_start": time(16, 0),
                "post_market_end": time(20, 0),
            }
            self.trading_days = [0, 1, 2, 3, 4]  # 周一到周五
            
        elif self.market_type == "HK_STOCK":
            # 港股市场
            self.market_hours = {
                "morning_open": time(9, 30),
                "morning_close": time(12, 0),
                "afternoon_open": time(13, 0),
                "afternoon_close": time(16, 0),
                "pre_market_start": time(9, 15),
                "pre_market_end": time(9, 30),
                "post_market_start": time(16, 0),
                "post_market_end": time(16, 30),
            }
            self.trading_days = [0, 1, 2, 3, 4]  # 周一到周五
    
    def get_market_status(self, current_time: datetime = None) -> MarketStatus:
        """获取当前市场状态"""
        if current_time is None:
            current_time = datetime.now()
        
        # 检查是否为工作日
        if current_time.weekday() not in self.trading_days:
            return MarketStatus.CLOSED
        
        # 检查是否为节假日
        if current_time.strftime("%Y-%m-%d") in self.holidays:
            return MarketStatus.CLOSED
        
        current_time_only = current_time.time()
        
        # A股市场状态判断
        if self.market_type == "A_SHARE":
            if self.market_hours["pre_market_start"] <= current_time_only < self.market_hours["pre_market_end"]:
                return MarketStatus.PRE_MARKET
            elif self.market_hours["morning_open"] <= current_time_only < self.market_hours["morning_close"]:
                return MarketStatus.OPEN
            elif self.market_hours["morning_close"] <= current_time_only < self.market_hours["afternoon_open"]:
                return MarketStatus.LUNCH_BREAK
            elif self.market_hours["afternoon_open"] <= current_time_only < self.market_hours["afternoon_close"]:
                return MarketStatus.OPEN
            elif self.market_hours["post_market_start"] <= current_time_only < self.market_hours["post_market_end"]:
                return MarketStatus.POST_MARKET
            else:
                return MarketStatus.CLOSED
        
        # 美股市场状态判断
        elif self.market_type == "US_STOCK":
            if self.market_hours["pre_market_start"] <= current_time_only < self.market_hours["pre_market_end"]:
                return MarketStatus.PRE_MARKET
            elif self.market_hours["morning_open"] <= current_time_only < self.market_hours["morning_close"]:
                return MarketStatus.OPEN
            elif self.market_hours["post_market_start"] <= current_time_only < self.market_hours["post_market_end"]:
                return MarketStatus.POST_MARKET
            else:
                return MarketStatus.CLOSED
        
        # 港股市场状态判断
        elif self.market_type == "HK_STOCK":
            if self.market_hours["pre_market_start"] <= current_time_only < self.market_hours["pre_market_end"]:
                return MarketStatus.PRE_MARKET
            elif self.market_hours["morning_open"] <= current_time_only < self.market_hours["morning_close"]:
                return MarketStatus.OPEN
            elif self.market_hours["morning_close"] <= current_time_only < self.market_hours["afternoon_open"]:
                return MarketStatus.LUNCH_BREAK
            elif self.market_hours["afternoon_open"] <= current_time_only < self.market_hours["afternoon_close"]:
                return MarketStatus.OPEN
            elif self.market_hours["post_market_start"] <= current_time_only < self.market_hours["post_market_end"]:
                return MarketStatus.POST_MARKET
            else:
                return MarketStatus.CLOSED
        
        return MarketStatus.CLOSED
    
    def is_market_open(self, current_time: datetime = None) -> bool:
        """检查市场是否开盘"""
        status = self.get_market_status(current_time)
        return status == MarketStatus.OPEN
    
    def is_trading_allowed(self, current_time: datetime = None) -> bool:
        """检查是否允许交易"""
        status = self.get_market_status(current_time)
        return status in [MarketStatus.OPEN, MarketStatus.PRE_MARKET]
    
    def get_next_trading_time(self, current_time: datetime = None) -> Optional[datetime]:
        """获取下一个交易时间"""
        if current_time is None:
            current_time = datetime.now()
        
        # 找到下一个交易日
        next_time = current_time
        for _ in range(10):  # 最多检查10天
            if next_time.weekday() in self.trading_days:
                if next_time.strftime("%Y-%m-%d") not in self.holidays:
                    # 找到有效交易日
                    if self.market_type == "A_SHARE":
                        morning_open = next_time.replace(
                            hour=self.market_hours["morning_open"].hour,
                            minute=self.market_hours["morning_open"].minute,
                            second=0,
                            microsecond=0
                        )
                        if next_time.time() < self.market_hours["morning_open"]:
                            return morning_open
                        elif next_time.time() < self.market_hours["afternoon_open"]:
                            return next_time.replace(
                                hour=self.market_hours["afternoon_open"].hour,
                                minute=self.market_hours["afternoon_open"].minute,
                                second=0,
                                microsecond=0
                            )
            
            # 检查下一天
            next_time += timedelta(days=1)
            next_time = next_time.replace(hour=0, minute=0, second=0, microsecond=0)
        
        return None
    
    def get_trading_calendar(self, start_date: datetime, end_date: datetime) -> List[datetime]:
        """获取交易日历"""
        trading_days = []
        current_date = start_date
        
        while current_date <= end_date:
            if (current_date.weekday() in self.trading_days and 
                current_date.strftime("%Y-%m-%d") not in self.holidays):
                trading_days.append(current_date)
            current_date += timedelta(days=1)
        
        return trading_days
    
    def get_market_hours_info(self) -> Dict:
        """获取市场时间信息"""
        return {
            "market_type": self.market_type,
            "market_hours": {k: v.strftime("%H:%M") for k, v in self.market_hours.items()},
            "trading_days": self.trading_days,
            "holidays_count": len(self.holidays)
        }
    
    def create_trading_session(self, date: datetime = None) -> Dict:
        """创建交易时段信息"""
        if date is None:
            date = datetime.now()
        
        session = {
            "date": date.strftime("%Y-%m-%d"),
            "is_trading_day": self.is_trading_day(date),
            "market_status": self.get_market_status(date),
            "sessions": []
        }
        
        if self.market_type == "A_SHARE":
            session["sessions"] = [
                {
                    "name": "集合竞价",
                    "start": self.market_hours["pre_market_start"].strftime("%H:%M"),
                    "end": self.market_hours["pre_market_end"].strftime("%H:%M"),
                    "type": "PRE_MARKET"
                },
                {
                    "name": "上午交易",
                    "start": self.market_hours["morning_open"].strftime("%H:%M"),
                    "end": self.market_hours["morning_close"].strftime("%H:%M"),
                    "type": "TRADING"
                },
                {
                    "name": "下午交易",
                    "start": self.market_hours["afternoon_open"].strftime("%H:%M"),
                    "end": self.market_hours["afternoon_close"].strftime("%H:%M"),
                    "type": "TRADING"
                }
            ]
        
        return session
    
    def is_trading_day(self, date: datetime) -> bool:
        """检查是否为交易日"""
        return (date.weekday() in self.trading_days and 
                date.strftime("%Y-%m-%d") not in self.holidays)
    
    def get_current_session_info(self, current_time: datetime = None) -> Dict:
        """获取当前交易时段信息"""
        if current_time is None:
            current_time = datetime.now()
        
        status = self.get_market_status(current_time)
        
        info = {
            "current_time": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "market_status": status.value,
            "is_trading_day": self.is_trading_day(current_time),
            "trading_allowed": self.is_trading_allowed(current_time),
            "next_trading_time": None
        }
        
        if not self.is_trading_allowed(current_time):
            next_trading = self.get_next_trading_time(current_time)
            if next_trading:
                info["next_trading_time"] = next_trading.strftime("%Y-%m-%d %H:%M:%S")
        
        return info


# 全局市场时间管理器实例
market_time_manager = MarketTimeManager("A_SHARE")
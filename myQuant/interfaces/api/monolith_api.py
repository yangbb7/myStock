# -*- coding: utf-8 -*-
"""
Monolith API Layer - 模块化单体的API接口层
提供简单、高性能的RESTful API，替代复杂的微服务网关
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from myQuant.core.enhanced_trading_system import EnhancedTradingSystem, SystemConfig, SystemModule
from .websocket_manager import WebSocketManager
from .user_authentication_api import router as auth_router
from .stock_screening_api import router as screening_router


class APIConfig(BaseModel):
    """API配置"""
    title: str = "myQuant Monolith API"
    description: str = "高性能模块化单体量化交易系统API"
    version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    enable_docs: bool = True


class MarketDataRequest(BaseModel):
    """市场数据请求"""
    symbol: str = Field(..., description="股票代码")
    period: str = Field(default="1d", description="周期")
    start_date: Optional[str] = Field(None, description="开始日期")
    end_date: Optional[str] = Field(None, description="结束日期")


class StrategyConfig(BaseModel):
    """策略配置"""
    initial_capital: float = Field(default=100000.0, description="初始资金")
    risk_tolerance: float = Field(default=0.02, description="风险容忍度")
    max_position_size: float = Field(default=0.1, description="最大仓位比例")
    stop_loss: Optional[float] = Field(None, description="止损比例")
    take_profit: Optional[float] = Field(None, description="止盈比例")
    indicators: Dict[str, Any] = Field(default_factory=dict, description="技术指标参数")
    additional_params: Dict[str, Any] = Field(default_factory=dict, description="其他参数")


class StrategyRequest(BaseModel):
    """策略请求"""
    name: str = Field(..., description="策略名称")
    config: StrategyConfig = Field(..., description="策略配置")


class OrderRequest(BaseModel):
    """订单请求"""
    symbol: str = Field(..., description="股票代码")
    side: str = Field(..., description="买卖方向 (BUY/SELL)")
    quantity: int = Field(..., description="数量")
    price: Optional[float] = Field(None, description="价格")
    order_type: str = Field(default="MARKET", description="订单类型")


class TickDataRequest(BaseModel):
    """Tick数据请求"""
    symbol: str = Field(..., description="股票代码")
    price: float = Field(..., description="最新价格")
    volume: int = Field(..., description="成交量")
    timestamp: str = Field(..., description="时间戳")
    bid: Optional[float] = Field(None, description="买一价")
    ask: Optional[float] = Field(None, description="卖一价")
    bid_size: Optional[int] = Field(None, description="买一量")
    ask_size: Optional[int] = Field(None, description="卖一量")


class SystemResponse(BaseModel):
    """系统响应"""
    success: bool = Field(True, description="是否成功")
    message: str = Field("", description="响应消息")
    data: Optional[Dict[str, Any]] = Field(None, description="响应数据")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


# 全局数据提供者实例（避免重复初始化）
_global_provider = None

def get_or_create_provider():
    """获取或创建全局数据提供者实例"""
    global _global_provider
    if _global_provider is None:
        from real_data_config import get_real_data_config
        from myQuant.infrastructure.data.providers.real_data_provider import RealDataProvider
        config = get_real_data_config()
        _global_provider = RealDataProvider(config)
        logging.getLogger(__name__).info("🌐 全局数据提供者初始化完成")
    return _global_provider

class MonolithAPI:
    """模块化单体API"""
    
    def __init__(self, 
                 trading_system: EnhancedTradingSystem,
                 api_config: APIConfig = None):
        self.trading_system = trading_system
        self.api_config = api_config or APIConfig()
        self.logger = logging.getLogger(__name__)
        
        # 创建FastAPI应用
        self.app = FastAPI(
            title=self.api_config.title,
            description=self.api_config.description,
            version=self.api_config.version,
            docs_url="/docs" if self.api_config.enable_docs else None,
            redoc_url="/redoc" if self.api_config.enable_docs else None
        )
        
        # 配置CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # 创建WebSocket管理器
        self.websocket_manager = WebSocketManager(trading_system)
        
        # 注册外部路由
        self.app.include_router(auth_router)
        self.app.include_router(screening_router)
        
        # 注册新的可视化策略API
        from myQuant.interfaces.api.visual_strategy_api import router as visual_strategy_router
        self.app.include_router(visual_strategy_router)
        
        # 注册AI助手API
        from myQuant.interfaces.api.ai_assistant_api import router as ai_assistant_router
        self.app.include_router(ai_assistant_router)
        
        # 注册智能风控API
        from myQuant.interfaces.api.intelligent_risk_api import router as intelligent_risk_router
        self.app.include_router(intelligent_risk_router)
        
        # 注册路由
        self._register_routes()
        
    def _register_routes(self):
        """注册API路由"""
        
        @self.app.get("/health")
        async def health_check():
            """健康检查"""
            health_status = self.trading_system.get_system_health()
            return SystemResponse(
                success=True,
                message="System is healthy",
                data=health_status
            )
        
        @self.app.get("/metrics")
        async def get_metrics():
            """获取系统指标"""
            metrics = self.trading_system.get_system_metrics()
            return SystemResponse(
                success=True,
                message="Metrics retrieved successfully",
                data=metrics
            )
        
        @self.app.post("/system/start")
        async def start_system():
            """启动系统"""
            try:
                await self.trading_system.start()
                return SystemResponse(
                    success=True,
                    message="System started successfully"
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/system/stop")
        async def stop_system():
            """停止系统"""
            try:
                await self.trading_system.stop()
                return SystemResponse(
                    success=True,
                    message="System stopped successfully"
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/system/restart")
        async def restart_system():
            """重启系统"""
            try:
                await self.trading_system.stop()
                await self.trading_system.start()
                return SystemResponse(
                    success=True,
                    message="System restarted successfully"
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # 数据相关API
        @self.app.get("/data/market/{symbol}")
        async def get_market_data(symbol: str, period: str = "1d"):
            """获取市场数据"""
            try:
                if 'data' not in self.trading_system.modules:
                    raise HTTPException(status_code=404, detail="Data module not enabled")
                    
                data = await self.trading_system.modules['data'].get_market_data(symbol, period)
                
                # 检查数据是否为空
                if hasattr(data, 'empty') and data.empty:
                    return SystemResponse(
                        success=False,
                        message=f"无法获取 {symbol} 的真实市场数据",
                        data=None
                    )
                
                # 获取股票信息（简化处理）
                stock_name_map = {
                    "000001.SZ": "平安银行",
                    "000002.SZ": "万科A", 
                    "600000.SH": "浦发银行",
                    "600036.SH": "招商银行"
                }
                stock_name = stock_name_map.get(symbol, symbol)
                
                # Convert DataFrame to dictionary for API response
                if hasattr(data, 'to_dict'):
                    data_dict = {
                        'symbol': symbol,
                        'name': stock_name,
                        'records': data.to_dict('records'),
                        'shape': data.shape,
                        'columns': list(data.columns) if hasattr(data, 'columns') else []
                    }
                else:
                    data_dict = {
                        'symbol': symbol,
                        'name': stock_name,
                        'data': data
                    }
                
                return SystemResponse(
                    success=True,
                    message="Market data retrieved successfully",
                    data=data_dict
                )
            except Exception as e:
                if "无法获取" in str(e) and "真实市场数据" in str(e):
                    return SystemResponse(
                        success=False,
                        message=str(e),
                        data=None
                    )
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/data/tick")
        async def process_tick_data(tick_data: TickDataRequest):
            """处理tick数据"""
            try:
                result = await self.trading_system.process_market_tick(tick_data.dict())
                return SystemResponse(
                    success=True,
                    message="Tick data processed successfully",
                    data=result
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/data/status")
        async def get_data_status():
            """获取数据处理状态"""
            try:
                if 'data' not in self.trading_system.modules:
                    raise HTTPException(status_code=404, detail="Data module not enabled")
                    
                status = self.trading_system.modules['data'].get_processing_status()
                return SystemResponse(
                    success=True,
                    message="Data status retrieved successfully",
                    data=status
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/data/realtime/{symbol}")
        async def get_realtime_price(symbol: str):
            """获取股票实时价格"""
            try:
                # 直接创建并使用数据提供者
                from real_data_config import get_real_data_config
                from myQuant.infrastructure.data.providers.real_data_provider import RealDataProvider
                
                config = get_real_data_config()
                provider = RealDataProvider(config)
                current_price = provider.get_current_price(symbol)
                
                if current_price > 0:
                    # 获取股票名称
                    stock_name_map = {
                        "000001.SZ": "平安银行",
                        "000002.SZ": "万科A", 
                        "600000.SH": "浦发银行",
                        "600036.SH": "招商银行"
                    }
                    stock_name = stock_name_map.get(symbol, symbol)
                    
                    return SystemResponse(
                        success=True,
                        message="Real-time price retrieved successfully",
                        data={
                            "symbol": symbol,
                            "name": stock_name,
                            "current_price": current_price,
                            "timestamp": datetime.now().isoformat(),
                            "source": "EastMoney API"
                        }
                    )
                else:
                    return SystemResponse(
                        success=False,
                        message=f"无法获取 {symbol} 的实时价格",
                        data=None
                    )
                    
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        # 使用模块级全局提供者

        @self.app.post("/data/realtime/batch")
        async def get_realtime_prices_batch(symbols: list[str]):
            """批量获取股票实时价格 - 优化版本（使用全局提供者实例）"""
            try:
                import asyncio
                import time
                
                start_time = time.time()
                
                # 使用全局提供者实例，避免重复初始化
                provider = get_or_create_provider()
                
                # 股票名称映射
                stock_name_map = {
                    "000001.SZ": "平安银行",
                    "000002.SZ": "万科A", 
                    "000003.SZ": "万科B",
                    "600000.SH": "浦发银行",
                    "600036.SH": "招商银行",
                    "600519.SH": "贵州茅台"
                }
                
                # 并发获取所有股票价格
                async def get_single_price(symbol):
                    try:
                        # 在线程池中执行同步操作
                        import concurrent.futures
                        
                        def sync_get_price():
                            return provider.get_current_price(symbol)
                            
                        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                            future = executor.submit(sync_get_price)
                            current_price = future.result(timeout=8)  # 缩短超时时间到8秒
                        
                        if current_price > 0:
                            stock_name = stock_name_map.get(symbol, symbol)
                            return {
                                "symbol": symbol,
                                "name": stock_name,
                                "current_price": current_price,
                                "timestamp": datetime.now().isoformat(),
                                "source": "EastMoney API",
                                "success": True
                            }
                        else:
                            return {
                                "symbol": symbol,
                                "success": False,
                                "error": "Price not available"
                            }
                    except Exception as e:
                        return {
                            "symbol": symbol,
                            "success": False,
                            "error": str(e)
                        }
                
                # 并发获取所有价格，最大并发数为8（增加并发数）
                semaphore = asyncio.Semaphore(8)
                
                async def limited_get_price(symbol):
                    async with semaphore:
                        return await get_single_price(symbol)
                
                # 创建任务列表
                tasks = [limited_get_price(symbol) for symbol in symbols]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # 处理结果
                data = []
                success_count = 0
                
                for result in results:
                    if isinstance(result, Exception):
                        data.append({
                            "symbol": "unknown",
                            "success": False,
                            "error": str(result)
                        })
                    else:
                        data.append(result)
                        if result.get("success"):
                            success_count += 1
                
                end_time = time.time()
                duration = end_time - start_time
                
                self.logger.info(f"批量价格获取完成：{success_count}/{len(symbols)} 成功，耗时 {duration:.2f}s")
                
                return SystemResponse(
                    success=True,
                    message=f"Batch price retrieval completed: {success_count}/{len(symbols)} successful in {duration:.2f}s",
                    data={
                        "results": data,
                        "total": len(symbols),
                        "successful": success_count,
                        "duration": duration,
                        "timestamp": datetime.now().isoformat()
                    }
                )
                    
            except Exception as e:
                self.logger.error(f"批量价格获取失败: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/data/symbols")
        async def get_symbols():
            """获取可用股票代码列表"""
            try:
                if 'data' not in self.trading_system.modules:
                    raise HTTPException(status_code=404, detail="Data module not enabled")
                    
                # 返回测试股票列表
                symbols = ["000001.SZ", "000002.SZ", "600000.SH", "600036.SH"]
                return SystemResponse(
                    success=True,
                    message="Symbols retrieved successfully",
                    data=symbols
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # 策略相关API
        @self.app.post("/strategy/add")
        async def add_strategy(strategy_request: StrategyRequest):
            """添加策略"""
            try:
                if 'strategy' not in self.trading_system.modules:
                    raise HTTPException(status_code=404, detail="Strategy module not enabled")
                    
                strategy_id = await self.trading_system.modules['strategy'].add_strategy(
                    strategy_request.name,
                    strategy_request.config.dict()
                )
                return SystemResponse(
                    success=True,
                    message="Strategy added successfully",
                    data={"strategy_id": strategy_id}
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/strategy/list")
        async def get_strategies():
            """获取策略列表"""
            try:
                if 'strategy' not in self.trading_system.modules:
                    raise HTTPException(status_code=404, detail="Strategy module not enabled")
                    
                strategies = self.trading_system.modules['strategy'].get_strategies()
                return SystemResponse(
                    success=True,
                    message="Strategies retrieved successfully",
                    data=strategies
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/strategy/performance")
        async def get_strategy_performance():
            """获取策略性能"""
            try:
                if 'strategy' not in self.trading_system.modules:
                    raise HTTPException(status_code=404, detail="Strategy module not enabled")
                    
                performance = self.trading_system.modules['strategy'].get_strategy_performance()
                return SystemResponse(
                    success=True,
                    message="Strategy performance retrieved successfully",
                    data=performance
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # 订单相关API
        @self.app.post("/order/create")
        async def create_order(order_request: OrderRequest):
            """创建订单"""
            try:
                if 'execution' not in self.trading_system.modules:
                    raise HTTPException(status_code=404, detail="Execution module not enabled")
                    
                # 构造信号
                signal = {
                    'symbol': order_request.symbol,
                    'side': order_request.side,
                    'quantity': order_request.quantity,
                    'price': order_request.price,
                    'type': order_request.order_type,
                    'timestamp': datetime.now()
                }
                
                order_id = await self.trading_system.modules['execution'].create_order(signal)
                return SystemResponse(
                    success=True,
                    message="Order created successfully",
                    data={"order_id": order_id}
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/order/status/{order_id}")
        async def get_order_status(order_id: str):
            """获取订单状态"""
            try:
                if 'execution' not in self.trading_system.modules:
                    raise HTTPException(status_code=404, detail="Execution module not enabled")
                    
                status = await self.trading_system.modules['execution'].get_order_status(order_id)
                return SystemResponse(
                    success=True,
                    message="Order status retrieved successfully",
                    data=status
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/order/history")
        async def get_order_history(start_date: Optional[str] = None, end_date: Optional[str] = None, page: int = 1, limit: int = 50):
            """获取订单历史"""
            try:
                if 'execution' not in self.trading_system.modules:
                    raise HTTPException(status_code=404, detail="Execution module not enabled")
                    
                orders = await self.trading_system.modules['execution'].get_order_history(
                    start_date=start_date, end_date=end_date, page=page, limit=limit
                )
                return SystemResponse(
                    success=True,
                    message="Order history retrieved successfully",
                    data=orders
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/order/active")
        async def get_active_orders():
            """获取活跃订单"""
            try:
                if 'execution' not in self.trading_system.modules:
                    raise HTTPException(status_code=404, detail="Execution module not enabled")
                    
                orders = await self.trading_system.modules['execution'].get_active_orders()
                return SystemResponse(
                    success=True,
                    message="Active orders retrieved successfully",
                    data=orders
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/order/stats")
        async def get_order_stats(start_date: Optional[str] = None, end_date: Optional[str] = None):
            """获取订单统计"""
            try:
                if 'execution' not in self.trading_system.modules:
                    raise HTTPException(status_code=404, detail="Execution module not enabled")
                    
                stats = await self.trading_system.modules['execution'].get_order_stats(
                    start_date=start_date, end_date=end_date
                )
                return SystemResponse(
                    success=True,
                    message="Order stats retrieved successfully",
                    data=stats
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # 投资组合相关API
        @self.app.get("/portfolio/summary")
        async def get_portfolio_summary():
            """获取投资组合摘要"""
            try:
                if 'portfolio' not in self.trading_system.modules:
                    raise HTTPException(status_code=404, detail="Portfolio module not enabled")
                    
                summary = self.trading_system.modules['portfolio'].get_portfolio_summary()
                return SystemResponse(
                    success=True,
                    message="Portfolio summary retrieved successfully",
                    data=summary
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/portfolio/history")
        async def get_portfolio_history(start_date: Optional[str] = None, end_date: Optional[str] = None):
            """获取投资组合历史"""
            try:
                if 'portfolio' not in self.trading_system.modules:
                    raise HTTPException(status_code=404, detail="Portfolio module not enabled")
                    
                history = self.trading_system.modules['portfolio'].get_portfolio_history(
                    start_date=start_date, end_date=end_date
                )
                return SystemResponse(
                    success=True,
                    message="Portfolio history retrieved successfully",
                    data=history
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/portfolio/positions")
        async def get_portfolio_positions():
            """获取投资组合持仓"""
            try:
                if 'portfolio' not in self.trading_system.modules:
                    raise HTTPException(status_code=404, detail="Portfolio module not enabled")
                    
                positions = self.trading_system.modules['portfolio'].get_positions()
                return SystemResponse(
                    success=True,
                    message="Portfolio positions retrieved successfully",
                    data=positions
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/portfolio/performance")
        async def get_portfolio_performance(start_date: Optional[str] = None, end_date: Optional[str] = None):
            """获取投资组合业绩"""
            try:
                if 'portfolio' not in self.trading_system.modules:
                    raise HTTPException(status_code=404, detail="Portfolio module not enabled")
                    
                performance = self.trading_system.modules['portfolio'].get_performance_metrics(
                    start_date=start_date, end_date=end_date
                )
                return SystemResponse(
                    success=True,
                    message="Portfolio performance retrieved successfully",
                    data=performance
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # 风险相关API
        @self.app.get("/risk/metrics")
        async def get_risk_metrics():
            """获取风险指标"""
            try:
                if 'risk' not in self.trading_system.modules:
                    raise HTTPException(status_code=404, detail="Risk module not enabled")
                    
                metrics = self.trading_system.modules['risk'].get_risk_metrics()
                return SystemResponse(
                    success=True,
                    message="Risk metrics retrieved successfully",
                    data=metrics
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/risk/config")
        async def get_risk_config():
            """获取风险配置"""
            try:
                if 'risk' not in self.trading_system.modules:
                    raise HTTPException(status_code=404, detail="Risk module not enabled")
                    
                config = self.trading_system.modules['risk'].get_risk_config()
                return SystemResponse(
                    success=True,
                    message="Risk config retrieved successfully",
                    data=config
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/risk/alerts")
        async def get_risk_alerts(start_date: Optional[str] = None, end_date: Optional[str] = None):
            """获取风险警报"""
            try:
                if 'risk' not in self.trading_system.modules:
                    raise HTTPException(status_code=404, detail="Risk module not enabled")
                    
                alerts = self.trading_system.modules['risk'].get_risk_alerts(
                    start_date=start_date, end_date=end_date
                )
                return SystemResponse(
                    success=True,
                    message="Risk alerts retrieved successfully",
                    data=alerts
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.put("/risk/limits")
        async def update_risk_limits(limits: dict):
            """更新风险限制"""
            try:
                if 'risk' not in self.trading_system.modules:
                    raise HTTPException(status_code=404, detail="Risk module not enabled")
                    
                result = self.trading_system.modules['risk'].update_risk_limits(limits)
                return SystemResponse(
                    success=True,
                    message="Risk limits updated successfully",
                    data=result
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # 分析相关API
        @self.app.get("/analytics/performance")
        async def get_performance_report(start_date: Optional[str] = None, end_date: Optional[str] = None):
            """获取性能报告"""
            try:
                if 'analytics' not in self.trading_system.modules:
                    raise HTTPException(status_code=404, detail="Analytics module not enabled")
                    
                report = self.trading_system.modules['analytics'].get_performance_report(
                    start_date=start_date, end_date=end_date
                )
                return SystemResponse(
                    success=True,
                    message="Performance report retrieved successfully",
                    data=report
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/analytics/backtest/history")
        async def get_backtest_history():
            """获取回测历史"""
            try:
                if 'analytics' not in self.trading_system.modules:
                    raise HTTPException(status_code=404, detail="Analytics module not enabled")
                    
                history = self.trading_system.modules['analytics'].get_backtest_history()
                return SystemResponse(
                    success=True,
                    message="Backtest history retrieved successfully",
                    data=history
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/analytics/backtest/run")
        async def run_backtest(config: dict):
            """运行回测"""
            try:
                if 'analytics' not in self.trading_system.modules:
                    raise HTTPException(status_code=404, detail="Analytics module not enabled")
                    
                result = await self.trading_system.modules['analytics'].run_backtest(config)
                return SystemResponse(
                    success=True,
                    message="Backtest started successfully",
                    data=result
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/analytics/report/{report_type}")
        async def generate_report(report_type: str, start_date: Optional[str] = None, end_date: Optional[str] = None):
            """生成报告"""
            try:
                if 'analytics' not in self.trading_system.modules:
                    raise HTTPException(status_code=404, detail="Analytics module not enabled")
                    
                report = await self.trading_system.modules['analytics'].generate_report(
                    report_type, start_date=start_date, end_date=end_date
                )
                return SystemResponse(
                    success=True,
                    message="Report generated successfully",
                    data=report
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/analytics/export/{data_type}/{format}")
        async def export_data(data_type: str, format: str, start_date: Optional[str] = None, end_date: Optional[str] = None):
            """导出数据"""
            try:
                if 'analytics' not in self.trading_system.modules:
                    raise HTTPException(status_code=404, detail="Analytics module not enabled")
                    
                result = await self.trading_system.modules['analytics'].export_data(
                    data_type, format, start_date=start_date, end_date=end_date
                )
                return SystemResponse(
                    success=True,
                    message="Data exported successfully",
                    data=result
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # WebSocket相关API
        @self.app.get("/ws/stats")
        async def get_websocket_stats():
            """获取WebSocket统计信息"""
            stats = self.websocket_manager.get_stats()
            return SystemResponse(
                success=True,
                message="WebSocket stats retrieved successfully",
                data=stats
            )
        
        @self.app.get("/stock/info/{symbol}")
        async def get_stock_info(symbol: str):
            """获取股票基本信息"""
            try:
                if 'data' not in self.trading_system.modules:
                    raise HTTPException(status_code=404, detail="Data module not enabled")
                    
                info = await self.trading_system.modules['data'].get_stock_info(symbol)
                return SystemResponse(
                    success=True,
                    message="Stock info retrieved successfully",
                    data=info
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/stock/search")
        async def search_stocks(keyword: str = "", limit: int = 20):
            """搜索股票"""
            try:
                if 'data' not in self.trading_system.modules:
                    raise HTTPException(status_code=404, detail="Data module not enabled")
                    
                results = await self.trading_system.modules['data'].search_stocks(keyword, limit)
                return SystemResponse(
                    success=True,
                    message="Stock search completed successfully",
                    data=results
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/ws/broadcast/market-data")
        async def broadcast_market_data(market_data: dict):
            """广播市场数据"""
            try:
                await self.websocket_manager.broadcast_market_data(market_data)
                return SystemResponse(
                    success=True,
                    message="Market data broadcasted successfully"
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/ws/broadcast/risk-alert")
        async def broadcast_risk_alert(alert_data: dict):
            """广播风险告警"""
            try:
                await self.websocket_manager.broadcast_risk_alert(alert_data)
                return SystemResponse(
                    success=True,
                    message="Risk alert broadcasted successfully"
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # 错误处理
        @self.app.exception_handler(Exception)
        async def global_exception_handler(request: Request, exc: Exception):
            """全局异常处理"""
            self.logger.error(f"API error: {exc}")
            return JSONResponse(
                status_code=500,
                content=SystemResponse(
                    success=False,
                    message=f"Internal server error: {str(exc)}",
                    data=None
                ).dict()
            )
        
        # 请求日志中间件
        @self.app.middleware("http")
        async def log_requests(request: Request, call_next):
            """请求日志中间件"""
            start_time = datetime.now()
            
            # 记录请求
            self.logger.info(f"API Request: {request.method} {request.url}")
            
            response = await call_next(request)
            
            # 记录响应
            process_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"API Response: {response.status_code} in {process_time:.3f}s")
            
            return response
    
    async def startup(self):
        """启动时的初始化"""
        await self.websocket_manager.start_background_tasks()
        self.logger.info("WebSocket manager started")
    
    async def shutdown(self):
        """关闭时的清理"""
        await self.websocket_manager.stop_background_tasks()
        self.logger.info("WebSocket manager stopped")
    
    def run(self):
        """运行API服务器"""
        self.logger.info(f"Starting API server on {self.api_config.host}:{self.api_config.port}")
        
        # 将WebSocket服务挂载到应用
        socket_app = self.websocket_manager.mount_to_app(self.app)
        
        # 添加启动和关闭事件
        @self.app.on_event("startup")
        async def startup_event():
            await self.startup()
        
        @self.app.on_event("shutdown")
        async def shutdown_event():
            await self.shutdown()
        
        if self.api_config.debug:
            # 开发模式：使用导入字符串以支持热重载
            uvicorn.run(
                socket_app,
                host=self.api_config.host,
                port=self.api_config.port,
                reload=False,  # Socket.IO不支持热重载
                access_log=True
            )
        else:
            # 生产模式：直接使用应用实例
            uvicorn.run(
                socket_app,
                host=self.api_config.host,
                port=self.api_config.port,
                reload=False,
                access_log=True
            )


# 创建应用实例的工厂函数
def create_app(system_config: SystemConfig = None, api_config: APIConfig = None) -> FastAPI:
    """创建FastAPI应用实例"""
    
    # 使用默认配置
    if system_config is None:
        system_config = SystemConfig(
            enabled_modules=[
                SystemModule.DATA,
                SystemModule.STRATEGY,
                SystemModule.EXECUTION,
                SystemModule.RISK,
                SystemModule.PORTFOLIO,
                SystemModule.ANALYTICS
            ]
        )
    
    # 创建交易系统
    trading_system = EnhancedTradingSystem(system_config)
    
    # 创建API
    api = MonolithAPI(trading_system, api_config)
    
    return api.app


if __name__ == "__main__":
    # 直接运行时的配置
    logging.basicConfig(level=logging.INFO)
    
    # 创建系统配置
    system_config = SystemConfig(
        enabled_modules=[
            SystemModule.DATA,
            SystemModule.STRATEGY,
            SystemModule.EXECUTION,
            SystemModule.RISK,
            SystemModule.PORTFOLIO,
            SystemModule.ANALYTICS
        ]
    )
    
    # 创建API配置
    api_config = APIConfig(
        port=8000,
        debug=True
    )
    
    # 创建并运行API
    trading_system = EnhancedTradingSystem(system_config)
    api = MonolithAPI(trading_system, api_config)
    api.run()
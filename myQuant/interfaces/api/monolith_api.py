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

from core.enhanced_trading_system import EnhancedTradingSystem, SystemConfig, SystemModule


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
        
        # 数据相关API
        @self.app.get("/data/market/{symbol}")
        async def get_market_data(symbol: str, period: str = "1d"):
            """获取市场数据"""
            try:
                if 'data' not in self.trading_system.modules:
                    raise HTTPException(status_code=404, detail="Data module not enabled")
                    
                data = await self.trading_system.modules['data'].get_market_data(symbol, period)
                
                # Convert DataFrame to dictionary for API response
                if hasattr(data, 'to_dict'):
                    data_dict = {
                        'records': data.to_dict('records'),
                        'shape': data.shape,
                        'columns': list(data.columns) if hasattr(data, 'columns') else []
                    }
                else:
                    data_dict = data
                
                return SystemResponse(
                    success=True,
                    message="Market data retrieved successfully",
                    data=data_dict
                )
            except Exception as e:
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
        
        # 分析相关API
        @self.app.get("/analytics/performance")
        async def get_performance_report():
            """获取性能报告"""
            try:
                if 'analytics' not in self.trading_system.modules:
                    raise HTTPException(status_code=404, detail="Analytics module not enabled")
                    
                report = self.trading_system.modules['analytics'].get_performance_report()
                return SystemResponse(
                    success=True,
                    message="Performance report retrieved successfully",
                    data=report
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
    
    def run(self):
        """运行API服务器"""
        self.logger.info(f"Starting API server on {self.api_config.host}:{self.api_config.port}")
        
        if self.api_config.debug:
            # 开发模式：使用导入字符串以支持热重载
            uvicorn.run(
                "myQuant.interfaces.api.monolith_api:create_app",
                host=self.api_config.host,
                port=self.api_config.port,
                reload=True,
                access_log=True,
                factory=True
            )
        else:
            # 生产模式：直接使用应用实例
            uvicorn.run(
                self.app,
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
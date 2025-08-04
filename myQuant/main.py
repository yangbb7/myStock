"""
myQuant FastAPI应用程序入口

提供REST API接口，包括用户认证、股票筛选、投资组合管理、风险管理等功能
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from interfaces.api.user_authentication_api import router as auth_router
from interfaces.api.stock_screening_api import router as screening_router
from interfaces.api.portfolio_api import router as portfolio_router
from interfaces.api.risk_management_api import router as risk_router
from interfaces.api.market_data_api import router as market_data_router
from interfaces.api.portfolio_performance_api import router as performance_router

# 创建FastAPI应用程序
app = FastAPI(
    title="myQuant量化交易系统",
    description="专业的量化交易系统API，提供完整的交易、分析和风险管理功能",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(auth_router)
app.include_router(screening_router)
app.include_router(portfolio_router)
app.include_router(risk_router)
app.include_router(market_data_router)
app.include_router(performance_router)

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "欢迎使用myQuant量化交易系统",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc"
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
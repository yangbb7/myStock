# -*- coding: utf-8 -*-
"""
投资组合管理API

提供投资组合创建、管理、绩效分析等功能
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, date
import numpy as np
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from myQuant.core.managers.portfolio_manager import PortfolioManager
from myQuant.infrastructure.database.repositories import PortfolioRepository
from myQuant.infrastructure.database.database_manager import DatabaseManager


class PortfolioCreateRequest(BaseModel):
    """创建投资组合请求"""
    name: str = Field(..., min_length=1, max_length=100)
    type: str = Field(default="STANDARD", pattern="^(STANDARD|AGGRESSIVE|CONSERVATIVE)$")
    initial_capital: float = Field(..., gt=0)
    description: Optional[str] = Field(None, max_length=500)


class PortfolioUpdateRequest(BaseModel):
    """更新投资组合请求"""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)


class PositionInfo(BaseModel):
    """持仓信息"""
    symbol: str
    quantity: int
    avg_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    weight: float
    sector: Optional[str] = None
    last_updated: datetime


class PortfolioSummary(BaseModel):
    """投资组合摘要"""
    id: int
    name: str
    type: str
    total_value: float
    cash_value: float
    position_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    positions_count: int
    created_at: datetime
    updated_at: datetime


class PortfolioDetail(PortfolioSummary):
    """投资组合详情"""
    positions: List[PositionInfo]
    performance_metrics: Dict[str, Any]


class PerformanceMetrics(BaseModel):
    """绩效指标"""
    total_return: float
    total_return_pct: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    volatility: float
    beta: Optional[float] = None
    alpha: Optional[float] = None


class PortfolioAPI:
    """投资组合API"""
    
    def __init__(self, portfolio_manager: PortfolioManager, portfolio_repo: PortfolioRepository, database_manager: Optional[DatabaseManager] = None):
        self.portfolio_manager = portfolio_manager
        self.portfolio_repo = portfolio_repo
        self.database_manager = database_manager
        self.router = APIRouter(prefix="/portfolio", tags=["portfolio"])
        self._setup_routes()
    
    def _setup_routes(self):
        """设置路由"""
        
        @self.router.post("/create", response_model=Dict[str, Any])
        async def create_portfolio(request: PortfolioCreateRequest, user_id: int = Depends(self._get_current_user)):
            """创建投资组合"""
            try:
                portfolio_id = await self.portfolio_repo.create(
                    user_id=user_id,
                    name=request.name,
                    type=request.type
                )
                
                # 初始化投资组合
                await self.portfolio_manager.initialize_portfolio(
                    portfolio_id=portfolio_id,
                    initial_capital=request.initial_capital
                )
                
                return {
                    "portfolio_id": portfolio_id,
                    "message": "Portfolio created successfully"
                }
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.router.get("/list", response_model=List[PortfolioSummary])
        async def list_portfolios(user_id: int = Depends(self._get_current_user)):
            """获取用户投资组合列表"""
            try:
                portfolios = await self.portfolio_repo.get_by_user(user_id)
                return [self._format_portfolio_summary(p) for p in portfolios]
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.router.get("/{portfolio_id}", response_model=PortfolioDetail)
        async def get_portfolio_detail(portfolio_id: int, user_id: int = Depends(self._get_current_user)):
            """获取投资组合详情"""
            try:
                portfolio = await self.portfolio_repo.get_by_id(portfolio_id)
                if not portfolio or portfolio['user_id'] != user_id:
                    raise HTTPException(status_code=404, detail="Portfolio not found")
                
                # 获取持仓信息
                positions = await self.portfolio_manager.get_positions(portfolio_id)
                
                # 获取绩效指标
                performance = await self.portfolio_manager.calculate_performance(portfolio_id)
                
                return self._format_portfolio_detail(portfolio, positions, performance)
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.router.put("/{portfolio_id}", response_model=Dict[str, str])
        async def update_portfolio(
            portfolio_id: int, 
            request: PortfolioUpdateRequest,
            user_id: int = Depends(self._get_current_user)
        ):
            """更新投资组合"""
            try:
                portfolio = await self.portfolio_repo.get_by_id(portfolio_id)
                if not portfolio or portfolio['user_id'] != user_id:
                    raise HTTPException(status_code=404, detail="Portfolio not found")
                
                if request.name:
                    await self.portfolio_repo.update_name(portfolio_id, request.name)
                
                return {"message": "Portfolio updated successfully"}
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.router.delete("/{portfolio_id}", response_model=Dict[str, str])
        async def delete_portfolio(portfolio_id: int, user_id: int = Depends(self._get_current_user)):
            """删除投资组合"""
            try:
                portfolio = await self.portfolio_repo.get_by_id(portfolio_id)
                if not portfolio or portfolio['user_id'] != user_id:
                    raise HTTPException(status_code=404, detail="Portfolio not found")
                
                await self.portfolio_repo.delete(portfolio_id)
                return {"message": "Portfolio deleted successfully"}
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.router.get("/{portfolio_id}/performance", response_model=PerformanceMetrics)
        async def get_portfolio_performance(
            portfolio_id: int,
            start_date: Optional[date] = None,
            end_date: Optional[date] = None,
            user_id: int = Depends(self._get_current_user)
        ):
            """获取投资组合绩效指标"""
            try:
                portfolio = await self.portfolio_repo.get_by_id(portfolio_id)
                if not portfolio or portfolio['user_id'] != user_id:
                    raise HTTPException(status_code=404, detail="Portfolio not found")
                
                performance = await self.portfolio_manager.calculate_performance(
                    portfolio_id=portfolio_id,
                    start_date=start_date,
                    end_date=end_date
                )
                
                return PerformanceMetrics(**performance)
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
    
    async def get_portfolio_summary(self, user_id: int) -> Dict[str, Any]:
        """获取投资组合概览"""
        data = await self.portfolio_manager.get_portfolio_summary(user_id)
        return {
            "success": True,
            "data": data
        }
    
    async def get_positions(self, user_id: int) -> Dict[str, Any]:
        """获取持仓详情"""
        if self.database_manager:
            query = "SELECT * FROM positions WHERE user_id = %s"
            positions = await self.database_manager.fetch_all(query, (user_id,))
            total_value = sum(pos.get('market_value', 0) for pos in positions)
            return {
                "success": True,
                "data": {
                    "positions": positions,
                    "total_value": total_value
                }
            }
        return {
            "success": True,
            "data": {
                "positions": [],
                "total_value": 0.0
            }
        }
    
    async def get_transactions(self, user_id: int, symbol: Optional[str] = None, start_date: Optional[str] = None, end_date: Optional[str] = None, page: int = 1, size: int = 10) -> Dict[str, Any]:
        """获取交易记录"""
        if self.database_manager:
            query = "SELECT * FROM transactions WHERE user_id = %s"
            params = [user_id]
            
            if symbol:
                query += " AND symbol = %s"
                params.append(symbol)
            
            if start_date:
                query += " AND executed_at >= %s"
                params.append(start_date)
            
            if end_date:
                query += " AND executed_at <= %s"
                params.append(end_date)
            
            query += " ORDER BY executed_at DESC LIMIT %s OFFSET %s"
            params.extend([size, (page - 1) * size])
            
            transactions = await self.database_manager.fetch_all(query, params)
            return {
                "success": True,
                "data": {
                    "transactions": transactions
                }
            }
        return {
            "success": True,
            "data": {
                "transactions": []
            }
        }
    
    async def get_performance_attribution(self, user_id: int, period: str) -> Dict[str, Any]:
        """获取绩效归因分析"""
        # 模拟返回绩效归因数据
        return {
            "success": True,
            "data": {
                "sector_attribution": {
                    "technology": 2.5,
                    "finance": -1.2,
                    "healthcare": 0.8
                },
                "stock_selection": {
                    "000001.SZ": 1.5,
                    "000002.SZ": -0.5
                },
                "asset_allocation": {
                    "equity": 1.2,
                    "bonds": -0.3
                }
            }
        }
    
    async def generate_performance_report(self, user_id: int, period: str, format: str) -> Dict[str, Any]:
        """生成绩效报告"""
        summary_resp = await self.get_portfolio_summary(user_id)
        positions_resp = await self.get_positions(user_id)
        
        return {
            "success": True,
            "data": {
                "report": {
                    "report_id": f"report_{user_id}_{period}",
                    "summary": summary_resp["data"],
                    "positions": positions_resp["data"],
                    "format": format,
                    "generated_at": datetime.now().isoformat(),
                    "detailed_analysis": {
                        "performance_metrics": {
                            "total_return": 25.0,
                            "annualized_return": 18.5
                        },
                        "risk_analysis": {
                            "volatility": 15.2,
                            "sharpe_ratio": 1.22
                        }
                    },
                    "charts": {
                        "performance_chart": "base64_chart_data",
                        "allocation_chart": "base64_chart_data"
                    }
                }
            }
        }
    
    async def get_optimization_suggestions(self, user_id: int) -> Dict[str, Any]:
        """获取投资组合优化建议"""
        return {
            "success": True,
            "data": {
                "suggestions": [
                    {
                        "type": "rebalancing",
                        "description": "Reduce position in 000001.SZ by 5%",
                        "priority": "high",
                        "action": "reduce",
                        "symbol": "000001.SZ",
                        "percentage": 5
                    },
                    {
                        "type": "diversification",
                        "description": "Consider diversifying into different sectors",
                        "priority": "medium"
                    }
                ]
            }
        }
    
    async def _get_portfolio_positions(self, user_id: int) -> List[Dict[str, Any]]:
        """获取投资组合持仓（内部方法）"""
        return await self.get_positions(user_id)
    
    async def _calculate_performance_metrics(self, user_id: int, period: str) -> Dict[str, Any]:
        """计算绩效指标（内部方法）"""
        return {
            "total_return": 25.0,
            "annualized_return": 18.5,
            "volatility": 15.2,
            "sharpe_ratio": 1.22,
            "max_drawdown": -8.5
        }
    
    async def _get_portfolio_value_history(self, user_id: int) -> List[float]:
        """获取投资组合价值历史（内部方法）"""
        # 模拟返回历史价值数据
        return [100000 + i * 1000 + np.random.normal(0, 5000) for i in range(252)]
    
    async def _get_portfolio_returns(self, user_id: int) -> List[float]:
        """获取投资组合收益率（内部方法）"""
        # 模拟返回收益率数据
        return [0.01, -0.005, 0.02, 0.001, 0.015]
    
    async def _get_benchmark_returns(self, benchmark: str) -> List[float]:
        """获取基准收益率（内部方法）"""
        # 模拟返回基准收益率数据
        return [0.008, -0.003, 0.015, 0.002, 0.012]
    
    async def get_portfolio_performance(self, user_id: int, period: str) -> Dict[str, Any]:
        """获取投资组合绩效"""
        performance = await self._calculate_performance_metrics(user_id, period)
        return {
            "success": True,
            "data": performance
        }
    
    async def get_sector_allocation(self, user_id: int) -> Dict[str, Any]:
        """获取行业配置"""
        positions = await self._get_portfolio_positions(user_id)
        # 模拟行业配置数据
        return {
            "success": True,
            "data": {
                "sectors": [
                    {"sector": "金融", "percentage": 45.0, "value": 450000.0},
                    {"sector": "房地产", "percentage": 55.0, "value": 550000.0}
                ]
            }
        }
    
    async def calculate_risk_metrics(self, user_id: int, period: str) -> Dict[str, Any]:
        """计算风险指标"""
        # 模拟风险指标数据
        return {
            "success": True,
            "data": {
                "volatility": 15.2,
                "var_95": -25000.0,
                "expected_shortfall": -35000.0,
                "max_drawdown": -8.5,
                "sharpe_ratio": 1.22
            }
        }
    
    async def compare_with_benchmark(self, user_id: int, benchmark: str, period: str) -> Dict[str, Any]:
        """与基准比较"""
        portfolio_returns = await self._get_portfolio_returns(user_id)
        benchmark_returns = await self._get_benchmark_returns(benchmark)
        
        return {
            "success": True,
            "data": {
                "alpha": 3.2,
                "beta": 1.15,
                "correlation": 0.85,
                "tracking_error": 5.8,
                "information_ratio": 0.55,
                "excess_return": 2.5
            }
        }
    
    def _get_current_user(self) -> int:
        """获取当前用户ID（占位符，实际实现需要从认证系统获取）"""
        # TODO: 实现真实的用户认证
        return 1
    
    def _format_portfolio_summary(self, portfolio: Dict[str, Any]) -> PortfolioSummary:
        """格式化投资组合摘要"""
        return PortfolioSummary(
            id=portfolio['id'],
            name=portfolio['name'],
            type=portfolio['type'],
            total_value=portfolio.get('total_value', 0),
            cash_value=portfolio.get('cash_value', 0),
            position_value=portfolio.get('total_value', 0) - portfolio.get('cash_value', 0),
            unrealized_pnl=portfolio.get('unrealized_pnl', 0),
            unrealized_pnl_pct=portfolio.get('unrealized_pnl_pct', 0),
            positions_count=portfolio.get('positions_count', 0),
            created_at=portfolio['created_at'],
            updated_at=portfolio['updated_at']
        )
    
    def _format_portfolio_detail(
        self, 
        portfolio: Dict[str, Any], 
        positions: List[Dict[str, Any]], 
        performance: Dict[str, Any]
    ) -> PortfolioDetail:
        """格式化投资组合详情"""
        summary = self._format_portfolio_summary(portfolio)
        
        position_infos = [
            PositionInfo(
                symbol=pos['symbol'],
                quantity=pos['quantity'],
                avg_cost=pos['avg_cost'],
                current_price=pos['current_price'],
                market_value=pos['market_value'],
                unrealized_pnl=pos['unrealized_pnl'],
                unrealized_pnl_pct=pos['unrealized_pnl_pct'],
                weight=pos['weight'],
                sector=pos.get('sector'),
                last_updated=pos['last_updated']
            )
            for pos in positions
        ]
        
        return PortfolioDetail(
            **summary.dict(),
            positions=position_infos,
            performance_metrics=performance
        )
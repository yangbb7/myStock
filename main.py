#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
myStock - 量化交易系统主入口
A comprehensive quantitative trading framework for Chinese stock market

使用方式:
    python main.py                    # 启动交互式界面
    python main.py --backtest        # 运行回测
    python main.py --live            # 实时交易模式
    python main.py --config config.yaml  # 使用自定义配置
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# 导入myQuant模块
from myQuant import (
    setup_logging,
    create_default_config,
    get_version,
    MAStrategy,
)
from myQuant.core.enhanced_trading_system import EnhancedTradingSystem, SystemConfig, SystemModule
from myQuant.core.engines.async_data_engine import AsyncDataEngine
from myQuant.interfaces.api.monolith_api import MonolithAPI, APIConfig


def create_production_system_config() -> SystemConfig:
    """创建生产环境系统配置"""
    return SystemConfig(
        # 基础配置
        initial_capital=1000000.0,
        commission_rate=0.0003,
        min_commission=5.0,
        
        # 性能配置
        max_concurrent_orders=50,
        order_timeout=10.0,
        data_buffer_size=1000,
        
        # 风险管理
        max_position_size=0.1,
        max_drawdown_limit=0.2,
        max_daily_loss=0.05,
        
        # 启用所有模块
        enabled_modules=[
            SystemModule.DATA,
            SystemModule.STRATEGY,
            SystemModule.EXECUTION,
            SystemModule.RISK,
            SystemModule.PORTFOLIO,
            SystemModule.ANALYTICS
        ],
        
        # 数据库配置
        database_url="sqlite:///data/myquant.db",
        enable_persistence=True,
        
        # 监控配置
        enable_metrics=True,
        metrics_port=8080,
        
        # 日志配置
        log_level="INFO",
        log_file="logs/trading_system.log"
    )


class MyQuantCLI:
    """myQuant命令行接口"""

    def __init__(self) -> None:
        self.config = create_default_config()
        self.logger = setup_logging()
        self.trading_system: Optional[EnhancedTradingSystem] = None

    def run_backtest(self, config_file: Optional[str] = None) -> None:
        """运行回测"""
        print("🚀 启动回测...")
        
        # 初始化交易系统
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
        self.trading_system = EnhancedTradingSystem(system_config)
        
        # 添加策略
        ma_strategy = MAStrategy(
            name="BacktestMAStrategy",
            symbols=["000001.SZ", "000002.SZ"],
            params={"short_window": 5, "long_window": 20},
        )
        self.trading_system.add_strategy(ma_strategy)
        
        print("✅ 回测配置完成")

    async def run_async_data_engine(self) -> None:
        """运行异步数据引擎"""
        print("🌐 启动异步数据引擎...")

        config = {
            "max_concurrent_requests": 5,
            "request_timeout": 30,
            "cache_ttl": 300,
        }

        async with AsyncDataEngine(config) as engine:
            # 健康检查
            health = await engine.health_check()
            print(f"💊 健康状态: {health.get('status', 'unknown')}")

            # 获取市场数据
            symbols = ["000001.SZ", "000002.SZ", "600000.SH"]
            print(f"📊 获取数据: {symbols}")

            async for data in engine.fetch_market_data(symbols):
                if "error" not in data:
                    print(f"✅ 获取 {data.get('symbol', 'unknown')} 数据成功")
                else:
                    print(f"❌ 获取数据失败: {data.get('error', 'unknown')}")

    def run_live_trading(self) -> None:
        """运行实时交易"""
        print("🔴 启动实时交易...")
        
        # 初始化交易系统
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
        self.trading_system = EnhancedTradingSystem(system_config)

        # 创建策略
        ma_strategy = MAStrategy(
            name="LiveMAStrategy",
            symbols=["000001.SZ", "000002.SZ"],
            params={"short_window": 5, "long_window": 20},
        )
        self.trading_system.add_strategy(ma_strategy)

        # 开盘前准备
        self.trading_system.pre_market_setup()
        print("✅ 实时交易系统已启动")

    def run_api_server(self, production: bool = False) -> None:
        """启动API服务器"""
        mode_str = "生产" if production else "开发"
        print(f"🌐 启动API服务器 ({mode_str}模式)...")
        
        # 创建系统配置
        if production:
            system_config = create_production_system_config()
            # 创建必要目录
            os.makedirs("logs", exist_ok=True)
            os.makedirs("data", exist_ok=True)
        else:
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
            title=f"myQuant 交易系统 ({mode_str}环境)",
            description="高性能、低延迟的量化交易系统" if production else "开发和测试环境",
            port=8000,
            debug=not production,
            enable_docs=True
        )
        
        try:
            # 创建交易系统
            trading_system = EnhancedTradingSystem(system_config)
            
            # 创建API实例
            api = MonolithAPI(trading_system, api_config)
            
            print("✅ API服务器配置完成")
            print(f"🌐 访问地址: http://localhost:8000")
            print(f"📚 API文档: http://localhost:8000/docs")
            print("按 Ctrl+C 停止服务器")
            
            # 启动服务器
            api.run()
            
        except KeyboardInterrupt:
            print("\n🛑 API服务器已停止")
        except Exception as e:
            print(f"❌ API服务器启动失败: {e}")
            self.logger.error(f"API服务器启动失败: {e}")

    def run_interactive_mode(self) -> None:
        """运行交互式模式"""
        print(f"🎯 myQuant v{get_version()} 交互式模式")
        print("=" * 50)

        while True:
            print("\n请选择操作:")
            print("1. 运行回测")
            print("2. 运行异步数据引擎") 
            print("3. 运行实时交易")
            print("4. 启动API服务器 (开发模式)")
            print("5. 启动API服务器 (生产模式)")
            print("6. 查看系统状态")
            print("0. 退出")

            choice = input("\n请输入选择 (0-6): ").strip()

            if choice == "0":
                print("👋 再见!")
                break
            elif choice == "1":
                self.run_backtest()
            elif choice == "2":
                try:
                    asyncio.run(self.run_async_data_engine())
                except Exception as e:
                    print(f"❌ 异步数据引擎失败: {e}")
            elif choice == "3":
                self.run_live_trading()
            elif choice == "4":
                self.run_api_server(production=False)
            elif choice == "5":
                self.run_api_server(production=True)
            elif choice == "6":
                self.show_system_status()
            else:
                print("❌ 无效选择，请重试")

    def show_system_status(self) -> None:
        """显示系统状态"""
        print("\n💻 系统状态:")
        print(f"  Python版本: {sys.version}")
        print(f"  myQuant版本: {get_version()}")
        print(f"  工作目录: {os.getcwd()}")

        if self.trading_system:
            status = self.trading_system.get_system_status()
            print(f"  运行状态: {'运行中' if status.get('is_running') else '已停止'}")
            print(f"  策略数量: {status.get('strategies_count', 0)}")
            print(f"  投资组合价值: ¥{status.get('portfolio_value', 0):,.2f}")
        else:
            print("  交易系统: 未初始化")


def main() -> None:
    """主函数"""
    parser = argparse.ArgumentParser(
        description="myQuant - 量化交易系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python main.py                    # 启动交互式界面
  python main.py --backtest        # 运行回测
  python main.py --live            # 实时交易
  python main.py --async-data      # 异步数据引擎
  python main.py --api-server      # 启动API服务器 (开发模式)
  python main.py --production      # 启动API服务器 (生产模式)
  python main.py --version         # 显示版本信息
        """,
    )

    parser.add_argument(
        "--version", action="version", version=f"myQuant {get_version()}"
    )
    parser.add_argument("--backtest", action="store_true", help="运行回测")
    parser.add_argument("--live", action="store_true", help="运行实时交易")
    parser.add_argument("--async-data", action="store_true", help="运行异步数据引擎")
    parser.add_argument("--api-server", action="store_true", help="启动API服务器 (开发模式)")
    parser.add_argument("--production", action="store_true", help="启动API服务器 (生产模式)")
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别",
    )
    parser.add_argument("--log-file", type=str, help="日志文件路径")

    args = parser.parse_args()

    # 初始化CLI
    cli = MyQuantCLI()

    # 设置日志级别
    if args.log_level or args.log_file:
        setup_logging(level=args.log_level, log_file=args.log_file)

    # 显示欢迎信息
    print("🎯 myQuant - 量化交易系统")
    print(f"版本: {get_version()}")
    print("=" * 50)

    try:
        if args.backtest:
            cli.run_backtest(args.config)
        elif args.live:
            cli.run_live_trading()
        elif args.async_data:
            asyncio.run(cli.run_async_data_engine())
        elif args.api_server:
            cli.run_api_server(production=False)
        elif args.production:
            cli.run_api_server(production=True)
        else:
            # 默认启动交互式模式
            cli.run_interactive_mode()

    except KeyboardInterrupt:
        print("\n\n👋 用户中断，程序退出")
    except Exception as e:
        print(f"\n❌ 程序异常: {e}")
        setup_logging().error(f"程序异常: {e}")
        sys.exit(1)


# Export app for testing
def create_app():
    """Create FastAPI app for testing"""
    from myQuant.interfaces.api.monolith_api import MonolithAPI, APIConfig
    from myQuant.core.enhanced_trading_system import EnhancedTradingSystem, SystemConfig, SystemModule
    
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
    
    api_config = APIConfig(
        title="myQuant Test Environment",
        description="Test environment for myQuant",
        port=8000,
        debug=True,
        enable_docs=True
    )
    
    trading_system = EnhancedTradingSystem(system_config)
    api = MonolithAPI(trading_system, api_config)
    return api.app

app = create_app()

if __name__ == "__main__":
    main()
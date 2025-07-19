# -*- coding: utf-8 -*-
"""
事件驱动架构演示脚本
"""

import asyncio
import logging
from datetime import datetime

# 导入事件系统组件
from core.events.event_bus import EventBus, get_event_bus
from core.events.event_handlers import (EventHandlerRegistry, event_handler,
                                        get_handler_registry)
from core.events.event_monitor import EventMonitor, get_event_monitor
from core.events.event_types import (MarketDataEvent, OrderEvent, RiskEvent,
                                     StrategyEvent, SystemEvent, TradeEvent,
                                     create_event)
# 导入数据管理器
from core.managers.data_manager import DataManager

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


class TradingSystemDemo:
    """交易系统演示"""

    def __init__(self):
        self.event_bus = get_event_bus()
        self.handler_registry = get_handler_registry()
        self.event_monitor = get_event_monitor()
        self.data_manager = DataManager({"enable_events": True})

        # 注册自定义事件处理器
        self._register_custom_handlers()

    def _register_custom_handlers(self):
        """注册自定义事件处理器"""

        # 策略信号处理器
        @event_handler(["strategy", "signal"], name="strategy_processor")
        async def strategy_signal_handler(event):
            """处理策略信号"""
            if event.type == "strategy":
                strategy_id = event.data.get("strategy_id", "unknown")
                action = event.data.get("action", "unknown")
                logger.info(f"策略事件: {strategy_id} - {action}")

                if action == "signal":
                    # 生成订单事件
                    order_data = {
                        "symbol": "AAPL",
                        "side": "buy",
                        "quantity": 100,
                        "price": 150.0,
                        "order_type": "limit",
                    }

                    order_event = OrderEvent(
                        order_id=f"order_{int(datetime.now().timestamp())}",
                        order_data=order_data,
                        action="create",
                        source="strategy_processor",
                    )

                    await get_event_bus().publish(order_event)

        # 订单执行处理器
        @event_handler(["order"], name="order_executor")
        async def order_execution_handler(event):
            """处理订单执行"""
            order_id = event.data.get("order_id", "unknown")
            action = event.data.get("action", "unknown")
            order_data = event.data.get("order_data", {})

            logger.info(f"订单事件: {order_id} - {action}")

            if action == "create":
                # 模拟订单执行
                await asyncio.sleep(0.1)  # 模拟执行延迟

                # 生成交易执行事件
                trade_data = {
                    "symbol": order_data.get("symbol", "UNKNOWN"),
                    "side": order_data.get("side", "buy"),
                    "quantity": order_data.get("quantity", 0),
                    "price": order_data.get("price", 0),
                    "commission": 5.0,
                    "timestamp": datetime.now().isoformat(),
                }

                trade_event = TradeEvent(
                    trade_id=f"trade_{int(datetime.now().timestamp())}",
                    order_id=order_id,
                    trade_data=trade_data,
                    source="order_executor",
                )

                await get_event_bus().publish(trade_event)

        # 风险管理处理器
        @event_handler(["trade"], name="risk_manager")
        async def risk_management_handler(event):
            """风险管理处理器"""
            trade_data = event.data.get("trade_data", {})
            symbol = trade_data.get("symbol", "UNKNOWN")
            quantity = trade_data.get("quantity", 0)

            logger.info(f"风险检查: {symbol} 数量 {quantity}")

            # 简单的风险检查逻辑
            if quantity > 500:
                # 触发风险告警
                risk_event = RiskEvent(
                    risk_type="position_limit",
                    risk_data={
                        "symbol": symbol,
                        "quantity": quantity,
                        "limit": 500,
                        "message": f"持仓数量 {quantity} 超过限制 500",
                    },
                    severity="warning",
                    source="risk_manager",
                )

                await get_event_bus().publish(risk_event)

        # 市场数据处理器
        @event_handler(["market_data"], name="market_data_processor")
        async def market_data_handler(event):
            """市场数据处理器"""
            symbol = event.data.get("symbol", "UNKNOWN")
            price_data = event.data.get("price_data", {})
            data_count = price_data.get("data_count", 0)

            logger.info(f"市场数据更新: {symbol} ({data_count} 条记录)")

            # 检查数据质量
            if data_count == 0:
                logger.warning(f"警告: {symbol} 没有数据")

    async def start(self):
        """启动演示系统"""
        logger.info("启动事件驱动交易系统演示...")

        # 启动事件系统组件
        await self.event_bus.start()
        await self.handler_registry.start_all_handlers()
        await self.event_monitor.start()

        logger.info("事件系统已启动")

    async def stop(self):
        """停止演示系统"""
        logger.info("停止事件驱动交易系统...")

        await self.handler_registry.stop_all_handlers()
        await self.event_monitor.stop()
        await self.event_bus.stop()

        logger.info("事件系统已停止")

    async def run_demo(self):
        """运行演示"""
        try:
            await self.start()

            logger.info("开始演示...")

            # 1. 发布系统启动事件
            startup_event = SystemEvent(
                system_data={"component": "trading_system", "version": "1.0"},
                action="startup",
            )
            await self.event_bus.publish(startup_event)

            # 2. 模拟数据获取（会触发市场数据事件）
            logger.info("获取市场数据...")
            data = self.data_manager.get_price_data("AAPL", "2023-01-01", "2023-01-05")
            logger.info(f"获取到 {len(data)} 条数据")

            # 3. 发布策略信号事件
            strategy_event = StrategyEvent(
                strategy_id="ma_crossover",
                strategy_data={
                    "signal": "buy",
                    "confidence": 0.85,
                    "reasons": ["ma_crossover", "volume_spike"],
                },
                action="signal",
            )
            await self.event_bus.publish(strategy_event)

            # 4. 模拟大量交易（触发风险告警）
            large_order_event = OrderEvent(
                order_id="large_order_001",
                order_data={
                    "symbol": "AAPL",
                    "side": "buy",
                    "quantity": 1000,  # 超过风险限制
                    "price": 150.0,
                    "order_type": "market",
                },
                action="create",
            )
            await self.event_bus.publish(large_order_event)

            # 等待事件处理
            await asyncio.sleep(2)

            # 5. 显示监控报告
            logger.info("\n=== 事件监控报告 ===")
            monitor_report = self.event_monitor.generate_monitoring_report()
            logger.info(f"监控报告: {monitor_report}")

            # 6. 显示事件总线统计
            logger.info("\n=== 事件总线统计 ===")
            bus_stats = self.event_bus.get_stats()
            logger.info(f"事件总线统计: {bus_stats}")

            # 7. 显示处理器统计
            logger.info("\n=== 处理器统计 ===")
            handler_stats = self.handler_registry.get_stats()
            logger.info(f"处理器统计: {handler_stats}")

            logger.info("演示完成!")

        except Exception as e:
            logger.error(f"演示过程中发生错误: {e}")
            raise
        finally:
            await self.stop()


async def main():
    """主函数"""
    demo = TradingSystemDemo()
    await demo.run_demo()


if __name__ == "__main__":
    asyncio.run(main())

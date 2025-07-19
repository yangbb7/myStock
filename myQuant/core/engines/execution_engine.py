# -*- coding: utf-8 -*-
"""
ExecutionEngine - 执行引擎模块
负责订单执行、成交管理和实时交易处理
"""

import logging
import threading
import time
from collections import defaultdict
from datetime import datetime
from enum import Enum
from queue import Empty, Queue
from typing import Any, Callable, Dict, List, Optional


class ExecutionStatus(Enum):
    """执行状态"""

    PENDING = "PENDING"
    EXECUTING = "EXECUTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ExecutionEngine:
    """执行引擎 - 负责订单的实际执行"""

    def __init__(self, config: Dict[str, Any] = None):
        # 兼容配置对象和字典
        if hasattr(config, '__dict__'):
            self.config = config.__dict__ if config else {}
        else:
            self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # 执行队列
        self.execution_queue = Queue()
        self.execution_results = {}

        # 执行状态跟踪
        self.executions = {}
        self.execution_history = []

        # 线程安全
        self._lock = threading.Lock()
        self._running = False
        self._execution_thread = None

        # 配置参数
        self.max_execution_threads = self.config.get("max_execution_threads", 4)
        self.execution_timeout = self.config.get("execution_timeout", 30.0)
        self.retry_attempts = self.config.get("retry_attempts", 3)

        # 执行回调
        self.execution_callbacks = []

    def start(self):
        """启动执行引擎"""
        if self._running:
            return

        self._running = True
        self._execution_thread = threading.Thread(target=self._execution_worker)
        self._execution_thread.daemon = True
        self._execution_thread.start()

        self.logger.info("Execution engine started")

    def stop(self):
        """停止执行引擎"""
        self._running = False
        if self._execution_thread:
            self._execution_thread.join(timeout=5.0)

        self.logger.info("Execution engine stopped")

    def submit_execution(self, order: Dict[str, Any]) -> str:
        """提交执行请求"""
        execution_id = f"exec_{int(time.time() * 1000)}"

        execution_request = {
            "execution_id": execution_id,
            "order": order,
            "timestamp": datetime.now(),
            "status": ExecutionStatus.PENDING,
            "attempts": 0,
        }

        with self._lock:
            self.executions[execution_id] = execution_request
            self.execution_queue.put(execution_request)

        self.logger.info(f"Execution request submitted: {execution_id}")
        return execution_id

    def cancel_execution(self, execution_id: str) -> bool:
        """取消执行"""
        with self._lock:
            if execution_id in self.executions:
                execution = self.executions[execution_id]
                if execution["status"] == ExecutionStatus.PENDING:
                    execution["status"] = ExecutionStatus.CANCELLED
                    self.logger.info(f"Execution cancelled: {execution_id}")
                    return True

        return False

    def get_execution_status(self, execution_id: str) -> Optional[ExecutionStatus]:
        """获取执行状态"""
        with self._lock:
            if execution_id in self.executions:
                return self.executions[execution_id]["status"]
        return None

    def get_execution_result(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """获取执行结果"""
        with self._lock:
            return self.execution_results.get(execution_id)

    def add_execution_callback(self, callback: Callable):
        """添加执行回调"""
        self.execution_callbacks.append(callback)

    def _execution_worker(self):
        """执行工作线程"""
        while self._running:
            try:
                # 从队列获取执行请求
                try:
                    execution_request = self.execution_queue.get(timeout=1.0)
                except Empty:
                    continue

                # 检查是否已取消
                if execution_request["status"] == ExecutionStatus.CANCELLED:
                    continue

                # 执行订单
                self._execute_order(execution_request)

            except Exception as e:
                self.logger.error(f"Execution worker error: {str(e)}")

    def _execute_order(self, execution_request: Dict[str, Any]):
        """执行具体订单"""
        execution_id = execution_request["execution_id"]
        order = execution_request["order"]

        try:
            # 更新状态为执行中
            execution_request["status"] = ExecutionStatus.EXECUTING
            execution_request["attempts"] += 1

            self.logger.info(f"Executing order: {execution_id}")

            # 模拟执行过程（实际环境中会调用券商API）
            execution_result = self._simulate_execution(order)

            # 更新执行结果
            with self._lock:
                execution_request["status"] = ExecutionStatus.COMPLETED
                self.execution_results[execution_id] = execution_result
                self.execution_history.append(
                    {
                        "execution_id": execution_id,
                        "order": order,
                        "result": execution_result,
                        "timestamp": datetime.now(),
                    }
                )

            # 调用回调函数
            for callback in self.execution_callbacks:
                try:
                    callback(execution_id, execution_result)
                except Exception as e:
                    self.logger.error(f"Execution callback error: {str(e)}")

            self.logger.info(f"Order executed successfully: {execution_id}")

        except Exception as e:
            self.logger.error(
                f"Order execution failed: {execution_id}, error: {str(e)}"
            )

            # 检查是否需要重试
            if execution_request["attempts"] < self.retry_attempts:
                # 重新加入队列重试
                self.execution_queue.put(execution_request)
            else:
                # 标记为失败
                execution_request["status"] = ExecutionStatus.FAILED
                with self._lock:
                    self.execution_results[execution_id] = {
                        "success": False,
                        "error": str(e),
                        "timestamp": datetime.now(),
                    }

    def _simulate_execution(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """模拟订单执行（实际环境中替换为真实执行逻辑）"""
        # 模拟执行延迟
        time.sleep(0.1)

        # 模拟执行结果
        symbol = order.get("symbol")
        side = order.get("side")
        quantity = order.get("quantity")
        price = order.get("price", 0)

        # 模拟市价单执行
        if order.get("order_type") == "MARKET":
            executed_price = price * (1.001 if side == "BUY" else 0.999)  # 模拟滑点
        else:
            executed_price = price

        return {
            "success": True,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "executed_price": executed_price,
            "executed_quantity": quantity,
            "commission": quantity * executed_price * 0.0003,  # 模拟佣金
            "timestamp": datetime.now(),
            "execution_id": f"trade_{int(time.time() * 1000)}",
        }

    def get_execution_statistics(self) -> Dict[str, Any]:
        """获取执行统计信息"""
        with self._lock:
            total_executions = len(self.execution_history)
            successful_executions = sum(
                1 for h in self.execution_history if h["result"].get("success", False)
            )

            return {
                "total_executions": total_executions,
                "successful_executions": successful_executions,
                "failure_rate": (total_executions - successful_executions)
                / max(total_executions, 1),
                "pending_executions": len(
                    [
                        e
                        for e in self.executions.values()
                        if e["status"] == ExecutionStatus.PENDING
                    ]
                ),
                "executing_count": len(
                    [
                        e
                        for e in self.executions.values()
                        if e["status"] == ExecutionStatus.EXECUTING
                    ]
                ),
            }

    def clear_execution_history(self, before_date: datetime = None):
        """清理执行历史"""
        if before_date is None:
            before_date = datetime.now()

        with self._lock:
            self.execution_history = [
                h for h in self.execution_history if h["timestamp"] > before_date
            ]

            # 清理已完成的执行记录
            completed_executions = [
                eid
                for eid, e in self.executions.items()
                if e["status"]
                in [
                    ExecutionStatus.COMPLETED,
                    ExecutionStatus.FAILED,
                    ExecutionStatus.CANCELLED,
                ]
                and e["timestamp"] < before_date
            ]

            for eid in completed_executions:
                del self.executions[eid]
                self.execution_results.pop(eid, None)

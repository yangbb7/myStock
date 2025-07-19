# -*- coding: utf-8 -*-
"""
Logging - 统一日志管理模块
"""

import json
import logging
import logging.handlers
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional


class StructuredLogger:
    """结构化日志记录器"""

    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(name)

        # 设置日志级别
        level = self.config.get("level", "INFO")
        self.logger.setLevel(getattr(logging, level.upper()))

        # 避免重复添加处理器
        if not self.logger.handlers:
            self._setup_handlers()

    def _setup_handlers(self):
        """设置日志处理器"""
        # 控制台处理器
        if self.config.get("console_output", True):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(self._get_console_formatter())
            self.logger.addHandler(console_handler)

        # 文件处理器
        if self.config.get("file_output", True):
            log_dir = self.config.get("log_dir", "logs")
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            log_file = os.path.join(log_dir, f"{self.name}.log")
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=self.config.get("max_file_size", 10 * 1024 * 1024),  # 10MB
                backupCount=self.config.get("backup_count", 5),
            )
            file_handler.setFormatter(self._get_file_formatter())
            self.logger.addHandler(file_handler)

        # JSON文件处理器（用于结构化日志）
        if self.config.get("json_output", False):
            log_dir = self.config.get("log_dir", "logs")
            json_file = os.path.join(log_dir, f"{self.name}.jsonl")
            json_handler = logging.handlers.RotatingFileHandler(
                json_file,
                maxBytes=self.config.get("max_file_size", 10 * 1024 * 1024),
                backupCount=self.config.get("backup_count", 5),
            )
            json_handler.setFormatter(self._get_json_formatter())
            self.logger.addHandler(json_handler)

    def _get_console_formatter(self):
        """获取控制台格式化器"""
        fmt = self.config.get(
            "console_format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        return logging.Formatter(fmt)

    def _get_file_formatter(self):
        """获取文件格式化器"""
        fmt = self.config.get(
            "file_format",
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
        )
        return logging.Formatter(fmt)

    def _get_json_formatter(self):
        """获取JSON格式化器"""
        return JsonFormatter()

    def info(self, message: str, **kwargs):
        """记录信息日志"""
        self._log_with_context(logging.INFO, message, **kwargs)

    def debug(self, message: str, **kwargs):
        """记录调试日志"""
        self._log_with_context(logging.DEBUG, message, **kwargs)

    def warning(self, message: str, **kwargs):
        """记录警告日志"""
        self._log_with_context(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs):
        """记录错误日志"""
        self._log_with_context(logging.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs):
        """记录严重错误日志"""
        self._log_with_context(logging.CRITICAL, message, **kwargs)

    def _log_with_context(self, level: int, message: str, **kwargs):
        """带上下文的日志记录"""
        # 构建额外的上下文信息
        extra = {
            "timestamp": datetime.now().isoformat(),
            "logger_name": self.name,
            **kwargs,
        }

        self.logger.log(level, message, extra=extra)


class JsonFormatter(logging.Formatter):
    """JSON格式化器"""

    def format(self, record):
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }

        # 添加额外的字段
        for key, value in record.__dict__.items():
            if key not in [
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "getMessage",
            ]:
                log_entry[key] = value

        return json.dumps(log_entry, default=str)


class LoggerManager:
    """日志管理器"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.loggers: Dict[str, StructuredLogger] = {}

        # 全局配置
        self.global_config = self.config.get("global", {})

        # 设置根日志级别
        root_level = self.global_config.get("level", "INFO")
        logging.getLogger().setLevel(getattr(logging, root_level.upper()))

    def get_logger(self, name: str, config: Dict[str, Any] = None) -> StructuredLogger:
        """获取或创建日志记录器"""
        if name not in self.loggers:
            # 合并配置
            logger_config = {**self.global_config}
            if config:
                logger_config.update(config)

            # 检查是否有特定的配置
            specific_config = self.config.get("loggers", {}).get(name, {})
            logger_config.update(specific_config)

            self.loggers[name] = StructuredLogger(name, logger_config)

        return self.loggers[name]

    def configure_logger(self, name: str, config: Dict[str, Any]):
        """配置特定的日志记录器"""
        if name in self.loggers:
            # 移除现有的处理器
            for handler in self.loggers[name].logger.handlers[:]:
                self.loggers[name].logger.removeHandler(handler)

            # 重新创建
            del self.loggers[name]

        # 使用新配置创建
        self.get_logger(name, config)

    def shutdown(self):
        """关闭所有日志记录器"""
        for logger in self.loggers.values():
            for handler in logger.logger.handlers:
                handler.close()

        logging.shutdown()


# 全局日志管理器实例
_logger_manager: Optional[LoggerManager] = None


def get_logger(name: str, config: Dict[str, Any] = None) -> StructuredLogger:
    """获取日志记录器（全局函数）"""
    global _logger_manager

    if _logger_manager is None:
        _logger_manager = LoggerManager()

    return _logger_manager.get_logger(name, config)


def configure_logging(config: Dict[str, Any]):
    """配置全局日志"""
    global _logger_manager
    _logger_manager = LoggerManager(config)


def shutdown_logging():
    """关闭全局日志"""
    global _logger_manager
    if _logger_manager:
        _logger_manager.shutdown()
        _logger_manager = None


# 预定义的日志记录器
def get_trading_logger() -> StructuredLogger:
    """获取交易日志记录器"""
    return get_logger(
        "trading", {"level": "INFO", "file_output": True, "json_output": True}
    )


def get_strategy_logger() -> StructuredLogger:
    """获取策略日志记录器"""
    return get_logger(
        "strategy", {"level": "DEBUG", "file_output": True, "json_output": True}
    )


def get_risk_logger() -> StructuredLogger:
    """获取风险管理日志记录器"""
    return get_logger(
        "risk", {"level": "WARNING", "file_output": True, "json_output": True}
    )


def get_data_logger() -> StructuredLogger:
    """获取数据管理日志记录器"""
    return get_logger(
        "data", {"level": "INFO", "file_output": True, "json_output": False}
    )


def get_performance_logger() -> StructuredLogger:
    """获取性能分析日志记录器"""
    return get_logger(
        "performance", {"level": "INFO", "file_output": True, "json_output": True}
    )

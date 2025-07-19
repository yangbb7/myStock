"""
异常日志记录器

提供专门的异常日志记录功能，包括结构化日志和异常监控。
"""

import json
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from ...core.exceptions import GlobalExceptionHandler, MyQuantException


class ExceptionLogger:
    """异常日志记录器"""

    def __init__(
        self,
        log_file: str = "logs/exceptions.log",
        log_level: int = logging.ERROR,
        enable_console: bool = True,
        enable_structured_logging: bool = True,
    ):
        self.log_file = log_file
        self.log_level = log_level
        self.enable_console = enable_console
        self.enable_structured_logging = enable_structured_logging

        # 创建日志目录
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

        # 设置日志器
        self.logger = self._setup_logger()

        # 异常处理器
        self.exception_handler = GlobalExceptionHandler(self.logger)

        # 统计信息
        self.exception_stats = {
            "total_exceptions": 0,
            "by_type": {},
            "by_hour": {},
            "by_severity": {},
        }

    def _setup_logger(self) -> logging.Logger:
        """设置日志器"""
        logger = logging.getLogger("myquant.exceptions")
        logger.setLevel(self.log_level)

        # 避免重复添加处理器
        if logger.handlers:
            return logger

        # 文件处理器
        file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
        file_handler.setLevel(self.log_level)

        # 控制台处理器
        if self.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)

            # 控制台格式器
            console_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

        # 文件格式器
        if self.enable_structured_logging:
            file_formatter = StructuredFormatter()
        else:
            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(pathname)s:%(lineno)d"
            )

        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        return logger

    def log_exception(
        self,
        exception: Exception,
        context: Optional[Dict[str, Any]] = None,
        severity: str = "ERROR",
    ):
        """记录异常"""
        try:
            # 更新统计信息
            self._update_stats(exception, severity)

            # 获取异常信息
            exc_info = sys.exc_info()
            tb_str = (
                "".join(traceback.format_exception(*exc_info)) if exc_info[0] else None
            )

            # 构建异常记录
            error_record = {
                "timestamp": datetime.now().isoformat(),
                "exception_type": exception.__class__.__name__,
                "message": str(exception),
                "severity": severity,
                "traceback": tb_str,
                "context": context or {},
            }

            # 如果是MyQuant异常，添加额外信息
            if isinstance(exception, MyQuantException):
                error_record.update(
                    {
                        "error_code": exception.error_code,
                        "details": exception.details,
                        "cause": str(exception.cause) if exception.cause else None,
                    }
                )

            # 记录日志
            log_level = getattr(logging, severity.upper(), logging.ERROR)
            self.logger.log(
                log_level,
                f"异常发生: {exception.__class__.__name__}: {str(exception)}",
                extra={"error_record": error_record},
            )

        except Exception as log_error:
            # 确保日志记录本身不会引发异常
            fallback_logger = logging.getLogger("myquant.fallback")
            fallback_logger.error(f"异常日志记录失败: {log_error}")

    def _update_stats(self, exception: Exception, severity: str):
        """更新异常统计信息"""
        self.exception_stats["total_exceptions"] += 1

        # 按类型统计
        exc_type = exception.__class__.__name__
        self.exception_stats["by_type"][exc_type] = (
            self.exception_stats["by_type"].get(exc_type, 0) + 1
        )

        # 按小时统计
        hour = datetime.now().hour
        self.exception_stats["by_hour"][hour] = (
            self.exception_stats["by_hour"].get(hour, 0) + 1
        )

        # 按严重程度统计
        self.exception_stats["by_severity"][severity] = (
            self.exception_stats["by_severity"].get(severity, 0) + 1
        )

    def get_stats(self) -> Dict[str, Any]:
        """获取异常统计信息"""
        return self.exception_stats.copy()

    def reset_stats(self):
        """重置统计信息"""
        self.exception_stats = {
            "total_exceptions": 0,
            "by_type": {},
            "by_hour": {},
            "by_severity": {},
        }


class StructuredFormatter(logging.Formatter):
    """结构化日志格式器"""

    def format(self, record):
        """格式化日志记录"""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # 添加异常记录（如果存在）
        if hasattr(record, "error_record"):
            log_entry["error_record"] = record.error_record

        # 添加异常信息（如果存在）
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False, indent=2)


# 全局异常日志记录器实例
exception_logger = ExceptionLogger()


def log_exception(
    exception: Exception,
    context: Optional[Dict[str, Any]] = None,
    severity: str = "ERROR",
):
    """全局异常记录函数"""
    exception_logger.log_exception(exception, context, severity)


def setup_exception_handling():
    """设置全局异常处理"""

    def handle_exception(exc_type, exc_value, exc_traceback):
        """全局异常处理器"""
        if issubclass(exc_type, KeyboardInterrupt):
            # 允许 Ctrl+C 正常退出
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        # 记录未捕获的异常
        exception_logger.logger.critical(
            "未捕获的异常", exc_info=(exc_type, exc_value, exc_traceback)
        )

    # 设置全局异常处理器
    sys.excepthook = handle_exception


# 异常监控装饰器
def monitor_exceptions(
    context: Optional[Dict[str, Any]] = None,
    severity: str = "ERROR",
    reraise: bool = True,
):
    """异常监控装饰器"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # 构建上下文信息
                func_context = {
                    "function": func.__name__,
                    "module": func.__module__,
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys()),
                }
                if context:
                    func_context.update(context)

                # 记录异常
                log_exception(e, func_context, severity)

                if reraise:
                    raise
                return None

        return wrapper

    return decorator

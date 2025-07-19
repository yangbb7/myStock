# -*- coding: utf-8 -*-
"""
异常日志记录器 - 提供统一的异常记录和处理功能
"""

import json
import logging
import traceback
from datetime import datetime
from typing import Any, Dict, Optional


class ExceptionLogger:
    """异常日志记录器"""

    def __init__(self, logger_name: str = None):
        self.logger = logging.getLogger(logger_name or __name__)

    def log_exception(self, exception: Exception, context: Dict[str, Any] = None):
        """
        记录异常

        Args:
            exception: 异常对象
            context: 上下文信息
        """
        context = context or {}

        error_info = {
            "timestamp": datetime.now().isoformat(),
            "exception_type": exception.__class__.__name__,
            "exception_message": str(exception),
            "traceback": traceback.format_exc(),
            "context": context,
        }

        self.logger.error(
            f"异常发生: {exception.__class__.__name__}: {str(exception)}",
            extra={"error_info": error_info},
        )

    def log_warning(self, message: str, context: Dict[str, Any] = None):
        """
        记录警告

        Args:
            message: 警告消息
            context: 上下文信息
        """
        context = context or {}
        warning_info = {
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "context": context,
        }

        self.logger.warning(message, extra={"warning_info": warning_info})

    def log_info(self, message: str, context: Dict[str, Any] = None):
        """
        记录信息

        Args:
            message: 信息消息
            context: 上下文信息
        """
        context = context or {}
        info = {
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "context": context,
        }

        self.logger.info(message, extra={"info": info})

"""
配置加载和管理模块

提供配置的加载、验证、热更新和环境切换功能。
"""

import json
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from ..core.exceptions import ConfigurationException

from .settings import Settings, validate_settings


class ConfigLoader:
    """配置加载器"""

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.logger = logging.getLogger(__name__)
        self._current_settings: Optional[Settings] = None
        self._config_cache: Dict[str, Settings] = {}

    def load_default_config(self) -> Settings:
        """加载默认配置"""
        try:
            settings = Settings()
            self._validate_config(settings)
            self._current_settings = settings
            return settings
        except Exception as e:
            raise ConfigurationException(
                f"加载默认配置失败: {str(e)}", config_file="default", cause=e
            )

    def load_from_env_file(self, env_file: str = ".env") -> Settings:
        """从环境文件加载配置"""
        env_path = Path(env_file)

        if not env_path.exists():
            self.logger.warning(f"环境文件不存在: {env_path}")
            return self.load_default_config()

        try:
            # 设置环境文件路径
            original_env_file = os.environ.get("ENV_FILE")
            os.environ["ENV_FILE"] = str(env_path)

            settings = Settings(_env_file=str(env_path))
            self._validate_config(settings)
            self._current_settings = settings

            # 恢复原环境变量
            if original_env_file:
                os.environ["ENV_FILE"] = original_env_file
            elif "ENV_FILE" in os.environ:
                del os.environ["ENV_FILE"]

            return settings

        except Exception as e:
            raise ConfigurationException(
                f"从环境文件加载配置失败: {str(e)}", config_file=str(env_path), cause=e
            )

    def load_from_json_file(self, json_file: str) -> Settings:
        """从JSON文件加载配置"""
        json_path = Path(json_file)

        if not json_path.exists():
            raise ConfigurationException(
                f"配置文件不存在: {json_path}", config_file=str(json_path)
            )

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)

            settings = Settings(**config_data)
            self._validate_config(settings)
            self._current_settings = settings

            return settings

        except json.JSONDecodeError as e:
            raise ConfigurationException(
                f"JSON配置文件格式错误: {str(e)}", config_file=str(json_path), cause=e
            )
        except Exception as e:
            raise ConfigurationException(
                f"从JSON文件加载配置失败: {str(e)}", config_file=str(json_path), cause=e
            )

    def load_environment_config(self, env: str) -> Settings:
        """加载环境特定配置"""
        if env in self._config_cache:
            return self._config_cache[env]

        # 尝试加载环境特定的配置文件
        env_files = [
            f".env.{env}",
            f"config/{env}.json",
            f"config/environments/{env}.json",
        ]

        for env_file in env_files:
            if Path(env_file).exists():
                try:
                    if env_file.endswith(".json"):
                        settings = self.load_from_json_file(env_file)
                    else:
                        settings = self.load_from_env_file(env_file)

                    self._config_cache[env] = settings
                    return settings

                except Exception as e:
                    self.logger.warning(f"加载环境配置文件失败 {env_file}: {e}")
                    continue

        # 如果没有找到环境特定配置，返回默认配置
        self.logger.info(f"未找到环境 {env} 的特定配置，使用默认配置")
        return self.load_default_config()

    def _validate_config(self, settings: Settings) -> None:
        """验证配置"""
        errors = validate_settings(settings)
        if errors:
            error_msg = "配置验证失败:\n" + "\n".join(f"- {error}" for error in errors)
            raise ConfigurationException(
                error_msg, details={"validation_errors": errors}
            )

    def get_current_settings(self) -> Optional[Settings]:
        """获取当前配置"""
        return self._current_settings

    def reload_config(self) -> Settings:
        """重新加载配置"""
        env = os.getenv("ENVIRONMENT", "development")
        return self.load_environment_config(env)

    def save_config(self, settings: Settings, file_path: str) -> None:
        """保存配置到文件"""
        try:
            # 创建目录
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)

            # 保存配置
            settings.save_to_file(file_path)

            self.logger.info(f"配置已保存到: {file_path}")

        except Exception as e:
            raise ConfigurationException(
                f"保存配置失败: {str(e)}", config_file=file_path, cause=e
            )


class ConfigManager:
    """配置管理器"""

    def __init__(self):
        self.loader = ConfigLoader()
        self.logger = logging.getLogger(__name__)
        self._settings: Optional[Settings] = None
        self._environment = os.getenv("ENVIRONMENT", "development")

    def initialize(self, environment: Optional[str] = None) -> Settings:
        """初始化配置管理器"""
        if environment:
            self._environment = environment
            os.environ["ENVIRONMENT"] = environment

        try:
            self._settings = self.loader.load_environment_config(self._environment)
            self.logger.info(f"配置管理器已初始化，环境: {self._environment}")
            return self._settings

        except Exception as e:
            self.logger.error(f"配置管理器初始化失败: {e}")
            # 降级到默认配置
            self._settings = self.loader.load_default_config()
            return self._settings

    @property
    def settings(self) -> Settings:
        """获取配置设置"""
        if self._settings is None:
            self._settings = self.initialize()
        return self._settings

    @property
    def environment(self) -> str:
        """获取当前环境"""
        return self._environment

    def switch_environment(self, environment: str) -> Settings:
        """切换环境"""
        old_env = self._environment
        self._environment = environment
        os.environ["ENVIRONMENT"] = environment

        try:
            self._settings = self.loader.load_environment_config(environment)
            self.logger.info(f"环境已切换: {old_env} -> {environment}")
            return self._settings

        except Exception as e:
            # 回滚到原环境
            self._environment = old_env
            os.environ["ENVIRONMENT"] = old_env
            raise ConfigurationException(
                f"切换环境失败: {str(e)}",
                details={
                    "target_environment": environment,
                    "current_environment": old_env,
                },
                cause=e,
            )

    def update_setting(self, key_path: str, value: Any) -> None:
        """更新配置项"""
        if self._settings is None:
            raise ConfigurationException("配置管理器未初始化")

        try:
            # 解析键路径 (例如: "risk.max_position_size")
            keys = key_path.split(".")
            current = self._settings

            # 导航到父对象
            for key in keys[:-1]:
                current = getattr(current, key)

            # 设置值
            setattr(current, keys[-1], value)

            # 重新验证配置
            self.loader._validate_config(self._settings)

            self.logger.info(f"配置项已更新: {key_path} = {value}")

        except AttributeError:
            raise ConfigurationException(f"配置项不存在: {key_path}")
        except Exception as e:
            raise ConfigurationException(
                f"更新配置项失败: {str(e)}", config_key=key_path, cause=e
            )

    def get_setting(self, key_path: str, default: Any = None) -> Any:
        """获取配置项"""
        if self._settings is None:
            self.initialize()

        try:
            keys = key_path.split(".")
            current = self._settings

            for key in keys:
                current = getattr(current, key)

            return current

        except AttributeError:
            if default is not None:
                return default
            raise ConfigurationException(f"配置项不存在: {key_path}")

    def reload(self) -> Settings:
        """重新加载配置"""
        self._settings = self.loader.reload_config()
        self.logger.info("配置已重新加载")
        return self._settings

    @contextmanager
    def temporary_setting(self, key_path: str, value: Any):
        """临时设置配置项（上下文管理器）"""
        original_value = self.get_setting(key_path)
        try:
            self.update_setting(key_path, value)
            yield
        finally:
            self.update_setting(key_path, original_value)

    def export_config(self, file_path: str) -> None:
        """导出当前配置"""
        if self._settings is None:
            raise ConfigurationException("配置管理器未初始化")

        self.loader.save_config(self._settings, file_path)

    def get_config_summary(self) -> Dict[str, Any]:
        """获取配置摘要"""
        if self._settings is None:
            return {}

        return {
            "environment": self._environment,
            "app_name": self._settings.app_name,
            "version": self._settings.version,
            "debug": self._settings.debug,
            "database_type": self._settings.database.type,
            "trading_mode": self._settings.trading.mode,
            "primary_data_source": self._settings.data.primary_source,
            "log_level": self._settings.logging.level,
        }


# 全局配置管理器实例
config_manager = ConfigManager()


def get_config() -> Settings:
    """获取配置"""
    return config_manager.settings


def get_config_manager() -> ConfigManager:
    """获取配置管理器"""
    return config_manager


def init_config(environment: Optional[str] = None) -> Settings:
    """初始化配置"""
    return config_manager.initialize(environment)

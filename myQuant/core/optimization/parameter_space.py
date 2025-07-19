# -*- coding: utf-8 -*-
"""
参数空间定义 - 定义策略参数的搜索空间和约束
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import numpy as np
import itertools
from scipy.stats import uniform, randint, loguniform


class ParameterType(Enum):
    """参数类型"""
    INTEGER = "integer"
    FLOAT = "float"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"


@dataclass
class Parameter:
    """参数定义"""
    name: str
    param_type: ParameterType
    low: Optional[Union[int, float]] = None
    high: Optional[Union[int, float]] = None
    choices: Optional[List[Any]] = None
    default: Optional[Any] = None
    log_scale: bool = False
    constraint: Optional[Callable[[Any], bool]] = None
    description: str = ""
    
    def __post_init__(self):
        """参数验证"""
        if self.param_type in [ParameterType.INTEGER, ParameterType.FLOAT, ParameterType.CONTINUOUS]:
            if self.low is None or self.high is None:
                raise ValueError(f"Parameter {self.name}: low and high must be specified")
            if self.low >= self.high:
                raise ValueError(f"Parameter {self.name}: low must be less than high")
        
        elif self.param_type == ParameterType.CATEGORICAL:
            if not self.choices:
                raise ValueError(f"Parameter {self.name}: choices must be specified")
        
        elif self.param_type == ParameterType.BOOLEAN:
            self.choices = [True, False]
    
    def sample(self, random_state: Optional[np.random.RandomState] = None) -> Any:
        """随机采样参数值"""
        if random_state is None:
            random_state = np.random.RandomState()
        
        if self.param_type == ParameterType.INTEGER:
            return random_state.randint(self.low, self.high + 1)
        
        elif self.param_type == ParameterType.FLOAT or self.param_type == ParameterType.CONTINUOUS:
            if self.log_scale:
                # 对数尺度采样
                log_low = np.log(self.low)
                log_high = np.log(self.high)
                return np.exp(random_state.uniform(log_low, log_high))
            else:
                return random_state.uniform(self.low, self.high)
        
        elif self.param_type in [ParameterType.CATEGORICAL, ParameterType.BOOLEAN]:
            return random_state.choice(self.choices)
        
        elif self.param_type == ParameterType.DISCRETE:
            return random_state.choice(self.choices)
        
        else:
            raise ValueError(f"Unknown parameter type: {self.param_type}")
    
    def validate(self, value: Any) -> bool:
        """验证参数值"""
        if self.param_type == ParameterType.INTEGER:
            if not isinstance(value, int) or value < self.low or value > self.high:
                return False
        
        elif self.param_type in [ParameterType.FLOAT, ParameterType.CONTINUOUS]:
            if not isinstance(value, (int, float)) or value < self.low or value > self.high:
                return False
        
        elif self.param_type in [ParameterType.CATEGORICAL, ParameterType.BOOLEAN, ParameterType.DISCRETE]:
            if value not in self.choices:
                return False
        
        # 检查约束条件
        if self.constraint and not self.constraint(value):
            return False
        
        return True
    
    def clip(self, value: Any) -> Any:
        """将值裁剪到有效范围"""
        if self.param_type == ParameterType.INTEGER:
            return int(np.clip(value, self.low, self.high))
        
        elif self.param_type in [ParameterType.FLOAT, ParameterType.CONTINUOUS]:
            return float(np.clip(value, self.low, self.high))
        
        elif self.param_type in [ParameterType.CATEGORICAL, ParameterType.BOOLEAN, ParameterType.DISCRETE]:
            if value in self.choices:
                return value
            else:
                # 返回最接近的选择或默认值
                return self.default if self.default in self.choices else self.choices[0]
        
        return value
    
    def get_grid_values(self, num_points: int) -> List[Any]:
        """获取网格搜索的值"""
        if self.param_type == ParameterType.INTEGER:
            step = max(1, (self.high - self.low) // (num_points - 1))
            return list(range(self.low, self.high + 1, step))
        
        elif self.param_type in [ParameterType.FLOAT, ParameterType.CONTINUOUS]:
            if self.log_scale:
                return np.logspace(np.log10(self.low), np.log10(self.high), num_points).tolist()
            else:
                return np.linspace(self.low, self.high, num_points).tolist()
        
        elif self.param_type in [ParameterType.CATEGORICAL, ParameterType.BOOLEAN, ParameterType.DISCRETE]:
            return self.choices
        
        return []


class ParameterSpace:
    """参数空间管理器"""
    
    def __init__(self, parameters: Optional[List[Parameter]] = None):
        """
        初始化参数空间
        
        Args:
            parameters: 参数列表
        """
        self.parameters: Dict[str, Parameter] = {}
        self.constraints: List[Callable[[Dict[str, Any]], bool]] = []
        
        if parameters:
            for param in parameters:
                self.add_parameter(param)
    
    def add_parameter(self, parameter: Parameter) -> None:
        """
        添加参数
        
        Args:
            parameter: 参数定义
        """
        self.parameters[parameter.name] = parameter
    
    def add_constraint(self, constraint: Callable[[Dict[str, Any]], bool]) -> None:
        """
        添加全局约束
        
        Args:
            constraint: 约束函数，接受参数字典，返回是否满足约束
        """
        self.constraints.append(constraint)
    
    def sample(self, n_samples: int = 1, random_state: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        随机采样参数组合
        
        Args:
            n_samples: 采样数量
            random_state: 随机种子
            
        Returns:
            List[Dict[str, Any]]: 参数组合列表
        """
        if random_state is not None:
            rng = np.random.RandomState(random_state)
        else:
            rng = np.random.RandomState()
        
        samples = []
        max_attempts = n_samples * 100  # 防止无限循环
        attempts = 0
        
        while len(samples) < n_samples and attempts < max_attempts:
            sample = {}
            for name, param in self.parameters.items():
                sample[name] = param.sample(rng)
            
            # 检查约束
            if self.validate_sample(sample):
                samples.append(sample)
            
            attempts += 1
        
        return samples
    
    def validate_sample(self, sample: Dict[str, Any]) -> bool:
        """
        验证参数组合
        
        Args:
            sample: 参数组合
            
        Returns:
            bool: 是否有效
        """
        # 检查各个参数
        for name, value in sample.items():
            if name not in self.parameters:
                return False
            if not self.parameters[name].validate(value):
                return False
        
        # 检查全局约束
        for constraint in self.constraints:
            if not constraint(sample):
                return False
        
        return True
    
    def clip_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        裁剪参数组合到有效范围
        
        Args:
            sample: 参数组合
            
        Returns:
            Dict[str, Any]: 裁剪后的参数组合
        """
        clipped = {}
        for name, value in sample.items():
            if name in self.parameters:
                clipped[name] = self.parameters[name].clip(value)
            else:
                clipped[name] = value
        
        return clipped
    
    def get_grid(self, num_points_per_param: Union[int, Dict[str, int]] = 10) -> List[Dict[str, Any]]:
        """
        生成网格搜索的参数组合
        
        Args:
            num_points_per_param: 每个参数的网格点数
            
        Returns:
            List[Dict[str, Any]]: 参数组合列表
        """
        if isinstance(num_points_per_param, int):
            num_points = {name: num_points_per_param for name in self.parameters.keys()}
        else:
            num_points = num_points_per_param
        
        # 生成每个参数的值列表
        param_values = {}
        for name, param in self.parameters.items():
            points = num_points.get(name, 10)
            param_values[name] = param.get_grid_values(points)
        
        # 生成笛卡尔积
        param_names = list(param_values.keys())
        combinations = itertools.product(*[param_values[name] for name in param_names])
        
        # 过滤满足约束的组合
        valid_combinations = []
        for combo in combinations:
            sample = dict(zip(param_names, combo))
            if self.validate_sample(sample):
                valid_combinations.append(sample)
        
        return valid_combinations
    
    def get_bounds(self) -> List[Tuple[float, float]]:
        """
        获取参数边界（用于某些优化算法）
        
        Returns:
            List[Tuple[float, float]]: 参数边界列表
        """
        bounds = []
        for param in self.parameters.values():
            if param.param_type in [ParameterType.INTEGER, ParameterType.FLOAT, ParameterType.CONTINUOUS]:
                bounds.append((param.low, param.high))
            else:
                # 对于分类参数，使用索引
                bounds.append((0, len(param.choices) - 1))
        
        return bounds
    
    def encode_sample(self, sample: Dict[str, Any]) -> List[float]:
        """
        将参数组合编码为数值向量（用于某些优化算法）
        
        Args:
            sample: 参数组合
            
        Returns:
            List[float]: 编码后的向量
        """
        encoded = []
        for name, param in self.parameters.items():
            value = sample.get(name, param.default)
            
            if param.param_type in [ParameterType.INTEGER, ParameterType.FLOAT, ParameterType.CONTINUOUS]:
                if param.log_scale:
                    # 对数尺度编码
                    encoded_value = np.log(value) / np.log(param.high / param.low)
                else:
                    # 线性归一化
                    encoded_value = (value - param.low) / (param.high - param.low)
                encoded.append(encoded_value)
            
            elif param.param_type in [ParameterType.CATEGORICAL, ParameterType.BOOLEAN, ParameterType.DISCRETE]:
                # 使用索引
                encoded.append(param.choices.index(value))
        
        return encoded
    
    def decode_sample(self, encoded: List[float]) -> Dict[str, Any]:
        """
        将数值向量解码为参数组合
        
        Args:
            encoded: 编码的向量
            
        Returns:
            Dict[str, Any]: 参数组合
        """
        sample = {}
        for i, (name, param) in enumerate(self.parameters.items()):
            if i >= len(encoded):
                sample[name] = param.default
                continue
            
            encoded_value = encoded[i]
            
            if param.param_type == ParameterType.INTEGER:
                if param.log_scale:
                    value = int(param.low * (param.high / param.low) ** encoded_value)
                else:
                    value = int(param.low + encoded_value * (param.high - param.low))
                sample[name] = np.clip(value, param.low, param.high)
            
            elif param.param_type in [ParameterType.FLOAT, ParameterType.CONTINUOUS]:
                if param.log_scale:
                    value = param.low * (param.high / param.low) ** encoded_value
                else:
                    value = param.low + encoded_value * (param.high - param.low)
                sample[name] = np.clip(value, param.low, param.high)
            
            elif param.param_type in [ParameterType.CATEGORICAL, ParameterType.BOOLEAN, ParameterType.DISCRETE]:
                index = int(np.clip(encoded_value, 0, len(param.choices) - 1))
                sample[name] = param.choices[index]
        
        return sample
    
    def get_parameter_names(self) -> List[str]:
        """获取参数名称列表"""
        return list(self.parameters.keys())
    
    def get_parameter(self, name: str) -> Optional[Parameter]:
        """获取参数定义"""
        return self.parameters.get(name)
    
    def remove_parameter(self, name: str) -> bool:
        """移除参数"""
        if name in self.parameters:
            del self.parameters[name]
            return True
        return False
    
    def update_parameter(self, name: str, **kwargs) -> bool:
        """更新参数属性"""
        if name not in self.parameters:
            return False
        
        param = self.parameters[name]
        for attr, value in kwargs.items():
            if hasattr(param, attr):
                setattr(param, attr, value)
        
        return True
    
    def get_summary(self) -> Dict[str, Any]:
        """获取参数空间摘要"""
        return {
            'parameter_count': len(self.parameters),
            'parameters': {
                name: {
                    'type': param.param_type.value,
                    'low': param.low,
                    'high': param.high,
                    'choices': param.choices,
                    'default': param.default,
                    'log_scale': param.log_scale,
                    'description': param.description
                }
                for name, param in self.parameters.items()
            },
            'constraint_count': len(self.constraints)
        }


# 预定义参数空间构建器
class ParameterSpaceBuilder:
    """参数空间构建器"""
    
    def __init__(self):
        self.space = ParameterSpace()
    
    def add_integer(self, name: str, low: int, high: int, **kwargs) -> 'ParameterSpaceBuilder':
        """添加整数参数"""
        param = Parameter(name=name, param_type=ParameterType.INTEGER, low=low, high=high, **kwargs)
        self.space.add_parameter(param)
        return self
    
    def add_float(self, name: str, low: float, high: float, log_scale: bool = False, **kwargs) -> 'ParameterSpaceBuilder':
        """添加浮点数参数"""
        param = Parameter(name=name, param_type=ParameterType.FLOAT, low=low, high=high, log_scale=log_scale, **kwargs)
        self.space.add_parameter(param)
        return self
    
    def add_categorical(self, name: str, choices: List[Any], **kwargs) -> 'ParameterSpaceBuilder':
        """添加分类参数"""
        param = Parameter(name=name, param_type=ParameterType.CATEGORICAL, choices=choices, **kwargs)
        self.space.add_parameter(param)
        return self
    
    def add_boolean(self, name: str, **kwargs) -> 'ParameterSpaceBuilder':
        """添加布尔参数"""
        param = Parameter(name=name, param_type=ParameterType.BOOLEAN, **kwargs)
        self.space.add_parameter(param)
        return self
    
    def add_constraint(self, constraint: Callable[[Dict[str, Any]], bool]) -> 'ParameterSpaceBuilder':
        """添加约束"""
        self.space.add_constraint(constraint)
        return self
    
    def build(self) -> ParameterSpace:
        """构建参数空间"""
        return self.space


# 常用参数空间模板
class CommonParameterSpaces:
    """常用参数空间模板"""
    
    @staticmethod
    def moving_average_strategy() -> ParameterSpace:
        """移动平均策略参数空间"""
        return (ParameterSpaceBuilder()
                .add_integer('fast_period', 5, 50, default=10)
                .add_integer('slow_period', 20, 200, default=50)
                .add_float('signal_threshold', 0.001, 0.1, default=0.01)
                .add_categorical('ma_type', ['SMA', 'EMA', 'WMA'], default='SMA')
                .add_constraint(lambda params: params['fast_period'] < params['slow_period'])
                .build())
    
    @staticmethod
    def rsi_strategy() -> ParameterSpace:
        """RSI策略参数空间"""
        return (ParameterSpaceBuilder()
                .add_integer('rsi_period', 5, 30, default=14)
                .add_float('oversold_threshold', 10, 40, default=30)
                .add_float('overbought_threshold', 60, 90, default=70)
                .add_constraint(lambda params: params['oversold_threshold'] < params['overbought_threshold'])
                .build())
    
    @staticmethod
    def bollinger_bands_strategy() -> ParameterSpace:
        """布林带策略参数空间"""
        return (ParameterSpaceBuilder()
                .add_integer('period', 10, 50, default=20)
                .add_float('std_multiplier', 1.0, 3.0, default=2.0)
                .add_float('position_size', 0.1, 1.0, default=0.5)
                .build())
    
    @staticmethod
    def macd_strategy() -> ParameterSpace:
        """MACD策略参数空间"""
        return (ParameterSpaceBuilder()
                .add_integer('fast_period', 5, 20, default=12)
                .add_integer('slow_period', 20, 50, default=26)
                .add_integer('signal_period', 5, 15, default=9)
                .add_constraint(lambda params: params['fast_period'] < params['slow_period'])
                .build())
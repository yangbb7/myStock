"""
指标工厂

用于创建和管理各种技术指标实例
"""

from typing import Dict, Any, List, Callable, Optional
import logging


class Indicator:
    """指标基类"""
    
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.parameters = kwargs
        
    def __getattr__(self, name):
        return self.parameters.get(name)


class SMAIndicator(Indicator):
    """简单移动平均线指标"""
    
    def __init__(self, period: int):
        super().__init__('sma', period=period)


class EMAIndicator(Indicator):
    """指数移动平均线指标"""
    
    def __init__(self, period: int):
        super().__init__('ema', period=period)


class MACDIndicator(Indicator):
    """MACD指标"""
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        super().__init__('macd', 
                        fast_period=fast_period, 
                        slow_period=slow_period, 
                        signal_period=signal_period)


class RSIIndicator(Indicator):
    """RSI指标"""
    
    def __init__(self, period: int = 14):
        super().__init__('rsi', period=period)


class BollingerBandsIndicator(Indicator):
    """布林带指标"""
    
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        super().__init__('bollinger_bands', period=period, std_dev=std_dev)


class StochasticIndicator(Indicator):
    """随机震荡指标"""
    
    def __init__(self, period: int = 14, k_smoothing: int = 3, d_smoothing: int = 3):
        super().__init__('stochastic', 
                        period=period, 
                        k_smoothing=k_smoothing, 
                        d_smoothing=d_smoothing)


class ATRIndicator(Indicator):
    """平均真实波幅指标"""
    
    def __init__(self, period: int = 14):
        super().__init__('atr', period=period)


class IndicatorFactory:
    """指标工厂"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._custom_indicators: Dict[str, Callable] = {}
        
        # 注册内置指标
        self._builtin_indicators = {
            'sma': SMAIndicator,
            'ema': EMAIndicator,
            'macd': MACDIndicator,
            'rsi': RSIIndicator,
            'bollinger_bands': BollingerBandsIndicator,
            'stochastic': StochasticIndicator,
            'atr': ATRIndicator
        }
    
    def create_indicator(self, indicator_type: str, **kwargs) -> Optional[Indicator]:
        """创建指标实例"""
        try:
            if indicator_type in self._builtin_indicators:
                indicator_class = self._builtin_indicators[indicator_type]
                return indicator_class(**kwargs)
            elif indicator_type in self._custom_indicators:
                # 对于自定义指标，返回一个通用指标对象
                return Indicator(indicator_type, **kwargs)
            else:
                self.logger.error(f"未知指标类型: {indicator_type}")
                return None
                
        except Exception as e:
            self.logger.error(f"创建指标失败: {e}")
            return None
    
    def register_custom_indicator(self, name: str, calculation_func: Callable, **default_params) -> Optional[Indicator]:
        """注册自定义指标"""
        try:
            self._custom_indicators[name] = calculation_func
            self.logger.info(f"已注册自定义指标: {name}")
            
            # 返回一个指标实例
            return Indicator(name, **default_params)
            
        except Exception as e:
            self.logger.error(f"注册自定义指标失败: {e}")
            return None
    
    def list_available_indicators(self) -> List[str]:
        """列出所有可用指标"""
        return list(self._builtin_indicators.keys()) + list(self._custom_indicators.keys())
    
    def get_indicator_parameters(self, indicator_type: str) -> Dict[str, Any]:
        """获取指标的参数信息"""
        if indicator_type == 'sma':
            return {'period': {'type': 'int', 'default': 20, 'min': 1, 'max': 200}}
        elif indicator_type == 'ema':
            return {'period': {'type': 'int', 'default': 12, 'min': 1, 'max': 200}}
        elif indicator_type == 'macd':
            return {
                'fast_period': {'type': 'int', 'default': 12, 'min': 1, 'max': 100},
                'slow_period': {'type': 'int', 'default': 26, 'min': 1, 'max': 200},
                'signal_period': {'type': 'int', 'default': 9, 'min': 1, 'max': 50}
            }
        elif indicator_type == 'rsi':
            return {'period': {'type': 'int', 'default': 14, 'min': 2, 'max': 100}}
        elif indicator_type == 'bollinger_bands':
            return {
                'period': {'type': 'int', 'default': 20, 'min': 2, 'max': 200},
                'std_dev': {'type': 'float', 'default': 2.0, 'min': 0.1, 'max': 5.0}
            }
        elif indicator_type == 'stochastic':
            return {
                'period': {'type': 'int', 'default': 14, 'min': 1, 'max': 100},
                'k_smoothing': {'type': 'int', 'default': 3, 'min': 1, 'max': 10},
                'd_smoothing': {'type': 'int', 'default': 3, 'min': 1, 'max': 10}
            }
        elif indicator_type == 'atr':
            return {'period': {'type': 'int', 'default': 14, 'min': 1, 'max': 100}}
        else:
            return {}
    
    def validate_parameters(self, indicator_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """验证指标参数"""
        param_info = self.get_indicator_parameters(indicator_type)
        validated_params = {}
        errors = []
        
        for param_name, param_config in param_info.items():
            value = parameters.get(param_name, param_config.get('default'))
            
            if value is None:
                errors.append(f"缺少必需参数: {param_name}")
                continue
            
            # 类型检查
            param_type = param_config.get('type', 'any')
            if param_type == 'int' and not isinstance(value, int):
                try:
                    value = int(value)
                except (ValueError, TypeError):
                    errors.append(f"参数 {param_name} 必须是整数")
                    continue
            elif param_type == 'float' and not isinstance(value, (int, float)):
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    errors.append(f"参数 {param_name} 必须是数字")
                    continue
            
            # 范围检查
            min_val = param_config.get('min')
            max_val = param_config.get('max')
            
            if min_val is not None and value < min_val:
                errors.append(f"参数 {param_name} 不能小于 {min_val}")
                continue
            
            if max_val is not None and value > max_val:
                errors.append(f"参数 {param_name} 不能大于 {max_val}")
                continue
            
            validated_params[param_name] = value
        
        return {
            'parameters': validated_params,
            'errors': errors,
            'is_valid': len(errors) == 0
        }
    
    def get_indicator_description(self, indicator_type: str) -> str:
        """获取指标描述"""
        descriptions = {
            'sma': '简单移动平均线 - 计算指定周期内价格的算术平均值',
            'ema': '指数移动平均线 - 对近期价格给予更高权重的移动平均线',
            'macd': 'MACD指标 - 显示两条移动平均线之间关系的动量指标',
            'rsi': 'RSI相对强弱指标 - 衡量价格变动速度和变化的动量震荡器',
            'bollinger_bands': '布林带 - 由移动平均线和标准差构成的技术分析工具',
            'stochastic': '随机震荡指标 - 比较收盘价与特定周期内价格范围的位置',
            'atr': '平均真实波幅 - 衡量市场波动性的指标'
        }
        
        return descriptions.get(indicator_type, '未知指标')
    
    def create_indicator_set(self, config: Dict[str, Dict[str, Any]]) -> Dict[str, Indicator]:
        """批量创建指标集合"""
        indicators = {}
        
        for name, indicator_config in config.items():
            indicator_type = indicator_config.get('type')
            parameters = indicator_config.get('parameters', {})
            
            if indicator_type:
                indicator = self.create_indicator(indicator_type, **parameters)
                if indicator:
                    indicators[name] = indicator
                else:
                    self.logger.warning(f"创建指标 {name} 失败")
        
        return indicators
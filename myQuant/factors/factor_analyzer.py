# -*- coding: utf-8 -*-
"""
因子分析器 - 提供因子有效性检验、IC分析、因子筛选等功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import warnings
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
import joblib

warnings.filterwarnings('ignore')


class AnalysisMethod(Enum):
    """分析方法枚举"""
    PEARSON = "pearson"         # 皮尔逊相关
    SPEARMAN = "spearman"       # 斯皮尔曼相关
    KENDALL = "kendall"         # 肯德尔相关
    MUTUAL_INFO = "mutual_info" # 互信息
    RANK_IC = "rank_ic"         # 排序IC


class FactorQuality(Enum):
    """因子质量等级"""
    EXCELLENT = "excellent"     # 优秀 (IC > 0.1)
    GOOD = "good"              # 良好 (0.05 < IC <= 0.1)
    MODERATE = "moderate"       # 一般 (0.02 < IC <= 0.05)
    POOR = "poor"              # 较差 (IC <= 0.02)
    INVALID = "invalid"         # 无效 (IC < 0)


@dataclass
class ICAnalysisResult:
    """IC分析结果"""
    ic_value: float = 0.0           # IC值
    ic_std: float = 0.0             # IC标准差
    ic_ir: float = 0.0              # IC信息比率 (IC均值/IC标准差)
    ic_win_rate: float = 0.0        # IC胜率
    ic_mean: float = 0.0            # IC均值
    ic_median: float = 0.0          # IC中位数
    ic_skewness: float = 0.0        # IC偏度
    ic_kurtosis: float = 0.0        # IC峰度
    ic_series: pd.Series = None     # IC时间序列
    quality: FactorQuality = FactorQuality.INVALID  # 因子质量


@dataclass
class FactorAnalysisResult:
    """因子分析结果"""
    factor_name: str = ""           # 因子名称
    ic_analysis: ICAnalysisResult = None  # IC分析结果
    return_analysis: Dict[str, float] = None  # 收益分析
    risk_analysis: Dict[str, float] = None    # 风险分析
    stability_analysis: Dict[str, float] = None  # 稳定性分析
    correlation_analysis: Dict[str, float] = None  # 相关性分析
    factor_exposure: pd.DataFrame = None  # 因子暴露
    factor_returns: pd.Series = None  # 因子收益
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            'factor_name': self.factor_name,
            'ic_analysis': {
                'ic_value': self.ic_analysis.ic_value if self.ic_analysis else 0,
                'ic_std': self.ic_analysis.ic_std if self.ic_analysis else 0,
                'ic_ir': self.ic_analysis.ic_ir if self.ic_analysis else 0,
                'ic_win_rate': self.ic_analysis.ic_win_rate if self.ic_analysis else 0,
                'quality': self.ic_analysis.quality.value if self.ic_analysis else FactorQuality.INVALID.value
            },
            'return_analysis': self.return_analysis or {},
            'risk_analysis': self.risk_analysis or {},
            'stability_analysis': self.stability_analysis or {},
            'correlation_analysis': self.correlation_analysis or {}
        }
        return result


class FactorAnalyzer:
    """因子分析器"""
    
    def __init__(self, 
                 method: AnalysisMethod = AnalysisMethod.SPEARMAN,
                 forward_periods: List[int] = [1, 5, 10, 20],
                 quantiles: List[float] = [0.2, 0.4, 0.6, 0.8],
                 min_periods: int = 20,
                 outlier_method: str = 'winsorize',
                 outlier_threshold: float = 0.05):
        """
        初始化因子分析器
        
        Args:
            method: 分析方法
            forward_periods: 前瞻期列表
            quantiles: 分位数列表
            min_periods: 最小观测期数
            outlier_method: 异常值处理方法
            outlier_threshold: 异常值阈值
        """
        self.method = method
        self.forward_periods = forward_periods
        self.quantiles = quantiles
        self.min_periods = min_periods
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        
        # 缓存
        self.analysis_cache = {}
        self.factor_data_cache = {}
        
        self.logger = logging.getLogger(__name__)
        
        # 标准化器
        self.scaler = RobustScaler()
    
    def analyze_factor(self, 
                      factor_data: pd.DataFrame,
                      return_data: pd.DataFrame,
                      factor_name: str,
                      benchmark_returns: Optional[pd.Series] = None) -> FactorAnalysisResult:
        """
        分析单个因子
        
        Args:
            factor_data: 因子数据 (date, symbol, factor_value)
            return_data: 收益数据 (date, symbol, return)
            factor_name: 因子名称
            benchmark_returns: 基准收益
            
        Returns:
            FactorAnalysisResult: 分析结果
        """
        try:
            self.logger.info(f"Analyzing factor: {factor_name}")
            
            # 数据预处理
            clean_factor_data, clean_return_data = self._preprocess_data(
                factor_data, return_data, factor_name
            )
            
            if clean_factor_data.empty or clean_return_data.empty:
                self.logger.warning(f"No valid data for factor {factor_name}")
                return FactorAnalysisResult(factor_name=factor_name)
            
            # IC分析
            ic_analysis = self._analyze_ic(clean_factor_data, clean_return_data)
            
            # 收益分析
            return_analysis = self._analyze_returns(
                clean_factor_data, clean_return_data, benchmark_returns
            )
            
            # 风险分析
            risk_analysis = self._analyze_risk(clean_factor_data, clean_return_data)
            
            # 稳定性分析
            stability_analysis = self._analyze_stability(clean_factor_data, clean_return_data)
            
            # 相关性分析
            correlation_analysis = self._analyze_correlation(clean_factor_data)
            
            # 因子暴露和收益
            factor_exposure, factor_returns = self._calculate_factor_exposure_returns(
                clean_factor_data, clean_return_data
            )
            
            return FactorAnalysisResult(
                factor_name=factor_name,
                ic_analysis=ic_analysis,
                return_analysis=return_analysis,
                risk_analysis=risk_analysis,
                stability_analysis=stability_analysis,
                correlation_analysis=correlation_analysis,
                factor_exposure=factor_exposure,
                factor_returns=factor_returns
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing factor {factor_name}: {e}")
            return FactorAnalysisResult(factor_name=factor_name)
    
    def batch_analyze_factors(self,
                            factor_data_dict: Dict[str, pd.DataFrame],
                            return_data: pd.DataFrame,
                            benchmark_returns: Optional[pd.Series] = None,
                            n_jobs: int = 1) -> Dict[str, FactorAnalysisResult]:
        """
        批量分析因子
        
        Args:
            factor_data_dict: 因子数据字典 {factor_name: factor_data}
            return_data: 收益数据
            benchmark_returns: 基准收益
            n_jobs: 并行任务数
            
        Returns:
            Dict[str, FactorAnalysisResult]: 分析结果字典
        """
        results = {}
        
        if n_jobs == 1:
            # 串行处理
            for factor_name, factor_data in factor_data_dict.items():
                results[factor_name] = self.analyze_factor(
                    factor_data, return_data, factor_name, benchmark_returns
                )
        else:
            # 并行处理
            from joblib import Parallel, delayed
            
            factor_items = list(factor_data_dict.items())
            parallel_results = Parallel(n_jobs=n_jobs)(
                delayed(self.analyze_factor)(
                    factor_data, return_data, factor_name, benchmark_returns
                )
                for factor_name, factor_data in factor_items
            )
            
            results = dict(zip(factor_data_dict.keys(), parallel_results))
        
        return results
    
    def _preprocess_data(self, 
                        factor_data: pd.DataFrame,
                        return_data: pd.DataFrame,
                        factor_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """数据预处理"""
        # 确保列名正确
        if factor_name not in factor_data.columns:
            factor_data = factor_data.rename(columns={factor_data.columns[-1]: factor_name})
        
        # 确保索引对齐
        factor_data = factor_data.set_index(['date', 'symbol']) if 'date' in factor_data.columns else factor_data
        return_data = return_data.set_index(['date', 'symbol']) if 'date' in return_data.columns else return_data
        
        # 对齐数据
        common_index = factor_data.index.intersection(return_data.index)
        factor_data = factor_data.loc[common_index]
        return_data = return_data.loc[common_index]
        
        # 删除缺失值
        valid_idx = factor_data[factor_name].notna() & return_data.iloc[:, 0].notna()
        factor_data = factor_data[valid_idx]
        return_data = return_data[valid_idx]
        
        # 异常值处理
        factor_data = self._handle_outliers(factor_data, factor_name)
        
        return factor_data, return_data
    
    def _handle_outliers(self, data: pd.DataFrame, factor_name: str) -> pd.DataFrame:
        """处理异常值"""
        if self.outlier_method == 'winsorize':
            # Winsorize处理
            from scipy.stats import mstats
            data[factor_name] = mstats.winsorize(
                data[factor_name], 
                limits=[self.outlier_threshold, self.outlier_threshold]
            )
        elif self.outlier_method == 'clip':
            # 截断处理
            lower = data[factor_name].quantile(self.outlier_threshold)
            upper = data[factor_name].quantile(1 - self.outlier_threshold)
            data[factor_name] = data[factor_name].clip(lower, upper)
        elif self.outlier_method == 'zscore':
            # Z-score处理
            z_scores = np.abs(stats.zscore(data[factor_name]))
            data = data[z_scores < 3]
        
        return data
    
    def _analyze_ic(self, 
                   factor_data: pd.DataFrame,
                   return_data: pd.DataFrame) -> ICAnalysisResult:
        """IC分析"""
        try:
            factor_name = factor_data.columns[0]
            return_name = return_data.columns[0]
            
            # 计算各期IC
            ic_results = {}
            
            for period in self.forward_periods:
                # 计算前瞻收益
                forward_returns = self._calculate_forward_returns(return_data, period)
                
                # 对齐数据
                aligned_factor = factor_data[factor_name].reindex(forward_returns.index).dropna()
                aligned_returns = forward_returns.reindex(aligned_factor.index).dropna()
                
                common_idx = aligned_factor.index.intersection(aligned_returns.index)
                if len(common_idx) < self.min_periods:
                    continue
                
                # 按日期分组计算IC
                ic_series = []
                dates = common_idx.get_level_values('date').unique()
                
                for date in dates:
                    date_factor = aligned_factor.loc[date] if date in aligned_factor.index.get_level_values('date') else pd.Series()
                    date_returns = aligned_returns.loc[date] if date in aligned_returns.index.get_level_values('date') else pd.Series()
                    
                    if len(date_factor) < 5 or len(date_returns) < 5:  # 至少需要5个样本
                        continue
                    
                    # 计算IC
                    if self.method == AnalysisMethod.PEARSON:
                        ic, p_value = pearsonr(date_factor, date_returns)
                    elif self.method == AnalysisMethod.SPEARMAN:
                        ic, p_value = spearmanr(date_factor, date_returns)
                    elif self.method == AnalysisMethod.KENDALL:
                        ic, p_value = kendalltau(date_factor, date_returns)
                    elif self.method == AnalysisMethod.RANK_IC:
                        # 排序IC
                        factor_ranks = date_factor.rank()
                        return_ranks = date_returns.rank()
                        ic, p_value = spearmanr(factor_ranks, return_ranks)
                    else:
                        ic = 0.0
                    
                    if not np.isnan(ic):
                        ic_series.append(ic)
                
                if ic_series:
                    ic_results[f'period_{period}'] = pd.Series(ic_series)
            
            # 使用第一个周期的结果作为主要IC
            if ic_results:
                main_ic_series = ic_results[f'period_{self.forward_periods[0]}']
                
                ic_value = main_ic_series.mean()
                ic_std = main_ic_series.std()
                ic_ir = ic_value / ic_std if ic_std > 0 else 0
                ic_win_rate = (main_ic_series > 0).mean()
                ic_median = main_ic_series.median()
                ic_skewness = main_ic_series.skew()
                ic_kurtosis = main_ic_series.kurtosis()
                
                # 判断因子质量
                if ic_value > 0.1:
                    quality = FactorQuality.EXCELLENT
                elif ic_value > 0.05:
                    quality = FactorQuality.GOOD
                elif ic_value > 0.02:
                    quality = FactorQuality.MODERATE
                elif ic_value > 0:
                    quality = FactorQuality.POOR
                else:
                    quality = FactorQuality.INVALID
                
                return ICAnalysisResult(
                    ic_value=ic_value,
                    ic_std=ic_std,
                    ic_ir=ic_ir,
                    ic_win_rate=ic_win_rate,
                    ic_mean=ic_value,
                    ic_median=ic_median,
                    ic_skewness=ic_skewness,
                    ic_kurtosis=ic_kurtosis,
                    ic_series=main_ic_series,
                    quality=quality
                )
            
            return ICAnalysisResult()
            
        except Exception as e:
            self.logger.error(f"Error in IC analysis: {e}")
            return ICAnalysisResult()
    
    def _calculate_forward_returns(self, 
                                  return_data: pd.DataFrame,
                                  periods: int) -> pd.Series:
        """计算前瞻收益"""
        return_series = return_data.iloc[:, 0]
        
        # 按股票分组计算前瞻收益
        forward_returns = return_series.groupby('symbol').apply(
            lambda x: x.shift(-periods)
        )
        
        return forward_returns.dropna()
    
    def _analyze_returns(self,
                        factor_data: pd.DataFrame,
                        return_data: pd.DataFrame,
                        benchmark_returns: Optional[pd.Series] = None) -> Dict[str, float]:
        """收益分析"""
        try:
            factor_name = factor_data.columns[0]
            
            # 分层回测
            layer_returns = self._calculate_layer_returns(factor_data, return_data)
            
            if not layer_returns:
                return {}
            
            # 计算多空收益
            if len(layer_returns) >= 2:
                long_short_return = layer_returns['top'] - layer_returns['bottom']
            else:
                long_short_return = 0.0
            
            # 计算信息比率
            if len(layer_returns) >= 2:
                long_short_volatility = np.std([layer_returns['top'], layer_returns['bottom']])
                information_ratio = long_short_return / long_short_volatility if long_short_volatility > 0 else 0
            else:
                information_ratio = 0.0
            
            # 计算最大回撤
            max_drawdown = self._calculate_max_drawdown(layer_returns.get('top', [0]))
            
            return {
                'long_short_return': long_short_return,
                'information_ratio': information_ratio,
                'max_drawdown': max_drawdown,
                'top_layer_return': layer_returns.get('top', 0),
                'bottom_layer_return': layer_returns.get('bottom', 0),
                **{f'layer_{i}_return': ret for i, ret in enumerate(layer_returns.values())}
            }
            
        except Exception as e:
            self.logger.error(f"Error in return analysis: {e}")
            return {}
    
    def _calculate_layer_returns(self,
                                factor_data: pd.DataFrame,
                                return_data: pd.DataFrame) -> Dict[str, float]:
        """计算分层收益"""
        factor_name = factor_data.columns[0]
        return_name = return_data.columns[0]
        
        # 合并数据
        combined_data = factor_data.join(return_data, how='inner')
        
        if combined_data.empty:
            return {}
        
        # 按日期分组，计算分层收益
        daily_layer_returns = []
        
        dates = combined_data.index.get_level_values('date').unique()
        
        for date in dates:
            date_data = combined_data.loc[date] if date in combined_data.index.get_level_values('date') else pd.DataFrame()
            
            if len(date_data) < len(self.quantiles) + 1:
                continue
            
            # 按因子值排序并分层
            date_data_sorted = date_data.sort_values(factor_name)
            n_stocks = len(date_data_sorted)
            
            layer_returns = {}
            
            for i, quantile in enumerate(self.quantiles):
                start_idx = int(n_stocks * (quantile - 0.2 if i > 0 else 0))
                end_idx = int(n_stocks * quantile)
                
                layer_data = date_data_sorted.iloc[start_idx:end_idx]
                if not layer_data.empty:
                    layer_return = layer_data[return_name].mean()
                    layer_returns[f'layer_{i}'] = layer_return
            
            # 最高和最低分层
            top_10_pct = int(n_stocks * 0.9)
            bottom_10_pct = int(n_stocks * 0.1)
            
            layer_returns['top'] = date_data_sorted.iloc[top_10_pct:][return_name].mean()
            layer_returns['bottom'] = date_data_sorted.iloc[:bottom_10_pct][return_name].mean()
            
            daily_layer_returns.append(layer_returns)
        
        # 计算平均收益
        if not daily_layer_returns:
            return {}
        
        avg_layer_returns = {}
        all_keys = set()
        for day_returns in daily_layer_returns:
            all_keys.update(day_returns.keys())
        
        for key in all_keys:
            returns = [day_returns.get(key, 0) for day_returns in daily_layer_returns]
            avg_layer_returns[key] = np.mean(returns)
        
        return avg_layer_returns
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """计算最大回撤"""
        if not returns:
            return 0.0
        
        cumulative = np.cumprod(1 + np.array(returns))
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        return np.min(drawdown)
    
    def _analyze_risk(self,
                     factor_data: pd.DataFrame,
                     return_data: pd.DataFrame) -> Dict[str, float]:
        """风险分析"""
        try:
            # 计算因子暴露的波动率
            factor_name = factor_data.columns[0]
            factor_volatility = factor_data[factor_name].std()
            
            # 计算分层组合的风险
            layer_returns = self._calculate_layer_returns(factor_data, return_data)
            
            if not layer_returns:
                return {'factor_volatility': factor_volatility}
            
            # 多空组合风险
            if 'top' in layer_returns and 'bottom' in layer_returns:
                long_short_volatility = abs(layer_returns['top'] - layer_returns['bottom']) * 0.1  # 简化计算
            else:
                long_short_volatility = 0.0
            
            # VaR计算 (简化)
            returns_list = list(layer_returns.values())
            if returns_list:
                var_95 = np.percentile(returns_list, 5)
            else:
                var_95 = 0.0
            
            return {
                'factor_volatility': factor_volatility,
                'long_short_volatility': long_short_volatility,
                'var_95': var_95,
                'factor_skewness': factor_data[factor_name].skew(),
                'factor_kurtosis': factor_data[factor_name].kurtosis()
            }
            
        except Exception as e:
            self.logger.error(f"Error in risk analysis: {e}")
            return {}
    
    def _analyze_stability(self,
                          factor_data: pd.DataFrame,
                          return_data: pd.DataFrame) -> Dict[str, float]:
        """稳定性分析"""
        try:
            factor_name = factor_data.columns[0]
            
            # 时间稳定性：计算滚动IC
            rolling_ics = self._calculate_rolling_ic(factor_data, return_data)
            
            if rolling_ics.empty:
                return {}
            
            # IC稳定性指标
            ic_stability = 1 - rolling_ics.std() / abs(rolling_ics.mean()) if rolling_ics.mean() != 0 else 0
            ic_consistency = (rolling_ics > 0).mean()  # IC为正的比例
            
            # 因子值稳定性
            factor_stability = self._calculate_factor_stability(factor_data, factor_name)
            
            return {
                'ic_stability': ic_stability,
                'ic_consistency': ic_consistency,
                'factor_stability': factor_stability,
                'rolling_ic_mean': rolling_ics.mean(),
                'rolling_ic_std': rolling_ics.std()
            }
            
        except Exception as e:
            self.logger.error(f"Error in stability analysis: {e}")
            return {}
    
    def _calculate_rolling_ic(self,
                             factor_data: pd.DataFrame,
                             return_data: pd.DataFrame,
                             window: int = 20) -> pd.Series:
        """计算滚动IC"""
        factor_name = factor_data.columns[0]
        return_name = return_data.columns[0]
        
        # 合并数据
        combined_data = factor_data.join(return_data, how='inner')
        
        if combined_data.empty:
            return pd.Series()
        
        # 按日期分组计算IC时间序列
        dates = combined_data.index.get_level_values('date').unique().sort_values()
        ic_series = []
        
        for date in dates:
            date_data = combined_data.loc[date] if date in combined_data.index.get_level_values('date') else pd.DataFrame()
            
            if len(date_data) < 5:
                continue
            
            if self.method == AnalysisMethod.SPEARMAN:
                ic, _ = spearmanr(date_data[factor_name], date_data[return_name])
            else:
                ic, _ = pearsonr(date_data[factor_name], date_data[return_name])
            
            if not np.isnan(ic):
                ic_series.append(ic)
        
        ic_series = pd.Series(ic_series, index=dates[:len(ic_series)])
        
        # 计算滚动IC
        rolling_ic = ic_series.rolling(window=window, min_periods=window//2).mean()
        
        return rolling_ic.dropna()
    
    def _calculate_factor_stability(self,
                                   factor_data: pd.DataFrame,
                                   factor_name: str) -> float:
        """计算因子稳定性"""
        # 计算因子值的时间序列稳定性
        dates = factor_data.index.get_level_values('date').unique()
        
        if len(dates) < 2:
            return 0.0
        
        # 计算每个时间点的因子分布
        date_means = []
        for date in dates:
            date_data = factor_data.loc[date] if date in factor_data.index.get_level_values('date') else pd.Series()
            if not date_data.empty:
                date_means.append(date_data[factor_name].mean())
        
        if len(date_means) < 2:
            return 0.0
        
        # 计算变异系数
        mean_of_means = np.mean(date_means)
        std_of_means = np.std(date_means)
        
        if mean_of_means == 0:
            return 0.0
        
        # 稳定性 = 1 - 变异系数
        stability = 1 - std_of_means / abs(mean_of_means)
        
        return max(0, stability)
    
    def _analyze_correlation(self, factor_data: pd.DataFrame) -> Dict[str, float]:
        """相关性分析"""
        try:
            factor_name = factor_data.columns[0]
            
            # 自相关性分析
            factor_series = factor_data[factor_name]
            
            # 滞后1期自相关
            lag1_corr = factor_series.autocorr(lag=1) if len(factor_series) > 1 else 0
            
            # 滞后5期自相关
            lag5_corr = factor_series.autocorr(lag=5) if len(factor_series) > 5 else 0
            
            return {
                'autocorr_lag1': lag1_corr if not np.isnan(lag1_corr) else 0,
                'autocorr_lag5': lag5_corr if not np.isnan(lag5_corr) else 0,
                'factor_range': factor_series.max() - factor_series.min(),
                'factor_iqr': factor_series.quantile(0.75) - factor_series.quantile(0.25)
            }
            
        except Exception as e:
            self.logger.error(f"Error in correlation analysis: {e}")
            return {}
    
    def _calculate_factor_exposure_returns(self,
                                          factor_data: pd.DataFrame,
                                          return_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """计算因子暴露和因子收益"""
        try:
            factor_name = factor_data.columns[0]
            return_name = return_data.columns[0]
            
            # 合并数据
            combined_data = factor_data.join(return_data, how='inner')
            
            if combined_data.empty:
                return pd.DataFrame(), pd.Series()
            
            # 计算因子收益时间序列
            factor_returns = []
            dates = combined_data.index.get_level_values('date').unique().sort_values()
            
            for date in dates:
                date_data = combined_data.loc[date] if date in combined_data.index.get_level_values('date') else pd.DataFrame()
                
                if len(date_data) < 5:
                    continue
                
                # 回归计算因子收益: r_i = α + β * f_i + ε_i
                X = date_data[factor_name].values.reshape(-1, 1)
                y = date_data[return_name].values
                
                try:
                    from sklearn.linear_model import LinearRegression
                    model = LinearRegression()
                    model.fit(X, y)
                    factor_return = model.coef_[0]
                    factor_returns.append(factor_return)
                except:
                    factor_returns.append(0.0)
            
            factor_returns_series = pd.Series(
                factor_returns,
                index=dates[:len(factor_returns)],
                name=f'{factor_name}_return'
            )
            
            # 因子暴露（简化为因子值的标准化）
            factor_exposure = factor_data.copy()
            factor_exposure[factor_name] = self.scaler.fit_transform(
                factor_exposure[factor_name].values.reshape(-1, 1)
            ).flatten()
            
            return factor_exposure, factor_returns_series
            
        except Exception as e:
            self.logger.error(f"Error calculating factor exposure and returns: {e}")
            return pd.DataFrame(), pd.Series()
    
    def generate_factor_report(self, 
                              analysis_result: FactorAnalysisResult,
                              save_path: Optional[str] = None) -> str:
        """生成因子分析报告"""
        factor_name = analysis_result.factor_name
        ic_analysis = analysis_result.ic_analysis
        
        report = f"""
因子分析报告: {factor_name}
{'='*60}

IC分析:
  IC值: {ic_analysis.ic_value:.4f}
  IC标准差: {ic_analysis.ic_std:.4f}
  IC信息比率: {ic_analysis.ic_ir:.4f}
  IC胜率: {ic_analysis.ic_win_rate:.2%}
  因子质量: {ic_analysis.quality.value}

收益分析:
"""
        
        if analysis_result.return_analysis:
            for key, value in analysis_result.return_analysis.items():
                report += f"  {key}: {value:.4f}\n"
        
        report += "\n风险分析:\n"
        if analysis_result.risk_analysis:
            for key, value in analysis_result.risk_analysis.items():
                report += f"  {key}: {value:.4f}\n"
        
        report += "\n稳定性分析:\n"
        if analysis_result.stability_analysis:
            for key, value in analysis_result.stability_analysis.items():
                report += f"  {key}: {value:.4f}\n"
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
        
        return report
    
    def create_factor_charts(self,
                            analysis_result: FactorAnalysisResult,
                            save_dir: Optional[str] = None,
                            figsize: Tuple[int, int] = (15, 12)) -> None:
        """创建因子分析图表"""
        try:
            factor_name = analysis_result.factor_name
            
            fig, axes = plt.subplots(2, 3, figsize=figsize)
            
            # 1. IC时间序列
            if analysis_result.ic_analysis and analysis_result.ic_analysis.ic_series is not None:
                ic_series = analysis_result.ic_analysis.ic_series
                axes[0, 0].plot(ic_series.index, ic_series.values, 'b-', linewidth=2)
                axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.7)
                axes[0, 0].set_title(f'{factor_name} - IC Time Series')
                axes[0, 0].set_ylabel('IC Value')
                axes[0, 0].grid(True, alpha=0.3)
            
            # 2. IC分布
            if analysis_result.ic_analysis and analysis_result.ic_analysis.ic_series is not None:
                ic_series = analysis_result.ic_analysis.ic_series
                axes[0, 1].hist(ic_series.values, bins=30, alpha=0.7, density=True)
                axes[0, 1].axvline(ic_series.mean(), color='r', linestyle='--', 
                                  label=f'Mean: {ic_series.mean():.4f}')
                axes[0, 1].set_title(f'{factor_name} - IC Distribution')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
            
            # 3. 因子收益
            if analysis_result.factor_returns is not None and not analysis_result.factor_returns.empty:
                cumulative_returns = (1 + analysis_result.factor_returns).cumprod()
                axes[0, 2].plot(cumulative_returns.index, cumulative_returns.values, 'g-', linewidth=2)
                axes[0, 2].set_title(f'{factor_name} - Cumulative Factor Returns')
                axes[0, 2].set_ylabel('Cumulative Return')
                axes[0, 2].grid(True, alpha=0.3)
            
            # 4. 分层收益
            if analysis_result.return_analysis:
                layer_keys = [k for k in analysis_result.return_analysis.keys() if 'layer_' in k]
                if layer_keys:
                    layer_returns = [analysis_result.return_analysis[k] for k in layer_keys]
                    layer_names = [k.replace('layer_', 'Q') for k in layer_keys]
                    
                    axes[1, 0].bar(layer_names, layer_returns)
                    axes[1, 0].set_title(f'{factor_name} - Layer Returns')
                    axes[1, 0].set_ylabel('Return')
                    axes[1, 0].grid(True, alpha=0.3)
            
            # 5. 因子暴露分布
            if analysis_result.factor_exposure is not None and not analysis_result.factor_exposure.empty:
                factor_values = analysis_result.factor_exposure.iloc[:, 0]
                axes[1, 1].hist(factor_values, bins=50, alpha=0.7, density=True)
                axes[1, 1].set_title(f'{factor_name} - Factor Distribution')
                axes[1, 1].set_xlabel('Factor Value')
                axes[1, 1].grid(True, alpha=0.3)
            
            # 6. 滚动IC
            if analysis_result.ic_analysis and analysis_result.ic_analysis.ic_series is not None:
                ic_series = analysis_result.ic_analysis.ic_series
                rolling_ic = ic_series.rolling(window=20, min_periods=10).mean()
                axes[1, 2].plot(rolling_ic.index, rolling_ic.values, 'purple', linewidth=2)
                axes[1, 2].axhline(y=0, color='r', linestyle='--', alpha=0.7)
                axes[1, 2].set_title(f'{factor_name} - Rolling IC (20-period)')
                axes[1, 2].set_ylabel('Rolling IC')
                axes[1, 2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_dir:
                from pathlib import Path
                save_dir = Path(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                plt.savefig(save_dir / f"{factor_name}_analysis_{timestamp}.png", 
                           dpi=300, bbox_inches='tight')
            
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Error creating charts for {analysis_result.factor_name}: {e}")


class ICAnalyzer:
    """IC分析器 - 专门用于IC相关分析"""
    
    def __init__(self):
        """初始化IC分析器"""
        self.logger = logging.getLogger(__name__)
    
    def calculate_ic_decay(self,
                          factor_data: pd.DataFrame,
                          return_data: pd.DataFrame,
                          periods: List[int] = [1, 3, 5, 10, 20]) -> pd.DataFrame:
        """计算IC衰减"""
        factor_name = factor_data.columns[0]
        return_name = return_data.columns[0]
        
        ic_decay_results = []
        
        for period in periods:
            # 计算前瞻收益
            forward_returns = self._calculate_forward_returns(return_data, period)
            
            # 对齐数据
            aligned_data = factor_data.join(forward_returns, how='inner')
            
            if aligned_data.empty:
                ic_decay_results.append({'period': period, 'ic': np.nan})
                continue
            
            # 计算IC
            ic_values = []
            dates = aligned_data.index.get_level_values('date').unique()
            
            for date in dates:
                date_data = aligned_data.loc[date] if date in aligned_data.index.get_level_values('date') else pd.DataFrame()
                
                if len(date_data) < 5:
                    continue
                
                ic, _ = spearmanr(date_data[factor_name], date_data.iloc[:, 1])
                if not np.isnan(ic):
                    ic_values.append(ic)
            
            mean_ic = np.mean(ic_values) if ic_values else np.nan
            ic_decay_results.append({'period': period, 'ic': mean_ic})
        
        return pd.DataFrame(ic_decay_results)
    
    def _calculate_forward_returns(self, return_data: pd.DataFrame, periods: int) -> pd.Series:
        """计算前瞻收益"""
        return_series = return_data.iloc[:, 0]
        
        # 按股票分组计算前瞻收益
        forward_returns = return_series.groupby('symbol').apply(
            lambda x: x.shift(-periods)
        )
        
        return forward_returns.dropna()
    
    def calculate_ic_breakdown(self,
                              factor_data: pd.DataFrame,
                              return_data: pd.DataFrame,
                              breakdown_by: str = 'industry') -> pd.DataFrame:
        """计算IC分解（按行业、市值等）"""
        # 简化实现，需要额外的分类数据
        factor_name = factor_data.columns[0]
        return_name = return_data.columns[0]
        
        # 合并数据
        combined_data = factor_data.join(return_data, how='inner')
        
        if combined_data.empty:
            return pd.DataFrame()
        
        # 这里需要根据breakdown_by添加分类逻辑
        # 简化示例：按股票代码首字母分组
        combined_data['group'] = combined_data.index.get_level_values('symbol').str[0]
        
        ic_breakdown = []
        
        for group in combined_data['group'].unique():
            group_data = combined_data[combined_data['group'] == group]
            
            if len(group_data) < 10:
                continue
            
            ic, _ = spearmanr(group_data[factor_name], group_data[return_name])
            
            ic_breakdown.append({
                'group': group,
                'ic': ic if not np.isnan(ic) else 0,
                'sample_size': len(group_data)
            })
        
        return pd.DataFrame(ic_breakdown)
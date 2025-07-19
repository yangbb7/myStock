import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .base_provider import BaseDataProvider, DataRequest, DataResponse, DataType, DataQuality
from .quality_monitor import DataQualityMonitor
from .storage_manager import ShardedStorageManager

@dataclass
class ProviderConfig:
    name: str
    provider_class: str
    config: Dict[str, Any]
    priority: int = 1
    enabled: bool = True
    fallback_providers: List[str] = None

class MultiSourceDataManager:
    def __init__(self, providers_config: List[ProviderConfig]):
        self.providers: Dict[str, BaseDataProvider] = {}
        self.providers_config = providers_config
        self.logger = logging.getLogger(__name__)
        self.quality_monitor = DataQualityMonitor()
        self.storage_manager = ShardedStorageManager()
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.circuit_breaker = {}
        
    async def initialize(self):
        for config in self.providers_config:
            if config.enabled:
                try:
                    provider_class = self._get_provider_class(config.provider_class)
                    provider = provider_class(config.config)
                    await provider.connect()
                    self.providers[config.name] = provider
                    self.circuit_breaker[config.name] = {
                        'failure_count': 0,
                        'last_failure': None,
                        'is_open': False
                    }
                    self.logger.info(f"Initialized provider: {config.name}")
                except Exception as e:
                    self.logger.error(f"Failed to initialize provider {config.name}: {e}")
    
    async def get_data(self, request: DataRequest, preferred_provider: str = None) -> DataResponse:
        if preferred_provider and preferred_provider in self.providers:
            providers_to_try = [preferred_provider]
        else:
            providers_to_try = self._get_ordered_providers(request.data_type)
        
        for provider_name in providers_to_try:
            if self._is_circuit_breaker_open(provider_name):
                continue
                
            try:
                provider = self.providers[provider_name]
                start_time = datetime.now()
                response = await provider.get_data(request)
                
                response.latency_ms = (datetime.now() - start_time).total_seconds() * 1000
                
                await self.quality_monitor.evaluate_data(response)
                
                if response.quality_metrics.overall_score >= 0.7:
                    await self.storage_manager.store_data(response)
                    self._reset_circuit_breaker(provider_name)
                    return response
                    
            except Exception as e:
                self.logger.error(f"Provider {provider_name} failed: {e}")
                self._increment_circuit_breaker(provider_name)
                continue
        
        raise Exception("All providers failed to deliver data")
    
    async def subscribe_realtime(self, symbols: List[str], data_type: DataType, callback) -> Dict[str, bool]:
        results = {}
        
        for provider_name, provider in self.providers.items():
            if data_type in provider.get_supported_data_types():
                try:
                    success = await provider.subscribe_realtime(symbols, data_type, callback)
                    results[provider_name] = success
                except Exception as e:
                    self.logger.error(f"Failed to subscribe to {provider_name}: {e}")
                    results[provider_name] = False
        
        return results
    
    async def get_multi_source_data(self, request: DataRequest) -> Dict[str, DataResponse]:
        tasks = []
        
        for provider_name, provider in self.providers.items():
            if request.data_type in provider.get_supported_data_types():
                task = asyncio.create_task(
                    self._get_data_from_provider(provider_name, provider, request)
                )
                tasks.append((provider_name, task))
        
        results = {}
        for provider_name, task in tasks:
            try:
                response = await task
                results[provider_name] = response
            except Exception as e:
                self.logger.error(f"Provider {provider_name} failed: {e}")
                results[provider_name] = None
        
        return results
    
    async def backfill_historical_data(self, symbols: List[str], start_date: datetime, end_date: datetime, data_type: DataType):
        tasks = []
        
        for symbol in symbols:
            request = DataRequest(
                symbol=symbol,
                data_type=data_type,
                start_time=start_date,
                end_time=end_date
            )
            
            task = asyncio.create_task(self._backfill_symbol(request))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful_backfills = sum(1 for r in results if not isinstance(r, Exception))
        self.logger.info(f"Backfilled {successful_backfills}/{len(symbols)} symbols")
        
        return successful_backfills
    
    async def get_health_status(self) -> Dict[str, Any]:
        status = {
            'total_providers': len(self.providers),
            'active_providers': 0,
            'provider_status': {},
            'circuit_breakers': self.circuit_breaker,
            'quality_metrics': await self.quality_monitor.get_overall_metrics()
        }
        
        for name, provider in self.providers.items():
            health = await provider.health_check()
            status['provider_status'][name] = health
            if health['is_connected']:
                status['active_providers'] += 1
        
        return status
    
    def _get_provider_class(self, class_name: str):
        if class_name == 'WindDataProvider':
            from .providers.wind_provider import WindDataProvider
            return WindDataProvider
        elif class_name == 'BloombergDataProvider':
            from .providers.bloomberg_provider import BloombergDataProvider
            return BloombergDataProvider
        elif class_name == 'ReutersDataProvider':
            from .providers.reuters_provider import ReutersDataProvider
            return ReutersDataProvider
        elif class_name == 'AlphaDataProvider':
            from .providers.alpha_provider import AlphaDataProvider
            return AlphaDataProvider
        else:
            raise ValueError(f"Unknown provider class: {class_name}")
    
    def _get_ordered_providers(self, data_type: DataType) -> List[str]:
        providers = []
        for config in sorted(self.providers_config, key=lambda x: x.priority):
            if config.enabled and config.name in self.providers:
                provider = self.providers[config.name]
                if data_type in provider.get_supported_data_types():
                    providers.append(config.name)
        return providers
    
    def _is_circuit_breaker_open(self, provider_name: str) -> bool:
        breaker = self.circuit_breaker.get(provider_name, {})
        return breaker.get('is_open', False)
    
    def _increment_circuit_breaker(self, provider_name: str):
        breaker = self.circuit_breaker.get(provider_name, {})
        breaker['failure_count'] = breaker.get('failure_count', 0) + 1
        breaker['last_failure'] = datetime.now()
        
        if breaker['failure_count'] >= 5:
            breaker['is_open'] = True
            self.logger.warning(f"Circuit breaker opened for {provider_name}")
    
    def _reset_circuit_breaker(self, provider_name: str):
        if provider_name in self.circuit_breaker:
            self.circuit_breaker[provider_name]['failure_count'] = 0
            self.circuit_breaker[provider_name]['is_open'] = False
    
    async def _get_data_from_provider(self, provider_name: str, provider: BaseDataProvider, request: DataRequest) -> DataResponse:
        try:
            return await provider.get_data(request)
        except Exception as e:
            self.logger.error(f"Provider {provider_name} failed: {e}")
            raise
    
    async def _backfill_symbol(self, request: DataRequest) -> bool:
        try:
            response = await self.get_data(request)
            await self.storage_manager.store_data(response)
            return True
        except Exception as e:
            self.logger.error(f"Backfill failed for {request.symbol}: {e}")
            return False
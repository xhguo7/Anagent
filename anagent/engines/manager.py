"""
Engine manager for sharing engines across agents.
"""

from typing import Dict, Any, Optional
from .engine import BaseEngine, EngineHub


class EngineManager:
    """Singleton manager for sharing engines across multiple agents."""
    
    _instance = None
    _engines: Dict[str, BaseEngine] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EngineManager, cls).__new__(cls)
        return cls._instance
    
    def get_engine(
        self,
        model_path: str,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        **engine_kwargs
    ) -> BaseEngine:
        """Get or create an engine for the given model."""
        # Create a cache key based on model_path and critical engine parameters
        cache_key = self._create_cache_key(model_path, engine_kwargs)
        
        if cache_key not in self._engines:
            engine = EngineHub.create_engine(
                model_path=model_path,
                temperature=temperature,
                max_tokens=max_tokens,
                **engine_kwargs
            )
            self._engines[cache_key] = engine

        return self._engines[cache_key]
    
    def _create_cache_key(self, model_path: str, engine_kwargs: Dict[str, Any]) -> str:
        """Create a cache key that includes model path and GPU-affecting parameters."""
        # Parameters that affect GPU allocation and should be part of cache key
        gpu_params = ['device', 'device_map', 'load_in_8bit', 'load_in_4bit', 
                      'torch_dtype', 'attn_implementation']
        
        key_parts = [model_path]
        
        for param in gpu_params:
            if param in engine_kwargs:
                key_parts.append(f"{param}={engine_kwargs[param]}")
        
        return "||".join(key_parts)
    
    def clear_cache(self):
        """Clear all cached engines."""
        self._engines.clear()
    
    def get_cached_models(self) -> list:
        """Get list of cached model keys."""
        return list(self._engines.keys())
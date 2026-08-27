"""
Base engine interface.
"""

import os
from PIL import Image
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union

from utils.logger import setup_logger
logger = setup_logger(__name__)


class BaseEngine(ABC):
    """Abstract base class for all model engines."""
    
    def __init__(
        self,
        model_path: str,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        top_p: float = 0.9,
        repetition_penalty: float = 1.05,
        device: str = "cuda",
        cache_dir: Optional[str] = None,
        **kwargs
    ):
        """Initialize base engine."""
        self.model_path = model_path
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.device = device
        self.cache_dir = cache_dir or os.getenv("HF_HOME", "/tmp/cache")
        self.model = None
        self.processor = None
        self.tokenizer = None
        
    @abstractmethod
    def load_model(self):
        """Load the model and necessary components."""
        pass
    
    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate response from messages."""
        pass
    
    def format_messages(
        self,
        messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Format messages for the specific engine."""
        return messages
    
    def __str__(self):
        return f"{self.__class__.__name__}(model={self.model_path})"
    
    def __repr__(self):
        return self.__str__()


class EngineHub:
    """Central hub for managing different engines."""
    
    # Registry of available engines
    _engines = {}
    
    @classmethod
    def register_engine(cls, name: str, engine_class: type):
        """Register an engine class."""
        cls._engines[name.lower()] = engine_class
    
    @classmethod
    def create_engine(
        cls,
        model_path: str,
        engine_type: Optional[str] = None,
        **kwargs
    ) -> BaseEngine:
        """Factory method to create appropriate engine based on model path."""
        if engine_type:
            engine_type = engine_type.lower()
            if engine_type not in cls._engines:
                raise ValueError(f"Unknown engine type: {engine_type}")
            return cls._engines[engine_type](model_path=model_path, **kwargs)
        
        # Auto-detect engine type from model_path
        model_path_lower = model_path.lower()
        
        # API / served engines
        if model_path.startswith("vllm/"):
            return cls._engines["vllm"](model_path=model_path, **kwargs)
        elif model_path.startswith("azure/"):
            return cls._engines["azure"](model_path=model_path, **kwargs)
        elif model_path.startswith("openai/"):
            return cls._engines["openai"](model_path=model_path, **kwargs)
        elif model_path.startswith("gemini/"):
            return cls._engines["gemini"](model_path=model_path, **kwargs)
        elif model_path.startswith("anthropic/"):
            return cls._engines["anthropic"](model_path=model_path, **kwargs)
        
        # Local models
        elif "qwen2" in model_path_lower:
            return cls._engines["qwen2_5vl"](model_path=model_path, **kwargs)
        elif "qwen3" in model_path_lower:
            return cls._engines["qwen3vl"](model_path=model_path, **kwargs)
        elif "internvl" in model_path_lower:
            return cls._engines["internvl"](model_path=model_path, **kwargs)
        elif "llama" in model_path_lower:
            return cls._engines["llama"](model_path=model_path, **kwargs)
        elif "gemma" in model_path_lower:
            return cls._engines["gemma"](model_path=model_path, **kwargs)
        elif "llava" in model_path_lower:
            return cls._engines["llava"](model_path=model_path, **kwargs)
        elif "phi" in model_path_lower:
            return cls._engines["phi"](model_path=model_path, **kwargs)
        
        else:
            raise ValueError(
                f"Cannot auto-detect engine type for model: {model_path}. "
                f"Please specify engine_type explicitly."
            )
    
    @classmethod
    def list_engines(cls) -> List[str]:
        """List all registered engines."""
        return list(cls._engines.keys())
"""
Engines module.
"""

from .engine import BaseEngine, EngineHub
from .manager import EngineManager
from .openai import OpenAIEngine
from .azure import AzureEngine
from .vllm import VLLMEngine
from .qwen2_5_vl import Qwen2_5VLEngine
from .qwen3_vl import Qwen3VLEngine
from .internvl import InternVLEngine
from .internvl3_5 import InternVL3_5Engine
from .gemini import GeminiEngine


__all__ = [
    'BaseEngine',
    'EngineHub',
    'EngineManager',
    'OpenAIEngine',
    'AzureEngine',
    'VLLMEngine',
    'Qwen2_5VLEngine',
    'Qwen3VLEngine',
    'InternVLEngine',
    'InternVL3_5Engine',
    'GeminiEngine',
]


def create_engine(model_path: str, engine_type: str = None, **kwargs) -> BaseEngine:
    """Convenience function to create an engine."""
    return EngineHub.create_engine(model_path=model_path, engine_type=engine_type, **kwargs)
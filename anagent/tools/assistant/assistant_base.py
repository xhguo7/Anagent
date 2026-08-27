"""
Base class for LLM-powered tools.
"""

import os
from typing import Dict, Any, List, Optional
from abc import abstractmethod


from utils.logger import setup_logger
logger = setup_logger(__name__)

from anagent.tools.base import BaseTool, ToolResult, ToolStatus
from anagent.engines import EngineHub


class BaseLLMTool(BaseTool):
    """Base class for LLM-powered tools."""
    
    def __init__(
        self,
        name: str,
        description: str,
        tool_model: str = "openai/gpt-4o-mini",
        temperature: float = 0.3,
        max_tokens: int = 2048,
        engine_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Initialize LLM tool."""
        super().__init__(
            name=name,
            description=description,
            **kwargs
        )
        
        self.tool_model = tool_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize engine
        engine_kwargs = engine_kwargs or {}
        self.engine = EngineHub.create_engine(
            model_path=tool_model,
            temperature=temperature,
            max_tokens=max_tokens,
            **engine_kwargs
        )
        
        logger.info(f"[{self.name}] Initialized with engine: {self.engine}")
    @abstractmethod
    def _build_prompt(self, **kwargs) -> List[Dict[str, Any]]:
        """Build prompt messages for the LLM."""
        pass
    
    def _call_llm(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Call the LLM engine."""
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        
        response = self.engine.generate(
            messages=messages,
            temperature=temp,
            max_tokens=max_tok
        )
        
        return response
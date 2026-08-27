"""
Base agent foundation.
"""

import os
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
from datetime import datetime


from utils.logger import setup_logger
logger = setup_logger(__name__)

from anagent.engines import EngineHub


class BaseAgent(ABC):
    """Base class for all agents."""
    
    def __init__(
        self,
        name: str,
        role: str,
        args: Any,
        engine_kwargs: Optional[Dict[str, Any]] = None,
        engine: Optional[Any] = None,
        **kwargs
    ):
        """Initialize base agent."""
        self.name = name
        self.role = role
        self.args = args
        self.max_retries = args.max_retries
        
        # Model configuration from args
        self.model_path = self._get_model_path()
        self.temperature = self._get_temperature()
        self.max_tokens = args.max_output_tokens
        
        # Use provided engine or create new one
        if engine is not None:
            self.engine = engine
            self._owns_engine = False  # Flag to track if we created the engine
            logger.info(f"[{self.name}] Using shared engine: {self.engine}")
        else:
            # Initialize engine
            engine_kwargs = engine_kwargs or {}
            self.engine = EngineHub.create_engine(
                model_path=self.model_path,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **engine_kwargs
            )
            self._owns_engine = True
            logger.info(f"[{self.name}] Initialized with new engine: {self.engine}")
        # Conversation history
        self.conversation_history = []

    def _get_model_path(self) -> str:
        """Get model path based on agent type."""
        # This will be overridden by subclasses
        return "openai/gpt-4o-mini"

    def _get_temperature(self) -> float:
        """Get temperature based on agent type and args."""
        # This will be overridden by subclasses
        return self.args.temperature
        
    @abstractmethod
    def run(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Main execution logic for the agent."""
        pass
    
    def _call_llm(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        repetition_penalty: Optional[float] = None
    ) -> str:
        """Call the LLM engine."""
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        
        response = self.engine.generate(
            messages=messages,
            temperature=temp,
            max_tokens=max_tok,
            repetition_penalty=repetition_penalty
        )
        
        return response
    
    def reset_conversation(self):
        """Reset conversation history."""
        self.conversation_history = []
    
    def get_output_format(self) -> Dict[str, Any]:
        """Get standard output format."""
        return {
            "agent": self.name,
            "role": self.role,
            "timestamp": datetime.now().isoformat(),
            "model": self.model_path,
        }
    
    def __str__(self) -> str:
        return f"{self.name} ({self.role})"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, model={self.model_path})"
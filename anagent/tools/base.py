"""
Base tool class.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum


class ToolStatus(Enum):
    """Status of tool execution."""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    ERROR = "error"


@dataclass
class ToolResult:
    """Standardized tool execution result."""
    status: ToolStatus
    data: Any
    formatted: Any
    metadata: Dict[str, Any] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        # Convert enum to value
        result["status"] = self.status.value
        return result
    
    def is_success(self) -> bool:
        """Check if execution was successful."""
        return self.status == ToolStatus.SUCCESS


class BaseTool(ABC):
    """Abstract base class for all tools."""
    
    def __init__(
        self,
        name: str,
        description: str,
        **kwargs
    ):
        """Initialize base tool."""
        self.name = name
        self.description = description
        self.config = kwargs
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get tool parameter schema."""
        pass
    
    def get_description(self) -> str:
        """Get tool description."""
        return self.description
    
    def get_name(self) -> str:
        """Get tool name."""
        return self.name
    
    def validate_parameters(self, params: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate input parameters against schema."""
        schema = self.get_parameters()
        required = schema.get("required", [])
        
        # Check required parameters
        for param in required:
            if param not in params:
                return False, f"Missing required parameter: {param}"
        
        # Check parameter types if specified
        properties = schema.get("properties", {})
        for param, value in params.items():
            if param in properties:
                expected_type = properties[param].get("type")
                if expected_type:
                    # Simple type checking
                    type_map = {
                        "string": str,
                        "integer": int,
                        "number": (int, float),
                        "boolean": bool,
                        "array": list,
                        "object": dict
                    }
                    if expected_type in type_map:
                        expected = type_map[expected_type]
                        if not isinstance(value, expected):
                            return False, f"Parameter '{param}' must be {expected_type}, got {type(value).__name__}"
        
        return True, None
    
    def __call__(self, **kwargs) -> ToolResult:
        """Allow tool to be called directly."""
        return self.execute(**kwargs)
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
    
    def __repr__(self) -> str:
        return self.__str__()


class ToolRegistry:
    """Central registry for managing tools."""
    
    _tools: Dict[str, BaseTool] = {}
    
    @classmethod
    def register(cls, tool: BaseTool):
        """Register a tool."""
        cls._tools[tool.get_name()] = tool
    
    @classmethod
    def get(cls, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return cls._tools.get(name)
    
    @classmethod
    def list_tools(cls) -> List[str]:
        """List all registered tool names."""
        return list(cls._tools.keys())
    
    @classmethod
    def get_all_descriptions(cls) -> Dict[str, str]:
        """Get descriptions of all tools."""
        return {
            name: tool.get_description()
            for name, tool in cls._tools.items()
        }
    
    @classmethod
    def clear(cls):
        """Clear all registered tools."""
        cls._tools.clear()
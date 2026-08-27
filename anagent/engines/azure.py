"""
Azure OpenAI engine.
"""

import os
import io
import base64
from typing import List, Dict, Any, Optional
from PIL import Image
import litellm
from litellm import completion

from utils.logger import setup_logger
logger = setup_logger(__name__)

from .engine import BaseEngine, EngineHub


class AzureEngine(BaseEngine):
    """Engine for Azure OpenAI models via LiteLLM."""
    
    def __init__(
        self,
        model_path: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        api_version: str = "2024-08-01-preview",
        timeout: int = 300,
        **kwargs
    ):
        """Initialize Azure OpenAI engine."""
        super().__init__(model_path=model_path, **kwargs)
        
        self.api_key = api_key or os.getenv("AZURE_API_KEY")
        self.base_url = base_url or os.getenv("AZURE_API_BASE")
        self.api_version = api_version or os.getenv("AZURE_API_VERSION", "2024-08-01-preview")
        self.timeout = timeout
        
        if not self.api_key or not self.base_url:
            raise ValueError(
                "Azure API key and base URL required. Set AZURE_API_KEY and "
                "AZURE_API_BASE environment variables or pass as parameters."
            )
        
        # Set up LiteLLM
        self._setup_litellm()
        
        logger.info(f"Initialized Azure engine: {self.model_path}")
    def _setup_litellm(self):
        """Configure LiteLLM for Azure."""
        os.environ["AZURE_API_KEY"] = self.api_key
        os.environ["AZURE_API_BASE"] = self.base_url
        os.environ["AZURE_API_VERSION"] = self.api_version
        litellm.set_verbose = False
    
    def load_model(self):
        """No model loading needed for API."""
        pass
    
    def _prepare_image_content(self, image: Any) -> str:
        """Convert image to base64 data URL for Azure API."""
        if isinstance(image, str):
            if image.startswith('data:image'):
                return image
            elif image.startswith('http://') or image.startswith('https://'):
                return image
            else:
                # Local file path
                with open(image, "rb") as img_file:
                    image_data = img_file.read()
                    base64_image = base64.b64encode(image_data).decode('utf-8')
                    ext = image.lower().split('.')[-1]
                    mime_type = 'image/jpeg' if ext in ['jpg', 'jpeg'] else 'image/png'
                    return f"data:{mime_type};base64,{base64_image}"
        
        elif isinstance(image, Image.Image):
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            image_data = buffer.getvalue()
            base64_image = base64.b64encode(image_data).decode('utf-8')
            return f"data:image/png;base64,{base64_image}"
        
        elif isinstance(image, bytes):
            base64_image = base64.b64encode(image).decode('utf-8')
            return f"data:image/png;base64,{base64_image}"
        
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    def format_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format messages for Azure OpenAI API."""
        formatted = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content")
            
            if isinstance(content, str):
                # Simple text message
                formatted.append({"role": role, "content": content})
            
            elif isinstance(content, list):
                # Multimodal message
                formatted_content = []
                for item in content:
                    if item.get("type") == "text":
                        formatted_content.append({
                            "type": "text",
                            "text": item.get("text", "")
                        })
                    elif item.get("type") == "image":
                        image_url = self._prepare_image_content(item.get("image"))
                        formatted_content.append({
                            "type": "image_url",
                            "image_url": {"url": image_url}
                        })
                    elif item.get("type") == "image_url":
                        # Already in correct format
                        formatted_content.append(item)
                
                formatted.append({"role": role, "content": formatted_content})
        
        return formatted
    
    def generate(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate response using Azure OpenAI API."""
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        
        # Format messages for Azure
        formatted_messages = self.format_messages(messages)
        
        try:
            response = completion(
                model=self.model_path,
                messages=formatted_messages,
                temperature=temp,
                max_tokens=max_tok,
                top_p=self.top_p,
                api_key=self.api_key,
                api_base=self.base_url,
                api_version=self.api_version,
                timeout=self.timeout,
            )
            
            
            # Extract response (Azure may have reasoning in different location)
            model_response = response.choices[0].message.content.strip()
            
            # Check for reasoning content (Azure-specific)
            try:
                reasoning = response.choices[0].message.reasoning_content
                if reasoning:
                    model_response = f"<reasoning>{reasoning.strip()}</reasoning>\n{model_response}"
            except (AttributeError, KeyError):
                pass
            
            logger.info(f"[Azure] Response: {model_response}...")
            return model_response
            
        except Exception as e:
            logger.error(f"Error in Azure API call: {e}")
            raise


# Register the engine
EngineHub.register_engine("azure", AzureEngine)
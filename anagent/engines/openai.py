"""
OpenAI engine.
"""

import os
import io
import base64
from typing import List, Dict, Any, Optional

import litellm
from litellm import completion

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from utils.logger import setup_logger
logger = setup_logger(__name__)

from .engine import BaseEngine, EngineHub


class OpenAIEngine(BaseEngine):
    """Engine for OpenAI models via LiteLLM."""
    
    def __init__(
        self,
        model_path: str,
        api_key: Optional[str] = None,
        timeout: int = 300,
        max_dimension: int = 2048,
        **kwargs
    ):
        """Initialize OpenAI engine."""
        super().__init__(model_path=model_path, **kwargs)
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.timeout = timeout
        self.max_dimension = max_dimension
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Set up LiteLLM
        self._setup_litellm()
        
        logger.info(f"Initialized OpenAI engine: {self.model_path}")
    def _setup_litellm(self):
        """Configure LiteLLM for OpenAI."""
        os.environ["OPENAI_API_KEY"] = self.api_key
        litellm.set_verbose = False
    
    def load_model(self):
        """No model loading needed for API."""
        pass
    
    def _prepare_image_content(self, image: Any) -> str:
        """Convert image to base64 data URL for OpenAI API."""
        if isinstance(image, str):
            if image.startswith('data:image'):
                return image
            elif image.startswith('http://') or image.startswith('https://'):
                return image
            else:
                # Local file path - validate it exists
                if not os.path.isfile(image):
                    raise FileNotFoundError(f"Image file not found: {image}")
                
                # Load image with PIL to handle large images and convert to base64
                try:
                    with Image.open(image) as img:
                        # Convert to RGB if necessary
                        if img.mode not in ('RGB', 'RGBA'):
                            img = img.convert('RGB')
                        
                        # Resize if too large to avoid decompression bomb
                        if max(img.size) > self.max_dimension:
                            ratio = self.max_dimension / max(img.size)
                            new_size = tuple(int(dim * ratio) for dim in img.size)
                            img = img.resize(new_size, Image.Resampling.LANCZOS)
                        
                        # Convert to base64
                        buffer = io.BytesIO()
                        img.save(buffer, format='PNG')
                        image_data = buffer.getvalue()
                        base64_image = base64.b64encode(image_data).decode('utf-8')
                        return f"data:image/png;base64,{base64_image}"
                except Exception as e:
                    raise ValueError(f"Failed to process image {image}: {e}")
        
        elif isinstance(image, Image.Image):
            # Resize if too large
            if max(image.size) > self.max_dimension:
                ratio = self.max_dimension / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            image_data = buffer.getvalue()
            base64_image = base64.b64encode(image_data).decode('utf-8')
            return f"data:image/png;base64,{base64_image}"
        
        elif isinstance(image, bytes):
            # Don't blindly label arbitrary bytes as PNG
            try:
                with Image.open(io.BytesIO(image)) as img:
                    if img.mode not in ('RGB', 'RGBA'):
                        img = img.convert('RGB')
                    if max(img.size) > self.max_dimension:
                        ratio = self.max_dimension / max(img.size)
                        img = img.resize(tuple(int(d * ratio) for d in img.size), Image.Resampling.LANCZOS)
                    buffer = io.BytesIO()
                    img.save(buffer, format='PNG')
                    base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    return f"data:image/png;base64,{base64_image}"
            except Exception as e:
                raise ValueError(f"Failed to process image bytes: {e}")
        
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    def format_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format messages for OpenAI API."""
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
                        # Usually already correct (http/https or a data: URL)
                        iu = item.get("image_url") or {}
                        url = iu.get("url") if isinstance(iu, dict) else None
                        if isinstance(url, str) and not url.startswith(("http://", "https://", "data:")):
                            try:
                                url = self._prepare_image_content(url)
                                formatted_content.append({"type": "image_url", "image_url": {"url": url}})
                            except Exception as e:
                                logger.warning(f"[image] dropping unreadable image_url {url!r}: {e}")
                        else:
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
        """Generate response using OpenAI API."""
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        
        # Format messages for OpenAI
        formatted_messages = self.format_messages(messages)
        
        try:
            # Determine max_completion_tokens based on model
            if "gpt-4o-mini" in self.model_path:
                max_completion_tokens = min(max_tok, 16384)
            elif "gpt-4" in self.model_path:
                max_completion_tokens = min(max_tok, 128000)
            else:
                max_completion_tokens = max_tok
            
            response = completion(
                model=self.model_path,
                messages=formatted_messages,
                temperature=temp,
                max_completion_tokens=max_completion_tokens,
                top_p=self.top_p,
                api_key=self.api_key,
                timeout=self.timeout,
            )
            
            
            # Extract response
            model_response = response.choices[0].message.content.strip()
            
            logger.info(f"[OpenAI] Response: {model_response}...")
            return model_response
            
        except Exception as e:
            logger.error(f"Error in OpenAI API call: {e}")
            raise


# Register the engine
EngineHub.register_engine("openai", OpenAIEngine)
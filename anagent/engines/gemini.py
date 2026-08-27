"""
Google Gemini engine.
"""

import os
import io
import base64
import time
from typing import List, Dict, Any, Optional

import litellm
from litellm import completion

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from utils.logger import setup_logger
logger = setup_logger(__name__)

from .engine import BaseEngine, EngineHub


class GeminiEngine(BaseEngine):
    """Engine for Gemini models via LiteLLM."""
    
    def __init__(
        self,
        model_path: str,
        api_key: Optional[str] = None,
        timeout: int = 600,
        max_retries: int = 5,
        max_resolution: int = 2048,
        **kwargs
    ):
        """Initialize Gemini engine."""
        super().__init__(model_path=model_path, **kwargs)
        
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_resolution = max_resolution
        
        if not self.api_key:
            raise ValueError(
                "Gemini API key required. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Set up LiteLLM
        self._setup_litellm()
        
        logger.info(f"Initialized Gemini engine: {self.model_path}")
    def _setup_litellm(self):
        """Configure LiteLLM for Gemini."""
        os.environ["GEMINI_API_KEY"] = self.api_key
    
    def load_model(self):
        """No model loading needed for API."""
        pass
    
    def _prepare_image_content(self, image: Any) -> str:
        """Convert image to base64 data URL for Gemini API."""
        if isinstance(image, str):
            if image.startswith('data:image'):
                return image
            elif image.startswith('http://') or image.startswith('https://'):
                return image
            else:
                # Local file path - validate it exists and convert to base64
                if not os.path.isfile(image):
                    raise FileNotFoundError(f"Image file not found: {image}")
                
                # Load image with PIL to handle large images and convert to base64
                try:
                    with Image.open(image) as img:
                        # Convert to RGB if necessary
                        if img.mode not in ('RGB', 'RGBA'):
                            img = img.convert('RGB')
                        
                        # Resize if too large to avoid decompression bomb
                        if max(img.size) > self.max_resolution:
                            ratio = self.max_resolution / max(img.size)
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
            if max(image.size) > self.max_resolution:
                ratio = self.max_resolution / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
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
        """Format messages for Gemini API."""
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
                        # Convert image to base64 data URL
                        image_url = self._prepare_image_content(item.get("image"))
                        formatted_content.append({
                            "type": "image_url",
                            "image_url": {"url": image_url}
                        })
                    elif item.get("type") == "image_url":
                        # If it's a local path in image_url format, convert it
                        url = item.get("image_url", {}).get("url", "")
                        if url and not url.startswith(('http://', 'https://', 'data:image')):
                            # It's a local path, process it
                            url = self._prepare_image_content(url)
                        formatted_content.append({
                            "type": "image_url",
                            "image_url": {"url": url}
                        })
                
                formatted.append({"role": role, "content": formatted_content})
        
        return formatted
    
    def generate(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate response using Gemini API with retry logic."""
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        
        # Format messages for Gemini
        formatted_messages = self.format_messages(messages)
        
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                if attempt > 0:
                    wait_time = 2 ** attempt  # Exponential backoff: 2, 4, 8 seconds
                    logger.warning(f"Retrying Gemini API call (attempt {attempt + 1}/{self.max_retries}) after {wait_time}s...")
                    time.sleep(wait_time)
                
                response = completion(
                    model=self.model_path,
                    messages=formatted_messages,
                    temperature=temp,
                    max_tokens=max_tok,
                    top_p=self.top_p,
                    api_key=self.api_key,
                    timeout=self.timeout,
                    num_retries=0,  # Handle retries ourselves
                )
                
                
                # Extract response
                model_response = response.choices[0].message.content.strip()
                
                logger.info(f"[Gemini] Response: {model_response[:100]}...")
                return model_response
                
            except litellm.exceptions.ServiceUnavailableError as e:
                last_exception = e
                logger.warning(f"Gemini API timeout/unavailable (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    logger.error(f"All {self.max_retries} retry attempts failed for Gemini API")
                    raise
                    
            except litellm.exceptions.RateLimitError as e:
                last_exception = e
                wait_time = 10 * (attempt + 1)  # Longer wait for rate limits
                logger.warning(f"Gemini API rate limit hit. Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                if attempt == self.max_retries - 1:
                    raise
                    
            except Exception as e:
                logger.error(f"Error in Gemini API call: {e}")
                raise
        
        # Should not reach here, but just in case
        if last_exception:
            raise last_exception
        raise RuntimeError("Unexpected error in Gemini API call")


# Register the engine
EngineHub.register_engine("gemini", GeminiEngine)
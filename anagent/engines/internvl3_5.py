"""
InternVL3.5 engine.
"""

import os
import torch
from typing import List, Dict, Any, Optional
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText

from utils.logger import setup_logger
logger = setup_logger(__name__)

from .engine import BaseEngine, EngineHub


class InternVL3_5Engine(BaseEngine):
    """Engine for InternVL 3.5 vision-language models."""
    
    def __init__(
        self,
        model_path: str,
        max_image_size: int = 1024,  # Max dimension for image resizing
        use_flash_attn: bool = True,  # Use flash attention for efficiency
        **kwargs
    ):
        """Initialize InternVL3.5 engine."""
        super().__init__(model_path=model_path, **kwargs)
        
        self.max_image_size = max_image_size
        self.use_flash_attn = use_flash_attn
        self.use_multi_gpu = False

        # Load model
        self.load_model()
        
        logger.info(f"Initialized InternVL3.5 engine: {self.model_path}")
        logger.info(f"Image settings - max_size: {max_image_size}, flash_attn: {use_flash_attn}")
    def load_model(self):
        """Load InternVL model, processor, and tokenizer."""
        logger.info(f"Loading InternVL3.5 model from {self.model_path}...")
        # Check for multi-GPU setup
        use_multi_gpu = "," in os.environ.get("CUDA_VISIBLE_DEVICES", "")
        
        if use_multi_gpu:
            # Multi-GPU: use device_map="auto"
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                use_flash_attn=self.use_flash_attn,
                trust_remote_code=True,
                device_map="auto",
                cache_dir=self.cache_dir
            )
            self.use_multi_gpu = True
        else:
            # Single GPU
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
                use_flash_attn=self.use_flash_attn if self.device == "cuda" else False,
                trust_remote_code=True,
                device_map=None,
                cache_dir=self.cache_dir
            )
            self.model = self.model.to(self.device)
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            cache_dir=self.cache_dir
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            cache_dir=self.cache_dir
        )
        
        self.model.eval()
        
        logger.info("InternVL3.5 model loaded successfully!")
    def _resize_image_if_needed(self, image: Image.Image) -> Image.Image:
        """Resize extremely large images to prevent immediate OOM errors."""
        if not isinstance(image, Image.Image):
            return image
        
        width, height = image.size
        max_dim = max(width, height)
        
        # Only resize extremely large images initially
        if max_dim > self.max_image_size:
            # Calculate new dimensions maintaining aspect ratio
            scale = self.max_image_size / max_dim
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            logger.info(f"Initial resize: {width}x{height} to {new_width}x{new_height}")
            return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return image

    def _adaptive_resize_images(self, formatted_messages: List[Dict[str, Any]], max_context_length: int = 200000) -> List[Dict[str, Any]]:
        """Adaptively resize images if token length exceeds limit."""
        # Try with current image sizes first
        attempt = 0
        max_attempts = 10
        resize_factor = 1.0
        
        while attempt < max_attempts:
            # Apply current resize factor to images
            current_messages = []
            for msg in formatted_messages:
                role = msg.get("role")
                content = msg.get("content", [])
                
                resized_content = []
                for item in content:
                    if item.get("type") == "image":
                        # Handle both 'url' and 'image' keys
                        image = item.get("image") or item.get("url")
                        if isinstance(image, Image.Image) and resize_factor < 1.0:
                            width, height = image.size
                            # Ensure minimum size of 64 pixels on smallest dimension
                            new_width = max(64, int(width * resize_factor))
                            new_height = max(64, int(height * resize_factor))
                            resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                            resized_content.append({"type": "image", "url": resized_image})
                            if attempt > 0:
                                logger.info(f"Attempt {attempt+1}: Resizing image from {width}x{height} to {new_width}x{new_height}")
                        else:
                            # Keep original format
                            if "url" in item:
                                resized_content.append({"type": "image", "url": image})
                            else:
                                resized_content.append({"type": "image", "url": image})
                    else:
                        resized_content.append(item)
                
                current_messages.append({"role": role, "content": resized_content})
            
            # Test token length with current resize factor
            try:
                inputs = self.processor.apply_chat_template(
                    current_messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt"
                )
                
                input_length = inputs.input_ids.shape[1]
                
                logger.info(f"Attempt {attempt+1}: Input token length: {input_length}")
                if input_length <= max_context_length:
                    # Success! Return the current messages and inputs
                    return current_messages, inputs
                
                # Too long, try smaller resize factor
                attempt += 1
                resize_factor *= 0.5
                
                logger.warning(f"Token length {input_length} exceeds limit {max_context_length}. Reducing image size (factor: {resize_factor:.2f})")
            except Exception as e:
                logger.error(f"Error during adaptive resize attempt {attempt+1}: {e}")
                attempt += 1
                resize_factor *= 0.5
        
        # Count total images for better error message
        total_images = sum(
            len([item for item in msg.get("content", []) if item.get("type") == "image"])
            for msg in formatted_messages
        )
        
        # If all attempts failed, raise error
        raise ValueError(
            f"Unable to reduce sequence length below {max_context_length} tokens after {max_attempts} attempts. "
            f"Found {total_images} image(s) in the input. Last attempt had token count exceeding limit. "
            f"Consider: (1) reducing the number of images, (2) using lower resolution images initially, "
            f"or (3) splitting the task into multiple requests."
        )
    
    def format_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format messages for InternVL."""
        formatted = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content")
            
            if isinstance(content, str):
                # Simple text message
                formatted.append({
                    "role": role,
                    "content": [{"type": "text", "text": content}]
                })
            
            elif isinstance(content, list):
                # Multimodal message
                formatted_content = []
                for item in content:
                    if item.get("type") == "text":
                        formatted_content.append(item)
                    elif item.get("type") == "image":
                        image = item.get("image") or item.get("url")
                        # Ensure image is PIL Image
                        if isinstance(image, str):
                            image = Image.open(image).convert('RGB')
                        elif not isinstance(image, Image.Image):
                            raise ValueError(f"Unsupported image type: {type(image)}")
                        
                        # Resize if needed to prevent OOM
                        image = self._resize_image_if_needed(image)
                        # InternVL uses 'url' key for images
                        formatted_content.append({"type": "image", "url": image})
                    elif item.get("type") == "image_url":
                        # Convert image_url format to InternVL format
                        url = item.get("image_url", {}).get("url", "")
                        if url:
                            image = Image.open(url).convert('RGB')
                            image = self._resize_image_if_needed(image)
                            formatted_content.append({"type": "image", "url": image})
                
                formatted.append({"role": role, "content": formatted_content})
        
        return formatted
    
    def generate(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        max_context_length: int = 200000,
        **kwargs
    ) -> str:
        """Generate response using InternVL model."""
        temp = temperature if temperature is not None else self.temperature
        rep_pen = repetition_penalty if repetition_penalty is not None else self.repetition_penalty
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        
        # Format messages for InternVL
        formatted_messages = self.format_messages(messages)
        
        # Adaptively resize images if token length exceeds limit
        formatted_messages, inputs = self._adaptive_resize_images(formatted_messages, max_context_length=max_context_length)
        
        # Move to device
        if self.use_multi_gpu:
            # For multi-GPU with device_map="auto", the model handles device placement
            model_device = next(self.model.parameters()).device
            inputs = inputs.to(model_device, dtype=torch.bfloat16)
        else:
            # For single GPU, move to specified device
            if self.device == "cuda":
                inputs = inputs.to(self.device, dtype=torch.bfloat16)
            else:
                inputs = inputs.to(self.device)
        
        # Clear cache before generation to free up memory
        if self.device == "cuda" or self.use_multi_gpu:
            torch.cuda.empty_cache()
        
        # Generate with better memory management
        try:
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tok,
                    temperature=temp,
                    top_p=self.top_p,
                    repetition_penalty=rep_pen,
                    do_sample=True if temp > 0 else False,
                    use_cache=True,  # Enable KV cache for efficiency
                )
        finally:
            # Clear cache after generation
            if self.device == "cuda" or self.use_multi_gpu:
                torch.cuda.empty_cache()
        
        # Decode output (only the new tokens)
        output_text = self.processor.decode(
            generated_ids[0, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        logger.info(f"\n[InternVL3.5] Response: {output_text}...")
        return output_text


# Register the engine
EngineHub.register_engine("internvl3_5", InternVL3_5Engine)
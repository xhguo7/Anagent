"""
Qwen3-VL engine.
"""

import os
import torch
from typing import List, Dict, Any, Optional
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

from utils.logger import setup_logger
logger = setup_logger(__name__)

from .engine import BaseEngine, EngineHub


class Qwen3VLEngine(BaseEngine):
    """Engine for Qwen vision-language models."""
    
    def __init__(
        self,
        model_path: str,
        min_pixels: int = 128*28*28,  # Minimum pixels for small images
        max_pixels: int = 1280*28*28,  # Maximum pixels for large images
        max_image_size: int = 1024,  # Max dimension for image resizing
        **kwargs
    ):
        """Initialize Qwen engine."""
        super().__init__(model_path=model_path, **kwargs)
        
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.max_image_size = max_image_size
        self.use_multi_gpu = False

        # Load model
        self.load_model()
        
        logger.info(f"Initialized Qwen engine: {self.model_path}")
        logger.info(f"Image settings - min_pixels: {min_pixels}, max_pixels: {max_pixels}, max_size: {max_image_size}")
    def load_model(self):
        """Load Qwen model, processor, and tokenizer."""
        logger.info(f"Loading Qwen model from {self.model_path}...")
        from transformers import Qwen3VLForConditionalGeneration
        
        # Check for multi-GPU setup
        use_multi_gpu = "," in os.environ.get("CUDA_VISIBLE_DEVICES", "")
        
        if use_multi_gpu:
            # Multi-GPU: use device_map="auto"
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                cache_dir=self.cache_dir
            )
            self.use_multi_gpu = True
        else:
            # Single GPU
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map=None,
                cache_dir=self.cache_dir
            )
            self.model = self.model.to(self.device)
        
        # Load processor and tokenizer
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            cache_dir=self.cache_dir
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            cache_dir=self.cache_dir
        )
        
        self.model.eval()
        
        logger.info("Qwen model loaded successfully!")
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
                        image = item.get("image")
                        if isinstance(image, Image.Image) and resize_factor < 1.0:
                            width, height = image.size
                            # Ensure minimum size of 64 pixels on smallest dimension
                            new_width = max(64, int(width * resize_factor))
                            new_height = max(64, int(height * resize_factor))
                            resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                            resized_content.append({"type": "image", "image": resized_image})
                            if attempt > 0:
                                logger.info(f"Attempt {attempt+1}: Resizing image from {width}x{height} to {new_width}x{new_height}")
                        else:
                            resized_content.append(item)
                    else:
                        resized_content.append(item)
                
                current_messages.append({"role": role, "content": resized_content})
            
            # Test token length with current resize factor
            try:
                text = self.processor.apply_chat_template(
                    current_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                image_inputs, video_inputs = process_vision_info(current_messages)
                
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
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
        
        # If all attempts failed, raise error
        total_images = sum(
            len([item for item in msg.get("content", []) if item.get("type") == "image"])
            for msg in formatted_messages
        )
        
        # If all attempts failed, raise error
        raise ValueError(
            f"Unable to reduce sequence length below {max_context_length} tokens after {max_attempts} attempts. "
            f"Found {total_images} image(s) in the input. Last attempt had {input_length} tokens. "
            f"Consider: (1) reducing the number of images, (2) using lower resolution images initially, "
            f"or (3) splitting the task into multiple requests."
        )
    
    def format_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format messages for Qwen."""
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
                        image = item.get("image")
                        # Ensure image is PIL Image
                        if isinstance(image, str):
                            image = Image.open(image).convert('RGB')
                        elif not isinstance(image, Image.Image):
                            raise ValueError(f"Unsupported image type: {type(image)}")
                        
                        # Resize if needed to prevent OOM
                        image = self._resize_image_if_needed(image)
                        formatted_content.append({"type": "image", "image": image})
                    elif item.get("type") == "image_url":
                        # Convert image_url format to Qwen format
                        url = item.get("image_url", {}).get("url", "")
                        if url:
                            image = Image.open(url).convert('RGB')
                            image = self._resize_image_if_needed(image)
                            formatted_content.append({"type": "image", "image": image})
                
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
        """Generate response using Qwen model."""
        temp = temperature if temperature is not None else self.temperature
        rep_pen = repetition_penalty if repetition_penalty is not None else self.repetition_penalty
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        
        # Format messages for Qwen
        formatted_messages = self.format_messages(messages)
        
        # Adaptively resize images if token length exceeds limit
        formatted_messages, inputs = self._adaptive_resize_images(formatted_messages, max_context_length=max_context_length)
        
        # Move to device
        if self.use_multi_gpu:
            # For multi-GPU with device_map="auto", move to the first device
            first_device = next(self.model.parameters()).device
            inputs = inputs.to(first_device)
        else:
            # For single GPU, move to specified device
            inputs = inputs.to(self.device)
        
        # Clear cache before generation to free up memory
        if self.device == "cuda":
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
            if self.device == "cuda":
                torch.cuda.empty_cache()
        
        # Decode output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0].strip()
        
        logger.info(f"\n[Qwen3-VL] Response: {output_text}...")
        return output_text


# Register the engine
EngineHub.register_engine("qwen3vl", Qwen3VLEngine)
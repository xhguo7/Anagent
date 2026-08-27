"""
Qwen2.5-VL engine.
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


class Qwen2_5VLEngine(BaseEngine):
    """Engine for Qwen vision-language models."""
    
    def __init__(
        self,
        model_path: str,
        min_pixels: int = 256*28*28,
        max_pixels: int = 512*28*28,  # Reduced from 1280 to 512 to save memory
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
        from transformers import Qwen2_5_VLForConditionalGeneration
        
        # Check for multi-GPU setup
        use_multi_gpu = "," in os.environ.get("CUDA_VISIBLE_DEVICES", "")
        
        if use_multi_gpu:
            # Multi-GPU: use device_map="auto"
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                cache_dir=self.cache_dir
            )
            self.use_multi_gpu = True
        else:
            # Single GPU
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
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
    @property
    def model_device(self):
        """Get the device where the model's first parameter is located."""
        if self.use_multi_gpu:
            # For multi-GPU, get device of first parameter
            return next(self.model.parameters()).device
        return self.device
    
    def _resize_image_if_needed(self, image: Image.Image) -> Image.Image:
        """Resize image if it exceeds max_image_size to prevent OOM errors."""
        if not isinstance(image, Image.Image):
            return image
        
        width, height = image.size
        max_dim = max(width, height)
        
        if max_dim > self.max_image_size:
            # Calculate new dimensions maintaining aspect ratio
            scale = self.max_image_size / max_dim
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            logger.info(f"Resizing image from {width}x{height} to {new_width}x{new_height}")
            return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return image
    
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
        **kwargs
    ) -> str:
        """Generate response using Qwen model."""
        temp = temperature if temperature is not None else self.temperature
        rep_pen = repetition_penalty if repetition_penalty is not None else self.repetition_penalty
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        
        # Format messages for Qwen
        formatted_messages = self.format_messages(messages)
        
        # Apply chat template
        text = self.processor.apply_chat_template(
            formatted_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Process vision info
        image_inputs, video_inputs = process_vision_info(formatted_messages)
        
        # Prepare inputs
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # Move to device - works for both single and multi-GPU
        inputs = inputs.to(self.model_device)
        
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
        
        logger.info(f"\n[Qwen2.5-VL] Response: {output_text}...")
        return output_text


# Register the engine
EngineHub.register_engine("qwen2_5vl", Qwen2_5VLEngine)
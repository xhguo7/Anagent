"""
InternVL engine.
"""

import os
import torch
from typing import List, Dict, Any, Optional
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

from utils.logger import setup_logger
logger = setup_logger(__name__)

from .engine import BaseEngine, EngineHub


class InternVLEngine(BaseEngine):
    """Engine for InternVL vision-language models."""
    
    def __init__(self, model_path: str, **kwargs):
        """Initialize InternVL engine."""
        super().__init__(model_path=model_path, **kwargs)
        
        # Load model
        self.load_model()
        
        logger.info(f"Initialized InternVL engine: {self.model_path}")
    def load_model(self):
        """Load InternVL model and processor."""
        logger.info(f"Loading InternVL model from {self.model_path}...")
        # Check for multi-GPU setup
        use_multi_gpu = "," in os.environ.get("CUDA_VISIBLE_DEVICES", "")
        
        if use_multi_gpu:
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                cache_dir=self.cache_dir
            )
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map=None,
                cache_dir=self.cache_dir
            )
            self.model = self.model.to(self.device)
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            cache_dir=self.cache_dir
        )
        
        self.model.eval()
        
        logger.info("InternVL model loaded successfully!")
    def format_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format messages for InternVL."""
        formatted = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content")
            
            if isinstance(content, str):
                # Convert string to proper format
                formatted.append({
                    "role": role,
                    "content": [{"type": "text", "text": content}]
                })
            elif isinstance(content, list):
                # Ensure images are proper format
                formatted_content = []
                for item in content:
                    if item.get("type") == "text":
                        formatted_content.append(item)
                    elif item.get("type") == "image":
                        image = item.get("image")
                        # Ensure PIL Image
                        if isinstance(image, str):
                            image = Image.open(image).convert('RGB')
                        formatted_content.append({"type": "image", "image": image})
                    elif item.get("type") == "image_url":
                        url = item.get("image_url", {}).get("url", "")
                        if url:
                            image = Image.open(url).convert('RGB')
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
        """Generate response using InternVL model."""
        temp = temperature if temperature is not None else self.temperature
        rep_pen = repetition_penalty if repetition_penalty is not None else self.repetition_penalty
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        
        # Format messages
        formatted_messages = self.format_messages(messages)
        
        # Apply chat template
        inputs = self.processor.apply_chat_template(
            formatted_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)
        
        # Move to device
        inputs = inputs.to(self.device)
        
        # Generate
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tok,
                temperature=temp,
                top_p=self.top_p,
                repetition_penalty=rep_pen,
                do_sample=True if temp > 0 else False,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )
        
        # Decode output
        decoded_output = self.processor.decode(
            generated_ids[0, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        logger.info(f"\n[InternVL] Response: {decoded_output}...")
        return decoded_output


# Register the engine
EngineHub.register_engine("internvl", InternVLEngine)
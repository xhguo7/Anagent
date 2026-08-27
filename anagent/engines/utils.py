"""
Engine utilities.
"""

import json
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from typing import Any, Dict, List

from utils.logger import setup_logger
logger = setup_logger(__name__)


def numpy_json_encoder(obj):
    """Custom JSON encoder for numpy types, PIL Images, bytes, and complex numbers."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray) and obj.ndim == 0:
        return obj.item()
    elif isinstance(obj, (complex, np.complexfloating)):
        return {"real": obj.real, "imag": obj.imag}
    elif isinstance(obj, (bytes, bytearray)):
        try:
            return obj.decode('utf-8')
        except UnicodeDecodeError:
            return base64.b64encode(obj).decode('utf-8')
    elif hasattr(obj, 'mode') and hasattr(obj, 'size'):  # PIL Image detection
        return f"<PIL_Image_{obj.mode}_{obj.size[0]}x{obj.size[1]}>"
    elif hasattr(obj, 'items'):
        return {k: numpy_json_encoder(v) for k, v in obj.items()}
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        return [numpy_json_encoder(i) for i in obj]
    else:
        raise TypeError(f'Object of type {type(obj).__name__} is not JSON serializable')


def save_json(data: Any, filepath: str, indent: int = 4):
    """Save data to JSON file with custom encoder."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, default=numpy_json_encoder, ensure_ascii=False)


def load_json(filepath: str) -> Any:
    """Load data from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def prepare_multimodal_message(
    text: str,
    images: List[Any] = None,
    role: str = "user"
) -> Dict[str, Any]:
    """Create a multimodal message with text and optional images."""
    if not images:
        return {"role": role, "content": text}
    
    content = []
    
    # Add images first
    for img in images:
        if isinstance(img, (str, Image.Image)):
            content.append({"type": "image", "image": img})
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")
    
    # Add text
    content.append({"type": "text", "text": text})
    
    return {"role": role, "content": content}


def create_text_message(text: str, role: str = "user") -> Dict[str, Any]:
    """Create a simple text message."""
    return {"role": role, "content": text}


def format_conversation(
    messages: List[str],
    roles: List[str] = None,
    images: List[List[Any]] = None
) -> List[Dict[str, Any]]:
    """Format a conversation with alternating messages."""
    if roles is None:
        roles = ["user" if i % 2 == 0 else "assistant" for i in range(len(messages))]
    
    if images is None:
        images = [None] * len(messages)
    
    if len(messages) != len(roles):
        raise ValueError("Number of messages must match number of roles")
    
    conversation = []
    for msg, role, imgs in zip(messages, roles, images):
        conversation.append(prepare_multimodal_message(msg, imgs, role))
    
    return conversation
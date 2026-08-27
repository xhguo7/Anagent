"""
OCR tool.
"""

import io
import numpy as np
from PIL import Image
from datetime import datetime
from typing import Dict, Any, List, Optional, Union


from utils.logger import setup_logger
logger = setup_logger(__name__)

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("Warning: pytesseract not installed. Install with: pip install pytesseract")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logger.warning("Warning: easyocr not installed. Install with: pip install easyocr")

from anagent.tools.base import BaseTool, ToolResult, ToolStatus


class OCRTool(BaseTool):
    """OCR tool for extracting text from images."""
    
    def __init__(
        self,
        backend: str = "tesseract",
        languages: List[str] = ["en"],
        **kwargs
    ):
        """Initialize OCR tool."""
        super().__init__(
            name="ocr_tool",
            description="Extract text from images using OCR",
            **kwargs
        )
        
        self.backend = backend
        self.languages = languages
        
        # Check backend availability
        if backend == "tesseract" and not TESSERACT_AVAILABLE:
            raise ImportError("Tesseract not available. Install with: pip install pytesseract")
        elif backend == "easyocr" and not EASYOCR_AVAILABLE:
            raise ImportError("EasyOCR not available. Install with: pip install easyocr")
        
        # Initialize EasyOCR reader if needed
        self.reader = None
        if backend == "easyocr":
            logger.info(f"[OCR] Initializing EasyOCR for languages: {languages}")
            self.reader = easyocr.Reader(languages, gpu=True)
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get parameter schema."""
        return {
            "type": "object",
            "properties": {
                "image": {
                    "type": "string",
                    "description": "Image path, URL, or PIL Image object"
                },
                "languages": {
                    "type": "array",
                    "description": "Languages to recognize (default: ['en'])"
                },
                "extract_bboxes": {
                    "type": "boolean",
                    "description": "Whether to extract bounding boxes (default: False)"
                },
                "confidence_threshold": {
                    "type": "number",
                    "description": "Minimum confidence threshold for EasyOCR (0-1)"
                }
            },
            "required": ["image"]
        }
    
    def execute(
        self,
        image: Union[str, Image.Image, bytes],
        languages: Optional[List[str]] = None,
        extract_bboxes: bool = False,
        confidence_threshold: float = 0.5,
        **kwargs
    ) -> ToolResult:
        """Execute OCR on image."""
        logger.info(f"[OCR] Processing image with {self.backend}")
        # Validate parameters
        is_valid, error_msg = self.validate_parameters({"image": str(image)})
        if not is_valid:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                formatted=f"[OCR Error] {error_msg}",
                error=error_msg
            )
        
        try:
            # Load image
            pil_image = self._load_image(image)
            
            # Run OCR
            langs = languages or self.languages
            
            if self.backend == "tesseract":
                result = self._ocr_tesseract(pil_image, langs, extract_bboxes)
            else:  # easyocr
                result = self._ocr_easyocr(pil_image, langs, extract_bboxes, confidence_threshold)
            
            text_length = len(result.get("text", ""))
            logger.info(f"[OCR] Extracted {text_length} characters")
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=result,
                formatted=self.format_results_for_agent(result),
                metadata={
                    "backend": self.backend,
                    "languages": langs,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            error_msg = f"OCR failed: {str(e)}"
            logger.error(f"[OCR] Error: {error_msg}")
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                formatted=f"[OCR Error] {error_msg}",
                error=error_msg
            )
    
    def _load_image(self, image: Union[str, Image.Image, bytes]) -> Image.Image:
        """Load image from various formats."""
        if isinstance(image, Image.Image):
            return image
        elif isinstance(image, bytes):
            return Image.open(io.BytesIO(image)).convert('RGB')
        elif isinstance(image, str):
            if image.startswith('http://') or image.startswith('https://'):
                import requests
                response = requests.get(image)
                return Image.open(io.BytesIO(response.content)).convert('RGB')
            else:
                return Image.open(image).convert('RGB')
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    def _ocr_tesseract(
        self,
        image: Image.Image,
        languages: List[str],
        extract_bboxes: bool
    ) -> Dict[str, Any]:
        """Run OCR using Tesseract."""
        # Convert language codes (e.g., ['en'] -> 'eng' for tesseract)
        lang_map = {'en': 'eng', 'eng': 'eng', 'zh': 'chi_sim', 'es': 'spa', 'fr': 'fra'}
        converted_langs = [lang_map.get(lang, lang) for lang in languages]
        lang_code = '+'.join(converted_langs)
        
        if extract_bboxes:
            # Get detailed data with bounding boxes
            data = pytesseract.image_to_data(image, lang=lang_code, output_type=pytesseract.Output.DICT)
            
            # Extract text and bboxes
            text_blocks = []
            full_text = []
            
            for i in range(len(data['text'])):
                if data['text'][i].strip():
                    text_blocks.append({
                        'text': data['text'][i],
                        'confidence': float(data['conf'][i]),
                        'bbox': [
                            data['left'][i],
                            data['top'][i],
                            data['left'][i] + data['width'][i],
                            data['top'][i] + data['height'][i]
                        ]
                    })
                    full_text.append(data['text'][i])
            
            return {
                'text': ' '.join(full_text),
                'text_blocks': text_blocks,
                'num_blocks': len(text_blocks)
            }
        else:
            # Just get text
            text = pytesseract.image_to_string(image, lang=lang_code)
            return {
                'text': text.strip(),
                'text_blocks': None,
                'num_blocks': None
            }
    
    def _ocr_easyocr(
        self,
        image: Image.Image,
        languages: List[str],
        extract_bboxes: bool,
        confidence_threshold: float
    ) -> Dict[str, Any]:
        """Run OCR using EasyOCR."""
        # Re-initialize reader if languages changed
        if set(languages) != set(self.languages):
            logger.info(f"[OCR] Reinitializing EasyOCR for languages: {languages}")
            self.reader = easyocr.Reader(languages, gpu=True)
            self.languages = languages
        
        # Convert PIL Image to numpy array for EasyOCR
        image_array = np.array(image)
        
        # Run detection and recognition
        results = self.reader.readtext(image_array)  # ← Use numpy array
        
        # Process results
        text_blocks = []
        full_text = []
        
        for bbox, text, confidence in results:
            if confidence >= confidence_threshold:
                text_blocks.append({
                    'text': text,
                    'confidence': float(confidence),
                    'bbox': [int(coord) for point in bbox for coord in point]  # Flatten bbox
                })
                full_text.append(text)
        
        return {
            'text': ' '.join(full_text),
            'text_blocks': text_blocks if extract_bboxes else None,
            'num_blocks': len(text_blocks)
        }
    
    def extract_from_pdf_page(self, pdf_page_image: Image.Image) -> str:
        """Specialized method for extracting text from PDF page images."""
        result = self.execute(
            image=pdf_page_image,
            extract_bboxes=False
        )
        
        if result.is_success():
            return result.data.get('text', '')
        else:
            return ""

    def format_results_for_agent(self, results: Dict[str, Any], block_limit: int = 20) -> str:
        """Format OCR results as a readable string for the agent."""
        if not results:
            return "No OCR results available."
        
        output = []
        
        # Header
        output.append("=== OCR Extracted Text ===")
        
        # Add extracted text
        text = results.get('text', '').strip()
        if text:
            output.append("\nExtracted Text:")
            output.append(text)
        else:
            output.append("\nNo text extracted from image.")
        
        # Add metadata about number of blocks
        num_blocks = results.get('num_blocks')
        if num_blocks is not None and num_blocks > 0:
            output.append(f"\nTotal Text Blocks: {num_blocks}")
        
        # Add individual blocks with details if available
        text_blocks = results.get('text_blocks')
        if text_blocks and len(text_blocks) > 0:
            output.append(f"\n--- Detailed Text Blocks (with positions) ---")
            
            # Show up to block_limit blocks to avoid overwhelming output
            blocks_to_show = text_blocks[:block_limit]
            for i, block in enumerate(blocks_to_show, 1):
                block_text = block.get('text', '')
                confidence = block.get('confidence', 0)
                bbox = block.get('bbox', [])
                
                output.append(f"\nBlock {i}:")
                output.append(f"  Text: {block_text}")
                output.append(f"  Confidence: {confidence/100:.2%}")
                
                if bbox and len(bbox) >= 4:
                    # Handle both flat [x1,y1,x2,y2] and nested bbox formats
                    if len(bbox) == 4:
                        output.append(f"  Position: ({bbox[0]}, {bbox[1]}) to ({bbox[2]}, {bbox[3]})")
                    else:
                        # EasyOCR format (flattened list of coordinates)
                        output.append(f"  Position: {bbox[:4]}")
            
            if len(text_blocks) > block_limit:
                output.append(f"\n... and {len(text_blocks) - block_limit} more blocks")
        
        return '\n'.join(output).strip()

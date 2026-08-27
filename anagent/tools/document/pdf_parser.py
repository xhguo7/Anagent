"""
PDF parser tool.
"""

import os
import io
import re
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import requests
from urllib.parse import urlparse
import base64
import hashlib

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False


from utils.logger import setup_logger
logger = setup_logger(__name__)

from anagent.tools.base import BaseTool, ToolResult, ToolStatus

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logger.warning("Warning: pdfplumber not installed. Install with: pip install pdfplumber")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("Warning: Pillow not installed. Install with: pip install Pillow")


class PDFParserTool(BaseTool):
    """PDF parser tool for extracting content from PDF documents."""
    
    def __init__(
        self,
        parse_format: str = "structure",  # "full" or "structure"
        extract_tables: bool = True,
        extract_images: bool = False,
        save_images: bool = True,
        image_output_dir: str = "./agent_outputs/expert/figures/",
        min_image_size: int = 50,  # Minimum width/height in pixels
        **kwargs
    ):
        """Initialize PDF parser tool."""
        super().__init__(
            name="pdf_parser",
            description="Extract text, tables, figures, and metadata from PDF documents",
            **kwargs
        )
        
        self.parse_format = parse_format
        self.extract_tables = extract_tables
        self.extract_images = extract_images
        self.save_images = save_images
        self.image_output_dir = image_output_dir
        self.min_image_size = min_image_size
        
        # Check dependencies
        if not PYPDF2_AVAILABLE:
            raise ImportError("PyPDF2 is required. Install with: pip install PyPDF2")
        
        if self.extract_tables and not PDFPLUMBER_AVAILABLE:
            raise ImportError("pdfplumber is required for table extraction. Install with: pip install pdfplumber")
        
        if self.extract_images and not PIL_AVAILABLE:
            logger.warning("Pillow is recommended for image extraction. Install with: pip install Pillow")
        
        # Create image output directory
        if self.save_images and self.extract_images:
            Path(self.image_output_dir).mkdir(parents=True, exist_ok=True)
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get parameter schema."""
        return {
            "type": "object",
            "properties": {
                "pdf_path": {
                    "type": "string",
                    "description": "Path to PDF file, bytes content, or URL to PDF document"
                },
                "pages": {
                    "type": "array",
                    "description": "Specific pages to extract (e.g., [0, 1, 2] for first 3 pages). None for all pages."
                },
                "extract_text": {
                    "type": "boolean",
                    "description": "Whether to extract text (default: True)"
                },
                "extract_tables": {
                    "type": "boolean",
                    "description": "Whether to extract tables (default: True)"
                },
                "extract_metadata": {
                    "type": "boolean",
                    "description": "Whether to extract PDF metadata (default: True)"
                },
                "extract_figures": {
                    "type": "boolean",
                    "description": "Whether to extract figures/images (default: False)"
                }
            },
            "required": ["pdf_path"]
        }
    
    def _is_url(self, path: str) -> bool:
        """Check if the given path is a URL."""
        try:
            result = urlparse(path)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def _fetch_pdf_from_url(self, url: str, save_dir: str = "./agent_outputs/expert/pdfs/") -> bytes:
        """Fetch PDF content from URL or convert webpage to PDF."""
        try:
            # Create save directory if it doesn't exist
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            
            # Generate filename from URL
            filename = self._url_to_filename(url)
            filepath = os.path.join(save_dir, filename)
            
            # Fetch content with headers to mimic browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, timeout=30, headers=headers)
            response.raise_for_status()
            
            # Check if response is already a PDF
            content_type = response.headers.get('Content-Type', '').lower()
            is_pdf = 'application/pdf' in content_type or url.lower().endswith('.pdf')
            
            if is_pdf:
                # Direct PDF download
                pdf_content = response.content
            else:
                # Convert webpage to PDF
                pdf_content = self._convert_html_to_pdf(response.text, url)
            
            # Save PDF to file
            with open(filepath, 'wb') as f:
                f.write(pdf_content)
            
            print(f"PDF saved to: {filepath}")
            return pdf_content
            
        except requests.RequestException as e:
            raise Exception(f"Failed to fetch content from URL: {str(e)}")
        except Exception as e:
            raise Exception(f"Failed to process PDF: {str(e)}")


    def _url_to_filename(self, url: str) -> str:
        """Convert URL to a safe filename."""
        # Parse URL to get the relevant parts
        parsed = urlparse(url)
        
        # Combine domain and path
        domain = parsed.netloc
        path = parsed.path.strip('/')
        
        # Combine and sanitize
        full_path = f"{domain}/{path}" if path else domain
        
        # Replace special characters
        filename = full_path.replace('.', '_dot_')
        filename = filename.replace('/', '_slash_')
        filename = re.sub(r'[^\w\-_]', '_', filename)
        
        # Ensure it ends with .pdf
        if not filename.endswith('.pdf'):
            filename += '.pdf'
        
        # Limit filename length (max 255 characters on most systems)
        if len(filename) > 250:
            filename = filename[:246] + '.pdf'
        
        return filename


    def _convert_html_to_pdf(self, html_content: str, url: str) -> bytes:
        """Convert HTML content to PDF."""
        # Method 1: Try using pdfkit (requires wkhtmltopdf installed)
        try:
            import pdfkit
            options = {
                'encoding': 'UTF-8',
                'enable-local-file-access': None,
                'page-size': 'A4',
                'margin-top': '0.75in',
                'margin-right': '0.75in',
                'margin-bottom': '0.75in',
                'margin-left': '0.75in',
            }
            return pdfkit.from_string(html_content, False, options=options)
        except (ImportError, OSError):
            pass
        
        # Method 2: Try using weasyprint
        try:
            from weasyprint import HTML
            return HTML(string=html_content, base_url=url).write_pdf()
        except ImportError:
            pass
        
        # Method 3: Try using playwright (requires installation)
        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                browser = p.chromium.launch()
                page = browser.new_page()
                page.set_content(html_content)
                pdf_bytes = page.pdf(format='A4')
                browser.close()
                return pdf_bytes
        except ImportError:
            pass
        
        raise Exception(
            "HTML to PDF conversion failed. Please install one of: "
            "pdfkit (with wkhtmltopdf), weasyprint, or playwright"
        )
    
    def execute(
        self,
        pdf_path: Union[str, bytes],
        pages: Optional[List[int]] = None,
        extract_text: bool = True,
        extract_tables: bool = True,
        extract_metadata: bool = True,
        extract_figures: bool = True,
        **kwargs
    ) -> ToolResult:
        """Execute PDF parsing."""
        logger.info(f"[PDFParser] Processing PDF in '{self.parse_format}' format")
        # Override instance setting with parameter if provided
        should_extract_figures = extract_figures or self.extract_images
        
        # Validate parameters - skip validation for bytes input since it's valid
        if isinstance(pdf_path, str):
            is_valid, error_msg = self.validate_parameters({"pdf_path": pdf_path})
            if not is_valid:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    data=None,
                    formatted=f"[PDF Parser Error] {error_msg}",
                    error=error_msg
                )
        elif not isinstance(pdf_path, bytes):
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                formatted=f"[PDF Parser Error] pdf_path must be string or bytes, got {type(pdf_path).__name__}",
                error=f"pdf_path must be string or bytes, got {type(pdf_path).__name__}"
            )
        
        try:
            # Handle file path, URL, or bytes
            if isinstance(pdf_path, str):
                # Check if it's a URL
                if self._is_url(pdf_path):
                    logger.info(f"[PDFParser] Fetching PDF from URL: {pdf_path}")
                    pdf_content = self._fetch_pdf_from_url(pdf_path)
                    pdf_file = io.BytesIO(pdf_content)
                    file_name = pdf_path.split('/')[-1] or "remote.pdf"
                # Check if it's a file path
                elif os.path.exists(pdf_path):
                    pdf_file = open(pdf_path, 'rb')
                    file_name = os.path.basename(pdf_path)
                    # Read content for figures if needed
                    if should_extract_figures:
                        pdf_content = pdf_file.read()
                        pdf_file.seek(0)
                else:
                    return ToolResult(
                        status=ToolStatus.ERROR,
                        data=None,
                        formatted=f"[PDF Parser Error] PDF file not found: {pdf_path}",
                        error=f"PDF file not found: {pdf_path}"
                    )
            else:
                pdf_content = pdf_path
                pdf_file = io.BytesIO(pdf_path)
                file_name = "uploaded.pdf"
            
            # Extract content
            result_data = {
                "file_name": file_name,
                "num_pages": 0,
                "metadata": {},
                "title": "",
                "abstract": "",
                "sections": [],
                "text": {},
                "tables": [],
                "figures": []
            }
            
            # Extract metadata and text with PyPDF2
            if extract_metadata or extract_text:
                extracted = self._extract_with_pypdf2(pdf_file, pages, extract_text, extract_metadata)
                result_data.update(extracted)
            
            # Detect sections from text if in structure mode
            if extract_text and self.parse_format == "structure":
                sections = self._detect_sections_hierarchical(result_data["text"])
                result_data["sections"] = sections
            
            # Extract tables with pdfplumber
            if extract_tables and self.extract_tables:
                # Reset file pointer or reopen
                if isinstance(pdf_path, str):
                    if not self._is_url(pdf_path) and os.path.exists(pdf_path):
                        pdf_file.close()
                        pdf_file = open(pdf_path, 'rb')
                    elif self._is_url(pdf_path):
                        pdf_file = io.BytesIO(pdf_content)  # Use stored content
                else:
                    # For bytes input, create new BytesIO from original bytes
                    pdf_file = io.BytesIO(pdf_content)
                
                tables = self._extract_tables_with_pdfplumber(pdf_file, pages)
                result_data["tables"] = tables
            
            # Extract figures/images
            if should_extract_figures:
                # Reset file pointer or reopen
                if isinstance(pdf_path, str):
                    if not self._is_url(pdf_path) and os.path.exists(pdf_path):
                        if not hasattr(pdf_file, 'closed') or pdf_file.closed:
                            pdf_file = open(pdf_path, 'rb')
                        else:
                            pdf_file.seek(0)
                    elif self._is_url(pdf_path):
                        pdf_file = io.BytesIO(pdf_content)
                else:
                    pdf_file = io.BytesIO(pdf_content)
                
                figures = self._extract_figures_with_pdfplumber(pdf_file, pages, file_name)
                result_data["figures"] = figures
            
            # Close file if opened from path
            if isinstance(pdf_path, str) and not self._is_url(pdf_path):
                if hasattr(pdf_file, 'close'):
                    pdf_file.close()
            
            # Generate formatted output for agent
            formatted_output = self.format_results_for_agent(result_data)
            result_data["formatted"] = formatted_output
            
            logger.info(f"[PDFParser] Extracted {result_data['num_pages']} pages")
            logger.info(f"[PDFParser] Found {len(result_data['sections'])} sections")
            logger.info(f"[PDFParser] Found {len(result_data['tables'])} tables")
            logger.info(f"[PDFParser] Found {len(result_data['figures'])} figures")
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=result_data,
                formatted=formatted_output,
                metadata={
                    "file_name": file_name,
                    "num_pages": result_data["num_pages"],
                    "parse_format": self.parse_format,
                    "extraction_type": "pdf"
                }
            )
            
        except Exception as e:
            error_msg = f"PDF parsing failed: {str(e)}"
            logger.error(f"[PDFParser] Error: {error_msg}")
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                formatted=f"[PDF Parser Error] {error_msg}",
                error=error_msg
            )
    
    def _extract_with_pypdf2(
        self,
        pdf_file: io.BytesIO,
        pages: Optional[List[int]],
        extract_text: bool,
        extract_metadata: bool
    ) -> Dict[str, Any]:
        """Extract text and metadata using PyPDF2."""
        reader = PyPDF2.PdfReader(pdf_file)
        
        result = {
            "num_pages": len(reader.pages),
            "metadata": {},
            "title": "",
            "abstract": "",
            "text": {}
        }
        
        # Extract metadata
        if extract_metadata and reader.metadata:
            result["metadata"] = {
                "title": reader.metadata.get("/Title", ""),
                "author": reader.metadata.get("/Author", ""),
                "subject": reader.metadata.get("/Subject", ""),
                "creator": reader.metadata.get("/Creator", ""),
                "producer": reader.metadata.get("/Producer", ""),
                "creation_date": reader.metadata.get("/CreationDate", ""),
            }
            result["title"] = result["metadata"].get("title", "")
        
        # Extract text
        if extract_text:
            page_indices = pages if pages is not None else range(len(reader.pages))
            all_text = []
            
            for page_num in page_indices:
                if page_num < len(reader.pages):
                    page = reader.pages[page_num]
                    text = page.extract_text()
                    result["text"][f"page_{page_num}"] = text
                    all_text.append(text)
            
            # Try to detect title and abstract from first page if not in metadata
            if not result["title"] and all_text:
                result["title"] = self._extract_title_from_text(all_text[0])
            
            if all_text:
                result["abstract"] = self._extract_abstract_from_text(all_text)
        
        return result
    
    def _extract_title_from_text(self, first_page_text: str) -> str:
        """Extract title from first page text."""
        lines = [line.strip() for line in first_page_text.split('\n') if line.strip()]
        if lines:
            # Title is usually the first substantial line
            for line in lines[:5]:
                if len(line) > 10 and not line.isupper():
                    return line
        return ""
    
    def _extract_abstract_from_text(self, all_text: List[str], max_len: int = 2000) -> str:
        """Extract abstract from text."""
        full_text = " ".join(all_text[:3])  # Check first 3 pages
        
        # Try to find abstract section
        abstract_patterns = [
            r"(?i)abstract[\s:]+(.+?)(?=\n\s*(?:introduction|keywords|1\.|I\.))",
            r"(?i)abstract[\s:]+(.+?)(?=\n\n)",
        ]
        
        for pattern in abstract_patterns:
            match = re.search(pattern, full_text, re.DOTALL)
            if match:
                abstract = match.group(1).strip()
                # Limit abstract length
                if len(abstract) > max_len:
                    abstract = abstract[:max_len] + "..."
                return abstract
        
        return ""
    
    def _detect_sections_hierarchical(self, text_dict: Dict[str, str]) -> List[Dict[str, Any]]:
        """Detect hierarchical sections from text."""
        # Combine all text
        full_text = "\n".join([text for _, text in sorted(text_dict.items())])
        
        sections = []
        
        # Common section patterns for academic papers
        section_patterns = [
            # Numbered sections: 1. Introduction, 2.1 Methods
            (r'^(\d+(?:\.\d+)*)\s+([A-Z][^\n]{5,100})$', 'numbered'),
            # Roman numerals: I. Introduction
            (r'^([IVX]+)\.\s+([A-Z][^\n]{5,100})$', 'roman'),
            # All caps: INTRODUCTION
            (r'^([A-Z\s]{5,50})$', 'allcaps'),
            # Bold-style markers (common in PDFs)
            (r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*):?\s*$', 'title'),
        ]
        
        lines = full_text.split('\n')
        current_section = None
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            for pattern, pattern_type in section_patterns:
                match = re.match(pattern, line)
                if match:
                    # Determine level based on numbering
                    level = 0
                    title = ""
                    
                    if pattern_type == 'numbered':
                        numbering = match.group(1)
                        title = match.group(2).strip()
                        level = numbering.count('.')
                    elif pattern_type == 'roman':
                        title = match.group(2).strip()
                        level = 0
                    elif pattern_type == 'allcaps':
                        title = match.group(1).strip().title()
                        level = 0
                    elif pattern_type == 'title':
                        title = match.group(1).strip()
                        level = 0
                    
                    # Skip if title is too short or common false positive
                    if len(title) < 3 or title.lower() in ['the', 'and', 'for', 'with']:
                        continue
                    
                    # Count paragraphs until next section
                    para_count = 0
                    for j in range(i + 1, len(lines)):
                        next_line = lines[j].strip()
                        if not next_line:
                            continue
                        # Check if it's a new section
                        is_new_section = False
                        for p, _ in section_patterns:
                            if re.match(p, next_line):
                                is_new_section = True
                                break
                        if is_new_section:
                            break
                        # Count as paragraph if substantial
                        if len(next_line) > 50:
                            para_count += 1
                    
                    section_info = {
                        "level": level,
                        "title": title,
                        "paragraph_count": para_count
                    }
                    
                    sections.append(section_info)
                    break
        
        # Build hierarchy
        hierarchical_sections = self._build_section_hierarchy(sections)
        
        return hierarchical_sections
    
    def _build_section_hierarchy(self, flat_sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build hierarchical structure from flat section list."""
        if not flat_sections:
            return []
        
        root_sections = []
        stack = []
        
        for section in flat_sections:
            section["subsections"] = []
            level = section["level"]
            
            # Pop stack until we find parent level
            while stack and stack[-1]["level"] >= level:
                stack.pop()
            
            # Add to parent or root
            if stack:
                stack[-1]["subsections"].append(section)
            else:
                root_sections.append(section)
            
            stack.append(section)
        
        return root_sections
    
    def _extract_tables_with_pdfplumber(
        self,
        pdf_file: io.BytesIO,
        pages: Optional[List[int]]
    ) -> List[Dict[str, Any]]:
        """Extract tables using pdfplumber."""
        tables = []
        
        with pdfplumber.open(pdf_file) as pdf:
            page_indices = pages if pages is not None else range(len(pdf.pages))
            
            for page_num in page_indices:
                if page_num < len(pdf.pages):
                    page = pdf.pages[page_num]
                    page_tables = page.extract_tables()
                    
                    if page_tables:
                        for table_idx, table in enumerate(page_tables):
                            table_info = {
                                "page": page_num,
                                "table_index": table_idx,
                                "label": f"Table on Page {page_num + 1}",
                                "caption": ""
                            }
                            
                            if self.parse_format == "full":
                                table_info["content"] = table
                            else:  # structure
                                rows = len(table)
                                cols = len(table[0]) if table else 0
                                table_info["dimensions"] = {"rows": rows, "columns": cols}
                            
                            tables.append(table_info)
        
        return tables
    
    def _extract_figures_with_pdfplumber(
        self,
        pdf_file: io.BytesIO,
        pages: Optional[List[int]],
        pdf_filename: str
    ) -> List[Dict[str, Any]]:
        """Extract figures/images using pdfplumber."""
        figures = []
        
        # Create a base filename from the PDF name
        base_name = Path(pdf_filename).stem
        
        try:
            with pdfplumber.open(pdf_file) as pdf:
                page_indices = pages if pages is not None else range(len(pdf.pages))
                
                for page_num in page_indices:
                    if page_num < len(pdf.pages):
                        page = pdf.pages[page_num]
                        
                        # Extract images from the page
                        page_images = page.images
                        
                        if page_images:
                            for img_idx, img_info in enumerate(page_images):
                                try:
                                    # Get image dimensions
                                    width = float(img_info.get('width', 0))
                                    height = float(img_info.get('height', 0))
                                    
                                    # Filter out small images (likely decorative elements)
                                    if width < self.min_image_size or height < self.min_image_size:
                                        continue
                                    
                                    # Generate unique identifier for the image
                                    img_hash = self._generate_image_hash(img_info, page_num, img_idx)
                                    
                                    # Create filename
                                    fig_filename = f"{base_name}_page{page_num + 1}_fig{img_idx + 1}_{img_hash[:8]}.png"
                                    
                                    figure_info = {
                                        "page": page_num,
                                        "figure_index": img_idx,
                                        "label": f"Figure on Page {page_num + 1}",
                                        "filename": fig_filename,
                                        "dimensions": {
                                            "width": width,
                                            "height": height
                                        },
                                        "bbox": {
                                            "x0": float(img_info.get('x0', 0)),
                                            "y0": float(img_info.get('y0', 0)),
                                            "x1": float(img_info.get('x1', 0)),
                                            "y1": float(img_info.get('y1', 0))
                                        }
                                    }
                                    
                                    # Try to extract and save the image
                                    if self.save_images:
                                        try:
                                            # Extract image from the page
                                            img_obj = self._extract_image_object(page, img_info)
                                            
                                            if img_obj is not None:
                                                # Save image
                                                img_path = os.path.join(self.image_output_dir, fig_filename)
                                                os.makedirs(self.image_output_dir, exist_ok=True)
                                                img_obj.save(img_path)
                                                figure_info["file_path"] = img_path
                                                
                                                # Add base64 encoding if requested (for full format)
                                                if self.parse_format == "full":
                                                    with open(img_path, 'rb') as f:
                                                        img_data = f.read()
                                                        figure_info["base64"] = base64.b64encode(img_data).decode('utf-8')
                                        
                                        except Exception as img_save_error:
                                            logger.warning(f"Failed to save image: {str(img_save_error)}")
                                            figure_info["error"] = str(img_save_error)
                                    
                                    figures.append(figure_info)
                                
                                except Exception as img_error:
                                    logger.warning(f"Error processing image {img_idx} on page {page_num}: {str(img_error)}")
                                    continue
        
        except Exception as e:
            logger.error(f"Error extracting figures: {str(e)}")
        
        return figures
    
    def _generate_image_hash(self, img_info: Dict[str, Any], page_num: int, img_idx: int) -> str:
        """Generate a unique hash for an image based on its properties."""
        # Create a string representation of the image info
        hash_string = f"{page_num}_{img_idx}_{img_info.get('width')}_{img_info.get('height')}"
        hash_string += f"_{img_info.get('x0')}_{img_info.get('y0')}"
        
        # Generate hash
        return hashlib.md5(hash_string.encode()).hexdigest()
    
    def _extract_image_object(self, page, img_info: Dict[str, Any]) -> Optional[Image.Image]:
        """Extract image object from page using coordinates."""
        try:
            # Get bounding box
            bbox = (
                img_info.get('x0', 0),
                img_info.get('top', 0),
                img_info.get('x1', 0),
                img_info.get('bottom', 0)
            )
            
            # Crop the page to the image bbox and convert to PIL Image
            cropped_page = page.crop(bbox)
            img = cropped_page.to_image(resolution=150)
            
            # Convert to PIL Image
            pil_img = img.original
            
            return pil_img
        
        except Exception as e:
            logger.warning(f"Failed to extract image object: {str(e)}")
            return None
    
    def _format_sections_hierarchical(self, sections: List[Dict[str, Any]], indent: int = 0) -> str:
        """Format hierarchical sections for agent."""
        formatted = []
        
        for section in sections:
            prefix = "  " * indent
            title = section.get("title", "Untitled")
            para_count = section.get("paragraph_count", 0)

            if title.lower().strip() == "abstract":
                self.abstract_reached = True
            
            if self.abstract_reached and para_count > 0:
                formatted.append(f"{prefix}{'#' * (indent + 1)} {title} [{para_count} paragraphs]")
            
            # Recursively format subsections
            if section.get("subsections"):
                formatted.append(self._format_sections_hierarchical(section["subsections"], indent + 1))
        
        return "\n".join(formatted).strip()
    
    def _format_tables(self, tables: List[Dict[str, Any]]) -> str:
        """Format tables for agent."""
        formatted = []
        
        for table in tables:
            label = table.get('label', 'Table')
            page = table.get('page', 0)
            
            formatted.append(f"[{label}]")
            
            if 'content' in table:
                # Full format
                content = table.get('content', [])
                if content:
                    formatted.append("Content (first 5 rows):")
                    for row in content[:5]:
                        formatted.append("  " + " | ".join(str(cell) if cell else "" for cell in row))
            elif 'dimensions' in table:
                # Structure format
                dims = table.get('dimensions', {})
                formatted.append(f"Dimensions: {dims.get('rows', 0)} rows × {dims.get('columns', 0)} columns")
            
            formatted.append("")
        
        return "\n".join(formatted).strip()
    
    def _format_figures(self, figures: List[Dict[str, Any]]) -> str:
        """Format figures for agent."""
        formatted = []
        
        for figure in figures:
            label = figure.get('label', 'Figure')
            page = figure.get('page', 0)
            filename = figure.get('filename', 'N/A')
            
            formatted.append(f"[{label}]")
            formatted.append(f"Filename: {filename}")
            
            if 'dimensions' in figure:
                dims = figure.get('dimensions', {})
                formatted.append(f"Dimensions: {dims.get('width', 0):.1f} × {dims.get('height', 0):.1f} pixels")
            
            if 'file_path' in figure:
                formatted.append(f"Saved to: {figure['file_path']}")
            
            if 'bbox' in figure:
                bbox = figure.get('bbox', {})
                formatted.append(f"Position: ({bbox.get('x0', 0):.1f}, {bbox.get('y0', 0):.1f}) to ({bbox.get('x1', 0):.1f}, {bbox.get('y1', 0):.1f})")
            
            if 'error' in figure:
                formatted.append(f"Note: {figure['error']}")
            
            formatted.append("").strip()
        
        return "\n".join(formatted)

    def format_results_for_agent(self, data: Dict[str, Any], max_len: int = 100) -> str:
        """Generate formatted output for agent combining all extracted contents."""
        formatted = []
        
        # Title
        if data.get("title"):
            formatted.append(f"TITLE: {data['title']}")
            formatted.append("")
        
        # Metadata
        if data.get("metadata"):
            metadata = data["metadata"]
            if any(metadata.values()):
                formatted.append("METADATA:")

            # Tables
            if data.get("tables"):
                formatted.append(f"Number of Tables: {len(data['tables'])}")
            
            # Figures
            if data.get("figures"):
                formatted.append(f"Number of Figures: {len(data['figures'])}")
            
            formatted.append("")
        
        # Abstract
        if data.get("abstract"):
            formatted.append(f"ABSTRACT: {data['abstract']}")
            formatted.append("")
        
        # Sections
        if data.get("sections"):
            formatted.append("SECTIONS:")
            self.abstract_reached = False
            formatted.append(self._format_sections_hierarchical(data['sections']))
            formatted.append("")
        elif self.parse_format == "full" and data.get("text"):
            # If no sections detected, show full text
            formatted.append("FULL TEXT:")
            for page_key, text in sorted(data['text'].items()):
                page_num = page_key.replace("page_", "")
                formatted.append(f"--- Page {page_num} ---")
                # Limit text length
                display_text = text[:max_len]
                if len(text) > max_len:
                    display_text += "... [truncated]"
                formatted.append(display_text)
                formatted.append("")
        
        return "\n".join(formatted).strip()

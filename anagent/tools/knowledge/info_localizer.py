"""
Information localizer tool.
"""

import os
import re
import requests
import tempfile
import tarfile
import zipfile
import shutil
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse
from typing import Dict, Any, List, Optional, Tuple

# PDF extraction
try:
    import PyPDF2
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

# HTML parsing
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

# XML parsing
try:
    import xml.etree.ElementTree as ET
    HAS_XML = True
except ImportError:
    HAS_XML = False


from utils.logger import setup_logger
logger = setup_logger(__name__)

from anagent.tools.base import BaseTool, ToolResult, ToolStatus


class InfoLocalizer(BaseTool):
    """Information localizer for academic papers and documents."""
    
    def __init__(
        self,
        **kwargs
    ):
        """Initialize InfoLocalizer."""
        super().__init__(
            name="info_localizer",
            description="Localize and extract complete content sections from academic papers",
            **kwargs
        )
        
        self.temp_dir = None
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get parameter schema."""
        return {
            "type": "object",
            "properties": {
                "search_string": {
                    "type": "string",
                    "description": "String to localize (word, sentence, caption, section title, equation, etc.)"
                },
                "document_url": {
                    "type": "string",
                    "description": "Document URL (arXiv, PubMed, HTML, PDF) or local file path"
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Number of context lines to include (default: 5)"
                }
            },
            "required": ["search_string", "document_url"]
        }
    
    def _cleanup_temp_dir(self):
        """Clean up temporary directory."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                logger.info(f"[InfoLocalizer] Cleaned up temp directory: {self.temp_dir}")
            except Exception as e:
                logger.warning(f"[InfoLocalizer] Failed to clean up temp dir: {e}")
    
    def _detect_url_type(self, url: str) -> str:
        """Detect the type of document URL."""
        url_lower = url.lower()
        
        # Check for arXiv
        if 'arxiv.org' in url_lower:
            return 'arxiv'
        
        # Check for PubMed
        if 'pubmed' in url_lower or 'ncbi.nlm.nih.gov' in url_lower:
            return 'pubmed'
        
        # Check for PDF
        if url_lower.endswith('.pdf') or '/pdf/' in url_lower:
            return 'pdf'
        
        # Check for local file
        if os.path.exists(url):
            if url_lower.endswith('.pdf'):
                return 'pdf'
            elif url_lower.endswith(('.tex', '.bib')):
                return 'latex'
            elif url_lower.endswith('.xml'):
                return 'xml'
            else:
                return 'file'
        
        # Default to HTML
        return 'html'
    
    def _download_arxiv_source(self, arxiv_url: str) -> str:
        """Download arXiv LaTeX source package."""
        # Extract arXiv ID
        arxiv_id = None
        patterns = [
            r'arxiv\.org/abs/(\d+\.\d+)',
            r'arxiv\.org/pdf/(\d+\.\d+)',
            r'(\d{4}\.\d{4,5})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, arxiv_url)
            if match:
                arxiv_id = match.group(1)
                break
        
        if not arxiv_id:
            raise ValueError(f"Could not extract arXiv ID from URL: {arxiv_url}")
        
        logger.info(f"[InfoLocalizer] Extracted arXiv ID: {arxiv_id}")
        # Create temp directory
        self.temp_dir = tempfile.mkdtemp(prefix="arxiv_")
        
        # Download source
        source_url = f"https://arxiv.org/e-print/{arxiv_id}"
        logger.info(f"[InfoLocalizer] Downloading source from: {source_url}")
        try:
            response = requests.get(source_url, timeout=60, headers={
                'User-Agent': 'Mozilla/5.0 (InfoLocalizer/1.0)'
            })
            response.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"Failed to download arXiv source: {e}")
        
        # Save and extract
        archive_path = os.path.join(self.temp_dir, f"{arxiv_id}.tar.gz")
        with open(archive_path, 'wb') as f:
            f.write(response.content)
        
        extract_dir = os.path.join(self.temp_dir, "source")
        os.makedirs(extract_dir, exist_ok=True)
        
        # Try to extract as tar.gz first
        try:
            with tarfile.open(archive_path, 'r:gz') as tar:
                tar.extractall(extract_dir)
                logger.info(f"[InfoLocalizer] Extracted tar.gz to: {extract_dir}")
        except tarfile.ReadError:
            # Try as plain .tex file
            logger.info("[InfoLocalizer] Not a tar archive, treating as single .tex file")
            with open(os.path.join(extract_dir, "main.tex"), 'wb') as f:
                f.write(response.content)
        
        return extract_dir
    
    def _find_latex_files(self, directory: str) -> List[str]:
        """Find all .tex and .bib files in directory."""
        latex_files = []
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(('.tex', '.bib')):
                    latex_files.append(os.path.join(root, file))
        
        return sorted(latex_files)
    
    def _read_file_safe(self, filepath: str) -> str:
        """Read file with multiple encoding attempts."""
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        
        # Last resort: read as binary and decode with errors='ignore'
        with open(filepath, 'rb') as f:
            return f.read().decode('utf-8', errors='ignore')
    
    def _localize_in_latex(self, search_string: str, source_dir: str) -> Dict[str, Any]:
        """Localize string in LaTeX source files and extract complete content."""
        latex_files = self._find_latex_files(source_dir)
        
        if not latex_files:
            return {
                "found": False,
                "error": "No LaTeX or BibTeX files found in source"
            }
        
        logger.info(f"[InfoLocalizer] Found {len(latex_files)} LaTeX/BibTeX files")
        results = []
        
        for filepath in latex_files:
            try:
                content = self._read_file_safe(filepath)
                
                # Search for the string
                if search_string.lower() in content.lower():
                    # Found! Now extract the complete context
                    extraction = self._extract_latex_context(
                        search_string, content, filepath
                    )
                    
                    if extraction:
                        results.append({
                            "file": os.path.basename(filepath),
                            "full_path": filepath,
                            "extraction": extraction
                        })
            
            except Exception as e:
                logger.warning(f"[InfoLocalizer] Error reading {filepath}: {e}")
                continue
        
        if not results:
            return {
                "found": False,
                "searched_files": len(latex_files),
                "message": "Search string not found in any LaTeX files"
            }
        
        return {
            "found": True,
            "num_occurrences": len(results),
            "results": results
        }
    
    def _extract_latex_context(
        self, 
        search_string: str, 
        content: str, 
        filepath: str
    ) -> Optional[Dict[str, Any]]:
        """Extract complete LaTeX context around the search string."""
        # Find the position of the search string
        search_lower = search_string.lower()
        content_lower = content.lower()
        
        pos = content_lower.find(search_lower)
        if pos == -1:
            return None
        
        # Check what type of content this is
        extraction_type, extracted_content = self._determine_and_extract_type(
            content, pos, filepath
        )
        
        return {
            "type": extraction_type,
            "content": extracted_content,
            "position": pos
        }
    
    def _determine_and_extract_type(
        self, 
        content: str, 
        pos: int, 
        filepath: str
    ) -> Tuple[str, str]:
        """Determine the type of content and extract it completely."""
        
        # Check if it's in a BibTeX file
        if filepath.endswith('.bib'):
            return self._extract_bibtex_entry(content, pos)
        
        # Check for figure environment
        figure_match = self._find_latex_environment(content, pos, 'figure')
        if figure_match:
            return ('figure', figure_match)
        
        # Check for table environment
        table_match = self._find_latex_environment(content, pos, 'table')
        if table_match:
            return ('table', table_match)
        
        # Check for equation environments
        for eq_env in ['equation', 'align', 'gather', 'multline', 'eqnarray']:
            eq_match = self._find_latex_environment(content, pos, eq_env)
            if eq_match:
                return ('equation', eq_match)
        
        # Check for algorithm environment
        algo_match = self._find_latex_environment(content, pos, 'algorithm')
        if algo_match:
            return ('algorithm', algo_match)
        
        # Check for abstract
        abstract_match = self._find_latex_environment(content, pos, 'abstract')
        if abstract_match:
            return ('abstract', abstract_match)
        
        # Check for section/subsection
        section_match = self._extract_section(content, pos)
        if section_match:
            return ('section', section_match)
        
        # Default: extract paragraph
        para_match = self._extract_paragraph(content, pos)
        return ('paragraph', para_match)
    
    def _find_latex_environment(
        self, 
        content: str, 
        pos: int, 
        env_name: str
    ) -> Optional[str]:
        """Find and extract complete LaTeX environment containing position."""
        
        # Look backwards for \begin{env_name}
        begin_pattern = f'\\\\begin{{{env_name}}}'
        end_pattern = f'\\\\end{{{env_name}}}'
        
        # Find the nearest \begin before pos
        begin_pos = content.rfind(begin_pattern, 0, pos)
        if begin_pos == -1:
            return None
        
        # Find the matching \end after begin_pos
        end_pos = content.find(end_pattern, begin_pos)
        if end_pos == -1:
            return None
        
        # Check if pos is within this environment
        end_pos += len(end_pattern)
        if begin_pos <= pos <= end_pos:
            return content[begin_pos:end_pos]
        
        return None
    
    def _extract_bibtex_entry(self, content: str, pos: int) -> Tuple[str, str]:
        """Extract complete BibTeX entry containing position."""
        
        # Find the nearest @ before pos
        at_pos = content.rfind('@', 0, pos)
        if at_pos == -1:
            # Extract surrounding context
            start = max(0, pos - 200)
            end = min(len(content), pos + 200)
            return ('reference', content[start:end])
        
        # Find the matching closing brace
        brace_count = 0
        in_entry = False
        entry_end = at_pos
        
        for i in range(at_pos, len(content)):
            char = content[i]
            if char == '{':
                brace_count += 1
                in_entry = True
            elif char == '}':
                brace_count -= 1
                if in_entry and brace_count == 0:
                    entry_end = i + 1
                    break
        
        entry = content[at_pos:entry_end]
        return ('reference', entry)
    
    def _extract_section(self, content: str, pos: int) -> Optional[str]:
        """Extract complete section containing position."""
        
        # Section patterns
        section_patterns = [
            r'\\section\*?\{[^}]+\}',
            r'\\subsection\*?\{[^}]+\}',
            r'\\subsubsection\*?\{[^}]+\}',
        ]
        
        # Find all section boundaries
        section_starts = []
        
        for pattern in section_patterns:
            for match in re.finditer(pattern, content):
                section_starts.append(match.start())
        
        section_starts = sorted(set(section_starts))
        
        if not section_starts:
            return None
        
        # Find the section containing pos
        current_section_start = None
        next_section_start = len(content)
        
        for i, start in enumerate(section_starts):
            if start <= pos:
                current_section_start = start
                if i + 1 < len(section_starts):
                    next_section_start = section_starts[i + 1]
            else:
                break
        
        if current_section_start is not None:
            section_content = content[current_section_start:next_section_start].strip()
            return section_content
        
        return None
    
    def _extract_paragraph(self, content: str, pos: int) -> str:
        """Extract paragraph containing position."""
        
        # Find paragraph boundaries (double newlines)
        para_start = content.rfind('\n\n', 0, pos)
        if para_start == -1:
            para_start = 0
        else:
            para_start += 2
        
        para_end = content.find('\n\n', pos)
        if para_end == -1:
            para_end = len(content)
        
        return content[para_start:para_end].strip()
    
    def _localize_in_pdf(self, search_string: str, pdf_path: str) -> Dict[str, Any]:
        """Localize string in PDF and extract context."""
        
        if not HAS_PDF:
            return {
                "found": False,
                "error": "PyPDF2 not installed. Install with: pip install PyPDF2"
            }
        
        try:
            with open(pdf_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                
                results = []
                
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    
                    if search_string.lower() in text.lower():
                        # Extract context around the match
                        context = self._extract_context_from_text(
                            text, search_string, lines=10
                        )
                        
                        results.append({
                            "page": page_num + 1,
                            "context": context
                        })
                
                if not results:
                    return {
                        "found": False,
                        "message": f"Search string not found in PDF ({len(pdf_reader.pages)} pages searched)"
                    }
                
                return {
                    "found": True,
                    "num_occurrences": len(results),
                    "results": results
                }
        
        except Exception as e:
            return {
                "found": False,
                "error": f"Error reading PDF: {str(e)}"
            }
    
    def _localize_in_html(self, search_string: str, html_content: str) -> Dict[str, Any]:
        """Localize string in HTML and extract context."""
        
        if not HAS_BS4:
            return {
                "found": False,
                "error": "BeautifulSoup4 not installed. Install with: pip install beautifulsoup4"
            }
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.decompose()
            
            # Search for the string in text
            text = soup.get_text()
            
            if search_string.lower() not in text.lower():
                return {
                    "found": False,
                    "message": "Search string not found in HTML content"
                }
            
            # Find all text nodes containing the search string
            results = []
            
            for element in soup.find_all(text=True):
                if search_string.lower() in element.lower():
                    parent = element.parent
                    
                    # Get parent context
                    if parent:
                        context = parent.get_text(strip=True)
                        tag_name = parent.name
                        
                        results.append({
                            "tag": tag_name,
                            "context": context[:500] if len(context) > 500 else context
                        })
            
            return {
                "found": True,
                "num_occurrences": len(results),
                "results": results
            }
        
        except Exception as e:
            return {
                "found": False,
                "error": f"Error parsing HTML: {str(e)}"
            }
    
    def _extract_context_from_text(
        self, 
        text: str, 
        search_string: str, 
        lines: int = 5
    ) -> str:
        """Extract context lines around search string in plain text."""
        
        text_lines = text.split('\n')
        search_lower = search_string.lower()
        
        for i, line in enumerate(text_lines):
            if search_lower in line.lower():
                start = max(0, i - lines)
                end = min(len(text_lines), i + lines + 1)
                
                context_lines = text_lines[start:end]
                return '\n'.join(context_lines)
        
        return ""
    
    def _download_url_content(self, url: str) -> str:
        """Download content from URL."""
        
        try:
            response = requests.get(url, timeout=60, headers={
                'User-Agent': 'Mozilla/5.0 (InfoLocalizer/1.0)'
            })
            response.raise_for_status()
            
            return response.text
        
        except Exception as e:
            raise RuntimeError(f"Failed to download URL: {str(e)}")
    
    def execute(
        self,
        search_string: str,
        document_url: str,
        context_lines: int = 100,
        **kwargs
    ) -> ToolResult:
        """Execute information localization."""
        
        try:
            # Detect document type
            doc_type = self._detect_url_type(document_url)
            
            logger.info(f"[InfoLocalizer] Document type: {doc_type}")
            logger.info(f"[InfoLocalizer] Searching for: '{search_string}'")
            # Process based on document type
            if doc_type == 'arxiv':
                # Download and extract arXiv source
                source_dir = self._download_arxiv_source(document_url)
                localization_result = self._localize_in_latex(search_string, source_dir)
            
            elif doc_type == 'pdf':
                # Handle PDF (local or URL)
                if document_url.startswith(('http://', 'https://')):
                    # Download PDF
                    response = requests.get(document_url, timeout=60)
                    pdf_path = os.path.join(tempfile.gettempdir(), 'temp.pdf')
                    with open(pdf_path, 'wb') as f:
                        f.write(response.content)
                else:
                    pdf_path = document_url
                
                localization_result = self._localize_in_pdf(search_string, pdf_path)
            
            elif doc_type == 'pubmed':
                # Download and parse PubMed as HTML
                html_content = self._download_url_content(document_url)
                localization_result = self._localize_in_html(search_string, html_content)
            
            elif doc_type == 'html':
                # Download and parse HTML
                html_content = self._download_url_content(document_url)
                localization_result = self._localize_in_html(search_string, html_content)
            
            elif doc_type == 'latex':
                # Direct LaTeX file
                content = self._read_file_safe(document_url)
                extraction = self._extract_latex_context(
                    search_string, content, document_url
                )
                
                if extraction:
                    localization_result = {
                        "found": True,
                        "num_occurrences": 1,
                        "results": [{
                            "file": os.path.basename(document_url),
                            "full_path": document_url,
                            "extraction": extraction
                        }]
                    }
                else:
                    localization_result = {
                        "found": False,
                        "message": "Search string not found"
                    }
            
            else:
                # Generic text file
                content = self._read_file_safe(document_url)
                
                if search_string.lower() in content.lower():
                    context = self._extract_context_from_text(
                        content, search_string, lines=context_lines
                    )
                    
                    localization_result = {
                        "found": True,
                        "num_occurrences": 1,
                        "results": [{
                            "context": context
                        }]
                    }
                else:
                    localization_result = {
                        "found": False,
                        "message": "Search string not found"
                    }
            
            # Prepare result data
            result_data = {
                "search_string": search_string,
                "document_url": document_url,
                "document_type": doc_type,
                "localization": localization_result
            }
            
            # Determine status
            if localization_result.get("found"):
                status = ToolStatus.SUCCESS
            else:
                status = ToolStatus.ERROR
            
            return ToolResult(
                status=status,
                data=result_data,
                formatted=self.format_results_for_agent(result_data),
                metadata={
                    "doc_type": doc_type,
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        except Exception as e:
            logger.error(f"[InfoLocalizer] Error: {str(e)}")
            
            return ToolResult(
                status=ToolStatus.ERROR,
                data={"error": str(e)},
                formatted=f"[Info Localizer Error] Error localizing information: {str(e)}",
                metadata={"timestamp": datetime.now().isoformat()}
            )
        
        finally:
            # Clean up temporary files
            self._cleanup_temp_dir()
    
    def format_results_for_agent(self, data: Dict[str, Any]) -> str:
        """Format localization results for agent consumption."""
        
        if not data or "localization" not in data:
            return "No localization results."
        
        localization = data["localization"]
        
        if not localization.get("found"):
            error_msg = localization.get("error", localization.get("message", "Not found"))
            return f"❌ Search string not found.\n{error_msg}\nPlease ensure your search string EXACTLY EXISTS in the document. If not sure, USE A KEYWORD OR TRY A DIFFERENT TOOL."
        
        # Build formatted output
        formatted = []
        
        formatted.append(f"Search String: \"{data['search_string']}\"")
        formatted.append(f"Document: {data['document_url']}")
        formatted.append(f"Document Type: {data['document_type'].upper()}")
        formatted.append(f"Occurrences Found: {localization.get('num_occurrences', 0)}")
        formatted.append("")
        formatted.append("---\n\nEXTRACTED CONTENT")
        formatted.append("")
        
        # Display each occurrence
        results = localization.get("results", [])
        
        for idx, result in enumerate(results, 1):
            if len(results) > 1:
                formatted.append(f"({idx}) Occurrence {idx}")
            
            # Display based on result structure
            if "file" in result:
                # LaTeX source result
                
                extraction = result.get("extraction", {})
                content_type = extraction.get("type", "unknown")
                content = extraction.get("content", "")
                
                formatted.append(f"Type: {content_type.upper()}")
                formatted.append("Content:")
                formatted.append(content)
            
            elif "page" in result:
                # PDF result
                formatted.append(f"Page: {result['page']}")
                formatted.append("Context:")
                formatted.append(result.get("context", ""))
            
            elif "tag" in result:
                # HTML result
                formatted.append(f"Tag: <{result['tag']}>")
                formatted.append("Context:")
                formatted.append(result.get("context", ""))
            
            else:
                # Generic result
                formatted.append("Context:")
                formatted.append(result.get("context", ""))
            
            formatted.append("")
        
        return "\n".join(formatted).strip()


# Example usage

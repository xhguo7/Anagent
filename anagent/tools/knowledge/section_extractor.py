"""
Section extractor tool.
"""

import os
import re
import requests
import tempfile
import tarfile
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


from utils.logger import setup_logger
logger = setup_logger(__name__)

from anagent.tools.base import BaseTool, ToolResult, ToolStatus
from anagent.tools.knowledge.context_finder import ContextFinder


class SectionExtractor(BaseTool):
    """Section extractor for academic papers and documents."""
    
    def __init__(
        self,
        **kwargs
    ):
        """Initialize SectionExtractor."""
        super().__init__(
            name="section_extractor",
            description="Extract complete sections/figures/tables from academic papers",
            **kwargs
        )
        
        self.temp_dir = None
        self.all_content = {}
        self.latex_files = []
        self.context_finder = ContextFinder(**kwargs)
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get parameter schema."""
        return {
            "type": "object",
            "properties": {
                "section_spec": {
                    "type": "string",
                    "description": "Section specification (e.g., 'Figure 1', 'Section 4.2', 'Conclusion', '5.2 Results')"
                },
                "document_url": {
                    "type": "string",
                    "description": "Document URL (arXiv, PubMed, HTML, PDF) or local file path"
                }
            },
            "required": ["section_spec", "document_url"]
        }
    
    def _cleanup_temp_dir(self):
        """Clean up temporary directory."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                logger.info(f"[SectionExtractor] Cleaned up temp directory")
            except Exception as e:
                logger.warning(f"[SectionExtractor] Failed to clean up: {e}")
    
    def _detect_url_type(self, url: str) -> str:
        """Detect the type of document URL."""
        url_lower = url.lower()
        
        if 'arxiv.org' in url_lower:
            return 'arxiv'
        if 'pubmed' in url_lower or 'ncbi.nlm.nih.gov' in url_lower:
            return 'pubmed'
        if url_lower.endswith('.pdf') or '/pdf/' in url_lower:
            return 'pdf'
        if os.path.exists(url):
            if url_lower.endswith('.pdf'):
                return 'pdf'
            elif url_lower.endswith(('.tex', '.bib')):
                return 'latex'
        
        return 'html'
    
    def _parse_section_spec(self, section_spec: str) -> Dict[str, Any]:
        """Parse section specification into structured format."""
        spec = section_spec.strip()

        # Pattern 0: an explicit \label{...} or \ref{...} key
        label_match = re.search(r'\\(?:label|ref|autoref|cref|Cref)\s*{([^}]+)}', spec)
        if label_match:
            return {
                'type': 'label',
                'label': label_match.group(1).strip(),
                'original': spec
            }

        # Pattern 1: Figure/Table/Algorithm X
        fig_match = re.match(r'(?:Figure|Fig\.?|Table|Algorithm|Equation|Eq\.?)\s+(\d+)', spec, re.IGNORECASE)
        if fig_match:
            return {
                'type': 'figure_table',
                'env_type': re.match(r'(\w+)', spec, re.IGNORECASE).group(1).lower(),
                'number': fig_match.group(1),
                'original': spec
            }
        
        # Pattern 2: Section X.Y.Z (with or without "Section" keyword)
        section_match = re.match(r'(?:Section\s+)?(\d+(?:\.\d+)*)', spec, re.IGNORECASE)
        if section_match:
            section_number = section_match.group(1)
            # Check if there's a title after the number
            title_match = re.search(rf'{re.escape(section_number)}\s*\.?\s+(.+)', spec)
            title = title_match.group(1) if title_match else None
            
            return {
                'type': 'numbered_section',
                'number': section_number,
                'title': title,
                'original': spec
            }
        
        # Pattern 3: Keyword sections (Abstract, Conclusion, etc.)
        keyword_patterns = [
            'abstract', 'introduction', 'related work', 'background',
            'methodology', 'method', 'approach', 'model',
            'experiments', 'results', 'evaluation', 'analysis',
            'discussion', 'conclusion', 'future work',
            'acknowledgments', 'acknowledgements', 'references',
            'appendix', 'supplementary'
        ]
        
        spec_lower = spec.lower()
        for keyword in keyword_patterns:
            if keyword in spec_lower:
                return {
                    'type': 'keyword',
                    'keyword': keyword,
                    'original': spec
                }
        
        # Pattern 4: Appendix X
        appendix_match = re.match(r'Appendix\s+([A-Z]|\d+)', spec, re.IGNORECASE)
        if appendix_match:
            return {
                'type': 'appendix',
                'identifier': appendix_match.group(1),
                'original': spec
            }
        
        # Default: treat as keyword search
        return {
            'type': 'keyword',
            'keyword': spec_lower,
            'original': spec
        }
    
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
        
        logger.info(f"[SectionExtractor] Extracted arXiv ID: {arxiv_id}")
        # Create temp directory
        self.temp_dir = tempfile.mkdtemp(prefix="section_extractor_")
        
        # Download source
        source_url = f"https://arxiv.org/e-print/{arxiv_id}"
        logger.info(f"[SectionExtractor] Downloading source from: {source_url}")
        response = requests.get(source_url, timeout=60, headers={
            'User-Agent': 'Mozilla/5.0 (SectionExtractor/1.0)'
        })
        response.raise_for_status()
        
        # Save and extract
        archive_path = os.path.join(self.temp_dir, f"{arxiv_id}.tar.gz")
        with open(archive_path, 'wb') as f:
            f.write(response.content)
        
        extract_dir = os.path.join(self.temp_dir, "source")
        os.makedirs(extract_dir, exist_ok=True)
        
        try:
            with tarfile.open(archive_path, 'r:gz') as tar:
                tar.extractall(extract_dir)
        except tarfile.ReadError:
            # Single .tex file
            with open(os.path.join(extract_dir, "main.tex"), 'wb') as f:
                f.write(response.content)
        
        return extract_dir
    
    def _load_all_latex_files(self, directory: str):
        """Load all LaTeX files into memory."""
        self.latex_files = []
        self.all_content = {}
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.tex'):
                    filepath = os.path.join(root, file)
                    self.latex_files.append(filepath)
                    content = self._read_file_safe(filepath)
                    self.all_content[filepath] = content
        
        logger.info(f"[SectionExtractor] Loaded {len(self.latex_files)} LaTeX files")
    def _read_file_safe(self, filepath: str) -> str:
        """Read file with multiple encoding attempts."""
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        
        with open(filepath, 'rb') as f:
            return f.read().decode('utf-8', errors='ignore')
    
    def _extract_by_label(self, content: str, label: str) -> Optional[str]:
        r"""Return the environment carrying \label{label}, matched exactly."""
        # Labels are whitespace-insensitive in practice; compare on a squashed form.
        wanted = re.sub(r'\s+', '', label)
        for m in re.finditer(r'\\label\s*{([^}]+)}', content):
            if re.sub(r'\s+', '', m.group(1)) != wanted:
                continue

            # Walk back to the nearest \begin{...} and forward to its \end{...}.
            begins = list(re.finditer(r'\\begin\s*{([a-zA-Z]+\*?)}', content[:m.start()]))
            if not begins:
                continue
            opener = begins[-1]
            env = opener.group(1)
            closer = re.search(r'\\end\s*{' + re.escape(env) + r'}', content[m.end():])
            if not closer:
                continue
            return content[opener.start():m.end() + closer.end()]
        return None

    def _extract_figure_table(
        self, 
        content: str, 
        env_type: str, 
        number: str
    ) -> Optional[str]:
        """Extract figure, table, algorithm, or equation by number."""
        # Normalize environment type
        env_map = {
            'fig': 'figure',
            'figure': 'figure',
            'table': 'table',
            'algorithm': 'algorithm',
            'eq': 'equation',
            'equation': 'equation'
        }
        env_name = env_map.get(env_type.lower(), env_type.lower())
        
        # Find all environments of this type
        if env_name == 'equation':
            # Handle numbered equations, starred forms included
            patterns = [
                r'\\begin{equation\*?}.*?\\end{equation\*?}',
                r'\\begin{align\*?}.*?\\end{align\*?}',
                r'\\begin{gather\*?}.*?\\end{gather\*?}',
            ]
        else:
            # figure* / table* span both columns and are numbered in the same
            patterns = [f'\\\\begin{{{env_name}\\*?}}.*?\\\\end{{{env_name}\\*?}}']
        
        environments = []
        for pattern in patterns:
            matches = re.finditer(pattern, content, re.DOTALL)
            for match in matches:
                env_content = match.group(0)
                environments.append({
                    'content': env_content,
                    'start': match.start()
                })
        
        if not environments:
            return None

        # Patterns are collected one at a time, so restore document order before
        environments.sort(key=lambda e: e['start'])
        
        # Count which one is the Nth occurrence
        target_index = int(number) - 1
        
        if target_index < len(environments):
            return environments[target_index]['content']
        
        return None
    
    def _walk_numbered_section(self, content, parts, section_commands) -> Optional[int]:
        """Locate a section like 3.2.5 by walking each level of the hierarchy."""
        span_start, span_end = 0, len(content)
        found_at = None

        for level, part in enumerate(parts, start=1):
            if not part.isdigit():
                return None
            cmd = section_commands.get(level)
            if not cmd:
                return None

            # (?!\*) keeps unnumbered \section*{} out of the count.
            pattern = cmd + r'(?!\*)\s*{'
            siblings = list(re.finditer(pattern, content[span_start:span_end]))

            index = int(part) - 1
            if index >= len(siblings):
                return None

            current = siblings[index]
            found_at = span_start + current.start()

            # The next level lives between this sibling and the following one.
            next_start = span_start + current.end()
            next_end = (span_start + siblings[index + 1].start()
                        if index + 1 < len(siblings) else span_end)
            span_start, span_end = next_start, next_end

        return found_at

    def _extract_numbered_section(
        self, 
        content: str, 
        section_number: str,
        section_title: Optional[str] = None
    ) -> Optional[str]:
        """Extract section by number (e.g., "3.2.5")."""
        # Determine section depth
        parts = section_number.split('.')
        depth = len(parts)
        
        # Map depth to LaTeX commands
        section_commands = {
            1: r'\\section',
            2: r'\\subsection',
            3: r'\\subsubsection',
            4: r'\\paragraph',
            5: r'\\subparagraph'
        }
        
        # Build pattern to find the section
        cmd = section_commands.get(depth, r'\\section')
        
        # Pattern 1: Explicit numbering in title
        if section_title:
            pattern = f'{cmd}\\*?{{\\s*{re.escape(section_number)}\\s*\\.?\\s*{re.escape(section_title)}'
        else:
            pattern = f'{cmd}\\*?{{\\s*{re.escape(section_number)}\\s*\\.?\\s*([^}}]+)}}'
        
        match = re.search(pattern, content, re.IGNORECASE)
        
        if not match:
            # Pattern 2: walk the numbered section hierarchy
            match_start = self._walk_numbered_section(content, parts, section_commands)
            if match_start is not None:
                match = type('Match', (), {
                    'start': lambda self, _s=match_start: _s,
                    'group': lambda self, n=0: '',
                })()

        if not match:
            return None
        
        # Found the section start, now find where it ends
        section_start = match.start()
        
        # Find the end: next section of same or higher level
        end_patterns = []
        for d in range(1, depth + 1):
            cmd_level = section_commands.get(d, r'\\section')
            end_patterns.append(cmd_level)
        
        # Combine patterns
        combined_pattern = '|'.join(end_patterns)
        
        # Find next section
        next_sections = re.finditer(combined_pattern, content[section_start + 1:])
        
        try:
            next_section = next(next_sections)
            section_end = section_start + 1 + next_section.start()
        except StopIteration:
            # No next section, go to end of document
            section_end = len(content)
        
        return content[section_start:section_end].strip()
    
    def _extract_keyword_section(
        self, 
        content: str, 
        keyword: str
    ) -> Optional[str]:
        """Extract section by keyword (e.g., "Conclusion", "Methodology")."""
        # Try different section levels
        section_commands = [
            r'\\section',
            r'\\subsection',
            r'\\subsubsection',
            r'\\chapter'
        ]
        
        for cmd in section_commands:
            # Pattern: \section{...keyword...}
            pattern = cmd + r'\*?\{([^\}]*' + re.escape(keyword) + r'[^\}]*)\}'
            
            match = re.search(pattern, content, re.IGNORECASE)
            
            if match:
                section_start = match.start()
                
                # Find next section of same or higher level
                next_match = re.search(f'{cmd}\\*?{{', content[section_start + 1:])
                
                if next_match:
                    section_end = section_start + 1 + next_match.start()
                else:
                    section_end = len(content)
                
                return content[section_start:section_end].strip()
        
        # Special handling for Abstract
        if 'abstract' in keyword.lower():
            abstract_match = re.search(
                r'\\begin{abstract}(.*?)\\end{abstract}',
                content,
                re.DOTALL | re.IGNORECASE
            )
            if abstract_match:
                return abstract_match.group(0)
        
        # Special handling for References
        if 'reference' in keyword.lower():
            # Try to find bibliography
            bib_match = re.search(
                r'\\begin{thebibliography}(.*?)\\end{thebibliography}',
                content,
                re.DOTALL
            )
            if bib_match:
                return bib_match.group(0)
            
            # Or bibliography command
            bib_cmd = re.search(r'\\bibliography{[^}]+}', content)
            if bib_cmd:
                return bib_cmd.group(0)
        
        return None
    
    def _extract_appendix(
        self, 
        content: str, 
        identifier: str
    ) -> Optional[str]:
        """Extract appendix by identifier (e.g., "A", "B", "1")."""
        # Pattern: \appendix or \section{Appendix A}
        patterns = [
            f'\\\\appendix\\s*\\\\section\\*?{{.*?{re.escape(identifier)}',
            f'\\\\section\\*?{{\\s*Appendix\\s+{re.escape(identifier)}',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            
            if match:
                section_start = match.start()
                
                # Find next appendix or end of document
                next_appendix = re.search(
                    r'\\section\*?{.*?Appendix',
                    content[section_start + 1:],
                    re.IGNORECASE
                )
                
                if next_appendix:
                    section_end = section_start + 1 + next_appendix.start()
                else:
                    section_end = len(content)
                
                return content[section_start:section_end].strip()
        
        return None
    
    def _search_all_files(
        self, 
        spec_info: Dict[str, Any]
    ) -> Optional[Tuple[str, str]]:
        """Search through all LaTeX files for the specified section."""
        for filepath, content in self.all_content.items():
            result = None
            
            if spec_info['type'] == 'label':
                result = self._extract_by_label(
                    content,
                    spec_info['label']
                )

            elif spec_info['type'] == 'figure_table':
                result = self._extract_figure_table(
                    content,
                    spec_info['env_type'],
                    spec_info['number']
                )
            
            elif spec_info['type'] == 'numbered_section':
                result = self._extract_numbered_section(
                    content,
                    spec_info['number'],
                    spec_info.get('title')
                )
            
            elif spec_info['type'] == 'keyword':
                result = self._extract_keyword_section(
                    content,
                    spec_info['keyword']
                )
            
            elif spec_info['type'] == 'appendix':
                result = self._extract_appendix(
                    content,
                    spec_info['identifier']
                )
            
            if result:
                return (result, os.path.basename(filepath))
        
        return None

    def _run_context_finder_fallback(
        self,
        section_spec: str,
        document_url: str,
        max_depth: int = 1,
        context_limit: int = 200,
    ) -> Optional[ToolResult]:
        """Try using ContextFinder as fallback when section extraction fails."""
        if not self.context_finder:
            logger.info("[SectionExtractor] ContextFinder not available for fallback")
            return None
        
        logger.info("[SectionExtractor] Attempting ContextFinder fallback")
        logger.info(f"[SectionExtractor] Using section_spec as search_string: '{section_spec[:100]}...'")
        try:
            # Extract a reasonable search string from section_spec
            search_string = section_spec
            
            # If it starts with \section{, extract the title
            section_match = re.match(r'\\(?:sub)*section\*?\{([^}]+)\}', section_spec)
            if section_match:
                search_string = section_match.group(1)
                logger.info(f"[SectionExtractor] Extracted section title: '{search_string}'")
            # If it's very long, take first context_limit chars or first sentence
            elif len(search_string) > context_limit:
                # Try to find first sentence
                first_sentence = re.match(r'^[^.!?]+[.!?]', search_string)
                if first_sentence:
                    search_string = first_sentence.group(0)
                else:
                    search_string = search_string[:context_limit]
                logger.info(f"[SectionExtractor] Truncated long spec to: '{search_string}'")
            # Call ContextFinder
            context_result = self.context_finder.execute(
                search_string=search_string,
                document_url=document_url,
                max_depth=max_depth,
                include_citations=True
            )
            
            if context_result.is_success():
                logger.info("[SectionExtractor] ContextFinder fallback succeeded")
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    data=context_result.data.copy(),
                    formatted=context_result.formatted,
                    metadata={
                        **context_result.metadata,
                        'fallback_used': True,
                        'original_tool': 'section_extractor',
                        'fallback_tool': 'context_finder'
                    }
                )
            else:
                logger.warning("[SectionExtractor] ContextFinder fallback also failed")
                return None
                
        except Exception as e:
            logger.error(f"[SectionExtractor] ContextFinder fallback error: {e}")
            return None
    
    def execute(
        self,
        section_spec: str,
        document_url: str,
        **kwargs
    ) -> ToolResult:
        """Execute section extraction."""
        
        try:
            logger.info(f"[SectionExtractor] Extracting: '{section_spec}'")
            # Parse section specification
            spec_info = self._parse_section_spec(section_spec)
            
            logger.info(f"[SectionExtractor] Parsed as: {spec_info['type']}")
            # Detect document type
            doc_type = self._detect_url_type(document_url)
            
            logger.info(f"[SectionExtractor] Document type: {doc_type}")
            # Handle different document types
            if doc_type == 'arxiv':
                # Download and extract arXiv source
                source_dir = self._download_arxiv_source(document_url)
                self._load_all_latex_files(source_dir)
            
            elif doc_type == 'latex':
                # Load local LaTeX files
                if os.path.isdir(document_url):
                    self._load_all_latex_files(document_url)
                else:
                    # Single file
                    self.latex_files = [document_url]
                    content = self._read_file_safe(document_url)
                    self.all_content[document_url] = content
            
            else:
                # For non-LaTeX documents, limited support
                return ToolResult(
                    status=ToolStatus.ERROR,
                    data={
                        "error": f"Document type '{doc_type}' not fully supported",
                        "message": "Section extraction works best with LaTeX/arXiv sources"
                    },
                    formatted=f"[Section Extractor Error] Section extraction requires LaTeX source. Document type '{doc_type}' not fully supported.",
                    metadata={"doc_type": doc_type, "timestamp": datetime.now().isoformat()}
                )
            
            # Search for the section
            logger.info(f"[SectionExtractor] Searching through {len(self.latex_files)} files")
            result = self._search_all_files(spec_info)

            if not result:
                # Try ContextFinder fallback before giving up
                logger.info(f"[SectionExtractor] Section not found, trying ContextFinder fallback")
                fallback_result = self._run_context_finder_fallback(
                    section_spec=section_spec,
                    document_url=document_url
                )
                
                if fallback_result and fallback_result.is_success():
                    return fallback_result
                
                # Fallback also failed or not available
                return ToolResult(
                    status=ToolStatus.ERROR,
                    data={
                        "section_spec": section_spec,
                        "spec_info": spec_info,
                        "found": False
                    },
                    formatted=f"[Section Extractor Error] Could not find: '{section_spec}'\n\nSearched {len(self.latex_files)} files but no match found.",
                    metadata={"timestamp": datetime.now().isoformat()}
                )
            
            extracted_content, filename = result
            
            # Prepare result data
            result_data = {
                "section_spec": section_spec,
                "document_url": document_url,
                "document_type": doc_type,
                "spec_info": spec_info,
                "found": True,
                "filename": filename,
                "content": extracted_content,
                "content_length": len(extracted_content)
            }
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=result_data,
                formatted=self.format_results_for_agent(result_data),
                metadata={
                    "doc_type": doc_type,
                    "filename": filename,
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        except Exception as e:
            logger.error(f"[SectionExtractor] Error: {str(e)}")
            
            return ToolResult(
                status=ToolStatus.ERROR,
                data={"error": str(e)},
                formatted=f"[Section Extractor Error] Error extracting section: {str(e)}",
                metadata={"timestamp": datetime.now().isoformat()}
            )
        
        finally:
            # Clean up
            self._cleanup_temp_dir()
    
    def format_results_for_agent(self, data: Dict[str, Any]) -> str:
        """Format extraction results for agent consumption."""
        
        if not data or not data.get('found'):
            return "❌ Section not found."
        
        formatted = []
        
        formatted.append("**SECTION EXTRACTION RESULTS**")
        formatted.append("")
        formatted.append(f"Requested: \"{data['section_spec']}\"")
        formatted.append(f"Document: {data['document_url']}")
        formatted.append(f"Found in file: {data['filename']}")
        formatted.append("")
        
        # Show what type of section was extracted
        spec_info = data.get('spec_info', {})
        spec_type = spec_info.get('type', 'unknown')
        
        if spec_type == 'label':
            formatted.append(f"* Type: Labelled environment ({spec_info['label']})")
        elif spec_type == 'figure_table':
            formatted.append(f"* Type: {spec_info['env_type'].upper()} #{spec_info['number']}")
        elif spec_type == 'numbered_section':
            formatted.append(f"* Type: Section {spec_info['number']}")
            if spec_info.get('title'):
                formatted.append(f"    Title: {spec_info['title']}")
        elif spec_type == 'keyword':
            formatted.append(f"* Type: Keyword Section ({spec_info['keyword']})")
        elif spec_type == 'appendix':
            formatted.append(f"* Type: Appendix {spec_info['identifier']}")
        
        formatted.append(f"* Content length: {data['content_length']} characters")
        formatted.append("")
        formatted.append("---\n\nEXTRACTED CONTENT:")
        formatted.append("")
        
        # Display the content
        content = data.get('content', '')
        formatted.append(content)
        
        return "\n".join(formatted).strip()


# Example usage

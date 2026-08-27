"""
Context finder tool.
"""

import os
import re
import time
import json
import requests
import tempfile
import tarfile
import shutil
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse
from typing import Dict, Any, List, Optional, Set, Tuple

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

# Import InfoLocalizer
from anagent.tools.knowledge.info_localizer import InfoLocalizer


class ContextFinder(BaseTool):
    """Context finder that handles both .bib files and embedded bibliographies."""
    
    def __init__(self, **kwargs):
        super().__init__(
            name="context_finder",
            description="Find all related contexts (references to/from) for a string in a document",
            **kwargs
        )
        
        self.localizer = InfoLocalizer()
        self.temp_dir = None
        self.source_dir = None
        self.all_content = {}
        self.latex_files = []
        self.embedded_bibliography = {}  # store embedded \bibitem entries

        # Semantic scholar
        self.ss_paper_data = None  # Store raw Semantic Scholar data
        self.ss_api_delay = 0.1  # Rate limiting: 100ms between requests
        self.ss_last_request = 0
        self.ss_api_key = kwargs.get('semanticscholar_api_key', None)
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "search_string": {"type": "string", "description": "String to localize and find contexts for"},
                "document_url": {"type": "string", "description": "Document URL (arXiv, Semantic Scholar, PubMed, HTML, PDF) or local file path"},
                "max_depth": {"type": "integer", "description": "Maximum reference depth to follow (default: 2)"},
                "include_citations": {"type": "boolean", "description": "Include citation contexts (default: True)"}
            },
            "required": ["search_string", "document_url"]
        }
    
    def _cleanup_temp_dir(self):
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                logger.info(f"[ContextFinder] Cleaned up temp directory: {self.temp_dir}")
            except Exception as e:
                logger.warning(f"[ContextFinder] Failed to clean up temp dir: {e}")
    
    def _detect_url_type(self, url: str) -> str:
        url_lower = url.lower()
        if 'semanticscholar.org' in url_lower or '/semantic-scholar/' in url_lower:
            return 'semanticscholar'
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
    
    def _download_arxiv_source(self, arxiv_url: str) -> str:
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
        
        logger.info(f"[ContextFinder] Extracted arXiv ID: {arxiv_id}")
        self.temp_dir = tempfile.mkdtemp(prefix="context_finder_")
        source_url = f"https://arxiv.org/e-print/{arxiv_id}"
        
        logger.info(f"[ContextFinder] Downloading source from: {source_url}")
        response = requests.get(source_url, timeout=60, headers={
            'User-Agent': 'Mozilla/5.0 (ContextFinder/1.0)'
        })
        response.raise_for_status()
        
        tar_path = os.path.join(self.temp_dir, f"{arxiv_id}.tar.gz")
        with open(tar_path, 'wb') as f:
            f.write(response.content)
        
        extract_dir = os.path.join(self.temp_dir, 'source')
        os.makedirs(extract_dir, exist_ok=True)
        
        try:
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(extract_dir)
        except tarfile.ReadError:
            shutil.move(tar_path, os.path.join(extract_dir, f"{arxiv_id}.tex"))
        
        logger.info(f"[ContextFinder] Extracted source to: {extract_dir}")
        return extract_dir
    
    def _download_semanticscholar_source(self, ss_url: str) -> str:
        """Download paper metadata from Semantic Scholar API."""
        # Extract paper ID from various URL formats
        paper_id = None
        patterns = [
            r'semanticscholar\.org/reader/([a-f0-9]+)',  # /reader/ID (most common)
            r'semanticscholar\.org/paper/[^/]+/([a-f0-9]+)',  # /paper/title/ID
            r'semanticscholar\.org/paper/([a-f0-9]+)',  # /paper/ID
            r'api\.semanticscholar\.org/[^/]+/paper/([a-f0-9]+)',  # API URL
            r'/([a-f0-9]{40})(?:[/?#]|$)',  # Direct 40-char hash in URL
            r'(?:arXiv:)?(\d{4}\.\d{4,5})',  # arXiv ID (also supported by S2)
        ]
        
        for pattern in patterns:
            match = re.search(pattern, ss_url)
            if match:
                paper_id = match.group(1)
                # Validate it looks like a paper ID (40 hex chars or arxiv format)
                if len(paper_id) == 40 or re.match(r'\d{4}\.\d{4,5}', paper_id):
                    break
                paper_id = None  # Invalid match, keep searching
        
        if not paper_id:
            raise ValueError(f"Could not extract Semantic Scholar paper ID from URL: {ss_url}")
        
        logger.info(f"[ContextFinder] Extracted Semantic Scholar paper ID: {paper_id}")
        # Create temp directory
        self.temp_dir = tempfile.mkdtemp(prefix="context_finder_ss_")
        
        # Fetch paper metadata from Semantic Scholar API
        api_url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
        fields = "title,authors,abstract,url,year,venue,citationCount,referenceCount,openAccessPdf,externalIds"
        
        logger.info(f"[ContextFinder] Fetching from Semantic Scholar API: {api_url}")
        # Retry logic for rate limiting
        max_retries = 3
        retry_delay = 2.0
        
        for attempt in range(max_retries):
            try:
                # Rate limiting
                if hasattr(self, 'ss_last_request') and self.ss_last_request > 0:
                    time_since_last = time.time() - self.ss_last_request
                    if time_since_last < 1.0:  # Wait at least 1 second between requests
                        time.sleep(1.0 - time_since_last)
                
                headers = {'User-Agent': 'Mozilla/5.0 (ContextFinder/1.0)'}
                if self.ss_api_key:
                    headers['x-api-key'] = self.ss_api_key

                response = requests.get(
                    api_url,
                    params={"fields": fields},
                    timeout=30,
                    headers=headers
                )

                self.ss_last_request = time.time()
                
                if response.status_code == 429:
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"[ContextFinder] Rate limited. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise requests.exceptions.HTTPError(
                            f"Rate limited after {max_retries} attempts. "
                            "Semantic Scholar API has strict rate limits. "
                            "Consider getting an API key from https://www.semanticscholar.org/product/api"
                        )
                
                response.raise_for_status()
                paper_data = response.json()
                break  # Success, exit retry loop
                
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1 and '429' not in str(e):
                    logger.warning(f"[ContextFinder] Request failed: {e}. Retrying...")
                    time.sleep(retry_delay)
                    continue
                raise ValueError(f"Failed to fetch paper from Semantic Scholar: {e}")
        
        # Save metadata to JSON file
        metadata_path = os.path.join(self.temp_dir, 'paper_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(paper_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[ContextFinder] Saved metadata to: {metadata_path}")
        logger.info(f"[ContextFinder] Paper title: {paper_data.get('title', 'N/A')}")
        return self.temp_dir
    
    def _read_file_safe(self, filepath: str) -> str:
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        for encoding in encodings:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    return f.read()
            except (UnicodeDecodeError, Exception):
                continue
        return ""
    
    def _load_all_latex_files(self, directory: str):
        """Load all LaTeX/BibTeX files AND parse embedded bibliographies."""
        self.latex_files = []
        self.all_content = {}
        self.embedded_bibliography = {}
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(('.tex', '.bib')):
                    filepath = os.path.join(root, file)
                    self.latex_files.append(filepath)
                    content = self._read_file_safe(filepath)
                    self.all_content[filepath] = content
                    
                    # NEW: Parse embedded bibliography from .tex files
                    if file.endswith('.tex'):
                        self._parse_embedded_bibliography(content, filepath)
        
        tex_count = sum(1 for f in self.latex_files if f.endswith('.tex'))
        bib_count = sum(1 for f in self.latex_files if f.endswith('.bib'))
        
        logger.info(f"[ContextFinder] Loaded {tex_count} .tex files and {bib_count} .bib files")
        if bib_count > 0:
            logger.info(f"[ContextFinder] BibTeX files: {[os.path.basename(f) for f in self.latex_files if f.endswith('.bib')]}")
        if self.embedded_bibliography:
            logger.info(f"[ContextFinder] Found {len(self.embedded_bibliography)} embedded bibliography entries")
    def _parse_embedded_bibliography(self, content: str, filepath: str):
        r"""Parse embedded bibliography from \begin{thebibliography}...\end{thebibliography}."""
        # Find thebibliography environment
        bib_pattern = r'\\begin\{thebibliography\}.*?\\end\{thebibliography\}'
        matches = re.finditer(bib_pattern, content, re.DOTALL)
        
        for match in matches:
            bib_content = match.group(0)
            
            # Extract individual \bibitem entries
            bibitem_pattern = r'\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}(.*?)(?=\\bibitem|\\end\{thebibliography\})'
            
            for item_match in re.finditer(bibitem_pattern, bib_content, re.DOTALL):
                citation_key = item_match.group(1).strip()
                citation_content = item_match.group(2).strip()
                
                # Store the citation
                if citation_key not in self.embedded_bibliography:
                    self.embedded_bibliography[citation_key] = {
                        'citation_key': citation_key,
                        'content': f'\\bibitem{{{citation_key}}}\n{citation_content}',
                        'file': os.path.basename(filepath),
                        'full_path': filepath,
                        'type': 'embedded_bibitem',
                        'raw_content': citation_content
                    }
                    
                    # Try to parse metadata from content
                    metadata = self._parse_bibitem_metadata(citation_content)
                    self.embedded_bibliography[citation_key].update(metadata)
    
    def _load_semanticscholar_content(self):
        """Load Semantic Scholar paper metadata into all_content, including references and citations."""
        metadata_path = os.path.join(self.temp_dir, 'paper_metadata.json')
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            paper_data = json.load(f)
        
        # Build a text representation of the paper metadata ONLY
        content_parts = []
        
        # Title
        title = paper_data.get('title', '')
        if title:
            content_parts.append(f"TITLE: {title}")
        
        # Authors
        authors = paper_data.get('authors', [])
        if authors:
            author_names = [a.get('name', '') for a in authors if a.get('name')]
            if author_names:
                content_parts.append(f"AUTHORS: {', '.join(author_names)}")
        
        # Year and Venue
        year = paper_data.get('year')
        venue = paper_data.get('venue', '')
        if year:
            content_parts.append(f"YEAR: {year}")
        if venue:
            content_parts.append(f"VENUE: {venue}")
        
        # Citation/Reference counts
        citation_count = paper_data.get('citationCount', 0)
        reference_count = paper_data.get('referenceCount', 0)
        content_parts.append(f"CITATIONS: {citation_count}")
        content_parts.append(f"REFERENCES: {reference_count}")
        
        # URL
        url = paper_data.get('url', '')
        if url:
            content_parts.append(f"URL: {url}")
        
        # Abstract
        abstract = paper_data.get('abstract', '')
        if abstract:
            content_parts.append(f"\nABSTRACT:\n{abstract}")
        
        # Combine into single content string
        full_content = '\n'.join(content_parts)
        
        # Store ONLY metadata in all_content
        self.all_content['paper_metadata'] = full_content
        
        # Store raw paper data for later use
        self.ss_paper_data = paper_data
        
        # Now fetch references and citations in separate API calls
        paper_id = paper_data.get('paperId')
        if paper_id:
            logger.info(f"[ContextFinder] Fetching references and citations for paper {paper_id}")
            # Fetch references
            try:
                references = self._fetch_semanticscholar_references(paper_id, limit=100)
                if references:
                    logger.info(f"[ContextFinder] Fetched {len(references)} references")
                    self._process_semanticscholar_references(references)
            except Exception as e:
                logger.warning(f"[ContextFinder] Failed to fetch references: {e}")
            # Fetch citations
            try:
                citations = self._fetch_semanticscholar_citations(paper_id, limit=100)
                if citations:
                    logger.info(f"[ContextFinder] Fetched {len(citations)} citations")
                    self._process_semanticscholar_citations(citations)
            except Exception as e:
                logger.warning(f"[ContextFinder] Failed to fetch citations: {e}")
        logger.info(f"[ContextFinder] Loaded Semantic Scholar paper: {title}")
        logger.info(f"[ContextFinder] Abstract length: {len(abstract)} chars")
        logger.info(f"[ContextFinder] Embedded bibliography entries: {len(self.embedded_bibliography)}")
    def _fetch_semanticscholar_references(self, paper_id: str, limit: int = 10) -> List[Dict]:
        """Fetch references for a paper from Semantic Scholar API."""
        api_url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/references"
        fields = "title,authors,year,venue,paperId,url"
        
        all_references = []
        offset = 0
        batch_size = min(100, limit)  # API allows max 100 per request
        
        while len(all_references) < limit:
            # Rate limiting
            if hasattr(self, 'ss_last_request') and self.ss_last_request > 0:
                time_since_last = time.time() - self.ss_last_request
                if time_since_last < 1.0:
                    time.sleep(1.0 - time_since_last)
            
            try:
                headers = {'User-Agent': 'Mozilla/5.0 (ContextFinder/1.0)'}
                if hasattr(self, 'ss_api_key') and self.ss_api_key:
                    headers['x-api-key'] = self.ss_api_key
                
                response = requests.get(
                    api_url,
                    params={
                        "fields": fields,
                        "limit": batch_size,
                        "offset": offset
                    },
                    timeout=30,
                    headers=headers
                )
                
                self.ss_last_request = time.time()
                
                if response.status_code == 429:
                    logger.warning("[ContextFinder] Rate limited while fetching references. Waiting 5s...")
                    time.sleep(5)
                    continue
                
                response.raise_for_status()
                data = response.json()
                
                references_data = data.get('data', [])
                if not references_data:
                    break  # No more references
                
                # Extract cited paper info
                for ref_item in references_data:
                    cited_paper = ref_item.get('citedPaper', {})
                    if cited_paper:
                        all_references.append(cited_paper)
                
                if len(references_data) < batch_size:
                    break  # Last page
                
                offset += batch_size
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"[ContextFinder] Error fetching references at offset {offset}: {e}")
                break
        
        return all_references[:limit]

    def _fetch_semanticscholar_pdf_text(self, paper_data: dict) -> Optional[str]:
        """Try to fetch and extract text from the paper's PDF."""
        if not HAS_PDF:
            logger.warning("[ContextFinder] PyPDF2 not available for PDF extraction")
            return None
        
        # Try to get PDF URL from various sources
        pdf_url = None
        
        # Check for open access PDF in the paper data
        if 'openAccessPdf' in paper_data and paper_data['openAccessPdf']:
            pdf_url = paper_data['openAccessPdf'].get('url')
        
        # If no open access PDF, try the external IDs (arXiv)
        if not pdf_url and 'externalIds' in paper_data:
            external_ids = paper_data['externalIds']
            if 'ArXiv' in external_ids:
                arxiv_id = external_ids['ArXiv']
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        
        if not pdf_url:
            logger.warning("[ContextFinder] No PDF URL found for this paper")
            return None
        
        logger.info(f"[ContextFinder] Downloading PDF from: {pdf_url}")
        try:
            # Download PDF
            response = requests.get(pdf_url, timeout=60, headers={
                'User-Agent': 'Mozilla/5.0 (ContextFinder/1.0)'
            })
            response.raise_for_status()
            
            # Save to temp file
            pdf_path = os.path.join(self.temp_dir, 'paper.pdf')
            with open(pdf_path, 'wb') as f:
                f.write(response.content)
            
            # Extract text
            text_parts = []
            with open(pdf_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        text = page.extract_text()
                        if text:
                            text_parts.append(text)
                    except Exception as e:
                        logger.warning(f"[ContextFinder] Failed to extract page {page_num}: {e}")
            full_text = '\n\n'.join(text_parts)
            
            logger.info(f"[ContextFinder] Extracted {len(full_text)} chars from PDF")
            return full_text
            
        except Exception as e:
            logger.warning(f"[ContextFinder] Failed to fetch/extract PDF: {e}")
            return None
    
    def _fetch_semanticscholar_citations(self, paper_id: str, limit: int = 10) -> List[Dict]:
        """Fetch citations for a paper from Semantic Scholar API."""
        api_url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations"
        fields = "title,authors,year,venue,paperId,url"
        
        all_citations = []
        offset = 0
        batch_size = min(100, limit)  # API allows max 100 per request
        
        while len(all_citations) < limit:
            # Rate limiting
            if hasattr(self, 'ss_last_request') and self.ss_last_request > 0:
                time_since_last = time.time() - self.ss_last_request
                if time_since_last < 1.0:
                    time.sleep(1.0 - time_since_last)
            
            try:
                headers = {'User-Agent': 'Mozilla/5.0 (ContextFinder/1.0)'}
                if hasattr(self, 'ss_api_key') and self.ss_api_key:
                    headers['x-api-key'] = self.ss_api_key
                
                response = requests.get(
                    api_url,
                    params={
                        "fields": fields,
                        "limit": batch_size,
                        "offset": offset
                    },
                    timeout=30,
                    headers=headers
                )
                
                self.ss_last_request = time.time()
                
                if response.status_code == 429:
                    logger.warning("[ContextFinder] Rate limited while fetching citations. Waiting 5s...")
                    time.sleep(5)
                    continue
                
                response.raise_for_status()
                data = response.json()
                
                citations_data = data.get('data', [])
                if not citations_data:
                    break  # No more citations
                
                # Extract citing paper info
                for cit_item in citations_data:
                    citing_paper = cit_item.get('citingPaper', {})
                    if citing_paper:
                        all_citations.append(citing_paper)
                
                if len(citations_data) < batch_size:
                    break  # Last page
                
                offset += batch_size
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"[ContextFinder] Error fetching citations at offset {offset}: {e}")
                break
        
        return all_citations[:limit]

    def _process_semanticscholar_references(self, references: List[Dict]):
        """Process Semantic Scholar references into embedded bibliography format."""
        for ref in references:
            if not ref or not isinstance(ref, dict):
                continue
            
            paper_id = ref.get('paperId', '')
            if not paper_id:
                continue
            
            # Create a citation key from paper ID
            citation_key = f"ss_ref_{paper_id[:8]}"
            
            # Skip if already processed
            if citation_key in self.embedded_bibliography:
                continue
            
            # Build reference entry
            ref_authors = ref.get('authors', [])
            author_str = ', '.join([a.get('name', '') for a in ref_authors[:3] if a.get('name')])
            if len(ref_authors) > 3:
                author_str += ' et al.'
            
            title = ref.get('title', 'Unknown Title')
            year = ref.get('year', '')
            venue = ref.get('venue', '')
            url = ref.get('url', '')
            
            # Build content string similar to BibTeX entry
            content_parts = []
            if author_str:
                content_parts.append(author_str)
            content_parts.append(title)
            if venue:
                content_parts.append(venue)
            if year:
                content_parts.append(str(year))
            
            content_str = '. '.join(content_parts)
            
            self.embedded_bibliography[citation_key] = {
                'citation_key': citation_key,
                'title': title,
                'author': author_str,
                'year': str(year) if year else '',
                'venue': venue,
                'url': url,
                'paperId': paper_id,
                'type': 'semanticscholar_reference',
                'content': content_str,
                'file': 'Semantic Scholar API',
                'full_path': f'semanticscholar://{paper_id}'
            }

    def _process_semanticscholar_citations(self, citations: List[Dict]):
        """Process Semantic Scholar citations (papers that cite this paper)."""
        for cit in citations:
            if not cit or not isinstance(cit, dict):
                continue
            
            paper_id = cit.get('paperId', '')
            if not paper_id:
                continue
            
            # Create a citation key from paper ID
            citation_key = f"ss_cit_{paper_id[:8]}"
            
            # Skip if already processed
            if citation_key in self.embedded_bibliography:
                continue
            
            # Build citation entry
            cit_authors = cit.get('authors', [])
            author_str = ', '.join([a.get('name', '') for a in cit_authors[:3] if a.get('name')])
            if len(cit_authors) > 3:
                author_str += ' et al.'
            
            title = cit.get('title', 'Unknown Title')
            year = cit.get('year', '')
            venue = cit.get('venue', '')
            url = cit.get('url', '')
            
            # Build content string
            content_parts = []
            if author_str:
                content_parts.append(author_str)
            content_parts.append(title)
            if venue:
                content_parts.append(venue)
            if year:
                content_parts.append(str(year))
            
            content_str = '. '.join(content_parts)
            
            self.embedded_bibliography[citation_key] = {
                'citation_key': citation_key,
                'title': title,
                'author': author_str,
                'year': str(year) if year else '',
                'venue': venue,
                'url': url,
                'paperId': paper_id,
                'type': 'semanticscholar_citation',
                'content': content_str,
                'file': 'Semantic Scholar API',
                'full_path': f'semanticscholar://{paper_id}'
            }

    def _find_in_semanticscholar_content(self, search_string: str) -> Optional[Dict[str, Any]]:
        """Find search string in Semantic Scholar paper content."""
        # First, try searching in metadata
        if 'paper_metadata' in self.all_content:
            content = self.all_content['paper_metadata']
            pos = content.find(search_string)
            
            # Try case-insensitive if exact match fails
            if pos == -1:
                content_lower = content.lower()
                search_lower = search_string.lower()
                pos = content_lower.find(search_lower)
            
            if pos != -1:
                # Found in metadata, extract paragraph
                para_start = content.rfind('\n\n', 0, pos)
                if para_start == -1:
                    para_start = 0
                else:
                    para_start += 2
                
                para_end = content.find('\n\n', pos)
                if para_end == -1:
                    para_end = len(content)
                
                paragraph = content[para_start:para_end].strip()
                
                return {
                    'content': paragraph,
                    'matched_text': content[pos:pos+len(search_string)],
                    'position': pos,
                    'source': 'metadata'
                }
        
        # Not found in metadata, try PDF
        logger.info(f"[ContextFinder] '{search_string}' not found in metadata, trying PDF...")
        pdf_text = self._fetch_semanticscholar_pdf_text(self.ss_paper_data)
        if not pdf_text:
            return None
        
        # Search in PDF text
        pos = pdf_text.find(search_string)
        
        # Try case-insensitive if exact match fails
        if pos == -1:
            pdf_lower = pdf_text.lower()
            search_lower = search_string.lower()
            pos = pdf_lower.find(search_lower)
        
        if pos == -1:
            logger.warning(f"[ContextFinder] '{search_string}' not found in PDF either")
            return None
        
        # Found in PDF - extract the relevant section/paragraph ONLY
        para_start = pdf_text.rfind('\n\n', 0, pos)
        if para_start == -1:
            para_start = max(0, pos - 500)
        else:
            para_start += 2
        
        para_end = pdf_text.find('\n\n', pos)
        if para_end == -1:
            para_end = min(len(pdf_text), pos + 500)
        
        # Extract the section containing the match
        extracted_section = pdf_text[para_start:para_end].strip()
        
        # Try to find section heading before this paragraph
        section_start = pdf_text.rfind('\n', 0, para_start - 10)
        if section_start != -1:
            potential_heading = pdf_text[section_start:para_start].strip()
            # Check if it looks like a heading (short, starts with number or capital)
            if len(potential_heading) < 100 and (potential_heading[0].isupper() if potential_heading else False):
                extracted_section = potential_heading + '\n\n' + extracted_section
        
        logger.info(f"[ContextFinder] Found '{search_string}' in PDF at position {pos}")
        logger.info(f"[ContextFinder] Extracted section length: {len(extracted_section)} chars")
        return {
            'content': extracted_section,
            'matched_text': pdf_text[pos:pos+len(search_string)],
            'position': pos,
            'source': 'pdf'
        }

    def _ss_api_request(self, endpoint: str, params: dict = None) -> dict:
        """Make a rate-limited request to Semantic Scholar API."""
        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self.ss_last_request
        if time_since_last < self.ss_api_delay:
            time.sleep(self.ss_api_delay - time_since_last)
        
        try:
            response = requests.get(
                endpoint,
                params=params,
                timeout=30,
                headers={'User-Agent': 'Mozilla/5.0 (ContextFinder/1.0)'}
            )
            response.raise_for_status()
            self.ss_last_request = time.time()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"[ContextFinder] Semantic Scholar API error: {e}")
            return {}
    
    def _parse_bibitem_metadata(self, content: str) -> Dict[str, str]:
        """Parse metadata from bibitem content."""
        metadata = {}
        
        # Split by \newblock to get structured parts
        parts = re.split(r'\\newblock\s*', content)
        
        # Clean function to remove LaTeX commands but keep content
        def clean_latex(text):
            # Remove \em, \bf, etc but keep the content
            text = re.sub(r'\\(?:em|bf|it|tt|sc)\s+', '', text)
            # Remove {, } around emphasized text  
            text = re.sub(r'\{\\em\s+([^}]+)\}', r'\1', text)
            # Remove curly braces around em content
            text = re.sub(r'\{\\em ([^}]+)\}', r'\1', text)
            # Remove remaining curly braces
            text = re.sub(r'[{}]', '', text)
            # Remove LaTeX tilde (non-breaking space)
            text = re.sub(r'~', ' ', text)
            # Remove backslash commands
            text = re.sub(r'\\[a-zA-Z]+\s*', ' ', text)
            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            # Remove trailing commas and periods
            text = re.sub(r'[,\.]+$', '', text).strip()
            return text
        
        # Part 0: Usually authors (before first \newblock)
        if len(parts) > 0:
            authors = clean_latex(parts[0])
            # Remove trailing period
            authors = re.sub(r'\.$', '', authors).strip()
            if authors and len(authors) < 300:  # Reasonable author length
                metadata['author'] = authors
        
        # Part 1: Usually title (after first \newblock)
        if len(parts) > 1:
            title = clean_latex(parts[1])
            # Remove trailing period
            title = re.sub(r'\.$', '', title).strip()
            if title:
                metadata['title'] = title
        
        # Part 2+: Usually venue and year
        if len(parts) > 2:
            venue_year = ' '.join(parts[2:])
            venue_year = clean_latex(venue_year)
            # Remove the year from venue if present (it's shown separately)
            venue_year = re.sub(r',?\s*(19|20)\d{2}\.?$', '', venue_year).strip()
            if venue_year:
                metadata['venue'] = venue_year
        
        # Extract year from anywhere in content (4 digits)
        year_match = re.search(r'\b(19|20)\d{2}\b', content)
        if year_match:
            metadata['year'] = year_match.group(0)
        
        return metadata
    
    def _extract_labels_from_content(self, content: str) -> Set[str]:
        labels = set()
        label_pattern = r'\\label\{([^}]+)\}'
        for match in re.finditer(label_pattern, content):
            labels.add(match.group(1))
        return labels
    
    def _extract_references_from_content(self, content: str) -> Set[str]:
        """Extract all references and citations with comprehensive pattern support."""
        references = set()
        
        # Cross-references - comprehensive patterns including section references with \S
        ref_patterns = [
            # Basic references
            r'\\ref\{([^}]+)\}', 
            r'\\pageref\{([^}]+)\}', 
            r'\\eqref\{([^}]+)\}',
            r'\\autoref\{([^}]+)\}', 
            r'\\nameref\{([^}]+)\}', 
            r'\\hyperref\[([^\]]+)\]',
            
            # cleveref package
            r'\\cref\{([^}]+)\}', 
            r'\\Cref\{([^}]+)\}',
            r'\\crefrange\{([^}]+)\}\{([^}]+)\}',  # Handle range start and end
            r'\\Crefrange\{([^}]+)\}\{([^}]+)\}',
            r'\\cpageref\{([^}]+)\}',
            r'\\Cpageref\{([^}]+)\}',
            r'\\labelcref\{([^}]+)\}',
            r'\\labelcpageref\{([^}]+)\}',
            
            # varioref package
            r'\\vref\{([^}]+)\}',
            r'\\vpageref\{([^}]+)\}',
            
            # Custom reference commands
            r'\\figref\{([^}]+)\}',
            r'\\tabref\{([^}]+)\}',
            r'\\equationref\{([^}]+)\}',
            r'\\secref\{([^}]+)\}',
            r'\\chapref\{([^}]+)\}',
            
            # Section references with \S (with various spacing)
            r'\\S\\ref\{([^}]+)\}',  # \S\ref{label}
            r'\\S~\\ref\{([^}]+)\}',  # \S~\ref{label}
            r'\\S\s+\\ref\{([^}]+)\}',  # \S \ref{label}
            
            # Figure/Table/Section with various spacing
            r'(?:Fig\.|Figure)~\\ref\{([^}]+)\}',
            r'(?:Fig\.|Figure)\s+\\ref\{([^}]+)\}',
            r'(?:Fig\.|Figure)\\ref\{([^}]+)\}',
            r'(?:Tab\.|Table)~\\ref\{([^}]+)\}',
            r'(?:Tab\.|Table)\s+\\ref\{([^}]+)\}',
            r'(?:Tab\.|Table)\\ref\{([^}]+)\}',
            r'(?:Sec\.|Section)~\\ref\{([^}]+)\}',
            r'(?:Sec\.|Section)\s+\\ref\{([^}]+)\}',
            r'(?:Sec\.|Section)\\ref\{([^}]+)\}',
        ]
        
        for pattern in ref_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                # Handle patterns with multiple groups (like crefrange)
                for group_idx in range(1, len(match.groups()) + 1):
                    group_content = match.group(group_idx)
                    if group_content:
                        refs = group_content.split(',')
                        for ref in refs:
                            clean_ref = ref.strip()
                            if clean_ref:
                                references.add(clean_ref)
        
        # Citations - ALL formats including natbib and biblatex
        cite_patterns = [
            # Standard LaTeX
            r'\\cite\{([^}]+)\}', 
            r'\\nocite\{([^}]+)\}',
            
            # natbib package
            r'\\citep\{([^}]+)\}', 
            r'\\citet\{([^}]+)\}',
            r'\\citealp\{([^}]+)\}', 
            r'\\citealt\{([^}]+)\}',
            r'\\citeauthor\{([^}]+)\}', 
            r'\\citeyear\{([^}]+)\}',
            r'\\citeyearpar\{([^}]+)\}',
            r'\\citetext\{([^}]+)\}',
            
            # natbib with capitalized versions
            r'\\Citep\{([^}]+)\}',
            r'\\Citet\{([^}]+)\}',
            r'\\Citealp\{([^}]+)\}',
            r'\\Citealt\{([^}]+)\}',
            r'\\Citeauthor\{([^}]+)\}',
            
            # natbib with optional arguments (pre and post notes)
            r'\\citep\[[^\]]*\]\{([^}]+)\}', 
            r'\\citet\[[^\]]*\]\{([^}]+)\}',
            r'\\citep\[[^\]]*\]\[[^\]]*\]\{([^}]+)\}',  # \citep[pre][post]{key}
            r'\\citet\[[^\]]*\]\[[^\]]*\]\{([^}]+)\}',
            
            # biblatex package
            r'\\textcite\{([^}]+)\}', 
            r'\\parencite\{([^}]+)\}',
            r'\\footcite\{([^}]+)\}', 
            r'\\fullcite\{([^}]+)\}',
            r'\\citeurl\{([^}]+)\}',
            r'\\citetitle\{([^}]+)\}',
            r'\\autocite\{([^}]+)\}',
            r'\\supercite\{([^}]+)\}',
            r'\\smartcite\{([^}]+)\}',
            
            # biblatex capitalized
            r'\\Textcite\{([^}]+)\}',
            r'\\Parencite\{([^}]+)\}',
            r'\\Footcite\{([^}]+)\}',
            r'\\Autocite\{([^}]+)\}',
            r'\\Smartcite\{([^}]+)\}',
            
            # biblatex variants
            r'\\footcitetext\{([^}]+)\}',
            r'\\notecite\{([^}]+)\}',
            r'\\pnotecite\{([^}]+)\}',
            r'\\fnotecite\{([^}]+)\}',
            
            # Multiple citations
            r'\\cites\{([^}]+)\}',
            r'\\parencites\{([^}]+)\}',
            r'\\textcites\{([^}]+)\}',
            r'\\footcites\{([^}]+)\}',
            r'\\autocites\{([^}]+)\}',
            
            # Custom citation commands
            r'\\citework\{([^}]+)\}',
            r'\\shortcite\{([^}]+)\}',
            r'\\citefullauthor\{([^}]+)\}',
        ]
        
        for pattern in cite_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                citations = match.group(1).split(',')
                for citation in citations:
                    clean_citation = citation.strip()
                    if clean_citation:
                        references.add(clean_citation)
        
        return references
    
    def _find_content_with_label(self, label: str) -> List[Dict[str, Any]]:
        """Find content with a specific label."""
        results = []
        
        for filepath, content in self.all_content.items():
            label_pattern = f'\\\\label{{{re.escape(label)}}}'
            if re.search(label_pattern, content):
                pos = content.find(f'\\label{{{label}}}')
                if pos != -1:
                    extracted = self._extract_environment_at_position(content, pos, filepath)
                    if extracted:
                        results.append({
                            'label': label,
                            'file': os.path.basename(filepath),
                            'full_path': filepath,
                            'type': extracted['type'],
                            'content': extracted['content']
                        })
        
        return results
    
    def _find_content_with_citation(self, citation_key: str) -> List[Dict[str, Any]]:
        """Find citation - CHECKS BOTH .bib files AND embedded bibliography!"""
        results = []
        
        # First, check embedded bibliography (most common for arXiv papers)
        if citation_key in self.embedded_bibliography:
            logger.info(f"[ContextFinder]   ✓ FOUND in embedded bibliography: {citation_key}")
            results.append(self.embedded_bibliography[citation_key].copy())
        
        # Also check .bib files
        bib_files = [f for f in self.all_content.keys() if f.endswith('.bib')]
        
        if bib_files:
            logger.info(f"[ContextFinder]   Searching in {len(bib_files)} .bib files for: {citation_key}")
            for filepath, content in self.all_content.items():
                if not filepath.endswith('.bib'):
                    continue
                
                citation_patterns = [
                    f'@\\w+{{{re.escape(citation_key)},',
                    f'@\\w+{{{re.escape(citation_key)}}}',
                    f'@\\w+\\s*{{\\s*{re.escape(citation_key)}\\s*,',
                ]
                
                for pattern in citation_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    
                    for match in matches:
                        pos = match.start()
                        entry = self._extract_bibtex_entry_at_position(content, pos)
                        
                        if entry:
                            metadata = self._parse_bibtex_entry(entry)
                            entry_hash = hash(entry)
                            already_added = any(
                                hash(r.get('content', '')) == entry_hash 
                                for r in results
                            )
                            
                            if not already_added:
                                result_dict = {
                                    'citation_key': citation_key,
                                    'file': os.path.basename(filepath),
                                    'full_path': filepath,
                                    'type': 'bibtex_entry',
                                    'content': entry
                                }
                                result_dict.update(metadata)
                                results.append(result_dict)
                                
                                logger.info(f"[ContextFinder]   ✓ FOUND in .bib file: {citation_key}")
        if not results:
            logger.warning(f"[ContextFinder]   ✗ Citation '{citation_key}' not found")
        
        return results
    
    def _parse_bibtex_entry(self, entry: str) -> Dict[str, str]:
        """Parse BibTeX entry for metadata."""
        metadata = {}
        
        title_match = re.search(r'title\s*=\s*[{"]\s*([^"}]+)\s*["}]', entry, re.IGNORECASE)
        if title_match:
            metadata['title'] = title_match.group(1).strip()
        
        author_match = re.search(r'author\s*=\s*[{"]\s*([^"}]+)\s*["}]', entry, re.IGNORECASE)
        if author_match:
            metadata['author'] = author_match.group(1).strip()
        
        year_match = re.search(r'year\s*=\s*[{"]\s*(\d{4})\s*["}]', entry, re.IGNORECASE)
        if year_match:
            metadata['year'] = year_match.group(1)
        
        url_match = re.search(r'url\s*=\s*[{"]\s*([^"}]+)\s*["}]', entry, re.IGNORECASE)
        if url_match:
            metadata['url'] = url_match.group(1).strip()
        
        return metadata
    
    def _extract_bibtex_entry_at_position(self, content: str, pos: int) -> Optional[str]:
        """Extract complete BibTeX entry starting at position."""
        at_pos = content.rfind('@', 0, pos + 50)
        if at_pos == -1:
            return None
        
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
        
        return content[at_pos:entry_end]
    
    def _extract_environment_at_position(self, content: str, pos: int, filepath: str) -> Optional[Dict[str, str]]:
        """Extract LaTeX environment or section containing a position."""
        environments = ['figure', 'table', 'equation', 'align', 'gather', 'algorithm', 'theorem', 'lemma', 'proof', 'definition']
        
        for env in environments:
            env_content = self._find_latex_environment(content, pos, env)
            if env_content:
                return {'type': env, 'content': env_content}
        
        section_content = self._extract_section_at_position(content, pos)
        if section_content:
            return {'type': 'section', 'content': section_content}
        
        para_content = self._extract_paragraph_at_position(content, pos)
        if para_content:
            return {'type': 'paragraph', 'content': para_content}
        
        return None
    
    def _find_latex_environment(self, content: str, pos: int, env_name: str) -> Optional[str]:
        """Find LaTeX environment containing position."""
        begin_pattern = f'\\\\begin{{{env_name}}}'
        end_pattern = f'\\\\end{{{env_name}}}'
        
        begin_pos = content.rfind(begin_pattern, 0, pos + 100)
        if begin_pos == -1:
            return None
        
        end_pos = content.find(end_pattern, begin_pos)
        if end_pos == -1:
            return None
        
        end_pos += len(end_pattern)
        
        if begin_pos <= pos <= end_pos:
            return content[begin_pos:end_pos]
        
        return None
    
    def _extract_section_at_position(self, content: str, pos: int) -> Optional[str]:
        """Extract section containing position."""
        section_patterns = [
            r'\\section\*?\{[^}]+\}',
            r'\\subsection\*?\{[^}]+\}',
            r'\\subsubsection\*?\{[^}]+\}',
        ]
        
        section_starts = []
        for pattern in section_patterns:
            for match in re.finditer(pattern, content):
                section_starts.append(match.start())
        
        section_starts = sorted(set(section_starts))
        if not section_starts:
            return None
        
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
            return content[current_section_start:next_section_start].strip()
        
        return None
    
    def _extract_paragraph_at_position(self, content: str, pos: int) -> Optional[str]:
        """Extract paragraph containing position."""
        start = pos
        end = pos
        
        while start > 0:
            if content[start:start+2] == '\n\n':
                start += 2
                break
            start -= 1
        
        while end < len(content) - 1:
            if content[end:end+2] == '\n\n':
                break
            end += 1
        
        para = content[start:end].strip()
        if len(para) > 20:
            return para
        
        return None
    
    def _find_all_content_referring_to_labels(self, labels: Set[str]) -> List[Dict[str, Any]]:
        """Find all content that REFERS TO the given labels with comprehensive pattern support."""
        referring_content = []
        
        for label in labels:
            escaped_label = re.escape(label)
            for filepath, content in self.all_content.items():
                # Comprehensive reference patterns including \S\ref, \S~\ref, etc.
                ref_patterns = [
                    # Basic references
                    f'\\\\ref{{{escaped_label}}}',
                    f'\\\\pageref{{{escaped_label}}}',
                    f'\\\\eqref{{{escaped_label}}}',
                    f'\\\\autoref{{{escaped_label}}}',
                    f'\\\\nameref{{{escaped_label}}}',
                    f'\\\\hyperref\\[{escaped_label}\\]',
                    
                    # cleveref
                    f'\\\\cref{{{escaped_label}}}',
                    f'\\\\Cref{{{escaped_label}}}',
                    f'\\\\cpageref{{{escaped_label}}}',
                    f'\\\\Cpageref{{{escaped_label}}}',
                    f'\\\\labelcref{{{escaped_label}}}',
                    f'\\\\labelcpageref{{{escaped_label}}}',
                    
                    # varioref
                    f'\\\\vref{{{escaped_label}}}',
                    f'\\\\vpageref{{{escaped_label}}}',
                    
                    # Custom commands
                    f'\\\\figref{{{escaped_label}}}',
                    f'\\\\tabref{{{escaped_label}}}',
                    f'\\\\equationref{{{escaped_label}}}',
                    f'\\\\secref{{{escaped_label}}}',
                    f'\\\\chapref{{{escaped_label}}}',
                    
                    # Section references with \S
                    f'\\\\S\\\\ref{{{escaped_label}}}',  # \S\ref{label}
                    f'\\\\S~\\\\ref{{{escaped_label}}}',  # \S~\ref{label}
                    f'\\\\S\\s+\\\\ref{{{escaped_label}}}',  # \S \ref{label}
                    
                    # Figure/Table/Section with various spacing
                    f'(?:Fig\\.|Figure)~\\\\ref{{{escaped_label}}}',
                    f'(?:Fig\\.|Figure)\\s+\\\\ref{{{escaped_label}}}',
                    f'(?:Fig\\.|Figure)\\\\ref{{{escaped_label}}}',
                    f'(?:Tab\\.|Table)~\\\\ref{{{escaped_label}}}',
                    f'(?:Tab\\.|Table)\\s+\\\\ref{{{escaped_label}}}',
                    f'(?:Tab\\.|Table)\\\\ref{{{escaped_label}}}',
                    f'(?:Sec\\.|Section)~\\\\ref{{{escaped_label}}}',
                    f'(?:Sec\\.|Section)\\s+\\\\ref{{{escaped_label}}}',
                    f'(?:Sec\\.|Section)\\\\ref{{{escaped_label}}}',
                ]
                
                # Patterns for comma-separated lists (e.g., \cref{fig1,fig2,label})
                list_patterns = [
                    f'\\\\ref{{[^}}]*{escaped_label}[^}}]*}}',
                    f'\\\\cref{{[^}}]*{escaped_label}[^}}]*}}',
                    f'\\\\Cref{{[^}}]*{escaped_label}[^}}]*}}',
                    f'\\\\crefrange{{[^}}]*{escaped_label}[^}}]*}}{{[^}}]*}}',
                    f'\\\\crefrange{{[^}}]*}}{{[^}}]*{escaped_label}[^}}]*}}',
                    f'\\\\autoref{{[^}}]*{escaped_label}[^}}]*}}',
                ]
                
                for pattern in ref_patterns + list_patterns:
                    for match in re.finditer(pattern, content, re.IGNORECASE):
                        pos = match.start()
                        extracted = self._extract_environment_at_position(content, pos, filepath)
                        
                        if extracted:
                            referring_content.append({
                                'refers_to_label': label,
                                'file': os.path.basename(filepath),
                                'full_path': filepath,
                                'type': extracted['type'],
                                'content': extracted['content'],
                                'reference_position': pos,
                                'reference_command': match.group(0)
                            })
        
        # Remove duplicates
        unique_content = []
        seen = set()
        for item in referring_content:
            content_hash = (hash(item['content'][:200]), item.get('reference_position', 0))
            if content_hash not in seen:
                seen.add(content_hash)
                unique_content.append(item)
        
        return unique_content
    
    def _find_all_references_from_content(self, localized_content: str) -> List[Dict[str, Any]]:
        """Find all content that the localized content REFERS TO."""
        references = self._extract_references_from_content(localized_content)
        
        logger.info(f"[ContextFinder] Found {len(references)} references in localized content")
        logger.info(f"[ContextFinder] References: {list(references)}")
        referenced_content = []
        
        for ref in references:
            logger.info(f"[ContextFinder] Processing reference: '{ref}'")
            # Try as label first
            label_results = self._find_content_with_label(ref)
            
            if label_results:
                logger.info(f"[ContextFinder]   Found as LABEL")
                referenced_content.extend(label_results)
            else:
                logger.info(f"[ContextFinder]   Not a label, trying as CITATION...")
                # Try as citation (checks both .bib and embedded)
                citation_results = self._find_content_with_citation(ref)
                if citation_results:
                    referenced_content.extend(citation_results)
        
        return referenced_content
    
    def _extract_next_level_contexts(
        self, 
        prev_refers_to: List[Dict[str, Any]], 
        prev_referred_by: List[Dict[str, Any]], 
        include_citations: bool
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Extract the next level of contexts from previous level contexts."""
        next_refers_to = []
        next_referred_by = []
        
        # Track what we've already processed to avoid duplicates
        processed_labels = set()
        processed_citations = set()
        
        # Process contexts that referred TO the previous level
        for context in prev_refers_to:
            content_str = context.get('content', '')
            
            # Extract labels from this context
            labels = self._extract_labels_from_content(content_str)
            for label in labels:
                if label not in processed_labels:
                    processed_labels.add(label)
                    # Find what refers to this label
                    referring = self._find_all_content_referring_to_labels([label])
                    next_refers_to.extend(referring)
            
            # Extract references from this context
            if include_citations:
                refs = self._extract_references_from_content(content_str)
                for ref in refs:
                    if ref not in processed_citations:
                        processed_citations.add(ref)
                        # Find what this reference points to
                        referenced = self._find_all_references_from_content(f"\\ref{{{ref}}}")
                        next_referred_by.extend(referenced)
        
        # Process contexts that were REFERRED BY the previous level
        for context in prev_referred_by:
            content_str = context.get('content', '')
            
            # Extract labels from this context
            labels = self._extract_labels_from_content(content_str)
            for label in labels:
                if label not in processed_labels:
                    processed_labels.add(label)
                    # Find what refers to this label
                    referring = self._find_all_content_referring_to_labels([label])
                    next_refers_to.extend(referring)
            
            # Extract references from this context
            if include_citations:
                refs = self._extract_references_from_content(content_str)
                for ref in refs:
                    if ref not in processed_citations:
                        processed_citations.add(ref)
                        # Find what this reference points to
                        referenced = self._find_all_references_from_content(f"\\ref{{{ref}}}")
                        next_referred_by.extend(referenced)
        
        # Deduplicate results
        next_refers_to = self._deduplicate_contexts(next_refers_to)
        next_referred_by = self._deduplicate_contexts(next_referred_by)
        
        return next_refers_to, next_referred_by
    
    def _deduplicate_contexts(self, contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate contexts based on content hash."""
        seen = set()
        unique_contexts = []
        
        for context in contexts:
            # Create a hash of the content and label/citation_key
            content = context.get('content', '')
            label = context.get('label', context.get('citation_key', ''))
            context_hash = hash((content[:1000], label))  # Use first 1000 chars for efficiency
            
            if context_hash not in seen:
                seen.add(context_hash)
                unique_contexts.append(context)
        
        return unique_contexts
    
    def _candidate_queries(self, search_string: str, limit: int = 4) -> List[str]:
        """Progressively simpler queries to try when the verbatim string is absent."""
        seen, out = set(), []

        def add(candidate: str):
            candidate = candidate.strip(" .,:;()[]")
            key = candidate.lower()
            if len(candidate) >= 4 and key not in seen:
                seen.add(key)
                out.append(candidate)

        # Longest list/conjunction fragments first - they carry the most context.
        parts = re.split(r",| and | or |:|;|/", search_string)
        for part in sorted(parts, key=len, reverse=True):
            if part.strip().lower() != search_string.strip().lower():
                add(part)

        # Then single tokens, but only ones distinctive enough to localise safely:
        tokens = re.findall(r'[A-Za-z][\w.\-]{3,}', search_string)
        distinctive = re.compile(r'\d|[-.]|[a-z][A-Z]')
        for token in sorted(tokens, key=len, reverse=True):
            if distinctive.search(token):
                add(token)

        return out[:limit]

    def _localize_with_retry(self, search_string: str, document_url: str):
        """Localize the string, falling back to simpler forms of the query."""
        result = self.localizer.execute(
            search_string=search_string,
            document_url=document_url
        )
        if result.is_success():
            return result, search_string

        for candidate in self._candidate_queries(search_string):
            logger.info(f"[ContextFinder] Retrying localization with: {candidate!r}")
            retry = self.localizer.execute(
                search_string=candidate,
                document_url=document_url
            )
            if retry.is_success():
                logger.info(
                    f"[ContextFinder] Localized via fallback {candidate!r} "
                    f"after {search_string!r} was not found"
                )
                return retry, candidate

        return result, search_string

    def execute(
        self,
        search_string: str,
        document_url: str,
        max_depth: int = 2,
        include_citations: bool = True,
        **kwargs
    ) -> ToolResult:
        """Execute context finding."""
        
        try:
            logger.info(f"[ContextFinder] Starting context search for: '{search_string}'")
            # Step 1: Localize
            doc_type = self._detect_url_type(document_url)

            logger.info("[ContextFinder] Step 1: Localizing string")
            logger.info(f"[ContextFinder] Detected document type: {doc_type}")
            # For Semantic Scholar, check if it has arXiv source first; if not, download first, then localize
            if doc_type == 'semanticscholar':
                logger.info("[ContextFinder] Downloading Semantic Scholar metadata first")
                self.source_dir = self._download_semanticscholar_source(document_url)
                self._load_semanticscholar_content()
                
                # Check if this paper has an arXiv version
                arxiv_id = None
                if hasattr(self, 'ss_paper_data') and self.ss_paper_data:
                    external_ids = self.ss_paper_data.get('externalIds', {})
                    if 'ArXiv' in external_ids:
                        arxiv_id = external_ids['ArXiv']
                        logger.info(f"[ContextFinder] Paper has arXiv version: {arxiv_id}")
                # If arXiv version exists, use arXiv workflow instead
                if arxiv_id:
                    logger.info(f"[ContextFinder] Switching to arXiv workflow for better source quality")
                    # Clean up Semantic Scholar temp directory
                    self._cleanup_temp_dir()
                    
                    # Use arXiv URL and let the normal localization handle it
                    arxiv_url = f"https://arxiv.org/abs/{arxiv_id}"
                    document_url = arxiv_url
                    doc_type = 'arxiv'
                    
                    # Now use normal localization with arXiv URL
                    if not self.localizer:
                        raise RuntimeError("ContextFinder not available")
                    
                    localization_result, search_string = self._localize_with_retry(
                        search_string, arxiv_url
                    )
                    
                    if not localization_result.is_success():
                        return ToolResult(
                            status=ToolStatus.ERROR,
                            data={"error": "Failed to localize string in arXiv version"},
                            formatted="[Context Finder Error] Could not localize the search string in the arXiv document.",
                            metadata={"timestamp": datetime.now().isoformat()}
                        )
                    
                    localization_data = localization_result.data
                    localized_content_str = ""
                    
                    loc_results = localization_data.get("localization", {}).get("results", [])
                    if loc_results:
                        if "extraction" in loc_results[0]:
                            localized_content_str = loc_results[0]["extraction"].get("content", "")
                        elif "context" in loc_results[0]:
                            localized_content_str = loc_results[0].get("context", "")
                    
                    logger.info(f"[ContextFinder] Localized content length: {len(localized_content_str)} chars")
                    # Extract labels from localized content
                    localized_labels = self._extract_labels_from_content(localized_content_str)
                    
                else:
                    # No arXiv version, use Semantic Scholar PDF workflow
                    logger.info("[ContextFinder] No arXiv version available, using Semantic Scholar PDF")
                    # Try to find in the loaded content
                    ss_result = self._find_in_semanticscholar_content(search_string)
                    if not ss_result:
                        return ToolResult(
                            status=ToolStatus.ERROR,
                            data={"error": f"String '{search_string}' not found in Semantic Scholar paper"},
                            formatted=f"[Context Finder Error] Could not find '{search_string}' in the paper abstract or metadata.",
                            metadata={"timestamp": datetime.now().isoformat()}
                        )
                    
                    localized_content_str = ss_result.get('content', '')
                    localized_labels = set()  # Semantic Scholar PDF doesn't have LaTeX labels
                    
                    # Create a mock localization_data structure for consistency
                    localization_data = {
                        "localization": {
                            "results": [{
                                "extraction": {
                                    "content": localized_content_str,
                                    "matched_text": ss_result.get('matched_text', search_string),
                                    "source": ss_result.get('source', 'unknown')
                                },
                                "context": localized_content_str,
                                "position": ss_result.get('position', 0)
                            }]
                        },
                        "search_string": search_string,
                        "document_type": "semanticscholar"
                    }
                
            else:
                # For other document types, use normal localization
                if not self.localizer:
                    raise RuntimeError("ContextFinder not available")
                
                localization_result, search_string = self._localize_with_retry(
                    search_string, document_url
                )
                
                if not localization_result.is_success():
                    return ToolResult(
                        status=ToolStatus.ERROR,
                        data={"error": "Failed to localize string"},
                        formatted="[Context Finder Error] Could not localize the search string in the document.",
                        metadata={"timestamp": datetime.now().isoformat()}
                    )
                
                localization_data = localization_result.data
                localized_content_str = ""
                
                loc_results = localization_data.get("localization", {}).get("results", [])
                if loc_results:
                    if "extraction" in loc_results[0]:
                        localized_content_str = loc_results[0]["extraction"].get("content", "")
                    elif "context" in loc_results[0]:
                        localized_content_str = loc_results[0].get("context", "")
                
                logger.info(f"[ContextFinder] Localized content length: {len(localized_content_str)} chars")
                # Extract labels from localized content
                localized_labels = self._extract_labels_from_content(localized_content_str)
            
            # Step 2: Load document (if not already loaded)
            if doc_type != 'semanticscholar':  # Already loaded for Semantic Scholar
                if doc_type == 'arxiv':
                    logger.info("[ContextFinder] Step 2: Downloading arXiv source")
                    self.source_dir = self._download_arxiv_source(document_url)
                    self._load_all_latex_files(self.source_dir)
                elif doc_type == 'latex':
                    logger.info("[ContextFinder] Step 2: Loading LaTeX files")
                    if os.path.isdir(document_url):
                        self._load_all_latex_files(document_url)
                    else:
                        self.latex_files = [document_url]
                        content = self._read_file_safe(document_url)
                        self.all_content[document_url] = content
                        self._parse_embedded_bibliography(content, document_url)
                else:
                    return ToolResult(
                        status=ToolStatus.SUCCESS,
                        data={
                            "search_string": search_string,
                            "document_url": document_url,
                            "document_type": doc_type,
                            "localization": localization_data,
                            "localized_content": localized_content_str,
                            "warning": "Deep context finding only available for LaTeX/arXiv documents"
                        },
                        formatted=self._format_limited_results(localization_data, doc_type),
                        metadata={"doc_type": doc_type, "timestamp": datetime.now().isoformat()}
                    )
            
            # Step 3: Extract labels
            logger.info("[ContextFinder] Step 3: Extracting labels")
            localized_labels = self._extract_labels_from_content(localized_content_str)
            
            logger.info(f"[ContextFinder] Found {len(localized_labels)} labels")
            # Step 4: Find referring content
            logger.info("[ContextFinder] Step 4: Finding content that refers to localized part")
            refers_to_localized = []
            if localized_labels:
                refers_to_localized = self._find_all_content_referring_to_labels(localized_labels)
            
            logger.info(f"[ContextFinder] Found {len(refers_to_localized)} pieces of referring content")
            # Step 5: Find referenced content
            logger.info("[ContextFinder] Step 5: Finding content referred by localized part")
            referred_by_localized = []
            if include_citations:
                referred_by_localized = self._find_all_references_from_content(localized_content_str)
            
            logger.info(f"[ContextFinder] Found {len(referred_by_localized)} referenced contexts")
            # Step 6: Multi-level extraction (if max_depth > 1)
            all_refers_to = [refers_to_localized]  # List of lists by level
            all_referred_by = [referred_by_localized]
            
            if max_depth > 1:
                logger.info(f"[ContextFinder] Step 6: Extracting deeper levels (up to depth {max_depth})")
                # Extract deeper levels
                for depth in range(2, max_depth + 1):
                    logger.info(f"[ContextFinder] Extracting Level {depth} contexts...")
                    prev_refers_to = all_refers_to[-1]
                    prev_referred_by = all_referred_by[-1]
                    
                    # Extract next level from both directions
                    current_refers_to, current_referred_by = self._extract_next_level_contexts(
                        prev_refers_to, prev_referred_by, include_citations
                    )
                    
                    all_refers_to.append(current_refers_to)
                    all_referred_by.append(current_referred_by)
                    
                    logger.info(f"[ContextFinder] Level {depth}: {len(current_refers_to)} referring, {len(current_referred_by)} referenced")
                    # Stop if no new contexts found
                    if not current_refers_to and not current_referred_by:
                        logger.info(f"[ContextFinder] No new contexts at depth {depth}, stopping")
                        break
            
            # Flatten all levels for backward compatibility
            refers_to_flattened = [item for level in all_refers_to for item in level]
            referred_by_flattened = [item for level in all_referred_by for item in level]
            
            result_data = {
                "search_string": search_string,
                "document_url": document_url,
                "document_type": doc_type,
                "max_depth": max_depth,
                "localization": localization_data,
                "localized_content": localized_content_str,
                "localized_labels": list(localized_labels),
                "refers_to_localized": refers_to_flattened,
                "referred_by_localized": referred_by_flattened,
                "refers_to_by_level": all_refers_to,
                "referred_by_by_level": all_referred_by,
                "total_contexts_found": len(refers_to_flattened) + len(referred_by_flattened)
            }
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=result_data,
                formatted=self.format_results_for_agent(result_data),
                metadata={
                    "doc_type": doc_type,
                    "max_depth": max_depth,
                    "num_files_searched": len(self.latex_files),
                    "num_bib_files": sum(1 for f in self.latex_files if f.endswith('.bib')),
                    "num_embedded_citations": len(self.embedded_bibliography),
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        except Exception as e:
            logger.error(f"[ContextFinder] Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return ToolResult(
                status=ToolStatus.ERROR,
                data={"error": str(e)},
                formatted=f"[Context Finder Error] Error finding contexts: {str(e)}",
                metadata={"timestamp": datetime.now().isoformat()}
            )
        
        finally:
            self._cleanup_temp_dir()
    
    def _format_limited_results(self, localization_data: Dict[str, Any], doc_type: str) -> str:
        formatted = []
        formatted.append("=" * 80)
        formatted.append("CONTEXT FINDER - LIMITED RESULTS")
        formatted.append("=" * 80)
        formatted.append(f"Document Type: {doc_type.upper()}")
        formatted.append("⚠️  Deep context finding only available for LaTeX/arXiv documents")
        formatted.append("=" * 80)
        return "\n".join(formatted)
    
    def format_results_for_agent(self, data: Dict[str, Any], max_limit: int = 1000) -> str:
        """Format results for agent consumption."""
        formatted = []
        
        formatted.append("**CONTEXT FINDER RESULTS**\n")
        formatted.append(f"Search String: \"{data['search_string']}\"")
        formatted.append(f"Document: {data['document_url']}")
        formatted.append(f"Maximum Depth: {data.get('max_depth', 1)}")

        doc_type = data['document_type']
        formatted.append(f"Document Type: {doc_type.upper()}")

        # Add Semantic Scholar specific info
        if doc_type == 'semanticscholar' and hasattr(self, 'ss_paper_data') and self.ss_paper_data:
            formatted.append(f"Paper ID: {self.ss_paper_data.get('paperId', 'N/A')}")
            formatted.append(f"Citations: {self.ss_paper_data.get('citationCount', 0)}")
            formatted.append(f"References: {self.ss_paper_data.get('referenceCount', 0)}")        
        
        num_refers_to = len(data.get('refers_to_localized', []))
        num_referred_by = len(data.get('referred_by_localized', []))
        total_contexts = data.get('total_contexts_found', 0)
        
        formatted.append("---")
        formatted.append(f"Total Related Contexts Found: {total_contexts}")
        formatted.append(f"  * Content referring TO localized part (all levels): {num_refers_to}")
        formatted.append(f"  * Content referred BY localized part (all levels): {num_referred_by}")
        
        # Show per-level breakdown if multi-level
        refers_to_by_level = data.get('refers_to_by_level', [])
        referred_by_by_level = data.get('referred_by_by_level', [])
        if len(refers_to_by_level) > 1 or len(referred_by_by_level) > 1:
            formatted.append("")
            formatted.append("Per-Level Breakdown:")
            max_levels = max(len(refers_to_by_level), len(referred_by_by_level))
            for level in range(max_levels):
                num_refers = len(refers_to_by_level[level]) if level < len(refers_to_by_level) else 0
                num_referred = len(referred_by_by_level[level]) if level < len(referred_by_by_level) else 0
                formatted.append(f"  Level {level + 1}: {num_refers} referring, {num_referred} referenced")
        
        formatted.append("")
        
        formatted.append("---\n\n1. LOCALIZED CONTENT (Original Target)")
        formatted.append("")

        labels = data.get('localized_labels', [])
        if labels:
            formatted.append(f"Labels: {', '.join(labels)}")
        
        localized_content = data.get('localized_content', '')

        # NOT truncate localized original content
        if localized_content:
            formatted.append(localized_content)
        formatted.append("")

        """
        # Truncate
        if localized_content:
            formatted.append(localized_content[:max_limit] if len(localized_content) > max_limit else localized_content)
            if len(localized_content) > max_limit:
                formatted.append(f"\n... (truncated, total length: {len(localized_content)} chars)")
        formatted.append("")
        """
        
        formatted.append("---\n\n2. CONTENT REFERRING TO THE LOCALIZED PART (ALL LEVELS)")
        formatted.append("")
        
        # Group by level if available
        if refers_to_by_level and len(refers_to_by_level) > 1:
            for level_idx, level_contexts in enumerate(refers_to_by_level, 1):
                if level_contexts:
                    formatted.append(f"--- LEVEL {level_idx} ({len(level_contexts)} contexts) ---")
                    formatted.append("")
                    for idx, item in enumerate(level_contexts, 1):
                        formatted.append(f"[L{level_idx}-{idx}] Reference")
                        formatted.append(f"File: {item.get('file', 'Unknown')}")
                        formatted.append(f"Refers to label: {item.get('refers_to_label', 'N/A')}")
                        formatted.append(f"Type: {item.get('type', 'Unknown').upper()}")
                        formatted.append("Content:")
                        formatted.append("---")
                        content = item.get('content', '').replace(localized_content, '')
                        if len(content) == 0:
                            content = 'Same as "1. LOCALIZED CONTENT (Original Target)"'
                        content_lines = (content[:max_limit] if len(content) > max_limit else content).split('\n')
                        for line in content_lines:
                            formatted.append(f"  {line}")
                        if len(content) > max_limit:
                            formatted.append(f"  ... (truncated)")
                        formatted.append("---")
                        formatted.append("")
        else:
            # Fallback to flat list
            refers_to = data.get('refers_to_localized', [])
            if refers_to:
                formatted.append(f"Found {len(refers_to)} piece(s):")
                formatted.append("")
                for idx, item in enumerate(refers_to, 1):
                    formatted.append(f"--- Reference {idx} ---")
                    formatted.append(f"File: {item.get('file', 'Unknown')}")
                    formatted.append(f"Refers to label: {item.get('refers_to_label', 'N/A')}")
                    formatted.append(f"Type: {item.get('type', 'Unknown').upper()}")
                    formatted.append("Content:")
                    formatted.append("---")
                    content = item.get('content', '')
                    formatted.append(content[:max_limit] if len(content) > max_limit else content)
                    if len(content) > max_limit:
                        formatted.append(f"\n... (truncated)")
                    formatted.append("---")
                    formatted.append("")
            else:
                formatted.append("(No content found that refers to the localized part)")
                formatted.append("")
        
        formatted.append("---\n\n3. CONTENT REFERRED BY THE LOCALIZED PART (ALL LEVELS)")
        formatted.append("")
        
        # Group by level if available
        if referred_by_by_level and len(referred_by_by_level) > 1:
            for level_idx, level_contexts in enumerate(referred_by_by_level, 1):
                if level_contexts:
                    formatted.append(f"--- LEVEL {level_idx} ({len(level_contexts)} contexts) ---")
                    formatted.append("")
                    for idx, item in enumerate(level_contexts, 1):
                        formatted.append(f"[L{level_idx}-{idx}] Referenced Content")
                        
                        if 'label' in item:
                            formatted.append(f"    * Label: {item.get('label')}")
                        if 'citation_key' in item:
                            formatted.append(f"    * Citation Key: {item.get('citation_key')}")
                        if 'title' in item:
                            formatted.append(f"    * Title: {item.get('title')}")
                        if 'author' in item:
                            formatted.append(f"    * Author: {item.get('author')}")
                        if 'year' in item:
                            formatted.append(f"    * Year: {item.get('year')}")
                        if 'venue' in item:
                            formatted.append(f"    * Venue: {item.get('venue')}")
                        if 'url' in item:
                            formatted.append(f"    * URL: {item.get('url')}")
                        formatted.append("")
        else:
            # Fallback to flat list
            referred_by = data.get('referred_by_localized', [])
            if referred_by:
                formatted.append(f"Found {len(referred_by)} piece(s):")
                formatted.append("")
                for idx, item in enumerate(referred_by, 1):
                    formatted.append(f"[{idx}] Referenced Content {idx}")
                    
                    if 'label' in item:
                        formatted.append(f"    * Label: {item.get('label')}")
                    if 'citation_key' in item:
                        formatted.append(f"    * Citation Key: {item.get('citation_key')}")
                    if 'title' in item:
                        formatted.append(f"    * Title: {item.get('title')}")
                    if 'author' in item:
                        formatted.append(f"    * Author: {item.get('author')}")
                    if 'year' in item:
                        formatted.append(f"    * Year: {item.get('year')}")
                    if 'venue' in item:
                        formatted.append(f"    * Venue: {item.get('venue')}")
                    if 'url' in item:
                        formatted.append(f"    * URL: {item.get('url')}")
                    formatted.append("")
            else:
                formatted.append("(No content found that is referenced by the localized part)")
                formatted.append("")
        
        return "\n".join(formatted).strip()

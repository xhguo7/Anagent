"""
Abstract collector tool.
"""

import os
import re
import requests
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
from urllib.parse import urlparse
import xml.etree.ElementTree as ET

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


class AbstractCollector(BaseTool):
    """Collects abstracts for a list of papers from various sources."""
    
    def __init__(self, **kwargs):
        super().__init__(
            name="abstract_collector",
            description="Collect paper abstracts from multiple sources (arXiv, PubMed, Semantic Scholar, etc.)",
            **kwargs
        )
        
        # Rate limiting
        self.last_request_time = {}
        self.rate_limit_delay = {
            'arxiv': 3.0,  # 3 seconds between requests
            'pubmed': 0.34,  # ~3 requests per second
            'semantic_scholar': 1.0,  # 1 second between requests (free tier)
            'crossref': 1.0  # 1 second between requests
        }
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "papers": {
                    "type": "array",
                    "description": "List of paper dictionaries with title (required), author, year, url (optional)",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string", "description": "Paper title (required)"},
                            "author": {"type": "string", "description": "Paper author (optional)"},
                            "year": {"type": "string", "description": "Publication year (optional)"},
                            "url": {"type": "string", "description": "Paper URL (optional)"}
                        },
                        "required": ["title"]
                    }
                }
            },
            "required": ["papers"]
        }
    
    def _rate_limit(self, source: str):
        """Enforce rate limiting for API calls."""
        current_time = time.time()
        if source in self.last_request_time:
            elapsed = current_time - self.last_request_time[source]
            if elapsed < self.rate_limit_delay[source]:
                sleep_time = self.rate_limit_delay[source] - elapsed
                logger.info(f"[AbstractCollector] Rate limiting {source}: sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)
        self.last_request_time[source] = time.time()
    
    def _detect_url_type(self, url: str) -> str:
        """Detect the type of URL."""
        if not url:
            return 'none'
        
        url_lower = url.lower()
        if 'arxiv.org' in url_lower:
            return 'arxiv'
        if 'pubmed' in url_lower or 'ncbi.nlm.nih.gov' in url_lower:
            return 'pubmed'
        if url_lower.endswith('.pdf') or '/pdf/' in url_lower:
            return 'pdf'
        if os.path.exists(url) and url_lower.endswith('.pdf'):
            return 'local_pdf'
        if url_lower.startswith('http'):
            return 'html'
        return 'unknown'
    
    def _extract_arxiv_id(self, url: str) -> Optional[str]:
        """Extract arXiv ID from URL."""
        patterns = [
            r'arxiv\.org/abs/(\d+\.\d+)',
            r'arxiv\.org/pdf/(\d+\.\d+)',
            r'(\d{4}\.\d{4,5})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    def _extract_abstract_from_arxiv_url(self, url: str) -> Optional[str]:
        """Extract abstract from arXiv URL."""
        try:
            arxiv_id = self._extract_arxiv_id(url)
            if not arxiv_id:
                return None
            
            # Use arXiv API
            self._rate_limit('arxiv')
            api_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
            
            logger.info(f"[AbstractCollector] Fetching arXiv abstract from API: {arxiv_id}")
            response = requests.get(api_url, timeout=30)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.content)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            entry = root.find('atom:entry', ns)
            if entry is not None:
                summary = entry.find('atom:summary', ns)
                if summary is not None and summary.text:
                    abstract = summary.text.strip()
                    logger.info(f"[AbstractCollector] Successfully extracted arXiv abstract")
                    return abstract
            
            return None
            
        except Exception as e:
            logger.error(f"[AbstractCollector] Error extracting from arXiv URL: {e}")
            return None
    
    def _extract_abstract_from_pubmed_url(self, url: str) -> Optional[str]:
        """Extract abstract from PubMed URL."""
        try:
            # Extract PMID from URL
            pmid_match = re.search(r'(\d{7,8})', url)
            if not pmid_match:
                return None
            
            pmid = pmid_match.group(1)
            
            # Use PubMed E-utilities API
            self._rate_limit('pubmed')
            api_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pmid}&retmode=xml"
            
            logger.info(f"[AbstractCollector] Fetching PubMed abstract: PMID {pmid}")
            response = requests.get(api_url, timeout=30)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.content)
            
            # Find abstract
            abstract_elem = root.find('.//Abstract/AbstractText')
            if abstract_elem is not None and abstract_elem.text:
                abstract = abstract_elem.text.strip()
                logger.info(f"[AbstractCollector] Successfully extracted PubMed abstract")
                return abstract
            
            return None
            
        except Exception as e:
            logger.error(f"[AbstractCollector] Error extracting from PubMed URL: {e}")
            return None
    
    def _extract_abstract_from_pdf_url(self, url: str) -> Optional[str]:
        """Extract abstract from PDF URL."""
        if not HAS_PDF:
            logger.warning("[AbstractCollector] PyPDF2 not available, cannot extract from PDF")
            return None
        
        try:
            logger.info(f"[AbstractCollector] Downloading PDF from URL")
            response = requests.get(url, timeout=60, headers={
                'User-Agent': 'Mozilla/5.0 (AbstractCollector/1.0)'
            })
            response.raise_for_status()
            
            # Save to temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(response.content)
                tmp_path = tmp_file.name
            
            try:
                abstract = self._extract_abstract_from_local_pdf(tmp_path)
                return abstract
            finally:
                os.unlink(tmp_path)
                
        except Exception as e:
            logger.error(f"[AbstractCollector] Error extracting from PDF URL: {e}")
            return None
    
    def _extract_abstract_from_local_pdf(self, filepath: str) -> Optional[str]:
        """Extract abstract from local PDF file."""
        if not HAS_PDF:
            return None
        
        try:
            logger.info(f"[AbstractCollector] Extracting from local PDF: {filepath}")
            with open(filepath, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                
                # Extract text from first few pages
                text = ""
                for i in range(min(3, len(reader.pages))):
                    text += reader.pages[i].extract_text()
                
                # Look for abstract section
                abstract_patterns = [
                    r'Abstract[\s\n]+(.*?)(?:\n\n|Introduction|Keywords|1\s+Introduction)',
                    r'ABSTRACT[\s\n]+(.*?)(?:\n\n|INTRODUCTION|KEYWORDS|1\s+INTRODUCTION)',
                ]
                
                for pattern in abstract_patterns:
                    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
                    if match:
                        abstract = match.group(1).strip()
                        # Clean up
                        abstract = re.sub(r'\s+', ' ', abstract)
                        if len(abstract) > 100:  # Reasonable abstract length
                            logger.info(f"[AbstractCollector] Successfully extracted abstract from PDF")
                            return abstract
                
                return None
                
        except Exception as e:
            logger.error(f"[AbstractCollector] Error extracting from local PDF: {e}")
            return None
    
    def _extract_abstract_from_html(self, url: str) -> Optional[str]:
        """Extract abstract from HTML page."""
        if not HAS_BS4:
            logger.warning("[AbstractCollector] BeautifulSoup not available, cannot extract from HTML")
            return None
        
        try:
            logger.info(f"[AbstractCollector] Fetching HTML page")
            response = requests.get(url, timeout=30, headers={
                'User-Agent': 'Mozilla/5.0 (AbstractCollector/1.0)'
            })
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Try common abstract selectors
            abstract_selectors = [
                {'name': 'meta', 'attrs': {'name': 'description'}},
                {'name': 'meta', 'attrs': {'property': 'og:description'}},
                {'name': 'div', 'attrs': {'class': re.compile(r'abstract', re.I)}},
                {'name': 'section', 'attrs': {'class': re.compile(r'abstract', re.I)}},
                {'name': 'p', 'attrs': {'class': re.compile(r'abstract', re.I)}},
            ]
            
            for selector in abstract_selectors:
                element = soup.find(**selector)
                if element:
                    if element.name == 'meta':
                        abstract = element.get('content', '').strip()
                    else:
                        abstract = element.get_text().strip()
                    
                    # Clean up
                    abstract = re.sub(r'\s+', ' ', abstract)
                    if len(abstract) > 100:
                        logger.info(f"[AbstractCollector] Successfully extracted abstract from HTML")
                        return abstract
            
            return None
            
        except Exception as e:
            logger.error(f"[AbstractCollector] Error extracting from HTML: {e}")
            return None
    
    def _search_arxiv_by_title(self, title: str) -> Optional[str]:
        """Search arXiv by title and extract abstract."""
        try:
            self._rate_limit('arxiv')
            
            # Clean title for search
            search_title = re.sub(r'[^\w\s]', ' ', title)
            search_title = re.sub(r'\s+', '+', search_title.strip())
            
            api_url = f"http://export.arxiv.org/api/query?search_query=ti:{search_title}&max_results=1"
            
            logger.info(f"[AbstractCollector] Searching arXiv for: {title}")
            response = requests.get(api_url, timeout=30)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.content)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            entry = root.find('atom:entry', ns)
            if entry is not None:
                summary = entry.find('atom:summary', ns)
                if summary is not None and summary.text:
                    abstract = summary.text.strip()
                    logger.info(f"[AbstractCollector] Found abstract on arXiv")
                    return abstract
            
            return None
            
        except Exception as e:
            logger.error(f"[AbstractCollector] Error searching arXiv: {e}")
            return None
    
    def _search_pubmed_by_title(self, title: str) -> Optional[str]:
        """Search PubMed by title and extract abstract."""
        try:
            self._rate_limit('pubmed')
            
            # First, search for PMID
            search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            params = {
                'db': 'pubmed',
                'term': title,
                'retmax': 1,
                'retmode': 'xml'
            }
            
            logger.info(f"[AbstractCollector] Searching PubMed for: {title}")
            response = requests.get(search_url, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse search results
            root = ET.fromstring(response.content)
            id_elem = root.find('.//Id')
            
            if id_elem is not None:
                pmid = id_elem.text
                
                # Fetch abstract
                self._rate_limit('pubmed')
                fetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pmid}&retmode=xml"
                
                response = requests.get(fetch_url, timeout=30)
                response.raise_for_status()
                
                root = ET.fromstring(response.content)
                abstract_elem = root.find('.//Abstract/AbstractText')
                
                if abstract_elem is not None and abstract_elem.text:
                    abstract = abstract_elem.text.strip()
                    logger.info(f"[AbstractCollector] Found abstract on PubMed")
                    return abstract
            
            return None
            
        except Exception as e:
            logger.error(f"[AbstractCollector] Error searching PubMed: {e}")
            return None
    
    def _search_semantic_scholar(self, title: str) -> Optional[str]:
        """Search Semantic Scholar by title and extract abstract."""
        try:
            self._rate_limit('semantic_scholar')
            
            # Use Semantic Scholar API
            api_url = "https://api.semanticscholar.org/graph/v1/paper/search"
            params = {
                'query': title,
                'limit': 1,
                'fields': 'title,abstract'
            }
            
            logger.info(f"[AbstractCollector] Searching Semantic Scholar for: {title}")
            response = requests.get(api_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('data') and len(data['data']) > 0:
                paper = data['data'][0]
                abstract = paper.get('abstract')
                if abstract:
                    logger.info(f"[AbstractCollector] Found abstract on Semantic Scholar")
                    return abstract
            
            return None
            
        except Exception as e:
            logger.error(f"[AbstractCollector] Error searching Semantic Scholar: {e}")
            return None
    
    def _search_crossref(self, title: str) -> Optional[str]:
        """Search CrossRef by title and extract abstract."""
        try:
            self._rate_limit('crossref')
            
            # Use CrossRef API
            api_url = "https://api.crossref.org/works"
            params = {
                'query.title': title,
                'rows': 1
            }
            
            headers = {
                'User-Agent': 'AbstractCollector/1.0 (mailto:research@example.com)'
            }
            
            logger.info(f"[AbstractCollector] Searching CrossRef for: {title}")
            response = requests.get(api_url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('message', {}).get('items'):
                item = data['message']['items'][0]
                abstract = item.get('abstract')
                if abstract:
                    # CrossRef may return abstract with XML tags, clean it
                    if HAS_BS4:
                        abstract = BeautifulSoup(abstract, 'html.parser').get_text()
                    abstract = re.sub(r'\s+', ' ', abstract).strip()
                    logger.info(f"[AbstractCollector] Found abstract on CrossRef")
                    return abstract
            
            return None
            
        except Exception as e:
            logger.error(f"[AbstractCollector] Error searching CrossRef: {e}")
            return None
    
    def _extract_abstract_for_paper(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """Extract abstract for a single paper."""
        title = paper.get('title', '')
        url = paper.get('url', '')
        
        result = {
            'title': title,
            'author': paper.get('author', ''),
            'year': paper.get('year', ''),
            'url': url,
            'abstract': None,
            'source': None,
            'error': None
        }
        
        logger.info(f"[AbstractCollector] Processing paper: {title}")
        # Step 1: Try URL if provided
        if url:
            url_type = self._detect_url_type(url)
            
            if url_type == 'arxiv':
                abstract = self._extract_abstract_from_arxiv_url(url)
                if abstract:
                    result['abstract'] = abstract
                    result['source'] = 'arxiv_url'
                    return result
            
            elif url_type == 'pubmed':
                abstract = self._extract_abstract_from_pubmed_url(url)
                if abstract:
                    result['abstract'] = abstract
                    result['source'] = 'pubmed_url'
                    return result
            
            elif url_type == 'pdf':
                abstract = self._extract_abstract_from_pdf_url(url)
                if abstract:
                    result['abstract'] = abstract
                    result['source'] = 'pdf_url'
                    return result
            
            elif url_type == 'local_pdf':
                abstract = self._extract_abstract_from_local_pdf(url)
                if abstract:
                    result['abstract'] = abstract
                    result['source'] = 'local_pdf'
                    return result
            
            elif url_type == 'html':
                abstract = self._extract_abstract_from_html(url)
                if abstract:
                    result['abstract'] = abstract
                    result['source'] = 'html'
                    return result
        
        # Step 2: Search by title if URL didn't work or wasn't provided
        if not result['abstract'] and title:
            # Try arXiv
            abstract = self._search_arxiv_by_title(title)
            if abstract:
                result['abstract'] = abstract
                result['source'] = 'arxiv_search'
                return result
            
            # Try PubMed
            abstract = self._search_pubmed_by_title(title)
            if abstract:
                result['abstract'] = abstract
                result['source'] = 'pubmed_search'
                return result
            
            # Try Semantic Scholar
            abstract = self._search_semantic_scholar(title)
            if abstract:
                result['abstract'] = abstract
                result['source'] = 'semantic_scholar'
                return result
            
            # Try CrossRef as last resort
            abstract = self._search_crossref(title)
            if abstract:
                result['abstract'] = abstract
                result['source'] = 'crossref'
                return result
        
        # If we got here, we couldn't find the abstract
        result['error'] = 'Abstract not found in any source'
        
        return result
    
    def execute(self, papers: List[Dict[str, Any]], **kwargs) -> ToolResult:
        """Execute abstract collection for a list of papers."""
        try:
            logger.info(f"[AbstractCollector] Starting collection for {len(papers)} papers")
            results = []
            
            for idx, paper in enumerate(papers, 1):
                logger.info(f"[AbstractCollector] Processing paper {idx}/{len(papers)}")
                if not isinstance(paper, dict) or 'title' not in paper:
                    results.append({
                        'error': 'Invalid paper format: must be dict with "title" field',
                        'paper': paper
                    })
                    continue
                
                result = self._extract_abstract_for_paper(paper)
                results.append(result)
            
            # Count successes and failures
            success_count = sum(1 for r in results if r.get('abstract'))
            failure_count = len(results) - success_count
            
            logger.info(f"[AbstractCollector] Collection complete: {success_count} successes, {failure_count} failures")
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    'papers': results,
                    'total': len(results),
                    'success': success_count,
                    'failure': failure_count
                },
                formatted=self.format_results_for_agent({
                    'papers': results,
                    'total': len(results),
                    'success': success_count,
                    'failure': failure_count
                }),
                metadata={
                    'timestamp': datetime.now().isoformat()
                }
            )
        
        except Exception as e:
            logger.error(f"[AbstractCollector] Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return ToolResult(
                status=ToolStatus.ERROR,
                data={"error": str(e)},
                formatted=f"[Abstract Collector Error] Error collecting abstracts: {str(e)}",
                metadata={"timestamp": datetime.now().isoformat()}
            )
    
    def format_results_for_agent(self, data: Dict[str, Any]) -> str:
        """Format results for agent consumption."""
        formatted = []
        
        formatted.append("**ABSTRACT COLLECTOR RESULTS**")
        
        total = data.get('total', 0)
        success = data.get('success', 0)
        failure = data.get('failure', 0)
        
        formatted.append(f"\nTotal Papers: {total}")
        formatted.append(f"Successful: {success}")
        formatted.append(f"Failed: {failure}")
        formatted.append(f"Success Rate: {(success/total*100):.1f}%" if total > 0 else "N/A")
        
        formatted.append("\n---\n\nCOLLECTED ABSTRACTS")
        formatted.append("")
        
        papers = data.get('papers', [])
        
        for idx, paper in enumerate(papers, 1):
            formatted.append(f"[{idx}] {paper.get('title', 'Unknown Title')}")
            
            if paper.get('author'):
                formatted.append(f"Author: {paper['author']}")
            if paper.get('year'):
                formatted.append(f"Year: {paper['year']}")
            if paper.get('url'):
                formatted.append(f"URL: {paper['url']}")
            
            if paper.get('abstract'):
                formatted.append("ABSTRACT:")
                formatted.append(paper['abstract'].replace("\n ", " ").replace("\n", "").strip())
            else:
                error_msg = paper.get('error', 'Unknown error')
                formatted.append(f"❌ Abstract not found: {error_msg}")
            
            formatted.append("")
        
        return "\n".join(formatted).strip()

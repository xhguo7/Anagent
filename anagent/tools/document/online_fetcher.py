"""
Online document fetcher tool.
"""

import os
import io
import re
import time
import tarfile
import gzip
import requests
from typing import Dict, Any, List, Optional, Union, Tuple
from urllib.parse import urlparse


from utils.logger import setup_logger
logger = setup_logger(__name__)

from anagent.tools.base import BaseTool, ToolResult, ToolStatus
from anagent.tools.document.pdf_parser import PDFParserTool
from anagent.tools.document.xml_parser import XMLParserTool

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    logger.error("Warning: BeautifulSoup not installed. Install with: pip install beautifulsoup4")


class OnlineDocumentFetcher(BaseTool):
    """Online document fetcher for papers and web content."""
    
    def __init__(
        self,
        parse_format: str = "full",  # "full" or "structure"
        timeout: int = 60,
        user_agent: str = None,
        semantic_scholar_api_key: str = None,
        **kwargs
    ):
        """Initialize online document fetcher."""
        super().__init__(
            name="online_document_fetcher",
            description="Fetch and parse documents from online sources (arXiv, Semantic Scholar, PubMed, URLs)",
            **kwargs
        )
        
        self.parse_format = parse_format
        self.timeout = timeout
        self.user_agent = user_agent or "Mozilla/5.0 (compatible; DocumentFetcher/1.0)"
        
        # Semantic Scholar API
        self.semantic_scholar_api_key = semantic_scholar_api_key
        self.s2_api_base = "https://api.semanticscholar.org/graph/v1"
        
        # Initialize parsers
        try:
            self.pdf_parser = PDFParserTool(parse_format=parse_format)
        except ImportError:
            self.pdf_parser = None
            logger.warning("Warning: PDF parser not available")
        self.xml_parser = XMLParserTool(parse_mode="pmc", parse_format=parse_format)
        
        # Rate limiting
        self.last_request_time = 0
        self.request_delay = 1.0  # 1 second between requests
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get parameter schema."""
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to fetch (arXiv PDF/XML, Semantic Scholar paper, PubMed XML, or general URL)"
                },
                "source_type": {
                    "type": "string",
                    "description": "Source type: 'arxiv', 'pubmed', 'url', 'auto' (default: auto)"
                },
                "format": {
                    "type": "string",
                    "description": "Preferred format: 'pdf', 'xml', 'html', 'auto' (default: auto)"
                },
                "extract_content": {
                    "type": "boolean",
                    "description": "Whether to extract and parse content (default: True)"
                },
                "query": {
                    "type": "string",
                    "description": "Search query for Semantic Scholar paper search"
                },
                "paper_id": {
                    "type": "string",
                    "description": "Semantic Scholar paper ID or DOI for direct paper lookup"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results for Semantic Scholar search (default: 10, max: 100)"
                }
            },
            "required": ["url"]
        }
    
    def _rate_limit(self):
        """Enforce rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.request_delay:
            time.sleep(self.request_delay - time_since_last)
        self.last_request_time = time.time()
    
    def execute(
        self,
        url: str,
        source_type: str = "auto",
        format: str = "auto",
        extract_content: bool = True,
        query: str = None,
        paper_id: str = None,
        limit: int = 10,
        **kwargs
    ) -> ToolResult:
        """Execute online document fetching."""
        if url:
            logger.info(f"[OnlineFetcher] Fetching from: {url}")
        elif query:
            logger.info(f"[OnlineFetcher] Searching Semantic Scholar: {query}")
        elif paper_id:
            logger.info(f"[OnlineFetcher] Fetching Semantic Scholar paper: {paper_id}")
        # Handle Semantic Scholar search/fetch without URL
        if source_type == "semantic_scholar" or query or paper_id:
            if query:
                return self._search_semantic_scholar(query, limit)
            elif paper_id:
                return self._fetch_semantic_scholar_paper(paper_id, extract_content)
            elif url:
                return self._fetch_semantic_scholar_url(url, extract_content)
        
        # Validate URL for other source types
        if not url:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                formatted=f"[Semantic Scholar Error] URL is required for non-Semantic Scholar sources",
                error="URL is required for non-Semantic Scholar sources"
            )
        
        # Validate parameters
        is_valid, error_msg = self.validate_parameters({"url": url})
        if not is_valid:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                formatted=f"[Semantic Scholar Error] {error_msg}",
                error=error_msg
            )
        
        try:
            # Auto-detect source type if needed
            if source_type == "auto":
                source_type = self._detect_source_type(url)
            
            logger.info(f"[OnlineFetcher] Detected source type: {source_type}")
            # Fetch based on source type
            if source_type == "arxiv":
                result = self._fetch_arxiv(url, format, extract_content)
            elif source_type == "semantic_scholar":
                result = self._fetch_semantic_scholar_url(url, extract_content)
            elif source_type == "xml":
                # Parse XML and wrap in our format
                xml_result = self.xml_parser.execute(url)
                if xml_result.is_success():
                    result = ToolResult(
                        status=ToolStatus.SUCCESS,
                        data={
                            "url": url,
                            "source": "xml",
                            "content_type": "xml",
                            "parsed_content": xml_result.data,
                            "raw_content": None
                        },
                        formatted=str(xml_result.data),
                        metadata=xml_result.metadata
                    )
                else:
                    result = xml_result
            elif source_type == "pdf":
                result = self.pdf_parser.execute(url)
            else:
                # All other source types (pubmed, xml, url, etc.) are just URLs
                result = self._fetch_url(url, format, extract_content)
            
            # Add formatted output for agent
            if result.is_success() and result.data:
                result.data["formatted"] = self.format_results_for_agent(result.data)
            
            return result
            
        except Exception as e:
            error_msg = str(e)
            if "Max retries exceeded" in error_msg or "Failed to resolve" in error_msg:
                error_msg = f"Network error: Unable to reach {url}"
            elif "NameResolutionError" in error_msg or "Name or service not known" in error_msg:
                error_msg = f"Invalid URL or domain not found: {url}"
            else:
                error_msg = f"Online fetch failed: {str(e)}"
            
            logger.error(f"[OnlineFetcher] Error: {error_msg}")
            error_data = {
                "error": error_msg,
                "url": url,
                "formatted": f"[Online Fetcher ERROR] {error_msg}"
            }
            
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                formatted=f"[Semantic Scholar Error] {error_msg}",
                error=error_msg
            )
    
    def _detect_source_type(self, url: str) -> str:
        """Auto-detect source type from URL."""
        url_lower = url.lower()
        
        if "arxiv.org" in url_lower:
            return "arxiv"
        elif "semanticscholar.org" in url_lower or "s2" in url_lower:
            return "semantic_scholar"
        else:
            # Everything else is just a URL (pubmed, xml, html, pdf, etc.)
            return "url"

    def _search_semantic_scholar(self, query: str, limit: int = 10) -> ToolResult:
        """Search for papers on Semantic Scholar."""
        try:
            self._rate_limit()
            
            # Prepare API request
            endpoint = f"{self.s2_api_base}/paper/search"
            params = {
                "query": query,
                "limit": min(limit, 100),
                "fields": "paperId,title,abstract,authors,year,venue,citationCount,referenceCount,externalIds,url,publicationDate,fieldsOfStudy,influentialCitationCount,isOpenAccess,openAccessPdf"
            }
            
            headers = {"User-Agent": self.user_agent}
            if self.semantic_scholar_api_key:
                headers["x-api-key"] = self.semantic_scholar_api_key
            
            logger.info(f"[SemanticScholar] Searching: {query}")
            response = requests.get(
                endpoint,
                params=params,
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            papers = data.get("data", [])
            
            if not papers:
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    data={
                        "source": "semantic_scholar",
                        "query": query,
                        "total_results": 0,
                        "papers": [],
                        "formatted": f"No papers found for query: {query}"
                    },
                    formatted=f"No papers found for query: {query}",
                    metadata={"total": 0}
                )
            
            # Format results
            formatted_papers = []
            for i, paper in enumerate(papers, 1):
                formatted_paper = {
                    "rank": i,
                    "paper_id": paper.get("paperId"),
                    "title": paper.get("title"),
                    "authors": [author.get("name") for author in paper.get("authors", [])],
                    "year": paper.get("year"),
                    "venue": paper.get("venue"),
                    "abstract": paper.get("abstract"),
                    "citation_count": paper.get("citationCount"),
                    "influential_citation_count": paper.get("influentialCitationCount"),
                    "reference_count": paper.get("referenceCount"),
                    "fields_of_study": paper.get("fieldsOfStudy", []),
                    "url": paper.get("url"),
                    "publication_date": paper.get("publicationDate"),
                    "is_open_access": paper.get("isOpenAccess"),
                    "open_access_pdf": paper.get("openAccessPdf", {}).get("url") if paper.get("openAccessPdf") else None,
                    "external_ids": paper.get("externalIds", {})
                }
                formatted_papers.append(formatted_paper)
            
            result_data = {
                "source": "semantic_scholar",
                "query": query,
                "total_results": len(papers),
                "papers": formatted_papers,
                "formatted": self._format_search_results(formatted_papers, query)
            }
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=result_data,
                formatted=result_data["formatted"],
                metadata={"total": len(papers), "query": query}
            )
            
        except requests.exceptions.HTTPError as e:
            error_msg = f"Semantic Scholar API error: {e.response.status_code}"
            if e.response.status_code == 429:
                error_msg = "Semantic Scholar API rate limit exceeded. Please try again later or use an API key."
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                formatted=f"[Semantic Scholar Error] {error_msg}",
                error=error_msg
            )
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                formatted=f"[Semantic Scholar Error] Semantic Scholar search failed: {str(e)}",
                error=f"Semantic Scholar search failed: {str(e)}"
            )
    
    def _fetch_semantic_scholar_paper(self, paper_id: str, extract_content: bool = True) -> ToolResult:
        """Fetch a specific paper from Semantic Scholar by ID or DOI."""
        try:
            self._rate_limit()
            
            # Prepare API request
            endpoint = f"{self.s2_api_base}/paper/{paper_id}"
            params = {
                "fields": "paperId,title,abstract,authors,year,venue,citationCount,referenceCount,externalIds,url,publicationDate,fieldsOfStudy,influentialCitationCount,isOpenAccess,openAccessPdf,tldr,citations,references"
            }
            
            headers = {"User-Agent": self.user_agent}
            if self.semantic_scholar_api_key:
                headers["x-api-key"] = self.semantic_scholar_api_key
            
            logger.info(f"[SemanticScholar] Fetching paper: {paper_id}")
            response = requests.get(
                endpoint,
                params=params,
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            paper = response.json()
            
            # Format paper metadata
            paper_data = {
                "source": "semantic_scholar",
                "paper_id": paper.get("paperId"),
                "title": paper.get("title"),
                "authors": [author.get("name") for author in paper.get("authors", [])],
                "year": paper.get("year"),
                "venue": paper.get("venue"),
                "abstract": paper.get("abstract"),
                "tldr": paper.get("tldr", {}).get("text") if paper.get("tldr") else None,
                "citation_count": paper.get("citationCount"),
                "influential_citation_count": paper.get("influentialCitationCount"),
                "reference_count": paper.get("referenceCount"),
                "fields_of_study": paper.get("fieldsOfStudy", []),
                "url": paper.get("url"),
                "publication_date": paper.get("publicationDate"),
                "is_open_access": paper.get("isOpenAccess"),
                "open_access_pdf": paper.get("openAccessPdf", {}).get("url") if paper.get("openAccessPdf") else None,
                "external_ids": paper.get("externalIds", {}),
                "citations": paper.get("citations", [])[:10] if paper.get("citations") else [],
                "references": paper.get("references", [])[:10] if paper.get("references") else []
            }
            
            # If extract_content is True and PDF is available, fetch it
            parsed_content = None
            if extract_content and paper_data["open_access_pdf"]:
                logger.info(f"[SemanticScholar] Fetching PDF content from: {paper_data['open_access_pdf']}")
                try:
                    pdf_result = self.pdf_parser.execute(paper_data["open_access_pdf"])
                    if pdf_result.is_success():
                        parsed_content = pdf_result.data
                except Exception as e:
                    logger.warning(f"[SemanticScholar] Failed to fetch PDF: {str(e)}")
            result_data = {
                "source": "semantic_scholar",
                "content_type": "metadata" if not parsed_content else "full",
                "metadata": paper_data,
                "parsed_content": parsed_content,
                "formatted": self._format_paper_details(paper_data, parsed_content)
            }
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=result_data,
                formatted=result_data["formatted"],
                metadata={"paper_id": paper_id}
            )
            
        except requests.exceptions.HTTPError as e:
            error_msg = f"Semantic Scholar API error: {e.response.status_code}"
            if e.response.status_code == 404:
                error_msg = f"Paper not found: {paper_id}"
            elif e.response.status_code == 429:
                error_msg = "Semantic Scholar API rate limit exceeded. Please try again later or use an API key."
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                formatted=f"[Semantic Scholar Error] {error_msg}",
                error=error_msg
            )
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                formatted=f"[Semantic Scholar Error] Failed to fetch Semantic Scholar paper: {str(e)}",
                error=f"Failed to fetch Semantic Scholar paper: {str(e)}"
            )
    
    def _fetch_semantic_scholar_url(self, url: str, extract_content: bool = True) -> ToolResult:
        """Fetch a paper from a Semantic Scholar URL."""
        # Extract paper ID from URL
        
        paper_id_match = re.search(r'/paper/[^/]+/([a-f0-9]+)', url)
        if not paper_id_match:
            paper_id_match = re.search(r'/paper/([a-f0-9]+)', url)
        if not paper_id_match:
            paper_id_match = re.search(r'/reader/([a-f0-9]+)', url)
        
        if not paper_id_match:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                formatted=f"[Semantic Scholar Error] Could not extract paper ID from URL: {url}",
                error=f"Could not extract paper ID from URL: {url}"
            )
        
        paper_id = paper_id_match.group(1)
        return self._fetch_semantic_scholar_paper(paper_id, extract_content)
    
    def _format_search_results(self, papers: List[Dict], query: str) -> str:
        """Format search results for display."""
        lines = [
            f"Semantic Scholar Search Results for: {query}",
            f"Total Results: {len(papers)}",
            "---",
            ""
        ]
        
        for paper in papers:
            lines.append(f"{paper['rank']}. {paper['title']}")
            if paper['authors']:
                authors_str = ", ".join(paper['authors'][:3])
                if len(paper['authors']) > 3:
                    authors_str += f" et al. ({len(paper['authors'])} authors)"
                lines.append(f"* Authors: {authors_str}")
            if paper['year']:
                lines.append(f"* ear: {paper['year']}")
            if paper['venue']:
                lines.append(f"* Venue: {paper['venue']}")
            if paper['citation_count'] is not None:
                lines.append(f"* Citations: {paper['citation_count']}")
            if paper['abstract']:
                abstract = paper['abstract']
                lines.append(f"* Abstract: {abstract}")
            if paper['open_access_pdf']:
                lines.append(f"* Open Access PDF: {paper['open_access_pdf']}")
            lines.append(f"* Paper ID: {paper['paper_id']}")
            lines.append(f"* URL: {paper['url']}")
            lines.append("")
        
        return "\n".join(lines).strip()
    
    def _format_paper_details(self, metadata: Dict, parsed_content: Optional[Dict] = None) -> str:
        """Format paper details for display."""
        lines = [
            "---"
            "Semantic Scholar Paper Details",
            "",
            f"Title: {metadata['title']}",
            ""
        ]
        
        if metadata['authors']:
            lines.append(f"* Authors: {', '.join(metadata['authors'])}")
            lines.append("")
        
        if metadata['year']:
            lines.append(f"* Year: {metadata['year']}")
        if metadata['venue']:
            lines.append(f"* Venue: {metadata['venue']}")
        if metadata['publication_date']:
            lines.append(f"* Publication Date: {metadata['publication_date']}")
        lines.append("")
        
        if metadata['citation_count'] is not None:
            lines.append(f"* Citations: {metadata['citation_count']}")
        if metadata['influential_citation_count'] is not None:
            lines.append(f"* Influential Citations: {metadata['influential_citation_count']}")
        if metadata['reference_count'] is not None:
            lines.append(f"* References: {metadata['reference_count']}")
        lines.append("")
        
        if metadata['fields_of_study']:
            lines.append(f"* Fields of Study: {', '.join(metadata['fields_of_study'])}")
            lines.append("")
        
        if metadata['tldr']:
            lines.append(f"* TL;DR: {metadata['tldr']}")
            lines.append("")
        
        if metadata['abstract']:
            lines.append(f"* Abstract: {metadata['abstract']}")
            lines.append("")
        
        if metadata['is_open_access']:
            lines.append("* Open Access: Yes")
            if metadata['open_access_pdf']:
                lines.append(f"* PDF URL: {metadata['open_access_pdf']}")
            lines.append("")
        
        if metadata['external_ids']:
            lines.append("* External IDs:")
            for id_type, id_value in metadata['external_ids'].items():
                if id_value:
                    lines.append(f"    * {id_type}: {id_value}")
            lines.append("")
        
        lines.append(f"* Semantic Scholar URL: {metadata['url']}")
        lines.append(f"* Paper ID: {metadata['paper_id']}")
        lines.append("")
        
        if parsed_content:
            lines.append("---")
            lines.append("Full Content Extracted from PDF")
            if parsed_content.get('sections'):
                lines.append(f"* Sections: {len(parsed_content['sections'])}")
            if parsed_content.get('figures'):
                lines.append(f"* Figures: {len(parsed_content['figures'])}")
            if parsed_content.get('tables'):
                lines.append(f"* Tables: {len(parsed_content['tables'])}")
            if parsed_content.get('references'):
                lines.append(f"* References: {len(parsed_content['references'])}")
        
        return "\n".join(lines)
    
    def _fetch_arxiv(
        self,
        url: str,
        format: str,
        extract_content: bool
    ) -> ToolResult:
        """Fetch arXiv paper from LaTeX source."""
        # Extract arXiv ID from URL
        arxiv_id = self._extract_arxiv_id(url)
        if not arxiv_id:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                formatted=f"[Semantic Scholar Error] Could not extract arXiv ID from URL",
                error="Could not extract arXiv ID from URL"
            )
        
        logger.info(f"[OnlineFetcher] arXiv ID: {arxiv_id}")
        # Construct URLs
        source_url = f"https://arxiv.org/e-print/{arxiv_id}"
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        
        result_data = {
            "source": "arxiv",
            "arxiv_id": arxiv_id,
            "url": url,
            "pdf_url": pdf_url,
            "source_url": source_url,
            "content_type": "latex",
            "raw_content": None
        }
        
        if not extract_content:
            # Just download source without parsing
            self._rate_limit()
            response = requests.get(
                source_url,
                headers={"User-Agent": self.user_agent},
                timeout=self.timeout
            )
            response.raise_for_status()
            result_data["raw_content"] = response.content
        else:
            # Download and parse LaTeX source
            try:
                logger.info(f"[OnlineFetcher] Downloading LaTeX source...")
                latex_data = self._download_arxiv_source(arxiv_id)
                
                logger.info(f"[OnlineFetcher] Parsing LaTeX structure...")
                parsed_content = self._parse_latex_structure(latex_data)
                result_data["parsed_content"] = parsed_content
                result_data["extraction_method"] = "latex_source"
                
            except Exception as e:
                logger.warning(f"[OnlineFetcher] LaTeX parsing failed: {e}, falling back to PDF")
                # Fallback to PDF parsing
                self._rate_limit()
                response = requests.get(
                    pdf_url,
                    headers={"User-Agent": self.user_agent},
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                if self.pdf_parser:
                    pdf_result = self.pdf_parser.execute(
                        pdf_path=response.content,
                        extract_text=True,
                        extract_tables=True,
                        extract_metadata=True
                    )
                    
                    if pdf_result.is_success():
                        result_data["parsed_content"] = pdf_result.data
                        result_data["extraction_method"] = "pdf_fallback"
                        result_data["content_type"] = "pdf"
        
        return ToolResult(
            status=ToolStatus.SUCCESS,
            data=result_data,
            formatted=self.format_results_for_agent(result_data),
            metadata={
                "source_type": "arxiv",
                "arxiv_id": arxiv_id,
                "parse_format": self.parse_format
            }
        )
    
    def _download_arxiv_source(self, arxiv_id: str) -> Dict[str, bytes]:
        """Download and extract arXiv source files."""
        source_url = f"https://arxiv.org/e-print/{arxiv_id}"
        
        self._rate_limit()
        response = requests.get(
            source_url,
            headers={"User-Agent": self.user_agent},
            timeout=self.timeout
        )
        response.raise_for_status()
        
        # Handle different archive formats
        content = response.content
        files = {}
        
        try:
            # Try as tar.gz first
            with tarfile.open(fileobj=io.BytesIO(content), mode='r:gz') as tar:
                for member in tar.getmembers():
                    if member.isfile():
                        files[member.name] = tar.extractfile(member).read()
        except:
            try:
                # Try as plain gzip
                tex_content = gzip.decompress(content)
                files['main.tex'] = tex_content
            except:
                # Assume plain tex file
                files['main.tex'] = content
        
        if not files:
            raise ValueError("No files extracted from arXiv source")
        
        return files
    
    def _identify_main_tex(self, files: Dict[str, bytes]) -> str:
        """Identify the main .tex file from multiple files."""
        tex_files = {name: content for name, content in files.items() 
                     if name.endswith('.tex')}
        
        if not tex_files:
            raise ValueError("No .tex files found in source")
        
        if len(tex_files) == 1:
            return list(tex_files.values())[0].decode('utf-8', errors='ignore')
        
        # Look for main indicators
        for name, content in tex_files.items():
            text = content.decode('utf-8', errors='ignore')
            if r'\documentclass' in text and r'\begin{document}' in text:
                return text
        
        # Fallback: use the largest file
        largest = max(tex_files.items(), key=lambda x: len(x[1]))
        return largest[1].decode('utf-8', errors='ignore')
    
    def _parse_latex_structure(self, files: Dict[str, bytes]) -> Dict[str, Any]:
        """Parse LaTeX content into comprehensive structured format."""
        
        # Identify main tex file
        tex_content = self._identify_main_tex(files)
        
        # Extract metadata
        title = self._extract_latex_field(tex_content, 'title')
        authors = self._extract_latex_field(tex_content, 'author')
        abstract = self._extract_latex_environment(tex_content, 'abstract')
        
        # Extract document content (between \begin{document} and \end{document})
        doc_match = re.search(
            r'\\begin\{document\}(.*?)\\end\{document\}',
            tex_content,
            re.DOTALL
        )
        doc_content = doc_match.group(1) if doc_match else tex_content
        
        # Check if appendix exists
        appendix_match = re.search(r'\\appendix', doc_content, re.IGNORECASE)
        has_appendix = appendix_match is not None
        
        if has_appendix:
            appendix_start = appendix_match.start()
            main_content = doc_content[:appendix_start]
            appendix_content = doc_content[appendix_start:]
        else:
            main_content = doc_content
            appendix_content = ""
        
        # Parse main content
        main_sections = self._extract_latex_sections(main_content, is_appendix=False)
        main_tables = self._extract_latex_tables(main_content, main_sections, is_appendix=False)
        main_figures = self._extract_latex_figures(main_content, main_sections, is_appendix=False)
        main_equations = self._extract_latex_equations(main_content, main_sections, is_appendix=False)
        main_algorithms = self._extract_latex_algorithms(main_content, main_sections, is_appendix=False)
        
        # Parse appendix content
        appendix_sections = []
        appendix_tables = []
        appendix_figures = []
        appendix_equations = []
        appendix_algorithms = []
        
        if has_appendix:
            appendix_sections = self._extract_latex_sections(appendix_content, is_appendix=True)
            appendix_tables = self._extract_latex_tables(appendix_content, appendix_sections, is_appendix=True)
            appendix_figures = self._extract_latex_figures(appendix_content, appendix_sections, is_appendix=True)
            appendix_equations = self._extract_latex_equations(appendix_content, appendix_sections, is_appendix=True)
            appendix_algorithms = self._extract_latex_algorithms(appendix_content, appendix_sections, is_appendix=True)
        
        # Extract bibliography
        references = self._extract_bibliography(tex_content)
        
        result = {
            "title": self._clean_latex(title),
            "authors": self._clean_latex(authors),
            "abstract": self._clean_latex(abstract),
            "main_content": {
                "sections": main_sections,
                "tables": main_tables,
                "figures": main_figures,
                "equations": main_equations,
                "algorithms": main_algorithms
            },
            "extraction_method": "latex_source",
            "accuracy": "high"
        }
        
        if has_appendix:
            result["appendix"] = {
                "sections": appendix_sections,
                "tables": appendix_tables,
                "figures": appendix_figures,
                "equations": appendix_equations,
                "algorithms": appendix_algorithms
            }
        
        if references:
            result["references"] = references
        
        return result
    
    def _extract_latex_field(self, tex_content: str, field: str) -> str:
        """Extract a LaTeX field like \\title{...} or \\author{...}."""
        pattern = rf'\\{field}\{{([^}}]+)\}}'
        match = re.search(pattern, tex_content, re.DOTALL)
        if match:
            return match.group(1)
        return ""
    
    def _extract_latex_environment(self, tex_content: str, env_name: str) -> str:
        """Extract content from LaTeX environment like \\begin{abstract}...\\end{abstract}."""
        pattern = rf'\\begin\{{{env_name}\}}(.*?)\\end\{{{env_name}\}}'
        match = re.search(pattern, tex_content, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""
    
    def _extract_latex_sections(self, content: str, is_appendix: bool) -> List[Dict[str, Any]]:
        """Extract hierarchical sections from LaTeX content."""
        sections = []
        
        # Section hierarchy patterns
        section_patterns = [
            (0, r'\\section\*?\{([^}]+)\}'),
            (1, r'\\subsection\*?\{([^}]+)\}'),
            (2, r'\\subsubsection\*?\{([^}]+)\}'),
            (3, r'\\paragraph\*?\{([^}]+)\}'),
            (4, r'\\subparagraph\*?\{([^}]+)\}'),
        ]
        
        for level, pattern in section_patterns:
            for match in re.finditer(pattern, content):
                position = match.start()
                title = match.group(1)
                
                # Extract content until next section of same or higher level
                section_content = self._extract_section_content(content, position, level)
                
                # Count paragraphs
                para_count = len(re.findall(r'\n\s*\n', section_content))
                
                section_data = {
                    "level": level,
                    "title": self._clean_latex(title),
                    "position": position,
                    "is_appendix": is_appendix
                }
                
                if self.parse_format == "full":
                    section_data["content"] = self._clean_latex(section_content)
                else:
                    section_data["paragraph_count"] = para_count
                
                sections.append(section_data)
        
        # Sort by position to maintain document order
        sections.sort(key=lambda x: x['position'])
        
        # Remove position field and build hierarchy
        for section in sections:
            del section['position']
        
        return sections
    
    def _extract_section_content(self, content: str, start_pos: int, level: int) -> str:
        """Extract content from section start until next section of same/higher level."""
        section_patterns = [
            r'\\section\*?\{',
            r'\\subsection\*?\{',
            r'\\subsubsection\*?\{',
            r'\\paragraph\*?\{',
            r'\\subparagraph\*?\{',
        ]
        
        # Only look for same or higher level sections
        relevant_patterns = section_patterns[:level + 1]
        
        # Find nearest next section
        remaining_content = content[start_pos:]
        min_next_pos = len(remaining_content)
        
        for pattern in relevant_patterns:
            # Skip the current section marker (look from position 20 onwards)
            match = re.search(pattern, remaining_content[20:])
            if match:
                min_next_pos = min(min_next_pos, match.start() + 20)
        
        return remaining_content[:min_next_pos]
    
    def _extract_latex_tables(
        self,
        content: str,
        sections: List[Dict[str, Any]],
        is_appendix: bool
    ) -> List[Dict[str, Any]]:
        """Extract all tables with captions, labels, and localization."""
        tables = []
        table_counter = 1
        
        # Find all table environments
        table_pattern = r'\\begin\{table\*?\}(.*?)\\end\{table\*?\}'
        
        for match in re.finditer(table_pattern, content, re.DOTALL):
            table_content = match.group(1)
            position = match.start()
            
            # Extract caption
            caption_match = re.search(r'\\caption\{([^}]+)\}', table_content, re.DOTALL)
            caption = caption_match.group(1) if caption_match else ""
            
            # Extract label
            label_match = re.search(r'\\label\{([^}]+)\}', table_content)
            label = label_match.group(1) if label_match else ""
            
            # Find localization (which section this table belongs to)
            localization = self._find_localization(position, sections)
            
            # Extract tabular content if in full mode
            tabular_content = ""
            if self.parse_format == "full":
                tabular_match = re.search(
                    r'\\begin\{tabular\}(.*?)\\end\{tabular\}',
                    table_content,
                    re.DOTALL
                )
                if tabular_match:
                    tabular_content = tabular_match.group(1)
            
            table_data = {
                "index": f"Table {table_counter}",
                "caption": self._clean_latex(caption),
                "label": label,
                "localization": localization,
                "is_appendix": is_appendix
            }
            
            if self.parse_format == "full" and tabular_content:
                table_data["content"] = self._clean_latex(tabular_content)
            
            tables.append(table_data)
            table_counter += 1
        
        return tables
    
    def _extract_latex_figures(
        self,
        content: str,
        sections: List[Dict[str, Any]],
        is_appendix: bool
    ) -> List[Dict[str, Any]]:
        """Extract all figures with captions, labels, and localization."""
        figures = []
        figure_counter = 1
        
        # Find all figure environments
        figure_pattern = r'\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}'
        
        for match in re.finditer(figure_pattern, content, re.DOTALL):
            figure_content = match.group(1)
            position = match.start()
            
            # Extract caption
            caption_match = re.search(r'\\caption\{([^}]+)\}', figure_content, re.DOTALL)
            caption = caption_match.group(1) if caption_match else ""
            
            # Extract label
            label_match = re.search(r'\\label\{([^}]+)\}', figure_content)
            label = label_match.group(1) if label_match else ""
            
            # Extract graphics filename if in full mode
            graphics_file = ""
            if self.parse_format == "full":
                graphics_match = re.search(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', figure_content)
                if graphics_match:
                    graphics_file = graphics_match.group(1)
            
            # Find localization
            localization = self._find_localization(position, sections)
            
            figure_data = {
                "index": f"Figure {figure_counter}",
                "caption": self._clean_latex(caption),
                "label": label,
                "localization": localization,
                "is_appendix": is_appendix
            }
            
            if self.parse_format == "full" and graphics_file:
                figure_data["graphics_file"] = graphics_file
            
            figures.append(figure_data)
            figure_counter += 1
        
        return figures
    
    def _extract_latex_equations(
        self,
        content: str,
        sections: List[Dict[str, Any]],
        is_appendix: bool
    ) -> List[Dict[str, Any]]:
        """Extract all equations with labels and localization."""
        equations = []
        equation_counter = 1
        
        # Patterns for different equation environments
        equation_patterns = [
            (r'\\begin\{equation\}(.*?)\\end\{equation\}', 'equation'),
            (r'\\begin\{align\}(.*?)\\end\{align\}', 'align'),
            (r'\\begin\{align\*\}(.*?)\\end\{align\*\}', 'align*'),
            (r'\\begin\{eqnarray\}(.*?)\\end\{eqnarray\}', 'eqnarray'),
            (r'\\begin\{multline\}(.*?)\\end\{multline\}', 'multline'),
            (r'\\begin\{gather\}(.*?)\\end\{gather\}', 'gather'),
            (r'\\\[(.*?)\\\]', 'display'),
        ]
        
        for pattern, env_type in equation_patterns:
            for match in re.finditer(pattern, content, re.DOTALL):
                equation_content = match.group(1).strip()
                position = match.start()
                
                # Extract label if present
                label_match = re.search(r'\\label\{([^}]+)\}', equation_content)
                label = label_match.group(1) if label_match else ""
                
                # Remove label from equation content
                if label:
                    equation_content = re.sub(r'\\label\{[^}]+\}', '', equation_content).strip()
                
                # Find localization
                localization = self._find_localization(position, sections)
                
                equation_data = {
                    "index": f"Equation {equation_counter}",
                    "equation": equation_content,  # Keep LaTeX format
                    "label": label,
                    "environment": env_type,
                    "localization": localization,
                    "is_appendix": is_appendix
                }
                
                equations.append(equation_data)
                equation_counter += 1
        
        return equations
    
    def _extract_latex_algorithms(
        self,
        content: str,
        sections: List[Dict[str, Any]],
        is_appendix: bool
    ) -> List[Dict[str, Any]]:
        """Extract all algorithms with captions, labels, and localization."""
        algorithms = []
        algorithm_counter = 1
        
        # Patterns for different algorithm environments
        algorithm_patterns = [
            r'\\begin\{algorithm\}(.*?)\\end\{algorithm\}',
            r'\\begin\{algorithm\*\}(.*?)\\end\{algorithm\*\}',
            r'\\begin\{algorithmic\}(.*?)\\end\{algorithmic\}',
        ]
        
        for pattern in algorithm_patterns:
            for match in re.finditer(pattern, content, re.DOTALL):
                algorithm_content = match.group(1)
                position = match.start()
                
                # Extract caption
                caption_match = re.search(r'\\caption\{([^}]+)\}', algorithm_content, re.DOTALL)
                caption = caption_match.group(1) if caption_match else ""
                
                # Extract label
                label_match = re.search(r'\\label\{([^}]+)\}', algorithm_content)
                label = label_match.group(1) if label_match else ""
                
                # Find localization
                localization = self._find_localization(position, sections)
                
                algorithm_data = {
                    "index": f"Algorithm {algorithm_counter}",
                    "caption": self._clean_latex(caption),
                    "label": label,
                    "localization": localization,
                    "is_appendix": is_appendix
                }
                
                # Extract algorithm content if in full mode
                if self.parse_format == "full":
                    # Remove caption and label from content
                    algo_content_clean = re.sub(r'\\caption\{[^}]+\}', '', algorithm_content)
                    algo_content_clean = re.sub(r'\\label\{[^}]+\}', '', algo_content_clean)
                    algorithm_data["content"] = self._clean_latex(algo_content_clean.strip())
                
                algorithms.append(algorithm_data)
                algorithm_counter += 1
        
        return algorithms
    
    def _find_localization(self, position: int, sections: List[Dict[str, Any]]) -> List[str]:
        """Find which section hierarchy an element belongs to based on position."""
        # Find the section that contains this position
        containing_sections = []
        
        for section in sections:
            # We need to track section positions in the original implementation
            pass
        
        # Simplified: return the most recent section titles
        if sections:
            # Find sections that appear before this position
            return [sections[-1].get('title', 'Unknown')]
        
        return []
    
    def _extract_bibliography(self, tex_content: str) -> List[Dict[str, str]]:
        """Extract bibliography entries."""
        refs = []
        
        # Find thebibliography environment
        bib_match = re.search(
            r'\\begin\{thebibliography\}(.*?)\\end\{thebibliography\}',
            tex_content,
            re.DOTALL
        )
        
        if bib_match:
            bib_content = bib_match.group(1)
            
            # Extract individual bibitem entries
            for item in re.finditer(r'\\bibitem\{([^}]+)\}(.*?)(?=\\bibitem|$)', bib_content, re.DOTALL):
                refs.append({
                    "key": item.group(1),
                    "citation": self._clean_latex(item.group(2).strip())
                })
        
        return refs
    
    def _clean_latex(self, text: str) -> str:
        """Remove LaTeX commands and clean text."""
        if not text:
            return ""
        
        # Remove comments
        text = re.sub(r'%.*$', '', text, flags=re.MULTILINE)
        
        # Remove common formatting commands but keep their content
        text = re.sub(r'\\(?:textbf|textit|emph|textrm|texttt)\{([^}]*)\}', r'\1', text)
        
        # Remove citations
        text = re.sub(r'\\cite\{[^}]*\}', '', text)
        
        # Remove labels and refs
        text = re.sub(r'\\(?:label|ref|eqref)\{[^}]*\}', '', text)
        
        # Remove general commands with arguments
        text = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', text)
        
        # Remove commands without arguments
        text = re.sub(r'\\[a-zA-Z]+', '', text)
        
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    
    def _fetch_url(
        self,
        url: str,
        format: str,
        extract_content: bool
    ) -> ToolResult:
        """Fetch content from general URL."""
        self._rate_limit()
        
        response = requests.get(
            url,
            headers={"User-Agent": self.user_agent},
            timeout=self.timeout,
            allow_redirects=True
        )
        response.raise_for_status()
        
        content_type = response.headers.get('Content-Type', '').lower()
        
        result_data = {
            "source": "url",
            "url": url,
            "content_type": content_type,
            "raw_content": None
        }
        
        # Handle PDF
        if 'pdf' in content_type or url.lower().endswith('.pdf'):
            result_data["content_type"] = "pdf"
            if not extract_content:
                result_data["raw_content"] = response.content
            elif self.pdf_parser:
                pdf_result = self.pdf_parser.execute(
                    pdf_path=response.content,
                    extract_text=True,
                    extract_tables=True
                )
                if pdf_result.is_success():
                    result_data["parsed_content"] = pdf_result.data
        
        # Handle HTML
        elif 'html' in content_type:
            result_data["content_type"] = "html"
            if extract_content:
                html_content = self._parse_html(response.text, url)
                result_data["parsed_content"] = html_content
            else:
                result_data["raw_content"] = response.text
        
        # Handle XML
        elif 'xml' in content_type or url.lower().endswith('.xml'):
            result_data["content_type"] = "xml"
            if extract_content:
                xml_result = self.xml_parser.execute(
                    xml_path=response.content,
                    extract_sections=True,
                    extract_tables=True,
                    extract_figures=True,
                    extract_references=True
                )
                if xml_result.is_success():
                    result_data["parsed_content"] = xml_result.data
            else:
                result_data["raw_content"] = response.content
        
        # Unknown type
        else:
            result_data["raw_content"] = response.content
        
        return ToolResult(
            status=ToolStatus.SUCCESS,
            data=result_data,
            formatted=self.format_results_for_agent(result_data),
            metadata={
                "source_type": "url",
                "content_type": result_data["content_type"],
                "parse_format": self.parse_format
            }
        )
    
    def _parse_html(self, html_content: str, url: str) -> Dict[str, Any]:
        """Parse HTML content with hierarchical structure."""
        if not BS4_AVAILABLE:
            return {"text": html_content, "error": "BeautifulSoup not available"}
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract title
        title = soup.title.string if soup.title else ""
        
        # Remove scripts, styles, and other non-content elements
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()
        
        # Extract hierarchical sections
        sections = self._extract_html_sections(soup)
        
        # Extract text if full format
        if self.parse_format == "full":
            text = soup.get_text(separator='\n', strip=True)
        else:
            text = None
        
        # Extract links
        links = []
        for link in soup.find_all('a', href=True):
            link_text = link.get_text(strip=True)
            if link_text:  # Only include links with text
                links.append({
                    "text": link_text,
                    "href": link['href']
                })
        
        result = {
            "title": title,
            "url": url,
            "sections": sections
        }
        
        if self.parse_format == "full":
            result["text"] = text
            result["links"] = links[:50]  # Limit to 50 links
        else:
            result["link_count"] = len(links)
        
        return result
    
    def _extract_html_sections(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract hierarchical sections from HTML."""
        sections = []
        
        # Find all headings
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        
        for heading in headings:
            level = int(heading.name[1])  # Extract number from h1, h2, etc.
            title = heading.get_text(strip=True)
            
            section_info = {
                "level": level - 1,  # 0-indexed
                "title": title
            }
            
            if self.parse_format == "full":
                # Get content until next heading of same or higher level
                content = []
                for sibling in heading.find_next_siblings():
                    if sibling.name and sibling.name.startswith('h') and int(sibling.name[1]) <= level:
                        break
                    text = sibling.get_text(strip=True)
                    if text:
                        content.append(text)
                section_info["content"] = "\n\n".join(content)
            else:
                # Count paragraphs in this section
                para_count = 0
                for sibling in heading.find_next_siblings():
                    if sibling.name and sibling.name.startswith('h') and int(sibling.name[1]) <= level:
                        break
                    if sibling.name == 'p':
                        para_count += 1
                section_info["paragraph_count"] = para_count
            
            sections.append(section_info)
        
        return sections

    def _format_latex_for_agent(self, content: Dict[str, Any]) -> str:
        """Format LaTeX-parsed arXiv content for agent."""
        formatted = []
        
        # Title and authors
        if content.get("title"):
            formatted.append(f"Title: {content['title']}")
        if content.get("authors"):
            formatted.append(f"Authors: {content['authors']}")
        if content.get("title") or content.get("authors"):
            formatted.append("")
        
        # Abstract
        if content.get("abstract"):
            formatted.append(f"Abstract: {content['abstract']}")
            formatted.append("")
        
        # Extraction method
        if content.get("extraction_method"):
            formatted.append(f"Extraction Method: {content['extraction_method']}")
            formatted.append("")
        
        # Main content
        if content.get("main_content"):
            main = content["main_content"]
            formatted.append("=== MAIN CONTENT ===")
            formatted.append("")
            
            # Sections
            if main.get("sections"):
                formatted.append(f"Sections ({len(main['sections'])}):")
                formatted.append(self._format_sections_for_agent(main["sections"]))
                formatted.append("")
            
            # Tables overview
            if main.get("tables"):
                formatted.append(f"Tables ({len(main['tables'])}):")
                for table in main["tables"]:
                    loc = " > ".join(table.get("localization", ["Unknown"]))
                    formatted.append(f"  • {table['index']}: {table['caption']}")
                    if table.get("label"):
                        formatted.append(f"    Label: {table['label']}")
                    formatted.append(f"    Location: {loc}")
                    if self.parse_format == "full" and table.get("content"):
                        formatted.append(f"    Content: {table['content'][:200]}...")
                formatted.append("")
            
            # Figures overview
            if main.get("figures"):
                formatted.append(f"Figures ({len(main['figures'])}):")
                for figure in main["figures"]:
                    loc = " > ".join(figure.get("localization", ["Unknown"]))
                    formatted.append(f"  • {figure['index']}: {figure['caption']}")
                    if figure.get("label"):
                        formatted.append(f"    Label: {figure['label']}")
                    formatted.append(f"    Location: {loc}")
                    if self.parse_format == "full" and figure.get("graphics_file"):
                        formatted.append(f"    File: {figure['graphics_file']}")
                formatted.append("")
            
            # Equations (always show full content)
            if main.get("equations"):
                formatted.append(f"Equations ({len(main['equations'])}):")
                for eq in main["equations"]:
                    loc = " > ".join(eq.get("localization", ["Unknown"]))
                    formatted.append(f"  • {eq['index']} ({eq['environment']}):")
                    formatted.append(f"    {eq['equation']}")
                    if eq.get("label"):
                        formatted.append(f"    Label: {eq['label']}")
                    formatted.append(f"    Location: {loc}")
                formatted.append("")
            
            # Algorithms overview
            if main.get("algorithms"):
                formatted.append(f"Algorithms ({len(main['algorithms'])}):")
                for algo in main["algorithms"]:
                    loc = " > ".join(algo.get("localization", ["Unknown"]))
                    formatted.append(f"  • {algo['index']}: {algo['caption']}")
                    if algo.get("label"):
                        formatted.append(f"    Label: {algo['label']}")
                    formatted.append(f"    Location: {loc}")
                    if self.parse_format == "full" and algo.get("content"):
                        formatted.append(f"    Content: {algo['content'][:200]}...")
                formatted.append("")
        
        # Appendix
        if content.get("appendix"):
            appendix = content["appendix"]
            formatted.append("=== APPENDIX ===")
            formatted.append("")
            
            # Appendix sections
            if appendix.get("sections"):
                formatted.append(f"Appendix Sections ({len(appendix['sections'])}):")
                formatted.append(self._format_sections_for_agent(appendix["sections"]))
                formatted.append("")
            
            # Appendix tables
            if appendix.get("tables"):
                formatted.append(f"Appendix Tables ({len(appendix['tables'])}):")
                for table in appendix["tables"]:
                    loc = " > ".join(table.get("localization", ["Unknown"]))
                    formatted.append(f"  • {table['index']}: {table['caption']}")
                    if table.get("label"):
                        formatted.append(f"    Label: {table['label']}")
                    formatted.append(f"    Location: {loc}")
                formatted.append("")
            
            # Appendix figures
            if appendix.get("figures"):
                formatted.append(f"Appendix Figures ({len(appendix['figures'])}):")
                for figure in appendix["figures"]:
                    loc = " > ".join(figure.get("localization", ["Unknown"]))
                    formatted.append(f"  • {figure['index']}: {figure['caption']}")
                    if figure.get("label"):
                        formatted.append(f"    Label: {figure['label']}")
                    formatted.append(f"    Location: {loc}")
                formatted.append("")
            
            # Appendix equations
            if appendix.get("equations"):
                formatted.append(f"Appendix Equations ({len(appendix['equations'])}):")
                for eq in appendix["equations"]:
                    loc = " > ".join(eq.get("localization", ["Unknown"]))
                    formatted.append(f"  • {eq['index']} ({eq['environment']}):")
                    formatted.append(f"    {eq['equation']}")
                    if eq.get("label"):
                        formatted.append(f"    Label: {eq['label']}")
                    formatted.append(f"    Location: {loc}")
                formatted.append("")
            
            # Appendix algorithms
            if appendix.get("algorithms"):
                formatted.append(f"Appendix Algorithms ({len(appendix['algorithms'])}):")
                for algo in appendix["algorithms"]:
                    loc = " > ".join(algo.get("localization", ["Unknown"]))
                    formatted.append(f"  • {algo['index']}: {algo['caption']}")
                    if algo.get("label"):
                        formatted.append(f"    Label: {algo['label']}")
                    formatted.append(f"    Location: {loc}")
                formatted.append("")
        
        # References
        if content.get("references"):
            formatted.append(f"References: {len(content['references'])} citations")
        
        return "\n".join(formatted).strip()

    def _format_pdf_for_agent(self, content: Dict[str, Any]) -> str:
        """Format PDF content for agent."""
        formatted = []
        
        # Title and metadata
        if content.get("title"):
            formatted.append(f"Title: {content['title']}")
        
        if content.get("metadata"):
            meta = content["metadata"]
            meta_parts = []
            if meta.get("author"):
                meta_parts.append(f"Author: {meta['author']}")
            if meta.get("creation_date"):
                meta_parts.append(f"Date: {meta['creation_date']}")
            if meta_parts:
                formatted.append(", ".join(meta_parts))
        
        if content.get("title") or content.get("metadata"):
            formatted.append("")
        
        # Abstract
        if content.get("abstract"):
            formatted.append(f"Abstract: {content['abstract']}")
            formatted.append("")
        
        # Structure
        if content.get("sections"):
            formatted.append("Structure:")
            formatted.append(self._format_sections_for_agent(content["sections"]))
            formatted.append("")
        
        # Summary counts
        summary_parts = []
        if content.get("num_pages"):
            summary_parts.append(f"{content['num_pages']} pages")
        if content.get("tables"):
            summary_parts.append(f"{len(content['tables'])} tables")
        if content.get("figures"):
            summary_parts.append(f"{len(content['figures'])} figures")
        if content.get("equations"):
            summary_parts.append(f"{len(content['equations'])} equations")
        if summary_parts:
            formatted.append(", ".join(summary_parts))
            formatted.append("")
        
        # Tables overview
        if content.get("tables"):
            formatted.append("Tables:")
            for table in content["tables"]:
                formatted.append(f"  • {table.get('index', 'Table')}: {table.get('caption', 'No caption')}")
                if table.get("localization"):
                    loc = " > ".join(table["localization"])
                    formatted.append(f"    Location: {loc}")
            formatted.append("")
        
        # Figures overview
        if content.get("figures"):
            formatted.append("Figures:")
            for figure in content["figures"]:
                formatted.append(f"  • {figure.get('index', 'Figure')}: {figure.get('caption', 'No caption')}")
                if figure.get("localization"):
                    loc = " > ".join(figure["localization"])
                    formatted.append(f"    Location: {loc}")
            formatted.append("")
        
        # Equations (show full content)
        if content.get("equations"):
            formatted.append("Equations:")
            for eq in content["equations"]:
                formatted.append(f"  • {eq.get('index', 'Equation')}:")
                formatted.append(f"    {eq.get('equation', '')}")
                if eq.get("localization"):
                    loc = " > ".join(eq["localization"])
                    formatted.append(f"    Location: {loc}")
            formatted.append("")
        
        return "\n".join(formatted).strip()

    def _format_xml_for_agent(self, content: Dict[str, Any]) -> str:
        """Format XML content for agent."""
        formatted = []
        
        # Title
        if content.get("title"):
            formatted.append(f"Title: {content['title']}")
            formatted.append("")
        
        # Abstract
        if content.get("abstract"):
            abstract = content['abstract']
            formatted.append(f"Abstract: {abstract}")
            formatted.append("")
        
        # Structure
        if content.get("sections"):
            formatted.append("Structure:")
            formatted.append(self._format_sections_for_agent(content['sections']))
            formatted.append("")
        
        # Summary counts
        summary_parts = []
        if content.get("tables"):
            summary_parts.append(f"{len(content['tables'])} tables")
        if content.get("figures"):
            summary_parts.append(f"{len(content['figures'])} figures")
        if content.get("equations"):
            summary_parts.append(f"{len(content['equations'])} equations")
        if content.get("references"):
            summary_parts.append(f"{len(content['references'])} references")
        if summary_parts:
            formatted.append(", ".join(summary_parts))
            formatted.append("")
        
        # Tables overview
        if content.get("tables"):
            formatted.append("Tables:")
            for table in content["tables"]:
                # Handle both 'index' and 'label' keys from different parsers
                label = table.get('label') or table.get('index', 'Table')
                caption = table.get('caption', 'No caption')
                formatted.append(f"  • {label}: {caption}")
                if table.get("localization"):
                    loc = " > ".join(table["localization"])
                    formatted.append(f"    Location: {loc}")
            formatted.append("")
        
        # Figures overview
        if content.get("figures"):
            formatted.append("Figures:")
            for figure in content["figures"]:
                # Handle both 'index' and 'label' keys from different parsers
                label = figure.get('label') or figure.get('index', 'Figure')
                caption = figure.get('caption', 'No caption')
                formatted.append(f"  • {label}: {caption}")
                if figure.get("localization"):
                    loc = " > ".join(figure["localization"])
                    formatted.append(f"    Location: {loc}")
            formatted.append("")
        
        # Equations (show full content)
        if content.get("equations"):
            formatted.append("Equations:")
            for eq in content["equations"]:
                formatted.append(f"  • {eq.get('index', 'Equation')}:")
                formatted.append(f"    {eq.get('equation', '')}")
                if eq.get("localization"):
                    loc = " > ".join(eq["localization"])
                    formatted.append(f"    Location: {loc}")
            formatted.append("")
        
        return "\n".join(formatted).strip()

    def _format_html_for_agent(self, content: Dict[str, Any]) -> str:
        """Format HTML content for agent."""
        formatted = []
        
        # Title
        if content.get("title"):
            formatted.append(f"Title: {content['title']}")
            formatted.append("")
        
        # Structure
        if content.get("sections"):
            formatted.append("Structure:")
            sections_preview = content["sections"][:8]  # Show first 8 sections
            for section in sections_preview:
                level = section.get('level', 0)
                title = section.get('title', 'Untitled')
                indent = "  " * level
                
                if "paragraph_count" in section:
                    formatted.append(f"{indent}• {title} ({section['paragraph_count']} ¶)")
                else:
                    formatted.append(f"{indent}• {title}")
            
            if len(content["sections"]) > 8:
                formatted.append(f"  ... and {len(content['sections']) - 8} more sections")
            formatted.append("")
        
        # Content preview (only for full format)
        if self.parse_format == "full" and content.get("text"):
            text = content["text"]
            preview_len = min(400, len(text))
            formatted.append(f"Content: {text[:preview_len]}...")
            formatted.append("")
        
        # Link count
        if content.get("links"):
            formatted.append(f"{len(content['links'])} links found")
        elif content.get("link_count"):
            formatted.append(f"{content['link_count']} links found")
        
        return "\n".join(formatted).strip()
    
    def _format_sections_for_agent(self, sections: List[Dict[str, Any]]) -> str:
        """Format hierarchical sections for agent."""
        formatted = []
        
        for section in sections:
            level = section.get('level', 0)
            title = section.get('title', 'Untitled')
            indent = "  " * level
            
            if self.parse_format == "full":
                formatted.append(f"{indent}{'#' * (level + 1)} {title}")
                content = section.get('content', '')
                if content:
                    # Show preview
                    preview_len = min(150, len(content))
                    preview = content[:preview_len] + ('...' if len(content) > 150 else '')
                    for line in preview.split('\n'):
                        if line.strip():
                            formatted.append(f"{indent}  {line}")
            else:
                para_count = section.get('paragraph_count', 0)
                formatted.append(f"{indent}{'#' * (level + 1)} {title} ({para_count} paragraphs)")
            
            # Handle subsections recursively
            if section.get('subsections'):
                formatted.append(self._format_sections_for_agent(section['subsections']))
        
        return "\n".join(formatted).strip()
    
    def format_results_for_agent(self, data: Dict[str, Any]) -> str:
        """Format fetched document results for agent consumption."""
        formatted = []
        
        # Header
        source_type = data.get('source', 'unknown').upper()
        formatted.append(f"Fetched from {source_type}")
        formatted.append(f"URL: {data.get('url', 'N/A')}")
        
        # Add source-specific identifiers
        if data.get('arxiv_id'):
            formatted.append(f"arXiv ID: {data['arxiv_id']}")
            if data.get('pdf_url'):
                formatted.append(f"PDF: {data['pdf_url']}")
            if data.get('source_url'):
                formatted.append(f"Source: {data['source_url']}")
        if data.get('pmc_id'):
            formatted.append(f"PMC ID: {data['pmc_id']}")
        
        content_type = data.get('content_type', 'unknown')
        formatted.append(f"Content Type: {content_type.upper()}")
        formatted.append("")
        
        # Format parsed content based on type
        if "parsed_content" in data:
            content = data["parsed_content"]
            
            # Check if this is LaTeX-parsed arXiv content
            if content_type == "latex" or content.get("extraction_method") == "latex_source":
                formatted.append(self._format_latex_for_agent(content))
            elif content_type == "pdf":
                formatted.append(self._format_pdf_for_agent(content))
            elif content_type == "xml":
                formatted.append(self._format_xml_for_agent(content))
            elif content_type == "html":
                formatted.append(self._format_html_for_agent(content))
            else:
                formatted.append("Content parsed successfully.")
        
        elif "raw_content" in data and data["raw_content"]:
            raw_size = len(data["raw_content"])
            if raw_size > 1024 * 1024:
                size_str = f"{raw_size / (1024*1024):.2f} MB"
            elif raw_size > 1024:
                size_str = f"{raw_size / 1024:.2f} KB"
            else:
                size_str = f"{raw_size} bytes"
            formatted.append(f"Raw content fetched: {size_str}")
        
        else:
            formatted.append("No content available.")
        
        return "\n".join(formatted).strip()
    
    def _extract_arxiv_id(self, url: str) -> Optional[str]:
        """Extract arXiv ID from URL."""
        patterns = [
            r'arxiv\.org/(?:abs|pdf)/(\d+\.\d+)',
            r'arxiv\.org/(?:abs|pdf)/([a-z\-]+/\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url.lower())
            if match:
                return match.group(1)
        
        return None

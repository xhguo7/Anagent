"""
arXiv search tool.
"""

import os
import time
import requests
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Optional
from datetime import datetime


from utils.logger import setup_logger
logger = setup_logger(__name__)

from anagent.tools.base import BaseTool, ToolResult, ToolStatus


class ArXivSearchTool(BaseTool):
    """arXiv search tool using arXiv API."""
    
    def __init__(
        self,
        max_results: int = 10,
        **kwargs
    ):
        """Initialize arXiv search tool."""
        super().__init__(
            name="arxiv_search",
            description="Search arXiv repository for academic papers and preprints",
            **kwargs
        )
        
        self.max_results = max_results
        self.api_url = "http://export.arxiv.org/api/query"
        
        # Rate limiting (arXiv recommends max 1 request per 3 seconds)
        self.request_delay = 3.0
        self.last_request_time = 0
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get parameter schema."""
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (supports arXiv query syntax)"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results (default: 10)"
                },
                "sort_by": {
                    "type": "string",
                    "description": "Sort by: 'relevance', 'lastUpdatedDate', 'submittedDate' (default: relevance)"
                },
                "sort_order": {
                    "type": "string",
                    "description": "Sort order: 'ascending', 'descending' (default: descending)"
                },
                "category": {
                    "type": "string",
                    "description": "Filter by category (e.g., 'cs.AI', 'physics.gen-ph')"
                }
            },
            "required": ["query"]
        }
    
    def _rate_limit(self):
        """Enforce rate limiting for arXiv API."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.request_delay:
            time.sleep(self.request_delay - time_since_last)
        self.last_request_time = time.time()
    
    def execute(
        self,
        query: str,
        max_results: Optional[int] = None,
        sort_by: str = "relevance",
        sort_order: str = "descending",
        category: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """Execute arXiv search."""
        logger.info(f"[arXiv] Searching for: '{query}'")
        # Validate parameters
        is_valid, error_msg = self.validate_parameters({"query": query})
        if not is_valid:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                formatted=f"[ArXiv Search Error] {error_msg}",
                error=error_msg
            )
        
        try:
            # Apply rate limiting
            self._rate_limit()
            
            # Build search query
            search_query = query
            if category:
                search_query = f"{query} AND cat:{category}"
            
            # Prepare parameters
            params = {
                "search_query": search_query,
                "start": 0,
                "max_results": max_results or self.max_results,
                "sortBy": sort_by,
                "sortOrder": sort_order
            }
            
            # Make API request
            response = requests.get(self.api_url, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse XML response
            articles = self._parse_response(response.content)
            
            logger.info(f"[arXiv] Found {len(articles)} papers")
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "results": articles,  # wrap in dict with "results" key for consistency
                    "total_results": len(articles)
                },
                formatted=self.format_results_for_agent(articles),
                metadata={
                    "query": query,
                    "total_results": len(articles),
                    "timestamp": datetime.now().isoformat(),
                    "search_type": "arxiv",
                    "category": category,
                    "sort_by": sort_by,
                    "sort_order": sort_order
                }
            )
            
        except Exception as e:
            error_msg = f"arXiv search failed: {str(e)}"
            logger.error(f"[arXiv] Error: {error_msg}")
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                formatted=f"[ArXiv Search Error] {error_msg}",
                error=error_msg
            )
    
    def _parse_response(self, xml_content: bytes) -> List[Dict[str, Any]]:
        """Parse arXiv API XML response."""
        # Parse XML
        root = ET.fromstring(xml_content)
        
        # Define namespaces
        namespaces = {
            'atom': 'http://www.w3.org/2005/Atom',
            'arxiv': 'http://arxiv.org/schemas/atom'
        }
        
        articles = []
        
        for entry in root.findall('atom:entry', namespaces):
            # Extract basic info
            title_elem = entry.find('atom:title', namespaces)
            title = title_elem.text.strip().replace('\n', ' ') if title_elem is not None else ""
            
            summary_elem = entry.find('atom:summary', namespaces)
            abstract = summary_elem.text.strip().replace('\n', ' ') if summary_elem is not None else ""
            
            # Extract authors
            authors = []
            for author_elem in entry.findall('atom:author', namespaces):
                name_elem = author_elem.find('atom:name', namespaces)
                if name_elem is not None:
                    authors.append(name_elem.text)
            
            # Extract dates
            published_elem = entry.find('atom:published', namespaces)
            published = published_elem.text if published_elem is not None else ""
            
            updated_elem = entry.find('atom:updated', namespaces)
            updated = updated_elem.text if updated_elem is not None else ""
            
            # Extract arXiv ID and categories
            arxiv_id = ""
            categories = []
            
            id_elem = entry.find('atom:id', namespaces)
            if id_elem is not None:
                arxiv_id = id_elem.text.split('/')[-1]
            
            for category_elem in entry.findall('atom:category', namespaces):
                term = category_elem.get('term')
                if term:
                    categories.append(term)
            
            # Primary category
            primary_category_elem = entry.find('arxiv:primary_category', namespaces)
            primary_category = primary_category_elem.get('term') if primary_category_elem is not None else ""
            
            # DOI
            doi_elem = entry.find('arxiv:doi', namespaces)
            doi = doi_elem.text if doi_elem is not None else ""
            
            # PDF link
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id else ""
            
            articles.append({
                "arxiv_id": arxiv_id,
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "published_date": published[:10] if published else "",  # YYYY-MM-DD
                "updated_date": updated[:10] if updated else "",
                "primary_category": primary_category,
                "categories": categories,
                "doi": doi,
                "url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "",
                "pdf_url": pdf_url
            })
        
        return articles
    
    def format_results_for_agent(self, results: List[Dict[str, Any]]) -> str:
        """Format search results as text for agent consumption."""
        if not results:
            return "No results found."
        
        formatted = []
        for i, paper in enumerate(results, 1):
            formatted.append(f"{i}. {paper['title']}")
            formatted.append(f"* arXiv ID: {paper['arxiv_id']}")
            if paper.get('authors'):
                authors_str = ", ".join(paper['authors'][:3])
                if len(paper['authors']) > 3:
                    authors_str += " et al."
                formatted.append(f"* Authors: {authors_str}")
            formatted.append(f"* Category: {paper['primary_category']}")
            formatted.append(f"* Published: {paper['published_date']}")
            formatted.append(f"* URL: {paper['url']}")
            # formatted.append(f"PDF: {paper['pdf_url']}")
            if paper.get('abstract'):
                abstract_preview = paper['abstract']
                formatted.append(f"* Abstract: {abstract_preview}")
            formatted.append("")
        
        return "\n".join(formatted).strip()

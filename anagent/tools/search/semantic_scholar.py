"""
Semantic Scholar search tool.
"""

import os
import time
import requests
from typing import Dict, Any, List, Optional
from datetime import datetime


from utils.logger import setup_logger
logger = setup_logger(__name__)

from anagent.tools.base import BaseTool, ToolResult, ToolStatus


class SemanticScholarSearchTool(BaseTool):
    """Semantic Scholar search tool using Semantic Scholar API."""
    
    def __init__(
        self,
        max_results: int = 10,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """Initialize Semantic Scholar search tool."""
        super().__init__(
            name="semantic_scholar_search",
            description="Search Semantic Scholar for academic papers and citations",
            **kwargs
        )
        
        self.max_results = max_results
        self.api_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        self.api_key = api_key
        
        # Rate limiting (1 request per second for public API, higher with API key)
        self.request_delay = 1.0 if not api_key else 0.1
        self.last_request_time = 0
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get parameter schema."""
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for papers"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results (default: 10, max: 100)"
                },
                "year": {
                    "type": "string",
                    "description": "Filter by year or year range (e.g., '2020' or '2018-2020')"
                },
                "venue": {
                    "type": "string",
                    "description": "Filter by venue (e.g., 'CVPR', 'NeurIPS')"
                },
                "fields_of_study": {
                    "type": "string",
                    "description": "Filter by field (e.g., 'Computer Science', 'Medicine')"
                },
                "min_citation_count": {
                    "type": "integer",
                    "description": "Minimum number of citations"
                },
                "publication_types": {
                    "type": "string",
                    "description": "Filter by type (e.g., 'JournalArticle', 'Conference')"
                },
                "open_access_pdf": {
                    "type": "boolean",
                    "description": "Only return papers with open access PDFs"
                }
            },
            "required": ["query"]
        }
    
    def _rate_limit(self):
        """Enforce rate limiting for Semantic Scholar API."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.request_delay:
            time.sleep(self.request_delay - time_since_last)
        self.last_request_time = time.time()
    
    def execute(
        self,
        query: str,
        max_results: Optional[int] = None,
        year: Optional[str] = None,
        venue: Optional[str] = None,
        fields_of_study: Optional[str] = None,
        min_citation_count: Optional[int] = None,
        publication_types: Optional[str] = None,
        open_access_pdf: Optional[bool] = None,
        **kwargs
    ) -> ToolResult:
        """Execute Semantic Scholar search."""
        logger.info(f"[Semantic Scholar] Searching for: '{query}'")
        # Validate parameters
        is_valid, error_msg = self.validate_parameters({"query": query})
        if not is_valid:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                formatted=f"[Semantic Scholar Search Error] {error_msg}",
                error=error_msg
            )
        
        try:
            # Apply rate limiting
            self._rate_limit()
            
            # Prepare parameters
            limit = min(max_results or self.max_results, 100)  # API max is 100
            
            params = {
                "query": query,
                "limit": limit,
                "fields": "paperId,title,abstract,authors,year,citationCount,influentialCitationCount,venue,publicationTypes,publicationDate,fieldsOfStudy,s2FieldsOfStudy,externalIds,url,openAccessPdf"
            }
            
            # Add optional filters
            if year:
                params["year"] = year
            if venue:
                params["venue"] = venue
            if fields_of_study:
                params["fieldsOfStudy"] = fields_of_study
            if min_citation_count is not None:
                params["minCitationCount"] = min_citation_count
            if publication_types:
                params["publicationTypes"] = publication_types
            if open_access_pdf:
                params["openAccessPdf"] = ""
            
            # Prepare headers
            headers = {}
            if self.api_key:
                headers["x-api-key"] = self.api_key
            
            # Make API request
            response = requests.get(
                self.api_url,
                params=params,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            
            # Parse JSON response
            data = response.json()
            papers = self._parse_response(data)
            
            logger.info(f"[Semantic Scholar] Found {len(papers)} papers")
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "results": papers,
                    "total_results": len(papers)
                },
                formatted=self.format_results_for_agent(papers),
                metadata={
                    "query": query,
                    "total_results": len(papers),
                    "timestamp": datetime.now().isoformat(),
                    "search_type": "semantic_scholar",
                    "year": year,
                    "venue": venue,
                    "fields_of_study": fields_of_study,
                    "min_citation_count": min_citation_count
                }
            )
            
        except requests.exceptions.HTTPError as e:
            error_msg = f"Semantic Scholar API error: {e.response.status_code} - {e.response.text}"
            logger.error(f"[Semantic Scholar] Error: {error_msg}")
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                formatted=f"[Semantic Scholar Search Error] {error_msg}",
                error=error_msg
            )
        except Exception as e:
            error_msg = f"Semantic Scholar search failed: {str(e)}"
            logger.error(f"[Semantic Scholar] Error: {error_msg}")
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                formatted=f"[Semantic Scholar Search Error] {error_msg}",
                error=error_msg
            )
    
    def _parse_response(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse Semantic Scholar API JSON response."""
        papers = []
        
        for item in data.get("data", []):
            # Extract authors
            authors = []
            for author in item.get("authors", []):
                name = author.get("name", "")
                if name:
                    authors.append(name)
            
            # Extract fields of study
            fields = []
            for field in item.get("s2FieldsOfStudy", []):
                category = field.get("category", "")
                if category:
                    fields.append(category)
            
            # Legacy fields of study (if s2FieldsOfStudy not available)
            if not fields:
                fields = item.get("fieldsOfStudy", [])
            
            # Extract external IDs
            external_ids = item.get("externalIds", {})
            arxiv_id = external_ids.get("ArXiv", "")
            doi = external_ids.get("DOI", "")
            
            # Open access PDF
            open_access = item.get("openAccessPdf")
            pdf_url = open_access.get("url", "") if open_access else ""
            
            papers.append({
                "paper_id": item.get("paperId", ""),
                "title": item.get("title", ""),
                "authors": authors,
                "abstract": item.get("abstract", ""),
                "year": item.get("year"),
                "publication_date": item.get("publicationDate", ""),
                "venue": item.get("venue", ""),
                "citation_count": item.get("citationCount", 0),
                "influential_citation_count": item.get("influentialCitationCount", 0),
                "fields_of_study": fields,
                "publication_types": item.get("publicationTypes", []),
                "arxiv_id": arxiv_id,
                "doi": doi,
                "url": item.get("url", ""),
                "pdf_url": pdf_url
            })
        
        return papers
    
    def format_results_for_agent(self, results: List[Dict[str, Any]]) -> str:
        """Format search results as text for agent consumption."""
        if not results:
            return "No results found."
        
        formatted = []
        for i, paper in enumerate(results, 1):
            formatted.append(f"{i}. {paper['title']}")
            formatted.append(f"* Paper ID: {paper['paper_id']}")
            if paper.get('authors'):
                authors_str = ", ".join(paper['authors'][:3])
                if len(paper['authors']) > 3:
                    authors_str += " et al."
                formatted.append(f"* Authors: {authors_str}")
            if paper.get('year'):
                formatted.append(f"* Year: {paper['year']}")
            if paper.get('venue'):
                formatted.append(f"* Venue: {paper['venue']}")
            formatted.append(f"* Citations: {paper['citation_count']} (Influential: {paper['influential_citation_count']})")
            if paper.get('fields_of_study'):
                fields_str = ", ".join(paper['fields_of_study'][:3])
                formatted.append(f"* Fields: {fields_str}")
            formatted.append(f"* URL: {paper['url']}")
            if paper.get('pdf_url'):
                formatted.append(f"* PDF: {paper['pdf_url']}")
            if paper.get('abstract'):
                abstract_preview = paper['abstract']
                formatted.append(f"* Abstract: {abstract_preview}")
            formatted.append("")
        
        return "\n".join(formatted).strip()

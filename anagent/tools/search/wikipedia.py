"""
Wikipedia search tool.
"""

import os
import requests
from typing import Dict, Any, List, Optional
from datetime import datetime


from utils.logger import setup_logger
logger = setup_logger(__name__)

from anagent.tools.base import BaseTool, ToolResult, ToolStatus


class WikipediaSearchTool(BaseTool):
    """Wikipedia search tool with intelligent API routing."""
    
    def __init__(
        self,
        language: str = "en",
        max_results: int = 5,
        **kwargs
    ):
        """Initialize Wikipedia search tool."""
        super().__init__(
            name="wikipedia_search",
            description="Search Wikipedia for general knowledge and encyclopedic information",
            **kwargs
        )
        
        self.language = language
        self.max_results = max_results
        self.api_url = f"https://{language}.wikipedia.org/w/api.php"
        self.rest_api_url = f"https://{language}.wikipedia.org/api/rest_v1"

        # Create session with proper headers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'WikipediaSearchTool/1.0 (Educational/Research Purpose; Python/requests)'
        })
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get parameter schema."""
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query with optional mode prefix: 'search:', 'title:', 'summary:', 'sections:', or plain text"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results (default: 5)"
                },
                "sentences": {
                    "type": "integer",
                    "description": "Number of sentences for summary (default: 3)"
                }
            },
            "required": ["query"]
        }
    
    def execute(
        self,
        query: str,
        max_results: Optional[int] = None,
        sentences: int = 5,
        **kwargs
    ) -> ToolResult:
        """Execute Wikipedia search with intelligent API routing."""
        logger.info(f"[Wikipedia] Processing query: '{query}'")
        # Validate parameters
        is_valid, error_msg = self.validate_parameters({"query": query})
        if not is_valid:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                formatted=f"[Wikipedia Search Error] {error_msg}",
                error=error_msg
            )
        
        try:
            # Detect query mode and route to appropriate API
            mode, clean_query = self._detect_mode(query)
            
            logger.info(f"[Wikipedia] Mode: {mode}, Query: '{clean_query}'")
            # Route to appropriate API method
            if mode == "search":
                result = self._opensearch(clean_query, max_results or self.max_results)
            elif mode == "title":
                result = self._query_by_title(clean_query, sentences)
            elif mode == "summary":
                result = self._rest_summary(clean_query)
            elif mode == "sections":
                result = self._parse_sections(clean_query)
            else:  # auto mode - smart search
                result = self._smart_search(clean_query, max_results or self.max_results, sentences)
            
            logger.info(f"[Wikipedia] Found {len(result)} results")
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=result,
                formatted=self.format_results_for_agent(result),
                metadata={
                    "query": query,
                    "mode": mode,
                    "total_results": len(result),
                    "timestamp": datetime.now().isoformat(),
                    "search_type": "wikipedia",
                    "language": self.language
                }
            )
            
        except Exception as e:
            error_msg = f"Wikipedia search failed: {str(e)}"
            logger.error(f"[Wikipedia] Error: {error_msg}")
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                formatted=f"[Wikipedia Search Error] {error_msg}",
                error=error_msg
            )
    
    def _detect_mode(self, query: str) -> tuple:
        """Detect query mode from prefix."""
        query_lower = query.strip().lower()
        
        if query_lower.startswith("search:"):
            return "search", query[7:].strip()
        elif query_lower.startswith("title:"):
            return "title", query[6:].strip()
        elif query_lower.startswith("summary:"):
            return "summary", query[8:].strip()
        elif query_lower.startswith("sections:"):
            return "sections", query[9:].strip()
        else:
            return "auto", query.strip()
    
    def _opensearch(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """OpenSearch API - Fast article title search."""
        params = {
            "action": "opensearch",
            "search": query,
            "limit": max_results,
            "namespace": 0,
            "format": "json"
        }
        
        response = self.session.get(self.api_url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        titles = data[1] if len(data) > 1 else []
        descriptions = data[2] if len(data) > 2 else []
        urls = data[3] if len(data) > 3 else []
        
        results = []
        for i in range(len(titles)):
            results.append({
                "title": titles[i],
                "description": descriptions[i] if i < len(descriptions) else "",
                "url": urls[i] if i < len(urls) else "",
                "content": None,
                "api_method": "opensearch"
            })
        
        return results
    
    def _query_by_title(self, title: str, sentences: int) -> List[Dict[str, Any]]:
        """Query API with TextExtracts - Get article by exact title."""
        params = {
            "action": "query",
            "prop": "extracts|info|pageimages",
            "exintro": True,
            "exsentences": sentences,
            "explaintext": True,
            "inprop": "url",
            "piprop": "thumbnail",
            "pithumbsize": 300,
            "titles": title,
            "format": "json"
        }
        
        response = self.session.get(self.api_url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        pages = data.get("query", {}).get("pages", {})
        
        results = []
        for page_id, page_data in pages.items():
            if page_id == "-1":  # Page not found
                continue
                
            results.append({
                "title": page_data.get("title", ""),
                "description": page_data.get("extract", "")[:200],
                "content": page_data.get("extract", ""),
                "url": page_data.get("fullurl", ""),
                "page_id": page_id,
                "thumbnail": page_data.get("thumbnail", {}).get("source", ""),
                "api_method": "query_extracts"
            })
        
        return results
    
    def _rest_summary(self, title: str) -> List[Dict[str, Any]]:
        """REST API Summary - Modern endpoint for article summaries."""
        # URL encode the title
        encoded_title = requests.utils.quote(title.replace(" ", "_"))
        url = f"{self.rest_api_url}/page/summary/{encoded_title}"

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"[Wikipedia] REST API request failed: {str(e)}")
            return []
        except ValueError as e:
            logger.error(f"[Wikipedia] REST API returned non-JSON: {response.text[:200]}")
            return []
        
        # Check for error in response
        if data.get("type") == "https://mediawiki.org/wiki/HyperSwitch/errors/not_found":
            logger.warning(f"[Wikipedia] Article not found: {title}")
            return []
        
        result = {
            "title": data.get("title", ""),
            "description": data.get("description", ""),
            "content": data.get("extract", ""),
            "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
            "thumbnail": data.get("thumbnail", {}).get("source", ""),
            "page_id": data.get("pageid", ""),
            "api_method": "rest_summary"
        }
        
        return [result] if result["title"] else []
    
    def _parse_sections(self, title: str) -> List[Dict[str, Any]]:
        """Parse API - Get article structure with sections."""
        params = {
            "action": "parse",
            "page": title,
            "prop": "sections|displaytitle",
            "format": "json",
            "formatversion": "2"
        }
        
        try:
            response = self.session.get(self.api_url, params=params, timeout=30)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"[Wikipedia] Request failed: {str(e)}")
            return []
        
        # Check if response is JSON before parsing
        try:
            data = response.json()
        except ValueError as e:
            logger.error(f"[Wikipedia] Parse API returned non-JSON: {response.text[:200]}")
            return []
        
        if "error" in data:
            logger.warning(f"[Wikipedia] Parse API error: {data['error'].get('info', 'Unknown error')}")
            return []
        
        parse_data = data.get("parse", {})
        sections = parse_data.get("sections", [])
        display_title = parse_data.get("displaytitle", title)
        
        # Get article URL using title
        article_url = f"https://{self.language}.wikipedia.org/wiki/{requests.utils.quote(title.replace(' ', '_'))}"
        
        # Format sections
        section_list = []
        for section in sections:
            section_list.append({
                "level": section.get("toclevel", 1),
                "number": section.get("number", ""),
                "title": section.get("line", ""),
                "index": section.get("index", "")
            })
        
        result = {
            "title": display_title,
            "description": f"Article structure with {len(sections)} sections",
            "content": self._format_sections(section_list),
            "sections": section_list,
            "url": article_url,
            "api_method": "parse_sections"
        }
        
        return [result]
    
    def _smart_search(self, query: str, max_results: int, sentences: int) -> List[Dict[str, Any]]:
        """Smart search - OpenSearch for titles, then TextExtracts for content."""
        # First, use OpenSearch to find matching titles
        search_results = self._opensearch(query, max_results)
        
        if not search_results:
            return []
        
        # Then fetch content for each using Query API
        titles = [result["title"] for result in search_results]
        
        params = {
            "action": "query",
            "prop": "extracts|info",
            "exintro": True,
            "exsentences": sentences,
            "explaintext": True,
            "inprop": "url",
            "titles": "|".join(titles),
            "format": "json"
        }
        
        response = self.session.get(self.api_url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        pages = data.get("query", {}).get("pages", {})
        
        # Merge search results with content
        for result in search_results:
            for page_id, page_data in pages.items():
                if page_data.get("title") == result["title"]:
                    result["content"] = page_data.get("extract", "")
                    result["url"] = page_data.get("fullurl", result.get("url", ""))
                    result["page_id"] = page_id
                    result["api_method"] = "smart_search"
                    break
        
        return search_results
    
    def _format_sections(self, sections: List[Dict[str, Any]]) -> str:
        """Format sections list as text."""
        lines = []
        for section in sections:
            indent = "  " * (section["level"] - 1)
            number = section["number"]
            title = section["title"]
            lines.append(f"{indent}{number} {title}")
        return "\n".join(lines)
    
    def format_results_for_agent(self, results: List[Dict[str, Any]]) -> str:
        """Format search results as text for agent consumption."""
        if not results:
            return "No results found."
        
        formatted = []
        for i, article in enumerate(results, 1):
            formatted.append(f"{i}. {article['title']}")
            formatted.append(f"URL: {article['url']}")
            formatted.append(f"[via {article.get('api_method', 'unknown')}]")
            
            if article.get('description'):
                formatted.append(f"- Results: {article['description']}")
            
            if article.get('content'):
                formatted.append(f"{article['content']}")
            
            if article.get('sections'):
                formatted.append(f"Sections: {len(article['sections'])} found")
            
            formatted.append("")
        
        return "\n".join(formatted).strip()

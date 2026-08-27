"""
Web search tool.
"""

import os
import requests
from typing import Dict, Any, List, Optional
from datetime import datetime


from utils.logger import setup_logger
logger = setup_logger(__name__)

from anagent.tools.base import BaseTool, ToolResult, ToolStatus


class WebSearchTool(BaseTool):
    """Web search tool using Google Custom Search API."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        search_engine_id: Optional[str] = None,
        max_results: int = 10,
        **kwargs
    ):
        """Initialize web search tool."""
        super().__init__(
            name="web_search",
            description="Search the web for general information using Google Custom Search API",
            **kwargs
        )
        
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.search_engine_id = search_engine_id or os.getenv("GOOGLE_CSE_ID")
        
        if not self.api_key:
            raise ValueError(
                "Google API key required. Set GOOGLE_API_KEY environment "
                "variable or pass api_key parameter. Get one at: "
                "https://console.cloud.google.com/apis/credentials"
            )
        
        if not self.search_engine_id:
            raise ValueError(
                "Google Custom Search Engine ID required. Set GOOGLE_CSE_ID environment "
                "variable or pass search_engine_id parameter. Create one at: "
                "https://programmablesearchengine.google.com/controlpanel/all"
            )
        
        self.max_results = max_results
        self.base_url = "https://www.googleapis.com/customsearch/v1"
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get parameter schema."""
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query string"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results (default: 10, max: 10 per API call)"
                },
                "date_restrict": {
                    "type": "string",
                    "description": "Restrict by date: 'd[number]' (days), 'w[number]' (weeks), 'm[number]' (months), 'y[number]' (years). e.g., 'd7' for last 7 days"
                },
                "language": {
                    "type": "string",
                    "description": "Language code (e.g., 'en', 'es', 'fr')"
                },
                "safe": {
                    "type": "string",
                    "description": "Safe search: 'off', 'medium', 'high' (default: medium)"
                }
            },
            "required": ["query"]
        }
    
    def execute(
        self,
        query: str,
        max_results: Optional[int] = None,
        date_restrict: Optional[str] = None,
        language: str = "en",
        safe: str = "medium",
        **kwargs
    ) -> ToolResult:
        """Execute web search."""
        logger.info(f"[WebSearch] Searching for: '{query}'")
        # Validate parameters
        is_valid, error_msg = self.validate_parameters({"query": query})
        if not is_valid:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                formatted=f"[Web Search Error] {error_msg}",
                error=error_msg
            )
        
        # Limit results to 10 (Google API limit per call)
        num_results = min(max_results or self.max_results, 10)
        
        # Prepare request parameters
        params = {
            "key": self.api_key,
            "cx": self.search_engine_id,
            "q": query,
            "num": num_results,
            "lr": f"lang_{language}",
            "safe": safe
        }
        
        if date_restrict:
            params["dateRestrict"] = date_restrict
        
        try:
            # Make API request
            response = requests.get(
                self.base_url,
                params=params,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Extract relevant results
            results = self._parse_results(data)
            
            logger.info(f"[WebSearch] Found {len(results)} results")
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=results,
                formatted=self.format_results_for_agent(results),
                metadata={
                    "query": query,
                    "total_results": len(results),
                    "timestamp": datetime.now().isoformat(),
                    "search_type": "web",
                    "source": "google_search"
                }
            )
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Web search failed: {str(e)}"
            logger.error(f"[WebSearch] Error: {error_msg}")
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                formatted=f"[Web Search Error] {error_msg}",
                error=error_msg
            )
        except Exception as e:
            error_msg = f"Unexpected error in web search: {str(e)}"
            logger.error(f"[WebSearch] Error: {error_msg}")
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                formatted=f"[Web Search Error] {error_msg}",
                error=error_msg
            )
    
    def _parse_results(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse API response into clean result format."""
        results = []
        
        # Extract search results
        items = data.get("items", [])
        
        for item in items:
            result = {
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "description": item.get("snippet", ""),
                "displayed_url": item.get("displayLink", ""),
            }
            
            # Add page metadata if available
            pagemap = item.get("pagemap", {})
            if pagemap:
                # Extract meta description if available
                metatags = pagemap.get("metatags", [{}])[0]
                if metatags.get("og:description"):
                    result["meta_description"] = metatags["og:description"]
                
                # Extract publication date if available
                if metatags.get("article:published_time"):
                    result["published_date"] = metatags["article:published_time"]
            
            results.append(result)
        
        return results
    
    def format_results_for_agent(self, results: List[Dict[str, Any]]) -> str:
        """Format search results as text for agent consumption."""
        if not results:
            return "No results found."
        
        formatted = []
        for i, result in enumerate(results, 1):
            formatted.append(f"{i}. {result['title']}")
            formatted.append(f"* URL: {result['url']}")
            if result.get('published_date'):
                formatted.append(f"* Published: {result['published_date']}")
            formatted.append(f"* Result: {result['description']}")
            formatted.append("")
        
        return "\n".join(formatted).strip()

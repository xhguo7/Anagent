"""
PubMed search tool.
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


class PubMedSearchTool(BaseTool):
    """PubMed search tool using NCBI E-utilities API."""
    
    def __init__(
        self,
        email: Optional[str] = None,
        api_key: Optional[str] = None,
        max_results: int = 10,
        **kwargs
    ):
        """Initialize PubMed search tool."""
        super().__init__(
            name="pubmed_search",
            description="Search PubMed database for biomedical literature",
            **kwargs
        )
        
        self.email = email or os.getenv("NCBI_EMAIL", "user@example.com")
        self.api_key = api_key or os.getenv("NCBI_API_KEY")
        self.max_results = max_results
        
        # E-utilities endpoints
        self.esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        self.esummary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        self.efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        
        # Rate limiting (3 requests/sec without API key, 10/sec with key)
        self.request_delay = 0.5 if not self.api_key else 0.25
        self.last_request_time = 0
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get parameter schema."""
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "PubMed search query (supports PubMed syntax)"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results (default: 10)"
                },
                "sort": {
                    "type": "string",
                    "description": "Sort order: 'relevance', 'pub_date' (default: relevance)"
                },
                "date_range": {
                    "type": "object",
                    "description": "Date range filter with 'from' and 'to' dates (YYYY/MM/DD)"
                },
                "retrieve_abstracts": {
                    "type": "boolean",
                    "description": "Whether to retrieve full abstracts (default: True)"
                }
            },
            "required": ["query"]
        }
    
    def _rate_limit(self):
        """Enforce rate limiting for NCBI API."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.request_delay:
            time.sleep(self.request_delay - time_since_last)
        self.last_request_time = time.time()
    
    def execute(
        self,
        query: str,
        max_results: Optional[int] = 10,
        sort: str = "relevance",
        date_range: Optional[Dict[str, str]] = None,
        retrieve_abstracts: bool = True,
        **kwargs
    ) -> ToolResult:
        """Execute PubMed search."""
        logger.info(f"[PubMed] Searching for: '{query}'")
        # Validate parameters
        params_to_validate = {"query": query}
        if max_results is not None:
            params_to_validate["max_results"] = max_results
        if sort:
            params_to_validate["sort"] = sort
        if date_range:
            params_to_validate["date_range"] = date_range
        if retrieve_abstracts is not None:
            params_to_validate["retrieve_abstracts"] = retrieve_abstracts
        
        is_valid, error_msg = self.validate_parameters(params_to_validate)
        if not is_valid:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                formatted=f"[PubMed Search Error] {error_msg}",
                error=error_msg
            )
        
        try:
            # Step 1: Search for PMIDs
            pmids = self._search_pmids(query, max_results or self.max_results, sort, date_range)
            
            if not pmids:
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    data=[],
                    formatted="NO paper found",
                    metadata={
                        "query": query,
                        "total_results": 0,
                        "timestamp": datetime.now().isoformat()
                    }
                )
            
            logger.info(f"[PubMed] Found {len(pmids)} articles")
            # Step 2: Fetch article details
            if retrieve_abstracts:
                articles = self._fetch_articles(pmids)
            else:
                articles = self._fetch_summaries(pmids)
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=articles,
                formatted=self.format_results_for_agent(articles),
                metadata={
                    "query": query,
                    "total_results": len(articles),
                    "timestamp": datetime.now().isoformat(),
                    "search_type": "pubmed",
                    "pmids": pmids
                }
            )
            
        except Exception as e:
            error_msg = f"PubMed search failed: {str(e)}"
            logger.error(f"[PubMed] Error: {error_msg}")
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                formatted=f"[PubMed Search Error] {error_msg}",
                error=error_msg
            )
    
    def _search_pmids(
        self,
        query: str,
        max_results: int,
        sort: str,
        date_range: Optional[Dict[str, str]]
    ) -> List[str]:
        """Search PubMed and get PMIDs."""
        self._rate_limit()
        
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": sort,
            "email": self.email
        }
        
        if self.api_key:
            params["api_key"] = self.api_key
        
        if date_range:
            if "from" in date_range and "to" in date_range:
                params["mindate"] = date_range["from"]
                params["maxdate"] = date_range["to"]
                params["datetype"] = "pdat"  # publication date
        
        response = requests.get(self.esearch_url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        pmids = data.get("esearchresult", {}).get("idlist", [])
        
        return pmids
    
    def _fetch_summaries(self, pmids: List[str]) -> List[Dict[str, Any]]:
        """Fetch article summaries using ESummary."""
        self._rate_limit()
        
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "json",
            "email": self.email
        }
        
        if self.api_key:
            params["api_key"] = self.api_key
        
        response = requests.get(self.esummary_url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        results = data.get("result", {})
        
        articles = []
        for pmid in pmids:
            if pmid in results:
                article_data = results[pmid]
                articles.append({
                    "pmid": pmid,
                    "title": article_data.get("title", ""),
                    "authors": [
                        author.get("name", "") 
                        for author in article_data.get("authors", [])
                    ],
                    "journal": article_data.get("fulljournalname", ""),
                    "pub_date": article_data.get("pubdate", ""),
                    "doi": article_data.get("elocationid", "").replace("doi: ", ""),
                    "abstract": None,
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                })
        
        return articles
    
    def _fetch_articles(self, pmids: List[str]) -> List[Dict[str, Any]]:
        """Fetch full article details including abstracts using EFetch."""
        self._rate_limit()
        
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "email": self.email
        }
        
        if self.api_key:
            params["api_key"] = self.api_key
        
        response = requests.get(self.efetch_url, params=params, timeout=30)
        response.raise_for_status()
        
        # Parse XML response
        root = ET.fromstring(response.content)
        
        articles = []
        for article_elem in root.findall(".//PubmedArticle"):
            article = self._parse_article_xml(article_elem)
            articles.append(article)
        
        return articles
    
    def _parse_article_xml(self, article_elem: ET.Element) -> Dict[str, Any]:
        """Parse PubMed article XML element."""
        # PMID
        pmid_elem = article_elem.find(".//PMID")
        pmid = pmid_elem.text if pmid_elem is not None else ""
        
        # Title
        title_elem = article_elem.find(".//ArticleTitle")
        title = title_elem.text if title_elem is not None else ""
        
        # Authors
        authors = []
        for author_elem in article_elem.findall(".//Author"):
            lastname = author_elem.find("LastName")
            forename = author_elem.find("ForeName")
            if lastname is not None and forename is not None:
                authors.append(f"{forename.text} {lastname.text}")
        
        # Journal
        journal_elem = article_elem.find(".//Journal/Title")
        journal = journal_elem.text if journal_elem is not None else ""
        
        # Publication date
        pub_date = ""
        year_elem = article_elem.find(".//PubDate/Year")
        month_elem = article_elem.find(".//PubDate/Month")
        if year_elem is not None:
            pub_date = year_elem.text
            if month_elem is not None:
                pub_date = f"{month_elem.text} {pub_date}"
        
        # DOI
        doi = ""
        for article_id in article_elem.findall(".//ArticleId"):
            if article_id.get("IdType") == "doi":
                doi = article_id.text
                break
        
        # Abstract
        abstract_texts = []
        for abstract_elem in article_elem.findall(".//Abstract/AbstractText"):
            label = abstract_elem.get("Label", "")
            text = abstract_elem.text or ""
            if label:
                abstract_texts.append(f"{label}: {text}")
            else:
                abstract_texts.append(text)
        abstract = " ".join(abstract_texts) if abstract_texts else None
        
        return {
            "pmid": pmid,
            "title": title,
            "authors": authors,
            "journal": journal,
            "pub_date": pub_date,
            "doi": doi,
            "abstract": abstract,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        }
    
    def format_results_for_agent(self, results: List[Dict[str, Any]]) -> str:
        """Format search results as text for agent consumption."""
        if not results:
            return "No results found."
        
        formatted = []
        for i, article in enumerate(results, 1):
            formatted.append(f"{i}. {article['title']}")
            formatted.append(f"* PMID: {article['pmid']}")
            if article.get('authors'):
                authors_str = ", ".join(article['authors'][:5])
                if len(article['authors']) > 3:
                    authors_str += " et al."
                formatted.append(f"* Authors: {authors_str}")
            
            journal_info = article['journal']
            if article['pub_date']:
                journal_info += f" ({article['pub_date']})"
            formatted.append(f"* Journal: {journal_info}")
            
            if article.get('doi'):
                formatted.append(f"* DOI: {article['doi']}")
            formatted.append(f"* URL: {article['url']}")
            
            if article.get('abstract'):
                # Format abstract with proper indentation
                abstract_lines = article['abstract'].split('\n')
                formatted.append(f"* Abstract:")
                for line in abstract_lines:
                    if line.strip():
                        formatted.append(f"{line.strip()}")
            
            formatted.append("")
        
        return "\n".join(formatted).strip()

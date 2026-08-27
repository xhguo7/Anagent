"""
XML parser tool.
"""

import os
import re
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import requests
from urllib.parse import urlparse
import time


from utils.logger import setup_logger
logger = setup_logger(__name__)

from anagent.tools.base import BaseTool, ToolResult, ToolStatus


class XMLParserTool(BaseTool):
    """XML parser tool for structured document parsing."""
    
    def __init__(
        self,
        parse_mode: str = "pmc",  # "pmc" or "generic"
        parse_format: str = "structure",  # "full" or "structure"
        **kwargs
    ):
        """Initialize XML parser tool."""
        super().__init__(
            name="xml_parser",
            description="Parse XML documents and extract structured content",
            **kwargs
        )
        
        self.parse_mode = parse_mode
        self.parse_format = parse_format
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get parameter schema."""
        return {
            "type": "object",
            "properties": {
                "xml_path": {
                    "type": "string",
                    "description": "Path to XML file, XML string content, or URL to XML document"
                },
                "extract_sections": {
                    "type": "boolean",
                    "description": "Extract document sections (default: True)"
                },
                "extract_tables": {
                    "type": "boolean",
                    "description": "Extract table metadata (default: True)"
                },
                "extract_figures": {
                    "type": "boolean",
                    "description": "Extract figure metadata (default: True)"
                },
                "extract_references": {
                    "type": "boolean",
                    "description": "Extract references/citations (default: True)"
                }
            },
            "required": ["xml_path"]
        }
    
    def _is_url(self, path: str) -> bool:
        """Check if the given path is a URL."""
        try:
            result = urlparse(path)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def _fetch_xml_from_url(self, url: str) -> str:
        """Fetch XML content from URL."""
        try:
            # Add delay to respect rate limits (NCBI requires 3 requests per second max)
            time.sleep(0.34)  # ~3 requests per second
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            raise Exception(f"Failed to fetch XML from URL: {str(e)}")
    
    def execute(
        self,
        xml_path: Union[str, bytes],
        extract_sections: bool = True,
        extract_tables: bool = True,
        extract_figures: bool = True,
        extract_references: bool = True,
        **kwargs
    ) -> ToolResult:
        """Execute XML parsing."""
        logger.info(f"[XMLParser] Parsing XML in '{self.parse_mode}' mode with '{self.parse_format}' format")
        # Validate parameters
        is_valid, error_msg = self.validate_parameters({"xml_path": xml_path})
        if not is_valid:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                formatted=f"Error: {error_msg}",
                error=error_msg
            )
        
        try:
            # Load XML content
            if isinstance(xml_path, str):
                # Check if it's a URL
                if self._is_url(xml_path):
                    logger.info(f"[XMLParser] Fetching XML from URL: {xml_path}")
                    xml_content = self._fetch_xml_from_url(xml_path)
                    file_name = xml_path.split('/')[-1] or "remote.xml"
                # Check if it's a file path
                elif os.path.exists(xml_path):
                    with open(xml_path, 'r', encoding='utf-8') as f:
                        xml_content = f.read()
                    file_name = os.path.basename(xml_path)
                # Assume it's XML string
                else:
                    xml_content = xml_path
                    file_name = "inline.xml"
            else:
                xml_content = xml_path.decode('utf-8')
                file_name = "uploaded.xml"
            
            # Parse XML
            root = ET.fromstring(xml_content)
            
            # Extract content based on mode
            if self.parse_mode == "pmc":
                result_data = self._parse_pmc_xml(
                    root,
                    extract_sections,
                    extract_tables,
                    extract_figures,
                    extract_references
                )
            else:
                result_data = self._parse_generic_xml(root)
            
            result_data["file_name"] = file_name
            
            # Generate formatted output for agent
            formatted_output = self.format_results_for_agent(result_data)
            result_data["formatted"] = formatted_output
            
            logger.info(f"[XMLParser] Extracted title: {result_data.get('title', 'N/A')}")
            logger.info(f"[XMLParser] Extracted abstract: {len(result_data.get('abstract', ''))} characters")
            logger.info(f"[XMLParser] Extracted {len(result_data.get('sections', []))} sections")
            logger.info(f"[XMLParser] Found {len(result_data.get('tables', []))} tables")
            logger.info(f"[XMLParser] Found {len(result_data.get('figures', []))} figures")
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=result_data,
                formatted=formatted_output,
                metadata={
                    "file_name": file_name,
                    "parse_mode": self.parse_mode,
                    "parse_format": self.parse_format,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
        except ET.ParseError as e:
            error_msg = f"XML parsing error: {str(e)}"
            logger.error(f"[XMLParser] Error: {error_msg}")
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                formatted=f"Error: {error_msg}",
                error=error_msg
            )
        except Exception as e:
            error_msg = f"XML parsing failed: {str(e)}"
            logger.error(f"[XMLParser] Error: {error_msg}")
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                formatted=f"Error: {error_msg}",
                error=error_msg
            )
    
    def _parse_pmc_xml(
        self,
        root: ET.Element,
        extract_sections: bool,
        extract_tables: bool,
        extract_figures: bool,
        extract_references: bool
    ) -> Dict[str, Any]:
        """Parse PubMed Central XML format or PubMed XML format."""
        result = {
            "title": "",
            "abstract": "",
            "sections": [],
            "tables": [],
            "figures": [],
            "references": []
        }
        
        # Detect if this is PubMed XML (not PMC) by checking for PubmedArticle
        if root.tag == "PubmedArticleSet" or root.find(".//PubmedArticle") is not None:
            return self._parse_pubmed_xml(root, extract_sections, extract_tables, 
                                         extract_figures, extract_references)
        
        # Extract title (always) - PMC format
        title_elem = root.find(".//article-title")
        if title_elem is not None:
            result["title"] = self._get_element_text(title_elem)
        
        # Extract abstract (always) - PMC format
        abstract_elem = root.find(".//abstract")
        if abstract_elem is not None:
            result["abstract"] = self._get_element_text(abstract_elem)
        
        # Extract sections with hierarchy
        if extract_sections:
            body = root.find(".//body")
            if body is not None:
                sections = self._extract_pmc_sections_hierarchical(body)
                result["sections"] = sections
        
        # Extract tables
        if extract_tables:
            tables = self._extract_pmc_tables(root)
            result["tables"] = tables
        
        # Extract figures
        if extract_figures:
            figures = self._extract_pmc_figures(root)
            result["figures"] = figures
        
        # Extract references
        if extract_references:
            references = self._extract_pmc_references(root)
            result["references"] = references
        
        return result
    
    def _parse_pubmed_xml(
        self,
        root: ET.Element,
        extract_sections: bool,
        extract_tables: bool,
        extract_figures: bool,
        extract_references: bool
    ) -> Dict[str, Any]:
        """Parse PubMed XML format (from efetch.fcgi?db=pubmed)."""
        result = {
            "title": "",
            "abstract": "",
            "sections": [],
            "tables": [],
            "figures": [],
            "references": []
        }
        
        # Find the PubmedArticle element
        article = root.find(".//PubmedArticle")
        if article is None:
            return result
        
        # Extract title
        title_elem = article.find(".//ArticleTitle")
        if title_elem is not None:
            result["title"] = self._get_element_text(title_elem)
        
        # Extract abstract - can have multiple AbstractText elements
        abstract_parts = []
        for abstract_text in article.findall(".//AbstractText"):
            # Check if it has a label attribute
            label = abstract_text.get("Label", "")
            text = self._get_element_text(abstract_text)
            if text:
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)
        
        if abstract_parts:
            result["abstract"] = "\n\n".join(abstract_parts)
        
        # PubMed XML typically doesn't have full text sections/tables/figures
        
        # Extract publication info as a "section" if requested
        if extract_sections:
            pub_info = self._extract_pubmed_publication_info(article)
            if pub_info:
                result["sections"].append(pub_info)
        
        # Extract references/citations if requested
        if extract_references:
            references = self._extract_pubmed_references(article)
            result["references"] = references
        
        return result
    
    def _extract_pubmed_publication_info(self, article: ET.Element) -> Dict[str, Any]:
        """Extract publication information from PubMed XML."""
        info = {
            "level": 0,
            "title": "Publication Information",
            "subsections": []
        }
        
        content_parts = []
        
        # Journal
        journal = article.find(".//Journal")
        if journal is not None:
            journal_title = journal.find(".//Title")
            if journal_title is not None:
                content_parts.append(f"Journal: {self._get_element_text(journal_title)}")
        
        # Authors
        authors = []
        for author in article.findall(".//Author"):
            last_name = author.find(".//LastName")
            fore_name = author.find(".//ForeName")
            if last_name is not None:
                name = self._get_element_text(last_name)
                if fore_name is not None:
                    name = f"{self._get_element_text(fore_name)} {name}"
                authors.append(name)
        
        if authors:
            content_parts.append(f"Authors: {', '.join(authors[:5])}" + 
                               (" et al." if len(authors) > 5 else ""))
        
        # Publication date
        pub_date = article.find(".//PubDate")
        if pub_date is not None:
            year = pub_date.find(".//Year")
            month = pub_date.find(".//Month")
            if year is not None:
                date_str = self._get_element_text(year)
                if month is not None:
                    date_str = f"{self._get_element_text(month)} {date_str}"
                content_parts.append(f"Published: {date_str}")
        
        # PMID
        pmid = article.find(".//PMID")
        if pmid is not None:
            content_parts.append(f"PMID: {self._get_element_text(pmid)}")
        
        # DOI
        for article_id in article.findall(".//ArticleId"):
            if article_id.get("IdType") == "doi":
                content_parts.append(f"DOI: {self._get_element_text(article_id)}")
                break
        
        if self.parse_format == "full":
            info["content"] = "\n".join(content_parts)
        else:
            info["paragraph_count"] = len(content_parts)
        
        return info
    
    def _extract_pubmed_references(self, article: ET.Element) -> List[Dict[str, Any]]:
        """Extract references from PubMed XML."""
        references = []
        
        # PubMed XML includes referenced PMIDs
        for ref_list in article.findall(".//ReferenceList"):
            for reference in ref_list.findall(".//Reference"):
                ref_info = {}
                
                # Get citation text
                citation = reference.find(".//Citation")
                if citation is not None:
                    ref_info["text"] = self._get_element_text(citation)
                    ref_info["preview"] = ref_info["text"][:100] + "..." if len(ref_info["text"]) > 100 else ref_info["text"]
                
                # Get PMID if available
                for article_id in reference.findall(".//ArticleId"):
                    if article_id.get("IdType") == "pubmed":
                        ref_info["pmid"] = self._get_element_text(article_id)
                        ref_info["id"] = f"pmid_{ref_info['pmid']}"
                        break
                
                if not ref_info.get("id"):
                    ref_info["id"] = f"ref{len(references) + 1}"
                
                if ref_info:
                    references.append(ref_info)
        
        return references
    
    def _extract_pmc_sections_hierarchical(self, body: ET.Element, level: int = 0) -> List[Dict[str, Any]]:
        """Extract sections from PMC body with hierarchical structure."""
        sections = []
        
        for sec in body.findall("sec"):  # Only direct children, not all descendants
            section_info = {
                "level": level,
                "title": "",
                "subsections": []
            }
            
            # Get section title
            title_elem = sec.find("title")
            section_info["title"] = self._get_element_text(title_elem) if title_elem is not None else "Untitled"
            
            if self.parse_format == "full":
                # Get section content (only direct paragraphs, not from subsections)
                paragraphs = []
                for p in sec.findall("p"):  # Only direct children
                    p_text = self._get_element_text(p)
                    if p_text:
                        paragraphs.append(p_text)
                
                section_info["content"] = "\n\n".join(paragraphs) if paragraphs else ""
            else:  # structure
                # Count only direct paragraphs
                paragraph_count = len(sec.findall("p"))
                section_info["paragraph_count"] = paragraph_count
            
            # Recursively extract subsections
            subsections = self._extract_pmc_sections_hierarchical(sec, level + 1)
            if subsections:
                section_info["subsections"] = subsections
            
            sections.append(section_info)
        
        return sections
    
    def _extract_pmc_tables(self, root: ET.Element) -> List[Dict[str, Any]]:
        """Extract table metadata from PMC XML based on parse_format."""
        tables = []
        
        for table_wrap in root.findall(".//table-wrap"):
            table_info = {}
            
            # Table ID
            table_info["id"] = table_wrap.get("id", "")
            
            # Table label (e.g., "Table 1")
            label_elem = table_wrap.find("label")
            table_info["label"] = self._get_element_text(label_elem) if label_elem is not None else ""
            
            # Table caption
            caption_elem = table_wrap.find("caption")
            if caption_elem is not None:
                title = caption_elem.find("title")
                para = caption_elem.find("p")
                caption_text = []
                if title is not None:
                    caption_text.append(self._get_element_text(title))
                if para is not None:
                    caption_text.append(self._get_element_text(para))
                table_info["caption"] = " ".join(caption_text)
            else:
                table_info["caption"] = ""
            
            # Table content based on parse_format
            table_elem = table_wrap.find(".//table")
            if table_elem is not None:
                if self.parse_format == "full":
                    # Extract full table content
                    rows = []
                    for tr in table_elem.findall(".//tr"):
                        row = []
                        for td in tr.findall(".//td"):
                            row.append(self._get_element_text(td))
                        for th in tr.findall(".//th"):
                            row.append(self._get_element_text(th))
                        if row:
                            rows.append(row)
                    table_info["content"] = rows
                else:  # structure
                    # Just count rows and columns
                    rows = table_elem.findall(".//tr")
                    row_count = len(rows)
                    col_count = 0
                    if rows:
                        first_row = rows[0]
                        col_count = len(first_row.findall(".//td")) + len(first_row.findall(".//th"))
                    table_info["dimensions"] = {"rows": row_count, "columns": col_count}
            else:
                if self.parse_format == "full":
                    table_info["content"] = []
                else:
                    table_info["dimensions"] = {"rows": 0, "columns": 0}
            
            tables.append(table_info)
        
        return tables
    
    def _extract_pmc_figures(self, root: ET.Element) -> List[Dict[str, Any]]:
        """Extract figure metadata from PMC XML."""
        figures = []
        
        for fig in root.findall(".//fig"):
            fig_info = {}
            
            # Figure ID
            fig_info["id"] = fig.get("id", "")
            
            # Figure label (e.g., "Figure 1")
            label_elem = fig.find("label")
            fig_info["label"] = self._get_element_text(label_elem) if label_elem is not None else ""
            
            # Figure caption
            caption_elem = fig.find("caption")
            if caption_elem is not None:
                title = caption_elem.find("title")
                para = caption_elem.find("p")
                caption_text = []
                if title is not None:
                    caption_text.append(self._get_element_text(title))
                if para is not None:
                    caption_text.append(self._get_element_text(para))
                fig_info["caption"] = " ".join(caption_text)
            else:
                fig_info["caption"] = ""
            
            # Figure graphic reference
            graphic = fig.find(".//graphic")
            if graphic is not None:
                fig_info["graphic_ref"] = graphic.get("{http://www.w3.org/1999/xlink}href", "")
            else:
                fig_info["graphic_ref"] = ""
            
            figures.append(fig_info)
        
        return figures
    
    def _extract_pmc_references(self, root: ET.Element) -> List[Dict[str, str]]:
        """Extract references from PMC XML based on parse_format."""
        references = []
        
        ref_list = root.find(".//ref-list")
        if ref_list is not None:
            for ref in ref_list.findall(".//ref"):
                ref_info = {
                    "id": ref.get("id", "")
                }
                
                # Try to get structured citation
                citation = ref.find(".//mixed-citation") or ref.find(".//element-citation")
                if citation is not None:
                    full_text = self._get_element_text(citation)
                else:
                    full_text = self._get_element_text(ref)
                
                if self.parse_format == "full":
                    ref_info["text"] = full_text
                else:  # structure
                    # Extract only first 100 characters as preview
                    ref_info["preview"] = full_text[:100] + "..." if len(full_text) > 100 else full_text
                
                if full_text:
                    references.append(ref_info)
        
        return references
    
    def _parse_generic_xml(self, root: ET.Element) -> Dict[str, Any]:
        """Parse generic XML structure."""
        return {
            "root_tag": root.tag,
            "attributes": root.attrib,
            "content": self._element_to_dict(root)
        }
    
    def _element_to_dict(self, element: ET.Element) -> Dict[str, Any]:
        """Convert XML element to dictionary."""
        result = {
            "tag": element.tag,
            "attributes": element.attrib,
            "text": element.text.strip() if element.text else "",
            "children": []
        }
        
        for child in element:
            result["children"].append(self._element_to_dict(child))
        
        return result
    
    def _get_element_text(self, element: ET.Element) -> str:
        """Extract all text content from an element and its children."""
        if element is None:
            return ""
        
        # Get all text including from children
        texts = [element.text or ""]
        for child in element.iter():
            if child.text:
                texts.append(child.text)
            if child.tail:
                texts.append(child.tail)
        
        return " ".join(texts).strip()
    
    def format_results_for_agent(self, data: Dict[str, Any]) -> str:
        """Generate formatted output for agent combining all extracted contents."""
        formatted = []

        formatted.append("METADATA:")

        # Tables
        if data.get("tables"):
            formatted.append(f"Total Number of Tables: {len(data['tables'])}")
        
        # Figures
        if data.get("figures"):
            formatted.append(f"Total Number of Figures: {len(data['figures'])}")

        if data.get("sections"):
            formatted.append(f"Total Number of Sections: {len(data['sections'])}")
        
        # Title
        if data.get("title"):
            formatted.append(f"TITLE: {data['title']}")
            formatted.append("")
        
        # Abstract
        if data.get("abstract"):
            formatted.append(f"ABSTRACT: {data['abstract']}")
            formatted.append("")
        
        # Sections with hierarchy
        if data.get("sections"):
            formatted.append("SECTIONS:")
            formatted.append(self._format_sections_hierarchical(data['sections']))
            formatted.append("")
        
        # References
        if data.get("references"):
            formatted.append("REFERENCES:")
            formatted.append(self._format_references(data['references']))
            formatted.append("")
        
        return "\n".join(formatted).strip()
    
    def _format_sections_hierarchical(self, sections: List[Dict[str, Any]], indent: int = 0) -> str:
        """Format hierarchical sections for agent."""
        formatted = []
        
        for section in sections:
            prefix = "  " * indent
            title = section.get("title", "Untitled")
            
            if self.parse_format == "full":
                # Full format with content
                formatted.append(f"{prefix}{'#' + '#' * (indent + 1)} {title}")
                content = section.get("content", "")
                if content:
                    # Indent content
                    for line in content.split("\n"):
                        formatted.append(f"{prefix}  {line}")
            else:
                # Structure format
                para_count = section.get("paragraph_count", 0)
                formatted.append(f"{prefix}{'#' + '#' * (indent + 1)} {title} [{para_count} paragraphs]")
            
            # Recursively format subsections
            if section.get("subsections"):
                formatted.append(self._format_sections_hierarchical(section["subsections"], indent + 1))
        
        return "\n".join(formatted).strip()
    
    def _format_tables(self, tables: List[Dict[str, Any]]) -> str:
        """Format tables for agent."""
        formatted = []
        
        for i, table in enumerate(tables):
            label = table.get('label', f'Table {i+1}')
            caption = table.get('caption', '')
            
            formatted.append(f"[{label}]")
            if caption:
                formatted.append(f"Caption: {caption}")
            
            if 'content' in table:
                # Full format
                content = table.get('content', [])
                if content:
                    formatted.append("Content (first 5 rows):")
                    for row in content[:5]:
                        formatted.append("  " + " | ".join(str(cell) for cell in row))
            elif 'dimensions' in table:
                # Structure format
                dims = table.get('dimensions', {})
                formatted.append(f"Dimensions: {dims.get('rows', 0)} rows × {dims.get('columns', 0)} columns")
            
            formatted.append("")
        
        return "\n".join(formatted).strip()
    
    def _format_figures(self, figures: List[Dict[str, Any]]) -> str:
        """Format figures for agent."""
        formatted = []
        
        for i, figure in enumerate(figures):
            label = figure.get('label', f'Figure {i+1}')
            caption = figure.get('caption', '')
            graphic_ref = figure.get('graphic_ref', '')
            
            formatted.append(f"[{label}]")
            if caption:
                formatted.append(f"Caption: {caption}")
            if graphic_ref:
                formatted.append(f"Graphic: {graphic_ref}")
            formatted.append("")
        
        return "\n".join(formatted).strip()
    
    def _format_references(self, references: List[Dict[str, str]]) -> str:
        """Format references for agent."""
        formatted = []
        
        for i, ref in enumerate(references):
            ref_id = ref.get('id', f'ref{i+1}')
            
            if 'text' in ref:
                # Full format
                formatted.append(f"[{ref_id}] {ref['text']}")
            elif 'preview' in ref:
                # Structure format
                formatted.append(f"[{ref_id}] {ref['preview']}")
        
        return "\n".join(formatted).strip()

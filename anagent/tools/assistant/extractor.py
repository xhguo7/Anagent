"""
Entity extraction tool.
"""

import os
import json
import re
from typing import Dict, Any, List, Optional
from pathlib import Path


from utils.logger import setup_logger
logger = setup_logger(__name__)

from anagent.tools.assistant.assistant_base import BaseLLMTool
from anagent.tools.base import ToolResult, ToolStatus
from datetime import datetime


class EntityExtractionTool(BaseLLMTool):
    """LLM-powered entity extraction tool."""
    
    def __init__(
        self,
        tool_model: str = "openai/gpt-4o-mini",
        temperature: float = 0.1,  # Low temperature for consistent extraction
        max_tokens: int = 2048,
        engine_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Initialize entity extraction tool."""
        super().__init__(
            name="entity_extractor",
            description="Extract named entities, technical terms, and structured information from text",
            tool_model=tool_model,
            temperature=temperature,
            max_tokens=max_tokens,
            engine_kwargs=engine_kwargs,
            **kwargs
        )
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get parameter schema."""
        return {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text as string, local file path, or URL to extract entities from"
                },
                "entity_types": {
                    "type": "array",
                    "description": "Types of entities to extract (e.g., ['PERSON', 'ORG', 'TECH_TERM'])"
                },
                "domain": {
                    "type": "string",
                    "description": "Domain context: 'biomedical', 'technical', 'general', 'legal' (default: general)"
                },
                "structured_output": {
                    "type": "boolean",
                    "description": "Whether to return JSON-structured output (default: True)"
                },
                "include_context": {
                    "type": "boolean",
                    "description": "Include surrounding context for each entity (default: False)"
                }
            },
            "required": ["text"]
        }
    
    def _detect_context_type(self, context: str) -> str:
        """Detect whether context is a string, file path, or URL."""
        # Check if it's a URL
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
            r'localhost|'  # localhost
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # or IP
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        
        if url_pattern.match(context.strip()):
            return 'url'
        
        # Check if it's a file path
        if os.path.exists(context) and os.path.isfile(context):
            return 'file'
        
        # Otherwise, treat as string content
        return 'string'
    
    def _extract_text_from_file(self, file_path: str) -> str:
        """Extract text from local file."""
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        elif file_ext == '.pdf':
            try:
                import PyPDF2
                text = []
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    for page in pdf_reader.pages:
                        text.append(page.extract_text())
                return '\n'.join(text)
            except ImportError:
                raise ImportError("PyPDF2 is required for PDF files. Install with: pip install PyPDF2")
        
        elif file_ext in ['.docx', '.doc']:
            try:
                import docx
                doc = docx.Document(file_path)
                return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
            except ImportError:
                raise ImportError("python-docx is required for DOCX files. Install with: pip install python-docx")
        
        else:
            raise ValueError(f"Unsupported file type: {file_ext}. Supported types: .txt, .pdf, .docx")
    
    def _extract_text_from_url(self, url: str) -> str:
        """Extract text from URL (web page, arxiv, pubmed, etc.)."""
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("requests and beautifulsoup4 are required for URL extraction. Install with: pip install requests beautifulsoup4")
        
        # Handle arxiv URLs
        if 'arxiv.org' in url:
            # Convert to PDF URL if abstract page
            if '/abs/' in url:
                pdf_url = url.replace('/abs/', '/pdf/') + '.pdf'
            else:
                pdf_url = url
            
            # Download and extract PDF
            response = requests.get(pdf_url)
            response.raise_for_status()
            
            try:
                import PyPDF2
                import io
                pdf_file = io.BytesIO(response.content)
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                text = []
                for page in pdf_reader.pages:
                    text.append(page.extract_text())
                return '\n'.join(text)
            except ImportError:
                raise ImportError("PyPDF2 is required for arxiv PDFs. Install with: pip install PyPDF2")
        
        # Handle PubMed URLs
        elif 'pubmed.ncbi.nlm.nih.gov' in url or 'ncbi.nlm.nih.gov/pmc' in url:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract abstract and body
            text_parts = []
            
            # Try to get abstract
            abstract = soup.find('div', class_='abstract') or soup.find('div', id='abstract')
            if abstract:
                text_parts.append(abstract.get_text(strip=True))
            
            # Try to get full text
            article_body = soup.find('div', class_='article-body') or soup.find('div', class_='body')
            if article_body:
                text_parts.append(article_body.get_text(strip=True))
            
            return '\n\n'.join(text_parts) if text_parts else soup.get_text(strip=True)
        
        # Handle general web pages
        else:
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text
    
    def _process_context(self, context: str) -> tuple[str, str]:
        """Process context based on its type."""
        context_type = self._detect_context_type(context)
        
        logger.info(f"[EntityExtractor] Context type detected: {context_type}")
        if context_type == 'string':
            return context, 'string'
        
        elif context_type == 'file':
            logger.info(f"[EntityExtractor] Extracting text from file: {context}")
            text = self._extract_text_from_file(context)
            return text, 'file'
        
        elif context_type == 'url':
            logger.info(f"[EntityExtractor] Extracting text from URL: {context}")
            text = self._extract_text_from_url(context)
            return text, 'url'
        
        else:
            raise ValueError(f"Unknown context type: {context_type}")
    
    def execute(
        self,
        text: str,
        entity_types: Optional[List[str]] = None,
        domain: str = "general",
        structured_output: bool = True,
        include_context: bool = False,
        **kwargs
    ) -> ToolResult:
        """Execute entity extraction."""
        logger.info(f"[EntityExtractor] Processing input...")
        # Validate parameters
        is_valid, error_msg = self.validate_parameters({"text": text})
        if not is_valid:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                formatted=None,
                error=error_msg
            )
        
        try:
            # Process context to extract text
            processed_text, context_type = self._process_context(text)
            
            logger.info(f"[EntityExtractor] Extracting entities from {len(processed_text)} chars")
            # Build prompt
            messages = self._build_prompt(
                text=processed_text,
                entity_types=entity_types,
                domain=domain,
                structured_output=structured_output,
                include_context=include_context
            )
            
            # Generate extraction
            response = self._call_llm(messages)
            
            # Parse response if structured output requested
            entities = response
            if structured_output:
                try:
                    # Try to parse JSON from response
                    entities = json.loads(response)
                except json.JSONDecodeError:
                    # If not valid JSON, try to extract JSON from markdown code blocks
                    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
                    if json_match:
                        entities = json.loads(json_match.group(1))
                    else:
                        entities = {"raw_response": response}
            
            entity_count = len(entities) if isinstance(entities, dict) else 0
            logger.info(f"[EntityExtractor] Extracted entities")
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "entities": entities,
                    "domain": domain,
                    "text_length": len(processed_text),
                    "context_type": context_type,
                    "context_source": text if context_type in ['file', 'url'] else None,
                    "model": self.tool_model
                },
                formatted=response,
                metadata={
                    "domain": domain,
                    "entity_types": entity_types or "all",
                    "context_type": context_type,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            error_msg = f"Entity extraction failed: {str(e)}"
            logger.info(f"[EntityExtractor] Error: {error_msg}")
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                formatted=None,
                error=error_msg
            )
    
    def _build_prompt(
        self,
        text: str,
        entity_types: Optional[List[str]],
        domain: str,
        structured_output: bool,
        include_context: bool
    ) -> List[Dict[str, Any]]:
        """Build prompt for entity extraction."""
        
        # Domain-specific entity types
        domain_entities = {
            "biomedical": ["GENE", "PROTEIN", "DISEASE", "DRUG", "CELL_TYPE", "ORGANISM"],
            "technical": ["ALGORITHM", "FRAMEWORK", "METRIC", "DATASET", "MODEL", "METHOD"],
            "general": ["PERSON", "ORGANIZATION", "LOCATION", "DATE", "CONCEPT"],
            "legal": ["STATUTE", "CASE", "PARTY", "JURISDICTION", "LEGAL_CONCEPT"]
        }
        
        # Determine entity types to extract
        if entity_types:
            types_to_extract = entity_types
        else:
            types_to_extract = domain_entities.get(domain, domain_entities["general"])
        
        # Build system prompt
        system_prompt = f"You are an expert entity extraction system specialized in {domain} domain. Extract named entities accurately and comprehensively."
        
        # Build user instruction
        instruction = f"Extract the following types of entities from the text: {', '.join(types_to_extract)}\n\n"
        
        if structured_output:
            instruction += """Provide the output as a JSON object with this structure:
{
    "entity_type": [
        {
            "text": "entity text",
            "type": "ENTITY_TYPE"
"""
            if include_context:
                instruction += ',\n            "context": "surrounding text"'
            
            instruction += """
        }
    ]
}

Return ONLY the JSON, no additional text."""
        else:
            instruction += "List each entity with its type, one per line."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{instruction}\n\nText:\n{text}"}
        ]
        
        return messages
    
    def extract_from_paper(
        self,
        abstract: str,
        sections: Dict[str, str],
        focus_on: str = "biomedical",
        max_len: int = 1000,
    ) -> ToolResult:
        """Specialized extraction for academic papers."""
        # Process abstract if it's a file or URL
        processed_abstract, _ = self._process_context(abstract)
        
        # Combine abstract and key sections
        paper_text = f"Abstract: {processed_abstract}\n\n"
        
        # Prioritize certain sections
        priority_sections = ["Methods", "Results", "Introduction"]
        for section_name in priority_sections:
            if section_name in sections:
                # Process each section (could also be file/URL)
                if isinstance(sections[section_name], str):
                    processed_section, _ = self._process_context(sections[section_name])
                    paper_text += f"{section_name}: {processed_section[:max_len]}\n\n"
                else:
                    paper_text += f"{section_name}: {sections[section_name][:max_len]}\n\n"
        
        return self.execute(
            text=paper_text,
            domain=focus_on,
            structured_output=True
        )

    def format_results_for_agent(self, result: ToolResult, max_entities_per_type: int = 10) -> str:
        """Format extraction results for agent consumption."""
        if result.status != ToolStatus.SUCCESS:
            return f"Error: {result.error}"
        
        data = result.data
        formatted = []
        
        # Header with metadata
        domain = data.get("domain", "general").title()
        text_length = data.get("text_length", 0)
        context_type = data.get("context_type", "unknown")
        
        formatted.append(f"=== Entity Extraction ({domain} Domain) ===")
        formatted.append(f"Source Type: {context_type.upper()}")
        
        if data.get("context_source"):
            source = data["context_source"]
            # Truncate long URLs/paths
            if len(source) > 80:
                source = source[:40] + "..." + source[-37:]
            formatted.append(f"Source: {source}")
        
        formatted.append(f"Text Length: {text_length:,} characters")
        formatted.append("")
        
        # The actual entities
        entities = data.get("entities", {})
        
        if isinstance(entities, dict):
            # Check if it's structured JSON output
            if any(isinstance(v, list) for v in entities.values()):
                # Group entities by type
                total_entities = sum(len(v) for v in entities.values() if isinstance(v, list))
                formatted.append(f"Extracted Entities ({total_entities} total):")
                formatted.append("")
                
                for entity_type, entity_list in entities.items():
                    if isinstance(entity_list, list) and entity_list:
                        formatted.append(f"## {entity_type} ({len(entity_list)} found)")
                        
                        # Show limited entities per type
                        for i, entity in enumerate(entity_list[:max_entities_per_type], 1):
                            if isinstance(entity, dict):
                                text = entity.get("text", "")
                                entity_type_detail = entity.get("type", entity_type)
                                
                                formatted.append(f"  {i}. {text} [{entity_type_detail}]")
                                
                                # Include context if available
                                if entity.get("context"):
                                    context = entity["context"]
                                    if len(context) > 100:
                                        context = context[:100].rsplit(' ', 1)[0] + "..."
                                    formatted.append(f"     Context: ...{context}...")
                            else:
                                # Simple string entity
                                formatted.append(f"  {i}. {entity}")
                        
                        if len(entity_list) > max_entities_per_type:
                            formatted.append(f"  ... and {len(entity_list) - max_entities_per_type} more")
                        
                        formatted.append("")
            
            elif "raw_response" in entities:
                # Fallback: LLM returned non-JSON response
                formatted.append("Entities (unstructured format):")
                formatted.append(entities["raw_response"])
            
            else:
                # Unknown structure, just show the dict
                formatted.append("Entities:")
                import json
                try:
                    formatted.append(json.dumps(entities, indent=2))
                except:
                    formatted.append(str(entities))
        
        else:
            # Plain text or other format
            formatted.append("Entities:")
            formatted.append(str(entities))
        
        return "\n".join(formatted)
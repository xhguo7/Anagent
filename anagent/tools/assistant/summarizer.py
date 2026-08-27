"""
Summarization tool.
"""

import os
import re
from typing import Dict, Any, List, Optional
from pathlib import Path


from utils.logger import setup_logger
logger = setup_logger(__name__)

from anagent.tools.assistant.assistant_base import BaseLLMTool
from anagent.tools.base import ToolResult, ToolStatus
from datetime import datetime


class SummarizationTool(BaseLLMTool):
    """LLM-powered summarization tool."""
    
    def __init__(
        self,
        tool_model: str = "openai/gpt-4o-mini",
        temperature: float = 0.3,
        max_tokens: int = 2048,
        engine_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Initialize summarization tool."""
        super().__init__(
            name="summarizer",
            description="Generate summaries of documents, papers, or long text using LLM",
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
                    "description": "Text as string, local file path, or URL to summarize"
                },
                "summary_type": {
                    "type": "string",
                    "description": "Type of summary: 'abstract', 'detailed', 'bullets', 'technical' (default: abstract)"
                },
                "max_length": {
                    "type": "integer",
                    "description": "Target summary length in words (approximate)"
                },
                "focus_areas": {
                    "type": "array",
                    "description": "Specific areas to focus on (e.g., ['methodology', 'results'])"
                },
                "custom_instruction": {
                    "type": "string",
                    "description": "Custom instruction for summarization"
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
        
        logger.info(f"[Summarizer] Context type detected: {context_type}")
        if context_type == 'string':
            return context, 'string'
        
        elif context_type == 'file':
            logger.info(f"[Summarizer] Extracting text from file: {context}")
            text = self._extract_text_from_file(context)
            return text, 'file'
        
        elif context_type == 'url':
            logger.info(f"[Summarizer] Extracting text from URL: {context}")
            text = self._extract_text_from_url(context)
            return text, 'url'
        
        else:
            raise ValueError(f"Unknown context type: {context_type}")
    
    def execute(
        self,
        text: str,
        summary_type: str = "abstract",
        max_length: Optional[int] = None,
        focus_areas: Optional[List[str]] = None,
        custom_instruction: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """Execute summarization."""
        logger.info(f"[Summarizer] Processing input for {summary_type} summary...")
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
            
            logger.info(f"[Summarizer] Generating {summary_type} summary (length: {len(processed_text)} chars)")
            # Build prompt
            messages = self._build_prompt(
                text=processed_text,
                summary_type=summary_type,
                max_length=max_length,
                focus_areas=focus_areas,
                custom_instruction=custom_instruction
            )
            
            # Generate summary
            summary = self._call_llm(messages)
            
            logger.info(f"[Summarizer] Generated summary: {len(summary)} chars")
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "summary": summary,
                    "summary_type": summary_type,
                    "original_length": len(processed_text),
                    "summary_length": len(summary),
                    "compression_ratio": round(len(summary) / len(processed_text), 2),
                    "context_type": context_type,
                    "context_source": text if context_type in ['file', 'url'] else None,
                    "model": self.tool_model
                },
                formatted=summary,
                metadata={
                    "summary_type": summary_type,
                    "context_type": context_type,
                    "timestamp": datetime.now().isoformat(),
                    "model": self.tool_model
                }
            )
            
        except Exception as e:
            error_msg = f"Summarization failed: {str(e)}"
            logger.info(f"[Summarizer] Error: {error_msg}")
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                formatted=None,
                error=error_msg
            )
    
    def _build_prompt(
        self,
        text: str,
        summary_type: str,
        max_length: Optional[int],
        focus_areas: Optional[List[str]],
        custom_instruction: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Build prompt for summarization."""
        
        # Base system prompt
        system_prompts = {
            "abstract": "You are an expert at creating concise, informative abstracts. Generate a brief 1-2 paragraph summary that captures the essence of the content.",
            "detailed": "You are an expert at creating comprehensive summaries. Generate a detailed summary that covers all major points, key findings, and important details.",
            "bullets": "You are an expert at extracting key takeaways. Generate a bulleted list of the most important points from the content.",
            "technical": "You are an expert at technical summarization. Focus on methodology, technical approach, results, and implications. Use precise technical language."
        }
        
        system_prompt = system_prompts.get(summary_type, system_prompts["abstract"])
        
        # Build user instruction
        instruction_parts = []
        
        if custom_instruction:
            instruction_parts.append(custom_instruction)
        else:
            instruction_parts.append(f"Please provide a {summary_type} summary of the following text.")
        
        if max_length:
            instruction_parts.append(f"Target length: approximately {max_length} words.")
        
        if focus_areas:
            focus_str = ", ".join(focus_areas)
            instruction_parts.append(f"Focus particularly on: {focus_str}.")
        
        # Format based on summary type
        if summary_type == "bullets":
            instruction_parts.append("Format the summary as clear, concise bullet points.")
        elif summary_type == "technical":
            instruction_parts.append("Include technical details, methodologies, and quantitative results where applicable.")
        
        instruction = " ".join(instruction_parts)
        
        # Build messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{instruction}\n\nText to summarize:\n\n{text}"}
        ]
        
        return messages
    
    def summarize_paper(
        self,
        title: str,
        abstract: str,
        sections: Dict[str, str],
        summary_type: str = "detailed"
    ) -> ToolResult:
        """Specialized method for summarizing academic papers."""
        # Process title if it's a file or URL (edge case, but handled)
        processed_title, _ = self._process_context(title)
        
        # Process abstract if it's a file or URL
        processed_abstract, _ = self._process_context(abstract)
        
        # Construct paper text
        paper_text = f"Title: {processed_title}\n\n"
        paper_text += f"Abstract: {processed_abstract}\n\n"
        
        # Process each section
        for section_name, section_text in sections.items():
            if isinstance(section_text, str):
                processed_section, _ = self._process_context(section_text)
                paper_text += f"## {section_name}\n{processed_section}\n\n"
            else:
                paper_text += f"## {section_name}\n{section_text}\n\n"
        
        # Generate summary with focus on paper structure
        return self.execute(
            text=paper_text,
            summary_type=summary_type,
            focus_areas=["main findings", "methodology", "conclusions"]
        )
    
    def format_results_for_agent(self, result: ToolResult, max_length: int = 2000) -> str:
        """Format summarization results for agent consumption."""
        if result.status != ToolStatus.SUCCESS:
            return f"Summarization Error: {result.error}"
        
        data = result.data
        formatted = []
        
        # Header with metadata
        summary_type = data.get("summary_type", "abstract").title()
        original_len = data.get("original_length", 0)
        summary_len = data.get("summary_length", 0)
        compression = data.get("compression_ratio", 0)
        context_type = data.get("context_type", "unknown")
        
        formatted.append(f"=== {summary_type} Summary ===")
        formatted.append(f"Source Type: {context_type.upper()}")
        
        if data.get("context_source"):
            source = data["context_source"]
            # Truncate very long URLs/paths for readability
            if len(source) > 80:
                source = source[:40] + "..." + source[-37:]
            formatted.append(f"Source: {source}")
        
        formatted.append(f"Original: {original_len:,} chars → Summary: {summary_len:,} chars (compression: {compression:.1%})")
        formatted.append("")
        
        # The actual summary content
        summary_text = data.get("summary", "")
        
        # Truncate if summary is extremely long (shouldn't happen, but just in case)
        if len(summary_text) > max_length:
            summary_text = summary_text[:max_length] + "\n\n[Summary truncated for length...]"
        
        formatted.append("Summary Content:")
        formatted.append(summary_text)
        
        return "\n".join(formatted)
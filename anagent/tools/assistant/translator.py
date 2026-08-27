"""
Translation tool.
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


class TranslationTool(BaseLLMTool):
    """LLM-powered translation tool with context awareness."""
    
    def __init__(
        self,
        tool_model: str = "openai/gpt-4o-mini",
        temperature: float = 0.3,
        max_tokens: int = 2048,
        engine_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Initialize translation tool."""
        super().__init__(
            name="translator",
            description="Translate text between languages with context awareness",
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
                    "description": "Text as string, local file path, or URL to translate"
                },
                "source_lang": {
                    "type": "string",
                    "description": "Source language (e.g., 'en', 'zh', 'auto' for auto-detect)"
                },
                "target_lang": {
                    "type": "string",
                    "description": "Target language (e.g., 'en', 'zh', 'fr')"
                },
                "domain": {
                    "type": "string",
                    "description": "Domain/field: 'technical', 'medical', 'legal', 'general' (default: general)"
                },
                "preserve_terms": {
                    "type": "array",
                    "description": "Technical terms to preserve untranslated"
                },
                "style": {
                    "type": "string",
                    "description": "Translation style: 'formal', 'casual', 'academic' (default: formal)"
                }
            },
            "required": ["text", "target_lang"]
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
        
        logger.info(f"[Translator] Context type detected: {context_type}")
        if context_type == 'string':
            return context, 'string'
        
        elif context_type == 'file':
            logger.info(f"[Translator] Extracting text from file: {context}")
            text = self._extract_text_from_file(context)
            return text, 'file'
        
        elif context_type == 'url':
            logger.info(f"[Translator] Extracting text from URL: {context}")
            text = self._extract_text_from_url(context)
            return text, 'url'
        
        else:
            raise ValueError(f"Unknown context type: {context_type}")
    
    def execute(
        self,
        text: str,
        target_lang: str,
        source_lang: str = "auto",
        domain: str = "general",
        preserve_terms: Optional[List[str]] = None,
        style: str = "formal",
        **kwargs
    ) -> ToolResult:
        """Execute translation."""
        logger.info(f"[Translator] Processing input for translation to {target_lang}...")
        # Validate parameters
        is_valid, error_msg = self.validate_parameters({
            "text": text,
            "target_lang": target_lang
        })
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
            
            logger.info(f"[Translator] Translating {len(processed_text)} chars to {target_lang}")
            # Build prompt
            messages = self._build_prompt(
                text=processed_text,
                source_lang=source_lang,
                target_lang=target_lang,
                domain=domain,
                preserve_terms=preserve_terms,
                style=style
            )
            
            # Generate translation
            translation = self._call_llm(messages)
            
            logger.info(f"[Translator] Generated translation: {len(translation)} chars")
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "original_text": processed_text,
                    "translated_text": translation,
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "domain": domain,
                    "context_type": context_type,
                    "context_source": text if context_type in ['file', 'url'] else None,
                    "model": self.tool_model
                },
                formatted=translation,
                metadata={
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "domain": domain,
                    "context_type": context_type,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            error_msg = f"Translation failed: {str(e)}"
            logger.error(f"[Translator] Error: {error_msg}")
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                formatted=None,
                error=error_msg
            )
    
    def _build_prompt(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        domain: str,
        preserve_terms: Optional[List[str]],
        style: str
    ) -> List[Dict[str, Any]]:
        """Build prompt for translation."""
        
        # Domain-specific system prompts
        domain_prompts = {
            "technical": "You are an expert technical translator. Preserve technical terminology and maintain precision.",
            "medical": "You are an expert medical translator. Maintain medical accuracy and use proper medical terminology.",
            "legal": "You are an expert legal translator. Preserve legal terminology and formal language structure.",
            "general": "You are an expert translator. Provide natural, accurate translations."
        }
        
        system_prompt = domain_prompts.get(domain, domain_prompts["general"])
        
        # Build translation instruction
        lang_instruction = f"Translate the following text to {target_lang}"
        if source_lang != "auto":
            lang_instruction += f" from {source_lang}"
        lang_instruction += "."
        
        # Add style guidance
        style_guidance = {
            "formal": "Use formal language and professional tone.",
            "casual": "Use casual, natural language.",
            "academic": "Use academic style appropriate for scholarly work."
        }
        
        instruction = f"{lang_instruction} {style_guidance.get(style, '')}"
        
        # Add term preservation if specified
        if preserve_terms:
            terms_str = ", ".join(preserve_terms)
            instruction += f"\n\nPreserve these technical terms unchanged: {terms_str}"
        
        instruction += "\n\nProvide only the translation, without explanations."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{instruction}\n\nText to translate:\n{text}"}
        ]
        
        return messages

    def format_results_for_agent(self, result: ToolResult) -> str:
        """Format translation results for agent consumption."""
        if result.status != ToolStatus.SUCCESS:
            return f"Translation failed: {result.error}"
        
        data = result.data
        formatted = []
        
        # Header with translation metadata
        source_lang = data.get("source_lang", "auto").upper()
        target_lang = data.get("target_lang", "unknown").upper()
        domain = data.get("domain", "general").title()
        context_type = data.get("context_type", "unknown")
        
        formatted.append(f"=== Translation ({source_lang} → {target_lang}) ===")
        formatted.append(f"Domain: {domain}")
        formatted.append(f"Source Type: {context_type.upper()}")
        
        # Add source reference if it's a file or URL
        if data.get("context_source"):
            source = data["context_source"]
            # Truncate long URLs/paths for readability
            if len(source) > 80:
                source = source[:40] + "..." + source[-37:]
            formatted.append(f"Source: {source}")
        
        # Add text length statistics
        original_text = data.get("original_text", "")
        translated_text = data.get("translated_text", "")
        
        if original_text:
            formatted.append(f"Original Length: {len(original_text):,} characters")
        if translated_text:
            formatted.append(f"Translated Length: {len(translated_text):,} characters")
        
        formatted.append("")
        
        # The actual translated text
        formatted.append("Translation:")
        formatted.append(translated_text)
        
        # Add model info for reference
        if data.get("model"):
            formatted.append("")
            formatted.append(f"[Model: {data['model']}]")
        
        return "\n".join(formatted)
"""
Question answering tool.
"""

import os
from typing import Dict, Any, List, Optional
from pathlib import Path
import re


from utils.logger import setup_logger
logger = setup_logger(__name__)

from anagent.tools.assistant.assistant_base import BaseLLMTool
from anagent.tools.base import ToolResult, ToolStatus
from datetime import datetime


class QATool(BaseLLMTool):
    """LLM-powered question answering tool."""
    
    def __init__(
        self,
        tool_model: str = "openai/gpt-4o-mini",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        engine_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Initialize QA tool."""
        super().__init__(
            name="qa_tool",
            description="Answer questions about documents or provided context using LLM",
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
                "question": {
                    "type": "string",
                    "description": "Question to answer"
                },
                "context": {
                    "type": "string",
                    "description": "Context as string, local file path, or URL (supports pdf, docx, txt, web pages, arxiv, pubmed, etc.)"
                },
                "answer_style": {
                    "type": "string",
                    "description": "Answer style: 'concise', 'detailed', 'technical' (default: concise)"
                },
                "citation_required": {
                    "type": "boolean",
                    "description": "Whether to include citations/evidence (default: True)"
                },
                "previous_qa": {
                    "type": "array",
                    "description": "Previous Q&A pairs for context"
                }
            },
            "required": ["question", "context"]
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
        
        logger.info(f"[QA] Context type detected: {context_type}")
        if context_type == 'string':
            return context, 'string'
        
        elif context_type == 'file':
            logger.info(f"[QA] Extracting text from file: {context}")
            text = self._extract_text_from_file(context)
            return text, 'file'
        
        elif context_type == 'url':
            logger.info(f"[QA] Extracting text from URL: {context}")
            text = self._extract_text_from_url(context)
            return text, 'url'
        
        else:
            raise ValueError(f"Unknown context type: {context_type}")
    
    def execute(
        self,
        question: str,
        context: str,
        answer_style: str = "concise",
        citation_required: bool = True,
        previous_qa: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> ToolResult:
        """Execute question answering."""
        logger.info(f"[QA] Question: {question[:100]}...")
        # Validate parameters
        if not question or not question.strip():
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                formatted=None,
                error="Question cannot be empty"
            )
        
        if not context or not context.strip():
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                formatted=None,
                error="Context cannot be empty"
            )
        
        # Validate answer_style
        if answer_style not in ["concise", "detailed", "technical"]:
            logger.warning(f"[QA] Invalid answer_style '{answer_style}', using 'concise'")
            answer_style = "concise"
        
        try:
            # Process context to extract text
            processed_context, context_type = self._process_context(context)
            
            logger.info(f"[QA] Processed context length: {len(processed_context)}")
            logger.info(f"[QA] Answer style: {answer_style}")
            logger.info(f"[QA] Citation required: {citation_required}")
            # Build prompt
            messages = self._build_prompt(
                question=question,
                context=processed_context,
                answer_style=answer_style,
                citation_required=citation_required,
                previous_qa=previous_qa
            )
            
            # Generate answer
            answer = self._call_llm(messages)
            
            logger.info(f"[QA] Generated answer length: {len(answer)}")
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "question": question,
                    "answer": answer,
                    "answer_style": answer_style,
                    "citation_required": citation_required,
                    "context_length": len(processed_context),
                    "context_type": context_type,
                    "context_source": context if context_type in ['file', 'url'] else None,
                    "model": self.tool_model
                },
                formatted=answer,
                metadata={
                    "answer_style": answer_style,
                    "context_type": context_type,
                    "timestamp": datetime.now().isoformat(),
                    "model": self.tool_model
                }
            )
            
        except Exception as e:
            error_msg = f"QA failed: {str(e)}"
            logger.error(f"[QA] Error: {error_msg}")
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                formatted=None,
                error=error_msg
            )
    
    def _build_prompt(
        self,
        question: str,
        context: str,
        answer_style: str,
        citation_required: bool,
        previous_qa: Optional[List[Dict[str, str]]]
    ) -> List[Dict[str, Any]]:
        """Build prompt for QA."""
        
        # Style-specific system prompts
        style_prompts = {
            "concise": "You are a helpful assistant that provides concise, accurate answers based on the given context. Keep answers brief and right to the point.",
            "detailed": "You are an expert assistant that provides comprehensive, detailed answers based on the given context. Include relevant details and explanations.",
            "technical": "You are a technical expert that provides precise, technical answers based on the given context. Use technical terminology and include specific details."
        }
        
        system_prompt = style_prompts.get(answer_style, style_prompts["concise"])
        
        if citation_required:
            system_prompt += " Always support your answers with evidence from the context."
        
        # Build user message
        user_message = f"Context:\n{context}\n\n"
        
        # Add previous Q&A if provided
        if previous_qa:
            user_message += "Previous conversation:\n"
            for qa_pair in previous_qa:
                user_message += f"Q: {qa_pair.get('question', '')}\n"
                user_message += f"A: {qa_pair.get('answer', '')}\n\n"
        
        user_message += f"Question: {question}\n\nAnswer:"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        return messages
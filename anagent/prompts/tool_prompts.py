"""
Tool prompts.
"""

import re
from typing import Dict, List


SEARCH_TOOLS = """
**Search Tools:**
* **web** - Search the web for general information: <tool>web</tool><query>your search query</query>
* **pubmed** - Search PubMed database for biomedical literature: <tool>pubmed</tool><query>your search query</query>
* **wikipedia** - Search Wikipedia for encyclopedic information: <tool>wikipedia</tool><query>your search query</query>
* **arxiv** - Search arXiv papers: <tool>arxiv</tool><query>arxiv paper keyword</query>
* **semantic_scholar** - Search Semantic Scholar papers: <tool>semantic_scholar</tool><query>semantic scholar paper keyword</query>
""".strip()



DOCUMENT_TOOLS = """
**Document Processing Tools:**
* **pdf_parser** - Parse and extract the structure of a PDF document: <tool>pdf_parser</tool><query>PDF path, or URL, or NONE to use task source</query>
* **xml_parser** - Parse and extract the structure of a XML document: <tool>xml_parser</tool><query>XML file path, or URL, or XML content string, or NONE to use task source</query>
* **online_fetcher** - Fetch and parse document from online sources: <tool>online_fetcher</tool><query>URL (html/website/xml/pdf/arxiv/semantic scholar/blog/general/etc.), or NONE to use task source</query>
""".strip()



KNOWLEDGE_TOOLS = r"""
**Knowledge Tools:**
* **info_localizer** - Localize and extract complete content from the specified document: <tool>info_localizer</tool><query>text to find (e.g., keyword, phrase, etc.)</query><context>URL, or local file path, or NONE to use task source</context>
* **context_finder** - Find and retrieve all related contexts (references to/from) for specified contents in the document: <tool>context_finder</tool><query>text to find all its related contexts (e.g., keyword, phrase, etc.)</query><context>URL, or local file path, or NONE to use task source</context>
* **section_extractor** - Extract complete sections/figures/tables as specified: <tool>section_extractor</tool><query>section specification (e.g., Section 3.2, 3.2.5, Section 4.1.3, Figure 1, Table 3, Algorithm 2, Equation 5, Abstract, Introduction, Conclusion, Methodology, Results, 3.2 Attention Mechanism, 5.2.3 Ablation Results, 6. Conclusion, Appendix A, Appendix B, etc.) WITHOUT TITLE</query><context>URL, or local path, or NONE to use task source</context>
* **abstract_collector** - Collect paper abstracts of all specified papers: <tool>abstract_collector</tool><papers>a list of paper dictionaries where each dictionary stores a paper metadata, e.g., [{"title": "Paper Title", "author": "Author Name", "year": "2025", "url": "https://arxiv.org/abs/XXXX.XXXXX"}, ...]</papers>
""".strip()



VISION_TOOLS = """
**Vision Tools:**
* **ocr** - Extract text from images using Optical Character Recognition (OCR): <tool>ocr</tool><query>image path, or image URL, or NONE to use task source</query>
* **figure_parser** - Parse and extract structured information from scientific figures/charts/plots: <tool>figure_parser</tool><query>image path, or image URL, or NONE to use task source</query>
* **image_analyzer** - Analyze and describe images using vision-language models: <tool>image_analyzer</tool><query>image path, or image URL, or NONE to use task source</query><task>"describe" (default, general description), or "detect" (identify objects), or "qa" (visual question answering), or "chart" (analyze figures/charts), or "text" (extract text)</task><question>your question (ONLY VALID WHEN you set task to "qa")</question><focus>text describing what to focus on, (e.g., objects, text, colors, layout, people, white animals, etc.)</focus><detail_level>"brief" (concise summary), or "detailed" (default, thorough analysis), or "comprehensive" (comprehensively detailed)</detail_level><context>any image contexts you want to provide, or leave blank</context>
""".strip()



SANDBOX_TOOLS = """
**Self-Exploration Tools:**
* **sandbox_explore** - Execute custom Python code in isolated sandbox for self-exploration and analysis: <tool>sandbox_explore</tool><code>your Python code here</code><install>a list of required packages (e.g., ["scikit-learn", "scipy>=1.7.0", "plotly"])</install>
    - Execution Environment:
        - Python 3.12 in isolated sandbox
        - Working directory: /workspace
        - Task data available in: /workspace/data/
        - Save outputs to: /workspace/output/
    - Important Notes:
        - Code runs in isolated environment - cannot access external file systems
        - Use helper functions to access task data (don't try to read files directly outside "/workspace")
        - Code execution has timeout (5 minutes default) and memory limits (2GB)
        - Use print() to see output - all stdout/stderr captured and returned
        - Network access available for API calls and downloads
        - Code should be self-contained and complete
        - Avoid infinite loops or excessive resource consumption
""".strip()



ASSISTANT_TOOLS = """
**MLLM-powered Multimodal Language Processing Tools**:
* Generate summaries of documents, papers, or text: <tool>summarizer</tool><query>text, or local path, or URL, or NONE to use task source</query><summary_type>"abstract" (default, 1-2 paragraphs), "detailed" (comprehensive with key points), "bullets" (key takeaways list), "technical" (methodology and results focus)</summary_type><focus_areas>a list of focus topic, (e.g., ["methodology", "results", "conclusions"])</focus_areas><custom_instruction>text instruction to guide LLM summarization (e.g., Summarize from a clinical perspective)</custom_instruction>
* Extract named entities, technical terms, and structured information from text: <tool>extractor</tool><query>text, or local path, or URL, or NONE to use task source</query><entity_types>a list of domain-specific entity types (e.g., ["GENE", "PROTEIN", "DISEASE", "DRUG", "CELL_TYPE", "ORGANISM"], ["ALGORITHM", "FRAMEWORK", "METRIC", "DATASET", "MODEL", "METHOD"], ["PERSON", "ORGANIZATION", "LOCATION", "DATE", "CONCEPT"], etc.)</entity_types><domain>"general" (default), "biomedical", "technical", "legal", etc.</domain><structured_output>true (default, returns JSON structured output) OR false (returns plain text)</structured_output><include_context>false (default, entity text only) OR true (includes surrounding text context)</include_context>
* Translate text between languages with context awareness: <tool>translator</tool><query>target language</query><context>text, or local path, or URL, or NONE to use task source</context><source_language>(e.g., 'en', 'zh', 'es', 'fr', 'de', 'ja', 'ko') or 'auto' (default, auto-detect)</source_language><domain>'general' (default), 'technical', 'medical', 'legal', etc.</domain><style>'formal' (default), OR 'casual', OR 'academic'</style><preserve_terms>a list of terms to keep untranslated (e.g., ["API", "HTTP", "JSON"])</preserve_terms>
* Answer your questions based on your provided contexts: <tool>qa</tool><query>your question</query><context>text, or local path, or URL, or NONE to use task source</context>><answer_style>'concise' (default, brief answers), OR 'detailed' (comprehensive explanations), OR 'technical' (precise technical details)</answer_style><citation_required>true (default, include evidence), OR false (answer only, no citations)</citation_required>
""".strip()



def _filter_tool_block(block: str, tools: Dict) -> str:
    """Keep only the bullets whose tool is registered; drop the block if none remain."""
    header, groups, current = [], [], None
    for line in block.split("\n"):
        match = re.match(r"\*\s*\*\*([A-Za-z_][\w]*)\*\*", line)
        if match:
            current = (match.group(1), [line])
            groups.append(current)
        elif current is not None:
            current[1].append(line)
        else:
            header.append(line)
    kept = [line for name, body in groups if name in tools for line in body]
    if not kept:
        return ""
    return "\n".join(header + kept).strip()


def get_tool_prompts(tools: Dict) -> str:
    """Get descriptions of the registered tools only."""
    blocks = [
        KNOWLEDGE_TOOLS,
        DOCUMENT_TOOLS,
        VISION_TOOLS,
        SEARCH_TOOLS,
        SANDBOX_TOOLS,
        ASSISTANT_TOOLS,
    ]
    descriptions = [b for b in (_filter_tool_block(x, tools) for x in blocks) if b]

    return "\n\n".join(descriptions)
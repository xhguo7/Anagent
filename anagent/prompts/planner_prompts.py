"""
Planner Agent prompts.
"""

# Tool description for planning
TOOL_DESCRIPTION = """
## 1. Search Tools
* **web**: Search the web for general information, current events, and recent news (supports date/language filters)
* **pubmed**: Search biomedical literature in PubMed database with abstracts and metadata
* **wikipedia**: Search Wikipedia for encyclopedic knowledge and factual information
* **arxiv**: Search arXiv repository for scientific preprints and research papers

## 2. Document Processing Tools
* **pdf_parser**: Extract and parse content from PDF documents (structure, text, tables, metadata)
* **xml_parser**: Parse XML documents, especially PubMed Central articles with structured content
* **online_fetcher**: Fetch and parse documents from URLs (auto-detects arXiv, PubMed, PDFs, HTML)

## 3. Multimodal Language Processing Tools
* **summarizer**: Generate summaries of documents with customizable styles (abstract, detailed, bullets, technical)
* **extractor**: Extract named entities and structured information using domain-specific recognition
* **translator**: Translate text between languages with domain awareness and terminology preservation
* **qa**: Answer questions based on provided context with citation support and multi-turn conversation

## 4. Vision Tools
* **ocr**: Extract text from images using OCR with multilingual support and bounding box detection
* **figure_parser**: Parse scientific figures, charts, and plots to extract data and insights
* **image_analyzer**: Analyze images for general description, object detection, or visual question answering

## 5. Knowledge Tools
* **document_chunker**: Split documents into chunks for vector storage using various strategies (fixed, sentence, paragraph, section)
* **semantic_search**: Find relevant information across documents using semantic similarity search
* **knowledge_merger**: Merge and synthesize information from multiple sources, removing redundancy

## 6. Self-Exploration Tool
* **sandbox_explore**: Execute custom Python code in isolated sandbox for specialized analysis, computations, and data processing

For more constructive planning, please consider which tools are most appropriate for the overall task, each subtask, and the optimal sequence for using them.
"""



# Prompt selection function
def get_planner_prompt(
    args,
    role: str,
    agent_model: str = "openai/gpt-4.1-mini",
    task_query: str = None,
    task_image: str = None,
    task_source: str = None,
    task_context: str = None,
    plan_limit: int = 512
) -> str:
    """Get appropriate planner prompt based on task type."""
    # Base planner prompt
    SYSTEM_PROMPT = """
You are a Task Planning agent specialized in analyzing and task problems, decomposing tasks into concrete subtasks, and generating high-level plans that can guide resolving the given tasks.
""".strip()
    
    BASE_PLANNER_PROMPT = f"""
**Task Problem**
{task_query}

**Additional Contexts**
1. Task Source
{task_source}
2. Task Contexts
{task_context}

---

**What You Need To Do**
1. Understand the **Task Problem** and its requirements, and analyze the **analysis width, depth, and objective** of this task
2. Analyze what contextual information and domain-specific knowledge are needed to resolve this task
3. Decompose complex tasks into clear, actionable subtasks
4. Create a concise and instructive **Problem-Solving Plan** that can effectively guide resolving the **Task Problem**

**Requirements**
* Be systematic, thorough, and strategic in your planning
* As contextual information and domain-specific knowledge are highly important, please ALWAYS highlight "Find, retrieve, distill, and summarize all the related contexts" at the very beginning of your plan
* Please specify finding and adding necessary citations and references to support support the analysis completeness and format accuracy of the final answer
* Your **Problem-Solving Plan** should be around {plan_limit} characters, consisting several subtasks in bullet-point format using "*" WITHOUT indents
* Provide both your reasoning and planning in the following format with proper enclosure:
<think>your reasoning and analysis</think>
<plan>your **Problem-Solving Plan**</plan>
""".strip()

    if task_image:
        if "qwen" in agent_model.lower():
            user_initial_content = [
                {
                    "type": "image",
                    "image": task_image,
                },
                {"type": "text", "text": BASE_PLANNER_PROMPT},
            ]
        else:
            user_initial_content = [
                {"type": "image", "image": task_image},
                {"type": "text", "text": BASE_PLANNER_PROMPT},
            ]
    else:
        user_initial_content = BASE_PLANNER_PROMPT

    prompts = {
        "system": SYSTEM_PROMPT,
        "user": user_initial_content,
    }
    
    return prompts[role]
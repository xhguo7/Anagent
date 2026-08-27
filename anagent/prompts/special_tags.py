SPECIAL_TAGS = {}

'''
SPECIAL_TAGS = {
    "answer_style": {"start": "<answer_style>", "end": "</answer_style>"},
    "category": {"start": "<category>", "end": "</category>"},
    "chunk_overlap": {"start": "<chunk_overlap>", "end": "</chunk_overlap>"},
    "chunk_size": {"start": "<chunk_size>", "end": "</chunk_size>"},
    "citation_required": {"start": "<citation_required>", "end": "</citation_required>"},
    "clear_before_add": {"start": "<clear_before_add>", "end": "</clear_before_add>"},
    "code": {"start": "<code>", "end": "</code>"},
    "confidence_threshold": {"start": "<confidence_threshold>", "end": "</confidence_threshold>"},
    "context": {"start": "<context>", "end": "</context>"},
    "custom_instruction": {"start": "<custom_instruction>", "end": "</custom_instruction>"},
    "date": {"start": "<date>", "end": "</date>"},
    "date_from": {"start": "<date_from>", "end": "</date_from>"},
    "date_to": {"start": "<date_to>", "end": "</date_to>"},
    "detail_level": {"start": "<detail_level>", "end": "</detail_level>"},
    "domain": {"start": "<domain>", "end": "</domain>"},
    "entity_types": {"start": "<entity_types>", "end": "</entity_types>"},
    "extract_bboxes": {"start": "<extract_bboxes>", "end": "</extract_bboxes>"},
    "extract_content": {"start": "<extract_content>", "end": "</extract_content>"},
    "extract_data: {"start": "<extract_data>", "end": "</extract_data>"},
    "extract_figures": {"start": "<extract_figures>", "end": "</extract_figures>"},
    "extract_metadata": {"start": "<extract_metadata>", "end": "</extract_metadata>"},
    "extract_references": {"start": "<extract_references>", "end": "</extract_references>"},
    "extract_sections": {"start": "<extract_sections>", "end": "</extract_sections>"},
    "extract_text": {"start": "<extract_text>", "end": "</extract_text>"},
    "extract_tables": {"start": "<extract_tables>", "end": "</extract_tables>"},
    "include_context": {"start": "<include_context>", "end": "</include_context>"},
    "install": {"start": "<install>", "end": "</install>"},
    "figure_caption": {"start": "<figure_caption>", "end": "</figure_caption>"},
    "filter_metadata": {"start": "<filter_metadata>", "end": "</filter_metadata>"},
    "focus": {"start": "<focus>", "end": "</focus>"},
    "focus_areas": {"start": "<focus_areas>", "end": "</focus_areas>"},
    "format": {"start": "<format>", "end": "</format>"},
    "heirarchy": {"start": "<heirarchy>", "end": "</heirarchy>"},
    "language": {"start": "<language>", "end": "</language>"},
    "languages": {"start": "<languages>", "end": "</languages>"},
    "max_length": {"start": "<max_length>", "end": "</max_length>"},
    "max_results": {"start": "<max_results>", "end": "</max_results>"},
    "min_similarity": {"start": "<min_similarity>", "end": "</min_similarity>"},
    "mode": {"start": "<mode>", "end": "</mode>"},
    "pages": {"start": "<pages>", "end": "</pages>"},
    "preserve_sentences": {"start": "<preserve_sentences>", "end": "</preserve_sentences>"},
    "preserve_terms": {"start": "<preserve_terms>", "end": "</preserve_terms>"},
    "previous_qa": {"start": "<previous_qa>", "end": "</previous_qa>"},
    "query": {"start": "<query>", "end": "</query>"},
    "question": {"start": "<question>", "end": "</question>"},
    "retrieve_abstracts": {"start": "<retrieve_abstracts>", "end": "</retrieve_abstracts>"},
    "sentences": {"start": "<sentences>", "end": "</sentences>"},
    "sort": {"start": "<sort>", "end": "</sort>"},
    "sort_by": {"start": "<sort_by>", "end": "</sort_by>"},
    "sort_order": {"start": "<sort_order>", "end": "</sort_order>"},
    "source_language": {"start": "<source_language>", "end": "</source_language>"},
    "source_type": {"start": "<source_type>", "end": "</source_type>"},
    "structured_output": {"start": "<structured_output>", "end": "</structured_output>"},
    "style": {"start": "<style>", "end": "</style>"},
    "summary_type": {"start": "<summary_type>", "end": "</summary_type>"},
    "task": {"start": "<task>", "end": "</task>"},
    "think": {"start": "<think>", "end": "</think>"},
    "tool": {"start": "<tool>", "end": "</tool>"},
    "top_k": {"start": "<top_k>", "end": "</top_k>"},
}
'''


THINK_START = "<think>"
THINK_END = "</think>"
PLAN_START = "<plan>"
PLAN_END = "</plan>"
EXPERT_START = "<summary>"
EXPERT_END = "</summary>"
SOLVER_START = "<answer>"
SOLVER_END = "</answer>"
CRITIC_START = "<feedback>"
CRITIC_END = "</feedback>"
CONFIDENCE_START = "<confidence>"
CONFIDENCE_END = "</confidence>"



def response_processor(agent: str, agent_response: str) -> str:
    """Response processing for problem solving"""

    if agent == "planner":
        if PLAN_START not in agent_response:
            if THINK_END in agent_response:
                agent_response = agent_response.replace(THINK_END, f"{THINK_END}{PLAN_START}").strip()
            else:
                agent_response = f"{PLAN_START}{agent_response.strip()}"
        
        if PLAN_END not in agent_response:
            agent_response = f"{agent_response.strip()}{PLAN_END}"

    elif agent == "expert":
        if EXPERT_START not in agent_response:
            if THINK_END in agent_response:
                agent_response = agent_response.replace(THINK_END, f"{THINK_END}{EXPERT_START}").strip()
            else:
                agent_response = f"{EXPERT_START}{agent_response.strip()}"
        if EXPERT_END not in agent_response:
            agent_response = f"{agent_response.strip()}{EXPERT_END}"

    elif agent == "solver":
        if SOLVER_START not in agent_response:
            if THINK_END in agent_response:
                agent_response = agent_response.replace(THINK_END, f"{THINK_END}{SOLVER_START}").strip()
            else:
                agent_response = f"{SOLVER_START}{agent_response.strip()}"
        if SOLVER_END not in agent_response:
            if CONFIDENCE_START in agent_response:
                agent_response = agent_response.replace(CONFIDENCE_START, f"{SOLVER_END}{CONFIDENCE_START}").strip()
            else:
                agent_response = f"{agent_response.strip()}{SOLVER_END}"

    elif agent == "critic":
        if CRITIC_START not in agent_response:
            if THINK_END in agent_response:
                agent_response = agent_response.replace(THINK_END, f"{THINK_END}{CRITIC_START}").strip()
            else:
                agent_response = f"{CRITIC_START}{agent_response.strip()}"
        if CRITIC_END not in agent_response:
            agent_response = f"{agent_response.strip()}{CRITIC_END}"

    if agent != "solver":
        agent_response = agent_response.replace(SOLVER_START, "").replace(SOLVER_END, "").strip()

    return agent_response
        
        
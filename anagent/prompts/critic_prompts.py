"""
Critic Agent prompts.
"""

def get_critic_prompt(
    args,
    role: str,
    agent_model: str = "openai/gpt-4.1-mini",
    planner_plan: str = None,
    expert_summary: str = None,
    solver_solution: dict = None,
    task_query: str = None,
    task_image: str = None,
    task_source: str = None,
    task_context: str = None,
    task_type: str = "general",
    feedback_limit: int = 1024
) -> str:
    """Get appropriate critic prompt based on task type."""

    data_type = "figure" if task_image else "table"
    
    '''
    CRITIC_SYSTEM_PROMPT = "You are a CRITIC ADVISOR responsible for evaluating and improving STUDENT's **STUDENT Solution** to the given **Task Problem** based on available **Additional Contexts** and STUDEUNT's **STUDEUNT Reasoning**."
    GENERAL_CRITIC_PROMPT = f"""
**Task Problem**
{task_query}

**Additional Contexts**
1. Task Source
{task_source}
2. Task Contexts
{task_context}
3. Additional Information
{expert_summary}

**STUDENT Reasoning**
Planning: {planner_plan}
Reasoning: {solver_solution.get("reasoning", "None") if solver_solution else "None"}
Confidence: {solver_solution.get("confidence", "None") if solver_solution else "None"}

**STUDENT Solution**
{solver_solution.get("solution", "None") if solver_solution else "None"}

**What You Need To Do**
1. Analyze the **Task Problem** and its requirements, and conclude the **analysis width, depth, and objective** of this task
2. Review the **Additional Contexts**, reasoning and analyzing all the available information and knowledge systematically and thoroughly
3. Evaluate STUDENT's **STUDENT Reasoning** and **STUDENT Solution** in the following five dimensions:
    (1) **Plan-Reason-Solution Consistency**: Whether STUDENT's planning, reasoning, and solution are consistent with the given information in **Task Problem** and **Additional Contexts**
    (2) **Query-Solution Alignment**: Whether STUDENT's planning, reasoning, and solution are targeted at resolving **Task Problem** and properly meet and cover all the requirements, and whether **STUDENT Solution** perfectly match the **goal, style, and type** of this task
    (3) **Knowledge Utilization**: Whether all necessary observations and contexts are covered in **STUDENT Solution**, whether the required level of detail (shallow description or in-depth analysis) are reached, and whether all important information are analyzed and included according to the **Task Problem**
    (4) **Format Correctness**: Whether STUDENT's solution satisfies the required format, whether contains typos or missing citations/references, and whether necessary \\cite and \\ref are properly added
    (5) **Solution Grounding**: Whether there are groundless information in **STUDENT Solution**, NOTE that both referred contents (e.g., referring to other tables/figures/sections/equations/etc., within the paper or referring to other papers) or reasonably inferred contents are both grounded (CAREFUL: tasks involving in-depth analysis SHOULD have inferred contents in the solution, as long as they are reasonable and inferable from the given information)
4. Generate your detailed improvement guidance feedback to STUDENT's **Task Solution**, which should at least cover the five evaluation dimensions above, and less than {feedback_limit} characters.

**Requirements**
* Ensure accuracy, precision, and professional quality in your evaluation
* Provide reasoning and assign grades for each evaluation dimension
* Deliver final assessment feedback as concise and instructive bullet points (addressing the STUDENT as "You")
* Follow the exact enclosure format below:
<think>your evaluation and analysis</think>
<consistency>grade 0 OR 1 for **Plan-Reason-Solution Consistency**, 0 for inconsistant, 1 for consistent</consistency>
<alignment>grade 0 OR 1 for **Query-Solution Alignment**, 0 for misaligned solution, 1 for aligned solution</alignment>
<utilization>grade 0 OR 1 for **Knowledge Utilization**, 0 for missing important information</utilization>
<format>grade 0 OR 1 for **Format Correctness**, 0 for incorrect format or missing format specifications</format>
<grounding>grade 0 OR 1 for **Solution Grounding**, 0 for containing groundless information, 1 for grounded solution</grounding>
<feedback>detailed bullet-point improvement guidance, NO MORE THAN {feedback_limit} characters</feedback>
""".strip()'''

    # Base critic prompt
    CRITIC_SYSTEM_PROMPT = "You are a CRITIC ADVISOR responsible for evaluating and improving STUDENT's **STUDENT Answer** to the given **Task Problem** based on available **Additional Contexts**."
    
    BASE_CRITIC_PROMPT = f"""
**Task Problem**
{task_query}

**Additional Contexts**
1. Task Source
{task_source}
2. Task Contexts
{task_context}
3. Additional Information
{expert_summary}

**STUDENT Problem-Solving Plan**
{planner_plan}

**STUDENT Answer**
{solver_solution.get("solution", "None") if solver_solution else "None"}

---

**Evaluation Criteria**
1. **Content Accuracy**: Does **STUDENT Answer** accurately and correctly reflect the explicitly presented information in the {data_type} and its relevant **Additional Contexts**, such as reported values, trends, stated findings, and observed patterns, without distortion or omission?
2. **Analytical Completeness**: Does **STUDENT Answer** sufficiently develop the derived analyses, interpretations, and conclusions implied by the {data_type} and its relevant **Additional Contexts**, capturing key insights without missing important analytical findings?
3. **Format Correctness**: Whether **STUDENT Answer** satisfies the required format, whether **STUDENT Answer** contains typos or missing citations/references, and whether necessary \\cite and \\ref are properly added?
4. **Clarity & Coherence**: Is **STUDENT Answer** clearly articulated, properly structured, and logically coherent, with sound scientific writing throughout?
5. **Reliability & Faithfulness**: Are there any hallucinated information in model analysis that is never covered, analyzed, and inferred in the ground-truth analysis? NOTE that both referred contents (e.g., referring to other tables/figures/sections/equations/etc., within the paper or referring to other papers) or reasonably inferred contents are considered as grounded (CAREFUL: analysis writing SHOULD have inferred contents, as long as they are reasonable and inferable from the given information)

**What You Need To Do**
1. Analyze the **Task Problem** and its requirements, and infer the **goal, style, and type** of this task (e.g., academic writing, creative generation, technical report, etc.)
2. Review the **Additional Contexts**, reasoning and analyzing all the available information and knowledge systematically and thoroughly
3. Evaluate STUDENT's **STUDENT Answer** according to the five **Evaluation Criteria**
4. Generate your detailed improvement guidance feedback to STUDENT, which should at least cover the five evaluation dimensions above, and less than {feedback_limit} characters.

**Response Format:**
<think>your evaluation reasoning</think>
<accuracy>grade 0-2 for **Content Accuracy**: 0 = mostly inaccurate; 1 = partially accurate; 2 = fully accurate</accuracy>
<completeness>grade 0-2 for **Analytical Completeness**: 0 = most key analyses missing; 1 = some key analyses missing; 2 = all key analyses present</completeness>
<format>grade 0-2 for **Format Correctness**: 0 = many format errors; 1 = some format errors; 2 = no format error</format>
<writing>grade 0-2 for **Clarity & Coherence**: 0 = poorly written; 1 = moderately clear; 2 = clear and coherent</writing>
<faithfulness>grade 0-2 for **Reliability & Faithfulness**: 0 = mostly hallucinated; 1 = partially hallucinated; 2 = fully reasonable</faithfulness>
<feedback>
- [Specific improvement point 1]
- [Specific improvement point 2]
- [Continue as needed, staying under {feedback_limit} characters total]
</feedback>
""".strip()

    if task_image:
        if "qwen" in agent_model.lower():
            user_initial_content = [
                {
                    "type": "image",
                    "image": task_image,
                },
                {"type": "text", "text": BASE_CRITIC_PROMPT},
            ]
        else:
            user_initial_content = [
                {"type": "image", "image": task_image},
                {"type": "text", "text": BASE_CRITIC_PROMPT},
            ]
    else:
        user_initial_content = BASE_CRITIC_PROMPT

    prompts = {
        "system": CRITIC_SYSTEM_PROMPT,
        "user": user_initial_content,
    }
    
    return prompts[role]
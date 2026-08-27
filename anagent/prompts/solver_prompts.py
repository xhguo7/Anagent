"""
Solver Agent prompts.
"""

def get_solver_prompt(
    args,
    role: str,
    planner_plan: str,
    expert_summary: str,
    agent_model: str = "openai/gpt-4.1-mini",
    task_query: str = None,
    task_image: str = None,
    task_source: str = None,
    task_context: str = None,
    task_type: str = "general",
    solution_limit: int = 2048,
    critic_feedback: dict = None,
) -> str:
    """Get appropriate solver prompt based on task type."""
    # Reasoning is optional
    enable_reasoning = getattr(args, "enable_reasoning", False)

    reasoning_tag = "<think>your reasoning</think>\n" if enable_reasoning else ""
    thinking_word = "thinking, " if enable_reasoning else ""

    if args.solver_awareness:
        output_format = f"""
* Provide your {thinking_word}solution, and confidence in the following format with proper enclosure:
{reasoning_tag}<answer>your scientific analysis</answer>
<confidence>your confidence level and any caveats</confidence>
""".strip()
    else:
        output_format = f"""
* Provide your solution in the following format with proper enclosure:
{reasoning_tag}<answer>your scientific analysis</answer>
""".strip()

    if enable_reasoning:
        solution_instruction = (
            "4. Generate your reasoning enclosed in: <think>your reasoning</think>, "
            "then your solution enclosed in: <answer>your solution</answer>"
        )
    else:
        solution_instruction = "4. Generate your solution enclosed in: <answer>your solution</answer>"

    analysis_writing = """
* Please carefully review all the given information and contexts, and generate semantically coherent, logically structured, and scientifically grounded analysis
* Always use the correct format in line with the input table/figure. For example, if the input uses LaTeX, you SHOULD use Latex Bib and \\cite for reference citations, and specify Latex \\ref when referring to other tables/figures/sections/equations/etc."""
    
    # Base solver prompt
    SOLVER_SYSTEM_PROMPT = "You are a Expert Scientist that writes high-quality scientific analysis according to the given **Task Problem**."
    BASE_SOLVER_PROMPT = f"""
**Task Problem**
{task_query}

**Additional Contexts**
1. Task Source
{task_source}
2. Task Contexts
{task_context}
3. Additional Information
{expert_summary}

**Problem-Solving Plan**
{planner_plan}

---

**What You Need To Do**
1. Analyze the **Task Problem** and its requirements, and conclude the **analysis width, depth, and objective** of this task
2. Review the **Problem-Solving Plan** and **Additional Contexts**
3. Reason and analyze all the available information and knowledge systematically and thoroughly
{solution_instruction}

**Requirements**
* Please ensure the accuracy, precision, completeness, profession, and quality of your scientific analysis.{analysis_writing}
{output_format}
""".strip()

    # Refine solution based on critic feedback
    if critic_feedback:
        critic_feedback_refinement = f"""
**Solution Refinement**
Suggestions to refine your solution:
{critic_feedback["feedback"]}

**Requirements**
* Please use **Solution Refinement** suggestions wisely, accepting helpful ones and ignoring those that are not.
* Please stick to your original version and refine your solution with necessary changes.
{output_format}
""".strip()

        '''critic_feedback_refinement_with_scores = f"""
**Solution Evaluation**
Your solution is evaluated in the following four aspects:
(1) **Plan-Reason-Solution Consistency** (i.e., Whether the solution is consistent with the given information in **Task Problem** and **Additional Contexts**): {critic_feedback["consistency"]}
(2) **Query-Solution Alignment** (i.e., Whether the solution is targeted at resolving **Task Problem** and properly meet and cover all the requirements, and perfectly match the **goal, style, and type** of this task): {critic_feedback["alignment"]}
(3) **Knowledge Utilization** (i.e., Whether all necessary observations and contexts are covered in the solution, whether the required level of detail are reached, and whether all important information are analyzed and included according to the **Task Problem**): {critic_feedback["utilization"]}
(4) **Format Correctness** (i.e., Whether the format is correct, and whether there are missing format specifications like missing citations or references): {critic_feedback["format"]}
(5) **Solution Grounding** (i.e., Whether there are groundless information in the solution): {critic_feedback["grounding"]}

**Solution Refinement**
Suggestions to refine your solution:
{critic_feedback["feedback"]}

**Requirements**
* Please use **Solution Refinement** suggestions wisely, accepting helpful ones and ignoring those that are not.
* Please stick to your original version and refine your solution with small necessary changes.{analysis_writing}
{output_format}
""".strip()'''
    else:
        critic_feedback_refinement = None

    if task_image:
        if "qwen" in agent_model.lower():
            user_initial_content = [
                {
                    "type": "image",
                    "image": task_image,
                },
                {"type": "text", "text": BASE_SOLVER_PROMPT},
            ]
        else:
            user_initial_content = [
                {"type": "image", "image": task_image},
                {"type": "text", "text": BASE_SOLVER_PROMPT},
            ]
    else:
        user_initial_content = BASE_SOLVER_PROMPT

    prompts = {
        "system": SOLVER_SYSTEM_PROMPT,
        "user": user_initial_content,
        "critic": critic_feedback_refinement,
    }
    
    return prompts[role]
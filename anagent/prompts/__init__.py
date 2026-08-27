"""
Prompts module.
"""

from .special_tags import SPECIAL_TAGS, response_processor
from .planner_prompts import get_planner_prompt
from .expert_prompts import get_expert_prompt
from .solver_prompts import get_solver_prompt
from .critic_prompts import get_critic_prompt
from .memory_prompts import get_memory_prompt
from .tool_prompts import get_tool_prompts

from utils.logger import setup_logger
logger = setup_logger(__name__)


__all__ = [
    'get_planner_prompt',
    'get_expert_prompt',
    'get_solver_prompt',
    'get_memory_prompt',
    'get_tool_prompts',
    'response_processor',
]


def load_special_tags(tag_key: str):
    if tag_key not in SPECIAL_TAGS:
        SPECIAL_TAGS[tag_key] = {
            "start": f"<{tag_key}>",
            "end": f"</{tag_key}>",
        }
    return SPECIAL_TAGS[tag_key]

def load_prompts(
    args,
    agent: str,
    role: str,
    agent_model: str = None,
    task_query: str = None,
    task_image: str = None,
    task_source: str = None,
    task_context: str = None,
    planner_plan: str = None,
    expert_summary: str = None,
    solver_solution: dict = None,
    critic_feedback: dict = None,
    max_turns: str = 10,
    expert_turn: int = 1,
    tool_description: str = None,
    tool_execution_output: str = None,
):
    """Load prompts for an agent."""
    assert agent in ["planner", "expert", "solver", "critic"]

    if agent == "planner":
        return get_planner_prompt(
            args=args,
            role=role,
            agent_model=agent_model,
            task_query=task_query,
            task_image=task_image,
            task_source=task_source,
            task_context=task_context,
        )

    elif agent == "expert":
        # The Expert always uses the multi-agent prompt family
        if not planner_plan:
            logger.warning(
                f"[load_prompts] Planner produced no plan; rendering the Expert prompt "
                f"with planner_plan='None' (turn {expert_turn}/{max_turns})"
            )
        return get_expert_prompt(
            args=args,
            role=role,
            agent_model=agent_model,
            task_query=task_query,
            task_image=task_image,
            task_source=task_source,
            task_context=task_context,
            planner_plan=planner_plan if planner_plan else "None",
            max_turns=max_turns,
            expert_turn=expert_turn,
            tool_description=tool_description,
            tool_execution_output=tool_execution_output,
        )

    elif agent == "solver":
        return get_solver_prompt(
            args=args,
            role=role,
            agent_model=agent_model,
            planner_plan=planner_plan,
            expert_summary=expert_summary,
            task_query=task_query,
            task_image=task_image,
            task_source=task_source,
            task_context=task_context,
            critic_feedback=critic_feedback,
        )

    elif agent == "critic":
        return get_critic_prompt(
            args=args,
            role=role,
            agent_model=agent_model,
            planner_plan=planner_plan,
            expert_summary=expert_summary,
            solver_solution=solver_solution,
            task_query=task_query,
            task_image=task_image,
            task_source=task_source,
            task_context=task_context,
        )
    
    else:
        raise ValueError(f"Error: Invalid agent: {agent}")
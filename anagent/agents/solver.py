"""
Solver Agent: final solution generation.
"""

import re
import os
from typing import Dict, Any, List, Optional


from utils.logger import setup_logger
logger = setup_logger(__name__)

from anagent.agents import BaseAgent
from utils.parser import response_parser
from anagent.prompts import load_prompts, load_special_tags, response_processor


class SolverAgent(BaseAgent):
    """Solver agent that generates final solutions."""
    
    def __init__(
        self,
        args,
        engine_kwargs: Optional[Dict[str, Any]] = None,
        engine: Optional[Any] = None,
        prompt_template: Optional[str] = None,
        **kwargs
    ):
        """Initialize Solver agent."""
        super().__init__(
            name="Solver",
            role="Solution Reasoning and Generation",
            args=args,
            engine_kwargs=engine_kwargs,
            engine=engine,
            **kwargs
        )
        
        # Agent
        self.agent_role = "solver"
        self.prompt_template = prompt_template
        self.solver_awareness =  args.solver_awareness

        # Task
        self.task = None
        self.memory = None
        self.planner_plan = None
        self.expert_summary = None
        self.solver_solution = None
        self.critic_feedback = None

    def _get_model_path(self) -> str:
        """Get model path for Solver."""
        return self.args.solver_model

    def _get_temperature(self) -> float:
        """Get temperature for Solver."""
        return self.args.solver_temperature if self.args.solver_temperature is not None else self.args.temperature
        
    def _generate_solution(self, messages: List) -> str:
        """Generate solution"""
        if self.critic_feedback and len(self.critic_feedback) > 0 and self.args.critic_model:
            messages_with_critic = self._build_prompt(messages)
            solution = self._call_llm(messages_with_critic)
        else:
            solution = self._call_llm(messages)

        return solution
        
    def run(self, task: Dict[str, Any], memory: Dict[str, Any]) -> Dict[str, Any]:
        """Generate solution based on collected information."""
        logger.info(f"\n[{self.name}] Generating solution...")
        # Store task info
        self.task = task
        self.memory = memory
        self.planner_plan = memory.get("planner_output", {}).get("plan", "")
        self.expert_summary = memory.get("expert_output", {}).get("expert_summary", "").get("expert_summary", memory.get("expert_output", {}).get("expert_summary", ""))

        # If critic
        if self.args.critic_model:
            self.solver_solution = memory.get("solver_output", {}).get("raw_response", "")
            self.critic_feedback = memory.get("critic_output", {})
        
        # Build prompt
        messages = self._build_prompt()        
        
        # Generate solution
        solution = self._generate_solution(messages)
        solution = response_processor(self.agent_role, solution)
        
        # Parse solution
        parsed_solution = self._parse_solution(solution)
        
        # Prepare output
        output = self.get_output_format()
        update_dict = {
            "raw_response": solution,
            "solution": parsed_solution.get("solution", solution),
            "reasoning": parsed_solution.get("think", "")
        }
        if self.solver_awareness:
            update_dict["confidence"] = parsed_solution.get("confidence", "")
        output.update(update_dict)
        
        logger.info(f"[{self.name}] Solution generated ({len(solution)} chars):\n{solution}")
        return output
    
    def _build_prompt(self, messages: List = None) -> List[Dict[str, Any]]:
        """Build prompt for solution generation."""
        
        # System prompt
        if self.prompt_template:
            system_prompt = self.prompt_template
        else:
            system_prompt = load_prompts(
                args=self.args,
                agent="solver",
                role="system",
                task_query=self.task["query"],
                task_source=self.task.get("task_sources", None),
                task_context=self.task.get("contexts", None),
            )
        
        # Refine solution based on critic feedback
        if messages and self.args.critic_model:
            critic_msg = load_prompts(
                args=self.args,
                agent="solver",
                role="critic",
                critic_feedback=self.critic_feedback,
            )
            messages.extend([
                {
                    "role": "assistant",
                    "content": self.solver_solution
                },
                {
                    "role": "user",
                    "content": critic_msg
                }
            ])
            return messages

        # User message
        task_image = self.task.get("image", None)
        user_message = load_prompts(
            args=self.args,
            agent="solver",
            role="user",
            agent_model=self.model_path,
            task_image=self.task["image"] if task_image and len(task_image)>0 else None,
            planner_plan=self.planner_plan,
            expert_summary=self.expert_summary,
            task_query=self.task["query"],
            task_source=self.task.get("task_sources", None),
            task_context=self.task.get("contexts", None),
        )
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add user message
        messages.append({"role": "user", "content": user_message})
        
        return messages
    
    def _parse_solution(self, response: str) -> Dict[str, Any]:
        """Parse solution from response."""
        
        parsed = {"think": "", "solution": ""}
        if self.solver_awareness:
            parsed["confidence"] = ""
        
        # Extract reasoning
        think_match, think_content = response_parser(response, load_special_tags("think"), "relaxed")
        parsed["think"] = think_content.strip() if think_match else ""
        
        # Extract solution
        solution_match, solution_content = response_parser(response, load_special_tags("answer"), "relaxed")
        parsed["solution"] = solution_content.strip() or (response or "").strip()
        
        # Extract confidence
        if self.solver_awareness:
            confidence_match, confidence_content = response_parser(response, load_special_tags("confidence"))
            if confidence_match:
                parsed["confidence"] = confidence_content.strip()
        
        return parsed
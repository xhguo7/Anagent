"""
Critic Agent: evaluate and refine solutions.
"""

import re
import os
from typing import Dict, Any, List, Optional


from utils.logger import setup_logger
logger = setup_logger(__name__)

from anagent.agents import BaseAgent
from utils.parser import response_parser
from anagent.prompts import load_prompts, load_special_tags, response_processor


class CriticAgent(BaseAgent):
    """Critic agent that evaluation and refine final solutions."""
    
    def __init__(
        self,
        args,
        engine_kwargs: Optional[Dict[str, Any]] = None,
        engine: Optional[Any] = None,
        prompt_template: Optional[str] = None,
        **kwargs
    ):
        """Initialize Critic agent."""
        super().__init__(
            name="Critic",
            role="Solution Evaluation and Refinement",
            args=args,
            engine_kwargs=engine_kwargs,
            engine=engine,
            **kwargs
        )
        
        # Agent
        self.agent_role = "critic"
        self.prompt_template = prompt_template

        # Task
        self.task = None
        self.memory = None
        self.plan = None
        self.expert_summary = None

    def _get_model_path(self) -> str:
        """Get model path for Critic."""
        return self.args.critic_model

    def _get_temperature(self) -> float:
        """Get temperature for Critic."""
        return self.args.critic_temperature if self.args.critic_temperature is not None else self.args.temperature
        
    
    def run(self, task: Dict[str, Any], memory: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate and refine solution based on collected information."""
        logger.info(f"\n[{self.name}] Evaluating and refining solution...")
        # Store task info
        self.task = task
        self.memory = memory
        self.planner_plan = memory.get("planner_output", {}).get("plan", "")
        self.expert_summary = memory.get("expert_output", {}).get("expert_summary", "")
        self.solver_reasoning = memory.get("solver_output", {}).get("reasoning", "")
        self.solver_confidence = memory.get("solver_output", {}).get("confidence", "")
        self.solver_solution = memory.get("solver_output", {}).get("solution", "")
        
        # Build prompt
        messages = self._build_prompt()
        
        # Generate response
        response = self._call_llm(messages)
        response = response_processor(self.agent_role, response)
        
        # Parse response
        parsed_response = self._parse_response(response)
        if not parsed_response["feedback"] or len(parsed_response["feedback"])==0:
            parsed_response["feedback"] = response
        
        # Prepare output
        output = self.get_output_format()
        output.update({"raw_response": response})
        output.update(parsed_response)
        critic_feedback = parsed_response["feedback"]
        
        logger.info(f"[{self.name}] Feedback generated ({len(critic_feedback)} chars):\n{critic_feedback}\n")
        return output
    
    def _build_prompt(self) -> List[Dict[str, Any]]:
        """Build prompt for solution generation."""
        
        # System prompt
        if self.prompt_template:
            system_prompt = self.prompt_template
        else:
            system_prompt = load_prompts(
                args=self.args,
                agent="critic",
                role="system",
                task_query=self.task["query"],
                task_source=self.task.get("task_sources", None),
                task_context=self.task.get("contexts", None),
            )

        # User message
        task_image = self.task.get("image", None)
        user_message = load_prompts(
            args=self.args,
            agent="critic",
            role="user",
            agent_model=self.model_path,
            planner_plan=self.planner_plan,
            expert_summary=self.expert_summary,
            solver_solution={"reasoning": self.solver_reasoning, "confidence": self.solver_confidence, "solution": self.solver_solution},
            task_image=self.task["image"] if task_image and len(task_image)>0 else None,
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
        
        parsed = {
            "think": "",
            "solution": "",
            "confidence": "",
        }
        
        # Extract reasoning
        think_match, think_content = response_parser(response, load_special_tags("think"), "relaxed")
        parsed["think"] = think_content.strip() if think_match else ""
        
        # Extract solution
        solution_match, solution_content = response_parser(response, load_special_tags("answer"), "relaxed")
        parsed["solution"] = solution_content.strip()
        
        # Extract confidence
        confidence_match, confidence_content = response_parser(response, load_special_tags("confidence"))
        if confidence_match:
            parsed["confidence"] = confidence_content.strip()
        
        return parsed

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse critic evaluation from response."""
        
        parsed_response = {
            "think": "",
            "accuracy": "",
            "completeness": "",
            "writing": "",
            "faithfulness": "",
            "format": "",
            "feedback": "",
        }

        # Extract response fields
        for tag in parsed_response:
            tag_match, tag_content = response_parser(response, load_special_tags(tag), "relaxed")
            if tag_match:
                parsed_response[tag] = tag_content.strip()
        
        return parsed_response
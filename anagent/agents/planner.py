"""
Planner Agent: task analysis and execution planning.
"""

import re
import os
import json
from typing import Dict, Any, List, Optional


from utils.logger import setup_logger
logger = setup_logger(__name__)

from anagent.agents import BaseAgent
from utils.parser import response_parser
from anagent.prompts import load_prompts, load_special_tags, response_processor


class PlannerAgent(BaseAgent):
    """Planner agent that analyzes tasks and creates execution plans."""
    
    def __init__(
        self,
        args,
        engine_kwargs: Optional[Dict[str, Any]] = None,
        engine: Optional[Any] = None,
        prompt_template: Optional[str] = None,
        **kwargs
    ):
        """Initialize Planner agent."""
        super().__init__(
            name="Planner",
            role="Task Analysis and Planning",
            args=args,
            engine_kwargs=engine_kwargs,
            engine=engine,
            **kwargs
        )
        
        self.prompt_template = prompt_template
        self.agent_role = "planner"

        # Task
        self.task = None
        self.memory = None
        self.max_retries = args.max_retries

    def _get_model_path(self) -> str:
        """Get model path for Planner."""
        return self.args.planner_model

    def _get_temperature(self) -> float:
        """Get temperature for Planner."""
        return self.args.planner_temperature if self.args.planner_temperature is not None else self.args.temperature
    
    def run(self, task: Dict[str, Any], memory: Dict[str, Any]) -> Dict[str, Any]:
        """Execute planning for the given task."""
        logger.info(f"\n[{self.name}] Analyzing task and creating plan...")
        # Store task info
        self.task = task
        self.memory = memory
        
        # Build prompt
        messages = self._build_prompt()
        
        # Generate plan
        plan_response = self._call_llm(messages)
        plan_response = response_processor(self.agent_role, plan_response)
            
        # Parse plan
        parsed_plan = self._parse_plan(plan_response)
        
        # Prepare output
        output = self.get_output_format()
        output.update({
            "raw_response": plan_response,
            "think": parsed_plan.get("think", plan_response),
            "plan": parsed_plan.get("plan", plan_response),
        })
        
        logger.info(f"[{self.name}] Plan created with {len(output['plan'])} steps")
        return output
    
    def _build_prompt(self) -> List[Dict[str, Any]]:
        """Build prompt for planning."""
        
        # System prompt
        if self.prompt_template:
            system_prompt = self.prompt_template
        else:
            system_prompt = load_prompts(
                args=self.args,
                agent="planner",
                role="system",
            )

        # User message
        task_image = self.task.get("image", None)
        user_message = load_prompts(
            args=self.args,
            agent="planner",
            role="user",
            agent_model=self.model_path,
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
    
    def _parse_plan(self, response: str) -> Dict[str, Any]:
        """Parse plan from response."""
        
        parsed = {
            "think": "",
            "plan": "",
        }
        
        # Extract reasoning
        think_match, think_content = response_parser(response, load_special_tags("think"), "relaxed")
        parsed["think"] = think_content.strip() if think_match else ""
        
        # Extract planning
        plan_match, plan_content = response_parser(response, load_special_tags("plan"), "relaxed")
        parsed["plan"] = plan_content.strip()
        
        return parsed
"""
Multi-agent orchestrator: coordinates the agent workflow.
"""

import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime


from utils.logger import setup_logger
logger = setup_logger(__name__)

from anagent.engines import EngineManager
from anagent.agents import PlannerAgent, ExpertAgent, SolverAgent, CriticAgent
from utils.json_util import save_to_json


def _extract_expert_summary(expert_output) -> str:
    """Robustly pull the expert-summary text from a (possibly malformed) expert output."""
    if not isinstance(expert_output, dict):
        return ""
    es = expert_output.get("expert_summary", "")
    if isinstance(es, dict):
        return es.get("expert_summary", "") or ""
    return es or ""


class MultiAgentOrchestrator:
    """Orchestrates the multi-agent system workflow."""
    
    def __init__(
        self,
        args,
        engine_kwargs: Optional[Dict[str, Any]] = None,
        enable_tools: Optional[List[str]] = None
    ):
        """Initialize orchestrator."""
        self.args = args
        self.save_dir = args.output_dir
        # Iterative Solver<->Critic refinement rounds (0 = off).
        self.critic_turns = int(getattr(args, "critic_turns", 0) or 0)
        engine_kwargs = engine_kwargs or {}

        # Agents
        self.planner = None
        self.expert = None
        self.solver = None
        self.critic = None
        
        # Engine manager for sharing engines
        self.engine_manager = EngineManager()
        
        # Initialize agents
        logger.info("[Orchestrator] Initializing agents...")

        self._init_agents(
            engine_kwargs=engine_kwargs,
            enable_tools=enable_tools
        )
        
    def _init_agents(
        self,
        engine_kwargs: Optional[Dict[str, Any]] = None,
        enable_tools: Optional[List[str]] = None
    ):
        """Initialize all agents with shared engines."""
        engine_kwargs = engine_kwargs or {}
        
        # Collect all unique model configurations
        models_config = {
            'planner': (
                self.args.planner_model,
                self.args.planner_temperature if self.args.planner_temperature is not None else self.args.temperature
            ),
            'expert': (
                self.args.expert_model,
                self.args.expert_temperature if self.args.expert_temperature is not None else self.args.temperature
            ),
            'solver': (
                self.args.solver_model,
                self.args.solver_temperature if self.args.solver_temperature is not None else self.args.temperature
            ),
            'tool': (
                self.args.tool_model,
                self.args.tool_temperature if self.args.tool_temperature is not None else self.args.temperature
            ),
        }
        
        # The Critic is created ONLY for iterative refinement (--critic-turns > 0)
        if self.critic_turns > 0:
            if not self.args.critic_model:
                self.args.critic_model = self.args.solver_model
        elif self.args.critic_model:
            logger.warning(
                "[Orchestrator] --critic-model is set but --critic-turns is 0, "
                "so no refinement runs and the Critic is not created. "
                "Pass --critic-turns N to enable it."
            )
        if self.critic_turns > 0:
            models_config['critic'] = (
                self.args.critic_model,
                self.args.critic_temperature if self.args.critic_temperature is not None else self.args.temperature
            )
        
        # Create/retrieve engines using the manager
        engines = {}
        for agent_name, (model_path, temp) in models_config.items():
            engines[agent_name] = self.engine_manager.get_engine(
                model_path=model_path,
                temperature=temp,
                max_tokens=self.args.max_output_tokens,
                **engine_kwargs
            )
        
        logger.info(f"[Orchestrator] Engine sharing summary:")
        logger.info(f"  - Unique engines created: {len(self.engine_manager.get_cached_models())}")
        logger.info(f"  - Cached models: {self.engine_manager.get_cached_models()}")
        # Initialize agents with shared engines
        self.planner = PlannerAgent(
            args=self.args,
            engine_kwargs=engine_kwargs,
            engine=engines['planner']
        )
        
        self.expert = ExpertAgent(
            args=self.args,
            engine_kwargs=engine_kwargs,
            engine=engines['expert'],
            tool_engine=engines['tool'],
            enable_tools=enable_tools or ["all"]
        )
        
        self.solver = SolverAgent(
            args=self.args,
            engine_kwargs=engine_kwargs,
            engine=engines['solver']
        )

        if self.critic_turns > 0 and self.args.critic_model:
            self.critic = CriticAgent(
                args=self.args,
                engine_kwargs=engine_kwargs,
                engine=engines['critic']
            )
        
        logger.info(f"[Orchestrator] Initialized with models:")
        logger.info(f"  - Tool: {self.args.tool_model} (temp: {models_config['tool'][1]})")
        logger.info(f"  - Planner: {self.args.planner_model} (temp: {models_config['planner'][1]})")
        logger.info(f"  - Expert: {self.args.expert_model} (temp: {models_config['expert'][1]})")
        logger.info(f"  - Solver: {self.args.solver_model} (temp: {models_config['solver'][1]})")
        if self.args.critic_model:
            logger.info(f"  - Critic: {self.args.critic_model} (temp: {models_config['critic'][1]})")
        logger.info(f"  - Max output tokens: {self.args.max_output_tokens}")
    def run(
        self,
        task: Dict,
        expert_turns: int = 10,
    ) -> Dict[str, Any]:
        """Run the multi-agent system."""
        logger.info(
            f"\n{'='*70}"
            "\nMULTI-AGENT SYSTEM EXECUTION"
            f"\n{'='*70}"
            f"\nQuery: {task['query']}"
        )
        if task.get("contexts", None):
            logger.info(f"Context:\n{task.get("contexts", None)}")
        if task.get("image", None):
            logger.info(f"Image: Provided")
        logger.info(
            f"\n{'='*70}"
            f"\nMax Expert Turns: {expert_turns}"
            f"\n{'='*70}\n"
        )
        # Shared memory
        shared_memory = {}
        
        # Track execution
        execution_log = []
        start_time = datetime.now()
        
        # STAGE 1: Planner
        logger.info("\n=== STAGE 1: PLANNING ===")
        planner_output = self._execute_agent(self.planner, task, shared_memory, "Planning agent for task planning")
        shared_memory["planner_output"] = planner_output
        execution_log.append({"agent": "planner", "output": planner_output})
        logger.info(f"[Orchestrator] Planner Plan:\n{planner_output.get('plan', '(planner produced no plan)')}\n")
        
        # STAGE 2: Expert (multi-turn execution)
        logger.info(f"\n=== STAGE 2: EXPERT INFORMATION RETRIEVAL (max {expert_turns} turns) ===")
        shared_memory["max_expert_turns"] = expert_turns        
        expert_output = self._execute_agent(
            self.expert,
            task,
            shared_memory,
            "Expert agent for information retrieval and summary generation"
        )
        shared_memory["expert_output"] = expert_output
        execution_log.append({
            "agent": "expert",
            "output": expert_output
        })
        logger.info(f"[Orchestrator] Expert Summary:\n{_extract_expert_summary(expert_output)}\n")
        
        # STAGE 3: Solver
        logger.info("\n=== STAGE 3: SOLUTION GENERATION ===")
        solver_output = self._execute_agent(self.solver, task, shared_memory, "Solver agent for solution generation")
        shared_memory["solver_output"] = solver_output
        execution_log.append({"agent": "solver", "output": solver_output})
        logger.info(f"[Orchestrator] Solver Soltion:\n{solver_output.get('solution', '')}\n")

        # STAGES 4-5: iterative Solver<->Critic refinement, repeated --critic-turns times
        critic_output = None
        solver_refined_output = None
        solver_solution = (solver_output.get("solution") or "").strip()
        if self.critic and self.critic_turns > 0 and solver_solution:
            for r in range(self.critic_turns):
                tag = f"round {r + 1}/{self.critic_turns}"
                # STAGE 4: Critic critiques the current solution.
                logger.info(f"\n=== STAGE 4: SOLUTION CRITIC ({tag}) ===")
                critic_output = self._execute_agent(self.critic, task, shared_memory, "Critic agent for solution evaluation and refinement")
                shared_memory["critic_output"] = critic_output
                execution_log.append({"agent": "critic", "output": critic_output})
                logger.info(f"[Orchestrator] Critic Feedback:\n{critic_output.get('feedback', '')}\n")

                # STAGE 5: Solver refines against the feedback.
                logger.info(f"\n=== STAGE 5: SOLUTION REFINEMENT ({tag}) ===")
                solver_refined_output = self._execute_agent(self.solver, task, shared_memory, "Solver agent for solution refinement")
                shared_memory["solver_refined_output"] = solver_refined_output
                execution_log.append({"agent": "solver", "output": solver_refined_output})
                logger.info(f"[Orchestrator] Solver Soltion:\n{solver_refined_output.get('solution', '')}\n")

                # Chain: the next round critiques/refines THIS refined solution
                refined = (solver_refined_output.get("solution") or "").strip()
                if not refined:
                    logger.info(f"[Orchestrator] refinement produced no solution; stopping at {tag}")
                    break
                shared_memory["solver_output"] = solver_refined_output
        elif self.critic and self.critic_turns > 0:
            logger.info("\n=== STAGES 4-5 SKIPPED: Solver produced no solution to critique/refine ===")
        
        # Compile final result
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Get expert's organized information
        expert_final_output = shared_memory.get("expert_output", {})
        expert_summary = _extract_expert_summary(expert_final_output)
        num_tools_used = expert_final_output.get("num_tools_used", 0)
        
        # Create a serializable copy of expert_final_output to avoid circular references
        expert_output_clean = {
            "expert_summary": expert_summary,
            "num_tools_used": num_tools_used,
            "total_turns": len([conv for conv in expert_final_output.get("conversation_history", []) if conv.get("role") == "assistant"]),
            "conversation_history": expert_final_output.get("conversation_history", []),
            "tools_used": expert_final_output.get("tools_used", []),
        }
        
        result = {
            "task": task,
            "answer": solver_output.get("solution", ""),
            "reason": solver_output.get("reasoning", ""),
            "confidence": solver_output.get("confidence", ""),
            "total_duration": duration,
            "timestamp": end_time.isoformat(),
            "planner_output": planner_output,
            "expert_output": expert_output_clean,
            "solver_output": solver_output,
        }

        if critic_output:
            result["critic_output"] = critic_output
            result["solver_refined_output"] = solver_refined_output
            result["refined_answer"] = solver_refined_output.get("solution", "")
            result["refined_reason"] = solver_refined_output.get("think", "")
            result["refined_confidence"] = solver_refined_output.get("confidence", "")
        
        logger.info(f"{'='*70}")
        logger.info("EXECUTION COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"Duration: {duration:.2f}s")
        logger.info(f"Expert Tools Used: {num_tools_used}")
        logger.info(f"{'='*70}\n")
        save_path = os.path.join(self.save_dir, f"orchestrator_output__task_{task.get('id', 'single')}.json")
        try:
            save_to_json(result, save_path)
            logger.info(f"\nResults saved to: {save_path}")
        except Exception as e:
            logger.info(f"\nError: Failed to save results to '{save_path}'\n{e}\n")
        return result
    
    def _execute_agent(
        self,
        agent: Any,
        task: Dict[str, Any],
        shared_memory: Dict[str, Any],
        stage_name: str
    ) -> Dict[str, Any]:
        """Execute a single agent."""
        try:
            output = agent.run(task, shared_memory)
            logger.info(f"[{stage_name}] {agent.name} completed successfully")
            return output

        except Exception as e:
            logger.error(f"[{stage_name}] {agent.name} failed: {str(e)}")

            return {
                "agent": agent.name,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def save_result(self, result: Dict[str, Any], output_path: str):
        """Save execution result to file."""
        
        def remove_circular_refs(obj, seen=None):
            """Remove circular references from nested objects."""
            if seen is None:
                seen = set()
            
            obj_id = id(obj)
            if obj_id in seen:
                return "<circular reference>"
            
            if isinstance(obj, dict):
                seen.add(obj_id)
                return {k: remove_circular_refs(v, seen.copy()) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                seen.add(obj_id)
                return [remove_circular_refs(item, seen.copy()) for item in obj]
            else:
                return obj
        
        cleaned_result = remove_circular_refs(result)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(cleaned_result, f, indent=4, default=str)
        
        logger.info(f"[Orchestrator] Result saved to {output_path}")
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get a summary of the agents used in this run."""
        summary = {"agents": {}}

        for agent_name, agent in [
            ("planner", self.planner),
            ("expert", self.expert),
            ("solver", self.solver),
        ]:
            if agent is not None:
                summary["agents"][agent_name] = {"model": agent.model_path}

        return summary
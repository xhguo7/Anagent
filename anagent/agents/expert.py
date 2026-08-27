"""
Expert Agent: domain-specific information retrieval.
"""

import re
import os
import sys
import json
import copy
import traceback
from typing import Dict, Any, List, Optional


from utils.logger import setup_logger
logger = setup_logger(__name__)

from anagent.agents import BaseAgent
from utils.parser import response_parser
from anagent.prompts import load_prompts, load_special_tags, get_tool_prompts, response_processor
from anagent.tools.base import ToolResult
from anagent.tools.search import WebSearchTool, PubMedSearchTool, WikipediaSearchTool, ArXivSearchTool, SemanticScholarSearchTool
from anagent.tools.document import PDFParserTool, XMLParserTool, OnlineDocumentFetcher
from anagent.tools.assistant import SummarizationTool, QATool, TranslationTool, EntityExtractionTool
from anagent.tools.vision import OCRTool, ImageAnalyzer, FigureParser
from anagent.tools.knowledge import InfoLocalizer, ContextFinder, SectionExtractor, AbstractCollector
from anagent.tools.sandbox_utils import get_sandbox_tool


class ExpertAgent(BaseAgent):
    """Expert agent with access to all tools for information retrieval and processing."""
    
    # Names the model may use to signal it has gathered enough and wants to summarize.
    STOP_TOOL_ALIASES = {
        "stop_and_summarize", "stop", "summarize", "summary",
        "finish", "done", "complete",
    }

    def __init__(
        self,
        args,
        engine_kwargs: Optional[Dict[str, Any]] = None,
        engine: Optional[Any] = None,
        tool_engine: Optional[Any] = None,
        prompt_template: Optional[str] = None,
        enable_tools: Optional[List[str]] = None,
        **kwargs
    ):
        """Initialize Expert agent."""
        super().__init__(
            name="Expert",
            role="Information Retrieval and Tool Execution",
            args=args,
            engine_kwargs=engine_kwargs,
            engine=engine,
            **kwargs
        )
        
        # Store tool engine and model for tools that need LLM
        self.agent_role = "expert"
        self.tool_engine = tool_engine
        self.tool_model = args.tool_model
        self.tool_temperature = args.tool_temperature if args.tool_temperature is not None else args.temperature
        
        self.few_shot = args.few_shot
        self.prompt_template = prompt_template
        self.enable_tools = enable_tools or ["all"]

        # Task
        self.task = None
        self.memory = None
        self.planner_plan = None
        
        # Multi-turn state
        self.expert_turn = 0
        self.max_turns = 5  # default, can be overridden
        self.conversation_history = []
        self.tool_execution_history = []
        self.expert_agent_trajectory = []

        # Tool execution
        self.max_result_len = int((1280000 * 0.8) / self.max_turns)
        self.tool_execution_results = []
        self.stop_requested = False
        self.tool_call_log = []
        # True outcome of every executed tool call, recorded at execution time.
        
        # Initialize tools
        self._init_tools(engine_kwargs)
        
        # Report tools that failed to load before any work starts
        self._report_unavailable_tools()
        
        # Prepare tool descriptions
        self.tool_descriptions = self._get_tool_descriptions()

    def _get_model_path(self) -> str:
        """Get model path for Expert."""
        return self.args.expert_model

    def _get_temperature(self) -> float:
        """Get temperature for Expert."""
        return self.args.expert_temperature if self.args.expert_temperature is not None else self.args.temperature
    
    # Tools whose absence is explained by something the user can install or set.
    TOOL_REQUIREMENTS = {
        "pdf_parser": ("package", "PyPDF2", "pip install PyPDF2"),
        "ocr": ("package", "pytesseract", "pip install pytesseract"),
        "sandbox_explore": ("package", "docker", "pip install docker"),
        "web": ("credentials", "GOOGLE_API_KEY, GOOGLE_CSE_ID", None),
    }

    def _record_unavailable(self, name: str, error: Exception):
        """Record why a tool could not be constructed."""
        kind, requirement, _ = self.TOOL_REQUIREMENTS.get(name, ("error", None, None))
        if kind == "package" and not isinstance(error, ImportError):
            # Constructed for some other reason; report what actually happened.
            kind, requirement = "error", None
        if requirement:
            reason = f"missing {kind}: {requirement}"
        else:
            reason = f"{type(error).__name__}: {str(error)[:120]}"
        self.unavailable_tools[name] = reason
        logger.warning(f"[{self.name}] Tool '{name}' unavailable - {reason}")

    def _report_unavailable_tools(self):
        """Warn about tools that will not be offered to the model, and confirm if interactive."""
        if not self.unavailable_tools:
            return
        total = len(self.tools) + len(self.unavailable_tools)
        lines = [
            f"[!] {len(self.unavailable_tools)} of {total} tools are unavailable "
            f"and will be hidden from the agent:"
        ]
        for name, reason in sorted(self.unavailable_tools.items()):
            _, _, remedy = self.TOOL_REQUIREMENTS.get(name, (None, None, None))
            suffix = f"   ->  {remedy}" if remedy else ""
            lines.append(f"      - {name:<14} {reason}{suffix}")
        notice = "\n".join(lines)

        if not sys.stdin.isatty():
            # Batch runs must never block on input.
            logger.warning(notice)
            logger.warning(f"    Continuing with {len(self.tools)} tools.")
            return

        print(notice)
        print(f"    Continue with {len(self.tools)} tools [y]  |  Abort and install [N]")
        try:
            answer = input("    > ").strip().lower()
        except EOFError:
            answer = "y"
        if answer not in ("y", "yes"):
            raise SystemExit("Aborted: install the missing requirements and re-run.")

    def _init_tools(self, engine_kwargs: Optional[Dict[str, Any]]):
        """Initialize available tools."""
        self.tools = {}
        # Tools that could not be constructed, as {name: reason}.
        self.unavailable_tools = {}
        
        # Search tools
        if "all" in self.enable_tools or "search" in self.enable_tools:
            try:
                self.tools["web"] = WebSearchTool(
                    api_key=engine_kwargs.get("google_api_key") if engine_kwargs else None,
                    search_engine_id=engine_kwargs.get("google_cse_id") if engine_kwargs else None,
                )
            except Exception as e:
                self._record_unavailable("web", e)
            
            self.tools["pubmed"] = PubMedSearchTool()
            self.tools["wikipedia"] = WikipediaSearchTool()
            self.tools["arxiv"] = ArXivSearchTool()
            self.tools["semantic_scholar"] = SemanticScholarSearchTool()
        
        # Document tools
        if "all" in self.enable_tools or "document" in self.enable_tools:
            try:
                self.tools["pdf_parser"] = PDFParserTool()
            except Exception as e:
                self._record_unavailable("pdf_parser", e)
            
            self.tools["xml_parser"] = XMLParserTool()
            self.tools["online_fetcher"] = OnlineDocumentFetcher()
        
        # LLM tools
        if "all" in self.enable_tools or "llm" in self.enable_tools:
            # Pass shared tool engine if available
            tool_engine_param = {'engine': self.tool_engine} if self.tool_engine else {}
            
            self.tools["summarizer"] = SummarizationTool(
                tool_model=self.tool_model,
                engine_kwargs=engine_kwargs,
                **tool_engine_param
            )
            self.tools["extractor"] = EntityExtractionTool(
                tool_model=self.tool_model,
                engine_kwargs=engine_kwargs,
                **tool_engine_param
            )
            self.tools["translator"] = TranslationTool(
                tool_model=self.tool_model,
                engine_kwargs=engine_kwargs,
                **tool_engine_param
            )
            self.tools["qa"] = QATool(
                tool_model=self.tool_model,
                engine_kwargs=engine_kwargs,
                **tool_engine_param
            )
        
        # Vision tools
        if "all" in self.enable_tools or "vision" in self.enable_tools:
            tool_engine_param = {'engine': self.tool_engine} if self.tool_engine else {}
            
            try:
                self.tools["ocr"] = OCRTool(backend="tesseract")
            except Exception as e:
                self._record_unavailable("ocr", e)
            
            try:
                self.tools["image_analyzer"] = ImageAnalyzer(
                    tool_model=self.tool_model,
                    engine_kwargs=engine_kwargs,
                    **tool_engine_param
                )
            except Exception as e:
                self._record_unavailable("image_analyzer", e)
            
            try:
                self.tools["figure_parser"] = FigureParser(
                    tool_model=self.tool_model,
                    engine_kwargs=engine_kwargs,
                    **tool_engine_param
                )
            except Exception as e:
                self._record_unavailable("figure_parser", e)

        # Knowledge tools
        if "all" in self.enable_tools or "knowledge" in self.enable_tools:
            self.tools["info_localizer"] = InfoLocalizer()
            self.tools["context_finder"] = ContextFinder()
            self.tools["section_extractor"] = SectionExtractor()
            self.tools["abstract_collector"] = AbstractCollector()

        # Sandbox tool (auto-detect Docker or Apptainer)
        if "all" in self.enable_tools or "sandbox" in self.enable_tools:
            SandboxToolClass = get_sandbox_tool()
            if SandboxToolClass:
                self.tools["sandbox_explore"] = SandboxToolClass(
                    timeout=300,
                    memory_limit="2g",
                    auto_cleanup=True,
                )
            else:
                logger.warning(f"[{self.name}] Sandbox tool unavailable - no container runtime found")
        logger.info(f"[{self.name}] Initialized {len(self.tools)} tools: {list(self.tools.keys())}")
    def _update_tools(self, add_tools: List = None, remove_tools: List = None):
        """Update tool set"""
        if add_tools:
            self.tools = self.tools
    
    def _get_tool_descriptions(self) -> str:
        """Get comprehensive descriptions of all available tools."""
        available_tools = ", ".join(list(self.tools.keys()))
        tool_desc = f"Available Tools:\n{available_tools}\n\n{get_tool_prompts(self.tools)}"
        return tool_desc
    
    def _reset_state(self):
        """Reset multi-turn state."""
        self.expert_turn = 0
        self.conversation_history = []
        self.tool_execution_history = []
        self.expert_agent_trajectory = []
        self.tool_execution_results = []
        self.stop_requested = False
        self.tool_call_log = []

    def _get_few_shot_examples(self, shot: int):
        few_shot_examples = {
            1: f"""<think>For this in-depth analysis task, I need to first understand the core contents of the {self.task["type"]}. Let me call the info_localizer tool to localize the related information.</think><tool>info_localizer</tool><query>{self.task["few-shot"]["keywords"]}</query>""",  # one-shot
            2: f"""<think>I need to capture all the relevant contexts of this {self.task["type"]}. Let me call the context_finder tool to find and extract all the related contexts.</think><tool>context_finder</tool><query>{self.task["few-shot"]["keywords"]}</query>""",  # two-shot
            3: f"""<think>I notice that there are some related work that this {self.task["type"]} and its contexts referred to. I need to use the abstract_collector tool to extract their abstracts to better understand them and support in-depth analysis.</think><tool>abstract_collector</tool><query>{self.task["few-shot"]["references"]}</query>""",  # three-shot
        }
        return few_shot_examples[shot]

    def _tool_summary_target(self) -> int:
        """Target size (chars) for a kept tool result."""
        ctx = getattr(self.args, "expert_ctx_tokens", 32768) or 32768
        return max(1500, int(ctx * 3.5 * 0.4 / max(1, self.max_turns)))

    def _summarize_tool_result(self, formatted: str) -> str:
        """Compress an oversized tool result by SUMMARIZING it (toward the task) with the Expert model, rather than hard-truncating — so task-relevant facts are condensed, not lost."""
        if not formatted:
            return formatted
        target = self._tool_summary_target()
        if len(formatted) <= target:
            return formatted[:self.max_result_len]

        # Bound the summarizer INPUT (head + tail) so this compression call itself fits
        max_in = max(target * 6, 12000)
        if len(formatted) > max_in:
            h = max_in // 2
            src = formatted[:h] + "\n...[middle omitted]...\n" + formatted[len(formatted) - (max_in - h):]
        else:
            src = formatted

        query = self.task.get("query", "") if self.task else ""
        plan = self.planner_plan or ""
        instruction = (
            "Compress the TOOL RESULT below into a dense summary for later analysis. "
            "Preserve every task-relevant fact — numbers, statistics, findings, entity "
            "names, and citations (e.g. PMIDs) — and drop boilerplate/navigation. "
            f"Keep it under ~{max(150, target // 6)} words. Output ONLY the summary.\n\n"
            f"TASK: {query}\n\nPLAN:\n{plan}\n\nTOOL RESULT:\n{src}"
        )
        try:
            summary = (self._call_llm([{"role": "user", "content": instruction}]) or "").strip()
            if summary:
                # Bound to the per-result budget
                return summary[:target]
        except Exception as e:
            logger.warning(f"[{self.name}] tool-result summarization failed ({e}); truncating instead")
        # Fallback: head+tail slice, total length bounded to `target` (marker included).
        marker = "\n...[truncated]...\n"
        keep = max(0, target - len(marker))
        h = keep // 2
        return formatted[:h] + marker + formatted[len(formatted) - (keep - h):]

    def run(self, task: Dict[str, Any], memory: Dict[str, Any]) -> Dict[str, Any]:
        """Execute multi-turn information retrieval."""
        # Expert task
        self.task = task
        self.memory = memory
        self.planner_plan = self.memory.get("planner_output", {}).get("plan", "")

        # Determine which turn this is
        self.expert_turn = memory.get("expert_turn", 1)
        self.max_turns = memory.get("max_expert_turns", 5)
        self.max_result_len = int((1280000 * 0.8) / self.max_turns)
        
        logger.info(f"[{self.name}] Turn {self.expert_turn}/{self.max_turns}")
        # Init and run
        if self.expert_turn > 0:
            self._reset_state()
        
        # First turn: Initialize conversation
        new_agent_output = self._execute_first_turn()
        if new_agent_output["is_final"]:
            logger.info(f"[{self.name}] Turn {self.expert_turn} / {self.max_turns}\n\nExpert Summary:\n{new_agent_output['expert_summary']}\n")
            new_agent_output["expert_agent_trajectory"] = self.expert_agent_trajectory
            return new_agent_output
        else:
            logger.info(f"[{self.name}] Turn {self.expert_turn} / {self.max_turns}\n\nAgent Decision:\n{new_agent_output['raw_response']}\n\nTool Execution Output:\n{new_agent_output['tool_execution_result']}\n")
        # Middle turns
        while self.expert_turn < (self.max_turns - 1) and not self.stop_requested:
            new_agent_output = self._execute_middle_turn()
            if new_agent_output["is_final"]:
                logger.info(f"[{self.name}] Turn {self.expert_turn} / {self.max_turns}\n\nExpert Summary:\n{new_agent_output['expert_summary']}\n")
                new_agent_output["expert_agent_trajectory"] = self.expert_agent_trajectory
                return new_agent_output
            else:
                logger.info(f"[{self.name}] Turn {self.expert_turn} / {self.max_turns}\n\nAgent Decision:\n{new_agent_output['raw_response']}\n\nTool Execution Output:\n{new_agent_output['tool_execution_result']}\n")
        # Last turn: Finalize and organize
        new_agent_output = self._execute_final_turn()
        logger.info(f"[{self.name}] Turn {self.expert_turn} / {self.max_turns}\n\nExpert Summary:\n{new_agent_output['expert_summary']}\n")
        new_agent_output["expert_agent_trajectory"] = self.expert_agent_trajectory
        new_agent_output["tools_used"] = self._tool_call_parser()
        return new_agent_output
    
    def _execute_first_turn(self) -> Dict[str, Any]:
        """Execute first turn - initial tool selection."""
        assert self.expert_turn == 0

        # Update turn index
        self.expert_turn += 1
        
        logger.info(f"[{self.name}] Executing first turn - selecting initial tool...")
        # Build initial prompt
        messages = self._build_first_turn_prompt()
        self.conversation_history.extend(messages)
        
        # Generate decision
        if self.expert_turn <= self.few_shot:
            decision_response = self._get_few_shot_examples(shot=self.expert_turn)  # one-shot
        else:
            decision_response = self._call_llm(messages)  # zero-shot
        
        self.conversation_history.append({
            "role": "assistant",
            "content": decision_response
        })
        
        # Parse and execute tool
        tool_result = self._parse_and_execute_tool(decision_response)
        if tool_result["formatted"]:
            tool_result["formatted"] = self._summarize_tool_result(tool_result["formatted"].strip())
            self.tool_execution_results.append(tool_result["formatted"])

        # Check if agent wants to stop early
        should_continue = self._check_should_continue(decision_response)
        agent_output = self.get_output_format()
        if not should_continue:
            self.tool_execution_history.append({
                "turn": self.expert_turn,
                "is_final": True,
                "user_prompt": self.conversation_history[-2]["content"],
                "agent_response": tool_result,
            })
            agent_output.update({
                "turn": self.expert_turn,
                "is_final": True,
                "expert_summary": tool_result,
                "num_tools_used": len(self.tool_call_log),
                "conversation_history": self.conversation_history,
                "tool_execution_history": self.tool_execution_history,
            })
            self.expert_agent_trajectory.append(agent_output)
            logger.info(f"[{self.name}] Completed with {len(self.tool_execution_history)-1} tools used")
            return agent_output

        # User continuation prompt
        user_prompt = load_prompts(
            args=self.args,
            agent="expert",
            role="user_continue",
            agent_model=self.model_path,
            max_turns=self.max_turns,
            expert_turn=self.expert_turn,
            planner_plan=self.planner_plan,
            tool_execution_output=tool_result["formatted"],
        )        
        self.conversation_history.append({
            "role": "user",
            "content": user_prompt
        })
        
        # Store tool execution
        self.tool_execution_history.append({
            "turn": self.expert_turn,
            "is_final": False,
            "user_prompt": messages[-1]["content"],
            "agent_response": decision_response,
            "tool_execution_result": tool_result
        })
        
        # Prepare output
        agent_output = self.get_output_format()
        agent_output.update({
            "turn": self.expert_turn,
            "is_final": False,
            "raw_response": decision_response,
            "tool_execution_result": tool_result,
            "should_continue": should_continue,
            "conversation_history": self.conversation_history,
            "tool_execution_history": self.tool_execution_history,
        })
        self.expert_agent_trajectory.append(agent_output)

        return agent_output
    
    def _execute_middle_turn(self) -> Dict[str, Any]:
        """Execute middle turn - continue based on previous results."""
        # Update turn index
        self.expert_turn += 1

        logger.info(f"[{self.name}] Executing middle turn {self.expert_turn}...")
        # Generate decision
        if self.expert_turn <= self.few_shot:
            decision_response = self._get_few_shot_examples(shot=self.expert_turn)  # k-shot
        else:
            decision_response = self._call_llm(self.conversation_history)  # zero-shot
        self.conversation_history.append({
            "role": "assistant",
            "content": decision_response
        })
        
        # Parse and execute tool
        tool_result = self._parse_and_execute_tool(decision_response)
        if tool_result["formatted"]:
            tool_result["formatted"] = self._summarize_tool_result(tool_result["formatted"].strip())
            self.tool_execution_results.append(tool_result["formatted"])

        # Check if agent wants to stop early
        should_continue = self._check_should_continue(decision_response)
        agent_output = self.get_output_format()
        if not should_continue:
            self.tool_execution_history.append({
                "turn": self.expert_turn,
                "is_final": True,
                "user_prompt": self.conversation_history[-2]["content"],
                "agent_response": tool_result,
            })
            agent_output.update({
                "turn": self.expert_turn,
                "is_final": True,
                "expert_summary": tool_result,
                "num_tools_used": len(self.tool_call_log),
                "conversation_history": self.conversation_history,
                "tool_execution_history": self.tool_execution_history,
            })
            self.expert_agent_trajectory.append(agent_output)
            logger.info(f"[{self.name}] Completed with {len(self.tool_execution_history)-1} tools used")
            return agent_output

        # User continuation prompt
        if self.expert_turn < (self.max_turns - 1):
            user_prompt = load_prompts(
                args=self.args,
                agent="expert",
                role="user_continue",
                agent_model=self.model_path,
                max_turns=self.max_turns,
                expert_turn=self.expert_turn,
                planner_plan=self.planner_plan,
                tool_execution_output=tool_result["formatted"],
            )
        # User final prompt
        else:
            user_prompt = self._build_final_turn_prompt()
        self.conversation_history.append({
            "role": "user",
            "content": user_prompt
        })
        
        # Store tool execution
        self.tool_execution_history.append({
            "turn": self.expert_turn,
            "is_final": False,
            "user_prompt": user_prompt,
            "agent_response": decision_response,
            "tool_execution_result": tool_result
        })
        
        # Prepare output
        agent_output = self.get_output_format()
        agent_output.update({
            "turn": self.expert_turn,
            "is_final": False,
            "raw_response": decision_response,
            "tool_execution_result": tool_result,
            "should_continue": should_continue,
            "conversation_history": self.conversation_history,
            "tool_execution_history": self.tool_execution_history,
        })
        self.expert_agent_trajectory.append(agent_output)

        return agent_output
    
    def _execute_final_turn(self) -> Dict[str, Any]:
        """Execute final turn - organize and return knowledge."""
        # Update turn index
        self.expert_turn += 1

        logger.info(f"[{self.name}] Executing FINAL turn {self.expert_turn} - organizing knowledge...")
        # Generate final organized knowledge
        if self.max_retries > 1:
            for attempt in range(self.max_retries):
                decision_response = self._call_llm([self.conversation_history[-1]])
                if "<summary>" in decision_response:
                    break
        else:
            decision_response = self._call_llm([self.conversation_history[-1]])        
        decision_response = response_processor(self.agent_role, decision_response)

        self.conversation_history.append({
            "role": "assistant",
            "content": decision_response
        })
        logger.info(f"[Expert] Expert Final Summary:\n{decision_response}\n")
        # Store final conversation
        final_response = self._parse_and_execute_tool(decision_response)
        self.tool_execution_history.append({
            "turn": self.expert_turn,
            "is_final": True,
            "user_prompt": self.conversation_history[-2]["content"],
            "agent_response": final_response,
        })
        
        # Prepare output
        agent_output = self.get_output_format()
        agent_output.update({
            "turn": self.expert_turn,
            "is_final": True,
            "expert_summary": final_response,
            "num_tools_used": len(self.tool_call_log),
            "conversation_history": self.conversation_history,
            "tool_execution_history": self.tool_execution_history,
        })
        
        logger.info(f"[{self.name}] Completed with {len(self.tool_execution_history)-1} tools used")
        self.expert_agent_trajectory.append(agent_output)
        
        return agent_output
    
    def _build_first_turn_prompt(self) -> List[Dict[str, Any]]:
        """Build prompt for first turn."""
        
        system_prompt = load_prompts(
            args=self.args,
            agent="expert",
            role="system",
            planner_plan=self.planner_plan,
        )
        
        if self.prompt_template:
            system_prompt = self.prompt_template
        
        # Get task and progress
        task_query = self.task["query"]
        task_source = self.task.get("task_sources", None)
        task_context = self.task.get("contexts", None)
        task_image = self.task.get("image", None)
        
        # User message
        user_message = load_prompts(
            args=self.args,
            agent="expert",
            role="user_initial",
            agent_model=self.model_path,
            task_image=self.task["image"] if task_image and len(task_image)>0 else None,
            task_query=task_query,
            task_source=task_source,
            task_context=task_context,
            planner_plan=self.planner_plan,
            max_turns=self.max_turns,
            expert_turn=self.expert_turn,
            tool_description=self.tool_descriptions,
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        return messages

    def _build_final_turn_prompt(self) -> List[Dict[str, Any]]:
        """Build prompt for first turn."""
        
        # Get task and progress
        task_query = self.task["query"]
        task_source = self.task.get("task_sources", None)
        task_context = self.task.get("contexts", None)
        task_image = self.task.get("image", None)
        
        # User message
        user_message = load_prompts(
            args=self.args,
            agent="expert",
            role="user_final",
            agent_model=self.model_path,
            task_image=self.task["image"] if task_image and len(task_image)>0 else None,
            task_query=task_query,
            task_source=task_source,
            task_context=task_context,
            planner_plan=self.planner_plan if self.planner_plan else "None",
            tool_execution_output="\n\n".join(self.tool_execution_results).strip(),
        )
        
        return user_message

    def _parse_tool_name(self, tool_content: str) -> str:
        """Parse tool name"""
        parsed_tool_name = tool_content.strip().lower()
        for tool in self.tools:
            if tool in parsed_tool_name:
                return tool
        return parsed_tool_name
    
    def _parse_and_execute_tool(self, response: str) -> Dict[str, Any]:
        """Parse response and execute the specified tool."""
        
        # Check for completion signal
        think_match, think_content = response_parser(response, load_special_tags("think"), "relaxed")
        think_content = think_content.strip() if think_match else ""
        complete_match, expert_summary = response_parser(response, load_special_tags("summary"), "relaxed")
        if complete_match:
            return {
                "status": "success",
                "think": think_content,
                "expert_summary": expert_summary,
                "formatted": None,  # no user response
            }
        
        # Parse tool and query
        tool_match, tool_content = response_parser(response, load_special_tags("tool"))
        action = response.strip()

        # One tool call is executed per turn
        tool_starts = [m.start() for m in re.finditer(re.escape("<tool>"), response)]
        dropped_tool_calls = max(0, len(tool_starts) - 1)
        if dropped_tool_calls:
            action = response[:tool_starts[1]].strip()
            logger.warning(
                f"[{self.name}] {len(tool_starts)} tool calls in one response; "
                f"executing the first and dropping {dropped_tool_calls}"
            )

        if not tool_match:
            logger.warning(f"[{self.name}] Could not parse tool call from response")
            return {
                "status": "error",
                "think": None,
                "error": "Error: Failed to parse tool call from your response. Please make sure you use proper enclosure for <think>your reasoning and analysis</think>, <tool>tool call</tool>, <summary>your final summary</summary>.",
                "formatted": "Error: Failed to parse tool call from your response. Please make sure you use proper enclosure for <think>your reasoning and analysis</think>, <tool>tool call</tool>, or <summary>your final summary</summary>."
            }
        
        # Tool call
        tool_name = self._parse_tool_name(tool_content)

        # The model may signal it is finished by naming a stop action
        if tool_name.lower().strip() in self.STOP_TOOL_ALIASES:
            self.stop_requested = True
            return {
                "tool": tool_name,
                "action": action,
                "think": think_content,
                "status": "success",
                "formatted": "Organize and return your final summary now.",
            }

        # Execute tool
        if tool_name not in self.tools:
            self.tool_call_log.append({
                "turn": self.expert_turn, "tool": tool_name, "status": "error",
            })
            return {
                "tool": tool_name,
                "action": action,
                "think": think_content,
                "status": "error",
                "error": f"Tool '{tool_name}' not available",
                "formatted": f"Error: Tool '{tool_name}' not available"
            }
        
        logger.info(f"[{self.name}] Executing tool: {tool_name}")
        logger.info(f"[{self.name}] Query: {action}")
        logger.info(f"[{self.name}] Tool Context: {self.task["source"]}")
        result = self._run_tool_execution(tool_name, action)
        if isinstance(result, ToolResult):
            result = copy.deepcopy(result.to_dict())
        result["think"] = think_content
        self.tool_call_log.append({
            "turn": self.expert_turn,
            "tool": tool_name,
            "status": result.get("status", "success"),
        })

        if dropped_tool_calls and result.get("formatted"):
            result["formatted"] = (
                f"{result['formatted']}\n\n"
                f"[Note] Your response contained {len(tool_starts)} tool calls. "
                f"Only the first was executed. Issue one tool call per turn."
            )

        # Store in history
        self.tool_execution_history.append(result)
        
        return result

    def _validate_input_query_paper_list(self, s: str) -> bool:
        """Check whether a string can be successfully converted to a list of dictionaries."""
        try:
            # Try to parse the string as JSON
            parsed = json.loads(s)
            
            # Check if it's a list
            if not isinstance(parsed, list):
                return False
            
            # Check if all elements are dictionaries
            return all(isinstance(item, dict) for item in parsed)
        
        except (json.JSONDecodeError, TypeError, ValueError):
            return False

    def _parse_tool_parameters(self, tool_name: str, action: str) -> Dict[str, Any]:
        """Parse tool-specific parameters from action string."""
        params = {"query": action, "image": "NONE"}

        # Extract query - most tools have, except sandbox
        query_match, query_content = response_parser(action, load_special_tags("query"))
        if query_match:
            params["query"] = query_content
        
        ####################################
        #           1. Search
        ####################################
        # 1.1 Parse web search specific parameters
        if tool_name == "web":
            # Extract and remove date restriction
            date_match, date_content = response_parser(action, load_special_tags("date"))
            if date_match:
                params["date_restrict"] = date_content
            
            # Extract and remove language
            language_match, language_content = response_parser(action, load_special_tags("language"))
            if language_match:
                params["language"] = language_content
        
        # 1.2 Parse PubMed specific parameters
        elif tool_name == "pubmed":
            # Extract sort order
            sort_match, sort_content = response_parser(action, load_special_tags("sort"))
            if sort_match:
                params["sort"] = sort_content
            
            # Extract date_from
            date_from_match, date_from_content = response_parser(action, load_special_tags("date_from"))
            if date_from_match:
                if "date_range" not in params:
                    params["date_range"] = {}
                params["date_range"]["from"] = date_from_content
            
            # Extract date_to
            date_to_match, date_to_content = response_parser(action, load_special_tags("date_to"))
            if date_to_match:
                if "date_range" not in params:
                    params["date_range"] = {}
                params["date_range"]["to"] = date_to_content

        # 1.3 Parse Wikipedia specific parameters
        elif tool_name == "wikipedia":
            # Extract max_results
            max_results_match, max_results_content = response_parser(action, load_special_tags("max_results"))
            if max_results_match:
                try:
                    params["max_results"] = int(max_results_content)
                except ValueError:
                    pass  # keep default
            
            # Extract sentences
            sentences_match, sentences_content = response_parser(action, load_special_tags("sentences"))
            if sentences_match:
                try:
                    params["sentences"] = int(sentences_content)
                except ValueError:
                    pass  # keep default

        # 1.4 Parse arXiv specific parameters
        elif tool_name == "arxiv":
            # Extract sort_by
            sort_by_match, sort_by_content = response_parser(action, load_special_tags("sort_by"))
            if sort_by_match:
                params["sort_by"] = sort_by_content
            
            # Extract sort_order
            sort_order_match, sort_order_content = response_parser(action, load_special_tags("sort_order"))
            if sort_order_match:
                params["sort_order"] = sort_order_content
            
            # Extract category
            category_match, category_content = response_parser(action, load_special_tags("category"))
            if category_match:
                params["category"] = category_content
        
        # 1.5 Parse Semantic Scholar specific parameters
        elif tool_name == "semantic_scholar":
            # Extract venue
            venue_match, venue_content = response_parser(action, load_special_tags("venue"))
            if venue_match:
                params["venue"] = venue_content
            
            # Extract year
            year_match, year_content = response_parser(action, load_special_tags("year"))
            if year_match:
                params["year"] = year_content
            
            # Extract category
            category_match, category_content = response_parser(action, load_special_tags("category"))
            if category_match:
                params["fields_of_study"] = category_content

        ####################################
        #           2. Document
        ####################################
        # 2.1 Parse PDF parser specific parameters
        elif tool_name == "pdf_parser":
            # Update query
            if not query_match or query_content.upper() == "NONE":
                params["query"] = self.task.get("source", {}).get("url", "")
            
            # Fallback to source url
            if (params["query"].lower().strip() == "none") or (not params["query"].startswith("https://")):
                params["query"] = self.task.get("source", {}).get("url", "")

        # 2.2 Parse XML parser specific parameters
        elif tool_name == "xml_parser":
            # Update query
            if not query_match or params["query"].upper() == "NONE":
                params["query"] = self.task["source"].get("xml_url", self.task["source"]["url"])
            
            # Fallback to source url
            if (params["query"] is None) or (params["query"].lower().strip() == "none") or (not params["query"].startswith("https://")):
                params["query"] = self.task.get("source", {}).get("url", "")

        # 2.3 Parse online_fetcher specific parameters
        elif tool_name == "online_fetcher":
            # Update query
            if not query_match or params["query"].upper() == "NONE":
                params["query"] = self.task.get("source", {}).get("url", "")
            
            # Fallback to source url
            if (params["query"].lower().strip() == "none") or (not params["query"].startswith("https://")):
                params["query"] = self.task.get("source", {}).get("url", "")

        ####################################
        #           3. Knowledge
        ####################################
        # 3.1 Localize input and return the complete section
        elif tool_name in ["info_localizer", "context_finder"]:
            # Search string
            query_match, query_content = response_parser(action, load_special_tags("query"))
            if query_match:
                params["search_string"] = query_content.strip()
            else:
                # Fallback to source data keywords
                params["search_string"] = self.task["few-shot"]["keywords"]
            
            # Extract context
            context_match, context_content = response_parser(action, load_special_tags("context"))
            if context_match:
                params["document_url"] = context_content.strip()
            else:
                params["document_url"] = self.task.get("source", {}).get("url", "")
            # Fallback to source url
            if (params["document_url"].lower().strip() == "none") or (not params["document_url"].startswith("https://")):
                params["document_url"] = self.task.get("source", {}).get("url", "")

        # 3.3 Extract specified section
        elif tool_name == "section_extractor":
            # Search string
            query_match, query_content = response_parser(action, load_special_tags("query"))
            if query_match:
                params["section_spec"] = query_content.strip()
            else:
                # Fallback to source data type
                params["section_spec"] = self.task["type"]
            
            # Extract context
            context_match, context_content = response_parser(action, load_special_tags("context"))
            if context_match:
                params["document_url"] = context_content.strip()
            else:
                params["document_url"] = self.task.get("source", {}).get("url", "")
            # Fallback to source url
            if (params["document_url"].lower().strip() == "none") or (not params["document_url"].startswith("https://")):
                params["document_url"] = self.task.get("source", {}).get("url", "")
        
        # 3.4 Collect abstracts of all the specified paper(s)
        elif tool_name == "abstract_collector":
            # Search string
            query_match, query_content = response_parser(action, load_special_tags("papers"))
            if query_match and self._validate_input_query_paper_list(query_content):
                # Parse JSON string to list
                try:
                    params["papers"] = json.loads(query_content.strip())
                except (json.JSONDecodeError, ValueError):
                    # If parsing fails, keep as string (will likely cause error in tool)
                    params["papers"] = query_content.strip()
            else:
                # Fallback to source paper
                params["papers"] = [{
                    "title": self.task["source"]["title"],
                    "url": self.task["source"]["url"],
                }]

        ####################################
        #            4. Vision
        ####################################
        # 4.1 Parse OCR tool specific parameters
        elif tool_name == "ocr":
            # Extract image
            query_match, query_content = response_parser(action, load_special_tags("query"))
            if query_match:
                params["image"] = query_content.strip()
            if not query_match or params["image"].upper() == "NONE":
                params["image"] = self.task["image"]
            # Fall back to the task image unless the value is a readable file or an image URL.
            if not (os.path.isfile(params["image"])
                    or params["image"].startswith(("http://", "https://"))):
                params["image"] = self.task["image"]
            
            # Extract languages (JSON array format)
            languages_match, languages_content = response_parser(action, load_special_tags("languages"))
            if languages_match:
                try:
                    params["languages"] = json.loads(languages_content.strip())
                except (json.JSONDecodeError, ValueError):
                    pass  # keep default
            
            # Extract extract_bboxes
            extract_bboxes_match, extract_bboxes_content = response_parser(action, load_special_tags("extract_bboxes"))
            if extract_bboxes_match:
                value = extract_bboxes_content.strip().lower()
                params["extract_bboxes"] = value in ['true', 'yes', '1']
            
            # Extract confidence_threshold
            confidence_threshold_match, confidence_threshold_content = response_parser(action, load_special_tags("confidence_threshold"))
            if confidence_threshold_match:
                try:
                    threshold_value = float(confidence_threshold_content.strip())
                    # Clamp between 0.0 and 1.0
                    params["confidence_threshold"] = max(0.0, min(1.0, threshold_value))
                except ValueError:
                    pass  # keep default

        # 4.2 Parse figure_parser tool specific parameters
        elif tool_name == "figure_parser":
            # Extract image
            query_match, query_content = response_parser(action, load_special_tags("query"))
            if query_match:
                params["image"] = query_content.strip()
            if not query_match or params["image"].upper() == "NONE":
                params["image"] = self.task["image"]
            # Fall back to the task image unless the supplied value is a readable file.
            if not os.path.isfile(params["image"]):
                params["image"] = self.task["image"]
            
            # Extract extract_data
            extract_data_match, extract_data_content = response_parser(action, load_special_tags("extract_data"))
            if extract_data_match:
                value = extract_data_content.strip().lower()
                params["extract_data"] = value in ['true', 'yes', '1']
            
            # Extract figure_context
            context_match, context_content = response_parser(action, load_special_tags("context"))
            if context_match:
                params["contexts"] = context_content.strip()

        # 4.3 Parse image_analyzer tool specific parameters
        elif tool_name == "image_analyzer":
            # Extract image
            query_match, query_content = response_parser(action, load_special_tags("query"))
            if query_match:
                params["image"] = query_content.strip()
            if not query_match or params["image"].upper() == "NONE":
                params["image"] = self.task["image"]
            # Fall back to the task image unless the supplied value is a readable file.
            if not os.path.isfile(params["image"]):
                params["image"] = self.task["image"]
            
            # Extract task
            task_match, task_content = response_parser(action, load_special_tags("task"))
            if task_match:
                task_value = task_content.strip().lower()
                if task_value in ['describe', 'detect', 'qa', 'chart', 'text']:
                    params["task"] = task_value
            
            # Extract question (for qa task)
            question_match, question_content = response_parser(action, load_special_tags("question"))
            if question_match:
                params["question"] = question_content.strip()
            
            # Extract focus
            focus_match, focus_content = response_parser(action, load_special_tags("focus"))
            if focus_match:
                params["focus"] = focus_content.strip()
            
            # Extract detail_level
            detail_level_match, detail_level_content = response_parser(action, load_special_tags("detail_level"))
            if detail_level_match:
                detail_value = detail_level_content.strip().lower()
                if detail_value in ['brief', 'detailed', 'comprehensive']:
                    params["detail_level"] = detail_value

            # Extract figure_context
            context_match, context_content = response_parser(action, load_special_tags("context"))
            if context_match:
                params["contexts"] = context_content.strip()

        ####################################
        #            5. Sandbox
        ####################################
        # 5. Parse sandbox_explore specific parameters
        elif tool_name == "sandbox_explore":
            # Extract install packages (JSON array format or comma-separated)
            install_match = re.search(r'<install>\s*(\[.*?\]|.*?)\s*</install>', action, re.IGNORECASE | re.DOTALL)
            if install_match:
                install_str = install_match.group(1).strip()
                
                # Parse as JSON array if it looks like one
                if install_str.startswith('[') and install_str.endswith(']'):
                    try:
                        params["packages"] = json.loads(install_str)
                    except json.JSONDecodeError:
                        # Fallback: treat as comma-separated
                        params["packages"] = [p.strip().strip('"\'') for p in install_str.strip('[]').split(',') if p.strip()]
                else:
                    # Comma or newline separated
                    params["packages"] = [p.strip().strip('"\'') for p in re.split('[,\n]', install_str) if p.strip()]
                    
            else:
                params["packages"] = []
            
            # Extract code from <code> tags
            code_match, code_content = response_parser(action, load_special_tags("code"))
            if code_match:
                params["code"] = code_content.replace("```python", "").replace("```", "").strip()
            else:
                # Fallback: check for ```python code blocks
                code_block_match = re.search(r'```python\n(.*?)\n```', action, re.DOTALL)
                if code_block_match:
                    params["code"] = code_block_match.group(1).strip()
                else:
                    # If no code tags, treat remaining action as code
                    params["code"] = action.strip() if action.strip() else None

        ####################################
        #         6. AI Assistant
        ####################################
        # 6.1 Parse Summarizer specific parameters
        elif tool_name == "summarizer":
            # Update query
            if not query_match or params["query"].upper() == "NONE":
                params["query"] = self.task.get("source", {}).get("url", "")

            # Extract summary_type
            summary_type_match, summary_type_content = response_parser(action, load_special_tags("summary_type"))
            if summary_type_match:
                summary_type_value = summary_type_content.strip().lower()
                if summary_type_value in ['abstract', 'detailed', 'bullets', 'technical']:
                    params["summary_type"] = summary_type_value
            
            # Extract max_length
            max_length_match, max_length_content = response_parser(action, load_special_tags("max_length"))
            if max_length_match:
                try:
                    params["max_length"] = int(max_length_content.strip())
                except ValueError:
                    pass  # keep default
            
            # Extract focus_areas (JSON array format)
            focus_areas_match, focus_areas_content = response_parser(action, load_special_tags("focus_areas"))
            if focus_areas_match:
                try:
                    params["focus_areas"] = json.loads(focus_areas_content.strip())
                except (json.JSONDecodeError, ValueError):
                    pass  # keep default
            
            # Extract custom_instruction
            custom_instruction_match, custom_instruction_content = response_parser(action, load_special_tags("custom_instruction"))
            if custom_instruction_match:
                params["custom_instruction"] = custom_instruction_content.strip()

        # 6.2 Parse Entity Extractor specific parameters
        elif tool_name == "extractor":
            # Update query
            if not query_match or params["query"].upper() == "NONE":
                params["query"] = self.task.get("source", {}).get("url", "")
            
            # Extract entity_types (JSON array format)
            entity_types_match, entity_types_content = response_parser(action, load_special_tags("entity_types"))
            if entity_types_match:
                try:
                    params["entity_types"] = json.loads(entity_types_content.strip())
                except (json.JSONDecodeError, ValueError):
                    pass  # keep default
            
            # Extract domain
            domain_match, domain_content = response_parser(action, load_special_tags("domain"))
            if domain_match:
                domain_value = domain_content.strip().lower()
                if domain_value in ['general', 'biomedical', 'technical', 'legal']:
                    params["domain"] = domain_value
            
            # Extract structured_output
            structured_output_match, structured_output_content = response_parser(action, load_special_tags("structured_output"))
            if structured_output_match:
                value = structured_output_content.strip().lower()
                params["structured_output"] = value in ['true', 'yes', '1']
            
            # Extract include_context
            include_context_match, include_context_content = response_parser(action, load_special_tags("include_context"))
            if include_context_match:
                value = include_context_content.strip().lower()
                params["include_context"] = value in ['true', 'yes', '1']

        # 6.3 Parse Translator specific parameters
        elif tool_name == "translator":
            # Update query
            if not query_match:
                params["target_lang"] = "en"  # default to "en"
                
            # Extract context
            context_match, context_content = response_parser(action, load_special_tags("context"))
            if context_match:
                params["context"] = context_content.strip()
            else:
                params["context"] = self.task.get("source", {}).get("url", "")

            # Extract source_language
            source_lang_match, source_lang_content = response_parser(action, load_special_tags("source_language"))
            if source_lang_match:
                params["source_lang"] = source_lang_content.strip()
            
            # Extract domain
            domain_match, domain_content = response_parser(action, load_special_tags("domain"))
            if domain_match:
                domain_value = domain_content.strip().lower()
                if domain_value in ['general', 'technical', 'medical', 'legal']:
                    params["domain"] = domain_value
            
            # Extract style
            style_match, style_content = response_parser(action, load_special_tags("style"))
            if style_match:
                style_value = style_content.strip().lower()
                if style_value in ['formal', 'casual', 'academic']:
                    params["style"] = style_value
            
            # Extract preserve_terms (JSON array format)
            preserve_terms_match, preserve_terms_content = response_parser(action, load_special_tags("preserve_terms"))
            if preserve_terms_match:
                try:
                    params["preserve_terms"] = json.loads(preserve_terms_content.strip())
                except (json.JSONDecodeError, ValueError):
                    pass  # keep default (None)

        # 6.4 Parse QA tool specific parameters
        elif tool_name == "qa":
            # Extract context
            context_match, context_content = response_parser(action, load_special_tags("context"))
            if context_match:
                params["context"] = context_content.strip()
            else:
                params["context"] = self.task.get("source", {}).get("url", "")

            # Extract answer_style
            answer_style_match, answer_style_content = response_parser(action, load_special_tags("answer_style"))
            if answer_style_match:
                answer_style_value = answer_style_content.strip().lower()
                if answer_style_value in ['concise', 'detailed', 'technical']:
                    params["answer_style"] = answer_style_value
            
            # Extract citation_required
            citation_required_match, citation_required_content = response_parser(action, load_special_tags("citation_required"))
            if citation_required_match:
                value = citation_required_content.strip().lower()
                params["citation_required"] = value in ['true', 'yes', '1']
            
            # Extract previous_qa (JSON array format)
            previous_qa_match, previous_qa_content = response_parser(action, load_special_tags("previous_qa"))
            if previous_qa_match:
                try:
                    params["previous_qa"] = json.loads(previous_qa_content.strip())
                except (json.JSONDecodeError, ValueError):
                    pass  # keep default (None)

        return params
    
    def _run_tool_execution(self, tool_name: str, action: str) -> Dict[str, Any]:
        """Execute the tool."""
        tool = self.tools[tool_name]
        
        # Parse parameters from action for specific tools
        parsed_params = self._parse_tool_parameters(tool_name, action)
        query = parsed_params.get("query", action)
        
        # Different tools have different interfaces
        if tool_name in ["web", "wikipedia", "pubmed", "arxiv", "semantic_scholar"]:
            # 1.1 Search - web
            if tool_name == "web":
                result = tool.execute(
                    query=query,
                    date_restrict=parsed_params.get("date_restrict"),
                    language=parsed_params.get("language", "en"),
                    safe=parsed_params.get("safe", "medium")
                )

            # 1.2 Search - pubmed
            elif tool_name == "pubmed":
                result = tool.execute(
                    query=query,
                    sort=parsed_params.get("sort", "relevance"),
                    date_range=parsed_params.get("date_range"),
                    retrieve_abstracts=parsed_params.get("retrieve_abstracts", True)
                )

            # 1.3 Search - wikipedia
            elif tool_name == "wikipedia":
                result = tool.execute(
                    query=query,
                    max_results=parsed_params.get("max_results"),
                    sentences=parsed_params.get("sentences", 5)
                )

            # 1.4 Search - arxiv
            elif tool_name == "arxiv":
                result = tool.execute(
                    query=query,
                    sort_by=parsed_params.get("sort_by", "relevance"),
                    sort_order=parsed_params.get("sort_order", "descending"),
                    category=parsed_params.get("category")
                )

            # 1.5 Search - semantic_scholar
            elif tool_name == "semantic_scholar":
                result = tool.execute(
                    query=query,
                    venue=parsed_params.get("venue", None),
                    year=parsed_params.get("year", None),
                    fields_of_study=parsed_params.get("fields_of_study")
                )

            else:
                raise TypeError(f"Invalid Tool: {tool_name}")

        elif tool_name in ["pdf_parser", "xml_parser", "online_fetcher"]:
            # 2.1 Document - pdf_parser
            if tool_name == "pdf_parser":
                # Handle "NONE" to use task source
                pdf_path = query
                if query.upper() == "NONE":
                    pdf_path = self.task.get("source", {}).get("url", "")
                if not pdf_path:
                    return {
                        "tool": tool_name,
                        "action": action,
                        "pdf_path": pdf_path,
                        "status": "error",
                        "error": "No source URL available in task for PDF parsing",
                        "formatted": "Error: No source URL available for PDF parsing"
                    }
                
                # Apply parse_format to tool if provided
                if "parse_format" in parsed_params:
                    tool.parse_format = parsed_params["parse_format"]
                
                result = tool.execute(
                    pdf_path=pdf_path,
                    pages=parsed_params.get("pages", None),
                    extract_text=parsed_params.get("extract_text", True),
                    extract_tables=parsed_params.get("extract_tables", True),
                    extract_metadata=parsed_params.get("extract_metadata", True)
                )
            
            # 2.2 Document - xml_parser
            elif tool_name == "xml_parser":
                # Handle "NONE" to use task source
                xml_path = query
                if query.upper() == "NONE":
                    xml_path = self.task.get("source", {}).get("xml_url", "")
                if not xml_path:
                    return {
                        "tool": tool_name,
                        "action": action,
                        "xml_path": xml_path,
                        "status": "error",
                        "error": "No source URL available in task for XML parsing",
                        "formatted": "Error: No source URL available for XML parsing"
                    }

                # Apply parse_mode and parse_format to tool if provided
                if "parse_mode" in parsed_params:
                    tool.parse_mode = parsed_params["parse_mode"]
                if "parse_format" in parsed_params:
                    tool.parse_format = parsed_params["parse_format"]
                
                result = tool.execute(
                    xml_path=xml_path,
                    extract_sections=parsed_params.get("extract_sections", True),
                    extract_tables=parsed_params.get("extract_tables", True),
                    extract_figures=parsed_params.get("extract_figures", True),
                    extract_references=parsed_params.get("extract_references", True)
                )
            
            # 2.3 Document - online_fetcher
            elif tool_name == "online_fetcher":
                result = tool.execute(
                    url=query,
                    source_type=parsed_params.get("source_type", "auto"),
                    format=parsed_params.get("format", "auto"),
                    extract_content=parsed_params.get("extract_content", True)
                )

            else:
                raise TypeError(f"Invalid Tool: {tool_name}")

        elif tool_name in ["info_localizer", "context_finder", "section_extractor", "abstract_collector"]:
            # Handle "NONE" to use task source
            document_url = parsed_params.get("document_url", None)
            if tool_name != "abstract_collector" and not document_url:
                return {
                    "tool": tool_name,
                    "action": action,
                    "knowledge_context": knowledge_context,
                    "status": "error",
                    "error": "No source URL or context available for QA",
                    "formatted": "Error: No source URL or context available for QA"
                }

            # 3.1 - 3.2 Knowledge - info_localizer & context_finder
            if tool_name in ["info_localizer", "context_finder"]:
                result = tool.execute(
                    search_string=parsed_params["search_string"],
                    document_url=parsed_params["document_url"],
                )
                # Fallback to the first keyword
                if (tool_name == "info_localizer") and ("Search string not found" in result.formatted):
                    result = tool.execute(
                        search_string=parsed_params["search_string"].split(' ')[0].strip(),
                        document_url=parsed_params["document_url"],
                    )

            # 3.3 Knowledge - section_extractor
            elif tool_name == "section_extractor":
                result = tool.execute(
                    section_spec=parsed_params["section_spec"],
                    document_url=parsed_params["document_url"],
                )

            # 3.4 Knowledge - abstract_collector
            elif tool_name == "abstract_collector":
                result = tool.execute(papers=parsed_params["papers"])

        elif tool_name in ["ocr", "figure_parser", "image_analyzer"]:
            # Handle "NONE" to use task image
            image_to_use = parsed_params.get("image", "NONE")
            if not image_to_use:
                image_to_use = "NONE"
            image_contexts = parsed_params.get("contexts", None)
            if image_to_use.upper() == "NONE":
                image_to_use = self.task.get("image", None)
            if not image_to_use:
                return {
                    "tool": tool_name,
                    "image_to_use": image_to_use,
                    "action": action,
                    "status": "error",
                    "error": f"No image available in task for {tool_name}",
                    "formatted": f"Error: No image available in task for {tool_name}"
                }
            
            # 4.1 Vision - OCR
            if tool_name == "ocr":
                result = tool.execute(
                    image=image_to_use,
                    languages=parsed_params.get("languages"),
                    extract_bboxes=parsed_params.get("extract_bboxes", False),
                    confidence_threshold=parsed_params.get("confidence_threshold", 0.5)
                )
            # 4.2 Vision - figure_parser
            elif tool_name == "figure_parser":
                result = tool.execute(
                    image=image_to_use,
                    extract_data=parsed_params.get("extract_data", True),
                    contexts=image_contexts,
                )
            # 4.3 Vision - image_analyzer
            elif tool_name == "image_analyzer":
                result = tool.execute(
                    image=image_to_use,
                    task=parsed_params.get("task", "describe"),
                    question=parsed_params.get("question", None),
                    focus=parsed_params.get("focus", None),
                    detail_level=parsed_params.get("detail_level", "detailed"),
                    contexts=image_contexts,
                )

        elif tool_name in ["sandbox_explore"]:
            # 5.1 Sandbox - exploration
            if tool_name == "sandbox_explore":
                # Execute sandbox tool with query and task
                result = tool.execute(
                    query=action,
                    task=self.task,
                )

        elif tool_name in ["qa", "summarizer", "extractor", "translator"]:
            # 6.1 Parse Summarizer specific parameters
            if tool_name == "summarizer":
                # Handle "NONE" to use task source or tool_context
                text_to_summarize = parsed_params.get("query", self.task.get("source", {}).get("url", ""))
                if text_to_summarize.upper() == "NONE":
                    # Use tool_context (which defaults to task source URL)
                    text_to_summarize = self.task.get("source", {}).get("url", "")
                if not text_to_summarize:
                    return {
                        "tool": tool_name,
                        "action": action,
                        "text_to_summarize": text_to_summarize,
                        "status": "error",
                        "error": "No source URL or context available for summarization",
                        "formatted": "Error: No source URL or context available for summarization"
                    }
                
                result = tool.execute(
                    text=text_to_summarize,
                    summary_type=parsed_params.get("summary_type", "abstract"),
                    max_length=parsed_params.get("max_length"),
                    focus_areas=parsed_params.get("focus_areas"),
                    custom_instruction=parsed_params.get("custom_instruction")
                )
            
            # 6.2 Assistant - extractor
            elif tool_name == "extractor":
                # Handle "NONE" to use task source or tool_context
                text_to_extract = parsed_params.get("query", self.task.get("source", {}).get("url", ""))
                if text_to_extract.upper() == "NONE":
                    # Use tool_context (which defaults to task source URL)
                    text_to_extract = self.task.get("source", {}).get("url", "")
                if not text_to_extract:
                    return {
                        "tool": tool_name,
                        "action": action,
                        "text_to_extract": text_to_extract,
                        "status": "error",
                        "error": "No source URL or context available for entity extraction",
                        "formatted": "Error: No source URL or context available for entity extraction"
                    }
                
                result = tool.execute(
                    text=text_to_extract,
                    entity_types=parsed_params.get("entity_types"),
                    domain=parsed_params.get("domain", "general"),
                    structured_output=parsed_params.get("structured_output", True),
                    include_context=parsed_params.get("include_context", False)
                )
            
            # 6.3 Assistant - translator
            elif tool_name == "translator":
                # Handle "NONE" to use task source or tool_context
                text_to_translate = parsed_params.get("context", self.task.get("source", {}).get("url", ""))
                if text_to_translate.upper() == "NONE":
                    # Use task source URL
                    text_to_translate = self.task.get("source", {}).get("url", "")
                if not text_to_translate:
                    return {
                        "tool": tool_name,
                        "action": action,
                        "text_to_translate": text_to_translate,
                        "status": "error",
                        "error": "No source URL or context available for translation",
                        "formatted": "Error: No source URL or context available for translation"
                    }
                
                result = tool.execute(
                    text=text_to_translate,
                    target_lang=parsed_params.get("query", "en"),
                    source_lang=parsed_params.get("source_lang", "auto"),
                    domain=parsed_params.get("domain", "general"),
                    preserve_terms=parsed_params.get("preserve_terms"),
                    style=parsed_params.get("style", "formal")
                )

            # 6.4 Assistant - QA
            elif tool_name == "qa":
                # Handle "NONE" to use task source or context_to_qa
                context_to_qa = parsed_params.get("context", self.task.get("source", {}).get("url", ""))
                if context_to_qa.upper() == "NONE":
                    # Use task source URL
                    context_to_qa = self.task.get("source", {}).get("url", "")
                if not context_to_qa:
                    return {
                        "tool": tool_name,
                        "action": action,
                        "context_to_qa": context_to_qa,
                        "status": "error",
                        "error": "No source URL or context available for QA",
                        "formatted": "Error: No source URL or context available for QA"
                    }
                
                result = tool.execute(
                    question=query,
                    context=context_to_qa,
                    answer_style=parsed_params.get("answer_style", "concise"),
                    citation_required=parsed_params.get("citation_required", True),
                    previous_qa=parsed_params.get("previous_qa")
                )

        else:
            # Generic execution for tools not explicitly handled above
            result = tool.execute(query=action)

        return {
            "tool": tool_name,
            "action": action,
            "status": result.status.value if hasattr(result, 'status') else "success",
            "data": result.data if hasattr(result, 'data') else result,
            "formatted": result.formatted,
        }
    
    def _check_should_continue(self, response: str) -> bool:
        """Check if agent wants to continue or complete."""
        complete_match = re.search(r'<summary>', response, re.IGNORECASE)
        return not bool(complete_match)

    def _tool_call_parser(self) -> Dict:
        """Summarize tool calls from the outcomes recorded at execution time."""
        tool_usage, tool_usage_success, tool_usage_failure = {}, {}, {}
        for entry in self.tool_call_log:
            name = entry["tool"]
            if name not in tool_usage:
                tool_usage[name] = {"total": 0, "success": 0, "failure": 0}
            tool_usage[name]["total"] += 1

            if entry["status"] == "success":
                tool_usage[name]["success"] += 1
                tool_usage_success[name] = tool_usage_success.get(name, 0) + 1
            else:
                tool_usage[name]["failure"] += 1
                tool_usage_failure[name] = tool_usage_failure.get(name, 0) + 1

        tool_usage_summary = {
            "tool_usage": tool_usage,
            "tool_usage_success": tool_usage_success,
            "tool_usage_failure": tool_usage_failure,
        }
        return tool_usage_summary
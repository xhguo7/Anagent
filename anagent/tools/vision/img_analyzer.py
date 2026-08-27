"""
Image analysis tool.
"""

from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from PIL import Image


from utils.logger import setup_logger
logger = setup_logger(__name__)

from anagent.tools.base import BaseTool, ToolResult, ToolStatus
from anagent.engines import EngineHub


class ImageAnalyzer(BaseTool):
    """Image analysis tool using vision-language models."""
    
    def __init__(
        self,
        tool_model: str = "openai/gpt-4o",
        temperature: float = 0.3,
        max_tokens: int = 1024,
        engine_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Initialize image analyzer."""
        super().__init__(
            name="image_analyzer",
            description="Analyze and describe images using vision-language models",
            **kwargs
        )
        
        self.tool_model = tool_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize engine
        engine_kwargs = engine_kwargs or {}
        self.engine = EngineHub.create_engine(
            model_path=tool_model,
            temperature=temperature,
            max_tokens=max_tokens,
            **engine_kwargs
        )
        
        logger.info(f"[ImageAnalyzer] Initialized with engine: {self.engine}")
    def get_parameters(self) -> Dict[str, Any]:
        """Get parameter schema."""
        return {
            "type": "object",
            "properties": {
                "image": {
                    "type": "string",
                    "description": "Image path, URL, or PIL Image object"
                },
                "task": {
                    "type": "string",
                    "description": "Analysis task: 'describe', 'detect', 'qa', 'chart' (default: describe)"
                },
                "question": {
                    "type": "string",
                    "description": "Question about the image (for 'qa' task)"
                },
                "focus": {
                    "type": "string",
                    "description": "What to focus on (e.g., 'objects', 'text', 'colors', 'layout')"
                },
                "detail_level": {
                    "type": "string",
                    "description": "Detail level: 'brief', 'detailed', 'comprehensive' (default: detailed)"
                }
            },
            "required": ["image"]
        }
    
    def execute(
        self,
        image: Union[str, Image.Image],
        task: str = "describe",
        question: Optional[str] = None,
        focus: Optional[str] = None,
        detail_level: str = "detailed",
        contexts: str = None,
        **kwargs
    ) -> ToolResult:
        """Execute image analysis."""
        logger.info(f"[ImageAnalyzer] Analyzing image - task: {task}")
        # Validate parameters
        is_valid, error_msg = self.validate_parameters({"image": str(image)})
        if not is_valid:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                formatted=f"[Image Analyzer Error] {error_msg}",
                error=error_msg
            )
        
        try:
            # Build prompt based on task
            messages = self._build_prompt(task, question, focus, detail_level, image, contexts)
            
            # Generate analysis
            analysis = self.engine.generate(messages)
            
            logger.info(f"[ImageAnalyzer] Generated analysis: {len(analysis)} chars")
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "analysis": analysis,
                    "task": task,
                    "model": self.tool_model
                },
                formatted=analysis,
                metadata={
                    "task": task,
                    "detail_level": detail_level,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            error_msg = f"Image analysis failed: {str(e)}"
            logger.error(f"[ImageAnalyzer] Error: {error_msg}")
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                formatted=f"[Image Analyzer Error] {error_msg}",
                error=error_msg
            )
    
    def _build_prompt(
        self,
        task: str,
        question: Optional[str],
        focus: Optional[str],
        detail_level: str,
        image: Union[str, Image.Image],
        contexts: str = None
    ) -> List[Dict[str, Any]]:
        """Build prompt for image analysis."""
        
        # Task-specific prompts
        task_prompts = {
            "describe": "Describe this image in detail. Include information about objects, scene, layout, colors, and any notable features.",
            "detect": "Identify and list all objects, people, and items visible in this image. Be comprehensive and systematic.",
            "qa": question or "What do you see in this image?",
            "chart": "Analyze this figure/chart. Describe what type of visualization it is, what data it presents, key trends, and important findings.",
            "text": "Extract and transcribe all text visible in this image, maintaining the original layout where possible."
        }
        
        prompt = task_prompts.get(task, task_prompts["describe"])
        
        # Add detail level guidance
        detail_instructions = {
            "brief": " Provide a concise, high-level summary.",
            "detailed": " Provide a thorough analysis with relevant details.",
            "comprehensive": " Provide an extremely detailed, comprehensive analysis covering all aspects."
        }
        prompt += detail_instructions.get(detail_level, detail_instructions["detailed"])
        
        # Add focus if specified
        if focus:
            prompt += f" Focus particularly on: {focus}."

        # Add contexts if provided
        if contexts:
            prompt += f"\n**Additional Contexts:**\n{contexts}"
        
        # Build messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        return messages
    
    def analyze_figure(
        self,
        image: Union[str, Image.Image],
        figure_type: str = "auto",
        contexts: str = None
    ) -> ToolResult:
        """Specialized method for analyzing scientific figures/charts."""
        prompt_additions = {
            "plot": "This is a scientific plot. Describe the axes, data series, trends, and key findings.",
            "chart": "This is a chart or graph. Identify the chart type, describe the data, and summarize key insights.",
            "diagram": "This is a diagram or schematic. Describe the components, relationships, and overall structure.",
            "auto": "Analyze this scientific figure. Identify the type of visualization and provide a comprehensive analysis."
        }
        
        custom_prompt = prompt_additions.get(figure_type, prompt_additions["auto"])
        if contexts:
            custom_prompt += f"\n**Additional Contexts:**\n{contexts}"
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": custom_prompt}
                ]
            }
        ]
        
        try:
            analysis = self.engine.generate(messages)
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "analysis": analysis,
                    "figure_type": figure_type,
                    "model": self.tool_model
                },
                formatted=analysis,
                metadata={
                    "task": "figure_analysis",
                    "timestamp": datetime.now().isoformat()
                }
            )
        except Exception as e:
            error_msg = f"Figure analysis failed: {str(e)}"
            logger.error(f"[ImageAnalyzer] Error: {error_msg}")
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                formatted=f"[Image Analyzer Error] {error_msg}",
                error=str(e)
            )
    
    def compare_images(
        self,
        image1: Union[str, Image.Image],
        image2: Union[str, Image.Image],
        comparison_aspects: Optional[List[str]] = None
    ) -> ToolResult:
        """Compare two images and describe differences/similarities."""
        prompt = "Compare these two images and describe their similarities and differences."
        
        if comparison_aspects:
            aspects_str = ", ".join(comparison_aspects)
            prompt += f" Focus on: {aspects_str}."
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Image 1:"},
                    {"type": "image", "image": image1},
                    {"type": "text", "text": "Image 2:"},
                    {"type": "image", "image": image2},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        try:
            comparison = self.engine.generate(messages)
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "comparison": comparison,
                    "model": self.tool_model
                },
                formatted=comparison,
                metadata={
                    "task": "image_comparison",
                    "timestamp": datetime.now().isoformat()
                }
            )
        except Exception as e:
            error_msg = f"Image comparison failed: {str(e)}"
            logger.error(f"[ImageAnalyzer] Error: {error_msg}")
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                formatted=f"[Image Analyzer Error] {error_msg}",
                error=str(e)
            )

    def format_results_for_agent(self, result: ToolResult) -> str:
        """Format image analysis results for agent consumption."""
        if result.status != ToolStatus.SUCCESS:
            return f"Error: {result.error}"
        
        data = result.data
        formatted = []
        
        # Header
        task_type = data.get("task", "describe").title()
        
        # Model info
        model = data.get("model", "unknown")
        # formatted.append(f"Model: {model}")
        
        # Detail level from metadata if available
        metadata = result.metadata or {}
        
        # formatted.append("")
        
        # Main analysis
        analysis = data.get("analysis", "")
        formatted.append("Analysis:")
        formatted.append(analysis)
        
        return "\n".join(formatted).strip()

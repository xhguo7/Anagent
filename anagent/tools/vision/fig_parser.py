"""
Figure and chart parser tool.
"""

from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from PIL import Image
import re


from utils.logger import setup_logger
logger = setup_logger(__name__)

from anagent.tools.base import BaseTool, ToolResult, ToolStatus
from anagent.tools.vision.ocr_tool import OCRTool
from anagent.tools.vision.img_analyzer import ImageAnalyzer


class FigureParser(BaseTool):
    """Figure/chart parser for scientific visualizations."""
    
    def __init__(
        self,
        tool_model: str = "openai/gpt-4o",
        use_ocr: bool = True,
        engine_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Initialize figure parser."""
        super().__init__(
            name="figure_parser",
            description="Parse and extract data from scientific figures and charts",
            **kwargs
        )
        
        self.use_ocr = use_ocr
        
        # Initialize sub-tools
        self.image_analyzer = ImageAnalyzer(
            tool_model=tool_model,
            engine_kwargs=engine_kwargs,
        )
        
        if use_ocr:
            try:
                self.ocr_tool = OCRTool(backend="tesseract")
            except ImportError:
                logger.error("[FigureParser] OCR not available, proceeding without it")
                self.ocr_tool = None
        else:
            self.ocr_tool = None
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get parameter schema."""
        return {
            "type": "object",
            "properties": {
                "image": {
                    "type": "string",
                    "description": "Figure/chart image path or PIL Image"
                },
                "extract_data": {
                    "type": "boolean",
                    "description": "Whether to attempt numerical data extraction (default: True)"
                },
                "contexts": {
                    "type": "string",
                    "description": "Optional figure contexts"
                }
            },
            "required": ["image"]
        }
    
    def execute(
        self,
        image: Union[str, Image.Image],
        extract_data: bool = True,
        contexts: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """Execute figure parsing."""
        logger.info(f"[FigureParser] Parsing figure")
        # Validate parameters
        is_valid, error_msg = self.validate_parameters({"image": str(image)})
        if not is_valid:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                formatted=f"[Figure Parser Error] {error_msg}",
                error=error_msg
            )
        
        try:
            result_data = {
                "ocr_text": None,
                "analysis": None,
                "figure_type": None,
                "axes": None,
                "legend": None,
                "key_findings": None
            }
            
            # Step 1: OCR to extract text
            if self.ocr_tool:
                ocr_result = self.ocr_tool.execute(image=image)
                if ocr_result.is_success():
                    result_data["ocr_text"] = ocr_result.data.get("text", "")
                    logger.info(f"[FigureParser] Extracted {len(result_data['ocr_text'])} chars via OCR")
            # Step 2: VLM analysis for comprehensive understanding
            analysis_result = self.image_analyzer.analyze_figure(image=image,contexts=contexts)
            
            if analysis_result.is_success():
                result_data["analysis"] = analysis_result.data.get("analysis", "")
                
                # Parse structured information from analysis
                parsed_info = self._parse_figure_analysis(
                    result_data["analysis"],
                    result_data.get("ocr_text", "")
                )
                result_data.update(parsed_info)
            
            # Step 3: Data extraction if requested
            if extract_data:
                data_extraction = self._extract_data_points(
                    image,
                    result_data.get("analysis", ""),
                    result_data.get("ocr_text", "")
                )
                result_data["extracted_data"] = data_extraction
            
            logger.info(f"[FigureParser] Parsed figure type: {result_data.get('figure_type', 'unknown')}")
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=result_data,
                formatted=self.format_results_for_agent(result_data),
                metadata={
                    "has_ocr": self.ocr_tool is not None,
                    "has_context": contexts is not None,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            error_msg = f"Figure parsing failed: {str(e)}"
            logger.error(f"[FigureParser] Error: {error_msg}")
            return ToolResult(
                status=ToolStatus.ERROR,
                data={"error": error_msg},
                formatted=f"[Figure Parser Error] {error_msg}",
                error=error_msg
            )
    
    def _parse_figure_analysis(self, analysis: str, ocr_text: str) -> Dict[str, Any]:
        """Parse structured information from VLM analysis."""
        result = {
            "figure_type": None,
            "axes": {"x": None, "y": None},
            "legend": None,
            "key_findings": []
        }
        
        # Identify figure type
        figure_types = [
            ("bar chart", r"bar\s+chart|bar\s+graph|histogram"),
            ("line plot", r"line\s+plot|line\s+graph|line\s+chart"),
            ("scatter plot", r"scatter\s+plot|scatter\s+graph"),
            ("pie chart", r"pie\s+chart"),
            ("heatmap", r"heatmap|heat\s+map"),
            ("box plot", r"box\s+plot|boxplot"),
        ]
        
        analysis_lower = analysis.lower()
        for fig_type, pattern in figure_types:
            if re.search(pattern, analysis_lower):
                result["figure_type"] = fig_type
                break
        
        # Extract axis information
        x_axis_match = re.search(r"x[-\s]axis:?\s*([^\n.]+)", analysis_lower, re.IGNORECASE)
        if x_axis_match:
            result["axes"]["x"] = x_axis_match.group(1).strip()
        
        y_axis_match = re.search(r"y[-\s]axis:?\s*([^\n.]+)", analysis_lower, re.IGNORECASE)
        if y_axis_match:
            result["axes"]["y"] = y_axis_match.group(1).strip()
        
        # Extract key findings (sentences with keywords)
        finding_keywords = ["shows?", "indicates?", "demonstrates?", "reveals?", "trend", "increase", "decrease", "correlation"]
        sentences = re.split(r'[.!?]+', analysis)
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in finding_keywords):
                result["key_findings"].append(sentence.strip())
        
        return result
    
    def _extract_data_points(
        self,
        image: Union[str, Image.Image],
        analysis: str,
        ocr_text: str
    ) -> Dict[str, Any]:
        """Attempt to extract numerical data points."""
        data_points = {
            "numbers_found": [],
            "data_extraction_method": "pattern_matching"
        }
        
        # Extract numbers from OCR text and analysis
        number_pattern = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'
        
        if ocr_text:
            numbers_ocr = re.findall(number_pattern, ocr_text)
            data_points["numbers_found"].extend(numbers_ocr[:20])  # Limit to 20
        
        if analysis:
            # Look for numbers mentioned in analysis
            numbers_analysis = re.findall(number_pattern, analysis)
            data_points["numbers_found"].extend(numbers_analysis[:10])
        
        # Remove duplicates
        data_points["numbers_found"] = list(set(data_points["numbers_found"]))
        
        return data_points
    
    def parse_multiple_figures(
        self,
        images: List[Union[str, Image.Image]],
        contexts: Optional[List[str]] = None
    ) -> ToolResult:
        """Parse multiple figures in batch."""
        if contexts and len(contexts) != len(images):
            return ToolResult(
                status=ToolStatus.ERROR,
                data={"error": "Number of contexts must match number of images"},
                formatted="[Figure Parser Error] ERROR: Number of contexts must match number of images",
                error="Number of contexts must match number of images"
            )
        
        results = []
        
        for i, image in enumerate(images):
            curr_contexts = contexts[i] if contexts else None
            
            result = self.execute(
                image=image,
                extract_data=True,
                contexts=curr_contexts
            )
            
            if result.is_success() and result.data:
                fig_data = result.data.copy()
                fig_data["figure_index"] = i
                if curr_contexts:
                    fig_data["contexts"] = curr_contexts
                results.append(fig_data)
        
        # Prepare batch result data
        batch_data = {
            "figures": results,
            "num_figures": len(results)
        }
        
        return ToolResult(
            status=ToolStatus.SUCCESS,
            data=batch_data,
            formatted=self.format_results_for_agent(batch_data),
            metadata={
                "total_images": len(images),
                "successfully_parsed": len(results),
                "timestamp": datetime.now().isoformat()
            }
        )

    def format_results_for_agent(self, results: Dict[str, Any]) -> str:
        """Format results for agent consumption."""
        # Check if this is a batch result (multiple figures)
        if "figures" in results and "num_figures" in results:
            # Batch processing result
            formatted_output = []
            
            for idx, fig_result in enumerate(results["figures"]):
                formatted_output.append(f"---")
                formatted_output.append(f"Figure {idx + 1}:")
                
                # Format individual figure
                formatted_output.append(self._format_single_figure(fig_result))
                formatted_output.append("")
            
            return "\n".join(formatted_output).strip()
        else:
            # Single figure result
            return self._format_single_figure(results).strip()

    def _format_single_figure(self, results: Dict[str, Any], num_limit: int = 10, ocr_limit: int = 256, analysis_limit: int = 512) -> str:
        """Format a single figure's results with proper structure."""
        sections = []
        
        # 1. Figure Type
        if results.get("figure_type"):
            sections.append(f"Figure Type: {results['figure_type'].title()}")
        
        # 2. Figure Caption (if provided)
        if results.get("contexts"):
            sections.append(f"Caption: {results['contexts']}")
        
        # 3. Axes Information
        if results.get("axes"):
            axes = results["axes"]
            axes_lines = ["Axes:"]
            if axes.get("x"):
                axes_lines.append(f"  X-axis: {axes['x']}")
            if axes.get("y"):
                axes_lines.append(f"  Y-axis: {axes['y']}")
            if len(axes_lines) > 1:  # only add if found axes
                sections.append("\n".join(axes_lines))
        
        # 4. Legend
        if results.get("legend"):
            sections.append(f"Legend: {results['legend']}")
        
        # 5. Key Findings
        if results.get("key_findings") and len(results["key_findings"]) > 0:
            findings_lines = ["Key Findings:"]
            for i, finding in enumerate(results["key_findings"], 1):
                findings_lines.append(f"  {i}. {finding}")
            sections.append("\n".join(findings_lines))
        
        # 6. Extracted Data
        if results.get("extracted_data"):
            extracted = results["extracted_data"]
            if extracted.get("numbers_found") and len(extracted["numbers_found"]) > 0:
                numbers = extracted["numbers_found"][:num_limit]
                data_lines = [f"Extracted Numbers ({len(extracted['numbers_found'])} total):"]
                data_lines.append(f"  {', '.join(numbers)}")
                if len(extracted["numbers_found"]) > num_limit:
                    data_lines.append(f"  ... and {len(extracted['numbers_found']) - num_limit} more")
                sections.append("\n".join(data_lines))
        
        # 7. OCR Text (if available and meaningful)
        if results.get("ocr_text"):
            ocr_text = results["ocr_text"].strip()
            if ocr_text and len(ocr_text) > 10:  # Only show if substantial
                # Limit OCR text length
                if len(ocr_text) > ocr_limit:
                    ocr_text = ocr_text[:ocr_limit].rsplit(' ', 1)[0] + "..."
                sections.append(f"Text from Figure:\n  {ocr_text}")
        
        # 8. Comprehensive Analysis
        if results.get("analysis"):
            analysis = results["analysis"]
            # Limit analysis length
            if len(analysis) > analysis_limit:
                analysis = analysis[:analysis_limit].rsplit(' ', 1)[0] + "..."
            sections.append(f"Detailed Analysis:\n  {analysis}")
        
        # Join all sections with double newlines for readability
        return "\n\n".join(sections).strip()

"""
Self-exploration tool: Apptainer sandbox.
"""

import os
import re
import time
from typing import Dict, Any, Optional
from pathlib import Path


from utils.logger import setup_logger
logger = setup_logger(__name__)

from .sandbox import SandboxEnvironment
from .executor import ApptainerExecutor
from anagent.tools.base import BaseTool, ToolResult, ToolStatus


class SandboxExploreTool(BaseTool):
    """Tool for self-exploration through code execution in isolated Apptainer sandbox."""
    
    def __init__(
        self,
        timeout: int = 300,
        memory_limit: str = "2g",
        auto_cleanup: bool = True,
    ):
        """Initialize sandbox exploration tool."""
        super().__init__(
            name="sandbox_explore",
            description="Execute custom Python code in isolated Apptainer sandbox for self-exploration and analysis",
        )
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.auto_cleanup = auto_cleanup
        
        logger.info(f"[{self.name}] Initialized Apptainer version with timeout={timeout}s, memory={memory_limit}")
    def get_parameters(self) -> dict:
        """Get tool parameters schema."""
        return {
            "query": {
                "type": "string",
                "description": "Code with optional package installation. Format: <install>[packages]</install><code>python_code</code>",
                "required": True
            },
            "task": {
                "type": "object",
                "description": "Task information with query, context, image, source",
                "required": True
            }
        }
    
    def execute(
        self,
        query: str,
        task: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ToolResult:
        """Execute self-exploration code in sandbox."""
        if not task:
            return ToolResult(
                status=ToolStatus.ERROR,
                data={"error": "Task information required for sandbox exploration"},
                formatted="[Sandbox Exploror Error] Error: Task information required for sandbox exploration"
            )
        
        # Parse install packages and code
        packages = self._parse_packages(query)
        code = self._parse_code(query)
        
        if not code:
            return ToolResult(
                status=ToolStatus.ERROR,
                data={"error": "No code found in query. Expected format: <install>packages</install><code>python_code</code>"},
                formatted="[Sandbox Exploror Error] Error: No code found in query. Expected format: <install>packages</install><code>python_code</code>"
            )
        
        # Validate code safety (basic checks)
        if not self._validate_code(code):
            return ToolResult(
                status=ToolStatus.ERROR,
                data={"error": "Code validation failed. Potentially unsafe operations detected."},
                formatted="[Sandbox Exploror Error] Error: Code validation failed. Potentially unsafe operations detected."
            )
        
        # Execute in sandbox
        try:
            with SandboxEnvironment() as sandbox:
                # Prepare task data
                prepared_data = sandbox.prepare_task_data(task)
                
                # Add data loading helper to code
                code_with_helpers = self._inject_helpers(code, prepared_data)
                
                # Save code
                sandbox.save_code(code_with_helpers)
                
                # Create definition file
                sandbox.create_definition_file(packages)
                
                # Execute in Apptainer
                image_name = f"sandbox_explore_{int(time.time())}"
                
                with ApptainerExecutor(timeout=self.timeout, memory_limit=self.memory_limit) as executor:
                    # Build image
                    if not executor.build_image(
                        str(sandbox.workspace_path / "sandbox.def"),
                        image_name
                    ):
                        return ToolResult(
                            status=ToolStatus.ERROR,
                            data={"error": "Failed to build Apptainer image"},
                            formatted="[Sandbox Exploror Error] Error: Failed to build Apptainer image.",
                        )
                    
                    # Run container
                    success, stdout, stderr = executor.run_container(
                        image_name,
                        str(sandbox.workspace_path)
                    )
                    
                    # Cleanup image
                    if self.auto_cleanup:
                        executor.cleanup_image(image_name, str(sandbox.workspace_path))
                    
                    # Prepare result
                    if success:
                        return ToolResult(
                            status=ToolStatus.SUCCESS,
                            data={
                                "output": stdout,
                                "stderr": stderr if stderr else "",
                                "packages_installed": packages,
                                "execution_time": "completed",
                                "code": code,
                                "metadata": {
                                    "workspace": str(sandbox.workspace_path),
                                    "timeout": self.timeout,
                                    "memory_limit": self.memory_limit
                                }
                            },
                            formatted=self.format_results_for_agent(stdout, stderr)
                        )
                    else:
                        return ToolResult(
                            status=ToolStatus.ERROR,
                            data={
                                "error": "Code execution failed",
                                "stdout": stdout,
                                "stderr": stderr,
                                "formatted": f"Execution Error:\n{stderr}\n\nOutput:\n{stdout}"
                            },
                            formatted=f"[Sandbox Exploror Error] Execution Error:\n{stderr}\n\nOutput:\n{stdout}"
                        )
        
        except Exception as e:
            logger.error(f"[{self.name}] Sandbox execution error: {e}")
            return ToolResult(
                status=ToolStatus.ERROR,
                data={"error": f"Sandbox execution error: {str(e)}"},
                formatted=f"[Sandbox Exploror Error] Sandbox execution error: {str(e)}"
            )
    
    def _parse_packages(self, query: str) -> list:
        """Parse package list from <install> tags."""
        install_match = re.search(r'<install>\s*(.*?)\s*</install>', query, re.DOTALL | re.IGNORECASE)
        
        if not install_match:
            return []
        
        install_str = install_match.group(1).strip()
        
        if not install_str:
            return []
        
        # Parse as JSON array if it looks like one
        if install_str.startswith('[') and install_str.endswith(']'):
            try:
                import json
                packages = json.loads(install_str)
                # Ensure all packages are strings
                return [str(p).strip() for p in packages if p]
            except json.JSONDecodeError:
                # Fallback: treat as comma-separated within brackets
                packages = install_str.strip('[]').split(',')
                return [p.strip().strip('"\'') for p in packages if p.strip()]
        
        # Comma or newline separated (without brackets)
        packages = re.split('[,\n]', install_str)
        return [p.strip().strip('"\'') for p in packages if p.strip()]
    
    def _parse_code(self, query: str) -> Optional[str]:
        """Parse Python code from <code> tags or code blocks."""
        # Try <code> tags first (primary format)
        code_match = re.search(r'<code>(.*?)</code>', query, re.DOTALL | re.IGNORECASE)
        if code_match:
            return code_match.group(1).replace("```python", "").replace("```", "").strip()
        
        # Try ```python code blocks (fallback)
        code_match = re.search(r'```python\n(.*?)\n```', query, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        # Try ``` generic code blocks
        code_match = re.search(r'```\n(.*?)\n```', query, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        # Last resort: if query contains Python-like code (has import or def), use it
        if re.search(r'\b(import|def|class|print|for|while|if)\b', query):
            return query.strip()
        
        return None
    
    def _validate_code(self, code: str) -> bool:
        """Basic code safety validation."""
        # Critical dangerous patterns that should be blocked
        critical_patterns = [
            (r'os\.system\s*\([^)]*rm\s+-rf', 'Destructive system command'),
            (r'subprocess\..*\b(rm|del|format)\b', 'Destructive subprocess command'),
            (r'__import__\s*\(\s*["\']os["\'].*\.system', 'Malicious dynamic import'),
        ]
        
        # Warning patterns (logged but allowed due to Apptainer isolation)
        warning_patterns = [
            (r'\beval\s*\(', 'eval() usage'),
            (r'\bexec\s*\(', 'exec() usage'),
            (r'__import__', 'Dynamic import'),
            (r'compile\s*\(', 'Code compilation'),
        ]
        
        # Check critical patterns (block these)
        for pattern, description in critical_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                logger.error(f"[{self.name}] Blocked: {description} - Pattern: {pattern}")
                return False
        
        # Check warning patterns (log but allow)
        for pattern, description in warning_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                logger.warning(f"[{self.name}] Warning: {description} detected (allowed in sandbox)")
        # Basic syntax check
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            logger.error(f"[{self.name}] Syntax error in code: {e}")
            # Return True anyway - let Apptainer execution show the syntax error
            pass
        
        return True  # Apptainer provides isolation, so we allow most code
    
    def _inject_helpers(self, code: str, prepared_data: Dict[str, str]) -> str:
        """Inject helper functions and data loading code."""
        # Escape any quotes in data values for safe injection
        def escape_str(s):
            return s.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
        
        source_url = escape_str(prepared_data.get('source_url', ''))
        query = escape_str(prepared_data.get('query', ''))
        image_path = escape_str(prepared_data.get('image_path', ''))
        
        helpers = f'''
# Auto-generated helper code - DO NOT MODIFY
import os
from pathlib import Path

# Data paths
DATA_DIR = Path("/workspace/data")
OUTPUT_DIR = Path("/workspace/output")

# Task data
TASK_DATA = {{
    "source_url": "{source_url}",
    "query": "{query}",
}}

def load_source_info():
    """Load source document information.
    
    Returns:
        str: Source metadata including title, URL, IDs, etc.
        None: If source file not available
    """
    source_file = DATA_DIR / "source.txt"
    if source_file.exists():
        with open(source_file, 'r', encoding='utf-8') as f:
            return f.read()
    return None

def load_query():
    """Load task query.
    
    Returns:
        str: The task query/question
    """
    query_file = DATA_DIR / "query.txt"
    if query_file.exists():
        with open(query_file, 'r', encoding='utf-8') as f:
            return f.read()
    return TASK_DATA.get("query", "")

def load_context():
    """Load task context (extracted text, tables, or other data).
    
    Returns:
        str: Task context content
        None: If no context available
    """
    context_file = DATA_DIR / "context.txt"
    if context_file.exists():
        with open(context_file, 'r', encoding='utf-8') as f:
            return f.read()
    return None

def get_image_path():
    """Get path to task image if available.
    
    Returns:
        str: Path to image file
        None: If no image available
    """
    image_file = DATA_DIR / "image.png"
    if image_file.exists():
        return str(image_file)
    
    # Check for other image formats
    for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
        img_file = DATA_DIR / f"image{{ext}}"
        if img_file.exists():
            return str(img_file)
    
    # Return injected path if available
    injected_path = "{image_path}"
    if injected_path and injected_path != "":
        return injected_path
    
    return None

def save_output(data, filename="output.txt"):
    """Save output data to file.
    
    Args:
        data: Data to save (will be converted to string)
        filename: Output filename (saved in OUTPUT_DIR)
    
    Returns:
        str: Full path to saved file
    """
    output_file = OUTPUT_DIR / filename
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Convert data to string if needed
    if not isinstance(data, str):
        data = str(data)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(data)
    
    print(f"[Saved output to: {{output_file}}]")
    return str(output_file)

# Print workspace info for debugging
print(f"[Sandbox workspace initialized]")
print(f"[DATA_DIR: {{DATA_DIR}}]")
print(f"[OUTPUT_DIR: {{OUTPUT_DIR}}]")
print(f"[Task query: {{TASK_DATA['query'][:100] if TASK_DATA['query'] else 'None'}}...]")
print("")

# ============================================
# User code starts below
# ============================================

'''
        return helpers + code
    
    def format_results_for_agent(self, stdout: str, stderr: str = None) -> str:
        """Format execution output for display."""
        formatted = []
        
        if stdout and stdout.strip():
            formatted.append("=== Execution Output ===")
            formatted.append(stdout.strip())
        
        if stderr and stderr.strip():
            formatted.append("\n=== Warnings/Errors ===")
            formatted.append(stderr.strip())
        
        if not formatted:
            return "No output generated"
        
        return "\n".join(formatted).strip()
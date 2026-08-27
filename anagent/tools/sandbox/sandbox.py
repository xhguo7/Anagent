"""
Docker sandbox setup and management.
"""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional


from utils.logger import setup_logger
logger = setup_logger(__name__)


class SandboxEnvironment:
    """Manages Docker-based sandbox environment for code exploration."""
    
    def __init__(self, workspace_dir: Optional[str] = None):
        """Initialize sandbox environment."""
        self.workspace_dir = workspace_dir or tempfile.mkdtemp(prefix="sandbox_")
        self.workspace_path = Path(self.workspace_dir)
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.code_dir = self.workspace_path / "code"
        self.data_dir = self.workspace_path / "data"
        self.output_dir = self.workspace_path / "output"
        
        self.code_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Sandbox workspace created at: {self.workspace_dir}")
    
    def prepare_task_data(self, task: Dict[str, Any]) -> Dict[str, str]:
        """Prepare task data in sandbox environment."""
        prepared = {}
        
        # Save source metadata
        if task.get("source"):
            source_file = self.data_dir / "source.txt"
            with open(source_file, 'w') as f:
                f.write(f"Title: {task['source'].get('title', 'N/A')}\n")
                f.write(f"URL: {task['source'].get('url', 'N/A')}\n")
                f.write(f"XML URL: {task['source'].get('xml_url', 'N/A')}\n")
                f.write(f"PMC ID: {task['source'].get('pmc_id', 'N/A')}\n")
                f.write(f"BioC URL: {task['source'].get('bioc_url', 'N/A')}\n")
            prepared['source_file'] = str(source_file)
            prepared['source_url'] = task['source'].get('url', '')
        
        # Save query
        if task.get("query"):
            query_file = self.data_dir / "query.txt"
            with open(query_file, 'w') as f:
                f.write(task['query'])
            prepared['query_file'] = str(query_file)
            prepared['query'] = task['query']
        
        # Save context (text/table data)
        if task.get("context"):
            context_file = self.data_dir / "context.txt"
            with open(context_file, 'w') as f:
                f.write(task['context'])
            prepared['context_file'] = str(context_file)
            prepared['context'] = task['context']
        
        # Handle image
        if task["image"]:
            image = task["image"]
            if isinstance(image, str):
                # Image path or URL
                prepared['image_path'] = image
            else:
                # Save image object
                image_file = self.data_dir / "image.png"
                # Assume PIL Image or similar
                if hasattr(image, 'save'):
                    image.save(image_file, dpi=(300, 300))
                    prepared['image_path'] = str(image_file)
        
        return prepared
    
    def create_dockerfile(self, packages: list = None) -> str:
        """Create Dockerfile for sandbox environment."""
        packages = packages or []
        
        dockerfile_content = f"""FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    wget curl git \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy data and code
COPY data/ /workspace/data/
COPY code/ /workspace/code/

# Install Python packages
RUN pip install --no-cache-dir \\
    requests beautifulsoup4 pandas numpy \\
    matplotlib seaborn pillow \\
    {''.join(f'{pkg} ' for pkg in packages)}

# Set environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace

# Run code
CMD ["python", "/workspace/code/explore.py"]
"""
        
        dockerfile_path = self.workspace_path / "Dockerfile"
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
        
        return str(dockerfile_path)
    
    def save_code(self, code: str, filename: str = "explore.py") -> str:
        """Save exploration code to sandbox."""
        code_file = self.code_dir / filename
        with open(code_file, 'w') as f:
            f.write(code)
        return str(code_file)
    
    def get_env_info(self) -> Dict[str, str]:
        """Get environment information for code context."""
        return {
            'workspace': str(self.workspace_path),
            'data_dir': str(self.data_dir),
            'code_dir': str(self.code_dir),
            'output_dir': str(self.output_dir)
        }
    
    def cleanup(self):
        """Clean up sandbox workspace."""
        try:
            if self.workspace_path.exists():
                shutil.rmtree(self.workspace_path)
                logger.info(f"Sandbox workspace cleaned: {self.workspace_dir}")
        except Exception as e:
            logger.error(f"Error cleaning workspace: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
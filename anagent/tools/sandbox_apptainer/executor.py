"""
Apptainer executor.
"""

import os
import subprocess
import time
from typing import Dict, Any, Optional, Tuple
from pathlib import Path


from utils.logger import setup_logger
logger = setup_logger(__name__)


class ApptainerExecutor:
    """Executes code in isolated Apptainer container."""
    
    def __init__(self, timeout: int = 300, memory_limit: str = "2g"):
        """Initialize Apptainer executor."""
        self.timeout = timeout
        self.memory_limit = memory_limit
        
        # Check Apptainer availability
        if not self._check_apptainer():
            raise RuntimeError("Apptainer not found. Please install Apptainer.")
        
        logger.info("Apptainer executor initialized successfully")
    
    def _check_apptainer(self) -> bool:
        """Check if Apptainer is available."""
        try:
            result = subprocess.run(
                ['apptainer', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                logger.info(f"Apptainer found: {result.stdout.strip()}")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return False
    
    def build_image(self, definition_file: str, image_name: str) -> bool:
        """Build Apptainer image from definition file."""
        try:
            definition_path = Path(definition_file)
            image_path = definition_path.parent / f"{image_name}.sif"
            
            logger.info(f"Building Apptainer image: {image_path}")
            
            # Build the SIF image
            result = subprocess.run(
                [
                    'apptainer', 'build',
                    '--fakeroot',  # Build without root
                    str(image_path),
                    str(definition_file)
                ],
                capture_output=True,
                text=True,
                timeout=600  # Building can take time
            )
            
            if result.returncode == 0:
                logger.info(f"Apptainer image built successfully: {image_path}")
                return True
            else:
                logger.error(f"Apptainer build failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("Apptainer build timeout")
            return False
        except Exception as e:
            logger.error(f"Error building image: {e}")
            return False
    
    def run_container(
        self,
        image_name: str,
        workspace_path: str,
        container_name: Optional[str] = None
    ) -> Tuple[bool, str, str]:
        """Run Apptainer container and capture output."""
        try:
            workspace = Path(workspace_path)
            image_path = workspace / f"{image_name}.sif"
            
            if not image_path.exists():
                return False, "", f"Image not found: {image_path}"
            
            container_name = container_name or f"sandbox_{int(time.time())}"
            logger.info(f"Running container: {container_name}")
            
            # Run the container with workspace bound
            result = subprocess.run(
                [
                    'apptainer', 'run',
                    '--cleanenv',  # Clean environment
                    '--containall',  # Contain all
                    '--bind', f"{workspace}:/workspace",  # Bind workspace
                    '--pwd', '/workspace',  # Set working directory
                    str(image_path)
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=str(workspace)
            )
            
            stdout = result.stdout
            stderr = result.stderr
            success = result.returncode == 0
            
            if success:
                logger.info(f"Container executed successfully: {container_name}")
            else:
                logger.warning(f"Container exited with code {result.returncode}: {container_name}")
            
            return success, stdout, stderr
            
        except subprocess.TimeoutExpired:
            logger.error("Container execution timeout")
            return False, "", "Execution timeout"
        except Exception as e:
            logger.error(f"Error running container: {e}")
            return False, "", str(e)
    
    def cleanup_image(self, image_name: str, workspace_path: str):
        """Remove Apptainer image."""
        try:
            image_path = Path(workspace_path) / f"{image_name}.sif"
            if image_path.exists():
                image_path.unlink()
                logger.info(f"Apptainer image removed: {image_path}")
        except Exception as e:
            logger.error(f"Error removing image: {e}")
    
    def cleanup_all_sandbox_resources(self):
        """Clean up all sandbox-related resources."""
        # Apptainer containers are ephemeral, no persistent cleanup needed
        logger.info("Apptainer sandbox cleanup complete (no persistent resources)")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup_all_sandbox_resources()
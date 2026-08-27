"""
Sandbox runtime detection.
"""

import os
import subprocess
import importlib.util


from utils.logger import setup_logger
logger = setup_logger(__name__)


def check_docker_available() -> bool:
    """Check if Docker is available and running."""
    try:
        # Check Docker command exists
        result = subprocess.run(
            ['docker', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode != 0:
            return False
        
        # Check Docker daemon is running
        result = subprocess.run(
            ['docker', 'ps'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode != 0:
            return False

        # The sandbox also needs the docker Python package, not just the binary
        return importlib.util.find_spec('docker') is not None
        
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def check_apptainer_available() -> bool:
    """Check if Apptainer is available."""
    try:
        result = subprocess.run(
            ['apptainer', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
        
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def get_sandbox_tool():
    """Get the appropriate sandbox tool based on available container runtime."""
    # Check Docker first (preferred)
    if check_docker_available():
        logger.info("Docker detected - using Docker-based sandbox")
        try:
            from anagent.tools.sandbox import SandboxExploreTool
            return SandboxExploreTool
        except ImportError as e:
            logger.error(f"Failed to import Docker sandbox: {e}")
    
    # Fall back to Apptainer
    if check_apptainer_available():
        logger.info("Apptainer detected - using Apptainer-based sandbox")
        try:
            from anagent.tools.sandbox_apptainer import SandboxExploreTool
            return SandboxExploreTool
        except ImportError as e:
            logger.error(f"Failed to import Apptainer sandbox: {e}")
    
    # No container runtime available
    logger.warning("No container runtime (Docker or Apptainer) available - sandbox tool disabled")
    return None


def get_container_info() -> dict:
    """Get information about available container runtimes."""
    info = {
        'docker_available': check_docker_available(),
        'apptainer_available': check_apptainer_available(),
        'selected_runtime': None
    }
    
    if info['docker_available']:
        info['selected_runtime'] = 'docker'
    elif info['apptainer_available']:
        info['selected_runtime'] = 'apptainer'
    
    return info
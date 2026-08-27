"""
Docker executor.
"""

import os
import time
import docker
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


from utils.logger import setup_logger
logger = setup_logger(__name__)


class DockerExecutor:
    """Executes code in isolated Docker container."""
    
    def __init__(self, timeout: int = 300, memory_limit: str = "512m"):
        """Initialize Docker executor."""
        try:
            self.client = docker.from_env()
            self.timeout = timeout
            self.memory_limit = memory_limit
            logger.info("Docker client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Docker client: {e}")
            raise
    
    def build_image(self, dockerfile_path: str, image_name: str) -> bool:
        """Build Docker image from Dockerfile."""
        try:
            dockerfile_dir = Path(dockerfile_path).parent
            
            logger.info(f"Building Docker image: {image_name}")
            image, build_logs = self.client.images.build(
                path=str(dockerfile_dir),
                tag=image_name,
                rm=True,
                forcerm=True
            )
            
            # Log build process
            for log in build_logs:
                if 'stream' in log:
                    logger.debug(log['stream'].strip())
            
            logger.info(f"Docker image built successfully: {image_name}")
            return True
            
        except docker.errors.BuildError as e:
            logger.error(f"Docker build failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Error building image: {e}")
            return False
    
    def run_container(
        self,
        image_name: str,
        container_name: Optional[str] = None,
        workspace_path: Optional[str] = None,
    ) -> Tuple[bool, str, str]:
        """Run Docker container and capture output."""
        container = None
        
        try:
            # Generate container name if not provided
            if not container_name:
                container_name = f"sandbox_{int(time.time())}"
            
            logger.info(f"Running container: {container_name}")

            # Prepare volumes if workspace provided
            volumes = {}
            if workspace_path:
                data_path = Path(workspace_path) / "data"
                volumes[str(data_path)] = {'bind': '/workspace/data', 'mode': 'ro'}
            
            # Run container with resource limits
            container = self.client.containers.run(
                image_name,
                name=container_name,
                detach=True,
                mem_limit=self.memory_limit,
                network_disabled=False,  # Allow network for downloads if needed
                remove=False,  # Keep container for log retrieval
                volumes=volumes,
            )
            
            # Wait for completion with timeout
            result = container.wait(timeout=self.timeout)
            exit_code = result.get('StatusCode', -1)
            
            # Get logs
            stdout = container.logs(stdout=True, stderr=False).decode('utf-8', errors='replace')
            stderr = container.logs(stdout=False, stderr=True).decode('utf-8', errors='replace')
            
            success = exit_code == 0
            
            if success:
                logger.info(f"Container executed successfully: {container_name}")
            else:
                logger.warning(f"Container exited with code {exit_code}: {container_name}")
            
            return success, stdout, stderr
            
        except docker.errors.ContainerError as e:
            logger.error(f"Container error: {e}")
            return False, "", str(e)
        except Exception as e:
            logger.error(f"Error running container: {e}")
            return False, "", str(e)
        finally:
            # Clean up container
            if container:
                try:
                    container.remove(force=True)
                    logger.debug(f"Container removed: {container_name}")
                except Exception as e:
                    logger.error(f"Error removing container: {e}")
    
    def cleanup_image(self, image_name: str):
        """Remove Docker image."""
        try:
            self.client.images.remove(image_name, force=True)
            logger.info(f"Docker image removed: {image_name}")
        except docker.errors.ImageNotFound:
            logger.warning(f"Image not found: {image_name}")
        except Exception as e:
            logger.error(f"Error removing image: {e}")
    
    def cleanup_all_sandbox_resources(self):
        """Clean up all sandbox-related containers and images."""
        try:
            # Remove sandbox containers
            containers = self.client.containers.list(all=True, filters={'name': 'sandbox_'})
            for container in containers:
                try:
                    container.remove(force=True)
                    logger.debug(f"Removed container: {container.name}")
                except Exception as e:
                    logger.error(f"Error removing container {container.name}: {e}")
            
            # Remove sandbox images
            images = self.client.images.list(filters={'reference': 'sandbox_*'})
            for image in images:
                try:
                    self.client.images.remove(image.id, force=True)
                    logger.debug(f"Removed image: {image.tags}")
                except Exception as e:
                    logger.error(f"Error removing image: {e}")
            
            logger.info("All sandbox resources cleaned up")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup_all_sandbox_resources()
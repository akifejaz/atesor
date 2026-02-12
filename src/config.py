"""
Environment configuration and path management for the Atesor AI system.
Detects Docker context and sets up workspace directories.
"""
import os
from pathlib import Path

def is_running_in_docker() -> bool:
    """Detect if running inside Docker container."""
    if os.path.exists('/.dockerenv'):
        return True
    try:
        with open('/proc/1/cgroup', 'r') as f:
            content = f.read()
            return 'docker' in content or 'containerd' in content
    except (FileNotFoundError, PermissionError):
        pass
    if os.path.exists('/workspace'):
        try:
            test_file = '/workspace/.test_write'
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            return True
        except (PermissionError, OSError):
            pass
    return False

def get_workspace_root() -> str:
    """Get appropriate workspace root based on environment."""
    if is_running_in_docker():
        return "/workspace"
    else:
        project_root = Path(__file__).parent.parent.absolute()
        workspace = project_root / "workspace"
        workspace.mkdir(exist_ok=True)
        return str(workspace)

def get_output_dir() -> str:
    """Get output directory path."""
    workspace = get_workspace_root()
    output_dir = os.path.join(workspace, "output")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def get_repos_dir() -> str:
    """Get repositories directory path."""
    workspace = get_workspace_root()
    repos_dir = os.path.join(workspace, "repos")
    os.makedirs(repos_dir, exist_ok=True)
    return repos_dir

def get_cache_dir() -> str:
    """Get cache directory path."""
    workspace = get_workspace_root()
    cache_dir = os.path.join(workspace, ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

# Global configuration
_IN_DOCKER = is_running_in_docker()
WORKSPACE_ROOT = get_workspace_root()
OUTPUT_DIR = get_output_dir()
REPOS_DIR = get_repos_dir()
CACHE_DIR = get_cache_dir()

# Docker Configuration
CONTAINER_NAME = "atesor-ai-sandbox"
IMAGE_NAME = "atesor-ai-sandbox:latest"

def print_config():
    """Print current configuration."""
    print(f"[Config] Environment: {'Docker Container' if _IN_DOCKER else 'Host System'}")
    print(f"[Config] Workspace: {WORKSPACE_ROOT}")
    print(f"[Config] Output: {OUTPUT_DIR}")
    print(f"[Config] Repos: {REPOS_DIR}")

__all__ = [
    'is_running_in_docker', 'get_workspace_root', 'get_output_dir',
    'get_repos_dir', 'get_cache_dir', 'WORKSPACE_ROOT', 'OUTPUT_DIR',
    'REPOS_DIR', 'CACHE_DIR', 'print_config',
    'CONTAINER_NAME', 'IMAGE_NAME',
]

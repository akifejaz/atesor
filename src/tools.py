"""
Low-level utility functions for command execution, file I/O, and safety validation.
Provides Docker-aware file and command operations.
"""

import re
import subprocess
import logging
import os
from typing import Optional, Tuple

from src.state import CommandResult
from src.config import CONTAINER_NAME

logger = logging.getLogger(__name__)


# ============================================================================
# COMMAND VALIDATION
# ============================================================================

class CommandValidator:
    """
    Smart command validation that doesn't block legitimate operations.
    
    Uses whitelist approach: explicitly allow safe patterns, block dangerous ones.
    """
    
    # Allowed command patterns (whitelist)
    SAFE_COMMANDS = {
        # File operations
        r'^ls\s+',
        r'^cat\s+',
        r'^head\s+',
        r'^tail\s+',
        r'^find\s+',
        r'^tree\s+',
        r'^file\s+',
        r'^stat\s+',
        
        # Search operations (FIXED: Previously blocked!)
        r'^grep\s+(-[a-zA-Z]+\s+)*',  # grep with optional flags
        r'^awk\s+',
        r'^sed\s+',
        r'^wc\s+',
        
        # Build operations
        r'^cmake\s+',
        r'^make\s+',
        r'^ninja\s+',
        r'^./configure\s+',
        r'^./autogen\.sh',
        r'^cargo\s+',
        r'^npm\s+',
        r'^pip\s+',
        r'^python\s+',
        r'^go\s+',
        
        # Package management
        r'^apt-get\s+',
        r'^apt\s+',
        r'^apk\s+',
        r'^yum\s+',
        r'^dnf\s+',
        
        # Git operations
        r'^git\s+',
        
        # Compilation
        r'^gcc\s+',
        r'^g\+\+\s+',
        r'^clang\s+',
        r'^rustc\s+',
        
        # Testing
        r'^ctest\s+',
        r'^pytest\s+',
        r'^cargo\s+test',
        
        # Directory operations
        r'^mkdir\s+-p\s+',
        r'^cd\s+',
        r'^pwd',
        
        # Text processing
        r'^echo\s+',
        r'^printf\s+',
        r'^tr\s+',
        r'^cut\s+',
        r'^sort\s+',
        r'^uniq\s+',
        
        # Environment and Shell
        r'^export\s+',
        r'^env\s+',
        r'^[A-Z_][A-Z0-9_.]*=.*',  # Environment variable assignments
        
        # System
        r'^touch\s+',
        r'^chmod\s+',
        r'^patch\s+',
        r'^diff\s+',
        r'^tar\s+',
        r'^unzip\s+',
    }
    
    # Dangerous patterns to block (blacklist)
    DANGEROUS_PATTERNS = {
        r'rm\s+-rf\s+/',                      # Recursive root deletion
        r':\(\)\{\s*:\|:\&\s*\}',             # Fork bomb
        r'dd\s+if=/dev/zero\s+of=/dev/sd',    # Disk wipe
        r'mkfs\.',                            # Format filesystem
        r'fdisk',                             # Partition editing
        r'wget.*\|\s*bash',                   # Remote code execution
        r'curl.*\|\s*sh',                     # Remote code execution
        r'eval\s+',                           # Eval is dangerous
        r'exec\s+',                           # Exec is dangerous
        r'/etc/shadow',                       # System files
        r'/etc/passwd',                       # System files
    }
    
    def is_safe(self, command: str) -> Tuple[bool, str]:
        """
        Check if a command is safe to execute.
        
        Returns:
            (is_safe, reason)
        """
        # Check dangerous patterns first
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, command):
                return False, f"Blocked dangerous pattern: {pattern}"
        
        # Check if matches safe patterns
        for pattern in self.SAFE_COMMANDS:
            if re.match(pattern, command.strip()):
                return True, "Matches safe command pattern"
        
        # Default deny for unknown patterns
        logger.warning(f"Unknown command pattern (consider adding to whitelist): {command[:100]}")
        return False, "Unknown command pattern (not in whitelist)"


# Global validator instance
_validator = CommandValidator()


# ============================================================================
# DOCKER CONFIGURATION
# ============================================================================

class DockerConfig:
    """Docker container configuration."""
    
    CONTAINER_NAME = CONTAINER_NAME
    WORKSPACE_PATH = "/workspace"  # Path inside container
    
    @staticmethod
    def is_container_running() -> bool:
        """Check if Docker container is running."""
        try:
            result = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", DockerConfig.CONTAINER_NAME],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout.strip() == "true"
        except Exception as e:
            logger.error(f"Failed to check container status: {e}")
            return False


# ============================================================================
# COMMAND EXECUTION
# ============================================================================

def execute_command(
    command: str,
    cwd: Optional[str] = None,
    timeout: int = 300,
    validate: bool = True,
    use_docker: bool = True,  # NEW: Execute in Docker by default
) -> CommandResult:
    """
    Execute a shell command safely.
    
    Args:
        command: Shell command to execute
        cwd: Working directory (path inside container if use_docker=True)
        timeout: Timeout in seconds
        validate: Whether to validate command safety
        use_docker: Whether to execute in Docker container (default: True)
    
    Returns:
        CommandResult with output and status
    """
    import time
    start_time = time.time()
    
    # Validate command safety
    if validate:
        is_safe, reason = _validator.is_safe(command)
        if not is_safe:
            logger.warning(f"Command blocked: {command[:100]}")
            logger.warning(f"Reason: {reason}")
            return CommandResult(
                command=command,
                exit_code=1,
                stdout="",
                stderr=f"Command blocked by safety validation: {reason}",
                duration_seconds=0.0,
            )
    
    # Execute command
    try:
        # FIXED: Execute in Docker container by default
        if use_docker:
            if not DockerConfig.is_container_running():
                logger.error(f"Docker container '{DockerConfig.CONTAINER_NAME}' is not running")
                return CommandResult(
                    command=command,
                    exit_code=1,
                    stdout="",
                    stderr=f"Docker container '{DockerConfig.CONTAINER_NAME}' is not running",
                    duration_seconds=0.0,
                )
            
            # Build docker exec command
            docker_cmd = ["docker", "exec"]
            
            # Add working directory if specified
            if cwd:
                # Ensure we use the container path
                container_cwd = cwd if cwd.startswith(DockerConfig.WORKSPACE_PATH) else cwd
                docker_cmd.extend(["-w", container_cwd])
            
            # Add container name and command
            docker_cmd.extend([DockerConfig.CONTAINER_NAME, "bash", "-c", command])
            
            logger.debug(f"Executing in Docker: {' '.join(docker_cmd[:5])}... (cwd: {cwd})")
            
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        else:
            # Execute on host (for git clone, etc.)
            logger.debug(f"Executing on host: {command[:100]}")
            
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            logger.debug(f"Command succeeded: {command[:50]}...")
        else:
            logger.warning(f"Command failed (exit {result.returncode}): {command[:50]}...")
            logger.warning(f"stderr: {result.stderr[:200]}")
        
        return CommandResult(
            command=command,
            exit_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            duration_seconds=duration,
        )
    
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        logger.error(f"Command timed out after {timeout}s: {command[:50]}")
        return CommandResult(
            command=command,
            exit_code=-1,
            stdout="",
            stderr=f"Command timed out after {timeout} seconds",
            duration_seconds=duration,
        )
    
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Command failed with exception: {e}")
        return CommandResult(
            command=command,
            exit_code=-1,
            stdout="",
            stderr=str(e),
            duration_seconds=duration,
        )


# ============================================================================
# FILE OPERATIONS
# ============================================================================

def read_file(filepath: str, max_lines: int = 1000, use_docker: bool = True) -> str:
    """
    Read file content with line limit.
    
    Args:
        filepath: Path to file (inside container if use_docker=True)
        max_lines: Maximum lines to read
        use_docker: Whether to read from Docker container
    
    Returns:
        File content (truncated if needed)
    """
    if use_docker:
        # Read from Docker container
        result = execute_command(f"head -n {max_lines} {filepath}", use_docker=True)
        if result.success:
            return result.stdout
        else:
            logger.error(f"Failed to read file {filepath} from container: {result.stderr}")
            return f"Error reading file: {result.stderr}"
    else:
        # Read from host
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = []
                for i, line in enumerate(f):
                    if i >= max_lines:
                        lines.append(f"\n... (truncated after {max_lines} lines)")
                        break
                    lines.append(line)
                return ''.join(lines)
        except Exception as e:
            logger.error(f"Failed to read file {filepath}: {e}")
            return f"Error reading file: {e}"


def write_file(filepath: str, content: str, use_docker: bool = True) -> bool:
    """
    Write content to file.
    
    Args:
        filepath: Path to file
        content: Content to write
        use_docker: Whether to write in Docker container
    
    Returns:
        Success status
    """
    if use_docker:
        # Write to Docker container using base64 for robustness
        import base64
        encoded = base64.b64encode(content.encode('utf-8')).decode('ascii')
        
        # Ensure directory exists
        dir_path = '/'.join(filepath.split('/')[:-1])
        if dir_path:
            execute_command(f"mkdir -p {dir_path}", use_docker=True)
            
        result = execute_command(
            f"echo '{encoded}' | base64 -d > {filepath}",
            use_docker=True
        )
        return result.success
    else:
        # Write to host
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Wrote {len(content)} bytes to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to write file {filepath}: {e}")
            return False


def file_exists(filepath: str, use_docker: bool = True) -> bool:
    """
    Check if file exists.
    
    Args:
        filepath: Path to file
        use_docker: Whether to check in Docker container
    
    Returns:
        True if file exists
    """
    if use_docker:
        result = execute_command(f"test -e {filepath}", use_docker=True)
        return result.success
    else:
        return os.path.exists(filepath)


def list_directory(dirpath: str, max_depth: int = 1, use_docker: bool = True) -> str:
    """
    List directory contents.
    
    Args:
        dirpath: Directory path
        max_depth: Maximum depth to traverse
        use_docker: Whether to list in Docker container
    
    Returns:
        Directory listing
    """
    result = execute_command(f"ls -la {dirpath}", use_docker=use_docker)
    return result.stdout if result.success else result.stderr


def get_file_tree(dirpath: str, max_depth: int = 3, use_docker: bool = True) -> str:
    """
    Get tree view of directory.
    
    Args:
        dirpath: Directory path
        max_depth: Maximum depth
        use_docker: Whether to use Docker container
    
    Returns:
        Tree view
    """
    # Try tree command first
    result = execute_command(
        f"tree -L {max_depth} -I '.git|node_modules|__pycache__|.venv' {dirpath}",
        use_docker=use_docker
    )
    
    if result.success:
        return result.stdout
    
    # Fallback to find
    result = execute_command(
        f"find {dirpath} -maxdepth {max_depth} -type f | head -n 100",
        use_docker=use_docker
    )
    return result.stdout


# ============================================================================
# GREP AND SEARCH
# ============================================================================

def grep_content(pattern: str, filepath: str, context_lines: int = 0, use_docker: bool = True) -> str:
    """
    Search for pattern in file.
    
    Args:
        pattern: Regular expression pattern
        filepath: Path to file
        context_lines: Lines of context before/after match
        use_docker: Whether to search in Docker container
    
    Returns:
        Matching lines
    """
    cmd = f"grep -E '{pattern}' {filepath}"
    if context_lines > 0:
        cmd += f" -C {context_lines}"
    
    result = execute_command(cmd, use_docker=use_docker)
    return result.stdout if result.success else ""


# ============================================================================
# PATCH OPERATIONS
# ============================================================================

def apply_patch(patch_content: str, filepath: Optional[str] = None, cwd: Optional[str] = None, use_docker: bool = True) -> bool:
    """
    Apply a patch to one or more files.
    
    Args:
        patch_content: Unified diff or content to append
        filepath: Specific file to patch (if None, assumes unified diff with its own paths)
        cwd: Working directory
        use_docker: Whether to run in Docker
    
    Returns:
        Success status
    """
    if not patch_content:
        return False
        
    if use_docker:
        # Write patch to container's /tmp
        import uuid
        patch_filename = f"/tmp/agent_{uuid.uuid4().hex[:8]}.patch"
        if not write_file(patch_filename, patch_content, use_docker=True):
            return False
        
        try:
            if filepath:
                # Apply to specific file
                # First check if it's a unified diff or just raw content
                if "--- " in patch_content and "+++ " in patch_content:
                    cmd = f"patch {filepath} < {patch_filename}"
                else:
                    # Raw content append (legacy fallback)
                    # Use a more robust way to append in container
                    import base64
                    encoded = base64.b64encode(patch_content.encode('utf-8')).decode('ascii')
                    cmd = f"echo '{encoded}' | base64 -d >> {filepath}"
            else:
                # Standard unified diff - apply with -p1
                # Try dry-run first
                dry_run = execute_command(f"patch -p1 --dry-run < {patch_filename}", cwd=cwd, use_docker=True)
                if not dry_run.success:
                    logger.warning(f"Patch dry-run failed: {dry_run.stderr}")
                    # Try p0 as fallback
                    dry_run = execute_command(f"patch -p0 --dry-run < {patch_filename}", cwd=cwd, use_docker=True)
                    if not dry_run.success:
                        return False
                    cmd = f"patch -p0 < {patch_filename}"
                else:
                    cmd = f"patch -p1 < {patch_filename}"
            
            result = execute_command(cmd, cwd=cwd, use_docker=True)
            return result.success
        finally:
            execute_command(f"rm {patch_filename}", use_docker=True)
    else:
        # Non-docker implementation (for local testing/setup)
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.patch') as f:
                f.write(patch_content)
                patch_file = f.name
            
            if filepath:
                if "--- " in patch_content and "+++ " in patch_content:
                    cmd = f"patch {filepath} < {patch_file}"
                else:
                    with open(filepath, 'a') as f:
                        f.write(f"\n{patch_content}\n")
                    return True
            else:
                cmd = f"patch -p1 < {patch_file}"
            
            result = execute_command(cmd, cwd=cwd, use_docker=False)
            os.remove(patch_file)
            return result.success
        except Exception as e:
            logger.error(f"Failed to apply patch on host: {e}")
            return False


# ============================================================================
# DOCKER OPERATIONS
# ============================================================================

def docker_exec(container_name: str, command: str, cwd: Optional[str] = None, timeout: int = 300) -> CommandResult:
    """
    Execute command in Docker container.
    
    Args:
        container_name: Name of Docker container
        command: Command to execute
        cwd: Working directory inside container
        timeout: Timeout in seconds
    
    Returns:
        CommandResult
    """
    # Save original container name
    original_container = DockerConfig.CONTAINER_NAME
    DockerConfig.CONTAINER_NAME = container_name
    
    try:
        result = execute_command(command, cwd=cwd, timeout=timeout, use_docker=True)
        return result
    finally:
        # Restore original container name
        DockerConfig.CONTAINER_NAME = original_container


def clone_repository(url: str, target_path: str) -> CommandResult:
    """
    Clone a git repository.
    
    Args:
        url: Repository URL
        target_path: Target directory (on host)
    
    Returns:
        CommandResult
    """
    # Check if already exists (on host)
    if file_exists(f"{target_path}/.git", use_docker=False):
        logger.info(f"Repository already exists at {target_path}, pulling...")
        return execute_command(f"cd {target_path} && git pull", use_docker=False)
    
    # Clone (on host, not in Docker)
    logger.info(f"Cloning {url} to {target_path}...")
    return execute_command(f"git clone --depth 1 {url} {target_path}", use_docker=False)


# Export all functions
__all__ = [
    "execute_command",
    "read_file",
    "write_file",
    "file_exists",
    "list_directory",
    "get_file_tree",
    "grep_content",
    "apply_patch",
    "docker_exec",
    "clone_repository",
    "CommandValidator",
    "DockerConfig",
]
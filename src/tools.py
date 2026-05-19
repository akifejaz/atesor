"""
Low-level utility functions for command execution, file I/O, and safety validation.
Provides Docker-aware file and command operations.
"""

import random
import re
import subprocess
import logging
import os
import time
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
        r'^ls(\s+.*)?$',
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
        r'^cmake(\s+.*)?$',
        r'^make(\s+.*)?$',
        r'^ninja(\s+.*)?$',
        r'^./configure.*',
        r'^./Configure.*',
        r'^./autogen\.sh.*',
        r'^./bootstrap.*',
        r'^./buildconf\.sh.*',
        r'^autoreconf(\s+.*)?$',
        r'^autoconf(\s+.*)?$',
        r'^automake(\s+.*)?$',
        r'^libtoolize(\s+.*)?$',
        r'^aclocal(\s+.*)?$',
        r'^autoheader(\s+.*)?$',
        r'^meson(\s+.*)?$',
        r'^cargo(\s+.*)?$',
        r'^npm(\s+.*)?$',
        r'^pip(\s+.*)?$',
        r'^python(\s+.*)?$',
        r'^python3(\s+.*)?$',
        r'^perl(\s+.*)?$',
        r'^go(\s+.*)?$',
        r'^git(\s+.*)?$',
        
        # Package management
        r'^apt-get\s+',
        r'^apt\s+',
        r'^apk\s+',
        r'^yum\s+',
        r'^dnf\s+',

        # File downloads (without piping to shell)
        r'^wget\s+(?!.*\|\s*(bash|sh|zsh|fish))',   # wget but not wget | shell
        r'^curl\s+(?!.*\|\s*(bash|sh|zsh|fish))',   # curl but not curl | shell
        
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
        r'^sh\s+',
        r'^bash\s+',
        
        # System
        r'^touch\s+',
        r'^chmod\s+',
        r'^patch\s+',
        r'^diff\s+',
        r'^tar\s+',
        r'^unzip\s+',
        r'^cp\s+',
        r'^mv\s+',
        r'^rm\s+(?!-rf\s+/)', # Allow rm but not rm -rf /
        
        # Discovery
        r'^which\s+',
        r'^uname\s+',
        r'^test\s+',
        r'^base64\s+',
        
        # Shell conditionals and control flow
        r'^if\s+',
        r'^\[\s+',           # [ test ]
        r'^\[\[\s+',         # [[ test ]]
        r'^then\s*',
        r'^else\s*',
        r'^elif\s+',
        r'^fi\s*$',
        r'^for\s+',
        r'^while\s+',
        r'^do\s*',
        r'^done\s*$',
        r'^case\s+',
        r'^esac\s*$',
        r'^\{\s*$',
        r'^\}\s*$',
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
    timeout: int = 1800,
    validate: bool = True,
    use_docker: bool = True,
) -> CommandResult:
    """
    Execute a shell command safely.
    
    Args:
        command: Shell command to execute
        cwd: Working directory (will be translated to container path if use_docker=True)
        timeout: Timeout in seconds
        validate: Whether to validate command safety
        use_docker: Whether to execute in Docker container (default: True)
    
    Returns:
        CommandResult with output and status
    """
    from src.config import _IN_DOCKER, WORKSPACE_ROOT
    
    start_time = time.time()
    
    # If we are ALREADY in Docker, we cannot use docker exec normally
    # and we don't need to.
    if _IN_DOCKER:
        use_docker = False
    
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
    
    # Auto-correct wrong Alpine package names in apk commands
    if _is_apk_command(command):
        command = _fix_apk_package_names(command)

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
                # Translate host path to container path if necessary
                container_cwd = cwd
                if cwd.startswith(str(WORKSPACE_ROOT)):
                    container_cwd = cwd.replace(str(WORKSPACE_ROOT), DockerConfig.WORKSPACE_PATH)
                elif not cwd.startswith(DockerConfig.WORKSPACE_PATH):
                    # If it's not relative and not in /workspace, it might be an absolute host path
                    # try to see if it contains 'workspace'
                    if 'workspace' in cwd:
                        parts = cwd.split('workspace', 1)
                        container_cwd = DockerConfig.WORKSPACE_PATH + parts[1]
                
                docker_cmd.extend(["-w", container_cwd])
            
            # Add container name and command
            # Wrap apk commands with flock to serialize across concurrent agents
            exec_command = command
            if _is_apk_command(command):
                exec_command = f"flock -w 120 /tmp/apk.lock sh -c '{command}'"
            docker_cmd.extend([DockerConfig.CONTAINER_NAME, "bash", "-c", exec_command])
            
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
            # Retry apk commands that fail due to database lock contention
            # (common when multiple agents share one container in batch mode)
            if _is_apk_lock_error(result) and _is_apk_command(command):
                APK_LOCK_RETRIES = 5
                for retry in range(1, APK_LOCK_RETRIES + 1):
                    delay = 5 * retry + random.uniform(0, 3)
                    logger.info(
                        f"apk lock contention, retry {retry}/{APK_LOCK_RETRIES} "
                        f"after {delay:.0f}s: {command[:60]}"
                    )
                    time.sleep(delay)
                    retry_result = subprocess.run(
                        docker_cmd if use_docker else command,
                        shell=not use_docker,
                        cwd=None if use_docker else cwd,
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                    )
                    if not _is_apk_lock_error(retry_result):
                        result = retry_result
                        duration = time.time() - start_time
                        if result.returncode == 0:
                            logger.info(f"apk command succeeded on retry {retry}")
                        break

            if result.returncode != 0:
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


def _is_apk_lock_error(result) -> bool:
    """Check if a command failed due to apk database lock contention."""
    return result.returncode != 0 and (
        "Unable to lock database" in (getattr(result, 'stderr', '') or '')
        or "Unable to lock database" in (getattr(result, 'stdout', '') or '')
    )


def _is_apk_command(command: str) -> bool:
    """Check if a command involves the apk package manager."""
    cmd = command.strip()
    return cmd.startswith("apk ") or "apk add" in cmd or "apk update" in cmd or "apk del" in cmd


def _fix_apk_package_names(command: str) -> str:
    """Auto-correct common Debian/Ubuntu package names to Alpine equivalents."""
    from src.knowledge import ALPINE_PACKAGE_CORRECTIONS
    fixed = command
    for wrong, correct in ALPINE_PACKAGE_CORRECTIONS.items():
        # Only replace whole package names (word boundaries)
        fixed = re.sub(r'\b' + re.escape(wrong) + r'\b', correct, fixed)
    if fixed != command:
        logger.info(f"Auto-corrected apk package names: {command[:80]} → {fixed[:80]}")
    return fixed


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
                # Apply to specific file - always try as a diff, never raw append
                if "--- " in patch_content and "+++ " in patch_content:
                    cmd = f"patch {filepath} < {patch_filename}"
                elif "@@ " in patch_content:
                    # Has diff hunks but missing headers - try patch -p0
                    cmd = f"patch -p0 {filepath} < {patch_filename}"
                else:
                    # Not a valid diff format - reject to avoid corrupting the file
                    logger.warning(f"Patch content is not a valid unified diff - rejecting to avoid file corruption")
                    return False
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
                elif "@@ " in patch_content:
                    cmd = f"patch -p0 {filepath} < {patch_file}"
                else:
                    logger.warning(f"Patch content is not a valid unified diff - rejecting")
                    os.remove(patch_file)
                    return False
            else:
                cmd = f"patch -p1 < {patch_file}"
            
            result = execute_command(cmd, cwd=cwd, use_docker=False)
            os.remove(patch_file)
            return result.success
        except Exception as e:
            logger.error(f"Failed to apply patch on host: {e}")
            return False


# Export all functions
__all__ = [
    "execute_command",
    "read_file",
    "write_file",
    "file_exists",
    "apply_patch",
    "CommandValidator",
    "DockerConfig",
]
"""
Low-level utility functions for command execution, file I/O, and safety validation.
Provides Docker-aware file and command operations.
"""

import re
import subprocess
import logging
import os
import json
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
    DEFAULT_SAFE_COMMANDS = {
        # File operations
        r"^ls\s+",
        r"^cat\s+",
        r"^head\s+",
        r"^tail\s+",
        r"^find\s+",
        r"^tree\s+",
        r"^file\s+",
        r"^stat\s+",
        # Search operations (FIXED: Previously blocked!)
        r"^grep\s+(-[a-zA-Z]+\s+)*",  # grep with optional flags
        r"^awk\s+",
        r"^sed\s+",
        r"^wc\s+",
        # Build operations
        r"^cmake\s+",
        r"^make\s+",
        r"^ninja\s+",
        r"^./configure.*",
        r"^./autogen\.sh.*",
        r"^cargo\s+",
        r"^npm\s+",
        r"^pip\s+",
        r"^python\s+",
        r"^go\s+",
        # Package management
        r"^apt-get\s+",
        r"^apt\s+",
        r"^apk\s+",
        r"^yum\s+",
        r"^dnf\s+",
        # Git operations
        r"^git\s+",
        # Compilation
        r"^gcc\s+",
        r"^g\+\+\s+",
        r"^clang\s+",
        r"^rustc\s+",
        # Testing
        r"^ctest\s+",
        r"^pytest\s+",
        r"^cargo\s+test",
        # Directory operations
        r"^mkdir\s+-p\s+",
        r"^cd\s+",
        r"^pwd",
        # Text processing
        r"^echo\s+",
        r"^printf\s+",
        r"^tr\s+",
        r"^cut\s+",
        r"^sort\s+",
        r"^uniq\s+",
        # Environment and Shell
        r"^export\s+",
        r"^env\s+",
        r"^[A-Z_][A-Z0-9_.]*=.*",  # Environment variable assignments
        r"^sh\s+",
        r"^bash\s+",
        # System
        r"^touch\s+",
        r"^chmod\s+",
        r"^patch\s+",
        r"^diff\s+",
        r"^tar\s+",
        r"^unzip\s+",
        r"^cp\s+",
        r"^mv\s+",
        r"^rm\s+(?!-rf\s+/)",  # Allow rm but not rm -rf /
        # Discovery
        r"^which\s+",
        r"^uname\s+",
        r"^test\s+",
        r"^base64\s+",
        # Shell conditionals and control flow
        r"^if\s+",
        r"^\[\s+",  # [ test ]
        r"^\[\[\s+",  # [[ test ]]
        r"^then\s*",
        r"^else\s*",
        r"^elif\s+",
        r"^fi\s*$",
        r"^for\s+",
        r"^while\s+",
        r"^do\s*",
        r"^done\s*$",
        r"^case\s+",
        r"^esac\s*$",
        r"^\{\s*$",
        r"^\}\s*$",
    }

    # Dangerous patterns to block (blacklist)
    DEFAULT_DANGEROUS_PATTERNS = {
        r"rm\s+-rf\s+/",  # Recursive root deletion
        r":\(\)\{\s*:\|:\&\s*\}",  # Fork bomb
        r"dd\s+if=/dev/zero\s+of=/dev/sd",  # Disk wipe
        r"mkfs\.",  # Format filesystem
        r"fdisk",  # Partition editing
        r"wget.*\|\s*bash",  # Remote code execution
        r"curl.*\|\s*sh",  # Remote code execution
        r"eval\s+",  # Eval is dangerous
        r"exec\s+",  # Exec is dangerous
        r"/etc/shadow",  # System files
        r"/etc/passwd",  # System files
    }

    def __init__(self, policy_path: Optional[str] = None):
        self.safe_commands = set(self.DEFAULT_SAFE_COMMANDS)
        self.dangerous_patterns = set(self.DEFAULT_DANGEROUS_PATTERNS)
        self._load_external_policy(
            policy_path or os.getenv("ATESOR_COMMAND_POLICY_PATH", "")
        )

    def _load_external_policy(self, policy_path: str) -> None:
        """
        Optionally load allow/deny regex lists from JSON policy.
        Schema:
        {
          "mode": "append" | "replace",
          "allow": ["^ls\\s+"],
          "deny": ["rm\\s+-rf\\s+/"]
        }
        """
        if not policy_path:
            return
        if not os.path.exists(policy_path):
            logger.warning(f"Command policy file not found: {policy_path}")
            return

        try:
            with open(policy_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load command policy '{policy_path}': {e}")
            return

        mode = str(payload.get("mode", "append")).strip().lower()
        allow = payload.get("allow", [])
        deny = payload.get("deny", [])

        if mode == "replace":
            self.safe_commands = set(allow)
            self.dangerous_patterns = set(deny)
        else:
            self.safe_commands.update(allow)
            self.dangerous_patterns.update(deny)

    def is_safe(self, command: str) -> Tuple[bool, str]:
        """
        Check if a command is safe to execute.

        Returns:
            (is_safe, reason)
        """
        # Check dangerous patterns first
        for pattern in self.dangerous_patterns:
            if re.search(pattern, command):
                return False, f"Blocked dangerous pattern: {pattern}"

        # Check if matches safe patterns
        for pattern in self.safe_commands:
            if re.match(pattern, command.strip()):
                return True, "Matches safe command pattern"

        # Default deny for unknown patterns
        logger.warning(
            f"Unknown command pattern (consider adding to whitelist): {command[:100]}"
        )
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
                [
                    "docker",
                    "inspect",
                    "-f",
                    "{{.State.Running}}",
                    DockerConfig.CONTAINER_NAME,
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout.strip() == "true"
        except Exception as e:
            logger.error(f"Failed to check container status: {e}")
            return False


# ============================================================================
# COMMAND EXECUTION WITH INCREMENTAL POLLING
# ============================================================================


def _get_polling_intervals() -> list:
    """
    Generate incremental polling intervals to check if a command completed.
    
    Returns:
        List of (delay_seconds, cumulative_seconds) tuples
        - Checks every 5 seconds for 25 seconds (5 checks)
        - Checks every 10 seconds for 30 seconds (3 checks)
        - Checks every 30 seconds afterwards
    """
    intervals = []
    cumulative = 0
    
    # Phase 1: Every 5 seconds for up to 25 seconds
    for i in range(5):
        intervals.append((5, cumulative + 5))
        cumulative += 5
    
    # Phase 2: Every 10 seconds for next 30 seconds
    for i in range(3):
        intervals.append((10, cumulative + 10))
        cumulative += 10
    
    # Phase 3: Every 30 seconds onwards (will be capped by total timeout)
    # We calculate how many more 30-second intervals we need
    # This is handled dynamically in the polling loop
    intervals.append(("dynamic_30s", cumulative))
    
    return intervals


def execute_command_with_polling(
    docker_cmd: list,
    timeout: int = 1200,
    cwd: Optional[str] = None,
) -> CommandResult:
    """
    Execute a Docker command with incremental polling instead of blocking wait.
    
    This allows us to check on long-running processes periodically (every 5-10-30 seconds)
    instead of blocking for the entire timeout duration.
    
    Args:
        docker_cmd: List of command arguments (starts with "docker", "exec", etc.)
        timeout: Total timeout in seconds
        cwd: Working directory (for logging only)
    
    Returns:
        CommandResult with output and status
    """
    start_time = time.time()
    
    try:
        # Start the process without blocking
        process = subprocess.Popen(
            docker_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        
        logger.debug(f"Process started with PID {process.pid}, polling until timeout {timeout}s")
        
        # Get polling intervals
        intervals = _get_polling_intervals()
        poll_count = 0
        
        for check_delay, cumulative_time in intervals:
            elapsed = time.time() - start_time
            
            if elapsed >= timeout:
                # Total timeout reached
                logger.warning(f"Total timeout ({timeout}s) reached after {elapsed:.1f}s and {poll_count} polls")
                process.kill()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                return CommandResult(
                    command=" ".join(docker_cmd),
                    exit_code=-1,
                    stdout="",
                    stderr=f"Command timed out after {timeout} seconds ({poll_count} polling checks)",
                    duration_seconds=elapsed,
                )
            
            # Calculate actual wait time (don't exceed minimum remaining time)
            remaining_time = timeout - elapsed
            
            if check_delay == "dynamic_30s":
                actual_delay = min(30, remaining_time)
            else:
                actual_delay = min(check_delay, remaining_time)
            
            # Wait and check periodically
            time.sleep(actual_delay)
            poll_count += 1
            
            # Poll to see if process completed
            returncode = process.poll()
            
            if returncode is not None:
                # Process completed!
                elapsed_total = time.time() - start_time
                stdout, stderr = process.communicate()
                
                logger.debug(
                    f"Command completed after {elapsed_total:.1f}s and {poll_count} polling checks "
                    f"(exit code: {returncode})"
                )
                
                return CommandResult(
                    command=" ".join(docker_cmd),
                    exit_code=returncode,
                    stdout=stdout or "",
                    stderr=stderr or "",
                    duration_seconds=elapsed_total,
                )
            
            logger.debug(
                f"Poll #{poll_count}: Process still running after {elapsed:.1f}s "
                f"(next check in {actual_delay}s)"
            )
        
        # Fallback: if we exit the loop without completing, timeout
        elapsed = time.time() - start_time
        logger.error(f"Command polling loop ended without completion after {elapsed:.1f}s")
        process.kill()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        
        return CommandResult(
            command=" ".join(docker_cmd),
            exit_code=-1,
            stdout="",
            stderr=f"Command timed out after {timeout} seconds ({poll_count} polling checks)",
            duration_seconds=elapsed,
        )
    
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Polling execution failed: {e}")
        return CommandResult(
            command=" ".join(docker_cmd),
            exit_code=-1,
            stdout="",
            stderr=str(e),
            duration_seconds=elapsed,
        )


def execute_command(
    command: str,
    cwd: Optional[str] = None,
    timeout: int = 1200,
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

    import time

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

    # Execute command
    try:
        # FIXED: Execute in Docker container by default
        if use_docker:
            if not DockerConfig.is_container_running():
                logger.error(
                    f"Docker container '{DockerConfig.CONTAINER_NAME}' is not running"
                )
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
                    container_cwd = cwd.replace(
                        str(WORKSPACE_ROOT), DockerConfig.WORKSPACE_PATH
                    )
                elif not cwd.startswith(DockerConfig.WORKSPACE_PATH):
                    # If it's not relative and not in /workspace, it might be an absolute host path
                    # try to see if it contains 'workspace'
                    if "workspace" in cwd:
                        parts = cwd.split("workspace", 1)
                        container_cwd = DockerConfig.WORKSPACE_PATH + parts[1]

                docker_cmd.extend(["-w", container_cwd])

            # Add container name and command
            docker_cmd.extend([DockerConfig.CONTAINER_NAME, "bash", "-c", command])

            logger.debug(
                f"Executing in Docker: {' '.join(docker_cmd[:5])}... (cwd: {cwd})"
            )

            # Use polling-based execution for incremental feedback
            result = execute_command_with_polling(docker_cmd, timeout=timeout, cwd=cwd)
        else:
            # Execute on host (for git clone, etc.)
            logger.debug(f"Executing on host: {command[:100]}")

            # Use polling-based execution for consistent incremental feedback
            host_cmd = ["bash", "-c", command]
            result = execute_command_with_polling(host_cmd, timeout=timeout, cwd=cwd)

        duration = time.time() - start_time

        if result.exit_code == 0:
            logger.debug(f"Command succeeded: {command[:50]}...")
        else:
            logger.warning(
                f"Command failed (exit {result.exit_code}): {command[:50]}..."
            )
            logger.warning(f"stderr: {result.stderr[:200]}")

        return CommandResult(
            command=command,
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
            duration_seconds=result.duration_seconds,
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
            logger.error(
                f"Failed to read file {filepath} from container: {result.stderr}"
            )
            return f"Error reading file: {result.stderr}"
    else:
        # Read from host
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                lines = []
                for i, line in enumerate(f):
                    if i >= max_lines:
                        lines.append(f"\n... (truncated after {max_lines} lines)")
                        break
                    lines.append(line)
                return "".join(lines)
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

        encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")

        # Ensure directory exists
        dir_path = "/".join(filepath.split("/")[:-1])
        if dir_path:
            execute_command(f"mkdir -p {dir_path}", use_docker=True)

        result = execute_command(
            f"echo '{encoded}' | base64 -d > {filepath}", use_docker=True
        )
        return result.success
    else:
        # Write to host
        try:
            with open(filepath, "w", encoding="utf-8") as f:
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


def apply_patch(
    patch_content: str,
    filepath: Optional[str] = None,
    cwd: Optional[str] = None,
    use_docker: bool = True,
) -> bool:
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

    # CRITICAL SAFETY CHECK: Detect if patch_content contains raw shell commands
    # This is a common failure mode when LLM JSON parsing fails
    suspicious_patterns = [
        "sed -i",
        "grep -",
        "echo '",
        'echo "',
        "awk ",
        "rm -",
        "touch ",
    ]
    if any(pattern in patch_content for pattern in suspicious_patterns):
        logger.error(
            f"BLOCKED: Patch contains suspicious shell commands: {patch_content[:200]}..."
        )
        return False

    # Ensure patch input always has trailing newline for `patch` parser robustness.
    if not patch_content.endswith("\n"):
        patch_content = patch_content + "\n"

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
                    # Try standard unified patch application first.
                    dry_run = execute_command(
                        f"patch -p1 --dry-run < {patch_filename}",
                        cwd=cwd,
                        use_docker=True,
                    )
                    if dry_run.success:
                        cmd = f"patch -p1 < {patch_filename}"
                    else:
                        dry_run = execute_command(
                            f"patch -p0 --dry-run < {patch_filename}",
                            cwd=cwd,
                            use_docker=True,
                        )
                        if dry_run.success:
                            cmd = f"patch -p0 < {patch_filename}"
                        else:
                            # Fallback: patch an explicit target file.
                            cmd = f"patch {filepath} < {patch_filename}"
                else:
                    # Raw content append (legacy fallback)
                    # Use a more robust way to append in container
                    import base64

                    encoded = base64.b64encode(patch_content.encode("utf-8")).decode(
                        "ascii"
                    )
                    cmd = f"echo '{encoded}' | base64 -d >> {filepath}"
            else:
                # Standard unified diff - apply with -p1
                # Try dry-run first
                dry_run = execute_command(
                    f"patch -p1 --dry-run < {patch_filename}", cwd=cwd, use_docker=True
                )
                if not dry_run.success:
                    logger.warning(f"Patch dry-run failed: {dry_run.stderr}")
                    # Try p0 as fallback
                    dry_run = execute_command(
                        f"patch -p0 --dry-run < {patch_filename}",
                        cwd=cwd,
                        use_docker=True,
                    )
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

            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".patch"
            ) as f:
                f.write(patch_content)
                patch_file = f.name

            if filepath:
                if "--- " in patch_content and "+++ " in patch_content:
                    dry_run = execute_command(
                        f"patch -p1 --dry-run < {patch_file}",
                        cwd=cwd,
                        use_docker=False,
                    )
                    if dry_run.success:
                        cmd = f"patch -p1 < {patch_file}"
                    else:
                        dry_run = execute_command(
                            f"patch -p0 --dry-run < {patch_file}",
                            cwd=cwd,
                            use_docker=False,
                        )
                        if dry_run.success:
                            cmd = f"patch -p0 < {patch_file}"
                        else:
                            cmd = f"patch {filepath} < {patch_file}"
                else:
                    with open(filepath, "a") as f:
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

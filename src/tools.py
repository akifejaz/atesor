"""Low-level utilities for command execution and file I/O.

Provides Docker-aware file and command operations together with
safety validation of shell commands.
"""

import logging
import os
import random
import re
import subprocess
import time
from typing import Optional, Tuple

from src.state import CommandResult

# Config values (_IN_DOCKER, WORKSPACE_ROOT) are imported lazily inside
# functions to keep the module-level import surface minimal and avoid
# import cycles.

logger = logging.getLogger(__name__)


# ============================================================================
# COMMAND VALIDATION
# ============================================================================


class CommandValidator:
    """Smart command validation that allows legitimate operations.

    Uses a whitelist approach: explicitly allow safe patterns and
    block dangerous ones.
    """

    # Allowed command patterns (whitelist)
    SAFE_COMMANDS = {
        # File operations
        r"^ls(\s+.*)?$",
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
        r"^cmake(\s+.*)?$",
        r"^make(\s+.*)?$",
        r"^ninja(\s+.*)?$",
        # Configure/bootstrap scripts: catch ./configure, ./Configure,
        # ./config, ./buildconf, ./buildconf.sh, ./autogen.sh,
        # ./bootstrap[.sh], ./setup, etc. Some upstreams (openssl, curl)
        # ship driver scripts without extensions so we keep the suffix
        # optional.
        r"^\./[A-Za-z][A-Za-z0-9_.-]*(\s+.*)?$",
        r"^autoreconf(\s+.*)?$",
        r"^autoconf(\s+.*)?$",
        r"^automake(\s+.*)?$",
        r"^libtoolize(\s+.*)?$",
        r"^aclocal(\s+.*)?$",
        r"^autoheader(\s+.*)?$",
        r"^meson(\s+.*)?$",
        r"^cargo(\s+.*)?$",
        r"^npm(\s+.*)?$",
        r"^pip(\s+.*)?$",
        r"^python(\s+.*)?$",
        r"^python3(\s+.*)?$",
        r"^perl(\s+.*)?$",
        r"^go(\s+.*)?$",
        r"^git(\s+.*)?$",
        # Package management
        r"^apt-get\s+",
        r"^apt\s+",
        r"^apk\s+",
        r"^yum\s+",
        r"^dnf\s+",
        # File downloads (without piping to shell)
        r"^wget\s+(?!.*\|\s*(bash|sh|zsh|fish))",  # wget but not wget | shell
        r"^curl\s+(?!.*\|\s*(bash|sh|zsh|fish))",  # curl but not curl | shell
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
        r"^sleep\s+\d",  # backoff retries
        r"^ln\s+",  # symlinks (gosec used ln -s)
        r"^ldconfig(\s+.*)?$",
        r"^update-alternatives\s+",
        # Versioned Go binaries installed via `golang.org/dl/goX.Y` (cariddi)
        r"^/root/go/bin/go\d+\.\d+\b",
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
    DANGEROUS_PATTERNS = {
        r"rm\s+-rf\s+/",  # Recursive root deletion
        r":\(\)\{\s*:\|:\&\s*\}",  # Fork bomb
        r"dd\s+if=/dev/zero\s+of=/dev/sd",  # Disk wipe
        r"mkfs\.",  # Format filesystem
        r"fdisk",  # Partition editing
        r"wget.*\|\s*bash",  # Remote code execution
        r"curl.*\|\s*sh",  # Remote code execution
        r"\beval\s+",  # Eval is dangerous (word-boundary, see below)
        # NOTE: `exec` is NOT blocked outright because
        # `find … -exec sed -i {} \;` is a very common (and safe)
        # refactor pattern emitted by the fixer. Truly dangerous use of
        # `exec` is already covered by other blocks (no
        # shell-redirect-exec, no remote pipe, etc.). Observed root
        # cause for ecoji, garble, go-fasttld failures on 2026-05-23.
        r"(?:^|\s|;|&)exec\s+\S+",  # bare `exec foo` at start of cmd
        r"/etc/shadow",  # System files
        r"/etc/passwd",  # System files
        # Container swap fiddling (host-only, never appropriate).
        r"mkswap\s+|swapon\s+",
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
        logger.warning(
            f"Unknown command pattern (consider adding to whitelist): "
            f"{command[:100]}"
        )
        return False, "Unknown command pattern (not in whitelist)"


# Global validator instance
_validator = CommandValidator()


# ============================================================================
# DOCKER CONFIGURATION
# ============================================================================


class _DockerConfigMeta(type):
    """Metaclass exposing the current container name dynamically.

    ``DockerConfig.CONTAINER_NAME`` resolves the *current* container
    (respecting the ``ATESOR_CONTAINER`` override) at access time, not
    at import time.
    """

    @property
    def CONTAINER_NAME(cls) -> str:  # noqa: N802
        from src.platforms import get_container_name

        return get_container_name()


class DockerConfig(metaclass=_DockerConfigMeta):
    """Docker container configuration. CONTAINER_NAME is profile-driven."""

    WORKSPACE_PATH = "/workspace"  # Path inside container

    @staticmethod
    def is_container_running() -> bool:
        """Check if the active profile's Docker container is running."""
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
# COMMAND EXECUTION
# ============================================================================


def execute_command(
    command: str,
    cwd: Optional[str] = None,
    timeout: int = 1800,
    validate: bool = True,
    use_docker: bool = True,
) -> CommandResult:
    """Execute a shell command safely.

    Args:
        command: Shell command to execute.
        cwd: Working directory (translated to a container path when
            ``use_docker`` is True).
        timeout: Timeout in seconds.
        validate: Whether to validate command safety.
        use_docker: Whether to execute in the Docker container.
            Defaults to True.

    Returns:
        A ``CommandResult`` with output and status.
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

    # Auto-correct wrong package names for the active distro
    if _is_pkg_command(command):
        command = _fix_pkg_names(command)
        command = _strip_bundled_toolchain_packages(command)

    # Execute command
    try:
        # FIXED: Execute in Docker container by default
        if use_docker:
            if not DockerConfig.is_container_running():
                logger.error(
                    f"Docker container "
                    f"'{DockerConfig.CONTAINER_NAME}' is not running"
                )
                return CommandResult(
                    command=command,
                    exit_code=1,
                    stdout="",
                    stderr=(
                        f"Docker container "
                        f"'{DockerConfig.CONTAINER_NAME}' is not running"
                    ),
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
                    # If it's not relative and not in /workspace, it
                    # might be an absolute host path; try to see if it
                    # contains 'workspace'.
                    if "workspace" in cwd:
                        parts = cwd.split("workspace", 1)
                        container_cwd = DockerConfig.WORKSPACE_PATH + parts[1]

                docker_cmd.extend(["-w", container_cwd])

            # Add container name and command.
            # Wrap package-manager commands with flock to serialize
            # across concurrent agents.
            exec_command = command
            if _is_pkg_command(command):
                from src.platforms import get_active_profile

                lock = get_active_profile().pkg_lock_file
                exec_command = f"flock -w 120 {lock} sh -c '{command}'"

            # Wrap with in-container `timeout` so a runaway process gets
            # SIGKILLed by the container kernel even if the python-side
            # docker-exec client is torn down. Without this,
            # qemu-emulated processes (notably `go build`) orphan after
            # subprocess.TimeoutExpired and keep burning CPU forever —
            # see PID 2496758 incident on dnsx (2026-05-21).
            #
            # We give the in-container timeout a small head-start (~30s
            # shorter than python's timeout) so it fires first; python's
            # timeout remains the belt-and-braces fallback if
            # `timeout(1)` itself is missing.
            inner_timeout = max(timeout - 30, 10)
            shell_quoted = exec_command.replace("'", "'\"'\"'")
            exec_command = (
                f"timeout --signal=KILL {inner_timeout}s "
                f"bash -c '{shell_quoted}'"
            )
            docker_cmd.extend(
                [DockerConfig.CONTAINER_NAME, "bash", "-c", exec_command]
            )

            logger.debug(
                f"Executing in Docker: "
                f"{' '.join(docker_cmd[:5])}... (cwd: {cwd})"
            )

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
            # Retry package-manager commands that fail due to database
            # lock contention (common when multiple agents share one
            # container in batch mode).
            if _is_pkg_lock_error(result) and _is_pkg_command(command):
                pkg_lock_retries = 5
                for retry in range(1, pkg_lock_retries + 1):
                    delay = 5 * retry + random.uniform(0, 3)
                    logger.info(
                        f"pkg lock contention, retry "
                        f"{retry}/{pkg_lock_retries} "
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
                    if not _is_pkg_lock_error(retry_result):
                        result = retry_result
                        duration = time.time() - start_time
                        if result.returncode == 0:
                            logger.info(
                                f"pkg command succeeded on retry {retry}"
                            )
                        break

            if result.returncode != 0:
                logger.warning(
                    f"Command failed (exit {result.returncode}): "
                    f"{command[:50]}..."
                )
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

        # Belt-and-braces: even though we wrap with `timeout --signal=KILL`
        # in-container, also try to kill any matching process from the host so
        # we never leave qemu-emulated builds running forever after a timeout.
        # This is best-effort and never raises.
        if use_docker:
            try:
                cmd_token = (
                    command.strip().split()[0] if command.strip() else ""
                )
                if cmd_token:
                    subprocess.run(
                        [
                            "docker",
                            "exec",
                            DockerConfig.CONTAINER_NAME,
                            "pkill",
                            "-9",
                            "-f",
                            cmd_token,
                        ],
                        capture_output=True,
                        timeout=10,
                    )
            except Exception as kill_exc:  # pragma: no cover - best effort
                logger.debug(
                    f"post-timeout pkill failed (non-fatal): {kill_exc}"
                )

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


def _is_pkg_lock_error(result) -> bool:
    """Detect package-manager lock contention (apk, apt, dpkg)."""
    text = (getattr(result, "stderr", "") or "") + (
        getattr(result, "stdout", "") or ""
    )
    if result.returncode == 0:
        return False
    return (
        "Unable to lock database" in text  # apk
        or "Could not get lock" in text  # apt-get
        or "dpkg frontend lock" in text  # apt-get
        or "Resource temporarily unavailable" in text
        and "lock" in text
    )


# Backward-compat alias
_is_apk_lock_error = _is_pkg_lock_error


def _is_pkg_command(command: str) -> bool:
    """Check if a command invokes the system package manager.

    Recognizes apk, apt, apt-get, and dpkg invocations.
    """
    cmd = command.strip()
    for prefix in ("apk ", "apt-get ", "apt ", "dpkg "):
        if cmd.startswith(prefix):
            return True
    for needle in (
        "apk add",
        "apk update",
        "apk del",
        "apt-get install",
        "apt-get update",
        "apt-get remove",
        "apt install",
        "apt update",
        "apt remove",
    ):
        if needle in cmd:
            return True
    return False


# Backward-compat alias
_is_apk_command = _is_pkg_command


def _fix_pkg_names(command: str) -> str:
    """Auto-correct package names that are wrong for the active distro."""
    from src.platforms import get_active_profile

    profile = get_active_profile()
    corrections = profile.name_corrections
    if not corrections:
        return command
    fixed = command
    for wrong, correct in corrections.items():
        fixed = re.sub(r"\b" + re.escape(wrong) + r"\b", correct, fixed)
    if fixed != command:
        logger.info(
            f"Auto-corrected package names for {profile.name}: "
            f"{command[:80]} → {fixed[:80]}"
        )
    return fixed


# Backward-compat alias
_fix_apk_package_names = _fix_pkg_names


# Packages that are bundled in our sandbox images and must NEVER be installed
# via the distro package manager. Doing so on Debian/Ubuntu would pull Go 1.18
# (jammy) which cannot parse modern `go.mod` files (toolchain directive,
# 3-part `go 1.X.Y` version), and on Alpine would shadow the curated /usr/local
# tarball. The pattern is matched as a whole token so we don't strip
# `libgolang-foo` or similar.
_BUNDLED_TOOLCHAIN_TOKENS = (
    re.compile(
        r"^golang(-[a-z0-9.\-]+)?$"
    ),  # golang, golang-go, golang-1.21, golang-1.18-go, ...
    re.compile(r"^go-1\.[0-9]+$"),  # go-1.21, go-1.22, ...
    re.compile(r"^gccgo(-[a-z0-9.\-]+)?$"),  # gccgo, gccgo-12, ...
)


def _strip_bundled_toolchain_packages(command: str) -> str:
    """Strip bundled Go toolchain packages from an install command.

    Removes ``golang*`` / ``gccgo*`` / ``go-1.*`` tokens from any
    package-manager ``install`` command. The sandbox images bake a
    current ``go`` toolchain at ``/usr/local/go``; installing the distro
    package replaces ``/usr/bin/go`` with an outdated copy and breaks
    every modern Go build (observed root cause for ~25 % of Debian
    batch failures).

    Only the install verb is touched (``apk add``, ``apt-get install``,
    ``apt install``). If stripping leaves the install with no packages,
    the whole install clause is replaced with a harmless ``true`` so
    command chaining (``apt-get install … && go build …``) keeps
    working.

    Tolerates the option being placed either before OR after
    ``install``:
        - apt-get install -y golang
        - apt-get -y install golang
        - apt install --no-install-recommends golang
        - DEBIAN_FRONTEND=noninteractive apt-get install golang

    Args:
        command: The shell command to sanitize.

    Returns:
        The command with bundled-toolchain packages removed.
    """
    # Match the full install clause up to the next shell separator.
    # Options/flags can appear in any position between the manager and the
    # package list, so we accept them anywhere using a permissive token class.
    install_re = re.compile(
        r"(apt-get|apt|apk)\s+"  # 1: pkg manager
        # 2: pre-verb flags.
        r"((?:-{1,2}[A-Za-z0-9][A-Za-z0-9-]*(?:=\S+)?\s+)*)"
        r"(install|add)\s+"  # 3: verb
        # 4: post-verb flags.
        r"((?:-{1,2}[A-Za-z0-9][A-Za-z0-9-]*(?:=\S+)?\s+)*)"
        r"([^&|;]+?)"  # 5: package list (greedy w/in clause)
        r"(?=\s*(?:&&|\|\||;|$))",
        re.IGNORECASE,
    )

    def _rewrite(match: "re.Match[str]") -> str:
        pm, pre_flags, verb, post_flags, pkgs = (
            match.group(1),
            match.group(2) or "",
            match.group(3),
            match.group(4) or "",
            match.group(5),
        )
        tokens = pkgs.split()
        kept, dropped = [], []
        for tok in tokens:
            if any(rx.match(tok) for rx in _BUNDLED_TOOLCHAIN_TOKENS):
                dropped.append(tok)
            else:
                kept.append(tok)
        if not dropped:
            return match.group(0)
        if not kept:
            logger.warning(
                f"Stripped bundled-toolchain install (no other "
                f"packages requested): {dropped}. Skipping the install "
                f"clause; sandbox already provides a modern Go "
                f"toolchain."
            )
            return "true"
        logger.warning(
            f"Stripped bundled-toolchain package(s) from {pm} "
            f"{verb}: {dropped}. Sandbox already provides a modern Go "
            f"toolchain at /usr/local/go."
        )
        # Preserve original flag placement
        rebuilt = f"{pm} "
        if pre_flags.strip():
            rebuilt += pre_flags
        rebuilt += verb
        if post_flags.strip():
            rebuilt += " " + post_flags.strip()
        rebuilt += " " + " ".join(kept)
        return rebuilt

    return install_re.sub(_rewrite, command)


# ============================================================================
# FILE OPERATIONS
# ============================================================================


def read_file(
    filepath: str, max_lines: int = 1000, use_docker: bool = True
) -> str:
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
        result = execute_command(
            f"head -n {max_lines} {filepath}", use_docker=True
        )
        if result.success:
            return result.stdout
        else:
            logger.error(
                f"Failed to read file {filepath} from container: "
                f"{result.stderr}"
            )
            return f"Error reading file: {result.stderr}"
    else:
        # Read from host
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                lines = []
                for i, line in enumerate(f):
                    if i >= max_lines:
                        lines.append(
                            f"\n... (truncated after {max_lines} lines)"
                        )
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


def _convert_codex_envelope_to_unified_diff(
    patch_content: str,
) -> Optional[str]:
    """Convert a Codex/Aider envelope patch to a unified diff.

    Converts a Codex/Aider-style "*** Begin Patch" envelope to a
    standard unified diff that ``patch -p1`` understands.

    The fixer LLM frequently emits envelope patches like::

        *** Begin Patch
        *** Update File: go.mod
        @@
         module github.com/foo/bar
        -go 1.24.0
        +go 1.21
        -toolchain go1.24.0
        *** End Patch

    These are rejected by the unified-diff validator and the fix never
    lands, causing the agent to loop until escalation. Convert them on
    the fly.

    Args:
        patch_content: The raw patch text, possibly enveloped.

    Returns:
        The converted unified diff, or ``None`` if the input is not a
        recognisable envelope.
    """
    text = patch_content.strip()
    if (
        "*** Begin Patch" not in text
        and "*** Update File:" not in text
        and "*** Add File:" not in text
    ):
        return None

    out_chunks: list[str] = []
    current_path: Optional[str] = None
    current_op: Optional[str] = None  # "update" | "add" | "delete"
    body: list[str] = []

    def flush():
        nonlocal body
        if not current_path or current_op is None:
            body = []
            return
        if current_op == "add":
            # Synthesize a unified diff that adds a new file from /dev/null.
            added_lines = [
                ln[1:] if ln.startswith("+") else ln
                for ln in body
                if not ln.startswith("@@")
            ]
            out_chunks.append(
                f"--- /dev/null\n+++ b/{current_path}\n"
                f"@@ -0,0 +1,{len(added_lines)} @@\n"
                + "".join(f"+{ln}\n" for ln in added_lines)
            )
        elif current_op == "delete":
            out_chunks.append(f"--- a/{current_path}\n+++ /dev/null\n")
        else:
            # update: preserve hunks verbatim, just add file headers.
            # Compute hunk metadata best-effort.
            chunk = f"--- a/{current_path}\n+++ b/{current_path}\n"
            # Find @@ headers; if absent (LLM commonly omits them),
            # wrap in a single hunk that covers what we have.
            has_at = any(ln.startswith("@@") for ln in body)
            if has_at:
                chunk += "".join(ln + "\n" for ln in body)
            else:
                minus = sum(1 for ln in body if ln.startswith("-"))
                plus = sum(1 for ln in body if ln.startswith("+"))
                ctx = sum(
                    1
                    for ln in body
                    if ln.startswith(" ")
                    or (not ln.startswith(("+", "-", "@")))
                )
                old_len = minus + ctx
                new_len = plus + ctx
                chunk += f"@@ -1,{max(old_len,1)} +1,{max(new_len,1)} @@\n"
                for ln in body:
                    if ln.startswith(("+", "-")):
                        chunk += ln + "\n"
                    else:
                        chunk += " " + ln + "\n"
            out_chunks.append(chunk)
        body = []

    for raw in text.splitlines():
        if raw.startswith("*** Begin Patch") or raw.startswith(
            "*** End Patch"
        ):
            flush()
            current_path = None
            current_op = None
            continue
        if raw.startswith("*** Update File:"):
            flush()
            current_path = raw.split(":", 1)[1].strip()
            current_op = "update"
            continue
        if raw.startswith("*** Add File:"):
            flush()
            current_path = raw.split(":", 1)[1].strip()
            current_op = "add"
            continue
        if raw.startswith("*** Delete File:"):
            flush()
            current_path = raw.split(":", 1)[1].strip()
            current_op = "delete"
            continue
        if current_path is not None:
            body.append(raw)
    flush()

    if not out_chunks:
        return None
    return "".join(out_chunks)


def apply_patch(
    patch_content: str,
    filepath: Optional[str] = None,
    cwd: Optional[str] = None,
    use_docker: bool = True,
) -> bool:
    """
    Apply a patch to one or more files.

    Args:
        patch_content: Unified diff or content to append.
        filepath: Specific file to patch. If None, assumes the unified
            diff carries its own paths.
        cwd: Working directory.
        use_docker: Whether to run in Docker.

    Returns:
        The success status.
    """
    if not patch_content:
        return False

    # Transparently convert Codex/Aider "*** Begin Patch" envelope patches
    # (frequently emitted by the fixer LLM) into a standard unified diff
    # before the format check. Without this, every envelope patch is rejected
    # with "not a valid unified diff" and the fix never lands.
    converted = _convert_codex_envelope_to_unified_diff(patch_content)
    if converted is not None:
        logger.info("Converted Codex envelope patch to unified diff")
        patch_content = converted
        # An envelope patch always specifies its own paths, so applying with
        # patch -p1 from cwd is the correct strategy — drop any caller-supplied
        # filepath so we don't double-route the diff.
        if filepath:
            filepath = None

    if use_docker:
        # Write patch to container's /tmp
        import uuid

        patch_filename = f"/tmp/agent_{uuid.uuid4().hex[:8]}.patch"
        if not write_file(patch_filename, patch_content, use_docker=True):
            return False

        try:
            if filepath:
                # Apply to a specific file - always try as a diff,
                # never raw append.
                if "--- " in patch_content and "+++ " in patch_content:
                    cmd = f"patch {filepath} < {patch_filename}"
                elif "@@ " in patch_content:
                    # Has diff hunks but missing headers - try patch -p0
                    cmd = f"patch -p0 {filepath} < {patch_filename}"
                else:
                    # Not a valid diff format - reject to avoid
                    # corrupting the file.
                    logger.warning(
                        "Patch content is not a valid unified diff - "
                        "rejecting to avoid file corruption"
                    )
                    return False
            else:
                # Standard unified diff - apply with -p1
                # Try dry-run first
                dry_run = execute_command(
                    f"patch -p1 --dry-run < {patch_filename}",
                    cwd=cwd,
                    use_docker=True,
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
                    cmd = f"patch {filepath} < {patch_file}"
                elif "@@ " in patch_content:
                    cmd = f"patch -p0 {filepath} < {patch_file}"
                else:
                    logger.warning(
                        "Patch content is not a valid unified diff - "
                        "rejecting"
                    )
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

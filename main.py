#!/usr/bin/env python3
"""
Main entry point for Atesor AI.
Handles CLI, Docker setup, and starts the multi-agent porting workflow.
"""

import os
import sys
import time
import argparse
import logging
from typing import Optional
from datetime import datetime

import docker
from termcolor import colored
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.models import check_api_keys, print_model_info, ModelProvider
from src.state import AgentState, AgentRole, BuildStatus
from src.persistence import SessionStore
from src.runtime import get_runtime_settings

from src.config import CONTAINER_NAME, IMAGE_NAME, OUTPUT_DIR, LOGS_DIR
from src.tools import execute_command, DockerConfig as DC

# Configure logging
def configure_logging(verbose: bool):
    # Capture everything at root level
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Create a custom formatter for better readability
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"
    )

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 1. File Handler (Always Capture Everything)
    log_file = os.path.join(LOGS_DIR, "agent.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)

    # 2. Console Handler (Respect Verbose Flag)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    if verbose:
        console_handler.setLevel(logging.DEBUG)
    else:
        # User requested no in-depth logs in terminal unless verbose is set
        console_handler.setLevel(logging.ERROR)
        
    root_logger.addHandler(console_handler)
    
    # Reduce noise from sensitive/chatty packages
    logging.getLogger("docker").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.INFO)
    logging.getLogger("langsmith").setLevel(logging.ERROR)
    logging.getLogger("google.ai").setLevel(logging.WARNING)
    
    # Ensure stdout is flushed
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)


logger = logging.getLogger(__name__)


def check_keys() -> bool:
    """Verify required API keys are set."""
    is_valid, msg, provider = check_api_keys()
    if not is_valid:
        print(colored(f"ERROR: {msg}", "red"))
        return False

    print(colored(f"{msg}", "green"))
    print_model_info()
    return True


def setup_docker_environment() -> bool:
    """
    Set up the Docker environment for RISC-V development.

    This creates the sandbox container where all builds will happen.
    """
    try:
        client = docker.from_env()
    except docker.errors.DockerException as e:
        print(colored(f"ERROR: Docker is not running or not accessible", "red"))
        print(colored(f"Details: {e}", "yellow"))
        return False

    # Step 1: Check/Build Docker Image
    print(colored("\nSetting up RISC-V development environment...", "cyan"))

    force_rebuild = os.environ.get("REBUILD_IMAGE") == "true"

    try:
        if force_rebuild:
            print(colored(f"Forcing rebuild of image '{IMAGE_NAME}'...", "yellow"))
            raise docker.errors.ImageNotFound("Triggering rebuild")

        client.images.get(IMAGE_NAME)
        print(colored(f"Image '{IMAGE_NAME}' found", "green"))
    except docker.errors.ImageNotFound:
        print(colored(f"   Building image '{IMAGE_NAME}'...", "yellow"))
        try:
            # Build the image
            image, logs = client.images.build(
                path=".",
                tag=IMAGE_NAME,
                rm=True,
                forcerm=True,
                pull=True,  # Ensure base image is up to date
                platform="linux/riscv64",
            )
            print(colored(f"Image built successfully", "green"))
        except docker.errors.BuildError as e:
            print(colored(f"ERROR: Failed to build Docker image", "red"))
            print(colored(f"{e}", "yellow"))
            return False

    # Step 2: Create/Start Container
    container = None
    for attempt in range(2):
        try:
            try:
                container = client.containers.get(CONTAINER_NAME)
                if container.status != "running":
                    print(
                        colored(
                            f"   Starting existing container '{CONTAINER_NAME}'...",
                            "yellow",
                        )
                    )
                    container.start()
                    time.sleep(2)
                print(colored(f"Container '{CONTAINER_NAME}' is running", "green"))
            except docker.errors.NotFound:
                print(colored(f"   Creating container '{CONTAINER_NAME}'...", "yellow"))

                os.makedirs(OUTPUT_DIR, exist_ok=True)

                container = client.containers.run(
                    IMAGE_NAME,
                    name=CONTAINER_NAME,
                    detach=True,
                    tty=True,
                    platform="linux/riscv64",
                    network_mode="bridge",
                    dns=["8.8.8.8", "8.8.4.4"],
                    volumes={
                        os.path.abspath("./workspace"): {
                            "bind": "/workspace",
                            "mode": "rw",
                        }
                    },
                    mem_limit="8g",
                    cpu_quota=-1,
                    privileged=True,
                )
                print(colored(f"Container created and started", "green"))
                time.sleep(5)

            container.reload()
            time.sleep(1)
            if container.status != "running":
                print(
                    colored(
                        f"   Container status: {container.status}, recreating...",
                        "yellow",
                    )
                )
                try:
                    container.stop()
                    container.remove()
                except:
                    pass
                continue

            break
        except docker.errors.NotFound:
            continue
        except docker.errors.APIError as e:
            print(colored(f"ERROR: Failed to create or start container", "red"))
            print(colored(f"{e}", "yellow"))
            return False

    if container is None or container.status != "running":
        print(colored(f"ERROR: Failed to start container after 2 attempts", "red"))
        return False

    print(colored(f"   Container ID: {container.short_id}", "cyan"))

    logger.info(f"Container '{CONTAINER_NAME}' is running with ID: {container.short_id}")
    logger.info(f"Workspace mounted at /workspace in container")

    # Step 3: Verify container is working and architecture is correct
    try:
        # Check architecture
        arch_result = container.exec_run("uname -m")
        if arch_result.exit_code == 0:
            arch = arch_result.output.decode("utf-8").strip()
            if "riscv64" in arch:
                print(
                    colored(
                        f"Container architecture verified: {arch} (Correctly Emulated)",
                        "green",
                    )
                )
            else:
                print(
                    colored(
                        f"WARNING: Container architecture is {arch} (Expected riscv64!)",
                        "yellow",
                    )
                )
                print(
                    colored(
                        "Try running: docker run --privileged --rm tonistiigi/binfmt --install all",
                        "yellow",
                    )
                )
        else:
            print(colored("ERROR: Container is not responding correctly", "red"))
            return False

        # Simple health check
        exec_result = container.exec_run("echo 'Container ready!'")
        if exec_result.exit_code == 0:
            print(colored("Container is responsive", "green"))
        else:
            print(colored("ERROR: Container is not responding correctly", "red"))
            return False
    except Exception as e:
        print(colored(f"ERROR: Container health check failed: {e}", "red"))

        # Try to recreate container on health check failure
        print(colored("   Attempting to recreate container...", "yellow"))
        try:
            container.stop()
            container.remove()
        except:
            pass

        # Retry setup
        container = None
        for attempt in range(2):
            try:
                print(
                    colored(
                        f"   Creating container '{CONTAINER_NAME}' (attempt {attempt + 1})...",
                        "yellow",
                    )
                )
                os.makedirs(OUTPUT_DIR, exist_ok=True)

                container = client.containers.run(
                    IMAGE_NAME,
                    name=CONTAINER_NAME,
                    detach=True,
                    tty=True,
                    platform="linux/riscv64",
                    network_mode="bridge",
                    dns=["8.8.8.8", "8.8.4.4"],
                    volumes={
                        os.path.abspath("./workspace"): {
                            "bind": "/workspace",
                            "mode": "rw",
                        }
                    },
                    mem_limit="8g",
                    cpu_quota=-1,
                    privileged=True,
                )
                print(colored(f"Container created and started", "green"))
                time.sleep(5)

                container.reload()
                time.sleep(1)
                if container.status != "running":
                    continue

                exec_result = container.exec_run("echo 'Container ready!'")
                if exec_result.exit_code == 0:
                    print(colored("Container is responsive after recreation", "green"))
                    return True
            except docker.errors.APIError as recreate_error:
                print(
                    colored(f"   Recreation attempt failed: {recreate_error}", "yellow")
                )
                if container:
                    try:
                        container.stop()
                        container.remove()
                    except:
                        pass
            except Exception as recreate_error:
                print(
                    colored(f"   Recreation attempt failed: {recreate_error}", "yellow")
                )
                if container:
                    try:
                        container.stop()
                        container.remove()
                    except:
                        pass

        return False

    return True


def save_porting_outputs(state: AgentState, output_dir: str) -> None:
    """
    Save all porting outputs to files.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    repo_name = state.repo_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 0. Save complete state (JSON)
    state_path = os.path.join(output_dir, f"{repo_name}_state_{timestamp}.json")
    try:
        state.save_to_json(state_path)
        print(colored(f"Full state saved: {state_path}", "green"))
    except Exception as e:
        logger.error(f"Failed to save state JSON: {e}")

    # 1. Save porting recipe
    if state.porting_recipe:
        recipe_path = os.path.join(output_dir, f"{repo_name}_recipe.md")
        with open(recipe_path, "w") as f:
            f.write(state.porting_recipe)
        print(colored(f"Porting recipe saved: {recipe_path}", "green"))

    # 2. Save detailed build report
    report = generate_detailed_report(state.to_dict())
    report_path = os.path.join(output_dir, f"{repo_name}_report_{timestamp}.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(colored(f"Detailed report saved: {report_path}", "green"))

    # 3. Save patches
    if state.patches_generated:
        patches_dir = os.path.join(output_dir, f"{repo_name}_patches_{timestamp}")
        os.makedirs(patches_dir, exist_ok=True)
        for i, patch in enumerate(state.patches_generated):
            patch_path = os.path.join(patches_dir, f"patch_{i + 1}.patch")
            with open(patch_path, "w") as f:
                f.write(patch)
        print(
            colored(
                f"{len(state.patches_generated)} patch(es) saved: {patches_dir}/",
                "green",
            )
        )


def generate_detailed_report(state: dict) -> str:
    """Generate a comprehensive detailed report."""
    report = f"""# RISC-V Porting Report: {state.get("repo_name", "Unknown")}

## Executive Summary

**Status**: {state.get("build_status", "UNKNOWN")}  
**Repository**: {state.get("repo_url", "N/A")}  
**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## Metrics

| Metric | Value |
|--------|-------|
| Build Status | {state.get("build_status", "N/A")} |
| Total Duration | {state.get("execution_duration", 0):.1f}s |
| Attempts Made | {state.get("attempt_count", 0)} |
| API Calls | {state.get("api_calls_made", 0)} |
| Scripted Operations | {state.get("scripted_ops_count", 0)} |
| Estimated Cost | ${state.get("api_cost_usd", 0):.4f} |

---

## Build System Analysis

"""

    build_plan = state.get("build_plan")
    if build_plan:
        report += f"**Build System**: {build_plan.get('build_system', 'Unknown')}\n\n"

        phases = build_plan.get("phases", [])
        if phases:
            report += "### Build Phases\n\n"
            for i, phase in enumerate(phases, 1):
                phase_name = phase.get("name", f"Phase {i}")
                report += f"#### {i}. {phase_name}\n\n"

                commands = phase.get("commands", [])
                if commands:
                    report += "```bash\n"
                    for cmd in commands:
                        report += f"{cmd}\n"
                    report += "```\n\n"
    else:
        report += "*No build plan available*\n\n"

    # Dependencies
    dependencies = state.get("dependencies")
    if dependencies:
        report += "### Dependencies\n\n"

        build_tools = dependencies.get("build_tools", [])
        if build_tools:
            report += f"**Build Tools**: {', '.join(build_tools)}\n\n"

        system_packages = dependencies.get("system_packages", [])
        if system_packages:
            report += f"**System Packages**: {', '.join(system_packages)}\n\n"

    # Architecture-specific code
    arch_code = state.get("arch_specific_code", [])
    if arch_code:
        report += f"---\n\n## Architecture-Specific Code\n\n"
        report += f"Found {len(arch_code)} instances of architecture-specific code:\n\n"

        for i, code in enumerate(arch_code[:5], 1):  # Show first 5
            filepath = code.get("file", "Unknown")
            line_num = code.get("line", "N/A")
            snippet = code.get("code_snippet", "")

            report += f"### {i}. {filepath}:{line_num}\n\n"
            report += f"```c\n{snippet[:200]}\n```\n\n"

        if len(arch_code) > 5:
            report += f"*...and {len(arch_code) - 5} more instances*\n\n"

    # Patches
    patches = state.get("patches_generated", [])
    if patches:
        report += f"---\n\n## Patches Applied\n\n"
        report += f"{len(patches)} patch(es) were needed for RISC-V compatibility:\n\n"

        for i, patch in enumerate(patches, 1):
            report += f"### Patch {i}\n\n"
            report += f"```diff\n{patch[:500]}\n```\n\n"
            if len(patch) > 500:
                report += f"*Truncated (full patch available in separate file)*\n\n"

    # Error history
    error_log = state.get("error_history", [])
    if error_log:
        report += f"---\n\n## Error Resolution History\n\n"

        for i, error in enumerate(error_log, 1):
            if isinstance(error, dict):
                category = error.get("category", "Unknown")
                message = error.get("message", "N/A")
            else:
                category = getattr(error, "category", "Unknown")
                message = getattr(error, "message", "N/A")

            report += f"### Error {i}: {category}\n\n"
            report += f"```\n{message[:300]}\n```\n\n"

    feedback_notes = state.get("feedback_notes", [])
    if feedback_notes:
        report += f"---\n\n## Feedback Loop Findings\n\n"
        for note in feedback_notes[-10:]:
            report += f"- {note}\n"
        report += "\n"

    # Recommendations
    report += "---\n\n## Recommendations\n\n"

    status = state.get("build_status")
    if status == "SUCCESS":
        report += """
- Build completed successfully for RISC-V
- Test the binary on actual RISC-V hardware or QEMU
- Consider submitting patches upstream if modifications were needed
- Add RISC-V to your CI/CD pipeline
"""
    elif status == "ESCALATED":
        escalation_reason = state.get("escalation_reason", "Unknown")
        report += f"""
- Manual intervention required
- Reason: {escalation_reason}
- Review the error logs above
- Consider consulting RISC-V porting documentation
"""
    else:
        report += """
- Build did not complete successfully
- Review error logs above
- Check dependencies and build system compatibility
- Consider filing an issue with the upstream project
"""

    report += "\n---\n\n"
    report += f"*Report generated by RISC-V Porting Foundry v1.0*\n"

    return report


def format_build_plan(build_plan) -> str:
    """Format build plan as markdown."""
    content = f"""# Build Plan

**Build System**: {getattr(build_plan, "build_system", "Unknown")}

## Phases

"""

    phases = getattr(build_plan, "phases", [])
    for i, phase in enumerate(phases, 1):
        phase_name = getattr(phase, "name", f"Phase {i}")
        content += f"### {i}. {phase_name}\n\n"

        commands = getattr(phase, "commands", [])
        if commands:
            content += "```bash\n"
            for cmd in commands:
                content += f"{cmd}\n"
            content += "```\n\n"

    return content


def run_agent(repo_url: str, max_attempts: int = 5, verbose: bool = False) -> int:
    """
    Run the RISC-V porting agent on a repository.

    Args:
        repo_url: The GitHub/GitLab repository URL
        max_attempts: Maximum fix attempts before escalation
        verbose: Enable verbose output

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    from src.graph import app
    from src.state import create_initial_state, BuildStatus
    from langchain_core.messages import HumanMessage

    logger.info("Starting RISC-V Porting Agent")
    logger.info(f"Repository: {repo_url}")
    logger.info(f"Max attempts: {max_attempts}")
    logger.info("-" * 60)

    print(colored("\nStarting RISC-V Porting Agent", "cyan", attrs=["bold"]))
    print(colored(f"   Repository: {repo_url}", "white"))
    print(colored(f"   Max attempts: {max_attempts}", "white"))
    print(colored("-" * 60, "cyan"))

    settings = get_runtime_settings()
    session_store = None
    session_id = None
    if settings.persist_sessions:
        try:
            session_store = SessionStore(settings.session_db_path)
            session_id = session_store.create_session(repo_url, max_attempts)
            logger.info(f"Session ID: {session_id}")
            print(colored(f"   Session ID: {session_id}", "white"))
        except Exception as persistence_error:
            logger.warning(f"Session persistence unavailable: {persistence_error}")
            session_store = None
            session_id = None

    initial_state = create_initial_state(repo_url, max_attempts=max_attempts)
    if session_id:
        initial_state.context_cache["session_id"] = session_id
    initial_state.messages = [
        HumanMessage(
            content=f"Port this repository to RISC-V: {repo_url}. The repository is cloned at {initial_state.repo_path}"
        )
    ]

    final_state = initial_state
    step_count = 0
    last_persisted_event_index = 0

    def persist_step(node_name: str) -> None:
        nonlocal last_persisted_event_index
        if not session_store or not session_id:
            return
        try:
            session_store.save_snapshot(session_id, step_count, node_name, final_state)
            new_events = final_state.audit_trail[last_persisted_event_index:]
            if new_events:
                session_store.save_events(session_id, step_count, new_events)
                last_persisted_event_index = len(final_state.audit_trail)
        except Exception as persistence_error:
            logger.warning(f"Failed to persist workflow step: {persistence_error}")

    def finalize(exit_code: int) -> int:
        if session_store and session_id:
            try:
                session_store.finish_session(session_id, final_state, exit_code)
            except Exception as persistence_error:
                logger.warning(f"Failed to finalize session record: {persistence_error}")
        return exit_code

    # Run the workflow
    try:
        for output in app.stream(initial_state):
            for node_name, state_update in output.items():
                step_count += 1

                # In LangGraph, state_update can be a dict of updates
                if isinstance(state_update, dict):
                    for k, v in state_update.items():
                        if hasattr(final_state, k):
                            setattr(final_state, k, v)
                elif isinstance(state_update, AgentState):
                    final_state = state_update

                persist_step(node_name)

                # Print progress
                status = getattr(
                    final_state.build_status, "value", str(final_state.build_status)
                )
                status_color = {
                    "PENDING": "white",
                    "PLANNING": "cyan",
                    "SCOUTING": "blue",
                    "BUILDING": "yellow",
                    "FIXING": "magenta",
                    "SUCCESS": "green",
                    "FAILED": "red",
                    "ESCALATED": "red",
                }.get(status, "white")

                logger.info(f"[Step {step_count}] {node_name} - Status: {status}")

                print(
                    colored(
                        f"\n[Step {step_count}] {node_name}", "cyan", attrs=["bold"]
                    ),
                    flush=True,
                )
                print(colored(f"   Status: {status}", status_color), flush=True)

                if (
                    verbose
                    and hasattr(state_update, "messages")
                    and state_update.messages
                ):
                    for msg in state_update.messages:
                        role = getattr(msg, "name", "Assistant")
                        content = getattr(msg, "content", str(msg))

                        logger.debug(f"[{role}] {content}")

                        role_color = "magenta" if role == "Supervisor" else "cyan"
                        if role == "Scout":
                            role_color = "blue"
                        if role == "Builder":
                            role_color = "yellow"
                        if role == "Fixer":
                            role_color = "green"

                        print(
                            colored(f"   [{role}]", role_color, attrs=["bold"]), end=" "
                        )
                        print(colored(content, "white"), flush=True)
                elif not verbose and node_name != "Supervisor":
                    compact_status = (
                        final_state.build_status.value
                        if hasattr(final_state, "build_status")
                        else "ACTIVE"
                    )
                    print(
                        colored(f"   Working... ({compact_status})", "white"),
                        end="\r",
                        flush=True,
                    )

        logger.info("=" * 60)
        print(colored("\n" + "=" * 60, "cyan"))

        save_porting_outputs(final_state, OUTPUT_DIR)

        final_status = final_state.build_status
        if final_status == BuildStatus.SUCCESS:
            logger.info("PORTING SUCCESSFUL!")
            logger.info(
                f"Recipe generated at: {OUTPUT_DIR}/{final_state.repo_name}_recipe.md"
            )

            print(colored("\nPORTING SUCCESSFUL!", "green", attrs=["bold"]))
            print(
                colored(
                    f"   Recipe generated at: {OUTPUT_DIR}/{final_state.repo_name}_recipe.md",
                    "white",
                )
            )
            return finalize(0)

        logger.info("PORTING STOPPED / FAILED")
        logger.info(f"Final Status: {final_status.value}")

        print(colored("\nPORTING STOPPED / FAILED", "red", attrs=["bold"]))
        print(colored(f"   Final Status: {final_status.value}", "white"))

        if final_state.last_error:
            logger.info("Last Error Capture:")
            logger.info(
                f"Category: {final_state.last_error_category.value if final_state.last_error_category else 'Unknown'}"
            )
            logger.info(
                f"Severity: {final_state.last_error_severity.value if final_state.last_error_severity else 'unknown'}"
            )
            logger.info(f"{final_state.last_error[:500]}")

            print(colored("\nLast Error Capture:", "red", attrs=["bold"]))
            print(
                colored(
                    f"   Category: {final_state.last_error_category.value if final_state.last_error_category else 'Unknown'}",
                    "yellow",
                )
            )
            print(
                colored(
                    f"   Severity: {final_state.last_error_severity.value if final_state.last_error_severity else 'unknown'}",
                    "yellow",
                )
            )
            print(colored(f"   {final_state.last_error[:500]}", "white"))

        if final_state.audit_trail:
            logger.info("Agent Decision Audit:")
            print(colored("\nAgent Decision Audit:", "cyan", attrs=["bold"]))
            for event in final_state.get_last_audit_events(10):
                ts = event["timestamp"].split("T")[-1][:8]
                agent = event.get("agent", "system")
                etype = event.get("event")
                data = event.get("data", {})

                msg = ""
                if etype == "decision":
                    msg = f"Decided to {data.get('action')} - {data.get('reason')}"
                elif etype == "scripted_op":
                    msg = f"Ran scripted op: {data.get('operation')}"
                elif etype == "feedback":
                    msg = f"Feedback: {data.get('issues')}"
                elif etype == "error":
                    msg = f"ERROR: {data.get('message')[:100]}..."

                logger.info(f"[{ts}] {agent:12} | {etype:10} | {msg}")
                print(colored(f"   [{ts}] {agent:12} | {etype:10} | {msg}", "white"))

        return finalize(1)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        print(colored("\n\nInterrupted by user", "yellow"))
        return finalize(130)
    except Exception as e:
        logger.error(f"Unexpected error during agent execution: {e}")
        print(colored(f"\nERROR: {e}", "red"))
        logger.exception("Unexpected error during agent execution")
        return finalize(1)

    finally:
        # Ensure logs are flushed
        for handler in logger.handlers:
            handler.flush()


def cleanup_workspace(dry_run: bool = False) -> None:
    """
    Clean up workspace directory to manage size.

    Args:
        dry_run: If True, only show what would be deleted without actually deleting
    """
    import shutil
    from pathlib import Path

    workspace_path = Path("./workspace")
    if not workspace_path.exists():
        print(colored("Workspace directory does not exist", "yellow"))
        return

    # Calculate current size
    total_size = 0
    file_count = 0

    for item in workspace_path.rglob("*"):
        if item.is_file():
            total_size += item.stat().st_size
            file_count += 1

    size_mb = total_size / (1024 * 1024)
    size_gb = size_mb / 1024

    print(colored(f"\nWorkspace Statistics:", "cyan", attrs=["bold"]))
    print(colored(f"   Location: {workspace_path.absolute()}", "white"))
    print(colored(f"   Total Size: {size_gb:.2f} GB ({size_mb:.2f} MB)", "white"))
    print(colored(f"   Total Files: {file_count}", "white"))

    # List subdirectories with sizes
    print(colored(f"\nDirectory Breakdown:", "cyan"))
    subdirs = {}
    for subdir in ["repos", "output", "logs", ".cache", "patches"]:
        subdir_path = workspace_path / subdir
        if subdir_path.exists():
            subdir_size = sum(
                f.stat().st_size for f in subdir_path.rglob("*") if f.is_file()
            )
            subdirs[subdir] = subdir_size
            print(
                colored(f"   {subdir}: {subdir_size / (1024 * 1024):.2f} MB", "white")
            )

    # Offer cleanup options
    print(colored(f"\nCleanup Options:", "cyan"))
    print(colored("   1. Clean repos/ (cloned repositories)", "white"))
    print(colored("   2. Clean output/ (generated reports)", "white"))
    print(colored("   3. Clean logs/ (execution logs)", "white"))
    print(colored("   4. Clean .cache/ (cached data)", "white"))
    print(colored("   5. Clean ALL (complete workspace reset)", "white"))

    if dry_run:
        print(colored("\n[DRY RUN] No files will be deleted", "yellow"))
        return

    choice = input(colored("\nSelect option (1-5, or 'q' to quit): ", "cyan"))

    dirs_to_clean = []
    if choice == "1":
        dirs_to_clean = ["repos"]
    elif choice == "2":
        dirs_to_clean = ["output"]
    elif choice == "3":
        dirs_to_clean = ["logs"]
    elif choice == "4":
        dirs_to_clean = [".cache"]
    elif choice == "5":
        dirs_to_clean = ["repos", "output", "logs", ".cache", "patches"]
    elif choice.lower() == "q":
        print(colored("Cleanup cancelled", "yellow"))
        return
    else:
        print(colored("Invalid option", "red"))
        return

    # Perform cleanup
    for dirname in dirs_to_clean:
        dir_path = workspace_path / dirname
        if dir_path.exists():
            try:
                shutil.rmtree(dir_path)
                dir_path.mkdir(exist_ok=True)
                print(colored(f"   Cleaned {dirname}/", "green"))
            except Exception as e:
                print(colored(f"   Error cleaning {dirname}/: {e}", "red"))

    # Recalculate size
    new_total_size = sum(
        f.stat().st_size for f in workspace_path.rglob("*") if f.is_file()
    )
    new_size_mb = new_total_size / (1024 * 1024)
    freed_mb = size_mb - new_size_mb

    print(colored(f"\nCleanup Complete!", "green", attrs=["bold"]))
    print(colored(f"   Freed: {freed_mb:.2f} MB", "green"))
    print(colored(f"   New Size: {new_size_mb:.2f} MB", "white"))


def cleanup_container(remove_image: bool = False):
    """Stop and remove the sandbox container and optionally the image."""
    try:
        client = docker.from_env()

        # Remove container
        try:
            container = client.containers.get(CONTAINER_NAME)
            if container.status == "running":
                print(colored(f"Stopping container '{CONTAINER_NAME}'...", "yellow"))
                container.stop(timeout=10)
            print(colored(f"Removing container '{CONTAINER_NAME}'...", "yellow"))
            container.remove()
            print(colored("Container cleaned up", "green"))
        except docker.errors.NotFound:
            print(colored(f"Container '{CONTAINER_NAME}' not found", "yellow"))

        # Remove image if requested
        if remove_image:
            try:
                print(colored(f"Removing image '{IMAGE_NAME}'...", "yellow"))
                client.images.remove(IMAGE_NAME)
                print(colored(f"Image '{IMAGE_NAME}' removed", "green"))
            except docker.errors.ImageNotFound:
                print(colored(f"Image '{IMAGE_NAME}' not found", "yellow"))
            except Exception as e:
                print(colored(f"Error removing image: {e}", "red"))

    except Exception as e:
        print(colored(f"Error during cleanup: {e}", "red"))


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="RISC-V Porting Foundry - Automated software porting agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --repo https://github.com/madler/zlib
  python main.py --repo https://github.com/sqlite/sqlite --max-attempts 10 --verbose
  python main.py --cleanup
        """,
    )

    parser.add_argument(
        "--repo", "-r", type=str, help="GitHub/GitLab repository URL to port"
    )

    parser.add_argument(
        "--max-attempts",
        "-m",
        type=int,
        default=5,
        help="Maximum fix attempts before escalation (default: 5)",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    parser.add_argument(
        "--cleanup", action="store_true", help="Clean up Docker container and exit"
    )

    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Only set up the Docker environment, don't run agent",
    )

    parser.add_argument(
        "--clean-workspace",
        action="store_true",
        help="Clean up workspace directory to manage size",
    )

    parser.add_argument(
        "--clean-image",
        action="store_true",
        help="Remove the Docker image as well as the container",
    )

    parser.add_argument(
        "--rebuild", action="store_true", help="Force rebuild of the Docker image"
    )

    args = parser.parse_args()

    # Configure logging early
    configure_logging(args.verbose)

    # Handle cleanup
    if args.cleanup:
        cleanup_container(remove_image=args.clean_image)
        return 0

    if args.clean_image and not args.cleanup:
        # If only --clean-image is provided, still perform cleanup
        cleanup_container(remove_image=True)
        return 0

    # Handle workspace cleanup
    if args.clean_workspace:
        cleanup_workspace()
        return 0

    # Require repo URL before starting long-running setup or API checks
    if not args.repo and not args.setup_only:
        print(colored("ERROR: --repo argument is required", "red"))
        parser.print_help()
        return 1

    # Check API keys
    if not check_keys():
        return 1

    # Set up Docker environment
    if args.rebuild:
        os.environ["REBUILD_IMAGE"] = "true"

    if not setup_docker_environment():
        return 1

    # Setup only mode
    if args.setup_only:
        print(colored("\nSetup complete!", "green"))
        return 0

    # Run the agent
    exit_code = run_agent(
        repo_url=args.repo, max_attempts=args.max_attempts, verbose=args.verbose
    )

    return exit_code


if __name__ == "__main__":
    sys.exit(main())

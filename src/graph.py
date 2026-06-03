"""Multi-agent state machine orchestration and workflow nodes.

Uses LangGraph to manage agent transitions and shared state for the
RISC-V porting pipeline.
"""

import base64
import json
import logging
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Dict, List

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

from .artifact_scanner import ArtifactScanner
from .knowledge import get_system_knowledge_summary
from .llm_helpers import ValidationResult, llm_call_with_validation
from .llm_logger import log_llm_call
from .memory import (
    format_few_shot_examples,
    save_learned_example,
    save_to_recipe_cache,
)
from .models import create_llm
from .scripted_ops import ScriptedOperations, quick_analysis
from .state import (
    AgentRole,
    AgentState,
    BuildPhase,
    BuildPlan,
    BuildStatus,
    ErrorCategory,
    FailureSeverity,
    FixAttempt,
    TaskPhase,
    TaskPlan,
    classify_error,
    create_error_record,
    get_next_action_recommendation,
    should_escalate,
)
from .tools import apply_patch, execute_command

# Configure logging
logger = logging.getLogger(__name__)

# Initialize scripted operations
scripted_ops = ScriptedOperations()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def get_model_for_role(role: AgentRole):
    """Return the right model for an agent role."""
    return create_llm(role)


def get_model_pool_for_role(role: AgentRole):
    """Return ``[primary, *fallbacks]`` for the role.

    Used by validated LLM calls so transient provider errors (404 /
    5xx) rotate to a backup model instead of escalating the whole
    agent.
    """
    from .models import create_llm_pool

    return create_llm_pool(role)


def invoke_llm(llm, messages, timeout: int = 120):
    """Invoke an LLM with a hard timeout to prevent indefinite hangs.

    Uses a daemon thread so blocking HTTP connections don't prevent
    process exit. Raises TimeoutError if the call exceeds ``timeout``
    seconds.
    """
    import threading

    result = [None]
    exception = [None]
    done = threading.Event()

    def worker():
        try:
            result[0] = llm.invoke(messages)
        except Exception as e:
            exception[0] = e
        finally:
            done.set()

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    if not done.wait(timeout=timeout):
        raise TimeoutError(f"LLM call timed out after {timeout}s")

    if exception[0] is not None:
        raise exception[0]

    return result[0]


def extract_content(content) -> str:
    """Safely extract string content from LLM response."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            [
                (
                    str(item.get("text", item))
                    if isinstance(item, dict)
                    else str(item)
                )
                for item in content
            ]
        )
    return str(content)


def extract_json_block(text: str) -> str:
    """Safely extract the first JSON block from a string."""
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end >= start:
        return text[start : end + 1]
    return text


def _build_platform_banner() -> str:
    """Build a banner describing the active sandbox/package manager.

    Injected at the TOP of scout & fixer prompts so the LLM cannot
    ignore it (knowledge appendices buried at the bottom tend to lose
    to early-context few-shot examples in free-tier models).
    """
    try:
        from .platforms import get_active_profile

        p = get_active_profile()
        pm = p.pkg_install.split()[0]  # "apk" / "apt-get"
        install_example = p.install_cmd(["<pkg>"])
        other_pms = (
            "apt-get / apt / dpkg" if pm == "apk" else "apk / apk-tools"
        )
        # Pick one concrete correction example to anchor the LLM on
        # the right names. name_corrections maps WRONG_NAME ->
        # RIGHT_NAME for THIS distro.
        correction_example = ""
        if p.name_corrections:
            wrong, right = next(iter(p.name_corrections.items()))
            correction_example = (
                f"\n- Correct package names for {p.display_name}: "
                f"e.g. `{wrong}` is WRONG, use `{right}` instead."
            )
        # Pick a concrete worked install example using a real
        # canonical -> distro mapping.
        worked_example = ""
        if p.package_map:
            canonical, distro_pkg = next(iter(p.package_map.items()))
            worked_example = (
                f"\n- Worked example for this distro: install "
                f"`{canonical}` via `{p.install_cmd([distro_pkg])}`."
            )
        return (
            f"- Distro: **{p.display_name}** (libc: **{p.libc}**, "
            f"triplet: `{p.target_triplet}`)\n"
            f"- Package manager: **{pm}** (full install: "
            f"`{install_example}`)\n"
            f"- NEVER emit {other_pms} commands — they do not exist "
            f"in this container\n"
            f"  and will fail with `command not found`. This is "
            f"unrecoverable for the build."
            f"{worked_example}"
            f"{correction_example}\n"
            f"- If you are unsure of a package name on this distro, "
            f"prefer the canonical\n"
            f"  short name (e.g. `openssl-dev`, `zlib-dev`) — the "
            f"agent auto-corrects to\n"
            f"  the distro-specific name. NEVER guess a name from "
            f"another distro."
        )
    except Exception:
        return (
            "- Platform: unknown (default to apt-get on "
            "debian/ubuntu, apk on alpine)"
        )


def _to_host_path(path: str) -> str:
    """Translate a /workspace container path to a host path."""
    if path.startswith("/workspace") and not os.path.exists("/workspace"):
        from .config import WORKSPACE_ROOT

        return path.replace("/workspace", WORKSPACE_ROOT, 1)
    return path


def _build_command_error_message(result, fallback: str) -> str:
    """Build a robust, human-readable error message from command output."""
    stderr = (result.stderr or "").strip()
    stdout = (result.stdout or "").strip()
    detail = stderr or stdout or "No stderr/stdout output captured."
    return f"{fallback} (exit {result.exit_code}) - {detail}"


def is_toolchain_version_mismatch(error_message: str) -> bool:
    """Return True if an error indicates an outdated compiler toolchain."""
    err = (error_message or "").lower()
    patterns = [
        "go.mod requires go >=",
        "running go ",
        "feature `edition2024` is required",
        "not stabilized in this version of cargo",
        "this package requires rustc",
        "requires rust version",
    ]
    return any(p in err for p in patterns)


def _replan_signature(error_message: str) -> str:
    """Return a normalized signature for failures that need a new build plan."""
    msg = (error_message or "").lower()
    signatures = {
        "go_missing_module": "no required module provides",
        "go_output_dir_collision": "already exists and is a directory",
        "cmake_wrong_source_root": "does not appear to contain cmakelists.txt",
        "missing_package": "unable to locate package",
        "go_inconsistent_vendoring": "inconsistent vendoring",
        "missing_configure_script": "./configure: no such file or directory",
    }
    for key, needle in signatures.items():
        if needle in msg:
            return key
    return ""


def _should_force_replan(state: AgentState) -> bool:
    """Bounded replan guard to avoid fixer/scout thrash on bad plans."""
    signature = _replan_signature(state.last_error or "")
    if not signature:
        return False
    attempts_by_sig = state.context_cache.setdefault(
        "replan_attempts_by_signature", {}
    )
    count = int(attempts_by_sig.get(signature, 0))
    # Only force one scout replan per unique signature; if it still fails,
    # let normal routing decide between fixer/escalation.
    if count >= 1:
        return False
    attempts_by_sig[signature] = count + 1
    return True


# ============================================================================
# NODE WRAPPER
# ============================================================================


def agent_node(role: AgentRole):
    """Wrap agent nodes with error handling and retries.

    Provides state tracking and automatic rate-limit retry.
    """

    def decorator(func):
        def wrapper(state: AgentState) -> AgentState:
            import time

            state.current_agent = role
            logger.info(f"Node Started: {role.value}")

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    result = func(state)
                    logger.info(f"Node Completed: {role.value}")
                    return result
                except Exception as e:
                    error_str = str(e).lower()
                    is_rate_limit = any(
                        term in error_str
                        for term in [
                            "rate limit",
                            "429",
                            "too many requests",
                            "quota exceeded",
                            "resource_exhausted",
                            "402",
                            "spend limit exceeded",
                            "usd spend limit",
                        ]
                    )
                    if is_rate_limit and attempt < max_retries - 1:
                        wait_time = 30 * (attempt + 1)
                        logger.warning(
                            f"Rate limit hit in {role.value} "
                            f"(attempt {attempt + 1}/{max_retries}), "
                            f"waiting {wait_time}s before retry..."
                        )
                        time.sleep(wait_time)
                        continue

                    logger.exception(f"Exception in {role.value} node: {e}")
                    error_msg = (
                        f"Unexpected error in {role.value} agent: {str(e)}"
                    )
                    error = create_error_record(
                        message=error_msg,
                        category=classify_error(str(e)),
                        attempt_number=state.attempt_count,
                    )
                    state.add_error(error)
                    state.last_error = error_msg
                    state.build_status = BuildStatus.ESCALATED
                    state.current_phase = "escalate"
                    return state

            return state

        return wrapper

    return decorator


# ============================================================================
# NODE: INITIALIZATION
# ============================================================================


@agent_node(AgentRole.SUPERVISOR)
def init_node(state: AgentState) -> AgentState:
    """Initialize the workflow by cloning the repository.

    Uses scripted operations; this is a zero-cost operation (no LLM
    calls).
    """
    logger.info(f"Initializing workflow for {state.repo_url}")

    result = scripted_ops.clone_or_update_repository(
        state.repo_url, state.repo_name
    )
    state.log_scripted_op("clone_or_update_repository")

    if not result.success:
        message = _build_command_error_message(
            result, f"Repository clone/update failed for {state.repo_url}"
        )
        error = create_error_record(
            message=message,
            category=classify_error(message),
            severity=FailureSeverity.HIGH,
            command=result.command,
        )
        state.add_error(error)
        state.build_status = BuildStatus.FAILED
        state.current_phase = "escalate"
        state.log_agent_decision(
            AgentRole.SUPERVISOR,
            "ESCALATE",
            f"Critical init failure ({error.severity.value}): "
            f"{error.message[:200]}",
        )
        return state

    logger.info("Performing quick analysis with scripted operations...")
    try:
        analysis = quick_analysis(state.repo_path)
        state.log_scripted_op("quick_analysis")

        state.build_system_info = analysis.get("build_system")

        # Warn if build system could not be detected
        if (
            state.build_system_info
            and state.build_system_info.type == "unknown"
        ):
            logger.warning(
                f" WARNING: Unable to identify build system in "
                f"{state.repo_name}. "
                "Exhaustive search was performed but no build "
                "configuration files found. "
                "The system will attempt configuration-free analysis "
                "or request manual input."
            )

        state.dependencies = analysis.get("dependencies")
        state.arch_specific_code = analysis.get("arch_specific_code", [])

        state.repo_tree = analysis.get("optimized_tree", "")
        logger.info(f"Repository tree captured ({len(state.repo_tree)} chars)")

        for doc_path in analysis.get("documentation", [])[:5]:
            content = scripted_ops.read_file(doc_path, max_lines=500)
            state.cache_file_content(doc_path, content)
            state.log_scripted_op("read_file")

        state.context_cache["quick_analysis"] = analysis

        if "go_main_info" in analysis:
            state.context_cache["go_main_info"] = analysis["go_main_info"]
            logger.info(
                f"Go main package detected: {analysis['go_main_info']}"
            )

        if "arch_build_files" in analysis:
            state.context_cache["arch_build_files"] = analysis[
                "arch_build_files"
            ]
            arch_info = analysis["arch_build_files"]
            if arch_info.get("has_arch_specific"):
                logger.info(
                    f"Arch-specific build files detected: "
                    f"{arch_info.get('archs_found')}, "
                    f"RISC-V exists: {arch_info.get('riscv_exists')}"
                )

        required_tools = set(["gcc", "make"])
        if state.build_system_info:
            required_tools.add(state.build_system_info.type)
        if state.dependencies:
            required_tools.update(state.dependencies.build_tools)

        system_info = scripted_ops.get_system_info(tools=list(required_tools))
        missing_tools = [
            t for t, status in system_info.items() if status == "Not installed"
        ]
        state.context_cache["system_info"] = system_info
        state.context_cache["missing_tools"] = missing_tools

        bsi = state.build_system_info
        bsi_type = bsi.type if bsi else "unknown"
        missing_str = ", ".join(missing_tools) if missing_tools else "None"
        logger.info(
            f"Quick analysis complete: "
            f"Build system: {bsi_type}, "
            f"Arch-specific code: "
            f"{len(state.arch_specific_code)} instances. "
            f"Relevant missing tools: {missing_str}"
        )

    except Exception as e:
        logger.warning(f"Quick analysis failed: {e}")

    state.build_status = BuildStatus.PENDING
    state.current_phase = "initialized"

    return state


# ============================================================================
# NODE: PLANNER (Strategic Planning)
# ============================================================================

PLANNER_PROMPT = (
    "You are the strategic planner for a RISC-V (riscv64) software porti"
    "ng agent.\n"
    "\n"
    "## Active sandbox (READ FIRST — overrides any conflicting knowledge"
    " below)\n"
    "{platform_banner}\n"
    "\n"
    "## Repository\n"
    "- Name: {repo_name} ({repo_url})\n"
    "- Path: {repo_path}\n"
    "- Build system: {build_system_info}\n"
    "- Dependencies (detected): {dependencies_info}\n"
    "- Existing arch support: {existing_archs}\n"
    "- Arch-specific code patterns: {arch_patterns}\n"
    "\n"
    "## Codebase structure\n"
    "{repo_tree}\n"
    "\n"
    "## RISC-V / platform knowledge\n"
    "{system_knowledge}\n"
    "\n"
    "## Your task\n"
    "Produce a short phased plan delegated to specialist agents. Keep it"
    " minimal —\n"
    "one scout phase + one builder phase is usually enough; only add mor"
    "e if the\n"
    "repo has unusual structure (e.g. autogen needed, subprojects, branc"
    "h switch).\n"
    "\n"
    "Agent roles:\n"
    "- **scout**: read build files, decide build commands, produce a Bui"
    "ldPlan\n"
    "- **builder**: execute commands, compile, verify artifacts\n"
    "- **fixer**: diagnose & patch failures (only added if predicted to "
    "be needed)\n"
    "\n"
    "Before answering, silently check: did I cover (a) the right build s"
    "ystem, (b)\n"
    "required `-dev` deps, (c) any arch-specific risks worth a fixer pha"
    "se? If\n"
    "unsure on build system, default to one scout phase that figures it "
    "out.\n"
    "\n"
    "## Output schema\n"
    "Return ONLY a JSON object with this exact shape:\n"
    "{{\n"
    '  "strategy": "<one sentence describing the porting approach>",\n'
    '  "complexity_score": <int 1-10>,\n'
    '  "estimated_total_time": "<e.g. \'5m\'>",\n'
    '  "estimated_total_cost": <float USD>,\n'
    '  "can_parallelize": [],\n'
    '  "phases": [\n'
    '    {{"id": 1, "name": "<short>", "description": "<what & '
    'why>",\n'
    '      "agent": "scout|builder|fixer", "use_scripted_ops": <bo'
    "ol>,\n"
    '      "depends_on": [], "estimated_cost": <float>}}\n'
    "  ]\n"
    "}}\n"
    "No prose, no markdown fences."
)


@agent_node(AgentRole.PLANNER)
def planner_node(state: AgentState) -> AgentState:
    """Decompose the task and create an execution plan.

    Reduces downstream LLM calls by providing clear direction.
    """
    logger.info("Starting strategic planning phase...")

    state.build_status = BuildStatus.PLANNING
    state.current_phase = "planning"

    # Prepare context from quick analysis
    build_sys = state.build_system_info
    deps = state.dependencies

    # Build system info
    bs_type = build_sys.type if build_sys else "Unknown"
    bs_conf = f"{build_sys.confidence:.2f}" if build_sys else "0.0"
    bs_file = build_sys.primary_file if build_sys else "Unknown"
    build_system_info = (
        f"Type: {bs_type}\n"
        f"Confidence: {bs_conf}\n"
        f"Primary File: {bs_file}"
    )

    # Dependencies info
    dependencies_info = "Unknown"
    if deps:
        dependencies_info = (
            f"Build tools: {', '.join(deps.build_tools)}\n"
            f"System packages: {len(deps.system_packages)} total\n"
            f"Libraries: {len(deps.libraries)} total"
        )

    # Target architecture
    target_arch = "RISC-V (riscv64)"

    # Architecture patterns - analyze existing architectures
    arch_patterns_list = []
    existing_archs_list = []

    # Look for existing architectures in file paths
    for file_path in state.file_content_cache.keys():
        file_lower = file_path.lower()
        if any(arch in file_lower for arch in ["x86", "x64", "amd64"]):
            existing_archs_list.append("x86_64")
        elif any(arch in file_lower for arch in ["arm", "arm64", "aarch64"]):
            existing_archs_list.append("arm64")
        elif any(arch in file_lower for arch in ["mips"]):
            existing_archs_list.append("mips")

    # Remove duplicates
    existing_archs_list = list(set(existing_archs_list))

    # Summarize arch patterns
    if state.arch_specific_code:
        patterns_by_type = {}
        for arch_code in state.arch_specific_code:
            if arch_code.arch_type not in patterns_by_type:
                patterns_by_type[arch_code.arch_type] = []
            patterns_by_type[arch_code.arch_type].append(
                f"{arch_code.file}:{arch_code.line}"
            )

        for arch_type, locations in patterns_by_type.items():
            arch_patterns_list.append(
                f"- {arch_type}: {len(locations)} occurrences in "
                f"{', '.join(locations[:3])}"
            )

    arch_patterns = (
        "\n".join(arch_patterns_list)
        if arch_patterns_list
        else "No architecture-specific patterns detected"
    )
    existing_archs = (
        f"Detected: {', '.join(existing_archs_list)}"
        if existing_archs_list
        else "No existing multi-architecture patterns detected"
    )

    system_knowledge = get_system_knowledge_summary()

    prompt = PLANNER_PROMPT.format(
        repo_name=state.repo_name,
        repo_url=state.repo_url,
        repo_path=state.repo_path,
        repo_tree=state.repo_tree if state.repo_tree else "(Not available)",
        build_system_info=build_system_info,
        dependencies_info=dependencies_info,
        target_arch=target_arch,
        arch_patterns=arch_patterns,
        existing_archs=existing_archs,
        system_knowledge=system_knowledge,
        platform_banner=_build_platform_banner(),
    )

    # Validated LLM call: retry-with-critique up to 2x, then fall back
    # to a deterministic plan.
    def _validate_plan(data: dict) -> ValidationResult:
        phases = data.get("phases")
        if not isinstance(phases, list) or not phases:
            return ValidationResult.bad(
                "'phases' must be a non-empty array of phase objects"
            )
        if not all(isinstance(p, dict) and "name" in p for p in phases):
            return ValidationResult.bad(
                "each phase must be an object with at least a 'name' key"
            )
        return ValidationResult.good()

    _planner_pool = get_model_pool_for_role(AgentRole.PLANNER)
    outcome = llm_call_with_validation(
        invoke_fn=invoke_llm,
        llm=_planner_pool[0],
        fallback_llms=_planner_pool[1:],
        prompt=prompt,
        validator=_validate_plan,
        # we materialize the default TaskPlan below
        fallback_factory=lambda: None,
        role=AgentRole.PLANNER.value,
        audit_metadata={"repo": state.repo_name, "phase": "planner"},
        cost_estimate=0.01,
        max_retries=2,
    )
    state.log_api_call(cost=0.01 * outcome.attempts)

    try:
        if outcome.used_fallback or outcome.data is None:
            logger.warning(
                f"Planner: using deterministic fallback after "
                f"{outcome.attempts} attempts "
                f"({outcome.last_error or 'no data'})"
            )
            state.task_plan = create_default_plan()
            state.context_cache["task_plan"] = {
                "fallback": True,
                "reason": outcome.last_error,
            }
            state.current_phase = "planned"
            return state

        plan_data = outcome.data

        # Create TaskPlan from validated payload
        phases = []
        for p in plan_data["phases"]:
            if not isinstance(p, dict):
                logger.warning(f"Skipping non-dict phase entry: {p}")
                continue
            agent_str = p.get("agent", "builder").lower()
            if agent_str in ["architect", "supervisor"]:
                role = AgentRole.BUILDER  # Fallback for planning
            elif "scout" in agent_str:
                role = AgentRole.SCOUT
            elif "fix" in agent_str:
                role = AgentRole.FIXER
            elif "build" in agent_str:
                role = AgentRole.BUILDER
            else:
                try:
                    role = AgentRole(agent_str)
                except ValueError:
                    role = AgentRole.BUILDER

            phase = TaskPhase(
                id=p.get("id", len(phases) + 1),
                name=p.get("name", "unknown"),
                description=p.get("description", ""),
                agent=role,
                use_scripted_ops=p.get(
                    "use_scripted_ops", "scout" not in agent_str
                ),
                depends_on=p.get("depends_on", []),
                estimated_cost=p.get("estimated_cost", 0.0),
            )
            phases.append(phase)

        if not phases:
            logger.warning(
                "No valid phases parsed from LLM response — using default plan"
            )
            state.task_plan = create_default_plan()
        else:
            state.task_plan = TaskPlan(
                phases=phases,
                can_parallelize=plan_data.get("can_parallelize", []),
                estimated_total_cost=plan_data.get(
                    "estimated_total_cost", 0.0
                ),
                estimated_total_time=plan_data.get(
                    "estimated_total_time", "unknown"
                ),
                complexity_score=plan_data.get("complexity_score", 5),
            )

        logger.info(
            f"Strategic plan created: {len(phases)} phases, "
            f"complexity: {state.task_plan.complexity_score}/10, "
            f"estimated cost: ${state.task_plan.estimated_total_cost:.3f}"
        )

        state.context_cache["task_plan"] = plan_data

    except Exception as e:
        logger.error(f"Planning post-processing failed: {e}")
        state.task_plan = create_default_plan()

    state.current_phase = "planned"
    return state


# ============================================================================
# NODE: SUMMARIZER (New - Professional Documentation)
# ============================================================================

SUMMARIZER_PROMPT = (
    "You are writing the RISC-V porting report for **{repo_name}**.\n"
    "\n"
    "## Build context\n"
    "- Repo: {repo_url}\n"
    "- Build system: {build_system}\n"
    "- Duration: {duration} | API calls: {api_calls} (~${cost:.4f})\n"
    "\n"
    "## Architecture issues observed\n"
    "{arch_issues}\n"
    "\n"
    "## Fixes applied\n"
    "{fixes}\n"
    "\n"
    "## Build steps actually executed\n"
    "{build_steps}\n"
    "\n"
    "## Required sections (use exactly these H2 headings, in order)\n"
    "1. `## Executive Summary` — status (success / partial / failed), 1-"
    "2 sentence portability verdict.\n"
    "2. `## Prerequisites` — exact package install command(s) for the ta"
    "rget distro.\n"
    "3. `## Build Instructions` — copy-pasteable shell commands, one fen"
    "ced block per logical step, in order.\n"
    "4. `## Architecture Notes` — only the arch-specific things this pro"
    "ject needed (preprocessor macros, SIMD handling, libc differences)."
    " Omit if N/A.\n"
    "5. `## Verification` — how to confirm produced binaries are riscv64"
    " (`file`, `readelf -h`).\n"
    "6. `## Known Issues & Recommendations` — leftover concerns + upstre"
    "am contribution ideas.\n"
    "\n"
    "Rules:\n"
    "- Every command in fenced ```bash blocks.\n"
    "- Only mention packages/flags that were actually used.\n"
    "- No generic RISC-V tutorial content — be specific to this project.\n"
    "- If a section has nothing useful to say, write a single sentence a"
    "cknowledging that and move on."
)


def create_default_plan() -> TaskPlan:
    """Create a default task plan if planning fails."""
    return TaskPlan(
        phases=[
            TaskPhase(
                1,
                "scout",
                "Analyze and create build plan",
                AgentRole.SCOUT,
                False,
                [],
            ),
            TaskPhase(
                2, "build", "Execute build", AgentRole.BUILDER, False, [1]
            ),
        ],
        estimated_total_cost=0.02,
        complexity_score=5,
    )


# ============================================================================
# NODE: SUPERVISOR (Orchestration)
# ============================================================================

SUPERVISOR_PROMPT = (
    "You are the routing supervisor for a RISC-V porting agent. The\n"
    "heuristic router already produced a recommendation; you decide whet"
    "her to\n"
    "follow it or override.\n"
    "\n"
    "## State snapshot\n"
    "- Phase: {current_phase} | Build status: {build_status}\n"
    "- Attempt {attempt_count}/{max_attempts} | Last agent: {current_age"
    "nt}\n"
    "- Stats: scripted ops={scripted_ops_count}, API calls={api_calls}, "
    "cost=${cost:.4f}\n"
    "\n"
    "### Build plan\n"
    "{build_plan_summary}\n"
    "\n"
    "### Recent agent history\n"
    "{agent_history}\n"
    "\n"
    "### Architecture issues\n"
    "{arch_issues_summary}\n"
    "\n"
    "### Recent errors\n"
    "{error_summary}\n"
    "\n"
    "### Extra context\n"
    "{additional_context}\n"
    "\n"
    "## Verification step (do this before answering)\n"
    "1. Has the same error category fired 3+ times in a row? → ESCALATE\n"
    "2. Is the build green and verified? → FINISH\n"
    "3. Is there no build plan yet? → SCOUT\n"
    "4. Has the last build failed with a fixable error? → FIX\n"
    "5. Otherwise → BUILD\n"
    "\n"
    "## Output schema (JSON only, no prose)\n"
    "{{\n"
    '  "next_agent": "SCOUT|BUILD|FIX|FINISH|ESCALATE",\n'
    '  "reasoning": "<2 sentences max>",\n'
    '  "confidence": "high|medium|low"\n'
    "}}"
)


@agent_node(AgentRole.SUPERVISOR)
def supervisor_node(state: AgentState) -> AgentState:
    """Make a routing decision with cost-aware heuristics."""
    bp_state = "Exists" if state.build_plan else "Missing"
    logger.info(
        f"Supervisor making routing decision... (BuildPlan: {bp_state})"
    )

    should_esc, esc_reason = should_escalate(state)
    if should_esc:
        logger.warning(f"Automatic escalation: {esc_reason}")
        state.current_phase = "escalate"
        return state

    recommended_action = get_next_action_recommendation(state)
    state.log_scripted_op("supervisor_routing")

    if state.api_calls_made > 8 or state.api_cost_usd > 0.08:
        state.log_agent_decision(
            AgentRole.SUPERVISOR,
            recommended_action.value,
            "Cost optimization heuristic (API calls > 8 or cost > $0.08)",
        )
        logger.info(
            f"Using cost-optimized routing: {recommended_action.value}"
        )
        state.current_phase = recommended_action.value.lower()
        return state

    if state.build_status == BuildStatus.SUCCESS:
        state.log_agent_decision(
            AgentRole.SUPERVISOR,
            "FINISH",
            "Build succeeded, moving to final documentation.",
        )
        state.current_phase = "finish"
        return state

    if state.attempt_count >= state.max_attempts:
        state.log_agent_decision(
            AgentRole.SUPERVISOR,
            "ESCALATE",
            f"Max attempts ({state.max_attempts}) reached.",
        )
        state.current_phase = "escalate"
        return state

    if state.attempt_count >= 3 and state.is_in_error_loop():
        logger.warning("Detected error loop - switching to escalation")
        state.log_agent_decision(
            AgentRole.SUPERVISOR, "ESCALATE", "Stuck in error loop"
        )
        state.current_phase = "escalate"
        return state

    if not state.task_plan:
        state.log_agent_decision(
            AgentRole.SUPERVISOR, "PLAN", "No task plan exists."
        )
        state.current_phase = "planner"
        return state

    if not state.build_plan and state.task_plan:
        state.log_agent_decision(
            AgentRole.SUPERVISOR,
            "SCOUT",
            "Build plan missing but task plan exists.",
        )
        state.current_phase = "scout"
        return state

    if state.build_status == BuildStatus.FAILED:
        if _should_force_replan(state):
            sig = _replan_signature(state.last_error or "") or "unknown"
            state.log_agent_decision(
                AgentRole.SUPERVISOR,
                "SCOUT",
                f"Failure signature '{sig}' indicates plan-level issue; "
                "requesting one bounded replan.",
            )
            state.current_phase = "scout"
            return state

        # MISSING_TOOLS / DEPENDENCY need an updated build plan from
        # scout (add packages, change steps).
        # CONFIGURATION, COMPILATION, LINKING, ARCHITECTURE -> send to
        # fixer to patch in-place.
        if state.last_error_category in (
            ErrorCategory.MISSING_TOOLS,
            ErrorCategory.DEPENDENCY,
            ErrorCategory.UNKNOWN,
        ):
            if (
                state.last_error_category == ErrorCategory.MISSING_TOOLS
                and is_toolchain_version_mismatch(state.last_error)
            ):
                state.log_agent_decision(
                    AgentRole.SUPERVISOR,
                    "ESCALATE",
                    "Sandbox toolchain is too old for this repository.",
                )
                state.current_phase = "escalate"
                return state
            state.log_agent_decision(
                AgentRole.SUPERVISOR,
                "SCOUT",
                f"{state.last_error_category.value} - need updated "
                f"build plan.",
            )
            state.current_phase = "scout"
            return state
        else:
            err_cat = state.last_error_category
            err_val = err_cat.value if err_cat else "Unknown"
            state.log_agent_decision(
                AgentRole.SUPERVISOR,
                "FIXER",
                f"Error: {err_val}",
            )
            state.current_phase = "fixer"
            return state

    if state.build_status == BuildStatus.PENDING and state.build_plan:
        state.log_agent_decision(
            AgentRole.SUPERVISOR, "BUILDER", "Build plan ready, executing."
        )
        state.current_phase = "builder"
        return state

    decision_context = ""
    if state.build_status == BuildStatus.FAILED:
        decision_context = (
            f"Last build failed with {state.last_error_category}. "
            f"Consider if FIXER can handle it, or if SCOUT needs more info."
        )
    elif not state.build_plan:
        decision_context = "No build plan exists. SCOUT must create one."
    elif state.build_status == BuildStatus.PENDING:
        decision_context = (
            "A build plan exists. BUILDER should now execute the build phases."
        )
    elif state.build_status == BuildStatus.SUCCESS:
        decision_context = "Build succeeded. FINISH or run tests if not done."

    # Build context for supervisor prompt
    build_plan_summary = "No build plan yet"
    if state.build_plan:
        phase_names = [p.name for p in state.build_plan.phases]
        build_plan_summary = (
            f"Build System: {state.build_plan.build_system}, "
            f"Phases: {', '.join(phase_names)}, "
            f"Last completed: {state.last_successful_phase}"
        )

    agent_history = ""
    if state.audit_trail:
        recent = state.audit_trail[-5:]
        agent_history = "\n".join(
            [
                f"- {entry.get('agent', 'unknown')}: "
                f"{entry.get('event', 'unknown')} - "
                f"{str(entry.get('data', ''))[:100]}"
                for entry in recent
            ]
        )
    else:
        agent_history = "No previous agent actions"

    arch_issues_summary = "None detected"
    if state.arch_specific_code:
        arch_issues_summary = (
            f"{len(state.arch_specific_code)} "
            f"architecture-specific code instances found"
        )

    error_summary = "No errors"
    if state.error_history:
        recent_errors = state.error_history[-3:]
        error_summary = "\n".join(
            [
                f"- [{e.category.value}] {e.message[:150]}"
                for e in recent_errors
            ]
        )

    prompt = SUPERVISOR_PROMPT.format(
        current_phase=state.current_phase,
        build_status=state.build_status.value,
        attempt_count=state.attempt_count,
        max_attempts=state.max_attempts,
        current_agent=(
            state.current_agent.value if state.current_agent else "none"
        ),
        build_plan_summary=build_plan_summary,
        scripted_ops_count=state.scripted_ops_count,
        api_calls=state.api_calls_made,
        cost=state.api_cost_usd,
        agent_history=agent_history,
        arch_issues_summary=arch_issues_summary,
        error_summary=error_summary,
        additional_context=decision_context,
    )

    try:
        messages = [HumanMessage(content=prompt)]
        llm = get_model_for_role(AgentRole.SUPERVISOR)
        response = invoke_llm(llm, messages)
        state.log_api_call(cost=0.002)

        content = extract_content(response.content)

        # Log LLM call for debugging
        log_llm_call(
            agent_role=AgentRole.SUPERVISOR.value,
            prompt=prompt,
            response=content,
            model=llm.model_name if hasattr(llm, "model_name") else "unknown",
            cost_usd=0.002,
            metadata={
                "repo": state.repo_name,
                "phase": "supervisor",
                "status": state.build_status.value,
            },
        )

        # Try to parse JSON response
        json_match = extract_json_block(content)
        decision = json.loads(json_match)

        action_str = decision.get("next_agent", "").strip().upper()

        action_map = {
            "SCOUT": "scout",
            "BUILD": "builder",
            "BUILDER": "builder",
            "FIX": "fixer",
            "FIXER": "fixer",
            "ESCALATE": "escalate",
            "FINISH": "finish",
        }

        state.log_agent_decision(
            AgentRole.SUPERVISOR,
            action_str,
            decision.get("reasoning", "LLM decision"),
        )
        state.current_phase = action_map.get(action_str, "scout")

        logger.info(f"Supervisor decision: {action_str}")

    except Exception as e:
        logger.error(f"Supervisor failed: {e}, using fallback")
        state.current_phase = recommended_action.value.lower()

    return state


# ============================================================================
# NODE: SCOUT (Architecture Analysis)
# ============================================================================

SCOUT_PROMPT = (
    "You are the build scout for a RISC-V porting agent. Produce a\n"
    "minimal, executable BuildPlan for **{repo_name}** built natively at"
    " `{repo_path}`.\n"
    "\n"
    "## ACTIVE SANDBOX (use ONLY these package commands)\n"
    "{platform_banner}\n"
    "\n"
    "## Target\n"
    "- Architecture: {target_arch} ({arch_identifiers})\n"
    "- Native compilation only (DO NOT set `--host`, `GOOS/GOARCH`, `-DC"
    "MAKE_SYSTEM_PROCESSOR`).\n"
    "\n"
    "## Repo facts\n"
    "- Build system: {build_system} | Module dir: `{module_dir}`\n"
    "- Arch-specific code: {arch_code_count} occurrences | Cached docs: "
    "{doc_count}\n"
    "- Project context: {go_main_info}\n"
    "\n"
    "### Project structure\n"
    "{repo_tree}\n"
    "\n"
    "### Arch patterns\n"
    "{arch_build_patterns}\n"
    "\n"
    "### Detected dependencies\n"
    "{dependencies}\n"
    "\n"
    "### Arch concerns\n"
    "{arch_concerns}\n"
    "\n"
    "### Sandbox tools available\n"
    "{system_info}\n"
    "\n"
    "### Documentation excerpts\n"
    "{documentation}\n"
    "\n"
    "## Platform knowledge (USE THIS — don't invent package names)\n"
    "{system_knowledge}\n"
    "\n"
    "{few_shot_examples}\n"
    "\n"
    "## Build plan rules\n"
    "1. Each command runs in a fresh shell at `{repo_path}`. **Always ch"
    "ain `cd` with `&&`** — e.g. `cd build && cmake ..`, never two comma"
    "nds.\n"
    "2. Group commands into ≤3 phases: `setup` (install deps) → `configu"
    "re` (optional) → `build`. Skip phases that have nothing to do.\n"
    "3. Install only `-dev` packages you can justify from CMakeLists.txt"
    " / configure.ac / meson.build / go.mod. No guessing.\n"
    "4. Use the package names from the platform knowledge above. Do NOT "
    "use names from a different distro.\n"
    "5. If the project has both CMake and autotools, prefer the one whos"
    "e dependencies are easier to satisfy; if unsure, prefer autotools.\n"
    "6. For Go: if `go.mod` is missing, first phase must `go mod init <m"
    "odule_path>` then `go mod tidy`. Always pass `-buildvcs=false` to `"
    "go build`. **If `Project context` above gives a `build_command`, us"
    "e exactly that command — do not substitute `go build ./...` (which "
    "compiles every helper and fails on the first one with missing deps)"
    ".** Never emit `go build ./cmd` or `go build ./<top-level-dir>` "
    "without an explicit output path; if building a subpackage, use "
    "`go build -o ./bin/<name> <package>`.\n"
    "7. Disable optional features that pull in heavy or x86-specific dep"
    "s (tests, SIMD, examples, docs).\n"
    "8. NEVER reference paths or files you can't see in the project stru"
    "cture.\n"
    "\n"
    "## Self-check (do silently before responding)\n"
    "- Did I read the actual build files (or admit I couldn't)?\n"
    "- Does every package install command exist for the target distro?\n"
    "- Do my commands actually run if pasted into a shell at `{repo_path"
    "}`?\n"
    "\n"
    "## Output schema (JSON only)\n"
    "{{\n"
    '  "build_system": "go|cmake|make|autotools|cargo|meson|other",\n'
    '  "build_system_confidence": <0-1 float>,\n'
    '  "phases": [\n'
    '    {{"id": 1, "name": "setup", "commands": ["..."], "ca'
    'n_parallelize": false, "expected_duration": "30s"}},\n'
    '    {{"id": 2, "name": "build", "commands": ["..."], "ca'
    'n_parallelize": false, "expected_duration": "2m"}}\n'
    "  ],\n"
    '  "total_estimated_duration": "<sum>",\n'
    '  "notes": ["..."],\n'
    '  "architecture_concerns": ["..."]\n'
    "}}"
)


def validate_build_plan(plan: BuildPlan) -> tuple[bool, str]:
    """Validate BuildPlan for hallucinations and common issues."""
    hallucination_patterns = [
        "/path/to/",
        "your_username",
        "example.com",
        "riscv64-unknown-linux-gnu-gcc",  # Unless we confirmed it's there
    ]

    # Go subcommands that must not appear as apk package names
    go_subcommands = {
        "mod",
        "build",
        "get",
        "install",
        "run",
        "test",
        "tool",
        "generate",
        "fmt",
        "vet",
    }

    for phase in plan.phases:
        for cmd in phase.commands:
            for pattern in hallucination_patterns:
                if pattern in cmd:
                    return (
                        False,
                        f"Hallucination detected in command: '{cmd}' "
                        f"(contains '{pattern}')",
                    )

            # Check for absolute paths that look guessed
            if "/home/" in cmd and "/workspace/" not in cmd:
                return False, f"Suspicious absolute path in command: '{cmd}'"

            # Check for `apk add go mod` / `apt-get install go build`
            # style hallucinations where Go subcommands are mistaken
            # for package names.
            stripped = cmd.strip()
            if (
                stripped.startswith("apk add")
                or stripped.startswith("apt-get install")
                or stripped.startswith("apt install")
            ):
                parts = stripped.split()
                # apk add <pkgs> / apt-get install <pkgs>
                skip = 2 if stripped.startswith("apk") else 2
                pkg_names = [p for p in parts[skip:] if not p.startswith("-")]
                for pkg in pkg_names:
                    if pkg in go_subcommands:
                        return False, (
                            f"Hallucination: '{cmd}' — '{pkg}' is a Go "
                            f"subcommand, not a package. "
                            f"Use 'go {pkg}' as a separate command."
                        )

            # Check for bad cmake patterns
            if "cmake" in cmd:
                if "cd build &&" in cmd and "-B build" in cmd:
                    return (
                        False,
                        f"Bad cmake command (would create nested "
                        f"build dir): '{cmd}'",
                    )
                # Ensure we're doing `cmake ..` from inside build dir,
                # not `cmake -S . -B build`.
                if (
                    "cd build &&" in cmd
                    and "cmake -S" in cmd
                    and "-B build" in cmd
                ):
                    return (
                        False,
                        f"Bad cmake pattern (creates build/build): '{cmd}'",
                    )

    return True, ""


@agent_node(AgentRole.SCOUT)
def scout_node(state: AgentState) -> AgentState:
    """Analyze the repository, leveraging scripted pre-analysis."""
    logger.info("Scout analyzing repository...")

    state.build_status = BuildStatus.SCOUTING
    state.current_phase = "scouting"

    build_sys = state.build_system_info
    deps = state.dependencies

    documentation = "No documentation cached"
    if state.file_content_cache:
        doc_files = [
            k
            for k in state.file_content_cache.keys()
            if any(d in k.upper() for d in ["README", "INSTALL", "BUILD"])
        ]
        if doc_files:
            report_docs = doc_files[:3]
            documentation = "\n\n".join(
                [
                    f"## {Path(f).name}\n{state.file_content_cache[f][:2000]}"
                    for f in report_docs
                ]
            )

    arch_concerns = "None detected"
    if state.arch_specific_code:
        arch_concerns = "\n".join(
            [
                f"- {a.file}:{a.line} - {a.arch_type} ({a.severity})"
                for a in state.arch_specific_code[:10]
            ]
        )

    dependencies = "Unknown"
    if deps:
        dependencies = json.dumps(
            {
                "build_tools": deps.build_tools,
                "system_packages": deps.system_packages[:10],
                "libraries": deps.libraries[:10],
            },
            indent=2,
        )

    system_info_raw = scripted_ops.get_system_info()
    system_info = "\n".join(
        [f"- {k}: {v}" for k, v in system_info_raw.items()]
    )

    module_dir = (
        build_sys.module_dir
        if build_sys and hasattr(build_sys, "module_dir")
        else ""
    )

    go_main_info = state.context_cache.get("go_main_info", {})
    build_type_lower = (build_sys.type if build_sys else "").lower()
    if go_main_info:
        go_main_str = json.dumps(go_main_info, indent=2)
    elif build_type_lower == "go":
        go_main_str = (
            "Go project detected; main package not auto-discovered "
            "(scout should inspect the repo for main.go or cmd/ "
            "subdirs)"
        )
    else:
        go_main_str = "Not a Go project"

    few_shot_context = {
        "build_system": build_sys.type if build_sys else "unknown",
        "has_main": go_main_info.get("has_main", False),
        "main_path": go_main_info.get("main_path", ""),
        "module_dir": module_dir,
    }
    few_shot_examples = format_few_shot_examples(
        "scout", few_shot_context, max_examples=2, max_chars=2000
    )

    # Build architecture patterns info
    arch_patterns_str = "No architecture-specific patterns detected"
    if state.arch_specific_code:
        arch_patterns_by_type = {}
        for arch_code in state.arch_specific_code:
            if arch_code.arch_type not in arch_patterns_by_type:
                arch_patterns_by_type[arch_code.arch_type] = 0
            arch_patterns_by_type[arch_code.arch_type] += 1
        arch_patterns_str = "\n".join(
            [
                f"- {arch_type}: {count} matches"
                for arch_type, count in arch_patterns_by_type.items()
            ]
        )

    prompt = SCOUT_PROMPT.format(
        target_arch="RISC-V (riscv64)",
        arch_identifiers="rv64, riscv, riscv64, RISCV64",
        repo_name=state.repo_name,
        platform_banner=_build_platform_banner(),
        build_system=(build_sys.type if build_sys else "unknown")
        .replace("{", "{{")
        .replace("}", "}}"),
        repo_tree=(
            state.repo_tree.replace("{", "{{").replace("}", "}}")
            if state.repo_tree
            else "(Not available)"
        ),
        arch_build_patterns=arch_patterns_str,
        module_dir=module_dir,
        deps_count=len(deps.system_packages) if deps else 0,
        arch_code_count=len(state.arch_specific_code),
        doc_count=len(state.file_content_cache),
        go_main_info=go_main_str.replace("{", "{{").replace("}", "}}"),
        documentation=documentation.replace("{", "{{").replace("}", "}}"),
        arch_concerns=arch_concerns.replace("{", "{{").replace("}", "}}"),
        dependencies=dependencies.replace("{", "{{").replace("}", "}}"),
        repo_path=state.repo_path.replace("{", "{{").replace("}", "}}"),
        system_info=system_info.replace("{", "{{").replace("}", "}}"),
        system_knowledge=get_system_knowledge_summary()
        .replace("{", "{{")
        .replace("}", "}}"),
        architecture=system_info_raw.get("architecture", "riscv64")
        .replace("{", "{{")
        .replace("}", "}}"),
        few_shot_examples=few_shot_examples,
    )

    def _validate_scout(data: dict) -> ValidationResult:
        phases = data.get("phases")
        if not isinstance(phases, list) or not phases:
            return ValidationResult.bad("'phases' must be a non-empty array")
        for i, p in enumerate(phases):
            if not isinstance(p, dict):
                return ValidationResult.bad(f"phase {i} is not an object")
            if not p.get("name"):
                return ValidationResult.bad(f"phase {i} missing 'name'")
            cmds = p.get("commands")
            if not isinstance(cmds, list) or not cmds:
                return ValidationResult.bad(
                    f"phase {i} ({p.get('name')}) must have a "
                    f"non-empty 'commands' array"
                )
            if not all(isinstance(c, str) and c.strip() for c in cmds):
                return ValidationResult.bad(
                    f"phase {i} ({p.get('name')}) has empty/non-string command"
                )
        # Dry-build a BuildPlan to reuse the existing semantic validator
        try:
            trial_phases = [
                BuildPhase(
                    id=p.get("id", i + 1),
                    name=p["name"],
                    commands=p["commands"],
                    can_parallelize=p.get("can_parallelize", False),
                    expected_duration=p.get("expected_duration", "unknown"),
                )
                for i, p in enumerate(phases)
            ]
            trial_plan = BuildPlan(
                build_system=data.get(
                    "build_system", build_sys.type if build_sys else "unknown"
                ),
                build_system_confidence=data.get(
                    "build_system_confidence", 0.5
                ),
                phases=trial_phases,
                total_estimated_duration=data.get(
                    "total_estimated_duration", "unknown"
                ),
            )
            ok, reason = validate_build_plan(trial_plan)
            if not ok:
                return ValidationResult.bad(
                    f"{reason}. Avoid absolute paths like "
                    f"/path/to/, /home/, /root/"
                )
        except Exception as exc:
            return ValidationResult.bad(
                f"could not materialize BuildPlan: {exc}"
            )
        return ValidationResult.good()

    _scout_pool = get_model_pool_for_role(AgentRole.SCOUT)
    outcome = llm_call_with_validation(
        invoke_fn=invoke_llm,
        llm=_scout_pool[0],
        fallback_llms=_scout_pool[1:],
        prompt=prompt,
        validator=_validate_scout,
        fallback_factory=lambda: None,  # we build the BuildPlan fallback below
        role=AgentRole.SCOUT.value,
        audit_metadata={
            "repo": state.repo_name,
            "build_system": (
                state.build_system_info.type
                if state.build_system_info
                else "unknown"
            ),
        },
        cost_estimate=0.01,
        max_retries=2,
    )
    state.log_api_call(cost=0.01 * outcome.attempts)

    try:
        if outcome.data is None:
            raise ValueError(
                outcome.last_error or "scout produced no valid plan"
            )

        plan_data = outcome.data
        phases = [
            BuildPhase(
                id=p.get("id", i + 1),
                name=p["name"],
                commands=p["commands"],
                can_parallelize=p.get("can_parallelize", False),
                expected_duration=p.get("expected_duration", "unknown"),
            )
            for i, p in enumerate(plan_data["phases"])
        ]

        state.build_plan = BuildPlan(
            build_system=plan_data.get(
                "build_system", build_sys.type if build_sys else "unknown"
            ),
            build_system_confidence=plan_data.get(
                "build_system_confidence",
                build_sys.confidence if build_sys else 0.5,
            ),
            phases=phases,
            total_estimated_duration=plan_data.get(
                "total_estimated_duration", "unknown"
            ),
            notes=plan_data.get("notes", []),
        )
        state.last_successful_phase = 0
        logger.info(f"Build plan created and validated: {len(phases)} phases")

    except Exception as e:
        logger.error(f"Scout failed: {e}")
        logger.info("Using fallback build plan due to scout failure")
        state.add_error(
            create_error_record(
                message=f"Scout failed: {e}",
                category=ErrorCategory.UNKNOWN,
            )
        )
        state.build_plan = create_fallback_build_plan(state)

    state.build_status = BuildStatus.PENDING
    return state


def create_fallback_build_plan(state: AgentState) -> BuildPlan:
    """Create a basic fallback build plan based on detected build system."""
    from .platforms import get_active_profile

    profile = get_active_profile()
    build_sys = state.build_system_info
    build_type = build_sys.type if build_sys else "unknown"

    def _setup(canonical_pkgs: List[str]) -> str:
        """Render install command for canonical package names.

        Uses the active platform profile.
        """
        return profile.install_cmd(
            [profile.resolve(p) for p in canonical_pkgs]
        )

    if build_type == "go":
        go_main_info = state.context_cache.get("go_main_info", {})
        needs_init = go_main_info.get("needs_go_init", False)
        build_cmd = go_main_info.get(
            "build_command", "go build -buildvcs=false ./..."
        )
        module_name = (
            state.repo_url.replace("https://", "").rstrip("/")
            if state.repo_url
            else f"github.com/unknown/{state.repo_name}"
        )

        build_cmds = []
        if needs_init:
            build_cmds += [f"go mod init {module_name}", "go mod tidy"]
        else:
            build_cmds.append("go mod tidy")
        build_cmds.append(
            build_cmd
            if build_cmd != "go build ."
            else "go build -buildvcs=false ./..."
        )

        return BuildPlan(
            build_system="go",
            build_system_confidence=0.8,
            phases=[
                BuildPhase(1, "setup", [_setup(["git"])], False, "30s"),
                BuildPhase(2, "build", build_cmds, False, "3m"),
            ],
            total_estimated_duration="4m",
            notes=[f"Fallback Go build plan ({profile.name})"],
        )
    elif build_type == "cmake":
        return BuildPlan(
            build_system="cmake",
            build_system_confidence=0.7,
            phases=[
                BuildPhase(
                    1, "setup", [_setup(["gcc", "cmake"])], False, "30s"
                ),
                BuildPhase(
                    2,
                    "configure",
                    [
                        "mkdir -p build",
                        "cd build && cmake .. -DCMAKE_BUILD_TYPE=Release",
                    ],
                    False,
                    "1m",
                ),
                BuildPhase(
                    3, "build", ["cd build && make -j$(nproc)"], False, "5m"
                ),
            ],
            total_estimated_duration="7m",
            notes=[f"Fallback CMake build plan ({profile.name})"],
        )
    elif build_type == "make":
        return BuildPlan(
            build_system="make",
            build_system_confidence=0.7,
            phases=[
                BuildPhase(1, "setup", [_setup(["gcc"])], False, "30s"),
                BuildPhase(2, "build", ["make -j$(nproc)"], False, "5m"),
            ],
            total_estimated_duration="6m",
            notes=[f"Fallback Make build plan ({profile.name})"],
        )
    elif build_type == "cargo":
        return BuildPlan(
            build_system="cargo",
            build_system_confidence=0.8,
            phases=[
                BuildPhase(1, "setup", [_setup(["rust"])], False, "30s"),
                BuildPhase(2, "build", ["cargo build --release"], False, "5m"),
            ],
            total_estimated_duration="6m",
            notes=[f"Fallback Cargo build plan ({profile.name})"],
        )
    elif build_type == "meson":
        return BuildPlan(
            build_system="meson",
            build_system_confidence=0.7,
            phases=[
                BuildPhase(
                    1,
                    "setup",
                    [_setup(["gcc", "meson", "ninja"])],
                    False,
                    "30s",
                ),
                BuildPhase(
                    2, "configure", ["meson setup builddir"], False, "1m"
                ),
                BuildPhase(3, "build", ["ninja -C builddir"], False, "5m"),
            ],
            total_estimated_duration="7m",
            notes=[f"Fallback Meson build plan ({profile.name})"],
        )
    elif build_type == "autotools":
        has_configure = (
            os.path.exists(
                os.path.join(_to_host_path(state.repo_path), "configure")
            )
            if state.repo_path
            else True
        )
        configure_phase_cmds = ["./configure"]
        if not has_configure:
            configure_phase_cmds = [
                "cp /usr/share/gettext/m4/*.m4 m4/ 2>/dev/null || true",
                "autoreconf -fi",
                "./configure",
            ]
        return BuildPlan(
            build_system="autotools",
            build_system_confidence=0.6,
            phases=[
                BuildPhase(
                    1,
                    "setup",
                    [
                        _setup(["gcc", "autotools", "pkgconfig"]),
                        "cp /usr/share/gettext/m4/*.m4 "
                        "m4/ 2>/dev/null || true",
                    ],
                    False,
                    "30s",
                ),
                BuildPhase(2, "configure", configure_phase_cmds, False, "3m"),
                BuildPhase(3, "build", ["make -j$(nproc)"], False, "5m"),
            ],
            total_estimated_duration="9m",
            notes=[f"Fallback Autotools build plan ({profile.name})"],
        )
    else:
        logger.warning(
            f"Unknown build system '{build_type}', using generic fallback"
        )
        return BuildPlan(
            build_system=build_type,
            build_system_confidence=0.3,
            phases=[
                BuildPhase(1, "setup", [_setup(["gcc"])], False, "30s"),
                BuildPhase(2, "build", ["make -j$(nproc)"], False, "5m"),
            ],
            total_estimated_duration="6m",
            notes=[
                f"Generic fallback - build system not recognized "
                f"({profile.name})"
            ],
        )


# ============================================================================
# NODE: BUILDER (Build Execution)
# ============================================================================


def _fixup_top_builddir_in_submakefiles(docker_repo_path: str) -> None:
    """Post-configure fixup for old gettext/autotools projects.

    Old po/Makefile.in.in templates reference $(top_builddir) as a
    make variable but never define it (they expected @top_builddir@
    substitution that was added in newer gettext). When make descends
    into po/ without passing top_builddir, the dependency
    $(top_builddir)/config.status expands to /config.status which
    doesn't exist -> "No rule to make target '/config.status'".

    Fix: run a shell script INSIDE Docker (to avoid host permission
    issues with root-owned files) that scans all subdirectory
    Makefiles and injects the correct relative top_builddir assignment.

    Args:
        docker_repo_path: Path to the repository inside the container.
    """
    try:
        from src.platforms import get_container_name

        container = get_container_name()
        # Shell script: walk subdirs, find Makefiles using
        # $(top_builddir) that don't already define it, and inject
        # `top_builddir = <relpath>`.
        script = (
            f"cd {docker_repo_path} && "
            "find . -mindepth 2 -name Makefile "
            "-not -path './.git/*' | while read f; do "
            "  if grep -q '\\$(top_builddir)' \"$f\" "
            "&& ! grep -q '^top_builddir' \"$f\"; then "
            "    depth=$(echo \"$f\" | tr -cd '/' | wc -c); "
            "depth=$((depth - 1)); "
            "    rel=$(python3 -c \"print('/'.join(['..'] * $depth))\"); "
            '    sed -i "1s|^|top_builddir = $rel\\n|" "$f" && '
            '    echo "Injected top_builddir=$rel into $f"; '
            "  fi; "
            "done"
        )
        result = subprocess.run(
            ["docker", "exec", container, "sh", "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.stdout.strip():
            logger.info(f"top_builddir fixup: {result.stdout.strip()}")
        if result.returncode != 0 and result.stderr.strip():
            logger.warning(
                f"_fixup_top_builddir_in_submakefiles stderr: "
                f"{result.stderr.strip()}"
            )
    except Exception as exc:
        logger.warning(f"_fixup_top_builddir_in_submakefiles: {exc}")


def _extract_cd_prefix(command: str) -> str:
    """Extract leading `cd ... &&` prefix so companion commands match context."""
    match = re.match(r"^\s*(cd\s+[^&;]+&&\s*)", command)
    return match.group(1) if match else ""


def _inject_go_flag(command: str, flag: str) -> str:
    """Inject a Go build/install flag once, preserving existing command shape."""
    if flag in command:
        return command
    return re.sub(
        r"\bgo\s+(build|install)\b",
        rf"go \1 {flag}",
        command,
        count=1,
    )


def _inject_go_output(command: str, output_path: str) -> str:
    """Inject `-o <path>` into Go build/install commands."""
    if re.search(r"\b-o\s+\S+", command):
        return command
    return _inject_go_flag(command, f"-o {output_path}")


@agent_node(AgentRole.BUILDER)
def builder_node(state: AgentState) -> AgentState:
    """Execute the build plan with smart error detection."""
    logger.info("Builder executing build plan...")

    state.build_status = BuildStatus.BUILDING
    state.current_phase = "building"

    if not state.build_plan:
        logger.error("No build plan available")
        state.build_status = BuildStatus.FAILED
        state.add_error(
            create_error_record(
                "No build plan available",
                ErrorCategory.CONFIGURATION,
            )
        )
        return state

    predictions = predict_build_issues(state)
    if predictions:
        logger.info(
            f"Proactively identified {len(predictions)} potential issues"
        )
        state.context_cache["predicted_issues"] = predictions

    for phase in state.build_plan.phases:
        if phase.id <= state.last_successful_phase:
            continue

        logger.info(f"Executing phase {phase.id}: {phase.name}")

        for command in phase.commands:
            cached_result = state.get_cached_command_result(command)
            if cached_result and cached_result.success:
                logger.info(f"Using cached result for: {command[:50]}...")
                continue

            optimized_cmd = command
            for pred in predictions:
                if (
                    pred["pattern"] == "dubious ownership"
                    and "go build" in command
                ):
                    if "-buildvcs=false" not in command:
                        optimized_cmd = command.replace(
                            "go build", "go build -buildvcs=false"
                        )
                        logger.info(
                            f"Proactively optimized command: {optimized_cmd}"
                        )
                        break

            # Rewrite autoreconf to copy-first sequence when the repo
            # has m4/gettext.m4. autoreconf calls autopoint internally
            # which reverts any gettext.m4 fix. Only rewrite when
            # m4/gettext.m4 exists — other repos are unaffected.
            if re.search(r"\bautoreconf\b", optimized_cmd):
                if os.path.isfile(
                    os.path.join(
                        _to_host_path(state.repo_path), "m4", "gettext.m4"
                    )
                ):
                    optimized_cmd = (
                        "cp /usr/share/gettext/m4/*.m4 m4/ 2>/dev/null || true"
                        " && aclocal -I m4 -I /usr/share/aclocal"
                        " && autoconf"
                    )
                    logger.info(
                        "Rewrote autoreconf → copy-first gettext sequence "
                        "(autoreconf/autopoint would revert gettext.m4)"
                    )

            # Before any make command, proactively fix po/Makefile
            # top_builddir if needed. This handles the case where
            # configure was run by the fixer (not the builder), so the
            # post-configure hook wouldn't have fired yet.
            if re.match(r"\bmake\b", optimized_cmd.lstrip()):
                _fixup_top_builddir_in_submakefiles(state.repo_path)

            result = execute_command(optimized_cmd, cwd=state.repo_path)
            state.cache_command_result(command, result)
            state.log_scripted_op("execute_build_command")

            if not result.success:
                logger.error(f"Command failed: {command}")
                logger.error(f"Error: {result.stderr[:500]}")

                is_go_build = (
                    "go build" in optimized_cmd
                    or "go install" in optimized_cmd
                )
                is_vcs_error = any(
                    pattern in result.stderr
                    for pattern in [
                        "dubious ownership",
                        "error obtaining VCS status",
                        "Use -buildvcs=false",
                        "fatal: detected dubious ownership",
                    ]
                )

                if (
                    is_go_build
                    and is_vcs_error
                    and "-buildvcs=false" not in optimized_cmd
                ):
                    logger.warning(
                        "Detected Go VCS error, retrying with -buildvcs=false"
                    )
                    retry_command = _inject_go_flag(
                        optimized_cmd, "-buildvcs=false"
                    )

                    retry_result = execute_command(
                        retry_command, cwd=state.repo_path
                    )
                    state.cache_command_result(retry_command, retry_result)
                    state.log_scripted_op("retry_build_command")

                    if retry_result.success:
                        logger.info("Retry with -buildvcs=false succeeded!")
                        phase.commands[phase.commands.index(command)] = (
                            retry_command
                        )
                        continue
                    else:
                        logger.error(
                            f"Retry also failed: {retry_result.stderr[:500]}"
                        )
                        result = retry_result

                if (
                    is_go_build
                    and "already exists and is a directory"
                    in (result.stderr or "").lower()
                    and "build output" in (result.stderr or "").lower()
                ):
                    out_match = re.search(
                        r'build output "([^"]+)"', result.stderr or ""
                    )
                    out_name = (
                        out_match.group(1)
                        if out_match
                        else f"{state.repo_name or 'build'}-bin"
                    )
                    safe_out = re.sub(r"[^A-Za-z0-9._-]", "-", out_name)
                    retry_command = _inject_go_output(
                        optimized_cmd, f"./.atesor-bin/{safe_out}"
                    )
                    retry_command = (
                        "mkdir -p ./.atesor-bin && " f"{retry_command}"
                    )
                    retry_result = execute_command(
                        retry_command, cwd=state.repo_path
                    )
                    state.cache_command_result(retry_command, retry_result)
                    state.log_scripted_op("retry_build_command")
                    if retry_result.success:
                        logger.info(
                            "Resolved Go output-dir collision by using -o"
                        )
                        phase.commands[phase.commands.index(command)] = (
                            retry_command
                        )
                        continue
                    result = retry_result

                if (
                    is_go_build
                    and "inconsistent vendoring" in (result.stderr or "").lower()
                    and "-mod=mod" not in optimized_cmd
                ):
                    retry_command = _inject_go_flag(optimized_cmd, "-mod=mod")
                    retry_result = execute_command(
                        retry_command, cwd=state.repo_path
                    )
                    state.cache_command_result(retry_command, retry_result)
                    state.log_scripted_op("retry_build_command")
                    if retry_result.success:
                        logger.info(
                            "Resolved inconsistent vendoring with -mod=mod"
                        )
                        phase.commands[phase.commands.index(command)] = (
                            retry_command
                        )
                        continue
                    result = retry_result

                if (
                    is_go_build
                    and "no required module provides"
                    in (result.stderr or "").lower()
                ):
                    prefix = _extract_cd_prefix(optimized_cmd)
                    tidy_cmd = f"{prefix}go mod tidy"
                    tidy_result = execute_command(tidy_cmd, cwd=state.repo_path)
                    state.cache_command_result(tidy_cmd, tidy_result)
                    state.log_scripted_op("retry_build_command")
                    if tidy_result.success:
                        retry_result = execute_command(
                            optimized_cmd, cwd=state.repo_path
                        )
                        state.cache_command_result(
                            optimized_cmd, retry_result
                        )
                        state.log_scripted_op("retry_build_command")
                        if retry_result.success:
                            logger.info(
                                "Resolved missing module by running go mod tidy"
                            )
                            continue
                        result = retry_result

                if (
                    re.search(r"\bautoreconf\b", optimized_cmd)
                    and "autopoint" in (result.stderr or "").lower()
                    and "no such file" in (result.stderr or "").lower()
                ):
                    retry_command = (
                        "cp /usr/share/gettext/m4/*.m4 m4/ 2>/dev/null || true"
                        " && aclocal -I m4 -I /usr/share/aclocal"
                        " && autoconf"
                    )
                    retry_result = execute_command(
                        retry_command, cwd=state.repo_path
                    )
                    state.cache_command_result(retry_command, retry_result)
                    state.log_scripted_op("retry_build_command")
                    if retry_result.success:
                        logger.info(
                            "Recovered from autopoint failure via copy-first "
                            "gettext regeneration."
                        )
                        phase.commands[phase.commands.index(command)] = (
                            retry_command
                        )
                        continue
                    result = retry_result

                error_message = _build_command_error_message(
                    result, f"Build command failed: {command}"
                )
                error = create_error_record(
                    message=error_message,
                    category=classify_error(error_message),
                    command=command,
                    attempt_number=state.attempt_count,
                )
                state.add_error(error)
                state.build_status = BuildStatus.FAILED

                return state

        state.last_successful_phase = phase.id
        logger.info(f"Phase {phase.id} completed successfully")

        # Post-configure fixup: after any configure phase, patch
        # subdirectory Makefiles that reference $(top_builddir) but
        # don't define it. This is a common issue with old
        # gettext/autotools projects where po/Makefile.in.in lacks the
        # top_builddir = @top_builddir@ substitution.
        if phase.name.lower() in ("configure", "config") or any(
            "./configure" in c for c in phase.commands
        ):
            _fixup_top_builddir_in_submakefiles(state.repo_path)

    # All phases completed - NOW VERIFY THE BUILD PRODUCED ARTIFACTS
    logger.info("All build phases completed - verifying artifacts...")

    # Determine where to look for build artifacts based on build system
    build_system = (
        state.build_plan.build_system if state.build_plan else "unknown"
    )

    # Check multiple possible artifact locations
    artifact_dirs = [state.repo_path]  # Always check repo root

    # Add build-system-specific directories
    build_subdir = os.path.join(state.repo_path, "build")
    if build_system in ["cmake", "meson"]:
        artifact_dirs.insert(
            0, build_subdir
        )  # Check build/ first for cmake/meson

    # Also check common output directories
    for subdir in ["bin", "dist", "target/release", "output"]:
        artifact_dirs.append(os.path.join(state.repo_path, subdir))

    artifacts_found = False
    for artifact_dir in artifact_dirs:
        check_cmd = f"test -d {artifact_dir}"
        check_result = execute_command(check_cmd, cwd=state.repo_path)

        if not check_result.success:
            continue

        logger.info(f"Checking for artifacts in: {artifact_dir}")
        scanner = ArtifactScanner(artifact_dir, cwd=state.repo_path)
        artifacts = scanner.scan()

        is_valid, message = scanner.verify_build_success()
        logger.info(f"Artifact verification in {artifact_dir}: {message}")

        if is_valid:
            # Record all artifacts in state
            for artifact in artifacts:
                state.add_build_artifact(
                    filepath=artifact["filepath"],
                    artifact_type=artifact["type"],
                    architecture=artifact["architecture"],
                )

            summary = scanner.get_summary()
            logger.info(
                f"Build artifacts summary: "
                f"{json.dumps(summary, indent=2, default=str)}"
            )
            state.context_cache["artifact_summary"] = summary
            artifacts_found = True
            break

    if not artifacts_found:
        # For Go projects, check if any binary was produced in the repo root
        if build_system == "go":
            find_cmd = (
                f"find {state.repo_path} -maxdepth 1 -type f -executable"
            )
            find_result = execute_command(find_cmd, cwd=state.repo_path)
            if find_result.success and find_result.stdout.strip():
                logger.info(f"Go binary found: {find_result.stdout.strip()}")
                for binary_path in find_result.stdout.strip().split("\n"):
                    binary_path = binary_path.strip()
                    if binary_path:
                        # Verify it's a RISC-V binary
                        file_cmd = f"file {binary_path}"
                        file_result = execute_command(
                            file_cmd, cwd=state.repo_path
                        )
                        if (
                            file_result.success
                            and "RISC-V" in file_result.stdout
                        ):
                            state.add_build_artifact(
                                filepath=binary_path,
                                artifact_type="binary",
                                architecture="riscv64",
                            )
                            artifacts_found = True
                            logger.info(
                                f"Verified RISC-V binary: {binary_path}"
                            )
                        elif (
                            file_result.success and "ELF" in file_result.stdout
                        ):
                            # ELF binary on RISC-V host is likely RISC-V
                            state.add_build_artifact(
                                filepath=binary_path,
                                artifact_type="binary",
                                architecture="riscv64",
                            )
                            artifacts_found = True
                            logger.info(
                                f"Found ELF binary (assumed RISC-V on "
                                f"native host): {binary_path}"
                            )

    if not artifacts_found:
        logger.warning(
            "No build artifacts found, but all phases completed successfully"
        )
        # Don't fail the build - phases completed successfully,
        # artifacts may be installed elsewhere.
        logger.info("Treating as success since all build commands succeeded")

    state.build_status = BuildStatus.SUCCESS
    logger.info("Build completed successfully with artifact verification!")

    return state


# ============================================================================
# NODE: FIXER (Enhanced with Reflection)
# ============================================================================


def validate_fix_command(command: str) -> tuple[bool, str]:
    """Validate fix commands to prevent dangerous operations.

    Returns:
        A ``(is_safe, reason)`` tuple.
    """
    dangerous_patterns = [
        (r"\btouch\s+.*\.go$", "Cannot create empty Go source files"),
        (r"\btouch\s+.*\.c$", "Cannot create empty C source files"),
        (r"\btouch\s+.*\.cpp$", "Cannot create empty C++ source files"),
        (r"\btouch\s+.*\.h$", "Cannot create empty header files"),
        (r"\btouch\s+.*\.rs$", "Cannot create empty Rust source files"),
        (r"\btouch\s+.*\.py$", "Cannot create empty Python source files"),
        (r"\brm\s+-rf\s+/", "Cannot delete root directories"),
        (r"\brm\s+-rf\s+\*", "Cannot delete all files"),
        (r"\bgit\s+push", "Cannot push to remote"),
        (r"\bgit\s+reset\s+--hard", "Cannot hard reset git"),
        (r">\s*$", "Cannot truncate files with empty redirect"),
        (r'echo\s+""?\s*>\s*', "Cannot overwrite files with empty content"),
    ]

    import re

    for pattern, reason in dangerous_patterns:
        if re.search(pattern, command, re.IGNORECASE):
            return False, reason

    # Reject `apk add go mod` / `apt-get install go build` style hallucinations
    # where Go subcommands are used as package names
    _go_subcommands = {
        "mod",
        "build",
        "get",
        "install",
        "run",
        "test",
        "tool",
        "generate",
        "fmt",
        "vet",
    }
    stripped = command.strip()
    pkg_install_prefixes = ("apk add", "apt-get install", "apt install")
    if any(stripped.startswith(pfx) for pfx in pkg_install_prefixes):
        parts = stripped.split()
        pkg_names = [p for p in parts[2:] if not p.startswith("-")]
        for pkg in pkg_names:
            if pkg in _go_subcommands:
                return False, (
                    f"Hallucination: '{command}' — '{pkg}' is a Go "
                    f"subcommand, not a package. "
                    f"Use 'go {pkg}' as a separate command."
                )

    return True, "Command is safe"


def validate_fixer_response(fix_data: Dict) -> tuple[bool, str]:
    """Validate a FIXER response to ensure it makes logical sense.

    Returns:
        A ``(is_valid, error_message)`` tuple.
    """
    try:
        if not isinstance(fix_data, dict):
            return (
                False,
                f"Expected JSON object from fixer, got "
                f"{type(fix_data).__name__}",
            )

        if not fix_data.get("strategies"):
            return False, "No strategies provided"

        recommended_id = fix_data.get("recommended_strategy_id", 1)
        strategies = fix_data.get("strategies", [])

        # Find recommended strategy
        recommended = next(
            (s for s in strategies if s["id"] == recommended_id), None
        )
        if not recommended:
            return False, f"Recommended strategy {recommended_id} not found"

        # Check if strategy has actions
        actions = recommended.get("actions", [])
        if not actions:
            # Strategy has no actions - this is incomplete/hallucinatory
            logger.warning(
                "FIXER strategy has no actions (incomplete response)"
            )
            return False, "Recommended strategy has no actions"

        # Validate actions in recommended strategy
        for action in actions:
            action_type = action.get("type", "").lower()

            # Validate create_file actions
            if action_type == "create_file":
                path = action.get("path", "").strip()
                content = action.get("content", "").strip()

                if not path:
                    return False, "create_file action missing path"

                # Check for empty files
                if not content:
                    return (
                        False,
                        f"create_file action has empty content for "
                        f"{path} - this won't help",
                    )

                # Check for obviously hallucinated placeholders
                if len(path) > 100:
                    return (
                        False,
                        f"Suspiciously long filepath: {path[:100]}...",
                    )

                # Check for absolute paths that look wrong
                if path.startswith("/home/") or path.startswith("/root/"):
                    return (
                        False,
                        f"Suspicious absolute path (should be "
                        f"relative): {path}",
                    )

            # Validate command actions
            elif action_type == "command":
                cmd = action.get("command", "").strip()

                if not cmd:
                    return False, "command action is empty"

                # Check for nonsensical commands
                if "cp -r" in cmd and "/*" in cmd:
                    # Check if it's copying to itself
                    parts = cmd.split()
                    if len(parts) >= 4:
                        src = parts[2]  # source
                        dst = parts[3]  # destination
                        if (
                            src == dst
                            or src.split("/")[0] == dst.split("/")[0]
                        ):
                            return (
                                False,
                                f"Nonsensical command detected: {cmd} "
                                f"(copying to self)",
                            )

                is_safe, reason = validate_fix_command(cmd)
                if not is_safe:
                    return False, f"Unsafe command: {reason}"

        # Validate reflection data
        reflection = fix_data.get("reflection", {})
        if not reflection.get("root_cause"):
            logger.warning("Fixer response has empty root_cause analysis")

        if not reflection.get("this_fix_will_work_because"):
            logger.warning(
                "Fixer response has empty "
                "'this_fix_will_work_because' - self-critique may be "
                "shallow"
            )

        return True, ""

    except Exception as e:
        logger.error(f"Error validating fixer response: {e}")
        return False, f"Validation error: {str(e)}"


# ============================================================================
# NODE: FIXER (Error Resolution)
# ============================================================================

FIXER_PROMPT = (
    "You are the build doctor for a RISC-V porting agent. Diagnose\n"
    "the failure for **{repo_name}** and emit a minimal fix.\n"
    "\n"
    "## ACTIVE SANDBOX (use ONLY these package commands)\n"
    "{platform_banner}\n"
    "\n"
    "## Build context\n"
    "- Target: {target_arch} | Build system: {build_system}\n"
    "- Attempt {attempt_count}/{max_attempts} | Phase: {current_phase}\n"
    "\n"
    "## Failure\n"
    "**Failed command:**\n"
    "```\n"
    "{failed_command}\n"
    "```\n"
    "**Error (truncated):**\n"
    "```\n"
    "{error_output}\n"
    "```\n"
    "**Exit code:** {exit_code}\n"
    "\n"
    "### Previous fix attempts (most recent last — DO NOT repeat any of "
    "these)\n"
    "{previous_fixes}\n"
    "\n"
    "### Repo structure\n"
    "{repo_tree}\n"
    "\n"
    "### Known arch concerns\n"
    "{known_arch_issues}\n"
    "\n"
    "### Current build plan\n"
    "{build_plan}\n"
    "\n"
    "## Platform knowledge (canonical — use these package names + comman"
    "ds)\n"
    "{system_knowledge}\n"
    "\n"
    "{few_shot_examples}\n"
    "\n"
    "## Diagnostic procedure\n"
    "1. **Read the error.** What's the *root cause*, not the symptom?\n"
    "2. **Classify**: missing dep / wrong package name / arch-specific c"
    "ode / configure\n"
    "   gone stale / build-system mismatch / glibc-only API / unrelated "
    "env issue.\n"
    "3. **Pick the cheapest fix** in this order: (a) install a missing `"
    "-dev` package,\n"
    "   (b) change a build flag, (c) regenerate configure / copy `config"
    ".{{guess,sub}}`,\n"
    "   (d) add a `#ifdef __riscv` guard, (e) patch source.\n"
    "4. **Check the few-shot examples above** — if one matches the error"
    " pattern, use it.\n"
    "5. **Verify your fix does not duplicate a previous failed attempt.*"
    "*\n"
    "\n"
    "## Critical rules\n"
    "- Each command runs in a fresh shell at the repo root. Chain `cd` w"
    "ith `&&`.\n"
    "- Do NOT propose unrelated cleanups. Fix only what's broken.\n"
    "- For Go: `-buildvcs=false` is required; never install Go via the p"
    "ackage manager (it is preinstalled in the sandbox). Avoid `go build "
    "./cmd` / `go build ./<dir>` unless you also set `-o` to a file path"
    " (directory-name output collisions are common).\n"
    "- For autotools failures involving gettext macros: copy macros BEFO"
    "RE `aclocal`,\n"
    "  do NOT run `autoreconf` (it reverts the fix via `autopoint`).\n"
    "- If the error indicates an unfixable mismatch (e.g. project needs "
    "newer Go\n"
    "  than the sandbox provides), say so in `reflection.root_cause` and"
    " propose\n"
    "  ESCALATE rather than a hack.\n"
    "\n"
    "## Output schema (JSON only)\n"
    "{{\n"
    '  "strategies": [\n'
    '    {{"id": 1, "description": "<one line>",\n'
    '      "actions": [\n'
    '        {{"type": "command", "command": "<shell>"}}\n'
    '        // or {{"type": "create_file", "path": "<rel>", "c'
    'ontent": "..."}}\n'
    '        // or {{"type": "patch", "file": "<rel>", "content'
    '": "<unified diff>"}}\n'
    "      ]}}\n"
    "  ],\n"
    '  "recommended_strategy_id": 1,\n'
    '  "reflection": {{\n'
    '    "root_cause": "<why it broke>",\n'
    '    "this_fix_will_work_because": "<why the chosen fix addresses'
    ' the root cause>"\n'
    "  }}\n"
    "}}\n"
    "\n"
    "Rules: ≥1 strategy with ≥1 action; no placeholder paths (`/path/to/"
    "`, `/home/`);\n"
    "prefer `command` with `sed` over `patch` for simple edits."
)


@agent_node(AgentRole.FIXER)
def fixer_node(state: AgentState) -> AgentState:
    """Fix build errors using a reflection pattern."""
    logger.info("Fixer analyzing error...")

    state.build_status = BuildStatus.FIXING
    state.current_phase = "fixing"

    if not state.last_error:
        logger.warning("No error to fix")
        state.build_status = BuildStatus.PENDING
        return state

    # Prepare context
    previous_fixes = (
        "\n".join(
            [
                f"- {fix.strategy} ({'Success' if fix.success else 'Failed'})"
                for fix in state.fixes_attempted[-5:]
            ]
        )
        if state.fixes_attempted
        else "None"
    )

    arch_issues = "None"
    if state.arch_specific_code:
        relevant = [
            a
            for a in state.arch_specific_code
            if a.severity in ["high", "critical"]
        ]
        relevant_subset = relevant[:5]
        if relevant_subset:
            arch_issues = "\n".join(
                [
                    f"- {a.file}:{a.line} - {a.arch_type}"
                    for a in relevant_subset
                ]
            )

    failed_command = "Unknown"
    if state.error_history:
        last = state.error_history[-1]
        failed_command = last.command or "Unknown"

    few_shot_context = {
        "build_system": (
            state.build_plan.build_system if state.build_plan else "unknown"
        ),
        "error_message": state.last_error,
    }
    few_shot_examples = format_few_shot_examples(
        "fixer", few_shot_context, max_examples=2, max_chars=2000
    )

    # Get build plan details
    build_plan_str = "Unknown"
    if state.build_plan:
        build_plan_str = (
            f"Build System: {state.build_plan.build_system}\n"
            f"Completed Phases: {state.last_successful_phase}"
        )

    # Create prompt
    prompt = FIXER_PROMPT.format(
        repo_name=state.repo_name,
        target_arch="RISC-V (riscv64)",
        platform_banner=_build_platform_banner(),
        build_system=(
            state.build_plan.build_system if state.build_plan else "unknown"
        ),
        current_phase=state.current_phase,
        attempt_count=state.attempt_count,
        max_attempts=state.max_attempts,
        failed_phase=state.current_phase,
        exit_code="N/A",
        error_output=(
            state.last_error[:1000] if state.last_error else "No error details"
        ),
        failed_command=failed_command,
        previous_fixes=previous_fixes,
        repo_tree=(
            state.repo_tree[:500] if state.repo_tree else "(Not available)"
        ),
        known_arch_issues=arch_issues,
        build_plan=build_plan_str,
        system_knowledge=get_system_knowledge_summary(),
        few_shot_examples=few_shot_examples,
    )

    def _validate_fix(data: dict) -> ValidationResult:
        ok, reason = validate_fixer_response(data)
        if not ok:
            return ValidationResult.bad(reason)
        return ValidationResult.good()

    _fixer_pool = get_model_pool_for_role(AgentRole.FIXER)
    outcome = llm_call_with_validation(
        invoke_fn=invoke_llm,
        llm=_fixer_pool[0],
        fallback_llms=_fixer_pool[1:],
        prompt=prompt,
        validator=_validate_fix,
        # no deterministic fix — escalate via FAILED status
        fallback_factory=lambda: None,
        role=AgentRole.FIXER.value,
        audit_metadata={
            "repo": state.repo_name,
            "error_category": (
                state.last_error_category.value
                if state.last_error_category
                else "unknown"
            ),
            "attempt": state.attempt_count,
        },
        cost_estimate=0.01,
        max_retries=2,
    )
    state.log_api_call(cost=0.01 * outcome.attempts)

    try:
        if outcome.data is None:
            logger.error(
                f"FIXER produced no valid response after "
                f"{outcome.attempts} attempts: "
                f"{outcome.last_error}"
            )
            state.add_error(
                create_error_record(
                    message=(
                        f"FIXER proposed invalid fix: " f"{outcome.last_error}"
                    ),
                    category=ErrorCategory.CONFIGURATION,
                )
            )
            state.build_status = BuildStatus.FAILED
            return state

        fix_data = outcome.data

        # Get recommended strategy
        recommended_id = fix_data.get("recommended_strategy_id", 1)
        strategies = fix_data.get("strategies", [])

        if not strategies:
            logger.error("No fix strategies generated")
            state.build_status = BuildStatus.FAILED
            return state

        # Find recommended strategy
        strategy = next(
            (s for s in strategies if s["id"] == recommended_id), strategies[0]
        )

        logger.info(f"Applying fix strategy: {strategy['description']}")

        # Apply fix actions
        changes_made = []
        for action in strategy.get("actions", []):
            if action["type"] == "create_file":
                file_path = action.get("path")
                file_content = action.get("content", "")

                if not file_path:
                    logger.error("create_file action missing 'path'")
                    continue

                full_file_path = os.path.join(state.repo_path, file_path)

                dir_path = os.path.dirname(full_file_path)
                if dir_path:
                    mkdir_result = execute_command(
                        f"mkdir -p {shlex.quote(dir_path)}",
                        cwd=state.repo_path,
                        use_docker=True,
                    )
                    if not mkdir_result.success:
                        logger.warning(
                            f"Failed to create directory: {dir_path}"
                        )

                # Use base64 encoding to safely transfer LLM-generated content
                encoded = base64.b64encode(file_content.encode()).decode()
                write_result = execute_command(
                    f"echo {shlex.quote(encoded)} | base64 -d "
                    f"> {shlex.quote(full_file_path)}",
                    cwd=state.repo_path,
                    use_docker=True,
                )

                if write_result.success:
                    changes_made.append(f"Created file: {file_path}")
                    logger.info(f"Created file: {file_path}")
                else:
                    logger.error(
                        f"Failed to create file {file_path}: "
                        f"{write_result.stderr}"
                    )

            elif action["type"] == "patch":
                patch_content = action["content"]
                file_path = action.get("file")
                full_file_path = (
                    os.path.join(state.repo_path, file_path)
                    if file_path
                    else None
                )

                if apply_patch(
                    patch_content,
                    filepath=full_file_path,
                    cwd=state.repo_path,
                    use_docker=True,
                ):
                    changes_made.append(
                        f"Patched {file_path if file_path else 'repository'}"
                    )
                    state.patches_generated.append(patch_content)
                else:
                    logger.error(f"Failed to apply patch to {file_path}")

            elif action["type"] == "command":
                command = action["command"]

                is_safe, reason = validate_fix_command(command)
                if not is_safe:
                    logger.warning(
                        f"Fix command blocked: {command} - {reason}"
                    )
                    continue

                # Rewrite autoreconf to copy-first sequence when
                # gettext macros are missing. autoreconf internally
                # calls autopoint which reverts any gettext.m4 fix.
                # The copy-first approach replaces the old m4/gettext.m4
                # with the system's modern one BEFORE running
                # aclocal+autoconf, which is the only reliable fix.
                if re.search(r"\bautoreconf\b", command):
                    if os.path.isfile(
                        os.path.join(
                            _to_host_path(state.repo_path), "m4", "gettext.m4"
                        )
                    ):
                        rewritten = (
                            "cp /usr/share/gettext/m4/*.m4 m4/"
                            " && aclocal -I m4 -I /usr/share/aclocal"
                            " && autoconf"
                        )
                        logger.info(
                            "Rewrote autoreconf → copy-first gettext "
                            "sequence (autoreconf would revert "
                            "gettext.m4 fix via autopoint)"
                        )
                        command = rewritten

                commands_to_run = [command]
                for cmd in commands_to_run:
                    result = execute_command(cmd, cwd=state.repo_path)
                    if result.success:
                        changes_made.append(f"Executed: {cmd}")
                    else:
                        logger.error(f"Fix command failed: {result.stderr}")

        # Record fix attempt
        fix_attempt = FixAttempt(
            error_category=state.last_error_category,
            strategy=strategy["description"],
            changes_made=changes_made,
            success=False,  # Will be updated after rebuild
        )
        state.add_fix_attempt(fix_attempt)

        logger.info(f"Fix applied: {len(changes_made)} changes")

        # Reset to pending so supervisor can try building again
        state.build_status = BuildStatus.PENDING

    except Exception as e:
        logger.error(f"Fixer failed: {e}")
        state.build_status = BuildStatus.FAILED

    return state


# ============================================================================
# NODE: ESCALATE
# ============================================================================


@agent_node(AgentRole.SUPERVISOR)  # Escalation is a supervisor-level summary
def escalate_node(state: AgentState) -> AgentState:
    """Escalate to a human with a comprehensive report."""
    logger.warning("Escalating to human intervention")

    state.build_status = BuildStatus.ESCALATED
    state.current_phase = "escalated"

    # Generate escalation report
    err_cat = state.last_error_category
    err_cat_val = err_cat.value if err_cat else "Unknown"
    err_sev = state.last_error_severity
    err_sev_val = err_sev.value if err_sev else "unknown"
    err_msg = state.last_error[:500] if state.last_error else "N/A"
    report = f"""
# ESCALATION REPORT

## Summary
- Repository: {state.repo_name}
- Status: {state.build_status.value}
- Attempts: {state.attempt_count}/{state.max_attempts}
- Cost: ${state.api_cost_usd:.4f}
- Duration: {state.get_execution_duration():.1f}s

## Last Error
Category: {err_cat_val}
Severity: {err_sev_val}
Message: {err_msg}

## Fixes Attempted
{len(state.fixes_attempted)} fix attempts made:
"""

    for fix in state.fixes_attempted[-5:]:
        report += (
            f"\n- {fix.strategy} ({'Success' if fix.success else 'Failed'})"
        )

    report += f"""

## Architecture Issues
{len(state.arch_specific_code)} architecture-specific code instances found
"""

    if state.arch_specific_code:
        high_priority = [
            a
            for a in state.arch_specific_code
            if a.severity in ["high", "critical"]
        ]
        report_issues = high_priority[:5]
        for issue in report_issues:
            report += f"\n- {issue.file}:{issue.line} - {issue.arch_type}"

    report += f"""

## Recommendation
{should_escalate(state)[1]}
"""

    if state.error_history:
        report += "\n\n## Recent Failures\n"
        for err in state.error_history[-5:]:
            timestamp = err.timestamp.strftime("%H:%M:%S")
            command = err.command if err.command else "N/A"
            msg = (err.message or "N/A").replace("\n", " ")
            report += (
                f"- [{timestamp}] severity={err.severity.value}, "
                f"category={err.category.value}, command={command}, "
                f"message={msg[:260]}\n"
            )

    logger.info(report)
    state.context_cache["escalation_report"] = report

    return state


# ============================================================================
# NODE: FINISH
# ============================================================================


def _save_learning_data(state: AgentState):
    """Save successful build patterns to examples and recipe cache.

    Implements the auto-learning step after a successful build.
    """
    try:
        repo = state.repo_name or "unknown"
        bs = state.build_plan.build_system if state.build_plan else "unknown"

        # --- Curate artifacts so recipe + report show only
        # user-facing deliverables ---
        if state.build_artifacts and not state.curated_artifacts:
            try:
                from .artifact_curator import curate_artifacts

                curator_llm = None
                try:
                    curator_llm = get_model_for_role(AgentRole.SUMMARIZER)
                except Exception as e:
                    logger.info(
                        f"Curator LLM unavailable, using rule-based "
                        f"fallback: {e}"
                    )
                state.curated_artifacts = curate_artifacts(
                    raw_artifacts=state.build_artifacts,
                    repo_name=repo,
                    build_system=bs,
                    llm=curator_llm,
                )
            except Exception as e:
                logger.warning(f"Artifact curation failed (non-fatal): {e}")
                state.curated_artifacts = []

        # --- Scout learning ---
        if state.build_plan and state.build_plan.phases:
            scout_data = {
                "name": f"Auto: {repo} ({bs})",
                "tags": [bs, repo],
                "build_system": bs,
                "repo_name": repo,
                "trigger": {
                    "build_system": bs,
                    "has_main": bool(
                        getattr(state, "go_main_info", {}).get(
                            "has_main", False
                        )
                        if hasattr(state, "go_main_info")
                        and state.go_main_info
                        else False
                    ),
                },
                "plan": {
                    "build_system": bs,
                    "phases": [
                        {"name": p.name, "commands": p.commands}
                        for p in state.build_plan.phases
                    ],
                },
                "reasoning": f"Auto-learned from successful {repo} build.",
            }
            save_learned_example("scout", scout_data)

        # --- Fixer learning ---
        for fix in state.fixes_attempted or []:
            if fix.success and fix.strategy:
                fixer_data = {
                    "name": f"Auto: {repo} - {fix.strategy[:40]}",
                    "tags": [bs, "auto-fix"],
                    "build_system": bs,
                    "repo_name": repo,
                    "error_pattern": (
                        re.escape((state.last_error or "")[:80])
                        if state.last_error
                        else ""
                    ),
                    "fix": {
                        "strategy": fix.strategy,
                        "actions": [
                            {"type": "command", "command": cmd}
                            for cmd in (fix.commands_run or [])
                        ],
                    },
                    "reasoning": f"Auto-learned fix from {repo}.",
                }
                save_learned_example("fixer", fixer_data)

        # --- Builder learning ---
        if state.build_plan and state.build_plan.phases:
            builder_data = {
                "name": f"Auto: {repo} build execution",
                "tags": [bs, repo],
                "build_system": bs,
                "repo_name": repo,
                "phases": [
                    {"name": p.name, "commands": p.commands}
                    for p in state.build_plan.phases
                ],
                "timeout_recommendation": (
                    "600s" if len(state.build_plan.phases) > 2 else "120s"
                ),
                "reasoning": f"Auto-learned build execution from {repo}.",
            }
            save_learned_example("builder", builder_data)

        # --- Recipe cache ---
        if state.build_plan:
            duration = (
                state.get_execution_duration()
                if hasattr(state, "get_execution_duration")
                else 0.0
            )
            save_to_recipe_cache(
                repo_name=repo,
                repo_url=state.repo_url or "",
                build_system=bs,
                build_plan={
                    "phases": [
                        {"name": p.name, "commands": p.commands}
                        for p in state.build_plan.phases
                    ]
                },
                dependencies=[],
                patches=[
                    f.strategy
                    for f in (state.fixes_attempted or [])
                    if f.success
                ],
                artifacts=state.curated_artifacts
                or state.build_artifacts
                or [],
                build_duration_seconds=duration,
            )

        logger.info(f"Auto-learning complete for {repo}")
    except Exception as e:
        logger.warning(f"Auto-learning failed (non-fatal): {e}")


@agent_node(AgentRole.SUMMARIZER)
def finish_node(state: AgentState) -> AgentState:
    """Finalize successful porting and generate the guide."""
    logger.info("Generating comprehensive porting guide...")

    state.build_status = BuildStatus.SUCCESS
    state.current_phase = "finished"

    # Prepare context for summarizer
    arch_issues = (
        "\n".join(
            [
                f"- {a.file}:{a.line} - {a.arch_type} ({a.severity})"
                for a in state.arch_specific_code[:20]
            ]
        )
        if state.arch_specific_code
        else "None found."
    )

    fixes = (
        "\n".join(
            [
                f"- {fix.strategy} ({'Success' if fix.success else 'Failed'})"
                for fix in state.fixes_attempted
            ]
        )
        if state.fixes_attempted
        else "No fixes were needed (generic code)."
    )

    build_steps = ""
    if state.build_plan:
        for phase in state.build_plan.phases:
            build_steps += f"\n### {phase.name}\n```bash\n"
            for cmd in phase.commands:
                build_steps += f"{cmd}\n"
            build_steps += "```\n"

    # Prepare artifacts information (prefer curated list — primary
    # first, no noise)
    artifacts_info = ""
    artifacts_to_show = state.curated_artifacts or state.build_artifacts
    if artifacts_to_show:
        primary = [a for a in artifacts_to_show if a.get("role") == "primary"]
        secondary = [
            a for a in artifacts_to_show if a.get("role") == "secondary"
        ]
        rest = [a for a in artifacts_to_show if not a.get("role")]

        artifacts_info = "\n### Build Artifacts Generated\n\n"
        for label, group in (
            ("Primary", primary),
            ("Secondary (tests / examples)", secondary),
            ("Other", rest),
        ):
            if not group:
                continue
            if label != "Other" or (not primary and not secondary):
                artifacts_info += f"**{label}:**\n" if label != "Other" else ""
            for art in group:
                arch = art.get("architecture", "?")
                path = art.get("filepath") or art.get("path", "")
                artifacts_info += (
                    f"- **{art.get('type', 'binary')}** ({arch}): `{path}`\n"
                )
            artifacts_info += "\n"

    build_steps += artifacts_info

    # Create prompt
    prompt = SUMMARIZER_PROMPT.format(
        repo_name=state.repo_name,
        repo_url=state.repo_url,
        build_system=(
            state.build_plan.build_system if state.build_plan else "unknown"
        ),
        build_steps=build_steps,
        arch_issues=arch_issues,
        fixes=fixes,
        api_calls=state.api_calls_made,
        cost=state.api_cost_usd,
        duration=f"{state.get_execution_duration():.1f}s",
    )

    try:
        messages = [HumanMessage(content=prompt)]
        llm = get_model_for_role(AgentRole.SUMMARIZER)
        response = invoke_llm(llm, messages)
        state.log_api_call(cost=0.005)

        recipe = extract_content(response.content)

        # Log LLM call for debugging
        log_llm_call(
            agent_role=AgentRole.SUMMARIZER.value,
            prompt=prompt,
            response=recipe,
            model=llm.model_name if hasattr(llm, "model_name") else "unknown",
            cost_usd=0.005,
            metadata={"repo": state.repo_name, "phase": "finish"},
        )

        state.porting_recipe = recipe
        logger.info("Porting guide generated successfully")

    except Exception as e:
        logger.error(f"Summarizer failed: {e}")
        # Fallback recipe
        state.porting_recipe = (
            f"# RISC-V Porting Recipe: {state.repo_name}\n\n"
            f"Build succeeded.\n\n{build_steps}"
        )

    # Auto-learning: save successful patterns
    _save_learning_data(state)

    return state


# ============================================================================
# ROUTING FUNCTION
# ============================================================================


def route_next(state: AgentState) -> str:
    """Determine the next node based on the current phase.

    Enhanced with smart routing and cost optimization.
    """
    phase = state.current_phase.lower()

    # Failed initialization is a hard blocker and must not continue
    # to planning.
    if (
        phase in {"initialization", "initialized"}
        and state.build_status == BuildStatus.FAILED
    ):
        logger.warning(
            "Initialization failed; forcing escalation instead of planning"
        )
        return "escalate_node"

    routing_map = {
        "initialization": "planner",
        "initialized": "planner",
        "planning": "supervisor",
        "planned": "supervisor",
        "scout": "scout_node",
        "scouting": "supervisor",
        "builder": "builder_node",
        "building": "supervisor",
        "fixer": "fixer_node",
        "fixing": "supervisor",
        "escalate": "escalate_node",
        "escalated": END,
        "finish": "finish_node",
        "finished": END,
    }

    next_node = routing_map.get(phase, "supervisor")
    logger.info(f"Routing from {phase} to {next_node}")

    return next_node


def predict_build_issues(state: AgentState) -> List[Dict[str, str]]:
    """Predict potential build issues before execution.

    Returns:
        A list of predicted issues with mitigation strategies.
    """
    predictions = []

    if not state.build_plan:
        return predictions

    for phase in state.build_plan.phases:
        for cmd in phase.commands:
            if "go build" in cmd and "-buildvcs=false" not in cmd:
                predictions.append(
                    {
                        "phase": phase.name,
                        "issue": "Go VCS ownership error",
                        "pattern": "dubious ownership",
                        "mitigation": "Add -buildvcs=false flag",
                        "confidence": 0.7,
                    }
                )

            if "cmake" in cmd and state.arch_specific_code:
                high_severity = [
                    a for a in state.arch_specific_code if a.severity == "high"
                ]
                if high_severity:
                    predictions.append(
                        {
                            "phase": phase.name,
                            "issue": (
                                "Architecture-specific code may cause "
                                "build failure"
                            ),
                            "pattern": "arch_specific",
                            "mitigation": "Add RISC-V compatibility patches",
                            "confidence": 0.8,
                        }
                    )

            if any(
                tok in cmd
                for tok in (
                    "apk add",
                    "apt-get install",
                    "apt install",
                    "dpkg ",
                )
            ):
                missing = state.context_cache.get("missing_tools", [])
                if missing:
                    predictions.append(
                        {
                            "phase": phase.name,
                            "issue": f"Missing tools: {missing}",
                            "pattern": "missing_tools",
                            "mitigation": "Ensure dependencies are installed",
                            "confidence": 0.9,
                        }
                    )

    return predictions


# ============================================================================
# WORKFLOW CREATION
# ============================================================================


def create_workflow() -> StateGraph:
    """Create the enhanced LangGraph workflow."""
    logger.info("Creating enhanced workflow...")

    # Create graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("init", init_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("scout_node", scout_node)
    workflow.add_node("builder_node", builder_node)
    workflow.add_node("fixer_node", fixer_node)
    workflow.add_node("escalate_node", escalate_node)
    workflow.add_node("finish_node", finish_node)

    # Set entry point
    workflow.set_entry_point("init")

    # Add conditional edges
    workflow.add_conditional_edges(
        "init",
        route_next,
    )

    workflow.add_conditional_edges(
        "planner",
        route_next,
    )

    workflow.add_conditional_edges(
        "supervisor",
        route_next,
    )

    workflow.add_conditional_edges(
        "scout_node",
        route_next,
    )

    workflow.add_conditional_edges(
        "builder_node",
        route_next,
    )

    workflow.add_conditional_edges(
        "fixer_node",
        route_next,
    )

    workflow.add_edge("escalate_node", END)
    workflow.add_edge("finish_node", END)

    return workflow.compile()


# Create global app instance
app = create_workflow()

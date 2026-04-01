"""
Core orchestration logic defining the multi-agent state machine and workflow nodes.
Utilizes LangGraph to manage agent transitions and state.
"""

import json
import logging
import os
from typing import List, Dict
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

from .state import (
    AgentState,
    BuildStatus,
    ErrorCategory,
    AgentRole,
    classify_error,
    create_initial_state,
    should_escalate,
    get_next_action_recommendation,
    create_error_record,
    BuildPlan,
    BuildPhase,
    TaskPlan,
    TaskPhase,
    FixAttempt,
)
from .scripted_ops import ScriptedOperations, quick_analysis
from .models import create_llm
from .tools import execute_command, apply_patch
from .knowledge import get_system_knowledge_summary
from .memory import format_few_shot_examples

# Configure logging
logger = logging.getLogger(__name__)

# Initialize scripted operations
scripted_ops = ScriptedOperations()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def get_model_for_role(role: AgentRole):
    """Factory to get the right model for an agent role."""
    return create_llm(role)


def extract_content(content) -> str:
    """Safely extract string content from LLM response."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            [
                str(item.get("text", item)) if isinstance(item, dict) else str(item)
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


# ============================================================================
# NODE WRAPPER
# ============================================================================


def agent_node(role: AgentRole):
    """Decorator to wrap agent nodes with error handling and state tracking."""

    def decorator(func):
        def wrapper(state: AgentState) -> AgentState:
            state.current_agent = role
            logger.info(f"Node Started: {role.value}")
            try:
                result = func(state)
                logger.info(f"Node Completed: {role.value}")
                return result
            except Exception as e:
                logger.exception(f"Exception in {role.value} node: {e}")
                error_msg = f"Unexpected error in {role.value} agent: {str(e)}"
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

        return wrapper

    return decorator


# ============================================================================
# NODE: INITIALIZATION
# ============================================================================


@agent_node(AgentRole.SUPERVISOR)
def init_node(state: AgentState) -> AgentState:
    """
    Initialize the workflow by cloning the repository using scripted operations.
    This is a zero-cost operation (no LLM calls).
    """
    logger.info(f"Initializing workflow for {state.repo_url}")

    result = scripted_ops.clone_or_update_repository(state.repo_url, state.repo_name)
    state.log_scripted_op()

    if not result.success:
        error = create_error_record(
            message=result.stderr,
            category=ErrorCategory.NETWORK,
            command=result.command,
        )
        state.add_error(error)
        state.build_status = BuildStatus.FAILED
        return state

    logger.info("Performing quick analysis with scripted operations...")
    try:
        analysis = quick_analysis(state.repo_path)
        state.log_scripted_op()

        state.build_system_info = analysis.get("build_system")
        state.dependencies = analysis.get("dependencies")
        state.arch_specific_code = analysis.get("arch_specific_code", [])

        state.repo_tree = analysis.get("optimized_tree", "")
        logger.info(f"Repository tree captured ({len(state.repo_tree)} chars)")

        for doc_path in analysis.get("documentation", [])[:5]:
            content = scripted_ops.read_file(doc_path, max_lines=500)
            state.cache_file_content(doc_path, content)
            state.log_scripted_op()

        state.context_cache["quick_analysis"] = analysis

        if "go_main_info" in analysis:
            state.context_cache["go_main_info"] = analysis["go_main_info"]
            logger.info(f"Go main package detected: {analysis['go_main_info']}")

        if "arch_build_files" in analysis:
            state.context_cache["arch_build_files"] = analysis["arch_build_files"]
            arch_info = analysis["arch_build_files"]
            if arch_info.get("has_arch_specific"):
                logger.info(
                    f"Arch-specific build files detected: {arch_info.get('archs_found')}, "
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

        logger.info(
            f"Quick analysis complete: "
            f"Build system: {state.build_system_info.type if state.build_system_info else 'unknown'}, "
            f"Arch-specific code: {len(state.arch_specific_code)} instances. "
            f"Relevant missing tools: {', '.join(missing_tools) if missing_tools else 'None'}"
        )

    except Exception as e:
        logger.warning(f"Quick analysis failed: {e}")

    state.build_status = BuildStatus.PENDING
    state.current_phase = "initialized"

    return state


# ============================================================================
# NODE: PLANNER (New - Strategic Planning)
# ============================================================================

PLANNER_PROMPT = """You are the **Strategic Planner** for the RISC-V Porting Agent.

Your role is to decompose the porting task into phases and determine the optimal execution strategy.

# Current Context
Repository: {repo_name}
URL: {repo_url}

## Repository Structure
{repo_tree}

## Quick Analysis Results (from scripted operations):
- Build System: {build_system} (confidence: {build_system_confidence})
- Dependencies: {dependencies_summary}
- Architecture-Specific Code: {arch_code_count} instances found
- Documentation Available: {doc_count} files
- System Environment: {system_info}

{arch_concerns}

# Your Task
Create a strategic plan that:
1. Breaks down the porting task into logical phases
2. Identifies which tasks can be handled by scripts vs. agents
3. Determines dependencies between phases
4. Estimates complexity and potential blockers

# Output Format
Provide your plan in this JSON format:
```json
{{
  "phases": [
    {{
      "id": 1,
      "name": "dependency_setup",
      "description": "Install system dependencies",
      "agent": "scripted",
      "use_scripted_ops": true,
      "depends_on": [],
      "estimated_cost": 0.0
    }},
    {{
      "id": 2,
      "name": "build_planning",
      "description": "Create detailed build plan from documentation",
      "agent": "scout",
      "use_scripted_ops": false,
      "depends_on": [1],
      "estimated_cost": 0.01
    }},
    {{
      "id": 3,
      "name": "build_execution",
      "description": "Execute build commands",
      "agent": "builder",
      "use_scripted_ops": false,
      "depends_on": [2],
      "estimated_cost": 0.005
    }}
  ],
  "can_parallelize": [],
  "estimated_total_cost": 0.015,
  "complexity_score": 5,
  "estimated_total_time": "3-5 minutes",
  "potential_blockers": ["list any major concerns"]
}}
```

Repository path: {repo_path}
"""


@agent_node(AgentRole.PLANNER)
def planner_node(state: AgentState) -> AgentState:
    """
    Strategic planning phase - decomposes task and creates execution plan.
    This reduces downstream LLM calls by providing clear direction.
    """
    logger.info("Starting strategic planning phase...")

    state.build_status = BuildStatus.PLANNING
    state.current_phase = "planning"

    # Prepare context from quick analysis
    build_sys = state.build_system_info
    deps = state.dependencies

    dependencies_summary = "Unknown"
    if deps:
        dependencies_summary = (
            f"Build tools: {', '.join(deps.build_tools)}, "
            f"System packages: {len(deps.system_packages)}, "
            f"Libraries: {len(deps.libraries)}"
        )

    arch_concerns = ""
    if state.arch_specific_code:
        high_severity = [a for a in state.arch_specific_code if a.severity == "high"]
        if high_severity:
            report_severity = high_severity[:5]
            arch_concerns = f"\n## Architecture Concerns:\n"
            for concern in report_severity:
                arch_concerns += (
                    f"- {concern.arch_type} in {concern.file}:{concern.line}\n"
                )

    # System Environment summary
    system_info_raw = state.context_cache.get("system_info", {})
    system_info = "\n".join(
        [f"  - {tool}: {status}" for tool, status in system_info_raw.items()]
    )

    prompt = PLANNER_PROMPT.format(
        repo_name=state.repo_name,
        repo_url=state.repo_url,
        repo_tree=state.repo_tree if state.repo_tree else "(Not available)",
        build_system=build_sys.type if build_sys else "unknown",
        build_system_confidence=f"{build_sys.confidence:.2f}" if build_sys else "0.0",
        dependencies_summary=dependencies_summary,
        arch_code_count=len(state.arch_specific_code),
        doc_count=len(
            [
                k
                for k in state.file_content_cache.keys()
                if "README" in k or "INSTALL" in k
            ]
        ),
        arch_concerns=arch_concerns,
        repo_path=state.repo_path,
        system_info=system_info,
    )

    # Call LLM
    try:
        messages = [HumanMessage(content=prompt)]
        llm = get_model_for_role(AgentRole.PLANNER)
        response = llm.invoke(messages)
        state.log_api_call(cost=0.01)  # Estimate

        content = extract_content(response.content)

        # Parse JSON response
        json_match = extract_json_block(content)
        plan_data = json.loads(json_match)

        # Create TaskPlan
        phases = []
        for p in plan_data["phases"]:
            agent_str = p["agent"].lower()
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
                id=p["id"],
                name=p["name"],
                description=p["description"],
                agent=role,
                use_scripted_ops=p["use_scripted_ops"],
                depends_on=p.get("depends_on", []),
                estimated_cost=p.get("estimated_cost", 0.0),
            )
            phases.append(phase)

        state.task_plan = TaskPlan(
            phases=phases,
            can_parallelize=plan_data.get("can_parallelize", []),
            estimated_total_cost=plan_data.get("estimated_total_cost", 0.0),
            estimated_total_time=plan_data.get("estimated_total_time", "unknown"),
            complexity_score=plan_data.get("complexity_score", 5),
        )

        logger.info(
            f"Strategic plan created: {len(phases)} phases, "
            f"complexity: {state.task_plan.complexity_score}/10, "
            f"estimated cost: ${state.task_plan.estimated_total_cost:.3f}"
        )

        # Store plan in context
        state.context_cache["task_plan"] = plan_data

    except Exception as e:
        logger.error(f"Planning failed: {e}")
        # Fall back to default plan
        state.task_plan = create_default_plan()

    state.current_phase = "planned"
    return state


# ============================================================================
# NODE: SUMMARIZER (New - Professional Documentation)
# ============================================================================

SUMMARIZER_PROMPT = """You are the **RISC-V Documentation Agent**. Your task is to create a professional **RISC-V Porting Guide** for the project: {repo_name}.

# Repository
URL: {repo_url}
Build System: {build_system}

# Execution Metrics
- Duration: {duration}
- API Calls: {api_calls}
- Estimated Cost: ${cost:.4f}

# Porting Context
## Architecture Issues Found:
{arch_issues}

## Fixes Applied:
{fixes}

# Your Task
Write a comprehensive Markdown report with the following sections:
1. **Executive Summary**: High-level status and value.
2. **Environment Setup**: Necessary tools and dependencies.
3. **Build Instructions**: Step-by-step commands to build on RISC-V.
4. **Architecture Notes**: Detailed analysis of what was architecture-specific and how it was resolved.
5. **Validation**: How to verify the build.
6. **Future Recommendations**: What could be improved for better RISC-V support.

Use code blocks and clear headings. Be technical and precise.

# RISC-V Specific Instructions:
{build_steps}
"""


def create_default_plan() -> TaskPlan:
    """Create a default task plan if planning fails."""
    return TaskPlan(
        phases=[
            TaskPhase(
                1, "scout", "Analyze and create build plan", AgentRole.SCOUT, False, []
            ),
            TaskPhase(2, "build", "Execute build", AgentRole.BUILDER, False, [1]),
        ],
        estimated_total_cost=0.02,
        complexity_score=5,
    )


# ============================================================================
# NODE: SUPERVISOR
# ============================================================================

SUPERVISOR_PROMPT = """You are the **RISC-V Porting Architect**, the controller of a multi-agent system.

# Current State
- Repository: {repo_name}
- Build Status: {build_status}
- Attempt: {attempt_count} / {max_attempts}
- Current Phase: {current_phase}
- Last Error: {last_error_category}

# Task Plan Status
{task_plan_status}

# Quick Metrics
- API Calls Made: {api_calls}
- Scripted Ops: {scripted_ops}
- Cost: ${cost:.4f}

# Your Goal
Orchestrate the RISC-V porting process. You are the final authority on quality and correctness.

# Your Verification Duty:
Before recommending an action:
1. Check if the current agent provided nonsense (e.g., hallucinated compiler paths like '/path/to/riscv64-gcc').
2. If the BuildPlan has invalid absolute paths, route back to SCOUT with a critique.
3. If the Fixer suggested a patch that doesn't match the file structure, route back to FIXER.

# Available Actions
- SCOUT: Deep analysis and build plan creation
- BUILDER: Execute build commands
- FIXER: Debug and fix errors
- ESCALATE: Escalate to human
- FINISH: Mark as complete

# Decision Guidelines
{decision_context}

# Output Format
ACTION: [SCOUT|BUILDER|FIXER|ESCALATE|FINISH]
REASON: [Brief justification in one sentence]
"""


@agent_node(AgentRole.SUPERVISOR)
def supervisor_node(state: AgentState) -> AgentState:
    """
    Enhanced supervisor with better context awareness, cost optimization, and heuristic routing.
    """
    logger.info(
        f"Supervisor making routing decision... (BuildPlan: {'Exists' if state.build_plan else 'Missing'})"
    )

    should_esc, esc_reason = should_escalate(state)
    if should_esc:
        logger.warning(f"Automatic escalation: {esc_reason}")
        state.current_phase = "escalate"
        return state

    recommended_action = get_next_action_recommendation(state)
    state.log_scripted_op()

    if state.api_calls_made > 8 or state.api_cost_usd > 0.08:
        state.log_agent_decision(
            AgentRole.SUPERVISOR,
            recommended_action.value,
            "Cost optimization heuristic (API calls > 8 or cost > $0.08)",
        )
        logger.info(f"Using cost-optimized routing: {recommended_action.value}")
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
        state.log_agent_decision(AgentRole.SUPERVISOR, "PLAN", "No task plan exists.")
        state.current_phase = "planner"
        return state

    if not state.build_plan and state.task_plan:
        state.log_agent_decision(
            AgentRole.SUPERVISOR, "SCOUT", "Build plan missing but task plan exists."
        )
        state.current_phase = "scout"
        return state

    if state.build_status == BuildStatus.FAILED:
        if state.last_error_category == ErrorCategory.MISSING_TOOLS:
            state.log_agent_decision(
                AgentRole.SUPERVISOR, "SCOUT", "Missing tools - need better plan."
            )
            state.current_phase = "scout"
            return state
        else:
            state.log_agent_decision(
                AgentRole.SUPERVISOR,
                "FIXER",
                f"Error: {state.last_error_category.value if state.last_error_category else 'Unknown'}",
            )
            state.current_phase = "fixer"
            return state

    if state.build_status == BuildStatus.PENDING and state.build_plan:
        state.log_agent_decision(
            AgentRole.SUPERVISOR, "BUILDER", "Build plan ready, executing."
        )
        state.current_phase = "builder"
        return state

    task_plan_status = "No plan yet"
    if state.task_plan:
        completed = len([p for p in state.task_plan.phases if p.status == "completed"])
        total = len(state.task_plan.phases)
        task_plan_status = f"Progress: {completed}/{total} phases completed"

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

    prompt = SUPERVISOR_PROMPT.format(
        repo_name=state.repo_name,
        build_status=state.build_status.value,
        attempt_count=state.attempt_count,
        max_attempts=state.max_attempts,
        current_phase=state.current_phase,
        last_error_category=state.last_error_category.value
        if state.last_error_category
        else "None",
        task_plan_status=task_plan_status,
        api_calls=state.api_calls_made,
        scripted_ops=state.scripted_ops_count,
        cost=state.api_cost_usd,
        decision_context=decision_context,
    )

    try:
        messages = [HumanMessage(content=prompt)]
        llm = get_model_for_role(AgentRole.SUPERVISOR)
        response = llm.invoke(messages)
        state.log_api_call(cost=0.002)

        content = extract_content(response.content)

        action_line = [
            line for line in content.split("\n") if line.startswith("ACTION:")
        ][0]
        action_str = action_line.split(":")[1].strip().upper()

        action_map = {
            "SCOUT": "scout",
            "BUILDER": "builder",
            "FIXER": "fixer",
            "ESCALATE": "escalate",
            "FINISH": "finish",
        }

        decision_reason = f"LLM Routing: {action_str}"
        state.log_agent_decision(
            AgentRole.SUPERVISOR, action_str, content.split("\n")[-1]
        )
        state.current_phase = action_map.get(action_str, "scout")

        logger.info(f"Supervisor decision: {action_str}")

    except Exception as e:
        logger.error(f"Supervisor failed: {e}, using fallback")
        state.current_phase = recommended_action.value.lower()

    return state


# ============================================================================
# NODE: SCOUT
# ============================================================================

SCOUT_PROMPT = """You are the **Scout Agent**, expert at analyzing software projects for RISC-V porting.

# Repository Structure
{repo_tree}

# Quick Analysis Already Done (via scripts):
- Build System: {build_system}
- Module Directory: {module_dir}
- Dependencies Found: {deps_count}
- Arch-Specific Code: {arch_code_count} instances
- Documentation: {doc_count} files cached
- Build System Metadata: {go_main_info}

{few_shot_examples}

# Available Documentation:
{documentation}

# Architecture Concerns:
{arch_concerns}

# Dependencies:
{dependencies}

# System Environment (Actual tools available)
{system_info}

# System Knowledge (Feb 2026)
{system_knowledge}

# Your Task
Analyze the project and create a **Build Plan** tailored to the detected build system.

## CRITICAL: Deep Build System Analysis Required

You must perform a thorough analysis BEFORE creating build commands:

### Step 1: Detect Build System Type
- Look for CMakeLists.txt, Makefile, Cargo.toml, go.mod, configure.ac, meson.build, etc.
- Check for architecture-specific build files (e.g., `cmpl_gcc_x64.mak`, `cmpl_gcc_arm64.mak`)

### Step 2: Analyze Existing Build Patterns
**IMPORTANT**: Many projects have architecture-specific build files. Look for:
- Files named with arch suffixes: `*_x64.mak`, `*_arm64.mak`, `*_x86.mk`, `config.arm64`
- Build directories organized by architecture: `b/g_x64`, `b/g_arm64`
- Configuration files for different platforms

### Step 3: Determine RISC-V Support Status
- **If RISC-V files exist**: Use them directly (e.g., `cmpl_gcc_riscv64.mak`)
- **If only other arch files exist**: You must CREATE RISC-V equivalents based on patterns
- **If generic build exists**: Use standard commands

### Step 4: Build File Creation (When Needed)
If the project has architecture-specific build files but NO RISC-V support:
1. Identify the template file (usually x86_64 or arm64 variant)
2. Note what changes are needed for RISC-V (arch name, disable ASM, etc.)
3. Include file creation commands in your build plan

## Build System Specific Guidelines

### Custom Makefiles (like 7zip pattern)
Many projects use custom makefile patterns:
```
CPP/7zip/cmpl_gcc_x64.mak    -> Create cmpl_gcc_riscv64.mak
CPP/7zip/var_gcc_x64.mak     -> Create var_gcc_riscv64.mak
```
Build command: `make -C CPP/7zip/Bundles/Alone -f ../../cmpl_gcc_riscv64.mak`

### Standard Build Systems
- **Go**: Use build_system_metadata. Add `-buildvcs=false` flag.
- **CMake**: Create build/, run cmake, then make
- **Autotools**: Run ./configure, then make
- **Cargo**: Use `cargo build --release`
- **Meson**: Setup build dir with meson, compile with ninja

## Output Format
Provide a JSON object with:
```json
{{
  "build_system": "<detected_build_system>",
  "build_system_confidence": <0.0-1.0>,
  "module_dir": "<subdirectory_if_any>",
  "has_arch_specific_build": <true/false>,
  "riscv_support_exists": <true/false>,
  "files_to_create": [
    {{
      "source": "<template_file>",
      "target": "<riscv_file>",
      "content_template": "<key replacements>"
    }}
  ],
  "phases": [
    {{
      "id": 1,
      "name": "install_dependencies",
      "commands": ["<install_commands>"],
      "can_parallelize": false,
      "expected_duration": "<estimate>"
    }},
    {{
      "id": 2,
      "name": "create_build_files",
      "commands": ["<file_creation_commands>"],
      "can_parallelize": false,
      "expected_duration": "<estimate>"
    }},
    {{
      "id": 3,
      "name": "build",
      "commands": ["<build_commands>"],
      "can_parallelize": false,
      "expected_duration": "<estimate>"
    }}
  ],
  "total_estimated_duration": "<estimate>",
  "notes": ["<important_observations>"]
}}
```

## Target Architecture
Use the system architecture: {architecture}.

Repository: {repo_path}
"""


def validate_build_plan(plan: BuildPlan) -> tuple[bool, str]:
    """Validate BuildPlan for hallucinations and common issues."""
    hallucination_patterns = [
        "/path/to/",
        "your_username",
        "example.com",
        "riscv64-unknown-linux-gnu-gcc",  # Unless we confirmed it's there
    ]

    for phase in plan.phases:
        for cmd in phase.commands:
            for pattern in hallucination_patterns:
                if pattern in cmd:
                    return (
                        False,
                        f"Hallucination detected in command: '{cmd}' (contains '{pattern}')",
                    )

            # Check for absolute paths that look guessed
            if "/home/" in cmd and "/workspace/" not in cmd:
                return False, f"Suspicious absolute path in command: '{cmd}'"

    return True, ""


@agent_node(AgentRole.SCOUT)
def scout_node(state: AgentState) -> AgentState:
    """
    Enhanced scout that leverages pre-analysis from scripted operations.
    """
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
    system_info = "\n".join([f"- {k}: {v}" for k, v in system_info_raw.items()])

    module_dir = (
        build_sys.module_dir if build_sys and hasattr(build_sys, "module_dir") else ""
    )

    go_main_info = state.context_cache.get("go_main_info", {})
    go_main_str = (
        json.dumps(go_main_info, indent=2) if go_main_info else "Not a Go project"
    )

    few_shot_context = {
        "build_system": build_sys.type if build_sys else "unknown",
        "has_main": go_main_info.get("has_main", False),
        "main_path": go_main_info.get("main_path", ""),
        "module_dir": module_dir,
    }
    few_shot_examples = format_few_shot_examples(
        "scout", few_shot_context, max_examples=2, max_chars=2000
    )

    prompt = SCOUT_PROMPT.format(
        repo_tree=state.repo_tree.replace("{", "{{").replace("}", "}}")
        if state.repo_tree
        else "(Not available)",
        build_system=(build_sys.type if build_sys else "unknown")
        .replace("{", "{{")
        .replace("}", "}}"),
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

    try:
        messages = [HumanMessage(content=prompt)]
        llm = get_model_for_role(AgentRole.SCOUT)
        response = llm.invoke(messages)
        state.log_api_call(cost=0.01)

        content = extract_content(response.content)

        json_match = extract_json_block(content)
        plan_data = json.loads(json_match)

        phases = []
        for i, p in enumerate(plan_data["phases"]):
            phase = BuildPhase(
                id=p.get("id", i + 1),
                name=p["name"],
                commands=p["commands"],
                can_parallelize=p.get("can_parallelize", False),
                expected_duration=p.get("expected_duration", "unknown"),
            )
            phases.append(phase)

        state.build_plan = BuildPlan(
            build_system=plan_data.get(
                "build_system", build_sys.type if build_sys else "unknown"
            ),
            build_system_confidence=plan_data.get(
                "build_system_confidence", build_sys.confidence if build_sys else 0.5
            ),
            phases=phases,
            total_estimated_duration=plan_data.get(
                "total_estimated_duration", "unknown"
            ),
            notes=plan_data.get("notes", []),
        )
        state.last_successful_phase = 0

        is_valid, reason = validate_build_plan(state.build_plan)
        if not is_valid:
            logger.warning(f"Plan validation failed: {reason}")
            state.add_error(
                create_error_record(
                    message=f"Invalid build plan: {reason}. Please avoid absolute paths and placeholders.",
                    category=ErrorCategory.CONFIGURATION,
                )
            )
            state.build_plan = None

            if state.attempt_count >= 2:
                logger.info("Using fallback build plan after repeated failures")
                state.build_plan = create_fallback_build_plan(state)
        else:
            logger.info(f"Build plan created and validated: {len(phases)} phases")

    except Exception as e:
        logger.error(f"Scout failed: {e}")
        state.build_plan = create_fallback_build_plan(state)

    state.build_status = BuildStatus.PENDING
    return state


def create_fallback_build_plan(state: AgentState) -> BuildPlan:
    """Create a basic fallback build plan."""
    build_sys = state.build_system_info

    if build_sys and build_sys.type == "cmake":
        commands = [
            "cmake -B build .",
            "cmake --build build",
        ]
    elif build_sys and build_sys.type == "make":
        commands = ["make"]
    else:
        commands = ["./configure", "make"]

    return BuildPlan(
        build_system=build_sys.type if build_sys else "unknown",
        build_system_confidence=0.5,
        phases=[BuildPhase(1, "build", commands, False, "unknown")],
        total_estimated_duration="unknown",
    )


# Continued in next file...
"""
Enhanced LangGraph State Machine - Part 2
Builder, Fixer, and Workflow Assembly
"""

# Continued from part 1...

# ============================================================================
# NODE: BUILDER
# ============================================================================

BUILDER_PROMPT = """You are the **Builder Agent**, executing build commands in the RISC-V environment.

# Build Plan
{build_plan}

# Current State
- Attempt: {attempt_count}
- Last successful phase: {last_successful_phase}

# Your Task
Execute the build plan step by step. For each phase:
1. Run the commands
2. Check for errors
3. Report status

Use the execute_command tool to run commands.

# Important
- Stop at first error and report it clearly
- Capture full error messages
- Note which phase failed

Repository: {repo_path}
"""


@agent_node(AgentRole.BUILDER)
def builder_node(state: AgentState) -> AgentState:
    """
    Execute the build plan with smart error detection and proactive issue prediction.
    """
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
        logger.info(f"Proactively identified {len(predictions)} potential issues")
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
                if pred["pattern"] == "dubious ownership" and "go build" in command:
                    if "-buildvcs=false" not in command:
                        optimized_cmd = command.replace(
                            "go build", "go build -buildvcs=false"
                        )
                        logger.info(f"Proactively optimized command: {optimized_cmd}")
                        break

            result = execute_command(optimized_cmd, cwd=state.repo_path)
            state.cache_command_result(command, result)
            state.log_scripted_op()

            if not result.success:
                logger.error(f"Command failed: {command}")
                logger.error(f"Error: {result.stderr[:500]}")

                is_go_build = "go build" in command or "go install" in command
                is_vcs_error = any(
                    pattern in result.stderr
                    for pattern in [
                        "dubious ownership",
                        "error obtaining VCS status",
                        "Use -buildvcs=false",
                        "fatal: detected dubious ownership",
                    ]
                )

                if is_go_build and is_vcs_error and "-buildvcs=false" not in command:
                    logger.warning(
                        "Detected Go VCS error, retrying with -buildvcs=false"
                    )
                    retry_command = command.replace(
                        "go build", "go build -buildvcs=false"
                    )
                    retry_command = retry_command.replace(
                        "go install", "go install -buildvcs=false"
                    )

                    retry_result = execute_command(retry_command, cwd=state.repo_path)
                    state.cache_command_result(retry_command, retry_result)
                    state.log_scripted_op()

                    if retry_result.success:
                        logger.info("Retry with -buildvcs=false succeeded!")
                        phase.commands[phase.commands.index(command)] = retry_command
                        continue
                    else:
                        logger.error(f"Retry also failed: {retry_result.stderr[:500]}")
                        result = retry_result

                error = create_error_record(
                    message=result.stderr,
                    category=classify_error(result.stderr),
                    command=command,
                    attempt_number=state.attempt_count,
                )
                state.add_error(error)
                state.build_status = BuildStatus.FAILED

                return state

        state.last_successful_phase = phase.id
        logger.info(f"Phase {phase.id} completed successfully")

    state.build_status = BuildStatus.SUCCESS
    logger.info("Build completed successfully!")

    return state


# ============================================================================
# NODE: FIXER (Enhanced with Reflection)
# ============================================================================


def validate_fix_command(command: str) -> tuple[bool, str]:
    """
    Validate fix commands to prevent dangerous operations.
    Returns (is_safe, reason).
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

    return True, "Command is safe"


FIXER_PROMPT = """You are the **Fixer Agent**, expert at debugging RISC-V porting issues with REFLECTION capabilities.

# Error Information
Category: {error_category}
Message: {error_message}
Failed Command: {failed_command}

# Context
- Build System: {build_system}
- Attempt: {attempt_count}
- Previous Fixes Tried: {previous_fixes}

# Architecture-Specific Issues Found
{arch_issues}

# System Knowledge (Feb 2026)
{system_knowledge}

{few_shot_examples}

# Your Task - REFLECTION PATTERN
You MUST follow this 4-step reflection process:
1. ANALYZE: Deep dive into the root cause - is this a missing file, wrong command, or configuration issue?
2. HYPOTHESIZE: Generate 2-3 potential fixes with pros/cons
3. SELF-CRITIQUE: For each fix, ask "Will this actually work? What are the risks?"
4. SELECT: Choose the minimal, safest fix

## CRITICAL: Understanding the Real Problem

### Missing Build Files
If the error is "No such file or directory" for a build file:
- The project likely has architecture-specific build files
- You need to CREATE RISC-V equivalents based on existing patterns
- Look for files like `cmpl_gcc_x64.mak`, `cmpl_gcc_arm64.mak`
- Create `cmpl_gcc_riscv64.mak` based on the template

### Pattern for Creating Build Files
1. Find existing arch-specific file (x64 or arm64 variant)
2. Read its content
3. Replace architecture identifiers:
   - `x64` → `riscv64`
   - `x86_64` → `riscv64`
   - `arm64` → `riscv64`
   - `X64` → `RISCV64`
4. Set `USE_ASM = ` (empty, disable assembly for RISC-V)
5. Write the new file

### Common File Creation Patterns
```json
{{
  "type": "create_file",
  "path": "CPP/7zip/cmpl_gcc_riscv64.mak",
  "content": "include $(CURDIR)/../../var_gcc_riscv64.mak\\ninclude $(CURDIR)/../../warn_gcc.mak\\ninclude $(CURDIR)/makefile.gcc"
}}
```

## Fix Strategy Guidelines

### By Error Category
- **MISSING_TOOLS**: Install the missing tool using apk add
- **DEPENDENCY**: Install required libraries and dev packages
- **FILE_NOT_FOUND**: Create the missing file based on existing patterns
- **COMPILATION**: Fix syntax errors, type mismatches, missing includes
- **LINKING**: Install missing libraries, fix library paths
- **ARCHITECTURE**: Add RISC-V conditional compilation or scalar fallbacks
- **CONFIGURATION**: Fix build system configuration, flags, paths

### By Build System
- **Custom Makefiles**: Look for arch patterns, create RISC-V equivalents
- **Go**: Check CGO requirements, module paths, use -buildvcs=false
- **C/C++**: Check for missing headers, ASM code that needs porting
- **CMake**: Check for missing dependencies, configuration errors

# CRITICAL: Forbidden Actions
- NEVER create empty source files (touch *.go, *.c, etc.)
- NEVER delete files unless absolutely necessary
- NEVER create directories to "fix" missing paths - analyze the pattern instead

# Output Format
```json
{{
  "analysis": "Deep analysis of root cause - is this a missing file, wrong command, or config issue?",
  "reflection": {{
    "previous_attempts_failed_because": "Why previous fixes didn't work",
    "this_time_will_be_different_because": "Why this approach will succeed"
  }},
  "strategies": [
    {{
      "id": 1,
      "description": "Strategy description",
      "confidence": 0.8,
      "risks": ["potential issue 1"],
      "self_critique": "This might fail if X happens",
      "actions": [
        {{"type": "create_file", "path": "path/to/file", "content": "file content"}},
        {{"type": "patch", "file": "src/file.c", "content": "patch content"}},
        {{"type": "command", "command": "apk add libfoo"}}
      ]
    }}
  ],
  "recommended_strategy_id": 1,
  "fallback_if_fails": "What to try if this doesn't work"
}}
```

Repository: {repo_path}
"""


@agent_node(AgentRole.FIXER)
def fixer_node(state: AgentState) -> AgentState:
    """
    Intelligent error fixing with reflection pattern.
    """
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
            a for a in state.arch_specific_code if a.severity in ["high", "critical"]
        ]
        relevant_subset = relevant[:5]
        if relevant_subset:
            arch_issues = "\n".join(
                [f"- {a.file}:{a.line} - {a.arch_type}" for a in relevant_subset]
            )

    failed_command = "Unknown"
    if state.error_history:
        last = state.error_history[-1]
        failed_command = last.command or "Unknown"

    few_shot_context = {
        "build_system": state.build_plan.build_system
        if state.build_plan
        else "unknown",
        "error_message": state.last_error,
    }
    few_shot_examples = format_few_shot_examples(
        "fixer", few_shot_context, max_examples=2, max_chars=2000
    )

    # Create prompt
    prompt = FIXER_PROMPT.format(
        error_category=state.last_error_category.value
        if state.last_error_category
        else "Unknown",
        error_message=state.last_error[:1000],
        failed_command=failed_command,
        build_system=state.build_plan.build_system if state.build_plan else "unknown",
        attempt_count=state.attempt_count,
        previous_fixes=previous_fixes,
        arch_issues=arch_issues,
        repo_path=state.repo_path,
        system_knowledge=get_system_knowledge_summary(),
        few_shot_examples=few_shot_examples,
    )

    try:
        messages = [HumanMessage(content=prompt)]
        llm = get_model_for_role(AgentRole.FIXER)
        response = llm.invoke(messages)
        state.log_api_call(cost=0.01)

        content = extract_content(response.content)

        # Parse JSON
        json_match = extract_json_block(content)
        fix_data = json.loads(json_match)

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
                        f"mkdir -p {dir_path}", cwd=state.repo_path, use_docker=True
                    )
                    if not mkdir_result.success:
                        logger.warning(f"Failed to create directory: {dir_path}")

                write_result = execute_command(
                    f"cat > {full_file_path} << 'ATESOR_EOF'\n{file_content}\nATESOR_EOF",
                    cwd=state.repo_path,
                    use_docker=True,
                )

                if write_result.success:
                    changes_made.append(f"Created file: {file_path}")
                    logger.info(f"Created file: {file_path}")
                else:
                    logger.error(
                        f"Failed to create file {file_path}: {write_result.stderr}"
                    )

            elif action["type"] == "patch":
                patch_content = action["content"]
                file_path = action.get("file")
                full_file_path = (
                    os.path.join(state.repo_path, file_path) if file_path else None
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
                    logger.warning(f"Fix command blocked: {command} - {reason}")
                    continue

                result = execute_command(command, cwd=state.repo_path)
                if result.success:
                    changes_made.append(f"Executed: {command}")
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
    """
    Escalate to human with comprehensive report.
    """
    logger.warning("Escalating to human intervention")

    state.build_status = BuildStatus.ESCALATED
    state.current_phase = "escalated"

    # Generate escalation report
    report = f"""
# ESCALATION REPORT

## Summary
- Repository: {state.repo_name}
- Status: {state.build_status.value}
- Attempts: {state.attempt_count}/{state.max_attempts}
- Cost: ${state.api_cost_usd:.4f}
- Duration: {state.get_execution_duration():.1f}s

## Last Error
Category: {state.last_error_category.value if state.last_error_category else "Unknown"}
Message: {state.last_error[:500] if state.last_error else "N/A"}

## Fixes Attempted
{len(state.fixes_attempted)} fix attempts made:
"""

    for fix in state.fixes_attempted[-5:]:
        report += f"\n- {fix.strategy} ({'Success' if fix.success else 'Failed'})"

    report += f"""

## Architecture Issues
{len(state.arch_specific_code)} architecture-specific code instances found
"""

    if state.arch_specific_code:
        high_priority = [
            a for a in state.arch_specific_code if a.severity in ["high", "critical"]
        ]
        report_issues = high_priority[:5]
        for issue in report_issues:
            report += f"\n- {issue.file}:{issue.line} - {issue.arch_type}"

    report += f"""

## Recommendation
{should_escalate(state)[1]}
"""

    logger.info(report)
    state.context_cache["escalation_report"] = report

    return state


# ============================================================================
# NODE: FINISH
# ============================================================================


@agent_node(AgentRole.SUMMARIZER)
def finish_node(state: AgentState) -> AgentState:
    """
    Finalize successful porting with comprehensive guide generation.
    """
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

    # Create prompt
    prompt = SUMMARIZER_PROMPT.format(
        repo_name=state.repo_name,
        repo_url=state.repo_url,
        build_system=state.build_plan.build_system if state.build_plan else "unknown",
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
        response = llm.invoke(messages)
        state.log_api_call(cost=0.005)

        recipe = extract_content(response.content)
        state.porting_recipe = recipe
        logger.info("Porting guide generated successfully")

    except Exception as e:
        logger.error(f"Summarizer failed: {e}")
        # Fallback recipe
        state.porting_recipe = f"# RISC-V Porting Recipe: {state.repo_name}\n\nBuild succeeded.\n\n{build_steps}"

    return state


# ============================================================================
# ROUTING FUNCTION
# ============================================================================


def route_next(state: AgentState) -> str:
    """
    Determine next node based on current phase.
    Enhanced with smart routing and cost optimization.
    """
    phase = state.current_phase.lower()

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
    """
    Proactively predict potential build issues before execution.
    Returns a list of predicted issues with mitigation strategies.
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
                            "issue": "Architecture-specific code may cause build failure",
                            "pattern": "arch_specific",
                            "mitigation": "Add RISC-V compatibility patches",
                            "confidence": 0.8,
                        }
                    )

            if "apk add" in cmd or "apt-get" in cmd:
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
    """
    Create the enhanced LangGraph workflow.
    """
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

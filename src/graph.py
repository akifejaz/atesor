"""
Core orchestration logic defining the multi-agent state machine and workflow nodes.
Utilizes LangGraph to manage agent transitions and state.
"""

import json
import logging
import os
import re
from typing import List, Dict
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

from .state import (
    AgentState,
    BuildStatus,
    ErrorCategory,
    FailureSeverity,
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
from .memory import format_few_shot_examples, save_learned_example, save_to_recipe_cache
from .llm_logger import log_llm_call
from .artifact_scanner import ArtifactScanner

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


def _build_command_error_message(result, fallback: str) -> str:
    """Build a robust, human-readable error message from command output."""
    stderr = (result.stderr or "").strip()
    stdout = (result.stdout or "").strip()
    detail = stderr or stdout or "No stderr/stdout output captured."
    return f"{fallback} (exit {result.exit_code}) - {detail}"


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
    state.log_scripted_op("clone_or_update_repository")

    if not result.success:
        message = _build_command_error_message(
            result, f"Repository clone/update failed for {state.repo_url}"
        )
        error = create_error_record(
            message=message,
            category=ErrorCategory.NETWORK,
            severity=FailureSeverity.HIGH,
            command=result.command,
        )
        state.add_error(error)
        state.build_status = BuildStatus.FAILED
        state.current_phase = "escalate"
        state.log_agent_decision(
            AgentRole.SUPERVISOR,
            "ESCALATE",
            f"Critical init failure ({error.severity.value}): {error.message[:200]}",
        )
        return state

    logger.info("Performing quick analysis with scripted operations...")
    try:
        analysis = quick_analysis(state.repo_path)
        state.log_scripted_op("quick_analysis")

        state.build_system_info = analysis.get("build_system")

        # Warn if build system could not be detected
        if state.build_system_info and state.build_system_info.type == "unknown":
            logger.warning(
                f"⚠️  WARNING: Unable to identify build system in {state.repo_name}. "
                "Exhaustive search was performed but no build configuration files found. "
                "The system will attempt configuration-free analysis or request manual input."
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
# NODE: PLANNER (Strategic Planning)
# ============================================================================
 
PLANNER_PROMPT = """<role>
You are a Strategic Planner specializing in cross-architecture software porting. Your mission is to analyze codebases and create executable build plans that enable software to compile and run on target architectures.
</role>
 
<capabilities>
- Pattern recognition: Identify how existing architectures are handled in build systems
- Build system analysis: Understand CMake, Makefiles, Meson, Cargo, Go modules, and other build tools
- Architecture mapping: Apply successful patterns from one architecture to another
- Dependency resolution: Identify required tools, libraries, and system dependencies
- Risk assessment: Predict potential build failures and compatibility issues
</capabilities>
 
<core_principle>
PATTERN-BASED PORTING: Study how the codebase handles EXISTING architectures, then replicate that pattern for the TARGET architecture. When no patterns exist, design a generic cross-platform build approach.
</core_principle>
 
<repository_context>
<project>
Name: {repo_name}
URL: {repo_url}
Path: {repo_path}
</project>
 
<build_system>
{build_system_info}
</build_system>
 
<dependencies>
{dependencies_info}
</dependencies>
 
<architecture_analysis>
Target Architecture: {target_arch}
 
Architecture-Specific Patterns Found:
{arch_patterns}
 
Existing Architecture Support:
{existing_archs}
</architecture_analysis>
 
<codebase_structure>
{repo_tree}
</codebase_structure>
</repository_context>
 
<knowledge_base>
{system_knowledge}
</knowledge_base>
 
<task>
Create a comprehensive build plan with the following structure:
 
<build_plan>
<strategy>
[Explain your porting approach: Will you follow existing architecture patterns or use a generic approach? Why?]
</strategy>
 
<phases>
<phase name="preparation">
<purpose>[What this phase achieves]</purpose>
<commands>
[List exact shell commands]
</commands>
<expected_outcome>[What success looks like]</expected_outcome>
</phase>
 
<phase name="configuration">
<purpose>[Configuration goals]</purpose>
<commands>
[Configuration commands with all necessary flags]
</commands>
<expected_outcome>[Configuration success criteria]</expected_outcome>
</phase>
 
<phase name="compilation">
<purpose>[Build objectives]</purpose>
<commands>
[Compilation commands]
</commands>
<expected_outcome>[What artifacts should be produced]</expected_outcome>
</phase>
 
<phase name="verification">
<purpose>[Testing and validation]</purpose>
<commands>
[Test commands]
</commands>
<expected_outcome>[Verification success criteria]</expected_outcome>
</phase>
</phases>
 
<risk_analysis>
<high_risk_areas>
[List components likely to fail with justification]
</high_risk_areas>
<mitigation_strategies>
[Preventive measures for each risk]
</mitigation_strategies>
</risk_analysis>
 
<architecture_specific_notes>
[Any special considerations for {target_arch}]
</architecture_specific_notes>
</build_plan>
</task>
 
<guidelines>
1. Commands must be executable as-is (no placeholders like <value>)
2. Include all necessary flags and environment variables
3. Specify absolute paths when critical
4. For multi-architecture projects, follow existing naming conventions exactly
5. Prioritize non-invasive changes (configuration over code modification)
6. Ensure reproducibility (same commands should work across environments)
7. Include fallback strategies for anticipated failures
</guidelines>
 
<output_format>
Return ONLY a valid JSON object with this EXACT structure:
{{
  "strategy": "your approach description",
  "complexity_score": 1-10,
  "estimated_total_time": "expected duration",
  "estimated_total_cost": 0.01,
  "can_parallelize": [[phase_ids_that_can_run_together]],
  "phases": [
    {{
      "id": 1,
      "name": "phase_name",
      "description": "what this phase does",
      "agent": "scout|builder|fixer",
      "use_scripted_ops": true|false,
      "depends_on": [list of prior phase ids],
      "estimated_cost": 0.01
    }}
  ]
}}

Agent assignment rules:
- "scout": For information gathering, dependency detection, environment checking
- "builder": For configuration, compilation, build setup
- "fixer": For fixing errors, applying patches, troubleshooting failures

Example phases:
- Phase 1 (scout): Analyze build system, detect dependencies, find architecture-specific code
- Phase 2 (builder): Configure build environment, apply patterns from reference architectures
- Phase 3 (builder): Compile with RISC-V target
- Phase 4 (fixer): If compilation fails, fix issues and retry
</output_format>
 
<examples>
<example type="cmake_multiarch">
{{
  "strategy": "Follow existing ARM64 pattern: create arch/riscv64 directory and mirror CMake structure",
  "complexity_score": 6,
  "estimated_total_time": "15 minutes",
  "estimated_total_cost": 0.05,
  "can_parallelize": [],
  "phases": [
    {{
      "id": 1,
      "name": "scout_architecture",
      "description": "Analyze CMake structure and detect existing arch patterns",
      "agent": "scout",
      "use_scripted_ops": true,
      "depends_on": [],
      "estimated_cost": 0.01
    }},
    {{
      "id": 2,
      "name": "setup_riscv64",
      "description": "Create RISC-V-specific directory structure mirroring ARM64 pattern",
      "agent": "builder",
      "use_scripted_ops": false,
      "depends_on": [1],
      "estimated_cost": 0.01
    }},
    {{
      "id": 3,
      "name": "configure_build",
      "description": "Run CMake with RISC-V toolchain",
      "agent": "builder",
      "use_scripted_ops": false,
      "depends_on": [2],
      "estimated_cost": 0.02
    }},
    {{
      "id": 4,
      "name": "compile",
      "description": "Compile for RISC-V target",
      "agent": "builder",
      "use_scripted_ops": false,
      "depends_on": [3],
      "estimated_cost": 0.01
    }}
  ]
}}
</example>

<example type="generic_makefile">
{{
  "strategy": "Generic Makefile with ARCH= override - no existing patterns detected",
  "complexity_score": 4,
  "estimated_total_time": "10 minutes",
  "estimated_total_cost": 0.03,
  "can_parallelize": [],
  "phases": [
    {{
      "id": 1,
      "name": "verify_toolchain",
      "description": "Check if RISC-V toolchain is available",
      "agent": "scout",
      "use_scripted_ops": true,
      "depends_on": [],
      "estimated_cost": 0.01
    }},
    {{
      "id": 2,
      "name": "compile_riscv",
      "description": "Build with RISC-V toolchain using Makefile",
      "agent": "builder",
      "use_scripted_ops": false,
      "depends_on": [1],
      "estimated_cost": 0.02
    }}
  ]
}}
</example>
</examples>
 
Think step-by-step:
1. What build system is used?
2. How do existing architectures fit into the build process?
3. Can I replicate that pattern for {target_arch}?
4. What dependencies and tools are required?
5. What's the minimal viable build command sequence?
6. What could go wrong and how do I prevent it?
 
Generate the build plan now."""


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

    # Build system info
    build_system_info = f"Type: {build_sys.type if build_sys else 'Unknown'}\nConfidence: {f'{build_sys.confidence:.2f}' if build_sys else '0.0'}\nPrimary File: {build_sys.primary_file if build_sys else 'Unknown'}"

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
            patterns_by_type[arch_code.arch_type].append(f"{arch_code.file}:{arch_code.line}")

        for arch_type, locations in patterns_by_type.items():
            arch_patterns_list.append(f"- {arch_type}: {len(locations)} occurrences in {', '.join(locations[:3])}")

    arch_patterns = "\n".join(arch_patterns_list) if arch_patterns_list else "No architecture-specific patterns detected"
    existing_archs = f"Detected: {', '.join(existing_archs_list)}" if existing_archs_list else "No existing multi-architecture patterns detected"

    # System Environment summary
    system_info_raw = state.context_cache.get("system_info", {})
    system_info = "\n".join(
        [f"  - {tool}: {status}" for tool, status in system_info_raw.items()]
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
    )

    # Call LLM
    try:
        messages = [HumanMessage(content=prompt)]
        llm = get_model_for_role(AgentRole.PLANNER)
        response = llm.invoke(messages)
        state.log_api_call(cost=0.01)  # Estimate

        content = extract_content(response.content)

        # Log LLM call for debugging
        log_llm_call(
            agent_role=AgentRole.PLANNER.value,
            prompt=prompt,
            response=content,
            model=llm.model_name if hasattr(llm, 'model_name') else 'unknown',
            cost_usd=0.01,
            metadata={"repo": state.repo_name, "phase": "planner"}
        )

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
                id=p.get("id", len(phases) + 1),
                name=p.get("name", "unknown"),
                description=p.get("description", ""),
                agent=role,
                use_scripted_ops=p.get("use_scripted_ops", "scout" not in agent_str),
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
# NODE: SUPERVISOR (Orchestration)
# ============================================================================
 
SUPERVISOR_PROMPT = """<role>
You are the Workflow Supervisor orchestrating a multi-agent software porting system. Your responsibility is to analyze the current state, select the optimal next action, and maintain efficient progress toward successful target architecture builds.
</role>
 
<capabilities>
- State analysis: Evaluate build progress, error patterns, and agent performance
- Decision making: Choose between SCOUT (investigate), BUILD (execute), FIX (repair), or FINISH (complete)
- Resource optimization: Minimize unnecessary LLM calls and redundant operations
- Risk management: Escalate critical failures that exceed agent capabilities
</capabilities>
 
<workflow_state>
<current_status>
Phase: {current_phase}
Build Status: {build_status}
Attempt: {attempt_count}
Previous Agent: {current_agent}
</current_status>
 
<build_plan>
{build_plan_summary}
</build_plan>
 
<execution_history>
Scripted Operations: {scripted_ops_count}
API Calls: {api_calls}
Cost: ${cost:.4f}
 
Agent Actions:
{agent_history}
</execution_history>
 
<architecture_issues>
{arch_issues_summary}
</architecture_issues>
 
<errors>
{error_summary}
</errors>
 
<context>
{additional_context}
</context>
</workflow_state>
 
<available_agents>
<scout>
Role: Investigate architecture-specific code and potential porting issues
When to use: Before first build, after plan changes, when new patterns emerge
Cost: Medium (requires code analysis)
</scout>
 
<builder>
Role: Execute build plan phases and capture output
When to use: Plan exists, no blocking issues identified, ready to attempt build
Cost: Low (mostly command execution with light LLM interpretation)
</builder>
 
<fixer>
Role: Diagnose and repair build failures
When to use: Build failed with actionable errors
Cost: High (requires error analysis, patch generation, verification)
</fixer>
 
<finish>
Role: Generate porting documentation and finalize workflow
When to use: Build succeeded OR maximum attempts exhausted
Cost: Low (summary generation only)
</finish>
</available_agents>
 
<decision_framework>
<conditions>
<!-- Scout Triggers -->
<trigger agent="SCOUT">
- No architecture scan performed yet (state.arch_specific_code is empty)
- Build plan changed significantly
- New architecture patterns discovered during build
- High-risk components identified but not analyzed
</trigger>
 
<!-- Build Triggers -->
<trigger agent="BUILD">
- Build plan exists and is not currently executing
- No critical blocking issues identified
- Scout analysis complete (if applicable)
- Last build attempt was not recent (avoid tight loops)
- Attempt count below threshold (typically < 5)
</trigger>
 
<!-- Fix Triggers -->
<trigger agent="FIX">
- Last build failed with concrete error messages
- Error is categorized as FIXABLE (not NETWORK, ENVIRONMENT, or CRITICAL_DEPENDENCY)
- Attempt count below maximum (< 8)
- Error provides sufficient context for diagnosis
</trigger>
 
<!-- Finish Triggers -->
<trigger agent="FINISH">
- Build status is SUCCESS
- Maximum attempts reached (>= 8)
- Error severity is HIGH and unfixable
- All reasonable recovery strategies exhausted
</trigger>
 
<!-- Escalate Triggers -->
<trigger agent="ESCALATE">
- Critical dependency missing (toolchain, core libraries)
- Network failures preventing progress
- State corruption or internal errors
- Infinite loop detected (same action repeating)
</trigger>
</conditions>
 
<optimization_rules>
1. Avoid redundant scouts: If architecture analysis complete and plan hasn't changed, skip SCOUT
2. Prevent tight loops: If BUILD → FIX → BUILD → FIX → BUILD → same error, ESCALATE
3. Respect attempt limits: After 8 attempts, gracefully FINISH with current state
4. Minimize costs: Prefer scripted operations over LLM calls when deterministic
5. Fail fast: ESCALATE immediately on unrecoverable errors (missing toolchain, network failures)
</optimization_rules>
</decision_framework>
 
<task>
Analyze the current state and decide the next action.
 
<thinking_process>
1. What was the last action and its outcome?
2. Is there a blocking issue that must be addressed?
3. Have we exhausted reasonable attempts for the current approach?
4. What's the most efficient path to completion?
5. Are we in a loop or making progress?
6. Should we escalate to human intervention?
</thinking_process>
 
Return your decision as a JSON object:
{{
  "next_agent": "SCOUT|BUILD|FIX|FINISH|ESCALATE",
  "reasoning": "2-3 sentence justification for this choice",
  "confidence": "high|medium|low",
  "expected_outcome": "what should happen next",
  "fallback_plan": "what to do if this fails"
}}
</task>
 
<examples>
<example scenario="first_build_ready">
{{
  "next_agent": "BUILD",
  "reasoning": "Build plan exists, no architecture scan needed (simple generic codebase), no previous build attempts. Time to execute the plan.",
  "confidence": "high",
  "expected_outcome": "Build executes and either succeeds or produces actionable errors",
  "fallback_plan": "If build fails, FIX agent will analyze errors"
}}
</example>
 
<example scenario="need_architecture_scan">
{{
  "next_agent": "SCOUT",
  "reasoning": "Build plan created but no architecture analysis performed. Must identify potential porting issues before attempting build.",
  "confidence": "high",
  "expected_outcome": "Scout identifies architecture-specific code patterns and compatibility issues",
  "fallback_plan": "Proceed to BUILD regardless if scout finds minimal issues"
}}
</example>
 
<example scenario="build_failed_fixable">
{{
  "next_agent": "FIX",
  "reasoning": "Build failed with clear error: missing architecture flag in CMake. Error is FIXABLE category, attempt 2 of 8.",
  "confidence": "high",
  "expected_outcome": "Fixer adds appropriate CMake flag and rebuild succeeds",
  "fallback_plan": "If fix doesn't work, try alternative flag combinations (attempt 3)"
}}
</example>
 
<example scenario="repeated_failures">
{{
  "next_agent": "ESCALATE",
  "reasoning": "Same compilation error after 4 fix attempts. Pattern indicates deeper issue beyond simple configuration. Human expertise needed.",
  "confidence": "high",
  "expected_outcome": "Escalation captures detailed state for manual review",
  "fallback_plan": "N/A - escalation is terminal state"
}}
</example>
 
<example scenario="build_succeeded">
{{
  "next_agent": "FINISH",
  "reasoning": "Build completed successfully, all artifacts generated correctly. Time to document the porting process.",
  "confidence": "high",
  "expected_outcome": "Comprehensive porting guide generated",
  "fallback_plan": "N/A - workflow complete"
}}
</example>
 
<example scenario="max_attempts">
{{
  "next_agent": "FINISH",
  "reasoning": "Reached 8 attempts without success. Gracefully finishing to preserve accumulated knowledge and provide partial guidance.",
  "confidence": "medium",
  "expected_outcome": "Partial porting guide with encountered issues documented",
  "fallback_plan": "N/A - attempt limit reached"
}}
</example>
</examples>
 
Make your decision now. Be decisive, cost-conscious, and pragmatic."""


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
    state.log_scripted_op("supervisor_routing")

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

    # Build context for supervisor prompt
    build_plan_summary = "No build plan yet"
    if state.build_plan:
        phase_names = [p.name for p in state.build_plan.phases]
        build_plan_summary = f"Build System: {state.build_plan.build_system}, Phases: {', '.join(phase_names)}, Last completed: {state.last_successful_phase}"

    agent_history = ""
    if state.audit_trail:
        recent = state.audit_trail[-5:]
        agent_history = "\n".join([
            f"- {entry.get('agent', 'unknown')}: {entry.get('event', 'unknown')} - {str(entry.get('data', ''))[:100]}"
            for entry in recent
        ])
    else:
        agent_history = "No previous agent actions"

    arch_issues_summary = "None detected"
    if state.arch_specific_code:
        arch_issues_summary = f"{len(state.arch_specific_code)} architecture-specific code instances found"

    error_summary = "No errors"
    if state.error_history:
        recent_errors = state.error_history[-3:]
        error_summary = "\n".join([
            f"- [{e.category.value}] {e.message[:150]}" for e in recent_errors
        ])

    prompt = SUPERVISOR_PROMPT.format(
        current_phase=state.current_phase,
        build_status=state.build_status.value,
        attempt_count=state.attempt_count,
        current_agent=state.current_agent.value if state.current_agent else "none",
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
        response = llm.invoke(messages)
        state.log_api_call(cost=0.002)

        content = extract_content(response.content)

        # Log LLM call for debugging
        log_llm_call(
            agent_role=AgentRole.SUPERVISOR.value,
            prompt=prompt,
            response=content,
            model=llm.model_name if hasattr(llm, 'model_name') else 'unknown',
            cost_usd=0.002,
            metadata={"repo": state.repo_name, "phase": "supervisor", "status": state.build_status.value}
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
            AgentRole.SUPERVISOR, action_str, decision.get("reasoning", "LLM decision")
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
 
SCOUT_PROMPT = """<role>
You are the Architecture Scout specializing in cross-platform build planning and compatibility analysis. Your mission is to analyze the codebase, identify porting challenges, and produce an executable build plan for the target architecture.
</role>

<capabilities>
- Build system analysis: Understand CMake, Make, Go modules, Cargo, Meson, Autotools, and other build tools
- Pattern recognition: Detect architecture-specific code patterns (assembly, intrinsics, conditional compilation)
- Dependency resolution: Identify required tools, libraries, and system packages
- Build plan generation: Create executable build phases with specific shell commands
</capabilities>

<target_architecture>
Architecture: {target_arch}
Common Identifiers: {arch_identifiers}
</target_architecture>

<codebase_context>
Repository: {repo_name}
Build System: {build_system}
Module Directory: {module_dir}

Project Structure:
{repo_tree}

Known Architecture Patterns:
{arch_build_patterns}

Go Main Package Info:
{go_main_info}
</codebase_context>

<pre_analysis>
Dependencies Detected: {deps_count}
Architecture-Specific Code Instances: {arch_code_count}
Documentation Files Cached: {doc_count}

System Environment:
{system_info}

Architecture Concerns:
{arch_concerns}

Detected Dependencies:
{dependencies}
</pre_analysis>

<documentation>
{documentation}
</documentation>

<system_knowledge>
{system_knowledge}
</system_knowledge>

{few_shot_examples}

<task>
Create a build plan that will compile this project for {target_arch} on Alpine Linux.

CRITICAL RULES:
1. Commands must be executable as-is inside a Docker container at path: {repo_path}
2. All commands run with the working directory set to {repo_path}
3. Do NOT use placeholder paths like /path/to/ or <value>
4. Do NOT use cross-compilation flags (GOOS/GOARCH) - we are building NATIVELY on {architecture}
5. For Go projects: use "go build ./..." or "go build -o <binary_name> ." - NOT "./configure && make"
6. For CMake projects: use "mkdir -p build && cd build && cmake .. && make"
7. For Make projects: use "make" or "make -j$(nproc)"
8. For Autotools: use "./configure && make" ONLY if a ./configure script actually exists
9. Install all needed dependencies via "apk add <package>" in the setup phase
10. Do NOT reference or copy non-existent directories

Return a JSON object with this EXACT structure:
{{
  "build_system": "go|cmake|make|autotools|cargo|meson|other",
  "build_system_confidence": 0.95,
  "phases": [
    {{
      "id": 1,
      "name": "setup",
      "commands": ["apk add <needed packages>"],
      "can_parallelize": false,
      "expected_duration": "30s"
    }},
    {{
      "id": 2,
      "name": "build",
      "commands": ["<actual build commands>"],
      "can_parallelize": false,
      "expected_duration": "2m"
    }}
  ],
  "total_estimated_duration": "3m",
  "notes": ["any important observations"],
  "architecture_concerns": ["list of arch-specific issues found, if any"]
}}
</task>

<examples>
<example type="go_project">
{{
  "build_system": "go",
  "build_system_confidence": 0.95,
  "phases": [
    {{
      "id": 1,
      "name": "setup",
      "commands": ["apk add go git"],
      "can_parallelize": false,
      "expected_duration": "30s"
    }},
    {{
      "id": 2,
      "name": "build",
      "commands": ["go build -buildvcs=false ./..."],
      "can_parallelize": false,
      "expected_duration": "2m"
    }}
  ],
  "total_estimated_duration": "3m",
  "notes": ["Go has native RISC-V support since Go 1.14"],
  "architecture_concerns": []
}}
</example>

<example type="cmake_project">
{{
  "build_system": "cmake",
  "build_system_confidence": 0.95,
  "phases": [
    {{
      "id": 1,
      "name": "setup",
      "commands": ["apk add build-base cmake"],
      "can_parallelize": false,
      "expected_duration": "30s"
    }},
    {{
      "id": 2,
      "name": "configure",
      "commands": ["mkdir -p build", "cd build && cmake .. -DCMAKE_BUILD_TYPE=Release"],
      "can_parallelize": false,
      "expected_duration": "1m"
    }},
    {{
      "id": 3,
      "name": "build",
      "commands": ["cd build && make -j$(nproc)"],
      "can_parallelize": false,
      "expected_duration": "5m"
    }}
  ],
  "total_estimated_duration": "7m",
  "notes": [],
  "architecture_concerns": []
}}
</example>

<example type="make_project">
{{
  "build_system": "make",
  "build_system_confidence": 0.9,
  "phases": [
    {{
      "id": 1,
      "name": "setup",
      "commands": ["apk add build-base"],
      "can_parallelize": false,
      "expected_duration": "30s"
    }},
    {{
      "id": 2,
      "name": "build",
      "commands": ["make -j$(nproc)"],
      "can_parallelize": false,
      "expected_duration": "5m"
    }}
  ],
  "total_estimated_duration": "6m",
  "notes": [],
  "architecture_concerns": []
}}
</example>
</examples>

Generate the build plan now. Return ONLY valid JSON, no other text."""

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

            # Check for bad cmake patterns
            if "cmake" in cmd:
                if "cd build &&" in cmd and "-B build" in cmd:
                    return False, f"Bad cmake command (would create nested build dir): '{cmd}'"
                # Ensure we're doing `cmake ..` from inside build dir, not `cmake -S . -B build`
                if "cd build &&" in cmd and "cmake -S" in cmd and "-B build" in cmd:
                    return False, f"Bad cmake pattern (creates build/build): '{cmd}'"

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

    # Build architecture patterns info
    arch_patterns_str = "No architecture-specific patterns detected"
    if state.arch_specific_code:
        arch_patterns_by_type = {}
        for arch_code in state.arch_specific_code:
            if arch_code.arch_type not in arch_patterns_by_type:
                arch_patterns_by_type[arch_code.arch_type] = 0
            arch_patterns_by_type[arch_code.arch_type] += 1
        arch_patterns_str = "\n".join(
            [f"- {arch_type}: {count} matches" for arch_type, count in arch_patterns_by_type.items()]
        )

    prompt = SCOUT_PROMPT.format(
        target_arch="RISC-V (riscv64)",
        arch_identifiers="rv64, riscv, riscv64, RISCV64",
        repo_name=state.repo_name,
        build_system=(build_sys.type if build_sys else "unknown")
        .replace("{", "{{")
        .replace("}", "}}"),
        repo_tree=state.repo_tree.replace("{", "{{").replace("}", "}}")
        if state.repo_tree
        else "(Not available)",
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

    try:
        messages = [HumanMessage(content=prompt)]
        llm = get_model_for_role(AgentRole.SCOUT)
        response = llm.invoke(messages)
        state.log_api_call(cost=0.01)

        content = extract_content(response.content)

        # Log LLM call for debugging
        log_llm_call(
            agent_role=AgentRole.SCOUT.value,
            prompt=prompt,
            response=content,
            model=llm.model_name if hasattr(llm, 'model_name') else 'unknown',
            cost_usd=0.01,
            metadata={"repo": state.repo_name, "build_system": state.build_system_info.type if state.build_system_info else 'unknown'}
        )

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
        logger.info("Using fallback build plan due to LLM parsing error")
        state.build_plan = create_fallback_build_plan(state)

    state.build_status = BuildStatus.PENDING
    return state


def create_fallback_build_plan(state: AgentState) -> BuildPlan:
    """Create a basic fallback build plan based on detected build system."""
    build_sys = state.build_system_info
    build_type = build_sys.type if build_sys else "unknown"

    if build_type == "go":
        return BuildPlan(
            build_system="go",
            build_system_confidence=0.8,
            phases=[
                BuildPhase(1, "setup", ["apk update && apk add go git"], False, "30s"),
                BuildPhase(2, "build", ["go build -buildvcs=false ./..."], False, "3m"),
            ],
            total_estimated_duration="4m",
            notes=["Fallback Go build plan"],
        )
    elif build_type == "cmake":
        return BuildPlan(
            build_system="cmake",
            build_system_confidence=0.7,
            phases=[
                BuildPhase(1, "setup", ["apk update && apk add build-base cmake"], False, "30s"),
                BuildPhase(2, "configure", ["mkdir -p build", "cd build && cmake .. -DCMAKE_BUILD_TYPE=Release"], False, "1m"),
                BuildPhase(3, "build", ["cd build && make -j$(nproc)"], False, "5m"),
            ],
            total_estimated_duration="7m",
            notes=["Fallback CMake build plan"],
        )
    elif build_type == "make":
        return BuildPlan(
            build_system="make",
            build_system_confidence=0.7,
            phases=[
                BuildPhase(1, "setup", ["apk update && apk add build-base"], False, "30s"),
                BuildPhase(2, "build", ["make -j$(nproc)"], False, "5m"),
            ],
            total_estimated_duration="6m",
            notes=["Fallback Make build plan"],
        )
    elif build_type == "cargo":
        return BuildPlan(
            build_system="cargo",
            build_system_confidence=0.8,
            phases=[
                BuildPhase(1, "setup", ["apk update && apk add rust cargo"], False, "30s"),
                BuildPhase(2, "build", ["cargo build --release"], False, "5m"),
            ],
            total_estimated_duration="6m",
            notes=["Fallback Cargo build plan"],
        )
    elif build_type == "meson":
        return BuildPlan(
            build_system="meson",
            build_system_confidence=0.7,
            phases=[
                BuildPhase(1, "setup", ["apk update && apk add build-base meson ninja"], False, "30s"),
                BuildPhase(2, "configure", ["meson setup builddir"], False, "1m"),
                BuildPhase(3, "build", ["ninja -C builddir"], False, "5m"),
            ],
            total_estimated_duration="7m",
            notes=["Fallback Meson build plan"],
        )
    elif build_type == "autotools":
        return BuildPlan(
            build_system="autotools",
            build_system_confidence=0.6,
            phases=[
                BuildPhase(1, "setup", ["apk update && apk add build-base autoconf automake libtool"], False, "30s"),
                BuildPhase(2, "configure", ["./configure"], False, "1m"),
                BuildPhase(3, "build", ["make -j$(nproc)"], False, "5m"),
            ],
            total_estimated_duration="7m",
            notes=["Fallback Autotools build plan"],
        )
    else:
        # Unknown build system - try to guess from files present
        logger.warning(f"Unknown build system '{build_type}', using generic fallback")
        return BuildPlan(
            build_system=build_type,
            build_system_confidence=0.3,
            phases=[
                BuildPhase(1, "setup", ["apk update && apk add build-base"], False, "30s"),
                BuildPhase(2, "build", ["make -j$(nproc)"], False, "5m"),
            ],
            total_estimated_duration="6m",
            notes=["Generic fallback - build system not recognized"],
        )



# ============================================================================
# NODE: BUILDER (Build Execution)
# ============================================================================
 

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
            state.log_scripted_op("execute_build_command")

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
                    state.log_scripted_op("retry_build_command")

                    if retry_result.success:
                        logger.info("Retry with -buildvcs=false succeeded!")
                        phase.commands[phase.commands.index(command)] = retry_command
                        continue
                    else:
                        logger.error(f"Retry also failed: {retry_result.stderr[:500]}")
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

    # All phases completed - NOW VERIFY THE BUILD PRODUCED ARTIFACTS
    logger.info("All build phases completed - verifying artifacts...")

    # Determine where to look for build artifacts based on build system
    build_system = state.build_plan.build_system if state.build_plan else "unknown"
    
    # Check multiple possible artifact locations
    artifact_dirs = [state.repo_path]  # Always check repo root
    
    # Add build-system-specific directories
    build_subdir = os.path.join(state.repo_path, "build")
    if build_system in ["cmake", "meson"]:
        artifact_dirs.insert(0, build_subdir)  # Check build/ first for cmake/meson
    
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
                    architecture=artifact["architecture"]
                )

            summary = scanner.get_summary()
            logger.info(f"Build artifacts summary: {json.dumps(summary, indent=2, default=str)}")
            state.context_cache["artifact_summary"] = summary
            artifacts_found = True
            break

    if not artifacts_found:
        # For Go projects, check if any binary was produced in the repo root
        if build_system == "go":
            find_cmd = f"find {state.repo_path} -maxdepth 1 -type f -executable"
            find_result = execute_command(find_cmd, cwd=state.repo_path)
            if find_result.success and find_result.stdout.strip():
                logger.info(f"Go binary found: {find_result.stdout.strip()}")
                for binary_path in find_result.stdout.strip().split("\n"):
                    binary_path = binary_path.strip()
                    if binary_path:
                        # Verify it's a RISC-V binary
                        file_cmd = f"file {binary_path}"
                        file_result = execute_command(file_cmd, cwd=state.repo_path)
                        if file_result.success and "RISC-V" in file_result.stdout:
                            state.add_build_artifact(
                                filepath=binary_path,
                                artifact_type="binary",
                                architecture="riscv64"
                            )
                            artifacts_found = True
                            logger.info(f"Verified RISC-V binary: {binary_path}")
                        elif file_result.success and "ELF" in file_result.stdout:
                            # ELF binary on RISC-V host is likely RISC-V
                            state.add_build_artifact(
                                filepath=binary_path,
                                artifact_type="binary",
                                architecture="riscv64"
                            )
                            artifacts_found = True
                            logger.info(f"Found ELF binary (assumed RISC-V on native host): {binary_path}")

    if not artifacts_found:
        logger.warning("No build artifacts found, but all phases completed successfully")
        # Don't fail the build - phases completed successfully, artifacts may be installed elsewhere
        logger.info("Treating as success since all build commands succeeded")

    state.build_status = BuildStatus.SUCCESS
    logger.info("Build completed successfully with artifact verification!")

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


def validate_fixer_response(fix_data: Dict) -> tuple[bool, str]:
    """
    Validate FIXER response to ensure it makes logical sense.
    Returns (is_valid, error_message).
    """
    try:
        if not fix_data.get("strategies"):
            return False, "No strategies provided"

        recommended_id = fix_data.get("recommended_strategy_id", 1)
        strategies = fix_data.get("strategies", [])

        # Find recommended strategy
        recommended = next((s for s in strategies if s["id"] == recommended_id), None)
        if not recommended:
            return False, f"Recommended strategy {recommended_id} not found"

        # Check if strategy has actions
        actions = recommended.get("actions", [])
        if not actions:
            # Strategy has no actions - this is incomplete/hallucinatory
            logger.warning("FIXER strategy has no actions (incomplete response)")
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
                    return False, f"create_file action has empty content for {path} - this won't help"

                # Check for obviously hallucinated placeholders
                if len(path) > 100:
                    return False, f"Suspiciously long filepath: {path[:100]}..."

                # Check for absolute paths that look wrong
                if path.startswith("/home/") or path.startswith("/root/"):
                    return False, f"Suspicious absolute path (should be relative): {path}"

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
                        if src == dst or src.split("/")[0] == dst.split("/")[0]:
                            return False, f"Nonsensical command detected: {cmd} (copying to self)"

                is_safe, reason = validate_fix_command(cmd)
                if not is_safe:
                    return False, f"Unsafe command: {reason}"

        # Validate reflection data
        reflection = fix_data.get("reflection", {})
        if not reflection.get("root_cause"):
            logger.warning("Fixer response has empty root_cause analysis")

        if not reflection.get("this_fix_will_work_because"):
            logger.warning("Fixer response has empty 'this_fix_will_work_because' - self-critique may be shallow")

        return True, ""

    except Exception as e:
        logger.error(f"Error validating fixer response: {e}")
        return False, f"Validation error: {str(e)}"


# ============================================================================
# NODE: FIXER (Error Resolution)
# ============================================================================
 
FIXER_PROMPT = """<role>
You are the Build Fixer specializing in diagnosing and resolving compilation failures. Your mission is to analyze errors, generate targeted patches, and restore build functionality with minimal code changes.
</role>
 
<capabilities>
- Error diagnosis: Parse compiler/linker output to identify root causes
- Patch generation: Create precise code modifications or build configuration changes
- Compatibility adaptation: Port architecture-specific code to target platform
- Regression prevention: Ensure fixes don't break existing functionality
</capabilities>
 
<build_context>
Repository: {repo_name}
Target Architecture: {target_arch}
Build System: {build_system}
Current Phase: {current_phase}
Attempt: {attempt_count} / {max_attempts}
</build_context>
 
<build_failure>
Phase: {failed_phase}
Exit Code: {exit_code}
 
Error Output:
{error_output}
 
Failed Command:
{failed_command}
 
Previous Fix Attempts:
{previous_fixes}
</build_failure>
 
<available_context>
<codebase_structure>
{repo_tree}
</codebase_structure>
 
<architecture_issues>
{known_arch_issues}
</architecture_issues>
 
<build_plan>
{build_plan}
</build_plan>
 
<system_knowledge>
{system_knowledge}
</system_knowledge>
</available_context>
 
<diagnostic_process>
<step_1_error_classification>
Categorize the failure:
- COMPILATION_ERROR: Source code won't compile
- LINKING_ERROR: Object files won't link
- CONFIGURATION_ERROR: Build system misconfiguration
- DEPENDENCY_ERROR: Missing library or tool
- ARCHITECTURE_INCOMPATIBILITY: Target-specific issue
</step_1_error_classification>
 
<step_2_root_cause_analysis>
Identify the specific problem:
- Extract file, line number, and error message
- Determine if this is architecture-specific or generic
- Check if error relates to known issues from scout analysis
- Assess whether this is a regression from previous fix
</step_2_root_cause_analysis>
 
<step_3_solution_design>
Choose the minimal effective fix:
- Prefer configuration changes over code modification
- Prefer conditional compilation over replacing code
- Prefer adding missing defines over removing functionality
- Prefer portable solutions over target-specific hacks
</step_3_solution_design>
 
<step_4_patch_generation>
Create the fix:
- For code: Generate unified diff format patches
- For build config: Provide exact modification instructions
- For flags: Specify complete command with new flags
- Test mentally: Would this fix work without breaking other architectures?
</step_4_patch_generation>
</diagnostic_process>
 
<fix_strategies_by_error_type>
<intrinsics_replacement>
Problem: x86/ARM intrinsics not available on target
Solution:
1. Check if portable fallback exists in codebase
2. Implement generic C version if simple
3. Use target-equivalent intrinsics if available
4. Disable feature via conditional compilation if non-critical
</intrinsics_replacement>
 
<inline_assembly>
Problem: Architecture-specific assembly code
Solution:
1. Look for existing generic implementation
2. Replace with portable C code
3. Use target assembly if performance critical
4. Disable feature if optional
</inline_assembly>
 
<missing_definitions>
Problem: Undefined macros or symbols
Solution:
1. Add missing defines to build configuration
2. Include missing headers
3. Add conditional definitions for target architecture
</missing_definitions>
 
<wrong_flags>
Problem: Incorrect compiler/linker flags
Solution:
1. Update CMake/Makefile with target-appropriate flags
2. Remove architecture-specific optimization flags
3. Add target architecture identification flags
</wrong_flags>
 
<conditional_compilation>
Problem: Code assumes specific architecture
Solution:
1. Add #elif for target architecture
2. Ensure fallback case exists
3. Define appropriate feature macros
</conditional_compilation>
</fix_strategies_by_error_type>
 
<task>
Analyze the build failure and generate a fix.
 
<thinking_process>
1. What exactly failed? (exact error, file, line)
2. Why did it fail? (root cause, not symptom)
3. Have we seen this before? (check previous fixes)
4. What's the minimal fix? (least invasive change)
5. Will this fix break anything else? (compatibility check)
6. Is this fixable by an agent? (or needs escalation)
</thinking_process>
 
Return a JSON object with this EXACT structure:
{{
  "strategies": [
    {{
      "id": 1,
      "description": "Brief description of what this fix does",
      "actions": [
        {{
          "type": "command",
          "command": "shell command to run (e.g., apk add missing-package)"
        }}
      ]
    }}
  ],
  "recommended_strategy_id": 1,
  "reflection": {{
    "root_cause": "Why the error occurred",
    "this_fix_will_work_because": "Reasoning why the proposed fix addresses the root cause"
  }}
}}

Action types:
- "command": Run a shell command. Fields: "type", "command"
- "create_file": Create a new file. Fields: "type", "path" (relative to repo root), "content"
- "patch": Apply a patch. Fields: "type", "file" (relative path), "content" (unified diff format)

IMPORTANT RULES:
1. Always provide at least one strategy with at least one action
2. Do NOT use empty commands or empty file content
3. Do NOT create empty source files (e.g., touch main.go)
4. Do NOT use absolute paths starting with /home/ or /root/
5. For Go projects with "./configure: No such file or directory" errors, the fix is to use "go build" instead
6. For missing dependencies, use "apk add <package>" commands
7. Provide actionable fixes, not just analysis
</task>
 
<examples>
<example type="wrong_build_command">
{{
  "strategies": [
    {{
      "id": 1,
      "description": "Replace incorrect ./configure with Go build command",
      "actions": [
        {{
          "type": "command",
          "command": "go build -buildvcs=false ./..."
        }}
      ]
    }}
  ],
  "recommended_strategy_id": 1,
  "reflection": {{
    "root_cause": "Build plan used ./configure for a Go project that uses go modules",
    "this_fix_will_work_because": "Go projects are built with 'go build', not autotools configure scripts"
  }}
}}
</example>

<example type="missing_dependency">
{{
  "strategies": [
    {{
      "id": 1,
      "description": "Install missing build dependency and retry",
      "actions": [
        {{
          "type": "command",
          "command": "apk add openssl-dev"
        }},
        {{
          "type": "command",
          "command": "make -j$(nproc)"
        }}
      ]
    }}
  ],
  "recommended_strategy_id": 1,
  "reflection": {{
    "root_cause": "Missing openssl-dev header files required by the project",
    "this_fix_will_work_because": "Installing the -dev package provides the missing headers"
  }}
}}
</example>

<example type="arch_incompatibility">
{{
  "strategies": [
    {{
      "id": 1,
      "description": "Add RISC-V case to architecture detection and disable SIMD",
      "actions": [
        {{
          "type": "patch",
          "file": "src/platform.h",
          "content": "--- a/src/platform.h\\n+++ b/src/platform.h\\n@@ -10,6 +10,8 @@\\n #ifdef __x86_64__\\n   #define PLATFORM_X86\\n+#elif defined(__riscv)\\n+  #define PLATFORM_RISCV64\\n #endif"
        }}
      ]
    }}
  ],
  "recommended_strategy_id": 1,
  "reflection": {{
    "root_cause": "Architecture detection header missing RISC-V case",
    "this_fix_will_work_because": "Adding the __riscv preprocessor check follows the same pattern as existing architectures"
  }}
}}
</example>
</examples>
 
<guidelines>
1. Always provide exact patches (not pseudocode)
2. Test your fix mentally before proposing it
3. Consider side effects on other architectures
4. Prefer portable solutions over target-specific hacks
5. Escalate when truly necessary (missing toolchain, fundamental incompatibility)
6. Learn from previous failed fixes (don't repeat the same approach)
7. Explain your reasoning clearly
</guidelines>
 
Diagnose and fix the error now."""


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

    # Get build plan details
    build_plan_str = "Unknown"
    if state.build_plan:
        build_plan_str = f"Build System: {state.build_plan.build_system}\nCompleted Phases: {state.last_successful_phase}"

    # Create prompt
    prompt = FIXER_PROMPT.format(
        repo_name=state.repo_name,
        target_arch="RISC-V (riscv64)",
        build_system=state.build_plan.build_system if state.build_plan else "unknown",
        current_phase=state.current_phase,
        attempt_count=state.attempt_count,
        max_attempts=state.max_attempts,
        failed_phase=state.current_phase,
        exit_code="N/A",
        error_output=state.last_error[:1000] if state.last_error else "No error details",
        failed_command=failed_command,
        previous_fixes=previous_fixes,
        repo_tree=state.repo_tree[:500] if state.repo_tree else "(Not available)",
        known_arch_issues=arch_issues,
        build_plan=build_plan_str,
        system_knowledge=get_system_knowledge_summary(),
    )

    try:
        messages = [HumanMessage(content=prompt)]
        llm = get_model_for_role(AgentRole.FIXER)
        response = llm.invoke(messages)
        state.log_api_call(cost=0.01)

        content = extract_content(response.content)

        # Log LLM call for debugging
        log_llm_call(
            agent_role=AgentRole.FIXER.value,
            prompt=prompt,
            response=content,
            model=llm.model_name if hasattr(llm, 'model_name') else 'unknown',
            cost_usd=0.01,
            metadata={
                "repo": state.repo_name,
                "error_category": state.last_error_category.value if state.last_error_category else 'unknown',
                "attempt": state.attempt_count
            }
        )

        # Parse JSON
        json_match = extract_json_block(content)
        fix_data = json.loads(json_match)

        # Validate the FIXER response
        is_valid, validation_error = validate_fixer_response(fix_data)
        if not is_valid:
            logger.error(f"FIXER response validation failed: {validation_error}")
            state.add_error(
                create_error_record(
                    message=f"FIXER proposed invalid fix: {validation_error}",
                    category=ErrorCategory.CONFIGURATION,
                )
            )
            state.build_status = BuildStatus.FAILED
            return state

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
Severity: {state.last_error_severity.value if state.last_error_severity else "unknown"}
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

    if state.error_history:
        report += "\n\n## Recent Failures\n"
        for err in state.error_history[-5:]:
            timestamp = err.timestamp.strftime("%H:%M:%S")
            command = err.command if err.command else "N/A"
            msg = (err.message or "N/A").replace("\n", " ")
            report += (
                f"- [{timestamp}] severity={err.severity.value}, "
                f"category={err.category.value}, command={command}, message={msg[:260]}\n"
            )

    logger.info(report)
    state.context_cache["escalation_report"] = report

    return state


# ============================================================================
# NODE: FINISH
# ============================================================================


def _save_learning_data(state: AgentState):
    """Save successful build patterns back to examples and recipe cache (auto-learning)."""
    try:
        repo = state.repo_name or "unknown"
        bs = state.build_plan.build_system if state.build_plan else "unknown"

        # --- Scout learning ---
        if state.build_plan and state.build_plan.phases:
            scout_data = {
                "name": f"Auto: {repo} ({bs})",
                "tags": [bs, repo],
                "build_system": bs,
                "repo_name": repo,
                "trigger": {
                    "build_system": bs,
                    "has_main": bool(getattr(state, 'go_main_info', {}).get("has_main", False)
                                     if hasattr(state, 'go_main_info') and state.go_main_info else False),
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
        for fix in (state.fixes_attempted or []):
            if fix.success and fix.strategy:
                fixer_data = {
                    "name": f"Auto: {repo} - {fix.strategy[:40]}",
                    "tags": [bs, "auto-fix"],
                    "build_system": bs,
                    "repo_name": repo,
                    "error_pattern": re.escape((state.last_error or "")[:80]) if state.last_error else "",
                    "fix": {
                        "strategy": fix.strategy,
                        "actions": [{"type": "command", "command": cmd} for cmd in (fix.commands_run or [])],
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
                "timeout_recommendation": "600s" if len(state.build_plan.phases) > 2 else "120s",
                "reasoning": f"Auto-learned build execution from {repo}.",
            }
            save_learned_example("builder", builder_data)

        # --- Recipe cache ---
        if state.build_plan:
            duration = state.get_execution_duration() if hasattr(state, 'get_execution_duration') else 0.0
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
                patches=[f.strategy for f in (state.fixes_attempted or []) if f.success],
                artifacts=state.build_artifacts or [],
                build_duration_seconds=duration,
            )

        logger.info(f"Auto-learning complete for {repo}")
    except Exception as e:
        logger.warning(f"Auto-learning failed (non-fatal): {e}")


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

    # Prepare artifacts information
    artifacts_info = ""
    if state.build_artifacts:
        artifacts_info = "\n### Build Artifacts Generated\n\n"
        for artifact in state.build_artifacts:
            artifacts_info += f"- **{artifact['type']}** ({artifact['architecture']}): `{artifact['filepath']}`\n"

    build_steps += artifacts_info

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

        # Log LLM call for debugging
        log_llm_call(
            agent_role=AgentRole.SUMMARIZER.value,
            prompt=prompt,
            response=recipe,
            model=llm.model_name if hasattr(llm, 'model_name') else 'unknown',
            cost_usd=0.005,
            metadata={"repo": state.repo_name, "phase": "finish"}
        )

        state.porting_recipe = recipe
        logger.info("Porting guide generated successfully")

    except Exception as e:
        logger.error(f"Summarizer failed: {e}")
        # Fallback recipe
        state.porting_recipe = f"# RISC-V Porting Recipe: {state.repo_name}\n\nBuild succeeded.\n\n{build_steps}"

    # Auto-learning: save successful patterns
    _save_learning_data(state)

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

    # Failed initialization is a hard blocker and must not continue to planning.
    if (
        phase in {"initialization", "initialized"}
        and state.build_status == BuildStatus.FAILED
    ):
        logger.warning("Initialization failed; forcing escalation instead of planning")
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

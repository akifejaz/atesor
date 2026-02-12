"""
Core orchestration logic defining the multi-agent state machine and workflow nodes.
Utilizes LangGraph to manage agent transitions and state.
"""

import json
import logging
from typing import Literal, Optional, List
from datetime import datetime
from pathlib import Path

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from src.state import (
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
)
from src.scripted_ops import ScriptedOperations, quick_analysis
from src.models import create_llm
from src.tools import execute_command, apply_patch

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
        return "\n".join([
            str(item.get("text", item)) if isinstance(item, dict) else str(item) 
            for item in content
        ])
    return str(content)


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
                    attempt_number=state.attempt_count
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

@agent_node(AgentRole.SUPERVISOR) # init is special but handled by supervisor/scripted ops logic
def init_node(state: AgentState) -> AgentState:
    """
    Initialize the workflow by cloning the repository using scripted operations.
    This is a zero-cost operation (no LLM calls).
    """
    logger.info(f"Initializing workflow for {state.repo_url}")
    
    # Clone or update repository (scripted operation)
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
    
    # Perform quick analysis using scripted operations
    logger.info("Performing quick analysis with scripted operations...")
    try:
        analysis = quick_analysis(state.repo_path)
        state.log_scripted_op()
        
        # Store results in state
        state.build_system_info = analysis.get('build_system')
        state.dependencies = analysis.get('dependencies')
        state.arch_specific_code = analysis.get('arch_specific_code', [])
        
        # Cache documentation content
        for doc_path in analysis.get('documentation', [])[:5]:  # Top 5 docs
            content = scripted_ops.read_file(doc_path, max_lines=500)
            state.cache_file_content(doc_path, content)
            state.log_scripted_op()
        
        # Store in context cache
        state.context_cache['quick_analysis'] = analysis
        
        logger.info(f"Quick analysis complete: "
                   f"Build system: {state.build_system_info.type if state.build_system_info else 'unknown'}, "
                   f"Arch-specific code: {len(state.arch_specific_code)} instances")
        
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

## Quick Analysis Results (from scripted operations):
- Build System: {build_system} (confidence: {build_system_confidence})
- Dependencies: {dependencies_summary}
- Architecture-Specific Code: {arch_code_count} instances found
- Documentation Available: {doc_count} files

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
        high_severity = [a for a in state.arch_specific_code if a.severity == 'high']
        if high_severity:
            arch_concerns = f"\n## Architecture Concerns:\n"
            for concern in high_severity[:5]:
                arch_concerns += f"- {concern.arch_type} in {concern.file}:{concern.line}\n"
    
    # Create prompt
    prompt = PLANNER_PROMPT.format(
        repo_name=state.repo_name,
        repo_url=state.repo_url,
        build_system=build_sys.type if build_sys else "unknown",
        build_system_confidence=f"{build_sys.confidence:.2f}" if build_sys else "0.0",
        dependencies_summary=dependencies_summary,
        arch_code_count=len(state.arch_specific_code),
        doc_count=len([k for k in state.file_content_cache.keys() if 'README' in k or 'INSTALL' in k]),
        arch_concerns=arch_concerns,
        repo_path=state.repo_path,
    )
    
    # Call LLM
    try:
        messages = [HumanMessage(content=prompt)]
        llm = get_model_for_role(AgentRole.PLANNER)
        response = llm.invoke(messages)
        state.log_api_call(cost=0.01)  # Estimate
        
        content = extract_content(response.content)
        
        # Parse JSON response
        json_match = content[content.find('{'):content.rfind('}')+1]
        plan_data = json.loads(json_match)
        
        # Create TaskPlan
        phases = []
        for p in plan_data['phases']:
            agent_str = p['agent'].lower()
            if agent_str in ['architect', 'supervisor']:
                role = AgentRole.BUILDER # Fallback for planning
            elif 'scout' in agent_str:
                role = AgentRole.SCOUT
            elif 'fix' in agent_str:
                role = AgentRole.FIXER
            elif 'build' in agent_str:
                role = AgentRole.BUILDER
            else:
                try:
                    role = AgentRole(agent_str)
                except ValueError:
                    role = AgentRole.BUILDER
            
            phase = TaskPhase(
                id=p['id'],
                name=p['name'],
                description=p['description'],
                agent=role,
                use_scripted_ops=p['use_scripted_ops'],
                depends_on=p.get('depends_on', []),
                estimated_cost=p.get('estimated_cost', 0.0),
            )
            phases.append(phase)
        
        state.task_plan = TaskPlan(
            phases=phases,
            can_parallelize=plan_data.get('can_parallelize', []),
            estimated_total_cost=plan_data.get('estimated_total_cost', 0.0),
            estimated_total_time=plan_data.get('estimated_total_time', 'unknown'),
            complexity_score=plan_data.get('complexity_score', 5),
        )
        
        logger.info(f"Strategic plan created: {len(phases)} phases, "
                   f"complexity: {state.task_plan.complexity_score}/10, "
                   f"estimated cost: ${state.task_plan.estimated_total_cost:.3f}")
        
        # Store plan in context
        state.context_cache['task_plan'] = plan_data
        
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
            TaskPhase(1, "scout", "Analyze and create build plan", AgentRole.SCOUT, False, []),
            TaskPhase(2, "build", "Execute build", AgentRole.BUILDER, False, [1]),
        ],
        estimated_total_cost=0.02,
        complexity_score=5,
    )


# ============================================================================
# NODE: SUPERVISOR (Enhanced)
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
    Enhanced supervisor with better context awareness and cost optimization.
    """
    logger.info(f"Supervisor making routing decision... (BuildPlan: {'Exists' if state.build_plan else 'Missing'})")
    
    # Check for automatic escalation
    should_esc, esc_reason = should_escalate(state)
    if should_esc:
        logger.warning(f"Automatic escalation: {esc_reason}")
        state.current_phase = "escalate"
        return state
    
    # Use recommendation system first (free)
    recommended_action = get_next_action_recommendation(state)
    state.log_scripted_op()
    
    # For simple cases, use recommendation directly
    if state.api_calls_made > 10 or state.api_cost_usd > 0.10:
        # Cost optimization: use heuristic routing instead of LLM
        state.log_agent_decision(AgentRole.SUPERVISOR, recommended_action.value, "Cost optimization heuristic")
        logger.info(f"Using cost-optimized routing: {recommended_action.value}")
        state.current_phase = recommended_action.value.lower()
        return state

    # Deterministic guards (from Foundry)
    if state.build_status == BuildStatus.SUCCESS:
        state.log_agent_decision(AgentRole.SUPERVISOR, "FINISH", "Build succeeded, moving to final documentation.")
        state.current_phase = "finish"
        return state
    
    if state.attempt_count >= state.max_attempts:
        state.log_agent_decision(AgentRole.SUPERVISOR, "ESCALATE", f"Max attempts ({state.max_attempts}) reached.")
        state.current_phase = "escalate"
        return state
    
    # Prepare context
    task_plan_status = "No plan yet"
    if state.task_plan:
        completed = len([p for p in state.task_plan.phases if p.status == 'completed'])
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
        decision_context = "A build plan exists. BUILDER should now execute the build phases."
    elif state.build_status == BuildStatus.SUCCESS:
        decision_context = "Build succeeded. FINISH or run tests if not done."
    
    # Create prompt
    prompt = SUPERVISOR_PROMPT.format(
        repo_name=state.repo_name,
        build_status=state.build_status.value,
        attempt_count=state.attempt_count,
        max_attempts=state.max_attempts,
        current_phase=state.current_phase,
        last_error_category=state.last_error_category.value if state.last_error_category else "None",
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
        
        # Parse action
        action_line = [line for line in content.split('\n') if line.startswith('ACTION:')][0]
        action_str = action_line.split(':')[1].strip().upper()
        
        # Map to next phase
        action_map = {
            'SCOUT': 'scout',
            'BUILDER': 'builder',
            'FIXER': 'fixer',
            'ESCALATE': 'escalate',
            'FINISH': 'finish',
        }
        
        decision_reason = f"LLM Routing: {action_str}"
        state.log_agent_decision(AgentRole.SUPERVISOR, action_str, content.split('\n')[-1])
        state.current_phase = action_map.get(action_str, 'scout')
        
        logger.info(f"Supervisor decision: {action_str}")
        
    except Exception as e:
        logger.error(f"Supervisor failed: {e}, using fallback")
        state.current_phase = recommended_action.value.lower()
    
    return state


# ============================================================================
# NODE: SCOUT (Enhanced)
# ============================================================================

SCOUT_PROMPT = """You are the **Scout Agent**, expert at analyzing software projects for RISC-V porting.

# Quick Analysis Already Done (via scripts):
- Build System: {build_system}
- Dependencies Found: {deps_count}
- Arch-Specific Code: {arch_code_count} instances
- Documentation: {doc_count} files cached

# Your Task
Review the available information and create a detailed build plan.

## Available Documentation:
{documentation}

## Architecture Concerns:
{arch_concerns}

## Dependencies:
{dependencies}

# System Environment (Actual tools available)
{system_info}

# Your Task
Analyze the project and create a **Build Plan**.
The Build Plan MUST use only standard tools (gcc, cmake, make) without absolute paths unless they are standard (e.g., /usr/bin/gcc).
DO NOT guess or hallucinate compiler paths like '/path/to/riscv64-gcc'. Just use 'gcc' or appropriate environment variables.
Use the system architecture: {architecture}.

# Output Format
Provide a JSON object with:
```json
{{
  "build_system": "cmake",
  "build_system_confidence": 0.95,
  "phases": [
    {{
      "id": 1,
      "name": "install_dependencies",
      "commands": ["apt-get update", "apt-get install -y gcc cmake"],
      "can_parallelize": false,
      "expected_duration": "30s"
    }},
    {{
      "id": 2,
      "name": "configure",
      "commands": ["cmake -B build -DCMAKE_C_FLAGS='-march=rv64gc'"],
      "can_parallelize": false,
      "expected_duration": "10s"
    }},
    {{
      "id": 3,
      "name": "build",
      "commands": ["cmake --build build -j$(nproc)"],
      "can_parallelize": false,
      "expected_duration": "2m"
    }}
  ],
  "total_estimated_duration": "3m",
  "notes": ["Any important observations"]
}}
```

Repository: {repo_path}
"""

def validate_build_plan(plan: BuildPlan) -> tuple[bool, str]:
    """Validate BuildPlan for hallucinations and common issues."""
    hallucination_patterns = [
        '/path/to/',
        'your_username',
        'example.com',
        'riscv64-unknown-linux-gnu-gcc', # Unless we confirmed it's there
    ]
    
    for phase in plan.phases:
        for cmd in phase.commands:
            for pattern in hallucination_patterns:
                if pattern in cmd:
                    return False, f"Hallucination detected in command: '{cmd}' (contains '{pattern}')"
            
            # Check for absolute paths that look guessed
            if '/home/' in cmd and '/workspace/' not in cmd:
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
    
    # Prepare context from cached data
    build_sys = state.build_system_info
    deps = state.dependencies
    
    documentation = "No documentation cached"
    if state.file_content_cache:
        doc_files = [k for k in state.file_content_cache.keys() if any(d in k.upper() for d in ['README', 'INSTALL', 'BUILD'])]
        if doc_files:
            documentation = "\n\n".join([
                f"## {Path(f).name}\n{state.file_content_cache[f][:2000]}"
                for f in doc_files[:3]
            ])
    
    arch_concerns = "None detected"
    if state.arch_specific_code:
        arch_concerns = "\n".join([
            f"- {a.file}:{a.line} - {a.arch_type} ({a.severity})"
            for a in state.arch_specific_code[:10]
        ])
    
    dependencies = "Unknown"
    if deps:
        dependencies = json.dumps({
            'build_tools': deps.build_tools,
            'system_packages': deps.system_packages[:10],
            'libraries': deps.libraries[:10],
        }, indent=2)
    
    # Get system info
    from src.scripted_ops import ScriptedOperations
    ops = ScriptedOperations()
    system_info_raw = ops.get_system_info()
    system_info = "\n".join([f"- {k}: {v}" for k, v in system_info_raw.items()])
    
    # Create prompt
    prompt = SCOUT_PROMPT.format(
        build_system=(build_sys.type if build_sys else "unknown").replace('{', '{{').replace('}', '}}'),
        deps_count=len(deps.system_packages) if deps else 0,
        arch_code_count=len(state.arch_specific_code),
        doc_count=len(state.file_content_cache),
        documentation=documentation.replace('{', '{{').replace('}', '}}'),
        arch_concerns=arch_concerns.replace('{', '{{').replace('}', '}}'),
        dependencies=dependencies.replace('{', '{{').replace('}', '}}'),
        repo_path=state.repo_path.replace('{', '{{').replace('}', '}}'),
        system_info=system_info.replace('{', '{{').replace('}', '}}'),
        architecture=system_info_raw.get('architecture', 'riscv64').replace('{', '{{').replace('}', '}}')
    )
    
    try:
        messages = [HumanMessage(content=prompt)]
        llm = get_model_for_role(AgentRole.SCOUT)
        response = llm.invoke(messages)
        state.log_api_call(cost=0.01)
        
        content = extract_content(response.content)
        
        # Parse JSON
        json_match = content[content.find('{'):content.rfind('}')+1]
        plan_data = json.loads(json_match)
        
        # Create BuildPlan
        phases = []
        for i, p in enumerate(plan_data['phases']):
            phase = BuildPhase(
                id=p.get('id', i+1),
                name=p['name'],
                commands=p['commands'],
                can_parallelize=p.get('can_parallelize', False),
                expected_duration=p.get('expected_duration', 'unknown'),
            )
            phases.append(phase)
        
        state.build_plan = BuildPlan(
            build_system=plan_data.get('build_system', build_sys.type if build_sys else 'unknown'),
            build_system_confidence=plan_data.get('build_system_confidence', build_sys.confidence if build_sys else 0.5),
            phases=phases,
            total_estimated_duration=plan_data.get('total_estimated_duration', 'unknown'),
            notes=plan_data.get('notes', []),
        )
        
        # Validate plan
        is_valid, reason = validate_build_plan(state.build_plan)
        if not is_valid:
            logger.warning(f"Plan validation failed: {reason}")
            state.add_error(create_error_record(
                message=f"Invalid build plan: {reason}. Please avoid absolute paths and placeholders.",
                category=ErrorCategory.CONFIGURATION
            ))
            # Clear invalid plan so supervisor knows we need a real one
            state.build_plan = None
            
            # Create a simple fallback if this happened multiple times
            if state.attempt_count >= 2:
                logger.info("Using fallback build plan after repeated failures")
                state.build_plan = create_fallback_build_plan(state)
        else:
            logger.info(f"Build plan created and validated: {len(phases)} phases")
        
    except Exception as e:
        logger.error(f"Scout failed: {e}")
        # Create fallback plan
        state.build_plan = create_fallback_build_plan(state)
    
    state.build_status = BuildStatus.PENDING
    return state


def create_fallback_build_plan(state: AgentState) -> BuildPlan:
    """Create a basic fallback build plan."""
    build_sys = state.build_system_info
    
    if build_sys and build_sys.type == 'cmake':
        commands = [
            "cmake -B build .",
            "cmake --build build",
        ]
    elif build_sys and build_sys.type == 'make':
        commands = ["make"]
    else:
        commands = ["./configure", "make"]
    
    return BuildPlan(
        build_system=build_sys.type if build_sys else 'unknown',
        build_system_confidence=0.5,
        phases=[
            BuildPhase(1, "build", commands, False, "unknown")
        ],
        total_estimated_duration="unknown",
    )


# Continued in next file...
"""
Enhanced LangGraph State Machine - Part 2
Builder, Fixer, and Workflow Assembly
"""

# Continued from part 1...

# ============================================================================
# NODE: BUILDER (Enhanced)
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
    Execute the build plan with smart error detection.
    """
    logger.info("Builder executing build plan...")
    
    state.build_status = BuildStatus.BUILDING
    state.current_phase = "building"
    
    if not state.build_plan:
        logger.error("No build plan available")
        state.build_status = BuildStatus.FAILED
        state.add_error(create_error_record(
            "No build plan available",
            ErrorCategory.CONFIGURATION,
        ))
        return state
    
    # Execute each phase
    for phase in state.build_plan.phases:
        # Skip if already completed
        if phase.id <= state.last_successful_phase:
            continue
        
        logger.info(f"Executing phase {phase.id}: {phase.name}")
        
        for command in phase.commands:
            # Check cache first
            cached_result = state.get_cached_command_result(command)
            if cached_result and cached_result.success:
                logger.info(f"Using cached result for: {command[:50]}...")
                continue
            
            # Execute command
            result = execute_command(command, cwd=state.repo_path)
            state.cache_command_result(command, result)
            state.log_scripted_op()
            
            if not result.success:
                # Build failed
                logger.error(f"Command failed: {command}")
                logger.error(f"Error: {result.stderr[:500]}")
                
                error = create_error_record(
                    message=result.stderr,
                    category=classify_error(result.stderr),
                    command=command,
                    attempt_number=state.attempt_count,
                )
                state.add_error(error)
                state.build_status = BuildStatus.FAILED
                
                return state
        
        # Phase completed successfully
        state.last_successful_phase = phase.id
        logger.info(f"Phase {phase.id} completed successfully")
    
    # All phases completed
    state.build_status = BuildStatus.SUCCESS
    logger.info("Build completed successfully!")
    
    return state


# ============================================================================
# NODE: FIXER (Enhanced with Reflection)
# ============================================================================

FIXER_PROMPT = """You are the **Fixer Agent**, expert at debugging RISC-V porting issues.

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

# Your Task
1. Analyze the error
2. Generate 2-3 potential fixes
3. For each fix, explain the strategy
4. Implement the best fix

# Common Fix Strategies for RISC-V
- Add RISC-V conditional compilation (#ifdef __riscv)
- Use scalar fallbacks for SIMD code
- Update configure flags (-march=rv64gc)
- Install missing dependencies
- Fix incompatible assembly: Port x86 asm to RISC-V or use C fallback
- Endianness: RISC-V is little-endian, check for assumptions
- Alignment: RISC-V may have stricter alignment requirements
- Compiler Flags: Some x86-specific flags need removal
- Version Mismatch: If tool versions need update, patch files or install newer versions
- Reflection Loop: 1. ANALYZE error 2. HYPOTHESIZE fix 3. SELF-CRITIQUE (will it work? risks?) 4. APPLY minimal fix

# Output Format
```json
{{
  "analysis": "Brief analysis of the root cause",
  "strategies": [
    {{
      "id": 1,
      "description": "Strategy description",
      "confidence": 0.8,
      "actions": [
        {{"type": "patch", "file": "src/file.c", "content": "patch content"}},
        {{"type": "command", "command": "apt-get install -y libfoo"}}
      ]
    }}
  ],
  "recommended_strategy_id": 1
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
    previous_fixes = "\n".join([
        f"- {fix.strategy} ({'Success' if fix.success else 'Failed'})"
        for fix in state.fixes_attempted[-5:]
    ]) if state.fixes_attempted else "None"
    
    arch_issues = "None"
    if state.arch_specific_code:
        relevant = [
            a for a in state.arch_specific_code 
            if a.severity in ['high', 'critical']
        ][:5]
        if relevant:
            arch_issues = "\n".join([
                f"- {a.file}:{a.line} - {a.arch_type}"
                for a in relevant
            ])
    
    failed_command = "Unknown"
    if state.error_history:
        last = state.error_history[-1]
        failed_command = last.command or "Unknown"
    
    # Create prompt
    prompt = FIXER_PROMPT.format(
        error_category=state.last_error_category.value if state.last_error_category else "Unknown",
        error_message=state.last_error[:1000],
        failed_command=failed_command,
        build_system=state.build_plan.build_system if state.build_plan else "unknown",
        attempt_count=state.attempt_count,
        previous_fixes=previous_fixes,
        arch_issues=arch_issues,
        repo_path=state.repo_path,
    )
    
    try:
        messages = [HumanMessage(content=prompt)]
        llm = get_model_for_role(AgentRole.FIXER)
        response = llm.invoke(messages)
        state.log_api_call(cost=0.01)
        
        content = extract_content(response.content)
        
        # Parse JSON
        json_match = content[content.find('{'):content.rfind('}')+1]
        fix_data = json.loads(json_match)
        
        # Get recommended strategy
        recommended_id = fix_data.get('recommended_strategy_id', 1)
        strategies = fix_data.get('strategies', [])
        
        if not strategies:
            logger.error("No fix strategies generated")
            state.build_status = BuildStatus.FAILED
            return state
        
        # Find recommended strategy
        strategy = next((s for s in strategies if s['id'] == recommended_id), strategies[0])
        
        logger.info(f"Applying fix strategy: {strategy['description']}")
        
        # Apply fix actions
        changes_made = []
        for action in strategy.get('actions', []):
            if action['type'] == 'patch':
                # Apply patch
                patch_content = action['content']
                file_path = action.get('file')
                full_file_path = os.path.join(state.repo_path, file_path) if file_path else None
                
                # Use robust apply_patch tool
                if apply_patch(patch_content, filepath=full_file_path, cwd=state.repo_path, use_docker=True):
                    changes_made.append(f"Patched {file_path if file_path else 'repository'}")
                    state.patches_generated.append(patch_content)
                else:
                    logger.error(f"Failed to apply patch to {file_path}")
            
            elif action['type'] == 'command':
                # Execute command
                result = execute_command(action['command'], cwd=state.repo_path)
                if result.success:
                    changes_made.append(f"Executed: {action['command']}")
                else:
                    logger.error(f"Fix command failed: {result.stderr}")
        
        # Record fix attempt
        fix_attempt = FixAttempt(
            error_category=state.last_error_category,
            strategy=strategy['description'],
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

@agent_node(AgentRole.SUPERVISOR) # Escalation is a supervisor-level summary
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
Category: {state.last_error_category.value if state.last_error_category else 'Unknown'}
Message: {state.last_error[:500] if state.last_error else 'N/A'}

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
        high_priority = [a for a in state.arch_specific_code if a.severity in ['high', 'critical']]
        for issue in high_priority[:5]:
            report += f"\n- {issue.file}:{issue.line} - {issue.arch_type}"
    
    report += f"""

## Recommendation
{should_escalate(state)[1]}
"""
    
    logger.info(report)
    state.context_cache['escalation_report'] = report
    
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
    arch_issues = "\n".join([
        f"- {a.file}:{a.line} - {a.arch_type} ({a.severity})"
        for a in state.arch_specific_code[:20]
    ]) if state.arch_specific_code else "None found."
    
    fixes = "\n".join([
        f"- {fix.strategy} ({'Success' if fix.success else 'Failed'})"
        for fix in state.fixes_attempted
    ]) if state.fixes_attempted else "No fixes were needed (generic code)."
    
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
        duration=f"{state.get_execution_duration():.1f}s"
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
    """
    phase = state.current_phase.lower()
    
    routing_map = {
        'initialization': 'planner',
        'initialized': 'planner',
        'planning': 'supervisor',
        'planned': 'supervisor',
        'scout': 'scout_node',
        'scouting': 'supervisor',
        'builder': 'builder_node',
        'building': 'supervisor',
        'fixer': 'fixer_node',
        'fixing': 'supervisor',
        'escalate': 'escalate_node',
        'escalated': END,
        'finish': 'finish_node',
        'finished': END,
    }
    
    next_node = routing_map.get(phase, 'supervisor')
    logger.info(f"Routing from {phase} to {next_node}")
    
    return next_node


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


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_porting_agent(repo_url: str, max_attempts: int = 5):
    """
    Run the complete porting workflow.
    """
    logger.info(f"Starting RISC-V porting agent for {repo_url}")
    
    # Create initial state
    initial_state = create_initial_state(repo_url, max_attempts)
    
    # Create and run workflow
    app = create_workflow()
    
    try:
        final_state = None
        for output in app.stream(initial_state):
            node_name = list(output.keys())[0]
            node_state = output[node_name]
            logger.info(f"Completed node: {node_name}")
            logger.info(f"Status: {node_state.build_status.value}")
            logger.info(f"Progress: {node_state.get_progress_summary()}")
            final_state = node_state
        
        # Print final results
        logger.info("=" * 60)
        logger.info("FINAL RESULTS")
        logger.info("=" * 60)
        logger.info(f"Status: {final_state.build_status.value}")
        logger.info(f"Duration: {final_state.get_execution_duration():.1f}s")
        logger.info(f"API Calls: {final_state.api_calls_made}")
        logger.info(f"Scripted Ops: {final_state.scripted_ops_count}")
        logger.info(f"Cost: ${final_state.api_cost_usd:.4f}")
        logger.info(f"Cost Savings: {(final_state.scripted_ops_count / max(final_state.api_calls_made + final_state.scripted_ops_count, 1) * 100):.1f}%")
        
        if final_state.porting_recipe:
            logger.info("\nPorting recipe generated successfully!")
        
        return final_state
        
    except Exception as e:
        logger.error(f"Workflow failed: {e}", exc_info=True)
        raise


# Create global app instance
app = create_workflow()


if __name__ == "__main__":
    # Example usage
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    
    repo_url = sys.argv[1] if len(sys.argv) > 1 else "https://github.com/madler/zlib"
    
    result = run_porting_agent(repo_url)
    
    if result.porting_recipe:
        # Save recipe
        with open(f"/workspace/output/{result.repo_name}_recipe.md", 'w') as f:
            f.write(result.porting_recipe)
        print(f"\nRecipe saved to {result.repo_name}_recipe.md")

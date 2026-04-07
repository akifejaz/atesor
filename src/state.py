"""
Global state definitions, data structures, and status tracking for the porting process.
Manages the AgentState dataclass and system-wide constants.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime
from langchain_core.messages import BaseMessage


# ============================================================================
# ENUMS
# ============================================================================


class BuildStatus(str, Enum):
    """Current status of the build process."""

    PENDING = "PENDING"
    PLANNING = "PLANNING"
    SCOUTING = "SCOUTING"
    BUILDING = "BUILDING"
    TESTING = "TESTING"
    FIXING = "FIXING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    ESCALATED = "ESCALATED"


class ErrorCategory(str, Enum):
    """Classification of errors encountered."""

    UNKNOWN = "UNKNOWN"
    DEPENDENCY = "DEPENDENCY"
    COMPILATION = "COMPILATION"
    LINKING = "LINKING"
    ARCHITECTURE = "ARCHITECTURE"
    NETWORK = "NETWORK"
    RATE_LIMIT = "RATE_LIMIT"
    CONFIGURATION = "CONFIGURATION"
    MISSING_TOOLS = "MISSING_TOOLS"
    PERMISSION = "PERMISSION"
    DISK_SPACE = "DISK_SPACE"
    LICENSE_INCOMPATIBLE = "LICENSE_INCOMPATIBLE"
    REQUIRES_HARDWARE = "REQUIRES_HARDWARE"
    ARCHITECTURE_IMPOSSIBLE = "ARCHITECTURE_IMPOSSIBLE"


class FailureSeverity(str, Enum):
    """Severity level for command and execution failures."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class AgentRole(str, Enum):
    """Roles of different agents in the system."""

    PLANNER = "planner"
    SUPERVISOR = "supervisor"
    SCOUT = "scout"
    BUILDER = "builder"
    FIXER = "fixer"
    SUMMARIZER = "summarizer"
    AGENT = "agent"


class Action(str, Enum):
    """Available actions for the supervisor."""

    PLAN = "PLAN"
    SCOUT = "SCOUT"
    BUILDER = "BUILDER"
    FIXER = "FIXER"
    ESCALATE = "ESCALATE"
    FINISH = "FINISH"


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class CommandResult:
    """Result of a shell command execution."""

    command: str
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def success(self) -> bool:
        return self.exit_code == 0

    @property
    def failed(self) -> bool:
        return self.exit_code != 0


@dataclass
class ErrorRecord:
    """Record of an error that occurred."""

    category: ErrorCategory
    message: str
    severity: FailureSeverity = FailureSeverity.MEDIUM
    command: Optional[str] = None
    file: Optional[str] = None
    line: Optional[int] = None
    traceback: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    attempt_number: int = 0


@dataclass
class FixAttempt:
    """Record of a fix attempt."""

    error_category: ErrorCategory
    strategy: str
    changes_made: List[str]
    success: bool
    build_result: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ArchSpecificCode:
    """Information about architecture-specific code found."""

    file: str
    line: int
    code_snippet: str
    arch_type: str  # "x86", "arm", "simd", etc.
    severity: str  # "low", "medium", "high", "critical"
    suggested_fix: Optional[str] = None


@dataclass
class BuildPhase:
    """A single phase in the build plan."""

    id: int
    name: str
    commands: List[str]
    can_parallelize: bool = False
    expected_duration: str = "unknown"
    required_dependencies: List[str] = field(default_factory=list)
    success_criteria: Optional[str] = None


@dataclass
class BuildPlan:
    """Complete build plan for the package."""

    build_system: str
    build_system_confidence: float
    phases: List[BuildPhase]
    total_estimated_duration: str
    notes: List[str] = field(default_factory=list)


@dataclass
class DependencyInfo:
    """Information about package dependencies."""

    system_packages: List[str] = field(default_factory=list)
    libraries: List[str] = field(default_factory=list)
    build_tools: List[str] = field(default_factory=list)
    risc_v_available: bool = True
    install_method: str = "apt"
    version_constraints: Dict[str, str] = field(default_factory=dict)


@dataclass
class TaskPhase:
    """A phase in the overall task plan."""

    id: int
    name: str
    description: str
    agent: AgentRole
    use_scripted_ops: bool
    depends_on: List[int] = field(default_factory=list)
    estimated_cost: float = 0.0  # In USD
    status: str = "pending"  # pending, in_progress, completed, failed


@dataclass
class TaskPlan:
    """High-level decomposition of the porting task."""

    phases: List[TaskPhase]
    can_parallelize: List[List[int]] = field(
        default_factory=list
    )  # Groups of parallel phase IDs
    estimated_total_cost: float = 0.0
    estimated_total_time: str = "unknown"
    complexity_score: int = 5  # 1-10


@dataclass
class BuildSystemInfo:
    """Detected build system information."""

    type: str
    confidence: float
    primary_file: str
    additional_files: List[str] = field(default_factory=list)
    requires_configuration: bool = True
    module_dir: str = ""


@dataclass
class CommunityPortStatus:
    """Status of RISC-V port in the community."""

    existing_port: bool = False
    patches_available: bool = False
    patch_urls: List[str] = field(default_factory=list)
    community_discussion_urls: List[str] = field(default_factory=list)
    last_checked: datetime = field(default_factory=datetime.now)


# ============================================================================
# MAIN STATE CLASS
# ============================================================================


@dataclass
class AgentState:
    """
    Comprehensive state for the RISC-V porting agent.
    All agents read from and write to this shared state.
    """

    # ========== Repository Information ==========
    repo_url: str
    repo_name: str
    repo_path: str = "/workspace/repos"
    repo_tree: str = ""  # Optimized tree output for initial context

    # ========== Task Planning ==========
    task_plan: Optional[TaskPlan] = None
    current_phase: str = "initialization"

    # ========== Build Information ==========
    build_system_info: Optional[BuildSystemInfo] = None
    build_plan: Optional[BuildPlan] = None
    build_status: BuildStatus = BuildStatus.PENDING
    tests_run: bool = False
    tests_passed: bool = False

    # ========== Dependencies ==========
    dependencies: Optional[DependencyInfo] = None
    missing_dependencies: List[str] = field(default_factory=list)

    # ========== Architecture Analysis ==========
    arch_specific_code: List[ArchSpecificCode] = field(default_factory=list)
    risc_v_blockers: List[str] = field(default_factory=list)
    community_port_status: Optional[CommunityPortStatus] = None

    # ========== Execution Tracking ==========
    attempt_count: int = 0
    max_attempts: int = 5
    last_successful_phase: int = 0

    # ========== Error Handling ==========
    last_error: Optional[str] = None
    last_error_category: Optional[ErrorCategory] = None
    last_error_severity: Optional[FailureSeverity] = None
    error_history: List[ErrorRecord] = field(default_factory=list)
    fixes_attempted: List[FixAttempt] = field(default_factory=list)

    # ========== Performance Tracking ==========
    api_calls_made: int = 0
    api_cost_usd: float = 0.0
    scripted_ops_count: int = 0
    execution_start_time: datetime = field(default_factory=datetime.now)

    # ========== Caching & Memory ==========
    context_cache: Dict[str, Any] = field(default_factory=dict)
    file_content_cache: Dict[str, str] = field(default_factory=dict)
    command_results_cache: Dict[str, CommandResult] = field(default_factory=dict)

    # ========== Agent Communication ==========
    messages: List[BaseMessage] = field(default_factory=list)

    # ========== Output Artifacts ==========
    patches_generated: List[str] = field(default_factory=list)
    porting_recipe: Optional[str] = None

    # ========== Debugging & Audit ==========
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)
    current_agent: Optional[AgentRole] = None

    # ========== Metadata ==========
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

    def update_timestamp(self):
        """Update the last_updated timestamp."""
        self.last_updated = datetime.now()

    def add_error(self, error: ErrorRecord):
        """Add an error to history and update state."""
        self.error_history.append(error)
        self.last_error = error.message
        self.last_error_category = error.category
        self.last_error_severity = error.severity
        self.attempt_count += 1
        self.update_timestamp()

    def add_fix_attempt(self, fix: FixAttempt):
        """Record a fix attempt."""
        self.fixes_attempted.append(fix)
        self.update_timestamp()

    def log_api_call(self, cost: float = 0.0):
        """Track API usage."""
        self.api_calls_made += 1
        self.api_cost_usd += cost
        self.update_timestamp()

    def log_scripted_op(self, operation: str = "unknown"):
        """Track scripted operation usage."""
        self.scripted_ops_count += 1
        self.log_event("scripted_op", {"operation": operation})
        self.update_timestamp()

    def log_event(self, event_type: str, data: Dict[str, Any]):
        """Add an event to the audit trail."""
        self.audit_trail.append(
            {
                "timestamp": datetime.now().isoformat(),
                "event": event_type,
                "agent": self.current_agent.value if self.current_agent else None,
                "data": data,
            }
        )
        self.update_timestamp()

    def log_agent_decision(self, agent: AgentRole, action: str, reason: str):
        """Log a decision made by an agent."""
        self.log_event(
            "decision", {"agent": agent.value, "action": action, "reason": reason}
        )

    def cache_command_result(self, command: str, result: CommandResult):
        """Cache a command result for reuse."""
        cache_key = self._generate_cache_key(command)
        self.command_results_cache[cache_key] = result
        self.update_timestamp()

    def get_cached_command_result(self, command: str) -> Optional[CommandResult]:
        """Retrieve cached command result if available."""
        cache_key = self._generate_cache_key(command)
        return self.command_results_cache.get(cache_key)

    def cache_file_content(self, filepath: str, content: str):
        """Cache file content to avoid repeated reads."""
        self.file_content_cache[filepath] = content
        self.update_timestamp()

    def get_cached_file_content(self, filepath: str) -> Optional[str]:
        """Retrieve cached file content if available."""
        return self.file_content_cache.get(filepath)

    def _generate_cache_key(self, command: str) -> str:
        """Generate a cache key for a command."""
        import hashlib

        return hashlib.md5(command.encode()).hexdigest()

    def get_execution_duration(self) -> float:
        """Get total execution time in seconds."""
        return (datetime.now() - self.execution_start_time).total_seconds()

    def is_in_error_loop(self) -> bool:
        """Check if we're stuck in an error loop."""
        if len(self.error_history) < 3:
            return False

        # Check if last 3 errors are the same category
        recent_errors = self.error_history[-3:]
        categories = [e.category for e in recent_errors]

        return len(set(categories)) == 1  # All same category

    def get_progress_summary(self) -> str:
        """Get a human-readable progress summary."""
        duration = self.get_execution_duration()
        return (
            f"Status: {self.build_status.value}\n"
            f"Attempt: {self.attempt_count}/{self.max_attempts}\n"
            f"API Calls: {self.api_calls_made}\n"
            f"Scripted Ops: {self.scripted_ops_count}\n"
            f"Cost: ${self.api_cost_usd:.4f}\n"
            f"Duration: {duration:.1f}s\n"
            f"Phase: {self.current_phase}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to a dictionary for serialization."""
        from dataclasses import asdict

        # We need a custom converter for Enums and Datetime
        def custom_serializer(obj):
            if isinstance(obj, (datetime, Enum)):
                return str(obj)
            if isinstance(obj, list):
                return [custom_serializer(i) for i in obj]
            if isinstance(obj, dict):
                return {k: custom_serializer(v) for k, v in obj.items()}
            if hasattr(obj, "__dict__"):
                return custom_serializer(vars(obj))
            return obj

        return custom_serializer(asdict(self))

    def save_to_json(self, filepath: str):
        """Save the current state to a JSON file."""
        import json

        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def get_last_audit_events(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve the last N audit events."""
        return self.audit_trail[-limit:]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def create_initial_state(repo_url: str, max_attempts: int = 5) -> AgentState:
    """Create initial state for a new porting task."""
    repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")

    return AgentState(
        repo_url=repo_url,
        repo_name=repo_name,
        repo_path=f"/workspace/repos/{repo_name}",
        max_attempts=max_attempts,
    )


def classify_error(error_message: str) -> ErrorCategory:
    """
    Classify an error message into a category.
    This uses pattern matching on common error patterns.
    """
    error_lower = error_message.lower()

    # Network errors
    if any(
        term in error_lower
        for term in ["network", "connection", "timeout", "unreachable"]
    ):
        return ErrorCategory.NETWORK

    # Rate limiting
    if any(
        term in error_lower
        for term in ["rate limit", "too many requests", "429", "quota exceeded"]
    ):
        return ErrorCategory.RATE_LIMIT

    # Linking errors
    if any(
        term in error_lower
        for term in [
            "linking error",
            "linker",
            "undefined symbol",
            "cannot find -l",
            "ld returned",
            "collect2: error",
        ]
    ):
        return ErrorCategory.LINKING

    # Compilation errors
    if any(
        term in error_lower
        for term in [
            "compilation error",
            "syntax error",
            "parse error",
            "undeclared",
            "undefined reference",
            "implicit declaration",
        ]
    ):
        return ErrorCategory.COMPILATION

    # Architecture-specific errors
    if any(
        term in error_lower
        for term in [
            "architecture",
            "arch",
            "sse",
            "avx",
            "neon",
            "simd",
            "unsupported instruction",
            "illegal instruction",
            "x86_64",
            "amd64",
        ]
    ):
        return ErrorCategory.ARCHITECTURE

    # Configuration errors
    if any(
        term in error_lower
        for term in [
            "configure error",
            "cmake error",
            "configure: error",
            "unsupported option",
            "invalid argument",
        ]
    ):
        return ErrorCategory.CONFIGURATION

    # Missing tools
    if any(
        term in error_lower
        for term in [
            "command not found",
            "no such command",
            "not installed",
            "cmake: not found",
            "make: not found",
        ]
    ):
        return ErrorCategory.MISSING_TOOLS

    # Dependency errors
    if any(
        term in error_lower
        for term in [
            "cannot find",
            "not found",
            "no such file",
            "missing dependency",
            "package not found",
            "module not found",
            "import error",
        ]
    ):
        return ErrorCategory.DEPENDENCY

    # Permission errors
    if any(
        term in error_lower
        for term in ["permission denied", "access denied", "forbidden"]
    ):
        return ErrorCategory.PERMISSION

    # Disk space
    if any(
        term in error_lower for term in ["no space left", "disk full", "out of space"]
    ):
        return ErrorCategory.DISK_SPACE

    # Python/System errors
    if any(
        term in error_lower
        for term in [
            "keyerror",
            "indexerror",
            "attributeerror",
            "typeerror",
            "valueerror",
            "importerror",
        ]
    ):
        return ErrorCategory.CONFIGURATION  # Usually a code/config bug

    return ErrorCategory.UNKNOWN


def create_error_record(
    message: str,
    category: Optional[ErrorCategory] = None,
    severity: Optional[FailureSeverity] = None,
    command: Optional[str] = None,
    attempt_number: int = 0,
) -> ErrorRecord:
    """Create an error record with automatic classification."""
    if category is None:
        category = classify_error(message)
    if severity is None:
        severity = infer_failure_severity(category, command=command, message=message)

    return ErrorRecord(
        category=category,
        message=message,
        severity=severity,
        command=command,
        attempt_number=attempt_number,
    )


def infer_failure_severity(
    category: ErrorCategory,
    command: Optional[str] = None,
    message: str = "",
) -> FailureSeverity:
    """
    Infer failure severity from category + command context.
    Low: non-blocking probe failures.
    Medium: standard build/config failures that should be fixed before continuing.
    High: critical initialization/infrastructure blockers.
    """
    cmd = (command or "").strip().lower()
    msg = (message or "").lower()

    if cmd.startswith("which "):
        return FailureSeverity.LOW

    high_categories = {
        ErrorCategory.LICENSE_INCOMPATIBLE,
        ErrorCategory.REQUIRES_HARDWARE,
        ErrorCategory.ARCHITECTURE_IMPOSSIBLE,
        ErrorCategory.PERMISSION,
        ErrorCategory.DISK_SPACE,
    }
    if category in high_categories:
        return FailureSeverity.HIGH

    if any(
        pattern in cmd
        for pattern in [
            "git clone",
            "git pull",
            "apt-get update",
            "apt update",
            "apk update",
            "apk add",
        ]
    ):
        return FailureSeverity.HIGH

    if category in {
        ErrorCategory.CONFIGURATION,
        ErrorCategory.DEPENDENCY,
        ErrorCategory.COMPILATION,
        ErrorCategory.LINKING,
        ErrorCategory.NETWORK,
        ErrorCategory.RATE_LIMIT,
        ErrorCategory.MISSING_TOOLS,
        ErrorCategory.ARCHITECTURE,
    }:
        return FailureSeverity.MEDIUM

    if "not found" in msg and "which " in msg:
        return FailureSeverity.LOW

    return FailureSeverity.MEDIUM


def should_escalate(state: AgentState) -> tuple[bool, str]:
    """
    Determine if the task should be escalated to human intervention.
    Returns (should_escalate, reason).
    """
    # Max attempts reached
    if state.attempt_count >= state.max_attempts:
        return True, f"Maximum attempts ({state.max_attempts}) reached"

    # Stuck in error loop
    if state.is_in_error_loop():
        return True, "Stuck in error loop with no progress"

    # Fundamental blockers
    fundamental_categories = {
        ErrorCategory.LICENSE_INCOMPATIBLE,
        ErrorCategory.REQUIRES_HARDWARE,
        ErrorCategory.ARCHITECTURE_IMPOSSIBLE,
    }

    if state.last_error_category in fundamental_categories:
        return True, f"Fundamental blocker: {state.last_error_category.value}"

    # Cost limit (if set)
    cost_limit = 1.0  # $1 USD limit
    if state.api_cost_usd > cost_limit:
        return True, f"API cost limit (${cost_limit}) exceeded"

    return False, ""


def get_next_action_recommendation(state: AgentState) -> Action:
    """
    Recommend the next action based on current state.
    This is a helper for the Supervisor agent.
    """
    # Check for escalation first
    should_esc, _ = should_escalate(state)
    if should_esc:
        return Action.ESCALATE

    # Initial planning
    if not state.task_plan:
        return Action.PLAN

    # Initial scouting
    if not state.build_plan:
        return Action.SCOUT

    # Build execution
    if state.build_status == BuildStatus.PENDING:
        return Action.BUILDER

    # Error recovery
    if state.build_status == BuildStatus.FAILED:
        if state.last_error_category == ErrorCategory.DEPENDENCY:
            return Action.SCOUT  # Need more info
        else:
            return Action.FIXER  # Try to fix

    # Success
    if state.build_status == BuildStatus.SUCCESS:
        if not state.tests_run:
            return Action.BUILDER  # Run tests
        else:
            return Action.FINISH

    # Default to builder
    return Action.BUILDER

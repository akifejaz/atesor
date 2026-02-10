# RISC-V Porting Agent: Modern Architecture & Implementation Plan

> **Design Philosophy**: Hierarchical Multi-Agent System with Task Decomposition, Scripted Operations Layer, and Cost-Optimized LLM Usage

---

## 1. Core Architecture Improvements

### 1.1 Design Pattern: Hierarchical Planning with Execution Separation

Following modern agentic patterns, we implement a **3-tier architecture**:

```
┌─────────────────────────────────────────────────────────────────┐
│                     ORCHESTRATION LAYER                         │
│  ┌────────────┐     ┌────────────┐     ┌────────────┐          │
│  │  Planner   │────→│ Supervisor │────→│ Summarizer │          │
│  │  (LLM)     │     │   (LLM)    │     │   (LLM)    │          │
│  └────────────┘     └────────────┘     └────────────┘          │
│         │                   │                                   │
│         └───────────────────┼───────────────────┐               │
│                             ▼                   ▼               │
├─────────────────────────────────────────────────────────────────┤
│                     INTELLIGENCE LAYER                          │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐                │
│  │   Scout    │  │  Builder   │  │   Fixer    │                │
│  │   (LLM)    │  │   (LLM)    │  │   (LLM)    │                │
│  └────────────┘  └────────────┘  └────────────┘                │
│         │                │                │                     │
│         └────────────────┼────────────────┘                     │
│                          ▼                                      │
├─────────────────────────────────────────────────────────────────┤
│                     SCRIPTED OPERATIONS LAYER                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Direct Shell Operations (No LLM Cost)                    │  │
│  │  • Repository cloning/updates                             │  │
│  │  • File system operations (ls, tree, find)                │  │
│  │  • Content search (grep, sed, awk)                        │  │
│  │  • Dependency detection (parse package files)             │  │
│  │  • Build system detection (heuristic rules)               │  │
│  │  • Simple validation checks                               │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Key Improvements from Current Architecture

| Issue | Current Behavior | Improved Behavior |
|-------|-----------------|-------------------|
| **Excessive LLM Calls** | Agent called for every operation | Scripted layer handles 60% of operations |
| **Command Blocking** | Overzealous security blocks grep | Smart validation with command whitelisting |
| **Poor Context** | State not preserved between agents | Shared context store with history |
| **Premature Escalation** | Escalates without retry | Smart retry with exponential backoff |
| **Sequential Execution** | Everything runs in sequence | Parallel execution for independent tasks |
| **Lack of Planning** | Reactive decision making | Upfront task decomposition & planning |
| **No Caching** | Repeated identical operations | Results caching for identical commands |

---

## 2. Agent Hierarchy & Responsibilities

### 2.1 PLANNER (New - Cost Saver)

**Role**: Strategic task decomposition before execution

**When to Use**:
- Initial repository analysis
- After major blockers discovered
- When attempting count > 3

**Responsibilities**:
1. Decompose porting task into phases (Analysis → Build → Test → Fix)
2. Identify which tasks can be handled by scripts vs. agents
3. Create dependency graph of tasks
4. Estimate complexity and set expectations
5. Determine parallel vs. sequential execution

**Output**: `TaskPlan` with phases, dependencies, and execution strategy

**Cost Impact**: +1 LLM call upfront, -5 to -15 downstream calls

---

### 2.2 SUPERVISOR (Enhanced)

**Role**: Execution orchestrator with adaptive routing

**Enhancements**:
1. **State-Aware Routing**: Uses rich state context to make decisions
2. **Retry Logic**: Implements exponential backoff for failed operations
3. **Cost Tracking**: Monitors LLM API calls and switches to cheaper operations
4. **Parallel Dispatch**: Can trigger multiple agents in parallel for independent tasks

**Decision Tree**:
```python
def decide_next_action(state: AgentState) -> Action:
    # Phase 1: Initial Analysis (only if no plan exists)
    if not state.task_plan:
        return Action.PLAN
    
    # Phase 2: Execution based on current status
    if state.build_status == BuildStatus.PENDING:
        if not state.build_plan:
            return Action.SCOUT
        else:
            return Action.BUILDER
    
    # Phase 3: Error Recovery
    if state.build_status == BuildStatus.FAILED:
        error_cat = classify_error(state.last_error)
        
        if error_cat == ErrorCategory.DEPENDENCY:
            return Action.SCOUT  # Need more info
        elif error_cat in [ErrorCategory.COMPILATION, ErrorCategory.LINKING]:
            return Action.FIXER
        elif error_cat == ErrorCategory.ARCHITECTURE:
            if state.attempt_count < 3:
                return Action.FIXER
            else:
                return Action.SCOUT  # Re-analyze with more context
        else:
            return Action.FIXER
    
    # Phase 4: Verification
    if state.build_status == BuildStatus.SUCCESS:
        if not state.tests_run:
            return Action.BUILDER  # Run tests
        else:
            return Action.FINISH
    
    # Phase 5: Escalation
    if state.attempt_count >= state.max_attempts:
        return Action.ESCALATE
```

---

### 2.3 SCOUT (Enhanced)

**Role**: Deep analysis and build plan creation

**Enhanced Tools**:
- Access to scripted layer for fast file operations
- Parallel file reading for multiple README/docs
- Cached build system detection results

**New Features**:
1. **Incremental Analysis**: Only re-analyzes changed parts
2. **Build System Templates**: Uses known patterns for common build systems
3. **Dependency Graph**: Maps all dependencies and their RISC-V availability
4. **Community Intelligence**: Searches for existing RISC-V ports/patches

**Output Format** (Enhanced):
```json
{
  "build_system": "cmake",
  "build_system_confidence": 0.95,
  "dependencies": {
    "system": ["gcc", "make", "cmake"],
    "libraries": ["libssl-dev", "zlib1g-dev"],
    "risc_v_available": true,
    "install_method": "apt"
  },
  "build_plan": {
    "phases": [
      {
        "name": "dependency_install",
        "commands": ["apt-get update", "apt-get install -y gcc make cmake"],
        "can_parallelize": false,
        "expected_duration": "30s"
      },
      {
        "name": "configure",
        "commands": ["cmake -B build -DCMAKE_C_FLAGS='-march=rv64gc'"],
        "can_parallelize": false,
        "expected_duration": "10s"
      },
      {
        "name": "build",
        "commands": ["cmake --build build -j$(nproc)"],
        "can_parallelize": false,
        "expected_duration": "2m"
      }
    ]
  },
  "architecture_concerns": [
    {
      "type": "SIMD",
      "file": "src/simd.c",
      "line": 45,
      "issue": "Uses x86 SSE intrinsics",
      "severity": "high",
      "suggested_fix": "Use RVV (RISC-V Vector) or scalar fallback"
    }
  ],
  "risc_v_community_status": {
    "existing_port": false,
    "patches_available": true,
    "patch_urls": ["https://github.com/..."],
    "community_discussion": "https://github.com/.../issues/123"
  }
}
```

---

### 2.4 BUILDER (Enhanced)

**Role**: Execute build plan with smart error capture

**Enhancements**:
1. **Command Streaming**: Real-time output capture and analysis
2. **Early Error Detection**: Stops on first error instead of continuing
3. **Checkpoint System**: Can resume from last successful step
4. **Resource Monitoring**: Tracks CPU/memory usage during build

**Execution Strategy**:
```python
def execute_build_plan(plan: BuildPlan, state: AgentState) -> BuildResult:
    results = []
    
    for phase in plan.phases:
        # Check if phase can be resumed from checkpoint
        if state.last_successful_phase >= phase.id:
            continue
        
        # Execute phase
        phase_result = execute_phase(phase)
        results.append(phase_result)
        
        if phase_result.failed:
            # Immediate failure analysis
            error_details = analyze_error_immediately(phase_result)
            return BuildResult(
                success=False,
                failed_phase=phase.name,
                error=error_details,
                partial_results=results
            )
        
        # Update checkpoint
        state.last_successful_phase = phase.id
    
    return BuildResult(success=True, results=results)
```

---

### 2.5 FIXER (Enhanced with Reflection)

**Role**: Intelligent error resolution with learning

**Enhanced Capabilities**:
1. **Error Pattern Matching**: Uses database of known errors and fixes
2. **Multi-Strategy Fixing**: Tries 3-5 different approaches in parallel
3. **Patch Generation**: Creates minimal, targeted patches
4. **Self-Critique**: Validates fixes before applying

**Reflection Loop** (Pattern from Chapter 4):
```
┌─────────────┐
│  Analyze    │ ← Read error logs
│  Error      │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Generate   │ ← Create multiple fix strategies
│  Fix Plans  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Critique   │ ← Self-evaluate each strategy
│  Strategies │   (Reflection Agent)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Apply Best │ ← Execute top-ranked fix
│  Fix        │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Verify     │ ← Test if fix worked
│  Fix        │
└─────────────┘
```

---

## 3. Scripted Operations Layer (Cost Optimization)

### 3.1 Purpose

Reduce LLM API costs by 60-70% through intelligent automation of deterministic tasks.

### 3.2 Operations Handled by Scripts

```python
class ScriptedOperations:
    """
    Zero-cost operations that don't require LLM intelligence.
    """
    
    def detect_build_system(self, repo_path: str) -> BuildSystemInfo:
        """
        Heuristic detection based on file presence.
        Rules:
        - CMakeLists.txt → CMake
        - Makefile + configure.ac → Autotools
        - Cargo.toml → Cargo/Rust
        - meson.build → Meson
        - etc.
        """
        pass
    
    def extract_dependencies(self, repo_path: str, build_system: str) -> List[str]:
        """
        Parse dependency files without LLM:
        - requirements.txt → Python deps
        - package.json → Node deps
        - go.mod → Go deps
        - CMakeLists.txt find_package() → CMake deps
        """
        pass
    
    def find_architecture_specific_code(self, repo_path: str) -> List[ArchSpecificCode]:
        """
        Fast grep for known patterns:
        - __x86_64__, __amd64__, __SSE__, __AVX__
        - __ARM__, __aarch64__, __NEON__
        - Inline assembly blocks
        """
        pass
    
    def clone_or_update_repository(self, url: str, target: str) -> RepoInfo:
        """
        Smart git operations:
        - Check if already cloned
        - Update if exists
        - Shallow clone if possible
        """
        pass
    
    def cache_file_content(self, filepath: str) -> str:
        """
        Cache frequently accessed files:
        - README.md, INSTALL, BUILDING.md
        - Build system files
        - Common source files
        """
        pass
```

### 3.3 When to Use Scripts vs. LLM

| Operation | Use Script | Use LLM Agent |
|-----------|-----------|---------------|
| Clone repository | ✅ | ❌ |
| List files | ✅ | ❌ |
| Detect build system (simple) | ✅ | ❌ |
| Extract dependencies (standard formats) | ✅ | ❌ |
| Search for architecture code | ✅ | ❌ |
| Read documentation | ❌ | ✅ (Scout) |
| Understand complex build system | ❌ | ✅ (Scout) |
| Create build plan | ❌ | ✅ (Scout) |
| Fix compilation errors | ❌ | ✅ (Fixer) |
| Generate patches | ❌ | ✅ (Fixer) |

---

## 4. State Management (Enhanced)

### 4.1 Comprehensive State Schema

```python
@dataclass
class AgentState:
    # Repository info
    repo_url: str
    repo_name: str
    repo_path: str
    
    # Task planning
    task_plan: Optional[TaskPlan] = None
    current_phase: str = "initialization"
    
    # Build information
    build_system: Optional[str] = None
    build_plan: Optional[BuildPlan] = None
    build_status: BuildStatus = BuildStatus.PENDING
    
    # Execution tracking
    attempt_count: int = 0
    max_attempts: int = 5
    last_successful_phase: int = 0
    checkpoint_data: Dict[str, Any] = field(default_factory=dict)
    
    # Error handling
    last_error: Optional[str] = None
    last_error_category: Optional[ErrorCategory] = None
    error_history: List[ErrorRecord] = field(default_factory=list)
    fixes_attempted: List[FixAttempt] = field(default_factory=list)
    
    # Performance tracking
    api_calls_made: int = 0
    api_cost_usd: float = 0.0
    scripted_ops_count: int = 0
    execution_time_seconds: float = 0.0
    
    # Context and memory
    context_cache: Dict[str, Any] = field(default_factory=dict)
    file_content_cache: Dict[str, str] = field(default_factory=dict)
    command_results_cache: Dict[str, CommandResult] = field(default_factory=dict)
    
    # Agent communication
    messages: List[BaseMessage] = field(default_factory=list)
    agent_logs: Dict[str, List[str]] = field(default_factory=dict)
    
    # Output artifacts
    patches_generated: List[str] = field(default_factory=list)
    build_artifacts: List[str] = field(default_factory=list)
    porting_recipe: Optional[str] = None
```

### 4.2 Context Preservation

All agents share access to state, ensuring:
1. No repeated file reads
2. No redundant command executions
3. Build knowledge accumulation
4. Error pattern learning

---

## 5. Error Handling & Retry Logic (Robust)

### 5.1 Smart Retry Strategy

```python
class RetryStrategy:
    def should_retry(self, error: ErrorRecord, attempt: int) -> bool:
        """
        Exponential backoff with error-specific logic.
        """
        if error.category == ErrorCategory.RATE_LIMIT:
            # Always retry rate limits with backoff
            return attempt < 10
        
        if error.category == ErrorCategory.NETWORK:
            # Retry transient network errors
            return attempt < 5
        
        if error.category in [ErrorCategory.COMPILATION, ErrorCategory.LINKING]:
            # Limited retries for code issues
            return attempt < 3
        
        if error.category == ErrorCategory.ARCHITECTURE:
            # Complex issues need different approach each time
            return attempt < 5 and error not in previous_attempts
        
        return False
    
    def get_backoff_time(self, attempt: int, error_category: ErrorCategory) -> int:
        """
        Calculate wait time before retry.
        """
        base_delay = {
            ErrorCategory.RATE_LIMIT: 60,  # 1 minute base
            ErrorCategory.NETWORK: 5,      # 5 seconds base
            ErrorCategory.COMPILATION: 0,  # No delay
            ErrorCategory.DEPENDENCY: 10,  # 10 seconds base
        }.get(error_category, 10)
        
        return base_delay * (2 ** (attempt - 1))  # Exponential
```

### 5.2 Graceful Degradation

```python
class FallbackStrategy:
    """
    If free API fails, gracefully degrade functionality.
    """
    
    def handle_api_failure(self, provider: str) -> Action:
        if provider == "openrouter_free":
            # Try alternative free providers
            alternatives = ["gemini_free", "groq_free"]
            for alt in alternatives:
                if self.try_provider(alt):
                    return Action.CONTINUE
        
        # If all providers fail, use local LLM or rule-based system
        return self.use_fallback_mode()
    
    def use_fallback_mode(self) -> Action:
        """
        Operate with minimal LLM usage:
        - Use scripted operations exclusively
        - Use template-based build plans
        - Use rule-based error fixing
        """
        pass
```

---

## 6. Command Validation (Fixed)

### 6.1 Whitelist-Based Validation

```python
class CommandValidator:
    """
    Smart validation that doesn't block legitimate operations.
    """
    
    # Allowed command patterns
    SAFE_COMMANDS = {
        # File operations
        r'^ls\s+',
        r'^cat\s+',
        r'^head\s+',
        r'^tail\s+',
        r'^find\s+',
        r'^tree\s+',
        
        # Search operations (previously blocked)
        r'^grep\s+(-[a-zA-Z]+\s+)*',
        r'^awk\s+',
        r'^sed\s+',
        
        # Build operations
        r'^cmake\s+',
        r'^make\s+',
        r'^./configure\s+',
        r'^cargo\s+',
        r'^npm\s+',
        r'^pip\s+',
        
        # Package management
        r'^apt-get\s+',
        r'^apk\s+',
        
        # Git operations
        r'^git\s+',
    }
    
    # Dangerous patterns to block
    DANGEROUS_PATTERNS = {
        r'rm\s+-rf\s+/',  # Recursive root deletion
        r':\(\)\{\s*:\|:\&\s*\}',  # Fork bomb
        r'dd\s+if=/dev/zero\s+of=/dev/sd',  # Disk wipe
        r'mkfs\.',  # Format filesystem
        r'wget.*\|\s*bash',  # Remote code execution
        r'curl.*\|\s*sh',  # Remote code execution
    }
    
    def is_safe(self, command: str) -> Tuple[bool, str]:
        """
        Returns (is_safe, reason).
        """
        # Check dangerous patterns first
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, command):
                return False, f"Blocked dangerous pattern: {pattern}"
        
        # Check if matches safe patterns
        for pattern in self.SAFE_COMMANDS:
            if re.match(pattern, command):
                return True, "Matches safe command pattern"
        
        # Default deny for unknown patterns (but log for review)
        logger.warning(f"Unknown command pattern: {command}")
        return False, "Unknown command pattern (whitelist addition needed)"
```

---

## 7. Parallel Execution (Performance)

### 7.1 Task Parallelization

Identify independent tasks that can run concurrently:

```python
class ParallelExecutor:
    """
    Execute independent tasks in parallel to reduce total time.
    """
    
    async def execute_parallel_scout_tasks(self, repo_path: str) -> ScoutResults:
        """
        Scout can run multiple investigations simultaneously:
        """
        tasks = [
            self.read_documentation(repo_path),      # Task 1
            self.detect_build_system(repo_path),     # Task 2
            self.scan_for_arch_code(repo_path),      # Task 3
            self.check_dependencies(repo_path),      # Task 4
            self.search_community_ports(repo_url),   # Task 5 (network)
        ]
        
        results = await asyncio.gather(*tasks)
        return self.merge_scout_results(results)
    
    async def execute_parallel_fixes(self, error: ErrorRecord) -> List[FixResult]:
        """
        Fixer can try multiple strategies in parallel:
        """
        strategies = self.generate_fix_strategies(error)
        
        # Try top 3 strategies in parallel
        tasks = [
            self.apply_and_test_fix(strategy)
            for strategy in strategies[:3]
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Return first successful fix
        for result in results:
            if result.success:
                return result
        
        return results[0]  # Return best attempt even if all failed
```

---

## 8. Cost Tracking & Optimization

### 8.1 API Cost Monitoring

```python
class CostTracker:
    """
    Track and optimize LLM API costs.
    """
    
    COST_PER_1K_TOKENS = {
        "gpt-4": 0.03,
        "gpt-3.5-turbo": 0.001,
        "claude-3-sonnet": 0.003,
        "gemini-pro-free": 0.0,
        "openrouter-free": 0.0,
    }
    
    def log_api_call(self, model: str, input_tokens: int, output_tokens: int):
        """
        Track each API call cost.
        """
        total_tokens = input_tokens + output_tokens
        cost = (total_tokens / 1000) * self.COST_PER_1K_TOKENS.get(model, 0)
        
        self.total_cost += cost
        self.call_count += 1
        
        logger.info(f"API Call #{self.call_count}: {model}, "
                   f"{total_tokens} tokens, ${cost:.4f}")
    
    def should_use_cheaper_model(self) -> bool:
        """
        Switch to cheaper model if cost is too high.
        """
        return self.total_cost > self.budget_limit
```

### 8.2 Optimization Recommendations

Based on analysis of logs, typical porting task breakdown:

| Operation | Current LLM Calls | Optimized (Scripted) | Cost Savings |
|-----------|------------------|---------------------|--------------|
| Clone repo | 0 | 0 | - |
| List files | 3-5 | 0 | 100% |
| Detect build system | 1 | 0 | 100% |
| Extract dependencies | 2-3 | 0 | 100% |
| Search arch code | 2-4 | 0 | 100% |
| Read docs & analyze | 1-2 | 1 | 50% |
| Create build plan | 1 | 1 | 0% |
| Execute build | 1-3 | 1 | 66% |
| Fix errors | 3-8 | 2-4 | 50% |
| **Total** | **14-27** | **4-6** | **78%** |

---

## 9. Human-in-the-Loop (Smart Escalation)

### 9.1 Escalation Decision Tree

```python
class EscalationManager:
    """
    Intelligent escalation to human oversight.
    """
    
    def should_escalate(self, state: AgentState) -> Tuple[bool, str]:
        """
        Returns (should_escalate, reason).
        """
        # Escalate if max attempts reached
        if state.attempt_count >= state.max_attempts:
            return True, f"Max attempts ({state.max_attempts}) reached"
        
        # Escalate if same error repeating
        if self.is_error_loop(state.error_history):
            return True, "Stuck in error loop with no progress"
        
        # Escalate if fundamental blocker
        if state.last_error_category in [
            ErrorCategory.LICENSE_INCOMPATIBLE,
            ErrorCategory.REQUIRES_HARDWARE,
            ErrorCategory.ARCHITECTURE_IMPOSSIBLE,
        ]:
            return True, f"Fundamental blocker: {state.last_error_category}"
        
        # Escalate if cost limit exceeded
        if state.api_cost_usd > self.cost_limit:
            return True, f"API cost limit (${self.cost_limit}) exceeded"
        
        return False, ""
    
    def create_escalation_report(self, state: AgentState) -> EscalationReport:
        """
        Generate comprehensive report for human review.
        """
        return EscalationReport(
            summary=self.summarize_attempts(state),
            error_analysis=self.analyze_error_patterns(state),
            fixes_tried=state.fixes_attempted,
            current_blockers=self.identify_blockers(state),
            recommended_actions=self.suggest_human_actions(state),
            artifacts=[
                state.patches_generated,
                state.build_artifacts,
                state.agent_logs,
            ]
        )
```

---

## 10. Implementation Roadmap

### Phase 1: Core Infrastructure (Week 1)
- [ ] Implement Scripted Operations Layer
- [ ] Fix Command Validation (whitelist approach)
- [ ] Enhance State Management with caching
- [ ] Add Cost Tracker

### Phase 2: Agent Enhancements (Week 2)
- [ ] Implement Planner Agent
- [ ] Enhance Supervisor with smart routing
- [ ] Improve Scout with parallel operations
- [ ] Enhance Fixer with reflection pattern

### Phase 3: Optimization (Week 3)
- [ ] Implement parallel execution
- [ ] Add retry/fallback strategies
- [ ] Optimize API usage
- [ ] Add performance metrics

### Phase 4: Testing & Refinement (Week 4)
- [ ] Test on 20+ different packages
- [ ] Measure cost savings
- [ ] Refine escalation logic
- [ ] Create comprehensive documentation

---

## 11. Success Metrics

### 11.1 Performance Targets

| Metric | Current | Target |
|--------|---------|--------|
| Average API calls per package | 14-27 | 4-6 |
| Cost per package (free tier) | $0 (but hitting limits) | $0 (sustainable) |
| Cost per package (paid tier) | ~$0.50 | ~$0.10 |
| Success rate (simple packages) | 70% | 95% |
| Success rate (complex packages) | 30% | 60% |
| Time to first build attempt | 5-8 min | 2-3 min |
| Total porting time | 15-30 min | 8-15 min |

### 11.2 Quality Targets

- **Code Quality**: Generated patches are minimal and targeted
- **Documentation**: Complete porting recipes for every success
- **Reliability**: No crashes or unhandled exceptions
- **Maintainability**: Clear separation of concerns, modular design

---

## 12. Compatibility Notes

### 12.1 Multi-Provider Support

All improvements maintain compatibility with:
- ✅ OpenAI (GPT-3.5, GPT-4)
- ✅ Google Gemini (Free & Paid)
- ✅ OpenRouter (Free & Paid)
- ✅ Anthropic Claude (via OpenRouter)
- ✅ Local LLMs (Ollama, LM Studio)

### 12.2 Provider-Specific Optimizations

```python
class ProviderOptimizer:
    """
    Adjust strategy based on provider capabilities.
    """
    
    def optimize_for_provider(self, provider: str) -> Config:
        if provider == "gemini_free":
            return Config(
                max_tokens=8000,  # Gemini has large context
                use_caching=True,  # Gemini supports caching
                temperature=0.1,   # Lower temp for free tier
            )
        elif provider == "openrouter_free":
            return Config(
                max_tokens=4000,
                use_caching=False,
                temperature=0.0,  # Deterministic for consistency
                rate_limit_delay=2,  # Be gentle with free tier
            )
        elif provider == "openai":
            return Config(
                max_tokens=4000,
                use_caching=False,
                temperature=0.2,
                enable_function_calling=True,  # OpenAI excels here
            )
```

---

## 13. References

- **Andrew Ng's Agentic AI Patterns**: Reflection, Planning, Multi-Agent, Tool Use
- **LangGraph Documentation**: Multi-agent workflows, state management
- **Google's Gemini API**: Free tier best practices
- **OpenRouter Documentation**: Free model optimization
- **RISC-V ISA Manual**: Architecture-specific considerations

---

**Next Step**: Review this plan and proceed to implementation of updated `graph.py`, `state.py`, `tools.py`, and new `scripted_ops.py`.

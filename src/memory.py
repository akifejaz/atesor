"""
Agent Memory System - Few-shot Learning & Self-Learning for Multi-Agent RISC-V Porting.

Features:
1. Compact example format per agent (scout, fixer, builder)
2. Keyword + regex-based relevance matching
3. Auto-learning: agents save novel successful patterns back to examples
4. Recipe cache: incremental registry of successfully-built packages
5. Deduplication and pruning (max 100 per agent, oldest auto-learned first)
"""

import json
import os
import logging
import re
import hashlib
import filelock
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

EXAMPLES_DIR = Path(__file__).parent.parent / "data" / "examples"
RECIPE_CACHE_PATH = Path(__file__).parent.parent / "data" / "recipe_cache.json"
MAX_EXAMPLES_PER_AGENT = 100


@dataclass
class AgentExample:
    """A single few-shot example for an agent."""

    id: str
    name: str
    tags: List[str]
    build_system: str = ""
    source: str = "manual"
    repo_name: str = ""
    timestamp: str = ""

    # Scout-specific
    trigger: Optional[Dict[str, Any]] = None
    plan: Optional[Dict[str, Any]] = None

    # Fixer-specific
    error_pattern: str = ""
    fix: Optional[Dict[str, Any]] = None

    # Builder-specific
    phases: Optional[List[Dict[str, Any]]] = None
    timeout_recommendation: str = ""

    # Legacy compat
    context: Dict[str, Any] = field(default_factory=dict)
    expected_output: Optional[Dict[str, Any]] = None
    solution: Optional[Dict[str, Any]] = None
    execution: Optional[Dict[str, Any]] = None
    reasoning: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_prompt_text(self, example_type: str) -> str:
        """Convert example to compact text for prompt inclusion."""
        if example_type == "scout":
            return self._to_scout_prompt()
        elif example_type == "fixer":
            return self._to_fixer_prompt()
        elif example_type == "builder":
            return self._to_builder_prompt()
        return ""

    def _to_scout_prompt(self) -> str:
        plan_data = self.plan or self.expected_output or {}
        phases = plan_data.get("phases", [])
        trigger = self.trigger or self.context or {}

        phases_json = []
        for p in phases:
            phases_json.append({
                "name": p.get("name", "unknown"),
                "commands": p.get("commands", [])
            })

        return (
            f"## {self.name}\n"
            f"Build: {self.build_system or trigger.get('build_system', '?')} | "
            f"Main: {trigger.get('main_path', trigger.get('has_main', '?'))} | "
            f"ModuleDir: {trigger.get('module_dir', 'root')}\n"
            f"```json\n{json.dumps({'phases': phases_json}, indent=2)}\n```\n"
            f"Why: {self.reasoning}\n"
        )

    def _to_fixer_prompt(self) -> str:
        fix_data = self.fix or self.solution or {}
        error_ctx = self.raw.get("error_context", {})
        error_msg = error_ctx.get("error_message", self.error_pattern)[:120]

        actions_text = ""
        for action in fix_data.get("actions", []):
            atype = action.get("type", "")
            if atype == "command":
                actions_text += f"\n  - `{action.get('command', '')}`"
            elif atype == "create_file":
                actions_text += f"\n  - create: {action.get('path', '')}"
            elif atype == "patch":
                actions_text += f"\n  - patch: {action.get('file', 'repo')}"

        return (
            f"## {self.name}\n"
            f"Error: {error_msg}\n"
            f"Strategy: {fix_data.get('strategy', fix_data.get('analysis', 'N/A'))}\n"
            f"Actions:{actions_text}\n"
            f"Why: {self.reasoning}\n"
        )

    def _to_builder_prompt(self) -> str:
        ctx = self.context or {}
        phases = self.phases or ctx.get("phases", [])

        cmds = []
        for p in phases:
            for c in p.get("commands", []):
                cmds.append(c)

        return (
            f"## {self.name}\n"
            f"Build: {self.build_system or ctx.get('build_system', '?')} | "
            f"Timeout: {self.timeout_recommendation or 'default'}\n"
            f"Commands: {', '.join(cmds[:4])}\n"
            f"Notes: {self.reasoning or self.raw.get('notes', '')}\n"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON storage."""
        d: Dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "tags": self.tags,
            "build_system": self.build_system,
            "source": self.source,
            "repo_name": self.repo_name,
            "timestamp": self.timestamp or datetime.now().strftime("%Y-%m-%d"),
        }
        if self.trigger:
            d["trigger"] = self.trigger
        if self.plan:
            d["plan"] = self.plan
        if self.error_pattern:
            d["error_pattern"] = self.error_pattern
        if self.fix:
            d["fix"] = self.fix
        if self.phases:
            d["phases"] = self.phases
        if self.timeout_recommendation:
            d["timeout_recommendation"] = self.timeout_recommendation
        if self.reasoning:
            d["reasoning"] = self.reasoning
        if self.context:
            d["context"] = self.context
        if self.expected_output:
            d["expected_output"] = self.expected_output
        if self.solution:
            d["solution"] = self.solution
        if self.execution:
            d["execution"] = self.execution
        return d


class AgentMemory:
    """Lightweight memory system for few-shot learning with auto-learning support."""

    def __init__(self, agent_type: str, examples_dir: Path = EXAMPLES_DIR):
        self.agent_type = agent_type
        self.examples_dir = examples_dir
        self.examples: List[AgentExample] = []
        self._filepath = self.examples_dir / f"{self.agent_type}_examples.json"
        self._load_examples()

    def reload(self):
        """Reload examples from disk (call after auto-learning writes)."""
        self.examples = []
        self._load_examples()

    def _load_examples(self):
        """Load examples from JSON file, supporting both new and legacy formats."""
        if not self._filepath.exists():
            logger.warning(f"No examples file found: {self._filepath}")
            return

        try:
            with open(self._filepath, "r") as f:
                data = json.load(f)

            for ex_data in data.get("examples", []):
                example = self._parse_example(ex_data)
                self.examples.append(example)

            logger.info(f"Loaded {len(self.examples)} examples for {self.agent_type}")
        except Exception as e:
            logger.error(f"Failed to load examples for {self.agent_type}: {e}")

    def _parse_example(self, ex_data: Dict[str, Any]) -> AgentExample:
        """Parse a single example dict into AgentExample, handling both formats."""
        return AgentExample(
            id=ex_data.get("id", ""),
            name=ex_data.get("name", ""),
            tags=ex_data.get("tags", []),
            build_system=ex_data.get("build_system",
                                     ex_data.get("context", {}).get("build_system", "")),
            source=ex_data.get("source", "manual"),
            repo_name=ex_data.get("repo_name",
                                  ex_data.get("context", {}).get("repo_name", "")),
            timestamp=ex_data.get("timestamp", ""),
            trigger=ex_data.get("trigger"),
            plan=ex_data.get("plan"),
            error_pattern=ex_data.get("error_pattern", ""),
            fix=ex_data.get("fix"),
            phases=ex_data.get("phases"),
            timeout_recommendation=ex_data.get("timeout_recommendation", ""),
            context=ex_data.get("context", {}),
            expected_output=ex_data.get("expected_output"),
            solution=ex_data.get("solution"),
            execution=ex_data.get("execution"),
            reasoning=ex_data.get("reasoning", ""),
            raw=ex_data,
        )

    def get_relevant_examples(
        self,
        context: Dict[str, Any],
        max_examples: int = 3,
    ) -> List[AgentExample]:
        """Get top-K relevant examples based on context matching."""
        if not self.examples:
            return []

        scored = [
            (self._calculate_relevance(ex, context), ex)
            for ex in self.examples
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ex for _, ex in scored[:max_examples]]

    def _calculate_relevance(
        self, example: AgentExample, context: Dict[str, Any]
    ) -> float:
        """Calculate relevance score. Supports keyword + regex matching."""
        score = 0.0
        ex_bs = example.build_system or example.context.get("build_system", "")
        ex_trigger = example.trigger or example.context or {}
        ex_tags = set(t.lower() for t in example.tags)

        # Build system match (strongest signal)
        if context.get("build_system") and context["build_system"] == ex_bs:
            score += 0.5

        # Error pattern regex match (fixer-specific, very strong)
        if context.get("error_message") and example.error_pattern:
            try:
                if re.search(example.error_pattern, context["error_message"],
                             re.IGNORECASE):
                    score += 0.4
            except re.error:
                pass

        # Tag-in-error matching
        if context.get("error_message"):
            error_lower = context["error_message"].lower()
            for tag in ex_tags:
                if tag in error_lower:
                    score += 0.15

        # has_main match
        if context.get("has_main") is not None:
            ex_has_main = ex_trigger.get("has_main",
                                         example.context.get("has_main"))
            if context["has_main"] == ex_has_main:
                score += 0.1

        # module_dir match
        ctx_md = context.get("module_dir", "")
        ex_md = ex_trigger.get("module_dir",
                               example.context.get("module_dir", ""))
        if ctx_md and ex_md and ctx_md == ex_md:
            score += 0.15
        elif ctx_md and ex_md:
            score += 0.05

        # CGO tag
        if context.get("has_cgo") and "cgo" in ex_tags:
            score += 0.2

        return min(score, 1.0)

    def format_examples_for_prompt(
        self, examples: List[AgentExample], max_chars: int = 3000
    ) -> str:
        """Format examples for inclusion in prompt."""
        if not examples:
            return ""

        sections = [f"# Few-Shot Examples ({self.agent_type})\n"]
        total_chars = 0

        for example in examples:
            text = example.to_prompt_text(self.agent_type)
            if total_chars + len(text) > max_chars:
                break
            sections.append(text)
            total_chars += len(text)

        return "\n".join(sections)

    def save_learned_example(self, example_data: Dict[str, Any]) -> bool:
        """
        Save an auto-learned example to the JSON file.
        Returns True if saved, False if duplicate or invalid.
        """
        if not example_data.get("name") or not example_data.get("build_system"):
            logger.warning("Cannot save example: missing name or build_system")
            return False

        if self._is_duplicate(example_data):
            logger.info(f"Skipping duplicate example: {example_data.get('name')}")
            return False

        auto_ids = [
            ex.id for ex in self.examples
            if ex.id.startswith(f"{self.agent_type}-auto-")
        ]
        next_num = len(auto_ids) + 1
        example_data["id"] = f"{self.agent_type}-auto-{next_num:03d}"
        example_data["source"] = "auto"
        example_data["timestamp"] = datetime.now().strftime("%Y-%m-%d")

        lock_path = str(self._filepath) + ".lock"
        try:
            lock = filelock.FileLock(lock_path, timeout=5)
            with lock:
                data = self._read_json_file()
                examples_list = data.get("examples", [])
                examples_list.append(example_data)

                examples_list = self._prune_examples_list(examples_list)
                data["examples"] = examples_list
                self._write_json_file(data)

            self.reload()
            logger.info(f"Saved learned example: {example_data['id']} - {example_data['name']}")
            return True
        except Exception as e:
            logger.error(f"Failed to save learned example: {e}")
            return False

    def _is_duplicate(self, new_data: Dict[str, Any]) -> bool:
        """Check if an example is a duplicate of existing ones."""
        new_bs = new_data.get("build_system", "")
        new_repo = new_data.get("repo_name", "")

        for ex in self.examples:
            ex_bs = ex.build_system or ex.context.get("build_system", "")
            ex_repo = ex.repo_name or ex.context.get("repo_name", "")

            if ex_repo and new_repo and ex_repo == new_repo and ex_bs == new_bs:
                return True

            if (self.agent_type == "fixer"
                and new_data.get("error_pattern")
                and ex.error_pattern
                and new_data["error_pattern"] == ex.error_pattern
                and ex_bs == new_bs):
                return True

            if self.agent_type in ("scout", "builder"):
                new_cmds = self._extract_commands(new_data)
                ex_cmds = self._extract_commands(ex.raw if ex.raw else ex.to_dict())
                if new_cmds and ex_cmds and new_cmds == ex_cmds:
                    return True

        return False

    def _extract_commands(self, data: Dict[str, Any]) -> str:
        """Extract a fingerprint of commands from an example for dedup."""
        cmds = []
        for phase in data.get("phases", []):
            cmds.extend(phase.get("commands", []))
        plan = data.get("plan", {})
        if plan:
            for phase in plan.get("phases", []):
                cmds.extend(phase.get("commands", []))
        output = data.get("expected_output", {})
        if output:
            for phase in output.get("phases", []):
                cmds.extend(phase.get("commands", []))
        if not cmds:
            return ""
        normalized = "|".join(sorted(cmds))
        return hashlib.md5(normalized.encode()).hexdigest()

    def _prune_examples_list(self, examples: List[Dict]) -> List[Dict]:
        """Prune to MAX_EXAMPLES_PER_AGENT. Remove oldest auto-learned first."""
        if len(examples) <= MAX_EXAMPLES_PER_AGENT:
            return examples

        manual = [e for e in examples if e.get("source", "manual") == "manual"]
        auto = [e for e in examples if e.get("source", "manual") == "auto"]

        auto.sort(key=lambda e: e.get("timestamp", ""))
        overflow = len(examples) - MAX_EXAMPLES_PER_AGENT
        if overflow > 0 and len(auto) >= overflow:
            auto = auto[overflow:]
        elif overflow > 0:
            auto = []

        return manual + auto

    def _read_json_file(self) -> Dict[str, Any]:
        if not self._filepath.exists():
            return {
                "version": "2.0",
                "description": f"Examples for {self.agent_type} agent",
                "examples": []
            }
        with open(self._filepath, "r") as f:
            return json.load(f)

    def _write_json_file(self, data: Dict[str, Any]):
        os.makedirs(self._filepath.parent, exist_ok=True)
        with open(self._filepath, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


# ============================================================================
# RECIPE CACHE
# ============================================================================

def load_recipe_cache() -> Dict[str, Any]:
    """Load the recipe cache from disk."""
    if not RECIPE_CACHE_PATH.exists():
        return {"version": "1.0", "packages": {}}
    try:
        with open(RECIPE_CACHE_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load recipe cache: {e}")
        return {"version": "1.0", "packages": {}}


def get_cached_recipe(repo_name: str, architecture: str = "riscv64") -> Optional[Dict[str, Any]]:
    """Look up a cached recipe for a package + architecture."""
    cache = load_recipe_cache()
    entry = cache.get("packages", {}).get(repo_name)
    if entry and entry.get("architecture") == architecture:
        return entry
    return None


def save_to_recipe_cache(
    repo_name: str,
    repo_url: str,
    build_system: str,
    build_plan: Dict[str, Any],
    dependencies: List[str],
    patches: List[str],
    artifacts: List[Dict[str, Any]],
    build_duration_seconds: float,
    architecture: str = "riscv64",
    sandbox: str = "alpine-riscv64",
) -> bool:
    """Upsert a package entry in the recipe cache."""
    lock_path = str(RECIPE_CACHE_PATH) + ".lock"
    try:
        lock = filelock.FileLock(lock_path, timeout=5)
        with lock:
            cache = load_recipe_cache()
            packages = cache.setdefault("packages", {})

            compact_phases = []
            for phase in build_plan.get("phases", []):
                compact_phases.append({
                    "name": phase.get("name", "unknown"),
                    "commands": phase.get("commands", []),
                })

            packages[repo_name] = {
                "repo_url": repo_url,
                "build_system": build_system,
                "architecture": architecture,
                "sandbox": sandbox,
                "last_built": datetime.now().isoformat(),
                "build_plan": {"phases": compact_phases},
                "dependencies": dependencies,
                "patches": patches[:10],
                "artifacts": [
                    {"type": a.get("type", "binary"), "path": a.get("filepath", "")}
                    for a in artifacts[:20]
                ],
                "build_duration_seconds": round(build_duration_seconds, 1),
                "recipe_file": f"output/{repo_name}_recipe.md",
            }

            os.makedirs(RECIPE_CACHE_PATH.parent, exist_ok=True)
            with open(RECIPE_CACHE_PATH, "w") as f:
                json.dump(cache, f, indent=2, ensure_ascii=False)

        logger.info(f"Recipe cache updated for {repo_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to save recipe cache for {repo_name}: {e}")
        return False


# ============================================================================
# SINGLETON + CONVENIENCE FUNCTIONS
# ============================================================================

MEMORY_INSTANCES: Dict[str, AgentMemory] = {}


def get_agent_memory(agent_type: str) -> AgentMemory:
    """Get or create memory instance for an agent type."""
    if agent_type not in MEMORY_INSTANCES:
        MEMORY_INSTANCES[agent_type] = AgentMemory(agent_type)
    return MEMORY_INSTANCES[agent_type]


def reload_agent_memory(agent_type: str):
    """Force reload an agent's memory (after auto-learning writes)."""
    if agent_type in MEMORY_INSTANCES:
        MEMORY_INSTANCES[agent_type].reload()
    else:
        MEMORY_INSTANCES[agent_type] = AgentMemory(agent_type)


def format_few_shot_examples(
    agent_type: str,
    context: Dict[str, Any],
    max_examples: int = 2,
    max_chars: int = 2500,
) -> str:
    """
    Convenience function to get formatted few-shot examples.

    Args:
        agent_type: 'scout', 'fixer', 'builder', or 'supervisor'
        context: Current context for relevance matching
        max_examples: Maximum examples to include
        max_chars: Maximum characters for examples section

    Returns:
        Formatted string for prompt inclusion
    """
    memory = get_agent_memory(agent_type)
    examples = memory.get_relevant_examples(context, max_examples)
    return memory.format_examples_for_prompt(examples, max_chars)


def save_learned_example(agent_type: str, example_data: Dict[str, Any]) -> bool:
    """Convenience function to save a learned example."""
    memory = get_agent_memory(agent_type)
    return memory.save_learned_example(example_data)

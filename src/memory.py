"""
Agent Memory System - Few-shot Learning for Multi-Agent RISC-V Porting.

This module provides a lightweight memory system for agents that uses:
1. Keyword-based matching for fast retrieval
2. Optional embedding similarity for semantic search
3. Consistent example format across all agents

Based on best practices from LangGraph memory patterns and agentic design research.
"""

import json
import os
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import re

logger = logging.getLogger(__name__)

EXAMPLES_DIR = Path(__file__).parent.parent / "data" / "examples"


@dataclass
class AgentExample:
    """A single few-shot example for an agent."""

    id: str
    name: str
    tags: List[str]
    context: Dict[str, Any]
    expected_output: Optional[Dict[str, Any]] = None
    solution: Optional[Dict[str, Any]] = None
    execution: Optional[Dict[str, Any]] = None
    reasoning: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_prompt_text(self, example_type: str) -> str:
        """Convert example to text format for prompt inclusion."""
        if example_type == "scout":
            return self._to_scout_prompt()
        elif example_type == "fixer":
            return self._to_fixer_prompt()
        elif example_type == "builder":
            return self._to_builder_prompt()
        return ""

    def _to_scout_prompt(self) -> str:
        """Format as Scout example."""
        ctx = self.context
        output = self.expected_output or {}
        phases_text = ""
        for phase in output.get("phases", []):
            cmds = "\n    ".join(phase.get("commands", []))
            phases_text += f"""
    {{
      "id": {phase.get("id", 1)},
      "name": "{phase.get("name", "unknown")}",
      "commands": [
    {cmds}
      ]
    }},"""

        return f"""
## Example: {self.name}
**Tags**: {", ".join(self.tags)}
**Context**: {ctx.get("repo_name", "unknown")} | Build: {ctx.get("build_system", "unknown")} | Main: {ctx.get("main_path", "N/A")} | Module Dir: {ctx.get("module_dir", "root")}

**Expected Build Plan**:
```json
{{
  "build_system": "{output.get("build_system", "unknown")}",
  "module_dir": "{output.get("module_dir", "")}",
  "phases": [{phases_text}
  ],
  "notes": {json.dumps(output.get("notes", []))}
}}
```
**Why**: {self.reasoning}
"""

    def _to_fixer_prompt(self) -> str:
        """Format as Fixer example."""
        ctx = self.raw.get("error_context", {})
        sol = self.solution or {}
        actions_text = ""
        for action in sol.get("actions", []):
            if action.get("type") == "command":
                actions_text += f"\n  - Run: `{action.get('command')}`"
            elif action.get("type") == "patch":
                actions_text += f"\n  - Patch: {action.get('file', 'repo')}"

        return f"""
## Example: {self.name}
**Error**: {ctx.get("category", "Unknown")} - {ctx.get("error_message", "")[:100]}...
**Analysis**: {sol.get("analysis", "N/A")}
**Strategy**: {sol.get("strategy", "N/A")}
**Actions**:{actions_text}
**Why**: {self.reasoning}
"""

    def _to_builder_prompt(self) -> str:
        """Format as Builder example."""
        ctx = self.context
        exec_data = self.execution or {}

        results_text = ""
        for key, value in exec_data.items():
            if key.endswith("_result"):
                results_text += f"\n  - {key}: {'SUCCESS' if value.get('success') else 'FAILED'} ({value.get('duration', 'unknown')})"

        return f"""
## Example: {self.name}
**Build System**: {ctx.get("build_system", "unknown")}
**Phases**: {len(ctx.get("phases", []))}
**Results**:{results_text}
**Notes**: {exec_data.get("notes", self.reasoning)}
"""


class AgentMemory:
    """
    Lightweight memory system for few-shot learning.

    Features:
    - Loads examples from JSON files
    - Keyword-based relevance matching
    - Optional embedding similarity (lazy-loaded)
    - Caches frequently used examples
    """

    def __init__(self, agent_type: str, examples_dir: Path = EXAMPLES_DIR):
        self.agent_type = agent_type
        self.examples_dir = examples_dir
        self.examples: List[AgentExample] = []
        self._embedder = None
        self._example_embeddings = {}

        self._load_examples()

    def _load_examples(self):
        """Load examples from JSON file."""
        filename = f"{self.agent_type}_examples.json"
        filepath = self.examples_dir / filename

        if not filepath.exists():
            logger.warning(f"No examples file found: {filepath}")
            return

        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            for ex_data in data.get("examples", []):
                example = AgentExample(
                    id=ex_data.get("id", ""),
                    name=ex_data.get("name", ""),
                    tags=ex_data.get("tags", []),
                    context=ex_data.get("context", {}),
                    expected_output=ex_data.get("expected_output"),
                    solution=ex_data.get("solution"),
                    execution=ex_data.get("execution"),
                    reasoning=ex_data.get("reasoning", ""),
                    raw=ex_data,
                )
                self.examples.append(example)

            logger.info(f"Loaded {len(self.examples)} examples for {self.agent_type}")

        except Exception as e:
            logger.error(f"Failed to load examples: {e}")

    def get_relevant_examples(
        self,
        context: Dict[str, Any],
        max_examples: int = 3,
        use_embeddings: bool = False,
    ) -> List[AgentExample]:
        """
        Get relevant examples based on context matching.

        Args:
            context: Current context (build_system, error_type, etc.)
            max_examples: Maximum examples to return
            use_embeddings: Whether to use embedding similarity

        Returns:
            List of relevant examples
        """
        if not self.examples:
            return []

        scored_examples = []

        for example in self.examples:
            score = self._calculate_relevance(example, context)
            scored_examples.append((score, example))

        scored_examples.sort(key=lambda x: x[0], reverse=True)

        return [ex for score, ex in scored_examples[:max_examples]]

    def _calculate_relevance(
        self, example: AgentExample, context: Dict[str, Any]
    ) -> float:
        """Calculate relevance score based on keyword matching."""
        score = 0.0

        ex_context = example.context
        ex_tags = set(example.tags)

        if context.get("build_system") == ex_context.get("build_system"):
            score += 0.5

        if context.get("error_message"):
            error_lower = context["error_message"].lower()
            for tag in ex_tags:
                if tag.lower() in error_lower:
                    score += 0.2

        if context.get("has_main") is not None:
            if context.get("has_main") == ex_context.get("has_main"):
                score += 0.1

        if context.get("module_dir") and ex_context.get("module_dir"):
            if context["module_dir"] == ex_context["module_dir"]:
                score += 0.15
            elif context["module_dir"] and ex_context["module_dir"]:
                score += 0.05

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


MEMORY_INSTANCES: Dict[str, AgentMemory] = {}


def get_agent_memory(agent_type: str) -> AgentMemory:
    """Get or create memory instance for an agent type."""
    if agent_type not in MEMORY_INSTANCES:
        MEMORY_INSTANCES[agent_type] = AgentMemory(agent_type)
    return MEMORY_INSTANCES[agent_type]


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

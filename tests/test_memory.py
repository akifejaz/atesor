"""
Unit tests for the Agent Memory System.
"""

import pytest
import json
import os
import tempfile
import shutil
from pathlib import Path

from src.memory import (
    AgentMemory,
    AgentExample,
    format_few_shot_examples,
    get_agent_memory,
    save_learned_example,
    reload_agent_memory,
    load_recipe_cache,
    get_cached_recipe,
    save_to_recipe_cache,
    MEMORY_INSTANCES,
    RECIPE_CACHE_PATH,
)


class TestAgentExample:
    """Tests for AgentExample dataclass."""

    def test_example_creation(self):
        """Test creating an example with new fields."""
        example = AgentExample(
            id="test-001",
            name="Test Example",
            tags=["go", "test"],
            build_system="go",
            source="manual",
            repo_name="my-repo",
            reasoning="Test reasoning",
        )
        assert example.id == "test-001"
        assert example.tags == ["go", "test"]
        assert example.build_system == "go"
        assert example.source == "manual"
        assert example.repo_name == "my-repo"

    def test_example_creation_legacy(self):
        """Test creating an example with legacy context field."""
        example = AgentExample(
            id="test-002",
            name="Legacy Example",
            tags=["go"],
            context={"build_system": "go", "repo_name": "legacy"},
            reasoning="Legacy reasoning",
        )
        assert example.context["build_system"] == "go"

    def test_scout_prompt_format(self):
        """Test formatting example as Scout prompt (new format)."""
        example = AgentExample(
            id="scout-001",
            name="Test Scout",
            tags=["go"],
            build_system="go",
            trigger={"build_system": "go", "has_main": True, "main_path": "."},
            plan={"build_system": "go", "phases": [
                {"name": "build", "commands": ["go build ."]}
            ]},
            reasoning="Because test",
        )
        text = example.to_prompt_text("scout")
        assert "Test Scout" in text
        assert "go" in text
        assert "Because test" in text
        assert "go build ." in text

    def test_scout_prompt_format_legacy(self):
        """Test formatting example as Scout prompt (legacy format)."""
        example = AgentExample(
            id="scout-002",
            name="Legacy Scout",
            tags=["go"],
            context={"build_system": "go", "repo_name": "test"},
            expected_output={"build_system": "go", "phases": [
                {"name": "build", "commands": ["go build ."]}
            ]},
            reasoning="Legacy",
        )
        text = example.to_prompt_text("scout")
        assert "Legacy Scout" in text
        assert "go build ." in text

    def test_fixer_prompt_format(self):
        """Test formatting example as Fixer prompt (new format)."""
        example = AgentExample(
            id="fixer-001",
            name="Test Fixer",
            tags=["go", "error"],
            build_system="go",
            error_pattern="go: command not found",
            fix={"strategy": "Install Go", "actions": [
                {"type": "command", "command": "apk add go"}
            ]},
            reasoning="Test reasoning",
            raw={"error_context": {"category": "TEST", "error_message": "go: command not found"}},
        )
        text = example.to_prompt_text("fixer")
        assert "Test Fixer" in text
        assert "Install Go" in text
        assert "apk add go" in text

    def test_builder_prompt_format(self):
        """Test formatting example as Builder prompt (new format)."""
        example = AgentExample(
            id="builder-001",
            name="Test Builder",
            tags=["go"],
            build_system="go",
            phases=[
                {"name": "build", "commands": ["go build ."]}
            ],
            timeout_recommendation="60s",
            reasoning="Simple build",
        )
        text = example.to_prompt_text("builder")
        assert "Test Builder" in text
        assert "go" in text
        assert "60s" in text

    def test_to_dict(self):
        """Test serialization to dict."""
        example = AgentExample(
            id="test-003",
            name="Serialize Test",
            tags=["go"],
            build_system="go",
            source="auto",
            repo_name="test-repo",
            trigger={"build_system": "go"},
            plan={"phases": [{"name": "build", "commands": ["go build ."]}]},
            reasoning="Serializable",
        )
        d = example.to_dict()
        assert d["id"] == "test-003"
        assert d["build_system"] == "go"
        assert d["source"] == "auto"
        assert "trigger" in d
        assert "plan" in d


class TestAgentMemory:
    """Tests for AgentMemory class."""

    def test_load_scout_examples(self):
        """Test loading scout examples from file (new v2.0 format)."""
        memory = AgentMemory("scout")
        assert len(memory.examples) > 0
        assert all(isinstance(ex, AgentExample) for ex in memory.examples)
        # Check new fields are populated
        for ex in memory.examples:
            assert ex.build_system != ""

    def test_load_fixer_examples(self):
        """Test loading fixer examples from file."""
        memory = AgentMemory("fixer")
        assert len(memory.examples) > 0
        # Check error_pattern is populated for manual examples
        has_pattern = any(ex.error_pattern for ex in memory.examples)
        assert has_pattern

    def test_load_builder_examples(self):
        """Test loading builder examples from file."""
        memory = AgentMemory("builder")
        assert len(memory.examples) > 0
        has_phases = any(ex.phases for ex in memory.examples)
        assert has_phases

    def test_get_relevant_examples(self):
        """Test getting relevant examples based on context."""
        memory = AgentMemory("scout")
        context = {"build_system": "go", "has_main": True}
        examples = memory.get_relevant_examples(context, max_examples=2)
        assert len(examples) <= 2
        for ex in examples:
            assert ex.build_system == "go" or "go" in ex.tags

    def test_relevance_scoring_build_system(self):
        """Test relevance scoring prioritizes build system match."""
        memory = AgentMemory("scout")

        context_go = {"build_system": "go"}
        examples_go = memory.get_relevant_examples(context_go, max_examples=3)

        context_cmake = {"build_system": "cmake"}
        examples_cmake = memory.get_relevant_examples(context_cmake, max_examples=3)

        assert len(examples_go) > 0
        assert len(examples_cmake) > 0
        # Go examples should be first for go context
        assert examples_go[0].build_system == "go"

    def test_relevance_scoring_error_pattern(self):
        """Test regex error_pattern matching in fixer relevance scoring."""
        memory = AgentMemory("fixer")
        context = {"build_system": "go", "error_message": "go: command not found"}
        examples = memory.get_relevant_examples(context, max_examples=3)
        assert len(examples) > 0
        # The "Missing Go Command" example should score high
        names = [ex.name for ex in examples]
        assert any("Go" in name or "command" in name.lower() for name in names)

    def test_format_examples_for_prompt(self):
        """Test formatting examples for prompt."""
        memory = AgentMemory("scout")
        context = {"build_system": "go"}
        examples = memory.get_relevant_examples(context, max_examples=2)
        formatted = memory.format_examples_for_prompt(examples, max_chars=1000)
        assert "# Few-Shot Examples" in formatted
        assert len(formatted) <= 1200

    def test_reload(self):
        """Test reloading examples from disk."""
        memory = AgentMemory("scout")
        initial_count = len(memory.examples)
        memory.reload()
        assert len(memory.examples) == initial_count


class TestAutoLearning:
    """Tests for auto-learning (save_learned_example)."""

    def setup_method(self):
        """Create a temp examples directory for each test."""
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.examples_dir = self.tmp_dir / "examples"
        self.examples_dir.mkdir()
        # Write a minimal examples file
        test_file = self.examples_dir / "scout_examples.json"
        test_file.write_text(json.dumps({
            "version": "2.0",
            "examples": [
                {
                    "id": "scout-001",
                    "name": "Existing Example",
                    "tags": ["go"],
                    "build_system": "go",
                    "source": "manual",
                    "repo_name": "existing-repo",
                    "plan": {"phases": [{"name": "build", "commands": ["go build ."]}]},
                    "reasoning": "Existing",
                }
            ]
        }))

    def teardown_method(self):
        """Clean up temp directory."""
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_save_learned_example(self):
        """Test saving a new auto-learned example."""
        memory = AgentMemory("scout", examples_dir=self.examples_dir)
        assert len(memory.examples) == 1

        result = memory.save_learned_example({
            "name": "Auto: new-repo (go)",
            "tags": ["go"],
            "build_system": "go",
            "repo_name": "new-repo",
            "plan": {"phases": [{"name": "build", "commands": ["go build ./cmd/app"]}]},
            "reasoning": "Auto-learned",
        })
        assert result is True
        assert len(memory.examples) == 2
        new_ex = memory.examples[-1]
        assert new_ex.id == "scout-auto-001"
        assert new_ex.source == "auto"

    def test_save_duplicate_rejected(self):
        """Test that duplicate examples are rejected."""
        memory = AgentMemory("scout", examples_dir=self.examples_dir)

        result = memory.save_learned_example({
            "name": "Duplicate",
            "tags": ["go"],
            "build_system": "go",
            "repo_name": "existing-repo",
            "reasoning": "Should be rejected",
        })
        assert result is False
        assert len(memory.examples) == 1

    def test_save_missing_fields_rejected(self):
        """Test that examples without required fields are rejected."""
        memory = AgentMemory("scout", examples_dir=self.examples_dir)

        result = memory.save_learned_example({
            "tags": ["go"],
        })
        assert result is False

    def test_prune_oldest_auto(self):
        """Test pruning removes oldest auto-learned first."""
        memory = AgentMemory("scout", examples_dir=self.examples_dir)
        # Manually add many auto examples to test pruning
        data = memory._read_json_file()
        for i in range(105):
            data["examples"].append({
                "id": f"scout-auto-{i:03d}",
                "name": f"Auto Example {i}",
                "tags": ["go"],
                "build_system": "go",
                "source": "auto",
                "repo_name": f"repo-{i}",
                "timestamp": f"2026-01-{i % 28 + 1:02d}",
            })
        pruned = memory._prune_examples_list(data["examples"])
        assert len(pruned) <= 100
        # Manual example should survive
        assert any(e.get("source", "manual") == "manual" for e in pruned)


class TestFormatFewShotExamples:
    """Tests for the convenience function."""

    def setup_method(self):
        """Clear cached instances for clean tests."""
        MEMORY_INSTANCES.clear()

    def test_format_scout_examples(self):
        """Test formatting scout examples."""
        context = {"build_system": "go", "has_main": True, "main_path": "."}
        result = format_few_shot_examples("scout", context, max_examples=2)
        assert "# Few-Shot Examples" in result
        assert "scout" in result.lower()

    def test_format_fixer_examples(self):
        """Test formatting fixer examples."""
        context = {"build_system": "go", "error_message": "go: command not found"}
        result = format_few_shot_examples("fixer", context, max_examples=2)
        assert "# Few-Shot Examples" in result
        assert "fixer" in result.lower()

    def test_format_builder_examples(self):
        """Test formatting builder examples."""
        context = {"build_system": "go"}
        result = format_few_shot_examples("builder", context, max_examples=2)
        assert isinstance(result, str)

    def test_empty_context(self):
        """Test with empty context."""
        result = format_few_shot_examples("scout", {}, max_examples=2)
        assert isinstance(result, str)


class TestMemoryCaching:
    """Tests for memory instance caching."""

    def setup_method(self):
        MEMORY_INSTANCES.clear()

    def test_get_agent_memory_caching(self):
        """Test that memory instances are cached."""
        memory1 = get_agent_memory("scout")
        memory2 = get_agent_memory("scout")
        assert memory1 is memory2

    def test_different_agents_different_memory(self):
        """Test that different agents have different memory."""
        scout_memory = get_agent_memory("scout")
        fixer_memory = get_agent_memory("fixer")
        assert scout_memory is not fixer_memory
        assert scout_memory.agent_type == "scout"
        assert fixer_memory.agent_type == "fixer"

    def test_reload_agent_memory(self):
        """Test force reloading memory."""
        memory1 = get_agent_memory("scout")
        reload_agent_memory("scout")
        memory2 = get_agent_memory("scout")
        # After reload, should have fresh examples
        assert len(memory2.examples) > 0


class TestRecipeCache:
    """Tests for recipe cache functions."""

    def setup_method(self):
        """Back up recipe cache if it exists."""
        self.backup = None
        if RECIPE_CACHE_PATH.exists():
            self.backup = RECIPE_CACHE_PATH.read_text()

    def teardown_method(self):
        """Restore recipe cache."""
        if self.backup is not None:
            RECIPE_CACHE_PATH.write_text(self.backup)
        elif RECIPE_CACHE_PATH.exists():
            RECIPE_CACHE_PATH.write_text(json.dumps({"version": "1.0", "packages": {}}))

    def test_load_empty_cache(self):
        """Test loading when cache is empty."""
        RECIPE_CACHE_PATH.write_text(json.dumps({"version": "1.0", "packages": {}}))
        cache = load_recipe_cache()
        assert cache["version"] == "1.0"
        assert cache["packages"] == {}

    def test_save_and_get_recipe(self):
        """Test saving and retrieving a recipe."""
        RECIPE_CACHE_PATH.write_text(json.dumps({"version": "1.0", "packages": {}}))

        result = save_to_recipe_cache(
            repo_name="test-pkg",
            repo_url="https://github.com/test/test-pkg",
            build_system="go",
            build_plan={"phases": [{"name": "build", "commands": ["go build ."]}]},
            dependencies=["go"],
            patches=[],
            artifacts=[{"type": "binary", "filepath": "test-pkg"}],
            build_duration_seconds=30.5,
        )
        assert result is True

        cached = get_cached_recipe("test-pkg")
        assert cached is not None
        assert cached["build_system"] == "go"
        assert cached["architecture"] == "riscv64"
        assert cached["build_duration_seconds"] == 30.5

    def test_get_nonexistent_recipe(self):
        """Test cache miss returns None."""
        RECIPE_CACHE_PATH.write_text(json.dumps({"version": "1.0", "packages": {}}))
        cached = get_cached_recipe("nonexistent-pkg")
        assert cached is None

    def test_upsert_recipe(self):
        """Test that saving again updates the entry."""
        RECIPE_CACHE_PATH.write_text(json.dumps({"version": "1.0", "packages": {}}))

        save_to_recipe_cache(
            repo_name="update-test",
            repo_url="https://github.com/test/update-test",
            build_system="go",
            build_plan={"phases": [{"name": "build", "commands": ["go build ."]}]},
            dependencies=[],
            patches=[],
            artifacts=[],
            build_duration_seconds=10.0,
        )
        save_to_recipe_cache(
            repo_name="update-test",
            repo_url="https://github.com/test/update-test",
            build_system="go",
            build_plan={"phases": [{"name": "build", "commands": ["go build -v ."]}]},
            dependencies=["go"],
            patches=["fixed something"],
            artifacts=[],
            build_duration_seconds=20.0,
        )

        cached = get_cached_recipe("update-test")
        assert cached["build_duration_seconds"] == 20.0
        assert cached["dependencies"] == ["go"]

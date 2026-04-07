"""
Unit tests for the Agent Memory System.
"""

import pytest
import json
from pathlib import Path

from src.memory import (
    AgentMemory,
    AgentExample,
    format_few_shot_examples,
    get_agent_memory,
)


class TestAgentExample:
    """Tests for AgentExample dataclass."""

    def test_example_creation(self):
        """Test creating an example."""
        example = AgentExample(
            id="test-001",
            name="Test Example",
            tags=["go", "test"],
            context={"build_system": "go"},
            reasoning="Test reasoning",
        )
        assert example.id == "test-001"
        assert example.tags == ["go", "test"]
        assert example.context["build_system"] == "go"

    def test_scout_prompt_format(self):
        """Test formatting example as Scout prompt."""
        example = AgentExample(
            id="scout-001",
            name="Test Scout",
            tags=["go"],
            context={"build_system": "go", "repo_name": "test"},
            expected_output={"build_system": "go", "phases": []},
            reasoning="Because test",
        )
        text = example.to_prompt_text("scout")
        assert "Test Scout" in text
        assert "go" in text
        assert "Because test" in text

    def test_fixer_prompt_format(self):
        """Test formatting example as Fixer prompt."""
        example = AgentExample(
            id="fixer-001",
            name="Test Fixer",
            tags=["go", "error"],
            context={},
            solution={"analysis": "Test analysis", "strategy": "Test strategy"},
            reasoning="Test reasoning",
            raw={"error_context": {"category": "TEST", "error_message": "test error"}},
        )
        text = example.to_prompt_text("fixer")
        assert "Test Fixer" in text
        assert "Test analysis" in text


class TestAgentMemory:
    """Tests for AgentMemory class."""

    def test_load_scout_examples(self):
        """Test loading scout examples from file."""
        memory = AgentMemory("scout")
        assert len(memory.examples) > 0
        assert all(isinstance(ex, AgentExample) for ex in memory.examples)

    def test_load_fixer_examples(self):
        """Test loading fixer examples from file."""
        memory = AgentMemory("fixer")
        assert len(memory.examples) > 0

    def test_get_relevant_examples(self):
        """Test getting relevant examples based on context."""
        memory = AgentMemory("scout")

        context = {"build_system": "go", "has_main": True}
        examples = memory.get_relevant_examples(context, max_examples=2)

        assert len(examples) <= 2
        # Should prefer go examples
        for ex in examples:
            assert "go" in ex.tags or ex.context.get("build_system") == "go"

    def test_relevance_scoring(self):
        """Test relevance scoring algorithm."""
        memory = AgentMemory("scout")

        # Example with matching build system
        context1 = {"build_system": "go"}
        examples1 = memory.get_relevant_examples(context1, max_examples=5)

        # Example with different build system
        context2 = {"build_system": "cmake"}
        examples2 = memory.get_relevant_examples(context2, max_examples=5)

        # Both should return examples but possibly different ones
        assert len(examples1) > 0
        assert len(examples2) > 0

    def test_format_examples_for_prompt(self):
        """Test formatting examples for prompt."""
        memory = AgentMemory("scout")
        context = {"build_system": "go"}
        examples = memory.get_relevant_examples(context, max_examples=2)

        formatted = memory.format_examples_for_prompt(examples, max_chars=1000)

        assert "# Few-Shot Examples" in formatted
        assert len(formatted) <= 1100  # Some margin for formatting


class TestFormatFewShotExamples:
    """Tests for the convenience function."""

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

    def test_empty_context(self):
        """Test with empty context."""
        result = format_few_shot_examples("scout", {}, max_examples=2)
        # Should still return something
        assert isinstance(result, str)


class TestMemoryCaching:
    """Tests for memory instance caching."""

    def test_get_agent_memory_caching(self):
        """Test that memory instances are cached."""
        memory1 = get_agent_memory("scout")
        memory2 = get_agent_memory("scout")

        assert memory1 is memory2  # Same instance

    def test_different_agents_different_memory(self):
        """Test that different agents have different memory."""
        scout_memory = get_agent_memory("scout")
        fixer_memory = get_agent_memory("fixer")

        assert scout_memory is not fixer_memory
        assert scout_memory.agent_type == "scout"
        assert fixer_memory.agent_type == "fixer"

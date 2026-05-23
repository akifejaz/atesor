"""
Shared pytest fixtures for the Atesor AI test suite.

Goals:
  * Isolate every test from shared global state (MEMORY_INSTANCES, recipe cache,
    active platform profile, llm_logger file handle).
  * Never touch the network, never launch a real Docker exec, never call an LLM.
  * Make file-system side effects opt-in via the `tmp_path` builtin.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import pytest


# ---------------------------------------------------------------------------
# Isolate the global agent-memory singleton between tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_memory_instances() -> Iterator[None]:
    """Wipe the MEMORY_INSTANCES singleton dict so AgentMemory tests don't bleed."""
    from src import memory

    saved = dict(memory.MEMORY_INSTANCES)
    memory.MEMORY_INSTANCES.clear()
    try:
        yield
    finally:
        memory.MEMORY_INSTANCES.clear()
        memory.MEMORY_INSTANCES.update(saved)


# ---------------------------------------------------------------------------
# Isolate the cached active PlatformProfile so tests can swap it freely
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _restore_active_profile() -> Iterator[None]:
    """Reset the platforms cache before/after each test.

    We pre-seed the cache with ALPINE_RISCV so that any code path that calls
    get_active_profile() during a test does NOT trigger a real `docker exec`
    against a (likely non-running) container — which would block for ~10s
    per call and balloon the suite runtime.
    """
    from src import platforms

    saved = platforms._cached_profile
    try:
        platforms._cached_profile = platforms.ALPINE_RISCV
        yield
    finally:
        platforms._cached_profile = saved


# ---------------------------------------------------------------------------
# Back up + restore the recipe cache on disk so we don't pollute the repo
# ---------------------------------------------------------------------------


@pytest.fixture
def empty_recipe_cache() -> Iterator[Path]:
    """
    Replace the real recipe cache with an empty v2 cache for the test, then
    restore the original contents. Yields the cache path.

    Tests that want a clean slate should depend on this fixture explicitly;
    they should NOT just write to RECIPE_CACHE_PATH globally.
    """
    from src import memory

    backup: bytes | None = None
    if memory.RECIPE_CACHE_PATH.exists():
        backup = memory.RECIPE_CACHE_PATH.read_bytes()

    memory.RECIPE_CACHE_PATH.write_text(
        json.dumps({"version": "2.0", "packages": {}})
    )
    try:
        yield memory.RECIPE_CACHE_PATH
    finally:
        if backup is not None:
            memory.RECIPE_CACHE_PATH.write_bytes(backup)
        else:
            memory.RECIPE_CACHE_PATH.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Safe helper: stub execute_command in a single module so tests don't fork docker
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_execute_command(monkeypatch):
    """
    Return a helper that replaces execute_command in any module with a
    deterministic callable. Example:

        def fake(cmd, **kwargs):
            return CommandResult(cmd, 0, "ok", "", 0.0)
        stub_execute_command("src.scripted_ops", fake)
    """
    def _install(module_path: str, fake):
        import importlib

        mod = importlib.import_module(module_path)
        monkeypatch.setattr(mod, "execute_command", fake)
        return fake

    return _install


# ---------------------------------------------------------------------------
# Per-test temporary examples directory (so save_learned_example tests don't
# trample the real data/examples/*.json files).
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_examples_dir(tmp_path) -> Path:
    """Create a tmp dir + seed a minimal scout examples file."""
    d = tmp_path / "examples"
    d.mkdir()
    (d / "scout_examples.json").write_text(json.dumps({
        "version": "2.0",
        "examples": [
            {
                "id": "scout-001",
                "name": "Seed Example",
                "tags": ["go"],
                "build_system": "go",
                "source": "manual",
                "repo_name": "seed",
                "sandbox": "alpine-riscv64",
                "plan": {"phases": [{"name": "build", "commands": ["go build ."]}]},
                "reasoning": "seed",
            }
        ],
    }))
    return d

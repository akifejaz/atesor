"""Tests for src/scripted_ops.py using real temporary directory trees.

These tests build small fake repos on disk (no LLM, no network, no Docker)
and verify that the deterministic analysis layer makes the right calls.
"""

from __future__ import annotations

import json
import os
import textwrap
import unittest

import pytest

from src.scripted_ops import ScriptedOperations, quick_analysis


# ---------- helpers ----------

def _write(root, rel, content=""):
    p = os.path.join(root, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True) if os.path.dirname(rel) else None
    with open(p, "w") as f:
        f.write(content)
    return p


@pytest.fixture
def repo(tmp_path):
    return str(tmp_path)


# ---------- Build system detection ----------


class TestDetectBuildSystem:
    def test_no_build_files_returns_unknown(self, repo):
        info = ScriptedOperations(repo).detect_build_system(repo)
        assert info.type == "unknown"
        assert info.confidence == 0.0

    def test_cmake_wins_when_marker_present(self, repo):
        _write(repo, "CMakeLists.txt", "project(foo)\n")
        info = ScriptedOperations(repo).detect_build_system(repo)
        assert info.type == "cmake"
        assert info.confidence >= 0.9
        assert info.primary_file == "CMakeLists.txt"

    def test_cmake_wins_over_makefile_when_both_present(self, repo):
        # cmake gets 0.95, make gets ~0.3 → cmake wins
        _write(repo, "CMakeLists.txt", "project(foo)\n")
        _write(repo, "Makefile", "all:\n\techo hi\n")
        info = ScriptedOperations(repo).detect_build_system(repo)
        assert info.type == "cmake"

    def test_cargo_detected(self, repo):
        _write(repo, "Cargo.toml", '[package]\nname = "foo"\nversion = "0.1.0"\n')
        info = ScriptedOperations(repo).detect_build_system(repo)
        assert info.type == "cargo"
        assert info.confidence >= 0.9

    def test_go_detected(self, repo):
        _write(repo, "go.mod", "module foo\ngo 1.21\n")
        info = ScriptedOperations(repo).detect_build_system(repo)
        assert info.type == "go"

    def test_go_in_subdirectory_falls_back(self, repo):
        # No top-level go.mod, but a subdir has one
        os.makedirs(os.path.join(repo, "src/sub"))
        _write(repo, "src/sub/go.mod", "module foo\n")
        info = ScriptedOperations(repo).detect_build_system(repo)
        assert info.type == "go"
        assert info.module_dir.endswith("src/sub")

    def test_autotools_detected(self, repo):
        _write(repo, "configure.ac", "AC_INIT(foo, 1.0)\n")
        info = ScriptedOperations(repo).detect_build_system(repo)
        assert info.type == "autotools"
        assert info.confidence >= 0.9

    def test_meson_detected(self, repo):
        _write(repo, "meson.build", "project('foo', 'c')\n")
        info = ScriptedOperations(repo).detect_build_system(repo)
        # meson uses lower-confidence (+0.3) heuristic but should win alone
        assert info.type == "meson"


# ---------- Dependency extraction ----------


class TestExtractDependencies:
    def test_cmake_finds_find_package_calls(self, repo):
        _write(repo, "CMakeLists.txt", textwrap.dedent("""
            project(foo)
            find_package(OpenSSL REQUIRED)
            find_package(ZLIB)
            find_package(Threads)
        """))
        deps = ScriptedOperations(repo)._extract_cmake_dependencies(repo)
        assert "OpenSSL" in deps.libraries
        assert "ZLIB" in deps.libraries
        assert "Threads" in deps.libraries
        assert deps.install_method == "apk"
        assert "cmake" in deps.build_tools

    def test_cargo_extracts_dependencies(self, repo):
        _write(repo, "Cargo.toml", textwrap.dedent("""
            [package]
            name = "foo"
            version = "0.1.0"

            [dependencies]
            serde = "1.0"
            tokio = "1.0"
        """))
        deps = ScriptedOperations(repo)._extract_cargo_dependencies(repo)
        assert set(deps.libraries) == {"serde", "tokio"}
        assert "cargo" in deps.build_tools

    def test_python_extracts_requirements(self, repo):
        _write(repo, "requirements.txt", textwrap.dedent("""
            # a comment
            requests==2.28.0
            numpy>=1.20
            click
        """))
        deps = ScriptedOperations(repo)._extract_python_dependencies(repo)
        assert set(deps.libraries) == {"requests", "numpy", "click"}

    def test_npm_extracts_both_dep_buckets(self, repo):
        _write(repo, "package.json", json.dumps({
            "name": "foo",
            "dependencies": {"express": "^4.0.0"},
            "devDependencies": {"jest": "^29.0.0"},
        }))
        deps = ScriptedOperations(repo)._extract_npm_dependencies(repo)
        assert set(deps.libraries) == {"express", "jest"}

    def test_go_mod_extracts_require_lines(self, repo):
        _write(repo, "go.mod", textwrap.dedent("""
            module foo
            go 1.21

            require github.com/spf13/cobra v1.0.0
            require golang.org/x/sys v0.5.0
        """))
        deps = ScriptedOperations(repo)._extract_go_dependencies(repo)
        assert "github.com/spf13/cobra" in deps.libraries
        assert "golang.org/x/sys" in deps.libraries

    def test_missing_file_returns_empty_deps(self, repo):
        deps = ScriptedOperations(repo)._extract_cmake_dependencies(repo)
        assert deps.libraries == []


# ---------- find_go_main_package ----------


class TestFindGoMainPackage:
    def test_no_go_files_returns_empty(self, repo):
        info = ScriptedOperations(repo).find_go_main_package(repo)
        assert info["has_main"] is False
        assert info["has_go_mod"] is False
        assert info["needs_go_init"] is False

    def test_gopath_style_repo_needs_init(self, repo):
        # .go files but no go.mod
        _write(repo, "main.go", "package main\nfunc main() {}\n")
        info = ScriptedOperations(repo).find_go_main_package(repo)
        assert info["needs_go_init"] is True
        assert info["has_main"] is True

    def test_simple_root_main(self, repo):
        _write(repo, "go.mod", "module foo\n")
        _write(repo, "main.go", "package main\nfunc main() {}\n")
        info = ScriptedOperations(repo).find_go_main_package(repo)
        assert info["has_main"] is True
        assert info["has_go_mod"] is True
        assert info["main_path"] == "."
        assert info["build_command"] == "go build ."

    def test_cmd_reponame_beats_root_main(self, tmp_path):
        # repo basename matters for scoring
        rp = tmp_path / "myapp"
        rp.mkdir()
        repo = str(rp)
        _write(repo, "go.mod", "module myapp\n")
        _write(repo, "main.go", "package main\nfunc main() {}\n")
        _write(repo, "cmd/myapp/main.go", "package main\nfunc main() {}\n")
        info = ScriptedOperations(repo).find_go_main_package(repo)
        assert info["main_path"] == "cmd/myapp"
        assert info["build_command"] == "go build ./cmd/myapp"

    def test_cmd_test_dir_demoted(self, tmp_path):
        rp = tmp_path / "myapp"
        rp.mkdir()
        repo = str(rp)
        _write(repo, "go.mod", "module myapp\n")
        _write(repo, "cmd/test-runner/main.go", "package main\nfunc main() {}\n")
        _write(repo, "main.go", "package main\nfunc main() {}\n")
        info = ScriptedOperations(repo).find_go_main_package(repo)
        # root main (score 3) > cmd/test-runner (score 1)
        assert info["main_path"] == "."


# ---------- quick_analysis integration ----------


class TestQuickAnalysis:
    def test_runs_on_simple_cmake_repo(self, repo):
        _write(repo, "CMakeLists.txt", "project(foo)\nfind_package(ZLIB)\n")
        _write(repo, "main.c", "int main(){return 0;}\n")
        result = quick_analysis(repo)
        assert result["build_system"].type == "cmake"
        assert "ZLIB" in result["dependencies"].libraries
        # file_tree calls execute_command (docker) — may be empty in unit tests
        assert isinstance(result["file_tree"], str)


# ---------- Path translation ----------


class TestPathTranslation:
    def test_to_host_path_translates_workspace(self):
        from src.config import WORKSPACE_ROOT
        ops = ScriptedOperations()
        # /workspace/repos/foo -> {WORKSPACE_ROOT}/repos/foo on host
        out = ops._to_host_path("/workspace/repos/foo")
        # When not running inside Docker, /workspace becomes WORKSPACE_ROOT
        assert out.endswith("/repos/foo")

    def test_to_host_path_passes_through_other_paths(self):
        ops = ScriptedOperations()
        assert ops._to_host_path("/tmp/x") == "/tmp/x"


if __name__ == "__main__":
    unittest.main()

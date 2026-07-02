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
    """Write."""
    p = os.path.join(root, rel)
    (
        os.makedirs(os.path.dirname(p), exist_ok=True)
        if os.path.dirname(rel)
        else None
    )
    with open(p, "w") as f:
        f.write(content)
    return p


@pytest.fixture
def repo(tmp_path):
    """Repo."""
    return str(tmp_path)


# ---------- Build system detection ----------


class TestDetectBuildSystem:
    """Tests for DetectBuildSystem."""

    def test_no_build_files_returns_unknown(self, repo) -> None:
        """Test no build files returns unknown."""
        info = ScriptedOperations(repo).detect_build_system(repo)
        assert info.type == "unknown"
        assert info.confidence == 0.0

    def test_cmake_wins_when_marker_present(self, repo) -> None:
        """Test cmake wins when marker present."""
        _write(repo, "CMakeLists.txt", "project(foo)\n")
        info = ScriptedOperations(repo).detect_build_system(repo)
        assert info.type == "cmake"
        assert info.confidence >= 0.9
        assert info.primary_file == "CMakeLists.txt"

    def test_cmake_wins_over_makefile_when_both_present(self, repo) -> None:
        # cmake gets 0.95, make gets ~0.3 → cmake wins
        """Test cmake wins over makefile when both present."""
        _write(repo, "CMakeLists.txt", "project(foo)\n")
        _write(repo, "Makefile", "all:\n\techo hi\n")
        info = ScriptedOperations(repo).detect_build_system(repo)
        assert info.type == "cmake"

    def test_cargo_detected(self, repo) -> None:
        """Test cargo detected."""
        _write(
            repo, "Cargo.toml", '[package]\nname = "foo"\nversion = "0.1.0"\n'
        )
        info = ScriptedOperations(repo).detect_build_system(repo)
        assert info.type == "cargo"
        assert info.confidence >= 0.9

    def test_go_detected(self, repo) -> None:
        """Test go detected."""
        _write(repo, "go.mod", "module foo\ngo 1.21\n")
        info = ScriptedOperations(repo).detect_build_system(repo)
        assert info.type == "go"

    def test_go_in_subdirectory_falls_back(self, repo) -> None:
        # No top-level go.mod, but a subdir has one
        """Test go in subdirectory falls back."""
        os.makedirs(os.path.join(repo, "src/sub"))
        _write(repo, "src/sub/go.mod", "module foo\n")
        info = ScriptedOperations(repo).detect_build_system(repo)
        assert info.type == "go"
        assert info.module_dir.endswith("src/sub")

    def test_autotools_detected(self, repo) -> None:
        """Test autotools detected."""
        _write(repo, "configure.ac", "AC_INIT(foo, 1.0)\n")
        info = ScriptedOperations(repo).detect_build_system(repo)
        assert info.type == "autotools"
        assert info.confidence >= 0.9

    def test_meson_detected(self, repo) -> None:
        """Test meson detected."""
        _write(repo, "meson.build", "project('foo', 'c')\n")
        info = ScriptedOperations(repo).detect_build_system(repo)
        # meson uses lower-confidence (+0.3) heuristic but should win alone
        assert info.type == "meson"


# ---------- Dependency extraction ----------


class TestExtractDependencies:
    """Tests for ExtractDependencies."""

    def test_cmake_finds_find_package_calls(self, repo) -> None:
        """Test cmake finds find package calls."""
        _write(
            repo,
            "CMakeLists.txt",
            textwrap.dedent("""
            project(foo)
            find_package(OpenSSL REQUIRED)
            find_package(ZLIB)
            find_package(Threads)
        """),
        )
        deps = ScriptedOperations(repo)._extract_cmake_dependencies(repo)
        assert "OpenSSL" in deps.libraries
        assert "ZLIB" in deps.libraries
        assert "Threads" in deps.libraries
        assert deps.install_method == "apk"
        assert "cmake" in deps.build_tools

    def test_cargo_extracts_dependencies(self, repo) -> None:
        """Test cargo extracts dependencies."""
        _write(
            repo,
            "Cargo.toml",
            textwrap.dedent("""
            [package]
            name = "foo"
            version = "0.1.0"

            [dependencies]
            serde = "1.0"
            tokio = "1.0"
        """),
        )
        deps = ScriptedOperations(repo)._extract_cargo_dependencies(repo)
        assert set(deps.libraries) == {"serde", "tokio"}
        assert "cargo" in deps.build_tools

    def test_python_extracts_requirements(self, repo) -> None:
        """Test python extracts requirements."""
        _write(
            repo,
            "requirements.txt",
            textwrap.dedent("""
            # a comment
            requests==2.28.0
            numpy>=1.20
            click
        """),
        )
        deps = ScriptedOperations(repo)._extract_python_dependencies(repo)
        assert set(deps.libraries) == {"requests", "numpy", "click"}

    def test_npm_extracts_both_dep_buckets(self, repo) -> None:
        """Test npm extracts both dep buckets."""
        _write(
            repo,
            "package.json",
            json.dumps(
                {
                    "name": "foo",
                    "dependencies": {"express": "^4.0.0"},
                    "devDependencies": {"jest": "^29.0.0"},
                }
            ),
        )
        deps = ScriptedOperations(repo)._extract_npm_dependencies(repo)
        assert set(deps.libraries) == {"express", "jest"}

    def test_go_mod_extracts_require_lines(self, repo) -> None:
        """Test go mod extracts require lines."""
        _write(
            repo,
            "go.mod",
            textwrap.dedent("""
            module foo
            go 1.21

            require github.com/spf13/cobra v1.0.0
            require golang.org/x/sys v0.5.0
        """),
        )
        deps = ScriptedOperations(repo)._extract_go_dependencies(repo)
        assert "github.com/spf13/cobra" in deps.libraries
        assert "golang.org/x/sys" in deps.libraries

    def test_missing_file_returns_empty_deps(self, repo) -> None:
        """Test missing file returns empty deps."""
        deps = ScriptedOperations(repo)._extract_cmake_dependencies(repo)
        assert deps.libraries == []


# ---------- find_go_main_package ----------


class TestFindGoMainPackage:
    """Tests for FindGoMainPackage."""

    def test_no_go_files_returns_empty(self, repo) -> None:
        """Test no go files returns empty."""
        info = ScriptedOperations(repo).find_go_main_package(repo)
        assert info["has_main"] is False
        assert info["has_go_mod"] is False
        assert info["needs_go_init"] is False

    def test_gopath_style_repo_needs_init(self, repo) -> None:
        # .go files but no go.mod
        """Test gopath style repo needs init."""
        _write(repo, "main.go", "package main\nfunc main() {}\n")
        info = ScriptedOperations(repo).find_go_main_package(repo)
        assert info["needs_go_init"] is True
        assert info["has_main"] is True

    def test_simple_root_main(self, repo) -> None:
        """Test simple root main."""
        _write(repo, "go.mod", "module foo\n")
        _write(repo, "main.go", "package main\nfunc main() {}\n")
        info = ScriptedOperations(repo).find_go_main_package(repo)
        assert info["has_main"] is True
        assert info["has_go_mod"] is True
        assert info["main_path"] == "."
        assert info["build_command"] == "go build ."

    def test_cmd_reponame_beats_root_main(self, tmp_path) -> None:
        # repo basename matters for scoring
        """Test cmd reponame beats root main."""
        rp = tmp_path / "myapp"
        rp.mkdir()
        repo = str(rp)
        _write(repo, "go.mod", "module myapp\n")
        _write(repo, "main.go", "package main\nfunc main() {}\n")
        _write(repo, "cmd/myapp/main.go", "package main\nfunc main() {}\n")
        info = ScriptedOperations(repo).find_go_main_package(repo)
        assert info["main_path"] == "cmd/myapp"
        assert info["build_command"] == "go build ./cmd/myapp"

    def test_cmd_test_dir_demoted(self, tmp_path) -> None:
        """Test cmd test dir demoted."""
        rp = tmp_path / "myapp"
        rp.mkdir()
        repo = str(rp)
        _write(repo, "go.mod", "module myapp\n")
        _write(
            repo, "cmd/test-runner/main.go", "package main\nfunc main() {}\n"
        )
        _write(repo, "main.go", "package main\nfunc main() {}\n")
        info = ScriptedOperations(repo).find_go_main_package(repo)
        # root main (score 3) > cmd/test-runner (score 1)
        assert info["main_path"] == "."


# ---------- quick_analysis integration ----------


class TestQuickAnalysis:
    """Tests for QuickAnalysis."""

    def test_runs_on_simple_cmake_repo(self, repo) -> None:
        """Test runs on simple cmake repo."""
        _write(repo, "CMakeLists.txt", "project(foo)\nfind_package(ZLIB)\n")
        _write(repo, "main.c", "int main(){return 0;}\n")
        result = quick_analysis(repo)
        assert result["build_system"].type == "cmake"
        assert "ZLIB" in result["dependencies"].libraries
        # file_tree calls execute_command (docker) — may be empty in unit tests
        assert isinstance(result["file_tree"], str)


# ---------- Path translation ----------


class TestPathTranslation:
    """Tests for PathTranslation."""

    def test_to_host_path_translates_workspace(self) -> None:
        """Translate a /workspace path to the host root outside Docker."""
        ops = ScriptedOperations()
        # /workspace/repos/foo -> {WORKSPACE_ROOT}/repos/foo on host
        out = ops._to_host_path("/workspace/repos/foo")
        # When not running inside Docker, /workspace becomes WORKSPACE_ROOT
        assert out.endswith("/repos/foo")

    def test_to_host_path_passes_through_other_paths(self) -> None:
        """Test to host path passes through other paths."""
        ops = ScriptedOperations()
        assert ops._to_host_path("/tmp/x") == "/tmp/x"


# ---------- Homepage → git URL resolution ----------


class TestResolveHomepageToGitUrls:
    """Tests for ScriptedOperations._resolve_homepage_to_git_urls."""

    def _ops(self) -> ScriptedOperations:
        """Helper: build a ScriptedOperations bound to a tmp workspace."""
        return ScriptedOperations()

    def test_gnu_homepage_maps_to_savannah(self) -> None:
        """`gnu.org/software/wget/` -> savannah `wget.git`."""
        got = self._ops()._resolve_homepage_to_git_urls(
            "https://www.gnu.org/software/wget/", "wget"
        )
        assert got == ["https://git.savannah.gnu.org/git/wget.git"]

    def test_gnu_homepage_without_trailing_slash(self) -> None:
        """`gnu.org/software/grep` (no slash) also resolves."""
        got = self._ops()._resolve_homepage_to_git_urls(
            "http://www.gnu.org/software/grep", "grep"
        )
        assert got == ["https://git.savannah.gnu.org/git/grep.git"]

    def test_savannah_cgit_url_maps_to_git_dir(self) -> None:
        """`git.savannah.gnu.org/cgit/gawk.git` -> git/gawk.git (no dupes)."""
        got = self._ops()._resolve_homepage_to_git_urls(
            "https://git.savannah.gnu.org/cgit/gawk.git", "gawk"
        )
        assert got == ["https://git.savannah.gnu.org/git/gawk.git"]

    def test_yorhel_maps_to_blicky(self) -> None:
        """`dev.yorhel.nl` -> `code.blicky.net/yorhel/<name>.git`."""
        got = self._ops()._resolve_homepage_to_git_urls(
            "https://dev.yorhel.nl", "ncdu"
        )
        assert got == ["https://code.blicky.net/yorhel/ncdu.git"]

    def test_generic_github_url_yields_no_rewrite(self) -> None:
        """Regular GitHub URLs have no homepage rewrite."""
        got = self._ops()._resolve_homepage_to_git_urls(
            "https://github.com/foo/bar", "bar"
        )
        assert got == []


# ---------- Container-health precheck ----------


class TestEnsureContainerHealthy:
    """Tests for ScriptedOperations._ensure_container_healthy."""

    def test_git_missing_triggers_install(self, stub_execute_command) -> None:
        """Missing git in container -> profile.install_cmd(['git']) runs."""
        from src.state import CommandResult

        calls: list = []

        def fake(cmd, **kwargs):
            """Fake."""
            calls.append(cmd)
            # First probe: git --version fails with 127
            if isinstance(cmd, list) and cmd == ["git", "--version"]:
                return CommandResult(
                    "git --version", 127, "", "git: not found", 0.0
                )
            # Install command succeeds
            return CommandResult(str(cmd), 0, "installed", "", 0.0)

        stub_execute_command("src.scripted_ops", fake)
        ScriptedOperations()._ensure_container_healthy()

        # The install command should reference git and use the active
        # profile's package manager (alpine → apk).
        install_calls = [c for c in calls if isinstance(c, str) and "git" in c]
        assert any("apk" in c or "apt-get" in c for c in install_calls), calls

    def test_healthy_container_makes_only_probes(
        self, stub_execute_command
    ) -> None:
        """A working sandbox runs only the read-only probes, no install."""
        from src.state import CommandResult

        calls: list = []

        def fake(cmd, **kwargs):
            """Fake."""
            calls.append(cmd)
            return CommandResult(str(cmd), 0, "", "", 0.0)

        stub_execute_command("src.scripted_ops", fake)
        ScriptedOperations()._ensure_container_healthy()

        # Only the git --version probe should run (alpine skips the apt
        # probe entirely).
        assert any(
            isinstance(c, list) and c[:2] == ["git", "--version"]
            for c in calls
        )
        # No install command
        assert not any(isinstance(c, str) and "install" in c for c in calls)


# ---------- Submodule init after clone ----------


class TestInitSubmodulesIfPresent:
    """Tests for ScriptedOperations._init_submodules_if_present."""

    def test_no_gitmodules_is_no_op(self, stub_execute_command) -> None:
        """When ``.gitmodules`` is absent, no submodule call is made."""
        from src.state import CommandResult

        calls: list = []

        def fake(cmd, **kwargs):
            """Fake."""
            calls.append(cmd)
            # test -f returns non-zero when the file doesn't exist
            if isinstance(cmd, str) and cmd.startswith("test -f"):
                return CommandResult(cmd, 1, "", "", 0.0)
            return CommandResult(str(cmd), 0, "", "", 0.0)

        stub_execute_command("src.scripted_ops", fake)
        ScriptedOperations()._init_submodules_if_present(
            "/workspace/repos/foo"
        )

        # test -f check ran, but no submodule command should follow.
        assert any(isinstance(c, str) and "test -f" in c for c in calls)
        assert not any(
            isinstance(c, str) and "submodule update" in c for c in calls
        )

    def test_gitmodules_triggers_submodule_init(
        self, stub_execute_command
    ) -> None:
        """When ``.gitmodules`` exists, git submodule update runs."""
        from src.state import CommandResult

        calls: list = []

        def fake(cmd, **kwargs):
            """Fake."""
            calls.append(cmd)
            return CommandResult(str(cmd), 0, "", "", 0.0)

        stub_execute_command("src.scripted_ops", fake)
        ScriptedOperations()._init_submodules_if_present(
            "/workspace/repos/foo"
        )

        assert any(
            isinstance(c, str)
            and "git submodule update --init --recursive" in c
            for c in calls
        )


class TestCloneResetsExistingRepo:
    """Regression: reused clones must be reset to pristine upstream state.

    Rationale: a previous run's LLM may have authored broken files (e.g.
    a syntactically-invalid Makefile) or half-applied patches; ``git
    pull`` alone preserves them, causing every subsequent run to replay
    the same failure. See src/scripted_ops.py::clone_or_update_repository.
    """

    def test_existing_repo_runs_fetch_reset_and_clean(
        self, stub_execute_command, monkeypatch, tmp_path
    ) -> None:
        """When the clone directory exists, fetch+reset+clean must run."""
        from src.state import CommandResult

        # Pretend the .git directory exists so we hit the "exists" branch.
        monkeypatch.setattr("os.path.exists", lambda _p: True)
        # Suppress side effects from health check and submodule init.
        monkeypatch.setattr(
            "src.scripted_ops.ScriptedOperations." "_ensure_container_healthy",
            lambda _self: None,
        )
        monkeypatch.setattr(
            "src.scripted_ops.ScriptedOperations."
            "_init_submodules_if_present",
            lambda _self, _p: None,
        )

        commands: list = []

        def fake(cmd, **kwargs):
            """Fake execute_command that records every issued command."""
            commands.append(cmd)
            return CommandResult(str(cmd), 0, "", "", 0.0)

        stub_execute_command("src.scripted_ops", fake)
        ScriptedOperations().clone_or_update_repository(
            "https://github.com/foo/bar.git", "bar"
        )

        joined = " ".join(str(c) for c in commands)
        # Must reset to upstream HEAD and remove untracked files.
        assert "git fetch" in joined
        assert "git reset --hard" in joined
        assert "git clean -fdx" in joined
        # Must NOT do a bare `git pull` which would preserve the poison.
        assert not any(
            isinstance(c, str) and c.strip().endswith("git pull")
            for c in commands
        )


if __name__ == "__main__":
    unittest.main()

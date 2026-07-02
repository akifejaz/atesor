"""Tests for graph.py validators.

Covers build plan, fix command, fixer response, and predictions.
"""

import os
import unittest

from src.graph import (
    _download_go_toolchain_cmd,
    _inject_go_flag,
    _inject_go_output,
    _is_go_build_command,
    _is_suspected_oom,
    _npm_scripts,
    _replan_signature,
    _repo_has_gitmodules,
    _resolve_header_to_packages,
    _resolve_missing_python_modules,
    _serialize_build_command,
    is_toolchain_version_mismatch,
    predict_build_issues,
    validate_build_plan,
    validate_fix_command,
    validate_fixer_response,
)
from src.state import (
    ArchSpecificCode,
    BuildPhase,
    BuildPlan,
    create_initial_state,
)


def _plan(*commands) -> BuildPlan:
    """Plan."""
    return BuildPlan(
        build_system="cmake",
        build_system_confidence=0.95,
        phases=[BuildPhase(id=1, name="build", commands=list(commands))],
        total_estimated_duration="5m",
    )


# ---------- validate_build_plan ----------


class TestValidateBuildPlan(unittest.TestCase):
    """Tests for ValidateBuildPlan."""

    def test_clean_plan_passes(self) -> None:
        """Test clean plan passes."""
        ok, msg = validate_build_plan(_plan("cmake -B build -S .", "make -j4"))
        self.assertTrue(ok, msg)

    def test_placeholder_path_rejected(self) -> None:
        """Test placeholder path rejected."""
        ok, msg = validate_build_plan(_plan("cp /path/to/foo bar"))
        self.assertFalse(ok)
        self.assertIn("Hallucination", msg)

    def test_your_username_placeholder_rejected(self) -> None:
        """Test your username placeholder rejected."""
        ok, msg = validate_build_plan(_plan("cd /home/your_username/foo"))
        self.assertFalse(ok)

    def test_example_com_placeholder_rejected(self) -> None:
        """Test example com placeholder rejected."""
        ok, _ = validate_build_plan(
            _plan("wget https://example.com/foo.tar.gz")
        )
        self.assertFalse(ok)

    def test_riscv_unknown_linux_gnu_gcc_rejected(self) -> None:
        """Test riscv unknown linux gnu gcc rejected."""
        ok, _ = validate_build_plan(
            _plan("riscv64-unknown-linux-gnu-gcc -O2 main.c")
        )
        self.assertFalse(ok)

    def test_home_path_outside_workspace_rejected(self) -> None:
        """Test home path outside workspace rejected."""
        ok, _ = validate_build_plan(_plan("cp /home/akif/foo bar"))
        self.assertFalse(ok)

    def test_home_inside_workspace_path_passes(self) -> None:
        # /home/.../workspace/... is fine
        """Test home inside workspace path passes."""
        ok, _ = validate_build_plan(_plan("ls /home/akif/workspace/foo"))
        self.assertTrue(ok)

    def test_go_subcommand_as_apk_package_rejected(self) -> None:
        """Test go subcommand as apk package rejected."""
        ok, msg = validate_build_plan(_plan("apk add go mod"))
        self.assertFalse(ok)
        self.assertIn("Go subcommand", msg)

    def test_go_subcommand_as_apt_package_rejected(self) -> None:
        """Test go subcommand as apt package rejected."""
        ok, _ = validate_build_plan(_plan("apt-get install build"))
        self.assertFalse(ok)

    def test_apt_install_with_flags_handles_pkg_list(self) -> None:
        """Test apt install with flags handles pkg list."""
        ok, _ = validate_build_plan(
            _plan("apt-get install -y libssl-dev curl")
        )
        self.assertTrue(ok)

    def test_nested_cmake_build_dir_rejected(self) -> None:
        """Test nested cmake build dir rejected."""
        ok, msg = validate_build_plan(
            _plan("cd build && cmake -S . -B build ..")
        )
        self.assertFalse(ok)
        self.assertIn("nested build dir", msg)


# ---------- validate_fix_command ----------


class TestValidateFixCommand(unittest.TestCase):
    """Tests for ValidateFixCommand."""

    @classmethod
    def setUpClass(cls) -> None:
        """SetUpClass."""
        cls.dangerous = [
            ("touch src/foo.go", "Go source"),
            ("touch include/x.h", "header"),
            ("touch foo.py", "Python"),
            ("touch app.cpp", "C++"),
            ("rm -rf /", "root rm"),
            ("rm -rf *", "wildcard rm"),
            ("git push origin main", "git push"),
            ("git reset --hard HEAD~1", "hard reset"),
            ('echo "" > foo.c', "empty content overwrite"),
            ("apk add go mod", "go subcommand as package"),
            ("apt install build", "go subcommand as package"),
        ]
        cls.safe = [
            "make -j4",
            "cmake -B build -S .",
            "sed -i 's/a/b/' foo.c",
            "cp src.c src.c.bak",
            "rm -f build/CMakeCache.txt",
            "apk add zlib-dev",
        ]

    def test_dangerous_commands_rejected(self) -> None:
        """Test dangerous commands rejected."""
        for cmd, label in self.dangerous:
            with self.subTest(cmd=cmd, label=label):
                ok, _ = validate_fix_command(cmd)
                self.assertFalse(ok, f"SHOULD REJECT [{label}]: {cmd}")

    def test_safe_commands_pass(self) -> None:
        """Test safe commands pass."""
        for cmd in self.safe:
            with self.subTest(cmd=cmd):
                ok, reason = validate_fix_command(cmd)
                self.assertTrue(ok, f"SHOULD PASS: {cmd} -> {reason}")


# ---------- validate_fixer_response ----------


class TestValidateFixerResponse(unittest.TestCase):
    """Tests for ValidateFixerResponse."""

    def _base(self, actions):
        """Build a base strategy payload for tests."""
        return {
            "strategies": [{"id": 1, "actions": actions}],
            "recommended_strategy_id": 1,
            "reflection": {
                "root_cause": "x",
                "this_fix_will_work_because": "y",
            },
        }

    def test_non_dict_rejected(self) -> None:
        """Test non dict rejected."""
        ok, _ = validate_fixer_response([])  # type: ignore[arg-type]
        self.assertFalse(ok)

    def test_no_strategies_rejected(self) -> None:
        """Test no strategies rejected."""
        ok, msg = validate_fixer_response({})
        self.assertFalse(ok)
        self.assertIn("strategies", msg.lower())

    def test_recommended_id_not_found_rejected(self) -> None:
        """Test recommended id not found rejected."""
        data = self._base([{"type": "command", "command": "ls"}])
        data["recommended_strategy_id"] = 99
        ok, msg = validate_fixer_response(data)
        self.assertFalse(ok)
        self.assertIn("not found", msg)

    def test_empty_actions_rejected(self) -> None:
        """Test empty actions rejected."""
        ok, msg = validate_fixer_response(self._base([]))
        self.assertFalse(ok)
        self.assertIn("no actions", msg.lower())

    def test_create_file_with_empty_content_rejected(self) -> None:
        """Test create file with empty content rejected."""
        ok, msg = validate_fixer_response(
            self._base(
                [{"type": "create_file", "path": "foo.c", "content": "   "}]
            )
        )
        self.assertFalse(ok)
        self.assertIn("empty content", msg)

    def test_create_file_with_absolute_home_path_rejected(self) -> None:
        """Test create file with absolute home path rejected."""
        ok, _ = validate_fixer_response(
            self._base(
                [
                    {
                        "type": "create_file",
                        "path": "/home/x/foo.c",
                        "content": "int main(){}",
                    }
                ]
            )
        )
        self.assertFalse(ok)

    def test_create_file_missing_path_rejected(self) -> None:
        """Test create file missing path rejected."""
        ok, _ = validate_fixer_response(
            self._base([{"type": "create_file", "path": "", "content": "x"}])
        )
        self.assertFalse(ok)

    def test_empty_command_rejected(self) -> None:
        """Test empty command rejected."""
        ok, msg = validate_fixer_response(
            self._base([{"type": "command", "command": "   "}])
        )
        self.assertFalse(ok)
        self.assertIn("empty", msg.lower())

    def test_self_copy_command_rejected(self) -> None:
        """Test self copy command rejected."""
        ok, msg = validate_fixer_response(
            self._base([{"type": "command", "command": "cp -r src/* src/"}])
        )
        self.assertFalse(ok)
        self.assertIn("copying to self", msg)

    def test_unsafe_embedded_command_rejected(self) -> None:
        """Test unsafe embedded command rejected."""
        ok, msg = validate_fixer_response(
            self._base([{"type": "command", "command": "touch foo.go"}])
        )
        self.assertFalse(ok)
        self.assertIn("Unsafe", msg)

    def test_valid_response_accepted(self) -> None:
        """Test valid response accepted."""
        ok, msg = validate_fixer_response(
            self._base(
                [
                    {"type": "command", "command": "make clean"},
                    {
                        "type": "create_file",
                        "path": "patch.diff",
                        "content": "--- a\n+++ b\n",
                    },
                ]
            )
        )
        self.assertTrue(ok, msg)


# ---------- predict_build_issues ----------


class TestPredictBuildIssues(unittest.TestCase):
    """Tests for PredictBuildIssues."""

    def test_no_plan_returns_empty(self) -> None:
        """Test no plan returns empty."""
        state = create_initial_state("https://x/y.git")
        self.assertEqual(predict_build_issues(state), [])

    def test_go_build_without_buildvcs_flagged(self) -> None:
        """Test go build without buildvcs flagged."""
        state = create_initial_state("https://x/y.git")
        state.build_plan = _plan("go build ./cmd/foo")
        preds = predict_build_issues(state)
        self.assertTrue(
            any(p["issue"] == "Go VCS ownership error" for p in preds)
        )

    def test_go_build_with_buildvcs_false_not_flagged(self) -> None:
        """Test go build with buildvcs false not flagged."""
        state = create_initial_state("https://x/y.git")
        state.build_plan = _plan("go build -buildvcs=false ./cmd/foo")
        preds = predict_build_issues(state)
        self.assertFalse(
            any(p["issue"] == "Go VCS ownership error" for p in preds)
        )

    def test_cmake_with_high_severity_arch_code_flagged(self) -> None:
        """Test cmake with high severity arch code flagged."""
        state = create_initial_state("https://x/y.git")
        state.build_plan = _plan("cmake -B build -S .")
        state.arch_specific_code = [
            ArchSpecificCode(
                file="x.c",
                line=10,
                code_snippet="_mm_add_ps",
                arch_type="x86",
                severity="high",
                suggested_fix="fallback",
            )
        ]
        preds = predict_build_issues(state)
        self.assertTrue(any(p["pattern"] == "arch_specific" for p in preds))

    def test_apk_install_with_missing_tools_flagged(self) -> None:
        """Test apk install with missing tools flagged."""
        state = create_initial_state("https://x/y.git")
        state.build_plan = _plan("apk add zlib-dev")
        state.context_cache["missing_tools"] = ["protoc"]
        preds = predict_build_issues(state)
        self.assertTrue(any(p["pattern"] == "missing_tools" for p in preds))

    def test_cargo_build_not_flagged_as_go_vcs(self) -> None:
        """Regression: substring 'go build' must not match 'cargo build'."""
        state = create_initial_state("https://x/y.git")
        state.build_plan = _plan("cargo build --release")
        preds = predict_build_issues(state)
        self.assertFalse(
            any(p["issue"] == "Go VCS ownership error" for p in preds)
        )

    def test_apk_add_cargo_build_base_not_flagged_as_go_vcs(self) -> None:
        """Regression: 'cargo build-base' (alpine pkg) is not a go build."""
        state = create_initial_state("https://x/y.git")
        state.build_plan = _plan("apk add --no-cache rust cargo build-base")
        preds = predict_build_issues(state)
        self.assertFalse(
            any(p["issue"] == "Go VCS ownership error" for p in preds)
        )


class TestIsGoBuildCommand(unittest.TestCase):
    """Regression tests pinning the word-boundary go-build detector.

    The previous implementation used substring checks like
    `"go build" in cmd`, which falsely matched `cargo build` (at index 3)
    and the alpine package list `apk add ... cargo build-base`. Downstream
    code then mangled those commands (e.g. injecting `-buildvcs=false`
    into apk arguments), which is what caused the dalfox build failures.
    """

    def test_matches_plain_go_build(self) -> None:
        """Test matches plain go build."""
        self.assertTrue(_is_go_build_command("go build ."))

    def test_matches_go_build_with_flags(self) -> None:
        """Test matches go build with flags."""
        self.assertTrue(_is_go_build_command("go build -v ./cmd/foo"))

    def test_matches_go_install(self) -> None:
        """Test matches go install."""
        self.assertTrue(_is_go_build_command("go install ./..."))

    def test_matches_go_build_after_env(self) -> None:
        """Test env-prefixed go build is still detected."""
        self.assertTrue(
            _is_go_build_command("env GOMAXPROCS=1 go build -p 1 ./...")
        )

    def test_does_not_match_cargo_build(self) -> None:
        """Test does not match cargo build."""
        self.assertFalse(_is_go_build_command("cargo build --release"))

    def test_does_not_match_apk_cargo_build_base(self) -> None:
        """Test does not match 'apk add ... cargo build-base'."""
        self.assertFalse(
            _is_go_build_command("apk add --no-cache rust cargo build-base")
        )

    def test_does_not_match_hyphenated_token(self) -> None:
        """Test does not match 'do go-build' (hyphenated, not a go command)."""
        self.assertFalse(_is_go_build_command("do go-build"))

    def test_does_not_match_empty(self) -> None:
        """Test empty / non-string inputs return False."""
        self.assertFalse(_is_go_build_command(""))


class TestToolchainMismatchDetection(unittest.TestCase):
    """Tests for toolchain-version mismatch detection helper."""

    def test_detects_go_version_mismatch(self) -> None:
        """Go `go.mod requires go >=` errors are treated as mismatches."""
        self.assertTrue(
            is_toolchain_version_mismatch(
                "go: go.mod requires go >= 1.26.0 (running go 1.22.5)"
            )
        )

    def test_detects_rust_edition_mismatch(self) -> None:
        """Cargo edition gating errors are treated as mismatches."""
        self.assertTrue(
            is_toolchain_version_mismatch(
                "feature `edition2024` is required and not stabilized "
                "in this version of Cargo"
            )
        )

    def test_ignores_regular_dependency_errors(self) -> None:
        """Ordinary missing package errors are not toolchain mismatches."""
        self.assertFalse(
            is_toolchain_version_mismatch("unable to select packages: libssl")
        )


class TestGoCommandRewriters(unittest.TestCase):
    """Tests for Go command rewrite helpers."""

    def test_inject_go_flag_once(self) -> None:
        """Do not duplicate flags when they already exist."""
        cmd = "go build -buildvcs=false ./cmd/foo"
        self.assertEqual(_inject_go_flag(cmd, "-buildvcs=false"), cmd)

    def test_inject_go_output(self) -> None:
        """Inject -o path into go build command."""
        cmd = "go build ./cmd"
        out = _inject_go_output(cmd, "./.atesor-bin/cmd")
        self.assertIn("-o ./.atesor-bin/cmd", out)
        self.assertIn("go build", out)


class TestReplanSignature(unittest.TestCase):
    """Tests for plan-level error signature detection."""

    def test_go_output_collision_signature(self) -> None:
        """Output-dir collision should map to replan signature."""
        msg = 'go: build output "cmd" already exists and is a directory'
        self.assertEqual(_replan_signature(msg), "go_output_dir_collision")

    def test_unknown_signature(self) -> None:
        """Unrelated errors should return empty signature."""
        self.assertEqual(_replan_signature("some random error"), "")


class _Res:
    """Minimal CommandResult stand-in for retry-helper tests."""

    def __init__(self, exit_code, stdout, stderr, success):
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.success = success


class TestSerializeBuildCommand(unittest.TestCase):
    """Tests for the OOM serialized-build rewriter."""

    def test_go_build_serialized(self) -> None:
        """Go build commands are serialized with env and -p limits."""
        out = _serialize_build_command("go build ./cmd/x")
        self.assertIn("GOMAXPROCS=1", out)
        self.assertIn("-p 1", out)

    def test_go_build_preserves_cd_prefix(self) -> None:
        """Serialized go builds keep the leading cd command."""
        out = _serialize_build_command("cd sub && go build ./cmd/x")
        self.assertTrue(out.startswith("cd sub && env GOMAXPROCS=1 go build"))

    def test_make_serialized(self) -> None:
        """Make commands replace nproc parallelism with -j1."""
        out = _serialize_build_command("cd build && make -j$(nproc)")
        self.assertIn("-j1", out)
        self.assertNotIn("nproc", out)
        self.assertTrue(out.startswith("cd build &&"))

    def test_plain_make_gets_j1(self) -> None:
        """Plain make commands gain a -j1 serialization flag."""
        out = _serialize_build_command("make")
        self.assertIn("make -j1", out)

    def test_cmake_not_mistaken_for_make(self) -> None:
        """Build commands using cmake are not rewritten as make."""
        out = _serialize_build_command("cmake --build build")
        self.assertEqual(out, "cmake --build build")

    def test_cargo_fetch_gets_env_only(self) -> None:
        """cargo fetch has no -j flag; env CARGO_BUILD_JOBS=1 is added."""
        out = _serialize_build_command("cargo fetch")
        self.assertIn("CARGO_BUILD_JOBS=1", out)
        # `cargo fetch` must not gain a spurious -j 1 flag.
        self.assertNotIn(" -j ", out)

    def test_cargo_build_gets_j_and_env(self) -> None:
        """cargo build gets both -j 1 and CARGO_BUILD_JOBS env."""
        out = _serialize_build_command("cargo build --release")
        self.assertIn("cargo build -j 1", out)
        self.assertIn("CARGO_BUILD_JOBS=1", out)

    def test_pip_install_gets_makeflags(self) -> None:
        """pip install serialization is via MAKEFLAGS (setup.py fanout)."""
        out = _serialize_build_command("pip install .")
        self.assertIn("MAKEFLAGS=-j1", out)

    def test_npm_install_gets_jobs_flag(self) -> None:
        """npm install gains --jobs 1."""
        out = _serialize_build_command("npm install")
        self.assertIn("npm install --jobs 1", out)


class TestSuspectedOOM(unittest.TestCase):
    """Tests for the suspected-OOM detector."""

    def test_exit_137_go(self) -> None:
        """Exit 137 from a go build is treated as suspected OOM."""
        self.assertTrue(
            _is_suspected_oom(_Res(137, "", "", False), "go build ./x")
        )

    def test_empty_output_build(self) -> None:
        """Failed go builds with empty output are treated as OOM."""
        self.assertTrue(
            _is_suspected_oom(_Res(1, "", "", False), "go build ./x")
        )

    def test_real_error_not_oom(self) -> None:
        """Go build failures with stderr are not suspected OOM."""
        self.assertFalse(
            _is_suspected_oom(_Res(1, "", "boom", False), "go build ./x")
        )

    def test_exit_137_any_command(self) -> None:
        """Exit 137 is treated as OOM even for non-build commands."""
        self.assertTrue(
            _is_suspected_oom(
                _Res(137, "", "", False), "git clone https://x /tmp/y"
            )
        )

    def test_cargo_fetch_exit137(self) -> None:
        """cargo fetch OOM (exit 137) is caught."""
        self.assertTrue(
            _is_suspected_oom(_Res(137, "", "", False), "cargo fetch")
        )

    def test_npm_install_empty_output(self) -> None:
        """npm install failing with no output is treated as OOM."""
        self.assertTrue(
            _is_suspected_oom(_Res(1, "", "", False), "npm install")
        )


class TestHeaderResolution(unittest.TestCase):
    """Tests for header->package resolution."""

    def test_png_header_resolves(self) -> None:
        """Missing png.h resolves to the Alpine libpng package."""
        from src.platforms import ALPINE_RISCV

        pkgs = _resolve_header_to_packages(
            "fatal error: png.h: No such file or directory", ALPINE_RISCV
        )
        self.assertEqual(pkgs, ["libpng-dev"])

    def test_nested_ogg_header_resolves(self) -> None:
        """Nested ogg headers resolve to the Debian libogg package."""
        from src.platforms import DEBIAN_RISCV

        pkgs = _resolve_header_to_packages(
            "fatal error: ogg/ogg.h: No such file", DEBIAN_RISCV
        )
        self.assertEqual(pkgs, ["libogg-dev"])

    def test_unknown_header_not_guessed(self) -> None:
        """Unknown headers do not produce guessed package names."""
        from src.platforms import ALPINE_RISCV

        pkgs = _resolve_header_to_packages(
            "fatal error: my_internal_thing.h: No such file", ALPINE_RISCV
        )
        self.assertEqual(pkgs, [])


class TestPythonModuleResolution(unittest.TestCase):
    """Tests for missing-Python-module resolution."""

    def test_jinja2_module(self) -> None:
        """Missing jinja2 modules resolve to the jinja2 package."""
        pkgs = _resolve_missing_python_modules(
            "ModuleNotFoundError: No module named 'jinja2'"
        )
        self.assertEqual(pkgs, ["jinja2"])

    def test_yaml_maps_to_pyyaml(self) -> None:
        """Missing yaml modules resolve to the pyyaml package."""
        pkgs = _resolve_missing_python_modules(
            "ModuleNotFoundError: No module named 'yaml'"
        )
        self.assertEqual(pkgs, ["pyyaml"])

    def test_dotted_module_uses_top_level(self) -> None:
        """Dotted module errors resolve by top-level module name."""
        pkgs = _resolve_missing_python_modules(
            "No module named 'google.protobuf'"
        )
        self.assertEqual(pkgs, ["protobuf"])

    def test_no_module_error(self) -> None:
        """Errors without module names produce no Python packages."""
        self.assertEqual(_resolve_missing_python_modules("some error"), [])


class TestGoToolchainDownloadCommand(unittest.TestCase):
    """Tests for the go.dev tarball installer shell command."""

    def test_version_appears_in_url(self) -> None:
        """The command references the requested Go version in the URL."""
        cmd = _download_go_toolchain_cmd("1.25.0")
        self.assertIn("go1.25.0.linux-riscv64.tar.gz", cmd)
        self.assertIn("https://go.dev/dl/", cmd)

    def test_command_replaces_usr_local_go(self) -> None:
        """The command wipes /usr/local/go before untar to avoid mixing."""
        cmd = _download_go_toolchain_cmd("1.26.3")
        self.assertIn("rm -rf /usr/local/go", cmd)
        self.assertIn("tar -xzf", cmd)
        self.assertIn("/usr/local/go/bin/go version", cmd)

    def test_command_uses_set_e(self) -> None:
        """The command uses ``set -e`` so any step failing aborts install."""
        cmd = _download_go_toolchain_cmd("1.25.0")
        self.assertTrue(cmd.startswith("set -e && "))


class TestRepoGitmodules(unittest.TestCase):
    """Tests for ``.gitmodules`` presence detection."""

    def test_present(self) -> None:
        """Returns True when the file exists."""
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, ".gitmodules"), "w") as f:
                f.write("x")
            self.assertTrue(_repo_has_gitmodules(d))

    def test_absent(self) -> None:
        """Returns False when the file does not exist."""
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(_repo_has_gitmodules(d))

    def test_none_path_is_safe(self) -> None:
        """Passing ``None`` does not raise."""
        self.assertFalse(_repo_has_gitmodules(None))


class TestNpmScripts(unittest.TestCase):
    """Tests for reading npm scripts from package.json."""

    def _write_pkg(self, repo, payload) -> None:
        """Helper: write ``package.json`` inside ``repo`` from a dict."""
        with open(os.path.join(repo, "package.json"), "w") as f:
            import json as _json

            _json.dump(payload, f)

    def test_missing_package_json(self) -> None:
        """No package.json → empty list."""
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(_npm_scripts(d), [])

    def test_present_scripts(self) -> None:
        """Scripts dict is returned as an ordered list."""
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            self._write_pkg(d, {"scripts": {"test": "jest", "prod": "x"}})
            self.assertEqual(_npm_scripts(d), ["test", "prod"])

    def test_no_scripts_key(self) -> None:
        """Missing scripts key → empty list, not KeyError."""
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            self._write_pkg(d, {"name": "foo"})
            self.assertEqual(_npm_scripts(d), [])

    def test_unparseable_json(self) -> None:
        """Garbage in package.json → empty list, no crash."""
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "package.json"), "w") as f:
                f.write("{ not json")
            self.assertEqual(_npm_scripts(d), [])


if __name__ == "__main__":
    unittest.main()

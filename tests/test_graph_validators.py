"""Tests for graph.py validators: build plan, fix command, fixer response, predictions."""

import unittest

from src.graph import (
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
    return BuildPlan(
        build_system="cmake",
        build_system_confidence=0.95,
        phases=[BuildPhase(id=1, name="build", commands=list(commands))],
        total_estimated_duration="5m",
    )


# ---------- validate_build_plan ----------


class TestValidateBuildPlan(unittest.TestCase):
    def test_clean_plan_passes(self):
        ok, msg = validate_build_plan(_plan("cmake -B build -S .", "make -j4"))
        self.assertTrue(ok, msg)

    def test_placeholder_path_rejected(self):
        ok, msg = validate_build_plan(_plan("cp /path/to/foo bar"))
        self.assertFalse(ok)
        self.assertIn("Hallucination", msg)

    def test_your_username_placeholder_rejected(self):
        ok, msg = validate_build_plan(_plan("cd /home/your_username/foo"))
        self.assertFalse(ok)

    def test_example_com_placeholder_rejected(self):
        ok, _ = validate_build_plan(_plan("wget https://example.com/foo.tar.gz"))
        self.assertFalse(ok)

    def test_riscv_unknown_linux_gnu_gcc_rejected(self):
        ok, _ = validate_build_plan(_plan("riscv64-unknown-linux-gnu-gcc -O2 main.c"))
        self.assertFalse(ok)

    def test_home_path_outside_workspace_rejected(self):
        ok, _ = validate_build_plan(_plan("cp /home/akif/foo bar"))
        self.assertFalse(ok)

    def test_home_inside_workspace_path_passes(self):
        # /home/.../workspace/... is fine
        ok, _ = validate_build_plan(_plan("ls /home/akif/workspace/foo"))
        self.assertTrue(ok)

    def test_go_subcommand_as_apk_package_rejected(self):
        ok, msg = validate_build_plan(_plan("apk add go mod"))
        self.assertFalse(ok)
        self.assertIn("Go subcommand", msg)

    def test_go_subcommand_as_apt_package_rejected(self):
        ok, _ = validate_build_plan(_plan("apt-get install build"))
        self.assertFalse(ok)

    def test_apt_install_with_flags_handles_pkg_list(self):
        ok, _ = validate_build_plan(_plan("apt-get install -y libssl-dev curl"))
        self.assertTrue(ok)

    def test_nested_cmake_build_dir_rejected(self):
        ok, msg = validate_build_plan(_plan("cd build && cmake -S . -B build .."))
        self.assertFalse(ok)
        self.assertIn("nested build dir", msg)


# ---------- validate_fix_command ----------


class TestValidateFixCommand(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
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

    def test_dangerous_commands_rejected(self):
        for cmd, label in self.dangerous:
            with self.subTest(cmd=cmd, label=label):
                ok, _ = validate_fix_command(cmd)
                self.assertFalse(ok, f"SHOULD REJECT [{label}]: {cmd}")

    def test_safe_commands_pass(self):
        for cmd in self.safe:
            with self.subTest(cmd=cmd):
                ok, reason = validate_fix_command(cmd)
                self.assertTrue(ok, f"SHOULD PASS: {cmd} -> {reason}")


# ---------- validate_fixer_response ----------


class TestValidateFixerResponse(unittest.TestCase):
    def _base(self, actions):
        return {
            "strategies": [{"id": 1, "actions": actions}],
            "recommended_strategy_id": 1,
            "reflection": {"root_cause": "x", "this_fix_will_work_because": "y"},
        }

    def test_non_dict_rejected(self):
        ok, _ = validate_fixer_response([])  # type: ignore[arg-type]
        self.assertFalse(ok)

    def test_no_strategies_rejected(self):
        ok, msg = validate_fixer_response({})
        self.assertFalse(ok)
        self.assertIn("strategies", msg.lower())

    def test_recommended_id_not_found_rejected(self):
        data = self._base([{"type": "command", "command": "ls"}])
        data["recommended_strategy_id"] = 99
        ok, msg = validate_fixer_response(data)
        self.assertFalse(ok)
        self.assertIn("not found", msg)

    def test_empty_actions_rejected(self):
        ok, msg = validate_fixer_response(self._base([]))
        self.assertFalse(ok)
        self.assertIn("no actions", msg.lower())

    def test_create_file_with_empty_content_rejected(self):
        ok, msg = validate_fixer_response(self._base([
            {"type": "create_file", "path": "foo.c", "content": "   "}
        ]))
        self.assertFalse(ok)
        self.assertIn("empty content", msg)

    def test_create_file_with_absolute_home_path_rejected(self):
        ok, _ = validate_fixer_response(self._base([
            {"type": "create_file", "path": "/home/x/foo.c", "content": "int main(){}"}
        ]))
        self.assertFalse(ok)

    def test_create_file_missing_path_rejected(self):
        ok, _ = validate_fixer_response(self._base([
            {"type": "create_file", "path": "", "content": "x"}
        ]))
        self.assertFalse(ok)

    def test_empty_command_rejected(self):
        ok, msg = validate_fixer_response(self._base([
            {"type": "command", "command": "   "}
        ]))
        self.assertFalse(ok)
        self.assertIn("empty", msg.lower())

    def test_self_copy_command_rejected(self):
        ok, msg = validate_fixer_response(self._base([
            {"type": "command", "command": "cp -r src/* src/"}
        ]))
        self.assertFalse(ok)
        self.assertIn("copying to self", msg)

    def test_unsafe_embedded_command_rejected(self):
        ok, msg = validate_fixer_response(self._base([
            {"type": "command", "command": "touch foo.go"}
        ]))
        self.assertFalse(ok)
        self.assertIn("Unsafe", msg)

    def test_valid_response_accepted(self):
        ok, msg = validate_fixer_response(self._base([
            {"type": "command", "command": "make clean"},
            {"type": "create_file", "path": "patch.diff", "content": "--- a\n+++ b\n"},
        ]))
        self.assertTrue(ok, msg)


# ---------- predict_build_issues ----------


class TestPredictBuildIssues(unittest.TestCase):
    def test_no_plan_returns_empty(self):
        state = create_initial_state("https://x/y.git")
        self.assertEqual(predict_build_issues(state), [])

    def test_go_build_without_buildvcs_flagged(self):
        state = create_initial_state("https://x/y.git")
        state.build_plan = _plan("go build ./cmd/foo")
        preds = predict_build_issues(state)
        self.assertTrue(any(p["issue"] == "Go VCS ownership error" for p in preds))

    def test_go_build_with_buildvcs_false_not_flagged(self):
        state = create_initial_state("https://x/y.git")
        state.build_plan = _plan("go build -buildvcs=false ./cmd/foo")
        preds = predict_build_issues(state)
        self.assertFalse(any(p["issue"] == "Go VCS ownership error" for p in preds))

    def test_cmake_with_high_severity_arch_code_flagged(self):
        state = create_initial_state("https://x/y.git")
        state.build_plan = _plan("cmake -B build -S .")
        state.arch_specific_code = [
            ArchSpecificCode(file="x.c", line=10, code_snippet="_mm_add_ps",
                             arch_type="x86", severity="high",
                             suggested_fix="fallback")
        ]
        preds = predict_build_issues(state)
        self.assertTrue(any(p["pattern"] == "arch_specific" for p in preds))

    def test_apk_install_with_missing_tools_flagged(self):
        state = create_initial_state("https://x/y.git")
        state.build_plan = _plan("apk add zlib-dev")
        state.context_cache["missing_tools"] = ["protoc"]
        preds = predict_build_issues(state)
        self.assertTrue(any(p["pattern"] == "missing_tools" for p in preds))


if __name__ == "__main__":
    unittest.main()

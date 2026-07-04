"""Tests for the LLM-value paths in src/graph.py.

Covers the analyst node (consumed PackageAnalysis + real cost logging),
the fixer's read-only investigation helper, and expected-artifact
verification.
"""

import unittest
from unittest import mock

from src.llm_helpers import LLMCallOutcome
from src.state import (
    BuildStatus,
    ErrorCategory,
    PackageAnalysis,
    create_initial_state,
)


def _state(**overrides):
    """Build a base AgentState with field overrides."""
    s = create_initial_state("https://github.com/a/b.git")
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


_ANALYST_DATA = {
    "purpose": "A DNS enumeration CLI tool",
    "language": "Go",
    "build_system": {
        "type": "go",
        "confidence": 0.95,
        "reasoning": "go.mod declares module, main.go at root",
    },
    "dependencies": [{"name": "git", "reason": "go mod fetch"}],
    "riscv_risks": ["cgo disabled path unclear"],
    "build_strategy": "go build the root module with -buildvcs=false.",
    "expected_artifacts": ["assetfinder"],
    "needs_custom_plan": False,
    "complexity": 3,
}


class TestAnalystNode(unittest.TestCase):
    """Tests for analyst_node."""

    def _run(self, outcome):
        """Run analyst_node with the LLM layer mocked out."""
        from src import graph

        s = _state()
        with (
            mock.patch.object(
                graph, "llm_call_with_validation", return_value=outcome
            ),
            mock.patch.object(
                graph,
                "get_model_pool_for_role",
                return_value=[mock.MagicMock()],
            ),
            mock.patch.object(
                graph, "collect_build_evidence", return_value="### go.mod\n..."
            ),
        ):
            return graph.analyst_node(s)

    def test_valid_analysis_is_stored_and_consumable(self) -> None:
        """A valid LLM response becomes an LLM-grounded analysis."""
        out = self._run(
            LLMCallOutcome(
                data=dict(_ANALYST_DATA),
                used_fallback=False,
                attempts=1,
                input_tokens=1200,
                output_tokens=300,
                cost_usd=0.0,
            )
        )
        pa = out.package_analysis
        self.assertIsNotNone(pa)
        self.assertTrue(pa.llm_grounded)
        self.assertEqual(pa.build_system, "go")
        self.assertEqual(pa.expected_artifacts, ["assetfinder"])
        self.assertFalse(pa.needs_custom_plan)
        # Structural task plan exists without spending LLM tokens on it
        self.assertIsNotNone(out.task_plan)

    def test_real_usage_is_logged(self) -> None:
        """The node logs the outcome's REAL tokens/cost, not a flat fee."""
        out = self._run(
            LLMCallOutcome(
                data=dict(_ANALYST_DATA),
                used_fallback=False,
                attempts=2,
                input_tokens=2000,
                output_tokens=500,
                cost_usd=0.0125,
            )
        )
        self.assertEqual(out.api_tokens_in, 2000)
        self.assertEqual(out.api_tokens_out, 500)
        self.assertAlmostEqual(out.api_cost_usd, 0.0125)
        self.assertEqual(out.api_calls_made, 2)

    def test_llm_starvation_falls_back_deterministically(self) -> None:
        """No LLM data → deterministic, non-grounded analysis."""
        out = self._run(
            LLMCallOutcome(
                data=None,
                used_fallback=False,
                attempts=3,
                last_error="429 everywhere",
            )
        )
        pa = out.package_analysis
        self.assertIsNotNone(pa)
        self.assertFalse(pa.llm_grounded)
        self.assertIsNotNone(out.task_plan)


class TestFixerInvestigation(unittest.TestCase):
    """Tests for _run_fixer_investigation."""

    def _fake_exec(self, stdout="output", stderr="", exit_code=0):
        """Build a fake execute_command result."""
        return mock.MagicMock(
            stdout=stdout, stderr=stderr, exit_code=exit_code, success=True
        )

    def test_read_only_commands_execute(self) -> None:
        """Whitelisted commands run and their output is captured."""
        from src import graph

        s = _state()
        with mock.patch.object(
            graph, "execute_command", return_value=self._fake_exec("hello")
        ) as exec_mock:
            out = graph._run_fixer_investigation(
                s, ["cat Makefile", "grep -rn simd src/"]
            )
        self.assertEqual(exec_mock.call_count, 2)
        self.assertIn("$ cat Makefile", out)
        self.assertIn("hello", out)

    def test_mutating_commands_are_rejected(self) -> None:
        """Non-whitelisted or redirecting commands never execute."""
        from src import graph

        s = _state()
        with mock.patch.object(graph, "execute_command") as exec_mock:
            out = graph._run_fixer_investigation(
                s,
                [
                    "rm -rf /",
                    "sed -i s/a/b/ Makefile",
                    "cat foo > bar",
                ],
            )
        exec_mock.assert_not_called()
        self.assertEqual(out.count("[rejected"), 3)

    def test_command_count_is_capped(self) -> None:
        """At most 4 commands run per investigation round."""
        from src import graph

        s = _state()
        with mock.patch.object(
            graph, "execute_command", return_value=self._fake_exec()
        ) as exec_mock:
            graph._run_fixer_investigation(s, ["ls"] * 10)
        self.assertEqual(exec_mock.call_count, 4)

    def test_output_is_truncated(self) -> None:
        """Command output is capped so the prompt stays small."""
        from src import graph

        s = _state()
        with mock.patch.object(
            graph,
            "execute_command",
            return_value=self._fake_exec("x" * 5000),
        ):
            out = graph._run_fixer_investigation(s, ["cat big.txt"])
        self.assertIn("[... truncated ...]", out)
        self.assertLess(len(out), 2500)


class TestExpectedArtifactVerification(unittest.TestCase):
    """Tests for expected-artifact handling in verify_node."""

    def _scanner(self):
        """Fake ArtifactScanner that finds nothing scannable."""
        scanner = mock.MagicMock()
        scanner.scan.return_value = []
        scanner.verify_build_success.return_value = (False, "no artifacts")
        scanner.get_summary.return_value = {"by_architecture": {}}
        return scanner

    def test_riscv_expected_artifact_seals_success(self) -> None:
        """An expected artifact found as RISC-V verifies the build."""
        from src import graph

        s = _state(
            package_analysis=PackageAnalysis(
                purpose="t",
                expected_artifacts=["mytool"],
                llm_grounded=True,
            )
        )
        with (
            mock.patch.object(
                graph, "ArtifactScanner", return_value=self._scanner()
            ),
            mock.patch.object(
                graph,
                "_locate_expected_artifacts",
                return_value=[
                    ("/usr/local/bin/mytool", "ELF 64-bit LSB pie, RISC-V")
                ],
            ),
        ):
            out = graph.verify_node(s)
        self.assertEqual(out.build_status, BuildStatus.SUCCESS)
        self.assertTrue(
            any(
                a["filepath"] == "/usr/local/bin/mytool"
                for a in out.build_artifacts
            )
        )

    def test_wrong_arch_expected_artifact_fails(self) -> None:
        """An expected artifact that is x86 must FAIL verification."""
        from src import graph

        s = _state(
            package_analysis=PackageAnalysis(
                purpose="t",
                expected_artifacts=["mytool"],
                llm_grounded=True,
            )
        )
        with (
            mock.patch.object(
                graph, "ArtifactScanner", return_value=self._scanner()
            ),
            mock.patch.object(
                graph,
                "_locate_expected_artifacts",
                return_value=[("/usr/local/bin/mytool", "ELF 64-bit, x86-64")],
            ),
        ):
            out = graph.verify_node(s)
        self.assertEqual(out.build_status, BuildStatus.FAILED)
        self.assertEqual(out.last_error_category, ErrorCategory.ARCHITECTURE)

    def test_missing_expected_artifacts_recorded_in_caveat(self) -> None:
        """Unfound expected artifacts land in the verification caveat."""
        from src import graph

        s = _state(
            package_analysis=PackageAnalysis(
                purpose="t",
                expected_artifacts=["mytool"],
                llm_grounded=True,
            )
        )
        with (
            mock.patch.object(
                graph, "ArtifactScanner", return_value=self._scanner()
            ),
            mock.patch.object(
                graph, "_locate_expected_artifacts", return_value=[]
            ),
        ):
            out = graph.verify_node(s)
        self.assertEqual(out.build_status, BuildStatus.SUCCESS)
        caveat = out.context_cache["artifact_verification"]
        self.assertFalse(caveat["verified"])
        self.assertEqual(caveat["expected_missing"], ["mytool"])

    def test_expectation_names_are_sanitized(self) -> None:
        """Path-ish or globby expectations never reach find."""
        from src import graph

        s = _state()
        with mock.patch.object(
            graph,
            "execute_command",
            return_value=mock.MagicMock(stdout="", stderr="", exit_code=1),
        ) as exec_mock:
            hits = graph._locate_expected_artifacts(
                s, ["../evil", "a*b", "$(rm)", "ok-name"]
            )
        # Only "ok-name" is searched (one find; file only on matches)
        self.assertEqual(exec_mock.call_count, 1)
        self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()

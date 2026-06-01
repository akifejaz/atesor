"""Tests for src/artifact_scanner.py.

Covers architecture detection and build verification.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

from src.artifact_scanner import ArtifactScanner


def _ok(stdout=""):
    """Ok."""
    return SimpleNamespace(
        success=True,
        failed=False,
        stdout=stdout,
        stderr="",
        exit_code=0,
        command="",
        duration_seconds=0.0,
        timestamp="",
    )


class TestDetectArchitecture(unittest.TestCase):
    """_detect_architecture must classify `file` output correctly."""

    CASES = [
        (
            "ELF 64-bit LSB pie executable, UCB RISC-V, RVC, double-float ABI",
            "RISC-V",
        ),
        ("ELF 64-bit LSB executable, riscv64", "RISC-V"),
        ("ELF 64-bit LSB executable, x86-64, version 1 (SYSV)", "x64"),
        ("ELF 64-bit LSB executable, x86_64", "x64"),
        ("ELF 64-bit LSB executable, ARM aarch64", "ARM64"),
        ("ELF 64-bit LSB executable, arm64", "ARM64"),
        ("ELF 32-bit LSB executable, ARM, EABI5", "ARM32"),
        ("ELF 32-bit LSB executable, Intel 80386", "x86"),
        ("ASCII text", None),
        ("", None),
    ]

    def test_detect_each_architecture(self) -> None:
        """Test detect each architecture."""
        sc = ArtifactScanner("/tmp/build")
        for info, expected in self.CASES:
            with self.subTest(info=info):
                self.assertEqual(sc._detect_architecture(info), expected)

    def test_case_insensitive(self) -> None:
        """Test case insensitive."""
        sc = ArtifactScanner("/tmp/build")
        self.assertEqual(sc._detect_architecture("RISC-V 64-bit"), "RISC-V")
        self.assertEqual(sc._detect_architecture("X86_64 binary"), "x64")


class TestVerifyBuildSuccess(unittest.TestCase):
    """Tests for VerifyBuildSuccess."""

    def _scanner_with(self, artifacts):
        """Scanner with."""
        sc = ArtifactScanner("/tmp/build")
        sc.artifacts = artifacts
        return sc

    def test_no_artifacts_fails(self) -> None:
        """Test no artifacts fails."""
        sc = self._scanner_with([])
        ok, msg = sc.verify_build_success()
        self.assertFalse(ok)
        self.assertIn("No build artifacts", msg)

    def test_riscv_artifacts_succeed(self) -> None:
        """Test riscv artifacts succeed."""
        sc = self._scanner_with(
            [
                {
                    "filepath": "/build/a",
                    "type": "binary",
                    "architecture": "RISC-V",
                }
            ]
        )
        ok, msg = sc.verify_build_success()
        self.assertTrue(ok)
        self.assertIn("RISC-V", msg)

    def test_x64_only_artifacts_fail(self) -> None:
        """Test x64 only artifacts fail."""
        sc = self._scanner_with(
            [{"filepath": "/build/a", "type": "binary", "architecture": "x64"}]
        )
        ok, msg = sc.verify_build_success()
        self.assertFalse(ok)
        self.assertIn("not for RISC-V", msg)
        self.assertIn("x64", msg)

    def test_mixed_arches_with_riscv_succeed(self) -> None:
        # If even one RISC-V artifact is found, build counts as success
        """Test mixed arches with riscv succeed."""
        sc = self._scanner_with(
            [
                {
                    "filepath": "/build/a",
                    "type": "binary",
                    "architecture": "RISC-V",
                },
                {
                    "filepath": "/build/b",
                    "type": "binary",
                    "architecture": "x64",
                },
            ]
        )
        ok, _ = sc.verify_build_success()
        self.assertTrue(ok)

    def test_unknown_arch_artifacts_fail(self) -> None:
        """Test unknown arch artifacts fail."""
        sc = self._scanner_with(
            [{"filepath": "/build/a", "type": "binary", "architecture": ""}]
        )
        ok, msg = sc.verify_build_success()
        self.assertFalse(ok)
        self.assertIn("could not detect architecture", msg)


class TestGetSummary(unittest.TestCase):
    """Tests for GetSummary."""

    def test_summary_counts_by_type_and_arch(self) -> None:
        """Test summary counts by type and arch."""
        sc = ArtifactScanner("/tmp/build")
        sc.artifacts = [
            {"type": "binary", "architecture": "RISC-V"},
            {"type": "binary", "architecture": "RISC-V"},
            {"type": "library_static", "architecture": "RISC-V"},
            {"type": "library_shared", "architecture": "x64"},
        ]
        s = sc.get_summary()
        self.assertEqual(s["total_artifacts"], 4)
        self.assertEqual(
            s["by_type"],
            {"binary": 2, "library_static": 1, "library_shared": 1},
        )
        self.assertEqual(
            s["by_arch" if "by_arch" in s else "by_architecture"],
            {"RISC-V": 3, "x64": 1},
        )
        self.assertTrue(s["has_riscv"])

    def test_empty_summary(self) -> None:
        """Test empty summary."""
        sc = ArtifactScanner("/tmp/build")
        s = sc.get_summary()
        self.assertEqual(s["total_artifacts"], 0)
        self.assertFalse(s["has_riscv"])


class TestScanIntegration(unittest.TestCase):
    """Verify the scan() pipeline calls the right helpers."""

    def test_scan_calls_find_and_file(self):
        """Test scan calls find and file."""
        sc = ArtifactScanner("/tmp/build", cwd="/tmp/build")

        # Build a sequence of fake responses for the find/file/stat chain
        def fake_exec(cmd, cwd=None, **kw):
            """Fake exec."""
            if cmd.startswith("find") and "executable" in cmd:
                return _ok("/tmp/build/foo\n")
            if cmd.startswith("find") and "*.a" in cmd:
                return _ok("")
            if cmd.startswith("find") and "*.so*" in cmd:
                return _ok("")
            if cmd.startswith("file "):
                return _ok(
                    "/tmp/build/foo: ELF 64-bit LSB executable, UCB RISC-V"
                )
            if "stat" in cmd:
                return _ok("12345")
            return _ok("")

        with mock.patch(
            "src.artifact_scanner.execute_command", side_effect=fake_exec
        ):
            artifacts = sc.scan()

        self.assertEqual(len(artifacts), 1)
        self.assertEqual(artifacts[0]["type"], "binary")
        self.assertEqual(artifacts[0]["architecture"], "RISC-V")
        self.assertEqual(artifacts[0]["size_bytes"], 12345)

    def test_scan_handles_no_artifacts(self) -> None:
        """Test scan handles no artifacts."""
        sc = ArtifactScanner("/tmp/build")
        with mock.patch(
            "src.artifact_scanner.execute_command", return_value=_ok("")
        ):
            artifacts = sc.scan()
        self.assertEqual(artifacts, [])


if __name__ == "__main__":
    unittest.main()

"""Tests for the ArtifactScanner module."""
import unittest
from unittest.mock import patch, call
from src.state import CommandResult
from src.artifact_scanner import ArtifactScanner


def _make_result(stdout="", stderr="", exit_code=0, command="test"):
    """Helper to create CommandResult objects for mocking."""
    return CommandResult(
        command=command,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        duration_seconds=0.1,
    )


MOCK_PATH = "src.artifact_scanner.execute_command"


class TestInit(unittest.TestCase):
    def test_build_dir_stored(self):
        scanner = ArtifactScanner("/build")
        self.assertEqual(scanner.build_dir, "/build")

    def test_cwd_defaults_to_build_dir(self):
        scanner = ArtifactScanner("/build")
        self.assertEqual(scanner.cwd, "/build")

    def test_cwd_custom(self):
        scanner = ArtifactScanner("/build", cwd="/workspace")
        self.assertEqual(scanner.cwd, "/workspace")

    def test_artifacts_initially_empty(self):
        scanner = ArtifactScanner("/build")
        self.assertEqual(scanner.artifacts, [])


class TestDetectArchitecture(unittest.TestCase):
    def setUp(self):
        self.scanner = ArtifactScanner("/build")

    def test_riscv_lower(self):
        self.assertEqual(self.scanner._detect_architecture("ELF 64-bit RISC-V"), "RISC-V")

    def test_riscv_via_riscv64(self):
        self.assertEqual(self.scanner._detect_architecture("riscv64 binary"), "RISC-V")

    def test_x64_via_x86_64(self):
        self.assertEqual(self.scanner._detect_architecture("ELF x86-64"), "x64")

    def test_x64_via_x86_64_underscore(self):
        self.assertEqual(self.scanner._detect_architecture("x86_64 GNU/Linux"), "x64")

    def test_arm64_via_aarch64(self):
        self.assertEqual(self.scanner._detect_architecture("ELF 64-bit aarch64"), "ARM64")

    def test_arm64_keyword(self):
        self.assertEqual(self.scanner._detect_architecture("ARM64 binary"), "ARM64")

    def test_arm32(self):
        self.assertEqual(self.scanner._detect_architecture("ELF ARM 32-bit"), "ARM32")

    def test_x86_via_80386(self):
        self.assertEqual(self.scanner._detect_architecture("Intel 80386 binary"), "x86")

    def test_x86_via_80386_plain(self):
        self.assertEqual(self.scanner._detect_architecture("80386"), "x86")

    def test_unknown_returns_none(self):
        self.assertIsNone(self.scanner._detect_architecture("ASCII text"))

    def test_empty_string_returns_none(self):
        self.assertIsNone(self.scanner._detect_architecture(""))


class TestGetFileSize(unittest.TestCase):
    def setUp(self):
        self.scanner = ArtifactScanner("/build")

    @patch(MOCK_PATH)
    def test_successful_size(self, mock_exec):
        mock_exec.return_value = _make_result(stdout="12345\n")
        size = self.scanner._get_file_size("/build/out.bin")
        self.assertEqual(size, 12345)

    @patch(MOCK_PATH)
    def test_failed_command_returns_zero(self, mock_exec):
        mock_exec.return_value = _make_result(exit_code=1)
        size = self.scanner._get_file_size("/build/missing")
        self.assertEqual(size, 0)

    @patch(MOCK_PATH)
    def test_non_numeric_returns_zero(self, mock_exec):
        mock_exec.return_value = _make_result(stdout="not_a_number\n")
        size = self.scanner._get_file_size("/build/bad")
        self.assertEqual(size, 0)


class TestGetArchiveArchitecture(unittest.TestCase):
    def setUp(self):
        self.scanner = ArtifactScanner("/build")

    @patch(MOCK_PATH)
    def test_successful_riscv_archive(self, mock_exec):
        mock_exec.return_value = _make_result(stdout="foo.o: ELF 64-bit RISC-V")
        arch = self.scanner._get_archive_architecture("/build/libfoo.a")
        self.assertEqual(arch, "RISC-V")

    @patch(MOCK_PATH)
    def test_failed_command_returns_none(self, mock_exec):
        mock_exec.return_value = _make_result(exit_code=1, stdout="")
        arch = self.scanner._get_archive_architecture("/build/libfoo.a")
        self.assertIsNone(arch)

    @patch(MOCK_PATH)
    def test_exception_returns_none(self, mock_exec):
        mock_exec.side_effect = RuntimeError("boom")
        arch = self.scanner._get_archive_architecture("/build/libfoo.a")
        self.assertIsNone(arch)


class TestCheckArtifact(unittest.TestCase):
    def setUp(self):
        self.scanner = ArtifactScanner("/build")

    @patch(MOCK_PATH)
    def test_binary_artifact(self, mock_exec):
        mock_exec.side_effect = [
            # file command
            _make_result(stdout="/build/app: ELF 64-bit RISC-V executable"),
            # stat command for size
            _make_result(stdout="4096\n"),
        ]
        self.scanner._check_artifact("/build/app", "binary")
        self.assertEqual(len(self.scanner.artifacts), 1)
        art = self.scanner.artifacts[0]
        self.assertEqual(art["filepath"], "/build/app")
        self.assertEqual(art["type"], "binary")
        self.assertEqual(art["architecture"], "RISC-V")
        self.assertEqual(art["size_bytes"], 4096)

    @patch(MOCK_PATH)
    def test_static_lib_with_ar_archive(self, mock_exec):
        mock_exec.side_effect = [
            # file command — reports ar archive
            _make_result(stdout="/build/libfoo.a: current ar archive"),
            # ar x + file *.o for archive arch detection
            _make_result(stdout="foo.o: ELF 64-bit RISC-V"),
            # stat command for size
            _make_result(stdout="8192\n"),
        ]
        self.scanner._check_artifact("/build/libfoo.a", "library_static")
        self.assertEqual(len(self.scanner.artifacts), 1)
        art = self.scanner.artifacts[0]
        self.assertEqual(art["type"], "library_static")
        self.assertEqual(art["architecture"], "RISC-V")

    @patch(MOCK_PATH)
    def test_file_command_failure_skips(self, mock_exec):
        mock_exec.return_value = _make_result(exit_code=1)
        self.scanner._check_artifact("/build/bad", "binary")
        self.assertEqual(len(self.scanner.artifacts), 0)


class TestScan(unittest.TestCase):
    def setUp(self):
        self.scanner = ArtifactScanner("/build")

    @patch(MOCK_PATH)
    def test_scan_finds_binaries_and_libs(self, mock_exec):
        mock_exec.side_effect = [
            # find executables
            _make_result(stdout="/build/app\n"),
            # file for app
            _make_result(stdout="/build/app: ELF 64-bit RISC-V"),
            # stat for app
            _make_result(stdout="1024\n"),
            # find *.a
            _make_result(stdout="/build/libfoo.a\n"),
            # file for libfoo.a
            _make_result(stdout="/build/libfoo.a: current ar archive"),
            # ar x for archive arch
            _make_result(stdout="foo.o: ELF 64-bit RISC-V"),
            # stat for libfoo.a
            _make_result(stdout="2048\n"),
            # find *.so*
            _make_result(stdout="/build/libbar.so\n"),
            # file for libbar.so
            _make_result(stdout="/build/libbar.so: ELF shared object RISC-V"),
            # stat for libbar.so
            _make_result(stdout="4096\n"),
        ]
        results = self.scanner.scan()
        self.assertEqual(len(results), 3)
        types = {a["type"] for a in results}
        self.assertEqual(types, {"binary", "library_static", "library_shared"})

    @patch(MOCK_PATH)
    def test_scan_empty_results(self, mock_exec):
        mock_exec.side_effect = [
            _make_result(exit_code=1),  # find binaries fails
            _make_result(stdout=""),     # find *.a empty
            _make_result(stdout=""),     # find *.so* empty
        ]
        results = self.scanner.scan()
        self.assertEqual(results, [])

    @patch(MOCK_PATH)
    def test_scan_resets_artifacts(self, mock_exec):
        self.scanner.artifacts = [{"filepath": "old", "type": "binary"}]
        mock_exec.return_value = _make_result(stdout="", exit_code=1)
        self.scanner.scan()
        # Old artifacts should be cleared
        self.assertNotIn({"filepath": "old", "type": "binary"}, self.scanner.artifacts)


class TestGetSummary(unittest.TestCase):
    def setUp(self):
        self.scanner = ArtifactScanner("/build")

    def test_empty_summary(self):
        summary = self.scanner.get_summary()
        self.assertEqual(summary["total_artifacts"], 0)
        self.assertEqual(summary["by_type"], {})
        self.assertEqual(summary["by_architecture"], {})
        self.assertFalse(summary["has_riscv"])
        self.assertEqual(summary["artifacts"], [])

    def test_mixed_artifacts(self):
        self.scanner.artifacts = [
            {"filepath": "/build/app", "type": "binary", "architecture": "RISC-V"},
            {"filepath": "/build/lib.a", "type": "library_static", "architecture": "RISC-V"},
            {"filepath": "/build/tool", "type": "binary", "architecture": "x64"},
        ]
        summary = self.scanner.get_summary()
        self.assertEqual(summary["total_artifacts"], 3)
        self.assertEqual(summary["by_type"], {"binary": 2, "library_static": 1})
        self.assertEqual(summary["by_architecture"], {"RISC-V": 2, "x64": 1})
        self.assertTrue(summary["has_riscv"])

    def test_no_architecture_skipped_in_by_arch(self):
        self.scanner.artifacts = [
            {"filepath": "/build/x", "type": "binary", "architecture": ""},
        ]
        summary = self.scanner.get_summary()
        self.assertEqual(summary["by_architecture"], {})


class TestVerifyBuildSuccess(unittest.TestCase):
    def setUp(self):
        self.scanner = ArtifactScanner("/build")

    def test_no_artifacts(self):
        success, msg = self.scanner.verify_build_success()
        self.assertFalse(success)
        self.assertIn("No build artifacts", msg)

    def test_riscv_artifacts(self):
        self.scanner.artifacts = [
            {"filepath": "/build/app", "type": "binary", "architecture": "RISC-V"},
            {"filepath": "/build/lib.a", "type": "library_static", "architecture": "RISC-V"},
        ]
        success, msg = self.scanner.verify_build_success()
        self.assertTrue(success)
        self.assertIn("RISC-V", msg)
        self.assertIn("2", msg)

    def test_non_riscv_artifacts(self):
        self.scanner.artifacts = [
            {"filepath": "/build/app", "type": "binary", "architecture": "x64"},
        ]
        success, msg = self.scanner.verify_build_success()
        self.assertFalse(success)
        self.assertIn("not for RISC-V", msg)
        self.assertIn("x64", msg)

    def test_unknown_architecture(self):
        self.scanner.artifacts = [
            {"filepath": "/build/blob", "type": "binary", "architecture": ""},
        ]
        success, msg = self.scanner.verify_build_success()
        self.assertFalse(success)
        self.assertIn("could not detect architecture", msg)


class TestArtifactPatterns(unittest.TestCase):
    def test_is_dict(self):
        self.assertIsInstance(ArtifactScanner.ARTIFACT_PATTERNS, dict)

    def test_expected_keys(self):
        expected = {"library_static", "library_shared", "binary", "test", "header"}
        self.assertEqual(set(ArtifactScanner.ARTIFACT_PATTERNS.keys()), expected)

    def test_values_are_lists(self):
        for key, val in ArtifactScanner.ARTIFACT_PATTERNS.items():
            self.assertIsInstance(val, list, f"Pattern for {key} should be a list")


if __name__ == "__main__":
    unittest.main()

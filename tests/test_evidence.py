"""Tests for src/evidence.py — repo evidence bundling for LLM prompts."""

import os
import tempfile
import unittest

from src.evidence import collect_build_evidence, error_context_excerpts


class TestCollectBuildEvidence(unittest.TestCase):
    """Tests for collect_build_evidence."""

    def setUp(self) -> None:
        """Create a fake repo with build files and docs."""
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = self.tmp.name
        with open(os.path.join(self.repo, "CMakeLists.txt"), "w") as f:
            f.write(
                "cmake_minimum_required(VERSION 3.10)\n"
                "project(demo C)\n"
                "find_package(ZLIB REQUIRED)\n"
            )
        with open(os.path.join(self.repo, "README.md"), "w") as f:
            f.write("# Demo\n\nBuild with cmake and make.\n")
        os.makedirs(os.path.join(self.repo, "src"))

    def tearDown(self) -> None:
        """Remove the fake repo."""
        self.tmp.cleanup()

    def test_bundle_contains_real_file_contents(self) -> None:
        """The bundle must contain the ACTUAL build file contents."""
        bundle = collect_build_evidence(self.repo)
        self.assertIn("find_package(ZLIB REQUIRED)", bundle)
        self.assertIn("### CMakeLists.txt", bundle)
        self.assertIn("Build with cmake and make.", bundle)

    def test_bundle_contains_top_level_listing(self) -> None:
        """The bundle lists top-level entries, marking directories."""
        bundle = collect_build_evidence(self.repo)
        self.assertIn("### Top-level files", bundle)
        self.assertIn("src/", bundle)

    def test_missing_files_are_skipped(self) -> None:
        """Absent build files produce no empty sections."""
        bundle = collect_build_evidence(self.repo)
        self.assertNotIn("### go.mod", bundle)
        self.assertNotIn("### Cargo.toml", bundle)

    def test_large_file_is_truncated(self) -> None:
        """Oversized files are cut to the per-file cap."""
        with open(os.path.join(self.repo, "Makefile"), "w") as f:
            f.write("x" * 10000)
        bundle = collect_build_evidence(self.repo)
        self.assertIn("[... truncated ...]", bundle)
        self.assertLess(len(bundle), 20000)

    def test_unreadable_repo_degrades_gracefully(self) -> None:
        """A nonexistent repo yields a bundle, not an exception."""
        bundle = collect_build_evidence("/nonexistent/path/xyz")
        self.assertIn("repository not readable", bundle)


class TestErrorContextExcerpts(unittest.TestCase):
    """Tests for error_context_excerpts."""

    def setUp(self) -> None:
        """Create a fake repo with a source file."""
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = self.tmp.name
        os.makedirs(os.path.join(self.repo, "src"))
        with open(os.path.join(self.repo, "src", "main.c"), "w") as f:
            f.write("\n".join(f"line{i}" for i in range(1, 51)))

    def tearDown(self) -> None:
        """Remove the fake repo."""
        self.tmp.cleanup()

    def test_extracts_context_around_error_line(self) -> None:
        """A file:line reference yields numbered source context."""
        error = (
            "src/main.c:20:5: error: unknown type name 'simd_t'\n"
            "make: *** [main.o] Error 1"
        )
        out = error_context_excerpts(error, self.repo)
        self.assertIn("src/main.c (around line 20)", out)
        self.assertIn("line20", out)
        self.assertIn("line10", out)  # ±12 lines of context
        self.assertNotIn("line50", out)

    def test_ignores_files_outside_repo(self) -> None:
        """Path traversal references are never followed."""
        error = "../../etc/passwd:1: error"
        self.assertEqual(error_context_excerpts(error, self.repo), "")

    def test_ignores_unreadable_references(self) -> None:
        """References to missing files are skipped silently."""
        error = "src/ghost.c:5: error: boom"
        self.assertEqual(error_context_excerpts(error, self.repo), "")

    def test_empty_error_returns_empty(self) -> None:
        """No error text, no excerpts."""
        self.assertEqual(error_context_excerpts("", self.repo), "")

    def test_deduplicates_repeated_references(self) -> None:
        """Multiple refs to one file yield a single excerpt."""
        error = "src/main.c:5: error a\nsrc/main.c:6: error b"
        out = error_context_excerpts(error, self.repo)
        self.assertEqual(out.count("### src/main.c"), 1)


if __name__ == "__main__":
    unittest.main()

"""Tests for src/knowledge.py – static knowledge base for RISC-V porting."""

import unittest

from src.knowledge import ALPINE_TOOL_MAP, get_system_knowledge_summary


class TestAlpineToolMap(unittest.TestCase):
    """Tests for the ALPINE_TOOL_MAP constant."""

    def test_is_dict(self):
        self.assertIsInstance(ALPINE_TOOL_MAP, dict)

    def test_is_non_empty(self):
        self.assertTrue(len(ALPINE_TOOL_MAP) > 0)

    def test_has_build_system_keys(self):
        for key in ("cmake", "make", "ninja", "meson", "autotools"):
            with self.subTest(key=key):
                self.assertIn(key, ALPINE_TOOL_MAP)

    def test_has_compiler_keys(self):
        for key in ("gcc", "g++", "pkgconfig"):
            with self.subTest(key=key):
                self.assertIn(key, ALPINE_TOOL_MAP)

    def test_has_language_keys(self):
        for key in ("go", "rust", "python3", "node", "java"):
            with self.subTest(key=key):
                self.assertIn(key, ALPINE_TOOL_MAP)

    def test_has_library_keys(self):
        for key in ("zlib", "openssl", "curl", "git"):
            with self.subTest(key=key):
                self.assertIn(key, ALPINE_TOOL_MAP)

    def test_correct_values(self):
        expected = {
            "cmake": "cmake",
            "make": "make",
            "ninja": "ninja",
            "meson": "meson",
            "autotools": "automake autoconf libtool",
            "gcc": "build-base",
            "g++": "build-base",
            "pkgconfig": "pkgconf",
            "go": "go",
            "rust": "rust cargo",
            "python3": "python3 py3-pip",
            "node": "nodejs npm",
            "java": "openjdk17",
            "zlib": "zlib-dev",
            "openssl": "openssl-dev",
            "curl": "curl-dev",
            "git": "git",
        }
        for tool, pkg in expected.items():
            with self.subTest(tool=tool):
                self.assertEqual(ALPINE_TOOL_MAP[tool], pkg)

    def test_all_values_are_non_empty_strings(self):
        for tool, pkg in ALPINE_TOOL_MAP.items():
            with self.subTest(tool=tool):
                self.assertIsInstance(pkg, str)
                self.assertTrue(len(pkg) > 0, f"Value for '{tool}' is empty")

    def test_all_keys_are_non_empty_strings(self):
        for key in ALPINE_TOOL_MAP:
            with self.subTest(key=key):
                self.assertIsInstance(key, str)
                self.assertTrue(len(key) > 0)


class TestGetSystemKnowledgeSummary(unittest.TestCase):
    """Tests for get_system_knowledge_summary()."""

    def setUp(self):
        self.summary = get_system_knowledge_summary()

    def test_returns_string(self):
        self.assertIsInstance(self.summary, str)

    def test_contains_header(self):
        self.assertIn("## RISC-V Tool Installation Knowledge (Feb 2026)", self.summary)

    def test_contains_apk_add_instruction(self):
        self.assertIn("apk add <package>", self.summary)

    def test_contains_alpine_linux_mention(self):
        self.assertIn("Alpine Linux (riscv64)", self.summary)

    def test_contains_common_tool_mappings_label(self):
        self.assertIn("Common Tool mappings:", self.summary)

    def test_contains_all_tool_package_mappings(self):
        for tool, pkg in ALPINE_TOOL_MAP.items():
            with self.subTest(tool=tool):
                expected_line = f"- {tool}: {pkg}"
                self.assertIn(expected_line, self.summary)

    def test_summary_is_non_empty(self):
        self.assertTrue(len(self.summary) > 0)

    def test_summary_ends_with_newline(self):
        self.assertTrue(self.summary.endswith("\n"))


if __name__ == "__main__":
    unittest.main()

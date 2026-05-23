"""Tests for src/artifact_curator.py — rule-based + LLM curation."""

import unittest
from unittest import mock

from src.artifact_curator import curate_artifacts


def _art(path, type="binary"):
    return {"filepath": path, "type": type, "architecture": "RISC-V"}


class TestRuleBasedCurator(unittest.TestCase):
    def test_empty_input_returns_empty(self):
        self.assertEqual(curate_artifacts([], "foo"), [])

    def test_cmake_internals_dropped(self):
        out = curate_artifacts(
            [
                _art("/b/CMakeFiles/CompilerIdC/a.out"),
                _art("/b/_deps/foo-build/x"),
                _art("/b/zlib"),
            ],
            "zlib",
        )
        paths = [a["filepath"] for a in out]
        self.assertNotIn("/b/CMakeFiles/CompilerIdC/a.out", paths)
        self.assertNotIn("/b/_deps/foo-build/x", paths)
        self.assertIn("/b/zlib", paths)

    def test_libtool_shim_dropped(self):
        out = curate_artifacts([_art("/b/.libs/lt-foo")], "foo")
        self.assertEqual(out, [])

    def test_conftest_dropped(self):
        out = curate_artifacts([_art("/b/conftest")], "foo")
        self.assertEqual(out, [])

    def test_meson_internals_dropped(self):
        for p in ["/b/meson-private/foo", "/b/meson-info/x", "/b/meson-logs/y"]:
            with self.subTest(path=p):
                out = curate_artifacts([_art(p)], "foo")
                self.assertEqual(out, [])

    def test_library_always_primary(self):
        out = curate_artifacts(
            [_art("/b/libfoo.so.1.2", type="library_shared")], "foo"
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["role"], "primary")

    def test_static_lib_primary(self):
        out = curate_artifacts([_art("/b/libfoo.a", type="library_static")], "foo")
        self.assertEqual(out[0]["role"], "primary")

    def test_repo_name_match_marks_primary(self):
        # Even in a test/ subdir, name match wins
        out = curate_artifacts([_art("/b/myapp")], "myapp")
        self.assertEqual(out[0]["role"], "primary")

    def test_test_path_marks_secondary(self):
        out = curate_artifacts([_art("/b/tests/runner")], "myapp")
        self.assertEqual(out[0]["role"], "secondary")

    def test_examples_path_marks_secondary(self):
        out = curate_artifacts([_art("/b/examples/demo")], "myapp")
        self.assertEqual(out[0]["role"], "secondary")

    def test_primary_listed_before_secondary(self):
        out = curate_artifacts(
            [
                _art("/b/tests/runner"),
                _art("/b/myapp"),
                _art("/b/examples/demo"),
            ],
            "myapp",
        )
        self.assertEqual(out[0]["role"], "primary")
        self.assertEqual(out[0]["filepath"], "/b/myapp")
        self.assertEqual(out[-1]["role"], "secondary")


class TestLLMCurator(unittest.TestCase):
    def _llm(self, response_text):
        m = mock.MagicMock()
        m.invoke.return_value = mock.MagicMock(content=response_text)
        return m

    def test_llm_classification_respected(self):
        artifacts = [
            _art("/b/foo"),
            _art("/b/tests/foo-test"),
            _art("/b/aux/probe"),
        ]
        llm = self._llm('{"primary": [1], "secondary": [2], "drop": [3]}')
        out = curate_artifacts(artifacts, "foo", build_system="cmake", llm=llm)
        roles = {a["filepath"]: a["role"] for a in out}
        self.assertEqual(roles.get("/b/foo"), "primary")
        self.assertEqual(roles.get("/b/tests/foo-test"), "secondary")
        # Dropped item absent
        self.assertNotIn("/b/aux/probe", roles)

    def test_malformed_json_falls_back_to_rules(self):
        artifacts = [_art("/b/foo")]
        llm = self._llm("garbage not json")
        out = curate_artifacts(artifacts, "foo", llm=llm)
        # rule-based keeps foo as primary
        self.assertEqual(out[0]["role"], "primary")

    def test_llm_exception_falls_back_to_rules(self):
        artifacts = [_art("/b/foo")]
        llm = mock.MagicMock()
        llm.invoke.side_effect = RuntimeError("rate limit")
        out = curate_artifacts(artifacts, "foo", llm=llm)
        self.assertEqual(out[0]["role"], "primary")

    def test_omitted_ids_default_to_secondary(self):
        artifacts = [_art("/b/a"), _art("/b/b"), _art("/b/c")]
        # LLM only mentions id 1 as primary; 2 and 3 are forgotten
        llm = self._llm('{"primary": [1], "secondary": [], "drop": []}')
        out = curate_artifacts(artifacts, "x", llm=llm)
        roles = {a["filepath"]: a["role"] for a in out}
        self.assertEqual(roles["/b/a"], "primary")
        self.assertEqual(roles["/b/b"], "secondary")
        self.assertEqual(roles["/b/c"], "secondary")

    def test_llm_dropping_everything_falls_back_to_rules(self):
        artifacts = [_art("/b/foo")]
        llm = self._llm('{"primary": [], "secondary": [], "drop": [1]}')
        out = curate_artifacts(artifacts, "foo", llm=llm)
        # Rules should restore foo
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["role"], "primary")

    def test_hard_noise_filtered_before_llm(self):
        # Even when LLM is provided, CMakeFiles paths get stripped before the prompt
        artifacts = [
            _art("/b/CMakeFiles/CompilerIdC/a.out"),
            _art("/b/foo"),
        ]
        llm = self._llm('{"primary": [1], "secondary": [], "drop": []}')
        out = curate_artifacts(artifacts, "foo", llm=llm)
        paths = [a["filepath"] for a in out]
        self.assertNotIn("/b/CMakeFiles/CompilerIdC/a.out", paths)
        # The prompt only saw one item, so id 1 == /b/foo
        self.assertIn("/b/foo", paths)


if __name__ == "__main__":
    unittest.main()

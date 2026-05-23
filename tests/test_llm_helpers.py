"""Tests for src/llm_helpers.py — validated LLM call with retry + fallback."""

import json
import unittest
from types import SimpleNamespace
from unittest import mock

from src.llm_helpers import (
    LLMCallOutcome,
    ValidationResult,
    extract_content,
    extract_json_block,
    llm_call_with_validation,
)


class TestExtractHelpers(unittest.TestCase):
    def test_extract_content_str_passthrough(self):
        self.assertEqual(extract_content("hello"), "hello")

    def test_extract_content_list_of_dicts(self):
        out = extract_content([{"text": "a"}, {"text": "b"}])
        self.assertEqual(out, "a\nb")

    def test_extract_content_list_of_strings(self):
        out = extract_content(["x", "y"])
        self.assertEqual(out, "x\ny")

    def test_extract_json_block_strips_prose(self):
        text = "Some prose\n{\"a\": 1, \"b\": 2}\nmore prose"
        self.assertEqual(extract_json_block(text), '{"a": 1, "b": 2}')

    def test_extract_json_block_handles_no_braces(self):
        text = "no json here"
        self.assertEqual(extract_json_block(text), "no json here")

    def test_extract_json_block_nested(self):
        text = 'prose {"outer": {"inner": 1}} more'
        self.assertEqual(extract_json_block(text), '{"outer": {"inner": 1}}')


def _ok_validator(_data):
    return ValidationResult.good()


def _make_invoke(responses):
    """Build a fake invoke_fn that returns responses[i] on the i-th call."""
    calls = {"n": 0}

    def invoke_fn(llm, messages, timeout=120):
        i = calls["n"]
        calls["n"] += 1
        r = responses[i]
        if isinstance(r, Exception):
            raise r
        return SimpleNamespace(content=r)

    return invoke_fn, calls


class TestLLMCallWithValidation(unittest.TestCase):
    def test_first_attempt_success(self):
        invoke, calls = _make_invoke(['{"x": 1}'])
        with mock.patch("src.llm_helpers.log_llm_call"):
            out = llm_call_with_validation(
                invoke_fn=invoke, llm=mock.MagicMock(),
                prompt="hello", validator=_ok_validator,
            )
        self.assertIsInstance(out, LLMCallOutcome)
        self.assertEqual(out.data, {"x": 1})
        self.assertFalse(out.used_fallback)
        self.assertEqual(out.attempts, 1)
        self.assertEqual(calls["n"], 1)

    def test_json_parse_failure_then_retry_success(self):
        invoke, calls = _make_invoke(["not json", '{"x": 2}'])
        with mock.patch("src.llm_helpers.log_llm_call"):
            out = llm_call_with_validation(
                invoke_fn=invoke, llm=mock.MagicMock(),
                prompt="hello", validator=_ok_validator,
                max_retries=2,
            )
        self.assertEqual(out.data, {"x": 2})
        self.assertEqual(out.attempts, 2)

    def test_validation_failure_retries_with_critique(self):
        # First response parses but fails validation; second passes
        def validator(d):
            if d.get("x") == 1:
                return ValidationResult.bad("missing y")
            return ValidationResult.good()

        invoke, calls = _make_invoke(['{"x": 1}', '{"x": 2, "y": 3}'])
        with mock.patch("src.llm_helpers.log_llm_call"):
            out = llm_call_with_validation(
                invoke_fn=invoke, llm=mock.MagicMock(),
                prompt="hello", validator=validator,
                max_retries=2,
            )
        self.assertTrue(out.data is not None)
        self.assertEqual(out.data["y"], 3)
        self.assertEqual(out.attempts, 2)

    def test_exhausts_retries_uses_fallback(self):
        invoke, calls = _make_invoke(["bad", "still bad", "still bad"])
        with mock.patch("src.llm_helpers.log_llm_call"):
            out = llm_call_with_validation(
                invoke_fn=invoke, llm=mock.MagicMock(),
                prompt="hello", validator=_ok_validator,
                fallback_factory=lambda: {"fallback": True},
                max_retries=2,
            )
        self.assertTrue(out.used_fallback)
        self.assertEqual(out.data, {"fallback": True})
        self.assertEqual(out.attempts, 3)

    def test_exhausts_retries_no_fallback_returns_none(self):
        invoke, _ = _make_invoke(["bad", "bad"])
        with mock.patch("src.llm_helpers.log_llm_call"):
            out = llm_call_with_validation(
                invoke_fn=invoke, llm=mock.MagicMock(),
                prompt="hello", validator=_ok_validator,
                max_retries=1,
            )
        self.assertIsNone(out.data)
        self.assertFalse(out.used_fallback)
        self.assertTrue(out.last_error)

    def test_llm_exception_triggers_retry(self):
        invoke, _ = _make_invoke([RuntimeError("network blip"), '{"x": 1}'])
        with mock.patch("src.llm_helpers.log_llm_call"):
            out = llm_call_with_validation(
                invoke_fn=invoke, llm=mock.MagicMock(),
                prompt="hello", validator=_ok_validator,
                max_retries=2,
            )
        self.assertEqual(out.data, {"x": 1})
        self.assertEqual(out.attempts, 2)

    def test_top_level_non_dict_json_rejected(self):
        # `[1, 2]` is valid JSON but not an object
        invoke, _ = _make_invoke(['[1, 2, 3]'])
        with mock.patch("src.llm_helpers.log_llm_call"):
            out = llm_call_with_validation(
                invoke_fn=invoke, llm=mock.MagicMock(),
                prompt="hello", validator=_ok_validator,
                fallback_factory=lambda: {"fb": 1},
                max_retries=0,
            )
        self.assertTrue(out.used_fallback)


if __name__ == "__main__":
    unittest.main()

"""Tests for src/llm_helpers.py — validated LLM call with retry + fallback."""

import unittest
from types import SimpleNamespace
from unittest import mock

from src.llm_helpers import (
    LLMCallOutcome,
    ValidationResult,
    _is_provider_error,
    extract_content,
    extract_json_block,
    llm_call_with_validation,
)


class TestExtractHelpers(unittest.TestCase):
    """Tests for ExtractHelpers."""

    def test_extract_content_str_passthrough(self) -> None:
        """Test extract content str passthrough."""
        self.assertEqual(extract_content("hello"), "hello")

    def test_extract_content_list_of_dicts(self) -> None:
        """Test extract content list of dicts."""
        out = extract_content([{"text": "a"}, {"text": "b"}])
        self.assertEqual(out, "a\nb")

    def test_extract_content_list_of_strings(self) -> None:
        """Test extract content list of strings."""
        out = extract_content(["x", "y"])
        self.assertEqual(out, "x\ny")

    def test_extract_json_block_strips_prose(self) -> None:
        """Test extract json block strips prose."""
        text = 'Some prose\n{"a": 1, "b": 2}\nmore prose'
        self.assertEqual(extract_json_block(text), '{"a": 1, "b": 2}')

    def test_extract_json_block_handles_no_braces(self) -> None:
        """Test extract json block handles no braces."""
        text = "no json here"
        self.assertEqual(extract_json_block(text), "no json here")

    def test_extract_json_block_nested(self) -> None:
        """Test extract json block nested."""
        text = 'prose {"outer": {"inner": 1}} more'
        self.assertEqual(extract_json_block(text), '{"outer": {"inner": 1}}')


def _ok_validator(_data):
    """Ok validator."""
    return ValidationResult.good()


def _make_invoke(responses):
    """Build a fake invoke_fn that returns responses[i] on the i-th call."""
    calls = {"n": 0}

    def invoke_fn(llm, messages, timeout=120):
        """Invoke fn."""
        i = calls["n"]
        calls["n"] += 1
        r = responses[i]
        if isinstance(r, Exception):
            raise r
        return SimpleNamespace(content=r)

    return invoke_fn, calls


class TestLLMCallWithValidation(unittest.TestCase):
    """Tests for LLMCallWithValidation."""

    def test_first_attempt_success(self) -> None:
        """Test first attempt success."""
        invoke, calls = _make_invoke(['{"x": 1}'])
        with mock.patch("src.llm_helpers.log_llm_call"):
            out = llm_call_with_validation(
                invoke_fn=invoke,
                llm=mock.MagicMock(),
                prompt="hello",
                validator=_ok_validator,
            )
        self.assertIsInstance(out, LLMCallOutcome)
        self.assertEqual(out.data, {"x": 1})
        self.assertFalse(out.used_fallback)
        self.assertEqual(out.attempts, 1)
        self.assertEqual(calls["n"], 1)

    def test_json_parse_failure_then_retry_success(self) -> None:
        """Test json parse failure then retry success."""
        invoke, calls = _make_invoke(["not json", '{"x": 2}'])
        with mock.patch("src.llm_helpers.log_llm_call"):
            out = llm_call_with_validation(
                invoke_fn=invoke,
                llm=mock.MagicMock(),
                prompt="hello",
                validator=_ok_validator,
                max_retries=2,
            )
        self.assertEqual(out.data, {"x": 2})
        self.assertEqual(out.attempts, 2)

    def test_validation_failure_retries_with_critique(self):
        # First response parses but fails validation; second passes
        """Test validation failure retries with critique."""

        def validator(d):
            """Validate the decoded payload for tests."""
            if d.get("x") == 1:
                return ValidationResult.bad("missing y")
            return ValidationResult.good()

        invoke, calls = _make_invoke(['{"x": 1}', '{"x": 2, "y": 3}'])
        with mock.patch("src.llm_helpers.log_llm_call"):
            out = llm_call_with_validation(
                invoke_fn=invoke,
                llm=mock.MagicMock(),
                prompt="hello",
                validator=validator,
                max_retries=2,
            )
        self.assertTrue(out.data is not None)
        self.assertEqual(out.data["y"], 3)
        self.assertEqual(out.attempts, 2)

    def test_exhausts_retries_uses_fallback(self) -> None:
        """Test exhausts retries uses fallback."""
        invoke, calls = _make_invoke(["bad", "still bad", "still bad"])
        with mock.patch("src.llm_helpers.log_llm_call"):
            out = llm_call_with_validation(
                invoke_fn=invoke,
                llm=mock.MagicMock(),
                prompt="hello",
                validator=_ok_validator,
                fallback_factory=lambda: {"fallback": True},
                max_retries=2,
            )
        self.assertTrue(out.used_fallback)
        self.assertEqual(out.data, {"fallback": True})
        self.assertEqual(out.attempts, 3)

    def test_exhausts_retries_no_fallback_returns_none(self) -> None:
        """Test exhausts retries no fallback returns none."""
        invoke, _ = _make_invoke(["bad", "bad"])
        with mock.patch("src.llm_helpers.log_llm_call"):
            out = llm_call_with_validation(
                invoke_fn=invoke,
                llm=mock.MagicMock(),
                prompt="hello",
                validator=_ok_validator,
                max_retries=1,
            )
        self.assertIsNone(out.data)
        self.assertFalse(out.used_fallback)
        self.assertTrue(out.last_error)

    def test_llm_exception_triggers_retry(self) -> None:
        """Test llm exception triggers retry."""
        invoke, _ = _make_invoke([RuntimeError("network blip"), '{"x": 1}'])
        with mock.patch("src.llm_helpers.log_llm_call"):
            out = llm_call_with_validation(
                invoke_fn=invoke,
                llm=mock.MagicMock(),
                prompt="hello",
                validator=_ok_validator,
                max_retries=2,
            )
        self.assertEqual(out.data, {"x": 1})
        self.assertEqual(out.attempts, 2)

    def test_top_level_non_dict_json_rejected(self) -> None:
        # `[1, 2]` is valid JSON but not an object
        """Test top level non dict json rejected."""
        invoke, _ = _make_invoke(["[1, 2, 3]"])
        with mock.patch("src.llm_helpers.log_llm_call"):
            out = llm_call_with_validation(
                invoke_fn=invoke,
                llm=mock.MagicMock(),
                prompt="hello",
                validator=_ok_validator,
                fallback_factory=lambda: {"fb": 1},
                max_retries=0,
            )
        self.assertTrue(out.used_fallback)


class TestProviderErrorDetection(unittest.TestCase):
    """Tests for provider-error detection used for model rotation."""

    def test_detects_429_rate_limit(self) -> None:
        """429 should be treated as provider-side failure."""
        self.assertTrue(_is_provider_error(RuntimeError("Error code: 429")))

    def test_detects_rate_limit_phrase(self) -> None:
        """Explicit rate limit text should be recognized."""
        self.assertTrue(
            _is_provider_error(
                RuntimeError("provider temporarily rate limited")
            )
        )

    def test_ignores_local_validation_error(self) -> None:
        """Non-provider failures should not match."""
        self.assertFalse(
            _is_provider_error(RuntimeError("json parse failure"))
        )


if __name__ == "__main__":
    unittest.main()

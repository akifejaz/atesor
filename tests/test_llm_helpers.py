"""Tests for src/llm_helpers.py — validated LLM call with retry + fallback."""

import unittest
from types import SimpleNamespace
from unittest import mock

from src.llm_helpers import (
    LLMCallOutcome,
    ValidationResult,
    _build_replacement_llm,
    _extract_affordable_tokens,
    _extract_slug_hint,
    _is_provider_error,
    _shrink_max_tokens,
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


class TestSlugHintExtraction(unittest.TestCase):
    """Tests for parsing OpenRouter's ``use this slug instead:`` hint."""

    def test_extracts_slug_from_404_body(self) -> None:
        """The typical OpenRouter payload yields the replacement id."""
        exc = RuntimeError(
            "Error code: 404 - {'error': {'message': 'This model is "
            "unavailable for free. The paid version is available now "
            "- use this slug instead: qwen/qwen3-14b', 'code': 404}}"
        )
        self.assertEqual(_extract_slug_hint(exc), "qwen/qwen3-14b")

    def test_returns_none_when_no_hint(self) -> None:
        """No hint text → returns None."""
        self.assertIsNone(_extract_slug_hint(RuntimeError("no hint here")))

    def test_extracts_slug_with_version_and_colon(self) -> None:
        """Slugs may include ``:tag`` suffixes and dashes."""
        exc = RuntimeError(
            "use this slug instead: google/gemini-2.0-flash-exp:free "
            "(other prose)"
        )
        self.assertEqual(
            _extract_slug_hint(exc), "google/gemini-2.0-flash-exp:free"
        )


class TestHotAddedReplacementModel(unittest.TestCase):
    """Rotation kicks in for a hinted slug, even with an empty pool."""

    def test_hinted_slug_added_and_rotated_on_next_attempt(self) -> None:
        """When primary raises with a slug hint we swap and retry."""

        # Track which model each call used
        seen_models: list = []

        def invoke(llm, messages, timeout=120):
            """Invoke fn: fail on primary, succeed on the replacement."""
            seen_models.append(getattr(llm, "model_name", "primary"))
            if getattr(llm, "model_name", None) == "primary":
                raise RuntimeError(
                    "Error code: 404 - {'error': {'message': "
                    "'use this slug instead: replacement/free', "
                    "'code': 404}}"
                )
            return SimpleNamespace(content='{"ok": true}')

        primary = mock.MagicMock()
        primary.model_name = "primary"

        # `_build_replacement_llm` calls type(reference)(**kwargs); we
        # patch it so we don't need a real LangChain constructor.
        replacement = mock.MagicMock()
        replacement.model_name = "replacement/free"

        with mock.patch("src.llm_helpers.log_llm_call"), mock.patch(
            "src.llm_helpers._build_replacement_llm",
            return_value=replacement,
        ):
            out = llm_call_with_validation(
                invoke_fn=invoke,
                llm=primary,
                prompt="hi",
                validator=lambda d: ValidationResult.good(),
                max_retries=2,
            )

        self.assertEqual(out.data, {"ok": True})
        # First call → primary; second call → replacement
        self.assertEqual(seen_models[0], "primary")
        self.assertIn("replacement/free", seen_models)


class TestEmptyResponseRotation(unittest.TestCase):
    """Empty / whitespace-only content must trigger pool rotation."""

    def test_empty_string_rotates_to_fallback(self) -> None:
        """Provider returning ``""`` should rotate, not consume retries."""
        invoke, calls = _make_invoke(["", '{"ok": true}'])
        with mock.patch("src.llm_helpers.log_llm_call"):
            out = llm_call_with_validation(
                invoke_fn=invoke,
                llm=mock.MagicMock(),
                fallback_llms=[mock.MagicMock()],
                prompt="hi",
                validator=_ok_validator,
                max_retries=0,
            )
        self.assertEqual(out.data, {"ok": True})
        # Only 2 calls total: primary returned "", rotated to fallback.
        self.assertEqual(calls["n"], 2)

    def test_whitespace_only_rotates_to_fallback(self) -> None:
        """``"   \\n\\t "`` is treated the same as empty content."""
        invoke, calls = _make_invoke(["   \n\t ", '{"ok": true}'])
        with mock.patch("src.llm_helpers.log_llm_call"):
            out = llm_call_with_validation(
                invoke_fn=invoke,
                llm=mock.MagicMock(),
                fallback_llms=[mock.MagicMock()],
                prompt="hi",
                validator=_ok_validator,
                max_retries=0,
            )
        self.assertEqual(out.data, {"ok": True})
        self.assertEqual(calls["n"], 2)

    def test_empty_falls_back_to_critique_when_pool_exhausted(self):
        """No fallbacks left → still use critique-retry budget."""
        invoke, calls = _make_invoke(["", '{"ok": true}'])
        with mock.patch("src.llm_helpers.log_llm_call"):
            out = llm_call_with_validation(
                invoke_fn=invoke,
                llm=mock.MagicMock(),
                prompt="hi",
                validator=_ok_validator,
                max_retries=2,
            )
        self.assertEqual(out.data, {"ok": True})
        self.assertEqual(calls["n"], 2)


class TestJSONParseRotation(unittest.TestCase):
    """Persistent unparseable output should rotate models before critique."""

    def test_bad_json_rotates_before_critique(self) -> None:
        """Bad JSON on primary → rotate to fallback that returns good."""
        invoke, calls = _make_invoke(["not json", '{"ok": true}'])
        with mock.patch("src.llm_helpers.log_llm_call"):
            out = llm_call_with_validation(
                invoke_fn=invoke,
                llm=mock.MagicMock(),
                fallback_llms=[mock.MagicMock()],
                prompt="hi",
                validator=_ok_validator,
                max_retries=0,
            )
        self.assertEqual(out.data, {"ok": True})
        self.assertEqual(calls["n"], 2)


class TestTokenBudgetSelfHeal(unittest.TestCase):
    """HTTP 402 with ``can only afford N`` must shrink caps + retry."""

    def test_extract_affordable_tokens_from_402_body(self) -> None:
        """Regex captures the affordable count from the error body."""
        exc = RuntimeError(
            "Error code: 402 - {'error': {'message': "
            "'This request requires more credits, or fewer max_tokens. "
            "You requested up to 65536 tokens, but can only afford "
            "5702. To increase, visit...'}}"
        )
        self.assertEqual(_extract_affordable_tokens(exc), 5702)

    def test_extract_affordable_tokens_returns_none_without_hint(self):
        """Unrelated errors do not falsely match."""
        self.assertIsNone(
            _extract_affordable_tokens(RuntimeError("connection reset"))
        )

    def test_shrink_max_tokens_updates_attribute(self) -> None:
        """A model with a high cap is shrunk to ~90 % of the affordable."""
        llm = SimpleNamespace(max_tokens=65536)
        changed = _shrink_max_tokens(llm, 5702)
        self.assertTrue(changed)
        # 90 % headroom: floor(5702 * 0.9) == 5131
        self.assertEqual(llm.max_tokens, 5131)

    def test_shrink_max_tokens_leaves_already_smaller_cap(self) -> None:
        """No-op when the current cap is already under the affordable."""
        llm = SimpleNamespace(max_tokens=1000)
        changed = _shrink_max_tokens(llm, 5702)
        self.assertFalse(changed)
        self.assertEqual(llm.max_tokens, 1000)

    def test_402_error_shrinks_and_retries_same_model(self) -> None:
        """HTTP 402 shrinks the pool max_tokens then retries same model."""
        primary = SimpleNamespace(max_tokens=65536, model_name="p")
        fallback = SimpleNamespace(max_tokens=65536, model_name="f")

        seen_caps: list = []

        def invoke(llm, messages, timeout=120):
            """Fail with 402 while cap > 6000, then succeed."""
            seen_caps.append(getattr(llm, "max_tokens", None))
            if getattr(llm, "max_tokens", 0) > 6000:
                raise RuntimeError(
                    "Error code: 402 - {'error': {'message': "
                    "'requires more credits. You requested up to "
                    "65536 tokens, but can only afford 5702.'}}"
                )
            return SimpleNamespace(content='{"ok": true}')

        with mock.patch("src.llm_helpers.log_llm_call"):
            out = llm_call_with_validation(
                invoke_fn=invoke,
                llm=primary,
                fallback_llms=[fallback],
                prompt="hi",
                validator=_ok_validator,
                max_retries=1,
            )

        self.assertEqual(out.data, {"ok": True})
        # First call at 65536 → 402; shrink to 5131; second call
        # succeeds on the SAME model (no rotation needed).
        self.assertEqual(seen_caps[0], 65536)
        self.assertLess(seen_caps[1], 6000)
        # Fallback should have been shrunk too, ready for future calls.
        self.assertLess(fallback.max_tokens, 6000)


if __name__ == "__main__":
    unittest.main()

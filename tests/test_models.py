"""Tests for src.models LLM pool behavior."""

import os
import unittest
from unittest import mock

from src.models import (
    OPENROUTER_FREE_ROUTER,
    _create_llm_with_model,
    _openrouter_fallback_ids,
    create_llm_pool,
)
from src.state import AgentRole


class TestCreateLLMPool(unittest.TestCase):
    """Tests for create_llm_pool."""

    @mock.patch.dict(os.environ, {"LLM_PROVIDER": "openrouter"}, clear=False)
    @mock.patch("src.models._create_llm_with_model")
    @mock.patch("src.models.create_llm")
    def test_openrouter_uses_default_fallbacks_when_env_missing(
        self,
        mock_create_llm: mock.MagicMock,
        mock_create_with_model: mock.MagicMock,
    ) -> None:
        """Use built-in fallback models when env var is unset.

        Health-check probing was removed (self-inflicted rate limits and
        false rejections of throttled models). The pool now includes
        every curated fallback that instantiates successfully.
        """
        with mock.patch.dict(
            os.environ, {"OPENROUTER_FALLBACK_MODELS": ""}, clear=False
        ):
            mock_create_llm.return_value = "primary"
            mock_create_with_model.side_effect = [
                "fb1",
                "fb2",
                "fb3",
                "fb4",
                "fb5",
                "fb6",
                "fb7",
            ]

            pool = create_llm_pool(AgentRole.FIXER)

        self.assertEqual(pool[0], "primary")
        self.assertEqual(len(pool), 8)
        self.assertEqual(mock_create_with_model.call_count, 7)

    @mock.patch.dict(os.environ, {"LLM_PROVIDER": "gemini"}, clear=False)
    @mock.patch("src.models.create_llm")
    def test_non_openrouter_returns_primary_only(
        self, mock_create_llm: mock.MagicMock
    ) -> None:
        """Gemini/OpenAI should not build fallback pools."""
        mock_create_llm.return_value = "primary"
        pool = create_llm_pool(AgentRole.SCOUT)
        self.assertEqual(pool, ["primary"])


class TestOpenRouterFallbackIds(unittest.TestCase):
    """Tests for the shared OpenRouter fallback chain."""

    @mock.patch.dict(
        os.environ, {"OPENROUTER_FALLBACK_MODELS": ""}, clear=False
    )
    def test_free_router_is_terminal_entry(self) -> None:
        """openrouter/free must close the default chain."""
        ids = _openrouter_fallback_ids()
        self.assertEqual(ids[-1], OPENROUTER_FREE_ROUTER)

    @mock.patch.dict(
        os.environ, {"OPENROUTER_FALLBACK_MODELS": ""}, clear=False
    )
    def test_auto_router_not_in_defaults(self) -> None:
        """Keep openrouter/auto out of the default chain.

        The Auto Router is paid-only — a guaranteed 402 on a
        zero-credit account.
        """
        self.assertNotIn("openrouter/auto", _openrouter_fallback_ids())

    @mock.patch.dict(
        os.environ,
        {"OPENROUTER_FALLBACK_MODELS": "a/x:free, b/y:free"},
        clear=False,
    )
    def test_env_override_still_appends_free_router(self) -> None:
        """A custom chain still degrades to openrouter/free last."""
        self.assertEqual(
            _openrouter_fallback_ids(),
            ["a/x:free", "b/y:free", OPENROUTER_FREE_ROUTER],
        )


class TestServerSideFallback(unittest.TestCase):
    """The OpenRouter LLM must carry a server-side models array."""

    @mock.patch.dict(
        os.environ,
        {
            "LLM_PROVIDER": "openrouter",
            "OPENROUTER_API_KEY": "test-key",
            "OPENROUTER_FALLBACK_MODELS": "",
        },
        clear=False,
    )
    def test_extra_body_models_excludes_primary(self) -> None:
        """extra_body carries the fallback chain minus the primary."""
        llm = _create_llm_with_model(
            AgentRole.FIXER, "openai/gpt-oss-120b:free"
        )
        models = llm.extra_body["models"]
        self.assertNotIn("openai/gpt-oss-120b:free", models)
        self.assertEqual(models[-1], OPENROUTER_FREE_ROUTER)
        # OpenRouter rejects the request with HTTP 400 when the models
        # array has more than 3 entries — regression guard for the
        # 2026-07-02 zlib planner failure.
        self.assertLessEqual(len(models), 3)
        self.assertEqual(len(models), len(set(models)))


class TestCostForUsage(unittest.TestCase):
    """Tests for real token-based cost computation."""

    def test_free_slug_costs_zero(self) -> None:
        """Free-tier ``:free`` slugs bill nothing."""
        from src.models import cost_for_usage

        self.assertEqual(
            cost_for_usage("qwen/qwen3-coder:free", 100000, 50000), 0.0
        )

    def test_free_router_costs_zero(self) -> None:
        """The Free Models Router bills nothing."""
        from src.models import cost_for_usage

        self.assertEqual(cost_for_usage("openrouter/free", 1000, 1000), 0.0)

    def test_paid_model_priced_from_table(self) -> None:
        """Table-listed paid models price per million tokens."""
        from src.models import cost_for_usage

        self.assertAlmostEqual(cost_for_usage("gpt-4o", 1_000_000, 0), 2.50)
        self.assertAlmostEqual(
            cost_for_usage("gpt-4o-mini", 0, 1_000_000), 0.60
        )

    def test_unknown_paid_model_uses_conservative_default(self) -> None:
        """Unlisted paid models over-count rather than bill as free."""
        from src.models import cost_for_usage

        cost = cost_for_usage("some/unknown-model", 1_000_000, 0)
        self.assertGreater(cost, 0.0)

    def test_negative_tokens_clamped(self) -> None:
        """Bogus negative usage never produces negative cost."""
        from src.models import cost_for_usage

        self.assertEqual(cost_for_usage("gpt-4o", -5, -5), 0.0)


if __name__ == "__main__":
    unittest.main()

"""Tests for src.models LLM pool behavior."""

import os
import unittest
from unittest import mock

from src.models import create_llm_pool
from src.state import AgentRole


class TestCreateLLMPool(unittest.TestCase):
    """Tests for create_llm_pool."""

    @mock.patch.dict(os.environ, {"LLM_PROVIDER": "openrouter"}, clear=False)
    @mock.patch("src.models._create_llm_with_model")
    @mock.patch("src.models.create_llm")
    @mock.patch("src.models._health_check", return_value=True)
    def test_openrouter_uses_default_fallbacks_when_env_missing(
        self,
        mock_health_check: mock.MagicMock,
        mock_create_llm: mock.MagicMock,
        mock_create_with_model: mock.MagicMock,
    ) -> None:
        """Use built-in fallback models when env var is unset."""
        with mock.patch.dict(
            os.environ, {"OPENROUTER_FALLBACK_MODELS": ""}, clear=False
        ):
            mock_create_llm.return_value = "primary"
            mock_create_with_model.side_effect = [
                "fb1", "fb2", "fb3", "fb4", "fb5", "fb6"
            ]

            pool = create_llm_pool(AgentRole.FIXER)

        self.assertEqual(pool[0], "primary")
        self.assertEqual(len(pool), 7)
        self.assertEqual(mock_create_with_model.call_count, 6)

    @mock.patch.dict(os.environ, {"LLM_PROVIDER": "gemini"}, clear=False)
    @mock.patch("src.models.create_llm")
    def test_non_openrouter_returns_primary_only(
        self, mock_create_llm: mock.MagicMock
    ) -> None:
        """Gemini/OpenAI should not build fallback pools."""
        mock_create_llm.return_value = "primary"
        pool = create_llm_pool(AgentRole.SCOUT)
        self.assertEqual(pool, ["primary"])


if __name__ == "__main__":
    unittest.main()

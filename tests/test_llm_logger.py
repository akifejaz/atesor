"""Tests for src.llm_logger module."""

import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from src.llm_logger import LLMCallLogger


class TestLLMCallLogger(unittest.TestCase):
    """Tests for the LLMCallLogger class."""

    def setUp(self):
        LLMCallLogger._instance = None
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        LLMCallLogger._instance = None
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _make_logger(self):
        """Create a logger with LOGS_DIR pointing to the temp directory."""
        with patch("src.llm_logger.LLMCallLogger._ensure_log_file"):
            logger_inst = LLMCallLogger()
        # Now call _ensure_log_file with the mocked config
        with patch("src.config.LOGS_DIR", self.temp_dir):
            logger_inst._ensure_log_file()
        return logger_inst

    # --- singleton -----------------------------------------------------------

    def test_singleton_returns_same_instance(self):
        with patch("src.llm_logger.LLMCallLogger._ensure_log_file"):
            a = LLMCallLogger()
            b = LLMCallLogger()
        self.assertIs(a, b)

    def test_singleton_reset_gives_new_instance(self):
        with patch("src.llm_logger.LLMCallLogger._ensure_log_file"):
            a = LLMCallLogger()
        LLMCallLogger._instance = None
        with patch("src.llm_logger.LLMCallLogger._ensure_log_file"):
            b = LLMCallLogger()
        self.assertIsNot(a, b)

    # --- _ensure_log_file ----------------------------------------------------

    def test_ensure_log_file_creates_file_with_header(self):
        lgr = self._make_logger()
        self.assertIsNotNone(lgr.log_file)
        self.assertTrue(os.path.exists(lgr.log_file))
        with open(lgr.log_file) as f:
            content = f.read()
        self.assertIn("ATESOR AI - LLM CALL LOG", content)
        self.assertIn("Created:", content)
        self.assertIn("=" * 80, content)

    def test_ensure_log_file_sets_none_on_import_error(self):
        with patch("src.llm_logger.LLMCallLogger._ensure_log_file"):
            lgr = LLMCallLogger()
        # Simulate an import error inside _ensure_log_file
        with patch.dict("sys.modules", {"src.config": None}):
            lgr._initialized = True
            lgr.log_file = None
            lgr._ensure_log_file()
        self.assertIsNone(lgr.log_file)

    # --- log_call basic ------------------------------------------------------

    def test_log_call_returns_call_id(self):
        lgr = self._make_logger()
        call_id = lgr.log_call("SCOUT", "hello", "world", "gpt-4")
        self.assertTrue(call_id.startswith("call_"))
        self.assertEqual(len(call_id), len("call_") + 8)

    def test_log_call_appends_to_calls_list(self):
        lgr = self._make_logger()
        self.assertEqual(len(lgr.calls), 0)
        lgr.log_call("SCOUT", "p1", "r1", "gpt-4")
        lgr.log_call("BUILDER", "p2", "r2", "gpt-4")
        self.assertEqual(len(lgr.calls), 2)
        self.assertEqual(lgr.calls[0]["agent_role"], "SCOUT")
        self.assertEqual(lgr.calls[1]["agent_role"], "BUILDER")

    def test_log_call_record_fields(self):
        lgr = self._make_logger()
        lgr.log_call("FIXER", "prompt_text", "response_text", "gpt-3.5", cost_usd=0.005)
        rec = lgr.calls[0]
        self.assertEqual(rec["agent_role"], "FIXER")
        self.assertEqual(rec["model"], "gpt-3.5")
        self.assertAlmostEqual(rec["cost_usd"], 0.005)
        self.assertEqual(rec["prompt_length"], len("prompt_text"))
        self.assertEqual(rec["response_length"], len("response_text"))
        self.assertEqual(rec["metadata"], {})
        self.assertIn("call_id", rec)
        self.assertIn("timestamp", rec)

    def test_log_call_writes_to_file(self):
        lgr = self._make_logger()
        lgr.log_call("SCOUT", "my prompt", "my response", "gpt-4", cost_usd=0.01)
        with open(lgr.log_file) as f:
            content = f.read()
        self.assertIn("AGENT: SCOUT", content)
        self.assertIn("MODEL: gpt-4", content)
        self.assertIn("COST: $0.010000", content)
        self.assertIn("--- PROMPT ---", content)
        self.assertIn("my prompt", content)
        self.assertIn("--- RESPONSE ---", content)
        self.assertIn("my response", content)

    # --- metadata ------------------------------------------------------------

    def test_log_call_with_metadata(self):
        lgr = self._make_logger()
        meta = {"task": "analysis", "step": 3}
        lgr.log_call("SCOUT", "p", "r", "gpt-4", metadata=meta)
        self.assertEqual(lgr.calls[0]["metadata"], meta)
        with open(lgr.log_file) as f:
            content = f.read()
        self.assertIn("METADATA:", content)
        self.assertIn('"task": "analysis"', content)

    def test_log_call_without_metadata(self):
        lgr = self._make_logger()
        lgr.log_call("SCOUT", "p", "r", "gpt-4")
        self.assertEqual(lgr.calls[0]["metadata"], {})
        with open(lgr.log_file) as f:
            content = f.read()
        self.assertNotIn("METADATA:", content)

    # --- truncation ----------------------------------------------------------

    def test_long_prompt_is_truncated_in_file(self):
        lgr = self._make_logger()
        long_prompt = "A" * 15000
        lgr.log_call("SCOUT", long_prompt, "short", "gpt-4")
        with open(lgr.log_file) as f:
            content = f.read()
        self.assertIn("[... truncated 5000 characters ...]", content)
        # Only the first 10000 chars of the prompt should appear
        self.assertNotIn("A" * 10001, content)

    def test_long_response_is_truncated_in_file(self):
        lgr = self._make_logger()
        long_response = "B" * 12000
        lgr.log_call("SCOUT", "short", long_response, "gpt-4")
        with open(lgr.log_file) as f:
            content = f.read()
        self.assertIn("[... truncated 2000 characters ...]", content)

    def test_short_prompt_not_truncated(self):
        lgr = self._make_logger()
        lgr.log_call("SCOUT", "small", "tiny", "gpt-4")
        with open(lgr.log_file) as f:
            content = f.read()
        self.assertNotIn("truncated", content)

    # --- log_file is None ----------------------------------------------------

    def test_log_call_no_file_still_tracks_in_memory(self):
        with patch("src.llm_logger.LLMCallLogger._ensure_log_file"):
            lgr = LLMCallLogger()
        lgr.log_file = None
        call_id = lgr.log_call("SCOUT", "p", "r", "gpt-4")
        self.assertTrue(call_id.startswith("call_"))
        self.assertEqual(len(lgr.calls), 1)

    def test_log_call_no_file_does_not_raise(self):
        with patch("src.llm_logger.LLMCallLogger._ensure_log_file"):
            lgr = LLMCallLogger()
        lgr.log_file = None
        # Should not raise even though there's no file
        lgr.log_call("SCOUT", "p", "r", "gpt-4")

    # --- log_llm_call convenience function -----------------------------------

    def test_log_llm_call_delegates_to_logger(self):
        lgr = self._make_logger()
        # Patch the module-level _llm_logger so the convenience function uses our instance
        with patch("src.llm_logger._llm_logger", lgr):
            from src.llm_logger import log_llm_call
            call_id = log_llm_call("BUILDER", "pp", "rr", "gpt-4", cost_usd=0.1)
        self.assertTrue(call_id.startswith("call_"))
        self.assertEqual(len(lgr.calls), 1)
        self.assertEqual(lgr.calls[0]["agent_role"], "BUILDER")


if __name__ == "__main__":
    unittest.main()

#############################################################################
# Copyright (c) 2026 10xEngineers
#
# Author: Akif Ejaz <akif.ejaz@10xengineers.ai>
# This program and the accompanying materials are made available under the
# terms of the MIT License which is available at
# https://opensource.org/licenses/MIT.
#
# SPDX-License-Identifier: MIT
#############################################################################

"""Comprehensive LLM call logging and tracking system.

Logs all LLM invocations with their prompts, responses, and results for
debugging. Creates ``agent-call.log`` in the ``workspace/logs`` directory.
"""

import json
import logging
import os
import threading
import uuid
from collections import deque
from datetime import datetime
from typing import Any, Dict, Optional

# Configure logging for this module
logger = logging.getLogger(__name__)


class LLMCallLogger:
    """Centralized logger for all LLM calls across the agent system.

    Provides a detailed audit trail for debugging and analysis. The class
    is a thread-safe singleton, so every caller shares one log file.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Return the process-wide singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(LLMCallLogger, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._write_lock = threading.Lock()
        self.log_file = None
        self.calls: deque = deque(maxlen=1000)
        self._ensure_log_file()

    def _ensure_log_file(self):
        """Ensure the log file directory exists and is writable."""
        try:
            from src.config import LOGS_DIR

            # Create logs directory if it doesn't exist
            os.makedirs(LOGS_DIR, exist_ok=True)

            self.log_file = os.path.join(LOGS_DIR, "agent-call.log")
            self._logs_dir = LOGS_DIR

            # Write header if file is new
            if not os.path.exists(self.log_file):
                with open(self.log_file, "w") as f:
                    f.write("=" * 80 + "\n")
                    f.write("ATESOR AI - LLM CALL LOG\n")
                    f.write(f"Created: {datetime.now().isoformat()}\n")
                    f.write("=" * 80 + "\n\n")

            logger.debug(f"LLM call log initialized: {self.log_file}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM call log: {e}")
            self.log_file = None
            self._logs_dir = None

    def set_repo_name(self, repo_name: str) -> None:
        """Switch to a per-repo log file for multi-process batch safety.

        Args:
            repo_name: Name of the repository whose log to write to.
        """
        if not repo_name or not self._logs_dir:
            return
        self.log_file = os.path.join(
            self._logs_dir, f"agent-call_{repo_name}.log"
        )

    def log_call(
        self,
        agent_role: str,
        prompt: str,
        response: str,
        model: str,
        cost_usd: float = 0.0,
        tokens_in: int = 0,
        tokens_out: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Log an LLM call with complete context.

        Args:
            agent_role: The agent role making the call (e.g. SCOUT,
                BUILDER, FIXER).
            prompt: The full prompt sent to the LLM.
            response: The response received from the LLM.
            model: The model used.
            cost_usd: Real cost of the call from token usage (0 for
                free-tier models).
            tokens_in: Billed prompt tokens (0 when the provider did
                not report usage).
            tokens_out: Billed completion tokens.
            metadata: Additional metadata about the call.

        Returns:
            A call ID string for cross-referencing.
        """
        call_id = f"call_{uuid.uuid4().hex[:8]}"
        timestamp = datetime.now().isoformat()

        call_record = {
            "call_id": call_id,
            "timestamp": timestamp,
            "agent_role": agent_role,
            "model": model,
            "cost_usd": cost_usd,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "prompt_length": len(prompt),
            "response_length": len(response),
            "metadata": metadata or {},
        }

        self.calls.append(call_record)

        # Write to file immediately
        if self.log_file:
            with self._write_lock:
                try:
                    with open(self.log_file, "a") as f:
                        f.write("\n" + "=" * 80 + "\n")
                        f.write(f"CALL: {call_id}\n")
                        f.write(f"TIMESTAMP: {timestamp}\n")
                        f.write(f"AGENT: {agent_role}\n")
                        f.write(f"MODEL: {model}\n")
                        f.write(f"COST: ${cost_usd:.6f}\n")
                        f.write(f"TOKENS: in={tokens_in} out={tokens_out}\n")

                        if metadata:
                            meta_json = json.dumps(metadata, indent=2)
                            f.write(f"METADATA: {meta_json}\n")

                        f.write("\n--- PROMPT ---\n")
                        f.write(prompt[:10000])  # First 10k chars.
                        if len(prompt) > 10000:
                            dropped = len(prompt) - 10000
                            f.write(
                                f"\n[... truncated {dropped} characters"
                                " ...]\n"
                            )

                        f.write("\n--- RESPONSE ---\n")
                        f.write(response[:10000])  # First 10k chars.
                        if len(response) > 10000:
                            dropped = len(response) - 10000
                            f.write(
                                f"\n[... truncated {dropped} characters"
                                " ...]\n"
                            )

                        f.write("\n" + "=" * 80 + "\n\n")
                except Exception as e:
                    logger.error(f"Failed to write LLM call log: {e}")

        return call_id


# Global instance
_llm_logger = LLMCallLogger()


def log_llm_call(
    agent_role: str,
    prompt: str,
    response: str,
    model: str,
    cost_usd: float = 0.0,
    tokens_in: int = 0,
    tokens_out: int = 0,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Log an LLM call through the global logger instance."""
    return _llm_logger.log_call(
        agent_role,
        prompt,
        response,
        model,
        cost_usd,
        tokens_in,
        tokens_out,
        metadata,
    )


def set_llm_log_repo(repo_name: str) -> None:
    """Switch the LLM call log to a per-repo file for batch safety.

    Args:
        repo_name: Name of the repository whose log to write to.
    """
    _llm_logger.set_repo_name(repo_name)

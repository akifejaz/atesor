"""
Comprehensive LLM call logging and tracking system.

Logs ALL LLM invocations with their prompts, responses, and results for debugging.
Creates agent-call.log in the workspace/logs directory.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List
import uuid

# Configure logging for this module
logger = logging.getLogger(__name__)


class LLMCallLogger:
    """
    Centralized logger for all LLM calls across the agent system.
    Provides detailed audit trail for debugging and analysis.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMCallLogger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self.log_file = None
        self.calls: List[Dict[str, Any]] = []
        self._ensure_log_file()

    def _ensure_log_file(self):
        """Ensure the log file directory exists and is writable."""
        try:
            from src.config import LOGS_DIR

            # Create logs directory if it doesn't exist
            os.makedirs(LOGS_DIR, exist_ok=True)

            self.log_file = os.path.join(LOGS_DIR, "agent-call.log")

            # Write header if file is new
            if not os.path.exists(self.log_file):
                with open(self.log_file, 'w') as f:
                    f.write("=" * 80 + "\n")
                    f.write("ATESOR AI - LLM CALL LOG\n")
                    f.write(f"Created: {datetime.now().isoformat()}\n")
                    f.write("=" * 80 + "\n\n")

            logger.debug(f"LLM call log initialized: {self.log_file}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM call log: {e}")
            self.log_file = None

    def log_call(
        self,
        agent_role: str,
        prompt: str,
        response: str,
        model: str,
        cost_usd: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log an LLM call with complete context.

        Args:
            agent_role: The agent role making the call (SCOUT, BUILDER, FIXER, etc.)
            prompt: The full prompt sent to the LLM
            response: The response received from the LLM
            model: The model used
            cost_usd: Estimated cost of the call
            metadata: Additional metadata about the call

        Returns:
            Call ID for cross-referencing
        """
        call_id = f"call_{uuid.uuid4().hex[:8]}"
        timestamp = datetime.now().isoformat()

        call_record = {
            "call_id": call_id,
            "timestamp": timestamp,
            "agent_role": agent_role,
            "model": model,
            "cost_usd": cost_usd,
            "prompt_length": len(prompt),
            "response_length": len(response),
            "metadata": metadata or {}
        }

        self.calls.append(call_record)

        # Write to file immediately
        if self.log_file:
            try:
                with open(self.log_file, 'a') as f:
                    f.write("\n" + "=" * 80 + "\n")
                    f.write(f"CALL: {call_id}\n")
                    f.write(f"TIMESTAMP: {timestamp}\n")
                    f.write(f"AGENT: {agent_role}\n")
                    f.write(f"MODEL: {model}\n")
                    f.write(f"COST: ${cost_usd:.6f}\n")

                    if metadata:
                        f.write(f"METADATA: {json.dumps(metadata, indent=2)}\n")

                    f.write("\n--- PROMPT ---\n")
                    f.write(prompt[:10000])  # First 10k chars
                    if len(prompt) > 10000:
                        f.write(f"\n[... truncated {len(prompt) - 10000} characters ...]\n")

                    f.write("\n--- RESPONSE ---\n")
                    f.write(response[:10000])  # First 10k chars
                    if len(response) > 10000:
                        f.write(f"\n[... truncated {len(response) - 10000} characters ...]\n")

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
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """Convenience function to log an LLM call."""
    return _llm_logger.log_call(agent_role, prompt, response, model, cost_usd, metadata)

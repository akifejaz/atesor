"""Shared LLM-invocation helpers.

Provides timeout-guarded LLM calls plus JSON-schema validation with
retry-on-critique and a deterministic fallback.

This module removes the duplicated boilerplate of
``build_prompt -> invoke_llm -> extract_content -> log_llm_call ->
extract_json_block -> json.loads -> validate -> (retry | fallback)``
that previously lived in every agent node.

Design goals:
    * Keep prompts small (we run on free-tier models; tokens matter).
    * Never let a malformed LLM response stall the workflow.
    * Always log every attempt to the audit trail.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

from langchain_core.messages import HumanMessage

from .llm_logger import log_llm_call

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Public types
# ----------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Result of validating a parsed LLM response."""

    ok: bool
    reason: str = ""

    @classmethod
    def good(cls) -> "ValidationResult":
        """Return a successful validation result."""
        return cls(True, "")

    @classmethod
    def bad(cls, reason: str) -> "ValidationResult":
        """Return a failed validation result with a reason."""
        return cls(False, reason)


@dataclass
class LLMCallOutcome:
    """Outcome of an LLM call after validation+retry."""

    data: Optional[dict]  # parsed JSON if validation succeeded, else None
    used_fallback: bool  # True when the deterministic fallback was used
    attempts: int  # how many LLM invocations we actually made
    last_error: str = ""  # human-readable reason of the final failure


# ----------------------------------------------------------------------------
# Helpers re-exported from graph.py (kept here to avoid circular imports)
# ----------------------------------------------------------------------------


def extract_content(content: Any) -> str:
    """Turn an LLM message body into a plain string.

    Args:
        content: A message body (``str``, ``list[dict]``, or other).

    Returns:
        The content coerced to a single string.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            (
                str(item.get("text", item))
                if isinstance(item, dict)
                else str(item)
            )
            for item in content
        )
    return str(content)


def extract_json_block(text: str) -> str:
    """Slice from the first ``{`` to the last ``}``.

    Tolerant of surrounding prose around the JSON object.

    Args:
        text: Raw text that may contain a JSON object.

    Returns:
        The substring spanning the outermost braces, or ``text`` if no
        balanced pair is found.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end >= start:
        return text[start : end + 1]
    return text


# ----------------------------------------------------------------------------
# Core helper: validated LLM call with retry + fallback
# ----------------------------------------------------------------------------

# Default critique appended to the *next* attempt when validation fails.
# Kept short on purpose — the previous prompt is already in the LLM's context
# on the retry turn, we don't need to repeat it.
_DEFAULT_CRITIQUE = (
    "Your previous response was rejected. Reason: {reason}\n"
    "Re-read the original instructions and emit ONLY a valid JSON object "
    "that conforms to the schema. No prose, no markdown fences, no commentary."
)


def llm_call_with_validation(
    *,
    invoke_fn: Callable[[Any, list], Any],  # graph.invoke_llm
    llm: Any,  # BaseChatModel instance
    prompt: str,  # fully-rendered prompt
    validator: Callable[[dict], ValidationResult],
    fallback_factory: Optional[Callable[[], Optional[dict]]] = None,
    role: str = "agent",
    audit_metadata: Optional[dict] = None,
    cost_estimate: float = 0.01,
    max_retries: int = 2,  # total LLM calls = max_retries + 1
    critique_template: str = _DEFAULT_CRITIQUE,
    timeout: int = 120,
    fallback_llms: Optional[
        list
    ] = None,  # rotate to next model on provider errors
) -> LLMCallOutcome:
    """Call ``llm`` with ``prompt``, parse, validate, retry, fall back.

    Parses the response as JSON, validates it, retries with a critique
    on failure, then falls back to ``fallback_factory()`` once retries
    are exhausted. Every attempt is logged via ``log_llm_call``. Always
    returns; never raises.

    ``validator`` receives the parsed dict and returns a
    ``ValidationResult``. If ``fallback_factory`` is ``None``, returns
    ``(data=None, used_fallback=False)`` on exhaustion so the caller can
    decide.

    ``fallback_llms`` (optional): when the primary ``llm`` raises a
    provider error (404 / 500 / "Provider returned error" / "Model not
    found"), the helper rotates to the next LLM in the list and
    continues the retry loop with a fresh attempt budget. This handles
    transient OpenRouter free-tier outages without escalating the whole
    agent state.

    Returns:
        An ``LLMCallOutcome`` describing the parsed data and how it was
        obtained.
    """
    metadata = dict(audit_metadata or {})
    messages: list = [HumanMessage(content=prompt)]
    attempts = 0
    last_reason = "no attempts made"
    # LLM rotation: primary + optional fallbacks. We try each LLM up to
    # (max_retries + 1) times before swapping.
    llm_pool = [llm] + list(fallback_llms or [])
    pool_idx = 0
    active_llm = llm_pool[0]

    for attempt in range(max_retries + 1):
        attempts += 1
        try:
            response = (
                invoke_fn(active_llm, messages, timeout=timeout)
                if _accepts_timeout(invoke_fn)
                else invoke_fn(active_llm, messages)
            )
        except Exception as exc:
            last_reason = f"LLM invocation raised: {exc}"
            logger.warning(
                f"[{role}] attempt {attempts} "
                f"(model={_model_id(active_llm)}): {last_reason}"
            )
            # Provider-side error → try next model in the pool (if any).
            if _is_provider_error(exc) and pool_idx + 1 < len(llm_pool):
                pool_idx += 1
                active_llm = llm_pool[pool_idx]
                logger.warning(
                    f"[{role}] rotating to fallback model #{pool_idx}: "
                    f"{_model_id(active_llm)}"
                )
                # Keep the original prompt for the new model; no critique.
                continue
            # If we still have retries left, fall through to the critique loop.
            if attempt < max_retries:
                messages.append(
                    HumanMessage(
                        content=critique_template.format(reason=last_reason)
                    )
                )
                continue
            break

        raw = extract_content(getattr(response, "content", response))
        log_llm_call(
            agent_role=role,
            prompt=_format_messages_for_log(messages),
            response=raw,
            model=getattr(active_llm, "model_name", "unknown"),
            cost_usd=cost_estimate,
            metadata={**metadata, "attempt": attempts, "pool_idx": pool_idx},
        )

        try:
            data = json.loads(extract_json_block(raw))
            if not isinstance(data, dict):
                raise ValueError("top-level JSON is not an object")
        except (json.JSONDecodeError, ValueError) as exc:
            last_reason = f"JSON parse failure: {exc}"
            logger.warning(f"[{role}] attempt {attempts}: {last_reason}")
            if attempt < max_retries:
                messages.append(
                    HumanMessage(
                        content=critique_template.format(reason=last_reason)
                    )
                )
            continue

        verdict = validator(data)
        if verdict.ok:
            return LLMCallOutcome(
                data=data,
                used_fallback=False,
                attempts=attempts,
                last_error="",
            )

        last_reason = f"schema validation failed: {verdict.reason}"
        logger.warning(f"[{role}] attempt {attempts}: {last_reason}")
        if attempt < max_retries:
            messages.append(
                HumanMessage(
                    content=critique_template.format(reason=last_reason)
                )
            )

    # Exhausted retries — use fallback if provided.
    if fallback_factory is not None:
        try:
            fb = fallback_factory()
            logger.error(
                f"[{role}] all {attempts} attempts failed ({last_reason}); "
                f"using deterministic fallback."
            )
            return LLMCallOutcome(
                data=fb,
                used_fallback=True,
                attempts=attempts,
                last_error=last_reason,
            )
        except Exception as exc:
            logger.exception(f"[{role}] fallback factory raised: {exc}")

    return LLMCallOutcome(
        data=None,
        used_fallback=False,
        attempts=attempts,
        last_error=last_reason,
    )


# ----------------------------------------------------------------------------
# Internal utilities
# ----------------------------------------------------------------------------


def _accepts_timeout(fn: Callable) -> bool:
    """Return True if ``fn`` accepts a ``timeout=`` keyword argument."""
    try:
        import inspect

        return "timeout" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


# Patterns in raised exceptions that indicate a *provider-side* failure
# (model not found, rate limit, upstream 5xx) — i.e. retrying the same
# model is unlikely to help; rotating to a different model often does.
_PROVIDER_ERROR_PATTERNS = (
    "model not found",
    "404",
    "429",
    "rate limit",
    "too many requests",
    "provider returned error",
    "no endpoints found",
    "internal server error",
    "502",
    "503",
    "504",
    "upstream",
)


def _is_provider_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(p in msg for p in _PROVIDER_ERROR_PATTERNS)


def _model_id(llm: Any) -> str:
    return getattr(llm, "model_name", None) or getattr(llm, "model", "unknown")


def _format_messages_for_log(messages: list) -> str:
    """Render the message thread into a single string for the audit log."""
    parts = []
    for i, m in enumerate(messages):
        content = getattr(m, "content", str(m))
        parts.append(f"--- message {i} ({type(m).__name__}) ---\n{content}")
    return "\n".join(parts)


__all__ = [
    "ValidationResult",
    "LLMCallOutcome",
    "llm_call_with_validation",
    "extract_content",
    "extract_json_block",
]

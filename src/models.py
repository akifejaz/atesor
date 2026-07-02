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

"""LLM provider management, model configuration, and cost tracking.

Handles interactions with the OpenAI, Gemini, and OpenRouter providers.
"""

import logging
import os
from enum import Enum
from typing import List, Tuple

from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from .state import AgentRole

logger = logging.getLogger(__name__)

# Process-level cache for the OpenRouter live-model catalogue. Populated
# by _discover_openrouter_free_models() on first use, then reused so
# every create_llm_pool() call doesn't re-hit the /models endpoint.
_FREE_MODELS_CACHE: "List[str] | None" = None

# Default output-token cap applied to every provider. Kept intentionally
# conservative so free-tier OpenRouter accounts (which reject calls whose
# requested `max_tokens` exceeds their remaining credit with HTTP 402)
# stay well within budget. Agent JSON outputs are typically <4k tokens;
# 8k gives a healthy safety margin. Override per-role by wrapping the
# returned LLM if a specific agent needs more.
_DEFAULT_MAX_TOKENS = int(os.getenv("ATESOR_MAX_OUTPUT_TOKENS", "8192"))


class ModelProvider(str, Enum):
    """Supported LLM providers selectable via ``LLM_PROVIDER``."""

    OPENAI = "openai"
    GEMINI = "gemini"
    OPENROUTER = "openrouter"


# Model configurations per provider
MODEL_CONFIG = {
    "openai": {
        "supervisor": {"model": "gpt-3.5-turbo", "temperature": 0.0},
        "planner": {"model": "gpt-4", "temperature": 0.1},
        "scout": {"model": "gpt-4", "temperature": 0.1},
        "builder": {"model": "gpt-3.5-turbo", "temperature": 0.0},
        "fixer": {"model": "gpt-4", "temperature": 0.2},
        "summarizer": {"model": "gpt-3.5-turbo", "temperature": 0.1},
    },
    "gemini": {
        "supervisor": {
            "model": "gemini-flash-lite-latest",
            "temperature": 0.0,
        },
        "planner": {"model": "gemini-flash-lite-latest", "temperature": 0.1},
        "scout": {"model": "gemini-flash-lite-latest", "temperature": 0.1},
        "builder": {"model": "gemini-flash-lite-latest", "temperature": 0.0},
        "fixer": {"model": "gemini-flash-lite-latest", "temperature": 0.2},
        "summarizer": {
            "model": "gemini-flash-lite-latest",
            "temperature": 0.1,
        },
    },
    # OpenRouter free-tier slugs are retired without notice (see run
    # 28020958388 — ``openrouter/free`` and then
    # ``google/gemini-2.0-flash-exp:free`` both went dark). We therefore
    # (a) keep a currently-live coding-oriented slug as the per-role
    # default, (b) diversify across roles so a single retirement can't
    # blackhole every agent, and (c) let ``create_llm_pool`` refresh the
    # fallback list dynamically from ``/models`` at pool-creation time.
    # Override any of these via ``OPENROUTER_FALLBACK_MODELS``.
    "openrouter": {
        "supervisor": {
            "model": "qwen/qwen3-coder:free",
            "temperature": 0.0,
        },
        "planner": {
            "model": "openai/gpt-oss-120b:free",
            "temperature": 0.1,
        },
        "scout": {
            "model": "qwen/qwen3-coder:free",
            "temperature": 0.1,
        },
        "builder": {
            "model": "qwen/qwen3-coder:free",
            "temperature": 0.0,
        },
        "fixer": {
            "model": "openai/gpt-oss-120b:free",
            "temperature": 0.2,
        },
        "summarizer": {
            "model": "meta-llama/llama-3.3-70b-instruct:free",
            "temperature": 0.1,
        },
    },
}


def check_api_keys() -> Tuple[bool, str, str]:
    """Verify required API keys are set for the selected provider."""
    provider = os.getenv("LLM_PROVIDER", ModelProvider.GEMINI.value).lower()

    if provider == ModelProvider.OPENAI.value:
        key = os.getenv("OPENAI_API_KEY")
        if not key or key == "your_key_here":
            return False, "OPENAI_API_KEY not found in environment", provider
        return True, "OpenAI API key verified", provider

    elif provider == ModelProvider.GEMINI.value:
        key = os.getenv("GOOGLE_API_KEY")
        if not key or key == "your_key_here":
            return False, "GOOGLE_API_KEY not found in environment", provider
        return True, "Gemini API key verified (using GOOGLE_API_KEY)", provider

    elif provider == ModelProvider.OPENROUTER.value:
        key = os.getenv("OPENROUTER_API_KEY")
        if not key or key == "your_key_here":
            return (
                False,
                "OPENROUTER_API_KEY not found in environment",
                provider,
            )
        return True, "OpenRouter API key verified", provider

    return False, f"Unknown provider: {provider}", provider


def create_llm(role: AgentRole) -> BaseChatModel:
    """Create an LLM instance for a specific agent role."""
    return _create_llm_with_model(role, _resolve_model_name(role))


def create_llm_pool(role: AgentRole) -> List[BaseChatModel]:
    """Return ``[primary_llm, *fallback_llms]`` for the given role.

    Fallbacks are read from ``OPENROUTER_FALLBACK_MODELS`` (a
    comma-separated list of model ids) when the active provider is
    OpenRouter -- the most failure-prone of the three because the free
    tier returns 404 when upstream models go down. For OpenAI and Gemini
    only the primary is returned (those providers' SDKs already retry
    transient errors transparently).

    A lightweight health-check runs each fallback model with a tiny
    prompt at pool-creation time; models that return 404/429 are skipped
    so the pool only contains reachable models.

    Callers that want resilience to ``Model not found`` / 404 from the
    primary pass this list into
    ``llm_call_with_validation(fallback_llms=...)``.

    Args:
        role: The agent role to build the model pool for.

    Returns:
        A list whose first element is the primary model, followed by any
        reachable fallback models.
    """
    primary = create_llm(role)
    provider = os.getenv("LLM_PROVIDER", ModelProvider.GEMINI.value).lower()
    if provider != ModelProvider.OPENROUTER.value:
        return [primary]
    raw = os.getenv("OPENROUTER_FALLBACK_MODELS", "")
    fallback_ids = [m.strip() for m in raw.split(",") if m.strip()]
    if not fallback_ids:
        # Currently-live curated defaults, diversified across providers
        # to survive a single-provider outage. When OpenRouter retires
        # any of these, llm_call_with_validation rotates to the next on
        # a 404 (see _is_provider_error). Refresh this list by running:
        #   python3 -c "from src.models import
        #     _discover_openrouter_free_models as f; print(f())"
        fallback_ids = [
            "qwen/qwen3-coder:free",
            "openai/gpt-oss-120b:free",
            "openai/gpt-oss-20b:free",
            "meta-llama/llama-3.3-70b-instruct:free",
            "qwen/qwen3-next-80b-a3b-instruct:free",
            "nvidia/nemotron-3-super-120b-a12b:free",
            "openrouter/auto",
        ]
    primary_id = getattr(primary, "model_name", None) or getattr(
        primary, "model", None
    )
    # Skip proactive probing: it (a) self-inflicts rate limits by hitting
    # every candidate at pool-creation time, and (b) mis-classifies
    # transiently-throttled (429) models as permanently unhealthy,
    # collapsing the pool to size 1 exactly when we need diversity.
    # Rotation in llm_call_with_validation handles per-call failures.
    fallbacks = []
    for model_id in fallback_ids:
        if model_id == primary_id:
            continue
        try:
            fallbacks.append(_create_llm_with_model(role, model_id))
        except Exception as exc:
            logger.warning(f"Skipping fallback model '{model_id}': {exc}")
    return [primary, *fallbacks]


def _discover_openrouter_free_models(timeout: int = 8) -> List[str]:
    """Fetch the current OpenRouter ``/models`` catalogue and return live
    ``:free`` slugs.

    Returns an empty list on any error — callers must treat this as a
    best-effort augmentation, not a required step. Cached in-process for
    the lifetime of the run to avoid hammering the endpoint from every
    ``create_llm_pool`` call.
    """
    global _FREE_MODELS_CACHE
    if _FREE_MODELS_CACHE is not None:
        return _FREE_MODELS_CACHE
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        _FREE_MODELS_CACHE = []
        return _FREE_MODELS_CACHE
    try:
        import requests
        resp = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
        resp.raise_for_status()
        models = resp.json().get("data", [])
        _FREE_MODELS_CACHE = sorted(
            m["id"] for m in models if ":free" in m.get("id", "")
        )
        logger.info(
            f"Discovered {len(_FREE_MODELS_CACHE)} live OpenRouter "
            f":free slugs for fallback pool augmentation."
        )
    except Exception as exc:
        logger.warning(f"OpenRouter model discovery failed: {exc}")
        _FREE_MODELS_CACHE = []
    return _FREE_MODELS_CACHE


def _health_check(llm: BaseChatModel, max_wait: int = 5) -> bool:
    """Return True if *llm* responds to a tiny prompt within *max_wait*.

    Lightweight probe that catches 404/429 and unreachable models at
    pool-creation time instead of failing during a real agent call.
    """
    try:
        from langchain_core.messages import HumanMessage
        probe = llm.invoke(
            [HumanMessage(content="Say OK")],
            timeout=max_wait,
        )
        return bool(getattr(probe, "content", ""))
    except Exception:
        return False


def _resolve_model_name(role: AgentRole) -> str:
    provider = os.getenv("LLM_PROVIDER", ModelProvider.GEMINI.value).lower()
    if provider not in MODEL_CONFIG:
        provider = ModelProvider.GEMINI.value
    role_key = role.value if hasattr(role, "value") else str(role)
    if role_key not in MODEL_CONFIG[provider]:
        role_key = "supervisor"
    return MODEL_CONFIG[provider][role_key]["model"]


def _create_llm_with_model(role: AgentRole, model_name: str) -> BaseChatModel:
    """Build a configured LLM for `role` using an explicit `model_name`.

    Temperature comes from MODEL_CONFIG[provider][role]. Provider plumbing
    (api key, base_url) is identical across the role's primary and fallback
    models — only the model id changes.
    """
    provider = os.getenv("LLM_PROVIDER", ModelProvider.GEMINI.value).lower()
    if provider not in MODEL_CONFIG:
        logger.warning(f"Unknown provider {provider}, falling back to Gemini")
        provider = ModelProvider.GEMINI.value
    role_key = role.value if hasattr(role, "value") else str(role)
    if role_key not in MODEL_CONFIG[provider]:
        logger.warning(
            f"Role {role_key} not in {provider} config, using supervisor"
        )
        role_key = "supervisor"
    temperature = MODEL_CONFIG[provider][role_key]["temperature"]

    if provider == ModelProvider.OPENAI.value:
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            request_timeout=120,
            max_tokens=_DEFAULT_MAX_TOKENS,
        )
    elif provider == ModelProvider.GEMINI.value:
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            timeout=120,
            max_output_tokens=_DEFAULT_MAX_TOKENS,
        )
    elif provider == ModelProvider.OPENROUTER.value:
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
            request_timeout=120,
            # OpenRouter free-tier accounts get HTTP 402 when the
            # requested `max_tokens` exceeds available credit. Agent
            # responses are almost always <4k tokens; capping the default
            # at 8k keeps us well within any reasonable free budget
            # without truncating legitimate outputs. Larger caps can
            # still be requested per-call by wrapping the LLM.
            max_tokens=_DEFAULT_MAX_TOKENS,
        )
    raise ValueError(f"Unsupported provider: {provider}")


def print_model_info() -> None:
    """Print information about the selected provider and models."""
    provider = os.getenv("LLM_PROVIDER", ModelProvider.GEMINI.value).lower()
    print(f"   Provider: {provider}")
    if provider in MODEL_CONFIG:
        config = MODEL_CONFIG[provider]
        supervisor = config["supervisor"]["model"]
        planner = config["planner"]["model"]
        builder = config["builder"]["model"]
        print(f"   Models: {supervisor} (S), {planner} (P), " f"{builder} (B)")


# Export functions
__all__ = [
    "create_llm",
    "create_llm_pool",
    "AgentRole",
    "check_api_keys",
    "print_model_info",
    "ModelProvider",
]

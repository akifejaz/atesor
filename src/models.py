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
    "openrouter": {
        "supervisor": {"model": "openrouter/free", "temperature": 0.0},
        "planner": {"model": "openrouter/free", "temperature": 0.1},
        "scout": {"model": "openrouter/free", "temperature": 0.1},
        "builder": {"model": "openrouter/free", "temperature": 0.0},
        "fixer": {"model": "openrouter/free", "temperature": 0.2},
        "summarizer": {"model": "openrouter/free", "temperature": 0.1},
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

    Callers that want resilience to ``Model not found`` / 404 from the
    primary pass this list into
    ``llm_call_with_validation(fallback_llms=...)``.

    Args:
        role: The agent role to build the model pool for.

    Returns:
        A list whose first element is the primary model, followed by any
        configured fallback models.
    """
    primary = create_llm(role)
    provider = os.getenv("LLM_PROVIDER", ModelProvider.GEMINI.value).lower()
    if provider != ModelProvider.OPENROUTER.value:
        return [primary]
    raw = os.getenv("OPENROUTER_FALLBACK_MODELS", "")
    fallback_ids = [m.strip() for m in raw.split(",") if m.strip()]
    if not fallback_ids:
        # Default pool for free-tier OpenRouter where upstream model
        # availability can vary by minute.
        fallback_ids = [
            "meta-llama/llama-3.3-70b-instruct:free",
            "qwen/qwen3-14b:free",
            "deepseek/deepseek-r1-0528:free",
        ]
    fallbacks = []
    for model_id in fallback_ids:
        try:
            fallbacks.append(_create_llm_with_model(role, model_id))
        except Exception as exc:
            logger.warning(f"Skipping fallback model '{model_id}': {exc}")
    return [primary, *fallbacks]


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
            model=model_name, temperature=temperature, request_timeout=120
        )
    elif provider == ModelProvider.GEMINI.value:
        return ChatGoogleGenerativeAI(
            model=model_name, temperature=temperature, timeout=120
        )
    elif provider == ModelProvider.OPENROUTER.value:
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
            request_timeout=120,
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
        print(
            f"   Models: {supervisor} (S), {planner} (P), "
            f"{builder} (B)"
        )


# Export functions
__all__ = [
    "create_llm",
    "create_llm_pool",
    "AgentRole",
    "check_api_keys",
    "print_model_info",
    "ModelProvider",
]

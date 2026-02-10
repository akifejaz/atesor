"""
Enhanced Model Management with Multi-Provider Support.

Supports:
- OpenAI (GPT-3.5, GPT-4)
- Google Gemini (Free & Pro)
- OpenRouter (Free & Paid models)
- Anthropic Claude (via OpenRouter)
"""

import os
import logging
from typing import Optional, Tuple
from enum import Enum
from termcolor import colored
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models import BaseChatModel

from src.state import AgentRole

logger = logging.getLogger(__name__)

# Provider selection
class ModelProvider(str, Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    OPENROUTER = "openrouter"

PROVIDER = os.getenv("LLM_PROVIDER", ModelProvider.GEMINI.value)  # openai, gemini, openrouter

# Model configurations per provider
MODEL_CONFIG = {
    "openai": {
        "supervisor": {"model": "gpt-3.5-turbo", "temperature": 0.0},
        "planner": {"model": "gpt-4", "temperature": 0.1},
        "scout": {"model": "gpt-4", "temperature": 0.1},
        "builder": {"model": "gpt-3.5-turbo", "temperature": 0.0},
        "fixer": {"model": "gpt-4", "temperature": 0.2},
    },
    "gemini": {
        "supervisor": {"model": "gemini-pro", "temperature": 0.0},
        "planner": {"model": "gemini-pro", "temperature": 0.1},
        "scout": {"model": "gemini-pro", "temperature": 0.1},
        "builder": {"model": "gemini-pro", "temperature": 0.0},
        "fixer": {"model": "gemini-pro", "temperature": 0.2},
    },
    "openrouter": {
        "supervisor": {"model": "openrouter/free", "temperature": 0.0},
        "planner": {"model": "openrouter/free", "temperature": 0.1},
        "scout": {"model": "openrouter/free", "temperature": 0.1},
        "builder": {"model": "openrouter/free", "temperature": 0.0},
        "fixer": {"model": "openrouter/free", "temperature": 0.2},
    },
}

# Cost tracking per model (USD per 1K tokens)
COST_PER_1K_TOKENS = {
    "gpt-4": 0.03,
    "gpt-3.5-turbo": 0.001,
    "gemini-pro": 0.0005,
    "gemini-pro-free": 0.0,
    "openrouter/free": 0.0,
    "claude-3-sonnet": 0.003,
}


def create_llm(role: AgentRole, provider: Optional[str] = None) -> BaseChatModel:
    """
    Create an LLM for a specific agent role.
    
    Args:
        role: The agent role (supervisor, scout, builder, etc.)
        provider: Override default provider
    
    Returns:
        Configured LLM instance
    """
    provider = provider or PROVIDER
    config = MODEL_CONFIG[provider][role.value]
    
    logger.info(f"Creating LLM for {role.value}: provider={provider}, "
               f"model={config['model']}, temp={config['temperature']}")
    
    if provider == "openai":
        return ChatOpenAI(
            model=config["model"],
            temperature=config["temperature"],
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
    
    elif provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=config["model"],
            temperature=config["temperature"],
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
    
    elif provider == "openrouter":
        return ChatOpenAI(
            model=config["model"],
            temperature=config["temperature"],
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
        )
    
    else:
        raise ValueError(f"Unknown provider: {provider}")


def check_api_keys() -> Tuple[bool, str, str]:
    """Verify required API keys are set for the selected provider."""
    provider = os.getenv("LLM_PROVIDER", ModelProvider.GEMINI.value).lower()
    
    if provider == ModelProvider.GEMINI.value:
        key = os.getenv("GOOGLE_API_KEY")
        if not key:
            return False, "GOOGLE_API_KEY not found in environment", provider
        return True, "Found GOOGLE_API_KEY", provider
    
    elif provider == ModelProvider.OPENAI.value:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            return False, "OPENAI_API_KEY not found in environment", provider
        return True, "Found OPENAI_API_KEY", provider
        
    elif provider == ModelProvider.OPENROUTER.value:
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            return False, "OPENROUTER_API_KEY not found in environment", provider
        return True, "Found OPENROUTER_API_KEY", provider
        
    return False, f"Unknown provider: {provider}", provider


def print_model_info():
    """Print information about the selected provider and models."""
    provider = os.getenv("LLM_PROVIDER", ModelProvider.GEMINI.value).lower()
    print(f"   Provider: {provider}")
    if provider in MODEL_CONFIG:
        config = MODEL_CONFIG[provider]
        print(f"   Models: {config['supervisor']['model']} (S), {config['planner']['model']} (P), {config['builder']['model']} (B)")


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Estimate cost for a model invocation.
    
    Args:
        model: Model name
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
    
    Returns:
        Estimated cost in USD
    """
    total_tokens = input_tokens + output_tokens
    cost_per_1k = COST_PER_1K_TOKENS.get(model, 0.001)  # Default fallback
    return (total_tokens / 1000) * cost_per_1k


class CostTrackingLLM(BaseChatModel):
    """
    Wrapper LLM that tracks costs automatically.
    """
    
    def __init__(self, base_llm: BaseChatModel, role: AgentRole):
        self.base_llm = base_llm
        self.role = role
        self.total_cost = 0.0
        self.call_count = 0
    
    def invoke(self, messages, *args, **kwargs):
        """Invoke with cost tracking."""
        response = self.base_llm.invoke(messages, *args, **kwargs)
        
        # Track cost (rough estimate)
        input_tokens = sum(len(m.content.split()) * 1.3 for m in messages)  # Rough
        output_tokens = len(response.content.split()) * 1.3
        
        cost = estimate_cost(
            self.base_llm.model_name,
            int(input_tokens),
            int(output_tokens)
        )
        
        self.total_cost += cost
        self.call_count += 1
        
        logger.info(f"{self.role.value} LLM call #{self.call_count}: "
                   f"~${cost:.4f} (total: ${self.total_cost:.4f})")
        
        return response
    
    @property
    def _llm_type(self) -> str:
        return self.base_llm._llm_type


def create_cost_tracking_llm(role: AgentRole) -> CostTrackingLLM:
    """Create an LLM with automatic cost tracking."""
    base_llm = create_llm(role)
    return CostTrackingLLM(base_llm, role)


# Fallback strategy for API failures
class LLMFallbackHandler:
    """
    Handle LLM API failures with graceful fallback.
    """
    
    def __init__(self, preferred_provider: str = PROVIDER):
        self.preferred_provider = preferred_provider
        self.fallback_order = ["gemini", "openrouter", "openai"]
        self.failed_providers = set()
    
    def get_llm(self, role: AgentRole) -> BaseChatModel:
        """
        Get LLM with fallback logic.
        """
        # Try preferred provider first
        if self.preferred_provider not in self.failed_providers:
            try:
                return create_llm(role, self.preferred_provider)
            except Exception as e:
                logger.warning(f"Failed to create {self.preferred_provider} LLM: {e}")
                self.failed_providers.add(self.preferred_provider)
        
        # Try fallbacks
        for provider in self.fallback_order:
            if provider in self.failed_providers:
                continue
            
            try:
                logger.info(f"Falling back to provider: {provider}")
                return create_llm(role, provider)
            except Exception as e:
                logger.warning(f"Failed to create {provider} LLM: {e}")
                self.failed_providers.add(provider)
        
        raise RuntimeError("All LLM providers failed!")
    
    def reset(self):
        """Reset failure tracking."""
        self.failed_providers.clear()


# Global fallback handler
_fallback_handler = LLMFallbackHandler()


def get_llm_with_fallback(role: AgentRole) -> BaseChatModel:
    """
    Get LLM with automatic fallback to alternative providers.
    """
    return _fallback_handler.get_llm(role)


# Export functions
__all__ = [
    "create_llm",
    "create_cost_tracking_llm",
    "get_llm_with_fallback",
    "estimate_cost",
    "AgentRole",
    "check_api_keys",
    "print_model_info",
    "ModelProvider",
]

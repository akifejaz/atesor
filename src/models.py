"""
LLM provider management, model configuration, and cost tracking logic.
Handles interactions with OpenAI, Gemini, and OpenRouter.
"""

import os
import logging
from typing import Optional, Tuple
from enum import Enum
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models import BaseChatModel

from .state import AgentRole

logger = logging.getLogger(__name__)

# Provider selection
class ModelProvider(str, Enum):
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
       "supervisor": {"model": "gemini-flash-lite-latest", "temperature": 0.0},
       "planner": {"model": "gemini-flash-lite-latest", "temperature": 0.1},
       "scout": {"model": "gemini-flash-lite-latest", "temperature": 0.1},
       "builder": {"model": "gemini-flash-lite-latest", "temperature": 0.0},
       "fixer": {"model": "gemini-flash-lite-latest", "temperature": 0.2},
       "summarizer": {"model": "gemini-flash-lite-latest", "temperature": 0.1},
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
            return False, "OPENROUTER_API_KEY not found in environment", provider
        return True, "OpenRouter API key verified", provider
        
    return False, f"Unknown provider: {provider}", provider


def create_llm(role: AgentRole) -> BaseChatModel:
    """Create an LLM instance for a specific agent role."""
    provider = os.getenv("LLM_PROVIDER", ModelProvider.GEMINI.value).lower()
    
    if provider not in MODEL_CONFIG:
        logger.warning(f"Unknown provider {provider}, falling back to Gemini")
        provider = ModelProvider.GEMINI.value
        
    role_key = role.value if hasattr(role, 'value') else str(role)
    if role_key not in MODEL_CONFIG[provider]:
        logger.warning(f"Role {role_key} not found in {provider} config, using supervisor config")
        role_key = "supervisor"
        
    config = MODEL_CONFIG[provider][role_key]
    model_name = config["model"]
    temperature = config["temperature"]
    
    if provider == ModelProvider.OPENAI.value:
        return ChatOpenAI(model=model_name, temperature=temperature, request_timeout=120)
    elif provider == ModelProvider.GEMINI.value:
        return ChatGoogleGenerativeAI(model=model_name, temperature=temperature, timeout=120)
    elif provider == ModelProvider.OPENROUTER.value:
        # OpenRouter uses ChatOpenAI with a base_url
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
            request_timeout=120,
        )
    
    raise ValueError(f"Unsupported provider: {provider}")


def print_model_info():
    """Print information about the selected provider and models."""
    provider = os.getenv("LLM_PROVIDER", ModelProvider.GEMINI.value).lower()
    print(f"   Provider: {provider}")
    if provider in MODEL_CONFIG:
        config = MODEL_CONFIG[provider]
        print(f"   Models: {config['supervisor']['model']} (S), {config['planner']['model']} (P), {config['builder']['model']} (B)")


# Export functions
__all__ = [
    "create_llm",
    "AgentRole",
    "check_api_keys",
    "print_model_info",
    "ModelProvider",
]

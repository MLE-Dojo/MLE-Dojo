"""
utils.py - Utility functions and configurations for AI model tokenization, pricing, and API integrations.

This module provides:
    - ModelSettings: Dataclass to encapsulate local model generation settings.
    - API keys and configurations for various models including local, Azure-based, and other remote APIs.
    - Functions to calculate API usage cost and to prepare message payloads with proper tokenization.

This code adheres to high quality, modular, and user-friendly standards for top-tier open source projects.
"""

from dataclasses import dataclass
import tiktoken


@dataclass
class ModelSettings:
    """
    Encapsulates settings for local model generation.

    Attributes:
        max_completion_tokens (int): Maximum number of tokens allowed for the model's generated completion.
        temperature (float): Sampling temperature for token generation (default: 0.0).
        top_p (float): Nucleus sampling parameter (default: 1.0).
        frequency_penalty (float): Penalty applied based on token frequency (default: 0.0).
        presence_penalty (float): Penalty applied to encourage new token appearances (default: 0.0).
    """
    max_completion_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


# Consolidated API keys for models and integrations.
API_KEYS = {
    "LOCAL_MODEL": "",
    "OPENAI": "",
    "GEMINI": "",
    "DEEPSEEK": "",
    "CLAUDE": "",
    "GROK": "",
}

# Structured Azure API configurations (version 24-0718).
AZURE_API_CONFIG = {
    "gpt-4o-mini": [
        {
            "api_key": "",
            "api_version": "",
            "azure_endpoint": "",
            "engine": "",
        }
    ],
    "gpt-4o": [
        {
            "api_key": "",
            "api_version": "",
            "azure_endpoint": "",
            "engine": "",
        },
    ],
    "o3-mini": [
        {
            "api_key": "",
            "api_version": "",
            "azure_endpoint": "",
            "engine": "",
        },
    ],
    "o1-mini": [
        {
            "api_key": "",
            "api_version": "",
            "azure_endpoint": "",
            "engine": "",
        },
    ],
}

# Mapping for accessing Azure API configurations by model name.
GPT_API_MAPPING = AZURE_API_CONFIG

# Cost rates mapping for different models (prompt_cost, completion_cost) in USD per 1K tokens.
COST_RATE_MAPPING = {
    "gpt-4o-mini": (0.0003, 0.0012),
    "gpt-4o": (0.00375, 0.015),
    "o1-mini": (0.0011, 0.0044),
    "o3-mini": (0.0011, 0.0044),
    "gemini-2.0-flash": (0.0001, 0.0004),
    "gemini-2.5-pro-preview-03-25": (0.00125, 0.01),
    "deepseek-reasoner": (0.00055, 0.00219),
    "deepseek-chat": (0.00027, 0.0011),
    "claude-3.5-sonnet-latest": (0.003, 0.015),
    "claude-3.7-sonnet-latest": (0.003, 0.015),
}


def check_cost(prompt_tokens: int, completion_tokens: int, model_name: str = "gpt-4o-mini") -> float:
    """
    Compute the total API usage cost based on token counts.

    Args:
        prompt_tokens (int): Number of tokens in the input prompt.
        completion_tokens (int): Number of tokens generated in the completion.
        model_name (str): Identifier of the model used for cost lookup (default: "gpt-4o-mini").

    Returns:
        float: Total cost computed in USD.

    Raises:
        ValueError: If an unsupported model name is provided.
    """
    if model_name not in COST_RATE_MAPPING:
        raise ValueError(f"Unsupported model type: {model_name}")
    prompt_cost, completion_cost = COST_RATE_MAPPING[model_name]
    cost = (prompt_tokens / 1000 * prompt_cost) + (completion_tokens / 1000 * completion_cost)
    return cost


def wrap_message(prompt: str, system_prompt: str, max_prompt_tokens: int = 100000) -> list[dict[str, str]]:
    """
    Tokenize and wrap the provided prompt into a formatted message payload.

    This function encodes the given prompt using the tokenization scheme for the "gpt-4o" model.
    It truncates the prompt to ensure it does not exceed a maximum token limit, then reconstructs
    a message list with roles for system and user.

    Args:
        prompt (str): The input text prompt from the user.
        system_prompt (str): The system message providing context or directives.
        max_prompt_tokens (int): Maximum allowed tokens in the prompt (default: 100000).

    Returns:
        list[dict[str, str]]: A list containing two dictionaries structured with roles "system" and "user".
    """
    tokenizer = tiktoken.encoding_for_model("gpt-4o")
    encoded_tokens = tokenizer.encode(prompt)[:max_prompt_tokens]
    truncated_prompt = tokenizer.decode(encoded_tokens)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": truncated_prompt},
    ]

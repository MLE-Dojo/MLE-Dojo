"""
chat.py - Client implementation for various AI chat models with unified interface.

This module provides:
    - ChatClient: A unified client for interacting with various AI models including GPT, Claude, Gemini, etc.
    - Support for Azure OpenAI, direct OpenAI API, Anthropic, and other providers
    - Robust error handling and retry mechanisms
    - Cost tracking for API usage
    - Loading API keys from .env files

This code adheres to high quality, modular, and user-friendly standards for top-tier open source projects.
"""

import time
import os
from dotenv import load_dotenv # Import load_dotenv
from .utils import GPT_API_MAPPING, check_cost, ModelSettings, API_KEYS

# Load environment variables from the project root .env file
# Calculate the project root directory (two levels up from the current file's directory)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
dotenv_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path)

class ChatClient:
    """
    Unified client for interacting with various AI chat models.
    
    Supports multiple model providers including OpenAI (GPT), Anthropic (Claude),
    Google (Gemini), DeepSeek, Grok, and local models.
    
    Attributes:
        model_name (str): Name of the model to use
        model_category (str): Category of the model (gpt, claude, etc.)
        api_idx (int): Index for API configuration when multiple are available (primarily for Azure)
        _passed_api_key (str | None): API key explicitly passed during initialization.
        port (int): Port for local model server
        client: The underlying API client
        engine (str): Engine name for the model (used mainly for Azure)
        api_key (str | None): The API key determined based on priority logic.
        total_cost (float): Running total of API usage cost
        max_retries (int): Maximum number of retry attempts
        retry_delay (int): Base delay between retries in seconds
    """
    
    SUPPORTED_CATEGORIES = {"gpt", "gemini", "claude", "deepseek", "grok", "local"}
    # Map model category to environment variable name
    ENV_VAR_MAP = {
        "gpt": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "claude": "ANTHROPIC_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",  # Assuming this is the correct env var name
        "grok": "XAI_API_KEY",
        "local": "HF_TOKEN",  # User specified HF_TOKEN for local models
    }


    def __init__(
        self,
        model_name: str,
        api_idx: int = 0,
        api_key: str = None,
        model_category: str = "gpt",
        port: int = 8000,
    ):
        if model_category not in self.SUPPORTED_CATEGORIES:
            raise ValueError(f"Unsupported model category: {model_category}")

        self.model_name = model_name
        self.model_category = model_category
        self.api_idx = api_idx
        self._passed_api_key = api_key # Store passed api_key separately
        self.port = port

        self.client = None
        self.engine = None
        self.api_key = None # Will be determined in init_client
        self.total_cost = 0.0

        self.max_retries = 3
        self.retry_delay = 5

        self.init_client()

    def init_client(self) -> None:
        """
        Initialize the appropriate client based on model category and configuration.

        Loads API keys from a .env file first, then checks environment variables,
        Azure configuration, passed parameters, and default keys.

        Retrieves API keys based on the following priority:
        1. Environment Variable (potentially loaded from .env) specific to model category
        2. Azure Configuration via `api_idx` (for GPT models only)
        3. API key passed via `api_key` parameter during initialization
        4. Default key from `utils.API_KEYS` (if applicable)
        """
        from openai import OpenAI, AzureOpenAI
        import anthropic

        # --- Common Logic: Determine API Key ---
        # Environment variables are now potentially loaded from .env by load_dotenv() called earlier
        env_var_name = self.ENV_VAR_MAP.get(self.model_category)
        env_api_key = os.environ.get(env_var_name) if env_var_name else None
        
        # Default to environment variable if available
        self.api_key = env_api_key
        
        # --- GPT Specific Logic (Azure vs Direct OpenAI) ---
        if self.model_category == "gpt":
            # Priority 1: Environment Variable (already assigned to self.api_key)
            # Priority 2: Azure Configuration (if env var not set and api_idx is valid)
            if not self.api_key and self.api_idx >= 0 and self.model_name in GPT_API_MAPPING:
                 if 0 <= self.api_idx < len(GPT_API_MAPPING[self.model_name]):
                    api_info = GPT_API_MAPPING[self.model_name][self.api_idx]
                    self.client = AzureOpenAI(
                        azure_endpoint=api_info["azure_endpoint"],
                        api_key=api_info["api_key"], # Use Azure key directly
                        api_version=api_info["api_version"],
                    )
                    self.engine = api_info["engine"]
                    self.api_key = api_info["api_key"] # Store the used key
                    return # Azure client initialized, skip further steps
                 else:
                    raise IndexError(
                        f"Invalid api_idx={self.api_idx} for model_name={self.model_name}"
                    )

            # Priority 3: Passed API Key (if env var and Azure not used)
            if not self.api_key:
                self.api_key = self._passed_api_key
            
            # Priority 4: Fallback to API_KEYS dict (if others not provided) - only for non-Azure
            if not self.api_key:
                 self.api_key = API_KEYS.get("OPENAI") # Use .get for safety

            # Initialize Direct OpenAI client
            self.client = OpenAI(api_key=self.api_key)
            self.engine = self.model_name # Direct OpenAI uses model_name as engine
            return

        # --- Anthropic (Claude) Logic ---
        if self.model_category == "claude":
            # Priority 1: Environment Variable (already assigned to self.api_key)
            # Priority 2: Passed API Key
            if not self.api_key:
                self.api_key = self._passed_api_key
            # Priority 3: Fallback to API_KEYS dict
            if not self.api_key:
                self.api_key = API_KEYS.get("ANTHROPIC") # Use .get for safety
            
            self.client = anthropic.Anthropic(api_key=self.api_key)
            return

        # --- Other Providers (Gemini, DeepSeek, Grok, Local) Logic ---
        # Priority 1: Environment Variable (already assigned to self.api_key)
        # Priority 2: Passed API Key
        if not self.api_key:
            self.api_key = self._passed_api_key

        # Priority 3: Fallback to API_KEYS dict (for non-local)
        if not self.api_key and self.model_category != "local":
             self.api_key = API_KEYS.get(self.model_category.upper()) # Use .get for safety

        # Note: For 'local', if HF_TOKEN env var and passed api_key are not set,
        # self.api_key will be None, which might be acceptable for some local servers.

        base_url_map = {
            "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
            "deepseek": "https://api.deepseek.com",
            "grok": "https://api.x.ai/v1",
            "local": f"http://localhost:{self.port}/v1",
        }

        if self.model_category in base_url_map:
            self.client = OpenAI(api_key=self.api_key, base_url=base_url_map[self.model_category])
        else:
            # This case should theoretically not be reached due to __init__ check
            raise ValueError(f"Unsupported or unhandled model category: {self.model_category}")

    def get_client(self):
        """Return the underlying API client."""
        return self.client

    def chat_completion(
        self, messages: list[dict[str, str]], settings: ModelSettings
    ) -> tuple[str, float]:
        """
        Generate a chat completion based on the selected model.
        
        :param messages: List of message dictionaries with 'role' and 'content'
        :param settings: Model generation settings (max tokens, temperature, etc.)
        :return: Tuple of (response_text, cost)
        :raises: Exception with detailed message if the request fails permanently
        """
        attempt = 0
        while attempt < self.max_retries:
            try:
                # ---------------------------
                # Anthropic / Claude branch
                # ---------------------------
                if self.model_category == "claude":
                    from anthropic import (
                        APIConnectionError as AnthropicAPIConnectionError,
                        RateLimitError as AnthropicRateLimitError,
                        APIStatusError as AnthropicAPIStatusError,
                        BadRequestError as AnthropicBadRequestError,
                        AuthenticationError as AnthropicAuthError,
                        PermissionDeniedError as AnthropicPermissionDeniedError,
                        NotFoundError as AnthropicNotFoundError,
                        UnprocessableEntityError as AnthropicUnprocessableEntityError,
                        InternalServerError as AnthropicInternalServerError,
                    )
                    response = self.client.messages.create(
                        model=self.model_name,
                        messages=messages,
                        max_tokens=settings.max_completion_tokens,
                        temperature=settings.temperature,
                        top_p=settings.top_p,
                    )
                    # Anthropic returns text in a different field
                    response_text = response.content[0].text
                    cost = 0.0  # Or implement your own cost logic

                    return response_text, cost

                # ---------------------------
                # OpenAI-based branch
                # (GPT, Gemini, DeepSeek, Local, etc.)
                # ---------------------------
                else:
                    from openai import (
                        RateLimitError as OpenAIRateLimitError,
                        APIConnectionError as OpenAIApiConnectionError,
                        AuthenticationError as OpenAIAuthError,
                        APITimeoutError as OpenAITimeoutError,
                        BadRequestError as OpenAIBadRequestError,
                        ConflictError as OpenAIConflictError,
                        InternalServerError as OpenAIInternalServerError,
                    )

                    # A few providers have subtle differences in parameters
                    if self.model_name in ["o3-mini", "o1-mini"]:
                        # "Mini" GPT logic
                        response = self.client.chat.completions.create(
                            messages=messages,
                            model=self.engine,  # Notice these use self.engine, not self.model_name
                            max_completion_tokens=settings.max_completion_tokens,
                        )
                        # GPT uses token-based cost
                        cost = check_cost(
                            response.usage.prompt_tokens,
                            response.usage.completion_tokens,
                            self.model_name,
                        )
                        response_text = response.choices[0].message.content

                    elif self.model_name in ["gpt-4o-mini", "gpt-4o"]:
                        response = self.client.chat.completions.create(
                            messages=messages,
                            model=self.engine,
                            max_tokens=settings.max_completion_tokens,
                            temperature=settings.temperature,
                            top_p=settings.top_p,
                            frequency_penalty=settings.frequency_penalty,
                            presence_penalty=settings.presence_penalty,
                        )
                        response_text = response.choices[0].message.content
                        cost = check_cost(
                            response.usage.prompt_tokens,
                            response.usage.completion_tokens,
                            self.model_name,
                        )


                    elif self.model_category == "local":
                        # Local usage has different parameter names (e.g., max_tokens)
                        response = self.client.chat.completions.create(
                            messages=messages,
                            model=self.model_name,
                            max_tokens=settings.max_completion_tokens,  # distinct from max_completion_tokens
                            temperature=settings.temperature,
                            top_p=settings.top_p,
                            frequency_penalty=settings.frequency_penalty,
                            presence_penalty=settings.presence_penalty,
                        )
                        response_text = response.choices[0].message.content
                        cost = 0.0

                    else:
                        # Gemini, DeepSeek, or other models usage
                        response = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=messages,
                            max_tokens=settings.max_completion_tokens,
                            temperature=settings.temperature,
                            top_p=settings.top_p,
                        )
                        response_text = response.choices[0].message.content

                        # TODO: Calculate cost for models other than GPT
                        if self.model_category == "gemini":
                            cost = check_cost(
                                response.usage.prompt_tokens,
                                response.usage.completion_tokens,
                                self.model_name,
                            )
                        else:
                            cost = 0.0

                    # Accumulate cost for GPT, if any
                    self.total_cost += cost
                    return response_text, cost

            # ---------------------------
            # Exception handling based on model category
            # ---------------------------
            except Exception as e:
                if self.model_category == "claude":
                    if isinstance(e, (AnthropicAPIConnectionError, AnthropicRateLimitError, AnthropicInternalServerError)):
                        attempt += 1
                        if attempt < self.max_retries:
                            time.sleep(self.retry_delay * (2**attempt))
                            continue

                        if isinstance(e, AnthropicAPIConnectionError):
                            raise Exception("The server could not be reached") from e
                        elif isinstance(e, AnthropicRateLimitError):
                            raise Exception("Rate limit exceeded") from e
                        else:
                            raise Exception("Internal server error") from e

                    elif isinstance(e, AnthropicAPIStatusError):
                        if isinstance(e, AnthropicBadRequestError):
                            raise Exception(f"Bad request: {e.status_code}") from e
                        elif isinstance(e, AnthropicAuthError):
                            raise Exception(f"Authentication failed: {e.status_code}") from e
                        elif isinstance(e, AnthropicPermissionDeniedError):
                            raise Exception(f"Permission denied: {e.status_code}") from e
                        elif isinstance(e, AnthropicNotFoundError):
                            raise Exception(f"Resource not found: {e.status_code}") from e
                        elif isinstance(e, AnthropicUnprocessableEntityError):
                            raise Exception(f"Unprocessable entity: {e.status_code}") from e
                        else:
                            raise Exception(f"API error: {e.status_code}") from e
                else:
                    if isinstance(e, (OpenAIApiConnectionError, OpenAIRateLimitError, OpenAITimeoutError, OpenAIInternalServerError)):
                        attempt += 1
                        if attempt < self.max_retries:
                            time.sleep(self.retry_delay * (2**attempt))
                            continue

                        if isinstance(e, OpenAIApiConnectionError):
                            raise Exception("The server could not be reached") from e
                        elif isinstance(e, OpenAIRateLimitError):
                            raise Exception("Rate limit exceeded") from e
                        elif isinstance(e, OpenAITimeoutError):
                            raise Exception("Request timed out") from e
                        else:
                            raise Exception("Internal server error") from e

                    elif isinstance(e, OpenAIAuthError):
                        raise Exception("Authentication failed") from e
                    elif isinstance(e, OpenAIBadRequestError):
                        raise Exception(f"Invalid request: {str(e)}") from e
                    elif isinstance(e, OpenAIConflictError):
                        raise Exception(f"Request conflict: {str(e)}") from e

                raise Exception(f"Unexpected error: {str(e)}") from e

        raise Exception(f"Failed after {self.max_retries} retries")

from typing import Any, Dict, Union, List, Tuple
from dataclasses import dataclass
import json
import tiktoken
import os
import time
from mledojo.chat import ChatClient, ModelSettings
from google import genai
from google.genai import types
from mledojo.agent.mleagent.prompt import MLEAgentPrompts

@dataclass
class LLMConfig:
    """Configuration for LLM API settings"""
    model_mode: str  # "local", "gpt" or "gemini"
    model_name: str  # Model name or engine name
    port: int = 8314  # Port for local model
    temperature: float = 0.0
    top_p: float = 1.0
    max_completion_tokens: int = 8192
    max_prompt_tokens: int = 30000

class KaggleAgent:
    """Agent class to interact with KaggleEnvironment using LLM"""
    
    def __init__(self, 
                 api_idx: int,
                 api_key: str,
                 llm_config: LLMConfig,
                 history_length: Union[int, str] = "all",
                 init_method: str = "cold",
                 history_init: str = None):
        self.api_idx = api_idx
        self.total_cost = 0.0
        self.history_length = history_length
        self.fix_parse_history = []
        self.cost_history = []  # Track cost history for each action
        self.init_method = init_method
        self.history_init = history_init
        
        # LLM settings
        self.llm_config = llm_config
        self.tokenizer = tiktoken.encoding_for_model('gpt-4')
        self.total_tokens = 0
        
        # Initialize conversation history
        self.conversation_history = []
        self.message_history = ""  # For Gemini models
        
        # Special handling for experimental Gemini models
        self.is_experimental_gemini = llm_config.model_name in ["gemini-2.0-pro-exp", "gemini-2.5-pro-preview-03-25", "gemini-2.5-pro-exp-03-25"]
        
        if self.is_experimental_gemini:
            self.gemini_client = genai.Client(api_key=api_key)
        else:
            self.model_client = ChatClient(
                model_name=llm_config.model_name,
                model_category=llm_config.model_mode,
                api_idx=api_idx,
                port=llm_config.port,
                api_key=api_key
            )
            self.model_settings = ModelSettings(
                max_completion_tokens=llm_config.max_completion_tokens,
                temperature=llm_config.temperature,
                top_p=llm_config.top_p
            )
            
        # Initialize prompts
        self.agent_prompts = MLEAgentPrompts()

    def _add_message(self, role: str, content: str):
        """Add message to conversation history in appropriate format"""
        if self.is_experimental_gemini:
            self.message_history += f"{role}: {content}\n"
            # Still keep the conversation history for potential other uses or debugging
            self.conversation_history.append({"role": role, "content": content})
        else:
            # Treat all other models (including claude) the same way
            self.conversation_history.append({"role": role, "content": content})

    def _init_conversation(self, max_steps: int, max_time: int):
        """Initialize conversation history"""
        if self.init_method == "hot" and self.history_init and os.path.exists(self.history_init):
            with open(self.history_init, 'r') as f:
                history = json.load(f)
                for item in history[:11]:
                    self._add_message(item["role"], item["content"])
        else:
            user_msg = self.agent_prompts.instruction_prompt.format(
                num_actions=max_steps, 
                time_left=max_time
            )
            self._add_message("user", user_msg)

    def _check_token_limit(self) -> bool:
        """Check if token limit is exceeded"""
        text = self.message_history if self.is_experimental_gemini else " ".join(
            msg["content"] for msg in self._get_truncated_history()
        )
        self.total_tokens = len(self.tokenizer.encode(text))
        return self.total_tokens < self.llm_config.max_prompt_tokens

    def _get_gemini_response(self) -> Tuple[str, float]:
        """Get response from Gemini model with retry logic"""
        try:
            response = self.gemini_client.models.generate_content(
                model=self.llm_config.model_name,
                contents=self.message_history + "assistant: ",
                config=types.GenerateContentConfig(
                    max_output_tokens=self.llm_config.max_completion_tokens,
                    temperature=0.0
                )
            )
            response_text = response.text
            self._add_message("assistant", response_text)
            return response_text, 0.0
        except Exception as e:
            if "429 RESOURCE_EXHAUSTED" in str(e):
                print("Gemini rate limit reached. Waiting for 100 seconds...")
                time.sleep(100)
                return self._get_gemini_response()
            raise e

    def _get_llm_response(self) -> Tuple[str, float]:
        """Get response from LLM"""
        try:
            if self.is_experimental_gemini:
                return self._get_gemini_response()
            else:
                response, cost = self.model_client.chat_completion(
                    self._get_truncated_history(), 
                    self.model_settings
                )
                self.total_cost += cost
                self.cost_history.append({"action": "get_llm_response", "cost": cost})
                self._add_message("assistant", response)
                return response, cost
        except Exception as e:
            print(f"Failed to get LLM response: {e}")
            return "Error", str(e)

    def _add_env_response(self, obs: Any, action_left: int, time_left: int):
        """Add environment response to conversation history"""
        feedback = obs.get("feedback").get("base").get("feedback")
        status = obs.get("action_status")
        prompt_template = self.agent_prompts.error if status == "FAILED" else self.agent_prompts.reflection
        formatted_prompt = prompt_template.format(
            observation=feedback, 
            num_actions=action_left, 
            time_left=time_left
        )
        self._add_message("user", formatted_prompt)

    def _get_truncated_history(self) -> List[Dict[str, str]]:
        """Get truncated conversation history"""
        if self.history_length == "all":
            return self.conversation_history
        return self.conversation_history[:2] + self.conversation_history[2:][-2*self.history_length:]

    def _extract_and_parse_json(self, response: str) -> Dict[str, Any]:
        """Extract JSON from response and parse it without validation"""
        # Extract JSON from markdown code blocks if present
        if "```json" in response and "```" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            if start > 6 and end > start:
                response = response[start:end].strip()
        
        if not response.strip():
            raise ValueError("Empty response")
            
        response_dict = json.loads(response)
        
        if not isinstance(response_dict, dict):
            raise ValueError(f"Response must be a JSON object, got {type(response_dict)}")
            
        return response_dict

    def _validate_response_format(self, response_dict: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Validate the parsed response format and return action and params if valid"""
        allowed_keys = {"action", "params"}
        actual_keys = set(response_dict.keys())
        if actual_keys != allowed_keys:
            raise ValueError(f"Response must contain exactly these keys: {allowed_keys}, got: {actual_keys}")
            
        action = response_dict["action"]
        params = response_dict["params"]
        
        valid_actions = {"request_info", "validate_code", "execute_code"}
        if action not in valid_actions:
            raise ValueError(f"Invalid action '{action}'. Must be one of: {valid_actions}")
            
        if not isinstance(params, dict):
            raise ValueError(f"'params' must be a dict, got {type(params)}")
            
        # Validate params based on action type
        if action == "request_info":
            allowed_params_keys = {"info_type"}
            actual_params_keys = set(params.keys())
            if actual_params_keys != allowed_params_keys:
                raise ValueError(f"'request_info' params must contain exactly: {allowed_params_keys}, got: {actual_params_keys}")
            if not isinstance(params["info_type"], str):
                raise ValueError(f"'info_type' must be string, got {type(params['info_type'])}")
        else:  # validate_code or execute_code
            allowed_params_keys = {"code"}
            actual_params_keys = set(params.keys())
            if actual_params_keys != allowed_params_keys:
                raise ValueError(f"'{action}' params must contain exactly: {allowed_params_keys}, got: {actual_params_keys}")
            if not isinstance(params["code"], str):
                raise ValueError(f"'code' must be string, got {type(params['code'])}")
                
        return action, params

    def _fix_parse_error(self, error: str, response: str, max_attempts: int = 3) -> Tuple[str, Dict[str, Any]]:
        """Fix parsing errors by asking the LLM to correct the response format"""
        fix_prompt = self.agent_prompts.fix_parse_error.format(error=error, response=response)
        fix_history = []

        for attempt in range(max_attempts):
            try:
                if self.is_experimental_gemini:
                    self._add_message("user", fix_prompt)
                    fixed_response, cost = self._get_gemini_response()
                else:
                    # Treat all other models (including claude) the same way
                    fix_messages = [{"role": "user", "content": fix_prompt}]
                    fixed_response, cost = self.model_client.chat_completion(fix_messages, self.model_settings)
                    self.total_cost += cost
                    self.cost_history.append({"action": "fix_parse_error", "attempt": attempt + 1, "cost": cost})

                fix_history.append({
                    "attempt": attempt + 1,
                    "error": error,
                    "fixed": fixed_response
                })
                
                # First extract and parse the JSON without validation
                response_dict = self._extract_and_parse_json(fixed_response)
                # Then validate the format and get action and params
                action, params = self._validate_response_format(response_dict)
                
                self.fix_parse_history.append({
                    "original_error": error,
                    "original_response": response,
                    "fix_attempts": fix_history,
                    "success": True,
                    "final_attempt": attempt + 1
                })
                
                return action, params
                
            except ValueError as e:
                if attempt == max_attempts - 1:
                    self.fix_parse_history.append({
                        "original_error": error,
                        "original_response": response,
                        "fix_attempts": fix_history,
                        "success": False,
                        "final_error": str(e)
                    })
                    raise ValueError(f"Failed to fix parse error after {max_attempts} attempts")
    
    def _parse_llm_response(self, response: str) -> Tuple[str, Dict[str, Any]]:
        """Parse LLM response to extract action and parameters"""
        try:
            # First extract and parse the JSON without validation
            response_dict = self._extract_and_parse_json(response)
            # Then validate the format and get action and params
            return self._validate_response_format(response_dict)
            
        except ValueError as e:
            return self._fix_parse_error(str(e), response)
        except Exception as e:
            return self._fix_parse_error(f"Unexpected error: {str(e)}", response)

    def act(self, obs: Any, action_left: int, time_left: int) -> Tuple[str, Dict[str, Any]]:
        """Main interface to get action from agent"""
        if not self._check_token_limit():
            return "End", "Reached token limit"

        try:
            if not self.conversation_history:
                self._init_conversation(action_left, time_left)
            else:
                self._add_env_response(obs, action_left, time_left)
                
            response, cost = self._get_llm_response()
            action, params = self._parse_llm_response(response)
            return action, params
        except Exception as e:
            print(f"Error: {e}")
            return "Error", str(e)

from pydantic import BaseModel
from agents import Agent, Runner
from mledojo.agent.openaiagent.prompt import OpenaiAgentPrompts
from typing import Any, Dict, Union, List, Tuple
import json
import os

class AgentAction(BaseModel):
    """Defines the structure for agent action responses."""
    action: str  # Must be one of: "request_info", "validate_code", "execute_code" 
    params_key: str  # Must be one of: "info_type", "code"
    params_value: str  # Depends on the action type

class OpenaiAgent:
    """Agent class to interact with KaggleEnvironment using LLM"""
    
    def __init__(self, agent: Agent, history_length: Union[int, str] = "all", 
                 init_method: str = "cold", history_init: str = None):
        self.agent = agent
        self.history_length = history_length
        self.init_method = init_method
        self.history_init = history_init
        self.conversation_history = []
        self.history_to_save = []
        self.cost_history = []
        self.fix_parse_history = []
        self.agent_prompts = OpenaiAgentPrompts()

    async def _get_llm_response(self, input_messages) -> str:
        """Get response from LLM"""
        try:
            result = await Runner.run(self.agent, input_messages)
            return result.final_output
        except Exception as e:
            print(f"Failed to get LLM response: {e}")
            return None

    def _init_conversation(self, max_steps: int, max_time: int):
        """Initialize conversation history"""
        if self.init_method == "hot" and self.history_init and os.path.exists(self.history_init):
            with open(self.history_init) as f:
                self.conversation_history = json.load(f)[:12]
            return
            
        self.conversation_history = [{
            "role": "user", 
            "content": self.agent_prompts.instruction_prompt.format(
                num_actions=max_steps,
                time_left=max_time
            )
        }]
        self.history_to_save = self.conversation_history.copy()

    def _add_env_response(self, obs: Any, action_left: int, time_left: int):
        """Add environment response to conversation history"""
        feedback = obs.get("feedback", {}).get("base", {}).get("feedback", "")
        status = obs.get("action_status")
        prompt = (self.agent_prompts.error if status == "FAILED" else self.agent_prompts.reflection)
        self.conversation_history.append({
            "role": "user",
            "content": prompt.format(
                observation=feedback,
                num_actions=action_left,
                time_left=time_left
            )
        })
        self.history_to_save.append({
            "role": "user",
            "content": prompt.format(
                observation=feedback,
                num_actions=action_left,
                time_left=time_left
            )
        })

    def _parse_llm_response(self, response: AgentAction) -> Tuple[str, Dict[str, Any]]:
        """Parse LLM response to extract action and parameters"""
        try:
            action, params = response.action, {response.params_key: response.params_value}
            self.history_to_save.append({
                "role": "assistant",
                "content": f"action: {action}, params: {params}"
            })
            return action, params
        except Exception as e:
            return "Error", {"error": str(e)}

    async def act(self, obs: Any, action_left: int, time_left: int) -> Tuple[str, Dict[str, Any]]:
        """Main interface to get action from agent"""
        try:
            if not self.conversation_history:
                self._init_conversation(action_left, time_left)
            else:
                self._add_env_response(obs, action_left, time_left)
                
            response = await self._get_llm_response(self.conversation_history)
            if response:
                return self._parse_llm_response(response)
            return "Error", {"error": "Failed to get LLM response"}
        except Exception as e:
            return "Error", {"error": str(e)}

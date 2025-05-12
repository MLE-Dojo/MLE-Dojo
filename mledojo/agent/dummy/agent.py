from typing import Any, Dict, Tuple
import json
from mledojo.agent.dummy.prompt import DummyPrompts

class DummyAgent:
    """
    A simple agent that doesn't use LLM, just performs 5 predefined steps
    to request different types of information from the environment.
    """
    
    def __init__(self):
        self.step_count = 0
        self.info_types = [
            "overview", 
            "sample_submission", 
            "data_structure", 
            "data_path", 
            "output_path"
        ]
        self.conversation_history = []
        self.agent_prompts = DummyPrompts()
        self.fix_parse_history = []
        self.cost_history = []
        
    def _init_conversation(self, max_steps: int, max_time: int):
        """Initialize conversation history"""
        self.conversation_history = [{
            "role": "user", 
            "content": self.agent_prompts.instruction_prompt.format(
                num_actions=max_steps,
                time_left=max_time
            )
        }]

        
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
        
    def act(self, obs: Any, action_left: int, time_left: int) -> Tuple[str, Dict[str, Any]]:
        """
        Main interface to get action from agent.
        Simply requests information in a predefined sequence.
        
        Args:
            obs: Observation from the environment
            action_left: Number of actions left
            time_left: Time left in seconds
            
        Returns:
            Tuple of (action, parameters)
        """
        try:
            if not self.conversation_history:
                self._init_conversation(action_left, time_left)
            else:
                self._add_env_response(obs, action_left, time_left)
            
            if self.step_count >= len(self.info_types):
                # If we've gone through all info types, just repeat the last one
                info_type = self.info_types[-1]
            else:
                info_type = self.info_types[self.step_count]
                self.step_count += 1
            
            # Add the fixed response to conversation history
            response = {
                "action": "request_info",
                "params": {
                    "info_type": info_type
                }
            }
            
            self.conversation_history.append({
                "role": "assistant",
                "content": json.dumps(response)
            })
            # Always use the request_info action with the current info type
            return "request_info", {"info_type": info_type}
        except Exception as e:
            return "Error", {"error": str(e)}
